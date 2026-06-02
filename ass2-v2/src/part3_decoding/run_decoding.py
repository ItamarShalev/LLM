"""
Part 3 - Run constrained decoding.

Loads Qwen2.5-7B-Instruct and Mistral-7B-Instruct-v0.3, and for each of the ten
English queries generates: (a) an unconstrained answer and (b) a constrained
answer that may only use the Hebrew-allowed token set.

Writes two CSV files:
    * outputs/decoding_outputs_mask_only.csv
        - regular decoding flow (mask only, no extra instruction)
    * outputs/decoding_outputs_system_prompt.csv
        - same flow, but with an additional Hebrew instruction
            - Qwen: passed as a system message
            - Mistral: prepended to the user prompt (Mistral chat template path)

This script needs a GPU (or a patient CPU) and the model weights, so it is meant
to run on Claude Code's machine. Everything it depends on (the allowed-token JSON
files, the queries) is already produced by the lighter scripts.

Usage:
    python -m src.part3_decoding.run_decoding
    python -m src.part3_decoding.run_decoding --max-new-tokens 80 --device cuda
"""

from __future__ import annotations

import argparse
import csv
import gc
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as C
from data.fixtures import DECODING_QUERIES
from src.part3_decoding.constrained_decode import build_processor, load_allowed_ids

ALLOWED_FILES = {
    "Qwen/Qwen2.5-7B-Instruct": C.HEBREW_TOKENS_QWEN,
    "mistralai/Mistral-7B-Instruct-v0.3": C.HEBREW_TOKENS_MISTRAL,
}

MASK_ONLY_CSV = C.OUTPUTS / "decoding_outputs.csv"
SYSTEM_PROMPT_CSV = C.OUTPUTS / "decoding_outputs_system_prompt.csv"
CSV_FIELDS = ["prompt", "model", "unconstrained_output", "constrained_output"]
HEB_SYSTEM_INSTRUCTION = (
    "You are a helpful assistant. You ALWAYS answer in natural, fluent Hebrew, "
    "regardless of the language of the question. Keep answers concise (1-4 sentences)."
)


def load_model(model_id, device, dtype):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok: Any = AutoTokenizer.from_pretrained(model_id, token=C.hf_token())
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    # `torch_dtype` is the supported Hugging Face argument. Keep the caller's
    # requested dtype so we do not inflate memory usage on smaller machines.
    if dtype == "auto":
        resolved_dtype = torch.float16
    else:
        resolved_dtype = getattr(torch, dtype)

    # On CUDA we let accelerate place layers automatically. Forcing
    # device_map=cuda for a 7B bf16 model on a 16 GB card OOMs/segfaults;
    # "auto" plus a generous max_memory keeps as many layers on GPU as fit.
    kwargs: dict[str, Any] = {
        "token": C.hf_token(),
        "torch_dtype": resolved_dtype,
        "low_cpu_mem_usage": True,
    }
    # Mistral v0.3 has been more sensitive to fused attention paths on some
    # systems, so keep it on the safe eager attention implementation.
    if "mistral" in model_id.lower():
        kwargs["attn_implementation"] = "eager"
    if device == "cuda":
        free, _ = torch.cuda.mem_get_info()
        # Reserve ~1.5 GB for KV cache and runtime activations; let the rest go
        # to weights. accelerate will spill the remainder to CPU automatically.
        gpu_budget_gib = max(1.0, (free / (1024**3)) - 1.5)
        kwargs["device_map"] = "auto"
        kwargs["max_memory"] = {0: f"{gpu_budget_gib:.1f}GiB", "cpu": "48GiB"}
    model: Any = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if device != "cuda":
        model = model.to(device)
    model.eval()
    return tok, model


def chat_prompt(tok, query, system: str | None = None):
    if system:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": query}]
    else:
        msgs = [{"role": "user", "content": query}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def generate(tok, model, prompt_text, max_new_tokens, processor=None):
    import torch

    inputs = tok(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            logits_processor=processor,
            pad_token_id=tok.pad_token_id,
        )
    gen = out[0][inputs["input_ids"].shape[1] :]
    return tok.decode(gen, skip_special_tokens=True).strip()


def prompt_with_hebrew_instruction(tok, model_id: str, query: str) -> str:
    """Build instructed prompt:
    - Qwen: instruction as system message.
    - Mistral: instruction prepended to user query.
    """
    mid = model_id.lower()
    if mid.startswith("qwen/"):
        return chat_prompt(tok, query, system=HEB_SYSTEM_INSTRUCTION)
    if "mistral" in mid:
        instructed_query = HEB_SYSTEM_INSTRUCTION + "\n\n" + query
        return chat_prompt(tok, instructed_query)
    return chat_prompt(tok, query, system=HEB_SYSTEM_INSTRUCTION)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--device", default="cuda")
    # Loading bf16 safetensors weights segfaults on torch 2.12 + Blackwell laptop
    # GPUs (RTX 50-series); float16 has the same memory footprint and loads
    # cleanly. Override with --dtype bfloat16 on hardware that supports it.
    ap.add_argument("--dtype", default="float16")
    args = ap.parse_args()

    mask_fh = MASK_ONLY_CSV.open("w", encoding="utf-8", newline="")
    sys_fh = SYSTEM_PROMPT_CSV.open("w", encoding="utf-8", newline="")
    mask_writer = csv.DictWriter(mask_fh, fieldnames=CSV_FIELDS)
    sys_writer = csv.DictWriter(sys_fh, fieldnames=CSV_FIELDS)
    mask_writer.writeheader()
    sys_writer.writeheader()

    written = 0
    try:
        for model_id, allowed_file in ALLOWED_FILES.items():
            print(f"[load] {model_id}  ({len(DECODING_QUERIES)} prompts to do)")
            tok, model = load_model(model_id, args.device, args.dtype)
            allowed = load_allowed_ids(allowed_file)
            proc = build_processor(tok, allowed)
            for q in DECODING_QUERIES:
                # 1) Baseline file: mask-only flow (no extra Hebrew instruction).
                p_base = chat_prompt(tok, q)
                base_uncon = generate(tok, model, p_base, args.max_new_tokens, processor=None)
                base_con = generate(tok, model, p_base, args.max_new_tokens, processor=proc)
                base_rec = {
                    "prompt": q,
                    "model": model_id,
                    "unconstrained_output": base_uncon,
                    "constrained_output": base_con,
                }
                mask_writer.writerow(base_rec)
                mask_fh.flush()

                # 2) Instructed file: same unconstrained output as baseline,
                # but constrained output uses Hebrew instruction in prompt.
                p_sys = prompt_with_hebrew_instruction(tok, model_id, q)
                sys_con = generate(tok, model, p_sys, args.max_new_tokens, processor=proc)
                sys_rec = {
                    "prompt": q,
                    "model": model_id,
                    "unconstrained_output": base_uncon,
                    "constrained_output": sys_con,
                }
                sys_writer.writerow(sys_rec)
                sys_fh.flush()

                written += 1
                print(f"  [{model_id.split('/')[-1]}] {q[:40]!r} done")
            # Release references before loading the next model so RAM and GPU
            # memory are reclaimed as aggressively as possible.
            del proc
            del allowed
            del model
            del tok
            gc.collect()
            try:
                import torch

                torch.cuda.empty_cache()
            except RuntimeError:
                pass
    finally:
        mask_fh.close()
        sys_fh.close()

    print(f"\nWrote {MASK_ONLY_CSV} ({written} rows)")
    print(f"Wrote {SYSTEM_PROMPT_CSV} ({written} rows)")


if __name__ == "__main__":
    main()
