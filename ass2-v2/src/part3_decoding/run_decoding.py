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


Usage:
    python -m src.part3_decoding.run_decoding
    python -m src.part3_decoding.run_decoding --max-new-tokens 80 --device cuda
"""

from __future__ import annotations

import argparse
import csv
import subprocess
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
    # `dtype` is the current Hugging Face argument (replaces the deprecated
    # `torch_dtype`). Keep the caller's requested dtype so we do not inflate
    # memory usage on smaller machines.
    resolved_dtype = torch.float16 if dtype == "auto" else getattr(torch, dtype)

    # Load to CPU first, WITHOUT accelerate's device_map offload. On RTX
    # 50-series / Blackwell GPUs (torch 2.12, transformers 5.x) letting
    # accelerate split a 7B model across GPU+CPU via device_map="auto" +
    # max_memory SEGFAULTS during weight materialization - reproducibly and
    # flakily, depending on exactly how much VRAM happens to be free. We avoid
    # that path entirely: load into CPU RAM, then move the WHOLE model onto the
    # GPU only if it fully fits (with margin). Otherwise we run on CPU, which is
    # slower but never crashes.
    kwargs: dict[str, Any] = {
        "token": C.hf_token(),
        "dtype": resolved_dtype,
        "low_cpu_mem_usage": True,
    }
    # Mistral v0.3 has been more sensitive to fused attention paths on some
    # systems, so keep it on the safe eager attention implementation.
    if "mistral" in model_id.lower():
        kwargs["attn_implementation"] = "eager"
    model: Any = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()

    if device == "cuda" and torch.cuda.is_available():
        free_bytes = torch.cuda.mem_get_info()[0]
        weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        margin = int(2.5 * 1024**3)  # leave room for activations + KV cache
        short = model_id.split("/")[-1]
        if free_bytes > weight_bytes + margin:
            # Whole model fits on the GPU -> move it in one shot (no offload).
            model = model.to("cuda")
            print(f"  [info] {short}: loaded fully on GPU ({weight_bytes / 1024**3:.1f} GB).")
        else:
            print(
                f"  [info] {short} (~{weight_bytes / 1024**3:.1f} GB fp16) does not fit in "
                f"{free_bytes / 1024**3:.1f} GB free VRAM; running on CPU "
                "(slower but stable - avoids the offload segfault)."
            )
    elif device != "cuda":
        model = model.to(device)
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


def decode_one_model(model_id, allowed_file, device, dtype, max_new_tokens) -> None:
    """Load ONE model and APPEND its decoded rows to both CSV files.

    Runs in a dedicated worker process (see ``main``): loading two 7B models
    back-to-back in a single process can exhaust system RAM and crash, so each
    model gets a clean process.
    """
    print(f"[load] {model_id}  ({len(DECODING_QUERIES)} prompts to do)")
    tok, model = load_model(model_id, device, dtype)
    allowed = load_allowed_ids(allowed_file)
    proc = build_processor(tok, allowed)

    mask_fh = MASK_ONLY_CSV.open("a", encoding="utf-8", newline="")
    sys_fh = SYSTEM_PROMPT_CSV.open("a", encoding="utf-8", newline="")
    mask_writer = csv.DictWriter(mask_fh, fieldnames=CSV_FIELDS)
    sys_writer = csv.DictWriter(sys_fh, fieldnames=CSV_FIELDS)
    try:
        for q in DECODING_QUERIES:
            # 1) Baseline file: mask-only flow (no extra Hebrew instruction).
            p_base = chat_prompt(tok, q)
            base_uncon = generate(tok, model, p_base, max_new_tokens, processor=None)
            base_con = generate(tok, model, p_base, max_new_tokens, processor=proc)
            mask_writer.writerow(
                {
                    "prompt": q,
                    "model": model_id,
                    "unconstrained_output": base_uncon,
                    "constrained_output": base_con,
                }
            )
            mask_fh.flush()

            # 2) Instructed file: same unconstrained output as baseline, but the
            # constrained output uses the Hebrew instruction in the prompt.
            p_sys = prompt_with_hebrew_instruction(tok, model_id, q)
            sys_con = generate(tok, model, p_sys, max_new_tokens, processor=proc)
            sys_writer.writerow(
                {
                    "prompt": q,
                    "model": model_id,
                    "unconstrained_output": base_uncon,
                    "constrained_output": sys_con,
                }
            )
            sys_fh.flush()
            print(f"  [{model_id.split('/')[-1]}] {q[:40]!r} done")
    finally:
        mask_fh.close()
        sys_fh.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--device", default="cuda")
    # Loading bf16 safetensors weights segfaults on torch 2.12 + Blackwell laptop
    # GPUs (RTX 50-series); float16 has the same memory footprint and loads
    # cleanly. Override with --dtype bfloat16 on hardware that supports it.
    ap.add_argument("--dtype", default="float16")
    # Internal: process exactly one model and append to the CSVs. Without it the
    # default run orchestrates one worker subprocess per model.
    ap.add_argument("--single-model", default=None, choices=list(ALLOWED_FILES))
    args = ap.parse_args()

    # Worker mode: do exactly one model in this (clean) process and exit.
    if args.single_model:
        decode_one_model(
            args.single_model,
            ALLOWED_FILES[args.single_model],
            args.device,
            args.dtype,
            args.max_new_tokens,
        )
        return

    # Orchestrator mode: (re)create the CSVs with their headers, then run each
    # model in its OWN process so every 7B load starts from a clean memory
    # state. A worker that dies (e.g. native OOM on a RAM-starved box) is
    # contained - we record it and carry on with the next model.
    for csv_path in (MASK_ONLY_CSV, SYSTEM_PROMPT_CSV):
        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            csv.DictWriter(fh, fieldnames=CSV_FIELDS).writeheader()

    repo_root = Path(__file__).resolve().parents[2]
    failures: list[tuple[str, int]] = []
    for model_id in ALLOWED_FILES:
        cmd = [
            sys.executable,
            "-m",
            "src.part3_decoding.run_decoding",
            "--single-model",
            model_id,
            "--device",
            args.device,
            "--dtype",
            args.dtype,
            "--max-new-tokens",
            str(args.max_new_tokens),
        ]
        result = subprocess.run(cmd, cwd=str(repo_root), check=False)
        if result.returncode != 0:
            failures.append((model_id, result.returncode))
            print(f"  [warn] worker for {model_id} exited with code {result.returncode}")

    print(f"\nWrote {MASK_ONLY_CSV}")
    print(f"Wrote {SYSTEM_PROMPT_CSV}")
    if failures:
        print(f"[warn] {len(failures)} model(s) failed: {failures}")


if __name__ == "__main__":
    main()
