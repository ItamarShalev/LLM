"""
Part 3 - Run constrained decoding.

Loads Qwen2.5-7B-Instruct and Mistral-7B-Instruct-v0.3, and for each of the ten
English queries generates: (a) an unconstrained answer and (b) a constrained
answer that may only use the Hebrew-allowed token set. Writes one JSON object per
line to outputs/decoding_outputs.jsonl with: prompt, model, unconstrained_output,
constrained_output.

The script is incremental and resumable: each completed (model, prompt) is
appended to the JSONL immediately, and re-running skips pairs that are already
present. A long Part 3 can therefore be interrupted and resumed without losing
progress.

This script needs a GPU (or a patient CPU) and the model weights, so it is meant
to run on Claude Code's machine. Everything it depends on (the allowed-token JSON
files, the queries) is already produced by the lighter scripts.

Usage:
    python -m src.part3_decoding.run_decoding
    python -m src.part3_decoding.run_decoding --max-new-tokens 80 --device cuda
    python -m src.part3_decoding.run_decoding --fresh   # ignore existing output
"""

from __future__ import annotations

import argparse
import json
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


def load_model(model_id, device, dtype):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok: Any = AutoTokenizer.from_pretrained(model_id, token=C.hf_token())
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    resolved_dtype = getattr(torch, dtype) if dtype != "auto" else "auto"
    # On CUDA we let accelerate place layers automatically. Forcing
    # device_map=cuda for a 7B bf16 model on a 16 GB card OOMs/segfaults;
    # "auto" plus a generous max_memory keeps as many layers on GPU as fit.
    kwargs: dict[str, Any] = {
        "token": C.hf_token(),
        "dtype": resolved_dtype,
    }
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


def chat_prompt(tok, query):
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


def _read_existing() -> list[dict[str, Any]]:
    if not C.DECODING_OUTPUTS.exists():
        return []
    out: list[dict[str, Any]] = []
    for raw in C.DECODING_OUTPUTS.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--device", default="cuda")
    # Loading bf16 safetensors weights segfaults on torch 2.12 + Blackwell laptop
    # GPUs (RTX 50-series); float16 has the same memory footprint and loads
    # cleanly. Override with --dtype bfloat16 on hardware that supports it.
    ap.add_argument("--dtype", default="float16")
    ap.add_argument(
        "--fresh",
        action="store_true",
        help="ignore any existing decoding_outputs.jsonl and start from scratch",
    )
    args = ap.parse_args()

    existing = [] if args.fresh else _read_existing()
    done_keys = {(r.get("model", ""), r.get("prompt", "")) for r in existing}
    if existing:
        print(f"[resume] found {len(existing)} existing records; skipping those pairs")

    # Open in append mode (or overwrite if --fresh) so each completed pair is
    # persisted immediately. Crashes mid-run no longer wipe earlier work.
    mode = "w" if args.fresh else "a"
    out_fh = C.DECODING_OUTPUTS.open(mode, encoding="utf-8")
    if args.fresh:
        # truncate any old file content
        out_fh.truncate(0)

    written = 0
    try:
        for model_id, allowed_file in ALLOWED_FILES.items():
            pending = [q for q in DECODING_QUERIES if (model_id, q) not in done_keys]
            if not pending:
                print(f"[skip-load] {model_id}: all {len(DECODING_QUERIES)} prompts already done")
                continue
            print(f"[load] {model_id}  ({len(pending)} prompts to do)")
            tok, model = load_model(model_id, args.device, args.dtype)
            allowed = load_allowed_ids(allowed_file)
            proc = build_processor(tok, allowed)
            for q in pending:
                p = chat_prompt(tok, q)
                uncon = generate(tok, model, p, args.max_new_tokens, processor=None)
                con = generate(tok, model, p, args.max_new_tokens, processor=proc)
                rec = {
                    "prompt": q,
                    "model": model_id,
                    "unconstrained_output": uncon,
                    "constrained_output": con,
                }
                out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_fh.flush()
                written += 1
                print(f"  [{model_id.split('/')[-1]}] {q[:40]!r} done")
            del model
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass
    finally:
        out_fh.close()

    total = len(existing) + written
    print(f"\nWrote {C.DECODING_OUTPUTS} ({total} lines; {written} new this run)")


if __name__ == "__main__":
    main()
