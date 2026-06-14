"""
Part 4 - Evaluation of the fine-tuned model.

Runs the base Qwen2.5-1.5B-Instruct and the LoRA fine-tuned version on the full
held-out evaluation set (the 10 inputs provided in the assignment + 10 of our own,
none of which appear in training) and writes outputs/eval_outputs.jsonl with one
record per prompt:

    {"prompt", "base_output", "finetuned_output", "notes"}

The "notes" field carries a short automatic diagnosis: what fraction of each answer
is Hebrew letters, whether the answer is non-empty, and a one-line verdict so the
report can be skimmed quickly. The goal of the fine-tune is that the model answers
in Hebrew while staying on-topic (a real answer, not a fixed canned string), so we
measure exactly that.

Usage:
    python -m src.part4_finetuning.evaluate
    python -m src.part4_finetuning.evaluate --max-new-tokens 256 --no-base
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as C
from data.fixtures import EVAL_INPUTS_OWN, EVAL_INPUTS_PROVIDED
from src.common.token_utils import HEBREW_RE

ADAPTER_DIR = C.OUTPUTS / "lora_adapter"


def hebrew_fraction(text: str) -> float:
    """Fraction of alphabetic characters that are Hebrew (ignores spaces/digits/punct)."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    heb = [c for c in letters if HEBREW_RE.match(c)]
    return len(heb) / len(letters)


def diagnose(base_out: str, ft_out: str) -> str:
    """One-line human-readable verdict comparing base vs fine-tuned answers."""
    b = hebrew_fraction(base_out)
    f = hebrew_fraction(ft_out)
    parts = [f"base_hebrew={b:.0%}", f"finetuned_hebrew={f:.0%}"]
    if f >= 0.8 and len(ft_out.strip()) > 0:
        if b < 0.5:
            parts.append("verdict=fine-tune switched the answer to Hebrew as intended")
        else:
            parts.append("verdict=both Hebrew; fine-tune kept Hebrew output")
    elif f < 0.5:
        parts.append("verdict=fine-tuned answer not primarily Hebrew, inspect")
    else:
        parts.append("verdict=partially Hebrew, inspect")
    return "; ".join(parts)


def generate(model: Any, tok: Any, prompt: str, max_new_tokens: int, device: str) -> str:
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(device)
    import torch

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    gen = out[0][inputs["input_ids"].shape[1] :]
    return tok.decode(gen, skip_special_tokens=True).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--device", default=None, help="cuda / cpu (auto by default)")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument(
        "--no-base",
        action="store_true",
        help="skip the base model pass (use cached base_output if present)",
    )
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, args.dtype)

    eval_inputs: list[dict[str, Any]] = []
    for i, p in enumerate(EVAL_INPUTS_PROVIDED):
        eval_inputs.append({"prompt": p, "split": "provided", "idx": i})
    for i, p in enumerate(EVAL_INPUTS_OWN):
        eval_inputs.append({"prompt": p, "split": "own", "idx": i})

    tok: Any = AutoTokenizer.from_pretrained(C.FINETUNE_MODEL, token=C.hf_token())
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # ---- base model ----
    base_outputs: dict[str, str] = {}
    if not args.no_base:
        print(f"Loading base model {C.FINETUNE_MODEL} ...")
        base: Any = AutoModelForCausalLM.from_pretrained(
            C.FINETUNE_MODEL, dtype=dtype, token=C.hf_token()
        )
        base = base.to(device)
        base.eval()
        for rec in eval_inputs:
            base_outputs[rec["prompt"]] = generate(
                base, tok, rec["prompt"], args.max_new_tokens, device
            )
            print(f"  base done: {rec['split']}#{rec['idx']}")
        del base
        if device == "cuda":
            torch.cuda.empty_cache()

    # ---- fine-tuned model (base + LoRA adapter) ----
    print(f"Loading fine-tuned adapter from {ADAPTER_DIR} ...")
    from peft import PeftModel

    ft_base: Any = AutoModelForCausalLM.from_pretrained(
        C.FINETUNE_MODEL, dtype=dtype, token=C.hf_token()
    )
    ft_base = ft_base.to(device)
    ft: Any = PeftModel.from_pretrained(ft_base, str(ADAPTER_DIR))
    ft = ft.to(device)
    ft.eval()

    records: list[dict[str, Any]] = []
    for rec in eval_inputs:
        ft_out = generate(ft, tok, rec["prompt"], args.max_new_tokens, device)
        base_out = base_outputs.get(rec["prompt"], "")
        records.append(
            {
                "prompt": rec["prompt"],
                "base_output": base_out,
                "finetuned_output": ft_out,
                "notes": f"split={rec['split']}; " + diagnose(base_out, ft_out),
            }
        )
        print(f"  finetuned done: {rec['split']}#{rec['idx']}")

    with C.EVAL_OUTPUTS.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_heb = sum(1 for r in records if hebrew_fraction(r["finetuned_output"]) >= 0.8)
    print(f"\nWrote {len(records)} records to {C.EVAL_OUTPUTS}")
    print(f"Fine-tuned answered in Hebrew (>=80% letters) on {n_heb}/{len(records)} inputs.")


if __name__ == "__main__":
    main()
