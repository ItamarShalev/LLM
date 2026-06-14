"""
Part 3 - Hebrew Token Isolation for Constrained Decoding.

Filters the full vocabulary matrices of the primary decoding models (Qwen2.5 
and Mistral) to isolate a subset of tokens eligible for Hebrew generation. 
The script tracks allowed subwords and exports indices to target JSON manifests.

Allowed tokens are identified using the following criteria:
1. Subwords containing Hebrew characters with no mixed foreign scripts.
2. Language-neutral punctuation, structural whitespace, and numeric digits.
3. Raw byte-fragment tokens that map to the Hebrew UTF-8 character footprint, 
   ensuring that multi-token split characters can be safely reconstructed.

Note on Sequence Termination:
The End-of-Sequence (EOS) control token is excluded from these baseline manifests 
to maintain a pure language footprint. The constrained decoding framework adds 
the EOS token dynamically at runtime to allow generation to safely terminate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as C
from src.common.token_utils import detect_family, is_hebrew_participating

TARGETS = {
    "Qwen/Qwen2.5-7B-Instruct": C.HEBREW_TOKENS_QWEN,
    "mistralai/Mistral-7B-Instruct-v0.3": C.HEBREW_TOKENS_MISTRAL,
}


def build_allowed(model_id: str) -> list[int]:
    from transformers import AutoTokenizer

    tok: Any = AutoTokenizer.from_pretrained(model_id, token=C.hf_token())
    family = detect_family(tok)
    n = len(tok)
    allowed = [i for i in range(n) if is_hebrew_participating(tok, i, family)]
    print(
        f"  {model_id}: family={family} vocab={n} allowed={len(allowed)} "
        f"({100 * len(allowed) / n:.1f}%)"
    )
    return allowed


def main() -> None:
    for model_id, out_path in TARGETS.items():
        allowed = build_allowed(model_id)
        out_path.write_text(
            json.dumps({"model_id": model_id, "allowed_token_ids": allowed}, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[wrote] {out_path}")


if __name__ == "__main__":
    main()
