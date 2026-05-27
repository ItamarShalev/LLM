"""
Part 3 - Identify Hebrew-related tokens.

For each of the two decoding models (Qwen2.5-7B-Instruct, Mistral-7B-Instruct-v0.3)
we scan the entire vocabulary and keep every token that may participate in Hebrew
text: tokens whose surface contains a Hebrew character (and no foreign-script
letter), pure punctuation / digit / whitespace tokens, and byte-fragment tokens
that fall inside the Hebrew UTF-8 footprint (so Hebrew letters that are split
across tokens can still be reconstructed). See src/common/token_utils.py for the
exact rule; the report explains the strategy and its assumptions.

Writes:
    outputs/hebrew_allowed_tokens_qwen.json
    outputs/hebrew_allowed_tokens_mistral.json

Each file:  {"model_id": "...", "allowed_token_ids": [ ... ]}

Note: the EOS token is intentionally NOT in these files (the files are the pure
Hebrew-allowed set, as the assignment specifies). The constrained decoder adds
EOS at run time so generation can terminate; this is documented in the report.

Usage:
    python -m src.part3_decoding.identify_hebrew_tokens
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
