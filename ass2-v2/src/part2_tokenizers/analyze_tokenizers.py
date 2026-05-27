"""
Part 2 - Tokenizers.

For each of the ten models, loads the tokenizer and records: type/family, vocab
size, special-token inventory, the word-boundary strategy, whether it is byte
level or byte-fallback, and the average number of tokens per word for English and
Hebrew. Writes outputs/tokenizers.csv.

Average tokens-per-word method (documented in the report):
  * Take the fixed prose samples in data/fixtures.py (one English, one Hebrew).
  * Encode each WITHOUT special tokens.
  * words = number of whitespace-separated chunks (str.split()).
  * avg = total_tokens / words.
  Assumption: a "word" is a whitespace-delimited chunk, so attached punctuation
  counts with its word. This matches how these languages are normally spaced and
  keeps the English and Hebrew counts comparable.

Usage:
    python -m src.part2_tokenizers.analyze_tokenizers
    HF_TOKEN=... python -m src.part2_tokenizers.analyze_tokenizers   # includes Llama
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as C
from data.fixtures import ENGLISH_SAMPLE, HEBREW_SAMPLE
from src.common.token_utils import detect_family


def load_tokenizer(model_id: str):
    from transformers import AutoTokenizer

    repo = C.resolve_repo(model_id)
    try:
        return AutoTokenizer.from_pretrained(repo, token=C.hf_token(), trust_remote_code=True)
    except Exception:
        # Final fallback to a verified public mirror if the official repo is gated.
        mirror = C.PUBLIC_MIRROR.get(model_id)
        if mirror and mirror != repo:
            return AutoTokenizer.from_pretrained(mirror, trust_remote_code=True)
        raise


def word_boundary_strategy(family: str) -> str:
    if family == "byte_level":
        return (
            "byte-level BPE: a leading space is encoded as the meta byte 'Ġ' (U+0120); "
            "tokens with no 'Ġ' continue the previous word, so detokenizing concatenates "
            "byte strings and maps them back to bytes"
        )
    return (
        "SentencePiece: a leading space is the meta symbol '▁' (U+2581); detokenizing "
        "concatenates pieces and replaces '▁' with a space"
    )


def special_tokens_summary(tok) -> str:
    specials = []
    for t in tok.all_special_tokens:
        specials.append(t)
    added = list(tok.get_added_vocab())
    n = len(set(specials) | set(added))
    shown = ", ".join(specials[:6])
    return f"{n} special/added; e.g. {shown}"


def avg_tokens_per_word(tok, text: str) -> float:
    n_tokens = len(tok.encode(text, add_special_tokens=False))
    n_words = len(text.split())
    return round(n_tokens / n_words, 3)


def analyze_one(model_id: str) -> dict:
    tok = load_tokenizer(model_id)
    family = detect_family(tok)
    cls = type(tok).__name__
    byte_strategy = (
        "byte-level (full 256-byte alphabet)"
        if family == "byte_level"
        else "SentencePiece with byte fallback (<0xNN>)"
    )
    return {
        "model_id": model_id,
        "tokenizer_type": f"{family} BPE",
        "vocab_size": len(tok),
        "special_tokens": special_tokens_summary(tok),
        "word_boundary_strategy": word_boundary_strategy(family),
        "byte_fallback_or_byte_level": byte_strategy,
        "avg_tokens_per_english_word": avg_tokens_per_word(tok, ENGLISH_SAMPLE),
        "avg_tokens_per_hebrew_word": avg_tokens_per_word(tok, HEBREW_SAMPLE),
        "tokenizer_backend": cls,
    }


def main() -> None:
    columns = C.TOKENIZER_COLUMNS + C.TOKENIZER_EXTRA_COLUMNS
    rows = []
    for model_id in C.MODELS:
        try:
            rows.append(analyze_one(model_id))
            print(f"[ok]   {model_id}")
        except Exception as e:
            print(f"[warn] {model_id}: {e}", file=sys.stderr)
            rows.append(dict.fromkeys(columns, "NA") | {"model_id": model_id})

    with C.TOKENIZERS_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "NA") for c in columns})
    print(f"\nWrote {C.TOKENIZERS_CSV} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
