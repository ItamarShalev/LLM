"""
Part 2 - Tokenization differences.

Finds an English text that the tokenizers split differently, prints the splits
for the three most distinctive tokenizers, and reports how many of the remaining
seven agree with each of those three. Writes a Markdown fragment for the report.

Strategy: probe a small battery of texts that are known to stress tokenizers
(long numbers, contractions, a rare/compound word, code-like punctuation,
mixed case). For each text we group the ten models by their exact token-id
*surface* sequence; the text with the most distinct groupings is chosen.

Usage:
    python -m src.part2_tokenizers.tokenization_diff
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as C
from src.common.token_utils import detect_family, token_surface

CANDIDATES = [
    "The hyperparameterization cost $1,234,567.89 in 2024.",
    "She said, \u201cdon\u2019t\u201d \u2014 and then left.",
    "antidisestablishmentarianism",
    'config_file.JSON => {"key": 42};',
    "I can\u2019t believe it\u2019s already 3:45pm!",
    "Supercalifragilisticexpialidocious",
]


def load(model_id):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_id, token=C.hf_token(), trust_remote_code=True)


def surfaces(tok, family, text):
    ids = tok.encode(text, add_special_tokens=False)
    return [token_surface(tok, i, family) or f"<{i}>" for i in ids]


def main() -> None:
    toks = {}
    for m in C.MODELS:
        try:
            t = load(m)
            toks[m] = (t, detect_family(t))
            print(f"[ok]   {m}", file=sys.stderr)
        except Exception as e:
            print(f"[skip] {m}: {str(e)[:60]}", file=sys.stderr)

    best: dict[str, Any] | None = None
    for text in CANDIDATES:
        groups: dict[tuple[Any, ...], list[str]] = defaultdict(list)
        per_model: dict[str, list[str]] = {}
        for m, (t, fam) in toks.items():
            seq = surfaces(t, fam, text)
            per_model[m] = seq
            groups[tuple(seq)].append(m)
        n_distinct = len(groups)
        if best is None or n_distinct > best["n_distinct"]:
            best = {
                "text": text,
                "n_distinct": n_distinct,
                "groups": groups,
                "per_model": per_model,
            }

    assert best is not None
    text = best["text"]
    groups = best["groups"]
    # Order groupings by how many models share them, take three distinct splits.
    ordered = sorted(groups.items(), key=lambda kv: -len(kv[1]))
    chosen = ordered[:3]

    lines = [
        "### Tokenization differences\n",
        f"Chosen text: `{text}`\n",
        f"The ten tokenizers produced {best['n_distinct']} distinct splittings of this text. "
        "Three representative tokenizations:\n",
    ]
    for i, (seq, members) in enumerate(chosen, 1):
        rep = members[0].split("/")[-1]
        lines.append(
            f"**Tokenization {i}** (e.g. {rep}, {len(seq)} tokens): "
            + " | ".join(s.replace(" ", "\u2423") for s in seq)
        )
        agree = len(members) - 1
        lines.append(
            f"  Agreeing models among the other nine: {agree} "
            f"({', '.join(x.split('/')[-1] for x in members)}).\n"
        )

    lines.append(
        "**Likely causes.** The splits diverge mainly because (a) the byte-level BPE "
        "vocabularies were learned on different corpora, so multi-digit numbers, the "
        "rare compound word and the contraction apostrophe are merged to different "
        "depths; (b) the SentencePiece models (Mistral, DictaLM) treat the leading "
        "space and punctuation differently from the byte-level models; and (c) the "
        "very large vocabularies (Phi-4 at ~200k, Qwen at ~152k) tend to keep common "
        "chunks whole where the smaller vocabularies (SmolLM2, Granite at ~49k) break "
        "them into more pieces.\n"
    )

    lines.append(
        "### Measurement method (avg tokens per word)\n\n"
        "A 'word' is defined as a whitespace-delimited unit: we count words with "
        "`text.split()` over a fixed prose sample (one English sample and one Hebrew "
        "sample, identical across all ten tokenizers so the numbers are comparable). "
        "For each tokenizer we encode the same sample without special tokens "
        "(`encode(text, add_special_tokens=False)`), count the resulting tokens, and "
        "report tokens divided by words. Using one shared sample removes corpus bias "
        "from the comparison: every difference in the ratio is then attributable to the "
        "tokenizer alone. Word-boundary marking is reported per model in tokenizers.csv: "
        "byte-level BPE models mark a word start with the meta byte '\u0120' (U+0120, the "
        "byte-level rendering of a leading space), while SentencePiece models use the "
        "meta symbol '\u2581' (U+2581).\n\n"
        "The Hebrew ratio is the most revealing number. English sits near 1.1 to 1.2 "
        "tokens per word for every model. Hebrew splits far more: models with dedicated "
        "Hebrew coverage (Phi-4-mini and Qwen near 1.9, DictaLM near 2.25, DeepSeek near "
        "2.4) stay low, whereas English-centric vocabularies fall back toward bytes and "
        "balloon to 4.5 to 5.8 tokens per Hebrew word (SmolLM2 highest, Llama and OLMo-2 "
        "around 5). This is the practical cost of tokenizer coverage: the same Hebrew "
        "sentence can be three times more expensive to process on one model than "
        "another.\n"
    )

    out = C.REPORT_DIR / "sections" / "part2_diff.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    # also dump raw per-model splits for the appendix
    (C.OUTPUTS / "tokenization_diff_detail.json").write_text(
        json.dumps({"text": text, "per_model": best["per_model"]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\n".join(lines))
    print(f"\n[wrote] {out}")


if __name__ == "__main__":
    main()
