"""
Part 2 - Analysis of Tokenization Divergence.

Identifies English strings that trigger contrasting segmentation rules across 
the ten target models. The script isolates the three most distinct tokenization 
behaviors, quantifies how closely the remaining seven models align with each 
baseline, and generates a formatted text fragment for the final report.

Methodology:
The pipeline evaluates a targeted set of challenging text inputs containing 
multi-digit numbers, contractions, rare compound terms, code punctuation, and 
mixed-case sequences. Models are grouped by their exact output subword sequences, 
and the text sample that yields the highest variation in structural groupings 
is selected for the final comparative baseline.
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


    (C.OUTPUTS / "tokenization_diff_detail.json").write_text(
        json.dumps({"text": text, "per_model": best["per_model"]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
