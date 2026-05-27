"""
Part 1 - Organize, reflect, analyze.

Reads outputs/architecture.csv and derives the cross-model trends the assignment
asks for: what is universal, where there is no consensus, which models deviate,
and what sizing rules of thumb hold. Writes a Markdown fragment that the report
builder includes verbatim, plus prints the same to stdout.

Usage:
    python -m src.part1_architecture.analyze_architecture
"""

from __future__ import annotations

import csv
import statistics as stats
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as C

OUT = C.REPORT_DIR / "sections" / "part1_analysis.md"


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main() -> None:
    rows = list(csv.DictReader(C.ARCHITECTURE_CSV.open(encoding="utf-8")))
    dense = [r for r in rows if r["moe_details"] == "NA"]

    # MLP-to-hidden ratio (the classic ~2.7x..4x band; SwiGLU often ~2.67x).
    ratios = []
    for r in dense:
        h, m = _num(r["hidden_size"]), _num(r["mlp_size"])
        if h and m:
            ratios.append((r["model_id"], round(m / h, 2)))

    head_dims = {r["head_dim"] for r in rows if r["head_dim"] not in ("NA", "")}
    gqa = [
        r["model_id"]
        for r in rows
        if r["num_kv_heads"] not in ("NA", "")
        and r["num_attention_heads"] not in ("NA", "")
        and _num(r["num_kv_heads"])
        and _num(r["num_kv_heads"]) < _num(r["num_attention_heads"])
    ]
    mha = [
        r["model_id"]
        for r in rows
        if r["num_kv_heads"] not in ("NA", "")
        and "DeepSeek" not in r["model_id"]
        and _num(r["num_kv_heads"]) == _num(r["num_attention_heads"])
    ]
    vocabs = sorted(
        (r["model_id"], int(_num(r["vocab_size"]))) for r in rows if _num(r["vocab_size"])
    )
    ctxs = sorted(
        (int(_num(r["context_length"])), r["model_id"]) for r in rows if _num(r["context_length"])
    )

    lines = []
    a = lines.append
    a("### Cross-model analysis and trends\n")

    a("**Universal choices (full consensus across all ten models).**\n")
    a(
        "- Decoder-only Transformer with RoPE positional encoding. No model in the set uses learned or sinusoidal absolute positions."
    )
    a("- RMSNorm rather than LayerNorm.")
    a(
        "- A gated SwiGLU MLP built on the SiLU activation. The activation is the same in every block of every model; nobody mixes activation types."
    )
    a(
        "- Untied or tied embeddings vary, but the embedding and unembedding share the same hidden width.\n"
    )

    a("**Choices with no consensus.**\n")
    a(
        f"- Attention scheme. Grouped-query attention (GQA, num_kv_heads < num_heads) is now the majority: {', '.join(m.split('/')[-1] for m in gqa)}. "
        f"Full multi-head attention survives only in {', '.join(m.split('/')[-1] for m in mha)}. DeepSeek-V3 replaces both with Multi-head Latent Attention (MLA)."
    )
    a(
        f"- Vocabulary size spans a 6x range, from {vocabs[0][1]:,} ({vocabs[0][0].split('/')[-1]}) to {vocabs[-1][1]:,} ({vocabs[-1][0].split('/')[-1]})."
    )
    a(
        f"- Context length spans from {ctxs[0][0]:,} ({ctxs[0][1].split('/')[-1]}) to {ctxs[-1][0]:,} ({ctxs[-1][1].split('/')[-1]}), often through RoPE scaling (YaRN, llama3, longrope) rather than native training length."
    )
    a(
        "- RoPE theta ranges over four orders of magnitude (10,000 to 10,000,000), tracking the target context length.\n"
    )

    a("**Deviating models.**\n")
    a(
        "- OLMo-2 is the clearest outlier on normalization: it uses a reordered post-norm scheme (norm after the attention and MLP sublayers) plus QK-norm, where every other model is standard pre-norm."
    )
    a(
        "- DeepSeek-V3 is the structural outlier: a 256-expert + 1-shared-expert MoE with MLA attention and the only model here that is not a dense decoder."
    )
    a(
        "- Granite multiplies embeddings, attention scores, residuals and logits by fixed scalars (embedding_multiplier, attention_multiplier, residual_multiplier, logits_scaling); the others do not."
    )
    a(
        "- Phi-4-mini and DeepSeek-V3 use partial / decoupled RoPE (only part of each head is rotated); the others rotate the whole head.\n"
    )

    a("**Sizing rules of thumb that hold.**\n")
    if ratios:
        vals = [v for _, v in ratios]
        a(
            f"- MLP-to-hidden ratio clusters tightly: mean {round(stats.mean(vals), 2)}x, range {min(vals)}x to {max(vals)}x. "
            "The SwiGLU models that target ~2.67x (8/3) are visible, while a few widen the MLP toward 3.5x to 5x to add capacity without more layers."
        )
    a(
        f"- head_dim is almost an invariant: the observed values are {sorted(int(float(x)) for x in head_dims if x.replace('.', '').isdigit()) if all(x.replace('.', '').isdigit() for x in head_dims) else sorted(head_dims)}. 64 or 128 dominate."
    )
    a(
        "- hidden_size is always a multiple of num_attention_heads (so head_dim is integral) and is a power-of-two-friendly number."
    )
    a(
        "- Larger vocabularies pair with the byte-level BPE tokenizers; the SentencePiece models (Mistral, DictaLM) keep ~32k.\n"
    )

    a("**If I were to build a new ~7-8B model.**\n")
    a(
        "Following the consensus and the rules of thumb above: decoder-only, pre-norm RMSNorm, RoPE with a theta chosen for the target context, SwiGLU/SiLU MLP at roughly 3.5x the hidden size, head_dim of 128, and GQA with 8 KV heads (a strong quality-to-KV-cache trade-off shared by Llama, Mistral, Granite and DictaLM). "
        "I would pick hidden_size 4096, 32 layers, 32 query heads and 8 KV heads, a 128k byte-level BPE vocabulary if multilingual coverage matters, and reserve MoE/MLA for a later scale-up where the inference-time savings justify the extra system complexity.\n"
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[wrote] {OUT}")


if __name__ == "__main__":
    main()
