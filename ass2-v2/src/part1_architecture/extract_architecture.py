"""
Part 1 - Architectural choices.

Extracts the architectural facts for each of the ten models straight from their
`config.json` on the Hugging Face Hub, derives a few useful quantities
(head_dim, GQA group size, approximate parameter count, MLP-to-hidden ratio),
and writes `outputs/architecture.csv`.

Why config.json: it is the single authoritative, machine-readable source for the
structural hyper-parameters and it is what `transformers` itself reads to build
the model. Facts that are NOT in config.json (norm placement, exact activation
arrangement) are taken from the knowledge tables in config.py, which are cited in
the report.

Usage:
    python -m src.part1_architecture.extract_architecture
    python -m src.part1_architecture.extract_architecture --offline  # use .cache only

Network: fetches https://huggingface.co/<id>/resolve/main/config.json. Gated
models (Llama-3.1) need HF_TOKEN in the environment.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import requests

# Allow running as a script or a module.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as C


def fetch_config(model_id: str, offline: bool, refresh: bool = False) -> dict[str, Any]:
    """Return the parsed config.json for a model, caching it under .cache/."""
    cache_file = C.CACHE_DIR / (model_id.replace("/", "__") + ".config.json")
    if cache_file.exists() and not refresh:
        return json.loads(cache_file.read_text())
    if offline:
        raise FileNotFoundError(f"No cached config for {model_id} and --offline set")

    headers = {}
    tok = C.hf_token()
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    repo = C.resolve_repo(model_id)
    url = f"https://huggingface.co/{repo}/resolve/main/config.json"
    r = requests.get(url, headers=headers, timeout=60)
    if r.status_code == 401:
        # Gated and no token: fall back to a verified public mirror if available.
        mirror = C.PUBLIC_MIRROR.get(model_id)
        if mirror and mirror != repo:
            r = requests.get(
                f"https://huggingface.co/{mirror}/resolve/main/config.json", timeout=60
            )
        if r.status_code == 401:
            raise PermissionError(
                f"{model_id} is gated. Set HF_TOKEN and accept the license on the Hub."
            )
    r.raise_for_status()
    cache_file.write_text(r.text)
    return r.json()


def approx_params_billions(cfg: dict[str, Any]) -> float | None:
    """Rough parameter count from the main config fields (dense models only).

    Counts embeddings, per-layer attention (q/k/v/o) and a SwiGLU gated MLP
    (gate + up + down), plus final norm and the unembedding when untied. This is
    an estimate and is reported as such; it is intentionally skipped for MoE.
    """
    h = cfg.get("hidden_size")
    L = cfg.get("num_hidden_layers")
    V = cfg.get("vocab_size")
    inter = cfg.get("intermediate_size")
    n_heads = cfg.get("num_attention_heads")
    n_kv = cfg.get("num_key_value_heads", n_heads)
    if h is None or L is None or V is None or inter is None or n_heads is None:
        return None
    if cfg.get("model_type") == "deepseek_v3":
        return None  # MoE: not captured by the dense formula
    head_dim = cfg.get("head_dim", h // n_heads)
    q = h * (n_heads * head_dim)
    kv = 2 * h * (n_kv * head_dim)
    o = (n_heads * head_dim) * h
    attn = q + kv + o
    mlp = 3 * h * inter  # gate, up, down for SwiGLU
    per_layer = attn + mlp
    emb = V * h
    tied = cfg.get("tie_word_embeddings", False)
    total = emb * (1 if tied else 2) + per_layer * L
    return round(total / 1e9, 2)


def moe_details(cfg: dict[str, Any]) -> str:
    if cfg.get("model_type") != "deepseek_v3" and "n_routed_experts" not in cfg:
        return "NA"
    parts = [
        f"routed_experts={cfg.get('n_routed_experts')}",
        f"shared_experts={cfg.get('n_shared_experts')}",
        f"experts_per_token={cfg.get('num_experts_per_tok')}",
        f"expert_intermediate={cfg.get('moe_intermediate_size')}",
        f"first_k_dense={cfg.get('first_k_dense_replace')}",
        f"scoring={cfg.get('scoring_func')}",
        "attention=MLA("
        f"kv_lora_rank={cfg.get('kv_lora_rank')},"
        f"q_lora_rank={cfg.get('q_lora_rank')},"
        f"qk_rope={cfg.get('qk_rope_head_dim')},"
        f"qk_nope={cfg.get('qk_nope_head_dim')},"
        f"v_head={cfg.get('v_head_dim')})",
    ]
    return "; ".join(p for p in parts if "None" not in p)


def position_encoding(cfg: dict[str, Any]) -> str:
    mt = cfg.get("model_type", "")
    scaling = cfg.get("rope_scaling")
    base = "RoPE"
    if mt == "phi3":
        prf = cfg.get("partial_rotary_factor")
        base = f"RoPE (partial, rotary_factor={prf})"
    if mt == "deepseek_v3":
        base = "decoupled RoPE (applied to qk_rope_head_dim only)"
    if scaling:
        stype = scaling.get("type") or scaling.get("rope_type") or "scaled"
        base += f" + {stype} scaling"
    return base


def extract_row(model_id: str, cfg: dict[str, Any], source: str) -> dict[str, Any]:
    h = cfg.get("hidden_size")
    n_heads = cfg.get("num_attention_heads")
    n_kv = cfg.get("num_key_value_heads", n_heads)
    head_dim = cfg.get("head_dim", (h // n_heads) if (h and n_heads) else None)
    if cfg.get("model_type") == "deepseek_v3":
        # MLA does not have a single head_dim; report the qk/v decomposition.
        qk = (cfg.get("qk_nope_head_dim", 0) or 0) + (cfg.get("qk_rope_head_dim", 0) or 0)
        head_dim = f"MLA qk={qk}, v={cfg.get('v_head_dim')}"
    act = cfg.get("hidden_act", "NA")
    activation = f"{act} (SwiGLU gated MLP)" if act in ("silu", "swish") else act
    norm = C.NORM_PLACEMENT.get(cfg.get("model_type", ""), "RMSNorm (placement: see report)")
    return {
        "model_id": model_id,
        "hidden_size": h,
        "num_layers": cfg.get("num_hidden_layers"),
        "num_attention_heads": n_heads,
        "num_kv_heads": n_kv,
        "mlp_size": cfg.get("intermediate_size"),
        "activation": activation,
        "norm_type": norm,
        "position_encoding": position_encoding(cfg),
        "context_length": cfg.get("max_position_embeddings"),
        "vocab_size": cfg.get("vocab_size"),
        "moe_details": moe_details(cfg),
        # extras
        "head_dim": head_dim,
        "kv_group_size": (n_heads // n_kv) if (n_heads and n_kv) else None,
        "rope_theta": cfg.get("rope_theta", "NA"),
        "tie_word_embeddings": cfg.get("tie_word_embeddings", "NA"),
        "params_billions_approx": approx_params_billions(cfg) or "NA",
        "source": source,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="use only cached configs")
    ap.add_argument("--refresh", action="store_true", help="force refetch from the Hub")
    args = ap.parse_args()

    rows = []
    for model_id in C.MODELS:
        try:
            cfg = fetch_config(model_id, args.offline, args.refresh)
            src = "config.json (HF Hub)"
            rows.append(extract_row(model_id, cfg, src))
            print(f"[ok]   {model_id}")
        except (PermissionError, FileNotFoundError) as e:
            print(f"[warn] {model_id}: {e}", file=sys.stderr)
            # leave a placeholder row so the gap is explicit in the CSV
            rows.append(
                dict.fromkeys(C.ARCHITECTURE_COLUMNS + C.ARCHITECTURE_EXTRA_COLUMNS, "NA")
                | {"model_id": model_id, "source": f"MISSING: {e}"}
            )

    cols = C.ARCHITECTURE_COLUMNS + C.ARCHITECTURE_EXTRA_COLUMNS
    with C.ARCHITECTURE_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow({c: row.get(c, "NA") for c in cols})
    print(f"\nWrote {C.ARCHITECTURE_CSV} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
