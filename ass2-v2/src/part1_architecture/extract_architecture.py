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

from transformers import AutoConfig

# Allow running as a script or a module.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as C


def fetch_config(model_id: str, offline: bool, refresh: bool = False) -> dict[str, Any]:
    """Return the parsed configuration for a model using AutoConfig.
    
    Rely on HF's native caching. Maps gated models to mirrors if access fails.
    """
    tok = C.hf_token()
    repo = C.resolve_repo(model_id)
    
    # Try the main repo first
    try:
        config = AutoConfig.from_pretrained(
            repo,
            local_files_only=offline,
            force_download=refresh,
            token=tok,
            trust_remote_code=True,
        )
        return config.to_dict()
    except Exception as e:
        # Fall back to public mirror if primary failed (likely gated issue or connection)
        mirror = C.PUBLIC_MIRROR.get(model_id)
        if mirror and mirror != repo:
            try:
                config = AutoConfig.from_pretrained(
                    mirror,
                    local_files_only=offline,
                    force_download=refresh,
                    trust_remote_code=True,
                )
                return config.to_dict()
            except Exception:
                pass
        raise e


def approx_params_billions(cfg: dict[str, Any]) -> str | float | None:
    """Rough parameter count from the main config fields, supporting both dense models
    and MoE / MLA architectures (like DeepSeek-V3).

    Counts embeddings, attention (standard Q/K/V/O or MLA low-rank projections),
    and MLP (standard dense SwiGLU or Mixture-of-Experts routing + experts).
    """
    h = cfg.get("hidden_size")
    L = cfg.get("num_hidden_layers")
    V = cfg.get("vocab_size")
    inter = cfg.get("intermediate_size")
    n_heads = cfg.get("num_attention_heads")
    n_kv = cfg.get("num_key_value_heads", n_heads)
    if h is None or L is None or V is None or inter is None or n_heads is None:
        return None

    is_mla = cfg.get("kv_lora_rank") is not None
    is_moe = cfg.get("n_routed_experts") is not None or cfg.get("num_local_experts") is not None

    # 1. Attention layer parameters
    if is_mla:
        kv_lora_rank = cfg.get("kv_lora_rank")
        q_lora_rank = cfg.get("q_lora_rank")
        qk_nope_head_dim = cfg.get("qk_nope_head_dim")
        qk_rope_head_dim = cfg.get("qk_rope_head_dim")
        v_head_dim = cfg.get("v_head_dim")

        if kv_lora_rank and q_lora_rank and qk_nope_head_dim and qk_rope_head_dim and v_head_dim:
            # Query projections: q_a_proj (h -> q_lora_rank) + q_b_proj (q_lora_rank -> n_heads * (qk_nope_head_dim + qk_rope_head_dim))
            q_a = h * q_lora_rank
            q_b = q_lora_rank * n_heads * (qk_nope_head_dim + qk_rope_head_dim)
            q_proj = q_a + q_b

            # KV projections: kv_a_proj_with_mqa (h -> kv_lora_rank + qk_rope_head_dim)
            kv_a = h * (kv_lora_rank + qk_rope_head_dim)
            # kv_b_proj (kv_lora_rank -> n_heads * (qk_nope_head_dim + v_head_dim))
            kv_b = kv_lora_rank * n_heads * (qk_nope_head_dim + v_head_dim)
            kv_proj = kv_a + kv_b

            # O projection: o_proj (n_heads * v_head_dim -> h)
            o_proj = (n_heads * v_head_dim) * h
            attn = q_proj + kv_proj + o_proj
        else:
            # Fallback to standard attention formula if MLA keys are missing
            head_dim = cfg.get("head_dim", h // n_heads)
            q = h * (n_heads * head_dim)
            kv = 2 * h * (n_kv * head_dim)
            o = (n_heads * head_dim) * h
            attn = q + kv + o
    else:
        # Standard GQA / MHA Attention
        head_dim = cfg.get("head_dim", h // n_heads)
        q = h * (n_heads * head_dim)
        kv = 2 * h * (n_kv * head_dim)
        o = (n_heads * head_dim) * h
        attn = q + kv + o

    # 2. MLP layer parameters
    if is_moe:
        n_routed = cfg.get("n_routed_experts") or cfg.get("num_local_experts", 0)
        n_shared = cfg.get("n_shared_experts", 0)
        moe_inter = cfg.get("moe_intermediate_size") or cfg.get("expert_intermediate_size") or inter
        
        first_k_dense = cfg.get("first_k_dense_replace", 0)
        
        # Dense layers (e.g. first 3 layers in DeepSeek-V3 are dense)
        dense_mlp = 3 * h * inter
        
        # MoE layers (total experts)
        # Each expert is a SwiGLU MLP: 3 * h * moe_inter
        expert_params = 3 * h * moe_inter
        moe_mlp_total = (n_routed + n_shared) * expert_params + h * n_routed # expert weights + router weight
        
        # Total MLP parameters across all layers
        mlp_total = (dense_mlp * first_k_dense) + (moe_mlp_total * (L - first_k_dense))
        
        # Active MLP parameters (per token)
        n_active_routed = cfg.get("num_experts_per_tok", 0)
        moe_mlp_active = (n_active_routed + n_shared) * expert_params + h * n_routed
        mlp_active = (dense_mlp * first_k_dense) + (moe_mlp_active * (L - first_k_dense))
    else:
        # Standard dense MLP
        mlp_total = (3 * h * inter) * L
        mlp_active = mlp_total

    # 3. Embeddings
    emb = V * h
    tied = cfg.get("tie_word_embeddings", False)
    emb_total = emb * (1 if tied else 2)

    # 4. Total and Active calculations
    total = emb_total + (attn * L) + mlp_total
    active = emb_total + (attn * L) + mlp_active

    if is_moe:
        return f"{round(total / 1e9, 2)} ({round(active / 1e9, 2)} active)"
    else:
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
    scaling = cfg.get("rope_scaling")
    rope_params = cfg.get("rope_parameters")
    
    # Check rope_parameters if rope_scaling is not set
    if not scaling and isinstance(rope_params, dict):
        if "rope_type" in rope_params or "type" in rope_params:
            scaling = rope_params

    # Dynamically detect partial rotary factor (e.g. Phi models)
    prf = cfg.get("partial_rotary_factor")
    if prf is None and isinstance(rope_params, dict):
        prf = rope_params.get("partial_rotary_factor")

    # Dynamically detect decoupled RoPE used in MLA models (e.g. DeepSeek)
    qk_rope = cfg.get("qk_rope_head_dim")
    if qk_rope is None and isinstance(rope_params, dict):
        qk_rope = rope_params.get("qk_rope_head_dim")

    if prf is not None:
        base = f"RoPE (partial, rotary_factor={prf})"
    elif qk_rope is not None:
        base = f"decoupled RoPE (applied to qk_rope_head_dim={qk_rope})"
    else:
        base = "RoPE"

    # Extract theta
    rope_theta = cfg.get("rope_theta")
    if rope_theta is None and isinstance(rope_params, dict):
        rope_theta = rope_params.get("rope_theta")
    
    # Extract scaling details
    scale_str = "None"
    if scaling:
        stype = scaling.get("rope_type") or scaling.get("type") or "scaled"
        if stype != "default":
            factor = scaling.get("factor")
            if factor is not None:
                scale_str = f"{stype}x{factor}"
            else:
                scale_str = stype

    # Extract max context
    max_pos = cfg.get("max_position_embeddings")

    # Combine into a clean structured format
    details = []
    if rope_theta is not None:
        t_val = int(rope_theta) if float(rope_theta).is_integer() else rope_theta
        details.append(f"theta={t_val}")
    else:
        details.append("theta=NA")
        
    if scale_str != "None":
        details.append(f"scaling={scale_str}")
        
    if max_pos is not None:
        details.append(f"max={max_pos}")

    return f"{base} ({', '.join(details)})"


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
    
    rope_theta = cfg.get("rope_theta")
    if rope_theta is None and "rope_parameters" in cfg:
        rope_theta = cfg["rope_parameters"].get("rope_theta")
        
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
