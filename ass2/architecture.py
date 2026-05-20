from numba.core.typing.builtins import Iter
import logging
from collections.abc import Iterable
from typing import Mapping, Any
import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi
from transformers import AutoConfig, AutoModel
from dataclasses import dataclass

@dataclass(frozen=True)
class ModelSpec:
    """Immutable description of one Hugging Face model used in the assignment."""

    model_id: str
    gated: bool = False
    trust_remote_code: bool = False

    @property
    def short_name(self) -> str:
        """The repo name without the owner prefix, e.g. ``Qwen2.5-7B-Instruct``."""
        return self.model_id.split("/")[-1]


MODELS: tuple[ModelSpec, ...] = (
    ModelSpec("meta-llama/Llama-3.1-8B-Instruct", gated=True),
    ModelSpec("mistralai/Mistral-7B-Instruct-v0.3"),
    ModelSpec("Qwen/Qwen2.5-7B-Instruct"),
    ModelSpec("allenai/OLMo-2-1124-7B-Instruct"),
    ModelSpec("ibm-granite/granite-3.3-8b-instruct"),
    ModelSpec("deepseek-ai/DeepSeek-V3", trust_remote_code=True),
    ModelSpec("HuggingFaceTB/SmolLM2-1.7B-Instruct"),
    ModelSpec("microsoft/Phi-4-mini-instruct"),
    ModelSpec("tiiuae/Falcon3-7B-Instruct"),
    ModelSpec("dicta-il/dictalm2.0-instruct"),
)


root: Path = Path(__file__).parent.parent
current_ass_folder = root / "ass2"
data_folder = current_ass_folder / "data"

load_dotenv(root / ".env")
data_folder.mkdir(exist_ok=True)


architecture_raw_file = data_folder / "architecture_raw.txt"
architecture_txt_file = data_folder / "architecture.txt"
architecture_csv_file = data_folder / "architecture.csv"

hf_token = os.getenv("HF_TOKEN")

#: Manual notes for facts that cannot be read from the config alone.
#: Verify against the paper/modelling code before submitting.
_NORM_NOTES: dict[str, str] = {
    "allenai/OLMo-2-1124-7B-Instruct": "RMSNorm (reordered/post-norm + QK-norm)",
}

@dataclass(frozen=True)
class ArchitectureRow:
    """One row of ``architecture.csv`` (field order is the CSV column order)."""

    model_id: str
    hidden_size: str
    head_dim: str            # ממד כל attention head
    num_layers: str
    num_attention_heads: str
    num_kv_heads: str
    mlp_size: str
    activation: str
    norm_type: str
    position_encoding: str
    context_length: str
    vocab_size: str
    moe_details: str

    def __dict__(self):
        return {field: getattr(self, field) for field in self.__dataclass_fields__}

    def __str__(self):
        return ", ".join(f"{field}={getattr(self, field)}" for field in self.__dataclass_fields__)


def _first(cfg: Mapping[str, Any], *keys: str, default: Any = "NA") -> Any:
    """Return the first present, non-null value among ``keys``, else ``default``."""
    for key in keys:
        value = cfg.get(key)
        if value is not None:
            return value
    return default


def _norm_type(model_id: str, cfg: Mapping[str, Any]) -> str:
    if model_id in _NORM_NOTES:
        return _NORM_NOTES[model_id]
    if "rms_norm_eps" in cfg or "rms_norm_epsilon" in cfg:
        return "RMSNorm (pre-norm)"
    if "layer_norm_epsilon" in cfg or "layer_norm_eps" in cfg:
        return "LayerNorm (pre-norm)"
    return "RMSNorm (pre-norm)?"

def _position_encoding(cfg: Mapping[str, Any]) -> str:
    """Describe the positional encoding (RoPE), supporting transformers 4.x and 5.x.

    transformers 5.x nests RoPE parameters under ``rope_parameters``; 4.x uses the
    flat keys ``rope_theta`` / ``rope_scaling``.
    """
    raw_params = cfg.get("rope_parameters")
    params: Any = raw_params if isinstance(raw_params, dict) else {}
    raw_scaling = cfg.get("rope_scaling")
    scaling: Any = raw_scaling if isinstance(raw_scaling, dict) else {}

    theta = params.get("rope_theta", cfg.get("rope_theta"))
    rope_type = (
        params.get("rope_type")
        or params.get("type")
        or scaling.get("rope_type")
        or scaling.get("type")
    )
    factor = params.get("factor") or scaling.get("factor")
    max_pos = _first(cfg, "max_position_embeddings", "max_seq_len", default=None)
    partial = cfg.get("partial_rotary_factor")

    if theta is None and not rope_type and not scaling:
        return "NA"

    parts: list[str] = []
    if theta is not None:
        if isinstance(theta, float) and theta.is_integer():
            theta = int(theta)
        parts.append(f"theta={theta}")
    if partial not in (None, 1.0):
        parts.append(f"partial_rotary={partial}")
    if rope_type and rope_type != "default":
        parts.append(f"scaling={rope_type}" + (f"x{factor}" if factor else ""))
    if max_pos is not None:
        parts.append(f"max={max_pos}")
    return f"RoPE ({', '.join(parts)})" if parts else "RoPE"

def _moe_details(cfg: Mapping[str, Any]) -> str:
    n_routed = _first(cfg, "n_routed_experts", "num_experts", "num_local_experts", default=None)
    if n_routed is None:
        return "NA"
    bits = [f"{n_routed} routed experts"]
    if (shared := cfg.get("n_shared_experts")) is not None:
        bits.append(f"{shared} shared")
    if (top_k := _first(cfg, "num_experts_per_tok", "moe_top_k", default=None)) is not None:
        bits.append(f"top-{top_k} active/token")
    if (expert_dim := cfg.get("moe_intermediate_size")) is not None:
        bits.append(f"expert_dim={expert_dim}")
    if (first_dense := cfg.get("first_k_dense_replace")) is not None:
        bits.append(f"first {first_dense} layers dense")
    return "MoE: " + ", ".join(bits)

def _head_dim(cfg: Mapping[str, Any]) -> str:
    """head_dim מהקונפיג, או חישוב hidden_size / num_attention_heads כשחסר."""
    head_dim = cfg.get("head_dim")
    if head_dim:  # קיים ושונה מ-0/None
        return str(head_dim)
    hidden, heads = cfg.get("hidden_size"), cfg.get("num_attention_heads")
    return str(hidden // heads) if hidden and heads else "NA"

def extract(spec: ModelSpec) -> ArchitectureRow:
    """Read ``spec``'s config and build a fully populated :class:`ArchitectureRow`."""
    cfg = AutoConfig.from_pretrained(
        spec.model_id,
        token=hf_token,
        trust_remote_code=spec.trust_remote_code,
    ).to_dict()
    n_heads = _first(cfg, "num_attention_heads")
    return ArchitectureRow(
        model_id=spec.model_id,
        hidden_size=str(_first(cfg, "hidden_size", "d_model")),
        num_layers=str(_first(cfg, "num_hidden_layers", "n_layers")),
        num_attention_heads=str(n_heads),
        num_kv_heads=str(_first(cfg, "num_key_value_heads", default=n_heads)),
        head_dim=_head_dim(cfg),
        mlp_size=str(_first(cfg, "intermediate_size", "ffn_dim", "moe_intermediate_size")),
        activation=str(_first(cfg, "hidden_act", "hidden_activation", "activation_function")),
        norm_type=_norm_type(spec.model_id, cfg),
        position_encoding=_position_encoding(cfg),
        context_length=str(_first(cfg, "max_position_embeddings", "max_seq_len")),
        vocab_size=str(_first(cfg, "vocab_size")),
        moe_details=_moe_details(cfg),
    )

def validate_hf_token() -> None:
    if not hf_token:
        return

    token_info = HfApi().whoami(token=hf_token)
    access_token = token_info.get("auth", {}).get("accessToken", {})
    fine_grained = access_token.get("fineGrained", {})

    if fine_grained and not fine_grained.get("canReadGatedRepos", False):
        raise RuntimeError(
            "Your current Hugging Face token cannot read gated repositories. "
            "Create a new token with gated repo access enabled, or use a classic token, "
            "then rerun this script."
        )

def extract_raw_architectures(models: Iterable[ModelSpec]) -> None:
    validate_hf_token()

    with open(architecture_raw_file, "w") as f:
        for model in models:
            try:
                config = AutoConfig.from_pretrained(model.model_id, token=hf_token)
            except Exception as e:
                print(f"Error loading {model.model_id}: {e}")
                continue
            config_dict = config.to_dict()

            f.write(f"{model.model_id}: ")
            for key, value in config_dict.items():
                f.write(f"{key}={value}\n")

            f.write("\n\n\n")

def _na_row(model_id: str) -> ArchitectureRow:
    values = {name: ("NA" if name != "model_id" else model_id) for name in CSV_COLUMNS}
    return ArchitectureRow(**values)


def build_rows(models: Iterable[ModelSpec]) -> list[ArchitectureRow]:
    """Extract every model, substituting an all-``NA`` row on failure."""
    rows: list[ArchitectureRow] = []
    for spec in models:
        try:
            rows.append(extract(spec))
            print(f"Extracted {spec.model_id}")
        except Exception as exc:  # noqa: BLE001 - report and continue per model
            print(f"Error extracting {spec.model_id}: {exc}")
            rows.append(_na_row(spec.model_id))
    return rows


def export_txt(rows: Iterable[ArchitectureRow], path: Path) -> None:
    with open(path, "w") as f:
        for row in rows:
            for field in row.__dataclass_fields__:
                f.write(f"{field}: {getattr(row, field)}\n")
            f.write("\n\n\n")

def export_csv(rows: Iterable[ArchitectureRow], path: Path) -> None:
    import csv

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ArchitectureRow.__dataclass_fields__.keys())
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__())


def main():
    extract_raw_architectures(MODELS)
    rows = build_rows(MODELS)
    export_txt(rows, architecture_txt_file)
    export_csv(rows, architecture_csv_file)



if __name__ == "__main__":
    main()
