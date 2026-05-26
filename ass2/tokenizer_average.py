from collections.abc import Iterable
import csv
import json
import math
from dataclasses import fields
from pathlib import Path
import tqdm

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from architecture import MODELS, TokenizerRow, hf_token, tokenizer_csv_file


en_corpus = load_dataset("wikitext", "wikitext-103-v1", split="test")
english_texts = [item["text"] for item in en_corpus if item["text"].strip()]

def _load_hebrew_texts() -> list[str]:
    dataset_path = hf_hub_download(
        "YanFren/Hebrew_wikipedia",
        repo_type="dataset",
        filename="dataset.jsonl",
        token=hf_token,
    )
    texts: list[str] = []

    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            item = json.loads(line)
            paragraphs = item.get("paragraphs", [])
            if isinstance(paragraphs, list):
                texts.extend(paragraph for paragraph in paragraphs if paragraph.strip())

    return texts


hebrew_texts = _load_hebrew_texts()


def _load_tokenizer(model_id: str, trust_remote_code: bool) -> AutoTokenizer:
    load_kwargs = {
        "token": hf_token,
        "trust_remote_code": trust_remote_code,
    }

    try:
        return AutoTokenizer.from_pretrained(model_id, use_fast=True, **load_kwargs)
    except Exception:
        return AutoTokenizer.from_pretrained(model_id, use_fast=False, **load_kwargs)


def _average_tokens_per_word(tokenizer: AutoTokenizer, texts: Iterable[str]) -> float:
    ratios: list[float] = []

    for text in texts[:4000]: 
        words = text.split()    
        if not words:
            continue

        tokens = tokenizer.encode(text, add_special_tokens=False)
        ratios.append(len(tokens) / len(words))

    return sum(ratios) / len(ratios) if ratios else math.nan


def compute_tokens_per_word(model_id: str, trust_remote_code: bool) -> tuple[str, str]:
    tokenizer = _load_tokenizer(model_id, trust_remote_code)
    english_ratio = _average_tokens_per_word(tokenizer, english_texts)
    hebrew_ratio = _average_tokens_per_word(tokenizer, hebrew_texts)

    return _format_ratio(english_ratio), _format_ratio(hebrew_ratio)


def _format_ratio(value: float) -> str:
    return "NA" if math.isnan(value) else f"{value:.6f}"


def update_tokenizer_csv(path: Path, averages: dict[str, tuple[str, str]]) -> None:
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    else:
        rows = []

    rows_by_model_id = {row["model_id"]: row for row in rows if row.get("model_id")}

    for spec in MODELS:
        row = rows_by_model_id.get(spec.model_id)
        if row is None:
            row = {field.name: "NA" for field in fields(TokenizerRow)}
            row["model_id"] = spec.model_id
            rows.append(row)
            rows_by_model_id[spec.model_id] = row

        english_ratio, hebrew_ratio = averages.get(spec.model_id, ("NA", "NA"))
        row["avg_tokens_per_english_word"] = english_ratio
        row["avg_tokens_per_hebrew_word"] = hebrew_ratio

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[field.name for field in fields(TokenizerRow)])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    averages: dict[str, tuple[str, str]] = {}

    for spec in tqdm.tqdm(MODELS):
        try:
            averages[spec.model_id] = compute_tokens_per_word(spec.model_id, spec.trust_remote_code)
            print(f"Computed token averages for {spec.model_id}")
        except Exception as exc:  # noqa: BLE001 - report and continue per model
            print(f"Error computing token averages for {spec.model_id}: {exc}")
            averages[spec.model_id] = ("NA", "NA")

    update_tokenizer_csv(tokenizer_csv_file, averages)


if __name__ == "__main__":
    main()







