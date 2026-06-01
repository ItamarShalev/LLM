"""
Part 4 - Fine-tuning (LoRA SFT).

Fine-tunes Qwen2.5-1.5B-Instruct with LoRA so it answers English instructions in
Hebrew. Uses the model's own chat template and masks the prompt tokens so the loss
is computed only on the Hebrew assistant answer (standard SFT). Saves the LoRA
adapter to outputs/lora_adapter/.

Runs on a single modest GPU (Colab T4 is enough for a 1.5B model with LoRA).

Usage:
    python -m src.part4_finetuning.train_lora
    python -m src.part4_finetuning.train_lora --epochs 3 --batch-size 8 --lr 2e-4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as C

TRAIN_FILE = C.TRAIN_DIR / "train.jsonl"
ADAPTER_DIR = C.OUTPUTS / "lora_adapter"


def build_dataset(tokenizer, max_len: int):
    """Tokenize each chat into input_ids + labels (prompt masked to -100)."""
    import torch

    examples = [
        json.loads(line)
        for line in TRAIN_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    feats = []
    for ex in examples:
        msgs = ex["messages"]
        # Prompt = everything up to and including the assistant generation prefix.
        prompt_text = tokenizer.apply_chat_template(
            msgs[:-1], tokenize=False, add_generation_prompt=True
        )
        full_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"][:max_len]
        labels = list(full_ids)
        for i in range(min(len(prompt_ids), len(labels))):
            labels[i] = -100  # do not train on the English prompt
        feats.append(
            {"input_ids": full_ids, "labels": labels, "attention_mask": [1] * len(full_ids)}
        )

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(feats)

        def __getitem__(self, i):
            return feats[i]

    return DS()


def collate(batch, pad_id):
    import torch

    maxlen = max(len(b["input_ids"]) for b in batch)
    out = {"input_ids": [], "attention_mask": [], "labels": []}
    for b in batch:
        pad = maxlen - len(b["input_ids"])
        out["input_ids"].append(b["input_ids"] + [pad_id] * pad)
        out["attention_mask"].append(b["attention_mask"] + [0] * pad)
        out["labels"].append(b["labels"] + [-100] * pad)
    return {k: torch.tensor(v) for k, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=float, default=12)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--rank", type=int, default=16)
    args = ap.parse_args()

    from functools import partial

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    tok: Any = AutoTokenizer.from_pretrained(C.FINETUNE_MODEL, token=C.hf_token())
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model: Any = AutoModelForCausalLM.from_pretrained(
        C.FINETUNE_MODEL,
        token=C.hf_token(),
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    lora = LoraConfig(
        r=args.rank,
        lora_alpha=2 * args.rank,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora)  # type: ignore[assignment]
    model.print_trainable_parameters()

    ds = build_dataset(tok, args.max_len)
    targs = TrainingArguments(
        output_dir=str(C.OUTPUTS / "trainer_tmp"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=args.lr,
        warmup_steps=0.05,
        logging_steps=10,
        save_strategy="no",
        bf16=torch.cuda.is_available(),
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=partial(collate, pad_id=tok.pad_token_id),
    )
    trainer.train()

    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ADAPTER_DIR)
    tok.save_pretrained(ADAPTER_DIR)
    print(f"\nSaved LoRA adapter to {ADAPTER_DIR}")


if __name__ == "__main__":
    main()
