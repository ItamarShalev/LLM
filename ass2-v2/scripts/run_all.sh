#!/usr/bin/env bash
# Run the entire Assignment 2 pipeline end to end, via uv.
#
# Parts 1 and 2 run on any machine (CPU only). Parts 3 and 4 need a GPU.
# Set HF_TOKEN (and TOKEN_KEY for the online Part 4 data) in your environment
# or in a .env file before running. Safe to re-run; each step overwrites its
# own outputs.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -f .env ]; then
  set -a; . ./.env; set +a
fi

echo "==> Part 1: extract + analyze architecture"
uv run python -m src.part1_architecture.extract_architecture --refresh
uv run python -m src.part1_architecture.analyze_architecture

echo "==> Part 2: tokenizer analysis + cross-model diff"
uv run python -m src.part2_tokenizers.analyze_tokenizers
uv run python -m src.part2_tokenizers.tokenization_diff

echo "==> Part 3: Hebrew allowed-token sets + constrained decoding (GPU)"
uv run python -m src.part3_decoding.identify_hebrew_tokens
uv run python -m src.part3_decoding.run_decoding

echo "==> Part 4: build data, train LoRA, evaluate (GPU)"
uv run python -m src.part4_finetuning.make_data
uv run python -m src.part4_finetuning.train_lora
uv run python -m src.part4_finetuning.evaluate

echo "==> Build the final report (HTML + Word)"
uv run python -m report.build_report
npm install --no-audit --no-fund >/dev/null 2>&1 && node report/build_report_docx.js

echo "All done. See outputs/ and report/report.(html|pdf)."
