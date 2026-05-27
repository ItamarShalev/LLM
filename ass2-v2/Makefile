# Assignment 2 pipeline, driven by uv.
# Quality gate: make check  (ruff lint + ruff format check + ty + tests)
# Pipeline:     make all    (runs every part and builds the report)
.PHONY: help setup check lint fmt typecheck test p1 p2 p3 p4 report report-docx all clean

help:
	@echo "make setup     - create venv and install all deps (CPU + dev)"
	@echo "make check     - quality gate: ruff lint, format check, ty, pytest"
	@echo "make fmt       - auto-format and auto-fix with ruff"
	@echo "make p1        - Part 1 architecture (CPU)"
	@echo "make p2        - Part 2 tokenizers (CPU)"
	@echo "make p3        - Part 3 constrained decoding (GPU)"
	@echo "make p4        - Part 4 fine-tuning + eval (GPU)"
	@echo "make report    - build report.html / report.pdf"
	@echo "make report-docx - build the Word .docx submission (report/report.docx)"
	@echo "make all       - run the whole pipeline"

setup:
	uv sync --group dev

# ---- quality gate ----
check: lint typecheck test
	uv run ruff format --check .

lint:
	uv run ruff check .

fmt:
	uv run ruff check --fix .
	uv run ruff format .

typecheck:
	uv run ty check

test:
	uv run pytest -q

# ---- pipeline ----
p1:
	uv run python -m src.part1_architecture.extract_architecture --refresh
	uv run python -m src.part1_architecture.analyze_architecture

p2:
	uv run python -m src.part2_tokenizers.analyze_tokenizers
	uv run python -m src.part2_tokenizers.tokenization_diff

p3:
	uv run python -m src.part3_decoding.identify_hebrew_tokens
	uv run python -m src.part3_decoding.run_decoding

p4:
	uv run python -m src.part4_finetuning.make_data
	uv run python -m src.part4_finetuning.train_lora
	uv run python -m src.part4_finetuning.evaluate

report:
	uv run python -m report.build_report

report-docx:
	npm install --no-audit --no-fund
	node report/build_report_docx.js

all:
	bash scripts/run_all.sh

clean:
	rm -rf outputs/trainer_tmp report/report.html report/report.pdf
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
