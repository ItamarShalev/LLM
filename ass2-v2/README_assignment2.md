# Technical Inventory: Data, Source, and Outputs

This document contains a structured inventory of the `data`, `src`, and `outputs` directories, detailing the purpose of each folder and the main function of every file.

---

## 📂 1. Directory: `data/`
Houses static prompt configurations, text samples used in tokenizer metrics, and datasets generated for fine-tuning.

### Subfolders
*   **📂 `data/train/`**
    Contains the dataset generated for Supervised Fine-Tuning (SFT).
    *   **`train.jsonl`**: A JSON Lines dataset of SFT conversation pairs, mapping English user prompts to concise Hebrew assistant responses.

### Files
*   **`__init__.py`**: Empty package initializer.
*   **`fixtures.py`**: Contains standard evaluation prompts, benchmark queries, and large text samples used across all tasks.

---

## 📂 2. Directory: `src/`
Contains the Python source code scripts divided by assignment tasks, along with helper utilities.

### Subfolders
*   **📂 `src/common/`**
    *   **`__init__.py`**: Package initializer.
    *   **`token_utils.py`**: Linguistic utility module that maps token IDs to bytes and filters allowed Hebrew characters.
*   **📂 `src/part1_architecture/`**
    *   **`__init__.py`**: Package initializer.
    *   **`extract_architecture.py`**: Fetches configurations from the Hugging Face Hub, extracts architectural specifications, and calculates parameter counts.
*   **📂 `src/part2_tokenizers/`**
    *   **`__init__.py`**: Package initializer.
    *   **`analyze_tokenizers.py`**: Evaluates tokenizer vocabulary statistics, features, and English/Hebrew compression ratios.
    *   **`tokenization_diff.py`**: Identifies and documents splitting differences across tokenizers on high-variance prompts.
*   **📂 `src/part3_decoding/`**
    *   **`__init__.py`**: Package initializer.
    *   **`constrained_decode.py`**: Implements the custom PyTorch LogitsProcessor to restrict token generation to Hebrew-participating tokens.
    *   **`identify_hebrew_tokens.py`**: Scans Qwen and Mistral vocabularies to identify and output valid Hebrew token IDs.
    *   **`run_decoding.py`**: Loads models and runs generation trials under constrained and unconstrained conditions.
*   **📂 `src/part4_finetuning/`**
    *   **`__init__.py`**: Package initializer.
    *   **`make_data.py`**: Generates SFT training pairs via GPT (or offline seed) while preventing evaluation set leaks.
    *   **`train_lora.py`**: Fine-tunes Qwen2.5 using LoRA with prompt masking to optimize training on assistant responses.
    *   **`evaluate.py`**: Benchmarks and compares responses of the base model versus the trained LoRA adapter.



---

## 📂 3. Directory: `outputs/`
Stores intermediate outputs, datasets, generated logs.

### Files
*   **`architecture.csv`**: Extracted architectural parameters and configs of the 10 models.
*   **`tokenizers.csv`**: Evaluated tokenizer statistics, vocab sizes, and compression ratios.
*   **`tokenization_diff_detail.json`**: Subword splits of the 10 models on the chosen high-variance English text.
*   **`hebrew_allowed_tokens_qwen.json`**: List of allowed Hebrew-participating token IDs for Qwen.
*   **`hebrew_allowed_tokens_mistral.json`**: List of allowed Hebrew-participating token IDs for Mistral.
*   **`decoding_outputs.jsonl`: Generation results under baseline decoding.
*   **`decoding_outputs_system_prompt.csv`**: Generation results under system prompt instructions.
*   **`eval_outputs.jsonl`**: Benchmarked responses and diagnostics comparing base vs. LoRA-adapted models.
