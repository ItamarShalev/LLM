# Assignment 4 - Retrieval & RAG over "Kol Zchut"  

This repository contains our implementation for **Assignment 4** (parts 1 and 2) of the course.  
The code covers:

- **Part 1** - Classical IR and vector-based retrieval over a small corpus (`docs.jsonl`, `queries.jsonl`), files: ()`indexing_part1.py`, `utils.py`.  
  - Four indexing methods:
    - BM25 over raw text.
    - Static word2vec embeddings.
    - Contextual BERT embeddings.
    - SentenceTransformers embeddings.
  - Evaluation of Recall@20 and MRR.
  - Outputs:
    - Ranked document IDs for the first query (`q1.txt`).
    - Metrics per method (`scores.txt`).

- **Part 2** - End-to-end RAG system over *Kol Zchut*:
  - Pre-processing and indexing of the Kol Zchut dump.
  - Multiple retrieval methods (BM25, dense, hybrid, LLM-assisted).
  - Two RAG systems: **unconstrained** and **token-budget-constrained**.
  - Manual end-to-end evaluation pipeline.
- **Code Files:**
    - `indexing.py` - pre-processing and indexing.
    - `retrieval.py` - retrieval methods and evaluation.
    - `e2e_rag.py` - RAG systems and Streamlit demo.
    - `eval_e2e_rag.py` - end-to-end evaluation script.
    - `utils.py` - shared utilities and global paths.
    - Data files under `data/` as per assignment instructions.
 - **Data Files:**
    - `ranks.txt` - retrieved document IDs for each query.
    - `eval_results_manual.csv` - manual evaluation results for the E2E RAG systems.

---

## 1. Though not all filed were included here, the project structure used to run the code (Other structure needs adjustments in paths in `utils.py`)
## to run the code files, the assumption is the project sturcture matches the following structure

```bash
root/
├── data/
│   ├── docs.jsonl              # input documents for Part 1
│   ├── queries.jsonl           # input queries for Part 1
│   ├── eval-set.csv            # evaluation set for Part 2
│   ├── kolzchut/               # Kol Zchut dataset for Part 2
│   │   ├── pages/              # Kol Zchut pages html dataset
│   │   └── get_random_page.py  # Kol Zchut pages metadata
│   │   └── page_urls.jsonl     # Kol Zchut pages URLs
├── src/
│   ├── indexing_part1.py       # Part 1 indexing & evaluation
│   ├── indexing.py             # Part 2 indexing
│   ├── retrieval.py            # Part 2 retrieval & evaluation
│   ├── e2e_rag.py              # RAG systems + Streamlit demo
│   ├── eval_e2e_rag.py         # End-to-end evaluation
│   └── utils.py                # shared utilities & global paths
├── .env                        # environment variables file
├── README.md                   # this file
├── pyproject.toml              # project metadata & dependencies
├── uv.lock                     # uv environment lock file
└── requirements.txt            # dependencies for pip installation
```

---

## 2. Environment & Dependencies

### Python version

- Implemented and tested with **Python 3.13** (assumes a modern environment with `asyncio`, `pathlib`, type hints, etc.).

### Key libraries

- `torch` - GPU/CPU device detection, transformer models. 
- `bm25s` - BM25 indexing and retrieval (both parts).
- `numpy` - dense vectors, score fusion.
- `sentence_transformers` - dense embeddings (`SentenceTransformer`, cosine similarity).
- `transformers` - DictaBERT lexical model, BERT context encoders (Part 1), masked-LM for lemma expansion.
- `gensim.downloader` - pre-trained `word2vec-google-news-300` (Part 1).
- `openai` - `AsyncOpenAI` client for query variations, reranking, and answer generation.   
- `tiktoken` - token counting for constrained RAG. 
- `streamlit` - UI for the demo app.   
- `tqdm`, `python-dotenv`, `python-bidi` - quality-of-life utilities in `utils.py`. 

### API keys & configuration

- The code expects a `.env` file at project root (`ROOT/.env`), loaded via `utils.load_dotenv`:
  - Required environment variable:
    - `OPENAI_API_KEY` - used by:
      - `LLMQueryManager` (query paraphrasing & reranking).   
      - `RagSystem` subclasses for answer generation.   

No additional configuration is required beyond placing the dataset files under `data/` according to the assignment instructions.

---

## 3. Installation

### Using `pip`
```bash
pip install -r requirements.txt
```

### Using `uv`
```bash
# Use uv.lock and pyproject.toml to create the environment
uv sync
```

