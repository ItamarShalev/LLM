"""
Shared text fixtures.

`ENGLISH_SAMPLE` and `HEBREW_SAMPLE` are prose samples used only to estimate
average-tokens-per-word in Part 2. They are sampled from real corpora at first
use (English: Salesforce/wikitext wikitext-103; Hebrew: YanFren/Hebrew_wikipedia)
so the token ratios reflect natural text. If the datasets cannot be reached, we
fall back to short handwritten paragraphs so Part 2 still runs fully offline.

These two samples are loaded lazily (PEP 562 module ``__getattr__``): importing
the cheap constants below (``DECODING_QUERIES``, ``EVAL_INPUTS_*``) never touches
the network, so Parts 3 and 4 and the offline unit tests stay download-free. The
corpora are fetched only when ``ENGLISH_SAMPLE`` / ``HEBREW_SAMPLE`` are actually
read, and the result is cached for the process.

`EVAL_INPUTS_PROVIDED` are the ten English evaluation prompts mandated by the
assignment. `EVAL_INPUTS_OWN` are ten additional prompts of our choosing. The
union of these twenty is the held-out set that must NEVER appear in training data
(see src/part4_finetuning/make_data.py, which filters against it).
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

# Maximum number of corpus items to join into each sample. A few thousand lines
# is far more than enough for a stable tokens-per-word estimate while keeping
# tokenization in Part 2 fast.
_MAX_ITEMS = 10000

# Handwritten fallbacks (original paragraphs written for this assignment). Used
# only when the online corpora are unreachable, so Part 2 never hard-fails.
_FALLBACK_ENGLISH = (
    "The morning train was almost empty when she finally found a seat by the window. "
    "Outside, the fields slipped past in long green stripes, broken now and then by a "
    "farmhouse or a line of tall trees. She opened her notebook and began to write down "
    "everything she still had to do before the meeting at nine: print the report, send "
    "two emails, and buy a coffee strong enough to keep her awake. The city was still 40 "
    "minutes away, but already the sky over the river looked brighter than it had at home. "
    "By the time the announcement crackled through the speakers, she had filled three pages "
    "and felt, for the first time in weeks, that the day might actually go well."
)
_FALLBACK_HEBREW = (
    "הרכבת של הבוקר הייתה כמעט ריקה כשהיא סוף סוף מצאה מקום ליד החלון. "
    "בחוץ חלפו השדות בפסים ארוכים וירוקים, שנקטעו פה ושם בבית חווה או בשורה של עצים גבוהים. "
    "היא פתחה את המחברת והתחילה לרשום את כל מה שעוד נשאר לה לעשות לפני הפגישה בתשע: להדפיס את הדוח, "
    "לשלוח שני מיילים, ולקנות קפה חזק מספיק כדי להישאר ערה. העיר עדיין הייתה במרחק 40 דקות, "
    "אבל כבר עכשיו השמיים מעל הנהר נראו בהירים יותר מאשר בבית. עד שההודעה נשמעה ברמקולים, "
    "היא כבר מילאה שלושה עמודים והרגישה, לראשונה מזה שבועות, שהיום אולי באמת יעבור בטוב."
)


def _hf_token() -> str | None:
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")


def _load_english_sample() -> str:
    """Join the first lines of the wikitext-103 test split into one sample.

    The legacy bare id ``wikitext`` is no longer a valid Hub repo id (newer
    huggingface_hub requires ``namespace/name``); the canonical mirror is
    ``Salesforce/wikitext``.
    """
    try:
        from datasets import load_dataset

        corpus = load_dataset(
            "Salesforce/wikitext", "wikitext-103-v1", split="test", token=_hf_token()
        )
        items = [row["text"] for row in corpus if row["text"].strip()][:_MAX_ITEMS]
        text = " ".join(items).strip()
        return text or _FALLBACK_ENGLISH
    except Exception as exc:
        print(f"[fixtures] English corpus unavailable ({type(exc).__name__}); using fallback")
        return _FALLBACK_ENGLISH


def _load_hebrew_sample() -> str:
    """Join paragraphs from the Hebrew Wikipedia dataset into one sample."""
    try:
        from huggingface_hub import hf_hub_download

        dataset_path = hf_hub_download(
            "YanFren/Hebrew_wikipedia",
            repo_type="dataset",
            filename="dataset.jsonl",
            token=_hf_token(),
        )
        texts: list[str] = []
        with open(dataset_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                paragraphs = item.get("paragraphs", [])
                if isinstance(paragraphs, list):
                    texts.extend(p for p in paragraphs if p.strip())
                if len(texts) >= _MAX_ITEMS:
                    break
        text = " ".join(texts[:_MAX_ITEMS]).strip()
        return text or _FALLBACK_HEBREW
    except Exception as exc:
        print(f"[fixtures] Hebrew corpus unavailable ({type(exc).__name__}); using fallback")
        return _FALLBACK_HEBREW


# Cache so the corpora are downloaded/joined at most once per process.
_sample_cache: dict[str, str] = {}

_LAZY_LOADERS = {
    "ENGLISH_SAMPLE": _load_english_sample,
    "HEBREW_SAMPLE": _load_hebrew_sample,
}

if TYPE_CHECKING:  # help static checkers see the lazily-provided names
    ENGLISH_SAMPLE: str
    HEBREW_SAMPLE: str


def __getattr__(name: str) -> str:
    """PEP 562 lazy module attributes for the two heavy text samples."""
    loader = _LAZY_LOADERS.get(name)
    if loader is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if name not in _sample_cache:
        _sample_cache[name] = loader()
    return _sample_cache[name]


EVAL_INPUTS_PROVIDED = [
    "Explain why the sky looks blue during the day.",
    "Give two advantages and two disadvantages of public transportation.",
    "Write a short email asking a professor for an extension on an assignment.",
    "Describe how to make a simple omelette.",
    "What is the difference between supervised and unsupervised learning?",
    "Summarize the story of Cinderella in three sentences.",
    "Suggest three ways to reduce smartphone distraction while studying.",
    "Explain what happens when water boils.",
    "Give a polite refusal to an invitation to a party.",
    "Turn the idea “practice makes progress” into advice for a student.",
]

EVAL_INPUTS_OWN = [
    "Explain in simple terms how a rainbow forms.",
    "List three tips for staying focused while working from home.",
    "Write a two-sentence thank-you note to a friend who helped you move.",
    "What is the difference between weather and climate?",
    "Describe the steps to brew a cup of tea.",
    "Give one reason to learn a second language and one reason it can be hard.",
    "Summarize what photosynthesis does for a plant.",
    "Suggest a simple plan to start running for a complete beginner.",
    "Explain why we should back up important files.",
    "Turn the idea “mistakes are part of learning” into encouragement for a child.",
]

# The full held-out set used to filter training data.
EVAL_INPUTS_ALL = EVAL_INPUTS_PROVIDED + EVAL_INPUTS_OWN

# A short list of English queries used for Part 3 constrained decoding (10 total).
DECODING_QUERIES = [
    "Explain why leaves are green.",
    "What is the capital of France?",
    "Give one tip for sleeping better.",
    "Describe what a computer does in one sentence.",
    "Why is the ocean salty?",
    "Name three healthy breakfast foods.",
    "What causes thunder?",
    "How do bees make honey?",
    "Suggest a good reason to read books.",
    "What is the largest planet in our solar system?",
]
