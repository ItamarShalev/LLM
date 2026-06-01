"""
Shared text fixtures.

`ENGLISH_SAMPLE` and `HEBREW_SAMPLE` are original paragraphs written for this
assignment (not copied from any source) and are used only to estimate
average-tokens-per-word. They are deliberately ordinary prose covering common
vocabulary, light punctuation and a couple of numbers, so the estimate reflects
typical text rather than an adversarial edge case.

`EVAL_INPUTS_PROVIDED` are the ten English evaluation prompts mandated by the
assignment. `EVAL_INPUTS_OWN` are ten additional prompts of our choosing. The
union of these twenty is the held-out set that must NEVER appear in training data
(see src/part4_finetuning/make_data.py, which filters against it).
"""
import json
import os
from datasets import load_dataset
from huggingface_hub import hf_hub_download
_hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")

en_corpus = load_dataset("wikitext", "wikitext-103-v1", split="test")
ENGLISH_SAMPLE = " ".join([item["text"] for item in en_corpus if item["text"].strip()][:10000])

def _load_hebrew_texts() -> list[str]:
    dataset_path = hf_hub_download(
        "YanFren/Hebrew_wikipedia",
        repo_type="dataset",
        filename="dataset.jsonl",
        token=_hf_token,
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

    return texts[:10000]


HEBREW_SAMPLE = " ".join(_load_hebrew_texts())


"""
ENGLISH_SAMPLE = (
    "The morning train was almost empty when she finally found a seat by the window. "
    "Outside, the fields slipped past in long green stripes, broken now and then by a "
    "farmhouse or a line of tall trees. She opened her notebook and began to write down "
    "everything she still had to do before the meeting at nine: print the report, send "
    "two emails, and buy a coffee strong enough to keep her awake. The city was still 40 "
    "minutes away, but already the sky over the river looked brighter than it had at home. "
    "By the time the announcement crackled through the speakers, she had filled three pages "
    "and felt, for the first time in weeks, that the day might actually go well."
)
"""

"""
HEBREW_SAMPLE = (
    "הרכבת של הבוקר הייתה כמעט ריקה כשהיא סוף סוף מצאה מקום ליד החלון. "
    "בחוץ חלפו השדות בפסים ארוכים וירוקים, שנקטעו פה ושם בבית חווה או בשורה של עצים גבוהים. "
    "היא פתחה את המחברת והתחילה לרשום את כל מה שעוד נשאר לה לעשות לפני הפגישה בתשע: להדפיס את הדוח, "
    "לשלוח שני מיילים, ולקנות קפה חזק מספיק כדי להישאר ערה. העיר עדיין הייתה במרחק 40 דקות, "
    "אבל כבר עכשיו השמיים מעל הנהר נראו בהירים יותר מאשר בבית. עד שההודעה נשמעה ברמקולים, "
    "היא כבר מילאה שלושה עמודים והרגישה, לראשונה מזה שבועות, שהיום אולי באמת יעבור בטוב."
)
"""

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
    "Turn the idea \u201cpractice makes progress\u201d into advice for a student.",
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
    "Turn the idea \u201cmistakes are part of learning\u201d into encouragement for a child.",
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
