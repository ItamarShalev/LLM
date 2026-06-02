"""
Part 4 - Training data.

Builds a supervised fine-tuning dataset for the task "English instruction in,
Hebrew answer out". Each example is a chat with a user turn (English) and an
assistant turn (Hebrew). The data is written to data/train/train.jsonl in the
messages format that train_lora.py consumes.

Two sources, combined:
  * A handwritten offline seed (SEED_PAIRS below) so the whole pipeline runs and
    can be sanity-checked even before any API key is set.
    * A GPT-generated bulk set (default model gpt-5.4-mini, override with
    OPENAI_MODEL). GPT first proposes diverse English instructions across many
    categories, then answers each one concisely in natural Hebrew.

Guardrails:
  * Every candidate English prompt is filtered against the 20 held-out evaluation
    inputs (exact, whitespace/case-normalized match) so the eval set never leaks.
  * Answers that are not actually in Hebrew (fewer than 40% Hebrew letters) are
    dropped, so the model learns to answer in Hebrew, not to echo English.

Usage:
    python -m src.part4_finetuning.make_data --n 300      # uses OPENAI_API_KEY
    python -m src.part4_finetuning.make_data --offline    # seed only, no API
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as C
from data.fixtures import EVAL_INPUTS_ALL

TRAIN_FILE = C.TRAIN_DIR / "train.jsonl"
HEBREW_RE = re.compile(r"[\u0590-\u05FF]")

# A small, original, fully handwritten seed so training works with zero API calls.
SEED_PAIRS = [
    ("What is the capital of Italy?", "בירת איטליה היא רומא."),
    ("Name a fruit that is red.", "תפוח הוא פרי אדום נפוץ."),
    ("How many days are in a week?", "בשבוע יש שבעה ימים."),
    ("What color is the sky on a clear day?", "ביום בהיר השמיים נראים כחולים."),
    ("Give one benefit of drinking water.", "שתיית מים עוזרת לגוף להישאר רענן וממוקד."),
    ("What is two plus two?", "שתיים ועוד שתיים שווה ארבע."),
    ("Suggest a simple way to relax.", "דרך פשוטה להירגע היא לקחת כמה נשימות עמוקות ואיטיות."),
    ("What do bees collect from flowers?", "דבורים אוספות צוף מהפרחים כדי להכין דבש."),
    ("Name a planet in our solar system.", "מאדים הוא אחד הכוכבים במערכת השמש שלנו."),
    ("Why is sleep important?", "שינה חשובה כי היא מאפשרת לגוף ולמוח לנוח ולהתאושש."),
    ("What is the opposite of hot?", "ההפך מחם הוא קר."),
    ("Give one tip for learning a new language.", "טיפ אחד ללימוד שפה חדשה הוא לתרגל מעט בכל יום."),
    ("What animal says meow?", "החתול הוא בעל החיים שמשמיע מיאו."),
    ("Describe the color green in one sentence.", "ירוק הוא הצבע של עלים ושל דשא טרי."),
    ("Why do we wear a coat in winter?", "אנחנו לובשים מעיל בחורף כדי לשמור על הגוף חם."),
    ("What is a library used for?", "ספרייה משמשת לקריאה ולהשאלה של ספרים."),
]

CATEGORIES = [
    "everyday factual questions",
    "short how-to instructions",
    "simple science explanations",
    "writing a short polite message",
    "giving advice or tips",
    "summarizing a common story",
    "comparisons between two concepts",
    "definitions of common terms",
    "math word problems",
    "opinions and suggestions",
]


def is_hebrew(text: str, threshold: float = 0.4) -> bool:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    heb = sum(1 for c in letters if HEBREW_RE.match(c))
    return heb / len(letters) >= threshold


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def add_pair(
    pairs: list[tuple[str, str]],
    seen: set[str],
    held: set[str],
    en: str,
    he: str,
) -> bool:
    key = norm(en)
    if key in held or key in seen:
        return False
    seen.add(key)
    pairs.append((en, he))
    return True


def _client() -> object | None:
    try:
        from openai import OpenAI
    except ModuleNotFoundError:
        print(
            "[warn] openai is not installed; falling back to the handwritten seed only.",
            file=sys.stderr,
        )
        return None

    key = C.openai_api_key()
    if not key:
        print(
            "[warn] No OPENAI_API_KEY / TOKEN_KEY in environment; falling back to the handwritten seed only.",
            file=sys.stderr,
        )
        return None
    return OpenAI(api_key=key)


def _chat(client, system, user, max_tokens=600):
    # Newer OpenAI models (gpt-5+) require max_completion_tokens; older ones use
    # max_tokens. Try the new name first, fall back transparently so the same
    # script works against either model family.
    kwargs = {
        "model": C.openai_model(),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    try:
        resp = client.chat.completions.create(max_completion_tokens=max_tokens, **kwargs)
    except TypeError:
        resp = client.chat.completions.create(max_tokens=max_tokens, **kwargs)
    except Exception as e:
        msg = str(e)
        if "max_tokens" in msg and "not supported" not in msg:
            resp = client.chat.completions.create(max_completion_tokens=max_tokens, **kwargs)
        elif "max_completion_tokens" in msg:
            resp = client.chat.completions.create(max_tokens=max_tokens, **kwargs)
        else:
            raise
    return (resp.choices[0].message.content or "").strip()


def gen_english_prompts(client, n: int) -> list[str]:
    """Ask GPT for diverse English instructions, n total across categories."""
    out: list[str] = []
    per = max(5, n // len(CATEGORIES) + 1)
    for cat in CATEGORIES:
        txt = _chat(
            client,
            "You generate short, clear English instructions or questions that would come from a user for a chatbot. "
            "The question or instruction should not reference some context that was not included in the instruction itself."
            f"Questions or instructions are similar in nature to the following: {" ".join([qa[0] for qa in SEED_PAIRS])} ... but should be varied and cover the category of {cat}."
            "The instructions should be suitable for a general-purpose assistant to answer, not require specialized knowledge or niche expertise. Yet at the same time should not be too generic, common or bland. Avoid instructions that are just asking for lists of items (e.g. 'Name some fruits') or that are too similar to the held-out evaluation set which as follows "
            f"{' '.join(EVAL_INPUTS_ALL)}."
            "All questions or instructions should be different from each other" 
            "Return one instruction per line, no numbering, no quotes.",
            f"Write {per} varied English instructions about: {cat}. Keep each under 20 words.",
            max_tokens=800,
        )
        for line in txt.splitlines():
            line = line.strip(" -\t")
            if line and len(line.split()) <= 25:
                out.append(line)
    # de-dupe and drop any that collide with the held-out eval set
    held = {norm(x) for x in EVAL_INPUTS_ALL}
    seen, clean = set(), []
    for p in out:
        if norm(p) in held or norm(p) in seen:
            continue
        seen.add(norm(p))
        clean.append(p)
    return clean[:n]


def gen_hebrew_answer(client, english_prompt: str) -> str:
    return _chat(
        client,
        "You are a helpful assistant. You ALWAYS answer in natural, fluent Hebrew, "
        "regardless of the language of the question. Keep answers concise (1-4 "
        "sentences) and actually address the question.",
        english_prompt,
        max_tokens=400,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="minimum number of final training examples")
    ap.add_argument("--offline", action="store_true", help="seed only, no API calls")
    args = ap.parse_args()

    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    held = {norm(x) for x in EVAL_INPUTS_ALL}
    for en, he in SEED_PAIRS:
        add_pair(pairs, seen, held, en, he)

    if not args.offline:
        client = _client()
        if client is not None:
            max_rounds = max(5, min(40, args.n // 100 + 5))
            round_idx = 0
            while len(pairs) < args.n and round_idx < max_rounds:
                remaining = args.n - len(pairs)
                request_n = min(max(200, remaining * 2), 800)
                prompts = gen_english_prompts(client, request_n)
                print(
                    f"[gpt] round {round_idx + 1}/{max_rounds}: requested {request_n}, "
                    f"got {len(prompts)} candidate English prompts"
                )
                accepted_before = len(pairs)
                for i, p in enumerate(prompts, 1):
                    try:
                        ans = gen_hebrew_answer(client, p)
                    except Exception as e:
                        print(f"  [skip] {p[:40]!r}: {e}", file=sys.stderr)
                        continue
                    if is_hebrew(ans):
                        add_pair(pairs, seen, held, p, ans)
                    if i % 25 == 0:
                        print(f"  ...{i}/{len(prompts)} answered")
                accepted = len(pairs) - accepted_before
                print(f"  accepted {accepted} new examples (total {len(pairs)}/{args.n})")
                round_idx += 1

            if len(pairs) < args.n:
                raise RuntimeError(
                    f"Could only assemble {len(pairs)} examples after {round_idx} rounds; "
                    f"target was {args.n}."
                )

    with TRAIN_FILE.open("w", encoding="utf-8") as f:
        for en, he in pairs:
            rec = {
                "messages": [{"role": "user", "content": en}, {"role": "assistant", "content": he}]
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\nWrote {TRAIN_FILE} ({len(pairs)} examples; held-out leakage check passed)")


if __name__ == "__main__":
    main()
