# מטלה 2 - ארכיטקטורה, טוקנייזרים, דקודינג ופיין-טיונינג

פרויקט מלא ומוכן להרצה שעונה על כל ארבעת החלקים של המטלה, בתקן איכות עם uv, Ruff ו-ty.

## התחלה מהירה
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # אם uv לא מותקן
make setup
cp .env.example .env       # מלא HF_TOKEN ו-TOKEN_KEY
make check                 # שער איכות: ruff + ty + pytest
bash scripts/run_all.sh    # מריץ הכל ובונה את הדוח
```
חלקים 1 ו-2 רצים על CPU. חלקים 3 ו-4 דורשים GPU (Colab T4 מספיק). לחלקים אלה: `uv sync --extra gpu --extra data --extra report`.

## תקן איכות
פקודה אחת: `make check`, שמריצה:
- `uv run ruff check .` לינט
- `uv run ruff format --check .` פורמט
- `uv run ty check` טיפוסים (אפס שגיאות)
- `uv run pytest -q` בדיקות

יש גם pre-commit (`uv run pre-commit install`) ו-CI ב-.github/workflows/ci.yml.

## מה מתקבל
- outputs/architecture.csv, outputs/tokenizers.csv
- outputs/hebrew_allowed_tokens_qwen.json, outputs/hebrew_allowed_tokens_mistral.json
- outputs/decoding_outputs.jsonl, outputs/eval_outputs.jsonl
- report/report.docx (מסמך ההגשה ב-Word) וגם report/report.html או report.pdf

## מסמכים
- SPEC.md - האיפיון המלא, תקן האיכות, ומיפוי כל דרישה לתוצר.
- CLAUDE.md - הנחיות הרצה ובדיקה ל-Claude Code.

## הסודות הנדרשים
- HF_TOKEN - טוקן Hugging Face (כולל אישור רישיון Llama ב-Hub).
- TOKEN_KEY - מפתח GPT ליצירת דאטה ל-Part 4. מודל דיפולטי gpt-5.4-mini, ניתן לשינוי דרך OPENAI_MODEL.
