# CLAUDE.md - הנחיות ל-Claude Code

קרא קודם את SPEC.md לתמונה המלאה. קובץ זה מסכם את כללי העבודה.

## תקן איכות (חובה)
הקוד עומד בתקן עם uv, Ruff ו-ty. שמור על השער ירוק.
- שער אחד: `make check` (ruff check, ruff format --check, ty check, pytest).
- אחרי כל עריכת קוד הרץ שוב `make check`. אל תשאיר אותו אדום.
- לתיקון אוטומטי של פורמט ולינט: `make fmt`.
- אפס שגיאות ב-ty. אזהרות unresolved-import עבור torch, peft ו-openai תקינות בסביבת CPU ונעלמות אחרי `uv sync --all-extras`.

## עקרונות
- אל תכתוב קוד חדש אלא אם צעד נכשל ומחייב תיקון. הכל כבר ממומש ועובר את השער. תפקידך להריץ, לבדוק תוצרים, ולבנות את הדוח.
- כל סקריפט הוא idempotent. מותר להריץ שוב; הוא דורס את התוצר של עצמו בלבד.
- אל תשתמש בתווי em-dash או en-dash בפרוזה.
- אל תדפיס סודות. HF_TOKEN ו-TOKEN_KEY מגיעים ממשתני סביבה או מ-.env בלבד.
- כל הפקודות רצות דרך uv run (ראה ה-Makefile). אל תפעיל venv ידנית.

## סביבה
```
make setup        # uv sync --group dev  (deps CPU + כלי פיתוח)
cp .env.example .env   # מלא HF_TOKEN ו-TOKEN_KEY
# לחלקים 3-4 (GPU):
uv sync --group dev --extra gpu --extra data --extra report
```

## הרצה לפי חלקים
חלקים 1 ו-2 על CPU, חלקים 3 ו-4 על GPU.
- `make p1` בודק: outputs/architecture.csv עם 10 שורות וכל העמודות הנדרשות. אם שורת meta-llama ריקה, ודא HF_TOKEN מאושר לרישיון Llama והרץ עם --refresh.
- `make p2` בודק: outputs/tokenizers.csv עם שורה לכל מודל, ו-report/sections/part2_diff.md מצביע על טקסט הפיצול שנבחר.
- `make p3` בודק: שני קבצי hebrew_allowed_tokens_*.json ו-decoding_outputs.jsonl עם 20 שורות (10 שאילתות כפול 2 מודלים).
- `make p4` בודק: data/train/train.jsonl ללא דליפה, outputs/lora_adapter נשמר, ו-outputs/eval_outputs.jsonl עם 20 שורות.

## בניית הדוח
1. מלא את report/students.json בשמות ובת.ז.
2. `make report`. אם weasyprint מותקן ייווצר report/report.pdf; אחרת report/report.html להדפסה ל-PDF.
3. ודא שאין בלוקים של "טרם נוצר" אם הרצת את כל הצעדים.

## תקלות נפוצות
- מודל נעול: meta-llama דורש אישור רישיון ב-Hub בנוסף לטוקן.
- קונפיג ישן בקאש: הוסף --refresh ל-extract_architecture.
- אין torch בסביבה: חלקים 1 ו-2 ירוצו; 3 ו-4 דורשים `uv sync --extra gpu`.
