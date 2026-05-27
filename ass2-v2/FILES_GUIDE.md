# מדריך קבצים והרצה מלא - מטלה 2

מסמך זה מסביר מה כל קובץ בפרויקט עושה, ואיך מריצים את הכל מקצה לקצה. הוא נכתב כך שלא תצטרך לנחש דבר. כל הפקודות רצות דרך uv. אין מקפים ארוכים בפרוזה.

## הוראות הרצה מהירות
```
curl -LsSf https://astral.sh/uv/install.sh | sh   # התקנת uv אם חסר
uv python install 3.13                             # הפרויקט מכוון ל-3.13
make setup                                         # התקנת תלויות CPU וכלי פיתוח
cp .env.example .env                               # מלא HF_TOKEN ו-TOKEN_KEY
make check                                         # שער איכות: ruff, ty, pytest
make p1                                            # חלק 1 (CPU)
make p2                                            # חלק 2 (CPU)
# לחלקים 3 ו-4 דרוש GPU:
uv sync --group dev --extra gpu --extra data --extra report
make p3                                            # חלק 3 (GPU)
make p4                                            # חלק 4 (GPU)
# מלא את report/students.json עם שמות ות.ז.
make report                                        # בניית הדוח הסופי
```
או בפקודה אחת לכל הפייפליין: bash scripts/run_all.sh

## הקבצים בשורש
- pyproject.toml: מקור האמת לפרויקט. תלויות, dependency-groups, extras (gpu, data, report), והגדרות ruff ו-ty. כולל גם את אינדקס ה-PyTorch CUDA 13.0 לגלגלי GPU.
- uv.lock: נעילת גרסאות מלאה של כל החבילות, כך שההתקנה משוחזרת בדיוק.
- Makefile: כל הפקודות. setup, check, fmt, p1 עד p4, report, all, clean.
- config.py: רישום מרכזי. רשימת עשרת המודלים, המודלים לדקודינג, מודל הפיין-טיונינג, כל נתיבי הפלט, טבלאות ידע על נורמליזציה ומשפחות טוקנייזר, פונקציות לקריאת הסודות, רשימת המראות הציבוריים למודלים נעולים, והגדרת עמודות ה-CSV.
- .env.example: תבנית לשני הסודות. HF_TOKEN ו-TOKEN_KEY, ומשתנה OPENAI_MODEL לבחירת מודל ה-GPT.
- README.md: התחלה מהירה ותקציר.
- SPEC.md: מסמך האיפיון המלא, תקן האיכות, ומיפוי כל דרישה לתוצר.
- CLAUDE.md: הנחיות עבודה ל-Claude Code.
- ANSWERS.md: מסמך התשובות והתובנות שעונה על כל שאלות הניתוח בכל ארבעת החלקים.
- FILES_GUIDE.md: המסמך הזה.
- .pre-commit-config.yaml: hooks של ruff ו-ty לפני כל commit.
- .github/workflows/ci.yml: שער איכות אוטומטי ב-CI.
- .gitignore: התעלמות מ-venv, קאש, ותוצרים זמניים.
- requirements.txt: גיבוי pip בלבד; pyproject הוא המקור הקנוני.
- scripts/run_all.sh: מריץ את כל החלקים ובונה את הדוח.

## הקוד תחת src

### src/common/token_utils.py
לב הזיהוי הלשוני. כולל regex לתווים עבריים, מפענח ה-byte של GPT-2, פונקציית detect_family שמזהה אם הטוקנייזר הוא byte-level או SentencePiece לפי סוג ה-backend, פונקציות שממירות טוקן לבתים ולמשטח טקסט תוך טיפול ב-byte fallback, והמסווג is_hebrew_participating שמכריע אם טוקן מותר בעברית. כל שאר החלקים נשענים על הקובץ הזה.

### src/part1_architecture
- extract_architecture.py: מוריד את config.json של כל מודל, מחשב head_dim עם מודעות ל-MLA, יחס ראשי KV, אומדן פרמטרים, פרטי MoE וקידוד מיקום, ונופל למראה ציבורי עבור מודל נעול ללא טוקן. כותב את outputs/architecture.csv. דגלים: offline-- לעבודה ללא רשת, refresh-- לרענון הקאש.
- analyze_architecture.py: מחשב את המגמות וכותב את report/sections/part1_analysis.md, כולל מקרי הקצה והמלצת אם בונים מודל.

### src/part2_tokenizers
- analyze_tokenizers.py: לכל מודל מחשב משפחה, גודל אוצר, טוקנים מיוחדים, אסטרטגיית גבולות מילה, אסטרטגיית בתים, וממוצע טוקנים למילה באנגלית ובעברית, ועמודת בונוס של מחלקת ה-backend. כותב את outputs/tokenizers.csv.
- tokenization_diff.py: מחפש טקסט אנגלי שממקסם את מספר הפיצולים השונים בין המודלים, סופר הסכמה, ומסביר את שיטת המדידה. כותב את report/sections/part2_diff.md ואת outputs/tokenization_diff_detail.json.

### src/part3_decoding
- identify_hebrew_tokens.py: סורק את אוצר המילים של Qwen ו-Mistral ובונה את רשימת הטוקנים המותרים בעברית, כולל טיפול בטוקני byte fallback. כותב את שני קבצי hebrew_allowed_tokens.
- constrained_decode.py: מחלקת LogitsProcessor שממסכת לוגיטים אסורים למינוס אינסוף ומוסיפה EOS ו-pad בזמן ריצה.
- run_decoding.py: טוען את שני המודלים ומריץ 10 שאילתות בשתי גרסאות, מוגבלת ולא מוגבלת. כותב את outputs/decoding_outputs.jsonl. דורש GPU.

### src/part4_finetuning
- make_data.py: יוצר את נתוני האימון, מסנן דליפה מול 20 קלטי ההערכה ומסנן שפה. כותב את data/train/train.jsonl. במצב מקוון משתמש ב-GPT. דגל offline-- לזרע ידני.
- train_lora.py: אימון LoRA על Qwen2.5-1.5B-Instruct, עם מיסוך טוקני הפרומפט. שומר את המתאם ב-outputs/lora_adapter. דורש GPU.
- evaluate.py: מריץ בסיס מול מכוונן על 20 הקלטים, מחשב אחוז עברית ופסק דין, וכותב את outputs/eval_outputs.jsonl. דורש GPU.

## בדיקות ודוח
- tests/test_core.py: בדיקות יחידה מהירות וללא רשת על זיהוי העברית, מפענח הבתים, ושלמות סט ההערכה (אין דליפה, 20 קלטים ייחודיים).
- report/build_report.py: מרכיב דוח HTML ו-PDF עם שמות ות.ז. בראש, תוכן עניינים מקושר, טבלאות, תרשים יחס MLP לרוחב, וכל פלטי ה-jsonl. אם weasyprint לא מותקן, נוצר HTML שאפשר להדפיס ל-PDF.
- report/students.json: מלא כאן שמות ות.ז. לפני בניית הדוח.
- report/sections: ניתוחי הפרוזה לכל חלק, נוצרים אוטומטית על ידי הסקריפטים.

## איפה התוצרים
כל התוצרים הנדרשים נכתבים ל-outputs:
- architecture.csv, tokenizers.csv
- hebrew_allowed_tokens_qwen.json, hebrew_allowed_tokens_mistral.json
- tokenization_diff_detail.json
- decoding_outputs.jsonl (חלק 3, GPU)
- eval_outputs.jsonl ו-lora_adapter (חלק 4, GPU)
והדוח הסופי ב-report/report.html או report/report.pdf.

## מה כבר מוכן ומה דורש GPU
מוכן ואומת על CPU: כל הקוד, שתי טבלאות ה-CSV עם כל עשרת המודלים מלאים (כולל Llama דרך המראה הציבורי), שתי רשימות הטוקנים המותרים בעברית, ניתוחי חלקים 1 ו-2, התרשים, והדוח שמתאסף מהזמין. דורש את ה-GPU שלך: decoding_outputs.jsonl ו-eval_outputs.jsonl יחד עם lora_adapter, דרך make p3 ו-make p4.

## פתרון תקלות
- שורת Llama ריקה: כבר נפתר דרך מראה ציבורי. אם תרצה ערכים מהמאגר הרשמי, הגדר HF_TOKEN מאושר והרץ make p1 ו-make p2 מחדש.
- שגיאת זיכרון ב-GPU: הקטן batch או max-new-tokens דרך הדגלים של הסקריפטים (ראה help-- בכל מודול).
- אזהרות unresolved-import ב-ty עבור torch או peft או openai: צפויות בסביבת CPU, אינן שגיאות, ונעלמות אחרי uv sync עם ה-extras.
