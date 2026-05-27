# מסמך איפיון מלא - מטלה 2: ארכיטקטורה, טוקנייזרים, דקודינג ופיין-טיונינג

מסמך זה הוא מקור האמת היחיד לפרויקט. הוא נכתב כדי ש-Claude Code יוכל לעבוד ממנו מקצה לקצה: להתקין את הסביבה, להריץ את כל החלקים, לשמור על תקן האיכות, ולבנות את הדוח המסכם. קרא אותו במלואו לפני שאתה מתחיל לעבוד.

## 0. תקן האיכות (דרישת חובה)
הקוד חייב לעמוד בתקן איכותי עם uv, Ruff ו-ty. זו דרישה קשיחה: כל שינוי בקוד חייב להשאיר את שער האיכות ירוק.

הכלים (כולם של Astral, מוגדרים כולם ב-pyproject.toml):
- uv: מנהל הסביבה והתלויות. כל פקודה רצה דרך uv run, בלי הפעלת venv ידנית.
- Ruff: לינטר ופורמטר (כולל מיון imports ומודרניזציה).
- ty: בדיקת טיפוסים סטטית.
- pytest: בדיקות יחידה מהירות (offline).

שער האיכות, פקודה אחת:
```
make check
```
שמריץ בפועל:
```
uv run ruff check .            # לינט, חייב לעבור נקי
uv run ruff format --check .   # פורמט, חייב להיות תקין
uv run ty check                # טיפוסים, אפס שגיאות
uv run pytest -q               # בדיקות, חייבות לעבור
```

הגדרת "גמור" (Definition of Done) לכל שינוי קוד:
1. ruff check עובר עם "All checks passed".
2. ruff format --check עובר בלי קבצים לא מפורמטים.
3. ty check מסתיים באפס שגיאות (errors). אזהרות unresolved-import עבור torch, peft ו-openai הן תקינות ומכוונות כל עוד ה-extras של GPU לא הותקנו; הן נעלמות אחרי uv sync --all-extras.
4. pytest עובר.

מוסכמות קוד:
- אין שימוש בתווי em-dash או en-dash בפרוזה. במחרוזות נתונים שבהן תו מקף הוא נתון לגיטימי, יש noqa ממוקד עם הסבר.
- ספריות ML עם טיפוסים חלקיים (transformers, torch, peft) ממוטפסות בגבול הקריאה כ-Any, כדי לא לייצר false positives ב-ty.
- imports כבדים (torch, peft, openai) נטענים בעצלתיים בתוך הפונקציות, כדי שחלקים 1 ו-2 ירוצו על CPU בלבד.
- כל סקריפט הוא idempotent: מותר להריץ שוב, הוא דורס רק את התוצר של עצמו.

## 1. מה צריך לספק (וזה הכל)
שני סודות בלבד, דרך משתני סביבה או קובץ .env (ראה .env.example):
- HF_TOKEN: טוקן Hugging Face עם הרשאת קריאה. נדרש להורדת קונפיגים וטוקנייזרים, ובמיוחד עבור המודל הנעול meta-llama/Llama-3.1-8B-Instruct (יש לאשר את הרישיון בעמוד המודל ב-Hub לפני ההרצה).
- TOKEN_KEY: מפתח ל-GPT ליצירת דאטה לפיין-טיונינג. המודל הדיפולטי הוא gpt-5.4-mini, וניתן לשינוי דרך OPENAI_MODEL. השם OPENAI_API_KEY מתקבל גם הוא.

הכל כבר כתוב ועובר את שער האיכות. אין צורך לכתוב קוד חדש, רק להתקין, להריץ ולבנות את הדוח.

## 2. התקנה
הפרויקט מכוון לפייתון 3.13 (requires-python >=3.13, וגם ruff ו-ty מוגדרים ל-3.13).
```
curl -LsSf https://astral.sh/uv/install.sh | sh   # אם uv לא מותקן
uv python install 3.13                             # uv ידאג לגרסה הנכונה
make setup                                         # uv sync --group dev (deps CPU + כלי פיתוח)
cp .env.example .env                               # ומלא HF_TOKEN ו-TOKEN_KEY
```
עבור חלקים 3 ו-4 שדורשים GPU, התקן גם את ה-extras:
```
uv sync --group dev --extra gpu --extra data --extra report
```
ב-Linux וב-Windows, ה-extra של gpu מושך את torch מתוך אינדקס ה-CUDA 13.0 הרשמי של PyTorch (pytorch-cu130) שמוגדר ב-pyproject.toml תחת tool.uv.sources. ב-macOS האינדקס מתעלם ומותקנת גרסת ברירת המחדל.

## 3. מבנה הפרויקט
```
llm-ass2/
  pyproject.toml            מקור האמת: תלויות, dependency-groups, הגדרות ruff ו-ty
  uv.lock                   נעילת גרסאות מלאה
  Makefile                  setup, check, fmt, p1..p4, report, all
  .pre-commit-config.yaml   hooks ל-ruff ו-ty
  .github/workflows/ci.yml  שער איכות ב-CI
  config.py                 רישום מרכזי: מודלים, נתיבים, שמות תוצרים, טבלאות ידע
  data/
    fixtures.py             טקסטים לדגימה, 10 קלטי הערכה שסופקו, 10 משלנו, שאילתות דקודינג
    train/train.jsonl       דאטה אימון (נוצר ב-Part 4)
  src/
    common/token_utils.py   זיהוי עברית, פענוח byte-level מול sentencepiece
    part1_architecture/     extract_architecture.py, analyze_architecture.py
    part2_tokenizers/       analyze_tokenizers.py, tokenization_diff.py
    part3_decoding/         identify_hebrew_tokens.py, constrained_decode.py, run_decoding.py
    part4_finetuning/       make_data.py, train_lora.py, evaluate.py
  tests/test_core.py        בדיקות יחידה offline
  outputs/                  כל התוצרים הנדרשים נכתבים לכאן
  report/
    sections/               ניתוחי פרוזה לכל חלק
    build_report.py         מרכיב את הדוח הסופי (HTML ו-PDF)
    students.json           שמות ות.ז. של המגישים (למלא לפני בניית הדוח)
  scripts/run_all.sh        מריץ את כל הפייפליין דרך uv
```

## 4. מיפוי דרישות לתוצרים

### Part 1 - בחירות ארכיטקטוניות
הדרישה: עבור 10 המודלים, לחלץ מספר שכבות ורוחב, ראשי attention, כל הגדלים (MLP, MoE, attention, embedding), קידוד מיקום וגודל מקסימלי והיפר-פרמטרים, פונקציות אקטיבציה, נורמליזציה (pre או post), ומאפיינים נוספים. לדווח על שיטת החילוץ, אי-ודאויות, סיכום ההבדלים, ומגמות וניתוח כולל "אם היית בונה מודל".
- קוד: src/part1_architecture/extract_architecture.py מוריד את config.json מכל מודל, מחשב head_dim עם מודעות ל-MLA של DeepSeek, kv_group_size, אומדן פרמטרים, ופרטי MoE. analyze_architecture.py מחשב את המגמות וכותב את ניתוח הפרוזה.
- תוצר: outputs/architecture.csv עם העמודות בדיוק כנדרש: model_id, hidden_size, num_layers, num_attention_heads, num_kv_heads, mlp_size, activation, norm_type, position_encoding, context_length, vocab_size, moe_details. ערכים חסרים מסומנים NA. עמודות העשרה נוספות (head_dim, kv_group_size, rope_theta, tie_word_embeddings, params_billions_approx, source) מתווספות אחרי העמודות הנדרשות.
- ניתוח: report/sections/part1_analysis.md.

### Part 2 - טוקנייזרים
הדרישה: לכל מודל, מספר הטוקנים, אסטרטגיית גבולות מילה, טוקנים שאינם מילים, וממוצע טוקנים למילה באנגלית ובעברית (עם הסבר על שיטת המדידה וספירת המילים). בנוסף, למצוא טקסט באנגלית שמתפצל שונה בין לפחות 3 מודלים, ולספור על כמה מ-7 הנותרים יש הסכמה.
- קוד: src/part2_tokenizers/analyze_tokenizers.py ו-tokenization_diff.py.
- תוצר: outputs/tokenizers.csv עם העמודות: model_id, tokenizer_type, vocab_size, special_tokens, word_boundary_strategy, byte_fallback_or_byte_level, avg_tokens_per_english_word, avg_tokens_per_hebrew_word.
- ניתוח והפיצול הנבחר: report/sections/part2_diff.md ו-outputs/tokenization_diff_detail.json.

### Part 3 - דקודינג מוגבל (עברית בלבד)
הדרישה: לטעון את Qwen2.5-7B-Instruct ו-Mistral-7B-Instruct-v0.3, לזהות את אוסף הטוקנים שמותר להם להופיע בפלט עברי (כולל מספרים וסימני פיסוק, ללא שפות אחרות), לממש דקודינג מוגבל שמאפשר רק טוקנים אלה, ולהריץ 10 שאילתות באנגלית על שני המודלים בשתי גרסאות: לא מוגבל ומוגבל.
- קוד: identify_hebrew_tokens.py בונה את אוסף הטוקנים, constrained_decode.py מממש LogitsProcessor שממסך לוגיטים אסורים למינוס אינסוף ומוסיף EOS ו-pad בזמן ריצה, run_decoding.py מריץ את שני המודלים.
- תוצרים: outputs/hebrew_allowed_tokens_qwen.json ו-outputs/hebrew_allowed_tokens_mistral.json במבנה {"model_id":..., "allowed_token_ids":[...]}, וכן outputs/decoding_outputs.jsonl עם שורה לכל זוג שאילתה ומודל: prompt, model, unconstrained_output, constrained_output.

### Part 4 - פיין-טיונינג (קלט אנגלית, פלט עברית)
הדרישה: לכוונן את Qwen2.5-1.5B-Instruct כך שיענה על שאילתות באנגלית בעברית, עם תשובה רלוונטית ואמיתית ולא מחרוזת קבועה. ליצור דאטה (LoRA עם peft). להעריך על 10 הקלטים שסופקו ועוד 10 משלנו, כאשר האימון אינו כולל אף אחד מ-20 הקלטים.
- קוד: make_data.py יוצר את הדאטה ומסנן דליפה מול 20 קלטי ההערכה, train_lora.py מאמן LoRA, evaluate.py מריץ בסיס מול מכוונן.
- תוצר: outputs/eval_outputs.jsonl עם שורה לכל קלט: prompt, base_output, finetuned_output, notes.
- ניתוח: report/sections/part4_method.md.

### הדוח הסופי
report/build_report.py מרכיב דוח HTML ו-PDF עם שמות ות.ז. בראש, תוכן עניינים מקושר, וכל ארבעת החלקים כולל הטבלאות ופלטי ה-jsonl. יש למלא את report/students.json לפני הבנייה.

## 5. סדר ההרצה
חלקים 1 ו-2 רצים על CPU. חלקים 3 ו-4 דורשים GPU (T4 ב-Colab מספיק).
```
make check     # ודא ששער האיכות ירוק לפני ואחרי כל שינוי
make p1        # architecture.csv + ניתוח
make p2        # tokenizers.csv + diff
make p3        # אוספי טוקנים עבריים + decoding_outputs.jsonl   (GPU)
make p4        # train.jsonl + אימון LoRA + eval_outputs.jsonl   (GPU)
make report    # report.html ו-report.pdf
make report-docx # report.docx (מסמך ההגשה ב-Word)
```
או בפקודה אחת: bash scripts/run_all.sh.

## 6. תהליך העבודה ל-Claude Code
1. הרץ make setup ואז make check. ודא שהכל ירוק לפני שאתה נוגע במשהו.
2. הרץ את החלקים לפי הסדר. אם צעד נכשל, תקן את שורש הבעיה ולא את הסימפטום.
3. אחרי כל עריכת קוד, הרץ שוב make check. אל תשאיר את השער אדום.
4. מלא את report/students.json, ואז make report.
5. ודא שאין בדוח בלוקים של "טרם נוצר" אחרי שכל הצעדים רצו.

## 7. רשימת בונוסים ל-100 מלא
- תקן איכות מלא: uv לניהול, Ruff נקי, ty באפס שגיאות, בדיקות pytest, pre-commit ו-CI.
- עמודות העשרה ב-architecture.csv מעבר לנדרש, עם עמודת source לשקיפות החילוץ.
- טיפול נכון במקרי הקצה: MLA של DeepSeek-V3, MoE עם expert משותף ו-3 שכבות dense ראשונות, post-norm ו-QK-norm של OLMo-2, partial rotary של Phi-4-mini, ו-head_dim לא סטנדרטי של Falcon3.
- בחירת טקסט הפיצול שממקסם את מספר הפיצולים השונים בין המודלים, עם ספירת הסכמה מדויקת.
- מסווג עברית שמטפל גם בטוקני byte fallback של SentencePiece (Mistral) ולא רק בטוקני byte-level (Qwen).
- הפרדת naive מול optimal, סינון דליפה קשיח, וסינון שפה בדאטה של Part 4.
- דוח מנווט עם תוכן עניינים, טבלאות RTL לעברית, והרצה חוזרת בטוחה של כל צעד.

## 8. הערות מימוש
- שורת Llama מלאה אוטומטית גם בלי טוקן: עבור מודל נעול ללא HF_TOKEN, הקוד נופל למראה ציבורי מאומת (NousResearch) שמכיל את אותו config וטוקנייזר, כך שאין שורת NA. עם טוקן מאושר, ניתן להריץ מחדש make p1 ו-make p2 כדי למשוך מהמאגר הרשמי.
- ערכי NA שנותרו הם לגיטימיים בלבד: moe_details עבור מודלים צפופים, ו-params_billions_approx עבור DeepSeek-V3 (שהנוסחה הצפופה אינה ישימה לו כ-MoE).
- שני מסמכי הסבר נלווים: ANSWERS.md עונה על כל שאלות הניתוח בכל ארבעת החלקים בצורה מנומקת ומבוססת נתונים, ו-FILES_GUIDE.md מסביר כל קובץ ואיך מריצים הכל.
- חלק מהתוצרים כבר נוצרו מראש (architecture.csv, tokenizers.csv, אוספי הטוקנים, ה-diff) כך שניתן לבדוק את הפלט מיד. שני התוצרים שדורשים GPU (decoding_outputs.jsonl ו-eval_outputs.jsonl) יסומנו בדוח כ"טרם נוצר" עד שתריץ אותם.
- אזהרות unresolved-import ב-ty עבור torch, peft ו-openai צפויות בסביבת CPU. הן אינן שגיאות ואינן מפילות את השער. כדי לראות פלט ירוק לחלוטין, הרץ uv sync --all-extras.
