"""
Assemble the final submission report.

Reads every deliverable that exists (architecture.csv, tokenizers.csv, the two
Hebrew-allowed-token JSON files, decoding_outputs.jsonl, eval_outputs.jsonl) plus
the prose analysis sections under report/sections/, and renders a single navigable
HTML document. If weasyprint is installed it is also rendered to report/report.pdf;
otherwise the HTML is written and the user can print-to-PDF from a browser.

The report is robust to missing pieces: any deliverable that has not been produced
yet (for example the GPU-only Part 3 / Part 4 outputs before Claude Code runs them)
is shown as a clearly marked "not yet generated" placeholder rather than crashing.

Student identity is read from report/students.json if present, else a visible
placeholder is inserted at the very top so it cannot be missed.

Usage:
    python -m report.build_report
"""

from __future__ import annotations

import contextlib
import csv
import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

SECTIONS = C.REPORT_DIR / "sections"
STUDENTS_JSON = C.REPORT_DIR / "students.json"
HTML_OUT = C.REPORT_DIR / "report.html"
PDF_OUT = C.REPORT_DIR / "report.pdf"


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def esc(x) -> str:
    return html.escape(str(x), quote=False)


def read_students() -> list[dict]:
    if STUDENTS_JSON.exists():
        try:
            data = json.loads(STUDENTS_JSON.read_text(encoding="utf-8"))
            if isinstance(data, list) and data:
                return data
        except Exception:
            pass
    return [
        {"name": "<< Student 1 full name >>", "id": "<< Student 1 ID >>"},
        {"name": "<< Student 2 full name >>", "id": "<< Student 2 ID >>"},
    ]


def csv_to_table(path: Path) -> str:
    if not path.exists():
        return placeholder(f"{path.name} not generated yet")
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        return placeholder(f"{path.name} is empty")
    head, *body = rows
    th = "".join(f"<th>{esc(c)}</th>" for c in head)
    trs = []
    for r in body:
        tds = "".join(f"<td>{esc(c)}</td>" for c in r)
        trs.append(f"<tr>{tds}</tr>")
    return (
        f'<div class="tablewrap"><table class="data"><thead><tr>{th}</tr>'
        f"</thead><tbody>{''.join(trs)}</tbody></table></div>"
    )


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line:
            with contextlib.suppress(json.JSONDecodeError):
                out.append(json.loads(line))
    return out


def md_to_html(path: Path) -> str:
    """Minimal Markdown to HTML (headings, lists, bold, code, paragraphs)."""
    if not path.exists():
        return placeholder(f"{path.name} not generated yet")
    text = path.read_text(encoding="utf-8")
    try:
        import markdown  # type: ignore

        return markdown.markdown(text, extensions=["tables", "fenced_code"])
    except Exception:
        pass
    # Fallback: tiny converter good enough for our own section files.
    lines = text.splitlines()
    out, in_ul = [], False

    def close_ul():
        nonlocal in_ul
        if in_ul:
            out.append("</ul>")
            in_ul = False

    for ln in lines:
        s = ln.rstrip()
        if s.startswith("### "):
            close_ul()
            out.append(f"<h4>{esc(s[4:])}</h4>")
        elif s.startswith("## "):
            close_ul()
            out.append(f"<h3>{esc(s[3:])}</h3>")
        elif s.startswith("# "):
            close_ul()
            out.append(f"<h2>{esc(s[2:])}</h2>")
        elif s.startswith("- ") or s.startswith("* "):
            if not in_ul:
                out.append("<ul>")
                in_ul = True
            out.append(f"<li>{esc(s[2:])}</li>")
        elif not s:
            close_ul()
        else:
            close_ul()
            out.append(f"<p>{esc(s)}</p>")
    close_ul()
    body = "\n".join(out)
    # crude **bold** and `code`
    import re

    body = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", body)
    body = re.sub(r"`(.+?)`", r"<code>\1</code>", body)
    return body


def placeholder(msg: str) -> str:
    return f'<div class="placeholder">[ {esc(msg)} ]</div>'


# --------------------------------------------------------------------------- #
# Section builders
# --------------------------------------------------------------------------- #
def section_part3() -> str:
    blocks = []
    for label, path in [
        ("Qwen2.5-7B-Instruct", C.HEBREW_TOKENS_QWEN),
        ("Mistral-7B-Instruct-v0.3", C.HEBREW_TOKENS_MISTRAL),
    ]:
        if path.exists():
            d = json.loads(path.read_text(encoding="utf-8"))
            n = len(d.get("allowed_token_ids", []))
            blocks.append(
                f"<p><strong>{esc(label)}</strong>: "
                f"{n:,} allowed token ids "
                f"(file <code>{esc(path.name)}</code>).</p>"
            )
        else:
            blocks.append(placeholder(f"{path.name} not generated yet"))

    rows = read_jsonl(C.DECODING_OUTPUTS)
    if rows:
        trs = []
        for r in rows:
            trs.append(
                "<tr>"
                f"<td class='prompt'>{esc(r.get('prompt', ''))}</td>"
                f"<td>{esc(r.get('model', ''))}</td>"
                f"<td class='rtl'>{esc(r.get('unconstrained_output', ''))}</td>"
                f"<td class='rtl'>{esc(r.get('constrained_output', ''))}</td>"
                "</tr>"
            )
        table = (
            '<div class="tablewrap"><table class="data"><thead><tr>'
            "<th>prompt</th><th>model</th><th>unconstrained</th>"
            "<th>constrained (Hebrew-only)</th></tr></thead>"
            f"<tbody>{''.join(trs)}</tbody></table></div>"
        )
    else:
        table = placeholder("decoding_outputs.jsonl not generated yet (GPU step)")
    return "\n".join(blocks) + table


def mlp_ratio_chart() -> str:
    """Inline SVG horizontal bar chart of mlp_size / hidden_size per model."""
    if not C.ARCHITECTURE_CSV.exists():
        return placeholder("architecture.csv not generated yet")
    with C.ARCHITECTURE_CSV.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    data = []
    for r in rows:
        try:
            h = float(r["hidden_size"])
            mlp = float(r["mlp_size"])
            if h > 0 and mlp > 0:
                data.append((r["model_id"].split("/")[-1], mlp / h))
        except (ValueError, KeyError, ZeroDivisionError):
            continue
    if not data:
        return placeholder("no numeric MLP/hidden data")
    data.sort(key=lambda x: x[1], reverse=True)
    row_h, top, left, width = 26, 12, 210, 360
    max_ratio = max(v for _, v in data)
    height = top * 2 + row_h * len(data)
    bars = []
    for i, (name, ratio) in enumerate(data):
        y = top + i * row_h
        w = max(2, int(width * ratio / max_ratio))
        bars.append(
            f'<text x="{left - 8}" y="{y + 14}" text-anchor="end" '
            f'font-size="11" fill="#333">{esc(name)}</text>'
            f'<rect x="{left}" y="{y + 4}" width="{w}" height="16" rx="2" '
            f'fill="#2d6cdf"/>'
            f'<text x="{left + w + 5}" y="{y + 16}" font-size="10" '
            f'fill="#333">{ratio:.2f}x</text>'
        )
    return (
        f'<svg viewBox="0 0 {left + width + 60} {height}" '
        f'xmlns="http://www.w3.org/2000/svg" style="max-width:760px;width:100%">'
        f"{''.join(bars)}</svg>"
        '<p class="meta">היחס mlp_size חלקי hidden_size. ערכים סביב 2.67 (8/3) הם '
        "SwiGLU קלאסי; ערכים גבוהים יותר מרחיבים את ה-MLP להוספת קיבולת.</p>"
    )


def section_part4() -> str:
    rows = read_jsonl(C.EVAL_OUTPUTS)
    if not rows:
        return placeholder("eval_outputs.jsonl not generated yet (GPU step)")
    trs = []
    for r in rows:
        trs.append(
            "<tr>"
            f"<td class='prompt'>{esc(r.get('prompt', ''))}</td>"
            f"<td class='rtl'>{esc(r.get('base_output', ''))}</td>"
            f"<td class='rtl'>{esc(r.get('finetuned_output', ''))}</td>"
            f"<td class='notes'>{esc(r.get('notes', ''))}</td>"
            "</tr>"
        )
    return (
        '<div class="tablewrap"><table class="data"><thead><tr>'
        "<th>prompt</th><th>base output</th><th>fine-tuned output</th>"
        "<th>notes</th></tr></thead>"
        f"<tbody>{''.join(trs)}</tbody></table></div>"
    )


# --------------------------------------------------------------------------- #
# Page assembly
# --------------------------------------------------------------------------- #
CSS = """
body { font-family: 'Segoe UI', Arial, sans-serif; color: #1a1a1a; line-height: 1.5;
       max-width: 1100px; margin: 0 auto; padding: 24px; }
h1 { font-size: 26px; border-bottom: 3px solid #333; padding-bottom: 8px; }
h2 { font-size: 21px; margin-top: 34px; border-bottom: 1px solid #bbb; padding-bottom: 4px; }
h3 { font-size: 17px; margin-top: 22px; }
h4 { font-size: 15px; margin-top: 16px; color: #333; }
.rtl { direction: rtl; text-align: right; }
.students { background:#f4f6fb; border:1px solid #ccd; border-radius:8px; padding:12px 16px; }
.students .rtl { direction: rtl; text-align: right; }
.toc { background:#fafafa; border:1px solid #ddd; border-radius:8px; padding:10px 18px; }
.toc a { text-decoration:none; color:#1849a9; }
.tablewrap { overflow-x:auto; margin:12px 0; }
table.data { border-collapse: collapse; width:100%; font-size:12px; }
table.data th, table.data td { border:1px solid #ccc; padding:5px 7px; vertical-align:top; }
table.data th { background:#2d3a50; color:#fff; text-align:left; }
table.data tr:nth-child(even) td { background:#f7f8fa; }
td.prompt { max-width:230px; }
td.notes { max-width:230px; font-size:11px; color:#444; }
code { background:#f0f0f0; padding:1px 4px; border-radius:3px; font-size:90%; }
.placeholder { background:#fff7e6; border:1px dashed #e0a800; color:#7a5b00;
               padding:8px 12px; border-radius:6px; margin:8px 0; }
.meta { color:#666; font-size:12px; }
"""

TOC = [
    ("part1", "Part 1 - Architectural Choices"),
    ("part2", "Part 2 - Tokenizers"),
    ("part3", "Part 3 - Constrained Decoding (Hebrew-only)"),
    ("part4", "Part 4 - Fine-tuning (English in, Hebrew out)"),
]


def build_html() -> str:
    students = read_students()
    student_rows = "".join(
        f"<div>{esc(s.get('name', ''))} (ID: {esc(s.get('id', ''))})</div>" for s in students
    )
    toc_items = "".join(f'<li><a href="#{sid}">{esc(t)}</a></li>' for sid, t in TOC)

    parts = []
    parts.append(f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>LLM Class - Assignment 2 Report</title><style>{CSS}</style></head><body>""")
    parts.append("<h1>LLM Class - Assignment 2</h1>")
    parts.append(
        "<p class='meta'>Architectural Choices, Tokenizers, Constrained "
        "Decoding, and Fine-tuning</p>"
    )
    parts.append(f'<div class="students"><strong>Submitted by:</strong>{student_rows}</div>')
    parts.append(f'<div class="toc"><strong>Table of Contents</strong><ul>{toc_items}</ul></div>')

    parts.append('<h2 id="part1">Part 1 - Architectural Choices</h2>')
    parts.append("<h3>Architecture table (architecture.csv)</h3>")
    parts.append(csv_to_table(C.ARCHITECTURE_CSV))
    parts.append("<h3>MLP-to-hidden ratio per model</h3>")
    parts.append(mlp_ratio_chart())
    parts.append("<h3>Extraction method, uncertainties, trends and analysis</h3>")
    parts.append(md_to_html(SECTIONS / "part1_analysis.md"))

    parts.append('<h2 id="part2">Part 2 - Tokenizers</h2>')
    parts.append("<h3>Tokenizer table (tokenizers.csv)</h3>")
    parts.append(csv_to_table(C.TOKENIZERS_CSV))
    parts.append("<h3>Text tokenized differently across models</h3>")
    parts.append(md_to_html(SECTIONS / "part2_diff.md"))

    parts.append('<h2 id="part3">Part 3 - Constrained Decoding</h2>')
    parts.append(section_part3())

    parts.append('<h2 id="part4">Part 4 - Fine-tuning</h2>')
    parts.append(md_to_html(SECTIONS / "part4_method.md"))
    parts.append("<h3>Evaluation results on the 20 held-out inputs (eval_outputs.jsonl)</h3>")
    parts.append(section_part4())

    parts.append("</body></html>")
    return "\n".join(parts)


def _pdf_via_weasyprint(html_doc: str) -> tuple[bool, str]:
    """Render PDF with WeasyPrint. Works out of the box on Linux/macOS;
    on Windows it needs the GTK runtime, which is almost never installed,
    so we skip it there entirely to avoid the long stderr banner weasyprint
    prints when its native libs cannot load. Set FORCE_WEASYPRINT=1 to
    bypass that skip."""
    import os
    import sys

    if sys.platform == "win32" and not os.environ.get("FORCE_WEASYPRINT"):
        return False, "weasyprint: skipped on Windows (needs GTK runtime); using playwright"
    try:
        from weasyprint import HTML  # type: ignore

        HTML(string=html_doc, base_url=str(C.REPORT_DIR)).write_pdf(str(PDF_OUT))
        return True, "weasyprint"
    except Exception as e:
        return False, f"weasyprint: {type(e).__name__}: {e}"


def _pdf_via_playwright(html_path: Path) -> tuple[bool, str]:
    """Render PDF with headless Chromium via Playwright. Cross-platform; the
    first run downloads ~150 MB of Chromium. Auto-installs it on demand."""
    try:
        from playwright.sync_api import sync_playwright  # type: ignore
    except ImportError as e:
        return False, f"playwright not installed: {e}"
    try:
        import subprocess
        import sys

        def _render() -> None:
            with sync_playwright() as pw:
                browser = pw.chromium.launch()
                page = browser.new_page()
                page.goto(html_path.as_uri(), wait_until="load")
                page.pdf(
                    path=str(PDF_OUT),
                    format="A4",
                    margin={"top": "16mm", "right": "16mm", "bottom": "16mm", "left": "16mm"},
                    print_background=True,
                )
                browser.close()

        try:
            _render()
        except Exception as e:
            # If Chromium is missing, install it once and retry.
            msg = str(e).lower()
            if "executable doesn't exist" in msg or "playwright install" in msg:
                print("playwright: Chromium not found, downloading once...")
                subprocess.run(
                    [sys.executable, "-m", "playwright", "install", "chromium"],
                    check=True,
                )
                _render()
            else:
                raise
        return True, "playwright"
    except Exception as e:
        return False, f"playwright: {type(e).__name__}: {e}"


def main() -> None:
    doc = build_html()
    HTML_OUT.write_text(doc, encoding="utf-8")
    print(f"Wrote {HTML_OUT}")

    errors: list[str] = []
    ok, info = _pdf_via_weasyprint(doc)
    if ok:
        print(f"Wrote {PDF_OUT} (via {info})")
        return
    errors.append(info)

    ok, info = _pdf_via_playwright(HTML_OUT)
    if ok:
        print(f"Wrote {PDF_OUT} (via {info})")
        return
    errors.append(info)

    print("PDF step skipped. Tried backends:")
    for e in errors:
        print(f"  - {e}")
    print("HTML is ready; open it in a browser and Print > Save as PDF.")


if __name__ == "__main__":
    main()
