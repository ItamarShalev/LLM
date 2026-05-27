// Build the Assignment 2 submission as a Word (.docx) document.
//
// Reads the same real outputs as the HTML report (architecture.csv,
// tokenizers.csv, the Hebrew allowed-token JSONs, the markdown analysis
// sections, and the Part 3/Part 4 jsonl files when present) and produces a
// single navigable Word document at report/report.docx.
//
// Any deliverable that has not been generated yet (the GPU-only Part 3 / Part 4
// outputs before they are run) is shown as a clearly marked placeholder rather
// than failing. Student identity is read from report/students.json.
//
// Usage:  node report/build_report_docx.js

const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  HeadingLevel, AlignmentType, WidthType, BorderStyle, ShadingType,
  TableOfContents, PageBreak, LevelFormat,
} = require("docx");

const ROOT = path.resolve(__dirname, "..");
const OUT = path.join(ROOT, "outputs");
const SECTIONS = path.join(__dirname, "sections");

// ---------------------------------------------------------------------------
// Small IO helpers
// ---------------------------------------------------------------------------
function readText(p) {
  try { return fs.readFileSync(p, "utf-8"); } catch { return null; }
}
function parseCsvLine(line) {
  const out = [];
  let cur = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (inQuotes) {
      if (ch === '"') {
        if (line[i + 1] === '"') { cur += '"'; i++; } else { inQuotes = false; }
      } else { cur += ch; }
    } else if (ch === '"') {
      inQuotes = true;
    } else if (ch === ",") {
      out.push(cur); cur = "";
    } else { cur += ch; }
  }
  out.push(cur);
  return out;
}
function readCsv(p) {
  const t = readText(p);
  if (!t) return null;
  const lines = t.replace(/\r/g, "").split("\n").filter((l) => l.length);
  return lines.map(parseCsvLine);
}
function readJsonl(p) {
  const t = readText(p);
  if (!t) return [];
  return t.replace(/\r/g, "").split("\n").filter((l) => l.trim())
    .map((l) => { try { return JSON.parse(l); } catch { return null; } })
    .filter(Boolean);
}
function readStudents() {
  const t = readText(path.join(__dirname, "students.json"));
  if (t) { try { const d = JSON.parse(t); if (Array.isArray(d) && d.length) return d; } catch { /* fall through */ } }
  return [
    { name: "<< Student 1 full name >>", id: "<< Student 1 ID >>" },
    { name: "<< Student 2 full name >>", id: "<< Student 2 ID >>" },
  ];
}

// ---------------------------------------------------------------------------
// Styling constants
// ---------------------------------------------------------------------------
const NAVY = "2D3A50";
const HEADER_FILL = "2D3A50";
const ZEBRA_FILL = "F2F4F8";
const PLACEHOLDER_FILL = "FFF7E6";
const CONTENT_WIDTH = 9360; // US Letter, 1 inch margins
const thinBorder = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const cellBorders = { top: thinBorder, bottom: thinBorder, left: thinBorder, right: thinBorder };
const cellMargins = { top: 60, bottom: 60, left: 100, right: 100 };

function placeholder(msg) {
  return new Paragraph({
    shading: { fill: PLACEHOLDER_FILL, type: ShadingType.CLEAR },
    spacing: { before: 80, after: 80 },
    children: [new TextRun({ text: `[ ${msg} ]`, italics: true, color: "7A5B00" })],
  });
}

// A reference callout that points to a deliverable file instead of dumping a
// very wide table inline. Summarizes the file (rows x columns + column list)
// so the reader knows exactly what it contains.
function fileReference(filename, whatItIs, csv) {
  const REF_FILL = "EAF1FB";
  const runs = [
    new TextRun({ text: "Deliverable file: ", bold: true }),
    new TextRun({ text: filename, bold: true, font: "Consolas", color: NAVY }),
  ];
  if (csv && csv.length > 1) {
    const nRows = csv.length - 1;
    const cols = csv[0];
    runs.push(new TextRun({ text: `  (${nRows} models x ${cols.length} columns)` }));
  }
  const para1 = new Paragraph({
    shading: { fill: REF_FILL, type: ShadingType.CLEAR },
    spacing: { before: 80, after: 0 },
    children: runs,
  });
  const lines = [new Paragraph({
    shading: { fill: REF_FILL, type: ShadingType.CLEAR },
    spacing: { after: csv && csv.length > 1 ? 0 : 80 },
    children: [new TextRun({ text: whatItIs })],
  })];
  if (csv && csv.length > 1) {
    lines.push(new Paragraph({
      shading: { fill: REF_FILL, type: ShadingType.CLEAR },
      spacing: { after: 80 },
      children: [
        new TextRun({ text: "Columns: ", italics: true }),
        new TextRun({ text: csv[0].join(", "), font: "Consolas", size: 16 }),
      ],
    }));
  }
  return [para1, ...lines];
}

// Minimal markdown to docx paragraphs: handles #/##/### headings, "- " bullets,
// **bold** inline, and plain paragraphs. Sufficient for our section files.
function mdToParagraphs(md) {
  if (!md) return [placeholder("section not generated yet")];
  const out = [];
  for (const raw of md.replace(/\r/g, "").split("\n")) {
    const line = raw.replace(/\s+$/, "");
    if (!line.trim()) continue;
    if (line.startsWith("### ")) {
      out.push(new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(line.slice(4))] }));
    } else if (line.startsWith("## ")) {
      out.push(new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(line.slice(3))] }));
    } else if (line.startsWith("# ")) {
      out.push(new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(line.slice(2))] }));
    } else if (line.startsWith("- ") || line.startsWith("* ")) {
      out.push(new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: inlineRuns(line.slice(2)) }));
    } else {
      out.push(new Paragraph({ spacing: { after: 120 }, children: inlineRuns(line) }));
    }
  }
  return out;
}

// Split a line on **bold** and `code` markers into TextRuns.
function inlineRuns(text) {
  const runs = [];
  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g);
  for (const p of parts) {
    if (!p) continue;
    if (p.startsWith("**") && p.endsWith("**")) {
      runs.push(new TextRun({ text: p.slice(2, -2), bold: true }));
    } else if (p.startsWith("`") && p.endsWith("`")) {
      runs.push(new TextRun({ text: p.slice(1, -1), font: "Consolas" }));
    } else {
      runs.push(new TextRun(p));
    }
  }
  return runs.length ? runs : [new TextRun(text)];
}

function h2(text, bookmark) {
  return new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(text)] });
}
function h3(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(text)] });
}

// ---------------------------------------------------------------------------
// Part-specific sections
// ---------------------------------------------------------------------------
function part3Section() {
  const blocks = [];
  for (const [label, file] of [
    ["Qwen2.5-7B-Instruct", "hebrew_allowed_tokens_qwen.json"],
    ["Mistral-7B-Instruct-v0.3", "hebrew_allowed_tokens_mistral.json"],
  ]) {
    const t = readText(path.join(OUT, file));
    if (t) {
      let n = 0;
      try { n = (JSON.parse(t).allowed_token_ids || []).length; } catch { /* ignore */ }
      blocks.push(new Paragraph({
        spacing: { after: 80 },
        children: [
          new TextRun({ text: `${label}: `, bold: true }),
          new TextRun(`${n.toLocaleString()} allowed token ids (file ${file}).`),
        ],
      }));
    } else {
      blocks.push(placeholder(`${file} not generated yet`));
    }
  }
  const rows = readJsonl(path.join(OUT, "decoding_outputs.jsonl"));
  if (rows.length) {
    blocks.push(...fileReference(
      "outputs/decoding_outputs.jsonl",
      `Full unconstrained vs constrained outputs for all ${rows.length} runs (10 queries x 2 models) are provided in this JSONL deliverable; each line has prompt, model, unconstrained_output and constrained_output.`,
      null));
  } else {
    blocks.push(placeholder("decoding_outputs.jsonl not generated yet (GPU step: run make p3)"));
  }
  return blocks;
}

function part4Section() {
  const blocks = mdToParagraphs(readText(path.join(SECTIONS, "part4_method.md")));
  blocks.push(h3("Evaluation results on the 20 held-out inputs"));
  const rows = readJsonl(path.join(OUT, "eval_outputs.jsonl"));
  if (rows.length) {
    blocks.push(...fileReference(
      "outputs/eval_outputs.jsonl",
      `Base vs fine-tuned answers for all ${rows.length} held-out inputs are provided in this JSONL deliverable; each line has prompt, base_output, finetuned_output and a notes field with the Hebrew-letter fraction and a verdict.`,
      null));
  } else {
    blocks.push(placeholder("eval_outputs.jsonl not generated yet (GPU step: run make p4)"));
  }
  return blocks;
}

// ---------------------------------------------------------------------------
// Assemble the document
// ---------------------------------------------------------------------------
function build() {
  const students = readStudents();
  const archCsv = readCsv(path.join(OUT, "architecture.csv"));
  const tokCsv = readCsv(path.join(OUT, "tokenizers.csv"));

  const children = [];
  children.push(new Paragraph({ heading: HeadingLevel.TITLE, children: [new TextRun("LLM Class - Assignment 2")] }));
  children.push(new Paragraph({
    spacing: { after: 120 },
    children: [new TextRun({ text: "Architectural Choices, Tokenizers, Constrained Decoding, and Fine-tuning", italics: true, color: "555555" })],
  }));

  children.push(new Paragraph({ children: [new TextRun({ text: "Submitted by:", bold: true })] }));
  for (const s of students) {
    children.push(new Paragraph({ spacing: { after: 20 }, children: [new TextRun(`${s.name} (ID: ${s.id})`)] }));
  }

  children.push(new Paragraph({ spacing: { before: 160 }, heading: HeadingLevel.HEADING_2, children: [new TextRun("Table of Contents")] }));
  children.push(new TableOfContents("Table of Contents", { hyperlink: true, headingStyleRange: "2-3" }));
  children.push(new Paragraph({ children: [new PageBreak()] }));

  // Part 1
  children.push(h2("Part 1 - Architectural Choices"));
  children.push(h3("Architecture table"));
  if (archCsv) {
    children.push(...fileReference(
      "outputs/architecture.csv",
      "The full architecture table for all ten models is provided as a separate CSV deliverable. It is too wide to embed legibly inline; the analysis below summarizes and interprets it.",
      archCsv));
  } else {
    children.push(placeholder("architecture.csv not generated yet"));
  }
  children.push(h3("Extraction method, uncertainties, trends and analysis"));
  children.push(...mdToParagraphs(readText(path.join(SECTIONS, "part1_analysis.md"))));

  // Part 2
  children.push(new Paragraph({ children: [new PageBreak()] }));
  children.push(h2("Part 2 - Tokenizers"));
  children.push(h3("Tokenizer table"));
  if (tokCsv) {
    children.push(...fileReference(
      "outputs/tokenizers.csv",
      "The full per-model tokenizer comparison is provided as a separate CSV deliverable, including the bonus tokenizer_backend column. The analysis below summarizes the key findings.",
      tokCsv));
  } else {
    children.push(placeholder("tokenizers.csv not generated yet"));
  }
  children.push(h3("Text tokenized differently across models, and measurement method"));
  children.push(...mdToParagraphs(readText(path.join(SECTIONS, "part2_diff.md"))));

  // Part 3
  children.push(new Paragraph({ children: [new PageBreak()] }));
  children.push(h2("Part 3 - Constrained Decoding (Hebrew-only)"));
  children.push(...part3Section());

  // Part 4
  children.push(new Paragraph({ children: [new PageBreak()] }));
  children.push(h2("Part 4 - Fine-tuning (English in, Hebrew out)"));
  children.push(...part4Section());

  const doc = new Document({
    styles: {
      default: { document: { run: { font: "Arial", size: 22 } } },
      paragraphStyles: [
        { id: "Title", name: "Title", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 44, bold: true, color: NAVY, font: "Arial" },
          paragraph: { spacing: { after: 120 } } },
        { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 30, bold: true, color: NAVY, font: "Arial" },
          paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 1,
            border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "BBBBBB", space: 2 } } } },
        { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 24, bold: true, color: "333333", font: "Arial" },
          paragraph: { spacing: { before: 160, after: 80 }, outlineLevel: 2 } },
      ],
    },
    numbering: {
      config: [
        { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "\u2022",
          alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 260 } } } }] },
      ],
    },
    sections: [{
      properties: { page: { size: { width: 12240, height: 15840 },
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
      children,
    }],
  });

  return doc;
}

const doc = build();
const outPath = path.join(__dirname, "report.docx");
Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(outPath, buf);
  console.log("Wrote " + outPath);
});
