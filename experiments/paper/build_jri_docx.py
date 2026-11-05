"""Build manuscript_jri.docx from manuscript_jri.md using the JRI 2026 template.

The template dictates: A4, Cambria-10 body (title Cambria-14 bold), justified body
text at 1.5 line spacing, centred bold section headings, tables without vertical
lines with the caption above, figure captions below, single-spaced 150-250 word
abstracts, and APA hanging-indent references.

Usage:
    python experiments/paper/build_jri_docx.py
    python experiments/paper/build_jri_docx.py --embed-figures

With ``--embed-figures`` every ``[[INSERT FIGURE N HERE]]`` marker is replaced by the
matching ``figures/figure N.(png|jpg|jpeg)`` image from the repository root; markers
without a matching file keep their text placeholder. Override the output path with the
``HEALTHIER_DOCX_OUT`` environment variable.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import docx
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

HERE = Path(__file__).resolve().parent
MD_PATH = HERE / "manuscript_jri.md"
TEMPLATE = Path(r"C:\Users\tataa\Downloads\TEMPLATE JRI TAHUN-2026-ENGLISH.docx")
OUT = Path(os.environ.get("HEALTHIER_DOCX_OUT", str(HERE / "manuscript_jri.docx")))
FIGURES_DIR = HERE.parents[1] / "figures"
EMBED_FIGURES = "--embed-figures" in sys.argv

FONT = "Cambria"
BODY_SIZE = Pt(10)
TITLE_SIZE = Pt(14)


def set_font(run, size=BODY_SIZE, bold=None, italic=None):
    run.font.name = FONT
    run.font.size = size
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.insert(0, rfonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rfonts.set(qn(attr), FONT)


def add_runs(paragraph, text):
    for token in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*)", text):
        if not token:
            continue
        if token.startswith("**") and token.endswith("**"):
            set_font(paragraph.add_run(token[2:-2]), bold=True)
        elif token.startswith("*") and token.endswith("*"):
            set_font(paragraph.add_run(token[1:-1]), italic=True)
        else:
            set_font(paragraph.add_run(token))


def para(doc, align=WD_ALIGN_PARAGRAPH.JUSTIFY, spacing=1.5, indent=None, after=6):
    p = doc.add_paragraph()
    p.alignment = align
    pf = p.paragraph_format
    pf.line_spacing = spacing
    pf.space_after = Pt(after)
    pf.space_before = Pt(0)
    if indent is not None:
        pf.first_line_indent = Cm(indent)
    return p


def set_table_borders(table):
    """Three-line style: horizontal borders only, no vertical lines."""
    tbl_pr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "bottom", "insideH"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "4")
        el.set(qn("w:color"), "000000")
        borders.append(el)
    for edge in ("left", "right", "insideV"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "none")
        borders.append(el)
    tbl_pr.append(borders)


def add_table(doc, rows):
    header, body = rows[0], rows[1:]
    table = doc.add_table(rows=1, cols=len(header))
    table.autofit = True
    set_table_borders(table)
    for j, cell_text in enumerate(header):
        cell = table.rows[0].cells[j]
        cell.text = ""
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.line_spacing = 1.0
        p.paragraph_format.space_after = Pt(0)
        set_font(p.add_run(cell_text), bold=True)
    for row in body:
        cells = table.add_row().cells
        for j, cell_text in enumerate(row):
            cells[j].text = ""
            p = cells[j].paragraphs[0]
            p.alignment = (
                WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER
            )
            p.paragraph_format.line_spacing = 1.0
            p.paragraph_format.space_after = Pt(0)
            set_font(p.add_run(cell_text))
    doc.add_paragraph().paragraph_format.space_after = Pt(0)
    return table


def find_figure(number: int) -> Path | None:
    """Return the image file for ``figure N`` in the repo's figures/ folder."""
    for ext in (".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG"):
        candidate = FIGURES_DIR / f"figure {number}{ext}"
        if candidate.is_file():
            return candidate
    return None


def clear_body(document):
    body = document.element.body
    for child in list(body):
        if child.tag == qn("w:sectPr"):
            continue
        body.remove(child)


def build():
    doc = docx.Document(str(TEMPLATE))
    clear_body(doc)
    normal = doc.styles["Normal"]
    normal.font.name = FONT
    normal.font.size = BODY_SIZE

    lines = MD_PATH.read_text(encoding="utf-8").split("\n")

    section = None          # current top-level section name
    in_abstract = False
    in_refs = False
    i = 0
    n = len(lines)

    while i < n:
        raw = lines[i]
        line = raw.rstrip()
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        # Title
        if stripped.startswith("# "):
            p = para(doc, WD_ALIGN_PARAGRAPH.CENTER, spacing=1.0, after=10)
            set_font(p.add_run(stripped[2:]), size=TITLE_SIZE, bold=True)
            i += 1
            # author block until next '## '
            while i < n and not lines[i].startswith("## "):
                a = lines[i].strip()
                if a:
                    p = para(doc, WD_ALIGN_PARAGRAPH.CENTER, spacing=1.0, after=2)
                    add_runs(p, a)
                i += 1
            doc.add_paragraph().paragraph_format.space_after = Pt(4)
            continue

        # Headings
        if stripped.startswith("## "):
            name = stripped[3:].strip()
            section = name
            in_refs = name.upper() == "REFERENCES"
            in_abstract = name in ("Abstract", "Abstrak")
            p = para(doc, WD_ALIGN_PARAGRAPH.CENTER, spacing=1.0, after=8)
            p.paragraph_format.space_before = Pt(10)
            set_font(p.add_run(name.upper() if name.islower() or section.upper() in
                               ("RESEARCH METHODS", "RESULTS AND DISCUSSION",
                                "CONCLUSIONS AND SUGGESTIONS") else name), bold=True)
            i += 1
            continue
        if stripped.startswith("#### "):
            p = para(doc, WD_ALIGN_PARAGRAPH.LEFT, spacing=1.0, after=4)
            set_font(p.add_run(stripped[5:]), bold=True, italic=True)
            i += 1
            continue
        if stripped.startswith("### "):
            p = para(doc, WD_ALIGN_PARAGRAPH.LEFT, spacing=1.0, after=4)
            set_font(p.add_run(stripped[4:]), bold=True)
            i += 1
            continue

        # Figure placeholder
        if stripped.startswith("[["):
            match = re.match(r"\[\[INSERT FIGURE (\d+)", stripped)
            image = find_figure(int(match.group(1))) if (EMBED_FIGURES and match) else None
            if image is not None:
                p = para(doc, WD_ALIGN_PARAGRAPH.CENTER, spacing=1.0, after=4)
                p.add_run().add_picture(str(image), width=Cm(15))
            else:
                p = para(doc, WD_ALIGN_PARAGRAPH.CENTER, spacing=1.0, after=2)
                set_font(p.add_run(stripped), italic=True)
            i += 1
            continue

        # Table / figure caption
        if re.match(r"^(Table|Figure) \d+\.", stripped):
            p = para(doc, WD_ALIGN_PARAGRAPH.LEFT, spacing=1.0, after=4)
            add_runs(p, stripped)
            i += 1
            continue

        # Equations
        if stripped.startswith("EQ:"):
            p = para(doc, WD_ALIGN_PARAGRAPH.CENTER, spacing=1.0, after=6)
            set_font(p.add_run(stripped[3:].strip()))
            i += 1
            continue

        # Markdown table block
        if stripped.startswith("|"):
            block = []
            while i < n and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not re.match(r"^:?-{2,}", cells[0]):
                    block.append(cells)
                i += 1
            add_table(doc, block)
            continue

        # Keywords
        if stripped.startswith(("Keywords:", "Kata kunci:")):
            p = para(doc, WD_ALIGN_PARAGRAPH.JUSTIFY, spacing=1.0, after=8)
            add_runs(p, stripped)
            i += 1
            continue

        # Bullets / numbered items
        if re.match(r"^([-*]|\d+\.)\s", stripped):
            p = para(doc, WD_ALIGN_PARAGRAPH.JUSTIFY, spacing=1.5, after=4)
            p.paragraph_format.left_indent = Cm(0.75)
            add_runs(p, re.sub(r"^([-*]|\d+\.)\s", "", stripped))
            i += 1
            continue

        # References: hanging indent, single-spaced
        if in_refs:
            p = para(doc, WD_ALIGN_PARAGRAPH.JUSTIFY, spacing=1.0, after=6)
            p.paragraph_format.left_indent = Cm(0.75)
            p.paragraph_format.first_line_indent = Cm(-0.75)
            add_runs(p, stripped)
            i += 1
            continue

        # Abstract / Abstrak: single spaced, no indent
        if in_abstract:
            p = para(doc, WD_ALIGN_PARAGRAPH.JUSTIFY, spacing=1.0, after=6)
            add_runs(p, stripped)
            i += 1
            continue

        # Regular body paragraph
        p = para(doc, WD_ALIGN_PARAGRAPH.JUSTIFY, spacing=1.5, indent=1.0, after=6)
        add_runs(p, stripped)
        i += 1

    doc.save(str(OUT))
    print("Wrote", OUT)


if __name__ == "__main__":
    build()
