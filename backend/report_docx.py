"""Convert the report's Markdown subset to a .docx — stdlib only.

A .docx is a ZIP of OOXML parts. The report Markdown (from
``report_builder.render_markdown``) uses a *fixed, known* subset — headings
(#..####), paragraphs, **bold**, ordered/unordered lists, blockquotes and
GitHub tables — so a focused writer covers it without ``python-docx`` or
``pandoc`` (neither is installed on the CPU-only backend host).

Entry point: ``markdown_to_docx_bytes(md) -> bytes``.
"""
from __future__ import annotations

import re
import zipfile
from io import BytesIO
from typing import List, Tuple
from xml.sax.saxutils import escape

_CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>"""

_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""

_DOC_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"""

# Heading styles + a Normal default with a CJK-friendly font.
_STYLES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:docDefaults><w:rPrDefault><w:rPr>
    <w:rFonts w:ascii="Microsoft YaHei" w:eastAsia="Microsoft YaHei" w:hAnsi="Microsoft YaHei"/>
    <w:sz w:val="22"/></w:rPr></w:rPrDefault></w:docDefaults>
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style>
  <w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/>
    <w:pPr><w:spacing w:before="240" w:after="120"/></w:pPr>
    <w:rPr><w:b/><w:sz w:val="36"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/>
    <w:pPr><w:spacing w:before="200" w:after="100"/></w:pPr>
    <w:rPr><w:b/><w:sz w:val="30"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading3"><w:name w:val="heading 3"/>
    <w:pPr><w:spacing w:before="160" w:after="80"/></w:pPr>
    <w:rPr><w:b/><w:sz w:val="26"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading4"><w:name w:val="heading 4"/>
    <w:pPr><w:spacing w:before="120" w:after="60"/></w:pPr>
    <w:rPr><w:b/><w:sz w:val="24"/></w:rPr></w:style>
</w:styles>"""

_WNS = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'


def _runs(text: str) -> str:
    """Inline **bold** → a sequence of <w:r> runs."""
    out: List[str] = []
    parts = re.split(r"(\*\*[^*]+\*\*)", text)
    for p in parts:
        if not p:
            continue
        if p.startswith("**") and p.endswith("**"):
            out.append(f'<w:r><w:rPr><w:b/></w:rPr><w:t xml:space="preserve">{escape(p[2:-2])}</w:t></w:r>')
        else:
            out.append(f'<w:r><w:t xml:space="preserve">{escape(p)}</w:t></w:r>')
    return "".join(out) or '<w:r><w:t/></w:r>'


def _para(text: str, style: str = "", ind: int = 0) -> str:
    ppr = ""
    if style:
        ppr += f'<w:pStyle w:val="{style}"/>'
    if ind:
        ppr += f'<w:ind w:left="{ind}"/>'
    ppr = f"<w:pPr>{ppr}</w:pPr>" if ppr else ""
    return f"<w:p>{ppr}{_runs(text)}</w:p>"


def _split_row(line: str) -> List[str]:
    t = line.strip().strip("|")
    return [c.strip() for c in t.split("|")]


_SEP_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$")


def _table(headers: List[str], rows: List[List[str]]) -> str:
    ncol = len(headers)
    grid = "".join(f'<w:gridCol w:w="{9000 // max(ncol, 1)}"/>' for _ in range(ncol))
    borders = (
        '<w:tblBorders>'
        '<w:top w:val="single" w:sz="4" w:color="999999"/>'
        '<w:left w:val="single" w:sz="4" w:color="999999"/>'
        '<w:bottom w:val="single" w:sz="4" w:color="999999"/>'
        '<w:right w:val="single" w:sz="4" w:color="999999"/>'
        '<w:insideH w:val="single" w:sz="4" w:color="999999"/>'
        '<w:insideV w:val="single" w:sz="4" w:color="999999"/>'
        '</w:tblBorders>'
    )

    def cell(text: str, header: bool) -> str:
        shade = '<w:shd w:val="clear" w:fill="EAF1F8"/>' if header else ""
        return f"<w:tc><w:tcPr>{shade}</w:tcPr>{_para(text)}</w:tc>"

    def row(cells: List[str], header: bool) -> str:
        return "<w:tr>" + "".join(cell(c, header) for c in cells) + "</w:tr>"

    body = row(headers, True) + "".join(row(r + [""] * (ncol - len(r)), False) for r in rows)
    return (
        f'<w:tbl><w:tblPr><w:tblW w:w="9000" w:type="dxa"/>{borders}</w:tblPr>'
        f"<w:tblGrid>{grid}</w:tblGrid>{body}</w:tbl>"
    )


def _markdown_to_body(md: str) -> str:
    lines = md.split("\n")
    out: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        h = re.match(r"^(#{1,6})\s+(.*)$", line)
        if h:
            level = min(len(h.group(1)), 4)
            out.append(_para(h.group(2), style=f"Heading{level}"))
            i += 1
            continue
        # table
        if "|" in line and i + 1 < len(lines) and _SEP_RE.match(lines[i + 1]):
            headers = _split_row(line)
            rows: List[List[str]] = []
            i += 2
            while i < len(lines) and "|" in lines[i] and lines[i].strip():
                rows.append(_split_row(lines[i]))
                i += 1
            out.append(_table(headers, rows))
            continue
        if re.match(r"^\s*>\s?", line):
            out.append(_para(re.sub(r"^\s*>\s?", "", line), ind=360))
            i += 1
            continue
        m = re.match(r"^\s*\d+\.\s+(.*)$", line)
        if m:
            out.append(_para("• " + m.group(1), ind=360))
            i += 1
            continue
        m = re.match(r"^\s*[-*]\s+(.*)$", line)
        if m:
            out.append(_para("• " + m.group(1), ind=360))
            i += 1
            continue
        if line.strip() == "":
            i += 1
            continue
        out.append(_para(line))
        i += 1
    return "".join(out)


def markdown_to_docx_bytes(md: str) -> bytes:
    body = _markdown_to_body(md)
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f"<w:document {_WNS}><w:body>{body}"
        '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
        '<w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1134"/></w:sectPr>'
        "</w:body></w:document>"
    )
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", _CONTENT_TYPES)
        z.writestr("_rels/.rels", _RELS)
        z.writestr("word/_rels/document.xml.rels", _DOC_RELS)
        z.writestr("word/styles.xml", _STYLES)
        z.writestr("word/document.xml", document)
    return buf.getvalue()


__all__ = ["markdown_to_docx_bytes"]
