"""Small, dependency-free DOCX reader for the knowledge ingestion pipeline.

DOCX files are ZIP archives containing OOXML.  The source knowledge document
uses paragraphs, inline images and a simple table, so the standard library is
enough and the cloud backend does not need a new Word-processing dependency.
"""

from __future__ import annotations

import hashlib
import posixpath
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List
from xml.etree import ElementTree as ET


_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_A = "http://schemas.openxmlformats.org/drawingml/2006/main"
_R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_PR = "http://schemas.openxmlformats.org/package/2006/relationships"


@dataclass(frozen=True)
class DocxBlock:
    kind: str
    text: str = ""
    image_targets: List[str] = field(default_factory=list)
    table_rows: List[List[str]] = field(default_factory=list)


@dataclass(frozen=True)
class ParsedDocx:
    filename: str
    sha256: str
    blocks: List[DocxBlock]
    media: Dict[str, bytes]


def _text(element: ET.Element) -> str:
    parts: List[str] = []
    for node in element.iter():
        if node.tag == f"{{{_W}}}t" and node.text:
            parts.append(node.text)
        elif node.tag == f"{{{_W}}}tab":
            parts.append("\t")
        elif node.tag == f"{{{_W}}}br":
            parts.append("\n")
    return "".join(parts).strip()


def _relationships(archive: zipfile.ZipFile) -> Dict[str, str]:
    rel_path = "word/_rels/document.xml.rels"
    if rel_path not in archive.namelist():
        return {}
    root = ET.fromstring(archive.read(rel_path))
    relationships: Dict[str, str] = {}
    for rel in root.findall(f"{{{_PR}}}Relationship"):
        rel_id = rel.attrib.get("Id")
        target = rel.attrib.get("Target")
        if rel_id and target:
            relationships[rel_id] = posixpath.normpath(posixpath.join("word", target))
    return relationships


def _image_targets(element: ET.Element, relationships: Dict[str, str]) -> List[str]:
    targets: List[str] = []
    for blip in element.iter(f"{{{_A}}}blip"):
        rel_id = blip.attrib.get(f"{{{_R}}}embed")
        target = relationships.get(rel_id or "")
        if target and target not in targets:
            targets.append(target)
    return targets


def _table_rows(table: ET.Element) -> List[List[str]]:
    rows: List[List[str]] = []
    for row in table.findall(f"{{{_W}}}tr"):
        cells = [_text(cell) for cell in row.findall(f"{{{_W}}}tc")]
        if any(cells):
            rows.append(cells)
    return rows


def _iter_blocks(body: ET.Element, relationships: Dict[str, str]) -> Iterable[DocxBlock]:
    for child in body:
        if child.tag == f"{{{_W}}}p":
            yield DocxBlock(
                kind="paragraph",
                text=_text(child),
                image_targets=_image_targets(child, relationships),
            )
        elif child.tag == f"{{{_W}}}tbl":
            yield DocxBlock(kind="table", table_rows=_table_rows(child))


def parse_docx(path: str | Path) -> ParsedDocx:
    source = Path(path)
    raw = source.read_bytes()
    with zipfile.ZipFile(source) as archive:
        relationships = _relationships(archive)
        root = ET.fromstring(archive.read("word/document.xml"))
        body = root.find(f"{{{_W}}}body")
        if body is None:
            raise ValueError("DOCX does not contain word/document.xml body")
        blocks = list(_iter_blocks(body, relationships))
        media = {
            target: archive.read(target)
            for target in sorted({t for block in blocks for t in block.image_targets})
            if target in archive.namelist()
        }
    return ParsedDocx(
        filename=source.name,
        sha256=hashlib.sha256(raw).hexdigest(),
        blocks=blocks,
        media=media,
    )


__all__ = ["DocxBlock", "ParsedDocx", "parse_docx"]
