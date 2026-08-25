#!/usr/bin/env python3
"""Parse the extracted Russian Tax Code text into hierarchical JSONL chunks."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "data" / "raw" / "taxiation_doc.txt"
CHUNKS = ROOT / "data" / "parsed" / "taxiation_chunks.jsonl"
TABLES = ROOT / "data" / "parsed" / "taxiation_table_candidates.json"

RE_SECTION = re.compile(r"^РАЗДЕЛ\s+(\d+)\.\s*(.*)$", re.I)
RE_CHAPTER = re.compile(r"^Глава\s+(\d+)\.\s*(.*)$", re.I)
RE_PARAGRAPH = re.compile(r"^Параграф\s+(\d+)\.\s*(.*)$", re.I)
RE_ARTICLE = re.compile(r"^Статья\s+(\d+)\.\s*(.*)$", re.I)
RE_POINT = re.compile(r"^(\d+)\.\s+(.+)$")
RE_SUBPOINT = re.compile(r"^(\d+)\)\s+(.+)$")
RE_LETTER = re.compile(r"^([а-яё])\)\s+(.+)$", re.I)
RE_NUM = re.compile(r"(?<!\w)(?:\d+(?:[\s.,/]\d+)*\s*%?|\d+\s*-?кратн)", re.I)


def clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def make_record(kind, text, state, start_line, end_line, number=None, title=None, parent_id=None, record_id=None):
    return {
        "id": record_id or f"{kind}:{number if number is not None else start_line}",
        "level": kind,
        "chunk": clean(text),
        "metadata": {
            "section_number": state["section_number"],
            "section_title": state["section_title"],
            "chapter_number": state["chapter_number"],
            "chapter_title": state["chapter_title"],
            "paragraph_number": state["paragraph_number"],
            "paragraph_title": state["paragraph_title"],
            "article_number": state["article_number"],
            "article_title": state["article_title"],
            "number": number,
            "point_number": number if kind == "point" else None,
            "subpoint_number": number if kind == "subpoint" else None,
            "title": title,
            "source_lines": [start_line, end_line],
            "parent_id": parent_id,
        },
    }


def parse():
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    state = {k: None for k in (
        "section_number", "section_title", "chapter_number", "chapter_title",
        "paragraph_number", "paragraph_title", "article_number", "article_title")}
    records = []
    current = None
    article_id = None
    point_id = None
    subpoint_id = None

    def flush(end_line):
        nonlocal current
        if current and current["text"]:
            records.append(make_record(
                current["kind"], " ".join(current["text"]), state,
                current["start"], end_line, current["number"], current["title"], current["parent"], current["id"]
            ))
        current = None

    for line_no, raw in enumerate(lines, 1):
        line = clean(raw)
        if not line:
            continue
        m = RE_SECTION.match(line)
        if m:
            flush(line_no - 1)
            state.update(section_number=int(m.group(1)), section_title=m.group(2),
                         chapter_number=None, chapter_title=None, paragraph_number=None,
                         paragraph_title=None, article_number=None, article_title=None)
            article_id = point_id = subpoint_id = None
            continue
        m = RE_CHAPTER.match(line)
        if m:
            flush(line_no - 1)
            state.update(chapter_number=int(m.group(1)), chapter_title=m.group(2),
                         paragraph_number=None, paragraph_title=None,
                         article_number=None, article_title=None)
            article_id = point_id = subpoint_id = None
            continue
        m = RE_PARAGRAPH.match(line)
        if m:
            flush(line_no - 1)
            state.update(paragraph_number=int(m.group(1)), paragraph_title=m.group(2),
                         article_number=None, article_title=None)
            article_id = point_id = subpoint_id = None
            continue
        m = RE_ARTICLE.match(line)
        if m:
            flush(line_no - 1)
            n, title = int(m.group(1)), m.group(2)
            state.update(article_number=n, article_title=title)
            article_id = f"article:{n}:{line_no}"
            point_id = subpoint_id = None
            current = {"kind": "article", "number": n, "title": title,
                       "text": [line], "start": line_no, "parent": None, "id": article_id}
            continue
        m = RE_POINT.match(line)
        if m and state["article_number"] is not None:
            flush(line_no - 1)
            n, text = int(m.group(1)), m.group(2)
            point_id = f"article:{state['article_number']}:point:{n}:{line_no}"
            subpoint_id = None
            current = {"kind": "point", "number": n, "title": None,
                       "text": [line], "start": line_no, "parent": article_id, "id": point_id}
            continue
        m = RE_SUBPOINT.match(line)
        if m and state["article_number"] is not None:
            flush(line_no - 1)
            n, text = int(m.group(1)), m.group(2)
            subpoint_id = f"article:{state['article_number']}:subpoint:{n}:{line_no}"
            current = {"kind": "subpoint", "number": n, "title": None,
                       "text": [line], "start": line_no, "parent": point_id or article_id, "id": subpoint_id}
            continue
        m = RE_LETTER.match(line)
        if m and state["article_number"] is not None:
            flush(line_no - 1)
            current = {"kind": "subpoint", "number": m.group(1).lower(), "title": None,
                       "text": [line], "start": line_no, "parent": point_id or article_id,
                       "id": f"article:{state['article_number']}:subpoint:{m.group(1).lower()}:{line_no}"}
            continue
        if current is not None:
            current["text"].append(line)
        elif state["article_number"] is not None:
            current = {"kind": "article", "number": state["article_number"],
                       "title": state["article_title"], "text": [line],
                       "start": line_no, "parent": None, "id": article_id}

    flush(len(lines))

    # Detect likely tabular fragments in the flattened TXT. Exact cell structure
    # cannot be recovered from TXT, so candidates are emitted for review.
    candidates = []
    for i, line in enumerate(lines):
        s = clean(line)
        if not s or not RE_NUM.search(s):
            continue
        numbers = RE_NUM.findall(s)
        keywords = ("ставк", "лимит", "предел", "таблиц")
        if (len(numbers) >= 3 or ("%" in s and len(numbers) >= 1)) and any(k in s.lower() for k in keywords):
            window = [clean(x) for x in lines[max(0, i-2):min(len(lines), i+3)] if clean(x)]
            candidates.append({"source_line": i + 1, "text": s, "context": window,
                               "needs_manual_review": True})
    # De-duplicate overlapping candidate windows.
    unique = []
    seen = set()
    for c in candidates:
        key = (c["source_line"], c["text"])
        if key not in seen:
            seen.add(key); unique.append(c)

    with CHUNKS.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    TABLES.write_text(json.dumps({"source": str(SOURCE), "count": len(unique),
                                  "warning": "TXT extraction flattened HTML tables; review candidates against source HTML.",
                                  "candidates": unique}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"chunks": len(records), "articles": sum(r["level"] == "article" for r in records),
                      "points": sum(r["level"] == "point" for r in records),
                      "subpoints": sum(r["level"] == "subpoint" for r in records),
                      "table_candidates": len(unique), "chunks_file": str(CHUNKS),
                      "tables_file": str(TABLES)}, ensure_ascii=False))


if __name__ == "__main__":
    parse()
