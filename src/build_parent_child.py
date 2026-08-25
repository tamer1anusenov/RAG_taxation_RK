#!/usr/bin/env python3
"""Build parent-child retrieval chunks from the parsed Tax Code text."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "data" / "raw" / "taxiation_doc.txt"
INPUT = ROOT / "data" / "parsed" / "taxiation_chunks.jsonl"
PARENTS = ROOT / "data" / "chunks" / "taxiation_parents.jsonl"
CHILDREN = ROOT / "data" / "chunks" / "taxiation_children.jsonl"
MANIFEST = ROOT / "data" / "chunks" / "taxiation_parent_child_manifest.json"


def read_records():
    return [json.loads(line) for line in INPUT.read_text(encoding="utf-8").splitlines() if line.strip()]


def source_text(lines, start, end):
    # Source line numbers in metadata are 1-based.
    return " ".join(line.strip() for line in lines[start - 1:end] if line.strip())


def main():
    records = read_records()
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    articles = [r for r in records if r["level"] == "article"]
    points = [r for r in records if r["level"] == "point"]

    parents = []
    children = []
    article_ranges = {}
    for index, article in enumerate(articles):
        start = article["metadata"]["source_lines"][0]
        end = articles[index + 1]["metadata"]["source_lines"][0] - 1 if index + 1 < len(articles) else len(lines)
        article_ranges[article["id"]] = (start, end)
        metadata = dict(article["metadata"])
        metadata.update({"retrieval_role": "parent", "parent_id": None,
                         "article_start_line": start, "article_end_line": end})
        parents.append({"id": article["id"], "level": "article", "chunk": source_text(lines, start, end),
                        "metadata": metadata})

    points_by_article = {}
    for point in points:
        parent_id = point["metadata"]["parent_id"]
        points_by_article.setdefault(parent_id, []).append(point)
    for article in articles:
        parent_id = article["id"]
        article_start, article_end = article_ranges[parent_id]
        article_points = sorted(points_by_article.get(parent_id, []),
                                key=lambda r: r["metadata"]["source_lines"][0])
        if not article_points:
            # Short articles without numbered points are retrieved as one unit.
            metadata = dict(article["metadata"])
            metadata.update({"retrieval_role": "child", "child_type": "article",
                             "parent_id": parent_id})
            children.append({"id": f"{parent_id}:child:article", "level": "child",
                             "chunk": source_text(lines, article_start, article_end),
                             "metadata": metadata})
            continue
        for i, point in enumerate(article_points):
            start = point["metadata"]["source_lines"][0]
            end = (article_points[i + 1]["metadata"]["source_lines"][0] - 1
                   if i + 1 < len(article_points) else article_end)
            metadata = dict(point["metadata"])
            metadata.update({"retrieval_role": "child", "child_type": "point",
                             "parent_id": parent_id, "parent_article_id": parent_id,
                             "article_start_line": article_start, "article_end_line": article_end})
            children.append({"id": f"{point['id']}:child", "level": "child",
                             "chunk": source_text(lines, start, end), "metadata": metadata})

    with PARENTS.open("w", encoding="utf-8") as f:
        for row in parents:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with CHILDREN.open("w", encoding="utf-8") as f:
        for row in children:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "strategy": "parent-child",
        "retrieval_unit": "point",
        "generation_context": "full_article_parent",
        "short_article_rule": "article_without_points_is_one_child_chunk",
        "source": str(SOURCE),
        "parents_file": str(PARENTS),
        "children_file": str(CHILDREN),
        "parent_count": len(parents),
        "child_count": len(children),
        "articles_with_points": sum(bool(points_by_article.get(a["id"])) for a in articles),
        "articles_without_points": sum(not points_by_article.get(a["id"]) for a in articles),
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
