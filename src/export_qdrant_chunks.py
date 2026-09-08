#!/usr/bin/env python3
"""Export all Qdrant chunk texts for retrieval evaluation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOCAL_PATH = str(ROOT / "qdrant_storage") if (ROOT / "qdrant_storage").exists() else ""
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_LOCAL_PATH = os.getenv("QDRANT_LOCAL_PATH", DEFAULT_LOCAL_PATH)
COLLECTION = os.getenv("QDRANT_COLLECTION", "taxation_children")
DEFAULT_OUTPUT = ROOT / "eval" / "chunks.json"


def make_client() -> QdrantClient:
    if QDRANT_LOCAL_PATH:
        return QdrantClient(path=QDRANT_LOCAL_PATH)
    return QdrantClient(url=QDRANT_URL)


def export_chunks(limit: int, output: Path) -> int:
    client = make_client()
    rows: list[dict[str, str]] = []
    offset: Any = None

    while True:
        points, next_offset = client.scroll(
            collection_name=COLLECTION,
            limit=limit,
            offset=offset,
            with_payload=["chunk"],
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            text = payload.get("chunk")
            if not isinstance(text, str) or not text:
                raise ValueError(f"Point {point.id} has no non-empty payload.chunk")
            rows.append({"id": str(point.id), "text": text})

        if next_offset is None:
            break
        offset = next_offset

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=256, help="Qdrant scroll page size")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")

    count = export_chunks(args.limit, args.output)
    print(json.dumps({"collection": COLLECTION, "count": count, "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
