#!/usr/bin/env python3
"""Evaluate dense retrieval against eval/golden.json using Qdrant point IDs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import mean
from typing import Any

from fastembed import TextEmbedding
from qdrant_client import QdrantClient

ROOT = Path(__file__).resolve().parent.parent
GOLDEN_PATH = ROOT / "eval" / "golden.json"
OUTPUT_PATH = ROOT / "eval" / "metrics.json"
DEFAULT_LOCAL_PATH = str(ROOT / "qdrant_storage") if (ROOT / "qdrant_storage").exists() else ""
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_LOCAL_PATH = os.getenv("QDRANT_LOCAL_PATH", DEFAULT_LOCAL_PATH)
COLLECTION = os.getenv("QDRANT_COLLECTION", "taxation_children")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


def load_golden(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Expected a non-empty list in {path}")
    for index, row in enumerate(rows):
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("question"), str)
            or not row["question"].strip()
            or not isinstance(row.get("relevant_ids"), list)
            or not row["relevant_ids"]
            or not all(isinstance(item, str) and item for item in row["relevant_ids"])
        ):
            raise ValueError(f"Invalid golden record at index {index}")
    return rows


def make_client() -> QdrantClient:
    return QdrantClient(path=QDRANT_LOCAL_PATH) if QDRANT_LOCAL_PATH else QdrantClient(url=QDRANT_URL)


def evaluate(golden: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    client = make_client()
    embedder = TextEmbedding(model_name=EMBEDDING_MODEL)
    questions = [row["question"] for row in golden]
    vectors = list(embedder.embed(questions))
    per_question: list[dict[str, Any]] = []

    for row, vector in zip(golden, vectors):
        points = client.query_points(
            collection_name=COLLECTION,
            query=list(map(float, vector)),
            limit=top_k,
            with_payload=False,
            with_vectors=False,
        ).points
        retrieved_ids = [str(point.id) for point in points]
        relevant_ids = set(row["relevant_ids"])
        relevant_ranks = [rank for rank, point_id in enumerate(retrieved_ids, 1) if point_id in relevant_ids]
        first_rank = min(relevant_ranks) if relevant_ranks else None
        hits = len(set(retrieved_ids) & relevant_ids)
        per_question.append(
            {
                "question": row["question"],
                "relevant_ids": sorted(relevant_ids),
                "retrieved_ids": retrieved_ids,
                "precision_at_k": hits / top_k,
                "recall_at_k": hits / len(relevant_ids),
                "hit_rate_at_k": 1.0 if hits else 0.0,
                "mrr": 1.0 / first_rank if first_rank else 0.0,
            }
        )

    return {
        "collection": COLLECTION,
        "embedding_model": EMBEDDING_MODEL,
        "top_k": top_k,
        "question_count": len(per_question),
        "precision_at_k": mean(row["precision_at_k"] for row in per_question),
        "recall_at_k": mean(row["recall_at_k"] for row in per_question),
        "hit_rate_at_k": mean(row["hit_rate_at_k"] for row in per_question),
        "mrr": mean(row["mrr"] for row in per_question),
        "per_question": per_question,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", type=Path, default=GOLDEN_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")

    metrics = evaluate(load_golden(args.golden), args.top_k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: metrics[key] for key in (
        "question_count", "top_k", "precision_at_k", "recall_at_k", "hit_rate_at_k", "mrr"
    )}, ensure_ascii=False))


if __name__ == "__main__":
    main()
