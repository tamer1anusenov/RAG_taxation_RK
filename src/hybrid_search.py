#!/usr/bin/env python3
"""Hybrid dense + BM25 retrieval with lightweight cross-encoder reranking."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parent.parent
BM25_FILE = ROOT / "data" / "indexes" / "bm25_index.pkl"
DENSE_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")
COLLECTION = os.getenv("QDRANT_COLLECTION", "taxation_children")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
DEFAULT_LOCAL_PATH = str(ROOT / "qdrant_storage") if (ROOT / "qdrant_storage").exists() else ""
QDRANT_LOCAL_PATH = os.getenv("QDRANT_LOCAL_PATH", DEFAULT_LOCAL_PATH)
EMBEDDING_PREFIX = os.getenv("EMBEDDING_PREFIX", "")


def tokenize(text: str) -> list[str]:
    return re.findall(r"[\w№]+", text.lower(), flags=re.UNICODE)


def qdrant_client():
    from qdrant_client import QdrantClient
    if QDRANT_LOCAL_PATH:
        return QdrantClient(path=QDRANT_LOCAL_PATH)
    return QdrantClient(url=QDRANT_URL)


def dense_search(query: str, limit: int) -> list[dict]:
    from fastembed import TextEmbedding
    embedder = TextEmbedding(model_name=DENSE_MODEL)
    vector = list(embedder.embed([EMBEDDING_PREFIX + query]))[0]
    client = qdrant_client()
    response = client.query_points(collection_name=COLLECTION, query=list(map(float, vector)), limit=limit,
                                   with_payload=True)
    results = []
    for rank, point in enumerate(response.points, 1):
        payload = point.payload or {}
        results.append({"source_id": payload.get("source_id", str(point.id)),
                        "chunk": payload.get("chunk", ""), "metadata": payload,
                        "dense_score": float(point.score), "dense_rank": rank})
    return results


def bm25_search(query: str, limit: int, data: dict) -> list[dict]:
    bm25 = BM25Okapi(data["tokens"])
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(enumerate(scores), key=lambda item: float(item[1]), reverse=True)[:limit]
    results = []
    for rank, (index, score) in enumerate(ranked, 1):
        row = data["chunks"][index]
        results.append({"source_id": row["id"], "chunk": row["chunk"],
                        "metadata": row["metadata"], "bm25_score": float(score), "bm25_rank": rank})
    return results


def rrf_merge(dense: list[dict], sparse: list[dict], k: int) -> list[dict]:
    merged: dict[str, dict] = {}
    for result_set, rank_key in ((dense, "dense_rank"), (sparse, "bm25_rank")):
        for row in result_set:
            source_id = row["source_id"]
            target = merged.setdefault(source_id, {"source_id": source_id, "chunk": row["chunk"],
                                                   "metadata": row["metadata"], "rrf_score": 0.0})
            target["rrf_score"] += 1.0 / (k + row[rank_key])
            for key, value in row.items():
                if key not in target:
                    target[key] = value
    return sorted(merged.values(), key=lambda row: row["rrf_score"], reverse=True)


def rerank(query: str, candidates: list[dict], limit: int) -> list[dict]:
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    selected = candidates[:limit]
    if not selected:
        return []
    model = TextCrossEncoder(model_name=RERANKER_MODEL)
    scores = list(model.rerank(query, [row["chunk"] for row in selected], batch_size=limit))
    for row, score in zip(selected, scores):
        row["rerank_score"] = float(score)
    return sorted(selected, key=lambda row: row["rerank_score"], reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Поисковый запрос на русском или казахском")
    parser.add_argument("--dense-limit", type=int, default=30)
    parser.add_argument("--bm25-limit", type=int, default=30)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--rerank-limit", type=int, default=10)
    parser.add_argument("--output-limit", type=int, default=5)
    args = parser.parse_args()
    if args.rerank_limit < args.output_limit:
        raise ValueError("--rerank-limit must be >= --output-limit")
    with BM25_FILE.open("rb") as f:
        bm25_data = pickle.load(f)
    dense = dense_search(args.query, args.dense_limit)
    sparse = bm25_search(args.query, args.bm25_limit, bm25_data)
    fused = rrf_merge(dense, sparse, args.rrf_k)
    final = rerank(args.query, fused, args.rerank_limit)[:args.output_limit]
    print(json.dumps({"query": args.query, "dense_model": DENSE_MODEL,
                      "reranker_model": RERANKER_MODEL, "dense_hits": len(dense),
                      "bm25_hits": len(sparse), "rrf_candidates": len(fused),
                      "results": final}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
