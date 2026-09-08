#!/usr/bin/env python3
"""Embed child chunks into Qdrant and build a matching BM25 index."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import uuid
from pathlib import Path

from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parent.parent
CHILDREN = ROOT / "data" / "chunks" / "taxiation_children.jsonl"
INDEX_DIR = ROOT / "data" / "indexes"
BM25_FILE = INDEX_DIR / "bm25_index.pkl"
MANIFEST_FILE = INDEX_DIR / "index_manifest.json"
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
COLLECTION = os.getenv("QDRANT_COLLECTION", "taxation_children")
BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
EMBEDDING_PREFIX = os.getenv("EMBEDDING_PREFIX", "")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
DEFAULT_LOCAL_PATH = str(ROOT / "qdrant_storage") if (ROOT / "qdrant_storage").exists() else ""
QDRANT_LOCAL_PATH = os.getenv("QDRANT_LOCAL_PATH", DEFAULT_LOCAL_PATH)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[\w№]+", text.lower(), flags=re.UNICODE)


def load_children() -> list[dict]:
    return [json.loads(line) for line in CHILDREN.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_bm25(children: list[dict]) -> None:
    documents = [row["chunk"] for row in children]
    tokens = [tokenize(text) for text in documents]
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with BM25_FILE.open("wb") as f:
        pickle.dump({"documents": documents, "tokens": tokens, "chunks": children,
                     "tokenizer": "unicode-word-lowercase", "version": 1,
                     "model": "rank_bm25.BM25Okapi"}, f, protocol=pickle.HIGHEST_PROTOCOL)
    # Construct once here to validate tokenization and index compatibility.
    BM25Okapi(tokens)


def qdrant_client():
    from qdrant_client import QdrantClient
    if QDRANT_LOCAL_PATH:
        return QdrantClient(path=QDRANT_LOCAL_PATH)
    return QdrantClient(url=QDRANT_URL)


def upload_qdrant(children: list[dict]) -> tuple[int, int]:
    from fastembed import TextEmbedding
    from qdrant_client import models

    embedder = TextEmbedding(model_name=MODEL_NAME)
    texts = [EMBEDDING_PREFIX + row["chunk"] for row in children]
    vectors = []
    for batch_start in range(0, len(texts), BATCH_SIZE):
        batch = texts[batch_start:batch_start + BATCH_SIZE]
        vectors.extend(list(embedder.embed(batch, batch_size=BATCH_SIZE)))
    if not vectors:
        raise RuntimeError("No embeddings were generated")
    vector_size = len(vectors[0])
    client = qdrant_client()
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
    )
    points = []
    for row, vector in zip(children, vectors):
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, "rag-taxation:" + row["id"]))
        payload = {"chunk": row["chunk"], "source_id": row["id"], **row["metadata"]}
        points.append(models.PointStruct(id=point_id, vector=list(map(float, vector)), payload=payload))
    for start in range(0, len(points), BATCH_SIZE):
        client.upsert(collection_name=COLLECTION, points=points[start:start + BATCH_SIZE], wait=True)
    return vector_size, len(points)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bm25-only", action="store_true", help="Build BM25 without requiring Qdrant or the embedding model")
    args = parser.parse_args()
    children = load_children()
    if not children:
        raise RuntimeError(f"No child chunks found in {CHILDREN}")
    build_bm25(children)
    vector_size = None
    vector_count = 0
    if not args.bm25_only:
        vector_size, vector_count = upload_qdrant(children)
    manifest = {
        "embedding_model": MODEL_NAME if not args.bm25_only else None,
        "embedding_prefix": EMBEDDING_PREFIX,
        "qdrant_collection": None if args.bm25_only else COLLECTION,
        "qdrant_url": None if args.bm25_only else (QDRANT_LOCAL_PATH or QDRANT_URL),
        "vector_size": vector_size,
        "vector_count": vector_count,
        "bm25_file": str(BM25_FILE),
        "child_source": str(CHILDREN),
        "chunk_count": len(children),
        "mode": "bm25-only" if args.bm25_only else "full",
    }
    MANIFEST_FILE.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
