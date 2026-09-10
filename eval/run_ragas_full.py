#!/usr/bin/env python3
"""Run full RAGAS evaluation using generated references and the local RAG API."""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import re
import sys
import types
from pathlib import Path
from typing import Any

import httpx
from datasets import Dataset
from dotenv import load_dotenv
from fastembed import TextEmbedding
from openai import OpenAI

# RAGAS 0.4.3 imports an optional VertexAI module absent from the installed
# langchain-community build. It is unused with the OpenAI provider.
vertex_module = types.ModuleType("langchain_community.chat_models.vertexai")
vertex_module.ChatVertexAI = type("ChatVertexAI", (), {})
sys.modules.setdefault("langchain_community.chat_models.vertexai", vertex_module)

from ragas import evaluate
from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.metrics import AnswerCorrectness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas.llms import llm_factory

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
REFERENCES = ROOT / "eval" / "references.json"
OUTPUT = ROOT / "eval" / "ragas_full.json"
INPUT_CACHE = ROOT / "eval" / "ragas_inputs.json"
API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8000").rstrip("/")
JUDGE_MODEL = "gpt-4o"
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


class FastEmbedRagas(BaseRagasEmbeddings):
    """Bridge the installed FastEmbed model to RAGAS's embedding interface."""

    def __init__(self, model_name: str):
        super().__init__()
        self.embedder = TextEmbedding(model_name=model_name)

    def embed_query(self, text: str) -> list[float]:
        return list(map(float, next(iter(self.embedder.embed([text])))))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, vector)) for vector in self.embedder.embed(texts)]

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)


def load_chunks() -> dict[str, str]:
    path = ROOT / "data" / "chunks" / "taxiation_children.jsonl"
    return {
        row["id"]: row["chunk"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for row in [json.loads(line)]
    }


def make_compatible_judge() -> Any:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    client = OpenAI(api_key=api_key, base_url=base_url)
    original_create = client.chat.completions.create

    def compatible_create(*args: Any, **kwargs: Any):
        if "max_tokens" in kwargs and "max_completion_tokens" not in kwargs:
            kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
        if "max_completion_tokens" in kwargs:
            kwargs["max_completion_tokens"] = max(int(kwargs["max_completion_tokens"]), 4096)
        if "temperature" in kwargs and kwargs["temperature"] != 1:
            kwargs.pop("temperature")
        kwargs.pop("top_p", None)
        return original_create(*args, **kwargs)

    client.chat.completions.create = compatible_create  # type: ignore[method-assign]
    return llm_factory(model=JUDGE_MODEL, provider="openai", client=client)


def normalize_reference_answer(answer: str) -> str:
    """Unwrap an occasional nested JSON object from the generation step."""
    try:
        data = json.loads(answer)
    except json.JSONDecodeError:
        return answer.strip()
    if isinstance(data, dict) and isinstance(data.get("reference_answer"), str):
        return data["reference_answer"].strip()
    return answer.strip()


def collect_rows(selected: list[dict[str, Any]], chunks: dict[str, str]) -> list[dict[str, Any]]:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.hybrid_search import bm25_search, dense_search, rerank, rrf_merge

    with (ROOT / "data" / "indexes" / "bm25_index.pkl").open("rb") as handle:
        bm25_data = pickle.load(handle)
    rows = []
    with httpx.Client(timeout=240) as client:
        for index, reference in enumerate(selected, 1):
            response = client.post(f"{API_URL}/api/ask", json={"question": reference["question"]})
            response.raise_for_status()
            result = response.json()
            dense = dense_search(reference["question"], 30)
            sparse = bm25_search(reference["question"], 30, bm25_data)
            fused = rrf_merge(dense, sparse, 60)
            retrieved = rerank(reference["question"], fused, 10)[:5]
            contexts = [row["chunk"] for row in retrieved if row.get("chunk")]
            if not contexts:
                raise ValueError(f"Hybrid retrieval returned no contexts for row {index}")
            rows.append({
                "user_input": reference["question"],
                "response": result["answer"],
                "retrieved_contexts": contexts,
                "reference": normalize_reference_answer(reference["reference_answer"]),
            })
            print(f"collected {index}/{len(selected)}", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, default=REFERENCES)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache", type=Path, default=INPUT_CACHE)
    args = parser.parse_args()
    references = json.loads(args.references.read_text(encoding="utf-8"))
    if args.limit < 1 or args.limit > len(references):
        parser.error(f"--limit must be between 1 and {len(references)}")
    selected = list(references)
    if args.limit < len(selected):
        selected = random.Random(args.seed).sample(selected, args.limit)
    if args.cache.exists():
        cached = json.loads(args.cache.read_text(encoding="utf-8"))
        if len(cached) == len(selected) and {row["user_input"] for row in cached} == {row["question"] for row in selected}:
            rows = cached
            print(f"loaded cache {len(rows)} rows", flush=True)
        else:
            rows = collect_rows(selected, load_chunks())
    else:
        rows = collect_rows(selected, load_chunks())
    args.cache.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dataset = Dataset.from_list(rows)
    metrics = [ContextPrecision(), ContextRecall(), AnswerRelevancy(strictness=1), AnswerCorrectness(max_retries=3)]
    result = evaluate(
        dataset,
        metrics=metrics,
        llm=make_compatible_judge(),
        embeddings=FastEmbedRagas(EMBEDDING_MODEL),
        raise_exceptions=False,
        show_progress=True,
        batch_size=8,
    )
    scores = result.to_pandas().to_dict(orient="records")
    names = ["context_precision", "context_recall", "answer_relevancy", "answer_correctness"]
    averages = {}
    valid_counts = {}
    for name in names:
        values = [float(row[name]) for row in scores if math.isfinite(float(row[name]))]
        valid_counts[name] = len(values)
        averages[name] = sum(values) / len(values) if values else None
    payload = {
        "question_count": len(rows),
        "judge_model": JUDGE_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "valid_counts": valid_counts,
        "metrics": averages,
        "rows": scores,
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(averages, ensure_ascii=False))
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
