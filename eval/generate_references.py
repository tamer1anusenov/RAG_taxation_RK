#!/usr/bin/env python3
"""Generate grounded reference answers for a deterministic RAGAS subset."""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
GOLDEN_PATH = ROOT / "eval" / "golden.json"
OUTPUT_PATH = ROOT / "eval" / "references.json"
COLLECTION = os.getenv("QDRANT_COLLECTION", "taxation_children")
QDRANT_LOCAL_PATH = os.getenv("QDRANT_LOCAL_PATH", str(ROOT / "qdrant_storage"))
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
MODEL = "gpt-4o"

SYSTEM_PROMPT = """Ты формируешь эталонный ответ для оценки RAG-системы по Налоговому кодексу Казахстана.
Отвечай ТОЛЬКО на основании GOLD_CONTEXT.
Не добавляй сведения, которых нет в GOLD_CONTEXT.
Не используй общие знания и не делай юридических выводов сверх текста.
Ответ должен быть коротким: 1–3 предложения.
Если контекст не отвечает на вопрос полностью, честно укажи это и не додумывай.
Верни только JSON-объект с полем reference_answer."""


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Expected a non-empty list in {path}")
    return rows


def qdrant() -> QdrantClient:
    return QdrantClient(path=QDRANT_LOCAL_PATH) if QDRANT_LOCAL_PATH else QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))


def reference_context(client: QdrantClient, row: dict[str, Any]) -> str:
    points = client.retrieve(
        collection_name=COLLECTION,
        ids=row["relevant_ids"],
        with_payload=True,
        with_vectors=False,
    )
    by_id = {str(point.id): point for point in points}
    missing = [point_id for point_id in row["relevant_ids"] if point_id not in by_id]
    if missing:
        raise ValueError(f"Missing relevant Qdrant IDs: {missing}")
    chunks = []
    for point_id in row["relevant_ids"]:
        payload = by_id[point_id].payload or {}
        chunk = payload.get("chunk")
        if not isinstance(chunk, str) or not chunk.strip():
            raise ValueError(f"Qdrant point has no text payload: {point_id}")
        chunks.append(chunk.strip())
    return "\n\n---\n\n".join(chunks)


def generate_answer(client: OpenAI, question: str, context: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        max_tokens=250,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"QUESTION:\n{question}\n\nGOLD_CONTEXT:\n{context}",
            },
        ],
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("LLM returned empty content")
    data = json.loads(content)
    answer = data.get("reference_answer")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("LLM response lacks non-empty reference_answer")
    return answer.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", type=Path, default=GOLDEN_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = load_rows(args.golden)
    if not 1 <= args.sample_size <= len(rows):
        parser.error(f"--sample-size must be between 1 and {len(rows)}")
    rng = random.Random(args.seed)
    selected = rng.sample(rows, args.sample_size)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    client = OpenAI(api_key=api_key, base_url=BASE_URL)
    qdrant_client = qdrant()
    output: list[dict[str, Any]] = []
    for index, row in enumerate(selected, 1):
        context = reference_context(qdrant_client, row)
        answer = generate_answer(client, row["question"], context)
        output.append({
            "question": row["question"],
            "relevant_ids": row["relevant_ids"],
            "reference_context": context,
            "reference_answer": answer,
        })
        print(f"generated {index}/{len(selected)}", flush=True)
        time.sleep(0.05)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "records": len(output), "model": MODEL}, ensure_ascii=False))


if __name__ == "__main__":
    main()
