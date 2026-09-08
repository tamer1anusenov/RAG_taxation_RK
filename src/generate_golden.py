#!/usr/bin/env python3
"""Generate a deterministic golden question sample from exported evaluation chunks."""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parent.parent
IN_PATH = ROOT / "eval" / "chunks.json"
OUT_PATH = ROOT / "eval" / "golden.json"
MODEL = "gpt-4o-mini"
SEED = 42
SAMPLE_SIZE = 50
REQUEST_TIMEOUT = float(os.getenv("EVAL_LLM_TIMEOUT_SECONDS", "120"))

SYSTEM_PROMPT = """Ты составляешь evaluation-набор для RAG по Налоговому кодексу Республики Казахстан.
На основании переданного CONTEXT создай ровно два разных конкретных вопроса, на которые CONTEXT отвечает
полностью и напрямую. Формулируй их так, как спросил бы живой пользователь, а не как заголовок статьи
или пересказ структуры текста. Не используй внешние знания и не добавляй сведения, которых нет в CONTEXT.
Верни только JSON-объект строго такого вида: {\"questions\": [\"...\", \"...\"]}."""


def load_chunks(path: Path) -> list[dict[str, str]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Expected a non-empty JSON list in {path}")
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not isinstance(row.get("text"), str):
            raise ValueError(f"Invalid chunk at index {index}; expected string id and text")
    return rows


def sample_chunks(chunks: list[dict[str, str]], sample_size: int, seed: int) -> list[dict[str, str]]:
    if sample_size > len(chunks):
        raise ValueError(f"sample_size={sample_size} exceeds available chunks={len(chunks)}")
    return random.Random(seed).sample(chunks, sample_size)


def generate_item(client: httpx.Client, chunk: dict[str, str], model: str) -> dict[str, list[str]]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"CONTEXT:\n{chunk['text']}"},
        ],
        "response_format": {"type": "json_object"},
    }
    response = client.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    result = json.loads(content)
    questions = result.get("questions") if isinstance(result, dict) else None
    if (
        set(result) != {"questions"}
        or not isinstance(questions, list)
        or len(questions) != 2
        or not all(isinstance(question, str) and question.strip() for question in questions)
        or len({question.strip().casefold() for question in questions}) != 2
    ):
        raise ValueError(f"Invalid golden response for chunk {chunk['id']}")
    return {"questions": [question.strip() for question in questions]}


def generate_golden(
    chunks: list[dict[str, str]], sample_size: int, seed: int, model: str, source: Path, output: Path
) -> None:
    selected = sample_chunks(chunks, sample_size, seed)
    items: list[dict[str, Any]] = []
    with httpx.Client() as client:
        for index, chunk in enumerate(selected, 1):
            generated = generate_item(client, chunk, model)
            items.append({"chunk_id": chunk["id"], "context": chunk["text"], **generated})
            print(f"generated {index}/{len(selected)}: {chunk['id']}")

    document = {
        "seed": seed,
        "sample_size": sample_size,
        "questions_per_chunk": 2,
        "model": model,
        "source": str(source),
        "items": items,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=IN_PATH)
    parser.add_argument("--output", type=Path, default=OUT_PATH)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    args = parser.parse_args()
    if args.sample_size < 1:
        parser.error("--sample-size must be positive")
    generate_golden(load_chunks(args.input), args.sample_size, args.seed, args.model, args.input, args.output)
    print(json.dumps({"output": str(args.output), "sample_size": args.sample_size, "seed": args.seed, "model": args.model}, ensure_ascii=False))


if __name__ == "__main__":
    main()
