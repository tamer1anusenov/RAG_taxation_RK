#!/usr/bin/env python3
"""Run a small end-to-end RAGAS evaluation against the local RAG API."""
from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import httpx
from datasets import Dataset
from dotenv import load_dotenv

# RAGAS 0.4.3 imports an optional VertexAI module that is absent in the
# installed langchain-community build. It is not used with the OpenAI provider.
vertex_module = types.ModuleType("langchain_community.chat_models.vertexai")
vertex_module.ChatVertexAI = type("ChatVertexAI", (), {})
sys.modules.setdefault("langchain_community.chat_models.vertexai", vertex_module)

from ragas import evaluate
from ragas.llms import llm_factory
from ragas.metrics import Faithfulness
from openai import OpenAI


class CompatCompletions:
    def __init__(self, inner):
        self._inner = inner

    def create(self, *args, **kwargs):
        if "max_tokens" in kwargs and "max_completion_tokens" not in kwargs:
            kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
        return self._inner.create(*args, **kwargs)


class CompatChat:
    def __init__(self, inner):
        self.completions = CompatCompletions(inner.completions)


class CompatOpenAI:
    def __init__(self, inner):
        self.chat = CompatChat(inner.chat)


ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8000").rstrip("/")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
OUTPUT = ROOT / "eval" / "ragas_smoke.json"

QUESTIONS = [
    "Когда уплачивается акциз при передаче сырой нефти и газового конденсата на переработку?",
    "Как осуществляется деятельность нерезидента на основании договора о совместной деятельности в Казахстане?",
]


def load_chunks() -> dict[str, str]:
    path = ROOT / "data" / "chunks" / "taxiation_children.jsonl"
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            result[row["id"]] = row["chunk"]
    return result


def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    chunk_by_id = load_chunks()
    rows = []
    with httpx.Client(timeout=180) as client:
        for question in QUESTIONS:
            response = client.post(f"{API_URL}/api/ask", json={"question": question})
            response.raise_for_status()
            result = response.json()
            source_ids = [source["source_id"] for source in result["sources"]]
            contexts = [chunk_by_id[source_id] for source_id in source_ids if source_id in chunk_by_id]
            rows.append({
                "user_input": question,
                "response": result["answer"],
                "retrieved_contexts": contexts,
            })
    dataset = Dataset.from_list(rows)
    client = OpenAI(api_key=api_key, base_url=base_url)
    original_create = client.chat.completions.create

    def compatible_create(*args, **kwargs):
        if "max_tokens" in kwargs and "max_completion_tokens" not in kwargs:
            kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
        if "temperature" in kwargs and kwargs["temperature"] != 1:
            kwargs.pop("temperature")
        kwargs.pop("top_p", None)
        return original_create(*args, **kwargs)

    client.chat.completions.create = compatible_create
    judge = llm_factory(model=LLM_MODEL, provider="openai", client=client)
    result = evaluate(
        dataset,
        metrics=[Faithfulness()],
        llm=judge,
        raise_exceptions=True,
        show_progress=True,
    )
    scores = result.to_pandas().to_dict(orient="records")
    payload = {
        "api_url": API_URL,
        "judge_model": LLM_MODEL,
        "question_count": len(rows),
        "metrics": {
            key: sum(float(row[key]) for row in scores) / len(scores)
            for key in ("faithfulness",)
        },
        "rows": scores,
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["metrics"], ensure_ascii=False))
    print(f"saved={OUTPUT}")


if __name__ == "__main__":
    main()
