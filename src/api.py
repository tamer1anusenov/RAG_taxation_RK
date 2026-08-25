#!/usr/bin/env python3
"""Small API wrapper for hybrid Tax Code retrieval and grounded answers."""
from __future__ import annotations

import json
import logging
import os
import time
from functools import lru_cache
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

try:
    from .hybrid_search import BM25_FILE, COLLECTION, DENSE_MODEL, RERANKER_MODEL, dense_search, rrf_merge, rerank, bm25_search
    from .index_chunks import load_children
except ImportError:
    from hybrid_search import BM25_FILE, COLLECTION, DENSE_MODEL, RERANKER_MODEL, dense_search, rrf_merge, rerank, bm25_search
    from index_chunks import load_children

ROOT = Path(__file__).resolve().parent.parent
PROMPT_FILE = ROOT / "config" / "system_prompt.txt"
SCHEMA_FILE = ROOT / "config" / "response_schema.json"
PARENTS_FILE = ROOT / "data" / "chunks" / "taxiation_parents.jsonl"
LOG_DIR = ROOT / "logs"
LOG_FILE = LOG_DIR / "api.jsonl"
FRONTEND = ROOT / "frontend" / "index.html"
PARENTS_FILE = ROOT / "data" / "chunks" / "taxiation_parents.jsonl"
load_dotenv(ROOT / ".env")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("taxation_rag.api")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    handler = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=2, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)

app = FastAPI(title="Taxation RK RAG API", version="1.0.0")


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    dense_limit: int = Field(default=30, ge=1, le=100)
    bm25_limit: int = Field(default=30, ge=1, le=100)
    rerank_limit: int = Field(default=10, ge=1, le=30)
    output_limit: int = Field(default=5, ge=1, le=10)


class AskResponse(BaseModel):
    answer: str
    citations: list[dict[str, Any]]
    sources: list[dict[str, Any]]
    confidence: float
    timings_ms: dict[str, float]


@lru_cache(maxsize=1)
def get_children() -> list[dict]:
    return load_children()


@lru_cache(maxsize=1)
def get_bm25_data() -> dict:
    with BM25_FILE.open("rb") as f:
        import pickle
        return pickle.load(f)


@lru_cache(maxsize=1)
def get_parents() -> dict[str, dict]:
    rows = [json.loads(line) for line in PARENTS_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {row["id"]: row for row in rows}


@lru_cache(maxsize=1)
def get_system_prompt() -> str:
    return PROMPT_FILE.read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def get_response_schema() -> dict:
    return json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))


def refusal() -> dict:
    return {"answer": "В найденном контексте нет достаточной информации для ответа на этот вопрос.",
            "citations": [], "sources": [], "confidence": 0.0}


def normalize_answer(data: dict) -> dict:
    required = {"answer", "citations", "sources", "confidence"}
    if set(data) != required or not isinstance(data["answer"], str):
        raise ValueError("LLM returned an invalid structured response")
    confidence = float(data["confidence"])
    if not 0 <= confidence <= 1:
        raise ValueError("confidence must be between 0 and 1")
    return {"answer": data["answer"], "citations": data["citations"],
            "sources": data["sources"], "confidence": confidence}


def build_context(candidates: list[dict], limit: int = 5) -> str:
    parents = get_parents()
    blocks = []
    for candidate in candidates[:limit]:
        metadata = candidate["metadata"]
        parent = parents.get(metadata.get("parent_id"), {})
        blocks.append(
            f"[CHILD source_id={candidate['source_id']} article={metadata.get('article_number')} "
            f"point={metadata.get('point_number')} subpoint={metadata.get('subpoint_number')}]\n"
            f"{candidate['chunk']}\n"
            f"[PARENT article={metadata.get('article_number')} source_id={metadata.get('parent_id')}]\n"
            f"{parent.get('chunk', candidate['chunk'])}"
        )
    return "\n\n---\n\n".join(blocks)


def log_request(question: str, found: list[dict], answer: str | None, confidence: float | None,
                timings: dict[str, float], error: str | None = None) -> None:
    record = {
        "question": question,
        "found": [{"source_id": row["source_id"], "article": row["metadata"].get("article_number"),
                   "point": row["metadata"].get("point_number"), "rrf_score": row.get("rrf_score"),
                   "rerank_score": row.get("rerank_score")} for row in found],
        "answer": answer[:4000] if answer else None,
        "confidence": confidence,
        "timings_ms": timings,
    }
    if error:
        record["error"] = error[:500]
    logger.info(json.dumps(record, ensure_ascii=False))


async def generate_answer(question: str, context: str) -> dict:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="LLM is not configured: set OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    user_message = f"QUESTION:\n{question}\n\nFOUND_CONTEXT:\n{context}"
    body = {"model": model,
            "messages": [{"role": "system", "content": get_system_prompt()},
                         {"role": "user", "content": user_message}]}
    response_format = os.getenv("LLM_RESPONSE_FORMAT", "json_object")
    if response_format == "json_schema":
        body["response_format"] = {"type": "json_schema", "json_schema": {
            "name": "taxation_rag_answer", "strict": True, "schema": get_response_schema()}}
    elif response_format == "json_object":
        body["response_format"] = {"type": "json_object"}
    elif response_format != "none":
        raise HTTPException(status_code=500, detail="LLM_RESPONSE_FORMAT must be json_object, json_schema, or none")
    async with httpx.AsyncClient(timeout=float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))) as client:
        response = await client.post(base_url + "/chat/completions", headers={"Authorization": f"Bearer {api_key}"}, json=body)
        if response.is_error:
            detail = response.text[:500].replace("\n", " ")
            raise HTTPException(status_code=502, detail=f"Alem API returned HTTP {response.status_code}: {detail}")
    content = response.json()["choices"][0]["message"]["content"]
    return normalize_answer(json.loads(content))


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "llm_configured": bool(os.getenv("OPENAI_API_KEY")),
            "collection": COLLECTION, "dense_model": DENSE_MODEL, "reranker_model": RERANKER_MODEL}


@app.get("/", response_class=FileResponse)
def frontend() -> str:
    return str(FRONTEND)


@app.post("/api/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    started = time.perf_counter()
    timings: dict[str, float] = {}
    t = time.perf_counter()
    dense = dense_search(request.question, request.dense_limit)
    timings["dense_search"] = round((time.perf_counter() - t) * 1000, 2)
    t = time.perf_counter()
    sparse = bm25_search(request.question, request.bm25_limit, get_bm25_data())
    timings["bm25_search"] = round((time.perf_counter() - t) * 1000, 2)
    t = time.perf_counter()
    fused = rrf_merge(dense, sparse, 60)
    timings["rrf_merge"] = round((time.perf_counter() - t) * 1000, 2)
    t = time.perf_counter()
    reranked = rerank(request.question, fused, request.rerank_limit)
    timings["rerank"] = round((time.perf_counter() - t) * 1000, 2)
    context = build_context(reranked, request.output_limit)
    t = time.perf_counter()
    try:
        result = await generate_answer(request.question, context) if context else refusal()
    except HTTPException as exc:
        timings["generation"] = round((time.perf_counter() - t) * 1000, 2)
        timings["total"] = round((time.perf_counter() - started) * 1000, 2)
        log_request(request.question, reranked, None, None, timings, str(exc.detail))
        raise
    except Exception as exc:
        timings["generation"] = round((time.perf_counter() - t) * 1000, 2)
        timings["total"] = round((time.perf_counter() - started) * 1000, 2)
        log_request(request.question, reranked, None, None, timings, str(exc))
        raise HTTPException(status_code=502, detail="LLM generation failed") from exc
    timings["generation"] = round((time.perf_counter() - t) * 1000, 2)
    timings["total"] = round((time.perf_counter() - started) * 1000, 2)
    log_request(request.question, reranked, result["answer"], result["confidence"], timings)
    return AskResponse(**result, timings_ms=timings)
