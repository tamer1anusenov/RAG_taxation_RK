#!/usr/bin/env python3
"""Telegram long-polling adapter for the local Taxation RK RAG API."""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import httpx
from dotenv import load_dotenv

from .api import ROOT

load_dotenv(ROOT / ".env")

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
RAG_API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8000").rstrip("/")
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TOKEN}" if TOKEN else ""
POLL_TIMEOUT = int(os.getenv("TELEGRAM_POLL_TIMEOUT_SECONDS", "30"))
REQUEST_TIMEOUT = float(os.getenv("TELEGRAM_REQUEST_TIMEOUT_SECONDS", "120"))
MAX_MESSAGE_LENGTH = 4096

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("taxation_rag.telegram")


def split_message(text: str) -> list[str]:
    return [text[i : i + MAX_MESSAGE_LENGTH] for i in range(0, len(text), MAX_MESSAGE_LENGTH)] or [""]


def format_response(data: dict[str, Any]) -> str:
    answer = str(data.get("answer", "Ответ не получен."))
    citations = data.get("citations") or []
    parts = [answer]
    if citations:
        parts.append("\nИсточники:")
        for citation in citations:
            article = citation.get("article", "?")
            point = citation.get("point")
            subpoint = citation.get("subpoint")
            location = f"статья {article}"
            if point is not None:
                location += f", пункт {point}"
            if subpoint is not None:
                location += f", подпункт {subpoint}"
            parts.append(f"• {location}\n{citation.get('quote', '')}")
    timings = data.get("timings_ms") or {}
    if timings:
        labels = {
            "dense_search": "Dense-поиск",
            "bm25_search": "BM25-поиск",
            "rrf_merge": "RRF-объединение",
            "rerank": "Reranking",
            "generation": "Генерация ответа",
            "total": "Всего",
        }
        timing_lines = [
            f"• {labels.get(name, name)}: {float(value):.0f} мс"
            for name, value in timings.items()
            if name in labels
        ]
        if timing_lines:
            parts.append("\nВремя по этапам:\n" + "\n".join(timing_lines))
    return "\n\n".join(parts)


async def telegram_call(client: httpx.AsyncClient, method: str, payload: dict[str, Any]) -> Any:
    response = await client.post(f"{TELEGRAM_API_URL}/{method}", json=payload)
    response.raise_for_status()
    data = response.json()
    if not data.get("ok"):
        raise RuntimeError(f"Telegram {method} failed: {data.get('description', 'unknown error')}")
    return data.get("result")


async def ask_rag(client: httpx.AsyncClient, question: str) -> str:
    response = await client.post(f"{RAG_API_URL}/api/ask", json={"question": question}, timeout=REQUEST_TIMEOUT)
    if response.is_error:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise RuntimeError(f"RAG API HTTP {response.status_code}: {str(detail)[:500]}")
    return format_response(response.json())


async def handle_message(client: httpx.AsyncClient, message: dict[str, Any]) -> None:
    chat = message.get("chat", {})
    chat_id = chat.get("id")
    text = (message.get("text") or "").strip()
    if chat_id is None or not text:
        return

    if text in {"/start", "/help"}:
        await telegram_call(
            client,
            "sendMessage",
            {
                "chat_id": chat_id,
                "text": "Я RAG-ассистент по Налоговому кодексу Республики Казахстан. Отправьте вопрос обычным сообщением.",
            },
        )
        return

    try:
        await telegram_call(client, "sendChatAction", {"chat_id": chat_id, "action": "typing"})
        progress = await telegram_call(
            client,
            "sendMessage",
            {"chat_id": chat_id, "text": "Запрос принят. Выполняю поиск и формирую ответ..."},
        )
        answer = await ask_rag(client, text)
    except Exception as exc:
        logger.exception("Request failed for chat_id=%s", chat_id)
        answer = f"Не удалось получить ответ: {exc}"
        progress = None

    if progress:
        await telegram_call(
            client,
            "editMessageText",
            {"chat_id": chat_id, "message_id": progress["message_id"], "text": answer[:MAX_MESSAGE_LENGTH]},
        )
        if len(answer) <= MAX_MESSAGE_LENGTH:
            return
        answer = answer[MAX_MESSAGE_LENGTH:]
    for chunk in split_message(answer):
        await telegram_call(client, "sendMessage", {"chat_id": chat_id, "text": chunk})


async def run() -> None:
    if not TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")

    offset = 0
    async with httpx.AsyncClient(timeout=POLL_TIMEOUT + 10) as client:
        bot = await telegram_call(client, "getMe", {})
        logger.info("Started Telegram bot @%s; RAG API: %s", bot.get("username"), RAG_API_URL)
        while True:
            updates = await telegram_call(
                client,
                "getUpdates",
                {"offset": offset, "timeout": POLL_TIMEOUT, "allowed_updates": ["message"]},
            )
            for update in updates:
                offset = max(offset, int(update["update_id"]) + 1)
                try:
                    await handle_message(client, update.get("message", {}))
                except Exception:
                    logger.exception("Failed to process update_id=%s", update.get("update_id"))


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logger.info("Telegram bot stopped")
