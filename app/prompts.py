"""Сборка промптов и разбор цитат [n] из ответа модели."""

from __future__ import annotations

import re

from app.retrieval import RetrievedChunk

SYSTEM_PROMPT = (
    "Ты — ассистент по базе знаний из документации фреймворка FastAPI (Python). "
    "Отвечай строго по приведённым фрагментам документации. "
    "После каждого факта или абзаца добавляй ссылку на источник в формате [n], "
    "где n — номер фрагмента. Отвечай по-русски, кратко и по делу. "
    "Если во фрагментах нет ответа на вопрос — ответь ровно: «Вне базы знаний — ответа в документации нет.» "
    "Не выдумывай и не используй внешние знания."
)

CITATION_RE = re.compile(r"\[(\d+)\]")


def build_context(chunks: list[RetrievedChunk], max_chars: int = 12000) -> str:
    """Контекст из пронумерованных фрагментов (с 1)."""
    parts: list[str] = []
    total = 0
    for i, ch in enumerate(chunks, start=1):
        body = ch.content[:2000]
        block = f"[{i}] Файл: {ch.source_file} | Раздел: {ch.section}\n{body}"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n\n---\n\n".join(parts)


def build_messages(question: str, chunks: list[RetrievedChunk]) -> list[dict[str, str]]:
    context = build_context(chunks)
    user = (
        f"Фрагменты документации:\n\n{context}\n\n"
        f"Вопрос пользователя: {question}\n\n"
        "Ответь по фрагментам выше со ссылками [n] на источники."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def parse_citations(answer: str) -> list[int]:
    """Номера фрагментов, процитированных в ответе (уникальные, по порядку)."""
    seen: list[int] = []
    for m in CITATION_RE.finditer(answer):
        n = int(m.group(1))
        if n not in seen:
            seen.append(n)
    return seen
