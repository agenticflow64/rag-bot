"""Сборка промптов и разбор цитат [n] из ответа модели."""

from __future__ import annotations

import re

from app.retrieval import RetrievedChunk

SYSTEM_PROMPT = (
    "Ты — ассистент первой линии поддержки торговой компании «Меридиан» "
    "(бытовая техника и электроника). Тебе отвечает сотрудник поддержки, "
    "которому нужен ответ по внутренней базе знаний, чтобы сразу решить вопрос клиента. "
    "Отвечай строго по приведённым фрагментам базы знаний. "
    "После каждого факта или абзаца добавляй ссылку на источник в формате [n], "
    "где n — номер фрагмента. Отвечай по-русски, кратко и по делу: сроки, суммы, порядок действий. "
    "Система уже проверила релевантность: фрагменты ниже относятся к теме вопроса, "
    "поэтому отвечай по ним, даже если ответ неполный — и указывай, каких деталей нет. "
    "Фразу «Вне базы знаний — ответа в базе нет.» используй только если "
    "ни один фрагмент вообще не о теме вопроса. "
    "Не выдумывай и не используй внешние знания: правила компании есть только в этих фрагментах."
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
        f"Фрагменты базы знаний:\n\n{context}\n\n"
        f"Вопрос сотрудника поддержки: {question}\n\n"
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
