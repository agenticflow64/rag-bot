"""Векторный поиск top-k релевантных чанков в pgvector."""

from __future__ import annotations

import time
from dataclasses import dataclass

from psycopg_pool import ConnectionPool

from app.llm import LLMService

SEARCH_SQL = """
SELECT id, source_file, title, section, url, content, token_count,
       1 - (embedding <=> %(vec)s::vector) AS score
FROM chunks
WHERE embedding IS NOT NULL
ORDER BY embedding <=> %(vec)s::vector
LIMIT %(k)s
"""


@dataclass
class RetrievedChunk:
    id: int
    source_file: str
    title: str
    section: str
    url: str
    content: str
    token_count: int
    score: float


class Retriever:
    def __init__(self, pool: ConnectionPool, llm: LLMService, top_k: int = 5, threshold: float = 0.30):
        self.pool = pool
        self.llm = llm
        self.top_k = top_k
        self.threshold = threshold

    def search(self, question: str, top_k: int | None = None) -> tuple[list[RetrievedChunk], float]:
        """Возвращает (чанки, latency_сек). Чанки отсортированы по убыванию близости."""
        k = top_k or self.top_k
        t0 = time.perf_counter()
        (vec,) = self.llm.embed([question])
        rows = []
        with self.pool.connection() as conn:
            cur = conn.execute(SEARCH_SQL, {"vec": _pgvec(vec), "k": k})
            rows = cur.fetchall()
        latency = time.perf_counter() - t0
        chunks = [
            RetrievedChunk(id=r[0], source_file=r[1], title=r[2], section=r[3],
                           url=r[4], content=r[5], token_count=r[6], score=float(r[7]))
            for r in rows
        ]
        return chunks, latency

    def is_relevant(self, chunks: list[RetrievedChunk]) -> bool:
        """Есть ли хоть один чанк выше порога релевантности."""
        return bool(chunks) and chunks[0].score >= self.threshold


def _pgvec(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"
