"""Полный RAG-конвейер: вопрос → retrieval → (порог) → генерация с источниками."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from app.llm import LLMService
from app.prompts import build_messages, parse_citations
from app.retrieval import RetrievedChunk, Retriever

log = logging.getLogger("rag.pipeline")

REFUSAL_ANSWER = "Вне базы знаний — ответа в документации нет."


@dataclass
class AskResult:
    question: str
    answer: str
    refused: bool
    retrieval_latency: float
    total_latency: float
    sources: list[RetrievedChunk] = field(default_factory=list)
    cited_indices: list[int] = field(default_factory=list)  # номера процитированных фрагментов (с 1)


class RAGPipeline:
    def __init__(self, retriever: Retriever, llm: LLMService):
        self.retriever = retriever
        self.llm = llm

    def ask(self, question: str, top_k: int | None = None) -> AskResult:
        t0 = time.perf_counter()
        chunks, retrieval_latency = self.retriever.search(question, top_k=top_k)

        if not self.retriever.is_relevant(chunks):
            return AskResult(
                question=question,
                answer=REFUSAL_ANSWER,
                refused=True,
                retrieval_latency=retrieval_latency,
                total_latency=time.perf_counter() - t0,
                sources=[],
            )

        messages = build_messages(question, chunks)
        answer = self.llm.chat(messages)
        total = time.perf_counter() - t0

        # модель может сама отказаться, если контекст не отвечает на вопрос
        refused = answer.strip().startswith("Вне базы знаний")
        cited = parse_citations(answer) if not refused else []

        return AskResult(
            question=question,
            answer=answer.strip(),
            refused=refused,
            retrieval_latency=retrieval_latency,
            total_latency=total,
            sources=chunks,
            cited_indices=[n for n in cited if 1 <= n <= len(chunks)],
        )
