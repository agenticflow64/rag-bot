"""Интеграционные тесты: ингест мини-корпуса → retrieval → конвейер ответа.

Требует поднятый docker compose (postgres) и API-ключи в .env.
Запуск: pytest -m integration
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app import db
from app.ingest import ingest
from app.llm import LLMService
from app.pipeline import RAGPipeline
from app.retrieval import Retriever

pytestmark = pytest.mark.integration

FIXTURE_DIR = Path(__file__).parent / "fixtures"

QUESTIONS = [
    ("Как откалибровать гаджет?", "gadgets.md"),
    ("Сколько гаджетов может зарегистрировать один владелец?", "gadgets.md"),
    ("Какой префикс должен быть у идентификатора гаджета?", "gadgets.md"),
    ("С каким интервалом телеметрия сбрасывается на диск?", "telemetry.md"),
    ("Сколько раз пайплайн повторяет запись при ошибке?", "telemetry.md"),
]


@pytest.fixture(scope="module")
def ingested(test_db):
    stats = ingest(test_db, data_dir=FIXTURE_DIR)
    assert stats["chunks"] >= 5
    return stats


@pytest.fixture
def retriever(test_db, ingested):
    llm = LLMService(test_db)
    return Retriever(db.get_pool(test_db), llm, top_k=5, threshold=0.30)


@pytest.mark.parametrize("question,expected_file", QUESTIONS)
def test_retrieval_finds_expected_source(retriever, question, expected_file):
    chunks, latency = retriever.search(question)
    assert chunks, "должен найти хоть один чанк"
    top_files = [c.source_file for c in chunks]
    assert expected_file in top_files, f"{expected_file} не в top-k: {top_files}"
    assert latency < 15


def test_relevant_question_is_not_refusal(retriever):
    assert retriever.is_relevant(retriever.search("Как откалибровать гаджет?")[0])


def test_out_of_base_refuses(retriever):
    chunks, _ = retriever.search("Рецепт борща на три литра")
    assert not retriever.is_relevant(chunks)


def test_full_pipeline_answers_with_source(test_db, ingested):
    llm = LLMService(test_db)
    r = Retriever(db.get_pool(test_db), llm, top_k=5, threshold=0.30)
    result = RAGPipeline(r, llm).ask("Какой допустимый уровень калибровки гаджета?")
    assert not result.refused
    assert "9" in result.answer or "1" in result.answer
    assert any(s.source_file == "gadgets.md" for s in result.sources)
    assert result.total_latency > 0
