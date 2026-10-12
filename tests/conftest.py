"""Общие фикстуры тестов."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings  # noqa: E402


@pytest.fixture(scope="session")
def settings() -> Settings:
    from app.config import load_env_file

    load_env_file(str(Path(__file__).resolve().parent.parent / ".env"))
    return Settings()


@pytest.fixture(scope="session")
def test_db(settings):
    """Интеграционные тесты идут в отдельную БД rag_test того же контейнера.

    Пропускаются, если postgres не поднят (docker compose up -d)
    или нет ни одного API-ключа.
    """
    psycopg = pytest.importorskip("psycopg")
    if not (settings.glm_api_key or settings.openrouter_api_key):
        pytest.skip("нет API-ключей LLM")

    base = settings.database_url.rsplit("/", 1)[0]
    url = base + "/rag_test"
    try:
        with psycopg.connect(base, connect_timeout=3, autocommit=True) as conn:
            exists = conn.execute("SELECT 1 FROM pg_database WHERE datname = 'rag_test'").fetchone()
            if not exists:
                conn.execute("CREATE DATABASE rag_test")
    except psycopg.OperationalError:
        pytest.skip("postgres недоступен — docker compose up -d")

    os.environ["DATABASE_URL"] = url
    s = Settings()
    yield s
    os.environ.pop("DATABASE_URL", None)
