"""PostgreSQL + pgvector: схема, пул соединений, кэш эмбеддингов."""

from __future__ import annotations

import hashlib

import psycopg
from psycopg import sql
from psycopg_pool import ConnectionPool

from app.config import Settings

_pool: ConnectionPool | None = None


def get_pool(settings: Settings | None = None) -> ConnectionPool:
    global _pool
    if _pool is None:
        s = settings or get_settings_cached()
        _pool = ConnectionPool(s.database_url, min_size=1, max_size=8, open=True)
    return _pool


_settings_cache: Settings | None = None


def get_settings_cached() -> Settings:
    global _settings_cache
    if _settings_cache is None:
        from app.config import get_settings

        _settings_cache = get_settings()
    return _settings_cache


SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id          bigserial PRIMARY KEY,
    source_file text        NOT NULL,
    title       text        NOT NULL,
    section     text        NOT NULL,
    url         text        NOT NULL,
    content     text        NOT NULL,
    token_count integer     NOT NULL,
    content_hash text       NOT NULL,
    embedding   vector({dim})
);

CREATE TABLE IF NOT EXISTS embed_cache (
    content_hash text PRIMARY KEY,
    model        text NOT NULL,
    embedding    vector({dim}) NOT NULL,
    created_at   timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS meta (
    key   text PRIMARY KEY,
    value text NOT NULL
);
"""

INDEX_SQL = """
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_content_hash_idx ON chunks (content_hash);
"""


def init_schema(pool: ConnectionPool, dim: int) -> None:
    with pool.connection() as conn:
        conn.execute(SCHEMA_SQL.format(dim=dim))
        conn.execute(INDEX_SQL)


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_meta(pool: ConnectionPool, key: str) -> str | None:
    with pool.connection() as conn:
        row = conn.execute("SELECT value FROM meta WHERE key = %s", (key,)).fetchone()
    return row[0] if row else None


def set_meta(pool: ConnectionPool, key: str, value: str) -> None:
    with pool.connection() as conn:
        conn.execute(
            "INSERT INTO meta (key, value) VALUES (%s, %s) "
            "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
            (key, value),
        )


def drop_all_embeddings(pool: ConnectionPool) -> None:
    """Сброс эмбеддингов при смене модели (кэш и чанки)."""
    with pool.connection() as conn:
        conn.execute("TRUNCATE embed_cache")
        conn.execute("UPDATE chunks SET embedding = NULL")
