"""Конфигурация из переменных окружения (.env)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

DEFAULT_DATABASE_URL = "postgresql://rag:rag@localhost:5432/rag"


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


@dataclass
class Settings:
    # БД
    database_url: str = field(default_factory=lambda: _env("DATABASE_URL", DEFAULT_DATABASE_URL))

    # Retrieval
    top_k: int = field(default_factory=lambda: int(_env("TOP_K", "5")))
    # Порог косинусной близости: ниже — «нет ответа в базе»
    relevance_threshold: float = field(default_factory=lambda: float(_env("RELEVANCE_THRESHOLD", "0.30")))

    # Чанкинг: целевой размер в «токенах» (оценка: 1 токен ≈ 4 символа)
    chunk_target_tokens: int = field(default_factory=lambda: int(_env("CHUNK_TARGET_TOKENS", "600")))
    chunk_max_tokens: int = field(default_factory=lambda: int(_env("CHUNK_MAX_TOKENS", "800")))
    chunk_overlap_tokens: int = field(default_factory=lambda: int(_env("CHUNK_OVERLAP_TOKENS", "100")))

    # LLM-провайдеры: glm | openrouter | auto (glm с откатом на openrouter)
    chat_provider: str = field(default_factory=lambda: _env("CHAT_PROVIDER", "auto"))
    # Эмбеддинги: local (GPU, sentence-transformers) | glm | openrouter | auto
    embeddings_provider: str = field(default_factory=lambda: _env("EMBEDDINGS_PROVIDER", "local"))

    # Локальный эмбеддер (bge-m3 на GPU, векторы 1024 измерения)
    local_embedding_model: str = field(default_factory=lambda: _env("LOCAL_EMBEDDING_MODEL", "BAAI/bge-m3"))
    local_embedding_device: str = field(default_factory=lambda: _env("LOCAL_EMBEDDING_DEVICE", "cuda"))

    # GLM (основной)
    glm_api_key: str = field(default_factory=lambda: _env("GLM_API_KEY"))
    glm_base_url: str = field(default_factory=lambda: _env("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"))
    glm_chat_model: str = field(default_factory=lambda: _env("GLM_CHAT_MODEL", "glm-5.3-flash"))
    glm_embedding_model: str = field(default_factory=lambda: _env("GLM_EMBEDDING_MODEL", "embedding-3"))

    # OpenRouter (резерв)
    openrouter_api_key: str = field(default_factory=lambda: _env("OPENROUTER_API_KEY"))
    openrouter_base_url: str = field(default_factory=lambda: _env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"))
    openrouter_chat_model: str = field(
        default_factory=lambda: _env("OPENROUTER_CHAT_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")
    )
    openrouter_embedding_model: str = field(
        default_factory=lambda: _env("OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small")
    )

    embedding_dim: int = field(default_factory=lambda: int(_env("EMBEDDING_DIM", "1024")))


def load_env_file(path: str = ".env") -> None:
    """Мини-загрузчик .env (без внешних зависимостей). Вызывать ДО создания Settings."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip("'\"")
            os.environ.setdefault(key, value)


def get_settings() -> Settings:
    load_env_file()
    return Settings()


def estimate_tokens(text: str) -> int:
    """Грубая оценка числа токенов: ~4 символа на токен для англ. текста."""
    return max(1, len(text) // 4)
