"""Локальные эмбеддинги на GPU через sentence-transformers (bge-m3).

Модель грузится один раз на процесс. Векторы нормализуются, чтобы
косинусная близость в pgvector считалась корректно.
"""

from __future__ import annotations

import logging

from app.config import Settings

log = logging.getLogger("rag.embeddings")

_instance: "LocalEmbedder | None" = None


class LocalEmbedder:
    def __init__(self, model_name: str, device: str):
        self.model_name = model_name
        self.device = device
        self._model = None  # ленивая загрузка: импорт тяжёлый, юнит-тестам не нужен

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            log.info("грузим эмбеддер %s на %s (первый запуск — дольше минуты)", self.model_name, self.device)
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._get_model().encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
        return [v.tolist() for v in vectors]

    @property
    def dim(self) -> int:
        return self._get_model().get_sentence_embedding_dimension()


def get_local_embedder(settings: Settings) -> LocalEmbedder:
    global _instance
    if _instance is None:
        _instance = LocalEmbedder(settings.local_embedding_model, settings.local_embedding_device)
    return _instance
