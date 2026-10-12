"""LLM-провайдеры: GLM (основной) и OpenRouter (резерв), OpenAI-совместимый протокол.

Генерация — через API с авто-fallback. Эмбеддинги — либо там же через API,
либо локально на GPU (bge-m3 через sentence-transformers), зависит от
EMBEDDINGS_PROVIDER. Провайдер API выбирается один раз на процесс.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import Any

import httpx

from app.config import Settings

log = logging.getLogger("rag.llm")

EMBED_TIMEOUT = 60.0
CHAT_TIMEOUT = 180.0


class LLMError(RuntimeError):
    pass


class _Provider:
    """Один OpenAI-совместимый endpoint."""

    name: str

    def __init__(self, base_url: str, api_key: str, chat_model: str, embedding_model: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.chat_model = chat_model
        self.embedding_model = embedding_model

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def available(self) -> bool:
        return bool(self.api_key)

    async def _post(self, client: httpx.AsyncClient, path: str, payload: dict[str, Any]) -> httpx.Response:
        return await client.post(f"{self.base_url}{path}", headers=self._headers(), json=payload)


class LLMService:
    """Генерация и эмбеддинги с выбором провайдера и кэшированием выбора."""

    def __init__(self, settings: Settings):
        self.s = settings
        self._local_embedder = None  # LocalEmbedder, создаётся лениво при local-провайдере
        self._glm = _Provider(
            settings.glm_base_url, settings.glm_api_key,
            settings.glm_chat_model, settings.glm_embedding_model,
        )
        self._glm.name = "glm"
        self._or = _Provider(
            settings.openrouter_base_url, settings.openrouter_api_key,
            settings.openrouter_chat_model, settings.openrouter_embedding_model,
        )
        self._or.name = "openrouter"
        self._chat_provider: _Provider | None = None
        self._embed_provider: _Provider | None = None

    # ---------- выбор провайдера ----------

    def chat_endpoint(self) -> _Provider:
        if self._chat_provider is not None:
            return self._chat_provider
        if self.s.chat_provider == "openrouter":
            self._chat_provider = self._or
            return self._or
        if self.s.chat_provider == "glm":
            self._chat_provider = self._glm
            return self._glm
        # auto: быстрый пробный запрос к GLM
        if self._glm.available() and self._probe_chat(self._glm):
            self._chat_provider = self._glm
        elif self._or.available():
            self._chat_provider = self._or
        else:
            raise LLMError("нет доступных LLM-провайдеров: проверьте GLM_API_KEY / OPENROUTER_API_KEY")
        log.info("chat provider: %s (%s)", self._chat_provider.name, self._chat_provider.chat_model)
        return self._chat_provider

    def embed_endpoint(self) -> _Provider:
        if self.s.embeddings_provider == "local":
            raise LLMError("эмбеддинги локальные — API-endpoint не используется")
        if self._embed_provider is not None:
            return self._embed_provider
        if self.s.embeddings_provider == "openrouter":
            self._embed_provider = self._or
            return self._or
        if self.s.embeddings_provider == "glm":
            self._embed_provider = self._glm
            return self._glm
        if self._glm.available() and self._probe_embed(self._glm):
            self._embed_provider = self._glm
        elif self._or.available():
            self._embed_provider = self._or
        else:
            raise LLMError("нет доступных провайдеров эмбеддингов")
        log.info("embeddings provider: %s (%s)", self._embed_provider.name, self._embed_provider.embedding_model)
        return self._embed_provider

    def _fallback(self, p: _Provider) -> _Provider | None:
        """Другой провайдер, если он доступен."""
        other = self._or if p is self._glm else self._glm
        return other if other.available() else None

    def _probe_chat(self, p: _Provider) -> bool:
        try:
            with httpx.Client(timeout=CHAT_TIMEOUT) as c:
                r = c.post(
                    f"{p.base_url}/chat/completions", headers=p._headers(),
                    json={"model": p.chat_model, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 1},
                )
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    def _probe_embed(self, p: _Provider) -> bool:
        try:
            with httpx.Client(timeout=EMBED_TIMEOUT) as c:
                r = c.post(
                    f"{p.base_url}/embeddings", headers=p._headers(),
                    json={"model": p.embedding_model, "input": ["ping"]},
                )
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    # ---------- эмбеддинги ----------

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.s.embeddings_provider == "auto" and not (self._glm.available() or self._or.available()):
            # ключей API нет — эмбеддим локально на GPU
            self.s.embeddings_provider = "local"
        if self.s.embeddings_provider == "local":
            if self._local_embedder is None:
                from app.embeddings import get_local_embedder

                self._local_embedder = get_local_embedder(self.s)
            return self._local_embedder.embed(texts)
        try:
            return self._embed_with(self.embed_endpoint(), texts)
        except LLMError:
            # жёсткая ошибка провайдера (модели нет / нет баланса) — откат на резерв
            fb = self._fallback(self.embed_endpoint())
            if fb is None:
                raise
            log.warning("эмбеддинги: %s недоступен, откат на %s", self.embed_endpoint().name, fb.name)
            self._embed_provider = fb
            return self._embed_with(fb, texts)

    def _embed_with(self, p: _Provider, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        # батчами по 32
        for i in range(0, len(texts), 32):
            batch = texts[i : i + 32]
            with httpx.Client(timeout=EMBED_TIMEOUT) as c:
                r = c.post(
                    f"{p.base_url}/embeddings", headers=p._headers(),
                    json={"model": p.embedding_model, "input": batch},
                )
            if r.status_code != 200:
                raise LLMError(f"embeddings {p.name} http={r.status_code}: {r.text[:200]}")
            data = sorted(r.json()["data"], key=lambda d: d["index"])
            out.extend(d["embedding"] for d in data)
        return out

    def embedding_model_name(self) -> str:
        if self.s.embeddings_provider == "local":
            return f"{self.s.local_embedding_model} (local)"
        return self.embed_endpoint().embedding_model

    # ---------- генерация ----------

    def chat(self, messages: list[dict[str, str]], max_tokens: int = 1024, temperature: float = 0.2) -> str:
        try:
            return self._chat_with(self.chat_endpoint(), messages, max_tokens, temperature)
        except LLMError:
            fb = self._fallback(self.chat_endpoint())
            if fb is None:
                raise
            log.warning("чат: %s недоступен, откат на %s", self.chat_endpoint().name, fb.name)
            self._chat_provider = fb
            return self._chat_with(fb, messages, max_tokens, temperature)

    def _chat_with(self, p: _Provider, messages: list[dict[str, str]], max_tokens: int, temperature: float) -> str:
        with httpx.Client(timeout=CHAT_TIMEOUT) as c:
            r = c.post(
                f"{p.base_url}/chat/completions", headers=p._headers(),
                json={
                    "model": p.chat_model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
            )
        if r.status_code != 200:
            raise LLMError(f"chat {p.name} http={r.status_code}: {r.text[:200]}")
        return r.json()["choices"][0]["message"]["content"]

    def chat_stream(self, messages: list[dict[str, str]], max_tokens: int = 1024, temperature: float = 0.2) -> Iterator[str]:
        """Потоковая генерация: yields куски текста ответа."""
        p = self.chat_endpoint()
        with httpx.Client(timeout=CHAT_TIMEOUT) as c:
            with c.stream(
                "POST", f"{p.base_url}/chat/completions", headers=p._headers(),
                json={
                    "model": p.chat_model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": True,
                },
            ) as r:
                if r.status_code != 200:
                    raise LLMError(f"chat-stream {p.name} http={r.status_code}")
                for line in r.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[len("data:") :].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        delta = json.loads(payload)["choices"][0]["delta"]
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
                    piece = delta.get("content") or ""
                    if piece:
                        yield piece
