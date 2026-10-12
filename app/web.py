"""Веб-интерфейс: FastAPI + одна статическая страница, SSE-стриминг ответа.

Запуск: python -m app.web (uvicorn) → http://localhost:8800
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel

from app.ask import build_pipeline
from app.config import Settings
from app.prompts import build_messages, parse_citations
from app.pipeline import REFUSAL_ANSWER

log = logging.getLogger("rag.web")

app = FastAPI(title="Ассистент поддержки по базе знаний")

_settings: Settings | None = None

KB_DIR = Path(__file__).resolve().parent.parent / "data" / "kb"
_KB_FILE_RE = re.compile(r"^[a-z0-9-]+\.md$")


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        from app.config import get_settings as load_settings

        _settings = load_settings()
    return _settings


class AskBody(BaseModel):
    question: str
    top_k: int | None = None


@app.get("/")
def index() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/kb/{name}", response_class=PlainTextResponse)
def kb_doc(name: str) -> PlainTextResponse:
    """Выдача документа базы знаний: на него ведут ссылки из источников ответа."""
    if not _KB_FILE_RE.match(name):
        raise HTTPException(status_code=404, detail="нет такого документа")
    path = KB_DIR / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="нет такого документа")
    return PlainTextResponse(path.read_text(encoding="utf-8"), media_type="text/markdown; charset=utf-8")


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/api/ask")
def ask(body: AskBody) -> StreamingResponse:
    question = body.question.strip()
    if not question:
        return StreamingResponse(iter([_sse("error", {"message": "пустой вопрос"})]), media_type="text/event-stream")

    def generate():
        pipeline = build_pipeline(get_settings())
        chunks, retrieval_latency = pipeline.retriever.search(question, top_k=body.top_k)

        sources = [
            {
                "n": i,
                "file": c.source_file,
                "section": c.section,
                "url": c.url,
                "score": round(c.score, 3),
                "quote": c.content[:280].strip() + ("…" if len(c.content) > 280 else ""),
            }
            for i, c in enumerate(chunks, start=1)
        ]
        yield _sse("sources", {"sources": sources, "retrieval_latency": round(retrieval_latency, 3)})

        if not pipeline.retriever.is_relevant(chunks):
            yield _sse("delta", {"text": REFUSAL_ANSWER})
            yield _sse("done", {"refused": True, "retrieval_latency": round(retrieval_latency, 3), "total_latency": round(retrieval_latency, 3)})
            return

        answer_parts: list[str] = []
        try:
            for piece in pipeline.llm.chat_stream(build_messages(question, chunks)):
                answer_parts.append(piece)
                yield _sse("delta", {"text": piece})
        except Exception as e:  # noqa: BLE001 — стриму нельзя уронить соединение молча
            log.exception("ошибка генерации")
            yield _sse("error", {"message": f"ошибка генерации: {e}"})
            return

        answer = "".join(answer_parts).strip()
        refused = answer.startswith("Вне базы знаний")
        cited = [n for n in parse_citations(answer) if 1 <= n <= len(chunks)]
        yield _sse("done", {"refused": refused, "cited": cited, "retrieval_latency": round(retrieval_latency, 3)})

    return StreamingResponse(generate(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


def main() -> None:
    import os

    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    get_settings()  # прогрев конфига
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8800")), log_level="info")


if __name__ == "__main__":
    main()
