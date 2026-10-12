"""Ингест документации: markdown → чанки → эмбеддинги (с кэшем) → pgvector.

Запуск: python -m app.ingest [--data-dir data/docs]
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from app import db
from app.chunking import chunk_markdown
from app.config import Settings
from app.download_docs import parse_links
from app.llm import LLMService

log = logging.getLogger("rag.ingest")

DEFAULT_DATA_DIR = Path("data/docs")


def load_corpus(data_dir: Path) -> list[tuple[str, str, str]]:
    """[(source_file, doc_url, text)] — url берём из links.txt по имени файла."""
    url_by_name = {}
    links_path = data_dir.parent / "links.txt"
    if links_path.exists():
        for _raw, doc_url, name in parse_links(links_path):
            url_by_name[name] = doc_url
    corpus = []
    for md in sorted(data_dir.glob("*.md")):
        url = url_by_name.get(md.name, "")
        corpus.append((md.name, url, md.read_text(encoding="utf-8")))
    return corpus


def ingest(settings: Settings, data_dir: Path = DEFAULT_DATA_DIR) -> dict:
    """Возвращает статистику ингеста."""
    t0 = time.perf_counter()
    corpus = load_corpus(data_dir)
    if not corpus:
        raise SystemExit(f"в {data_dir} нет .md файлов — сначала python -m app.download_docs")

    llm = LLMService(settings)
    model = llm.embedding_model_name()

    pool = db.get_pool(settings)
    db.init_schema(pool, settings.embedding_dim)

    # смена модели эмбеддингов → кэш и векторы несовместимы, сбрасываем
    if db.get_meta(pool, "embedding_model") not in (None, model):
        log.warning("сменилась модель эмбеддингов (%s) — сбрасываю векторы", model)
        db.drop_all_embeddings(pool)
    db.set_meta(pool, "embedding_model", model)

    all_chunks = []
    for name, url, text in corpus:
        chunks = chunk_markdown(
            text, source_file=name, url=url,
            target_tokens=settings.chunk_target_tokens,
            max_tokens=settings.chunk_max_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
        )
        log.info("%s: %d чанков", name, len(chunks))
        all_chunks.extend(chunks)

    hashes = [db.content_hash(c.content) for c in all_chunks]

    # 1) кэш эмбеддингов в БД — не перегенерировать повторно
    with pool.connection() as conn:
        rows = conn.execute(
            "SELECT content_hash FROM embed_cache WHERE content_hash = ANY(%s) AND model = %s",
            (hashes, model),
        ).fetchall()
    cached = {r[0] for r in rows}

    todo = [(h, c) for h, c in zip(hashes, all_chunks) if h not in cached]
    log.info("эмбеддингов: всего %d, из кэша %d, посчитать %d", len(hashes), len(cached), len(todo))

    new_vectors: dict[str, list[float]] = {}
    for i in range(0, len(todo), 32):
        batch = todo[i : i + 32]
        vectors = llm.embed([c.content for _, c in batch])
        for (h, _c), v in zip(batch, vectors):
            new_vectors[h] = v
        log.info("эмбеддинги: %d/%d", min(i + 32, len(todo)), len(todo))

    # 2) новые векторы → кэш; чанки перезаписываются, векторы берём из кэша
    files = [name for name, _, _ in corpus]
    with pool.connection() as conn:
        conn.execute("DELETE FROM chunks WHERE source_file = ANY(%s)", (files,))
        for h, v in new_vectors.items():
            conn.execute(
                "INSERT INTO embed_cache (content_hash, model, embedding) VALUES (%s, %s, %s::vector) "
                "ON CONFLICT (content_hash) DO NOTHING",
                (h, model, _pgvec(v)),
            )
        for (h, c) in zip(hashes, all_chunks):
            conn.execute(
                "INSERT INTO chunks (source_file, title, section, url, content, token_count, content_hash, embedding) "
                "SELECT %s,%s,%s,%s,%s,%s,%s, embedding FROM embed_cache WHERE content_hash = %s AND model = %s",
                (c.source_file, c.title, c.section, c.url, c.content, c.token_count, h, h, model),
            )

    stats = {
        "files": len(files),
        "chunks": len(all_chunks),
        "embeddings_new": len(new_vectors),
        "embeddings_cached": len(cached),
        "seconds": round(time.perf_counter() - t0, 1),
    }
    log.info("ингест завершён: %s", json.dumps(stats, ensure_ascii=False))
    return stats


def _pgvec(vec: list[float] | None) -> str | None:
    if vec is None:
        return None
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Ингест документации в pgvector")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    settings = Settings()
    settings.load_env_file()
    stats = ingest(settings, args.data_dir)
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
