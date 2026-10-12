"""Прогон замеров качества RAG-бота по методике .claude/skills/rag-eval.

Набор вопросов: data/eval-questions.local (JSON, в .gitignore — чтобы вопросы
не «подгонялись» при разработке). Запуск:

    python scripts/eval_run.py

Выводит markdown-таблицу метрик и подробности по вопросам.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import db
from app.config import get_settings
from app.llm import LLMService
from app.pipeline import RAGPipeline
from app.retrieval import Retriever

QUESTIONS_FILE = Path("data/eval-questions.local")


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(0.95 * len(s)))]


def main() -> None:
    settings = get_settings()
    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))

    llm = LLMService(settings)
    retriever = Retriever(db.get_pool(settings), llm, top_k=settings.top_k, threshold=settings.relevance_threshold)
    pipeline = RAGPipeline(retriever, llm)

    rows = []
    for i, q in enumerate(questions, 1):
        t0 = time.perf_counter()
        res = pipeline.ask(q["question"])
        wall = time.perf_counter() - t0
        in_base = bool(q.get("expect_file"))
        cited_files = {res.sources[n - 1].source_file for n in res.cited_indices if 0 < n <= len(res.sources)}
        top_files = [s.source_file for s in res.sources]

        if in_base:
            hit = q["expect_file"] in cited_files or q["expect_file"] in top_files
            ok = hit and not res.refused
            rows.append(dict(n=i, kind="in", ok=ok, hit=hit, refused=res.refused,
                             ret=res.retrieval_latency, total=res.total_latency))
            status = "HIT" if hit and not res.refused else ("FALSE-REFUSAL" if res.refused else "MISS")
            print(f"[{i:2d}] {status:12s} {q['question'][:60]} "
                  f"(эталон: {q['expect_file']}, top: {top_files[:2]})")
        else:
            ok = res.refused
            rows.append(dict(n=i, kind="out", ok=ok, hit=False, refused=res.refused,
                             ret=res.retrieval_latency, total=res.total_latency))
            print(f"[{i:2d}] {'REFUSED-OK' if ok else 'HALLUCINATION?':12s} {q['question'][:60]}")

    in_rows = [r for r in rows if r["kind"] == "in"]
    out_rows = [r for r in rows if r["kind"] == "out"]
    ret_lat = [r["ret"] for r in rows]
    tot_lat = [r["total"] for r in rows]

    hits = sum(1 for r in in_rows if r["hit"] and not r["refused"])
    false_refusals = sum(1 for r in in_rows if r["refused"])
    refusals_ok = sum(1 for r in out_rows if r["refused"])

    md = f"""
| Метрика | Значение | Комментарий |
|---|---|---|
| Hit rate источника | {hits / len(in_rows) * 100:.0f} % | {hits} из {len(in_rows)} вопросов |
| Latency полная (медиана / p95) | {statistics.median(tot_lat):.1f} с / {p95(tot_lat):.1f} с | retrieval + генерация |
| Latency retrieval (медиана) | {statistics.median(ret_lat):.2f} с | только векторный поиск |
| Доля отказов (вне-базовые) | {refusals_ok / len(out_rows) * 100:.0f} % | корректные «нет ответа в базе» |
| Ложные отказы | {false_refusals} шт. | вопросы с ответом, где бот отказал |
"""
    print(md)
    Path("data/eval-results.local").write_text(md, encoding="utf-8")
    print("таблица сохранена: data/eval-results.local")
    db.get_pool(settings).close()


if __name__ == "__main__":
    main()
