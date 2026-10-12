"""CLI: вопрос боту из терминала.

Запуск: python -m app.ask "Как использовать сессии в requests?"
"""

from __future__ import annotations

import sys

from app import db
from app.config import Settings, get_settings
from app.llm import LLMService
from app.pipeline import RAGPipeline
from app.retrieval import Retriever


def build_pipeline(settings: Settings | None = None) -> RAGPipeline:
    s = settings or Settings()
    pool = db.get_pool(s)
    llm = LLMService(s)
    retriever = Retriever(pool, llm, top_k=s.top_k, threshold=s.relevance_threshold)
    return RAGPipeline(retriever, llm)


def main() -> int:
    if len(sys.argv) < 2 or not sys.argv[1].strip():
        print("использование: python -m app.ask \"вопрос\" [--top-k N]", file=sys.stderr)
        return 2

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    top_k = None
    if "--top-k" in sys.argv:
        i = sys.argv.index("--top-k")
        top_k = int(sys.argv[i + 1])

    settings = get_settings()
    result = build_pipeline(settings).ask(args[0], top_k=top_k)
    db.get_pool(settings).close()

    print(result.answer)
    print()
    if result.refused:
        print("(ретривер: релевантных фрагментов не найдено)")
    elif result.sources:
        print("Источники:")
        cited = set(result.cited_indices) or set(range(1, len(result.sources) + 1))
        for n in sorted(cited):
            ch = result.sources[n - 1]
            print(f"  [{n}] {ch.source_file} — {ch.section} (score {ch.score:.2f})")
            if ch.url:
                print(f"      {ch.url}")
    print()
    print(f"latency: retrieval {result.retrieval_latency:.2f} с, всего {result.total_latency:.2f} с")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
