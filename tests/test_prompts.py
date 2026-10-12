"""Юнит-тесты сборки промптов и разбора цитат."""

from app.prompts import SYSTEM_PROMPT, build_context, build_messages, parse_citations
from app.retrieval import RetrievedChunk


def _chunk(n: int, content: str = "текст фрагмента") -> RetrievedChunk:
    return RetrievedChunk(
        id=n, source_file=f"file{n}.md", title="T", section=f"S{n}",
        url=f"http://x/{n}", content=content, token_count=10, score=0.5,
    )


def test_system_prompt_rules():
    assert "строго" in SYSTEM_PROMPT
    assert "Вне базы знаний" in SYSTEM_PROMPT
    assert "[n]" in SYSTEM_PROMPT


def test_build_context_numbers_chunks():
    ctx = build_context([_chunk(1), _chunk(2)])
    assert ctx.startswith("[1] Файл: file1.md")
    assert "[2] Файл: file2.md" in ctx
    assert "---" in ctx


def test_build_context_truncates_long_content():
    ch = _chunk(1, content="x" * 5000)
    ctx = build_context([ch])
    assert len(ch.content[:2000]) <= len(ctx) < 5000 + 200


def test_build_messages_structure():
    msgs = build_messages("вопрос?", [_chunk(1)])
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert "вопрос?" in msgs[1]["content"]
    assert "file1.md" in msgs[1]["content"]


def test_parse_citations_unique_in_order():
    assert parse_citations("ответ [2] и ещё [1], потом снова [2]") == [2, 1]


def test_parse_citations_empty():
    assert parse_citations("без цитат") == []
