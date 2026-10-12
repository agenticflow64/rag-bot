"""Юнит-тесты чанкинга markdown."""

from app.chunking import _split_long, chunk_markdown, split_markdown
from app.config import estimate_tokens


MD = """# Gadgets API

Вводный раздел про реестр гаджетов и то, зачем он нужен команде.

## Настройка gadget

Раздел описывает настройку основных параметров гаджета в реестре.

### Калибровка

Раздел описывает калибровку: уровни, режим обслуживания, проверку.

## Телеметрия

Раздел описывает сбор телеметрии с гаджетов и её буферизацию.
"""


def test_split_by_headings():
    sections = split_markdown(MD)
    labels = [" > ".join(p) for p, _ in sections]
    assert labels == [
        "Gadgets API",                       # H1
        "Gadgets API > Настройка gadget",    # H2
        "Gadgets API > Настройка gadget > Калибровка",
        "Gadgets API > Телеметрия",
    ]
    texts = {t for _, t in sections}
    assert any("калибровку" in t for t in texts)


def test_anchor_suffix_stripped():
    md = "# T\n\n## Section { #section }\n\nтекст"
    sections = split_markdown(md)
    path, text = next(p for p in sections if "Section" in " > ".join(p[0]))
    assert path == ["T", "Section"]
    assert "{ #" not in text


def test_chunk_metadata():
    chunks = chunk_markdown(MD, source_file="gadgets.md", url="http://x/gadgets/")
    by_section = {c.section: c for c in chunks}
    # H1 не дублируется в пути: путь начинается с H2
    assert "Настройка gadget > Калибровка" in by_section
    cal = by_section["Настройка gadget > Калибровка"]
    assert cal.title == "Gadgets API"
    assert cal.source_file == "gadgets.md"
    assert cal.url == "http://x/gadgets/"
    assert cal.token_count == estimate_tokens(cal.content)
    assert cal.token_count > 0


def test_long_section_windowed_with_overlap():
    paras = [f"Параграф номер {i} с достаточно длинным текстом для наполнения окна. " * 3 for i in range(40)]
    long_text = "\n\n".join(paras)
    windows = _split_long(long_text, target_tokens=200, max_tokens=300, overlap_tokens=50)
    assert len(windows) > 1
    assert all(estimate_tokens(w) <= 600 for w in windows)  # max_tokens*2 жёсткий предел
    # перекрытие: конец предыдущего окна в начале следующего
    assert windows[1][:30].strip() in windows[0] or windows[1][:10] in windows[0]


def test_short_section_single_chunk():
    md = "# T\n\n" + "преамбула документа достаточной длины для фильтра. " * 2 + "\n\n## S\n\n" + "короткий, но осмысленный текст секции. " * 2
    chunks = chunk_markdown(md, source_file="t.md", url="u")
    assert len(chunks) == 2  # H1-преамбула + секция


def test_too_tiny_sections_dropped():
    md = "# Заголовок\n\n" + "достаточно длинная преамбула документа для прохода фильтра. " * 2 + "\n\n## S\n\nок"
    chunks = chunk_markdown(md, source_file="t.md", url="u")
    # секция «ок» короче 15 токенов — отбрасывается, остаётся только H1-преамбула
    assert len(chunks) == 1


def test_no_headings_still_chunks():
    text = "Просто длинный текст без какого-либо заголовка в документе. " * 100
    chunks = chunk_markdown(text, source_file="t.md", url="u")
    assert chunks
    assert all(c.section == "t.md" for c in chunks)  # без H1 section = имя файла
