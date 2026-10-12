"""Чанкинг markdown-документации: по заголовкам, окно ~500–800 токенов с перекрытием."""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.config import estimate_tokens

_HEADING_RE = re.compile(r"^(#{1,4})\s+(.*)$")
# некоторые генераторы доков добавляют к заголовкам явные якоря: "## Title { #anchor }"
_ANCHOR_RE = re.compile(r"\s*\{\s*#[^}]*\}\s*$")


def _clean_heading(title: str) -> str:
    return _ANCHOR_RE.sub("", title).strip()


@dataclass
class Chunk:
    source_file: str
    title: str      # заголовок документа (H1)
    section: str    # путь заголовков, например "Advanced usage > Sessions"
    url: str
    content: str
    token_count: int


@dataclass
class _RawSection:
    level: int | None  # None — преамбула до первого заголовка
    title: str
    lines: list[str]

    @property
    def text(self) -> str:
        return "\n".join(self.lines).strip()


def split_markdown(text: str) -> list[tuple[list[str], str]]:
    """Режет markdown на секции по заголовкам #..####.

    Возвращает список (путь_заголовков, текст_секции_со_строкой_заголовка).
    """
    raw: list[_RawSection] = [_RawSection(level=None, title="", lines=[])]
    for line in text.splitlines():
        m = _HEADING_RE.match(line)
        if m:
            level, title_txt = len(m.group(1)), _clean_heading(m.group(2))
            raw.append(_RawSection(level=level, title=title_txt, lines=[f"{'#' * level} {title_txt}"]))
        else:
            raw[-1].lines.append(line)

    stack: list[tuple[int, str]] = []
    result: list[tuple[list[str], str]] = []
    for sec in raw:
        if sec.level is None:
            if sec.text:
                result.append(([], sec.text))
            continue
        while stack and stack[-1][0] >= sec.level:
            stack.pop()
        stack.append((sec.level, sec.title))
        if sec.text:
            result.append(([t for _, t in stack], sec.text))
    return result


def _split_long(
    text: str, target_tokens: int, max_tokens: int, overlap_tokens: int, context: str = ""
) -> list[str]:
    """Длинную секцию режет на окна по абзацам с перекрытием.

    `context` — строка заголовка секции: добавляется в начало каждого окна,
    кроме первого (там заголовок уже есть), чтобы окно не теряло тему.
    """
    if estimate_tokens(text) <= max_tokens:
        return [text]

    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return [text]

    char_target, char_overlap = target_tokens * 4, overlap_tokens * 4
    windows: list[str] = []
    current: list[str] = []
    current_len = 0

    def flush():
        nonlocal current, current_len
        if not current:
            return
        windows.append("\n\n".join(current).strip())
        tail = "\n\n".join(current)[-char_overlap:]
        current = [tail] if tail.strip() else []
        current_len = len(tail)

    for para in paragraphs:
        if current and current_len + len(para) > char_target:
            flush()
        current.append(para)
        current_len += len(para) + 2
    if current:
        windows.append("\n\n".join(current).strip())

    # патологически длинный абзац (код без пустых строк) режем жёстко по символам
    final: list[str] = []
    for w in windows:
        while estimate_tokens(w) > max_tokens * 2:
            final.append(w[:char_target])
            w = w[char_target - char_overlap :]
        final.append(w)

    if context:
        final = [final[0]] + [f"{context}\n\n{w}" for w in final[1:]]
    return final


def chunk_markdown(
    text: str,
    source_file: str,
    url: str,
    target_tokens: int = 600,
    max_tokens: int = 800,
    overlap_tokens: int = 100,
) -> list[Chunk]:
    """Полный конвейер чанкинга одного markdown-файла."""
    doc_title = ""
    m = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    if m:
        doc_title = m.group(1).strip()
    if not doc_title:
        doc_title = source_file

    chunks: list[Chunk] = []
    for heading_path, sec_text in split_markdown(text):
        path = heading_path
        if path and path[0] == doc_title:  # H1 совпадает с заголовком документа — не дублируем
            path = path[1:]
        section_label = " > ".join(path) if path else doc_title
        ctx = section_label if section_label != doc_title else doc_title
        for piece in _split_long(sec_text, target_tokens, max_tokens, overlap_tokens, context=ctx):
            piece = piece.strip()
            if not piece or estimate_tokens(piece) < 15:
                continue
            chunks.append(
                Chunk(
                    source_file=source_file,
                    title=doc_title,
                    section=section_label,
                    url=url,
                    content=piece,
                    token_count=estimate_tokens(piece),
                )
            )
    return chunks
