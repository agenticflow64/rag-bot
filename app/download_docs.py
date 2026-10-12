"""Скачивание исходников документации (markdown) в data/docs/.

Список источников: data/links.txt — строки вида
    <raw_url> <док-url>
Запуск: python -m app.download_docs
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import httpx

log = logging.getLogger("rag.download")

LINKS_FILE = Path("data/links.txt")
DOCS_DIR = Path("data/docs")


def parse_links(path: Path = LINKS_FILE) -> list[tuple[str, str, str]]:
    """[(raw_url, doc_url, имя_файла)] из links.txt.

    Имя файла — путь внутри docs с «/» → «__», чтобы одноимённые index.md
    из разных папок не затирали друг друга.
    """
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        raw_url = parts[0]
        doc_url = parts[1] if len(parts) > 1 else raw_url
        marker = "/docs/en/docs/"
        if marker in raw_url:
            inner = raw_url.split(marker, 1)[1]
        else:
            inner = raw_url.rsplit("/", 1)[-1]
        out.append((raw_url, doc_url, inner.replace("/", "__")))
    return out


def download_all(links_path: Path = LINKS_FILE, docs_dir: Path = DOCS_DIR) -> list[Path]:
    docs_dir.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        for raw_url, _doc_url, name in parse_links(links_path):
            dest = docs_dir / name
            if dest.exists() and dest.stat().st_size > 0:
                files.append(dest)
                continue
            r = client.get(raw_url)
            if r.status_code != 200:
                log.warning("SKIP %s: http %s", raw_url, r.status_code)
                continue
            dest.write_bytes(r.content)
            log.info("ok %s (%d KB)", name, len(r.content) // 1024)
            files.append(dest)
    return files


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    got = download_all(Path(sys.argv[1]) if len(sys.argv) > 1 else LINKS_FILE)
    print(f"скачано файлов: {len(got)}")
