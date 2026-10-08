from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List


PAGE_RE = re.compile(r"_page_(\d+)")


def _page_number(path: Path) -> int:
    match = PAGE_RE.search(path.stem)
    return int(match.group(1)) if match else -1


def chunk_document(
    text_dir: str | Path,
    source_type: str,
    source_id: str,
    chunk_size: int = 1200,
    overlap: int = 250,
) -> List[Dict]:
    """Chunk only the supplied source directory."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text_dir = Path(text_dir)
    files = sorted(text_dir.rglob("*.txt"), key=lambda p: (_page_number(p), str(p)))
    chunks: List[Dict] = []

    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue

        page_number = _page_number(path)
        start = 0
        chunk_index = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    {
                        "id": f"{source_id}:{page_number}:{chunk_index}",
                        "text": chunk_text,
                        "source_type": source_type,
                        "source_id": source_id,
                        "document": path.name.rsplit("_page_", 1)[0],
                        "page_number": page_number,
                        "chunk_index": chunk_index,
                        "text_path": str(path),
                    }
                )
            if end >= len(text):
                break
            start = end - overlap
            chunk_index += 1

    return chunks
