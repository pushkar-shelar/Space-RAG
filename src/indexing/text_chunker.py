from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List


PAGE_RE = re.compile(r"_page_(\d+)")


def _page_number(path: Path) -> int:
    match = PAGE_RE.search(path.stem)
    return int(match.group(1)) if match else -1


def _is_table_line(line: str) -> bool:
    s = line.strip()
    return s.startswith("|") and s.endswith("|") and s.count("|") >= 2


def _split_into_blocks(text: str) -> List[str]:
    """Split text into paragraphs and contiguous markdown table blocks."""
    lines = text.splitlines()
    blocks: List[str] = []
    current_block: List[str] = []
    in_table = False

    for line in lines:
        is_tbl = _is_table_line(line)
        if is_tbl:
            if not in_table and current_block:
                blocks.append("\n".join(current_block).strip())
                current_block = []
            in_table = True
            current_block.append(line)
        else:
            if in_table and current_block:
                blocks.append("\n".join(current_block).strip())
                current_block = []
                in_table = False
            if line.strip() == "":
                if current_block:
                    blocks.append("\n".join(current_block).strip())
                    current_block = []
            else:
                current_block.append(line)

    if current_block:
        blocks.append("\n".join(current_block).strip())

    return [b for b in blocks if b]


def _chunk_table(table_block: str, chunk_size: int) -> List[str]:
    """Split a large Markdown table while repeating header rows on every chunk."""
    rows = table_block.splitlines()
    if len(rows) <= 2:
        return [table_block]

    header = rows[0]
    separator = rows[1] if len(rows) > 1 and "---" in rows[1] else "| " + " | ".join(["---"] * header.count("|"))
    header_prefix = f"{header}\n{separator}\n"

    chunks = []
    current_chunk = header_prefix
    for row in rows[2:]:
        candidate = f"{current_chunk}{row}\n"
        if len(candidate) > chunk_size and len(current_chunk) > len(header_prefix):
            chunks.append(current_chunk.strip())
            current_chunk = f"{header_prefix}{row}\n"
        else:
            current_chunk = candidate

    if current_chunk.strip() and current_chunk != header_prefix:
        chunks.append(current_chunk.strip())

    return chunks if chunks else [table_block]


def _chunk_text_content(text: str, chunk_size: int = 1200, overlap: int = 250) -> List[str]:
    """Structure-aware chunking preserving tables and semantic paragraph boundaries."""
    blocks = _split_into_blocks(text)
    if not blocks:
        return []

    chunks: List[str] = []
    current_chunk = ""

    for block in blocks:
        # Check if block is a Markdown table
        if _is_table_line(block.splitlines()[0]):
            if len(block) > chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                table_chunks = _chunk_table(block, chunk_size)
                chunks.extend(table_chunks)
                continue
            else:
                # If table fits into current chunk, append it; else flush and start new
                if len(current_chunk) + len(block) + 2 <= chunk_size:
                    current_chunk = f"{current_chunk}\n\n{block}" if current_chunk else block
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = block
                continue

        # Regular text block
        if len(block) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            # Fall back to sliding window for enormous single paragraphs
            start = 0
            while start < len(block):
                end = min(start + chunk_size, len(block))
                sub_chunk = block[start:end].strip()
                if sub_chunk:
                    chunks.append(sub_chunk)
                if end >= len(block):
                    break
                start = end - overlap
        else:
            if len(current_chunk) + len(block) + 2 <= chunk_size:
                current_chunk = f"{current_chunk}\n\n{block}" if current_chunk else block
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = block

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def chunk_document(
    text_dir: str | Path,
    source_type: str,
    source_id: str,
    chunk_size: int = 1200,
    overlap: int = 250,
) -> List[Dict]:
    """Chunk only the supplied source directory using structure-aware parsing."""
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
        doc_name = path.name.rsplit("_page_", 1)[0]
        text_chunks = _chunk_text_content(text, chunk_size=chunk_size, overlap=overlap)

        for chunk_index, chunk_text in enumerate(text_chunks):
            if chunk_text.strip():
                chunks.append(
                    {
                        "id": f"{source_id}:{doc_name}:{page_number}:{chunk_index}",
                        "text": chunk_text.strip(),
                        "source_type": source_type,
                        "source_id": source_id,
                        "document": doc_name,
                        "page_number": page_number,
                        "chunk_index": chunk_index,
                        "text_path": str(path),
                    }
                )

    return chunks
