from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pymupdf


def _format_table_markdown(table) -> str:
    """Convert an extracted PyMuPDF table into clean Markdown."""
    try:
        rows = table.extract()
        if not rows or len(rows) < 2:
            return ""
        clean_rows = []
        for r in rows:
            clean_rows.append([re.sub(r"\s+", " ", str(c or "")).strip() for c in r])
        headers = clean_rows[0]
        if not any(headers):
            return ""
        col_count = len(headers)
        md = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * col_count) + " |",
        ]
        for row in clean_rows[1:]:
            if len(row) < col_count:
                row = row + [""] * (col_count - len(row))
            md.append("| " + " | ".join(row[:col_count]) + " |")
        return "\n" + "\n".join(md) + "\n"
    except Exception:
        return ""


def extract_text(pdf_path, output_dir: Optional[str | Path] = None):
    """Extract page text into session/corpus-index-local text files, preserving structured tables."""
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir or (pdf_path.parent.parent / "text"))
    output_dir.mkdir(parents=True, exist_ok=True)

    document = pymupdf.open(pdf_path)
    pages = []
    try:
        for page_number, page in enumerate(document, start=1):
            text = (page.get_text() or "").strip()

            # Extract structured tables if present on the page
            try:
                tabs = page.find_tables()
                if tabs and tabs.tables:
                    tables_md = []
                    for t in tabs.tables:
                        t_str = _format_table_markdown(t)
                        if t_str.strip():
                            tables_md.append(t_str)
                    if tables_md:
                        tables_block = "### Page Tables (Structured Data):\n\n" + "\n\n".join(tables_md)
                        text = (tables_block + "\n\n" + text) if text else tables_block
            except Exception:
                pass

            text_path = output_dir / f"{pdf_path.stem}_page_{page_number:03d}.txt"
            text_path.write_text(text, encoding="utf-8")
            pages.append(
                {
                    "document": pdf_path.name,
                    "page_number": page_number,
                    "text_path": str(text_path),
                    "char_count": len(text),
                    "modality": "text",
                }
            )
    finally:
        document.close()

    return pages
