from __future__ import annotations

from pathlib import Path
from typing import Optional

import pymupdf


def extract_text(pdf_path, output_dir: Optional[str | Path] = None):
    """Extract page text into session/corpus-index-local text files."""
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir or (pdf_path.parent.parent / "text"))
    output_dir.mkdir(parents=True, exist_ok=True)

    document = pymupdf.open(pdf_path)
    pages = []
    try:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text()
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
