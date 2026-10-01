from __future__ import annotations

from pathlib import Path
from typing import Optional

import pymupdf


def render_pdf_pages(pdf_path, output_dir: Optional[str | Path] = None):
    """Render full PDF pages for visual retrieval."""
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir or (pdf_path.parent.parent / "pages"))
    output_dir.mkdir(parents=True, exist_ok=True)

    document = pymupdf.open(pdf_path)
    rendered = []
    try:
        for page_number, page in enumerate(document, start=1):
            pixmap = page.get_pixmap(
                matrix=pymupdf.Matrix(2, 2),
                alpha=False,
            )
            image_path = output_dir / f"{pdf_path.stem}_page_{page_number:03d}.png"
            pixmap.save(image_path)
            rendered.append(
                {
                    "document": pdf_path.name,
                    "page_number": page_number,
                    "image_path": str(image_path),
                    "modality": "page_image",
                    "width": pixmap.width,
                    "height": pixmap.height,
                }
            )
    finally:
        document.close()

    return rendered
