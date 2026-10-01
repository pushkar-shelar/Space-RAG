from __future__ import annotations

from pathlib import Path
from typing import Optional

import pymupdf


def load_pdf(pdf_path, image_output_dir: Optional[str | Path] = None):
    """Extract embedded raster images into a supplied directory."""
    pdf_path = Path(pdf_path)
    output_dir = Path(image_output_dir or (pdf_path.parent.parent / "images"))
    output_dir.mkdir(parents=True, exist_ok=True)

    document = pymupdf.open(pdf_path)
    pages = []

    try:
        for page_number, page in enumerate(document, start=1):
            page_data = {
                "document": pdf_path.name,
                "page_number": page_number,
                "images": [],
            }

            for image_number, image in enumerate(page.get_images(full=True), start=1):
                xref = image[0]
                image_data = document.extract_image(xref)
                extension = image_data["ext"]
                image_path = output_dir / (
                    f"{pdf_path.stem}_page_{page_number}_img_{image_number}.{extension}"
                )
                image_path.write_bytes(image_data["image"])
                page_data["images"].append(
                    {
                        "image_id": image_number,
                        "path": str(image_path),
                        "extension": extension,
                        "xref": xref,
                        "modality": "image",
                    }
                )
            pages.append(page_data)
    finally:
        document.close()

    return pages
