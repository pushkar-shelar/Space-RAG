from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


def build_manifest(
    pdf_path,
    text_pages,
    image_pages,
    rendered_pages,
    output_dir: Optional[str | Path] = None,
    session_id: Optional[str] = None,
    session_root: Optional[str | Path] = None,
    source_type: str = "private",
):
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir or (pdf_path.parent.parent / "metadata"))
    output_dir.mkdir(parents=True, exist_ok=True)

    total_pages = max(len(text_pages), len(image_pages), len(rendered_pages))
    pages = []

    for page_number in range(1, total_pages + 1):
        text_page = next((p for p in text_pages if p["page_number"] == page_number), None)
        image_page = next((p for p in image_pages if p["page_number"] == page_number), None)
        rendered_page = next((p for p in rendered_pages if p["page_number"] == page_number), None)

        page_record = {
            "page_number": page_number,
            "text": None,
            "page_image": None,
            "embedded_images": [],
        }
        if text_page:
            page_record["text"] = {
                "path": text_page["text_path"],
                "modality": "text",
                "char_count": text_page["char_count"],
            }
        if image_page:
            page_record["embedded_images"] = image_page["images"]
        if rendered_page:
            page_record["page_image"] = {
                "path": rendered_page["image_path"],
                "modality": "page_image",
                "width": rendered_page["width"],
                "height": rendered_page["height"],
            }
        pages.append(page_record)

    manifest = {
        "session_id": session_id,
        "session_root": str(session_root) if session_root is not None else None,
        "source_type": source_type,
        "document": pdf_path.name,
        "document_path": str(pdf_path),
        "total_pages": total_pages,
        "pages": pages,
    }

    path = output_dir / "document_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest["manifest_path"] = str(path)
    return manifest
