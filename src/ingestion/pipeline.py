from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.ingestion.manifest_builder import build_manifest
from src.ingestion.page_renderer import render_pdf_pages
from src.ingestion.pdf_loader import load_pdf
from src.ingestion.session_manager import create_session
from src.ingestion.text_extractor import extract_text


def process_document(
    pdf_path,
    session_root: str | Path,
    session_id: Optional[str] = None,
):
    """Process one private PDF entirely inside one runtime session."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    session_root = Path(session_root)
    session_root.mkdir(parents=True, exist_ok=True)

    text_dir = session_root / "text"
    pages_dir = session_root / "pages"
    images_dir = session_root / "images"
    metadata_dir = session_root / "metadata"

    image_pages = load_pdf(pdf_path, images_dir)
    text_pages = extract_text(pdf_path, text_dir)
    rendered_pages = render_pdf_pages(pdf_path, pages_dir)

    manifest = build_manifest(
        pdf_path=pdf_path,
        text_pages=text_pages,
        image_pages=image_pages,
        rendered_pages=rendered_pages,
        output_dir=metadata_dir,
        session_id=session_id,
        session_root=session_root,
        source_type="private",
    )
    return manifest


def create_and_process_session(original_filename: str, pdf_bytes: bytes):
    session = create_session(original_filename, pdf_bytes)
    manifest = process_document(
        session["pdf_path"],
        session["session_root"],
        session_id=session["session_id"],
    )
    return {"session": session, "manifest": manifest}
