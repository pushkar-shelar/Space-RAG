from __future__ import annotations

import json
import re
from pathlib import Path

from src.indexing.bm25_index import BM25SourceIndex
from src.indexing.chroma_indexer import ChromaTextIndex
from src.indexing.visual_indexer import ColPaliPageIndex
from src.ingestion.manifest_builder import build_manifest
from src.ingestion.page_renderer import render_pdf_pages
from src.ingestion.pdf_loader import load_pdf
from src.ingestion.text_extractor import extract_text

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS_DIR = PROJECT_ROOT / "data" / "corpus"
DEFAULT_CORPUS_INDEX = PROJECT_ROOT / "data" / "corpus_index"


def _safe_name(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._ -]", "_", name)
    return name.strip(" .") or "document"


def build_corpus_index(
    corpus_dir: str | Path = DEFAULT_CORPUS_DIR,
    corpus_index_root: str | Path = DEFAULT_CORPUS_INDEX,
) -> dict:
    """Build the permanent knowledge corpus from PDFs under data/corpus."""
    corpus_dir = Path(corpus_dir)
    corpus_index_root = Path(corpus_index_root)
    corpus_index_root.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(corpus_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in {corpus_dir}")

    document_roots = []
    visual_paths = []
    visual_metadata = []
    manifests = []

    for pdf_path in pdfs:
        doc_root = corpus_index_root / "documents" / _safe_name(pdf_path.stem)
        text_dir = doc_root / "text"
        pages_dir = doc_root / "pages"
        images_dir = doc_root / "images"
        metadata_dir = doc_root / "metadata"

        image_pages = load_pdf(pdf_path, images_dir)
        text_pages = extract_text(pdf_path, text_dir)
        rendered_pages = render_pdf_pages(pdf_path, pages_dir)
        manifest = build_manifest(
            pdf_path=pdf_path,
            text_pages=text_pages,
            image_pages=image_pages,
            rendered_pages=rendered_pages,
            output_dir=metadata_dir,
            source_type="corpus",
            session_root=corpus_index_root,
        )
        manifests.append(manifest)
        document_roots.append(doc_root)

        for page in rendered_pages:
            visual_paths.append(page["image_path"])
            visual_metadata.append(
                {
                    "path": page["image_path"],
                    "document": pdf_path.name,
                    "page_number": page["page_number"],
                }
            )

    text_dirs = [root / "text" for root in document_roots]
    bm25 = BM25SourceIndex(corpus_index_root / "bm25.pkl")
    bm25_count = bm25.build(text_dirs, "corpus", "corpus")

    chroma = ChromaTextIndex(corpus_index_root / "chroma", "corpus_text")
    dense_count = chroma.build(text_dirs, "corpus", "corpus")
    chroma.close()

    visual_count = ColPaliPageIndex(corpus_index_root / "colpali.pt").build(
        visual_paths,
        visual_metadata,
    )

    manifest_path = corpus_index_root / "corpus_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "source_type": "corpus",
                "corpus_dir": str(corpus_dir),
                "document_count": len(manifests),
                "documents": [m["document"] for m in manifests],
                "bm25_chunks": bm25_count,
                "dense_chunks": dense_count,
                "visual_pages": visual_count,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "documents": len(manifests),
        "bm25_chunks": bm25_count,
        "dense_chunks": dense_count,
        "visual_pages": visual_count,
        "manifest": str(manifest_path),
    }


if __name__ == "__main__":
    print(build_corpus_index())
