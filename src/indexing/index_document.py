from __future__ import annotations

from pathlib import Path

from src.indexing.bm25_index import BM25SourceIndex
from src.indexing.chroma_indexer import ChromaTextIndex


def index_private_session(session_root: str | Path) -> dict:
    """Build fast text indexes immediately. ColPali stays lazy until needed."""
    root = Path(session_root)
    text_dir = root / "text"
    chroma_dir = root / "chroma"

    index_dir = root / "indexes"
    index_dir.mkdir(parents=True, exist_ok=True)

    source_id = root.name

    bm25 = BM25SourceIndex(index_dir / "bm25.pkl")
    bm25_count = bm25.build([text_dir], "private", source_id)
    # Also copy or write to root / bm25.pkl for compatibility
    import shutil
    try:
        shutil.copy2(index_dir / "bm25.pkl", root / "bm25.pkl")
    except Exception:
        pass

    chroma = ChromaTextIndex(chroma_dir, "private_text")
    dense_count = chroma.build([text_dir], "private", source_id)
    chroma.close()

    return {
        "source_type": "private",
        "source_id": source_id,
        "bm25_chunks": bm25_count,
        "dense_chunks": dense_count,
        "visual_pages": len(list((root / "pages").glob("*.png"))),
    }
