import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.session_manager import (
    create_session,
    cleanup_session,
    cleanup_stale_sessions,
)
from src.ingestion.pipeline import process_document
from src.indexing.index_document import index_private_session


def test_session_lifecycle():
    print("\n" + "=" * 60)
    print("TEST: Private Session Lifecycle & Isolation")
    print("=" * 60)

    # Use existing sample PDF from corpus
    sample_pdf = PROJECT_ROOT / "data" / "corpus" / "GSFC-HDBK-8007_Admn.pdf"
    assert sample_pdf.exists(), f"Sample PDF not found at {sample_pdf}"

    pdf_bytes = sample_pdf.read_bytes()

    # 1. Create temporary session
    print("[1] Creating private session...")
    session = create_session("test_private_doc.pdf", pdf_bytes)
    session_id = session["session_id"]
    session_root = Path(session["session_root"])

    assert session_root.exists(), "Session root was not created"
    assert (session_root / "document" / "test_private_doc.pdf").exists()
    print(f"Session created at: {session_root}")

    # 2. Ingest document in session
    print("[2] Processing document in session...")
    manifest = process_document(
        session["pdf_path"],
        session["session_root"],
        session_id=session_id,
    )
    assert manifest["total_pages"] > 0, "No pages processed in manifest"
    print(f"Ingested {manifest['total_pages']} pages into session")

    # 3. Index private session
    print("[3] Indexing private session...")
    index_stats = index_private_session(session_root)
    assert index_stats["source_type"] == "private"
    assert (session_root / "indexes" / "bm25.pkl").exists(), "BM25 index missing"
    assert (session_root / "chroma").exists(), "Chroma dir missing"
    print(f"Indexes built: {index_stats['bm25_chunks']} BM25 chunks, {index_stats['dense_chunks']} dense chunks")

    # 4. Clean up session
    print("[4] Cleaning up session...")
    cleanup_session(session_id)
    assert not session_root.exists(), "Session root was not deleted after cleanup!"
    print("Session root successfully removed (Privacy verified).")

    print("\n" + "=" * 60)
    print("TEST RESULT: PASS")
    print("=" * 60)


if __name__ == "__main__":
    test_session_lifecycle()
