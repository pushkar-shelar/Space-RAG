import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.session_manager import (
    create_session,
    cleanup_session,
    get_session_root,
)
from src.ingestion.pipeline import process_document
from src.indexing.index_document import index_private_session
from src.retrieval.source_text_retriever import SourceTextRetriever


def test_first_5_tasks():
    print("\n" + "=" * 70)
    print("VERIFICATION SUITE: Tasks 1 to 5 (Session, Ingestion, Corpus, Private, Retrieval)")
    print("=" * 70)

    # ----------------------------------------------------
    # TASK 3: Corpus Index Verification
    # ----------------------------------------------------
    print("\n--- [TASK 3] Permanent Corpus Index Check ---")
    corpus_root = PROJECT_ROOT / "data" / "corpus_index"
    assert (corpus_root / "corpus_manifest.json").exists(), "corpus_manifest.json is missing!"
    assert (corpus_root / "bm25.pkl").exists(), "corpus bm25.pkl is missing!"
    assert (corpus_root / "chroma").exists(), "corpus chroma/ is missing!"
    assert (corpus_root / "colpali.pt").exists(), "corpus colpali.pt is missing!"
    print("[PASS] Task 3 PASSED: Permanent Corpus Index (BM25, Chroma, ColPali, Manifest) verified.")

    # ----------------------------------------------------
    # TASK 1: Session Manager Verification
    # ----------------------------------------------------
    print("\n--- [TASK 1] Session Manager Check ---")
    sample_pdf = PROJECT_ROOT / "data" / "corpus" / "TROPICS.pdf"
    assert sample_pdf.exists()

    session = create_session("my_private_satellite.pdf", sample_pdf.read_bytes())
    session_id = session["session_id"]
    session_root = Path(session["session_root"])

    assert session_root.exists(), "Session folder was not created!"
    assert (session_root / "document" / "my_private_satellite.pdf").exists()
    print(f"[PASS] Task 1 PASSED: Session created at {session_root}")

    # ----------------------------------------------------
    # TASK 2: Private Ingestion Pipeline Check
    # ----------------------------------------------------
    print("\n--- [TASK 2] Private Ingestion Check ---")
    t0 = time.time()
    manifest = process_document(session["pdf_path"], session_root, session_id=session_id)
    t_ingest = time.time() - t0
    assert manifest["total_pages"] == 11, f"Expected 11 pages, got {manifest['total_pages']}"
    assert manifest["source_type"] == "private"
    assert (session_root / "text").exists()
    assert (session_root / "pages").exists()
    print(f"[PASS] Task 2 PASSED: 11 pages ingested strictly inside session sandbox in {t_ingest:.2f}s.")

    # ----------------------------------------------------
    # TASK 4: Private Document Indexing Check
    # ----------------------------------------------------
    print("\n--- [TASK 4] Private Document Indexing Check ---")
    t0 = time.time()
    idx_stats = index_private_session(session_root)
    t_idx = time.time() - t0
    assert idx_stats["source_type"] == "private"
    assert idx_stats["bm25_chunks"] > 0
    assert idx_stats["dense_chunks"] > 0
    assert (session_root / "indexes" / "bm25.pkl").exists()
    assert (session_root / "chroma").exists()
    print(f"[PASS] Task 4 PASSED: Private BM25 & Chroma built in {t_idx:.2f}s (Chunks: {idx_stats['dense_chunks']}).")

    # ----------------------------------------------------
    # TASK 5: BM25 + Dense Source-Aware Retrieval Check
    # ----------------------------------------------------
    print("\n--- [TASK 5] Source-Aware Hybrid Retrieval Check ---")
    query = "What is the constellation and mission objective of TROPICS?"
    
    # 1. Search Corpus
    retriever_corpus = SourceTextRetriever(corpus_root, "corpus")
    corpus_results = retriever_corpus.search(query, top_k=3)
    retriever_corpus.close()

    assert len(corpus_results) > 0, "Corpus retriever returned 0 results"
    for r in corpus_results:
        assert r["source_type"] == "corpus", f"Expected source_type='corpus', got {r['source_type']}"
        assert "text_score" in r and r["text_score"] > 0
    print(f"[PASS] Corpus retrieval returned {len(corpus_results)} results (Top score: {corpus_results[0]['text_score']:.4f})")

    # 2. Search Private Session
    retriever_private = SourceTextRetriever(session_root, "private")
    private_results = retriever_private.search(query, top_k=3)
    retriever_private.close()

    assert len(private_results) > 0, "Private retriever returned 0 results"
    for r in private_results:
        assert r["source_type"] == "private", f"Expected source_type='private', got {r['source_type']}"
        assert r["metadata"]["document"] == "my_private_satellite"
    print(f"[PASS] Private retrieval returned {len(private_results)} results (Doc: {private_results[0]['metadata']['document']}, Page: {private_results[0]['metadata']['page_number']})")
    print("[PASS] Task 5 PASSED: Both corpus and private source-aware retrievers cleanly retrieve and score.")

    # ----------------------------------------------------
    # Teardown & Privacy Check
    # ----------------------------------------------------
    print("\n--- [Cleanup] Testing Session Purge ---")
    cleaned = cleanup_session(session_id)
    assert cleaned is True
    assert not session_root.exists(), "Session folder was not purged!"
    print("[PASS] Cleanup PASSED: Session directory purged completely.")

    print("\n" + "=" * 70)
    print("ALL 5 TASKS VERIFIED AND PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    test_first_5_tasks()
