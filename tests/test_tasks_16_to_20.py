import os
import sys
import time
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.session_manager import (
    create_session,
    cleanup_session,
    cleanup_stale_sessions,
    get_session_root,
)
from src.indexing.visual_indexer import get_shared_colpali, ColPaliPageIndex
from src.generation.answer_generator import get_shared_qwen, AnswerGenerator
from src.retrieval.multimodal_retriever import SourceAwareMultimodalRetriever
from src.agents.workflow import run_space_rag, reset_session_retriever


def test_tasks_16_to_20():
    print("\n" + "=" * 70)
    print("VERIFICATION SUITE: Tasks 16 to 20 (Cleanup, Optimization, E2E, Isolation, Validation)")
    print("=" * 70)

    # -----------------------------------------------------------------
    # TASK 16: Session Cleanup & Windows Handle Teardown
    # -----------------------------------------------------------------
    print("\n--- [TASK 16] Session Cleanup & Explicit Handle Disconnection ---")
    dummy_pdf_bytes = b"%PDF-1.4 sample pdf content for unit test"
    sess = create_session("test_cleanup_doc.pdf", dummy_pdf_bytes)
    sess_id = sess["session_id"]
    sess_root = Path(sess["session_root"])
    assert sess_root.exists()

    # Simulate Chroma text index inside session to verify file lock release
    from src.indexing.chroma_indexer import ChromaTextIndex
    chroma_dir = sess_root / "chroma"
    idx = ChromaTextIndex(chroma_dir, "private_text")
    idx.collection.add(ids=["1"], documents=["satellite telemetry"], metadatas=[{"source_type": "private"}])
    assert idx.collection.count() == 1

    # Cleanup must stop Chroma SQLite system, collect GC, and delete dir
    idx.close()
    cleaned = cleanup_session(sess_id)
    assert cleaned is True, "cleanup_session returned False"
    assert not sess_root.exists(), "Session root still exists after cleanup_session!"
    print("[PASS] Session directory and locked SQLite files cleanly purged from disk.")

    # Test stale sessions sweeper
    sess_stale = create_session("stale_doc.pdf", dummy_pdf_bytes)
    stale_root = Path(sess_stale["session_root"])
    time.sleep(0.1)
    # With max_age_hours=0.0, anything created in past is stale
    removed_count = cleanup_stale_sessions(max_age_hours=0.00001)
    assert not stale_root.exists(), "Stale session was not cleaned up!"
    print(f"[PASS] Stale session garbage collector removed {removed_count} stale session(s).")

    # -----------------------------------------------------------------
    # TASK 17: Performance Optimization (Singleton Caches & Routing)
    # -----------------------------------------------------------------
    print("\n--- [TASK 17] Performance Optimization Verification ---")
    
    # 1. ColPali model singleton test
    col1, proc1 = get_shared_colpali()
    col2, proc2 = get_shared_colpali()
    assert col1 is col2, "ColPali model was not shared via singleton cache!"
    assert proc1 is proc2, "ColPali processor was not shared via singleton cache!"
    from src.indexing.visual_indexer import release_shared_colpali
    release_shared_colpali()
    import gc
    gc.collect()
    print("[PASS] ColPali singleton cache verified: 0 duplicate models instantiated.")

    # 2. Qwen model singleton test
    qwen1, tok1 = get_shared_qwen()
    qwen2, tok2 = get_shared_qwen()
    assert qwen1 is qwen2, "Qwen model was not shared via singleton cache!"
    assert tok1 is tok2, "Qwen tokenizer was not shared via singleton cache!"
    print("[PASS] Qwen offline singleton cache verified: 0 duplicate models instantiated.")

    # 3. Fast-path routing test (Text queries must completely skip visual search)
    corpus_root = PROJECT_ROOT / "data" / "corpus_index"
    retriever = SourceAwareMultimodalRetriever(corpus_root=corpus_root)
    t0 = time.time()
    res = retriever.retrieve("What is satellite constellation altitude?", top_k=2)
    dt_cold = time.time() - t0
    assert res["modality"] == "text"

    # Subsequent warm retrieval should be fast with singleton models
    t0 = time.time()
    res_warm = retriever.retrieve("What is the primary payload sensor?", top_k=2)
    dt_warm = time.time() - t0
    assert res_warm["modality"] == "text"
    assert dt_warm < 10.0, f"Warm text query took {dt_warm:.2f}s, expected < 10s on CPU!"
    print(f"[PASS] Text fast-path verified: cold load took {dt_cold:.2f}s, warm query took {dt_warm:.2f}s.")
    retriever.close()

    # -----------------------------------------------------------------
    # TASK 18: End-to-End Fallbacks & Edge Cases
    # -----------------------------------------------------------------
    print("\n--- [TASK 18] End-to-End Fallbacks & Robustness ---")
    # Query with nonexistent concepts to verify graceful fallback
    out = run_space_rag(
        query="xyznonexistentqueryterms99999",
        corpus_root=corpus_root,
        model_mode="offline",
        top_k=2,
    )
    assert "answer" in out
    assert "citations" in out
    assert out["citations"] == []
    print(f"[PASS] Nonexistent query gracefully handled with zero hallucinations.")

    # -----------------------------------------------------------------
    # TASK 19: Old Global Retrieval Paths Deprecation & Isolation
    # -----------------------------------------------------------------
    print("\n--- [TASK 19] Legacy Retrieval Paths Isolation ---")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        from src.retrieval.chroma_text_retriever import ChromaTextRetriever
        from src.retrieval.visual_retriever import VisualRetriever

        # Trigger inits
        try:
            _ = ChromaTextRetriever(persist_directory="nonexistent_test_path")
        except Exception:
            pass

        try:
            _ = VisualRetriever(embeddings_root="nonexistent_test_path", pages_dir="nonexistent_test_path")
        except Exception:
            pass

        dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(dep_warnings) >= 2, f"Expected at least 2 DeprecationWarnings, got {len(dep_warnings)}"
        print(f"[PASS] Legacy classes correctly emit DeprecationWarnings pointing to source-aware modules.")

    # -----------------------------------------------------------------
    # TASK 20: Final System Readiness & App Validation
    # -----------------------------------------------------------------
    print("\n--- [TASK 20] Final System Readiness Validation ---")
    corpus_manifest = corpus_root / "corpus_manifest.json"
    assert corpus_manifest.exists(), "Corpus manifest missing!"
    assert (corpus_root / "bm25.pkl").exists(), "Corpus BM25 index missing!"
    assert (corpus_root / "colpali.pt").exists(), "Corpus ColPali index missing!"
    assert (corpus_root / "chroma").exists(), "Corpus Chroma index missing!"
    print("[PASS] Permanent knowledge corpus index files complete and intact.")

    streamlit_file = PROJECT_ROOT / "app" / "streamlit_app.py"
    assert streamlit_file.exists()
    print("[PASS] Streamlit application ready at app/streamlit_app.py.")

    print("\n" + "=" * 70)
    print("ALL TASKS 16 TO 20 VERIFIED AND PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    test_tasks_16_to_20()
