import time
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.session_manager import create_session, cleanup_session
from src.ingestion.pipeline import process_document
from src.indexing.index_document import index_private_session
from src.agents.workflow import run_space_rag, reset_session_retriever


def test_end_to_end_pipeline():
    print("\n" + "=" * 70)
    print("DEEP INTEGRATION TEST: End-to-End Space RAG Flow")
    print("=" * 70)

    # 1. Setup session with sample PDF
    sample_pdf = PROJECT_ROOT / "data" / "corpus" / "GSFC-HDBK-8007_Admn.pdf"
    assert sample_pdf.exists()

    print("[1/5] Ingesting document into ephemeral session...")
    t0 = time.time()
    session = create_session("GSFC-HDBK-8007.pdf", sample_pdf.read_bytes())
    session_id = session["session_id"]
    session_root = session["session_root"]

    manifest = process_document(session["pdf_path"], session_root, session_id=session_id)
    print(f"       Ingestion time: {time.time() - t0:.2f}s (Pages: {manifest['total_pages']})")

    # 2. Fast indexing
    print("[2/5] Building text indexes (BM25 + Chroma)...")
    t0 = time.time()
    stats = index_private_session(session_root)
    print(f"       Indexing time: {time.time() - t0:.2f}s (BM25: {stats['bm25_chunks']}, Dense: {stats['dense_chunks']})")

    # 3. Retrieval
    query = "What is the purpose of this handbook?"
    print(f"[3/5] Executing query: '{query}'...")
    t0 = time.time()

    # We use a dummy empty corpus_root since corpus isn't indexed yet
    corpus_root = PROJECT_ROOT / "data" / "corpus_index"
    
    result = run_space_rag(
        query=query,
        session_id=session_id,
        session_root=session_root,
        document_name="GSFC-HDBK-8007.pdf",
        model_mode="offline",
        corpus_root=corpus_root,
        top_k=3,
        retrieval_k=5,
    )
    rag_time = time.time() - t0
    print(f"       RAG Execution time (Retrieval + Inference): {rag_time:.2f}s")

    # 4. Validate output
    print("[4/5] Validating generated response & citations...")
    answer = result.get("answer", "")
    citations = result.get("citations", [])
    source_mix = result.get("source_mix", {})
    
    print("-" * 50)
    print(f"ANSWER:\n{answer}\n")
    print(f"CITATIONS:\n{citations}")
    print(f"SOURCE MIX:\n{source_mix}")
    print("-" * 50)

    assert len(answer) > 20, "Answer was empty or too short!"
    assert len(citations) > 0, "No citations generated for private document evidence!"
    assert any("GSFC-HDBK-8007" in c for c in citations), "Citation did not name private doc!"

    # 5. Clean up
    print("[5/5] Cleaning up session...")
    reset_session_retriever(session_id)
    cleanup_session(session_id)
    print("Session directory removed. End-to-end test passed successfully.")


if __name__ == "__main__":
    test_end_to_end_pipeline()
