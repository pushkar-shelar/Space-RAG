from __future__ import annotations

import gc
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.workflow import reset_session_retriever, run_space_rag
from src.indexing.corpus_builder import build_corpus_index
from src.indexing.index_document import index_private_session
from src.ingestion.pipeline import create_and_process_session
from src.ingestion.session_manager import cleanup_session, cleanup_stale_sessions

CORPUS_DIR = PROJECT_ROOT / "data" / "corpus"
CORPUS_INDEX = PROJECT_ROOT / "data" / "corpus_index"

st.set_page_config(
    page_title="Space RAG",
    page_icon="📄",
    layout="wide",
)

cleanup_stale_sessions(max_age_hours=12)

if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "session_root" not in st.session_state:
    st.session_state.session_root = None
if "document_name" not in st.session_state:
    st.session_state.document_name = None


def end_private_session():
    session_id = st.session_state.get("session_id")
    if session_id:
        reset_session_retriever(session_id)
        gc.collect()
        try:
            cleanup_session(session_id)
        except Exception as exc:
            st.error(f"Could not delete private session: {exc}")
            return
    st.session_state.session_id = None
    st.session_state.session_root = None
    st.session_state.document_name = None
    st.session_state.pop("uploaded_file_name", None)
    st.rerun()


st.title("Space RAG")
st.caption("Permanent space-research corpus + temporary private document retrieval")

with st.sidebar:
    st.header("Knowledge Sources")
    corpus_ready = (CORPUS_INDEX / "corpus_manifest.json").exists()
    if corpus_ready:
        st.success("Permanent corpus index: ready")
    else:
        st.warning("Permanent corpus index: not built")

    if st.button("Build / Refresh Corpus Index", use_container_width=True):
        try:
            with st.spinner("Building permanent corpus index..."):
                stats = build_corpus_index(CORPUS_DIR, CORPUS_INDEX)
            st.success(
                f"Corpus indexed: {stats['documents']} documents, "
                f"{stats['bm25_chunks']} text chunks, {stats['visual_pages']} visual pages."
            )
        except Exception as exc:
            st.error(f"Corpus build failed: {exc}")

    st.divider()
    st.header("Answer Model")
    model_mode = st.radio(
        "Mode",
        ["offline", "online"],
        format_func=lambda x: "Offline — Local Qwen" if x == "offline" else "Online — Grok API",
    )
    if model_mode == "online":
        st.info("Online mode sends retrieved evidence only, not the uploaded PDF.")

    st.divider()
    st.header("Private Session")
    uploaded_file = st.file_uploader(
        "Upload a space-research PDF",
        type=["pdf"],
    )

    if uploaded_file is not None:
        old_name = st.session_state.get("uploaded_file_name")
        if old_name != uploaded_file.name:
            if st.session_state.session_id:
                reset_session_retriever(st.session_state.session_id)
                gc.collect()
                cleanup_session(st.session_state.session_id)
                st.session_state.session_id = None
                st.session_state.session_root = None
                st.session_state.document_name = None
            try:
                with st.spinner("Creating private session and ingesting PDF..."):
                    result = create_and_process_session(
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                    )
                    session = result["session"]
                    manifest = result["manifest"]
                    index_private_session(session["session_root"])

                st.session_state.session_id = session["session_id"]
                st.session_state.session_root = session["session_root"]
                st.session_state.document_name = session["document_name"]
                st.session_state.uploaded_file_name = uploaded_file.name
                st.success(
                    f"Private document ready: {manifest['total_pages']} pages"
                )
            except Exception as exc:
                st.error(f"Private document processing failed: {exc}")

    if st.session_state.session_id:
        st.success(f"Private document: {st.session_state.document_name}")
        if st.button("END CHAT & DELETE PRIVATE DATA", use_container_width=True):
            end_private_session()
    else:
        st.info("No private PDF loaded. Corpus-only mode is available.")

st.divider()

query = st.text_area(
    "Ask a question",
    placeholder=(
        "Example: What are the mission objectives?\n"
        "Example: What does the graph on page 7 show?"
    ),
    height=100,
)

col1, col2 = st.columns([1, 5])
with col1:
    ask = st.button("Ask", type="primary", use_container_width=True)

if ask and query.strip():
    if not (CORPUS_INDEX / "corpus_manifest.json").exists() and not st.session_state.session_id:
        st.warning("Build the corpus index or upload a private PDF first.")
    else:
        try:
            with st.spinner("Retrieving evidence and generating answer..."):
                result = run_space_rag(
                    query=query.strip(),
                    session_id=st.session_state.session_id,
                    session_root=st.session_state.session_root,
                    document_name=st.session_state.document_name,
                    model_mode=model_mode,
                    corpus_root=CORPUS_INDEX,
                    top_k=5,
                    retrieval_k=10,
                )

            st.subheader("Answer")
            st.write(result.get("answer", "No answer generated."))

            source_mix = result.get("source_mix", {})
            st.caption(
                f"Evidence used: {source_mix.get('private', 0)} private sources, "
                f"{source_mix.get('corpus', 0)} corpus sources"
            )

            citations = result.get("citations", [])
            if citations:
                st.subheader("Private-document sources")
                for citation in citations:
                    st.write(f"- {citation}")

            # Referenced visual pages preview
            retrieval_results = result.get("retrieval_results", [])
            visual_items = [
                r for r in retrieval_results
                if r.get("page_image") and r.get("source_type") == "private" and Path(r.get("page_image", "")).exists()
            ]
            if visual_items:
                with st.expander("Referenced Document Visual Pages", expanded=False):
                    for item in visual_items[:3]:
                        st.image(
                            item["page_image"],
                            caption=f"{item.get('document', 'Document')} — Page {item.get('page_number', '?')}",
                            use_container_width=True,
                        )

        except Exception as exc:
            st.error(f"Space RAG failed: {exc}")

elif ask:
    st.warning("Enter a question first.")
