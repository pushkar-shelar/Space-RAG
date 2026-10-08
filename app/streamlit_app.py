from __future__ import annotations

import gc
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.workflow import (
    reset_session_retriever,
    run_space_rag,
    stream_space_rag_answer,
)
from src.indexing.corpus_builder import build_corpus_index
from src.indexing.index_document import index_private_session
from src.ingestion.pipeline import create_and_process_session
from src.ingestion.session_manager import cleanup_session, cleanup_stale_sessions

CORPUS_DIR = PROJECT_ROOT / "data" / "corpus"
CORPUS_INDEX = PROJECT_ROOT / "data" / "corpus_index"

st.set_page_config(
    page_title="Space RAG — Aerospace AI Assistant",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for a Professional Modern AI Chatbot
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    h1, h2, h3, .brand-title {
        font-family: 'Outfit', sans-serif !important;
        letter-spacing: -0.02em;
    }

    .stApp {
        background: radial-gradient(circle at 20% 15%, rgba(15, 23, 42, 0.98), rgba(3, 7, 18, 1) 95%);
        color: #f8fafc;
    }

    /* Minimalist executive header */
    .chat-header {
        display: flex;
        align-items: center;
        gap: 14px;
        padding: 0.6rem 0.2rem 1.1rem 0.2rem;
        margin-bottom: 0.8rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    .chat-title {
        font-size: 1.65rem;
        font-weight: 700;
        background: linear-gradient(90deg, #60a5fa, #a78bfa, #38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    
    .chat-subtitle {
        color: #94a3b8;
        font-size: 0.88rem;
        margin-top: 0.15rem;
    }

    /* Modern active document badge */
    .doc-badge {
        background: rgba(34, 197, 94, 0.08);
        border: 1px solid rgba(34, 197, 94, 0.3);
        border-radius: 8px;
        padding: 10px 12px;
        margin-top: 8px;
        margin-bottom: 8px;
    }

    /* Citation badge style */
    .citation-tag {
        display: inline-flex;
        align-items: center;
        background: rgba(56, 189, 248, 0.12);
        color: #7dd3fc;
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.82rem;
        font-weight: 500;
        margin: 3px 4px 3px 0;
    }

    /* Suggested query buttons */
    .stButton > button {
        border-radius: 8px;
        font-size: 0.85rem;
        text-align: left;
        transition: all 0.2s ease;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

cleanup_stale_sessions(max_age_hours=12)

# Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "session_root" not in st.session_state:
    st.session_state.session_root = None
if "document_name" not in st.session_state:
    st.session_state.document_name = None
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None


def end_private_session():
    """Cleanly purge ephemeral private sandbox and disconnect databases."""
    session_id = st.session_state.get("session_id")
    if session_id:
        reset_session_retriever(session_id)
        gc.collect()
        try:
            cleanup_session(session_id)
        except Exception as exc:
            st.error(f"Could not release session: {exc}")
            return
    st.session_state.session_id = None
    st.session_state.session_root = None
    st.session_state.document_name = None
    st.session_state.pop("uploaded_file_name", None)
    st.rerun()


# =============================================================================
# SIDEBAR CONTROLS
# =============================================================================
with st.sidebar:
    st.markdown("### 📎 Attach Document")
    uploaded_file = st.file_uploader(
        "Upload a PDF for analysis",
        type=["pdf"],
        help="Upload any space paper, technical manual, or mission spec. Analysis is completely private.",
        label_visibility="collapsed",
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
                with st.spinner("Processing attached document..."):
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
                st.success(f"Attached: {uploaded_file.name} ({manifest['total_pages']} pages)")
            except Exception as exc:
                st.error(f"Attachment failed: {exc}")

    if st.session_state.session_id:
        st.markdown(
            f"""
            <div class="doc-badge">
                <span style="color: #4ade80; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.05em;">● ACTIVE ATTACHMENT</span><br/>
                <span style="color: #f1f5f9; font-size: 0.88rem; font-weight: 500; word-break: break-all;">{st.session_state.document_name}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Detach & Remove Document", use_container_width=True):
            end_private_session()

    st.divider()

    st.markdown("### ⚙️ Intelligence Engine")
    model_mode = st.radio(
        "Model Mode",
        ["offline", "online"],
        format_func=lambda x: "Offline (Local Qwen 2.5)" if x == "offline" else "Online (xAI Grok API)",
        label_visibility="collapsed",
    )

    api_key_input = None
    if model_mode == "online":
        api_key_input = st.text_input(
            "xAI API Key (optional if in env)",
            type="password",
            placeholder="xai-...",
            help="Your API key stays strictly in memory for this session.",
        )
        st.caption("🔒 Document content never leaves your machine. Only relevant excerpts are analyzed.")

    st.divider()

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # System Status & Diagnostic Index Tools
    with st.expander("⚙️ System Status", expanded=False):
        st.caption("Synchronize or rebuild reference knowledge index.")
        if st.button("Re-sync Knowledge Index", use_container_width=True):
            try:
                with st.spinner("Synchronizing reference index..."):
                    stats = build_corpus_index(CORPUS_DIR, CORPUS_INDEX)
                st.success(f"Synchronized: {stats['documents']} collections, {stats['bm25_chunks']} items.")
                st.rerun()
            except Exception as exc:
                st.error(f"Sync failed: {exc}")


# =============================================================================
# MAIN CHAT INTERFACE
# =============================================================================
st.markdown(
    """
    <div class="chat-header">
        <div>
            <h1 class="chat-title">Space RAG Assistant</h1>
            <p class="chat-subtitle">Grounded conversational intelligence for satellite missions, remote sensing, and aerospace systems.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Suggested prompts when conversation is empty
if not st.session_state.messages:
    st.markdown("<p style='color: #64748b; font-size: 0.85rem; font-weight: 500; margin-bottom: 8px;'>SUGGESTED TOPICS</p>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    
    if st.session_state.document_name:
        with c1:
            if st.button(f"📄 Summarize the key payload specifications in {st.session_state.document_name}", use_container_width=True):
                st.session_state.pending_query = f"Summarize the key payload specifications and mission goals in {st.session_state.document_name}"
                st.rerun()
            if st.button("🔬 What methodologies or system architectures are described?", use_container_width=True):
                st.session_state.pending_query = "What methodologies or system architectures are described in the attached document?"
                st.rerun()
        with c2:
            if st.button("📊 Explain the diagrams and charts in this document", use_container_width=True):
                st.session_state.pending_query = "Explain the primary diagrams and charts shown in this document."
                st.rerun()
            if st.button("⚡ What are the operational margins and performance constraints?", use_container_width=True):
                st.session_state.pending_query = "What are the operational margins and performance constraints in this document?"
                st.rerun()
    else:
        with c1:
            if st.button("🛰️ What constellation trade-offs apply to LEO precipitation monitoring?", use_container_width=True):
                st.session_state.pending_query = "What constellation trade-offs apply to LEO precipitation monitoring?"
                st.rerun()
            if st.button("📡 How do multi-frequency radiometers resolve atmospheric moisture?", use_container_width=True):
                st.session_state.pending_query = "How do multi-frequency radiometers resolve atmospheric moisture and temperature?"
                st.rerun()
        with c2:
            if st.button("🚀 Compare CubeSat propulsion architectures for orbital maintenance", use_container_width=True):
                st.session_state.pending_query = "Compare CubeSat propulsion architectures for orbital maintenance"
                st.rerun()
            if st.button("🛡️ What space weather mitigation strategies protect polar orbit payloads?", use_container_width=True):
                st.session_state.pending_query = "What space weather mitigation strategies protect polar orbit payloads?"
                st.rerun()

# Display Conversation History
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user", avatar="🧑‍🚀"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant", avatar="🛰️"):
            st.markdown(msg["content"])

            citations = msg.get("citations", [])
            if citations:
                st.markdown("<p style='font-size:0.85rem; font-weight:600; color:#38bdf8; margin-top:0.6rem; margin-bottom:0.2rem;'>📎 Private-document sources</p>", unsafe_allow_html=True)
                for citation in citations:
                    st.markdown(f"- `{citation}`")

            visual_items = msg.get("visual_items", [])
            if visual_items:
                with st.expander("Referenced Document Visual Pages", expanded=False):
                    for item in visual_items[:3]:
                        st.image(
                            item["page_image"],
                            caption=f"{item.get('document', 'Attached Document')} — Page {item.get('page_number', '?')}",
                            use_container_width=True,
                        )

# Input handling
active_query = st.chat_input("Ask anything about space missions, satellite systems, or your attached document...")
if not active_query and "pending_query" in st.session_state:
    active_query = st.session_state.pop("pending_query")

if active_query:
    # 1. Render and record user message
    st.session_state.messages.append({"role": "user", "content": active_query})
    with st.chat_message("user", avatar="🧑‍🚀"):
        st.markdown(active_query)

    # 2. Assistant stream generation
    with st.chat_message("assistant", avatar="🛰️"):
        with st.spinner("Synthesizing response..."):
            try:
                # Use stream_space_rag_answer with conversational history
                stream_result = stream_space_rag_answer(
                    query=active_query.strip(),
                    session_id=st.session_state.session_id,
                    session_root=st.session_state.session_root,
                    document_name=st.session_state.document_name,
                    model_mode=model_mode,
                    corpus_root=CORPUS_INDEX,
                    top_k=5,
                    retrieval_k=10,
                    chat_history=st.session_state.messages[:-1],  # prior turns
                    api_key=api_key_input,
                )

                # Real-time token streaming
                full_answer = st.write_stream(stream_result["stream"])

                citations = stream_result.get("citations", [])
                if citations:
                    st.markdown("<p style='font-size:0.85rem; font-weight:600; color:#38bdf8; margin-top:0.6rem; margin-bottom:0.2rem;'>📎 Private-document sources</p>", unsafe_allow_html=True)
                    for citation in citations:
                        st.markdown(f"- `{citation}`")

                retrieval_results = stream_result.get("retrieval_results", [])
                visual_items = [
                    r for r in retrieval_results
                    if r.get("page_image") and r.get("source_type") == "private" and Path(r.get("page_image", "")).exists()
                ]
                if visual_items:
                    with st.expander("Referenced Document Visual Pages", expanded=False):
                        for item in visual_items[:3]:
                            st.image(
                                item["page_image"],
                                caption=f"{item.get('document', 'Attached Document')} — Page {item.get('page_number', '?')}",
                                use_container_width=True,
                            )

                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_answer,
                    "citations": citations,
                    "visual_items": visual_items,
                })

            except Exception as exc:
                # Fallback to run_space_rag if streaming hits an edge case
                try:
                    fallback_result = run_space_rag(
                        query=active_query.strip(),
                        session_id=st.session_state.session_id,
                        session_root=st.session_state.session_root,
                        document_name=st.session_state.document_name,
                        model_mode=model_mode,
                        corpus_root=CORPUS_INDEX,
                        top_k=5,
                        retrieval_k=10,
                        chat_history=st.session_state.messages[:-1],
                        api_key=api_key_input,
                    )
                    ans = fallback_result.get("answer", "")
                    st.markdown(ans)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": ans,
                        "citations": fallback_result.get("citations", []),
                        "visual_items": [],
                    })
                except Exception as inner_exc:
                    st.error(f"Space RAG Assistant encountered an error: {inner_exc}")
