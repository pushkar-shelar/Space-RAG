from __future__ import annotations

from pathlib import Path
from typing import Optional

from langgraph.graph import END, START, StateGraph

from src.agents.state import SpaceRAGState
from src.generation.answer_generator import AnswerGenerator
from src.retrieval.multimodal_retriever import SourceAwareMultimodalRetriever
from src.retrieval.query_modality import QueryModalityDetector

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS_ROOT = PROJECT_ROOT / "data" / "corpus_index"

_retrievers: dict[str, SourceAwareMultimodalRetriever] = {}
_answer_generator: Optional[AnswerGenerator] = None


def reset_session_retriever(session_id: str | None = None):
    if session_id is None:
        for r in _retrievers.values():
            if hasattr(r, "close"):
                r.close()
        _retrievers.clear()
    else:
        for k in list(_retrievers.keys()):
            if k.startswith(f"{session_id}::"):
                r = _retrievers.pop(k, None)
                if r and hasattr(r, "close"):
                    r.close()


def get_retriever(
    session_id: str | None,
    corpus_root: str | Path,
    session_root: str | Path | None,
):
    key = f"{session_id or 'corpus-only'}::{Path(corpus_root).resolve()}::{Path(session_root).resolve() if session_root else ''}"
    if key not in _retrievers:
        _retrievers[key] = SourceAwareMultimodalRetriever(
            corpus_root=corpus_root,
            session_root=session_root,
        )
    return _retrievers[key]


def get_answer_generator():
    global _answer_generator
    if _answer_generator is None:
        _answer_generator = AnswerGenerator()
    return _answer_generator


def understand_query(state: SpaceRAGState):
    info = QueryModalityDetector().detect(state["query"])
    return {
        "query_modality": info.modality,
        "page_reference": info.page_reference,
    }


def retrieve_evidence(state: SpaceRAGState):
    retriever = get_retriever(
        session_id=state.get("session_id"),
        corpus_root=state["corpus_root"],
        session_root=state.get("session_root"),
    )
    result = retriever.retrieve(
        state["query"],
        top_k=state.get("top_k", 5),
        retrieval_k=state.get("retrieval_k", 10),
    )
    return {
        "retrieval_results": result["results"],
        "query_modality": result["modality"],
        "page_reference": result["page_reference"],
        "source_mix": result["source_mix"],
    }


def assess_evidence(state: SpaceRAGState):
    count = len(state.get("retrieval_results", []))
    return {"sufficiency": "sufficient" if count > 0 else "insufficient"}


def verify_evidence(state: SpaceRAGState):
    """Aggregate cross-modal verification checks for multimodal evidence."""
    results = state.get("retrieval_results", [])
    summary = []
    for item in results:
        v = item.get("verification")
        if v and isinstance(v, dict):
            summary.append({
                "document": item.get("document"),
                "page_number": item.get("page_number"),
                "source_type": item.get("source_type"),
                "status": v.get("status", "UNVERIFIED"),
                "confidence": v.get("confidence", 0.0),
                "claim_type": v.get("claim_type", "general"),
            })
    return {"verification_summary": summary}


def generate_answer(state: SpaceRAGState):
    result = get_answer_generator().generate(
        query=state["query"],
        retrieval_results=state.get("retrieval_results", []),
        model_mode=state.get("model_mode", "offline"),
    )
    return {
        "answer": result["answer"],
        "citations": result["citations"],
        "model": result["model"],
        "source_mix": result["source_mix"],
    }


def build_graph():
    graph = StateGraph(SpaceRAGState)
    graph.add_node("understand_query", understand_query)
    graph.add_node("retrieve_evidence", retrieve_evidence)
    graph.add_node("assess_evidence", assess_evidence)
    graph.add_node("verify_evidence", verify_evidence)
    graph.add_node("generate_answer", generate_answer)

    graph.add_edge(START, "understand_query")
    graph.add_edge("understand_query", "retrieve_evidence")
    graph.add_edge("retrieve_evidence", "assess_evidence")
    graph.add_edge("assess_evidence", "verify_evidence")
    graph.add_edge("verify_evidence", "generate_answer")
    graph.add_edge("generate_answer", END)
    return graph.compile()


_GRAPH = build_graph()


def run_space_rag(
    query: str,
    session_id: str | None = None,
    session_root: str | Path | None = None,
    document_name: str | None = None,
    model_mode: str = "offline",
    corpus_root: str | Path = DEFAULT_CORPUS_ROOT,
    top_k: int = 5,
    retrieval_k: int = 10,
):
    state: SpaceRAGState = {
        "session_id": session_id,
        "session_root": str(session_root) if session_root else None,
        "document_name": document_name,
        "corpus_root": str(corpus_root),
        "query": query,
        "top_k": top_k,
        "retrieval_k": retrieval_k,
        "model_mode": model_mode,
    }
    return _GRAPH.invoke(state)
