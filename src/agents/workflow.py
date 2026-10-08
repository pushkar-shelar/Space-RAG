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


def condense_query_with_history(query: str, chat_history: list | None) -> str:
    """Resolve pronouns and context from conversation history without slow LLM calls."""
    if not chat_history:
        return query

    import re

    # If query is self-contained and detailed with specific identifier/acronym, do not alter
    has_specific_id = bool(re.search(r"\b[A-Z0-9#-]{3,}\b", query))
    if len(query.strip().split()) >= 7 and has_specific_id:
        return query

    # Words indicating anaphora/follow-up
    pronoun_pattern = re.compile(
        r"\b(it|its|they|them|their|this|that|these|those)\b",
        re.IGNORECASE,
    )
    is_followup = bool(pronoun_pattern.search(query)) or len(query.strip().split()) < 5

    if not is_followup:
        return query

    # Extract key technical terms and capitalized entities from previous messages
    entities = []
    for msg in reversed(chat_history[-4:]):
        content = msg.get("content", "")
        # Find acronyms / model names like TROPICS, NASA, GSFC, SAR, Qwen, etc.
        matches = re.findall(r"\b[A-Z0-9-]{3,}\b", content)
        for m in matches:
            if m.lower() not in ("what", "when", "where", "which", "this", "that", "from", "with", "have"):
                if m not in entities:
                    entities.append(m)
        # Check for page numbers
        p_matches = re.findall(r"\bpage\s+(\d+)\b", content, re.IGNORECASE)
        if p_matches and "page" not in query.lower():
            entities.append(f"page {p_matches[-1]}")

    if entities:
        # Prepend up to 2 unique key entities to search query
        key_context = " ".join(entities[:2])
        if key_context.lower() not in query.lower():
            return f"{key_context} {query}"

    return query


def understand_query(state: SpaceRAGState):
    raw_query = state["query"]
    history = state.get("chat_history")
    standalone = condense_query_with_history(raw_query, history)
    info = QueryModalityDetector().detect(standalone)
    return {
        "standalone_query": standalone,
        "query_modality": info.modality,
        "page_reference": info.page_reference,
    }


def retrieve_evidence(state: SpaceRAGState):
    retriever = get_retriever(
        session_id=state.get("session_id"),
        corpus_root=state["corpus_root"],
        session_root=state.get("session_root"),
    )
    search_query = state.get("standalone_query") or state["query"]
    result = retriever.retrieve(
        search_query,
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
        chat_history=state.get("chat_history"),
        api_key=state.get("api_key"),
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
    chat_history: list | None = None,
    api_key: str | None = None,
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
        "chat_history": chat_history or [],
        "api_key": api_key,
    }
    return _GRAPH.invoke(state)


def stream_space_rag_answer(
    query: str,
    session_id: str | None = None,
    session_root: str | Path | None = None,
    document_name: str | None = None,
    model_mode: str = "offline",
    corpus_root: str | Path = DEFAULT_CORPUS_ROOT,
    top_k: int = 5,
    retrieval_k: int = 10,
    chat_history: list | None = None,
    api_key: str | None = None,
):
    """Executes retrieval and verification, then yields streaming tokens."""
    retriever = get_retriever(session_id, corpus_root, session_root)
    search_query = condense_query_with_history(query, chat_history)
    retrieval_info = retriever.retrieve(search_query, top_k=top_k, retrieval_k=retrieval_k)
    results = retrieval_info["results"]

    gen = get_answer_generator()
    stream = gen.generate_stream(
        query=query,
        retrieval_results=results,
        model_mode=model_mode,
        chat_history=chat_history,
        api_key=api_key,
    )

    citations = []
    seen = set()
    for r in results:
        cit = gen._private_citation(r)
        if cit and cit not in seen:
            seen.add(cit)
            citations.append(cit)

    return {
        "stream": stream,
        "citations": citations,
        "source_mix": retrieval_info["source_mix"],
        "retrieval_results": results,
    }
