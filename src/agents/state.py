from __future__ import annotations

from typing import Any, Dict, List, TypedDict


class SpaceRAGState(TypedDict, total=False):
    session_id: str | None
    session_root: str | None
    document_name: str | None
    corpus_root: str
    query: str
    top_k: int
    retrieval_k: int
    query_modality: str
    page_reference: int | None
    retrieval_results: List[Dict[str, Any]]
    sufficiency: str
    verification_summary: List[Dict[str, Any]]
    model_mode: str
    model: str
    answer: str
    citations: List[str]
    source_mix: Dict[str, int]
    error: str | None
