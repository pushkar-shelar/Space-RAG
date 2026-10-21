from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from src.retrieval.query_modality import QueryModalityDetector
from src.retrieval.source_text_retriever import SourceTextRetriever
from src.retrieval.source_visual_retriever import SourceVisualRetriever

_SHARED_RERANKER = None
_RERANK_PAIR_CACHE: dict[tuple[str, str], float] = {}


def get_shared_reranker():
    global _SHARED_RERANKER
    if _SHARED_RERANKER is None:
        from sentence_transformers import CrossEncoder
        _SHARED_RERANKER = CrossEncoder("BAAI/bge-reranker-base")
    return _SHARED_RERANKER


class SourceAwareMultimodalRetriever:
    """
    Search the permanent corpus and the current private document separately.

    The source label is retained internally for correct citation behavior, but
    corpus provenance is intentionally hidden from the user-facing answer.
    """

    def __init__(
        self,
        corpus_root: str | Path,
        session_root: Optional[str | Path] = None,
    ):
        self.corpus_root = Path(corpus_root)
        self.session_root = Path(session_root) if session_root else None
        self.detector = QueryModalityDetector()

        self.corpus_text = (
            SourceTextRetriever(self.corpus_root, "corpus")
            if self.corpus_root.exists()
            else None
        )
        self.corpus_visual = (
            SourceVisualRetriever(self.corpus_root, "corpus")
            if self.corpus_root.exists()
            else None
        )

        self.private_text = (
            SourceTextRetriever(self.session_root, "private")
            if self.session_root and self.session_root.exists()
            else None
        )
        self.private_visual = (
            SourceVisualRetriever(self.session_root, "private")
            if self.session_root and self.session_root.exists()
            else None
        )

    def _add_text_results(self, results: List[dict], bucket: Dict):
        for result in results:
            meta = result["metadata"]
            key = (
                result["source_type"],
                meta.get("document"),
                meta.get("page_number"),
                meta.get("chunk_index"),
            )
            bucket[key] = {
                "text": result["text"],
                "source_type": result["source_type"],
                "source_id": result.get("source_id"),
                "document": meta.get("document"),
                "page_number": meta.get("page_number"),
                "text_path": meta.get("text_path"),
                "bm25_score": result.get("bm25_score", 0.0),
                "dense_score": result.get("dense_score", 0.0),
                "text_score": result.get("text_score", 0.0),
                "visual_score": 0.0,
            }

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        retrieval_k: int = 10,
    ) -> dict:
        info = self.detector.detect(query)
        candidates: Dict[tuple, dict] = {}

        # Always search corpus and private text so a private session can use
        # both the uploaded paper and the permanent knowledge corpus.
        if self.corpus_text:
            self._add_text_results(
                self.corpus_text.search(query, retrieval_k),
                candidates,
            )
        if self.private_text:
            self._add_text_results(
                self.private_text.search(query, retrieval_k),
                candidates,
            )

        visual_enabled = info.modality in {"visual", "multimodal"}
        if visual_enabled:
            self._add_visual_results(
                self.corpus_visual,
                query,
                retrieval_k,
                candidates,
            )
            self._add_visual_results(
                self.private_visual,
                query,
                retrieval_k,
                candidates,
            )

        if not candidates:
            return {
                "modality": info.modality,
                "page_reference": info.page_reference,
                "results": [],
                "source_mix": {"corpus": 0, "private": 0},
            }

        # Candidate-level adaptive fusion.
        values = list(candidates.values())
        text_values = [x["text_score"] for x in values]
        visual_values = [x["visual_score"] for x in values]

        tmin, tmax = min(text_values), max(text_values)
        vmin, vmax = min(visual_values), max(visual_values)

        for item in values:
            t = (
                (item["text_score"] - tmin) / (tmax - tmin)
                if tmax > tmin else item["text_score"]
            )
            v = (
                (item["visual_score"] - vmin) / (vmax - vmin)
                if vmax > vmin else 0.0
            )

            if info.modality == "text":
                item["fusion_score"] = 0.80 * t + 0.20 * v
            elif info.modality == "visual":
                item["fusion_score"] = 0.25 * t + 0.75 * v
            else:
                item["fusion_score"] = 0.40 * t + 0.60 * v

            # Small private-session preference when evidence is otherwise equal.
            # This is a tie-break, not the only retrieval signal.
            if item["source_type"] == "private":
                item["fusion_score"] += 0.03

        # Protect explicit page references.
        if info.page_reference is not None:
            for item in values:
                if item.get("page_number") == info.page_reference:
                    item["fusion_score"] += 0.40

        values.sort(key=lambda x: x["fusion_score"], reverse=True)
        values = values[:retrieval_k]

        # Rerank candidates with BGE cross-encoder if text candidates exist
        if values and len(values) > 1:
            self._rerank_candidates(values, query)

        # Cross-modal verification is only attempted for a visual/multimodal query.
        if info.modality in {"visual", "multimodal"}:
            self._verify_selected(values, query, limit=min(2, len(values)))

        # Check if query specifically targets the attached private document
        import re
        doc_intent_re = re.compile(r"\b(pdf|document|paper|attachment|attached|uploaded|mentioned in)\b", re.I)
        query_targets_private = bool(self.private_text) and bool(doc_intent_re.search(query))

        if query_targets_private:
            private_items = [x for x in values if x.get("source_type") == "private"]
            corpus_items = [x for x in values if x.get("source_type") == "corpus"]
            values = private_items + corpus_items

        final = values[:top_k]
        for rank, item in enumerate(final, start=1):
            item["rank"] = rank
            item["provenance"] = self._public_provenance(item)

        source_mix = {
            "corpus": sum(x["source_type"] == "corpus" for x in final),
            "private": sum(x["source_type"] == "private" for x in final),
        }

        return {
            "modality": info.modality,
            "page_reference": info.page_reference,
            "results": final,
            "source_mix": source_mix,
        }

    def _rerank_candidates(self, candidates: List[dict], query: str, max_candidates: int = 10):
        """Cross-encoder reranking over fused candidates with caching and batch optimization."""
        try:
            import re
            import torch
            doc_intent_re = re.compile(r"\b(pdf|document|paper|attachment|attached|uploaded|mentioned in)\b", re.I)
            query_targets_private = bool(self.private_text) and bool(doc_intent_re.search(query))

            reranker = get_shared_reranker()

            text_items = [c for c in candidates if (c.get("text") or "").strip()][:max_candidates]
            if not text_items:
                return

            unseen_pairs = []
            unseen_indices = []

            for idx, c in enumerate(text_items):
                pair_key = (query, c["text"][:1000])
                if pair_key in _RERANK_PAIR_CACHE:
                    c["bge_score"] = _RERANK_PAIR_CACHE[pair_key]
                else:
                    unseen_pairs.append(pair_key)
                    unseen_indices.append(idx)

            if unseen_pairs:
                with torch.inference_mode():
                    new_scores = reranker.predict(unseen_pairs, batch_size=8, show_progress_bar=False)
                for pair_key, idx, score in zip(unseen_pairs, unseen_indices, new_scores):
                    score_val = float(score)
                    if len(_RERANK_PAIR_CACHE) > 1024:
                        _RERANK_PAIR_CACHE.pop(next(iter(_RERANK_PAIR_CACHE)))
                    _RERANK_PAIR_CACHE[pair_key] = score_val
                    text_items[idx]["bge_score"] = score_val

            for item in text_items:
                bge_score = item.get("bge_score", 0.0)
                norm_bge = max(0.0, min(1.0, float(bge_score)))
                is_private = item.get("source_type") == "private"
                private_bonus = 0.25 if (query_targets_private and is_private) else (0.05 if is_private else 0.0)
                item["fusion_score"] = 0.60 * item.get("fusion_score", 0.0) + 0.40 * norm_bge + private_bonus

            candidates.sort(key=lambda x: (x["fusion_score"], x.get("source_type") == "private"), reverse=True)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Reranking error: {e}")

    @staticmethod
    def _add_visual_results(retriever, query, top_k, bucket):
        if retriever is None or not retriever.available():
            return
        for result in retriever.search(query, top_k):
            meta = result.get("metadata", {})
            page = meta.get("page_number")
            key = (
                result["source_type"],
                meta.get("document"),
                page,
                "visual",
            )
            existing = bucket.get(key)
            if existing is None:
                bucket[key] = {
                    "text": "",
                    "source_type": result["source_type"],
                    "source_id": result.get("source_id"),
                    "document": meta.get("document"),
                    "page_number": page,
                    "text_path": None,
                    "bm25_score": 0.0,
                    "dense_score": 0.0,
                    "text_score": 0.0,
                    "visual_score": result.get("visual_score", 0.0),
                    "page_image": result.get("path"),
                }
            else:
                existing["visual_score"] = max(
                    existing.get("visual_score", 0.0),
                    result.get("visual_score", 0.0),
                )
                existing["page_image"] = result.get("path")

    def _verify_selected(self, results, query, limit=2):
        try:
            from src.verification.cross_modal_verifier import CrossModalVerifier
        except Exception:
            return

        verifier = None
        for item in results[:limit]:
            image = item.get("page_image")
            if not image:
                continue

            # Resolve associated PDF path for verification
            pdf_path = None
            if item.get("source_type") == "corpus":
                corpus_dir = (
                    self.corpus_root.parent / "corpus"
                    if self.corpus_root.name == "corpus_index"
                    else self.corpus_root
                )
                doc_name = item.get("document") or ""
                candidate_pdf = corpus_dir / f"{doc_name}.pdf"
                if candidate_pdf.exists():
                    pdf_path = candidate_pdf
                else:
                    matches = list(corpus_dir.glob(f"*{doc_name}*.pdf"))
                    if matches:
                        pdf_path = matches[0]
            elif self.session_root:
                doc_dir = Path(self.session_root) / "document"
                pdfs = list(doc_dir.glob("*.pdf"))
                if pdfs:
                    pdf_path = pdfs[0]

            if not pdf_path or not Path(pdf_path).exists():
                continue

            page_num = item.get("page_number") or 1
            claim_text = (
                item.get("text")
                or f"Figure on page {page_num} describes remote sensing and mission architecture."
            )

            try:
                if verifier is None:
                    verifier = CrossModalVerifier()
                verification = verifier.verify(
                    pdf_path=str(pdf_path),
                    page_number=int(page_num),
                    text_claim=claim_text,
                    image_path=str(image),
                    query=query,
                )
                item["verification"] = verification
            except Exception as exc:
                item["verification"] = {
                    "status": "UNAVAILABLE",
                    "confidence": 0.0,
                    "error": str(exc),
                }

    @staticmethod
    def _public_provenance(item: dict):
        if item["source_type"] == "private":
            return {
                "source_type": "private",
                "document": item.get("document"),
                "page_number": item.get("page_number"),
            }
        # Never expose corpus document names, paths, or internal IDs to UI/LLM.
        return {"source_type": "corpus"}

    def close(self):
        """Cleanly close all underlying Chroma database clients."""
        if hasattr(self, "corpus_text") and self.corpus_text is not None:
            self.corpus_text.close()
        if hasattr(self, "private_text") and self.private_text is not None:
            self.private_text.close()
