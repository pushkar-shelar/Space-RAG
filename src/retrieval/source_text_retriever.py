from __future__ import annotations

from pathlib import Path
from typing import List

from src.indexing.bm25_index import BM25SourceIndex
from src.indexing.chroma_indexer import ChromaTextIndex


class SourceTextRetriever:
    """Search one source only: corpus OR one private session."""

    def __init__(self, root: str | Path, source_type: str):
        self.root = Path(root)
        self.source_type = source_type
        bm25_path = (
            self.root / "indexes" / "bm25.pkl"
            if (self.root / "indexes" / "bm25.pkl").exists()
            else self.root / "bm25.pkl"
        )
        self.bm25 = BM25SourceIndex(bm25_path)
        collection_name = "corpus_text" if source_type == "corpus" else "private_text"
        self.chroma = ChromaTextIndex(self.root / "chroma", collection_name)

    def search(self, query: str, top_k: int = 10) -> List[dict]:
        bm25_results = self.bm25.search(query, top_k)
        dense_results = self.chroma.search(query, top_k)

        merged = {}
        for result in bm25_results:
            meta = result["metadata"]
            key = (meta.get("document"), meta.get("page_number"), meta.get("chunk_index"))
            merged[key] = {
                "text": result["text"],
                "metadata": meta,
                "bm25_score": result["bm25_score"],
                "dense_score": 0.0,
            }
        for result in dense_results:
            meta = result["metadata"]
            key = (meta.get("document"), meta.get("page_number"), meta.get("chunk_index"))
            if key not in merged:
                merged[key] = {
                    "text": result["text"],
                    "metadata": meta,
                    "bm25_score": 0.0,
                    "dense_score": result["dense_score"],
                }
            else:
                merged[key]["dense_score"] = result["dense_score"]

        items = list(merged.values())
        if not items:
            return []

        bm_values = [max(0.0, x["bm25_score"]) for x in items]
        dense_values = [x["dense_score"] for x in items]

        bm_max = max(bm_values) if bm_values else 0.0
        dense_min = min(dense_values) if dense_values else 0.0
        dense_max = max(dense_values) if dense_values else 1.0

        for item in items:
            bm_norm = item["bm25_score"] / bm_max if bm_max > 0 else 0.0
            dense_norm = (
                (item["dense_score"] - dense_min) / (dense_max - dense_min)
                if dense_max > dense_min
                else 0.0
            )
            item["text_score"] = 0.45 * bm_norm + 0.55 * dense_norm
            item["source_type"] = self.source_type
            item["source_id"] = item["metadata"].get("source_id")

        items.sort(key=lambda x: x["text_score"], reverse=True)
        return items[:top_k]

    def close(self):
        """Release underlying Chroma resources."""
        if hasattr(self, "chroma") and self.chroma is not None:
            self.chroma.close()
