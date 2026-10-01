from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import List

from rank_bm25 import BM25Okapi

from src.indexing.text_chunker import chunk_document


TOKEN_RE = re.compile(r"\b\w+\b")


def _tokenize(text: str):
    return TOKEN_RE.findall(text.lower())


class BM25SourceIndex:
    """BM25 index stored inside one source's directory."""

    def __init__(self, index_path: str | Path):
        self.index_path = Path(index_path)
        self.bm25 = None
        self.chunks: List[dict] = []
        if self.index_path.exists():
            self._load()

    def build(
        self,
        text_dirs,
        source_type: str,
        source_id: str,
        chunk_size: int = 1000,
        overlap: int = 200,
    ):
        self.chunks = []
        for text_dir in text_dirs:
            self.chunks.extend(
                chunk_document(
                    text_dir,
                    source_type=source_type,
                    source_id=source_id,
                    chunk_size=chunk_size,
                    overlap=overlap,
                )
            )

        tokenized = [_tokenize(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized) if tokenized else None
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with self.index_path.open("wb") as file:
            pickle.dump({"chunks": self.chunks, "bm25": self.bm25}, file)
        return len(self.chunks)

    def _load(self):
        with self.index_path.open("rb") as file:
            data = pickle.load(file)
        self.chunks = data["chunks"]
        self.bm25 = data["bm25"]

    def search(self, query: str, top_k: int = 10) -> List[dict]:
        if self.bm25 is None or not self.chunks:
            return []
        scores = self.bm25.get_scores(_tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            {
                "text": self.chunks[i]["text"],
                "metadata": self.chunks[i],
                "bm25_score": float(scores[i]),
            }
            for i in order
        ]
