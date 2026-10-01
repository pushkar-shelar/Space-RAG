from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, List, Sequence

import chromadb
from sentence_transformers import SentenceTransformer

from src.indexing.text_chunker import chunk_document


class ChromaTextIndex:
    """Persistent Chroma index scoped to exactly one source."""

    def __init__(
        self,
        persist_directory: str | Path,
        collection_name: str,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.embedding_model_name = embedding_model
        self._model = None

    def _get_model(self):
        if self._model is None:
            self._model = SentenceTransformer(self.embedding_model_name)
        return self._model

    def build(
        self,
        text_dirs: Sequence[str | Path],
        source_type: str,
        source_id: str,
        chunk_size: int = 1000,
        overlap: int = 200,
    ) -> int:
        all_chunks: List[dict] = []
        for text_dir in text_dirs:
            all_chunks.extend(
                chunk_document(
                    text_dir,
                    source_type=source_type,
                    source_id=source_id,
                    chunk_size=chunk_size,
                    overlap=overlap,
                )
            )

        if not all_chunks:
            return 0

        documents = [c["text"] for c in all_chunks]
        embeddings = self._get_model().encode(
            documents,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        ids = []
        metadatas = []
        clean_embeddings = []
        clean_documents = []

        for chunk, embedding in zip(all_chunks, embeddings):
            chunk_id = hashlib.sha1(chunk["id"].encode()).hexdigest()
            ids.append(chunk_id)
            metadatas.append(
                {
                    "source_type": chunk["source_type"],
                    "source_id": chunk["source_id"],
                    "document": chunk["document"],
                    "page_number": int(chunk["page_number"]),
                    "chunk_index": int(chunk["chunk_index"]),
                    "text_path": chunk["text_path"],
                }
            )
            clean_documents.append(chunk["text"])
            clean_embeddings.append(embedding)

        # Session indexes are rebuilt from scratch, so deleting stale entries here
        # prevents an old uploaded PDF from surviving inside the same session dir.
        existing = self.collection.get(include=["metadatas"])
        if existing.get("ids"):
            self.collection.delete(ids=existing["ids"])

        batch_size = 64
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            self.collection.add(
                ids=ids[start:end],
                documents=clean_documents[start:end],
                metadatas=metadatas[start:end],
                embeddings=clean_embeddings[start:end],
            )

        return len(ids)

    def search(self, query: str, top_k: int = 10) -> List[dict]:
        if self.collection.count() == 0:
            return []

        query_embedding = self._get_model().encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0].tolist()

        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        output = []
        for document, metadata, distance in zip(documents, metadatas, distances):
            output.append(
                {
                    "text": document,
                    "metadata": metadata,
                    "dense_score": 1.0 - float(distance),
                }
            )
        return output

    def close(self):
        """Release underlying database handles to prevent file lock issues on Windows."""
        if hasattr(self, "client") and self.client is not None:
            try:
                if hasattr(self.client, "_system"):
                    self.client._system.stop()
                if hasattr(self.client, "close"):
                    self.client.close()
            except Exception:
                pass
