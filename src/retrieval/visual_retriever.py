import re
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.indexing.colpali_embedder import ColPaliEmbedder


class VisualRetriever:
    """DEPRECATED: Use src.retrieval.source_visual_retriever.SourceVisualRetriever instead."""

    def __init__(
        self,
        embeddings_root="data/processed/embeddings",
        pages_dir="data/processed/pages"
    ):
        import warnings
        warnings.warn(
            "VisualRetriever is deprecated. Use SourceVisualRetriever for source-aware retrieval.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.embeddings_root = Path(embeddings_root)
        self.pages_dir = Path(pages_dir)
        self.embedder = ColPaliEmbedder()

        print(f"ColPali embeddings root: {self.embeddings_root}")
        print(f"Rendered pages directory: {self.pages_dir}")

    @staticmethod
    def parse_embedding_name(embedding_path):
        stem = Path(embedding_path).stem
        match = re.match(
            r"^(?P<document>.+)_page_(?P<page>\d+)$",
            stem
        )
        if match is None:
            return None
        return {
            "document": match.group("document"),
            "page_number": int(match.group("page"))
        }

    def create_query_embedding(self, query):
        batch = self.embedder.processor.process_queries([query])
        batch = {
            key: value.to(self.embedder.device)
            for key, value in batch.items()
        }
        with torch.no_grad():
            return self.embedder.model(**batch)

    def retrieve(self, query, top_k=5):
        print("\n" + "=" * 60)
        print("SPACE RAG - VISUAL RETRIEVAL")
        print("=" * 60)
        print(f"Query: {query}")

        page_files = sorted(
            self.embeddings_root.glob("**/*.pt")
        )

        if not page_files:
            print("\nNo ColPali page embeddings found.")
            return []

        print(f"\nComparing against {len(page_files)} page embeddings...")
        print("\nGenerating query embedding...")
        query_embedding = self.create_query_embedding(query)

        page_embeddings = []
        valid_page_files = []

        for page_file in page_files:
            try:
                embedding = torch.load(
                    page_file,
                    map_location="cpu",
                    weights_only=True
                )
                page_embeddings.append(embedding.squeeze(0))
                valid_page_files.append(page_file)
            except Exception as error:
                print(f"Warning: Could not load {page_file.name}: {error}")

        if not page_embeddings:
            return []

        scores = self.embedder.processor.score_multi_vector(
            query_embedding.cpu(),
            page_embeddings
        )

        results = []

        for page_file, score in zip(valid_page_files, scores[0]):
            info = self.parse_embedding_name(page_file)
            if info is None:
                continue

            document = info["document"]
            page_number = info["page_number"]
            page_image = self.pages_dir / f"{document}_page_{page_number:03d}.png"

            results.append({
                "document": document,
                "page_number": page_number,
                "page_image": str(page_image),
                "image_path": str(page_image),
                "embedding_path": str(page_file),
                "score": float(score.item()),
                "modality": "page_image"
            })

        # Prevent duplicate document/page results when legacy and new
        # embedding directories both contain the same page.
        unique = {}
        for result in results:
            key = (result["document"], result["page_number"])
            if key not in unique or result["score"] > unique[key]["score"]:
                unique[key] = result

        results = list(unique.values())
        results.sort(key=lambda item: item["score"], reverse=True)

        for rank, result in enumerate(results[:top_k], start=1):
            print(
                f"{rank}. {result['document']} | "
                f"Page {result['page_number']} | "
                f"score: {result['score']:.4f}"
            )

        return results[:top_k]


if __name__ == "__main__":
    retriever = VisualRetriever()
    retriever.retrieve(
        "What does the graph on page 7 show?",
        top_k=5
    )
