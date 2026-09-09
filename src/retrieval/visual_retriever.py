import sys
import json
from pathlib import Path

import torch


# Find the indexing folder
INDEXING_DIR = (
    Path(__file__).resolve().parent.parent / "indexing"
)

sys.path.insert(0, str(INDEXING_DIR))

from colpali_embedder import ColPaliEmbedder


class VisualRetriever:

    def __init__(
        self,
        embeddings_dir="data/processed/embeddings/TROPICS"
    ):

        self.embeddings_dir = Path(
            embeddings_dir
        )

        # Path to the document manifest
        self.manifest_path = (
            Path("data")
            / "processed"
            / "metadata"
            / "document_manifest.json"
        )

        # Load the document manifest
        with open(
            self.manifest_path,
            "r",
            encoding="utf-8"
        ) as file:

            self.manifest = json.load(file)

        self.embedder = ColPaliEmbedder()


    def create_query_embedding(self, query):

        batch = self.embedder.processor.process_queries(
            [query]
        )

        batch = {
            key: value.to(self.embedder.device)
            for key, value in batch.items()
        }

        with torch.no_grad():

            query_embedding = self.embedder.model(
                **batch
            )

        return query_embedding


    def retrieve(self, query, top_k=5):

        print("=" * 60)
        print("SPACE RAG - VISUAL RETRIEVAL")
        print("=" * 60)

        print(f"Query: {query}")

        print("\nGenerating query embedding...")

        query_embedding = self.create_query_embedding(
            query
        )

        page_files = sorted(
            self.embeddings_dir.glob("*.pt")
        )

        print(
            f"Comparing against "
            f"{len(page_files)} pages..."
        )

        page_embeddings = []

        for page_file in page_files:

            embedding = torch.load(
                page_file,
                map_location="cpu"
            )

            page_embeddings.append(
                embedding.squeeze(0)
            )

        scores = (
            self.embedder.processor.score_multi_vector(
                query_embedding.cpu(),
                page_embeddings
            )
        )

        results = []

        for page_file, score in zip(
            page_files,
            scores[0]
        ):

            # Example:
            # TROPICS_page_007
            page_name = page_file.stem

            # Extract page number from filename
            # TROPICS_page_007 → 7
            page_number = int(
                page_name.split("_")[-1]
            )

            # Find matching page information
            # inside the document manifest
            page_info = next(
                (
                    page
                    for page in self.manifest["pages"]
                    if page["page_number"] == page_number
                ),
                None
            )

            # Safety check
            if page_info is None:
                print(
                    f"Warning: Page {page_number} "
                    f"not found in manifest."
                )
                continue

            # Create provenance information
            result = {
                "document": Path(
                self.manifest["document"]).stem,
                "page_number": page_number,
                "page_image": page_info["page_image"]["path"],
                "score": score.item(),
                "modality": "visual",
                "image_path": self.manifest["pages"][
                    page_number - 1
                ]["page_image"]["path"],
                "modality": "page_image"
            }

            results.append(result)

        results.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        print("\n" + "=" * 60)
        print("TOP RESULTS")
        print("=" * 60)

        for rank, result in enumerate(
            results[:top_k],
            start=1
        ):

            print(
                f"{rank}. "
                f"{result['document']} "
                f"| Page {result['page_number']} "
                f"| score: "
                f"{result['score']:.4f}"
            )

        return results[:top_k]