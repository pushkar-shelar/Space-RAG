import sys
from pathlib import Path

import chromadb

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent.parent
)

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)

from src.indexing.text_embedder import TextEmbedder


class ChromaTextRetriever:

    def __init__(
        self,
        persist_directory="data/processed/chroma",
        collection_name="space_rag_text"
    ):

        print("Loading ChromaDB...")

        self.client = chromadb.PersistentClient(
            path=persist_directory
        )

        self.collection = (
            self.client.get_collection(
                name=collection_name
            )
        )

        print(
            f"ChromaDB collection: "
            f"{collection_name}"
        )

        print(
            f"Stored documents: "
            f"{self.collection.count()}"
        )

        print("Loading text embedding model...")

        self.embedder = TextEmbedder()

        print("ChromaDB retriever ready.")

    def retrieve(
        self,
        query,
        top_k=5
    ):

        print("\n")
        print("=" * 60)
        print("SPACE RAG - CHROMADB TEXT RETRIEVAL")
        print("=" * 60)

        print(f"Query: {query}")

        print("\nGenerating query embedding...")

        query_embedding = (
            self.embedder.embed_query(query)
        )

        query_embedding = (
            query_embedding
            .cpu()
            .numpy()
            .tolist()
        )

        print("Query embedding generated.")

        results = self.collection.query(
            query_embeddings=[
                query_embedding
            ],
            n_results=top_k
        )

        retrieved_results = []

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        ids = results["ids"][0]

        for (
            document,
            metadata,
            distance,
            retrieval_id
        ) in zip(
            documents,
            metadatas,
            distances,
            ids
        ):

            result = {

                "retrieval_id": retrieval_id,

                "document": metadata[
                    "document"
                ],

                "page_number": metadata[
                    "page_number"
                ],

                "chunk_id": metadata[
                    "chunk_id"
                ],

                "text": document,

                "source": metadata[
                    "source"
                ],

                "modality": metadata[
                    "modality"
                ],

                "distance": float(
                    distance
                )
            }

            retrieved_results.append(
                result
            )

        print("\n")
        print("=" * 60)
        print("TOP RESULTS")
        print("=" * 60)

        for rank, result in enumerate(
            retrieved_results,
            start=1
        ):

            print(
                f"{rank}. "
                f"{result['document']} | "
                f"Page {result['page_number']} | "
                f"Chunk {result['chunk_id']} | "
                f"distance: "
                f"{result['distance']:.4f}"
            )

            print(
                result["text"][:300]
                .replace("\n", " ")
            )

            print("-" * 60)

        return retrieved_results


if __name__ == "__main__":

    retriever = ChromaTextRetriever()

    retriever.retrieve(
        "What are the main goals of the TROPICS mission?",
        top_k=5
    )