import sys
from pathlib import Path

# Find the indexing folder
PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent.parent
)

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)

from src.indexing.text_chunker import chunk_document


class TextRetriever:

    def __init__(self):

        print("Loading text chunks...")

        self.chunks = chunk_document()

        print(
            f"Loaded {len(self.chunks)} text chunks."
        )

        # Prepare documents for BM25
        self.documents = [
            chunk["text"]
            for chunk in self.chunks
        ]

        # Import BM25
        from rank_bm25 import BM25Okapi

        # Convert each document into words
        tokenized_documents = [
            document.lower().split()
            for document in self.documents
        ]

        # Create BM25 index
        self.bm25 = BM25Okapi(
            tokenized_documents
        )

        print("BM25 index created.")


    def retrieve(
        self,
        query,
        top_k=5
    ):

        print("\n" + "=" * 60)
        print("SPACE RAG - TEXT RETRIEVAL")
        print("=" * 60)

        print(
            f"Query: {query}"
        )

        # Convert query into words
        tokenized_query = (
            query.lower().split()
        )

        # Calculate BM25 scores
        scores = self.bm25.get_scores(
            tokenized_query
        )

        # Pair each chunk with its score
        scored_chunks = list(
            zip(
                self.chunks,
                scores
            )
        )

        # Sort highest score first
        scored_chunks.sort(
            key=lambda item: item[1],
            reverse=True
        )

        results = []

        for chunk, score in scored_chunks[:top_k]:

            result = {
                "document": chunk["document"],
                "page_number": chunk["page_number"],
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "source": chunk["source"],
                "score": float(score),
                "modality": "text"
            }

            results.append(result)

        print("\n" + "=" * 60)
        print("TOP RESULTS")
        print("=" * 60)

        for rank, result in enumerate(
            results,
            start=1
        ):

            print(
                f"{rank}. "
                f"{result['document']} "
                f"| Page {result['page_number']} "
                f"| Chunk {result['chunk_id']} "
                f"| score: "
                f"{result['score']:.4f}"
            )

            # Show the beginning of the retrieved text
            print(
                result["text"][:500]
            )

            print("-" * 60)

        return results


if __name__ == "__main__":

    retriever = TextRetriever()

    retriever.retrieve(
        "What are the objectives of the TROPICS mission?",
        top_k=5
    )