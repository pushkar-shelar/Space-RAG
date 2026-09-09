import sys
from pathlib import Path

# Find the project root
PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent.parent
)

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)

from src.indexing.text_chunker import chunk_document
from src.indexing.text_embedder import TextEmbedder


class DenseTextRetriever:

    def __init__(self):

        print("Loading text chunks...")

        self.chunks = chunk_document()

        print(
            f"Loaded {len(self.chunks)} text chunks."
        )

        # Create text embedding model
        self.embedder = TextEmbedder()

        print("\nGenerating embeddings...")

        # Get the text from every chunk
        texts = [
            chunk["text"]
            for chunk in self.chunks
        ]

        # Generate embeddings
        self.embeddings = (
            self.embedder.embed_texts(texts)
        )

        print("Embeddings generated.")


    def retrieve(
        self,
        query,
        top_k=5
    ):

        print("\n" + "=" * 60)
        print("SPACE RAG - DENSE TEXT RETRIEVAL")
        print("=" * 60)

        print(
            f"Query: {query}"
        )

        # Convert query into an embedding
        query_embedding = (
            self.embedder.embed_query(query)
        )

        # Calculate similarity between
        # query and every text chunk
        scores = (
            self.embeddings
            @ query_embedding
        )

        # Pair every chunk with its score
        scored_chunks = list(
            zip(
                self.chunks,
                scores
            )
        )

        # Sort from highest score to lowest
        scored_chunks.sort(
            key=lambda item: item[1].item(),
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
                "score": float(score.item()),
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

            print(
                result["text"][:500]
            )

            print("-" * 60)

        return results


if __name__ == "__main__":

    retriever = DenseTextRetriever()

    retriever.retrieve(
        "What are the main goals of the TROPICS mission?",
        top_k=5
    )