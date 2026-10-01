import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.chroma_text_retriever import ChromaTextRetriever

def print_results(query, results):
    print("\n" + "=" * 70)
    print(f"QUERY: {query}")
    print("=" * 70)

    for rank, result in enumerate(results, start=1):
        print(f"\nRank {rank}")
        print(f"Document: {result['document']}")
        print(f"Page: {result['page_number']}")
        print(f"Chunk: {result['chunk_id']}")
        print(f"Distance: {result['distance']:.4f}")
        print(f"Text: {result['text'][:300]}...")


def main():
    retriever = ChromaTextRetriever()

    test_queries = [
        "What are the main goals of the TROPICS mission?",
        "What is the overarching goal of the TROPICS mission?",
        "What are the TROPICS satellites used for?",
        "What happened to MicroMAS-2a?",
        "What is the constellation of TROPICS satellites?",
    ]

    for query in test_queries:
        results = retriever.retrieve(query, top_k=5)
        print_results(query, results)


if __name__ == "__main__":
    main()