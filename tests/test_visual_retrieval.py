import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.visual_retriever import VisualRetriever


def run_test(retriever, query, expected_page):

    print("\n")
    print("=" * 70)
    print("TEST")
    print("=" * 70)

    print(f"Query: {query}")
    print(f"Expected page: {expected_page}")

    results = retriever.retrieve(
        query,
        top_k=5
    )

    top_page = (
    f"TROPICS_page_"
    f"{results[0]['page_number']:03d}"
    )

    print("\nExpected:")
    print(expected_page)

    print("Retrieved:")
    print(top_page)

    if expected_page in top_page:
        print("RESULT: PASS")
    else:
        print("RESULT: FAIL")


if __name__ == "__main__":

    print("Loading Visual Retriever once...")
    retriever = VisualRetriever()

    run_test(
        retriever,
        "Which page contains the graph and table describing the TROPICS mission?",
        "TROPICS_page_007"
    )

    run_test(
        retriever,
        "Which page describes the TROPICS constellation and its mission?",
        "TROPICS_page_001"
    )

    run_test(
        retriever,
        "Which page has the MicroMAS-2a on-orbit data diagram?",
        "TROPICS_page_009"
    )

    run_test(
        retriever,
        "What are the objectives of the TROPICS mission?",
        "TROPICS_page_001"
    )