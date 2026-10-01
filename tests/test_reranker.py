import sys
from pathlib import Path

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)

from src.indexing.reranker import Reranker


if __name__ == "__main__":

    reranker = Reranker()

    query = (
        "What are the main goals "
        "of the TROPICS mission?"
    )

    results = [
        {
            "text_results": [
                {
                    "text": (
                        "The overarching goal "
                        "for TROPICS is to provide "
                        "nearly all-weather observations "
                        "of 3-D temperature and humidity, "
                        "as well as cloud ice and "
                        "precipitation structure at "
                        "high temporal resolution."
                    )
                }
            ]
        },
        {
            "text_results": [
                {
                    "text": (
                        "The TROPICS bus will provide "
                        "power, communications, thermal "
                        "management and attitude control."
                    )
                }
            ]
        }
    ]

    reranked = reranker.rerank(
        query,
        results,
        top_k=2
    )

    print("\n")
    print("=" * 60)
    print("RERANKER TEST")
    print("=" * 60)

    for rank, result in enumerate(
        reranked,
        start=1
    ):

        print(
            f"\n{rank}. "
            f"Score: "
            f"{result['reranker_score']:.4f}"
        )

        print(
            result["text_results"][0]["text"]
        )