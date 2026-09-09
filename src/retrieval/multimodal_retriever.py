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


from src.indexing.reranker import Reranker
from src.retrieval.text_retriever import TextRetriever
from src.retrieval.dense_text_retriever import DenseTextRetriever
from src.retrieval.visual_retriever import VisualRetriever

class MultimodalRetriever:

    def __init__(self):

        print("=" * 70)
        print("SPACE RAG - LOADING MULTIMODAL RETRIEVER")
        print("=" * 70)

        # BM25 retriever
        print("\nLoading BM25 retriever...")

        self.bm25_retriever = TextRetriever()


        # Dense text retriever
        print("\nLoading dense text retriever...")

        self.dense_retriever = DenseTextRetriever()


        # ColPali visual retriever
        print("\nLoading ColPali visual retriever...")

        self.visual_retriever = VisualRetriever()


        # Reranker
        print("\nLoading reranker...")

        self.reranker = Reranker()


        print("\n" + "=" * 70)
        print("MULTIMODAL RETRIEVER READY")
        print("=" * 70)


    def retrieve(
        self,
        query,
        top_k=5,
        retrieval_k=10
    ):

        print("\n")
        print("=" * 70)
        print("SPACE RAG - MULTIMODAL RETRIEVAL")
        print("=" * 70)

        print(
            f"Query: {query}"
        )


        # -------------------------------------------------
        # 1. BM25 retrieval
        # -------------------------------------------------

        print("\n[1/3] Running BM25 retrieval...")

        bm25_results = (
            self.bm25_retriever.retrieve(
                query,
                top_k=retrieval_k
            )
        )


        # -------------------------------------------------
        # 2. Dense text retrieval
        # -------------------------------------------------

        print("\n[2/3] Running dense text retrieval...")

        dense_results = (
            self.dense_retriever.retrieve(
                query,
                top_k=retrieval_k
            )
        )


        # -------------------------------------------------
        # 3. ColPali visual retrieval
        # -------------------------------------------------

        print("\n[3/3] Running ColPali visual retrieval...")

        visual_results = (
            self.visual_retriever.retrieve(
                query,
                top_k=retrieval_k
            )
        )


        # -------------------------------------------------
        # Store RRF scores
        # -------------------------------------------------

        fused_scores = {}

        # Store result information
        result_data = {}


        # -------------------------------------------------
        # BM25 → RRF
        # -------------------------------------------------

        for rank, result in enumerate(
            bm25_results,
            start=1
        ):

            key = (
                result["document"],
                result["page_number"]
            )

            rrf_score = 1 / (60 + rank)

            fused_scores[key] = (
                fused_scores.get(key, 0)
                + rrf_score
            )

            result_data[key] = {
                "document": result["document"],
                "page_number": result["page_number"],
                "text_results": [],
                "visual_results": []
            }

            result_data[key]["text_results"].append(
                result
            )


        # -------------------------------------------------
        # Dense → RRF
        # -------------------------------------------------

        for rank, result in enumerate(
            dense_results,
            start=1
        ):

            key = (
                result["document"],
                result["page_number"]
            )

            rrf_score = 1 / (60 + rank)

            fused_scores[key] = (
                fused_scores.get(key, 0)
                + rrf_score
            )

            if key not in result_data:

                result_data[key] = {
                    "document": result["document"],
                    "page_number": result["page_number"],
                    "text_results": [],
                    "visual_results": []
                }

            result_data[key]["text_results"].append(
                result
            )


        # -------------------------------------------------
        # ColPali → RRF
        # -------------------------------------------------

        for rank, result in enumerate(
            visual_results,
            start=1
        ):

            key = (
                result["document"],
                result["page_number"]
            )

            rrf_score = 1 / (60 + rank)

            fused_scores[key] = (
                fused_scores.get(key, 0)
                + rrf_score
            )

            if key not in result_data:

                result_data[key] = {
                    "document": result["document"],
                    "page_number": result["page_number"],
                    "text_results": [],
                    "visual_results": []
                }

            result_data[key]["visual_results"].append(
                result
            )


        # -------------------------------------------------
        # Sort fused results
        # -------------------------------------------------

        ranked_results = sorted(
            fused_scores.items(),
            key=lambda item: item[1],
            reverse=True
        )


        # -------------------------------------------------
        # Build RRF candidate results
        # -------------------------------------------------

        candidates = []

        for key, fused_score in ranked_results[:retrieval_k]:

            result = result_data[key].copy()

            result["fusion_score"] = (
                float(fused_score)
            )

            candidates.append(
                result
            )


        # -------------------------------------------------
        # Rerank candidates
        # -------------------------------------------------

        print("\n")
        print("=" * 70)
        print("RERANKING CANDIDATES")
        print("=" * 70)

        results = self.reranker.rerank(
            query,
            candidates,
            top_k=top_k
        )

        # -------------------------------------------------
        # Build clean provenance information
        # -------------------------------------------------

        for result in results:

            best_text = ""

            if result["text_results"]:

                best_text_result = max(
                    result["text_results"],
                    key=lambda item:
                        item.get("score", 0)
                )

                best_text = best_text_result.get(
                    "text",
                    ""
                )

            best_visual = None

            if result["visual_results"]:

                best_visual = max(
                    result["visual_results"],
                    key=lambda item:
                        item.get("score", 0)
                )

            result["provenance"] = {

                "document": result["document"],

                "page_number": result[
                    "page_number"
                ],

                "text_source": (
                    best_text_result.get("source")
                    if result["text_results"]
                    else None
                ),

                "page_image": (
                    best_visual.get("image_path")
                    if best_visual
                    else None
                )
            }

            result["evidence_summary"] = {

                "text_available": bool(
                    result["text_results"]
                ),

                "visual_available": bool(
                    result["visual_results"]
                ),

                "text_evidence_count": len(
                    result["text_results"]
                ),

                "visual_evidence_count": len(
                    result["visual_results"]
                )
            }

            result["scores"] = {

                "fusion": float(
                    result["fusion_score"]
                ),

                "reranker": float(
                    result["reranker_score"]
                ),

                "visual": (
                    float(best_visual["score"])
                    if best_visual
                    else None
                )
            }

            result["retrieval_id"] = (
                f"{result['document']}"
                f"_page_{result['page_number']:03d}"
            )
        # -------------------------------------------------
        # Display final results
        # -------------------------------------------------

        print("\n")
        print("=" * 70)
        print("FINAL MULTIMODAL RESULTS")
        print("=" * 70)

        for rank, result in enumerate(
            results,
            start=1
        ):

            print("\n" + "-" * 70)

            print(
                f"Rank: {rank}"
            )

            print(
                f"Document: {result['document']}"
            )

            print(
                f"Page: {result['page_number']}"
            )

            print(
                f"Retrieval ID: "
                f"{result['retrieval_id']}"
            )

            print(
                f"Fusion Score: "
                f"{result['scores']['fusion']:.6f}"
            )

            print(
                f"Reranker Score: "
                f"{result['scores']['reranker']:.4f}"
            )

            print(
                f"Visual Score: "
                f"{result['scores']['visual']}"
            )

            print(
                f"Text Evidence: "
                f"{result['evidence_summary']['text_evidence_count']}"
            )

            print(
                f"Visual Evidence: "
                f"{result['evidence_summary']['visual_evidence_count']}"
            )

            print(
                f"Text Available: "
                f"{result['evidence_summary']['text_available']}"
            )

            print(
                f"Visual Available: "
                f"{result['evidence_summary']['visual_available']}"
            )

            print(
                f"Text Source: "
                f"{result['provenance']['text_source']}"
            )

            print(
                f"Page Image: "
                f"{result['provenance']['page_image']}"
            )

        print("\n" + "=" * 70)
        print("MULTIMODAL RETRIEVAL COMPLETE")
        print("=" * 70)

        return results


# -------------------------------------------------
# Test the multimodal retriever
# -------------------------------------------------

if __name__ == "__main__":

    print("\n")
    print("=" * 70)
    print("LOADING MULTIMODAL RETRIEVER")
    print("=" * 70)

    retriever = MultimodalRetriever()

    results = retriever.retrieve(
        "What are the main goals of the TROPICS mission?",
        top_k=5,
        retrieval_k=10
    )