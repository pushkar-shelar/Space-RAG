from sentence_transformers import CrossEncoder


class Reranker:

    def __init__(
        self,
        model_name="BAAI/bge-reranker-base"
    ):

        print("Loading reranker model...")

        self.model = CrossEncoder(
            model_name
        )

        print("Reranker model loaded.")

    def get_best_text(
        self,
        result
    ):

        text_results = result.get(
            "text_results",
            []
        )

        if not text_results:
            return ""

        best_result = max(
            text_results,
            key=lambda item:
                item.get("score", 0)
        )

        return best_result.get(
            "text",
            ""
        )

    def rerank(
        self,
        query,
        results,
        top_k=5
    ):

        pairs = []

        for result in results:

            text = self.get_best_text(
                result
            )

            pairs.append(
                (
                    query,
                    text
                )
            )

        scores = self.model.predict(
            pairs
        )

        reranked_results = []

        for result, score in zip(
            results,
            scores
        ):

            result = result.copy()

            result["reranker_score"] = (
                float(score)
            )

            reranked_results.append(
                result
            )

        reranked_results.sort(
            key=lambda item:
                item["reranker_score"],
            reverse=True
        )

        return reranked_results[:top_k]