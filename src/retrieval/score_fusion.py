class ScoreNormalizer:

    def min_max_normalize(self, scores):

        if not scores:
            return []

        minimum = min(scores)
        maximum = max(scores)

        # If every score is identical,
        # give every result the same normalized score.
        if maximum == minimum:
            return [1.0 for _ in scores]

        normalized_scores = []

        for score in scores:

            normalized_score = (
                (score - minimum)
                / (maximum - minimum)
            )

            normalized_scores.append(
                normalized_score
            )

        return normalized_scores


class AdaptiveScoreFusion:

    def __init__(self):

        self.normalizer = ScoreNormalizer()

    def fuse(
        self,
        bm25_results,
        dense_results,
        visual_results,
        weights
    ):

        # -------------------------------------------------
        # Store page-level scores
        # -------------------------------------------------

        pages = {}

        def create_page(document, page_number):

            return {
                "document": document,
                "page_number": page_number,
                "bm25_score": 0.0,
                "dense_score": 0.0,
                "colpali_score": 0.0
            }


        # -------------------------------------------------
        # BM25 scores
        # -------------------------------------------------

        for result in bm25_results:

            key = (
                result["document"],
                result["page_number"]
            )

            if key not in pages:

                pages[key] = create_page(
                    result["document"],
                    result["page_number"]
                )

            pages[key]["bm25_score"] = max(
                pages[key]["bm25_score"],
                float(result["score"])
            )


        # -------------------------------------------------
        # Dense scores
        # -------------------------------------------------

        for result in dense_results:

            key = (
                result["document"],
                result["page_number"]
            )

            if key not in pages:

                pages[key] = create_page(
                    result["document"],
                    result["page_number"]
                )

            distance = float(result["distance"])

            similarity = 1.0 / (1.0 + distance)

            pages[key]["dense_score"] = max(
                pages[key]["dense_score"],
                similarity
            )


        # -------------------------------------------------
        # ColPali scores
        # -------------------------------------------------

        for result in visual_results:

            key = (
                result["document"],
                result["page_number"]
            )

            if key not in pages:

                pages[key] = create_page(
                    result["document"],
                    result["page_number"]
                )

            pages[key]["colpali_score"] = max(
                pages[key]["colpali_score"],
                float(result["score"])
            )


        # -------------------------------------------------
        # Prepare scores for normalization
        # -------------------------------------------------

        page_list = list(pages.values())

        bm25_scores = [
            page["bm25_score"]
            for page in page_list
        ]

        dense_scores = [
            page["dense_score"]
            for page in page_list
        ]

        colpali_scores = [
            page["colpali_score"]
            for page in page_list
        ]


        # -------------------------------------------------
        # Normalize scores
        # -------------------------------------------------

        normalized_bm25 = (
            self.normalizer.min_max_normalize(
                bm25_scores
            )
        )

        normalized_dense = (
            self.normalizer.min_max_normalize(
                dense_scores
            )
        )

        normalized_colpali = (
            self.normalizer.min_max_normalize(
                colpali_scores
            )
        )


        # -------------------------------------------------
        # Adaptive weighted fusion
        # -------------------------------------------------

        fused_results = []

        for index, page in enumerate(page_list):

            # Get weights for this specific page

            if callable(weights):

                page_weights = weights(
                    page["document"],
                    page["page_number"]
                )

            else:

                page_weights = weights


            bm25_weight = page_weights["bm25"]

            dense_weight = page_weights["dense"]

            colpali_weight = page_weights["colpali"]


            # Calculate final fusion score

            final_score = (

                normalized_bm25[index]
                * bm25_weight

                +

                normalized_dense[index]
                * dense_weight

                +

                normalized_colpali[index]
                * colpali_weight
            )


            fused_results.append({

                "document":
                    page["document"],

                "page_number":
                    page["page_number"],

                "bm25_raw":
                    page["bm25_score"],

                "dense_raw":
                    page["dense_score"],

                "colpali_raw":
                    page["colpali_score"],

                "bm25_normalized":
                    normalized_bm25[index],

                "dense_normalized":
                    normalized_dense[index],

                "colpali_normalized":
                    normalized_colpali[index],

                "bm25_weight":
                    bm25_weight,

                "dense_weight":
                    dense_weight,

                "colpali_weight":
                    colpali_weight,

                "fusion_score":
                    final_score
            })


        # -------------------------------------------------
        # Sort by fusion score
        # -------------------------------------------------

        fused_results.sort(
            key=lambda result:
                result["fusion_score"],
            reverse=True
        )


        return fused_results

        # --------------------------------------------------
        # COLLECT ALL PAGES
        # --------------------------------------------------

        pages = {}

        # BM25 results
        for result in bm25_results:

            key = (
                result["document"],
                result["page_number"]
            )

            pages.setdefault(
                key,
                {
                    "document": result["document"],
                    "page_number": result["page_number"],
                    "bm25_score": 0.0,
                    "dense_score": 0.0,
                    "colpali_score": 0.0
                }
            )

            pages[key]["bm25_score"] = max(
                pages[key]["bm25_score"],
                float(result["score"])
            )

        # Dense results
        for result in dense_results:

            key = (
                result["document"],
                result["page_number"]
            )

            pages.setdefault(
                key,
                {
                    "document": result["document"],
                    "page_number": result["page_number"],
                    "bm25_score": 0.0,
                    "dense_score": 0.0,
                    "colpali_score": 0.0
                }
            )

            # ChromaDB distance:
            # lower distance = more similar.
            #
            # Convert distance into similarity.
            distance = float(
                result["distance"]
            )

            similarity = 1.0 / (
                1.0 + distance
            )

            pages[key]["dense_score"] = max(
                pages[key]["dense_score"],
                similarity
            )

        # ColPali results
        for result in visual_results:

            key = (
                result["document"],
                result["page_number"]
            )

            pages.setdefault(
                key,
                {
                    "document": result["document"],
                    "page_number": result["page_number"],
                    "bm25_score": 0.0,
                    "dense_score": 0.0,
                    "colpali_score": 0.0
                }
            )

            pages[key]["colpali_score"] = max(
                pages[key]["colpali_score"],
                float(result["score"])
            )

        # --------------------------------------------------
        # NORMALIZE SCORES
        # --------------------------------------------------

        page_list = list(pages.values())

        bm25_scores = [
            page["bm25_score"]
            for page in page_list
        ]

        dense_scores = [
            page["dense_score"]
            for page in page_list
        ]

        colpali_scores = [
            page["colpali_score"]
            for page in page_list
        ]

        normalized_bm25 = (
            self.normalizer.min_max_normalize(
                bm25_scores
            )
        )

        normalized_dense = (
            self.normalizer.min_max_normalize(
                dense_scores
            )
        )

        normalized_colpali = (
            self.normalizer.min_max_normalize(
                colpali_scores
            )
        )

        # --------------------------------------------------
        # ADAPTIVE WEIGHTS
        # --------------------------------------------------

        bm25_weight = weights["bm25"]
        dense_weight = weights["dense"]
        colpali_weight = weights["colpali"]

        # --------------------------------------------------
        # WEIGHTED FUSION
        # --------------------------------------------------

        fused_results = []

        for index, page in enumerate(page_list):

            final_score = (

                normalized_bm25[index]
                * bm25_weight

                +

                normalized_dense[index]
                * dense_weight

                +

                normalized_colpali[index]
                * colpali_weight
            )

            fused_result = {

                "document": page["document"],

                "page_number": page["page_number"],

                "bm25_raw": page["bm25_score"],

                "dense_raw": page["dense_score"],

                "colpali_raw": page["colpali_score"],

                "bm25_normalized":
                    normalized_bm25[index],

                "dense_normalized":
                    normalized_dense[index],

                "colpali_normalized":
                    normalized_colpali[index],

                "bm25_weight":
                    bm25_weight,

                "dense_weight":
                    dense_weight,

                "colpali_weight":
                    colpali_weight,

                "fusion_score":
                    final_score
            }

            fused_results.append(
                fused_result
            )

        # --------------------------------------------------
        # SORT
        # --------------------------------------------------

        fused_results.sort(
            key=lambda result:
                result["fusion_score"],
            reverse=True
        )

        return fused_results


if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("SPACE RAG - SCORE NORMALIZATION + ADAPTIVE FUSION TEST")
    print("=" * 70)

    # --------------------------------------------------
    # SAMPLE RETRIEVAL RESULTS
    # --------------------------------------------------

    bm25_results = [

        {
            "document": "TROPICS",
            "page_number": 7,
            "score": 9.5
        },

        {
            "document": "TROPICS",
            "page_number": 2,
            "score": 6.2
        },

        {
            "document": "TROPICS",
            "page_number": 9,
            "score": 3.5
        }
    ]

    dense_results = [

        {
            "document": "TROPICS",
            "page_number": 7,
            "distance": 0.8
        },

        {
            "document": "TROPICS",
            "page_number": 2,
            "distance": 1.1
        },

        {
            "document": "TROPICS",
            "page_number": 9,
            "distance": 1.5
        }
    ]

    visual_results = [

        {
            "document": "TROPICS",
            "page_number": 7,
            "score": 18.8
        },

        {
            "document": "TROPICS",
            "page_number": 9,
            "score": 12.0
        },

        {
            "document": "TROPICS",
            "page_number": 2,
            "score": 5.0
        }
    ]

    # --------------------------------------------------
    # TEST 1
    # TEXT QUERY + TEXT PAGE
    # --------------------------------------------------

    print("\n" + "-" * 70)
    print("TEST 1: TEXT-ORIENTED FUSION")
    print("-" * 70)

    weights = {
        "bm25": 0.45,
        "dense": 0.50,
        "colpali": 0.05
    }

    fusion = AdaptiveScoreFusion()

    results = fusion.fuse(
        bm25_results,
        dense_results,
        visual_results,
        weights
    )

    for result in results:

        print(
            f"Page {result['page_number']} | "
            f"Fusion: {result['fusion_score']:.4f} | "
            f"BM25: {result['bm25_normalized']:.4f} | "
            f"Dense: {result['dense_normalized']:.4f} | "
            f"ColPali: {result['colpali_normalized']:.4f}"
        )

    # --------------------------------------------------
    # TEST 2
    # VISUAL QUERY + VISUAL PAGE
    # --------------------------------------------------

    print("\n" + "-" * 70)
    print("TEST 2: VISUAL-ORIENTED FUSION")
    print("-" * 70)

    weights = {
        "bm25": 0.10,
        "dense": 0.15,
        "colpali": 0.75
    }

    results = fusion.fuse(
        bm25_results,
        dense_results,
        visual_results,
        weights
    )

    for result in results:

        print(
            f"Page {result['page_number']} | "
            f"Fusion: {result['fusion_score']:.4f} | "
            f"BM25: {result['bm25_normalized']:.4f} | "
            f"Dense: {result['dense_normalized']:.4f} | "
            f"ColPali: {result['colpali_normalized']:.4f}"
        )