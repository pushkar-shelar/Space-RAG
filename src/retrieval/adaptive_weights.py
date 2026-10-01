class AdaptiveWeightCalculator:

    def __init__(self):

        # Default weights for a balanced multimodal query
        self.default_weights = {
            "bm25": 0.30,
            "dense": 0.30,
            "colpali": 0.40
        }

    def calculate_weights(self, query_modality, modality_profile):

        # Start with default weights
        weights = self.default_weights.copy()

        # --------------------------------------------------
        # QUERY MODALITY
        # --------------------------------------------------

        if query_modality == "text":

            weights["bm25"] = 0.40
            weights["dense"] = 0.45
            weights["colpali"] = 0.15

        elif query_modality == "visual":

            weights["bm25"] = 0.15
            weights["dense"] = 0.20
            weights["colpali"] = 0.65

        elif query_modality == "multimodal":

            weights["bm25"] = 0.30
            weights["dense"] = 0.30
            weights["colpali"] = 0.40

        # --------------------------------------------------
        # PAGE MODALITY
        # --------------------------------------------------

        page_modality = modality_profile.get(
            "modality",
            "mixed"
        )

        if page_modality == "text":

            weights["bm25"] += 0.05
            weights["dense"] += 0.05
            weights["colpali"] -= 0.10

        elif page_modality == "visual":

            weights["bm25"] -= 0.05
            weights["dense"] -= 0.05
            weights["colpali"] += 0.10

        # --------------------------------------------------
        # NORMALIZE
        # --------------------------------------------------

        total = sum(weights.values())

        if total > 0:

            for key in weights:
                weights[key] = weights[key] / total

        return weights


if __name__ == "__main__":

    calculator = AdaptiveWeightCalculator()

    # --------------------------------------------------
    # TEST 1: TEXT QUERY + TEXT PAGE
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("TEST 1: TEXT QUERY + TEXT PAGE")
    print("=" * 70)

    profile = {
        "modality": "text"
    }

    weights = calculator.calculate_weights(
        "text",
        profile
    )

    print(weights)

    # --------------------------------------------------
    # TEST 2: VISUAL QUERY + VISUAL PAGE
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("TEST 2: VISUAL QUERY + VISUAL PAGE")
    print("=" * 70)

    profile = {
        "modality": "visual"
    }

    weights = calculator.calculate_weights(
        "visual",
        profile
    )

    print(weights)

    # --------------------------------------------------
    # TEST 3: MULTIMODAL QUERY + MIXED PAGE
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("TEST 3: MULTIMODAL QUERY + MIXED PAGE")
    print("=" * 70)

    profile = {
        "modality": "mixed"
    }

    weights = calculator.calculate_weights(
        "multimodal",
        profile
    )

    print(weights)

    # --------------------------------------------------
    # TEST 4: TEXT QUERY + VISUAL PAGE
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("TEST 4: TEXT QUERY + VISUAL PAGE")
    print("=" * 70)

    profile = {
        "modality": "visual"
    }

    weights = calculator.calculate_weights(
        "text",
        profile
    )

    print(weights)