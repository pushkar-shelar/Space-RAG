import re

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

    # ---------------------------------------------------------
    # Tokenization
    # ---------------------------------------------------------

    def tokenize(self, text):

        return re.findall(
            r"\b[a-zA-Z0-9]+\b",
            text.lower()
        )

    # ---------------------------------------------------------
    # Normal lexical relevance
    # ---------------------------------------------------------

    def lexical_relevance(
        self,
        query,
        text
    ):

        query_tokens = set(
            self.tokenize(query)
        )

        text_tokens = set(
            self.tokenize(text)
        )

        if not query_tokens:

            return 0.0

        overlap = (
            query_tokens
            & text_tokens
        )

        return len(overlap) / len(
            query_tokens
        )

    # ---------------------------------------------------------
    # Intent-specific lexical relevance
    # ---------------------------------------------------------

    def intent_lexical_relevance(
        self,
        query,
        text
    ):

        query_lower = query.lower()
        text_lower = text.lower()

        intent_groups = {

            "objective": [
                "objective",
                "objectives",
                "goal",
                "goals",
                "purpose",
                "aim",
                "aims"
            ],

            "definition": [
                "what is",
                "what are",
                "define",
                "definition"
            ],

            "requirement": [
                "requirement",
                "requirements",
                "needed",
                "necessary"
            ],

            "process": [
                "process",
                "procedure",
                "method",
                "steps",
                "how"
            ],

            "cause": [
                "why",
                "reason",
                "cause"
            ]
        }

        query_intent = self.detect_query_intent(
            query
        )

        if query_intent not in intent_groups:

            return 0.0

        intent_terms = intent_groups[
            query_intent
        ]

        matched_terms = 0

        for term in intent_terms:

            if re.search(
                r"\b" + re.escape(term) + r"\b",
                query_lower
            ):

                if term in text_lower:

                    matched_terms += 1

        if not matched_terms:

            return 0.0

        query_term_count = sum(
            1
            for term in intent_terms
            if re.search(
                r"\b" + re.escape(term) + r"\b",
                query_lower
            )
        )

        if query_term_count == 0:

            return 0.0

        return (
            matched_terms
            / query_term_count
        )

    # ---------------------------------------------------------
    # Query intent detection
    # ---------------------------------------------------------

    def detect_query_intent(self, query):

        query_lower = query.lower()

        def contains_term(term):
            return re.search(
                r"\b" + re.escape(term) + r"\b",
                query_lower
            ) is not None

        # Objective
        if any(
            contains_term(term)
            for term in [
                "objective",
                "objectives",
                "goal",
                "goals",
                "purpose",
                "aim",
                "aims"
            ]
        ):
            return "objective"

        # Definition
        if any(
            contains_term(term)
            for term in [
                "what is",
                "what are",
                "define",
                "definition"
            ]
        ):
            return "definition"

        # Requirement
        if any(
            contains_term(term)
            for term in [
                "requirement",
                "requirements",
                "needed",
                "necessary"
            ]
        ):
            return "requirement"

        # Process
        if any(
            contains_term(term)
            for term in [
                "process",
                "procedure",
                "method",
                "steps",
                "how"
            ]
        ):
            return "process"

        # Cause
        if any(
            contains_term(term)
            for term in [
                "why",
                "reason",
                "cause"
            ]
        ):
            return "cause"

        return "general"

    # ---------------------------------------------------------
    # Final score calculation
    # ---------------------------------------------------------

    def calculate_final_score(
        self,
        query_intent,
        query_modality,
        reranker_score,
        lexical_score,
        intent_lexical_score,
        fusion_score,
        has_visual
    ):

        # =====================================================
        # VISUAL QUERY
        # =====================================================

        if query_modality == "visual":

            if has_visual:

                # Visual evidence should dominate.
                #
                # Fusion already contains ColPali
                # and adaptive modality weighting.
                #
                # BGE remains as supporting evidence only.

                return (
                    0.70 * fusion_score
                    + 0.20 * reranker_score
                    + 0.05 * lexical_score
                    + 0.05 * intent_lexical_score
                )

            else:

                # No visual evidence:
                # strongly penalize text-only candidates.

                return (
                    0.30 * fusion_score
                    + 0.10 * reranker_score
                    + 0.05 * lexical_score
                    + 0.05 * intent_lexical_score
                )

        # =====================================================
        # MULTIMODAL QUERY
        # =====================================================

        if query_modality == "multimodal":

            if has_visual:

                return (
                    0.40 * fusion_score
                    + 0.30 * reranker_score
                    + 0.20 * lexical_score
                    + 0.10 * intent_lexical_score
                )

            else:

                return (
                    0.45 * fusion_score
                    + 0.30 * reranker_score
                    + 0.20 * lexical_score
                    + 0.05 * intent_lexical_score
                )

        # =====================================================
        # TEXT QUERY
        # =====================================================

        if query_intent in [
            "objective",
            "definition",
            "requirement"
        ]:

            return (
                0.25 * reranker_score
                + 0.40 * lexical_score
                + 0.20 * intent_lexical_score
                + 0.15 * fusion_score
            )

        if query_intent in [
            "process",
            "cause"
        ]:

            return (
                0.30 * reranker_score
                + 0.35 * lexical_score
                + 0.15 * intent_lexical_score
                + 0.20 * fusion_score
            )

        # General text query

        return (
            0.35 * reranker_score
            + 0.30 * lexical_score
            + 0.10 * intent_lexical_score
            + 0.25 * fusion_score
        )

    # ---------------------------------------------------------
    # Rerank
    # ---------------------------------------------------------

    def rerank(
        self,
        query,
        results,
        top_k=5,
        query_modality="text"
    ):

        query_intent = self.detect_query_intent(
            query
        )

        print(
            f"Query intent: {query_intent}"
        )

        if not results:

            return []

        reranked_results = []

        for result in results:

            # -------------------------------------------------
            # Get best text evidence
            # -------------------------------------------------

            text_results = result.get(
                "text_results",
                []
            )

            if text_results:

                best_text_result = max(
                    text_results,
                    key=lambda item:
                        item.get(
                            "score",
                            0
                        )
                )

                text = best_text_result.get(
                    "text",
                    ""
                )

            else:

                text = ""

            # -------------------------------------------------
            # BGE score
            # -------------------------------------------------

            if text.strip():

                bge_score = float(
                    self.model.predict(
                        [
                            (
                                query,
                                text
                            )
                        ]
                    )[0]
                )

                # BGE scores can be negative
                # or greater than 1 depending
                # on the model/output.

                bge_score = max(
                    0.0,
                    min(
                        1.0,
                        bge_score
                    )
                )

            else:

                bge_score = 0.0

            # -------------------------------------------------
            # Lexical scores
            # -------------------------------------------------

            lexical_score = (
                self.lexical_relevance(
                    query,
                    text
                )
            )

            intent_lexical_score = (
                self.intent_lexical_relevance(
                    query,
                    text
                )
            )

            # -------------------------------------------------
            # Visual evidence
            # -------------------------------------------------

            visual_results = result.get(
                "visual_results",
                []
            )

            has_visual = bool(
                visual_results
            )

            # -------------------------------------------------
            # Fusion score
            # -------------------------------------------------

            fusion_score = float(
                result.get(
                    "fusion_score",
                    0.0
                )
            )

            # -------------------------------------------------
            # Final score
            # -------------------------------------------------

            final_score = (
                self.calculate_final_score(
                    query_intent=query_intent,
                    query_modality=query_modality,
                    reranker_score=bge_score,
                    lexical_score=lexical_score,
                    intent_lexical_score=
                        intent_lexical_score,
                    fusion_score=fusion_score,
                    has_visual=has_visual
                )
            )

            result["reranker_score"] = (
                final_score
            )
            result["final_score"] = (
                final_score
            )

            result["bge_score"] = (
                bge_score
            )

            result["lexical_score"] = (
                lexical_score
            )

            result["intent_lexical_score"] = (
                intent_lexical_score
            )

            result["query_intent"] = (
                query_intent
            )

            reranked_results.append(
                result
            )

        # -----------------------------------------------------
        # Sort
        # -----------------------------------------------------

        reranked_results.sort(
            key=lambda item:
                item.get(
                    "reranker_score",
                    0
                ),
            reverse=True
        )

        return reranked_results[
            :top_k
        ]