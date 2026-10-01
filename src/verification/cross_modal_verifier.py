import re
import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent.parent
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )

import pymupdf

from src.verification.visual_verifier import VisualVerifier


class CrossModalVerifier:

    def __init__(self):

        print(
            "Loading cross-modal verifier..."
        )

        # ---------------------------------------------------------
        # Visual verifier
        # ---------------------------------------------------------

        self.visual_verifier = (
            VisualVerifier()
        )

        # ---------------------------------------------------------
        # Semantic similarity model
        # ---------------------------------------------------------

        print(
            "Loading semantic similarity model..."
        )

        self.semantic_model = (
            SentenceTransformer(
                "all-MiniLM-L6-v2"
            )
        )

        print(
            "Semantic similarity model loaded."
        )

        print(
            "Cross-modal verifier ready."
        )

    # =========================================================
    # Number extraction
    # =========================================================

    def extract_numbers(
        self,
        text
    ):
        """
        Extract numeric values from text.
        """

        if not isinstance(
            text,
            str
        ):
            return []

        pattern = (
            r"(?<!\w)"
            r"-?\d+(?:\.\d+)?"
            r"%?"
        )

        return re.findall(
            pattern,
            text
        )

    # =========================================================
    # Number normalization
    # =========================================================

    def normalize_number(
        self,
        value
    ):

        if value is None:
            return None

        value = (
            str(value)
            .replace("%", "")
            .replace(",", "")
            .strip()
        )

        try:

            return float(
                value
            )

        except ValueError:

            return None

    # =========================================================
    # Extract page text
    # =========================================================

    def extract_page_text(
        self,
        pdf_path,
        page_number
    ):

        document = pymupdf.open(
            pdf_path
        )

        try:

            if (
                page_number < 1
                or page_number > len(document)
            ):
                return ""

            page = document[
                page_number - 1
            ]

            return page.get_text()

        finally:

            document.close()

    # =========================================================
    # Extract tables
    # =========================================================

    def extract_tables(
        self,
        pdf_path,
        page_number
    ):
        """
        Extract structured tables from a PDF page.
        """

        document = pymupdf.open(
            pdf_path
        )

        try:

            if (
                page_number < 1
                or page_number > len(document)
            ):
                return []

            page = document[
                page_number - 1
            ]

            tables = []

            try:

                table_finder = (
                    page.find_tables()
                )

                for table in (
                    table_finder.tables
                ):

                    extracted = (
                        table.extract()
                    )

                    if extracted:

                        tables.append(
                            extracted
                        )

            except Exception as error:

                print(
                    "Table extraction warning: "
                    f"{error}"
                )

            return tables

        finally:

            document.close()

    # =========================================================
    # Convert table to text
    # =========================================================

    def table_to_text(
        self,
        tables
    ):

        text_parts = []

        for table in tables:

            for row in table:

                row_values = []

                for cell in row:

                    if cell is not None:

                        cell_text = (
                            str(cell)
                            .strip()
                        )

                        if cell_text:

                            row_values.append(
                                cell_text
                            )

                if row_values:

                    text_parts.append(
                        " | ".join(
                            row_values
                        )
                    )

        return "\n".join(
            text_parts
        )

    # =========================================================
    # Detect OCR / graph dump
    # =========================================================

    def is_visual_ocr_dump(
        self,
        text
    ):
        """
        Detect raw OCR/layout extraction from a graph,
        chart, or figure.
        """

        if not isinstance(
            text,
            str
        ):
            return False

        cleaned = text.strip()

        if not cleaned:
            return False

        lines = [
            line.strip()
            for line in cleaned.splitlines()
            if line.strip()
        ]

        if not lines:
            return False

        numbers = self.extract_numbers(
            cleaned
        )

        numeric_density = (
            len(numbers)
            / max(len(lines), 1)
        )

        visual_terms = [
            "figure",
            "fig.",
            "graph",
            "plot",
            "chart",
            "diagram",
            "channel",
            "ch.",
            "delta",
            "pressure",
            "altitude",
            "temperature weighting",
            "passband"
        ]

        lower_text = (
            cleaned.lower()
        )

        visual_term_count = sum(
            1
            for term in visual_terms
            if term in lower_text
        )

        short_line_count = sum(
            1
            for line in lines
            if len(line) <= 20
        )

        short_line_ratio = (
            short_line_count
            / max(len(lines), 1)
        )

        # Strong OCR/layout pattern
        if (
            len(numbers) >= 15
            and numeric_density >= 1.0
            and short_line_ratio >= 0.45
        ):
            return True

        # Strong visual numeric pattern
        if (
            len(numbers) >= 25
            and visual_term_count >= 1
        ):
            return True

        # Mixed graph/table OCR pattern
        if (
            visual_term_count >= 2
            and short_line_ratio >= 0.40
            and len(numbers) >= 10
        ):
            return True

        return False

    # =========================================================
    # Determine claim type
    # =========================================================

    def classify_claim(
        self,
        text_claim
    ):
        """
        Classify evidence as:

        numeric
        table
        visual
        textual
        """

        if not isinstance(
            text_claim,
            str
        ):
            return "textual"

        claim = (
            text_claim
            .lower()
            .strip()
        )

        # -----------------------------------------------------
        # Raw visual OCR
        # -----------------------------------------------------

        if self.is_visual_ocr_dump(
            text_claim
        ):
            return "visual"

        # -----------------------------------------------------
        # Table
        # -----------------------------------------------------

        table_terms = [
            "table",
            "tab.",
            "tabulated",
            "row",
            "column",
            "cell"
        ]

        if any(
            re.search(
                r"\b"
                + re.escape(term)
                + r"\b",
                claim
            )
            for term in table_terms
        ):
            return "table"

        # -----------------------------------------------------
        # Numeric
        # -----------------------------------------------------

        numeric_terms = [
            "value",
            "values",
            "number",
            "rate",
            "percentage",
            "percent",
            "frequency",
            "bandwidth",
            "beamwidth",
            "resolution",
            "temperature",
            "altitude",
            "minutes",
            "hours",
            "km",
            "ghz",
            "mhz",
            "degrees",
            "deg"
        ]

        has_numbers = bool(
            self.extract_numbers(
                text_claim
            )
        )

        has_numeric_language = any(
            re.search(
                r"\b"
                + re.escape(term)
                + r"\b",
                claim
            )
            for term in numeric_terms
        )

        if (
            has_numbers
            and has_numeric_language
        ):
            return "numeric"

        # -----------------------------------------------------
        # Visual interpretation
        # -----------------------------------------------------

        visual_terms = [
            "graph",
            "figure",
            "plot",
            "chart",
            "diagram",
            "curve",
            "trend",
            "shows",
            "indicates",
            "illustrates",
            "visual"
        ]

        if any(
            re.search(
                r"\b"
                + re.escape(term)
                + r"\b",
                claim
            )
            for term in visual_terms
        ):
            return "visual"

        return "textual"

    # =========================================================
    # Check visual evidence
    # =========================================================

    def has_visual_evidence(
        self,
        page_text,
        tables
    ):

        if tables:
            return True

        visual_terms = [
            "figure",
            "fig.",
            "graph",
            "plot",
            "chart",
            "diagram",
            "table"
        ]

        page_lower = (
            page_text.lower()
        )

        return any(
            term in page_lower
            for term in visual_terms
        )

    # =========================================================
    # Compare visual description with PDF text
    # =========================================================

    def compare_visual_and_text(
        self,
        visual_description,
        page_text
    ):
        """
        Compare the VLM-generated visual description
        with the actual PDF page text using semantic
        similarity.

        Semantic similarity is calculated using
        all-MiniLM-L6-v2.

        Important:
        A semantic mismatch does NOT automatically mean
        contradiction.

        Low similarity means the textual evidence does
        not sufficiently support the visual description.
        """

        # -----------------------------------------------------
        # Basic validation
        # -----------------------------------------------------

        if not visual_description:

            return {
                "status": "INSUFFICIENT",
                "confidence": 0.0,
                "semantic_score": 0.0,
                "overlap_score": 0.0,
                "matched_terms": [],
                "matched_sentence": ""
            }

        if not page_text:

            return {
                "status": "INSUFFICIENT",
                "confidence": 0.0,
                "semantic_score": 0.0,
                "overlap_score": 0.0,
                "matched_terms": [],
                "matched_sentence": ""
            }

        # -----------------------------------------------------
        # Clean inputs
        # -----------------------------------------------------

        visual_description = (
            str(
                visual_description
            )
            .strip()
        )

        page_text = (
            str(
                page_text
            )
            .strip()
        )

        if not visual_description:
            return {
                "status": "INSUFFICIENT",
                "confidence": 0.0,
                "semantic_score": 0.0,
                "overlap_score": 0.0,
                "matched_terms": [],
                "matched_sentence": ""
            }

        if not page_text:
            return {
                "status": "INSUFFICIENT",
                "confidence": 0.0,
                "semantic_score": 0.0,
                "overlap_score": 0.0,
                "matched_terms": [],
                "matched_sentence": ""
            }

        # -----------------------------------------------------
        # Split PDF text into sentences
        # -----------------------------------------------------

        sentences = [
            sentence.strip()
            for sentence in re.split(
                r"(?<=[.!?])\s+",
                page_text
            )
            if len(sentence.strip()) >= 20
        ]

        # -----------------------------------------------------
        # Fallback
        # -----------------------------------------------------

        if not sentences:

            sentences = [
                page_text
            ]

        # -----------------------------------------------------
        # Encode visual description
        # -----------------------------------------------------

        visual_embedding = (
            self.semantic_model.encode(
                visual_description,
                convert_to_tensor=True,
                normalize_embeddings=True
            )
        )

        # -----------------------------------------------------
        # Encode page sentences
        # -----------------------------------------------------

        sentence_embeddings = (
            self.semantic_model.encode(
                sentences,
                convert_to_tensor=True,
                normalize_embeddings=True
            )
        )

        # -----------------------------------------------------
        # Cosine similarity
        #
        # Embeddings are normalized, so:
        #
        # cosine similarity =
        # normalized embedding dot product
        # -----------------------------------------------------

        similarity_scores = (
            sentence_embeddings
            @ visual_embedding
        )

        # -----------------------------------------------------
        # Find best matching sentence
        # -----------------------------------------------------

        best_index = int(
            similarity_scores
            .argmax()
            .item()
        )

        best_score = float(
            similarity_scores[
                best_index
            ].item()
        )

        best_sentence = (
            sentences[
                best_index
            ]
        )

        # -----------------------------------------------------
        # Token overlap
        #
        # This is ONLY for interpretability.
        # It does NOT determine consistency.
        # -----------------------------------------------------

        visual_tokens = set(
            re.findall(
                r"\b[a-zA-Z0-9]+\b",
                visual_description.lower()
            )
        )

        text_tokens = set(
            re.findall(
                r"\b[a-zA-Z0-9]+\b",
                best_sentence.lower()
            )
        )

        matched_terms = sorted(
            visual_tokens
            & text_tokens
        )

        # -----------------------------------------------------
        # Classification
        # -----------------------------------------------------

        if best_score >= 0.70:

            status = "CONSISTENT"

        elif best_score >= 0.45:

            status = "INSUFFICIENT"

        else:

            status = "INSUFFICIENT"

        return {
            "status": status,

            "confidence": best_score,

            "semantic_score": best_score,

            # Kept for compatibility with the
            # existing retrieval pipeline.
            #
            # This field now represents semantic
            # similarity rather than token overlap.
            "overlap_score": best_score,

            "matched_terms": matched_terms,

            "matched_sentence": best_sentence
        }

    # =========================================================
    # Numeric comparison
    # =========================================================

    def compare_numbers(
        self,
        claim_numbers,
        visual_numbers
    ):
        """
        Compare numeric claim values against
        structured table values.
        """

        matched_numbers = []

        for claim_number in (
            claim_numbers
        ):

            for visual_number in (
                visual_numbers
            ):

                if (
                    abs(
                        claim_number
                        - visual_number
                    )
                    < 0.0001
                ):

                    matched_numbers.append(
                        claim_number
                    )

                    break

        return matched_numbers

    # =========================================================
    # Visual verification
    # =========================================================

    def verify_visual_claim(
        self,
        image_path,
        query,
        page_text
    ):
        """
        Use SmolVLM to interpret a visual page
        and compare that description with PDF text.
        """

        print(
            "\n" + "-" * 60
        )

        print(
            "VISUAL SEMANTIC VERIFICATION"
        )

        print(
            "-" * 60
        )

        # -----------------------------------------------------
        # Run VLM
        # -----------------------------------------------------

        visual_result = (
            self.visual_verifier.analyze_image(
                image_path=image_path,
                question=query
            )
        )

        # -----------------------------------------------------
        # Extract VLM description
        # -----------------------------------------------------

        visual_description = (
            str(
                visual_result
            ).strip()
        )

        visual_description = (
            visual_description.strip()
        )

        print(
            "VLM description:"
        )

        print(
            visual_description
        )

        # -----------------------------------------------------
        # Compare against PDF text
        # -----------------------------------------------------

        comparison = (
            self.compare_visual_and_text(
                visual_description=
                    visual_description,

                page_text=
                    page_text
            )
        )

        print(
            f"Visual-text status: "
            f"{comparison['status']}"
        )

        print(
            f"Visual-text confidence: "
            f"{comparison['confidence']:.2f}"
        )

        print(
            f"Semantic score: "
            f"{comparison['semantic_score']:.4f}"
        )

        print(
            f"Matched sentence: "
            f"{comparison['matched_sentence']}"
        )

        print(
            f"Matched terms: "
            f"{comparison['matched_terms']}"
        )

        return {
            "status":
                comparison[
                    "status"
                ],

            "confidence":
                comparison[
                    "confidence"
                ],

            "visual_description":
                visual_description,

            "semantic_score":
                comparison[
                    "semantic_score"
                ],

            # Maintain compatibility
            # with existing pipeline.
            "overlap_score":
                comparison[
                    "overlap_score"
                ],

            "matched_sentence":
                comparison[
                    "matched_sentence"
                ],

            "matched_terms":
                comparison[
                    "matched_terms"
                ]
        }

    # =========================================================
    # Main verification
    # =========================================================

    def verify(
        self,
        pdf_path,
        page_number,
        text_claim,
        image_path=None,
        query=None
    ):
        """
        Verify textual, numeric, table, or visual evidence.

        Visual claims can optionally use SmolVLM when
        image_path and query are supplied.
        """

        print(
            "\n" + "=" * 60
        )

        print(
            "CROSS-MODAL VERIFICATION"
        )

        print(
            "=" * 60
        )

        print(
            f"Page: {page_number}"
        )

        print(
            f"Claim: {text_claim}"
        )

        # =====================================================
        # 1. Validate claim
        # =====================================================

        if not isinstance(
            text_claim,
            str
        ):

            return {
                "status": "INSUFFICIENT",
                "confidence": 0.0,
                "page_number": page_number,
                "claim": "",
                "claim_type": "unknown",
                "claim_numbers": [],
                "visual_numbers": [],
                "matched_numbers": [],
                "table_count": 0,
                "page_text": "",
                "visual_evidence": "",
                "visual_description": "",
                "semantic_score": 0.0,
                "overlap_score": 0.0,
                "matched_sentence": "",
                "matched_terms": []
            }

        # =====================================================
        # 2. Extract page evidence
        # =====================================================

        page_text = (
            self.extract_page_text(
                pdf_path,
                page_number
            )
        )

        tables = (
            self.extract_tables(
                pdf_path,
                page_number
            )
        )

        table_text = (
            self.table_to_text(
                tables
            )
        )

        # =====================================================
        # 3. Classify claim
        # =====================================================

        claim_type = (
            self.classify_claim(
                text_claim
            )
        )

        # =====================================================
        # 4. Extract numbers
        # =====================================================

        claim_numbers = [

            self.normalize_number(
                number
            )

            for number in (
                self.extract_numbers(
                    text_claim
                )
            )
        ]

        claim_numbers = [

            number

            for number in claim_numbers

            if number is not None
        ]

        visual_numbers = [

            self.normalize_number(
                number
            )

            for number in (
                self.extract_numbers(
                    table_text
                )
            )
        ]

        visual_numbers = [

            number

            for number in visual_numbers

            if number is not None
        ]

        # =====================================================
        # 5. Visual semantic verification
        # =====================================================

        if (
            claim_type == "visual"
            and image_path is not None
            and query is not None
        ):

            visual_verification = (
                self.verify_visual_claim(
                    image_path=image_path,
                    query=query,
                    page_text=page_text
                )
            )

            result = {

                "status":
                    visual_verification[
                        "status"
                    ],

                "confidence":
                    visual_verification[
                        "confidence"
                    ],

                "page_number":
                    page_number,

                "claim":
                    text_claim,

                "claim_type":
                    claim_type,

                "claim_numbers":
                    claim_numbers,

                "visual_numbers":
                    visual_numbers,

                "matched_numbers":
                    [],

                "table_count":
                    len(tables),

                "page_text":
                    page_text,

                "visual_evidence":
                    table_text,

                "visual_description":
                    visual_verification[
                        "visual_description"
                    ],

                "semantic_score":
                    visual_verification[
                        "semantic_score"
                    ],

                "overlap_score":
                    visual_verification[
                        "overlap_score"
                    ],

                "matched_sentence":
                    visual_verification[
                        "matched_sentence"
                    ],

                "matched_terms":
                    visual_verification[
                        "matched_terms"
                    ]
            }

            print(
                f"Claim type: {claim_type}"
            )

            print(
                f"Status: "
                f"{result['status']}"
            )

            print(
                f"Confidence: "
                f"{result['confidence']:.2f}"
            )

            return result

        # =====================================================
        # 6. Numeric/table verification
        # =====================================================

        matched_numbers = []

        if (
            claim_type
            in [
                "numeric",
                "table"
            ]
            and claim_numbers
            and visual_numbers
        ):

            matched_numbers = (
                self.compare_numbers(
                    claim_numbers,
                    visual_numbers
                )
            )

        # =====================================================
        # 7. Determine status
        # =====================================================

        if (
            not page_text
            and not tables
        ):

            status = (
                "INSUFFICIENT"
            )

            confidence = 0.0

        elif claim_type == "visual":

            # Visual claim without an image/query.
            #
            # We cannot perform semantic visual
            # verification, so do not claim consistency.

            status = (
                "INSUFFICIENT"
            )

            confidence = 0.20

        elif (
            claim_type
            in [
                "numeric",
                "table"
            ]
            and claim_numbers
        ):

            if matched_numbers:

                status = (
                    "CONSISTENT"
                )

                confidence = min(
                    1.0,
                    len(
                        matched_numbers
                    )
                    /
                    len(
                        claim_numbers
                    )
                )

            elif visual_numbers:

                status = (
                    "CONTRADICTORY"
                )

                confidence = 0.80

            else:

                status = (
                    "INSUFFICIENT"
                )

                confidence = 0.20

        else:

            status = (
                "INSUFFICIENT"
            )

            confidence = 0.20

        # =====================================================
        # 8. Build result
        # =====================================================

        result = {

            "status":
                status,

            "confidence":
                confidence,

            "page_number":
                page_number,

            "claim":
                text_claim,

            "claim_type":
                claim_type,

            "claim_numbers":
                claim_numbers,

            "visual_numbers":
                visual_numbers,

            "matched_numbers":
                matched_numbers,

            "table_count":
                len(tables),

            "page_text":
                page_text,

            "visual_evidence":
                table_text,

            "visual_description":
                "",

            "semantic_score":
                0.0,

            "overlap_score":
                0.0,

            "matched_sentence":
                "",

            "matched_terms":
                []
        }

        # =====================================================
        # 9. Display result
        # =====================================================

        print(
            f"Claim type: "
            f"{claim_type}"
        )

        print(
            f"Status: "
            f"{status}"
        )

        print(
            f"Confidence: "
            f"{confidence:.2f}"
        )

        print(
            f"Claim numbers: "
            f"{claim_numbers}"
        )

        print(
            f"Visual numbers: "
            f"{visual_numbers}"
        )

        print(
            f"Matched numbers: "
            f"{matched_numbers}"
        )

        print(
            f"Tables found: "
            f"{len(tables)}"
        )

        return result


# ============================================================
# Standalone test
# ============================================================

if __name__ == "__main__":

    verifier = (
        CrossModalVerifier()
    )

    pdf_path = (
        r"data\corpus\TROPICS.pdf"
    )

    image_path = (
        r"data\processed\pages"
        r"\TROPICS_page_007.png"
    )

    # --------------------------------------------------------
    # TEST 1: Matching table value
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "TEST 1: MATCHING TABLE VALUE"
    )

    print(
        "=" * 70
    )

    verifier.verify(
        pdf_path=pdf_path,
        page_number=7,
        text_claim=(
            "The table reports "
            "a value of 27.7."
        )
    )

    # --------------------------------------------------------
    # TEST 2: Non-matching table value
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "TEST 2: NON-MATCHING TABLE VALUE"
    )

    print(
        "=" * 70
    )

    verifier.verify(
        pdf_path=pdf_path,
        page_number=7,
        text_claim=(
            "The table reports "
            "a value of 10."
        )
    )

    # --------------------------------------------------------
    # TEST 3: Visual semantic verification
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "TEST 3: VISUAL SEMANTIC VERIFICATION"
    )

    print(
        "=" * 70
    )

    verifier.verify(
        pdf_path=pdf_path,
        page_number=7,
        text_claim=(
            "The graph shows "
            "temperature weighting "
            "functions for the "
            "TROPICS channels."
        ),
        image_path=image_path,
        query=(
            "What does the graph on "
            "page 7 show?"
        )
    )