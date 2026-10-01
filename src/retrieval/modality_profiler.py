import json
from pathlib import Path


class ModalityProfiler:

    def __init__(
        self,
        manifest_path="data/processed/metadata/document_manifest.json"
    ):
        self.manifest_path = Path(manifest_path)

        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {self.manifest_path}"
            )

        with open(
            self.manifest_path,
            "r",
            encoding="utf-8"
        ) as file:
            self.manifest = json.load(file)

        print("Modality profiler loaded.")
        print(f"Document: {self.manifest['document']}")
        print(
            f"Pages available: "
            f"{self.manifest['total_pages']}"
        )

    def get_page_profile(self, page_number):
        """
        Calculate the text and visual profile of one page.
        """

        # --------------------------------------------------
        # FIND PAGE
        # --------------------------------------------------

        page = next(
            (
                page
                for page in self.manifest["pages"]
                if page["page_number"] == page_number
            ),
            None
        )

        if page is None:
            raise ValueError(
                f"Page {page_number} not found in manifest."
            )

        # --------------------------------------------------
        # TEXT INFORMATION
        # --------------------------------------------------

        text_info = page.get("text")

        if text_info:
            text_length = text_info.get(
                "char_count",
                0
            )
        else:
            text_length = 0

        # Text density score
        if text_length == 0:
            text_score = 0

        elif text_length < 500:
            text_score = 1

        elif text_length < 1500:
            text_score = 2

        else:
            text_score = 3

        # --------------------------------------------------
        # READ PAGE TEXT
        # --------------------------------------------------

        text = ""

        if text_info and text_info.get("path"):

            text_path = Path(
                text_info["path"]
            )

            if text_path.exists():

                with open(
                    text_path,
                    "r",
                    encoding="utf-8"
                ) as file:
                    text = file.read().lower()

        # --------------------------------------------------
        # VISUAL REFERENCES
        # --------------------------------------------------

        visual_keywords = {
            "figure": [
                "figure",
                "fig."
            ],
            "table": [
                "table"
            ],
            "graph": [
                "graph"
            ],
            "plot": [
                "plot"
            ],
            "chart": [
                "chart"
            ],
            "diagram": [
                "diagram"
            ]
        }

        visual_references = {}

        for category, keywords in visual_keywords.items():

            count = 0

            for keyword in keywords:
                count += text.count(keyword)

            visual_references[category] = count

        # --------------------------------------------------
        # EMBEDDED IMAGES
        # --------------------------------------------------

        embedded_images = page.get(
            "embedded_images",
            []
        )

        image_count = len(
            embedded_images
        )

        # --------------------------------------------------
        # VISUAL SCORE
        # --------------------------------------------------

        visual_score = 0

        # Embedded images
        if image_count > 0:
            visual_score += 2

        # Figure references
        if visual_references["figure"] > 0:
            visual_score += 1

        # Table references
        if visual_references["table"] > 0:
            visual_score += 1

        # Graph references
        if visual_references["graph"] > 0:
            visual_score += 1

        # Plot references
        if visual_references["plot"] > 0:
            visual_score += 1

        # Chart references
        if visual_references["chart"] > 0:
            visual_score += 1

        # Diagram references
        if visual_references["diagram"] > 0:
            visual_score += 1

        # --------------------------------------------------
        # MODALITY CLASSIFICATION
        # --------------------------------------------------

        if visual_score > text_score:

            modality = "visual"

        elif text_score > visual_score:

            modality = "text"

        else:

            modality = "mixed"

        # --------------------------------------------------
        # RETURN PROFILE
        # --------------------------------------------------

        return {
            "document": self.manifest["document"],
            "page_number": page_number,

            "text_profile": {
                "text_length": text_length,
                "text_score": text_score
            },

            "visual_profile": {
                "image_count": image_count,
                "figure_references":
                    visual_references["figure"],
                "table_references":
                    visual_references["table"],
                "graph_references":
                    visual_references["graph"],
                "plot_references":
                    visual_references["plot"],
                "chart_references":
                    visual_references["chart"],
                "diagram_references":
                    visual_references["diagram"],
                "visual_score": visual_score
            },

            "modality": modality
        }


if __name__ == "__main__":

    profiler = ModalityProfiler()

    test_pages = [
        1,
        2,
        4,
        7,
        9
    ]

    print("\n" + "=" * 70)
    print(
        "SPACE RAG - MODALITY PROFILING TEST"
    )
    print("=" * 70)

    for page_number in test_pages:

        profile = profiler.get_page_profile(
            page_number
        )

        text_profile = profile[
            "text_profile"
        ]

        visual_profile = profile[
            "visual_profile"
        ]

        print("\n" + "-" * 70)
        print(
            f"PAGE {page_number}"
        )
        print("-" * 70)

        print(
            f"Text length:             "
            f"{text_profile['text_length']}"
        )

        print(
            f"Text score:              "
            f"{text_profile['text_score']}"
        )

        print(
            f"Embedded images:         "
            f"{visual_profile['image_count']}"
        )

        print(
            f"Figure references:       "
            f"{visual_profile['figure_references']}"
        )

        print(
            f"Table references:        "
            f"{visual_profile['table_references']}"
        )

        print(
            f"Graph references:        "
            f"{visual_profile['graph_references']}"
        )

        print(
            f"Plot references:         "
            f"{visual_profile['plot_references']}"
        )

        print(
            f"Chart references:        "
            f"{visual_profile['chart_references']}"
        )

        print(
            f"Diagram references:     "
            f"{visual_profile['diagram_references']}"
        )

        print(
            f"Visual score:            "
            f"{visual_profile['visual_score']}"
        )

        print(
            f"Classification:          "
            f"{profile['modality']}"
        )