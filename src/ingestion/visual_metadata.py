import pymupdf
from pathlib import Path


def extract_visual_metadata(pdf_path):

    pdf_path = Path(pdf_path)

    document = pymupdf.open(pdf_path)

    visual_pages = []

    for page_number, page in enumerate(document):

        page_number_display = page_number + 1

        text = page.get_text()

        images = page.get_images(full=True)

        visual_items = []

        # Embedded raster images
        for image_number, image in enumerate(images):

            visual_items.append(
                {
                    "type": "embedded_image",
                    "image_id": image_number + 1,
                    "xref": image[0]
                }
            )

        # Look for common visual keywords
        visual_keywords = [
            "figure",
            "fig.",
            "table",
            "graph",
            "diagram",
            "plot",
            "chart"
        ]

        text_lower = text.lower()

        detected_keywords = []

        for keyword in visual_keywords:

            if keyword in text_lower:

                detected_keywords.append(
                    keyword
                )

        if detected_keywords:

            visual_items.append(
                {
                    "type": "referenced_visual",
                    "keywords": detected_keywords
                }
            )

        if visual_items:

            visual_pages.append(
                {
                    "document": pdf_path.name,
                    "page_number": page_number_display,
                    "visual_items": visual_items
                }
            )

    document.close()

    return visual_pages


if __name__ == "__main__":

    results = extract_visual_metadata(
        r"data\corpus\TROPICS.pdf"
    )

    print("\n")
    print("=" * 60)
    print("VISUAL METADATA")
    print("=" * 60)

    for page in results:

        print(
            f"\nPage {page['page_number']}"
        )

        for item in page["visual_items"]:

            print(
                f"  {item}"
            )