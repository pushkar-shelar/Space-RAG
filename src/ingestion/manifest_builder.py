import json
from pathlib import Path


def build_manifest(
    pdf_path,
    text_pages,
    image_pages,
    rendered_pages,
    output_dir="data/processed/metadata"
):

    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    total_pages = max(
        len(text_pages),
        len(image_pages),
        len(rendered_pages)
    )

    pages = []

    for page_number in range(1, total_pages + 1):

        page_record = {
            "page_number": page_number,
            "text": None,
            "page_image": None,
            "embedded_images": []
        }

        # -------------------------
        # Text
        # -------------------------

        text_page = next(
            (
                page
                for page in text_pages
                if page["page_number"] == page_number
            ),
            None
        )

        if text_page:

            page_record["text"] = {
                "path": text_page["text_path"],
                "modality": "text",
                "char_count": text_page["char_count"]
            }

        # -------------------------
        # Embedded images
        # -------------------------

        image_page = next(
            (
                page
                for page in image_pages
                if page["page_number"] == page_number
            ),
            None
        )

        if image_page:

            page_record["embedded_images"] = (
                image_page["images"]
            )

        # -------------------------
        # Rendered page image
        # -------------------------

        rendered_page = next(
            (
                page
                for page in rendered_pages
                if page["page_number"] == page_number
            ),
            None
        )

        if rendered_page:

            page_record["page_image"] = {
                "path": rendered_page["image_path"],
                "modality": "page_image",
                "width": rendered_page["width"],
                "height": rendered_page["height"]
            }

        pages.append(page_record)

    manifest = {
        "document": pdf_path.name,
        "document_path": str(pdf_path),
        "total_pages": total_pages,
        "pages": pages
    }

    manifest_path = (
        output_dir
        / "document_manifest.json"
    )

    with open(
        manifest_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            manifest,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\n" + "=" * 60)
    print("UNIFIED MANIFEST CREATED")
    print("=" * 60)
    print(f"Document: {pdf_path.name}")
    print(f"Pages: {total_pages}")
    print(f"Manifest: {manifest_path}")

    return manifest