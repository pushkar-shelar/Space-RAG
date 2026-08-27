import json
from pathlib import Path


def create_manifest(
    pdf_path,
    rendered_pages,
    output_dir="data/processed/metadata"
):
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "document": pdf_path.name,
        "document_path": str(pdf_path),
        "total_pages": len(rendered_pages),
        "pages": rendered_pages
    }

    manifest_path = output_dir / "document_manifest.json"

    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(
            manifest,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(f"Manifest created: {manifest_path}")

    return manifest_path


if __name__ == "__main__":

    from page_renderer import render_pdf_pages

    pdf_path = "data/corpus/GSFC-HDBK-8007_Admn Ext_1.pdf"

    rendered_pages = render_pdf_pages(pdf_path)

    create_manifest(
        pdf_path,
        rendered_pages
    )