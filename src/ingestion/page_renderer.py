import pymupdf
from pathlib import Path


def render_pdf_pages(pdf_path, output_dir="data/processed/pages"):
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    document = pymupdf.open(pdf_path)

    rendered_pages = []

    print(f"Document: {pdf_path.name}")
    print(f"Pages: {len(document)}")

    for page_number, page in enumerate(document):

        page_number_display = page_number + 1

        pixmap = page.get_pixmap(
            matrix=pymupdf.Matrix(2, 2),
            alpha=False
        )

        image_filename = (
            f"{pdf_path.stem}"
            f"_page_{page_number_display:03d}"
            f".png"
        )

        image_path = output_dir / image_filename

        pixmap.save(image_path)

        page_info = {
            "document": pdf_path.name,
            "page_number": page_number_display,
            "image_path": str(image_path),
            "modality": "page_image",
            "width": pixmap.width,
            "height": pixmap.height,
        }

        rendered_pages.append(page_info)

        print(
            f"Rendered page {page_number_display}: "
            f"{image_filename}"
        )

    document.close()

    return rendered_pages


if __name__ == "__main__":

    pages = render_pdf_pages(
        r"C:\Users\Pushkar Shelar\Desktop\Space RAG\data\corpus\GSFC-HDBK-8007_Admn Ext_1.pdf"
    )

    print("\n")
    print("=" * 60)
    print("PAGE RENDERING COMPLETE")
    print("=" * 60)
    print(f"Total pages rendered: {len(pages)}") 