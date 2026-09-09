import pymupdf
from pathlib import Path


def extract_text(pdf_path):
    pdf_path = Path(pdf_path)

    output_dir = (
        Path("data")
        / "processed"
        / "text"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    document = pymupdf.open(pdf_path)

    pages = []

    print(f"Document: {pdf_path.name}")
    print(f"Pages: {len(document)}")

    for page_number, page in enumerate(document):

        page_number_display = page_number + 1

        text = page.get_text()

        text_filename = (
            f"{pdf_path.stem}"
            f"_page_{page_number_display}.txt"
        )

        text_path = output_dir / text_filename

        with open(
            text_path,
            "w",
            encoding="utf-8"
        ) as text_file:

            text_file.write(text)

        page_data = {
            "document": pdf_path.name,
            "page_number": page_number_display,
            "text_path": str(text_path),
            "char_count": len(text),
            "modality": "text"
        }

        pages.append(page_data)

        print(
            f"Page {page_number_display}: "
            f"{len(text)} characters → "
            f"{text_filename}"
        )

    document.close()

    print("\n" + "=" * 60)
    print("TEXT EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Total pages: {len(pages)}")

    return pages


if __name__ == "__main__":

    extract_text(
        r"data\corpus\GSFC-HDBK-8007_Admn Ext_1.pdf"
    )