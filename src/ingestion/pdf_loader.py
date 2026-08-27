import pymupdf
from pathlib import Path


def load_pdf(pdf_path):
    pdf_path = Path(pdf_path)

    document = pymupdf.open(pdf_path)

    pages = []

    print(f"Document: {pdf_path.name}")
    print(f"Pages: {len(document)}")

    for page_number, page in enumerate(document):

        text = page.get_text()

        page_data = {
            "document": pdf_path.name,
            "page_number": page_number + 1,
            "text": text,
            "char_count": len(text),
            "images": [],
        }

        images = page.get_images(full=True)

        print("\n" + "=" * 60)
        print(f"PAGE {page_number + 1}")
        print("=" * 60)
        print(f"Characters: {len(text)}")
        print(f"Images found: {len(images)}")

        for image_number, image in enumerate(images):

            xref = image[0]

            image_data = document.extract_image(xref)

            image_bytes = image_data["image"]
            image_extension = image_data["ext"]

            image_filename = (
                f"{pdf_path.stem}"
                f"_page_{page_number + 1}"
                f"_img_{image_number + 1}"
                f".{image_extension}"
            )

            image_path = (
                Path("data")
                / "processed"
                / "images"
                / image_filename
            )

            with open(image_path, "wb") as image_file:
                image_file.write(image_bytes)

            image_info = {
                "image_id": image_number + 1,
                "path": str(image_path),
                "extension": image_extension,
                "xref": xref,
                "modality": "image",
            }

            page_data["images"].append(image_info)

            print(
                f"  Image {image_number + 1}: "
                f"{image_filename}"
            )

        pages.append(page_data)

    document.close()

    return pages


if __name__ == "__main__":

    pages = load_pdf(
        r"C:\Users\Pushkar Shelar\Desktop\Space RAG\data\corpus\GSFC-HDBK-8007_Admn Ext_1.pdf"
    )

    print("\n")
    print("=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Total pages extracted: {len(pages)}")

    total_images = sum(
        len(page["images"])
        for page in pages
    )

    print(f"Total images extracted: {total_images}")