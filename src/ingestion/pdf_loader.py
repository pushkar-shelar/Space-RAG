import pymupdf
from pathlib import Path


def load_pdf(pdf_path):
    pdf_path = Path(pdf_path)

    image_output_dir = (
        Path("data")
        / "processed"
        / "images"
    )

    image_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    document = pymupdf.open(pdf_path)

    pages = []

    print(f"Document: {pdf_path.name}")
    print(f"Pages: {len(document)}")

    for page_number, page in enumerate(document):

        page_number_display = page_number + 1

        images = page.get_images(full=True)

        page_data = {
            "document": pdf_path.name,
            "page_number": page_number_display,
            "images": []
        }

        print("\n" + "=" * 60)
        print(f"PAGE {page_number_display}")
        print("=" * 60)
        print(f"Images found: {len(images)}")

        for image_number, image in enumerate(images):

            xref = image[0]

            image_data = document.extract_image(xref)

            image_bytes = image_data["image"]
            image_extension = image_data["ext"]

            image_filename = (
                f"{pdf_path.stem}"
                f"_page_{page_number_display}"
                f"_img_{image_number + 1}"
                f".{image_extension}"
            )

            image_path = (
                image_output_dir
                / image_filename
            )

            with open(
                image_path,
                "wb"
            ) as image_file:

                image_file.write(image_bytes)

            image_info = {
                "image_id": image_number + 1,
                "path": str(image_path),
                "extension": image_extension,
                "xref": xref,
                "modality": "image"
            }

            page_data["images"].append(
                image_info
            )

            print(
                f"  Image {image_number + 1}: "
                f"{image_filename}"
            )

        pages.append(page_data)

    document.close()

    print("\n" + "=" * 60)
    print("IMAGE EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Total pages processed: {len(pages)}")

    total_images = sum(
        len(page["images"])
        for page in pages
    )

    print(f"Total images extracted: {total_images}")

    return pages


if __name__ == "__main__":

    pages = load_pdf(
        r"data\corpus\GSFC-HDBK-8007_Admn Ext_1.pdf"
    )