import sys
from pathlib import Path

INGESTION_DIR = Path(__file__).resolve().parent

sys.path.insert(
    0,
    str(INGESTION_DIR)
)

from pdf_loader import load_pdf
from text_extractor import extract_text
from page_renderer import render_pdf_pages
from manifest_builder import build_manifest

def process_document(pdf_path):

    pdf_path = Path(pdf_path)

    print("\n")
    print("=" * 70)
    print("SPACE RAG - DOCUMENT INGESTION PIPELINE")
    print("=" * 70)
    print(f"Document: {pdf_path.name}")
    print("=" * 70)

    # --------------------------------------------------
    # STEP 1: Extract embedded images
    # --------------------------------------------------

    print("\n[1/4] Extracting embedded images...")

    image_pages = load_pdf(pdf_path)

    # --------------------------------------------------
    # STEP 2: Extract page text
    # --------------------------------------------------

    print("\n[2/4] Extracting page text...")

    text_pages = extract_text(pdf_path)

    # --------------------------------------------------
    # STEP 3: Render complete PDF pages
    # --------------------------------------------------

    print("\n[3/4] Rendering PDF pages...")

    rendered_pages = render_pdf_pages(pdf_path)

    # --------------------------------------------------
    # STEP 4: Build unified manifest
    # --------------------------------------------------

    print("\n[4/4] Building unified document manifest...")

    manifest = build_manifest(
        pdf_path=pdf_path,
        text_pages=text_pages,
        image_pages=image_pages,
        rendered_pages=rendered_pages
    )

    print("\n")
    print("=" * 70)
    print("SPACE RAG INGESTION COMPLETE")
    print("=" * 70)

    print(f"Document: {pdf_path.name}")
    print(f"Pages: {manifest['total_pages']}")
    print("=" * 70)

    return manifest


if __name__ == "__main__":

    process_document(
        r"C:\Users\Pushkar Shelar\Desktop\Space RAG\data\corpus\TROPICS.pdf"
    )