import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.indexing.colpali_embedder import ColPaliEmbedder


def embed_all_pages(
    pages_dir="data/processed/pages",
    embeddings_dir="data/processed/embeddings/pages"
):

    pages_dir = Path(pages_dir)
    embeddings_dir = Path(embeddings_dir)

    embeddings_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    page_images = sorted(
        pages_dir.glob("*.png")
    )

    print("=" * 60)
    print("SPACE RAG - BATCH COLPALI EMBEDDING")
    print("=" * 60)
    print(f"Pages found: {len(page_images)}")

    if not page_images:
        print("\nNo rendered page images found.")
        return

    embedder = ColPaliEmbedder()

    saved_count = 0
    skipped_count = 0

    for page_image in page_images:

        embedding_path = (
            embeddings_dir
            / f"{page_image.stem}.pt"
        )

        if embedding_path.exists():
            print(
                f"Skipping {page_image.name} "
                "(already embedded)"
            )
            skipped_count += 1
            continue

        embedding = embedder.embed_image(page_image)

        torch.save(
            embedding.cpu(),
            embedding_path
        )

        saved_count += 1

        print(
            f"Saved: {embedding_path}"
        )
        print(
            f"Shape: {embedding.shape}"
        )

    print("\n" + "=" * 60)
    print("BATCH EMBEDDING COMPLETE")
    print("=" * 60)
    print(f"New embeddings: {saved_count}")
    print(f"Skipped existing: {skipped_count}")
    print(
        f"Total embeddings: "
        f"{len(list(embeddings_dir.glob('*.pt')))}"
    )


if __name__ == "__main__":
    embed_all_pages()
