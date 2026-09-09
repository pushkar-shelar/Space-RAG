import torch
from pathlib import Path

from colpali_embedder import ColPaliEmbedder


def embed_all_pages():

    pages_dir = Path(
        "data/processed/pages"
    )

    embeddings_dir = Path(
        "data/processed/embeddings/TROPICS"
    )

    embeddings_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    page_images = sorted(
        pages_dir.glob("TROPICS_page_*.png")
    )

    print("=" * 60)
    print("SPACE RAG - BATCH COLPALI EMBEDDING")
    print("=" * 60)

    print(
        f"Pages found: {len(page_images)}"
    )

    embedder = ColPaliEmbedder()

    for page_image in page_images:

        embedding_filename = (
            page_image.stem + ".pt"
        )

        embedding_path = (
            embeddings_dir /
            embedding_filename
        )

        if embedding_path.exists():

            print(
                f"Skipping {page_image.name} "
                f"(already embedded)"
            )

            continue

        embedding = embedder.embed_image(
            page_image
        )

        torch.save(
            embedding.cpu(),
            embedding_path
        )

        print(
            f"Saved: {embedding_path}"
        )

        print(
            f"Shape: {embedding.shape}"
        )

    print("\n" + "=" * 60)
    print("BATCH EMBEDDING COMPLETE")
    print("=" * 60)

    saved_embeddings = list(
        embeddings_dir.glob("*.pt")
    )

    print(
        f"Embeddings saved: "
        f"{len(saved_embeddings)}"
    )


if __name__ == "__main__":

    embed_all_pages()