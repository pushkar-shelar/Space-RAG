import torch
from PIL import Image
from pathlib import Path

from colpali_engine.models import ColPali, ColPaliProcessor


class ColPaliEmbedder:

    def __init__(
        self,
        model_name="vidore/colpali-v1.3"
    ):

        self.model_name = model_name

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(f"Using device: {self.device}")

        print("Loading ColPali model...")

        self.model = ColPali.from_pretrained(
            self.model_name,
            device_map=self.device
        )

        self.model.eval()

        print("ColPali model loaded.")

        print("Loading ColPali processor...")

        self.processor = (
            ColPaliProcessor.from_pretrained(
                self.model_name
            )
        )

        print("ColPali processor loaded.")

    def embed_image(self, image_path):

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        print(
            f"Embedding page: "
            f"{image_path.name}"
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        batch = self.processor.process_images(
            [image]
        )

        batch = {
            key: value.to(self.device)
            for key, value in batch.items()
        }

        with torch.no_grad():

            embeddings = self.model(
                **batch
            )

        return embeddings


if __name__ == "__main__":

    image_path = (
        "data/processed/pages/"
        "TROPICS_page_007.png"
    )

    embedder = ColPaliEmbedder()

    embedding = embedder.embed_image(
        image_path
    )

    print("\n" + "=" * 60)
    print("EMBEDDING TEST COMPLETE")
    print("=" * 60)

    print(f"Shape: {embedding.shape}")
    print(f"Dtype: {embedding.dtype}")
    print(f"Device: {embedding.device}")