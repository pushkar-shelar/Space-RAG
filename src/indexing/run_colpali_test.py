import torch
from PIL import Image

from colpali_engine.models import ColPali, ColPaliProcessor


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_NAME = "vidore/colpali-v1.3"

IMAGE_PATH = (
    "data/processed/pages/"
    "GSFC-HDBK-8007_Admn Ext_1_page_001.png"
)


# --------------------------------------------------
# Select device
# --------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("SPACE RAG - COLPALI TEST")
print("=" * 60)

print(f"Device: {device}")
print(f"Model: {MODEL_NAME}")
print(f"Image: {IMAGE_PATH}")


# --------------------------------------------------
# Load ColPali model
# --------------------------------------------------

print("\nLoading ColPali model...")

model = ColPali.from_pretrained(
    MODEL_NAME,
    device_map=device
)

model.eval()

print("ColPali model loaded successfully!")


# --------------------------------------------------
# Load processor
# --------------------------------------------------

print("\nLoading ColPali processor...")

processor = ColPaliProcessor.from_pretrained(
    MODEL_NAME
)

print("ColPali processor loaded successfully!")


# --------------------------------------------------
# Load page image
# --------------------------------------------------

print("\nLoading page image...")

image = Image.open(
    IMAGE_PATH
).convert("RGB")

print(f"Image size: {image.size}")


# --------------------------------------------------
# Prepare image
# --------------------------------------------------

print("\nPreparing image for ColPali...")

batch_images = processor.process_images(
    [image]
)

batch_images = {
    key: value.to(device)
    for key, value in batch_images.items()
}


# --------------------------------------------------
# Generate visual embeddings
# --------------------------------------------------

print("\nGenerating ColPali embedding...")

with torch.no_grad():

    image_embeddings = model(
        **batch_images
    )


# --------------------------------------------------
# Display result
# --------------------------------------------------

print("\n")
print("=" * 60)
print("COLPALI EMBEDDING GENERATED")
print("=" * 60)

print(f"Embedding shape: {image_embeddings.shape}")
print(f"Embedding dtype: {image_embeddings.dtype}")
print(f"Embedding device: {image_embeddings.device}")