from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import torch
from PIL import Image

from colpali_engine.models import ColPali, ColPaliProcessor


_SHARED_COLPALI_MODEL = None
_SHARED_COLPALI_PROCESSOR = None
_SHARED_MODEL_KEY = None


def get_shared_colpali(model_name: str = "vidore/colpali-v1.3", device: str | None = None):
    global _SHARED_COLPALI_MODEL, _SHARED_COLPALI_PROCESSOR, _SHARED_MODEL_KEY
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    key = (model_name, dev)
    if _SHARED_COLPALI_MODEL is None or _SHARED_MODEL_KEY != key:
        dtype = torch.bfloat16 if dev != "cpu" else torch.float32
        _SHARED_COLPALI_MODEL = ColPali.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=dev,
        ).eval()
        _SHARED_COLPALI_PROCESSOR = ColPaliProcessor.from_pretrained(model_name)
        _SHARED_MODEL_KEY = key
    return _SHARED_COLPALI_MODEL, _SHARED_COLPALI_PROCESSOR


def release_shared_colpali():
    global _SHARED_COLPALI_MODEL, _SHARED_COLPALI_PROCESSOR, _SHARED_MODEL_KEY
    _SHARED_COLPALI_MODEL = None
    _SHARED_COLPALI_PROCESSOR = None
    _SHARED_MODEL_KEY = None
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class ColPaliPageIndex:
    """Stores per-page ColPali multi-vector embeddings for one source."""

    def __init__(
        self,
        index_path: str | Path,
        model_name: str = "vidore/colpali-v1.3",
        device: str | None = None,
    ):
        self.index_path = Path(index_path)
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None

    def _load_model(self):
        if self.model is None or self.processor is None:
            self.model, self.processor = get_shared_colpali(self.model_name, self.device)

    def build(
        self,
        page_images: Iterable[str | Path],
        metadata: Iterable[dict],
        batch_size: int = 2,
    ) -> int:
        page_images = [Path(p) for p in page_images]
        metadata = list(metadata)
        if not page_images:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"items": []}, self.index_path)
            return 0

        self._load_model()
        items = []

        for start in range(0, len(page_images), batch_size):
            batch_paths = page_images[start:start + batch_size]
            images = [Image.open(path).convert("RGB") for path in batch_paths]
            batch = self.processor.process_images(images).to(self.model.device)
            with torch.no_grad():
                embeddings = self.model(**batch).detach().cpu()

            for path, emb in zip(batch_paths, embeddings):
                item_meta = next(
                    (m for m in metadata if str(m["path"]) == str(path)),
                    None,
                )
                items.append(
                    {
                        "path": str(path),
                        "embedding": emb,
                        "metadata": item_meta or {"path": str(path)},
                    }
                )

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"items": items}, self.index_path)
        return len(items)

    def load_items(self):
        if not self.index_path.exists():
            return []
        data = torch.load(self.index_path, map_location="cpu", weights_only=False)
        return data.get("items", [])

    def search(self, query: str, top_k: int = 10) -> List[dict]:
        items = self.load_items()
        if not items:
            return []

        self._load_model()
        batch_queries = self.processor.process_queries([query]).to(self.model.device)
        with torch.no_grad():
            query_embedding = self.model(**batch_queries).detach().cpu()

        doc_embeddings = [item["embedding"] for item in items]
        scores = self.processor.score_multi_vector(
            query_embedding,
            doc_embeddings,
        )
        scores = scores.detach().cpu().flatten().tolist()

        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        output = []
        for i in order:
            item = items[i]
            output.append(
                {
                    "visual_score": float(scores[i]),
                    "path": item["path"],
                    "metadata": item["metadata"],
                }
            )
        return output
