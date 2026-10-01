from __future__ import annotations

from pathlib import Path

from src.indexing.visual_indexer import ColPaliPageIndex


class SourceVisualRetriever:
    def __init__(self, root: str | Path, source_type: str):
        self.root = Path(root)
        self.source_type = source_type
        self.index_path = (
            self.root / "colpali.pt"
            if source_type == "corpus"
            else self.root / "indexes" / "colpali.pt"
        )
        self.index = ColPaliPageIndex(self.index_path)

    def _ensure_private_index(self):
        if self.source_type != "private" or self.index_path.exists():
            return
        page_images = sorted((self.root / "pages").glob("*.png"))
        if not page_images:
            return
        metadata = []
        import re
        for path in page_images:
            match = re.search(r"_page_(\d+)", path.name)
            metadata.append({
                "path": str(path),
                "page_number": int(match.group(1)) if match else -1,
                "document": self.root.name,
            })
        self.index.build(page_images, metadata)

    def available(self) -> bool:
        if self.source_type == "private":
            self._ensure_private_index()
        return self.index_path.exists()

    def search(self, query: str, top_k: int = 10):
        self._ensure_private_index()
        results = self.index.search(query, top_k)
        for result in results:
            result["source_type"] = self.source_type
            result["source_id"] = "corpus" if self.source_type == "corpus" else self.root.name
        return results
