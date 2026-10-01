from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class QueryInfo:
    modality: str
    page_reference: int | None = None


class QueryModalityDetector:
    VISUAL = {
        "image", "picture", "photo", "figure", "fig", "graph", "plot",
        "chart", "diagram", "map", "illustration", "shown", "visual",
        "trend", "curve", "bar", "axis", "table", "caption",
    }
    TEXT = {
        "define", "definition", "explain", "describe", "what", "why", "how",
        "meaning", "purpose", "objective", "method", "architecture", "process",
        "difference", "compare", "summary", "summarize",
    }
    MULTIMODAL = {
        "according to the figure", "according to the graph", "from the graph",
        "based on the graph", "explain the table", "what does the figure",
        "what does the graph", "compare the text and figure", "compare the table and text",
    }

    def detect(self, query: str) -> QueryInfo:
        q = query.lower().strip()
        page_match = re.search(r"\bpage\s+(\d+)\b", q)
        page_reference = int(page_match.group(1)) if page_match else None

        if any(phrase in q for phrase in self.MULTIMODAL):
            return QueryInfo("multimodal", page_reference)

        tokens = set(re.findall(r"[a-z0-9]+", q))
        visual_hit = bool(tokens & self.VISUAL)
        text_hit = bool(tokens & self.TEXT)

        if visual_hit and text_hit:
            modality = "multimodal"
        elif visual_hit:
            modality = "visual"
        else:
            modality = "text"

        return QueryInfo(modality, page_reference)
