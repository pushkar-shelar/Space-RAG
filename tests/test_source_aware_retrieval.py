import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.query_modality import QueryModalityDetector, QueryInfo
from src.generation.answer_generator import AnswerGenerator


def test_query_modality_detector():
    print("\n" + "=" * 60)
    print("TEST: Query Modality Detection")
    print("=" * 60)

    detector = QueryModalityDetector()

    # Text query
    q_text = "What is the mission objective of TROPICS?"
    info_text = detector.detect(q_text)
    assert isinstance(info_text, QueryInfo)
    assert info_text.modality == "text"
    print(f"'{q_text}' -> modality={info_text.modality}, page={info_text.page_reference}")

    # Visual query with page reference
    q_vis = "Look at the figure on page 7"
    info_vis = detector.detect(q_vis)
    assert info_vis.modality == "visual"
    assert info_vis.page_reference == 7
    print(f"'{q_vis}' -> modality={info_vis.modality}, page={info_vis.page_reference}")

    # Multimodal query
    q_multi = "Explain the architecture from the graph on page 3"
    info_multi = detector.detect(q_multi)
    assert info_multi.modality == "multimodal"
    assert info_multi.page_reference == 3
    print(f"'{q_multi}' -> modality={info_multi.modality}, page={info_multi.page_reference}")

    print("Query Modality Detection: PASS")


def test_citation_policy():
    print("\n" + "=" * 60)
    print("TEST: Source-Aware Citation Policy")
    print("=" * 60)

    generator = AnswerGenerator()

    # Private source MUST produce transparent citations
    private_result = {
        "source_type": "private",
        "document": "my_satellite_report.pdf",
        "page_number": 4,
        "text": "The solar panel efficiency is 28.5%.",
    }
    citation = generator._private_citation(private_result)
    assert citation == "[my_satellite_report.pdf, p. 4]"
    print(f"Private evidence citation: {citation} (Expected: [my_satellite_report.pdf, p. 4])")

    # Corpus source MUST NOT produce a private citation (provenance masked)
    corpus_result = {
        "source_type": "corpus",
        "document": "TROPICS.pdf",
        "page_number": 1,
        "text": "TROPICS is a constellation of state-of-the-art CubeSats.",
    }
    corpus_citation = generator._private_citation(corpus_result)
    assert corpus_citation is None, "Corpus result generated citation when it should be masked!"
    print("Corpus evidence citation masked: PASS")

    print("\n" + "=" * 60)
    print("TEST RESULT: PASS")
    print("=" * 60)


if __name__ == "__main__":
    test_query_modality_detector()
    test_citation_policy()
