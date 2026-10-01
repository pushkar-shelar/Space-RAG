import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.workflow import _GRAPH, build_graph, run_space_rag
from src.generation.answer_generator import AnswerGenerator


def test_tasks_11_to_15():
    print("\n" + "=" * 70)
    print("VERIFICATION SUITE: Tasks 11 to 15 (Workflow, Qwen, Grok, Citations, UI)")
    print("=" * 70)

    # -----------------------------------------------------------------
    # TASK 11: LangGraph Workflow Architecture
    # -----------------------------------------------------------------
    print("\n--- [TASK 11] LangGraph Workflow Architecture ---")
    compiled_graph = build_graph()
    expected_nodes = {
        "understand_query",
        "retrieve_evidence",
        "assess_evidence",
        "verify_evidence",
        "generate_answer",
    }
    actual_nodes = set(compiled_graph.nodes.keys())
    assert expected_nodes.issubset(actual_nodes), f"Missing nodes: {expected_nodes - actual_nodes}"
    print(f"[PASS] Workflow contains all required nodes: {sorted(list(expected_nodes))}")

    # Test assess_evidence logic
    from src.agents.workflow import assess_evidence
    res_empty = assess_evidence({"retrieval_results": []})
    assert res_empty["sufficiency"] == "insufficient"
    res_full = assess_evidence({"retrieval_results": [{"text": "sample"}]})
    assert res_full["sufficiency"] == "sufficient"
    print("[PASS] Evidence sufficiency gating verified.")

    # -----------------------------------------------------------------
    # TASK 12: Qwen Offline Generation Formatting & Sanitization
    # -----------------------------------------------------------------
    print("\n--- [TASK 12] Qwen Offline Generation Core ---")
    generator = AnswerGenerator()
    assert generator.offline_model_name == "Qwen/Qwen2.5-1.5B-Instruct"

    # Test prompt synthesis
    mock_results = [
        {
            "source_type": "private",
            "document": "PrivateMission.pdf",
            "page_number": 4,
            "text": "The operational altitude of satellite Alpha is 550 km.",
        },
        {
            "source_type": "corpus",
            "document": "NASA_Corpus_Secret.pdf",
            "page_number": 99,
            "text": "Standard orbit inclination for constellation beta is 30 degrees.",
        },
    ]

    prompt_text, citations = generator._prompt("What is the altitude and inclination?", mock_results)
    assert "PRIVATE_EVIDENCE_1" in prompt_text
    assert "PrivateMission.pdf" in prompt_text
    assert "page=4" in prompt_text
    assert "CORPUS_EVIDENCE_2" in prompt_text
    # Corpus document title and page must NOT appear in evidence blocks
    assert "NASA_Corpus_Secret.pdf" not in prompt_text
    assert "page=99" not in prompt_text
    print("[PASS] Evidence blocks correctly formatted with corpus privacy isolation.")

    # Test output sanitization
    raw_answer = "According to CORPUS_EVIDENCE_2 and PRIVATE_EVIDENCE_1, the altitude is 550 km."
    clean = generator._sanitize_output(raw_answer)
    assert "CORPUS_EVIDENCE" not in clean
    assert "PRIVATE_EVIDENCE" not in clean
    assert "550 km" in clean
    print(f"[PASS] Sanitizer stripped internal labels: '{clean}'")

    # -----------------------------------------------------------------
    # TASK 13: Grok Online Generation Integration
    # -----------------------------------------------------------------
    print("\n--- [TASK 13] Grok Online Generation Integration ---")
    # Verify proper error message when API key is missing
    with patch.dict(os.environ, {}, clear=True):
        grok_gen = AnswerGenerator()
        try:
            grok_gen._load_grok()
            assert False, "Should have raised RuntimeError when XAI_API_KEY is missing"
        except RuntimeError as e:
            assert "XAI_API_KEY is not configured" in str(e)
            print("[PASS] Missing API key cleanly caught with descriptive error.")

    # Verify standard OpenAI chat completions endpoint structure
    with patch.dict(os.environ, {"XAI_API_KEY": "xai-test-key-mock"}):
        with patch("openai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_client.chat.completions.create.return_value.choices = [
                MagicMock(message=MagicMock(content="Mocked Grok response from space evidence."))
            ]

            grok_gen = AnswerGenerator()
            out = grok_gen.generate(
                query="Test query",
                retrieval_results=mock_results,
                model_mode="online",
            )
            assert out["model"] == "grok-2-latest"
            assert "Mocked Grok response" in out["answer"]
            # Verify chat.completions.create was called with system & user roles
            mock_client.chat.completions.create.assert_called_once()
            call_kwargs = mock_client.chat.completions.create.call_args[1]
            assert call_kwargs["model"] == "grok-2-latest"
            assert call_kwargs["messages"][0]["role"] == "system"
            assert call_kwargs["messages"][1]["role"] == "user"
            print("[PASS] Grok chat.completions.create verified with system & user prompts.")

    # -----------------------------------------------------------------
    # TASK 14: Source-Aware Citations Policy
    # -----------------------------------------------------------------
    print("\n--- [TASK 14] Source-Aware Citations Policy ---")
    assert len(citations) == 1
    assert citations[0] == "[PrivateMission.pdf, p. 4]"
    # Ensure corpus result has no citation
    for r in mock_results:
        if r["source_type"] == "corpus":
            assert generator._private_citation(r) is None
    print(f"[PASS] Citation extracted strictly for private document: {citations}")

    # Empty evidence handling
    empty_gen = generator.generate("What is X?", [], model_mode="offline")
    assert "sufficient evidence" in empty_gen["answer"].lower()
    assert empty_gen["citations"] == []
    print("[PASS] Insufficient evidence gracefully returns fallback answer without citations.")

    # -----------------------------------------------------------------
    # TASK 15: Streamlit UI File & Component Verification
    # -----------------------------------------------------------------
    print("\n--- [TASK 15] Streamlit UI Verification ---")
    app_file = PROJECT_ROOT / "app" / "streamlit_app.py"
    assert app_file.exists(), "streamlit_app.py not found!"

    # Verify syntax & imports in streamlit_app.py
    import py_compile
    py_compile.compile(str(app_file), doraise=True)
    print("[PASS] streamlit_app.py compiled successfully with zero syntax errors.")

    # Verify required UI logic tokens
    app_code = app_file.read_text(encoding="utf-8")
    assert "run_space_rag" in app_code
    assert "end_private_session" in app_code
    assert "cleanup_session" in app_code
    assert "build_corpus_index" in app_code
    assert "model_mode" in app_code
    assert "Private-document sources" in app_code
    assert "Referenced Document Visual Pages" in app_code
    print("[PASS] Streamlit UI contains all required session, retrieval, and citation controls.")

    print("\n" + "=" * 70)
    print("ALL TASKS 11 TO 15 VERIFIED AND PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    test_tasks_11_to_15()
