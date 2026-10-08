"""
Empirical Benchmark Evaluation for SpaceBot
Compares:
1. Baseline: Standard Dense-Only Retrieval (ChromaDB + bge-small)
2. Baseline: Pure Lexical Retrieval (BM25 only)
3. Proposed: SpaceBot (Hybrid BM25 + Dense Max-Signal Fusion + ColPali Visual Retrieval)

Measures:
- Hit Rate@3 (Overall)
- Hit Rate@3 (Tabular Sub-category)
- Hit Rate@3 (Visual Sub-category)
- Hit Rate@3 (Prose Sub-category)
- MRR@3 (Mean Reciprocal Rank)
- Average Retrieval Latency (s)
"""

import json
import time
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.source_text_retriever import SourceTextRetriever
from src.retrieval.source_visual_retriever import SourceVisualRetriever
from src.retrieval.multimodal_retriever import SourceAwareMultimodalRetriever
from src.ingestion.session_manager import create_session, cleanup_session
from src.ingestion.pipeline import process_document
from src.indexing.index_document import index_private_session

BENCHMARK_QUERIES = [
    # --- Category A: Structured Tabular Telemetry & Numerical Data ---
    {
        "id": "T1",
        "category": "Tabular",
        "query": "what was the Apogee of the second Earth Bound Manoeuvre (EBN #2)?",
        "target_doc": "publication_6_.pdf",
        "target_pages": [14, 15],
        "expected_value": "74715",
    },
    {
        "id": "T2",
        "category": "Tabular",
        "query": "what was the Apogee of the fifth Earth Bound Manoeuvre (EBN #5)?",
        "target_doc": "publication_6_.pdf",
        "target_pages": [14, 16],
        "expected_value": "379454",
    },
    {
        "id": "T3",
        "category": "Tabular",
        "query": "what was the Apogee of the third Earth Bound Manoeuvre (EBN #3)?",
        "target_doc": "publication_6_.pdf",
        "target_pages": [14, 16],
        "expected_value": "165016",
    },
    {
        "id": "T4",
        "category": "Tabular",
        "query": "delta-V realized for EBN #1 maneuver",
        "target_doc": "publication_6_.pdf",
        "target_pages": [14],
        "expected_value": "343.19",
    },
    {
        "id": "T5",
        "category": "Tabular",
        "query": "duration of EBN #4 orbit-raising burn",
        "target_doc": "publication_6_.pdf",
        "target_pages": [14],
        "expected_value": "192.18",
    },

    # --- Category B: Prose / Mission Specifications ---
    {
        "id": "P1",
        "category": "Prose",
        "query": "What is the primary overarching science goal of the TROPICS mission?",
        "target_doc": "TROPICS.pdf",
        "target_pages": [2, 3],
        "expected_value": "tropical cyclones",
    },
    {
        "id": "P2",
        "category": "Prose",
        "query": "What is the operational orbital inclination for TROPICS constellation satellites?",
        "target_doc": "TROPICS.pdf",
        "target_pages": [3, 4],
        "expected_value": "inclination",
    },
    {
        "id": "P3",
        "category": "Prose",
        "query": "What is the primary purpose and scope of the Goddard Space Flight Center handbook?",
        "target_doc": "GSFC-HDBK-8007_Admn.pdf",
        "target_pages": [1, 2],
        "expected_value": "handbook",
    },
    {
        "id": "P4",
        "category": "Prose",
        "query": "When did Chandrayaan-1 first enter deep space after crossing 150000 km?",
        "target_doc": "publication_6_.pdf",
        "target_pages": [15, 16],
        "expected_value": "October 26",
    },
    {
        "id": "P5",
        "category": "Prose",
        "query": "What liquid engine Newton thrust was fired for Chandrayaan-1 orbit-raising maneuvers?",
        "target_doc": "publication_6_.pdf",
        "target_pages": [15],
        "expected_value": "440 Newton",
    },

    # --- Category C: Visual / Multimodal Diagrams & Figures ---
    {
        "id": "V1",
        "category": "Visual",
        "query": "diagram showing the TROPICS constellation orbital plane architecture and spatial sampling",
        "target_doc": "TROPICS.pdf",
        "target_pages": [3, 4],
        "expected_value": "figure",
    },
    {
        "id": "V2",
        "category": "Visual",
        "query": "graph showing temperature weighting functions and sensor channel frequencies",
        "target_doc": "TROPICS.pdf",
        "target_pages": [7],
        "expected_value": "weighting",
    },
    {
        "id": "V3",
        "category": "Visual",
        "query": "flowchart illustrating Goddard administration workflow hierarchy and management process",
        "target_doc": "GSFC-HDBK-8007_Admn.pdf",
        "target_pages": [5, 6],
        "expected_value": "chart",
    },
    {
        "id": "V4",
        "category": "Visual",
        "query": "orbital trajectory diagram showing Earth Bound maneuvers to lunar insertion",
        "target_doc": "publication_6_.pdf",
        "target_pages": [13, 14],
        "expected_value": "orbit",
    },
    {
        "id": "V5",
        "category": "Visual",
        "query": "payload layout diagram showing microwave radiometer sensor deployment",
        "target_doc": "TROPICS.pdf",
        "target_pages": [5, 6],
        "expected_value": "radiometer",
    },
]


def run_benchmark():
    print("=" * 70)
    print("RUNNING EMPIRICAL BENCHMARK EVALUATION FOR SPACEBOT")
    print("=" * 70)

    # 1. Setup session with publication_6_.pdf
    pdf_source = Path(r"C:\Users\Pushkar Shelar\.gemini\antigravity-ide\brain\bf7021f0-e387-46ab-8c6e-340568fc3990\.user_uploaded\media_1790941417366.pdf")
    if not pdf_source.exists():
        raise FileNotFoundError(f"PDF source not found: {pdf_source}")

    session = create_session("publication_6_.pdf", pdf_source.read_bytes())
    session_id = session["session_id"]
    session_root = Path(session["session_root"])
    corpus_root = PROJECT_ROOT / "data" / "corpus_index"

    print(f"[*] Ingesting and indexing private session at: {session_root}")
    process_document(session["pdf_path"], session_root, session_id=session_id)
    index_private_session(session_root)

    # Load retrievers
    private_text = SourceTextRetriever(session_root, "private")
    corpus_text = SourceTextRetriever(corpus_root, "corpus")
    spacebot = SourceAwareMultimodalRetriever(corpus_root=corpus_root, session_root=session_root)
    # Use corpus pre-indexed ColPali for visual queries during evaluation to prevent CPU memory thrashing
    spacebot.private_visual = None

    results_dense = []
    results_bm25 = []
    results_spacebot = []

    print("\n[*] Evaluating queries across pipelines...")

    for item in BENCHMARK_QUERIES:
        q_id = item["id"]
        cat = item["category"]
        query = item["query"]
        target_pages = item["target_pages"]
        expected_val = item["expected_value"].lower()

        # ----------------------------------------------------
        # 1. Baseline: Dense Only
        # ----------------------------------------------------
        t0 = time.time()
        dense_hits = []
        if private_text.chroma:
            for d in private_text.chroma.search(query, top_k=5):
                meta = d.get("metadata", {})
                dense_hits.append((meta.get("page_number"), d.get("text", "")))
        if corpus_text.chroma:
            for d in corpus_text.chroma.search(query, top_k=5):
                meta = d.get("metadata", {})
                dense_hits.append((meta.get("page_number"), d.get("text", "")))
        dense_top3 = dense_hits[:3]
        dt_dense = time.time() - t0

        dense_hit = any(p in target_pages for p, txt in dense_top3)
        dense_rr = 0.0
        for rank, (p, txt) in enumerate(dense_top3, start=1):
            if p in target_pages:
                dense_rr = 1.0 / rank
                break
        dense_num_match = any(expected_val in txt.lower() for p, txt in dense_top3)

        results_dense.append({
            "id": q_id,
            "category": cat,
            "hit": dense_hit,
            "rr": dense_rr,
            "num_match": dense_num_match,
            "latency": dt_dense,
        })

        # ----------------------------------------------------
        # 2. Baseline: BM25 Only
        # ----------------------------------------------------
        t0 = time.time()
        bm25_hits = []
        if private_text.bm25:
            for d in private_text.bm25.search(query, top_k=5):
                meta = d.get("metadata", {})
                bm25_hits.append((meta.get("page_number"), d.get("text", "")))
        if corpus_text.bm25:
            for d in corpus_text.bm25.search(query, top_k=5):
                meta = d.get("metadata", {})
                bm25_hits.append((meta.get("page_number"), d.get("text", "")))
        bm25_top3 = bm25_hits[:3]
        dt_bm25 = time.time() - t0

        bm25_hit = any(p in target_pages for p, txt in bm25_top3)
        bm25_rr = 0.0
        for rank, (p, txt) in enumerate(bm25_top3, start=1):
            if p in target_pages:
                bm25_rr = 1.0 / rank
                break
        bm25_num_match = any(expected_val in txt.lower() for p, txt in bm25_top3)

        results_bm25.append({
            "id": q_id,
            "category": cat,
            "hit": bm25_hit,
            "rr": bm25_rr,
            "num_match": bm25_num_match,
            "latency": dt_bm25,
        })

        # ----------------------------------------------------
        # 3. Proposed: SpaceBot
        # ----------------------------------------------------
        t0 = time.time()
        sb_res = spacebot.retrieve(query, top_k=3, retrieval_k=6)
        dt_sb = time.time() - t0

        sb_top3 = sb_res["results"]
        sb_hit = any(r.get("page_number") in target_pages for r in sb_top3)
        sb_rr = 0.0
        for rank, r in enumerate(sb_top3, start=1):
            if r.get("page_number") in target_pages:
                sb_rr = 1.0 / rank
                break
        sb_num_match = any(expected_val in (r.get("text") or "").lower() for r in sb_top3)

        results_spacebot.append({
            "id": q_id,
            "category": cat,
            "hit": sb_hit,
            "rr": sb_rr,
            "num_match": sb_num_match,
            "latency": dt_sb,
        })

        print(f"[{q_id} - {cat:<7}] Dense Hit: {dense_hit} | BM25 Hit: {bm25_hit} | SpaceBot Hit: {sb_hit}")

    # Compute Aggregate Metrics
    def compute_stats(res_list):
        total = len(res_list)
        hit_rate = sum(x["hit"] for x in res_list) / total * 100.0
        mrr = sum(x["rr"] for x in res_list) / total
        avg_lat = sum(x["latency"] for x in res_list) / total

        # By category
        by_cat = {}
        for cat in ["Tabular", "Prose", "Visual"]:
            cat_items = [x for x in res_list if x["category"] == cat]
            c_tot = len(cat_items)
            c_hit = sum(x["hit"] for x in cat_items) / c_tot * 100.0 if c_tot else 0.0
            c_num = sum(x["num_match"] for x in cat_items) / c_tot * 100.0 if c_tot else 0.0
            by_cat[cat] = {"hit_rate": c_hit, "num_accuracy": c_num}

        return {
            "overall_hit_rate": hit_rate,
            "mrr": mrr,
            "avg_latency": avg_lat,
            "categories": by_cat,
        }

    stats_dense = compute_stats(results_dense)
    stats_bm25 = compute_stats(results_bm25)
    stats_sb = compute_stats(results_spacebot)

    final_report = {
        "Dense_Only_Baseline": stats_dense,
        "BM25_Only_Baseline": stats_bm25,
        "SpaceBot_Proposed": stats_sb,
    }

    out_path = PROJECT_ROOT / "data" / "benchmark_evaluation_results.json"
    out_path.write_text(json.dumps(final_report, indent=2), encoding="utf-8")
    print(f"\n[+] Results successfully written to {out_path}")

    # Cleanup
    spacebot.close()
    cleanup_session(session_id)

    print("\n" + "=" * 70)
    print("FINAL EMPIRICAL RESULTS SUMMARY (FOR IEEE RESEARCH PAPER)")
    print("=" * 70)
    print(f"{'Metric':<28} | {'Dense Baseline':<15} | {'BM25 Baseline':<15} | {'SpaceBot (Proposed)':<18}")
    print("-" * 84)
    print(f"{'Overall Hit Rate@3 (%)':<28} | {stats_dense['overall_hit_rate']:<15.1f} | {stats_bm25['overall_hit_rate']:<15.1f} | {stats_sb['overall_hit_rate']:<18.1f}")
    print(f"{'Tabular Hit Rate@3 (%)':<28} | {stats_dense['categories']['Tabular']['hit_rate']:<15.1f} | {stats_bm25['categories']['Tabular']['hit_rate']:<15.1f} | {stats_sb['categories']['Tabular']['hit_rate']:<18.1f}")
    print(f"{'Visual Hit Rate@3 (%)':<28} | {stats_dense['categories']['Visual']['hit_rate']:<15.1f} | {stats_bm25['categories']['Visual']['hit_rate']:<15.1f} | {stats_sb['categories']['Visual']['hit_rate']:<18.1f}")
    print(f"{'Prose Hit Rate@3 (%)':<28} | {stats_dense['categories']['Prose']['hit_rate']:<15.1f} | {stats_bm25['categories']['Prose']['hit_rate']:<15.1f} | {stats_sb['categories']['Prose']['hit_rate']:<18.1f}")
    print(f"{'MRR@3 (Rank Quality)':<28} | {stats_dense['mrr']:<15.3f} | {stats_bm25['mrr']:<15.3f} | {stats_sb['mrr']:<18.3f}")
    print(f"{'Average Latency (s)':<28} | {stats_dense['avg_latency']:<15.2f} | {stats_bm25['avg_latency']:<15.2f} | {stats_sb['avg_latency']:<18.2f}")
    print("=" * 84)


if __name__ == "__main__":
    run_benchmark()
