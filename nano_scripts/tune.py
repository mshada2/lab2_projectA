from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import retrieve
from eval import evaluate_run, load_query_file
from utils import PUBLIC_QUERIES_PATH


def eval_config(name: str, **kwargs):
    for key, value in kwargs.items():
        setattr(retrieve, key, value)
    retrieve._STATE = None
    rows = load_query_file(PUBLIC_QUERIES_PATH)
    queries = [r["query"] for r in rows]
    gt = [r["relevant_page_ids"] for r in rows]
    t0 = time.perf_counter()
    stats = evaluate_run(queries, gt, retrieve.search_batch)
    dt = time.perf_counter() - t0
    print(
        f"{name:24s} ndcg={stats['mean_ndcg@10']:.4f} time={dt:.2f}s "
        f"EW={retrieve.LITERAL_EVIDENCE_WEIGHT:.2f} BASE={retrieve.BASE_IN_RERANK:.2f} "
        f"RN={retrieve.RERANK_TOP_N} D={retrieve.DENSE_WEIGHT:.2f} B={retrieve.BM25_WEIGHT:.2f}"
    )


def main():
    original = {k: getattr(retrieve, k) for k in [
        "DENSE_WEIGHT", "BM25_WEIGHT", "DENSE_CANDIDATES", "BM25_CANDIDATES", "FINAL_POOL",
        "TITLE_MATCH_WEIGHT", "NUMBER_MATCH_WEIGHT", "RERANK_TOP_N", "BASE_IN_RERANK",
        "LITERAL_EVIDENCE_WEIGHT", "RARE_COVERAGE_WEIGHT", "EXACT_COVERAGE_WEIGHT", "PHRASE_WEIGHT",
        "NUMBER_EVIDENCE_WEIGHT", "RELATIVE_YEAR_WEIGHT", "SCORE_THRESHOLD",
    ]}
    configs = [
        ("v3_default", {}),
        ("no_rerank", {"BASE_IN_RERANK": 1.0, "LITERAL_EVIDENCE_WEIGHT": 0.0}),
        ("light_evidence", {"BASE_IN_RERANK": 0.88, "LITERAL_EVIDENCE_WEIGHT": 0.12}),
        ("medium_evidence", {"BASE_IN_RERANK": 0.78, "LITERAL_EVIDENCE_WEIGHT": 0.22}),
        ("strong_evidence", {"BASE_IN_RERANK": 0.68, "LITERAL_EVIDENCE_WEIGHT": 0.32}),
        ("rerank_150", {"RERANK_TOP_N": 150}),
        ("rerank_700", {"RERANK_TOP_N": 700}),
        ("bm25_70", {"DENSE_WEIGHT": 0.30, "BM25_WEIGHT": 0.70}),
        ("bm25_75", {"DENSE_WEIGHT": 0.25, "BM25_WEIGHT": 0.75}),
        ("no_title_aux", {"TITLE_MATCH_WEIGHT": 0.0, "NUMBER_MATCH_WEIGHT": 0.10}),
        ("no_aux", {"TITLE_MATCH_WEIGHT": 0.0, "NUMBER_MATCH_WEIGHT": 0.0}),
        ("phrase_heavy", {"PHRASE_WEIGHT": 0.34, "RARE_COVERAGE_WEIGHT": 0.36}),
        ("number_heavy", {"NUMBER_EVIDENCE_WEIGHT": 0.32, "RELATIVE_YEAR_WEIGHT": 0.22}),
    ]
    for name, cfg in configs:
        for k, v in original.items():
            setattr(retrieve, k, v)
        eval_config(name, **cfg)


if __name__ == "__main__":
    main()
