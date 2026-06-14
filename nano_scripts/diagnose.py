from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval import ndcg_at_k, load_query_file
from main import run
from utils import PUBLIC_QUERIES_PATH


def main():
    rows = load_query_file(PUBLIC_QUERIES_PATH)
    queries = [r["query"] for r in rows]
    gt = [r["relevant_page_ids"] for r in rows]
    t0 = time.perf_counter()
    ranked = run(queries)
    dt = time.perf_counter() - t0
    print(f"query_phase_time={dt:.2f}s\n")
    scores = []
    for i, (row, pred, rel) in enumerate(zip(rows, ranked, gt)):
        score = ndcg_at_k(pred, rel)
        scores.append(score)
        ranks = {}
        for pid in rel:
            pid = int(pid)
            ranks[pid] = pred.index(pid) + 1 if pid in pred else None
        print(f"{i:02d} {row['query_id']} ndcg={score:.4f}")
        print(f"query: {row['query']}")
        print(f"relevant_ranks: {ranks}")
        print(f"top10: {pred[:10]}\n")
    print(f"mean_ndcg@10={sum(scores)/len(scores):.4f}")


if __name__ == "__main__":
    main()
