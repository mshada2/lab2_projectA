#!/usr/bin/env python3
"""Per-query NDCG@10 diagnostic — shows exactly which queries pass and which fail."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieve import search_batch
from utils import load_public_queries


def ndcg_at_k(ranked, relevant_ids, k=10):
    rel = {int(r) for r in relevant_ids}
    dcg = sum(1.0 / math.log2(i + 2) for i, p in enumerate(ranked[:k]) if p in rel)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(rel), k)))
    return dcg / idcg if idcg > 0 else 0.0


rows = load_public_queries()
queries = [r["query"] for r in rows]
print(f"Running {len(queries)} queries...\n")
results = search_batch(queries, top_k=50)

total_ndcg = 0.0
zero_count = 0

for row, ranked in zip(rows, results):
    rel_ids = [int(p) for p in row["relevant_page_ids"]]
    ndcg = ndcg_at_k(ranked, rel_ids)
    total_ndcg += ndcg
    if ndcg == 0:
        zero_count += 1

    # Show position of each relevant page in the ranked list
    positions = []
    for pid in rel_ids:
        try:
            pos = ranked.index(pid) + 1
            positions.append(f"{pid}@{pos}")
        except ValueError:
            positions.append(f"{pid}@MISSING")

    if ndcg >= 0.9:
        mark = "✓✓"
    elif ndcg >= 0.3:
        mark = "~ "
    else:
        mark = "✗ "

    print(f"{mark} [{row['query_id']}] ndcg={ndcg:.3f}  {positions}")
    print(f"      {row['query'][:90]}")

print(f"\n{'='*60}")
print(f"mean_ndcg@10 = {total_ndcg / len(rows):.4f}")
print(f"Queries with ndcg=0 : {zero_count}/{len(rows)}")
print(f"Queries with ndcg>0 : {len(rows)-zero_count}/{len(rows)}")
