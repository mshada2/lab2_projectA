# Experiment log — Section B retrieval

Each row is one change, measured with `python scripts/eval_public.py` (mean
NDCG@10 over the 29 deduplicated public queries). We change **one idea at a
time** so every number is attributable. Keep a change only if it does not lower
the score; otherwise revert and note why.

## How to reproduce a measurement

```bash
python scripts/eval_public.py        # prints mean_ndcg@10 ; no rebuild needed
```

The dense/BM25 artifacts only depend on `chunk.py`, `embed.py`, `index.py`. All
the changes below are query-time (`retrieve.py`, `utils.py`), so they do **not**
require `python scripts/build_index.py`.

## Results

| # | Change | File(s) | NDCG@10 | Keep? | Rationale |
|---|--------|---------|--------:|:-----:|-----------|
| 0 | Baseline (RRF + literal rerank + MMR + query expansions) | — | **0.2674** | — | Starting point on the deduplicated public set. |
| 1 | Remove hand-built `QUERY_EXPANSIONS` table; literal query terms only | `utils.py` | _TBD_ | _?_ | Corpus is full of near-paraphrase pages; injecting common synonyms ("research", "team") pulls in distractors and dilutes the rare terms that identify the gold page. Table was also overfit to the old (duplicate) query set. |
| 2 | Remove MMR diversity reranking | `retrieve.py` | _TBD_ | _?_ | On deduplicated multi-relevant queries the gold pages are intentionally similar; MMR's diversity term demotes relevant siblings — the opposite of what NDCG@10 rewards. |
| 3 | Rare-key dominance: weight exact number/year matches by corpus rarity (IDF) | `retrieve.py` | _TBD_ | _?_ | Single-fact queries are pinned by a near-unique key (e.g. an exact population `1,456,779`). A rare number → strong boost; a common year → mild boost. |
| 4 | Facet-coverage reward for "links A, B, C" / "how do X, Y, Z connect" queries | `retrieve.py` | _TBD_ | _?_ | Gold pages of multi-part queries each cover one or more comma/"and"-separated facets; reward pages that span multiple facets. Targets the high-relevance-count queries (5–12 golds). |
| 1–4 | **All four applied together** | `retrieve.py`, `utils.py` | **0.3243** | ✅ | +0.057 over baseline (+21% relative); query time 18.0 s. |
| 5 | Tier-1 knob sweep, all applied together: `RRF_K` 60→20, RRF weights dense/body/title 0.25/0.55/0.20→0.40/0.45/0.15, `RERANK_TOP_N` 500→1000, `LITERAL_EVIDENCE_WEIGHT` 0.22→0.32, `FACET_COVERAGE_WEIGHT` 0.30→0.45 | `retrieve.py` | **0.3415** | ✅ | +0.017 combined. Dense was under-weighted; sharper RRF_K and deeper/heavier reranking all pull together. Applied as one batch — not individually attributed. |
| 6 | Per-facet retrieval (Tier 2): for multi-part queries, run BM25 **and** dense (MiniLM) per comma/"and" facet and merge (max score), so single-facet golds enter the pool | `retrieve.py` | 0.3398 | ❌ reverted | −0.0017 vs 0.3415 (noise). **Key finding:** widening the candidate pool did not help, so the multi-part queries are *ranking-limited, not recall-limited* — the golds are already retrieved but ranked below 10. Redirects effort from recall to ranking. |
| 7 | Rare-term **anchor boost** in fusion: lift pages containing the query's 2 rarest content words (IDF≥5), scaled by IDF — mirrors the number boost | `retrieve.py` | **0.3449** | ✅ | +0.0034. Confirmed mechanism: q11 ("fjord") went 0.0 → 0.356 as the gold climbed from rank 22 into the top-10. q13/q20/q24 unchanged (their discriminators are 2-word phrases or multi-gold clusters, not single rare words). |
| 8 | Rerank rebalance toward evidence: `BASE_IN_RERANK` 0.78→0.60, `LITERAL_EVIDENCE_WEIGHT` 0.32→0.50 | `retrieve.py` | 0.3415 | ❌ reverted | −0.0034; erased the anchor gain. Finding: fusion rank already carries good signal — trusting evidence more demotes well-fused golds. The 0.78/0.32 balance is near-optimal; **not** base-dominated as hypothesised. |
| 9 | **Finer chunking (Tier 3, rebuild):** SUMMARY/WINDOW 180→90, stride 150→70, max chunks 12→20; `DENSE_CANDIDATES` 2000→3000 | `chunk.py`, `retrieve.py` | 0.3369 | ❌ | −0.008. Finer chunking *hurt* (351k vectors). Relevance is page-level and pages are short → many small chunks add aggregation noise and over-weight the repeated title. |
| 10 | **Coarser / page-level chunking (Tier 3, rebuild):** SUMMARY/WINDOW 180→400, stride→350, max chunks→6 | `chunk.py`, `retrieve.py` | 0.3369 | ❌ reverted | Coarser did not beat 180 either. Both finer (#9) and coarser (#10) scored lower → **180-word chunking is the final choice.** Also keeps `dense.faiss` ~under the GitHub size limit. |

| 11 | **Dense PRF (Rocchio in embedding space):** nudge query toward centroid of top-8 retrieved chunk vectors, re-search, merge by max (`β=0.5`) | `retrieve.py` | **0.3497** | ✅ | +0.0049. Yu et al. SIGIR'21. Semantic query expansion using only MiniLM vectors (legal). First structural lever to beat the plateau — confirms embedding-space expansion helps where lexical PRF drifted. Knobs `K`/`β` still to sweep. |

**Best confirmed configuration: NDCG@10 = 0.3497** — 180-word chunks; RRF (dense/body/title 0.40/0.45/0.15, RRF_K=20); literal-evidence rerank with rare-key number+word anchors and facet coverage. +29% over the 0.2674 baseline, all from generalizable rules.

> Changes 1–4 were applied together for the 0.3243 result above. To attribute the
> gain to each rule individually (for the write-up/video), revert one at a time
> with git and re-run `eval_public.py`, filling the per-row `NDCG@10` cells.

## Observations that drove the design (from inspecting the data)

- **Two query types.** Single-fact (1–3 golds, identified by a rare key:
  number/year/entity) vs. multi-facet "links/connect" (5–12 golds, a cluster of
  pages sharing themes). They need opposite treatments — specificity vs. breadth.
- **Distractor swamp.** The synthetic corpus reuses the same vocabulary
  ("profit-sharing", "distribution agreements", "research division") across many
  pages, so generic semantic/lexical similarity is weak. Discrimination comes
  from rare keys and specific facet combinations — which is what changes 1, 3, 4
  optimise for.
