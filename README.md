# Section B — Wikipedia Retrieval Pipeline

End-to-end retrieval over a ~27,000-page Wikipedia-style corpus. The autograder
calls `run(queries)` from `main.py`, which loads the prebuilt index from
`artifacts/` and returns a ranked list of `page_id`s per query, scored by
**NDCG@10**.

**📹 Video presentation:** https://technionmail-my.sharepoint.com/:v:/g/personal/hila_litvak_campus_technion_ac_il/IQCFbEt_vhqpSKBqGwiVYuPyAXnLntV4JitnVxqvgmMuxl4?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=eSfi6k

## Pipeline

| Stage | File | Method |
|-------|------|--------|
| **Chunk** | `chunk.py` | Coarse, page-level chunks with the title prepended (relevance is page-level and pages are short; finer and coarser chunking both measured lower). |
| **Embed** | `embed.py` | `sentence-transformers/all-MiniLM-L6-v2`, L2-normalized. |
| **Index** | `index.py` | FAISS inner-product dense index + full-text BM25 + title-only BM25 (b=0). |
| **Retrieve** | `retrieve.py` | (1) RRF fusion of dense + body-BM25 + title-BM25; (2) Dense pseudo-relevance feedback (Rocchio in embedding space); (3) literal-evidence rerank — rare-term/title coverage, rare-key number+word anchors, facet coverage for multi-part "links A, B, C" queries; (4) two-stage cross-encoder rerank — `ms-marco-MiniLM-L-6-v2` over the top-60, then `ms-marco-MiniLM-L4-v2` over the top-3. |

The corpus reuses vocabulary across many look-alike pages, so the pipeline
rewards **specific discriminators** (rare keys, facet coverage) over generic
topical similarity. The full empirical process — including rejected experiments —
is in [`EXPERIMENTS.md`](EXPERIMENTS.md) (NDCG@10 0.267 → 0.4536).

## Setup

```bash
pip install -r requirements.txt
```

## Clone

This repository stores large artifacts (indexes + model weights) with Git LFS.
Install Git LFS before cloning:

```bash
git lfs install
git clone https://github.com/mshada2/lab2_projectA.git
cd lab2_projectA
```

## Evaluate (no rebuild needed)

```bash
python scripts/eval_public.py     # mean NDCG@10 on the public queries (use python3 if needed)
```

On a fresh clone, evaluation runs without rebuilding the index. The prebuilt
files under `artifacts/` are committed (large binaries via Git LFS).

> **No downloads at runtime.** All model weights — `all-MiniLM-L6-v2` and both
> cross-encoders (`ms-marco-MiniLM-L-6-v2`, `ms-marco-MiniLM-L4-v2`) — are
> committed under `artifacts/` (`minilm/`, `ce_stage1/`, `ce_stage2/`) and loaded
> from disk, so `run()` never downloads from HuggingFace.
>
> **Query time:** ~36 s on 29 public queries (GPU); ~50 s on 50 queries — within
> the 60 s grading limit.

## Build the index (offline only — not run at grading)

```bash
python scripts/build_index.py     # ~10–16 min; re-creates the index artifacts
```

Produces, under `artifacts/`:

| File / folder | Format | Purpose |
|---------------|--------|---------|
| `dense.faiss` | FAISS `IndexFlatIP` | chunk embedding vectors |
| `chunk_meta.json` | JSON | chunk → page_id mapping |
| `page_meta.json.gz` | gzipped JSON | page ids, titles, terms, text, numbers (for the reranker) |
| `bm25_meta.json`, `bm25_postings.npz`, `bm25_doc_len.npy` | JSON / npz / npy | full-text BM25 index |
| `title_bm25_meta.json`, `title_bm25_postings.npz` | JSON / npz | title-only BM25 index |
| `build_info.json` | JSON | build statistics |
| `minilm/`, `ce_stage1/`, `ce_stage2/` | model dirs | embedder + cross-encoder weights (shipped, loaded from disk) |

## Dev tools

```bash
python nano_scripts/diagnose.py   # per-query NDCG@10 + gold ranks (recall vs ranking)
python nano_scripts/tune.py       # parameter sweeps
```

## Layout

```
main.py                     run(queries) — autograder entry point
chunk.py embed.py index.py  offline build (chunk -> embed -> index)
retrieve.py utils.py        query-time retrieval + shared helpers
eval.py                     NDCG@10 utilities (read-only)
scripts/                    build_index.py, eval_public.py (read-only)
nano_scripts/               diagnose.py, tune.py (dev tools)
data/                       public_queries.json (full corpus not committed)
artifacts/                  prebuilt index + model weights (committed; Git LFS)
EXPERIMENTS.md              measured design process
```

## API

```python
from main import run
results = run(["Which city hosts light commuter rail on a fjord-lined coast?"])
# -> list[list[int]] : one ranked list of page_ids per query (top 10 scored)
```
