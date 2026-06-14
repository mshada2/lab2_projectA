# Section B — Wikipedia Retrieval Pipeline

End-to-end retrieval over a ~27,000-page Wikipedia-style corpus. The autograder
calls `run(queries)` from `main.py`, which loads the prebuilt index from
`artifacts/` and returns a ranked list of `page_id`s per query, scored by
**NDCG@10**.

**📹 Video presentation:** `<ADD LINK HERE>`

## Pipeline

| Stage | File | Method |
|-------|------|--------|
| **Chunk** | `chunk.py` | Coarse, page-level chunks with the title prepended (relevance is page-level and pages are short — finer chunks were measured to hurt). |
| **Embed** | `embed.py` | `sentence-transformers/all-MiniLM-L6-v2`, L2-normalized. |
| **Index** | `index.py` | FAISS inner-product dense index + full-text BM25 + title-only BM25 (b=0). |
| **Retrieve** | `retrieve.py` | Reciprocal Rank Fusion of the three signals, then a literal-evidence reranker: rare-term coverage, rare-key boosts for distinctive numbers/words, and facet coverage for multi-part "links A, B, C" queries. |

The corpus reuses vocabulary across many look-alike pages, so the pipeline
rewards **specific discriminators** (rare keys, facet coverage) over generic
topical similarity. The full empirical process behind these choices — including
rejected experiments — is in [`EXPERIMENTS.md`](EXPERIMENTS.md) (NDCG@10 0.267 → 0.451).

## Setup

```bash
pip install -r requirements.txt
```

## Evaluate (no rebuild needed)

```bash
python scripts/eval_public.py     # mean NDCG@10 on the public queries
```

On a fresh clone this runs **without rebuilding** — the `artifacts/` are
committed (large binaries via Git LFS).

## Build the index (offline only — not run at grading)

```bash
python scripts/build_index.py     # ~10–16 min; re-creates artifacts/
```

Produces, under `artifacts/`:

| File | Format | Purpose |
|------|--------|---------|
| `dense.faiss` | FAISS `IndexFlatIP` | chunk embedding vectors |
| `chunk_meta.json` | JSON | chunk → page_id mapping |
| `page_meta.json.gz` | gzipped JSON | page ids, titles, terms, text, numbers (for the reranker) |
| `bm25_meta.json`, `bm25_postings.npz`, `bm25_doc_len.npy` | JSON / npz / npy | full-text BM25 index |
| `title_bm25_meta.json`, `title_bm25_postings.npz` | JSON / npz | title-only BM25 index |
| `build_info.json` | JSON | build statistics |

## Dev tools

```bash
python nano_scripts/diagnose.py   # per-query NDCG@10 + gold ranks (recall vs ranking)
python nano_scripts/tune.py       # parameter sweeps
```

## Layout

```
main.py                     run(queries) — autograder entry point
chunk.py embed.py index.py  offline build (chunk → embed → index)
retrieve.py utils.py        query-time retrieval + shared helpers
eval.py                     NDCG@10 utilities (read-only)
scripts/                    build_index.py, eval_public.py (read-only)
nano_scripts/               diagnose.py, tune.py (dev tools)
data/                       public_queries.json + Wikipedia Entries/ (corpus)
artifacts/                  prebuilt index (committed; Git LFS for big files)
EXPERIMENTS.md              measured design process
```

## API

```python
from main import run
results = run(["Which city hosts light commuter rail on a fjord-lined coast?"])
# -> list[list[int]] : one ranked list of page_ids per query (top 10 scored)
```
