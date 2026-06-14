# Section B Retrieval Pipeline

This repository implements Section B: retrieval over a Wikipedia-style corpus.  The autograder imports `run(queries)` from `main.py` and expects prebuilt artifacts under `artifacts/`.

## Method

The current system is a hybrid retriever:

1. **Chunking**: each page is converted into compact title-weighted chunks for dense retrieval.
2. **Embedding**: chunks and queries are embedded with `sentence-transformers/all-MiniLM-L6-v2`.
3. **Dense index**: normalized MiniLM vectors are stored in a FAISS inner-product index.
4. **Lexical index**: BM25 over full page text, plus a title-only BM25 (b=0) as a separate signal.
5. **Fusion (RRF)**: dense, body-BM25, and title-BM25 are combined by Reciprocal Rank Fusion, which is robust to their different score distributions.
6. **Literal-evidence reranking**: the strongest candidates are reranked on *discriminating* signals — rare query-term coverage, exact phrase hits, and **rarity-weighted number/year matches** (a near-unique number is a decisive key) — plus a **facet-coverage** reward for multi-part "what links A, B, C" queries whose gold pages span several comma/"and"-separated facets.

This corpus contains many near-paraphrase pages that share generic vocabulary, so the gold page is identified by specific keys rather than topical similarity. For that reason we deliberately **removed** two earlier components — a hand-built query-synonym table and MMR diversity reranking — because both pulled in distractors / demoted relevant siblings and lowered NDCG@10. See [`EXPERIMENTS.md`](EXPERIMENTS.md) for the measured process behind these decisions.

## Artifacts

Build artifacts locally before submission:

```bash
python scripts/build_index.py
```

The build creates:

```text
artifacts/dense.faiss
artifacts/chunk_meta.json
artifacts/page_meta.json.gz
artifacts/bm25_meta.json
artifacts/bm25_postings.npz
artifacts/bm25_doc_len.npy
artifacts/build_info.json
```

`page_meta.json.gz` also contains compact normalized page text and page terms used by the v3 reranker. The grader does not rebuild the index, so these artifacts must be committed to the repository.

## Evaluate

```bash
python scripts/eval_public.py
```

Optional diagnostics:

```bash
python nano_scripts/diagnose.py    # per-query NDCG@10 + gold ranks (recall vs ranking)
python nano_scripts/tune.py         # parameter sweeps
```

## API

```python
from main import run
results = run(["query one", "query two"])
```

`run` returns `list[list[int]]`, one ranked list of page IDs per query. Only the first 10 IDs are scored by NDCG@10, but returning more IDs is allowed.
