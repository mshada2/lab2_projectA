"""Offline artifact build and loading helpers.

The grader does not call ``build_index``.  Build once locally, commit the produced
``artifacts/`` files, and keep query-time code in ``retrieve.py`` fast.

v4 additions:
- MAX_DF_FRACTION raised to 1.0: IDF naturally down-weights common terms;
  hard filtering risks removing content-bearing words that appear across many pages.
- Title-only BM25 (b=0, no length normalization) as a third retrieval signal.
  Titles are concise entity names; a title-only index provides strong exact-match
  signal independently of body length.
"""
from __future__ import annotations

import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from chunk import Chunk, chunk_corpus
from embed import embed_texts
from utils import (
    ARTIFACTS_DIR,
    EMBEDDING_MODEL_NAME,
    ensure_artifacts_dir,
    iter_entries,
    normalize_space,
    raw_tokenize,
    tokenize,
)

try:  # faiss-cpu or faiss is allowed by requirements.txt
    import faiss  # type: ignore
except Exception:  # pragma: no cover - grader should have faiss installed
    faiss = None

DENSE_INDEX_NAME = "dense.faiss"
CHUNK_META_NAME = "chunk_meta.json"
PAGE_META_NAME = "page_meta.json.gz"
BM25_META_NAME = "bm25_meta.json"
BM25_POSTINGS_NAME = "bm25_postings.npz"
DOC_LEN_NAME = "bm25_doc_len.npy"
TITLE_BM25_META_NAME = "title_bm25_meta.json"
TITLE_BM25_POSTINGS_NAME = "title_bm25_postings.npz"

BM25_K1 = 1.35
BM25_B = 0.72
# Raised from 0.35 → 1.0: BM25 IDF already down-weights common terms naturally.
# Hard-filtering risks removing content words that span many pages (e.g. "team",
# "research", "city") which are important for retrieval.
MAX_DF_FRACTION = 1.0
MIN_DF = 1
TITLE_WEIGHT = 3
PAGE_TEXT_MAX_CHARS = 7000

# Title-only BM25 parameters
TITLE_BM25_K1 = 1.2
TITLE_BM25_B = 0.0   # No length normalization — titles are uniformly short


def _match_text(record: Dict[str, Any]) -> str:
    """Compact normalized text used only for query-time literal reranking."""
    title = normalize_space(record.get("title", ""))
    content = normalize_space(record.get("content", ""))
    text = f"{title}. {content}" if title and content else title or content
    text = text.lower().replace("-", " ")
    text = " ".join(raw_tokenize(text))
    return text[:PAGE_TEXT_MAX_CHARS]


def _page_numbers_from_tokens(tokens: List[str]) -> List[str]:
    nums = sorted({t for t in tokens if t.isdigit() and len(t) >= 2})
    return nums


def _write_json(path: Path, obj: Any, *, gzip_it: bool = False) -> None:
    if gzip_it:
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
    else:
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _dense_build(out_dir: Path, records: List[Dict[str, Any]]) -> Tuple[int, int]:
    chunks: List[Chunk] = chunk_corpus(records)
    texts = [c.text for c in chunks]
    vectors = embed_texts(texts)
    vectors = np.ascontiguousarray(vectors.astype(np.float32, copy=False))

    if faiss is None:
        raise RuntimeError(
            "faiss is required to build the submitted dense index. Install faiss-cpu."
        )

    dim = int(vectors.shape[1]) if vectors.ndim == 2 and vectors.size else 384
    dense_index = faiss.IndexFlatIP(dim)
    dense_index.add(vectors)
    faiss.write_index(dense_index, str(out_dir / DENSE_INDEX_NAME))

    chunk_meta = {
        "model": EMBEDDING_MODEL_NAME,
        "num_vectors": int(len(chunks)),
        "dim": dim,
        "page_ids": [int(c.page_id) for c in chunks],
        "chunk_ids": [int(c.chunk_id) for c in chunks],
    }
    _write_json(out_dir / CHUNK_META_NAME, chunk_meta)
    return len(chunks), dim


def _doc_tokens(record: Dict[str, Any]) -> List[str]:
    title = normalize_space(record.get("title", ""))
    content = normalize_space(record.get("content", ""))
    tokens = []
    if title:
        title_tokens = tokenize(title)
        for _ in range(TITLE_WEIGHT):
            tokens.extend(title_tokens)
    tokens.extend(tokenize(content))
    return tokens


def _build_bm25(out_dir: Path, records: List[Dict[str, Any]]) -> Dict[str, Any]:
    n_docs = len(records)
    page_ids: List[int] = []
    titles: List[str] = []
    title_terms: List[List[str]] = []
    page_terms: List[List[str]] = []
    page_texts: List[str] = []
    page_numbers: List[List[str]] = []
    doc_len = np.zeros(n_docs, dtype=np.float32)
    df_counter: Counter[str] = Counter()

    print("BM25 pass 1/2: document frequencies")
    for i, rec in enumerate(records):
        page_ids.append(int(rec["page_id"]))
        title = normalize_space(rec.get("title", ""))
        titles.append(title)
        tterms = tokenize(title)
        title_terms.append(tterms)
        tokens = _doc_tokens(rec)
        doc_len[i] = max(1, len(tokens))
        df_counter.update(set(tokens))
        unique_terms = sorted(set(tokenize(title + " " + normalize_space(rec.get("content", "")))))
        page_terms.append(unique_terms)
        page_texts.append(_match_text(rec))
        page_numbers.append(_page_numbers_from_tokens(raw_tokenize(title + " " + normalize_space(rec.get("content", "")))))

    max_df = max(1, int(MAX_DF_FRACTION * n_docs))
    vocab_terms = [
        term for term, df in df_counter.items() if MIN_DF <= df <= max_df
    ]
    vocab_terms.sort()
    term_to_id = {term: i for i, term in enumerate(vocab_terms)}

    print(f"BM25 selected {len(vocab_terms):,} terms from {len(df_counter):,}")
    postings_docs: List[List[int]] = [[] for _ in vocab_terms]
    postings_tfs: List[List[int]] = [[] for _ in vocab_terms]

    print("BM25 pass 2/2: postings")
    for doc_idx, rec in enumerate(records):
        counts = Counter(t for t in _doc_tokens(rec) if t in term_to_id)
        for term, tf in counts.items():
            tid = term_to_id[term]
            postings_docs[tid].append(doc_idx)
            postings_tfs[tid].append(int(tf))

    starts = np.zeros(len(vocab_terms) + 1, dtype=np.int64)
    total_postings = 0
    for i, docs in enumerate(postings_docs):
        starts[i] = total_postings
        total_postings += len(docs)
    starts[len(vocab_terms)] = total_postings

    doc_indices = np.empty(total_postings, dtype=np.int32)
    term_freqs = np.empty(total_postings, dtype=np.float32)
    cursor = 0
    idfs = np.empty(len(vocab_terms), dtype=np.float32)
    for tid, term in enumerate(vocab_terms):
        docs = postings_docs[tid]
        tfs = postings_tfs[tid]
        n = len(docs)
        if n:
            doc_indices[cursor : cursor + n] = docs
            term_freqs[cursor : cursor + n] = tfs
            cursor += n
        idfs[tid] = math.log(1.0 + (n_docs - n + 0.5) / (n + 0.5))

    np.save(out_dir / DOC_LEN_NAME, doc_len)
    np.savez_compressed(
        out_dir / BM25_POSTINGS_NAME,
        doc_indices=doc_indices,
        term_freqs=term_freqs,
        starts=starts,
        idfs=idfs,
    )

    page_meta = {
        "page_ids": page_ids,
        "titles": titles,
        "title_terms": title_terms,
        "page_terms": page_terms,
        "page_texts": page_texts,
        "page_numbers": page_numbers,
        "page_text_max_chars": PAGE_TEXT_MAX_CHARS,
    }
    _write_json(out_dir / PAGE_META_NAME, page_meta, gzip_it=True)

    bm25_meta = {
        "n_docs": n_docs,
        "avgdl": float(doc_len.mean()) if n_docs else 1.0,
        "k1": BM25_K1,
        "b": BM25_B,
        "title_weight": TITLE_WEIGHT,
        "max_df_fraction": MAX_DF_FRACTION,
        "vocab": {term: i for i, term in enumerate(vocab_terms)},
    }
    _write_json(out_dir / BM25_META_NAME, bm25_meta)

    return {
        "n_docs": n_docs,
        "vocab_size": len(vocab_terms),
        "total_postings": int(total_postings),
        "avgdl": float(doc_len.mean()) if n_docs else 1.0,
    }


def _build_title_bm25(out_dir: Path, records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a title-only BM25 index (BM25F field variant).

    Titles are concise entity identifiers.  A dedicated title index with no
    length normalization (b=0) rewards pages whose titles directly match query
    terms without penalizing short titles.
    """
    print("Title BM25: building title-only index")
    n_docs = len(records)
    page_ids = [int(rec["page_id"]) for rec in records]

    title_tokens_list: List[List[str]] = []
    df_counter: Counter[str] = Counter()
    for rec in records:
        title = normalize_space(rec.get("title", ""))
        tokens = tokenize(title)
        title_tokens_list.append(tokens)
        df_counter.update(set(tokens))

    vocab_terms = sorted(df_counter.keys())
    term_to_id = {term: i for i, term in enumerate(vocab_terms)}

    postings_docs: List[List[int]] = [[] for _ in vocab_terms]
    postings_tfs: List[List[int]] = [[] for _ in vocab_terms]
    doc_len = np.zeros(n_docs, dtype=np.float32)

    for doc_idx, tokens in enumerate(title_tokens_list):
        counts = Counter(t for t in tokens if t in term_to_id)
        doc_len[doc_idx] = max(1, len(tokens))
        for term, tf in counts.items():
            tid = term_to_id[term]
            postings_docs[tid].append(doc_idx)
            postings_tfs[tid].append(int(tf))

    avgdl = float(doc_len.mean()) if n_docs else 1.0

    starts = np.zeros(len(vocab_terms) + 1, dtype=np.int64)
    total = 0
    for i, docs in enumerate(postings_docs):
        starts[i] = total
        total += len(docs)
    starts[len(vocab_terms)] = total

    doc_indices = np.empty(total, dtype=np.int32)
    term_freqs = np.empty(total, dtype=np.float32)
    idfs = np.empty(len(vocab_terms), dtype=np.float32)
    cursor = 0

    for tid in range(len(vocab_terms)):
        docs = postings_docs[tid]
        tfs = postings_tfs[tid]
        n = len(docs)
        if n:
            doc_indices[cursor : cursor + n] = docs
            term_freqs[cursor : cursor + n] = tfs
            cursor += n
        idfs[tid] = math.log(1.0 + (n_docs - n + 0.5) / (n + 0.5))

    np.savez_compressed(
        out_dir / TITLE_BM25_POSTINGS_NAME,
        doc_indices=doc_indices,
        term_freqs=term_freqs,
        starts=starts,
        idfs=idfs,
    )

    meta = {
        "n_docs": n_docs,
        "avgdl": avgdl,
        "k1": TITLE_BM25_K1,
        "b": TITLE_BM25_B,
        "page_ids": page_ids,
        "vocab": {term: i for i, term in enumerate(vocab_terms)},
    }
    _write_json(out_dir / TITLE_BM25_META_NAME, meta)
    print(f"Title BM25: {len(vocab_terms):,} terms, {total:,} postings")
    return {"vocab_size": len(vocab_terms), "total_postings": int(total)}


def build_index(
    *,
    entries_dir: Optional[Path] = None,
    artifacts_dir: Optional[Path] = None,
) -> Tuple[None, List[int]]:
    """Build all offline artifacts needed by ``main.run``."""
    out_dir = artifacts_dir or ensure_artifacts_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading corpus records")
    records = list(iter_entries(entries_dir))
    print(f"Loaded {len(records):,} pages")

    dense_count, dim = _dense_build(out_dir, records)
    bm25_stats = _build_bm25(out_dir, records)
    title_stats = _build_title_bm25(out_dir, records)

    build_info = {
        "model": EMBEDDING_MODEL_NAME,
        "dense_vectors": dense_count,
        "dense_dim": dim,
        "bm25": bm25_stats,
        "title_bm25": title_stats,
        "artifacts": [
            DENSE_INDEX_NAME,
            CHUNK_META_NAME,
            PAGE_META_NAME,
            BM25_META_NAME,
            BM25_POSTINGS_NAME,
            DOC_LEN_NAME,
            TITLE_BM25_META_NAME,
            TITLE_BM25_POSTINGS_NAME,
        ],
    }
    _write_json(out_dir / "build_info.json", build_info)
    print(json.dumps(build_info, indent=2))
    return None, [int(r["page_id"]) for r in records]


def load_index(artifacts_dir: Optional[Path] = None):
    """Backward-compatible loader name used by older starter code."""
    from retrieve import load_artifacts

    return load_artifacts(artifacts_dir or ARTIFACTS_DIR)
