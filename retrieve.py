"""Hybrid retrieval: RRF fusion of dense + body-BM25 + title-BM25, followed by
literal-evidence reranking.

Pipeline and design rationale (the empirical process behind these choices is
documented in the README and the presentation video):

1. **Reciprocal Rank Fusion (RRF)** combines the three retrievers by rank
   rather than raw score, so it is robust to their very different score
   distributions and avoids hand-tuned score scaling.

2. **Title-only BM25** (b=0, no length normalization) is a third RRF signal.
   Titles are concise entity names, so an exact title hit is strong evidence
   independent of body length.

3. **Literal-evidence reranking** re-scores the strongest candidates on
   *discriminating* signals: rare-term coverage, exact phrase hits, and
   number/year matches.  This corpus contains many near-paraphrase pages that
   share generic vocabulary, so the gold page is identified by specific keys
   (a rare entity, an exact number, a year) rather than by topical similarity.

Two earlier components were removed after they lowered NDCG@10 on the
deduplicated evaluation set, and the reasons are recorded here so the decision
is auditable:

* **Pseudo-Relevance Feedback (PRF)** — expanding the query with terms mined
  from the top BM25 docs drifted toward distractor pages at the current
  precision and hurt multi-relevant queries.
* **MMR diversity reranking** — its diversity term demoted genuinely-relevant
  sibling pages on multi-relevant queries (whose gold pages are intentionally
  similar), which is the opposite of what NDCG@10 rewards here.
"""
from __future__ import annotations

import gzip
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from embed import embed_queries
from index import (
    BM25_META_NAME,
    BM25_POSTINGS_NAME,
    CHUNK_META_NAME,
    DENSE_INDEX_NAME,
    DOC_LEN_NAME,
    PAGE_META_NAME,
    TITLE_BM25_META_NAME,
    TITLE_BM25_POSTINGS_NAME,
)
from utils import (
    ARTIFACTS_DIR,
    DEFAULT_RETURN_K,
    K_EVAL,
    expand_query_tokens,
    raw_tokenize,
    tokenize,
)

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None

try:
    import torch as _torch
    _DEVICE = "cuda" if _torch.cuda.is_available() else "cpu"
except ImportError:
    _DEVICE = "cpu"

# ---------------------------------------------------------------------------
# Cross-encoder reranking
# ---------------------------------------------------------------------------
CROSS_ENCODER_ENABLED = True
CROSS_ENCODE_TOP_N = 60

# Best public setting so far: keep cross-encoder dominant, but preserve a
# small amount of the pre-CE retrieval/literal score.
CROSS_ENCODER_BLEND_ENABLED = True
CROSS_ENCODER_CE_WEIGHT = 0.95
CROSS_ENCODER_BASE_WEIGHT = 0.05

_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_CROSS_ENCODER = None


def _get_cross_encoder():
    global _CROSS_ENCODER
    if _CROSS_ENCODER is None:
        from sentence_transformers import CrossEncoder
        _CROSS_ENCODER = CrossEncoder(
            _CROSS_ENCODER_MODEL, max_length=512, device=_DEVICE
        )
    return _CROSS_ENCODER

# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------
DENSE_CANDIDATES = 2000       # chunks pulled from FAISS per query

# Dense pseudo-relevance feedback (Rocchio in embedding space; Yu et al. SIGIR'21).
# Nudge the query vector toward the centroid of its top-K retrieved chunks, then
# re-search.  Expands the query semantically using only MiniLM embeddings.
DENSE_PRF = True
DENSE_PRF_K = 4               # top chunks used to form the feedback centroid
DENSE_PRF_BETA = 0.3          # centroid weight relative to the original query
BM25_CANDIDATES = 8000        # wide BM25 net for better recall
TITLE_BM25_CANDIDATES = 500
FINAL_POOL = 9000             # holds BM25+dense+title union
MIN_RESULTS = K_EVAL
MAX_RESULTS = DEFAULT_RETURN_K
SCORE_THRESHOLD = 0.02

# ---------------------------------------------------------------------------
# RRF fusion weights (must be positive; need not sum to 1)
# ---------------------------------------------------------------------------
RRF_K = 20 # try to sharpen the difference between rank 1 and rank 2 without overly penalising lower ranks
DENSE_RRF_WEIGHT = 0.4
BM25_RRF_WEIGHT = 0.45
TITLE_BM25_RRF_WEIGHT = 0.15

# Auxiliary boosts added on top of RRF (RRF scores are ~0.001–0.017).
# Title overlap stays tiny.  An exact number match, however, is a near-unique
# key, so its boost is scaled by the number's corpus rarity (IDF) and allowed to
# be larger — enough to lift a uniquely-matching page toward the top of fusion.
TITLE_MATCH_WEIGHT = 0.0008
NUMBER_MATCH_WEIGHT = 0.02     # scaled by idf/RARE_NUMBER_IDF in _auxiliary_boosts
RARE_NUMBER_IDF = 9.0          # IDF at/above which a number is treated as ~unique
DENSE_AGG_SUM_WEIGHT = 0.10

# Rare content-word "anchor" boost (mirrors the number boost). The gold page of
# a query usually owns the query's rarest term ("fjord", "maritime", "alloy"),
# while distractors share only the generic theme words.  Lifting pages that
# contain those rare anchors in fusion lets them survive into the reranked
# top-10.  Diagnostic showed this is the dominant failure mode for buried golds.
ANCHOR_MATCH_WEIGHT = 0.015
ANCHOR_TITLE_MATCH_WEIGHT = 0.030  # anchor in title is ~2× stronger than in body
ANCHOR_TERMS_N = 4             # the N rarest query terms act as discriminators
ANCHOR_MIN_IDF = 5.0           # only genuinely rare terms qualify as anchors
ANCHOR_IDF_NORM = 9.0          # normaliser for anchor IDF scaling

# ---------------------------------------------------------------------------
# Literal-evidence reranking (unchanged from v3)
# ---------------------------------------------------------------------------
RERANK_TOP_N = 1000
BASE_IN_RERANK = 0.78
LITERAL_EVIDENCE_WEIGHT = 0.32
RARE_COVERAGE_WEIGHT = 0.46
EXACT_COVERAGE_WEIGHT = 0.18
PHRASE_WEIGHT = 0.22
NUMBER_EVIDENCE_WEIGHT = 0.30   # raised: a rare exact number is a decisive key
RELATIVE_YEAR_WEIGHT = 0.12
FACET_COVERAGE_WEIGHT = 0.45    # reward covering several facets of "links A, B, C" queries
TITLE_EVIDENCE_WEIGHT = 0.40    # rare query terms in page title → very strong evidence

GENERIC_QUERY_TERMS = {
    "link", "links", "learned", "together", "connect", "connection",
    "combines", "combined", "involved", "involve", "fit", "called", "named",
    "year", "years", "page", "pages", "relevant", "information",
}

DECADE_RE = re.compile(r"\b(1[5-9]\d0|20\d0)s\b")
# Facets of a multi-part query are separated by commas or the word "and":
# "What links A, B, and C?" → ["A", "B", "C"].
FACET_SPLIT_RE = re.compile(r",|\band\b")


# ---------------------------------------------------------------------------
# Artifacts dataclass
# ---------------------------------------------------------------------------
@dataclass
class RetrievalArtifacts:
    # Core fields
    dense_index: object
    chunk_page_ids: np.ndarray
    page_ids: np.ndarray
    titles: List[str]
    title_term_sets: List[set]
    page_term_sets: List[set]
    page_texts: List[str]
    page_numbers: List[set]
    pid_to_idx: Dict[int, int]
    vocab: Dict[str, int]
    doc_len: np.ndarray
    avgdl: float
    k1: float
    b: float
    postings_doc_indices: np.ndarray
    postings_term_freqs: np.ndarray
    postings_starts: np.ndarray
    postings_idfs: np.ndarray
    # Optional title BM25 fields (None when title index not present)
    title_vocab: Optional[Dict[str, int]] = field(default=None)
    title_k1: float = field(default=1.2)
    title_b: float = field(default=0.0)
    title_postings_doc_indices: Optional[np.ndarray] = field(default=None)
    title_postings_term_freqs: Optional[np.ndarray] = field(default=None)
    title_postings_starts: Optional[np.ndarray] = field(default=None)
    title_postings_idfs: Optional[np.ndarray] = field(default=None)


_STATE: Optional[RetrievalArtifacts] = None


def _load_json(path: Path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(path.read_text(encoding="utf-8"))


def load_artifacts(artifacts_dir: Optional[Path] = None) -> RetrievalArtifacts:
    root = artifacts_dir or ARTIFACTS_DIR
    if faiss is None:
        raise RuntimeError("faiss is required at query time. Install faiss-cpu.")

    dense_index = faiss.read_index(str(root / DENSE_INDEX_NAME))
    chunk_meta = _load_json(root / CHUNK_META_NAME)
    page_meta = _load_json(root / PAGE_META_NAME)
    bm25_meta = _load_json(root / BM25_META_NAME)
    postings = np.load(root / BM25_POSTINGS_NAME)

    page_ids = np.asarray(page_meta["page_ids"], dtype=np.int64)
    pid_to_idx = {int(pid): i for i, pid in enumerate(page_ids.tolist())}
    n_pages = len(page_ids)

    page_terms = page_meta.get("page_terms", [[] for _ in range(n_pages)])
    page_texts = page_meta.get("page_texts", ["" for _ in range(n_pages)])
    page_numbers = page_meta.get("page_numbers", [[] for _ in range(n_pages)])

    # --- Optional title BM25 ---
    title_vocab: Optional[Dict[str, int]] = None
    title_k1, title_b = 1.2, 0.0
    t_doc_indices = t_term_freqs = t_starts = t_idfs = None

    t_meta_path = root / TITLE_BM25_META_NAME
    t_post_path = root / TITLE_BM25_POSTINGS_NAME
    if t_meta_path.exists() and t_post_path.exists():
        try:
            tmeta = _load_json(t_meta_path)
            tpost = np.load(t_post_path)
            title_vocab = {str(k): int(v) for k, v in tmeta["vocab"].items()}
            title_k1 = float(tmeta.get("k1", 1.2))
            title_b = float(tmeta.get("b", 0.0))
            t_doc_indices = tpost["doc_indices"].astype(np.int32, copy=False)
            t_term_freqs = tpost["term_freqs"].astype(np.float32, copy=False)
            t_starts = tpost["starts"].astype(np.int64, copy=False)
            t_idfs = tpost["idfs"].astype(np.float32, copy=False)
            print(f"Title BM25 loaded: {len(title_vocab):,} terms")
        except Exception as exc:
            print(f"Warning: could not load title BM25 ({exc}); skipping")
            title_vocab = None

    return RetrievalArtifacts(
        dense_index=dense_index,
        chunk_page_ids=np.asarray(chunk_meta["page_ids"], dtype=np.int64),
        page_ids=page_ids,
        titles=list(page_meta["titles"]),
        title_term_sets=[set(x) for x in page_meta.get("title_terms", [])],
        page_term_sets=[set(x) for x in page_terms],
        page_texts=[str(x) for x in page_texts],
        page_numbers=[set(str(v) for v in row) for row in page_numbers],
        pid_to_idx=pid_to_idx,
        vocab={str(k): int(v) for k, v in bm25_meta["vocab"].items()},
        doc_len=np.load(root / DOC_LEN_NAME).astype(np.float32, copy=False),
        avgdl=float(bm25_meta["avgdl"]),
        k1=float(bm25_meta["k1"]),
        b=float(bm25_meta["b"]),
        postings_doc_indices=postings["doc_indices"].astype(np.int32, copy=False),
        postings_term_freqs=postings["term_freqs"].astype(np.float32, copy=False),
        postings_starts=postings["starts"].astype(np.int64, copy=False),
        postings_idfs=postings["idfs"].astype(np.float32, copy=False),
        title_vocab=title_vocab,
        title_k1=title_k1,
        title_b=title_b,
        title_postings_doc_indices=t_doc_indices,
        title_postings_term_freqs=t_term_freqs,
        title_postings_starts=t_starts,
        title_postings_idfs=t_idfs,
    )


def _ensure_global_state(artifacts_dir: Optional[Path] = None) -> RetrievalArtifacts:
    global _STATE
    if artifacts_dir is not None:
        return load_artifacts(artifacts_dir)
    if _STATE is None:
        _STATE = load_artifacts(ARTIFACTS_DIR)
    return _STATE


# ---------------------------------------------------------------------------
# Dense retrieval
# ---------------------------------------------------------------------------

def _aggregate_dense(
    state: RetrievalArtifacts, scores: np.ndarray, indices: np.ndarray
) -> Dict[int, float]:
    """Aggregate chunk-level FAISS hits into page scores (max + mean blend)."""
    best: Dict[int, float] = {}
    sums: Dict[int, float] = {}
    counts: Dict[int, int] = {}
    for raw_score, raw_idx in zip(scores[0], indices[0]):
        idx = int(raw_idx)
        if idx < 0 or idx >= len(state.chunk_page_ids):
            continue
        pid = int(state.chunk_page_ids[idx])
        s = float(raw_score)
        if pid not in best or s > best[pid]:
            best[pid] = s
        if s > 0:
            sums[pid] = sums.get(pid, 0.0) + s
            counts[pid] = counts.get(pid, 0) + 1
    return {
        pid: s + DENSE_AGG_SUM_WEIGHT * (sums.get(pid, 0.0) / max(1, counts.get(pid, 1)))
        for pid, s in best.items()
    }


def _dense_page_scores(
    state: RetrievalArtifacts, query_vectors: np.ndarray, q_idx: int
) -> Dict[int, float]:
    qv = np.ascontiguousarray(
        query_vectors[q_idx : q_idx + 1].astype(np.float32, copy=False)
    )
    scores, indices = state.dense_index.search(qv, DENSE_CANDIDATES)
    pages = _aggregate_dense(state, scores, indices)

    if DENSE_PRF:
        # Rocchio in embedding space: move the query toward the centroid of its
        # top-K retrieved chunk vectors, re-search, and merge by max.  Recovers
        # related pages (esp. for multi-part queries) using only MiniLM vectors.
        top_ids = [int(i) for i in indices[0][:DENSE_PRF_K] if int(i) >= 0]
        if top_ids:
            vecs = np.vstack([state.dense_index.reconstruct(i) for i in top_ids])
            new_q = qv[0] + DENSE_PRF_BETA * vecs.mean(axis=0)
            norm = float(np.linalg.norm(new_q))
            if norm > 0.0:
                new_q = np.ascontiguousarray(
                    (new_q / norm).astype(np.float32)
                ).reshape(1, -1)
                prf_scores, prf_indices = state.dense_index.search(
                    new_q, DENSE_CANDIDATES
                )
                for pid, s in _aggregate_dense(state, prf_scores, prf_indices).items():
                    if s > pages.get(pid, -1e9):
                        pages[pid] = s
    return pages


# ---------------------------------------------------------------------------
# BM25 retrieval
# ---------------------------------------------------------------------------

def _add_bm25_term(
    state: RetrievalArtifacts,
    scores: np.ndarray,
    denom_const: np.ndarray,
    term: str,
) -> None:
    """Add one BM25 term's contribution into scores."""
    tid = state.vocab.get(term)
    if tid is None:
        return
    start = int(state.postings_starts[tid])
    end = int(state.postings_starts[tid + 1])
    if end <= start:
        return
    docs = state.postings_doc_indices[start:end]
    tf = state.postings_term_freqs[start:end]
    idf = float(state.postings_idfs[tid])
    denom = tf + denom_const[docs]
    scores[docs] += idf * (tf * (state.k1 + 1.0) / denom)


def _bm25_doc_scores(
    state: RetrievalArtifacts,
    query: str,
) -> Dict[int, float]:
    """BM25 over full-page text."""
    q_tokens = expand_query_tokens(tokenize(query))
    # Filter meta-query boilerplate ("links", "learned", "together", etc.) that
    # bias BM25 toward wrong pages for "What links …" / "What can be learned …"
    # query patterns.  These words are already excluded in _query_content_terms
    # used by the reranker; mirror that filter here for consistency.
    q_tokens = [t for t in q_tokens if t not in GENERIC_QUERY_TERMS]
    if not q_tokens:
        return {}

    scores = np.zeros(len(state.page_ids), dtype=np.float32)
    denom_const = state.k1 * (
        1.0 - state.b + state.b * (state.doc_len / state.avgdl)
    )
    seen: set = set()

    for term in q_tokens:
        if term in seen:
            continue
        seen.add(term)
        _add_bm25_term(state, scores, denom_const, term)

    if not np.any(scores > 0):
        return {}
    n = min(BM25_CANDIDATES, int(np.count_nonzero(scores > 0)))
    top_idx = np.argpartition(-scores, n - 1)[:n]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    return {int(state.page_ids[int(i)]): float(scores[int(i)]) for i in top_idx}


# ---------------------------------------------------------------------------
# Title BM25 retrieval (third RRF signal)
# ---------------------------------------------------------------------------

def _title_bm25_scores(
    state: RetrievalArtifacts, query: str
) -> Dict[int, float]:
    """BM25 over titles only (b=0, no length normalization).

    Returns empty dict if title index was not built.
    """
    if state.title_vocab is None or state.title_postings_starts is None:
        return {}

    q_tokens = expand_query_tokens(tokenize(query))
    q_tokens = [t for t in q_tokens if t not in GENERIC_QUERY_TERMS]
    if not q_tokens:
        return {}

    n_docs = len(state.page_ids)
    scores = np.zeros(n_docs, dtype=np.float32)
    seen: set = set()
    k1 = state.title_k1

    for term in q_tokens:
        if term in seen:
            continue
        seen.add(term)
        tid = state.title_vocab.get(term)
        if tid is None:
            continue
        start = int(state.title_postings_starts[tid])
        end = int(state.title_postings_starts[tid + 1])
        if end <= start:
            continue
        docs = state.title_postings_doc_indices[start:end]
        tf = state.title_postings_term_freqs[start:end]
        idf = float(state.title_postings_idfs[tid])
        # b=0 → denom = tf + k1 (scalar, same for every doc)
        denom = tf + k1
        scores[docs] += idf * (tf * (k1 + 1.0) / denom)

    if not np.any(scores > 0):
        return {}
    n_nz = int(np.count_nonzero(scores > 0))
    n = min(TITLE_BM25_CANDIDATES, n_nz)
    if n == 0:
        return {}
    top_idx = np.argpartition(-scores, n - 1)[:n]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    return {int(state.page_ids[int(i)]): float(scores[int(i)]) for i in top_idx}


# ---------------------------------------------------------------------------
# IDF helper
# ---------------------------------------------------------------------------

def _idf_for_token(state: RetrievalArtifacts, token: str) -> float:
    tid = state.vocab.get(token)
    if tid is None or tid < 0 or tid >= len(state.postings_idfs):
        return 1.0
    return float(state.postings_idfs[tid])


# ---------------------------------------------------------------------------
# RRF fusion (replaces linear score combination)
# ---------------------------------------------------------------------------

def _rrf_fuse(
    dense: Dict[int, float],
    bm25: Dict[int, float],
    title_bm25: Dict[int, float],
    aux: Dict[int, float],
) -> List[Tuple[int, float]]:
    """Weighted Reciprocal Rank Fusion over dense, BM25, and title BM25.

    Each retriever contributes 1/(RRF_K + rank) scaled by its weight.
    Pages absent from a retriever's list get a penalty rank beyond the
    list length, contributing a small but non-zero score.
    """
    # Build rank maps (0-based: rank 0 is the top document)
    signals: List[Tuple[Dict[int, int], float, int]] = []

    def _rank_map(d: Dict[int, float]) -> Dict[int, int]:
        return {
            pid: i
            for i, (pid, _) in enumerate(sorted(d.items(), key=lambda x: -x[1]))
        }

    if dense:
        signals.append((_rank_map(dense), DENSE_RRF_WEIGHT, len(dense)))
    if bm25:
        signals.append((_rank_map(bm25), BM25_RRF_WEIGHT, len(bm25)))
    if title_bm25:
        signals.append((_rank_map(title_bm25), TITLE_BM25_RRF_WEIGHT, len(title_bm25)))

    if not signals:
        return []

    total_w = sum(w for _, w, _ in signals)
    all_pids = set(dense) | set(bm25) | set(title_bm25) | set(aux)

    fused: List[Tuple[int, float]] = []
    for pid in all_pids:
        score = 0.0
        for rank_map, w, n in signals:
            # Unseen pages are penalised with rank = n (one past the last)
            r = rank_map.get(pid, n)
            score += (w / total_w) * (1.0 / (RRF_K + r))
        score += aux.get(pid, 0.0)
        fused.append((pid, score))

    fused.sort(key=lambda x: (-x[1], x[0]))
    return fused


# ---------------------------------------------------------------------------
# Auxiliary boosts (title overlap, number overlap, rare-term anchors)
# ---------------------------------------------------------------------------

def _auxiliary_boosts(
    state: RetrievalArtifacts,
    query_tokens: List[str],
    candidate_pids: Iterable[int],
) -> Dict[int, float]:
    boosts: Dict[int, float] = {}
    query_set = set(query_tokens)
    query_numbers = {t for t in query_tokens if t.isdigit() and len(t) >= 2}

    # The query's rarest content words act as discriminating "anchors".
    content_terms = {t for t in query_tokens if not t.isdigit() and len(t) > 1}
    anchors = sorted(
        ((t, _idf_for_token(state, t)) for t in content_terms),
        key=lambda x: -x[1],
    )[:ANCHOR_TERMS_N]
    anchors = [(t, idf) for t, idf in anchors if idf >= ANCHOR_MIN_IDF]

    for pid in candidate_pids:
        doc_idx = state.pid_to_idx.get(int(pid))
        if doc_idx is None:
            continue
        title_terms = (
            state.title_term_sets[doc_idx]
            if doc_idx < len(state.title_term_sets)
            else set()
        )
        if title_terms:
            overlap = len(query_set & title_terms)
            if overlap:
                boosts[pid] = boosts.get(pid, 0.0) + TITLE_MATCH_WEIGHT * min(
                    1.0, overlap / max(1, len(title_terms))
                )
        if query_numbers:
            page_nums = (
                state.page_numbers[doc_idx]
                if doc_idx < len(state.page_numbers)
                else set()
            )
            matched = query_numbers & page_nums
            if matched:
                # Scale by rarity: a near-unique number (high IDF) is a strong
                # key; a common year shared by many pages is only a mild signal.
                best_idf = max(_idf_for_token(state, n) for n in matched)
                boosts[pid] = boosts.get(pid, 0.0) + NUMBER_MATCH_WEIGHT * min(
                    1.0, best_idf / RARE_NUMBER_IDF
                )
        if anchors:
            page_terms = (
                state.page_term_sets[doc_idx]
                if doc_idx < len(state.page_term_sets)
                else set()
            )
            for term, idf in anchors:
                scale = min(1.0, idf / ANCHOR_IDF_NORM)
                if term in title_terms:
                    # Anchor in title is a near-definitive match — boost harder.
                    boosts[pid] = boosts.get(pid, 0.0) + ANCHOR_TITLE_MATCH_WEIGHT * scale
                elif term in page_terms:
                    boosts[pid] = boosts.get(pid, 0.0) + ANCHOR_MATCH_WEIGHT * scale
    return boosts


# ---------------------------------------------------------------------------
# Literal-evidence reranking
# ---------------------------------------------------------------------------

def _normalize_scores(items: Dict[int, float]) -> Dict[int, float]:
    if not items:
        return {}
    vals = np.asarray(list(items.values()), dtype=np.float32)
    lo, hi = float(vals.min()), float(vals.max())
    if hi <= lo + 1e-9:
        return {k: 1.0 for k in items}
    scale = hi - lo
    return {k: (float(v) - lo) / scale for k, v in items.items()}


def _number_tokens_from_query(query: str) -> Tuple[set, set]:
    raw = raw_tokenize(query)
    exact = {t for t in raw if t.isdigit() and len(t) >= 2}
    expanded = set(exact)
    for m in DECADE_RE.finditer(query.lower()):
        base = int(m.group(1))
        for year in range(base, base + 10):
            expanded.add(str(year))
    return exact, expanded


def _relative_year_signal(query: str, page_numbers: set) -> float:
    if not page_numbers:
        return 0.0
    years = sorted(int(x) for x in page_numbers if x.isdigit() and 1500 <= int(x) <= 2099)
    if len(years) < 2:
        return 0.0
    year_set = set(years)
    q = query.lower()
    score = 0.0
    if "two years before" in q or "2 years before" in q:
        if any((y + 2) in year_set for y in years):
            score += 1.0
    if "year before" in q:
        if any((y + 1) in year_set for y in years):
            score += 0.8
    if "decades after" in q:
        if any((b - a) >= 20 for a in years for b in years if b > a):
            score += 0.6
    return min(1.0, score)


def _query_content_terms(query: str) -> List[str]:
    return [t for t in tokenize(query) if t not in GENERIC_QUERY_TERMS]


def _query_facets(query: str) -> List[List[str]]:
    """Split a query into comma/"and"-separated facets, keeping each facet's
    content (non-generic) terms.

    Multi-part queries ("What links A, B, and C?") have gold pages that each
    cover one or more facets, so rewarding pages that span MULTIPLE facets
    targets these high-relevance-count queries directly.
    """
    facets: List[List[str]] = []
    for part in FACET_SPLIT_RE.split(query.lower()):
        terms = [t for t in tokenize(part) if t not in GENERIC_QUERY_TERMS]
        if terms:
            facets.append(terms)
    return facets


def _query_phrases(query: str) -> List[str]:
    raw = [
        t
        for t in raw_tokenize(query)
        if t not in {"which", "who", "what", "when", "where", "how"}
    ]
    phrases: List[str] = []
    seen: set = set()
    for n in (4, 3, 2):
        for i in range(0, len(raw) - n + 1):
            seq = raw[i : i + n]
            content_count = sum(1 for t in seq if t not in GENERIC_QUERY_TERMS)
            if content_count < 2:
                continue
            phrase = " ".join(seq)
            if phrase not in seen:
                phrases.append(phrase)
                seen.add(phrase)
    return phrases[:28]


def _literal_evidence_scores(
    state: RetrievalArtifacts,
    query: str,
    candidate_pids: Sequence[int],
) -> Dict[int, float]:
    base_terms = _query_content_terms(query)
    expanded_terms = expand_query_tokens(base_terms)
    exact_nums, expanded_nums = _number_tokens_from_query(query)
    phrases = _query_phrases(query)
    facets = _query_facets(query)
    multi_facet = len(facets) >= 2

    term_weights: Dict[str, float] = {}
    for t in expanded_terms:
        if t in GENERIC_QUERY_TERMS:
            continue
        w = _idf_for_token(state, t)
        w = w * 1.35 if t in base_terms else w * 0.55
        term_weights[t] = max(term_weights.get(t, 0.0), min(7.0, max(0.8, w)))

    denom = sum(term_weights.values()) or 1.0
    scores: Dict[int, float] = {}
    for pid in candidate_pids:
        doc_idx = state.pid_to_idx.get(int(pid))
        if doc_idx is None:
            continue
        terms = (
            state.page_term_sets[doc_idx] if doc_idx < len(state.page_term_sets) else set()
        )
        title_terms = (
            state.title_term_sets[doc_idx] if doc_idx < len(state.title_term_sets) else set()
        )
        text = state.page_texts[doc_idx] if doc_idx < len(state.page_texts) else ""
        nums = state.page_numbers[doc_idx] if doc_idx < len(state.page_numbers) else set()

        rare_hit_weight = sum(w for t, w in term_weights.items() if t in terms)
        rare_coverage = rare_hit_weight / denom
        # Rare query terms that appear in the page title are a much stronger
        # discriminator than the same terms appearing only in the body.
        title_rare_hit = sum(w for t, w in term_weights.items() if t in title_terms)
        title_coverage = title_rare_hit / denom
        exact_terms = [t for t in base_terms if t not in GENERIC_QUERY_TERMS]
        exact_coverage = (
            sum(1 for t in exact_terms if t in terms) / max(1, len(exact_terms))
            if exact_terms
            else 0.0
        )

        phrase_hits = sum(1 for ph in phrases if text and ph in text)
        phrase_score = min(1.0, phrase_hits / 3.0)

        number_score = 0.0
        matched_nums = (nums & exact_nums) or (nums & expanded_nums)
        if matched_nums:
            # Rarity-weighted: a long, near-unique number (e.g. an exact
            # population count) is a decisive key; a common year is a weak one.
            best_idf = max(_idf_for_token(state, n) for n in matched_nums)
            number_score = min(1.0, best_idf / RARE_NUMBER_IDF)
            if exact_nums and (nums & exact_nums):
                number_score = min(1.0, number_score + 0.1)

        facet_score = 0.0
        if multi_facet:
            covered = sum(1 for f in facets if any(t in terms for t in f))
            facet_score = covered / len(facets)

        rel_year = _relative_year_signal(query, nums)

        evidence = (
            RARE_COVERAGE_WEIGHT * rare_coverage
            + TITLE_EVIDENCE_WEIGHT * title_coverage
            + EXACT_COVERAGE_WEIGHT * exact_coverage
            + PHRASE_WEIGHT * phrase_score
            + NUMBER_EVIDENCE_WEIGHT * number_score
            + RELATIVE_YEAR_WEIGHT * rel_year
            + FACET_COVERAGE_WEIGHT * facet_score
        )
        scores[int(pid)] = float(evidence)
    return scores


def _rerank_with_literal_evidence(
    state: RetrievalArtifacts,
    query: str,
    fused: List[Tuple[int, float]],
) -> List[Tuple[int, float]]:
    if not fused:
        return fused
    head = fused[:RERANK_TOP_N]
    tail = fused[RERANK_TOP_N:]
    pids = [pid for pid, _ in head]
    evidence = _literal_evidence_scores(state, query, pids)
    base_norm = _normalize_scores(dict(head))
    evidence_norm = _normalize_scores(evidence)
    reranked: List[Tuple[int, float]] = []
    for pid, original_score in head:
        score = (
            BASE_IN_RERANK * base_norm.get(pid, 0.0)
            + LITERAL_EVIDENCE_WEIGHT * evidence_norm.get(pid, 0.0)
        )
        score += 0.015 * original_score
        reranked.append((pid, score))
    reranked.sort(key=lambda x: (-x[1], x[0]))
    return reranked + tail


# ---------------------------------------------------------------------------
# Cross-encoder reranking (final stage)
# ---------------------------------------------------------------------------

def _cross_encoder_rerank(
    state: RetrievalArtifacts,
    query: str,
    fused: List[Tuple[int, float]],
) -> List[Tuple[int, float]]:
    if not CROSS_ENCODER_ENABLED or not fused:
        return fused
    n = min(CROSS_ENCODE_TOP_N, len(fused))
    head = fused[:n]
    tail = fused[n:]
    pairs: List[Tuple[str, str]] = []
    for pid, _ in head:
        doc_idx = state.pid_to_idx.get(int(pid))
        if doc_idx is None:
            pairs.append((query, ""))
            continue
        title = state.titles[doc_idx] if doc_idx < len(state.titles) else ""
        body = state.page_texts[doc_idx] if doc_idx < len(state.page_texts) else ""
        pairs.append((query, f"{title}. {body}"))
    ce_scores = _get_cross_encoder().predict(
        pairs,
        batch_size=128,
        show_progress_bar=False,
    )

    if not CROSS_ENCODER_BLEND_ENABLED:
        reranked = sorted(
            [(pid, float(s)) for (pid, _), s in zip(head, ce_scores)],
            key=lambda x: -x[1],
        )
        return reranked + tail

    ce_raw: Dict[int, float] = {}
    base_raw: Dict[int, float] = {}
    for (pid, base_score), ce_score in zip(head, ce_scores):
        ce_raw[int(pid)] = float(ce_score)
        base_raw[int(pid)] = float(base_score)

    ce_norm = _normalize_scores(ce_raw)
    base_norm = _normalize_scores(base_raw)

    reranked: List[Tuple[int, float]] = []
    for pid, _ in head:
        pid = int(pid)
        final_score = (
            CROSS_ENCODER_CE_WEIGHT * ce_norm.get(pid, 0.0)
            + CROSS_ENCODER_BASE_WEIGHT * base_norm.get(pid, 0.0)
        )
        reranked.append((pid, float(final_score)))

    reranked.sort(key=lambda x: (-x[1], x[0]))
    return reranked + tail


# ---------------------------------------------------------------------------
# Threshold + final output
# ---------------------------------------------------------------------------

def _threshold_ranked(
    fused: List[Tuple[int, float]], *, max_results: int
) -> List[int]:
    if not fused:
        return []
    best = fused[0][1]
    threshold = max(SCORE_THRESHOLD, best * SCORE_THRESHOLD)
    ranked: List[int] = []
    seen: set = set()
    for pid, score in fused:
        if pid in seen:
            continue
        if len(ranked) >= MIN_RESULTS and score < threshold:
            break
        ranked.append(int(pid))
        seen.add(pid)
        if len(ranked) >= max_results:
            break
    return ranked


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def search_batch(
    queries: List[str],
    *,
    top_k: int = DEFAULT_RETURN_K,
    artifacts_dir: Optional[Path] = None,
) -> List[List[int]]:
    """Return ranked page_id lists, most relevant first."""
    if not queries:
        return []
    state = _ensure_global_state(artifacts_dir)
    query_vectors = embed_queries(queries)
    if query_vectors.size == 0:
        return [[] for _ in queries]

    max_results = max(MIN_RESULTS, int(top_k))
    out: List[List[int]] = []

    for qi, query in enumerate(queries):
        # --- Step 1: Base retrieval ---
        # PRF and per-facet recall expansion were both tested and removed: the
        # multi-part queries are ranking-limited, not recall-limited (the golds
        # are already retrieved), so widening the pool did not help.  See
        # EXPERIMENTS.md.
        dense = _dense_page_scores(state, query_vectors, qi)
        bm25 = _bm25_doc_scores(state, query)
        title = _title_bm25_scores(state, query)

        # --- Step 2: Build candidate pool ---
        prelim_pids = set(dense) | set(bm25) | set(title)
        if len(prelim_pids) > FINAL_POOL:
            prelim = _rrf_fuse(dense, bm25, title, {})[:FINAL_POOL]
            prelim_pids = {pid for pid, _ in prelim}

        # --- Step 3: Auxiliary boosts (title overlap, number overlap) ---
        q_tokens = [
            t for t in expand_query_tokens(tokenize(query))
            if t not in GENERIC_QUERY_TERMS
        ]
        aux = _auxiliary_boosts(state, q_tokens, prelim_pids)

        # --- Step 4: RRF fusion ---
        fused = _rrf_fuse(dense, bm25, title, aux)

        # --- Step 5: Literal-evidence reranking ---
        fused = _rerank_with_literal_evidence(state, query, fused)

        # --- Step 6: Cross-encoder reranking ---
        fused = _cross_encoder_rerank(state, query, fused)

        # --- Step 7: Threshold + output ---
        out.append(_threshold_ranked(fused, max_results=max_results))

    return out
