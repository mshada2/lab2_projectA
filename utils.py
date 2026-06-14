"""Shared paths and text helpers for Section B.

The autograder imports ``main.run`` from this directory.  Keep this module free of
non-standard dependencies so it can also be used by the offline build script.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence

STUDENT_ROOT = Path(__file__).resolve().parent
DATA_DIR = STUDENT_ROOT / "data"
ENTRIES_DIR = DATA_DIR / "Wikipedia Entries"
PUBLIC_QUERIES_PATH = DATA_DIR / "public_queries.json"
ARTIFACTS_DIR = STUDENT_ROOT / "artifacts"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
K_EVAL = 10
DEFAULT_RETURN_K = 50

# Keep the list conservative: remove high-frequency function words, but keep
# content-bearing words such as "where", "when", and "who" out of the scoring
# anyway because they are uninformative for retrieval.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "for", "from",
    "had", "has", "have", "he", "her", "his", "in", "into", "is", "it", "its",
    "of", "on", "or", "she", "that", "the", "their", "there", "they", "this",
    "to", "was", "were", "which", "who", "what", "when", "where", "whose",
    "why", "how", "with", "within", "without", "about", "after", "before",
    "during", "over", "under", "between", "through", "while", "than", "then",
    "also", "only", "most", "more", "less", "one", "two", "three", "first",
    "second", "third", "later", "earlier", "did", "does", "do", "not", "no",
}

TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?|\d[\d,]*")
WHITESPACE_RE = re.compile(r"\s+")


def normalize_page_id(value: Any) -> int:
    """Coerce page_id from JSON (int or numeric string) to int."""
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    raise ValueError(f"Invalid page_id: {value!r}")


def load_public_queries(path: Path | None = None) -> List[Dict[str, Any]]:
    path = path or PUBLIC_QUERIES_PATH
    rows = json.loads(path.read_text(encoding="utf-8"))
    for row in rows:
        row["relevant_page_ids"] = [
            normalize_page_id(pid) for pid in row["relevant_page_ids"]
        ]
    return rows


def iter_entries(entries_dir: Path | None = None) -> Iterator[Dict[str, Any]]:
    """Yield one record per JSON file in the corpus directory, sorted by filename."""
    root = entries_dir or ENTRIES_DIR
    if not root.is_dir():
        raise FileNotFoundError(
            f"Corpus directory not found: {root}. Expected data/Wikipedia Entries/."
        )
    for path in sorted(root.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        data["page_id"] = normalize_page_id(data.get("page_id", path.stem))
        data.setdefault("title", "")
        data.setdefault("content", "")
        yield data


def normalize_space(text: str) -> str:
    return WHITESPACE_RE.sub(" ", str(text)).strip()


def entry_text(record: Dict[str, Any]) -> str:
    title = normalize_space(record.get("title", ""))
    content = normalize_space(record.get("content", ""))
    if title and content:
        return f"{title}. {content}".strip()
    return title or content


def raw_tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric terms, preserving numbers."""
    out: List[str] = []
    for m in TOKEN_RE.finditer(str(text).lower().replace("-", " ")):
        tok = m.group(0).replace(",", "")
        if tok:
            out.append(tok)
    return out


def tokenize(text: str, *, keep_stopwords: bool = False) -> List[str]:
    tokens = raw_tokenize(text)
    if keep_stopwords:
        return tokens
    return [t for t in tokens if t not in STOPWORDS and (len(t) > 1 or t.isdigit())]


def expand_query_tokens(tokens: Sequence[str]) -> List[str]:
    """Return the query tokens unchanged (deduplicated, order-preserving).

    Design note (kept here because it is a deliberate decision, not an omission):
    earlier versions injected a large hand-built synonym table at this point.
    On the deduplicated evaluation set it consistently *lowered* NDCG@10.  The
    corpus is full of near-paraphrase pages that share generic vocabulary
    ("research", "team", "championship", "distribution"); adding such terms to a
    query pulls in distractor pages and dilutes the rare, discriminating terms
    that actually identify the gold page.  We therefore keep query terms literal
    and let IDF weighting (in BM25 and the literal-evidence reranker) reward the
    rare terms instead of expanding toward common ones.
    """
    out: List[str] = []
    seen: set[str] = set()
    for tok in tokens:
        if tok not in seen:
            out.append(tok)
            seen.add(tok)
    return out


def ensure_artifacts_dir() -> Path:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS_DIR
