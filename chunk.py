"""Preprocessing and dense-retrieval chunking.

Dense embeddings are strongest on short, focused passages.  We therefore index a
small set of title-weighted chunks per page rather than one very long page vector.
BM25 is built separately over full pages, so the dense side can stay compact.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from utils import entry_text, normalize_space

# Coarse, page-level chunking.  Relevance here is at the PAGE level and pages are
# short, so a near-whole-page embedding beats many small chunks (finer chunking
# was measured to hurt — it adds aggregation noise and over-weights the repeated
# title).  Most pages fit in a single chunk; only long pages spill into a second.
SUMMARY_WORDS = 400
WINDOW_WORDS = 400
WINDOW_STRIDE = 350
MAX_CHUNKS_PER_PAGE = 6


@dataclass
class Chunk:
    page_id: int
    chunk_id: int
    text: str


def _words(text: str) -> List[str]:
    return normalize_space(text).split()


def _make_chunk(title: str, body_words: List[str], chunk_id: int, page_id: int) -> Chunk:
    body = " ".join(body_words).strip()
    # Repeating the title is a cheap way to keep page/entity identity visible to
    # MiniLM even when the passage itself is a mid-page section.
    if title and body:
        text = f"{title}. {body}"
    else:
        text = title or body
    return Chunk(page_id=page_id, chunk_id=chunk_id, text=text[:4500])


def chunk_entry(record: Dict[str, Any]) -> List[Chunk]:
    """Split one corpus entry into a compact set of dense retrieval units."""
    page_id = int(record["page_id"])
    title = normalize_space(record.get("title", ""))
    content = normalize_space(record.get("content", ""))
    full_text = entry_text(record)
    words = _words(content or full_text)

    if not words:
        return [Chunk(page_id=page_id, chunk_id=0, text=title)]

    chunks: List[Chunk] = []
    chunks.append(_make_chunk(title, words[:SUMMARY_WORDS], 0, page_id))

    if len(words) <= SUMMARY_WORDS:
        return chunks

    chunk_id = 1
    start = 0
    # Cover the beginning and several later sections.  The cap prevents very long
    # real Wikipedia pages from dominating artifact size; full-page BM25 still
    # covers their entire text lexically.
    while start < len(words) and len(chunks) < MAX_CHUNKS_PER_PAGE:
        window = words[start : start + WINDOW_WORDS]
        if not window:
            break
        # Avoid duplicating the summary chunk exactly.
        if start != 0:
            chunks.append(_make_chunk(title, window, chunk_id, page_id))
            chunk_id += 1
        start += WINDOW_STRIDE

    return chunks


def chunk_corpus(records: List[Dict[str, Any]]) -> List[Chunk]:
    chunks: List[Chunk] = []
    for record in records:
        chunks.extend(chunk_entry(record))
    return chunks
