"""Section B entry point.

The autograder calls run(queries) once with all evaluation queries.  Artifacts
must already exist in artifacts/; staff do not rebuild them during grading.
"""
from __future__ import annotations

from typing import List

from index import build_index
from retrieve import search_batch
from utils import DEFAULT_RETURN_K


def run(queries: List[str]) -> List[List[int]]:
    """Rank corpus pages for each query, most relevant first."""
    return search_batch(queries, top_k=DEFAULT_RETURN_K)


def build_offline_index() -> None:
    """Run once locally to create artifacts/ (not timed at grading)."""
    build_index()


if __name__ == "__main__":
    build_offline_index()
    print("Index built under artifacts/. Run: python scripts/eval_public.py")
