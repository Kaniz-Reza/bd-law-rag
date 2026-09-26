"""Hybrid retrieval: combine dense (embedding) search with keyword (BM25) search.

FAISS/bge-m3 (dense search) finds chunks that are SEMANTICALLY similar to a
query, even if the exact wording differs. BM25 (keyword search) finds chunks
that LITERALLY contain the query's words. Each catches cases the other
misses -- dense search can under-weight a specific keyword (like "theft")
that BM25 would catch instantly; keyword search struggles with paraphrases
or cross-lingual queries that dense search handles well. Combining both
with Reciprocal Rank Fusion (RRF) gets the benefit of each.

NOTE (found during evaluation): treating FAISS and BM25 as equally trusted
(unweighted RRF) made results WORSE overall on our 38-question test set
(MRR 0.704 -> 0.448) -- most of our questions use common legal phrasing
with no rare/distinctive keyword, so BM25 mostly contributes noise rather
than signal for them. reciprocal_rank_fusion() below now accepts a
`weights` argument so FAISS's ranking can be trusted more than BM25's,
instead of splitting trust 50/50.

Usage (as a library, imported by evaluate_hybrid.py and, later, generate.py):
    from src.retrieval.hybrid_search import build_bm25_index, bm25_search, reciprocal_rank_fusion
"""

from __future__ import annotations

import re
from collections import defaultdict

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    """Split text into lowercase word tokens. \\w+ with Python's default
    Unicode matching handles Bangla and English words alike -- no
    language-specific stemming, just consistent splitting so the same word
    tokenizes the same way in a chunk and in a query."""
    return re.findall(r"\w+", text.lower())


def build_bm25_index(chunks: list[dict]) -> tuple[BM25Okapi, list[str]]:
    """Build an in-memory BM25 index over chunk texts. Returns the index and
    the chunk_id for each row, in matching order (row i of the index <->
    chunk_ids[i]) -- the same pattern chunk_ids.json uses for the FAISS
    index. Building this takes well under a second for ~3,920 chunks (BM25
    has no model weights to load, unlike the embedding model), so it is
    rebuilt fresh each run rather than saved to a file."""
    tokenized_docs = [tokenize(c["text"]) for c in chunks]
    chunk_ids = [c["chunk_id"] for c in chunks]
    bm25 = BM25Okapi(tokenized_docs)
    return bm25, chunk_ids


def bm25_search(query: str, bm25: BM25Okapi, chunk_ids: list[str], top_k: int) -> list[str]:
    """Return the top_k chunk_ids for this query, ranked by BM25 score
    (highest/most relevant first)."""
    scores = bm25.get_scores(tokenize(query))
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [chunk_ids[i] for i in ranked_indices[:top_k]]


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    weights: list[float] | None = None,
    k: int = 60,
    top_k: int = 10,
) -> list[str]:
    """Combine several ranked lists of chunk_ids into one merged ranking,
    using (weighted) Reciprocal Rank Fusion: each chunk's fused score is the
    sum of weight * 1/(k + rank) across every list it appears in (rank is
    1-based; a chunk missing from a list simply contributes 0 for that list,
    no penalty beyond not getting that list's points).

    RRF works on RANKS, not raw scores, which sidesteps the problem that
    FAISS similarity scores and BM25 scores live on completely different,
    incomparable scales -- there's no need to normalize either one. k=60 is
    the standard default from the original RRF paper; it just softens how
    much rank 1 dominates rank 2, etc.

    weights: one multiplier per list in ranked_lists, same order and length.
    Defaults to equal weight (1.0 each) when not given. Use this to trust
    one list's ranking more than another's instead of splitting trust
    50/50 -- e.g. weights=[0.7, 0.3] makes the first list (FAISS) count
    more than the second (BM25), so BM25 can still nudge a chunk upward
    but can no longer outvote a ranking FAISS was already confident about.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights must have the same length as ranked_lists")

    scores: dict[str, float] = defaultdict(float)
    for ranked_list, weight in zip(ranked_lists, weights, strict=True):
        for rank, chunk_id in enumerate(ranked_list, start=1):
            scores[chunk_id] += weight * (1.0 / (k + rank))
    fused = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)
    return fused[:top_k]
