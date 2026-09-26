"""Evaluate hybrid (dense FAISS + keyword BM25) retrieval, using the same
Recall@k / MRR methodology as evaluate.py so numbers stay comparable.

test_set.json holds 60 held-out questions (ts-01-ts-60), sampled from
chunks never used when choosing FAISS_WEIGHT/BM25_WEIGHT below -- so
running evaluate.py (plain FAISS) and this script back to back on this
file is a direct, apples-to-apples generalization check: does hybrid
search still help (or at least not hurt) on questions it never saw, not
just the original 38 it was tuned on?

History on the original 38 (for reference -- not reproducible from this
file anymore, since its content changed): plain FAISS scored Recall@1=0.553,
Recall@5=0.921, Recall@10=0.921, Recall@50=1.000, MRR=0.704. Unweighted RRF
(FAISS_WEIGHT=BM25_WEIGHT=1.0) made every metric worse (MRR -> 0.448).
0.85/0.15 (this version's weights) came closest on that set: Recall@10
tied plain FAISS, MRR=0.603, and the motivating theft/section-379 case
went from rank 66/3920 to rank 1.

On the 60 held-out questions actually in this file, hybrid ties plain
FAISS on Recall@1/5/10 (0.750/0.900/0.950 both) and edges ahead on
Recall@3/MRR (0.883 vs 0.867, 0.819 vs 0.818) -- confirming the weights
generalize rather than being overfit to the 38 they were tuned on.

Reads:  data/processed/chunks.jsonl
        data/processed/embeddings.faiss
        data/processed/chunk_ids.json
        src/evaluation/test_set.json

Usage:
    python -m src.evaluation.evaluate_hybrid
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.retrieval.hybrid_search import (
    bm25_search,
    build_bm25_index,
    reciprocal_rank_fusion,
)

CHUNKS_PATH = Path("data/processed/chunks.jsonl")
INDEX_PATH = Path("data/processed/embeddings.faiss")
CHUNK_IDS_PATH = Path("data/processed/chunk_ids.json")
TEST_SET_PATH = Path("src/evaluation/test_set.json")

MODEL_NAME = "BAAI/bge-m3"
FAISS_CANDIDATES = 100  # how many FAISS results feed into the fusion step
BM25_CANDIDATES = 100  # how many BM25 results feed into the fusion step
TOP_K = 50  # how many the FUSED list keeps -- matches evaluate.py's TOP_K
RECALL_KS = (1, 3, 5, 10, 20, 30, 50)

# Chosen by tuning on the (now-replaced) original 38 questions -- see the
# module docstring for that history. Not re-tuned here on purpose: the point
# of this run is to check whether these already-chosen weights generalize.
FAISS_WEIGHT = 0.85
BM25_WEIGHT = 0.15


def load_chunks(path: Path) -> list[dict]:
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def faiss_search(
    query: str, model: SentenceTransformer, index: faiss.Index, chunk_ids: list[str], top_k: int
) -> list[str]:
    query_vector = model.encode([query], normalize_embeddings=True)
    query_vector = np.asarray(query_vector, dtype="float32")
    _, row_indices = index.search(query_vector, top_k)
    return [chunk_ids[row] for row in row_indices[0]]


def find_rank(retrieved_chunk_ids: list[str], expected_chunk_ids: list[str]) -> int | None:
    expected = set(expected_chunk_ids)
    for rank, cid in enumerate(retrieved_chunk_ids, start=1):
        if cid in expected:
            return rank
    return None


def compute_metrics(results: list[dict]) -> dict:
    metrics = {}
    for k in RECALL_KS:
        hits = sum(1 for r in results if r["rank"] is not None and r["rank"] <= k)
        metrics[f"recall@{k}"] = hits / len(results)
    metrics["mrr"] = sum(r["reciprocal_rank"] for r in results) / len(results)
    return metrics


def main() -> None:
    chunks = load_chunks(CHUNKS_PATH)
    with CHUNK_IDS_PATH.open("r", encoding="utf-8") as f:
        faiss_chunk_ids = json.load(f)
    index = faiss.read_index(str(INDEX_PATH))

    model = SentenceTransformer(MODEL_NAME)
    model.max_seq_length = 512

    print("Building BM25 index...")
    bm25, bm25_chunk_ids = build_bm25_index(chunks)

    with TEST_SET_PATH.open("r", encoding="utf-8") as f:
        test_set = json.load(f)

    results = []
    for question in tqdm(test_set, desc="Evaluating questions"):
        dense_hits = faiss_search(
            question["question"], model, index, faiss_chunk_ids, FAISS_CANDIDATES
        )
        keyword_hits = bm25_search(question["question"], bm25, bm25_chunk_ids, BM25_CANDIDATES)
        fused = reciprocal_rank_fusion(
            [dense_hits, keyword_hits],
            weights=[FAISS_WEIGHT, BM25_WEIGHT],
            top_k=TOP_K,
        )

        rank = find_rank(fused, question["chunk_ids"])
        results.append(
            {
                "id": question["id"],
                "question": question["question"],
                "rank": rank,
                "reciprocal_rank": (1 / rank) if rank else 0.0,
            }
        )

    metrics = compute_metrics(results)
    print("\n" + "=" * 50)
    print(f"Hybrid (FAISS x{FAISS_WEIGHT} + BM25 x{BM25_WEIGHT}) retrieval results:")
    for k in RECALL_KS:
        print(f"  Recall@{k}: {metrics[f'recall@{k}']:.3f}")
    print(f"  MRR: {metrics['mrr']:.3f}")

    failures = [r for r in results if r["rank"] is None or r["rank"] > 10]
    if failures:
        print(f"\n{len(failures)} question(s) still missing from top-10:")
        for r in failures:
            print(f"  [{r['id']}] {r['question']} -- rank: {r['rank']}")


if __name__ == "__main__":
    main()
