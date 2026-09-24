"""Evaluate retrieval quality WITH a cross-encoder reranking step, for
comparison against the plain-FAISS baseline in evaluate.py.

Stage 1 (recall):    FAISS retrieves RERANK_CANDIDATES (50) nearest chunks
                      using the bi-encoder (bge-m3) embeddings -- same as
                      evaluate.py, just with a wider net.
Stage 2 (precision):  A cross-encoder (bge-reranker-v2-m3) re-scores each
                      of those 50 candidates against the query using full
                      cross-attention -- slower but more accurate than the
                      bi-encoder's separate embeddings. Candidates are
                      re-sorted by this score; only the top TOP_K are kept.

Reads:  data/processed/embeddings.faiss
        data/processed/chunk_ids.json
        data/processed/chunks.jsonl
        src/evaluation/test_set.json
Writes: src/evaluation/eval_results_reranked.json

Usage:
    python -m src.evaluation.evaluate_rerank
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer
from tqdm import tqdm

INDEX_PATH = Path("data/processed/embeddings.faiss")
CHUNK_IDS_PATH = Path("data/processed/chunk_ids.json")
CHUNKS_PATH = Path("data/processed/chunks.jsonl")
TEST_SET_PATH = Path("src/evaluation/test_set.json")
RESULTS_PATH = Path("src/evaluation/eval_results_reranked.json")

EMBED_MODEL_NAME = "BAAI/bge-m3"
RERANK_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
RERANK_CANDIDATES = 50  # how many FAISS results feed into the reranker
TOP_K = 10  # how many reranked results we keep / report recall@k for
RECALL_KS = (1, 3, 5, 10)


def load_chunk_lookup(path: Path) -> dict[str, dict]:
    """Read chunks.jsonl into a dict keyed by chunk_id, for text lookup."""
    lookup: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunk = json.loads(line)
                lookup[chunk["chunk_id"]] = chunk
    return lookup


def find_rank(retrieved_chunk_ids: list[str], expected_chunk_ids: list[str]) -> int | None:
    """1-based rank of the first retrieved chunk that is in expected_chunk_ids,
    or None if none of them appear at all."""
    expected = set(expected_chunk_ids)
    for rank, cid in enumerate(retrieved_chunk_ids, start=1):
        if cid in expected:
            return rank
    return None


def evaluate_question(
    question: dict,
    embed_model: SentenceTransformer,
    reranker: CrossEncoder,
    index: faiss.Index,
    chunk_ids: list[str],
    chunk_lookup: dict[str, dict],
) -> dict:
    """Stage 1: FAISS wide net. Stage 2: cross-encoder rerank down to TOP_K."""
    query = question["question"]

    query_vector = embed_model.encode([query], normalize_embeddings=True)
    query_vector = np.asarray(query_vector, dtype="float32")
    _, row_indices = index.search(query_vector, RERANK_CANDIDATES)
    candidate_ids = [chunk_ids[row] for row in row_indices[0]]

    pairs = [(query, chunk_lookup[cid]["text"]) for cid in candidate_ids]
    scores = reranker.predict(pairs, batch_size=16)

    scored = sorted(zip(scores, candidate_ids, strict=True), reverse=True)
    retrieved = [cid for _, cid in scored][:TOP_K]

    rank = find_rank(retrieved, question["chunk_ids"])

    return {
        "id": question["id"],
        "question": question["question"],
        "rank": rank,
        "reciprocal_rank": (1 / rank) if rank else 0.0,
        "retrieved": retrieved,
    }


def compute_metrics(results: list[dict]) -> dict:
    n = len(results)
    metrics = {"num_questions": n}
    for k in RECALL_KS:
        hits = sum(1 for r in results if r["rank"] is not None and r["rank"] <= k)
        metrics[f"recall@{k}"] = hits / n
    metrics["mrr"] = sum(r["reciprocal_rank"] for r in results) / n
    return metrics


def main() -> None:
    with TEST_SET_PATH.open("r", encoding="utf-8") as f:
        test_set = json.load(f)
    with CHUNK_IDS_PATH.open("r", encoding="utf-8") as f:
        chunk_ids = json.load(f)
    chunk_lookup = load_chunk_lookup(CHUNKS_PATH)

    index = faiss.read_index(str(INDEX_PATH))

    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    embed_model.max_seq_length = 512

    reranker = CrossEncoder(RERANK_MODEL_NAME, max_length=512)

    results = [
        evaluate_question(q, embed_model, reranker, index, chunk_ids, chunk_lookup)
        for q in tqdm(test_set, desc="Evaluating questions")
    ]
    metrics = compute_metrics(results)

    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "per_question": results}, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print("=" * 50)
    print(f"Evaluated {metrics['num_questions']} questions (WITH reranking)")
    print("-" * 50)
    for k in RECALL_KS:
        print(f"Recall@{k:<2}: {metrics[f'recall@{k}']:.3f}")
    print(f"MRR      : {metrics['mrr']:.3f}")
    print("=" * 50)

    failures = [r for r in results if r["rank"] is None]
    if failures:
        print(f"\n{len(failures)} question(s) still NOT found in top-{TOP_K}:")
        for r in failures:
            print(f"  - [{r['id']}] {r['question']}")


if __name__ == "__main__":
    main()
