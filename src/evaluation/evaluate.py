"""Evaluate retrieval quality against the hand-written test set.

Reads:  data/processed/embeddings.faiss
        data/processed/chunk_ids.json
        src/evaluation/test_set.json
Writes: src/evaluation/eval_results.json   (per-question detail, for later review)

For every question, embeds it, searches the FAISS index for the top-K
nearest chunks, and checks whether any of the question's known-correct
chunk_ids appears among them -- and if so, at what rank. From these
per-question ranks it computes Recall@1/3/5/10 and MRR across the whole
test set.

Usage:
    python -m src.evaluation.evaluate
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_PATH = Path("data/processed/embeddings.faiss")
CHUNK_IDS_PATH = Path("data/processed/chunk_ids.json")
CHUNKS_PATH = Path("data/processed/chunks.jsonl")
TEST_SET_PATH = Path("src/evaluation/test_set.json")
RESULTS_PATH = Path("src/evaluation/eval_results.json")

MODEL_NAME = "BAAI/bge-m3"
TOP_K = 10  # largest k we report recall@k for
RECALL_KS = (1, 3, 5, 10)


def find_rank(retrieved_chunk_ids: list[str], expected_chunk_ids: list[str]) -> int | None:
    """1-based rank of the first retrieved chunk that is in expected_chunk_ids,
    or None if none of them appear at all in the retrieved list."""
    expected = set(expected_chunk_ids)
    for rank, cid in enumerate(retrieved_chunk_ids, start=1):
        if cid in expected:
            return rank
    return None


def evaluate_question(
    question: dict, model: SentenceTransformer, index: faiss.Index, chunk_ids: list[str]
) -> dict:
    """Embed one question, search the index, and record where (if at all)
    a correct chunk showed up."""
    query_vector = model.encode([question["question"]], normalize_embeddings=True)
    query_vector = np.asarray(query_vector, dtype="float32")

    _, row_indices = index.search(query_vector, TOP_K)
    retrieved = [chunk_ids[row] for row in row_indices[0]]

    rank = find_rank(retrieved, question["chunk_ids"])

    return {
        "id": question["id"],
        "question": question["question"],
        "rank": rank,
        "reciprocal_rank": (1 / rank) if rank else 0.0,
        "retrieved": retrieved,
    }


def load_chunk_lookup(path: Path) -> dict[str, dict]:
    """Read chunks.jsonl into a dict keyed by chunk_id, for debug lookups."""
    lookup: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunk = json.loads(line)
                lookup[chunk["chunk_id"]] = chunk
    return lookup


def compute_metrics(results: list[dict]) -> dict:
    """Recall@k for each k in RECALL_KS, plus MRR, averaged over all results."""
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

    index = faiss.read_index(str(INDEX_PATH))
    model = SentenceTransformer(MODEL_NAME)
    model.max_seq_length = 512

    results = [evaluate_question(q, model, index, chunk_ids) for q in test_set]
    metrics = compute_metrics(results)

    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "per_question": results}, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print("=" * 50)
    print(f"Evaluated {metrics['num_questions']} questions")
    print("-" * 50)
    for k in RECALL_KS:
        print(f"Recall@{k:<2}: {metrics[f'recall@{k}']:.3f}")
    print(f"MRR      : {metrics['mrr']:.3f}")
    print("=" * 50)

    failures = [r for r in results if r["rank"] is None]
    if failures:
        test_set_by_id = {q["id"]: q for q in test_set}
        chunk_lookup = load_chunk_lookup(CHUNKS_PATH)

        print(f"\n{len(failures)} question(s) NOT found in top-{TOP_K}:")
        for r in failures:
            expected = test_set_by_id[r["id"]]
            print(f"\n  [{r['id']}] {r['question']}")
            print(
                f"    Expected: {expected['expected_act_title']} "
                f"— section {expected['expected_section_no']}"
            )
            print("    Top-3 actually retrieved instead:")
            for rank, cid in enumerate(r["retrieved"][:3], start=1):
                chunk = chunk_lookup.get(cid)
                if chunk:
                    snippet = chunk["text"][:120]
                    print(
                        f"      {rank}. [{cid}] {chunk['act_title']} "
                        f"— section {chunk['section_no']}"
                    )
                    print(f"         {snippet}...")


if __name__ == "__main__":
    main()
