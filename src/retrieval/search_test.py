"""Quick sanity check for the embedding + FAISS index built by embed.py.

Reads:  data/processed/chunks.jsonl
        data/processed/chunk_ids.json
        data/processed/embeddings.faiss

For a handful of sample questions (Bangla, English, cross-lingual), embeds
each question with the same model used to build the index, searches for the
top-k nearest chunks, and prints them for a manual look. This is NOT a
formal evaluation (recall@k / MRR come later) -- just a "does this
basically work" check.

Usage:
    python -m src.retrieval.search_test
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = Path("data/processed/chunks.jsonl")
INDEX_PATH = Path("data/processed/embeddings.faiss")
CHUNK_IDS_PATH = Path("data/processed/chunk_ids.json")

MODEL_NAME = "BAAI/bge-m3"
TOP_K = 5

# Edit / add questions here.
SAMPLE_QUERIES = [
    # --- IN-SCOPE: topics that ARE covered by the 30 pilot acts ---
    # Bangla -> Societies Registration Act (English chunk, cross-lingual test)
    "সোসাইটি নিবন্ধনের জন্য কমপক্ষে কতজন সদস্য প্রয়োজন?",
    # English -> same topic, direct-language check
    "How many people are required to register a society under this Act?",
    # Bangla -> Contract Act, 1872
    "চুক্তি ভঙ্গের শাস্তি কী?",
    # Bangla -> Penal Code, 1860 (theft)
    "চুরি করলে সর্বোচ্চ কত বছর জেল হতে পারে?",
    # English -> Divorce Act, 1869
    "What are the grounds for divorce under this Act?",
    # Bangla -> যৌতুক নিরোধ আইন, ২০১৮
    "যৌতুক দাবি করা কি শাস্তিযোগ্য অপরাধ?",
    # --- OUT-OF-SCOPE: topics NOT among the 30 pilot acts (negative test) ---
    # Bangla -> no rent-control act in the pilot set
    "বাড়িওয়ালা কীভাবে ভাড়াটিয়াকে উচ্ছেদ করতে পারবে?",
    # Bangla -> Digital Security Act NOT in the pilot set
    "ডিজিটাল নিরাপত্তা আইনে সাইবার অপরাধের শাস্তি কী?",
    # English -> no Motor Vehicles act in the pilot set
    "What are the traffic rules for overtaking on a highway?",
]


def load_chunk_lookup(path: Path) -> dict[str, dict]:
    """Read chunks.jsonl into a dict keyed by chunk_id, for fast lookup."""
    lookup: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunk = json.loads(line)
                lookup[chunk["chunk_id"]] = chunk
    return lookup


def load_chunk_ids(path: Path) -> list[str]:
    """Read the FAISS-row -> chunk_id mapping saved by embed.py."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def search(
    query: str,
    model: SentenceTransformer,
    index: faiss.Index,
    chunk_ids: list[str],
    chunk_lookup: dict[str, dict],
    top_k: int = TOP_K,
) -> None:
    """Embed one query, search the index, and print the top-k matches."""
    query_vector = model.encode([query], normalize_embeddings=True)
    query_vector = np.asarray(query_vector, dtype="float32")

    scores, row_indices = index.search(query_vector, top_k)

    print(f"\n{'=' * 70}")
    print(f"প্রশ্ন: {query}")
    print("-" * 70)

    for rank, (row, score) in enumerate(zip(row_indices[0], scores[0], strict=True), start=1):
        chunk_id = chunk_ids[row]
        chunk = chunk_lookup[chunk_id]
        print(f"{rank}. [{score:.3f}] {chunk['act_title']} — section {chunk['section_no']}")
        print(f"   {chunk['text'][:200]}...")


def main() -> None:
    chunk_lookup = load_chunk_lookup(CHUNKS_PATH)
    chunk_ids = load_chunk_ids(CHUNK_IDS_PATH)
    index = faiss.read_index(str(INDEX_PATH))

    model = SentenceTransformer(MODEL_NAME)
    model.max_seq_length = 512

    for query in SAMPLE_QUERIES:
        search(query, model, index, chunk_ids, chunk_lookup)


if __name__ == "__main__":
    main()
