"""Generate embeddings for law chunks and build a FAISS search index.

Reads:  data/processed/chunks.jsonl
Writes: data/processed/embeddings.faiss   (the vector index)
        data/processed/chunk_ids.json     (maps FAISS row -> chunk_id, for lookup)

Usage:
    python -m src.retrieval.embed
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CHUNKS_PATH = Path("data/processed/chunks.jsonl")
INDEX_PATH = Path("data/processed/embeddings.faiss")
CHUNK_IDS_PATH = Path("data/processed/chunk_ids.json")

MODEL_NAME = "BAAI/bge-m3"
BATCH_SIZE = 32


def load_chunks(path: Path) -> list[dict]:
    """Read chunks.jsonl into a list of dicts, one per chunk."""
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    logger.info("Loaded %d chunks from %s", len(chunks), path)
    return chunks


def embed_chunks(chunks: list[dict], model_name: str = MODEL_NAME) -> np.ndarray:
    """Encode each chunk's text into a dense vector."""
    model = SentenceTransformer(model_name)

    texts = [chunk["text"] for chunk in chunks]
    model.max_seq_length = 512
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,  # so cosine similarity == plain dot product
    )
    return np.asarray(embeddings, dtype="float32")


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Flat inner-product index. Since vectors are normalized above,
    inner product here is equivalent to cosine similarity."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def save_index(index: faiss.Index, chunk_ids: list[str]) -> None:
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    with CHUNK_IDS_PATH.open("w", encoding="utf-8") as f:
        json.dump(chunk_ids, f, ensure_ascii=False)
    logger.info("Saved index (%d vectors) to %s", index.ntotal, INDEX_PATH)


def main() -> None:
    chunks = load_chunks(CHUNKS_PATH)
    embeddings = embed_chunks(chunks)
    index = build_faiss_index(embeddings)
    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    save_index(index, chunk_ids)


if __name__ == "__main__":
    main()
