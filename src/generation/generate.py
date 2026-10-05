"""Generate a cited answer to a legal question using retrieved chunks + an LLM.

Reads:  data/processed/embeddings.faiss
        data/processed/chunk_ids.json
        data/processed/chunks.jsonl

For a question, retrieves the top-k most relevant chunks and assembles them
into a labelled prompt, and asks Gemini to answer using only that context --
citing the Act and section for each claim, and declining if the context
doesn't actually answer the question.

Retrieval is HYBRID (dense FAISS + keyword BM25, combined via weighted
Reciprocal Rank Fusion) -- see src/retrieval/hybrid_search.py. This
replaced plain-FAISS-only retrieval after evaluation showed FAISS alone can
bury a chunk containing a rare, specific keyword very deep (a "theft" query
ranked the correct Penal Code section at 66th out of 3920 chunks). Across
60 held-out test questions never used to choose FAISS_WEIGHT/BM25_WEIGHT
below, hybrid ties plain FAISS on Recall@1/5/10 (0.750/0.900/0.950 both)
and edges ahead on Recall@3 and MRR -- so this costs nothing measurable on
ordinary questions while fixing a real, demonstrated failure mode.

Uses the Gemini API free tier (no cost, no credit card) -- get a key at
https://aistudio.google.com/app/apikey

Gemini's servers are often overloaded (503) or a model's free-tier daily
quota runs out (429), and that has nothing to do with this code. call_llm()
below therefore tries a list of models in order (LLM_MODEL_NAMES) and moves
on to the next one as soon as one fails, instead of waiting on a busy model.

Usage:
    python -m src.generation.generate "চুরি করলে সর্বোচ্চ কত বছর জেল হতে পারে?"
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.retrieval.hybrid_search import (
    bm25_search,
    build_bm25_index,
    reciprocal_rank_fusion,
)

load_dotenv()

CHUNKS_PATH = Path("data/processed/chunks.jsonl")
INDEX_PATH = Path("data/processed/embeddings.faiss")
CHUNK_IDS_PATH = Path("data/processed/chunk_ids.json")

EMBED_MODEL_NAME = "BAAI/bge-m3"

# Tried in order. If one is overloaded (503) or out of free-tier quota (429,
# tracked per model), the next one is tried straight away.
LLM_MODEL_NAMES = [
    "gemini-3.8-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
]

# Hybrid retrieval settings -- chosen and validated in src/evaluation/
# evaluate_hybrid.py (see that file's docstring for the full history: an
# unweighted blend made results worse; 0.85/0.15 came out best and held up
# across 60 held-out questions never used to pick it).
FAISS_CANDIDATES = 100  # how many FAISS results feed into the fusion step
BM25_CANDIDATES = 100  # how many BM25 results feed into the fusion step
FAISS_WEIGHT = 0.85
BM25_WEIGHT = 0.15
TOP_K = 10  # how many of the FUSED results actually go to the LLM

SYSTEM_PROMPT = """You are a legal-information assistant for Bangladeshi law.
You will be given a question and several excerpts from Bangladeshi acts,
each labelled with its Act name and section number.

Rules:
- Answer using ONLY the information in the excerpts below. Do not add
  facts, numbers, or conditions that are not stated in them.
- You may rephrase and simplify the legal language for clarity, but the
  meaning must stay exactly what the excerpts say.
- Cite the Act and section for every claim you make.
- If the excerpts do not actually answer the question, say clearly that
  the provided acts do not cover this question -- do not guess.
- Answer in the same language the question was asked in (Bangla or
  English)."""


def load_chunk_lookup(path: Path) -> dict[str, dict]:
    """Read chunks.jsonl into a dict keyed by chunk_id."""
    lookup: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunk = json.loads(line)
                lookup[chunk["chunk_id"]] = chunk
    return lookup


def retrieve(
    question: str,
    embed_model: SentenceTransformer,
    index: faiss.Index,
    chunk_ids: list[str],
    chunk_lookup: dict[str, dict],
    bm25: BM25Okapi,
    bm25_chunk_ids: list[str],
    top_k: int = TOP_K,
) -> list[dict]:
    """Hybrid retrieval: run FAISS (dense/semantic) and BM25 (keyword)
    search independently, combine their rankings with weighted Reciprocal
    Rank Fusion (FAISS trusted more, per FAISS_WEIGHT/BM25_WEIGHT), then
    return the full chunk records for the fused top-k.

    Each method alone misses things the other catches: FAISS can under-rank
    a chunk that contains an exact rare keyword (see the module docstring's
    theft/section-379 example); BM25 alone would miss paraphrases and
    cross-lingual matches FAISS handles natively."""
    query_vector = embed_model.encode([question], normalize_embeddings=True)
    query_vector = np.asarray(query_vector, dtype="float32")
    _, row_indices = index.search(query_vector, FAISS_CANDIDATES)
    dense_hits = [chunk_ids[row] for row in row_indices[0]]

    keyword_hits = bm25_search(question, bm25, bm25_chunk_ids, BM25_CANDIDATES)

    fused_ids = reciprocal_rank_fusion(
        [dense_hits, keyword_hits],
        weights=[FAISS_WEIGHT, BM25_WEIGHT],
        top_k=top_k,
    )
    return [chunk_lookup[cid] for cid in fused_ids]


def build_prompt(question: str, chunks: list[dict]) -> str:
    """Turn retrieved chunks into labelled sources, followed by the question."""
    sources = []
    for i, chunk in enumerate(chunks, start=1):
        sources.append(
            f"[Source {i}: {chunk['act_title']}, section {chunk['section_no']}]\n{chunk['text']}"
        )
    sources_block = "\n\n".join(sources)
    return f"{sources_block}\n\nQuestion: {question}"


def call_llm(client: genai.Client, question: str, chunks: list[dict]) -> tuple[str, str]:
    """Send the assembled prompt to Gemini and return (answer_text, model_used).

    Tries each model in LLM_MODEL_NAMES in order. If one is overloaded (503)
    or rejects the request (e.g. 429 quota exhausted -- the free-tier quota
    is tracked per model), it moves straight on to the next one instead of
    waiting and retrying the same busy model. Only if EVERY model fails does
    the last error propagate."""
    prompt = build_prompt(question, chunks)
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        max_output_tokens=2048,
    )

    last_error: Exception | None = None
    for model_name in LLM_MODEL_NAMES:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            )
        except (genai_errors.ServerError, genai_errors.ClientError) as exc:
            last_error = exc
            print(f"  {model_name} unavailable ({type(exc).__name__}) -- trying the next model...")
            continue
        return response.text, model_name

    assert last_error is not None  # LLM_MODEL_NAMES is never empty
    raise last_error


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python -m src.generation.generate "your question"')
        return
    question = sys.argv[1]

    chunk_lookup = load_chunk_lookup(CHUNKS_PATH)
    with CHUNK_IDS_PATH.open("r", encoding="utf-8") as f:
        chunk_ids = json.load(f)
    index = faiss.read_index(str(INDEX_PATH))

    # BM25 has no model weights to load and builds in well under a second
    # for this corpus size, so it's rebuilt fresh on every run rather than
    # persisted to a file.
    bm25, bm25_chunk_ids = build_bm25_index(list(chunk_lookup.values()))

    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    embed_model.max_seq_length = 512

    chunks = retrieve(question, embed_model, index, chunk_ids, chunk_lookup, bm25, bm25_chunk_ids)

    client = genai.Client()  # reads GEMINI_API_KEY from the environment

    try:
        answer, model_used = call_llm(client, question, chunks)
    except (genai_errors.ServerError, genai_errors.ClientError):
        print(
            "\nEvery configured Gemini model was unavailable (overloaded or out of "
            "free-tier quota). This is on Google's side, not your setup -- wait a "
            "while and run the same command again."
        )
        return

    print("=" * 50)
    print(f"প্রশ্ন: {question}")
    print("-" * 50)
    print(answer)
    print("=" * 50)
    print(f"Answered by: {model_used}")
    print("\nSources used:")
    for chunk in chunks:
        print(f"  - {chunk['act_title']}, section {chunk['section_no']}")


if __name__ == "__main__":
    main()
