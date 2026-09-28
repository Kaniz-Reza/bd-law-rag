"""FastAPI service exposing the RAG pipeline over HTTP.

Wraps the same hybrid retrieval + Gemini generation pipeline used by
src/generation/generate.py, but as a long-running server instead of a
one-shot CLI command -- so heavy resources (the embedding model, the
FAISS index, the BM25 index) are loaded ONCE at startup and reused for
every request, instead of being reloaded on every single question.

Endpoints:
    GET  /health  -> liveness check, no heavy work
    POST /ask     -> {"question": "..."} -> {"answer": "...", "sources": [...]}

Usage:
    uvicorn src.serving.main:app --reload
    then open http://127.0.0.1:8000/docs for interactive testing
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager

import faiss
from fastapi import FastAPI, HTTPException
from google import genai
from google.genai import errors as genai_errors
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from src.generation.generate import (
    CHUNK_IDS_PATH,
    CHUNKS_PATH,
    EMBED_MODEL_NAME,
    INDEX_PATH,
    call_llm,
    load_chunk_lookup,
    retrieve,
)
from src.retrieval.hybrid_search import build_bm25_index

# Populated once at startup by `lifespan` below, read by every request --
# this is what avoids reloading the embedding model on every question.
resources: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once when the server starts (before it accepts any requests).
    Loading everything here -- instead of inside the /ask handler -- means
    the cost of loading the model/index is paid once at startup, not on
    every question."""
    resources["chunk_lookup"] = load_chunk_lookup(CHUNKS_PATH)
    with CHUNK_IDS_PATH.open("r", encoding="utf-8") as f:
        resources["chunk_ids"] = json.load(f)
    resources["index"] = faiss.read_index(str(INDEX_PATH))
    resources["bm25"], resources["bm25_chunk_ids"] = build_bm25_index(
        list(resources["chunk_lookup"].values())
    )
    resources["embed_model"] = SentenceTransformer(EMBED_MODEL_NAME)
    resources["embed_model"].max_seq_length = 512
    resources["client"] = genai.Client()  # reads GEMINI_API_KEY from the environment

    yield  # server runs and serves requests here

    resources.clear()


app = FastAPI(title="bd-law-rag", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    """Liveness check -- confirms the process is up. Deliberately does not
    touch the loaded model/index, so it stays fast even under load."""
    return {"status": "ok"}


class AskRequest(BaseModel):
    question: str


class Source(BaseModel):
    act_title: str
    section_no: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """Retrieve + generate, using the resources loaded once at startup --
    the exact same retrieve()/call_llm() functions the CLI (generate.py)
    uses, so the API and the CLI can never silently drift apart."""
    chunks = retrieve(
        request.question,
        resources["embed_model"],
        resources["index"],
        resources["chunk_ids"],
        resources["chunk_lookup"],
        resources["bm25"],
        resources["bm25_chunk_ids"],
    )

    try:
        answer = call_llm(resources["client"], request.question, chunks)
    except genai_errors.ServerError as exc:
        raise HTTPException(
            status_code=503,
            detail="Gemini's servers are temporarily overloaded -- try again shortly.",
        ) from exc

    sources = [Source(act_title=c["act_title"], section_no=c["section_no"]) for c in chunks]
    return AskResponse(answer=answer, sources=sources)
