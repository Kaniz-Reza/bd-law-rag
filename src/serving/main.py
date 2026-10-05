"""FastAPI service exposing the RAG pipeline over HTTP.

Wraps the same hybrid retrieval + Gemini generation pipeline used by
src/generation/generate.py, but as a long-running server instead of a
one-shot CLI command -- so heavy resources (the embedding model, the
FAISS index, the BM25 index) are loaded ONCE at startup and reused for
every request, instead of being reloaded on every single question.

Endpoints:
    GET  /health   -> liveness check, no heavy work
    POST /ask      -> {"question": "..."} -> {"answer": "...", "sources": [...]}
    GET  /metrics  -> Prometheus-format counters and timings (see below)

Usage:
    uvicorn src.serving.main:app --reload
    then open http://127.0.0.1:8000/docs for interactive testing
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager

import faiss
from fastapi import FastAPI, HTTPException, Response
from google import genai
from google.genai import errors as genai_errors
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
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

# --- Metrics -----------------------------------------------------------
# Prometheus scrapes these from GET /metrics. A Counter only ever goes up
# (it counts events); a Histogram records how long things took, bucketed,
# so Grafana can later show averages and percentiles (e.g. "95% of requests
# finished within X seconds").
#
# The bucket edges (seconds) stretch up to 80s on purpose: call_llm() retries
# a failing Gemini call up to 5 times with 2s/4s/8s/16s waits, so one slow
# request can legitimately take 30+ seconds.
SLOW_BUCKETS = (0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 40, 80)

# One counter, split by "outcome" label, instead of three separate counters:
#   success              -> answered normally
#   gemini_server_error  -> Gemini overloaded (the 503s we saw)
#   gemini_client_error  -> Gemini rejected the request, e.g. quota (the 429s)
ASK_REQUESTS = Counter(
    "ask_requests_total",
    "Total /ask requests, split by outcome.",
    ["outcome"],
)
ASK_LATENCY = Histogram(
    "ask_latency_seconds",
    "Total time to handle one /ask request (retrieval + Gemini).",
    buckets=SLOW_BUCKETS,
)
RETRIEVAL_LATENCY = Histogram(
    "retrieval_latency_seconds",
    "Time spent on hybrid retrieval (embedding + FAISS + BM25 + fusion).",
    buckets=SLOW_BUCKETS,
)
LLM_LATENCY = Histogram(
    "llm_latency_seconds",
    "Time spent waiting on Gemini, including internal retries.",
    buckets=SLOW_BUCKETS,
)

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


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    """The 'tally sheet' Prometheus reads. Hidden from /docs because it is
    for machines (Prometheus), not for people calling the API."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


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
    uses, so the API and the CLI can never silently drift apart.

    `with X.time():` starts a stopwatch on entering the block and records the
    elapsed time into histogram X on leaving it -- even if an error is raised
    inside, so failed requests are timed too."""
    with ASK_LATENCY.time():
        with RETRIEVAL_LATENCY.time():
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
            with LLM_LATENCY.time():
                answer = call_llm(resources["client"], request.question, chunks)
        except genai_errors.ServerError as exc:
            # Gemini's servers are transiently overloaded -- call_llm() already
            # retried 5 times internally (tenacity) before giving up.
            ASK_REQUESTS.labels(outcome="gemini_server_error").inc()
            raise HTTPException(
                status_code=503,
                detail="Gemini's servers are temporarily overloaded -- try again shortly.",
            ) from exc
        except genai_errors.ClientError as exc:
            # Covers things like the free tier's daily request quota being
            # exhausted (429 RESOURCE_EXHAUSTED) -- not retryable on a short
            # timescale, so call_llm() does not retry this one; we just report
            # it clearly instead of letting it crash into a bare 500.
            ASK_REQUESTS.labels(outcome="gemini_client_error").inc()
            raise HTTPException(
                status_code=429,
                detail=(
                    "Gemini rejected the request -- often a rate limit or quota "
                    "issue (common on the free tier). Try again later."
                ),
            ) from exc

    ASK_REQUESTS.labels(outcome="success").inc()
    sources = [Source(act_title=c["act_title"], section_no=c["section_no"]) for c in chunks]
    return AskResponse(answer=answer, sources=sources)
