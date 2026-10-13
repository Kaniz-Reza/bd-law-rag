"""Checks the drift signal: the similarity of the best FAISS match.

No real model, index or Gemini is used. Tiny fakes stand in for them, so these
tests run in a fraction of a second.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.generation import generate
from src.serving import main

CHUNK_LOOKUP = {
    "c1": {"chunk_id": "c1", "act_title": "Test Act", "section_no": "1", "text": "one"},
    "c2": {"chunk_id": "c2", "act_title": "Test Act", "section_no": "2", "text": "two"},
}


class FakeEmbedModel:
    def encode(self, texts, normalize_embeddings=True):
        return np.array([[1.0, 0.0]])


class FakeIndex:
    """Returns the same shape FAISS does: (scores, row_indices), one row per query."""

    def search(self, query_vector, k):
        return np.array([[0.73, 0.41]]), np.array([[0, 1]])


@pytest.fixture
def fake_keyword_search(monkeypatch):
    monkeypatch.setattr(generate, "bm25_search", lambda *args, **kwargs: ["c2"])


def call_retrieve_with_score():
    return generate.retrieve_with_score(
        "any question",
        FakeEmbedModel(),
        FakeIndex(),
        ["c1", "c2"],
        CHUNK_LOOKUP,
        None,
        ["c1", "c2"],
    )


def test_retrieve_with_score_returns_top_faiss_similarity(fake_keyword_search):
    chunks, top_similarity = call_retrieve_with_score()

    assert top_similarity == pytest.approx(0.73)
    assert {c["chunk_id"] for c in chunks} == {"c1", "c2"}


def test_retrieve_still_returns_only_the_chunks(fake_keyword_search):
    chunks, _ = call_retrieve_with_score()

    plain = generate.retrieve(
        "any question",
        FakeEmbedModel(),
        FakeIndex(),
        ["c1", "c2"],
        CHUNK_LOOKUP,
        None,
        ["c1", "c2"],
    )

    assert plain == chunks


def test_retrieve_endpoint_returns_similarity_and_records_metric(monkeypatch):
    fake_chunks = [{"act_title": "Test Act", "section_no": "7", "text": "seven"}]
    monkeypatch.setattr(main, "retrieve_with_score", lambda *args, **kwargs: (fake_chunks, 0.42))
    for key in ("embed_model", "index", "chunk_ids", "chunk_lookup", "bm25", "bm25_chunk_ids"):
        monkeypatch.setitem(main.resources, key, None)

    client = TestClient(main.app)
    response = client.post("/retrieve", json={"question": "anything"})

    assert response.status_code == 200
    body = response.json()
    assert body["top_similarity"] == pytest.approx(0.42)
    assert body["sources"] == [{"act_title": "Test Act", "section_no": "7"}]
    assert "retrieval_top1_similarity_count" in client.get("/metrics").text
