"""Checks that the /metrics endpoint exists and exposes our /ask metrics.

TestClient is deliberately NOT used as a context manager here: that would
run the server's startup (loading the embedding model and FAISS index),
which /metrics does not need and which would make this test slow.
"""

from fastapi.testclient import TestClient

from src.serving.main import app

client = TestClient(app)


def test_metrics_endpoint_exposes_ask_metrics():
    response = client.get("/metrics")

    assert response.status_code == 200
    body = response.text
    assert "ask_requests_total" in body
    assert "ask_latency_seconds" in body
    assert "retrieval_latency_seconds" in body
    assert "llm_latency_seconds" in body


def test_health_still_works():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
