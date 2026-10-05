"""Checks call_llm() moves on to the next model when one fails.

Uses a fake Gemini client, so no network call and no API quota is used.
"""

from types import SimpleNamespace

import pytest
from google.genai import errors as genai_errors

from src.generation import generate

CHUNKS = [{"act_title": "Test Act", "section_no": "1", "text": "Some legal text."}]


# Real Gemini errors need a response payload to build; these stand-ins are
# subclasses, so `except genai_errors.ServerError/ClientError` still catches them.
class FakeServerError(genai_errors.ServerError):
    def __init__(self):
        pass


class FakeClientError(genai_errors.ClientError):
    def __init__(self):
        pass


class FakeClient:
    """behaviors: {model_name: answer_text or an Exception to raise}"""

    def __init__(self, behaviors):
        self.calls = []
        self.models = SimpleNamespace(generate_content=self._generate)
        self._behaviors = behaviors

    def _generate(self, model, contents, config):
        self.calls.append(model)
        outcome = self._behaviors[model]
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(text=outcome)


@pytest.fixture(autouse=True)
def three_models(monkeypatch):
    monkeypatch.setattr(generate, "LLM_MODEL_NAMES", ["m1", "m2", "m3"])


def test_first_model_answers_and_no_fallback_used():
    client = FakeClient({"m1": "answer-1"})

    answer, model = generate.call_llm(client, "q?", CHUNKS)

    assert (answer, model) == ("answer-1", "m1")
    assert client.calls == ["m1"]


def test_overloaded_first_model_falls_back_to_second():
    client = FakeClient({"m1": FakeServerError(), "m2": "answer-2"})

    answer, model = generate.call_llm(client, "q?", CHUNKS)

    assert (answer, model) == ("answer-2", "m2")
    assert client.calls == ["m1", "m2"]


def test_quota_error_then_overload_falls_through_to_third():
    client = FakeClient({"m1": FakeClientError(), "m2": FakeServerError(), "m3": "answer-3"})

    answer, model = generate.call_llm(client, "q?", CHUNKS)

    assert (answer, model) == ("answer-3", "m3")
    assert client.calls == ["m1", "m2", "m3"]


def test_all_models_failing_raises_the_last_error():
    client = FakeClient({"m1": FakeServerError(), "m2": FakeServerError(), "m3": FakeClientError()})

    with pytest.raises(genai_errors.ClientError):
        generate.call_llm(client, "q?", CHUNKS)

    assert client.calls == ["m1", "m2", "m3"]
