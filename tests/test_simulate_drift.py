"""Checks the pure logic of the drift simulation (no API, no network).

The HTTP call is replaced by a tiny fake, so these tests run instantly.
"""

import pytest

from src.monitoring import simulate_drift


def test_summarize_reports_average_range_and_low_share():
    summary = simulate_drift.summarize([0.60, 0.40, 0.50, 0.30])

    assert summary["count"] == 4
    assert summary["average"] == pytest.approx(0.45)
    assert summary["lowest"] == pytest.approx(0.30)
    assert summary["highest"] == pytest.approx(0.60)
    # 0.40 and 0.30 are below LOW_SIMILARITY (0.45)
    assert summary["low_share"] == pytest.approx(0.5)


def test_drift_is_detected_only_when_the_average_drops_enough():
    normal = simulate_drift.summarize([0.60, 0.70])

    big_drop = simulate_drift.summarize([0.35, 0.36])
    tiny_drop = simulate_drift.summarize([0.58, 0.66])

    assert simulate_drift.drift_detected(normal, big_drop) is True
    assert simulate_drift.drift_detected(normal, tiny_drop) is False


def test_run_phase_cycles_through_the_questions_and_waits_between_requests():
    asked = []
    waits = []

    def fake_ask(question):
        asked.append(question)
        return 0.5

    similarities = simulate_drift.run_phase(
        "test phase", ["q1", "q2"], 5, fake_ask, delay=2.0, sleep=waits.append
    )

    assert asked == ["q1", "q2", "q1", "q2", "q1"]
    assert similarities == [0.5] * 5
    # a wait after every request except the last
    assert waits == [2.0] * 4


def test_the_two_question_lists_are_different():
    assert simulate_drift.NORMAL_QUESTIONS
    assert simulate_drift.DRIFTED_QUESTIONS
    assert not set(simulate_drift.NORMAL_QUESTIONS) & set(simulate_drift.DRIFTED_QUESTIONS)
