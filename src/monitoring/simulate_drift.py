"""Simulate drift in the questions users ask, and check the drift signal catches it.

"Drift" here means: the questions people ask slowly move away from what the
corpus covers. The signal is the cosine similarity of the best-matching chunk
(`top_similarity`, returned by POST /retrieve and recorded in the Prometheus
histogram `retrieval_top1_similarity`). Questions the 30 pilot acts cover score
high; questions about something else score low.

This script plays both roles against the running API:
  phase 1 (normal)  -> everyday legal questions the corpus covers
  phase 2 (drifted) -> questions with nothing to do with the corpus
then prints the average similarity of each phase and a verdict.

/retrieve is used (not /ask), so no Gemini quota is spent. The API records every
call in its metrics, so the Grafana dashboard shows the drop while this runs.

Only the standard library is used, so it runs anywhere Python runs.

Usage (the API must be running, e.g. via `docker compose up`):
    uv run python -m src.monitoring.simulate_drift
    uv run python -m src.monitoring.simulate_drift --per-phase 20 --delay 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable

# A best match below this similarity counts as "the corpus covers this poorly".
# Chosen from real runs: legal questions scored about 0.56-0.70, questions
# unrelated to the corpus about 0.35-0.36. It matches a bucket edge in
# SIMILARITY_BUCKETS (src/serving/main.py), so Grafana can count questions below it.
LOW_SIMILARITY = 0.45

# The drifted phase must score at least this much lower, on average, than the
# normal phase for the verdict to say drift was detected.
DRIFT_DROP = 0.10

# Everyday questions on topics the 30 pilot acts cover (Penal Code, Contract Act,
# Labour Act, Consumer Rights Act, ...), written the way a user would ask them.
NORMAL_QUESTIONS = [
    "মাতৃত্বকালীন ছুটি কত দিনের?",
    "ভেজাল পণ্য বিক্রি করলে কী শাস্তি হয়?",
    "চুরি করলে সর্বোচ্চ কত বছর জেল হতে পারে?",
    "যৌতুক চাইলে কী শাস্তি হয়?",
    "জন্ম নিবন্ধন না করলে কী হয়?",
    "তথ্য অধিকার আইনে কীভাবে তথ্য চাইতে হয়?",
    "পরিবেশ দূষণ করলে কী শাস্তি হয়?",
    "What is the punishment for theft under the Penal Code?",
    "What makes a contract void under the Contract Act?",
    "Who can be appointed guardian of a minor?",
    "On what grounds can a Muslim wife seek dissolution of marriage?",
    "When can a partner bind the firm under the Partnership Act?",
]

# Questions with nothing to do with the corpus: where drift would end up if
# users started asking about something else entirely.
DRIFTED_QUESTIONS = [
    "সেরা বিরিয়ানির রেসিপি কী?",
    "ঢাকা থেকে কক্সবাজার যাওয়ার সবচেয়ে ভালো উপায় কী?",
    "ডায়াবেটিস কমানোর উপায় কী?",
    "বাংলাদেশের সেরা ক্রিকেটার কে?",
    "ফেসবুক পাসওয়ার্ড ভুলে গেলে কী করব?",
    "আজকের আবহাওয়া কেমন থাকবে?",
    "Who won the 2022 FIFA World Cup?",
    "How do I train a neural network in PyTorch?",
    "What is the best smartphone under 30000 taka?",
    "How do I lose weight quickly?",
    "Recommend a good movie to watch tonight.",
    "How does a refrigerator work?",
]


def summarize(similarities: list[float]) -> dict:
    """Average, lowest, highest, and the share of questions below LOW_SIMILARITY."""
    if not similarities:
        raise ValueError("need at least one similarity value")
    return {
        "count": len(similarities),
        "average": sum(similarities) / len(similarities),
        "lowest": min(similarities),
        "highest": max(similarities),
        "low_share": sum(1 for s in similarities if s < LOW_SIMILARITY) / len(similarities),
    }


def drift_detected(normal: dict, drifted: dict) -> bool:
    """True when the drifted phase's average similarity is at least DRIFT_DROP
    lower than the normal phase's."""
    return normal["average"] - drifted["average"] >= DRIFT_DROP


def ask_retrieve(url: str, question: str, timeout: float = 120.0) -> float:
    """POST one question to /retrieve and return its top_similarity."""
    body = json.dumps({"question": question}).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return float(json.load(response)["top_similarity"])


def run_phase(
    title: str,
    questions: list[str],
    count: int,
    ask: Callable[[str], float],
    delay: float,
    sleep: Callable[[float], None] = time.sleep,
) -> list[float]:
    """Send `count` questions (cycling through the list), one at a time, and
    return the similarity of each. `delay` seconds pass between requests so the
    Grafana graph gets a steady stream of points."""
    print(f"\n{title}")
    similarities = []
    for i in range(count):
        question = questions[i % len(questions)]
        similarity = ask(question)
        similarities.append(similarity)
        print(f"  [{i + 1:>2}/{count}] {similarity:.3f}  {question}")
        if i < count - 1:
            sleep(delay)
    return similarities


def print_summary(label: str, summary: dict) -> None:
    print(
        f"  {label:<8} average {summary['average']:.3f} | "
        f"lowest {summary['lowest']:.3f} | highest {summary['highest']:.3f} | "
        f"below {LOW_SIMILARITY}: {summary['low_share']:.0%}"
    )


def main() -> int:
    # Git Bash pipes output through the Windows code page, which cannot print
    # Bangla; switch to UTF-8 so the questions show up correctly.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Simulate question drift against /retrieve.")
    parser.add_argument("--url", default="http://localhost:8000/retrieve")
    parser.add_argument("--per-phase", type=int, default=15, help="questions per phase")
    parser.add_argument("--delay", type=float, default=1.0, help="seconds between requests")
    args = parser.parse_args()

    def ask(question: str) -> float:
        return ask_retrieve(args.url, question)

    try:
        normal = run_phase(
            "Phase 1: normal questions (the corpus covers these)",
            NORMAL_QUESTIONS,
            args.per_phase,
            ask,
            args.delay,
        )
        drifted = run_phase(
            "Phase 2: drifted questions (nothing to do with the corpus)",
            DRIFTED_QUESTIONS,
            args.per_phase,
            ask,
            args.delay,
        )
    except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
        print(f"\nCould not reach {args.url} ({exc}).")
        print("Is the API running? Start it with: docker compose up -d")
        return 1

    normal_summary = summarize(normal)
    drifted_summary = summarize(drifted)

    print("\nSummary")
    print_summary("normal", normal_summary)
    print_summary("drifted", drifted_summary)

    drop = normal_summary["average"] - drifted_summary["average"]
    if drift_detected(normal_summary, drifted_summary):
        print(f"\nVerdict: DRIFT DETECTED (average similarity dropped by {drop:.3f}).")
    else:
        print(
            f"\nVerdict: no drift detected (average dropped by {drop:.3f}, less than {DRIFT_DROP})."
        )
    print("Open Grafana at http://localhost:3000 to see the same drop on the dashboard.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
