"""Resolve chunk_id for each question in test_set.json.

test_set.json was written by hand: some entries were confirmed against
real search_test.py output, others are educated guesses (see the
"source" field on each entry) about which act/section number is
correct in THIS dataset. This script is what actually checks those
guesses against your real data and fills in the exact chunk_id --
any question whose act_title/section_no does not match anything in
chunks.jsonl gets flagged instead of silently left wrong.

Reads:  data/processed/chunks.jsonl
        src/evaluation/test_set.json
Writes: src/evaluation/test_set.json  (chunk_id filled in, in place)

Usage:
    python -m src.evaluation.resolve_chunk_ids
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

CHUNKS_PATH = Path("data/processed/chunks.jsonl")
TEST_SET_PATH = Path("src/evaluation/test_set.json")


def normalize(text: str) -> str:
    """Unicode-normalize (NFC) then lowercase, so visually-identical Bangla
    text that happens to be encoded differently (common with complex
    conjuncts) still compares equal."""
    return unicodedata.normalize("NFC", text).lower()


def load_chunks(path: Path) -> list[dict]:
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def find_matches(chunks: list[dict], act_title: str, section_no: str) -> list[dict]:
    """Substring match on act_title (Unicode-normalized, case-insensitive),
    exact match on section_no. Multiple matches are expected when a section
    was split into several chunks during chunking -- all of them are
    equally correct retrieval targets, not a problem to resolve down to one."""
    needle = normalize(act_title)
    return [
        c for c in chunks if needle in normalize(c["act_title"]) and c["section_no"] == section_no
    ]


def available_sections(chunks: list[dict], act_title: str) -> list[str]:
    """All distinct section_no values that exist for this act -- used to help
    debug a NO MATCH by showing what sections actually exist."""
    needle = normalize(act_title)
    seen: list[str] = []
    for c in chunks:
        if needle in normalize(c["act_title"]) and c["section_no"] not in seen:
            seen.append(c["section_no"])
    return seen


def main() -> None:
    chunks = load_chunks(CHUNKS_PATH)
    with TEST_SET_PATH.open("r", encoding="utf-8") as f:
        test_set = json.load(f)

    resolved = 0
    problems: list[str] = []

    for item in test_set:
        matches = find_matches(chunks, item["expected_act_title"], item["expected_section_no"])
        item.pop("chunk_id", None)  # drop the old singular field if this item still has it

        if matches:
            item["chunk_ids"] = [m["chunk_id"] for m in matches]
            resolved += 1
        else:
            item["chunk_ids"] = []
            sections = available_sections(chunks, item["expected_act_title"])
            if sections:
                sample = ", ".join(sections[:15])
                more = " ..." if len(sections) > 15 else ""
                problems.append(
                    f"[{item['id']}] NO MATCH for section {item['expected_section_no']!r} "
                    f"in '{item['expected_act_title']}'. Sections that DO exist: {sample}{more}"
                )
            else:
                problems.append(
                    f"[{item['id']}] act_title '{item['expected_act_title']}' not found at "
                    f"all -- check spelling"
                )

    with TEST_SET_PATH.open("w", encoding="utf-8") as f:
        json.dump(test_set, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Resolved {resolved}/{len(test_set)} questions.")
    if problems:
        print("\nNeeds manual attention -- check the act/section number by hand:")
        for p in problems:
            print(f"  - {p}")


if __name__ == "__main__":
    main()
