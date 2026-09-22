"""Search the loaded acts by words in the title, to help choose the pilot acts.

Run: python -m src.ingestion.find_acts <word> [<word> ...] [--all]
Needs data/processed/acts.jsonl (create it with: python -m src.ingestion.load).
"""

import argparse
import json
import sys
from pathlib import Path

ACTS_FILE = Path("data/processed/acts.jsonl")


def read_acts(path: Path = ACTS_FILE) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def search(acts: list[dict], words: list[str], include_repealed: bool = False) -> list[dict]:
    """Acts whose title contains every word (case-insensitive), oldest first."""
    wanted = [word.lower() for word in words]
    found = []
    for act in acts:
        if act["is_repealed"] and not include_repealed:
            continue
        title = act["title"].lower()
        if all(word in title for word in wanted):
            found.append(act)
    return sorted(found, key=lambda act: (act["year"], act["act_id"]))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Search acts by title words.")
    parser.add_argument("words", nargs="+", help="words that must all appear in the title")
    parser.add_argument("--all", action="store_true", help="include repealed acts")
    args = parser.parse_args()

    if not ACTS_FILE.exists():
        print(f"{ACTS_FILE} not found. Run: python -m src.ingestion.load")
        return
    found = search(read_acts(), args.words, include_repealed=args.all)
    print(f"{len(found)} act(s) found")
    for act in found:
        note = " [repealed]" if act["is_repealed"] else ""
        print(
            f"id {act['act_id']:>5} | {act['year']} | {act['language']:<7} | "
            f"{act['n_sections']:>4} sections | {act['title']}{note}"
        )


if __name__ == "__main__":
    main()
