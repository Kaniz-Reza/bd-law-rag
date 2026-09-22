"""Split cleaned law sections into retrieval-sized chunks.

Reads data/processed/acts.jsonl and data/processed/sections.jsonl (written by
src.ingestion.load), keeps only the pilot acts, splits each section into
overlapping chunks of about chunking.max_tokens words each, and writes
data/processed/chunks.jsonl.

Run: python -m src.retrieval.chunk
"""

import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

# Cut points, tried in order from most natural to most desperate. Some law
# sections are tables or semicolon-separated lists flattened into plain text
# with no space after the punctuation, so later levels ignore spacing.
_LEVEL_PATTERNS = [
    re.compile(r"(?<=[।.!?])\s+"),  # 0: whole sentences (danda/period + a space)
    re.compile(r"(?<=[।.!?;])"),  # 1: same marks, or a semicolon, spaced or not
    re.compile(r"(?=[(（][০-৯০-৯0-9]+[ক-হA-Za-z]{0,2}[)）])"),  # 2: before "(১)" / "(a)"
]


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    act_id: int
    act_title: str
    act_year: int
    language: str
    official_url: str
    section_no: str | None
    section_key: str | None
    heading: str | None
    chunk_index: int
    n_chunks_in_section: int
    text: str
    word_count: int

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError(f"empty chunk text in {self.chunk_id}")


def word_count(text: str) -> int:
    """Approximate token count: each whitespace-separated word counts as one."""
    return len(text.split())


def atomic_units(text: str, max_tokens: int, level: int = 0) -> list[str]:
    """Break text into pieces, none longer than max_tokens words.

    Tries the most natural break point first (whole sentences). If a piece
    is still too big — a table row or a semicolon list with no spacing —
    it re-splits just that piece with a more aggressive pattern, and a
    plain word count is the last-resort fallback that always succeeds.
    """
    if word_count(text) <= max_tokens:
        return [text] if text.strip() else []
    if level >= len(_LEVEL_PATTERNS):
        words = text.split()
        return [" ".join(words[i : i + max_tokens]) for i in range(0, len(words), max_tokens)]
    parts = [p.strip() for p in _LEVEL_PATTERNS[level].split(text) if p.strip()]
    if len(parts) <= 1:
        return atomic_units(text, max_tokens, level + 1)
    units: list[str] = []
    for part in parts:
        units.extend(atomic_units(part, max_tokens, level + 1))
    return units


def carry_over(units: list[str], overlap_tokens: int) -> list[str]:
    """The trailing units of a chunk, repeated at the start of the next one."""
    kept: list[str] = []
    kept_words = 0
    for unit in reversed(units):
        words = word_count(unit)
        if kept and kept_words + words > overlap_tokens:
            break
        kept.insert(0, unit)
        kept_words += words
    return kept


def pack_units(units: list[str], max_tokens: int, overlap_tokens: int) -> list[str]:
    """Group small units back into pieces of about max_tokens words each."""
    if not units:
        return []
    pieces: list[str] = []
    current: list[str] = [units[0]]
    current_words = word_count(units[0])
    for unit in units[1:]:
        unit_words = word_count(unit)
        if current_words + unit_words > max_tokens:
            pieces.append(" ".join(current))
            current = carry_over(current, overlap_tokens)
            current_words = word_count(" ".join(current)) if current else 0
        current.append(unit)
        current_words += unit_words
    if current:
        pieces.append(" ".join(current))
    return pieces


def chunk_section(section: dict, act: dict, max_tokens: int, overlap_tokens: int) -> list[Chunk]:
    text = section["text"]
    if word_count(text) <= max_tokens:
        pieces = [text]
    else:
        units = atomic_units(text, max_tokens)
        pieces = pack_units(units, max_tokens, overlap_tokens)
    return [
        Chunk(
            chunk_id=f"{section['chunk_id']}-{i}" if len(pieces) > 1 else section["chunk_id"],
            act_id=act["act_id"],
            act_title=act["title"],
            act_year=act["year"],
            language=act["language"],
            official_url=act["official_url"],
            section_no=section["section_no"],
            section_key=section["section_key"],
            heading=section["heading"],
            chunk_index=i,
            n_chunks_in_section=len(pieces),
            text=piece,
            word_count=word_count(piece),
        )
        for i, piece in enumerate(pieces)
    ]


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, records: list) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def build_chunks(
    processed_dir: Path,
    pilot_act_ids: list[int],
    max_tokens: int,
    overlap_tokens: int,
) -> tuple[list[Chunk], dict]:
    acts_by_id = {a["act_id"]: a for a in read_jsonl(processed_dir / "acts.jsonl")}
    sections = read_jsonl(processed_dir / "sections.jsonl")
    pilot_ids = set(pilot_act_ids)

    chunks: list[Chunk] = []
    sections_used = 0
    sections_skipped_omitted = 0
    sections_split = 0
    for section in sections:
        if section["act_id"] not in pilot_ids:
            continue
        if section["is_omitted"]:
            sections_skipped_omitted += 1
            continue
        act = acts_by_id[section["act_id"]]
        section_chunks = chunk_section(section, act, max_tokens, overlap_tokens)
        if len(section_chunks) > 1:
            sections_split += 1
        chunks.extend(section_chunks)
        sections_used += 1

    write_jsonl(processed_dir / "chunks.jsonl", chunks)

    word_counts = sorted(c.word_count for c in chunks)
    n = len(word_counts)
    stats = {
        "pilot_acts": len(pilot_ids),
        "sections_used": sections_used,
        "sections_skipped_omitted": sections_skipped_omitted,
        "sections_split_into_multiple_chunks": sections_split,
        "chunks_written": len(chunks),
        "words_median": word_counts[n // 2] if n else 0,
        "words_p95": word_counts[int(n * 0.95)] if n else 0,
        "words_max": word_counts[-1] if n else 0,
        "chunks_over_max_tokens": sum(1 for w in word_counts if w > max_tokens),
    }
    return chunks, stats


def print_report(stats: dict) -> None:
    print(f"Pilot acts:                     {stats['pilot_acts']}")
    print(f"Sections used:                  {stats['sections_used']}")
    print(f"Sections skipped (omitted):     {stats['sections_skipped_omitted']}")
    print(f"Sections split into >1 chunk:   {stats['sections_split_into_multiple_chunks']}")
    print(f"Chunks written:                 {stats['chunks_written']}")
    print(
        f"Words per chunk:                median {stats['words_median']}, "
        f"p95 {stats['words_p95']}, max {stats['words_max']}"
    )
    print(f"Chunks over the max_tokens budget: {stats['chunks_over_max_tokens']}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    from src.config import load_config

    cfg = load_config()
    _, stats = build_chunks(
        Path(cfg.data.processed_dir),
        cfg.data.pilot_act_ids,
        cfg.chunking.max_tokens,
        cfg.chunking.overlap_tokens,
    )
    print_report(stats)


if __name__ == "__main__":
    main()
