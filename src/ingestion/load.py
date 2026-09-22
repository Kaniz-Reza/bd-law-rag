"""Load, clean and split the BLAD law dataset into acts and sections.

Reads data/raw/Contextualized_Bangladesh_Legal_Acts.json and writes
data/processed/acts.jsonl and data/processed/sections.jsonl.

Run: python -m src.ingestion.load
"""

import json
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

RAW_FILENAME = "Contextualized_Bangladesh_Legal_Acts.json"
OFFICIAL_URL = "http://bdlaws.minlaw.gov.bd/act-{act_id}.html"
LANGUAGES = {"bengali", "english", "mixed", "unknown"}

_BN_DIGITS = "\u09e6-\u09ef"
_DIGITS = "0-9" + _BN_DIGITS
_BN_TO_ASCII = str.maketrans({chr(0x09E6 + i): str(i) for i in range(10)})

# "৷" (U+09F7) is used as a danda in this data; "।" (U+0964) is the real danda.
_FAKE_DANDA = "\u09f7"
_DANDA = "\u0964"

# Footnote numbers glued to amended text, e.g. "7[ ***]" or "74[text]".
_PLACEHOLDER = re.compile(rf"(?:[{_DIGITS}]+)?\[\s*\*+\s*\]")
_FOOTNOTE_OPEN = re.compile(rf"[{_DIGITS}]+\[")
# Words glued together when HTML tags were removed: "theContract", "২০০৬নামে".
_ENGLISH_GLUE = re.compile(r"(?<=[a-z])(?=[A-Z][a-z])")
_DIGIT_WORD_GLUE = re.compile(rf"(?<=[{_BN_DIGITS}])(?=[\u0985-\u09b9][\u0980-\u09ff]{{2,}})")
_LINE_SPACES = re.compile(r"[ ]*\n[ ]*")
_SPACES = re.compile(r"[ ]{2,}")

# Section numbers: "12.", "১২।", "16A.", "১০ক।" (optionally after a stray "[").
_SECTION_START = re.compile(
    rf"^[\s\[]*(?P<no>[{_DIGITS}]{{1,4}}(?:[A-Z]{{1,2}}|[\u0995-\u09b9]{{1,2}})?)"
    rf"\s*[.{_DANDA}](?!\d)"
)

# Sections that only say "omitted" or "repealed by ...".
_OMITTED_EN = re.compile(
    r"\bRep\.\s+by\b|\bRepealed\s+by\b|\b(?:was|were)\s+omitted\b|\bomitted\s+by\b",
    re.IGNORECASE,
)
# A Bengali section is a stub only if the whole body is the word, e.g. "[বিলুপ্ত]".
_OMITTED_BN = re.compile(r"^[\s\[\(]*(?:বিলুপ্ত|অবলুপ্ত|রহিত)[\s\]\)\.\u0964]*$")
_LEADING_FOOTNOTE = re.compile(r"^\d{1,3}(?=[A-Za-z])")
_REPEALED_TAG = re.compile(r"\[\s*(?:Repealed|রহিত)\s*\]", re.IGNORECASE)
_ACT_ID = re.compile(r"act-print-(\d+)")
MIN_BODY_CHARS = 15
MAX_OMITTED_CHARS = 200


@dataclass(frozen=True)
class Act:
    act_id: int
    title: str
    year: int
    act_no: str
    language: str
    is_repealed: bool
    source_url: str
    official_url: str
    n_sections: int

    def __post_init__(self) -> None:
        if self.act_id <= 0:
            raise ValueError(f"bad act_id: {self.act_id}")
        if not self.title:
            raise ValueError(f"act {self.act_id} has an empty title")
        if not 1700 <= self.year <= 2100:
            raise ValueError(f"act {self.act_id} has an implausible year: {self.year}")
        if self.language not in LANGUAGES:
            raise ValueError(f"act {self.act_id} has unknown language: {self.language}")
        if self.is_repealed and self.n_sections:
            raise ValueError(f"act {self.act_id} is repealed but has sections")


@dataclass(frozen=True)
class Section:
    chunk_id: str
    act_id: int
    index: int
    section_no: str | None
    section_key: str | None
    heading: str | None
    text: str
    is_omitted: bool

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError(f"empty text in {self.chunk_id}")


def clean_text(text: str) -> str:
    """Fix the known quirks of the source text (see the notes in this file)."""
    text = text.replace(_FAKE_DANDA, _DANDA)
    text = text.replace("\u00a0", " ").replace("\t", " ")
    text = _PLACEHOLDER.sub(" ", text)
    text = _FOOTNOTE_OPEN.sub("[", text)
    text = _ENGLISH_GLUE.sub(" ", text)
    text = _DIGIT_WORD_GLUE.sub(" ", text)
    text = _LINE_SPACES.sub("\n", text)
    text = _SPACES.sub(" ", text)
    return text.strip()


def clean_title(title: str) -> str:
    title = _LEADING_FOOTNOTE.sub("", clean_text(title))
    title = _REPEALED_TAG.sub("", title)
    return _SPACES.sub(" ", title).strip()


def split_section_number(text: str) -> tuple[str | None, str]:
    """Return (section number as printed, text after the number)."""
    match = _SECTION_START.match(text)
    if match is None:
        return None, text
    return match.group("no"), text[match.end() :].strip()


def section_key(section_no: str | None) -> str | None:
    """Section number with ASCII digits, so Bangla and English numbers can be compared."""
    if section_no is None:
        return None
    return section_no.translate(_BN_TO_ASCII)


def is_omitted(body: str) -> bool:
    body = body.strip()
    if len(body) < MIN_BODY_CHARS:
        return True
    if len(body) > MAX_OMITTED_CHARS:
        return False
    return bool(_OMITTED_EN.search(body) or _OMITTED_BN.search(body))


def act_id_from_url(url: str) -> int:
    match = _ACT_ID.search(url)
    if match is None:
        raise ValueError(f"cannot find the act id in url: {url!r}")
    return int(match.group(1))


def build_sections(act_id: int, raw_sections: list[dict]) -> tuple[list[Section], Counter]:
    """Turn the raw text blocks of one act into sections.

    The source "sections" are text blocks, not always whole sections: sub-sections
    often come as separate blocks without a number. Those are merged into the
    section before them.
    """
    counts: Counter = Counter()
    heading: str | None = None
    working: list[dict] = []
    for block in raw_sections:
        title = clean_text(block.get("section_title") or "")
        if title:
            heading = title
        content = clean_text(block.get("section_content") or "")
        if not content:
            counts["empty_blocks_dropped"] += 1
            continue
        number, _ = split_section_number(content)
        if number is None and working:
            working[-1]["parts"].append(content)
            counts["fragments_merged"] += 1
            continue
        working.append({"no": number, "heading": heading, "parts": [content]})

    sections = []
    for index, item in enumerate(working):
        text = "\n".join(item["parts"])
        body = split_section_number(text)[1]
        sections.append(
            Section(
                chunk_id=f"{act_id}-{index}",
                act_id=act_id,
                index=index,
                section_no=item["no"],
                section_key=section_key(item["no"]),
                heading=item["heading"],
                text=text,
                is_omitted=is_omitted(body),
            )
        )
    return sections, counts


def build_act(raw: dict) -> tuple[Act, list[Section], Counter]:
    act_id = act_id_from_url(raw["source_url"])
    meta = raw.get("csv_metadata") or {}
    is_repealed = bool(meta.get("is_repealed"))
    sections, counts = build_sections(act_id, raw.get("sections") or [])
    act = Act(
        act_id=act_id,
        title=clean_title(meta.get("act_title_from_csv") or raw["act_title"]),
        year=int(raw["act_year"]),
        act_no=clean_text(str(raw.get("act_no", ""))),
        language=raw.get("language", "unknown"),
        is_repealed=is_repealed,
        source_url=raw["source_url"],
        official_url=OFFICIAL_URL.format(act_id=act_id),
        n_sections=len(sections),
    )
    return act, sections, counts


def write_jsonl(path: Path, records: list) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def build_dataset(raw_path: Path, out_dir: Path, exclude_repealed: bool = True) -> dict:
    """Read the raw file, write acts.jsonl and sections.jsonl, return statistics."""
    with open(raw_path, encoding="utf-8-sig") as f:
        raw_acts = json.load(f)["acts"]

    acts: list[Act] = []
    sections: list[Section] = []
    counts: Counter = Counter()
    problems: list[str] = []
    for raw in raw_acts:
        try:
            act, act_sections, act_counts = build_act(raw)
        except (KeyError, ValueError) as error:
            problems.append(f"{raw.get('source_url', '?')}: {error}")
            continue
        acts.append(act)
        counts.update(act_counts)
        if not (act.is_repealed and exclude_repealed):
            sections.extend(act_sections)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "acts.jsonl", acts)
    write_jsonl(out_dir / "sections.jsonl", sections)

    lengths = sorted(len(s.text) for s in sections)
    return {
        "acts_total": len(raw_acts),
        "acts_loaded": len(acts),
        "acts_repealed": sum(a.is_repealed for a in acts),
        "acts_with_text": sum(a.n_sections > 0 for a in acts),
        "languages": dict(Counter(a.language for a in acts)),
        "sections_total": len(sections),
        "sections_numbered": sum(s.section_no is not None for s in sections),
        "sections_omitted": sum(s.is_omitted for s in sections),
        "fragments_merged": counts["fragments_merged"],
        "empty_blocks_dropped": counts["empty_blocks_dropped"],
        "chars_median": lengths[len(lengths) // 2] if lengths else 0,
        "chars_p99": lengths[int(len(lengths) * 0.99)] if lengths else 0,
        "chars_max": lengths[-1] if lengths else 0,
        "problems": problems,
    }


def print_report(stats: dict) -> None:
    print(f"Acts in file:            {stats['acts_total']}")
    print(f"Acts loaded:             {stats['acts_loaded']}")
    print(f"  repealed (no text):    {stats['acts_repealed']}")
    print(f"  with text:             {stats['acts_with_text']}")
    print(f"  languages:             {stats['languages']}")
    print(f"Sections written:        {stats['sections_total']}")
    print(f"  with a section number: {stats['sections_numbered']}")
    print(f"  flagged as omitted:    {stats['sections_omitted']}")
    print(f"Fragments merged:        {stats['fragments_merged']}")
    print(f"Empty blocks dropped:    {stats['empty_blocks_dropped']}")
    print(
        f"Section length (chars):  median {stats['chars_median']}, "
        f"p99 {stats['chars_p99']}, max {stats['chars_max']}"
    )
    print(f"Problems:                {len(stats['problems'])}")
    for problem in stats["problems"][:10]:
        print(f"  - {problem}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    from src.config import load_config  # imported here so tests do not need the config

    cfg = load_config()
    stats = build_dataset(
        Path(cfg.data.raw_dir) / RAW_FILENAME,
        Path(cfg.data.processed_dir),
        cfg.data.exclude_repealed,
    )
    print_report(stats)


if __name__ == "__main__":
    main()
