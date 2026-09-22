import json
import tempfile
from pathlib import Path

import pytest

from src.retrieval.chunk import (
    Chunk,
    atomic_units,
    build_chunks,
    carry_over,
    pack_units,
    word_count,
)

ACT = {
    "act_id": 1,
    "title": "The Test Act, 1990",
    "year": 1990,
    "language": "english",
    "official_url": "http://bdlaws.minlaw.gov.bd/act-1.html",
}


def make_section(chunk_id="1-0", text="Short text.", section_no="1", is_omitted=False):
    return {
        "chunk_id": chunk_id,
        "act_id": 1,
        "section_no": section_no,
        "section_key": section_no,
        "heading": None,
        "text": text,
        "is_omitted": is_omitted,
    }


def test_word_count():
    assert word_count("এই আইন প্রযোজ্য হইবে।") == 4
    assert word_count("") == 0


def test_short_text_is_one_unit():
    assert atomic_units("A short sentence.", max_tokens=50) == ["A short sentence."]


def test_atomic_units_splits_on_sentences_first():
    text = "First sentence here. Second sentence here. Third one here."
    units = atomic_units(text, max_tokens=4)
    assert len(units) == 3
    assert all(word_count(u) <= 4 for u in units)


def test_atomic_units_falls_back_when_sentences_have_no_spacing():
    # A table-like run: words are spaced normally, but there is no space
    # after the danda between "rows" (the real Customs Act table pattern).
    row = "এক দুই তিন চার পাঁচ ছয় সাত আট নয় দশ।"
    text = row * 5
    units = atomic_units(text, max_tokens=15)
    assert all(word_count(u) <= 15 for u in units)
    assert len(units) > 1


def test_atomic_units_splits_on_subclause_markers_as_last_resort():
    # One giant "sentence" with no danda/semicolon at all, only (১), (২) markers.
    text = "শর্তাবলী নিম্নরূপ (১) প্রথম শর্ত অনেক লম্বা কথা (২) দ্বিতীয় শর্ত আরও লম্বা কথা (৩) তৃতীয় শর্ত"
    units = atomic_units(text, max_tokens=6)
    assert all(word_count(u) <= 6 for u in units)


def test_atomic_units_word_fallback_always_terminates():
    text = "শব্দ" * 500  # one single glued "word", no punctuation anywhere
    units = atomic_units(text, max_tokens=50)
    assert all(word_count(u) <= 50 for u in units)


def test_carry_over_always_keeps_at_least_the_last_unit():
    units = ["one two", "three four", "five six seven"]
    assert carry_over(units, overlap_tokens=4) == ["five six seven"]
    assert carry_over(units, overlap_tokens=0) == ["five six seven"]
    assert carry_over(units, overlap_tokens=100) == units
    assert carry_over([], overlap_tokens=10) == []


def test_pack_units_repeats_overlap_between_pieces():
    units = ["Alpha bravo.", "Charlie delta.", "Echo foxtrot.", "Golf hotel."]
    pieces = pack_units(units, max_tokens=4, overlap_tokens=2)
    assert len(pieces) >= 2
    assert pieces[0].split()[-2:] == pieces[1].split()[:2]


def test_build_chunks_end_to_end():
    acts = [ACT]
    sections = [
        make_section("1-0", "This is one short section about definitions."),
        make_section("1-1", "Rep. by the Repealing Act, 1874.", is_omitted=True),
        make_section(
            "1-2",
            " ".join(f"Clause {i} says something important." for i in range(1, 30)),
            section_no="2",
        ),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        with open(out / "acts.jsonl", "w", encoding="utf-8") as f:
            for a in acts:
                f.write(json.dumps(a, ensure_ascii=False) + "\n")
        with open(out / "sections.jsonl", "w", encoding="utf-8") as f:
            for s in sections:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        chunks, stats = build_chunks(out, pilot_act_ids=[1], max_tokens=20, overlap_tokens=5)
        written = (out / "chunks.jsonl").read_text(encoding="utf-8").splitlines()

    assert stats["sections_used"] == 2
    assert stats["sections_skipped_omitted"] == 1
    assert stats["sections_split_into_multiple_chunks"] == 1
    assert len(written) == len(chunks)
    assert all(c.act_title == "The Test Act, 1990" for c in chunks)
    assert all(c.word_count <= 20 for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_chunk_rejects_blank_text():
    with pytest.raises(ValueError):
        Chunk(
            chunk_id="x",
            act_id=1,
            act_title="T",
            act_year=1990,
            language="english",
            official_url="http://x",
            section_no="1",
            section_key="1",
            heading=None,
            chunk_index=0,
            n_chunks_in_section=1,
            text="   ",
            word_count=0,
        )


@pytest.mark.skipif(
    not Path("data/processed/sections.jsonl").exists(),
    reason="run src.ingestion.load first to produce data/processed",
)
def test_real_pilot_data():
    from src.config import load_config

    cfg = load_config()
    chunks, stats = build_chunks(
        Path(cfg.data.processed_dir),
        cfg.data.pilot_act_ids,
        cfg.chunking.max_tokens,
        cfg.chunking.overlap_tokens,
    )
    assert stats["pilot_acts"] == 30
    assert stats["chunks_written"] > 3000
    assert stats["words_max"] < 1000
    assert len({c.chunk_id for c in chunks}) == len(chunks)
