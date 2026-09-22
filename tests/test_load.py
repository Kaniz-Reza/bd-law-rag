import json
import tempfile
from pathlib import Path

import pytest

from src.ingestion.load import (
    RAW_FILENAME,
    Act,
    act_id_from_url,
    build_act,
    build_dataset,
    build_sections,
    clean_text,
    is_omitted,
    section_key,
    split_section_number,
)

RAW = Path("data/raw") / RAW_FILENAME
URL = "http://bdlaws.minlaw.gov.bd/act-print-99.html"


def make_raw_act(sections, title="The Test Act, 1990", repealed=False, year="1990", url=URL):
    return {
        "act_title": "1" + title,
        "act_no": "IV",
        "act_year": year,
        "sections": sections,
        "source_url": url,
        "language": "english",
        "csv_metadata": {"act_title_from_csv": title, "is_repealed": repealed},
    }


def test_clean_text_removes_footnote_markers():
    assert clean_text("the2[***] Government may 7[ ***] act") == "the Government may act"
    assert clean_text("Act74[text] here") == "Act[text] here"


def test_clean_text_fixes_fake_danda_and_spaces():
    assert clean_text("১\u09f7\u00a0 (১) এই\tআইন") == "১। (১) এই আইন"


def test_clean_text_splits_glued_words_but_keeps_ordinals():
    assert clean_text("called theContract Act") == "called the Contract Act"
    assert clean_text("আইন, ২০০৬নামে অভিহিত") == "আইন, ২০০৬ নামে অভিহিত"
    assert clean_text("১লা জুলাই এবং ৫টি") == "১লা জুলাই এবং ৫টি"


@pytest.mark.parametrize(
    ("text", "number", "body"),
    [
        ("১২। (১) এই আইন", "১২", "(১) এই আইন"),
        ("16A. Notwithstanding this", "16A", "Notwithstanding this"),
        ("১০ক। এই আইন", "১০ক", "এই আইন"),
        ("[৩ক। (১) অন্য", "৩ক", "(১) অন্য"),
        ("15.(1) It shall be lawful", "15", "(1) It shall be lawful"),
        ("(2) The Government may", None, "(2) The Government may"),
        ("10.5 per cent of the amount", None, "10.5 per cent of the amount"),
        ("WHEREAS it is expedient", None, "WHEREAS it is expedient"),
    ],
)
def test_split_section_number(text, number, body):
    assert split_section_number(text) == (number, body)


def test_split_section_number_after_cleaning():
    assert split_section_number(clean_text("১৮৷ (১) জনগণের")) == ("১৮", "(১) জনগণের")
    assert split_section_number(clean_text("22[16A. Notwithstanding")) == (
        "16A",
        "Notwithstanding",
    )


def test_section_key_uses_ascii_digits():
    assert section_key("১৬ক") == "16ক"
    assert section_key("16A") == "16A"
    assert section_key(None) is None


def test_is_omitted():
    assert is_omitted("")
    assert is_omitted("Rep. by the Repealing Act, 1874 (XII of 1874).")
    assert is_omitted("[This section was omitted by Article 3 of the Ordinance, 1973]")
    assert is_omitted("[বিলুপ্ত]")
    assert not is_omitted("উক্ত আইনের section 6 বিলুপ্ত হইবে।")
    assert not is_omitted("The Government may make rules for carrying out this Act.")
    assert not is_omitted("x omitted by " + "long text " * 30)


def test_build_sections_merges_fragments_and_carries_headings():
    blocks = [
        {"section_title": "PRELIMINARY", "section_content": "1. This Act may be called X."},
        {"section_content": "It extends to the whole of Bangladesh."},
        {"section_content": "2. In this Act, words have these meanings."},
        {"section_title": "CHAPTER II", "section_content": "3. The Government may act."},
        {"section_content": "   "},
    ]
    sections, counts = build_sections(7, blocks)
    assert [s.section_key for s in sections] == ["1", "2", "3"]
    assert [s.heading for s in sections] == ["PRELIMINARY", "PRELIMINARY", "CHAPTER II"]
    assert "It extends to the whole" in sections[0].text
    assert sections[0].chunk_id == "7-0"
    assert counts["fragments_merged"] == 1
    assert counts["empty_blocks_dropped"] == 1


def test_build_sections_first_block_without_number():
    sections, _ = build_sections(3, [{"section_content": "WHEREAS it is expedient."}])
    assert len(sections) == 1
    assert sections[0].section_no is None


def test_build_act_uses_clean_csv_title_and_official_url():
    raw = make_raw_act([{"section_content": "1. This Act may be called the Test Act."}])
    act, sections, _ = build_act(raw)
    assert act.title == "The Test Act, 1990"
    assert (act.act_id, act.year, act.n_sections) == (99, 1990, 1)
    assert act.official_url == "http://bdlaws.minlaw.gov.bd/act-99.html"
    assert len(sections) == 1


def test_build_act_repealed_has_no_sections_and_clean_title():
    raw = make_raw_act([], title="The Old Act, 1900 [Repealed]", repealed=True, year="১৯০০")
    act, sections, _ = build_act(raw)
    assert act.is_repealed
    assert act.title == "The Old Act, 1900"
    assert act.year == 1900
    assert sections == []


def test_act_validation():
    ok = {
        "act_id": 1,
        "title": "T",
        "year": 1990,
        "act_no": "1",
        "language": "english",
        "is_repealed": False,
        "source_url": URL,
        "official_url": URL,
        "n_sections": 1,
    }
    assert Act(**ok).year == 1990
    with pytest.raises(ValueError):
        Act(**{**ok, "year": 1500})
    with pytest.raises(ValueError):
        Act(**{**ok, "language": "klingon"})
    with pytest.raises(ValueError):
        Act(**{**ok, "is_repealed": True})
    with pytest.raises(ValueError):
        act_id_from_url("http://example.com/nothing")


def test_build_dataset_writes_bangla_jsonl_and_records_problems():
    good = make_raw_act([{"section_content": "১। (১) এই আইন প্রযোজ্য।"}])
    repealed = make_raw_act([], repealed=True, url="http://x/act-print-5.html")
    broken = {"act_title": "no url"}
    with tempfile.TemporaryDirectory() as tmp:
        raw_path = Path(tmp) / "raw.json"
        raw_path.write_text(json.dumps({"acts": [good, repealed, broken]}), encoding="utf-8")
        stats = build_dataset(raw_path, Path(tmp) / "out")
        acts = (Path(tmp) / "out" / "acts.jsonl").read_text(encoding="utf-8")
        sections = (Path(tmp) / "out" / "sections.jsonl").read_text(encoding="utf-8")
    assert stats["acts_total"] == 3
    assert stats["acts_loaded"] == 2
    assert stats["acts_repealed"] == 1
    assert len(stats["problems"]) == 1
    assert len(acts.splitlines()) == 2
    assert len(sections.splitlines()) == 1
    assert "এই আইন প্রযোজ্য" in sections  # written as real Bangla, not \u escapes


@pytest.mark.skipif(not RAW.exists(), reason="the raw dataset is not in data/raw")
def test_real_dataset():
    with tempfile.TemporaryDirectory() as tmp:
        stats = build_dataset(RAW, Path(tmp))
    assert stats["problems"] == []
    assert stats["acts_total"] == 1484
    assert stats["acts_repealed"] == 203
    assert stats["acts_with_text"] == 1281
    assert stats["sections_numbered"] / stats["sections_total"] > 0.99
