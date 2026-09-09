"""Focused retained-result publication checks; no inference or rescoring."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest
from pdfrw import PdfReader

from scripts import build_paper_figures as old
from scripts import build_poc_manuscript as paper

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper"


def read(path):
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def data():
    return {k: read(ROOT / p) for k, p in paper.SOURCES.items()}


def test_retained_inputs_and_generated_manifest():
    m = read(OUT / "manuscript_manifest.json")
    for path, digest in m["source_artifacts"].items():
        assert old.sha(ROOT / path) == digest
    for path, digest in m["inputs"].items():
        assert old.sha(ROOT / path) == digest
    for path, digest in m["outputs"].items():
        assert old.sha(OUT / path) == digest


@pytest.mark.parametrize("key,expected", [("compact", 12), ("prose", 7)])
def test_strict_parser_count_is_not_truthiness_of_a_diagnostic_dictionary(data, key, expected):
    registry = read(OUT / "tables/numerical_registry.json")
    count = sum(x["syntax_recovery"]["strict_parseable"] for x in data[key]["calls"])
    assert count == expected == registry[key + "_strict_json"]["value"]
    assert not any(x["syntax_recovery"]["applied"] for x in data[key]["calls"])


def test_compact_table_exact_values_and_full_denominators(data):
    rows = read(OUT / "tables/compact_comparison.json")
    assert len(rows) == 9
    for row in rows:
        for a in "AB":
            original = next(
                x
                for x in data["compact"]["results"]
                if x["story_id"] == row["story"]
                and x["task"] == row["question"]
                and x["approach"].startswith(a)
            )
            for k, v in original["evaluation"]["full"].items():
                if isinstance(v, (int, float)):
                    assert row[a + "_" + k] == v
            assert row[a + "_predicted"] == len(original["parsed"]["facts"])


def test_published_semantics_parse_failures_and_scores_are_separate(data):
    rows = read(OUT / "tables/published_comparison.json")
    assert rows == data["prose"]["table"]
    assert sum(x["metric_status"] != "scored" for x in rows) == 3
    for row in rows:
        if row["metric_status"] != "scored":
            assert row["predictions"] is None
            assert row["strict_qualified_f1"] is None


def test_cost_counts_unique_calls_not_selected_views(data):
    rows = read(OUT / "tables/allocation.json")
    for key, row in zip(paper.SOURCES, rows, strict=True):
        original = data[key]
        calls = original.get("calls", original.get("rows"))
        assert row["calls"] == len(calls)
        assert row["request_seconds"] == sum(c["request_seconds"] for c in calls)
        assert (
            row["allocated_seconds"]
            == original.get("allocation", original.get("accounting"))["new_allocated_seconds"]
        )
    assert sum(x["calls"] for x in rows[2:]) == 21
    assert sum(x["allocated_seconds"] for x in rows[2:]) == pytest.approx(454.817647)


def test_historical_versions_remain_unchanged(data):
    assert read(OUT / "tables/historical_syntax_recovery.json") == data["compact"]["historical"]
    for key in ["ladder", "ladder_v2"]:
        assert read(OUT / "tables" / (key + ".json")) == data[key]["rows"]


def test_main_figure_facts_and_statuses_match_retained_records():
    _, _, panels, _ = old.load_inputs()
    manifest = read(OUT / "manuscript_manifest.json")
    for fig in manifest["figures"]:
        assert fig["minimum_font_pt"] >= 8.5
        assert fig["width_mm"] == 178 and fig["height_mm"] <= 180
        for edge in fig["displayed_edges"]:
            r = next(x for x in panels[edge["panel"]]["records"] if x["index"] == edge["index"])
            assert edge["fact"] == r["fact"] and edge["status"] == r["status"]
        page = PdfReader(str(OUT / "figures" / (fig["name"] + ".pdf"))).pages[0]
        assert float(page.MediaBox[2]) * 25.4 / 72 == pytest.approx(178, abs=0.001)
        assert float(page.MediaBox[3]) * 25.4 / 72 <= 180
    harbor = manifest["figures"][1]
    assert harbor["display_callouts"] == panels["harbor_B_locations"]["records"][:2]
    assert (
        len([e for e in harbor["displayed_edges"] if e["panel"] == "harbor_B_locations"])
        + len(harbor["display_callouts"])
        == 6
    )


def test_source_words_questions_and_required_errors_are_visible(data):
    captions = (OUT / "FIGURE_CAPTIONS.md").read_text()
    _, _, panels, _ = old.load_inputs()
    for k in ["fable_A", "harbor_A_locations", "alice_B_actions", "alice_B_claims"]:
        assert panels[k]["result"]["question"] in captions
    for name, story, ids in [
        ("figure_1_fable", "fable", ["S1", "S2", "S3", "S4", "S5"]),
        ("figure_3_alice", "alice", ["S1", "S2"]),
    ]:
        text = subprocess.check_output(
            ["pdftotext", str(OUT / "figures" / (name + ".pdf")), "-"], text=True
        )
        normalized = " ".join(text.split())
        for eid in ids:
            assert " ".join(data["prose"]["stories"][story]["evidence"][eid].split()) in normalized
    text = (OUT / "PROOF_OF_CONCEPT.md").read_text()
    assert (
        "sentence-valued" in text
        and "wrong participant" in text
        and "extra terminal closing brace" in text
    )


def test_resolved_numbers_and_citations():
    source = (OUT / "manuscript_source.md").read_text()
    text = (OUT / "PROOF_OF_CONCEPT.md").read_text()
    registry = read(OUT / "tables/numerical_registry.json")
    for key in re.findall(r"\{\{(\w+)\}\}", source):
        assert registry[key]["display"] in text
    assert (
        "{{" not in text and "!TABLE:" not in text and "!FIGURE:" not in text and "[@" not in text
    )
    refs = read(OUT / "reference_sources.json")
    assert set(re.findall(r"\[@(\w+)\]", source)) == {r["id"] for r in refs}
    for r in refs:
        assert r["url"] in text and r["supports"]
    assert "12 parseable outputs from 12 calls" in text and "7 of 9 calls" in text


def streams(page):
    content = page.Contents
    return [x.stream for x in content] if isinstance(content, list) else [content.stream]


def test_supplement_appends_original_pages_without_rewriting_content():
    merged = PdfReader(str(OUT / "SUPPLEMENTARY_MATERIAL.pdf")).pages
    paths = [
        OUT / "supplement_front.pdf",
        ROOT / "reports/PAPER_FIGURES.pdf",
        ROOT / "reports/figures/paper/FULL_OUTPUT_SUPPLEMENT.pdf",
    ]
    expected = [page for p in paths for page in PdfReader(str(p)).pages]
    assert len(merged) == len(expected)
    for actual, original in zip(merged, expected, strict=True):
        assert list(actual.MediaBox) == list(original.MediaBox)
        assert streams(actual) == streams(original)


@pytest.mark.parametrize("filename", ["PROOF_OF_CONCEPT.pdf", "supplement_front.pdf"])
def test_every_page_text_is_in_bounds_and_searchable(filename):
    raw = subprocess.check_output(["pdftotext", "-bbox", str(OUT / filename), "-"])
    root = ET.fromstring(raw)
    ns = {"h": "http://www.w3.org/1999/xhtml"}
    pages = root.findall(".//h:page", ns)
    assert pages
    for page in pages:
        words = page.findall("h:word", ns)
        assert len(words) > 40
        for w in words:
            assert float(w.attrib["xMin"]) >= 20
            assert float(w.attrib["xMax"]) <= float(page.attrib["width"]) - 20
            assert float(w.attrib["yMin"]) >= 15
            assert float(w.attrib["yMax"]) <= float(page.attrib["height"]) - 15
            assert "\ufffd" not in (w.text or "")


def test_pdf_figures_are_vector_forms_not_raster_images():
    reader = PdfReader(str(OUT / "PROOF_OF_CONCEPT.pdf"))
    forms = []
    for page in reader.pages:
        for obj in (page.Resources.XObject or {}).values():
            assert str(obj.Subtype) != "/Image"
            forms.append(obj)
    assert len(forms) == 3


def test_review_word_count_and_scope():
    m = read(OUT / "manuscript_manifest.json")
    assert 3000 <= m["body_word_count_excluding_tables_figures_references_and_code"] <= 4000
    source = (OUT / "PROOF_OF_CONCEPT.md").read_text()
    assert "No new inference was performed" in source
    assert "not by independent human reviewers" in source
    assert "no new inference" in m["scope"]
