"""Actual frozen outputs and report consistency, not authored generation successes."""

import csv
import hashlib
import json
from pathlib import Path

import pytest

from story_projection_onto import compact_story as p
from story_projection_onto.scorer_only import compact_story as scorer
from story_projection_onto.simple_ladder import parse_text

ROOT = Path(__file__).resolve().parents[2]


def test_actual_outputs_separate_parseability_qualifications_and_relevance():
    path = ROOT / "reports/tables/compact_story_results.json"
    if not path.exists():
        pytest.skip("Actual compact-story batch has not run")
    data = json.loads(path.read_text())
    assert len(data["calls"]) == 8
    assert sum(c["parsed"] is not None for c in data["calls"]) == 6
    assert sum(c["response"]["prompt_tokens"] for c in data["calls"]) == 4022
    assert sum(c["response"]["completion_tokens"] for c in data["calls"]) == 2334
    for c in data["calls"]:
        parsed, _, _ = parse_text(c["raw_text"])
        assert parsed == c["parsed"]
        assert json.loads(json.dumps(scorer.evaluate(c["case_id"], parsed))) == c["evaluation"]
        assert c["transport_complete"] and c["response"]["finish_reason"] == "stop"
        assert c["response"]["completion_tokens"] < 2048
        assert not c["registered_result"]
    for c in data["calls"][:2]:
        assert c["evaluation"]["underlying"]["true_positive"] == 12
        assert c["evaluation"]["full"]["true_positive"] == 5
        assert c["evaluation"]["full"]["predicted"] == 14
        assert not any("valid_from" in r or "valid_until" in r for r in c["parsed"]["facts"])
    assert data["calls"][2]["parsed"] is None and data["calls"][5]["parsed"] is None
    for r in data["results"]:
        ev = scorer.evaluate_answer(r["story_id"], r["task"], r["parsed"])
        assert json.loads(json.dumps(ev)) == r["evaluation"]
        if r["approach"].startswith("A"):
            base = next(c for c in data["calls"] if c["case_id"] == r["source_case_id"])
            selected, trace = p.select(base["parsed"], r["story_id"], r["task"])
            assert selected == r["parsed"] and trace == r["selection_trace"]
    assert not data["registered_acceptance"]


def test_csv_and_report_manifest_are_generated_from_canonical_records():
    directory = ROOT / "reports"
    data = json.loads((directory / "tables/compact_story_results.json").read_text())
    rows = list(csv.DictReader((directory / "tables/compact_story_results.csv").open()))
    assert len(rows) == len(data["results"]) == 12
    for row, result in zip(rows, data["results"], strict=True):
        for prefix, key in (("bare", "underlying"), ("complete", "full")):
            for metric in ("precision", "recall", "f1"):
                assert float(row[prefix + "_" + metric]) == result["evaluation"][key][metric]
        assert int(row["predictions"]) == result["evaluation"]["full"]["predicted"]
    manifest = json.loads((directory / "tables/compact_story_manifest.json").read_text())
    assert all(
        hashlib.sha256((directory / f).read_bytes()).hexdigest() == h
        for f, h in manifest["files"].items()
    )
    for previous in ("simple_synthetic_manifest.json", "simple_synthetic_v2_manifest.json"):
        old = json.loads((directory / "tables" / previous).read_text())
        assert all(
            hashlib.sha256((directory / f).read_bytes()).hexdigest() == h
            for f, h in old.items()
        )
