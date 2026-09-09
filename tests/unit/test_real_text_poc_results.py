"""Actual-output reporting regressions; not new model executions or reliability tests."""

import csv
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.report_real_text_poc import assessment_key, assessments, render
from story_projection_onto import real_text_poc as p
from story_projection_onto.compact_syntax import recover
from story_projection_onto.scorer_only import real_text_poc as scorer

REPORTS = Path("reports")
RUN = Path("artifacts/restricted/real-text-poc-backup.leI8Rf/run-20260909T140211537585")


def result():
    return json.loads((REPORTS / "tables/real_text_proof_of_concept.json").read_text())


def test_all_nine_actual_outcomes_preserved_and_no_unauthorized_recovery():
    data = result()
    assert data["summary"] == dict(
        executed_calls=9,
        strict_json_calls=7,
        usable_calls=7,
        recovered_calls=0,
        input_tokens=6802,
        output_tokens=2927,
    )
    assert data["rules"] == scorer.frozen_rules()
    assert data["references"] == scorer.references()
    for call in data["calls"]:
        parsed, receipt = recover(call["raw_text"])
        assert parsed == call["parsed"]
        assert receipt["applied"] is False
        assert call["evaluation"] == json.loads(
            json.dumps(scorer.evaluate(call["case_id"], parsed))
        )
        assert call["response"]["finish_reason"] == "stop"
        assert call["response"]["completion_tokens"] < 2048
        assert call["registered_result"] is False
        if call["case_id"] in ("3", "8"):
            assert call["raw_text"].endswith("]}}") and parsed is None
            assert "Extra data" in call["parse_error"]
    assert data["allocation"]["actual_allocated_seconds"] == 10717.909776
    assert data["allocation"]["new_allocated_seconds"] == pytest.approx(205.365216)
    assert data["allocation"]["open_allocations"] == 0
    assert data["allocation"]["open_service_journals"] == 0
    assert data["resources"]["observed_violations"] == []


def test_every_selected_prediction_kept_and_failures_not_empty_successes():
    data = result()
    calls = {c["case_id"]: c for c in data["calls"]}
    assert len(data["results"]) == len(data["table"]) == 12
    expected_f1 = {
        ("alice", "actions", "A"): 2 / 7,
        ("alice", "actions", "B"): 0.4,
        ("fable", "actions", "A"): 0.625,
        ("fable", "actions", "B"): 4 / 13,
    }
    for r, table in zip(data["results"], data["table"], strict=True):
        base = calls[r["source_case_id"]]["parsed"]
        expected = (
            p.select(base, r["story_id"], r["task"])[0]
            if r["approach"][0] == "A" and base
            else base
        )
        assert r["parsed"] == expected
        ev = scorer.evaluate_answer(r["story_id"], r["task"], expected)
        assert json.loads(json.dumps(ev)) == r["evaluation"]
        assert len(r["semantic_assessments"]) == ev["full"]["predicted"]
        if not r["parseable"]:
            assert table["predictions"] is None
            assert table["strict_qualified_f1"] is None
            assert table["semantic_coverage"] is None
            assert table["metric_status"] == "unavailable_parse_failure"
        else:
            assert table["predictions"] == len(r["parsed"]["facts"])
            assert table["strict_qualified_f1"] == pytest.approx(
                expected_f1.get((r["story_id"], r["task"], r["approach"][0]), 0)
            )


def test_manual_component_credit_is_bound_explicit_and_separate(tmp_path):
    data = result()
    catalog = data["manual_assessment_catalog"]
    assert len(catalog) == 30
    calls = {c["case_id"]: c for c in data["calls"]}
    for key, note in catalog.items():
        assert key == assessment_key(note["story_id"], note["fact"])
        assert "not independent human review" in note["author"]
        for ref in note["source_responses"]:
            c = calls[ref["case_id"]]
            assert c["response"]["response_sha256"] == ref["response_sha256"]
            assert c["parsed"]["facts"][ref["fact_index"] - 1] == note["fact"]
    for r in data["results"]:
        assert all(n["author"] != "unassessed" for n in r["semantic_assessments"])
    alice = next(
        r
        for r in data["results"]
        if r["story_id"] == "alice" and r["task"] == "claims" and r["approach"][0] == "B"
    )
    assert alice["evaluation"]["full"]["true_positive"] == 0
    assert alice["semantic_assessments"][0]["covered_concepts"] == [5, 6]
    assert alice["semantic_assessments"][1]["covered_concepts"] == [7]
    assert alice["semantic_assessments"][2]["support"] == "unresolved"
    assert alice["semantic_assessments"][2]["covered_concepts"] == []
    bad = deepcopy(catalog)
    next(iter(bad.values()))["source_responses"][0]["response_sha256"] = "bad"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad))
    with pytest.raises(AssertionError, match="response binding"):
        assessments(deepcopy(data), path)


def test_tables_manifest_sources_and_historical_artifacts_unchanged():
    data = result()
    with (REPORTS / "tables/real_text_proof_of_concept.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 12
    for actual, expected in zip(rows, data["table"], strict=True):
        assert actual == {k: "" if v is None else str(v) for k, v in expected.items()}
    for name in (
        "real_text_proof_of_concept",
        "compact_story",
        "compact_story_v2",
        "simple_synthetic",
        "simple_synthetic_v2",
    ):
        manifest = json.loads((REPORTS / f"tables/{name}_manifest.json").read_text())
        for path, digest in manifest.get("files", manifest).items():
            assert hashlib.sha256((REPORTS / path).read_bytes()).hexdigest() == digest
    md = (REPORTS / "REAL_TEXT_PROOF_OF_CONCEPT.md").read_text()
    html = (REPORTS / "figures/real_text_proof_of_concept.html").read_text()
    # Table rows must immediately follow the separator, not a diagnostic heading.
    assert "|---|---|---|---|---|---|---|\n| alice/actions" in md
    for sid, source in p.stories().items():
        assert "".join(source["evidence"].values()) == source["excerpt"]
        assert hashlib.sha256(source["excerpt"].encode()).hexdigest() == source["excerpt_sha256"]
        for task in p.QUESTIONS[sid]:
            for method in ("A", "B"):
                assert f'id="{sid}-{task}-{method}"' in html
    assert "No mechanically displayable endpoint records" in html
    assert "not independent human review" in md


@pytest.mark.skipif(not RUN.exists(), reason="restricted raw run is intentionally not released")
def test_exact_reproduction_from_immutable_run(tmp_path):
    render(RUN, tmp_path)
    manifest = json.loads((REPORTS / "tables/real_text_proof_of_concept_manifest.json").read_text())
    for path, expected in manifest.items():
        assert hashlib.sha256((tmp_path / path).read_bytes()).hexdigest() == expected
