"""Replay actual v2 outputs and keep historical/derived/new results distinct."""

import csv
import hashlib
import json
from pathlib import Path

import pytest

from story_projection_onto import compact_story_v2 as p
from story_projection_onto.compact_syntax import recover
from story_projection_onto.scorer_only import compact_story_v2 as scorer

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"


def data():
    return json.loads((REPORTS / "tables/compact_story_v2_results.json").read_text())


def test_actual_outputs_replay_without_semantic_repair():
    d = data()
    assert len(d["calls"]) == 12 and len(d["results"]) == 18
    assert sum(len(c["parsed"]["facts"]) for c in d["calls"]) == 134
    for c in d["calls"]:
        parsed, syntax = recover(c["raw_text"])
        assert syntax["strict_parseable"] and not syntax["applied"]
        assert parsed == c["parsed"] and syntax == c["syntax_recovery"]
        assert json.loads(json.dumps(scorer.evaluate(c["case_id"], parsed))) == c["evaluation"]
        assert c["response"]["finish_reason"] == "stop"
        assert c["response"]["completion_tokens"] < 2048
        assert c["response"]["prompt_tokens"] + 2048 < 12288
        assert not c["registered_result"]
    for r in d["results"]:
        ev = scorer.evaluate_answer(r["story_id"], r["task"], r["parsed"])
        assert json.loads(json.dumps(ev)) == r["evaluation"]
        assert ev["full"]["predicted"] == len(r["parsed"]["facts"])
        if r["approach"].startswith("A"):
            base = next(c for c in d["calls"] if c["case_id"] == r["source_case_id"])
            selected, trace = p.select(base["parsed"], r["story_id"], r["task"])
            assert selected == r["parsed"] and trace == r["selection_trace"]
    assert not d["registered_acceptance"]
    assert d["allocation"]["actual_allocated_seconds"] == 10512.54456
    assert round(d["allocation"]["new_allocated_seconds"], 6) == 249.452431
    assert d["allocation"]["open_allocations"] == d["allocation"]["open_service_journals"] == 0


def test_historical_raw_recovered_and_new_categories_keep_their_denominators():
    d = data()
    old = json.loads((REPORTS / "tables/compact_story_results.json").read_text())
    assert d["historical"]["original_results"] == old["results"]
    assert d["historical"]["original_calls"] == old["calls"]
    for r in d["historical"]["derived"]:
        ev = r["evaluation"]["full"]
        assert (ev["true_positive"], ev["predicted"], ev["reference_count"]) == (3, 10, 5)
        assert not r["new_model_output"] and not r["syntax_recovery"]["strict_parseable"]
    rows = list(csv.DictReader((REPORTS / "tables/compact_story_v2_results.csv").open()))
    expected = d["historical"]["original_results"] + d["historical"]["derived"] + d["results"]
    assert len(rows) == len(expected) == 32
    assert [r["category"] for r in rows] == (
        ["historical_raw"] * 12 + ["historical_syntax_recovered"] * 2 + ["new_model"] * 18
    )
    for row, r in zip(rows, expected, strict=True):
        for prefix, key in (("bare", "underlying"), ("qualified", "full")):
            for metric in ("precision", "recall", "f1"):
                assert float(row[prefix + "_" + metric]) == r["evaluation"][key][metric]
        assert int(row["predictions"]) == r["evaluation"]["full"]["predicted"]
    # Direct contextual answers retain their irrelevant predictions, including the best one.
    assert all(
        int(r["irrelevant"]) > 0
        for r in rows
        if r["category"] == "new_model" and r["approach"].startswith("B")
    )


def test_report_hashes_manual_bindings_and_original_results_preserved():
    d = data()
    manifest = json.loads((REPORTS / "tables/compact_story_v2_manifest.json").read_text())
    for f, h in manifest.items():
        assert hashlib.sha256((REPORTS / f).read_bytes()).hexdigest() == h
    for k, note in d["manual_diagnosis"]["cases"].items():
        c = next(c for c in d["calls"] if c["case_id"] == k)
        assert note["response_sha256"] == c["response"]["response_sha256"]
        assert all(0 <= i < len(c["parsed"]["facts"]) for i in note["record_indices"])
    html = (REPORTS / "figures/compact_story_v2_comparison.html").read_text()
    for sid in p.stories():
        for task in p.questions(p.stories()[sid]):
            for method in ("A", "B"):
                assert f'id="{sid}-{task}-{method}"' in html
    for marker in ("213.173.110.36", "Authorization:", "Bearer ", "/workspace/", ".ssh/"):
        assert all(marker not in (REPORTS / f).read_text() for f in manifest)
    original = json.loads((REPORTS / "tables/compact_story_manifest.json").read_text())
    assert all(
        hashlib.sha256((REPORTS / f).read_bytes()).hexdigest() == h
        for f, h in original["files"].items()
    )


def test_report_reproduces_from_closed_restricted_run(tmp_path):
    from scripts.report_compact_story_v2 import render

    run = ROOT / "artifacts/restricted/compact-story-v2-backup.vxnmOm/run-20260909T063738739103"
    if not run.exists():
        pytest.skip("Restricted raw run intentionally excluded from public checkout")
    render(run, tmp_path, REPORTS / "tables/compact_story_results.json")
    actual = json.loads((tmp_path / "tables/compact_story_v2_manifest.json").read_text())
    assert actual == json.loads((REPORTS / "tables/compact_story_v2_manifest.json").read_text())
