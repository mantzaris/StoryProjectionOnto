"""Prospective compact v2 controls; no model success is inferred from fixtures."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from story_projection_onto.scorer_only.simple_ladder_v2 import (
    evaluate,
    frozen_rules,
    office_comparison,
    references,
)
from story_projection_onto.simple_ladder_v2 import BASELINE, admit, cases, next_case, prepare
from tests.unit.test_simple_ladder import pinned as pinned_fixture


@pytest.fixture(scope="module")
def pinned():
    return pinned_fixture.__wrapped__()


@pytest.mark.parametrize("case_id", [str(i) for i in range(1, 9)])
def test_all_frozen_requests_and_authored_capacity(case_id, pinned):
    t, m = pinned
    c = cases()[case_id]
    q = prepare(c, t, m)
    assert q.rendered_input_token_count < 500
    assert q.rendered_input_token_count + q.decoding.maximum_output_tokens <= 12288
    assert "guided_json" not in q.wire_payload()
    assert "evidence_ids" in q.messages[0].content and "evidence_id (" not in q.messages[0].content
    assert "reference" not in q.messages[1].content.lower()
    for alternative in references()[case_id]:
        authored = {"facts": alternative}
        e = evaluate(case_id, authored)
        assert e["full"]["f1"] == e["underlying"]["f1"] == 1
        assert e["format_compliant"] == e["citation_valid"] == len(alternative)
        assert (
            len(t.encode(json.dumps(authored), add_special_tokens=False))
            < q.decoding.maximum_output_tokens
        )


def test_citation_failure_does_not_erase_relationships():
    rows = copy.deepcopy(references()["5"][0])
    rows[0]["evidence_ids"] = [1]
    e = evaluate("5", {"facts": rows})
    assert e["underlying"]["f1"] == 1 and e["full"]["true_positive"] == 3
    assert e["citation_valid"] == 3 and e["prediction_count"] == 4


@pytest.mark.parametrize(
    "field,value", [("holder", "Invented holder"), ("attitude", "knows"), ("holder", None)]
)
def test_wrong_attribution_fails_without_erasing_location(field, value):
    rows = copy.deepcopy(references()["3"][0])
    rows[0][field] = value
    e = evaluate("3", {"facts": rows})
    assert e["full"]["f1"] == 0.5 and e["underlying"]["f1"] == 1
    assert e["qualifications"]["epistemic"]["correct"] == 0


def test_narration_is_not_assumed_known():
    rows = copy.deepcopy(references()["3"][0])
    rows[1].update(holder="Rina", attitude="believes")
    e = evaluate("3", {"facts": rows})
    assert e["full"]["f1"] == 0.5
    assert e["qualifications"]["epistemic"]["applicable_predictions"] == 2
    assert e["qualifications"]["epistemic"]["precision"] == 0.5


def test_boolean_bounds_are_not_numeric_time_and_unknown_format_is_not_pass():
    rows = copy.deepcopy(references()["2"][0])
    rows[0]["valid_from"] = False
    e = evaluate("2", {"facts": rows})
    assert e["qualifications"]["temporal"]["correct"] == 1
    assert not e["records"][0]["temporal_correct"]
    rows = copy.deepcopy(references()["3"][0])
    rows[0]["object"] = "unresolved expression"
    e = evaluate("3", {"facts": rows})
    assert e["shape_compliant"] == 2 and e["format_compliant"] == 1
    assert e["format_unresolved"] == 1


def test_missing_bounds_wrong_bounds_unsupported_precision():
    rows = copy.deepcopy(references()["2"][0])
    del rows[0]["valid_until"]
    rows[1]["valid_until"] = 99
    e = evaluate("2", {"facts": rows})
    assert e["full"]["f1"] == 0 and e["underlying"]["f1"] == 1
    assert e["qualifications"]["temporal"]["target_recall"] == 0
    assert e["qualifications"]["temporal"]["applicable_predictions"] == 2
    assert e["format_compliant"] == 1
    direct = copy.deepcopy(references()["1"][0])
    direct[0].update(valid_from=1, valid_until=2)
    assert evaluate("1", {"facts": direct})["full"]["true_positive"] == 2


def test_aliases_are_explicit_prospective_and_not_semantic_completion():
    rows = copy.deepcopy(references()["3"][0])
    rows[0]["subject"] = "the coin"
    rows[1]["object"] = "the chest"
    assert evaluate("3", {"facts": rows})["full"]["f1"] == 1
    rows[0]["object"] = "the coin is in the drawer"
    e = evaluate("3", {"facts": rows})
    assert e["full"]["f1"] == 0.5
    assert e["records"][0]["assessment"].startswith("unresolved")
    assert "the coin" in frozen_rules()["endpoint_aliases"]["3"]


def test_duplicates_wrong_directions_unresolved_and_omissions_retained():
    rows = copy.deepcopy(references()["1"][0])
    wrong = {**rows[0], "subject": "lantern", "object": "Mira"}
    rows += [rows[0], wrong, {**rows[0], "relation": "unrecognized paraphrase"}]
    e = evaluate("1", {"facts": rows})
    assert e["full"]["precision"] == 0.5 and e["prediction_count"] == 6
    assert e["records"][-1]["assessment"].startswith("unresolved")
    assert evaluate("1", {"facts": rows[:1]})["full"]["recall"] == 1 / 3


def test_pairs_and_organization_frozen_without_cosmetic_requirement():
    c = cases()
    assert c["5"]["evidence"] == c["6"]["evidence"] and c["7"]["evidence"] == c["8"]["evidence"]
    assert c["5"]["evidence"] != c["7"]["evidence"]
    r = references()
    a = {"facts": r["5"][0]}
    b = {"facts": list(reversed(r["6"][0]))}
    assert office_comparison(a, b, ("5", "6"))["verdict"] == "same graph organization and bindings"
    assert (
        office_comparison({"facts": r["5"][1]}, b, ("5", "6"))["verdict"]
        == "different semantic organization"
    )
    assert (
        office_comparison({"facts": r["5"][0][:3]}, b, ("5", "6"))["verdict"]
        == "different fact selection within same organization"
    )


def test_one_attempt_each_and_exact_shutdown_bounds():
    assert next_case([{}] * 7) == "8" and next_case([{}] * 8) is None
    admit(BASELINE, 0, 0, starting=True, seconds=352.260137)
    for kwargs in [
        dict(starts=1, attempts=0, starting=True),
        dict(starts=1, attempts=8, generating=True),
    ]:
        with pytest.raises(ValueError):
            admit(BASELINE, seconds=1, **kwargs)
    with pytest.raises(ValueError):
        admit(BASELINE + 350, 1, 1, seconds=50)


def test_original_results_and_frozen_evaluator_unchanged():
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads((root / "reports/tables/simple_synthetic_manifest.json").read_text())
    for p, h in manifest.items():
        assert hashlib.sha256((root / "reports" / p).read_bytes()).hexdigest() == h


def test_actual_v2_replay_and_report_consistency(tmp_path):
    import csv

    from scripts.report_simple_ladder_v2 import render

    run = Path("artifacts/restricted/simple-v2-backup.TBXLRG/run-20260909T012304816240")
    if not run.exists():
        pytest.skip("restricted real outputs unavailable")
    terminal = json.loads((run / "terminal.json").read_text())
    assert len(terminal["outcomes"]) == 8
    expected = [1, 1, 1, 1, 0, 0, 0.5, 0.5]
    packing = json.loads((run / "packing.json").read_text())
    for outcome, f1 in zip(terminal["outcomes"], expected, strict=True):
        k = outcome["case_id"]
        recomputed = json.loads(json.dumps(evaluate(k, outcome["parsed"])))
        assert recomputed == outcome["evaluation"]
        assert recomputed["full"]["f1"] == f1
        assert outcome["response"]["finish_reason"] == "stop"
        assert outcome["response"]["prompt_tokens"] == packing[k]["input_tokens"]
        assert outcome["response"]["completion_tokens"] < packing[k]["reserved_output_tokens"]
    rows = render(run, tmp_path)
    saved = list(csv.DictReader((tmp_path / "tables/simple_synthetic_v2_results.csv").open()))
    for row, record in zip(rows, saved, strict=True):
        for k, v in row.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                assert float(record[k]) == v
    manifest = json.loads((tmp_path / "tables/simple_synthetic_v2_manifest.json").read_text())
    for p, h in manifest.items():
        assert hashlib.sha256((tmp_path / p).read_bytes()).hexdigest() == h
    assert terminal["open_allocations"] == terminal["open_service_journals"] == 0
