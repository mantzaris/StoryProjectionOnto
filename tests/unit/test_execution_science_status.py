"""HTTP completion cannot certify science or successful-output throughput."""

import copy
import hashlib
import sqlite3
from pathlib import Path

import pytest

from story_projection_onto.ledger_verify import derive_call_status

LEDGER = Path(
    "artifacts/restricted/parent-repair-session-backup.vaJysn/artifacts/restricted/phase1_acceptance.sqlite"
)


def sources():
    call = dict(
        model_call_id="call", attempt_id="attempt", response_artifact_hash="hash", successful=1
    )
    event = {"succeeded": 1}
    validation = dict(
        validation_id="v",
        attempt_id="attempt",
        input_artifact_hash="hash",
        validation_status="accepted",
        evidence_support_status="supported",
        temporal_status="valid",
        commitment_status="valid",
        semantic_assessment_scope="posthoc_scorer_or_reviewer",
    )
    failure = dict(failure_id="f", attempt_id="attempt", failure_kind="invalid_output")
    return call, event, validation, failure


@pytest.mark.parametrize("successful", [0, 1])
def test_failure_overrides_execution_bit_and_even_conflicting_positive_validation(successful):
    call, event, validation, failure = sources()
    call["successful"] = successful
    before = copy.deepcopy((call, event, validation, failure))
    status = derive_call_status(call, event, [failure], [validation], diagnostic_only=False)
    assert status["scientific_status"] == "rejected"
    assert not any(status[k] for k in status if k.endswith("_eligible"))
    assert (call, event, validation, failure) == before


@pytest.mark.parametrize("case", ["missing", "structural_only", "wrong_artifact", "diagnostic"])
def test_positive_explicit_bound_assessment_and_protocol_scope_are_required(case):
    call, event, v, _ = sources()
    if case == "structural_only":
        v["semantic_assessment_scope"] = "runtime_structural_only_not_assessed"
    if case == "wrong_artifact":
        v["input_artifact_hash"] = "another-output"
    result = derive_call_status(
        call, event, [], [] if case == "missing" else [v], diagnostic_only=case == "diagnostic"
    )
    assert not any(result[k] for k in result if k.endswith("_eligible"))


def test_nine_actual_mismatches_remain_immutable_rejected_and_ineligible():
    if not LEDGER.exists():
        pytest.skip("restricted historical ledger not in public release")
    original = hashlib.sha256(LEDGER.read_bytes()).hexdigest()
    with sqlite3.connect(f"file:{LEDGER}?mode=ro", uri=True) as c:
        c.row_factory = sqlite3.Row
        calls = [dict(r) for r in c.execute("SELECT * FROM model_calls")]
        events = {r["event_id"]: dict(r) for r in c.execute("SELECT * FROM gpu_events")}
        failures = [dict(r) for r in c.execute("SELECT * FROM failures")]
        validations = [dict(r) for r in c.execute("SELECT * FROM validations")]
        mismatches = [
            r
            for r in calls
            if r["gpu_event_id"] in events
            and r["successful"] != events[r["gpu_event_id"]]["succeeded"]
        ]
        assert len(mismatches) == 9
        for call in mismatches:
            for flag in (0, 1):
                derived = derive_call_status(
                    {**call, "successful": flag},
                    events[call["gpu_event_id"]],
                    failures,
                    validations,
                    diagnostic_only=True,
                )
                assert derived["execution_completed"]
                assert derived["scientific_status"] == "rejected"
                assert derived["failure_ids"]
                assert not any(derived[k] for k in derived if k.endswith("_eligible"))
        actual = (
            c.execute("SELECT SUM(allocated_microseconds) FROM gpu_events").fetchone()[0]
            + c.execute("SELECT SUM(overhead_microseconds) FROM gpu_service_sessions").fetchone()[0]
        )
        assert actual == 6716108081
    assert hashlib.sha256(LEDGER.read_bytes()).hexdigest() == original
