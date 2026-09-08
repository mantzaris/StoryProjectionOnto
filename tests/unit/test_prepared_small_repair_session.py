"""Exact authorized repair replay; CPU only, no service start."""

from dataclasses import replace

import pytest

from story_projection_onto.gpu_runtime import TokenizerManifest
from story_projection_onto.store import AttemptKind, Ledger, ReleaseClass
from tests.unit.test_capacity_diagnostic_controller import ROOT, driver
from tests.unit.test_small_retry_path import RETAINED, retained_examples


@pytest.fixture(scope="module")
def frozen():
    from transformers import AutoTokenizer

    path = ROOT / "artifacts/restricted/pinned-tokenizer-cpu"
    if not path.exists() or not RETAINED.exists():
        pytest.skip("restricted request parents/tokenizer not in public release")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    manifest = TokenizerManifest(
        **driver.read(RETAINED / "rendered-semantic-first.json")["tokenizer_manifest"]
    )
    examples = list(retained_examples(tokenizer, manifest))
    return tokenizer, manifest, {x[0]: x[2] for x in examples}


def test_exact_requests_all_messages_and_template_counts(frozen):
    tokenizer, manifest, bases = frozen
    variants, parents, receipts = driver.verify_prepared_repairs(
        ROOT, bases, tokenizer, manifest, parent_root=RETAINED
    )
    assert [r["input_tokens"] for r in receipts.values()] == [8049, 8461]
    assert [r["total_tokens"] for r in receipts.values()] == [11633, 12045]
    for kind, request in variants.items():
        assert len(request.messages) == 4
        assert request.stream_response and request.decoding.maximum_output_tokens == 3584
        assert parents[kind].startswith(driver.PARENT_BLOCK)
        assert request.messages[1] == bases[kind].messages[1]


def test_changed_template_or_parent_request_is_rejected_before_allocation(frozen):
    tokenizer, manifest, bases = frozen
    altered = dict(bases)
    q = altered["semantic-first"]
    altered["semantic-first"] = replace(q, messages=(*q.messages, q.messages[-1]))
    with pytest.raises(ValueError, match="parent response/request"):
        driver.verify_prepared_repairs(ROOT, altered, tokenizer, manifest, parent_root=RETAINED)


def test_two_repair_lineages_run_despite_first_validation_failure(tmp_path, frozen):
    tokenizer, manifest, bases = frozen
    variants, parents, _ = driver.verify_prepared_repairs(
        ROOT, bases, tokenizer, manifest, parent_root=RETAINED
    )
    ledger = Ledger(tmp_path / "ledger.sqlite")
    for kind, parent in parents.items():
        job = ledger.create_or_resume_job({"fixture": kind}, release_class=ReleaseClass.RESTRICTED)
        ledger.record_attempt(
            attempt_id=parent,
            job_id=job.job_id,
            attempt_kind=AttemptKind.BASE,
            input_hash=bases[kind].request_hash,
            config_hash=bases[kind].decoding.content_hash,
            seed=1,
        )
    calls = []
    while True:
        kind, stop = driver.next_prepared_repair(
            completed=len(calls), healthy=True, remaining_seconds=500
        )
        if kind is None:
            break
        parent = ledger.attempt_lineage(parents[kind])[-1]
        attempt = ledger.record_attempt(
            attempt_id="repair-" + kind,
            job_id=parent.job_id,
            attempt_kind=AttemptKind.REPAIR,
            parent_attempt_id=parent.attempt_id,
            input_hash=variants[kind].request_hash,
            config_hash=variants[kind].decoding.content_hash,
            seed=1,
        )
        assert ledger.attempt_lineage(attempt.attempt_id) == (parent, attempt)
        # Simulated canonical/scientific failures do not affect frozen ordering.
        calls.append((kind, "canonical_failure" if not calls else "scientific_failure"))
    assert [x[0] for x in calls] == ["semantic-first", "semantic-second"]
    assert stop == "two_prepared_repairs_complete"


@pytest.mark.parametrize(
    "completed,healthy,remaining,reason",
    [
        (0, False, 860, "unsafe_service_state"),
        (1, True, 269, "insufficient_time_with_shutdown_reserve"),
        (2, True, 500, "two_prepared_repairs_complete"),
    ],
)
def test_no_third_call_or_unprotected_shutdown(completed, healthy, remaining, reason):
    assert driver.next_prepared_repair(
        completed=completed, healthy=healthy, remaining_seconds=remaining
    ) == (None, reason)
