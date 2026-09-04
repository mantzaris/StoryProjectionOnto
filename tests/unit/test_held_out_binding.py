from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from story_projection_onto.contracts import canonical_sha256
from story_projection_onto.held_out_binding import (
    FROZEN_HELD_OUT_PRODUCTION_FACTORY,
    DevelopmentPredecessorLedgerBinding,
    HeldOutRuntimeBinding,
    verify_predecessor_gpu_event_chain,
)
from story_projection_onto.held_out_primary import (
    HeldOutControlError,
    HeldOutReviewGateError,
    load_executable_held_out_configuration,
    load_held_out_control_configuration,
)
from story_projection_onto.store import GpuEventKind, GpuServiceJournalState, Ledger

ROOT = Path(__file__).resolve().parents[2]
HASH = "a" * 64


def _ledger_binding(ledger: Ledger) -> DevelopmentPredecessorLedgerBinding:
    events = ledger.gpu_events()
    sessions = ledger.gpu_service_sessions()
    journal = ledger.gpu_service_journal_records()
    summary = ledger.gpu_summary()
    summary_payload = {
        "total_allocated_microseconds": summary.total_allocated_microseconds,
        "event_count": summary.event_count,
        "service_session_count": summary.service_session_count,
        "by_kind_microseconds": {
            kind.value: microseconds for kind, microseconds in summary.by_kind_microseconds
        },
    }
    session_rows = tuple(
        (item.service_session_id, canonical_sha256(asdict(item))) for item in sessions
    )
    journal_rows = tuple(
        (item.journal_id, canonical_sha256(asdict(item))) for item in journal
    )
    return DevelopmentPredecessorLedgerBinding(
        development_execution_id="development-test",
        development_execution_result_hash=HASH,
        predecessor_gpu_event_count=len(events),
        predecessor_gpu_event_ids=tuple(item.event_id for item in events),
        predecessor_gpu_event_chain_hash=canonical_sha256(
            tuple(asdict(item) for item in events)
        ),
        predecessor_total_allocated_microseconds=summary.total_allocated_microseconds,
        predecessor_gpu_summary_hash=canonical_sha256(summary_payload),
        predecessor_gpu_service_session_count=summary.service_session_count,
        predecessor_gpu_service_session_rows=session_rows,
        predecessor_gpu_service_sessions_hash=canonical_sha256(session_rows),
        predecessor_gpu_service_journal_rows=journal_rows,
        predecessor_gpu_service_journal_hash=canonical_sha256(journal_rows),
        development_handoff_allocated_microseconds=1,
        development_gpu_event_ids=tuple(
            f"development-event-{index:02d}" for index in range(24)
        ),
        development_gpu_events_hash=HASH,
        development_model_call_ids=tuple(
            f"development-model-{index:02d}" for index in range(24)
        ),
        development_model_calls_hash=HASH,
        development_call_lineage_hash=HASH,
    )


def _binding_payload() -> dict[str, object]:
    return {
        "binding_id": "held-out-runtime-test",
        "control_configuration_path": "configs/study/held_out_primary.json",
        "control_configuration_file_sha256": HASH,
        "control_configuration_hash": HASH,
        "template_call_manifest_hash": HASH,
        "post_development_call_manifest_hash": HASH,
        "runtime_control_configuration_hash": HASH,
        "production_adapter_factory": FROZEN_HELD_OUT_PRODUCTION_FACTORY,
        "fallback_result_path": "artifacts/public/results/fallback.json",
        "fallback_result_file_sha256": HASH,
        "fallback_result_manifest_sha256": HASH,
        "fallback_run_id": "fallback-test",
        "development_execution_result_path": (
            "artifacts/restricted/fallback-development/development_execution_result.json"
        ),
        "development_execution_result_file_sha256": HASH,
        "development_execution_result_hash": HASH,
        "development_execution_id": "development-test",
        "development_execution_manifest_hash": HASH,
        "development_prequery_inputs_hash": HASH,
        "development_continuation_receipt_hash": HASH,
        "development_predecessor_ledger": {
            "development_execution_id": "development-test",
            "development_execution_result_hash": HASH,
            "predecessor_gpu_event_count": 1,
            "predecessor_gpu_event_ids": ("predecessor-event",),
            "predecessor_gpu_event_chain_hash": HASH,
            "predecessor_total_allocated_microseconds": 1,
            "predecessor_gpu_summary_hash": HASH,
            "predecessor_gpu_service_session_count": 0,
            "predecessor_gpu_service_session_rows": (),
            "predecessor_gpu_service_sessions_hash": canonical_sha256(()),
            "predecessor_gpu_service_journal_rows": (),
            "predecessor_gpu_service_journal_hash": canonical_sha256(()),
            "development_handoff_allocated_microseconds": 1,
            "development_gpu_event_ids": tuple(
                f"development-event-{index:02d}" for index in range(24)
            ),
            "development_gpu_events_hash": HASH,
            "development_model_call_ids": tuple(
                f"development-model-{index:02d}" for index in range(24)
            ),
            "development_model_calls_hash": HASH,
            "development_call_lineage_hash": HASH,
        },
        "selected_model_freeze_path": (
            "artifacts/restricted/fallback-development/selected_model_freeze.json"
        ),
        "selected_model_freeze_file_sha256": HASH,
        "selected_model_freeze_hash": HASH,
        "source_association_path": "artifacts/public/manifests/source.association.json",
        "source_association_file_sha256": HASH,
        "source_association_manifest_sha256": HASH,
        "source_tree_sha256": HASH,
        "created_at": datetime(2026, 9, 4, tzinfo=UTC),
    }


def test_runtime_binding_paths_are_restricted_and_repository_relative() -> None:
    binding = HeldOutRuntimeBinding.model_validate(_binding_payload())
    assert binding.held_out_review_still_required
    payload = _binding_payload()
    payload["development_execution_result_path"] = "artifacts/public/development.json"
    with pytest.raises(ValidationError, match="restricted namespace"):
        HeldOutRuntimeBinding.model_validate(payload)
    payload = _binding_payload()
    payload["source_association_path"] = "../outside.json"
    with pytest.raises(ValidationError, match="repository-relative"):
        HeldOutRuntimeBinding.model_validate(payload)


def test_source_template_freezes_factory_but_requires_dynamic_predecessor() -> None:
    template = load_held_out_control_configuration(ROOT)
    assert template.production_adapter_factory == FROZEN_HELD_OUT_PRODUCTION_FACTORY
    assert template.development_execution_result_file_sha256 == "PENDING"
    with pytest.raises(HeldOutReviewGateError, match="post-development binding"):
        load_executable_held_out_configuration(repository=ROOT)


def test_same_total_replacement_ledger_fails_exact_predecessor_chain(
    tmp_path: Path,
) -> None:
    original = Ledger(tmp_path / "original.sqlite3")
    replacement = Ledger(tmp_path / "replacement.sqlite3")
    try:
        original.record_gpu_event(
            event_id="accepted-development-event",
            event_kind=GpuEventKind.INFERENCE,
            allocated_seconds=7,
            started_at=datetime(2026, 9, 4, tzinfo=UTC),
            ended_at=datetime(2026, 9, 4, tzinfo=UTC) + timedelta(seconds=7),
            succeeded=True,
        )
        replacement.record_gpu_event(
            event_id="fabricated-replacement-event",
            event_kind=GpuEventKind.INFERENCE,
            allocated_seconds=7,
            started_at=datetime(2026, 9, 4, tzinfo=UTC),
            ended_at=datetime(2026, 9, 4, tzinfo=UTC) + timedelta(seconds=7),
            succeeded=True,
        )
        binding = _ledger_binding(original)
        assert original.gpu_summary().total_allocated_microseconds == (
            replacement.gpu_summary().total_allocated_microseconds
        )
        assert verify_predecessor_gpu_event_chain(original, binding) is False
        with pytest.raises(HeldOutControlError, match="event chain differs"):
            verify_predecessor_gpu_event_chain(replacement, binding)
    finally:
        original.close()
        replacement.close()


def test_same_total_altered_service_identity_and_journal_are_rejected(
    tmp_path: Path,
) -> None:
    original = Ledger(tmp_path / "service-original.sqlite3")
    replacement = Ledger(tmp_path / "service-replacement.sqlite3")
    try:
        for ledger in (original, replacement):
            ledger.record_gpu_event(
                event_id="shared-development-event",
                event_kind=GpuEventKind.INFERENCE,
                allocated_seconds=1,
                started_at=datetime(2026, 9, 4, tzinfo=UTC),
                ended_at=datetime(2026, 9, 4, tzinfo=UTC) + timedelta(seconds=1),
                succeeded=True,
            )
        for ledger, started_at, session_id in (
            (
                original,
                datetime(2026, 9, 4, tzinfo=UTC),
                "accepted-logical-service",
            ),
            (
                replacement,
                datetime(2026, 9, 4, tzinfo=UTC) + timedelta(seconds=10),
                "altered-logical-service",
            ),
        ):
            ledger.record_gpu_service_observation(
                service_session_id="shared-service",
                state=GpuServiceJournalState.OPENED,
                session_id=session_id,
                configuration_hash=HASH,
                service_started_at=started_at,
                elapsed_seconds=0,
                ledger_allocated_seconds_before_session=0,
                hard_limit_seconds=36_000,
                observed_at=started_at,
            )
            ledger.recover_gpu_service_journal(
                service_session_id="shared-service",
                recovered_at=started_at + timedelta(seconds=3),
            )
        binding = _ledger_binding(original)
        assert original.gpu_summary() == replacement.gpu_summary()
        with pytest.raises(HeldOutControlError, match="service-session rows differ"):
            verify_predecessor_gpu_event_chain(replacement, binding)
    finally:
        original.close()
        replacement.close()


def test_same_total_altered_service_journal_row_is_rejected(tmp_path: Path) -> None:
    original = Ledger(tmp_path / "journal-original.sqlite3")
    replacement = Ledger(tmp_path / "journal-replacement.sqlite3")
    try:
        for ledger in (original, replacement):
            ledger.record_gpu_event(
                event_id="shared-development-event",
                event_kind=GpuEventKind.INFERENCE,
                allocated_seconds=1,
                started_at=datetime(2026, 9, 4, tzinfo=UTC),
                ended_at=datetime(2026, 9, 4, tzinfo=UTC) + timedelta(seconds=1),
                succeeded=True,
            )
            ledger.record_gpu_service_session(
                service_session_id="shared-service",
                session_id="shared-logical-service",
                service_seconds=2,
                classified_event_seconds=1,
                started_at=datetime(2026, 9, 4, tzinfo=UTC),
                ended_at=datetime(2026, 9, 4, tzinfo=UTC) + timedelta(seconds=2),
            )
        for ledger, detail in ((original, "accepted"), (replacement, "altered")):
            ledger.record_gpu_service_observation(
                service_session_id="shared-service",
                state=GpuServiceJournalState.OPENED,
                session_id="shared-logical-service",
                configuration_hash=HASH,
                service_started_at=datetime(2026, 9, 4, tzinfo=UTC),
                elapsed_seconds=0,
                ledger_allocated_seconds_before_session=0,
                hard_limit_seconds=36_000,
                observed_at=datetime(2026, 9, 4, tzinfo=UTC),
                details={"identity": detail},
            )
        binding = _ledger_binding(original)
        assert original.gpu_summary() == replacement.gpu_summary()
        with pytest.raises(HeldOutControlError, match="service-journal rows differ"):
            verify_predecessor_gpu_event_chain(replacement, binding)
    finally:
        original.close()
        replacement.close()
