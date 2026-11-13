from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

import story_projection_onto.combined_gpu_factory as factory
from story_projection_onto.combined_gpu_factory import (
    CombinedFactoryError,
    _CombinedSemanticExecutor,
    _FrozenCombinedLifecycleOwner,
    _FrozenCombinedOwnedService,
    _OrdinarySemanticCompletion,
    _registered_combined_storage_preflights,
    _registered_repair_policy_hash,
    _require_registered_combined_storage_preflights,
    _require_registered_phase_five_transition_storage_preflight,
    _SemanticIntent,
)
from story_projection_onto.combined_gpu_production import (
    CombinedActivationSlot,
    CombinedProductionError,
    CombinedRecoveryRequired,
    CombinedRepairClaim,
)
from story_projection_onto.conditions.base import ConditionAttemptRecord
from story_projection_onto.contracts import ReleaseClass, RunOutcome, canonical_sha256
from story_projection_onto.held_out_execution import FrozenHeldOutSemanticExecutor
from story_projection_onto.store import (
    ArtifactStore,
    AttemptKind,
    BlobStore,
    Compression,
    GpuEventKind,
    JobState,
    Ledger,
    ModelBackend,
    ModelCallRole,
    RetryClass,
    StoragePreflight,
)
from tests.unit.test_combined_gpu_block import combined_fixture, digest
from tests.unit.test_combined_gpu_production import _prepared


def _activation_slot():
    configuration, _selections, _sources, runtime, gate, manifest = combined_fixture()
    return CombinedActivationSlot(
        run_id="test-combined-run",
        manifest_hash=manifest.content_hash,
        configuration_hash=configuration.content_hash,
        runtime_binding_hash=runtime.content_hash,
        upstream_gate_hash=gate.content_hash,
        global_accounting_id=gate.global_accounting_id,
        gpu_seconds_before=0.0,
        model_load_event_hashes_before=(),
        short_reserve_slots_consumed_before=0,
        remaining_registered_p95_seconds_before=10_000.0,
        created_at=manifest.created_at,
    )


def _validation_helper(
    artifacts: ArtifactStore,
    *,
    validator_hash: str,
) -> SimpleNamespace:
    """Bind real canonical validation/lifecycle methods to a small test shim."""

    helper = SimpleNamespace(artifacts=artifacts, validator_hash=validator_hash)
    helper._advance_state = lambda job_id, state, occurred_at: (  # type: ignore[attr-defined]
        FrozenHeldOutSemanticExecutor._advance_state(
            helper,
            job_id,
            state,
            occurred_at,
        )
    )
    helper._record_attempt_validation = lambda **kwargs: (  # type: ignore[attr-defined]
        FrozenHeldOutSemanticExecutor._record_attempt_validation(helper, **kwargs)
    )
    return helper


def test_factory_recomputes_exact_registered_repair_policy_hash() -> None:
    configuration, _selections, _sources, _runtime, _gate, _manifest = combined_fixture()
    prompt_hash = digest("repair-prompt")
    validator_hash = digest("validator")

    observed = _registered_repair_policy_hash(
        configuration=configuration,
        repair_prompt_file_sha256=prompt_hash,
        validator_hash=validator_hash,
    )

    assert observed == canonical_sha256(
        {
            "policy": "one-preservation-repair-with-short-global-reserve-v1",
            "repair_prompt_file_sha256": prompt_hash,
            "maximum_repairs_per_call": configuration.maximum_repairs_per_call,
            "repair_reserve_class": configuration.repair_reserve_class,
            "repair_watchdog_seconds": configuration.repair_watchdog_seconds,
            "validator_hash": validator_hash,
        }
    )


def test_combined_factory_uses_distinct_registered_phase_four_and_five_reservations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = Path(__file__).resolve().parents[2]
    storage = StoragePreflight(tmp_path)
    original_check = storage.check
    observed: list[dict[str, int]] = []
    observed_phases: list[str] = []

    registered_plan = factory.StorageAllocationPlan.load(
        repository / "configs/study/storage_phase_allocations.json"
    )

    class _ObservedPlan:
        def reservation_for(self, phase: str):
            observed_phases.append(phase)
            return registered_plan.reservation_for(phase)

    class _ObservedPlanLoader:
        @staticmethod
        def load(path: Path):
            assert path == repository / "configs/study/storage_phase_allocations.json"
            return _ObservedPlan()

    def capture_check(**arguments: int):
        observed.append(dict(arguments))
        return original_check(
            **arguments,
            current_occupied_bytes=0,
            filesystem_free_bytes=30_000_000_000,
        )

    monkeypatch.setattr(storage, "check", capture_check)
    monkeypatch.setattr(factory, "StorageAllocationPlan", _ObservedPlanLoader)

    reports = _registered_combined_storage_preflights(
        repository,
        storage,
    )

    expected = {
        "declared_growth_bytes": 500_000_000,
        "largest_atomic_temporary_bytes": 134_217_728,
        "quarantine_allowance_bytes": 134_217_728,
        "release_staging_bytes": 268_435_456,
    }
    assert observed_phases == ["phase_4", "phase_5"]
    assert tuple(phase for phase, _report in reports) == ("phase_4", "phase_5")
    assert observed == [expected, expected]
    assert all(
        report.additional_reserved_bytes == sum(expected.values())
        for _phase, report in reports
    )


def test_combined_factory_persists_both_storage_gates_before_rejecting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = Path(__file__).resolve().parents[2]
    storage = StoragePreflight(tmp_path)
    original_check = storage.check
    call_count = 0

    def second_phase_rejects(**arguments: int):
        nonlocal call_count
        call_count += 1
        return original_check(
            **arguments,
            current_occupied_bytes=0 if call_count == 1 else 25_000_000_000,
            filesystem_free_bytes=30_000_000_000,
        )

    monkeypatch.setattr(storage, "check", second_phase_rejects)

    with Ledger(tmp_path / "study.sqlite3") as ledger:
        with pytest.raises(CombinedFactoryError, match="storage preflight failed"):
            _require_registered_combined_storage_preflights(
                repository,
                storage,
                ledger,
            )
        phase_four = ledger.storage_samples_with_phase_prefix("phase_4:")
        phase_five = ledger.storage_samples_with_phase_prefix("phase_5:")

    assert call_count == 2
    assert len(phase_four) == 1
    assert phase_four[0].phase == "phase_4:combined_gpu_block:factory"
    assert phase_four[0].allowed is True
    assert len(phase_five) == 1
    assert phase_five[0].phase == "phase_5:combined_gpu_block:factory"
    assert phase_five[0].allowed is False


def test_combined_phase_five_transition_rechecks_exact_registered_reservation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = Path(__file__).resolve().parents[2]
    storage = StoragePreflight(tmp_path)
    original_check = storage.check
    observed: list[dict[str, int]] = []

    def capture_check(**arguments: int):
        observed.append(dict(arguments))
        return original_check(
            **arguments,
            current_occupied_bytes=0,
            filesystem_free_bytes=30_000_000_000,
        )

    monkeypatch.setattr(storage, "check", capture_check)

    with Ledger(tmp_path / "study.sqlite3") as ledger:
        report = _require_registered_phase_five_transition_storage_preflight(
            repository,
            storage,
            ledger,
        )
        samples = ledger.storage_samples_with_phase_prefix("phase_5:")

    assert observed == [
        {
            "declared_growth_bytes": 500_000_000,
            "largest_atomic_temporary_bytes": 134_217_728,
            "quarantine_allowance_bytes": 134_217_728,
            "release_staging_bytes": 268_435_456,
        }
    ]
    assert report.allowed is True
    assert len(samples) == 1
    assert samples[0].phase == "phase_5:combined_gpu_block:transition"


def test_combined_phase_five_transition_persists_rejection_before_stopping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = Path(__file__).resolve().parents[2]
    storage = StoragePreflight(tmp_path)
    original_check = storage.check
    rejected = original_check(
        current_occupied_bytes=25_000_000_000,
        filesystem_free_bytes=30_000_000_000,
        declared_growth_bytes=500_000_000,
        largest_atomic_temporary_bytes=134_217_728,
        quarantine_allowance_bytes=134_217_728,
        release_staging_bytes=268_435_456,
    )
    monkeypatch.setattr(storage, "check", lambda **_arguments: rejected)

    with Ledger(tmp_path / "study.sqlite3") as ledger:
        with pytest.raises(CombinedFactoryError, match="Phase 5 storage preflight"):
            _require_registered_phase_five_transition_storage_preflight(
                repository,
                storage,
                ledger,
            )
        samples = ledger.storage_samples_with_phase_prefix("phase_5:")

    assert len(samples) == 1
    assert samples[0].phase == "phase_5:combined_gpu_block:transition"
    assert samples[0].allowed is False


class _LeaseRecoveryService:
    def __init__(self, launcher_hash: str, *, terminal: bool = False) -> None:
        self.launcher_hash = launcher_hash
        self.terminal = terminal
        self.pid = 7_777
        self.start_calls = 0
        self.checkpoint_calls = 0
        self.live_resume_calls = 0
        self.stale_recovery_calls = 0

    def start(self, **kwargs: object) -> None:
        del kwargs
        self.start_calls += 1
        raise AssertionError("activation recovery must not launch a second model")

    def resume_from_checkpoint(self, path: Path) -> bool:
        assert not path.exists()
        return False

    def resume_live_service_lease(self, **kwargs: object) -> bool:
        assert kwargs["expected_session_id"] == "test-combined-run-combined-service"
        assert kwargs["expected_event_id"] == "test-combined-run-combined-model-load"
        self.live_resume_calls += 1
        return not self.terminal

    def recover_stale_service_lease(self):
        self.stale_recovery_calls += 1
        return SimpleNamespace(service_session_id="terminal") if self.terminal else None

    def write_resume_checkpoint(self, path: Path) -> None:
        self.checkpoint_calls += 1
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "session_id": "test-combined-run-combined-service",
                    "accounting_session_id": "test-combined-run-combined-model-load",
                    "configuration_hash": self.launcher_hash,
                    "pid": self.pid,
                    "process_start_ticks": 88_888,
                }
            )
            + "\n",
            encoding="utf-8",
        )

    def shutdown(self):
        return None


def _owner(tmp_path: Path):
    configuration, _selections, _sources, runtime, _gate, manifest = combined_fixture()
    artifacts = ArtifactStore(
        BlobStore(tmp_path / "cas", compression=Compression.GZIP),
        Ledger(tmp_path / "ledger.sqlite3"),
    )
    factory_calls: list[None] = []

    def service_factory():
        factory_calls.append(None)
        raise AssertionError("recovery classification must not instantiate a service")

    owner = _FrozenCombinedLifecycleOwner(
        run_id="test-combined-run",
        configuration=configuration,
        runtime=runtime,
        artifacts=artifacts,
        runtime_root=tmp_path / "runtime",
        service_factory=service_factory,
        semantic=SimpleNamespace(),
    )
    return owner, artifacts, factory_calls, manifest.created_at


def test_lifecycle_replays_only_an_activation_slot_with_no_physical_trace(
    tmp_path: Path,
) -> None:
    owner, artifacts, factory_calls, _created_at = _owner(tmp_path)
    try:
        assert owner.recover_active("a" * 64) is None
        assert factory_calls == []
    finally:
        artifacts.ledger.close()


def test_ordinary_terminal_failure_reconstructs_missing_completion_without_resend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configuration, _selections, _sources, runtime, _gate, manifest = combined_fixture()
    call = manifest.calls[21]
    prepared = _prepared(call, runtime)
    guided_hash = digest("recovered-guided")
    decoding_hash = digest("recovered-decoding")
    config_hash = digest("recovered-config")
    inputs = SimpleNamespace(
        run_config=SimpleNamespace(
            content_hash=config_hash,
            seed_block=call.seed_block,
            condition=call.condition,
        ),
        context=call.source.context,
        packet=SimpleNamespace(release_class=ReleaseClass.RESTRICTED),
        prequery_barrier=SimpleNamespace(
            sealed_at=manifest.created_at + timedelta(seconds=1)
        ),
        query_access=SimpleNamespace(
            accessed_at=manifest.created_at
            + timedelta(seconds=1, microseconds=1)
        ),
    )
    guided = SimpleNamespace(
        request_hash=guided_hash,
        request_id=f"{call.call_id}-base",
        decoding=SimpleNamespace(content_hash=decoding_hash),
    )
    monkeypatch.setattr(
        factory,
        "build_development_guided_request",
        lambda **_kwargs: guided,
    )
    artifacts = ArtifactStore(
        BlobStore(tmp_path / "cas", compression=Compression.GZIP),
        Ledger(tmp_path / "ledger.sqlite3"),
    )
    semantic = object.__new__(_CombinedSemanticExecutor)
    semantic.repository = Path(".")
    semantic.run_id = "terminal-row-recovery"
    semantic.manifest = manifest
    semantic.runtime = runtime
    semantic.calls_by_hash = {call.content_hash: call}
    semantic.provider = SimpleNamespace(produce_inputs=lambda *_args: inputs)
    semantic.tokenizer = SimpleNamespace()
    semantic.tokenizer_manifest = SimpleNamespace()
    semantic.artifacts = artifacts
    semantic.helper = _validation_helper(
        artifacts,
        validator_hash=runtime.validator_hash,
    )
    semantic.state_root = tmp_path / "semantic"
    semantic.state_root.mkdir()
    identity = SimpleNamespace(content_hash=digest("terminal-row-service"))
    intent = _SemanticIntent(
        request_kind="ordinary",
        run_id=semantic.run_id,
        call_id=call.call_id,
        call_spec_hash=call.content_hash,
        prepared_hash=prepared.content_hash,
        guided_request_hash=guided_hash,
        gpu_event_id=f"{semantic.run_id}-{call.call_id}-base-gpu",
        service_identity_hash=identity.content_hash,
        cumulative_gpu_seconds_before=0.0,
        remaining_required_seconds=100.0,
        created_at=manifest.created_at,
        ordinary_prepared_call=prepared,
    )
    semantic._append(
        semantic._intent_path(prepared.content_hash, "ordinary"),
        intent,
        "test terminal intent",
    )
    started = manifest.created_at + timedelta(seconds=1)
    ended = started + timedelta(seconds=2)
    try:
        job = artifacts.ledger.create_or_resume_job(
            {"execution_id": semantic.run_id, "call_id": call.call_id},
            release_class=ReleaseClass.RESTRICTED,
            created_at=started,
        )
        attempt_id = f"{semantic.run_id}-{call.call_id}-base"
        artifacts.ledger.record_attempt(
            attempt_id=attempt_id,
            job_id=job.job_id,
            attempt_kind=AttemptKind.BASE,
            input_hash=guided_hash,
            config_hash=config_hash,
            seed=call.vllm_seed,
            created_at=started,
        )
        event_id = f"{attempt_id}-gpu"
        artifacts.ledger.record_gpu_event(
            event_id=event_id,
            event_kind=GpuEventKind.FAILURE,
            allocated_seconds=2.0,
            started_at=started,
            ended_at=ended,
            succeeded=False,
            job_id=job.job_id,
            attempt_id=attempt_id,
        )
        artifacts.ledger.record_model_call(
            model_call_id=f"{attempt_id}-model",
            job_id=job.job_id,
            attempt_id=attempt_id,
            gpu_event_id=event_id,
            backend=ModelBackend.VLLM_GPU,
            call_role=ModelCallRole.QUERY_TIME,
            retry_class=RetryClass.STANDARD,
            model_manifest_hash=runtime.model_manifest_hash,
            decoding_manifest_hash=decoding_hash,
            request_hash=guided_hash,
            response_artifact_hash=None,
            construction_unit_hash=canonical_sha256(
                {
                    "manifest": manifest.content_hash,
                    "ordinal": call.ordinal,
                    "condition": call.condition,
                }
            ),
            served_context_count=1,
            prompt_tokens=0,
            completion_tokens=0,
            allocated_gpu_seconds=2.0,
            successful=False,
            created_at=ended,
        )

        recovered = semantic.recover_ordinary(prepared.content_hash, identity)

        assert recovered is not None
        assert recovered.outcome is RunOutcome.FAILED
        assert semantic._completion_path(prepared.content_hash).is_file()
        assert semantic._result_path(prepared.content_hash, "ordinary").is_file()
        assert artifacts.ledger.count_rows("model_calls") == 1
        assert artifacts.ledger.count_rows("gpu_events") == 1
        assert artifacts.ledger.count_rows("validations") == 1
        assert artifacts.ledger.count_rows("failures") == 1
        assert artifacts.ledger.get_job(job.job_id).state is JobState.FINALIZED
    finally:
        artifacts.ledger.close()


def test_lifecycle_persists_path_free_activation_intent_before_start(
    tmp_path: Path,
) -> None:
    configuration, _selections, _sources, runtime, _gate, _manifest = combined_fixture()
    artifacts = ArtifactStore(
        BlobStore(tmp_path / "cas", compression=Compression.GZIP),
        Ledger(tmp_path / "ledger.sqlite3"),
    )
    observed_intent: list[str] = []
    owner: _FrozenCombinedLifecycleOwner

    class CrashAtStart:
        def start(self, **_kwargs: object) -> None:
            observed_intent.append(owner.activation_intent_path.read_text(encoding="utf-8"))
            raise RuntimeError("TEST-ONLY loss at service start")

        def shutdown(self) -> None:
            return None

    owner = _FrozenCombinedLifecycleOwner(
        run_id="test-combined-run",
        configuration=configuration,
        runtime=runtime,
        artifacts=artifacts,
        runtime_root=tmp_path / "runtime",
        service_factory=CrashAtStart,  # type: ignore[arg-type]
        semantic=SimpleNamespace(),
    )
    try:
        with pytest.raises(RuntimeError, match="loss at service start"):
            owner.activate(_activation_slot())
        assert len(observed_intent) == 1
        assert str(tmp_path) not in observed_intent[0]
        assert '"activation_slot"' in observed_intent[0]
    finally:
        artifacts.ledger.close()


def test_lifecycle_refuses_duplicate_after_unbound_model_load_event(
    tmp_path: Path,
) -> None:
    owner, artifacts, factory_calls, created_at = _owner(tmp_path)
    try:
        artifacts.ledger.record_gpu_event(
            event_id="test-combined-run-combined-model-load",
            event_kind=GpuEventKind.MODEL_LOAD,
            allocated_seconds=1.0,
            started_at=created_at,
            ended_at=created_at + timedelta(seconds=1),
            succeeded=False,
        )
        with pytest.raises(CombinedRecoveryRequired, match="lacks a complete lifecycle binding"):
            owner.recover_active("b" * 64)
        assert factory_calls == []
    finally:
        artifacts.ledger.close()


def test_lifecycle_recovers_live_lease_before_checkpoint_without_second_load(
    tmp_path: Path,
) -> None:
    configuration, _selections, _sources, runtime, _gate, manifest = combined_fixture()
    artifacts = ArtifactStore(
        BlobStore(tmp_path / "cas", compression=Compression.GZIP),
        Ledger(tmp_path / "ledger.sqlite3"),
    )
    service = _LeaseRecoveryService(runtime.launcher_configuration_hash)
    owner = _FrozenCombinedLifecycleOwner(
        run_id="test-combined-run",
        configuration=configuration,
        runtime=runtime,
        artifacts=artifacts,
        runtime_root=tmp_path / "runtime",
        service_factory=lambda: service,  # type: ignore[arg-type]
        semantic=SimpleNamespace(),
    )
    slot = _activation_slot()
    try:
        owner._activation_intent(slot)
        artifacts.ledger.record_gpu_event(
            event_id="test-combined-run-combined-model-load",
            event_kind=GpuEventKind.MODEL_LOAD,
            allocated_seconds=2.0,
            started_at=manifest.created_at,
            ended_at=manifest.created_at + timedelta(seconds=2),
            succeeded=True,
        )

        recovered = owner.recover_active(slot.content_hash)

        assert recovered is not None
        assert recovered.identity().service_pid == service.pid
        assert recovered.identity().service_start_ticks == 88_888
        assert service.start_calls == 0
        assert service.live_resume_calls == 1
        assert service.checkpoint_calls == 1
        assert service.stale_recovery_calls == 0
        assert owner.binding_path.is_file()
        assert owner.checkpoint_path.is_file()
        activation_bytes = owner.activation_intent_path.read_text(encoding="utf-8")
        assert str(tmp_path) not in activation_bytes
        assert "_path" not in activation_bytes
        assert len(
            [
                event
                for event in artifacts.ledger.gpu_events()
                if event.event_kind is GpuEventKind.MODEL_LOAD
            ]
        ) == 1
    finally:
        artifacts.ledger.close()


def test_lifecycle_treats_stale_terminal_recovery_as_consumed_load(
    tmp_path: Path,
) -> None:
    configuration, _selections, _sources, runtime, _gate, manifest = combined_fixture()
    artifacts = ArtifactStore(
        BlobStore(tmp_path / "cas", compression=Compression.GZIP),
        Ledger(tmp_path / "ledger.sqlite3"),
    )
    service = _LeaseRecoveryService(
        runtime.launcher_configuration_hash,
        terminal=True,
    )
    owner = _FrozenCombinedLifecycleOwner(
        run_id="test-combined-run",
        configuration=configuration,
        runtime=runtime,
        artifacts=artifacts,
        runtime_root=tmp_path / "runtime",
        service_factory=lambda: service,  # type: ignore[arg-type]
        semantic=SimpleNamespace(),
    )
    slot = _activation_slot()
    try:
        owner._activation_intent(slot)
        artifacts.ledger.record_gpu_event(
            event_id="test-combined-run-combined-model-load",
            event_kind=GpuEventKind.MODEL_LOAD,
            allocated_seconds=1.0,
            started_at=manifest.created_at,
            ended_at=manifest.created_at + timedelta(seconds=1),
            succeeded=False,
        )

        with pytest.raises(CombinedRecoveryRequired, match="second model load is forbidden"):
            owner.recover_active(slot.content_hash)
        assert service.start_calls == 0
        assert service.live_resume_calls == 1
        assert service.stale_recovery_calls == 1
        assert not owner.binding_path.exists()
    finally:
        artifacts.ledger.close()


def test_lifecycle_free_view_threads_exact_remaining_forecast_receipt() -> None:
    observed: list[tuple[str, float]] = []
    ordinary_result = SimpleNamespace(kind="ordinary")
    phase5_result = SimpleNamespace(kind="phase5")

    class Semantic:
        def execute_ordinary(self, **kwargs: object):
            observed.append(("ordinary", float(kwargs["remaining_required_seconds"])))
            return ordinary_result

        def execute_phase5(self, **kwargs: object):
            observed.append(("phase5", float(kwargs["remaining_required_seconds"])))
            return phase5_result

    owned = _FrozenCombinedOwnedService(
        service=SimpleNamespace(actual_allocated_service_seconds=1.0),
        identity=SimpleNamespace(),
        semantic=Semantic(),  # type: ignore[arg-type]
    )
    assert owned.execute_combined(
        SimpleNamespace(),
        {},
        SimpleNamespace(),
        321.25,
    ) is ordinary_result
    assert owned.execute_phase5_owned(
        SimpleNamespace(),
        {},
        SimpleNamespace(),
        210.5,
    ) is phase5_result
    assert observed == [("ordinary", 321.25), ("phase5", 210.5)]


def test_phase5_output_reference_reuses_existing_authoritative_cas_metadata(
    tmp_path: Path,
) -> None:
    configuration, *_rest = combined_fixture()
    artifacts = ArtifactStore(
        BlobStore(tmp_path / "cas", compression=Compression.GZIP),
        Ledger(tmp_path / "ledger.sqlite3"),
    )
    payload = (configuration.to_canonical_json() + "\n").encode("utf-8")
    existing = artifacts.put_bytes(
        payload,
        media_type="application/vnd.story-projection.ontology-projection+json",
        release_class=ReleaseClass.RESTRICTED,
        created_at=_activation_slot().created_at,
    )
    semantic = object.__new__(_CombinedSemanticExecutor)
    semantic.artifacts = artifacts
    try:
        reference = semantic._phase5_existing_record_reference(
            configuration,
            object_kind="phase5_after_projection",
        )

        assert reference.artifact_hash == existing.content_hash
        assert reference.logical_content_hash == configuration.content_hash
        assert reference.media_type == existing.media_type
        assert reference.release_class is ReleaseClass.RESTRICTED
        assert artifacts.ledger.get_artifact(existing.content_hash) == existing
    finally:
        artifacts.ledger.close()


def test_semantic_generation_rejects_zero_remaining_forecast_before_gpu() -> None:
    semantic = object.__new__(_CombinedSemanticExecutor)
    with pytest.raises(CombinedFactoryError, match="nonzero remaining forecast"):
        semantic._run_generation(
            service=SimpleNamespace(),
            call=SimpleNamespace(),
            semantic=SimpleNamespace(),
            config=SimpleNamespace(),
            inputs=SimpleNamespace(),
            repair_authority=SimpleNamespace(),
            authority_request_hash="a" * 64,
            remaining_required_seconds=0,
        )


@pytest.mark.parametrize("phase5", [False, True], ids=["ordinary", "phase5"])
@pytest.mark.parametrize(
    "reserve_available",
    [True, False],
    ids=["repair-eligible", "reserve-exhausted"],
)
def test_invalid_terminal_base_recovery_makes_one_durable_repair_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase5: bool,
    reserve_available: bool,
) -> None:
    """A crash before the repair claim never resends base or strands its ITT row."""

    _configuration, _selections, _sources, runtime, _gate, manifest = combined_fixture()
    call = manifest.calls[12] if phase5 else manifest.calls[21]
    request_hash = digest(f"repair-boundary-{'phase5' if phase5 else 'ordinary'}")
    base_guided = SimpleNamespace(
        request_hash=digest("repair-boundary-base-guided"),
        decoding=SimpleNamespace(content_hash=digest("base-decoding")),
    )
    repair_guided = SimpleNamespace(
        request_hash=digest("repair-boundary-repair-guided"),
        decoding=SimpleNamespace(content_hash=digest("repair-decoding")),
    )
    repair_semantic = SimpleNamespace(content_hash=digest("repair-semantic"))
    base_config = SimpleNamespace(content_hash=digest("base-config"))
    repair_config = SimpleNamespace(content_hash=digest("repair-config"))

    class Inputs:
        run_config = base_config
        prequery_barrier = SimpleNamespace(
            sealed_at=manifest.created_at + timedelta(seconds=1)
        )
        query_access = SimpleNamespace(
            accessed_at=manifest.created_at + timedelta(seconds=1)
        )

        @staticmethod
        def model_dump(**_kwargs: object) -> dict[str, object]:
            return {"run_config": base_config}

    inputs = Inputs()

    class InputModelShim:
        @staticmethod
        def model_validate(_values: object) -> Inputs:
            return inputs

    monkeypatch.setattr(factory, "HeldOutProduceInputs", InputModelShim)
    monkeypatch.setattr(
        factory,
        "build_development_guided_request",
        lambda **_kwargs: base_guided,
    )
    monkeypatch.setattr(
        factory,
        "failed_c2_attempt",
        lambda _inputs, *, outcome, failure_code, raw_output_hash: (
            ConditionAttemptRecord(
                attempt_id=f"recovered-{call.call_id}",
                condition=call.condition,
                unit_id=call.source.unit_id,
                seed_block=call.seed_block,
                outcome=outcome,
                raw_output_hash=raw_output_hash,
                failure_code=failure_code,
                release_class=ReleaseClass.RESTRICTED,
            )
        ),
    )

    artifacts = ArtifactStore(
        BlobStore(tmp_path / "cas", compression=Compression.GZIP),
        Ledger(tmp_path / "ledger.sqlite3"),
    )
    semantic = object.__new__(_CombinedSemanticExecutor)
    semantic.repository = Path(".")
    semantic.run_id = "repair-boundary-recovery"
    semantic.manifest = manifest
    semantic.runtime = runtime
    semantic.tokenizer = SimpleNamespace()
    semantic.tokenizer_manifest = SimpleNamespace()
    semantic.artifacts = artifacts
    semantic.state_root = tmp_path / "semantic"
    semantic.state_root.mkdir()
    repair_trace = {"exists": False}
    persist_count = {"ordinary": 0, "phase5": 0}
    fake_job_id = "TEST-ONLY-ledger-job"
    base = SimpleNamespace(
        job_id=fake_job_id,
        attempt_id=f"{semantic.run_id}-{call.call_id}-base",
        generated=SimpleNamespace(parsed_object={"nodes": []}),
        failure=None,
        raw_reference=SimpleNamespace(artifact_hash=digest("invalid-base-raw")),
        model_call_id=f"{semantic.run_id}-{call.call_id}-base-model",
        event_id=f"{semantic.run_id}-{call.call_id}-base-gpu",
        semantic=SimpleNamespace(content_hash=digest("base-semantic")),
        guided=base_guided,
    )
    repair = SimpleNamespace(
        job_id=fake_job_id,
        attempt_id=f"{semantic.run_id}-{call.call_id}-repair",
        generated=None,
        failure=RuntimeError("TEST-ONLY terminal repair failure"),
        raw_reference=None,
        model_call_id=f"{semantic.run_id}-{call.call_id}-repair-model",
        event_id=f"{semantic.run_id}-{call.call_id}-repair-gpu",
        semantic=repair_semantic,
        guided=repair_guided,
    )
    validation_failures: set[str] = set()

    def record_validation(**kwargs: object) -> str:
        attempt = kwargs["attempt"]
        if kwargs.get("error") is not None:
            validation_failures.add(attempt.attempt_id)
        return f"{attempt.attempt_id}-validation"

    semantic.helper = SimpleNamespace(
        _repair_request=lambda **_kwargs: (repair_semantic, repair_config),
        _repair_guided_request=lambda **_kwargs: repair_guided,
        _record_attempt_validation=record_validation,
        _advance_state=lambda *_args, **_kwargs: None,
        _record_projection=lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        artifacts.ledger,
        "failures_for_lineage",
        lambda attempt_id: (
            (SimpleNamespace(attempt_id=attempt_id),)
            if attempt_id in validation_failures
            else ()
        ),
    )
    semantic._attempt_trace_exists = lambda _call, suffix: (  # type: ignore[method-assign]
        suffix == "repair" and repair_trace["exists"]
    )
    semantic._restore_terminal_attempt = (  # type: ignore[method-assign]
        lambda **kwargs: repair if kwargs["repair"] else base
    )

    def persist_ordinary(**kwargs: object) -> object:
        assert kwargs["repair"] is True
        persist_count["ordinary"] += 1
        repair_trace["exists"] = True
        artifacts.ledger.record_gpu_event(
            event_id=repair.event_id,
            event_kind=GpuEventKind.REPAIR,
            allocated_seconds=1.0,
            started_at=started + timedelta(seconds=3),
            ended_at=started + timedelta(seconds=4),
            succeeded=False,
        )
        return repair

    def persist_phase5(**kwargs: object) -> object:
        assert kwargs["repair"] is True
        persist_count["phase5"] += 1
        repair_trace["exists"] = True
        artifacts.ledger.record_gpu_event(
            event_id=repair.event_id,
            event_kind=GpuEventKind.REPAIR,
            allocated_seconds=1.0,
            started_at=started + timedelta(seconds=3),
            ended_at=started + timedelta(seconds=4),
            succeeded=False,
        )
        return repair

    semantic.helper._persist_attempt = persist_ordinary
    semantic._persist_phase5_attempt = persist_phase5  # type: ignore[method-assign]

    def reject_base(**_kwargs: object) -> object:
        raise ValueError("TEST-ONLY invalid base")

    semantic._validate_generation = reject_base  # type: ignore[method-assign]
    started = manifest.created_at + timedelta(seconds=1)
    artifacts.ledger.record_gpu_event(
        event_id=base.event_id,
        event_kind=GpuEventKind.INFERENCE,
        allocated_seconds=2.0,
        started_at=started,
        ended_at=started + timedelta(seconds=2),
        succeeded=True,
    )

    class Authority:
        claim_calls = 0
        recover_calls = 0
        recovered_claim: CombinedRepairClaim | None = None

        def recover_claim(self, **kwargs: object) -> CombinedRepairClaim | None:
            self.recover_calls += 1
            assert kwargs["request_hash"] == request_hash
            assert kwargs["base_model_call_id"] == base.model_call_id
            return self.recovered_claim

        def claim(self, **kwargs: object) -> CombinedRepairClaim:
            self.claim_calls += 1
            if not reserve_available:
                raise CombinedProductionError(
                    "registered global short-repair reserve is exhausted"
                )
            if self.recovered_claim is None:
                self.recovered_claim = CombinedRepairClaim(
                    claim_id="combined-short-repair-1",
                    call_id=call.call_id,
                    call_spec_hash=call.content_hash,
                    request_hash=request_hash,
                    base_model_call_id=base.model_call_id,
                    global_slot_number=1,
                    claimed_at=kwargs["claimed_at"],
                )
            return self.recovered_claim

    authority = Authority()
    try:
        kwargs = {
            "service": SimpleNamespace(),
            "call": call,
            "semantic": SimpleNamespace(content_hash=digest("base-semantic")),
            "inputs": inputs,
            "repair_authority": authority,
            "authority_request_hash": request_hash,
            "remaining_required_seconds": 123.0,
            "phase5_request_hash": request_hash if phase5 else None,
        }
        first, first_attempts = semantic._recover_terminal_condition_attempt(**kwargs)
        second, second_attempts = semantic._recover_terminal_condition_attempt(**kwargs)

        disposition = semantic._read(
            semantic._repair_disposition_path(request_hash),
            factory._RepairDisposition,
        )
        expected_persists = 1 if reserve_available else 0
        assert sum(persist_count.values()) == expected_persists
        assert authority.claim_calls == 1
        assert disposition.decision == (
            "repair_claimed" if reserve_available else "reserve_exhausted"
        )
        assert len(first_attempts) == len(second_attempts) == (
            2 if reserve_available else 1
        )
        assert first.outcome is second.outcome
        assert artifacts.ledger.count_rows("gpu_events") == (
            2 if reserve_available else 1
        )
    finally:
        artifacts.ledger.close()


def test_semantic_completion_rebuilds_terminal_result_without_resending(
    tmp_path: Path,
) -> None:
    _configuration, _selections, _sources, _runtime, _gate, manifest = combined_fixture()
    call = manifest.calls[21]
    artifacts = ArtifactStore(
        BlobStore(tmp_path / "cas", compression=Compression.GZIP),
        Ledger(tmp_path / "ledger.sqlite3"),
    )
    semantic = object.__new__(_CombinedSemanticExecutor)
    semantic.run_id = "test-combined-run"
    semantic.calls_by_hash = {call.content_hash: call}
    semantic.artifacts = artifacts
    semantic.state_root = tmp_path / "semantic"
    semantic.state_root.mkdir()
    prepared_hash = digest("prepared-terminal-recovery")
    service_identity_hash = digest("service-terminal-recovery")
    started_at = manifest.created_at + timedelta(seconds=1)
    completed_at = started_at + timedelta(seconds=2)
    intent = _SemanticIntent(
        run_id=semantic.run_id,
        call_id=call.call_id,
        call_spec_hash=call.content_hash,
        prepared_hash=prepared_hash,
        guided_request_hash=digest("guided-terminal-recovery"),
        gpu_event_id=f"{semantic.run_id}-{call.call_id}-base-gpu",
        service_identity_hash=service_identity_hash,
        cumulative_gpu_seconds_before=0.0,
        remaining_required_seconds=100.0,
        created_at=started_at - timedelta(seconds=1),
    )
    try:
        job = artifacts.ledger.create_or_resume_job(
            {"call_id": call.call_id},
            release_class=ReleaseClass.RESTRICTED,
            created_at=started_at,
        )
        attempt = artifacts.ledger.record_attempt(
            attempt_id=f"{semantic.run_id}-{call.call_id}-base",
            job_id=job.job_id,
            attempt_kind=AttemptKind.BASE,
            input_hash=intent.guided_request_hash,
            config_hash=digest("terminal-config"),
            seed=call.vllm_seed,
            created_at=started_at,
        )
        event = artifacts.ledger.record_gpu_event(
            event_id=intent.gpu_event_id,
            event_kind=GpuEventKind.FAILURE,
            allocated_seconds=2.0,
            started_at=started_at,
            ended_at=completed_at,
            succeeded=False,
            job_id=job.job_id,
            attempt_id=attempt.attempt_id,
        )
        model_call = artifacts.ledger.record_model_call(
            model_call_id=f"{semantic.run_id}-{call.call_id}-base-model",
            job_id=job.job_id,
            attempt_id=attempt.attempt_id,
            gpu_event_id=event.event_id,
            backend=ModelBackend.VLLM_GPU,
            call_role=ModelCallRole.QUERY_TIME,
            retry_class=RetryClass.STANDARD,
            model_manifest_hash=digest("model"),
            decoding_manifest_hash=digest("decoder"),
            request_hash=intent.guided_request_hash,
            response_artifact_hash=None,
            construction_unit_hash=digest("construction"),
            served_context_count=1,
            prompt_tokens=0,
            completion_tokens=0,
            allocated_gpu_seconds=2.0,
            successful=False,
            created_at=completed_at,
        )
        condition_attempt = ConditionAttemptRecord(
            attempt_id="terminal-failed-attempt",
            condition=call.condition,
            unit_id=call.source.context_id,
            seed_block=call.seed_block,
            outcome=RunOutcome.FAILED,
            failure_code="test_terminal_failure",
            release_class=ReleaseClass.RESTRICTED,
        )
        completion = _OrdinarySemanticCompletion(
            run_id=semantic.run_id,
            call_id=call.call_id,
            call_spec_hash=call.content_hash,
            prepared_hash=prepared_hash,
            semantic_request_hash=digest("semantic-request"),
            semantic_intent_hash=intent.content_hash,
            service_identity_hash=service_identity_hash,
            condition_attempt=condition_attempt,
            model_call_ids=(model_call.model_call_id,),
            model_request_hashes=(intent.guided_request_hash,),
            cumulative_gpu_seconds_before=0.0,
            cumulative_gpu_seconds_after=2.0,
            started_at=started_at,
            completed_at=completed_at,
        )
        semantic._append(
            semantic._intent_path(prepared_hash, "ordinary"),
            intent,
            "test intent",
        )
        identity = SimpleNamespace(content_hash=service_identity_hash)
        with pytest.raises(CombinedRecoveryRequired, match="inflight or ambiguous"):
            semantic.recover_ordinary(prepared_hash, identity)
        semantic._append(
            semantic._completion_path(prepared_hash),
            completion,
            "test completion",
        )

        recovered = semantic.recover_ordinary(prepared_hash, identity)

        assert recovered is not None
        assert recovered.model_call_ids == (model_call.model_call_id,)
        assert recovered.outcome is RunOutcome.FAILED
        assert semantic.recover_ordinary(prepared_hash, identity) == recovered
        assert artifacts.ledger.count_rows("gpu_events") == 1
        assert artifacts.ledger.count_rows("model_calls") == 1
    finally:
        artifacts.ledger.close()
