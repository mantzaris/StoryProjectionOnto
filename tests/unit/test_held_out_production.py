from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from story_projection_onto.contracts import ConditionName, PreQueryInventory
from story_projection_onto.development_runtime import RunOutcome
from story_projection_onto.held_out_primary import (
    HeldOutCallSpec,
    HeldOutCASReference,
    HeldOutServiceResult,
)
from story_projection_onto.held_out_production import (
    HeldOutArtifactResolver,
    HeldOutProductionError,
    HeldOutScheduleState,
    HeldOutServiceActivationIntent,
    ProductionHeldOutBundle,
    ProductionHeldOutRuntime,
    ProductionHeldOutSession,
    SequentialVLLMConditionActivator,
)
from story_projection_onto.store import (
    ArtifactStore,
    BlobStore,
    Compression,
    GpuEventKind,
    Ledger,
    ReleaseClass,
)

NOW = datetime(2026, 1, 1, tzinfo=UTC)


class _FakeResumableService:
    def __init__(
        self,
        pid: int,
        *,
        resumable: bool = True,
        live_lease_resumable: bool = False,
        terminal_recovery: object | None = None,
        ledger: Ledger | None = None,
    ) -> None:
        self._pid = pid
        self.resumable = resumable
        self.starts = []
        self.resume_calls = 0
        self.shutdown_calls = 0
        self.detach_calls = 0
        self.stale_recovery_calls = 0
        self.live_lease_recovery_calls = 0
        self.live_lease_resumable = live_lease_resumable
        self.terminal_recovery = terminal_recovery
        self.ledger = ledger
        self.last_recovered_process_identity = None
        self.session_id: str | None = None
        self.accounting_session_id: str | None = None

    @property
    def pid(self) -> int:
        return self._pid

    @property
    def actual_allocated_service_seconds(self) -> float:
        if self.ledger is None:
            return 0.0
        return self.ledger.gpu_summary().total_allocated_seconds

    def start(self, **kwargs) -> None:
        self.starts.append(kwargs)
        self.session_id = kwargs["session_id"]
        self.accounting_session_id = kwargs["event_id"]
        if self.ledger is not None:
            self.ledger.record_gpu_event(
                event_id=self.accounting_session_id,
                event_kind=GpuEventKind.MODEL_LOAD,
                allocated_seconds=1,
                started_at=NOW,
                ended_at=NOW + timedelta(seconds=1),
                succeeded=True,
            )

    def write_resume_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        assert self.session_id is not None
        assert self.accounting_session_id is not None
        path.write_text(
            json.dumps(
                {
                    "pid": self.pid,
                    "process_start_ticks": self.pid * 10,
                    "accounting_session_id": self.accounting_session_id,
                    "session_id": self.session_id,
                    "configuration_hash": "a" * 64,
                }
            ),
            encoding="utf-8",
        )

    def resume_from_checkpoint(self, path: Path) -> bool:
        self.resume_calls += 1
        if not self.resumable:
            return False
        value = json.loads(path.read_text(encoding="utf-8"))
        self._pid = value["pid"]
        self.session_id = value["session_id"]
        self.accounting_session_id = value["accounting_session_id"]
        return True

    def generate(self, *_args, **_kwargs):
        raise AssertionError("activation tests must not generate")

    def shutdown(self) -> None:
        self.shutdown_calls += 1
        if (
            self.ledger is not None
            and self.session_id is not None
            and self.accounting_session_id is not None
        ):
            self.ledger.record_gpu_service_session(
                service_session_id=self.accounting_session_id,
                session_id=self.session_id,
                service_seconds=2,
                classified_event_seconds=1,
                started_at=NOW,
                ended_at=NOW + timedelta(seconds=2),
                details={"test_only": True},
            )

    def recover_stale_service_lease(self):
        self.stale_recovery_calls += 1
        if self.terminal_recovery is not None:
            self.last_recovered_process_identity = SimpleNamespace(
                pid=self.pid,
                process_start_ticks=self.pid * 10,
                configuration_hash="a" * 64,
                session_id=self.session_id,
                accounting_session_id=self.accounting_session_id,
            )
        return self.terminal_recovery

    def resume_live_service_lease(self, **kwargs) -> bool:
        self.live_lease_recovery_calls += 1
        if self.live_lease_resumable:
            self.session_id = kwargs["expected_session_id"]
            self.accounting_session_id = kwargs["expected_event_id"]
        return self.live_lease_resumable

    def detach_for_controller_restart(self, path: Path) -> None:
        self.detach_calls += 1
        self.write_resume_checkpoint(path)


def _activation_intent(
    activator: SequentialVLLMConditionActivator,
    *,
    remaining_required_seconds: float,
) -> HeldOutServiceActivationIntent:
    return HeldOutServiceActivationIntent(
        condition=activator.condition,
        call_manifest_hash="e" * 64,
        development_execution_result_hash="f" * 64,
        global_accounting_id="TEST-ONLY-global",
        expected_session_id=(
            f"{activator.owner_run_id}-{activator.condition.value}-service"
        ),
        expected_model_load_event_id=(
            f"{activator.owner_run_id}-{activator.condition.value}-model-load"
        ),
        launcher_configuration_hash=activator.launcher_configuration_hash,
        service_start_watchdog_seconds=activator.service_start_watchdog_seconds,
        remaining_registered_seconds_before=(
            remaining_required_seconds + 180
        ),
        requested_at=NOW,
    )


def _activate(
    activator: SequentialVLLMConditionActivator,
    *,
    remaining_required_seconds: float,
    resume_required: bool = False,
):
    return activator(
        remaining_required_seconds=remaining_required_seconds,
        resume_required=resume_required,
        activation_intent=_activation_intent(
            activator,
            remaining_required_seconds=remaining_required_seconds,
        ),
    )


def _production_runtime_for_activation_failure(
    tmp_path: Path,
) -> tuple[
    ProductionHeldOutRuntime,
    _FakeResumableService,
    SequentialVLLMConditionActivator,
    Ledger,
]:
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    artifacts = ArtifactStore(
        BlobStore(tmp_path / "cas", compression=Compression.GZIP),
        ledger,
    )
    manifest_id = "TEST-ONLY-held-out-orphan"
    condition = ConditionName.C1_LLM_PRE
    service = _FakeResumableService(4101, ledger=ledger)
    activator = SequentialVLLMConditionActivator(
        condition=condition,
        service_factory=lambda: service,
        checkpoint_path=tmp_path / "runtime" / "c1-service.json",
        owner_run_id=manifest_id,
        launcher_configuration_hash="a" * 64,
        model_snapshot_hash="b" * 64,
        selected_model_freeze_hash="c" * 64,
        source_execution_hash="d" * 64,
    )
    runtime = object.__new__(ProductionHeldOutRuntime)
    runtime.repository = tmp_path.resolve()
    runtime.reviewed_plan = SimpleNamespace(
        call_manifest=SimpleNamespace(
            manifest_id=manifest_id,
            content_hash="e" * 64,
        )
    )
    runtime.configuration = SimpleNamespace(
        hard_gpu_seconds_limit=20_000,
        scheduled_gpu_seconds_limit=10_000,
        service_start_watchdog_seconds=180,
        gpu_call_inventory_file_sha256="1" * 64,
    )
    runtime.development_result = SimpleNamespace(content_hash="f" * 64)
    runtime.development_predecessor_ledger_hash = "2" * 64
    runtime.artifacts = artifacts
    runtime.tokenizer = object()
    runtime.neutral_by_model_visible_hash = {}
    runtime.state_path = tmp_path / "runtime" / "held-out-state.json"
    runtime.semantic_executor = object()
    runtime.activators = {condition: activator}
    runtime.clock = lambda: NOW
    runtime._openings = {}
    runtime._packets = {}
    runtime._services = {}
    runtime._identities = {}
    runtime._shutdowns = {}
    runtime._write_state(
        HeldOutScheduleState(
            call_manifest_hash="e" * 64,
            development_execution_result_hash="f" * 64,
            development_predecessor_ledger_hash="2" * 64,
            development_predecessor_allocated_gpu_seconds=0,
            remaining_registered_p95_seconds=1_000,
            updated_at=NOW,
        )
    )
    return runtime, service, activator, ledger


@pytest.mark.parametrize("error_type", (OSError, RuntimeError))
def test_post_activation_state_failure_is_terminal_accounted_and_not_duplicated(
    tmp_path: Path,
    monkeypatch,
    error_type: type[Exception],
) -> None:
    runtime, service, activator, ledger = _production_runtime_for_activation_failure(
        tmp_path
    )
    condition = ConditionName.C1_LLM_PRE
    original_replace_state = ProductionHeldOutRuntime._replace_state
    injected_writes = 0

    def fail_identity_writes(self, state, **updates) -> None:
        nonlocal injected_writes
        if self is runtime and "session_identities" in updates:
            injected_writes += 1
            raise error_type("private post-start state-write failure")
        original_replace_state(self, state, **updates)

    monkeypatch.setattr(
        ProductionHeldOutRuntime,
        "_replace_state",
        fail_identity_writes,
    )
    try:
        with pytest.raises(error_type, match="post-start state-write"):
            runtime.activate(condition)
        monkeypatch.setattr(
            ProductionHeldOutRuntime,
            "_replace_state",
            original_replace_state,
        )

        event_id = f"{activator.owner_run_id}-{condition.value}-model-load"
        terminal = ledger.get_gpu_service_session(event_id)
        assert terminal is not None
        assert terminal.session_id == (
            f"{activator.owner_run_id}-{condition.value}-service"
        )
        assert service.shutdown_calls == 1
        assert len(service.starts) == 1
        assert len(ledger.gpu_events()) == 1
        assert ledger.gpu_summary().service_session_count == 1
        assert not runtime.service_available(condition)
        assert injected_writes == 3

        state = runtime._read_state()
        assert state.consumed_primary_service_start_count == 0
        assert state.remaining_registered_p95_seconds == 1_000
        assert condition.value not in state.session_identities
        assert condition.value not in state.shutdown_intents
        assert condition.value not in state.shutdown_receipts

        cleanup_calls: list[str] = []
        bundle = ProductionHeldOutBundle(
            cpu=object(),
            sessions={},
            runtime=runtime,
            close_callback=lambda: cleanup_calls.append("closed"),
        )
        if error_type is OSError:
            bundle.preserve_for_resume()
        else:
            bundle.close()
        assert cleanup_calls == ["closed"]
        assert service.shutdown_calls == 1

        replacement = _FakeResumableService(
            4102,
            resumable=False,
            terminal_recovery=object(),
            ledger=ledger,
        )
        runtime._identities.clear()
        runtime._shutdowns.clear()
        runtime.activators = {
            condition: SequentialVLLMConditionActivator(
                condition=condition,
                service_factory=lambda: replacement,
                checkpoint_path=activator.checkpoint_path,
                owner_run_id=activator.owner_run_id,
                launcher_configuration_hash="a" * 64,
                model_snapshot_hash="b" * 64,
                selected_model_freeze_hash="c" * 64,
                source_execution_hash="d" * 64,
            )
        }
        recovered_identity = runtime.activate(condition)
        assert replacement.starts == []
        assert replacement.stale_recovery_calls == 1
        assert len(ledger.gpu_events()) == 1
        assert ledger.gpu_summary().service_session_count == 1
        recovered_state = runtime._read_state()
        assert recovered_state.consumed_primary_service_start_count == 1
        assert recovered_state.remaining_registered_p95_seconds == 820
        assert recovered_state.session_identities[condition.value] == recovered_identity
        assert condition.value in recovered_state.shutdown_intents
        assert condition.value in recovered_state.shutdown_receipts
    finally:
        ledger.close()


def test_typed_held_out_cas_resolves_physical_and_logical_hashes(
    tmp_path: Path,
) -> None:
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    try:
        resolver = HeldOutArtifactResolver(
            ArtifactStore(
                BlobStore(tmp_path / "cas", compression=Compression.GZIP),
                ledger,
            )
        )
        inventory = PreQueryInventory(
            inventory_id="TEST-ONLY-empty-c2",
            snapshot_hash="a" * 64,
            recorded_at=NOW,
        )
        reference = resolver.persist_record(
            inventory,
            object_kind="pre_query_inventory",
            release_class=ReleaseClass.RESTRICTED,
            created_at=NOW,
        )
        assert (
            resolver.resolve_record(
                reference,
                PreQueryInventory,
                required_release=ReleaseClass.RESTRICTED,
            )
            == inventory
        )

        changed = HeldOutCASReference(
            **reference.model_dump(
                mode="python",
                exclude={"content_hash", "logical_content_hash"},
            ),
            logical_content_hash="b" * 64,
        )
        with pytest.raises(HeldOutProductionError, match="logical object hash"):
            resolver.resolve_record(changed, PreQueryInventory)
    finally:
        ledger.close()


def test_durable_schedule_refuses_unmetered_or_duplicate_service_identity() -> None:
    with pytest.raises(ValidationError, match="service-start count"):
        HeldOutScheduleState(
            call_manifest_hash="a" * 64,
            development_execution_result_hash="b" * 64,
            development_predecessor_ledger_hash="c" * 64,
            development_predecessor_allocated_gpu_seconds=100,
            remaining_registered_p95_seconds=200,
            consumed_primary_service_start_count=1,
            updated_at=NOW,
        )


def test_three_condition_activators_create_three_sequential_180_second_services(
    tmp_path: Path,
) -> None:
    services: list[_FakeResumableService] = []

    def factory() -> _FakeResumableService:
        service = _FakeResumableService(1000 + len(services))
        services.append(service)
        return service

    activators = tuple(
        SequentialVLLMConditionActivator(
            condition=condition,
            service_factory=factory,
            checkpoint_path=tmp_path / f"{condition.value}.json",
            owner_run_id="TEST-ONLY-held-out",
            launcher_configuration_hash="a" * 64,
            model_snapshot_hash="b" * 64,
            selected_model_freeze_hash="c" * 64,
            source_execution_hash="d" * 64,
        )
        for condition in (
            ConditionName.C1_LLM_PRE,
            ConditionName.C2_LLM_QUERY,
            ConditionName.A_FIXED_SELECT,
        )
    )
    identities = []
    for index, activator in enumerate(activators):
        service, identity, _ = _activate(
            activator,
            remaining_required_seconds=1000 - index,
        )
        identities.append(identity)
        service.shutdown()

    assert len(services) == 3
    assert len({item.pid for item in services}) == 3
    assert all(len(item.starts) == 1 for item in services)
    assert all(item.starts[0]["watchdog_seconds"] == 180 for item in services)
    assert len({item.content_hash for item in identities}) == 3
    assert all(item.shutdown_calls == 1 for item in services)
    with pytest.raises(HeldOutProductionError, match="allocate twice"):
        _activate(activators[0], remaining_required_seconds=0)


def test_startup_watchdog_is_separate_from_registered_180_second_charge(
    tmp_path: Path,
) -> None:
    service = _FakeResumableService(1777)
    activator = SequentialVLLMConditionActivator(
        condition=ConditionName.C1_LLM_PRE,
        service_factory=lambda: service,
        checkpoint_path=tmp_path / "watchdog.json",
        owner_run_id="TEST-ONLY-watchdog",
        launcher_configuration_hash="a" * 64,
        model_snapshot_hash="b" * 64,
        selected_model_freeze_hash="c" * 64,
        source_execution_hash="d" * 64,
        service_start_watchdog_seconds=300,
    )
    _activate(activator, remaining_required_seconds=900)
    assert service.starts[0]["watchdog_seconds"] == 300

    with pytest.raises(HeldOutProductionError, match="between 180 and 600"):
        SequentialVLLMConditionActivator(
            condition=ConditionName.C1_LLM_PRE,
            service_factory=lambda: _FakeResumableService(1888),
            checkpoint_path=tmp_path / "too-short.json",
            owner_run_id="TEST-ONLY-too-short",
            launcher_configuration_hash="a" * 64,
            model_snapshot_hash="b" * 64,
            selected_model_freeze_hash="c" * 64,
            source_execution_hash="d" * 64,
            service_start_watchdog_seconds=179,
        )


def test_condition_activator_adopts_checkpoint_without_second_load(tmp_path: Path) -> None:
    checkpoint = tmp_path / "c1.json"
    first = _FakeResumableService(2001)
    first_activator = SequentialVLLMConditionActivator(
        condition=ConditionName.C1_LLM_PRE,
        service_factory=lambda: first,
        checkpoint_path=checkpoint,
        owner_run_id="TEST-ONLY-resume",
        launcher_configuration_hash="a" * 64,
        model_snapshot_hash="b" * 64,
        selected_model_freeze_hash="c" * 64,
        source_execution_hash="d" * 64,
    )
    _activate(first_activator, remaining_required_seconds=900)
    adopted = _FakeResumableService(9999)
    second_activator = SequentialVLLMConditionActivator(
        condition=first_activator.condition,
        service_factory=lambda: adopted,
        checkpoint_path=checkpoint,
        owner_run_id="TEST-ONLY-resume",
        launcher_configuration_hash="a" * 64,
        model_snapshot_hash="b" * 64,
        selected_model_freeze_hash="c" * 64,
        source_execution_hash="d" * 64,
    )
    _, identity, _ = _activate(
        second_activator,
        remaining_required_seconds=900,
        resume_required=True,
    )
    assert adopted.resume_calls == 1
    assert adopted.starts == []
    assert identity.service_pid == first.pid


def test_condition_activator_adopts_live_lease_after_precheckpoint_crash(
    tmp_path: Path,
) -> None:
    service = _FakeResumableService(2051, live_lease_resumable=True)
    activator = SequentialVLLMConditionActivator(
        condition=ConditionName.C1_LLM_PRE,
        service_factory=lambda: service,
        checkpoint_path=tmp_path / "missing-after-start.json",
        owner_run_id="TEST-ONLY-precheckpoint",
        launcher_configuration_hash="a" * 64,
        model_snapshot_hash="b" * 64,
        selected_model_freeze_hash="c" * 64,
        source_execution_hash="d" * 64,
    )
    outcome = _activate(
        activator,
        remaining_required_seconds=900,
        resume_required=True,
    )
    assert outcome.service_ready
    assert service.live_lease_recovery_calls == 1
    assert service.starts == []
    assert activator.checkpoint_path.is_file()
    assert outcome.live_identity.service_pid == 2051


def test_condition_activator_rejects_unpersisted_or_changed_intent(
    tmp_path: Path,
) -> None:
    created: list[_FakeResumableService] = []
    activator = SequentialVLLMConditionActivator(
        condition=ConditionName.C1_LLM_PRE,
        service_factory=lambda: created.append(_FakeResumableService(2071)) or created[-1],
        checkpoint_path=tmp_path / "intent.json",
        owner_run_id="TEST-ONLY-intent",
        launcher_configuration_hash="a" * 64,
        model_snapshot_hash="b" * 64,
        selected_model_freeze_hash="c" * 64,
        source_execution_hash="d" * 64,
    )
    changed = _activation_intent(activator, remaining_required_seconds=900)
    changed_payload = changed.model_dump(mode="python", exclude={"content_hash"})
    changed_payload["expected_model_load_event_id"] = "TEST-ONLY-wrong-event"
    with pytest.raises(HeldOutProductionError, match="durable intent"):
        activator(
            remaining_required_seconds=900,
            activation_intent=HeldOutServiceActivationIntent.model_validate(changed_payload),
        )
    assert created == []


def test_condition_activator_rejects_checkpoint_from_another_run_and_stops_it(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "c1-other-run.json"
    original = _FakeResumableService(2111)
    first = SequentialVLLMConditionActivator(
        condition=ConditionName.C1_LLM_PRE,
        service_factory=lambda: original,
        checkpoint_path=checkpoint,
        owner_run_id="TEST-ONLY-original-run",
        launcher_configuration_hash="a" * 64,
        model_snapshot_hash="b" * 64,
        selected_model_freeze_hash="c" * 64,
        source_execution_hash="d" * 64,
    )
    _activate(first, remaining_required_seconds=900)

    adopted = _FakeResumableService(2222)
    wrong_run = SequentialVLLMConditionActivator(
        condition=ConditionName.C1_LLM_PRE,
        service_factory=lambda: adopted,
        checkpoint_path=checkpoint,
        owner_run_id="TEST-ONLY-other-run",
        launcher_configuration_hash="a" * 64,
        model_snapshot_hash="b" * 64,
        selected_model_freeze_hash="c" * 64,
        source_execution_hash="d" * 64,
    )
    with pytest.raises(HeldOutProductionError, match="checkpoint identity changed"):
        _activate(wrong_run, remaining_required_seconds=900, resume_required=True)
    assert adopted.resume_calls == 0
    assert adopted.shutdown_calls == 1


def test_condition_activator_reconciles_stopped_timeout_without_new_load(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "c2-stopped.json"
    original = _FakeResumableService(2444)
    first = SequentialVLLMConditionActivator(
        condition=ConditionName.C2_LLM_QUERY,
        service_factory=lambda: original,
        checkpoint_path=checkpoint,
        owner_run_id="TEST-ONLY-timeout-resume",
        launcher_configuration_hash="a" * 64,
        model_snapshot_hash="b" * 64,
        selected_model_freeze_hash="c" * 64,
        source_execution_hash="d" * 64,
    )
    _activate(first, remaining_required_seconds=900)

    stopped = _FakeResumableService(
        2555,
        resumable=False,
        terminal_recovery=object(),
    )
    stopped.session_id = "TEST-ONLY-timeout-resume-C2-service"
    stopped.accounting_session_id = "TEST-ONLY-timeout-resume-C2-model-load"
    resumed = SequentialVLLMConditionActivator(
        condition=ConditionName.C2_LLM_QUERY,
        service_factory=lambda: stopped,
        checkpoint_path=checkpoint,
        owner_run_id="TEST-ONLY-timeout-resume",
        launcher_configuration_hash="a" * 64,
        model_snapshot_hash="b" * 64,
        selected_model_freeze_hash="c" * 64,
        source_execution_hash="d" * 64,
    )
    outcome = _activate(resumed, remaining_required_seconds=900, resume_required=True)
    _, identity, _ = outcome
    assert stopped.resume_calls == 1
    assert stopped.stale_recovery_calls == 1
    assert stopped.starts == []
    assert identity.service_pid == original.pid
    assert not outcome.service_ready


def test_condition_activator_cannot_replace_a_missing_persisted_service(
    tmp_path: Path,
) -> None:
    created: list[_FakeResumableService] = []

    def factory() -> _FakeResumableService:
        service = _FakeResumableService(3001)
        created.append(service)
        return service

    activator = SequentialVLLMConditionActivator(
        condition=ConditionName.C2_LLM_QUERY,
        service_factory=factory,
        checkpoint_path=tmp_path / "missing.json",
        owner_run_id="TEST-ONLY-resume-required",
        launcher_configuration_hash="a" * 64,
        model_snapshot_hash="b" * 64,
        selected_model_freeze_hash="c" * 64,
        source_execution_hash="d" * 64,
    )
    with pytest.raises(HeldOutProductionError, match="replacement model load is forbidden"):
        _activate(activator, remaining_required_seconds=900, resume_required=True)
    assert len(created) == 1
    assert created[0].starts == []


def test_unstarted_fixed_select_slot_decrements_registered_forecast() -> None:
    class Runtime:
        def __init__(self) -> None:
            self.state = HeldOutScheduleState(
                call_manifest_hash="a" * 64,
                development_execution_result_hash="b" * 64,
                development_predecessor_ledger_hash="c" * 64,
                development_predecessor_allocated_gpu_seconds=100,
                remaining_registered_p95_seconds=1000,
                active_call_id="TEST-ONLY-fixed-skip",
                updated_at=NOW,
            )

        def _read_state(self):
            return self.state

        def _replace_state(self, state, **updates) -> None:
            payload = state.model_dump(mode="python", exclude={"content_hash"})
            payload.update(updates)
            payload["updated_at"] = NOW
            self.state = HeldOutScheduleState.model_validate(payload)

        def schedule_snapshot(self):
            return self.state.remaining_registered_p95_seconds

    call = HeldOutCallSpec(
        ordinal=1,
        call_id="TEST-ONLY-fixed-skip",
        call_class="test_fixed_select",
        condition=ConditionName.A_FIXED_SELECT,
        unit_id="TEST-ONLY-unit",
        seed_block=1,
        frozen_seed=1,
        vllm_seed=1,
        p95_seconds=72,
        watchdog_seconds=90,
        repair_reserve_class="reserve_short",
        prequery_stage_hash="c" * 64,
        query_stage_hash="d" * 64,
        source_c1_call_id="TEST-ONLY-c1-source",
        construction_operations_permitted=False,
    )
    result = HeldOutServiceResult(
        call_id=call.call_id,
        condition=call.condition,
        outcome=RunOutcome.FAILED,
        request_started=False,
        global_ledger_chain_hash="e" * 64,
        allocated_gpu_seconds=0,
        repair_attempts=0,
        failure_code="source_c1_unavailable",
        completed_at=NOW,
    )
    runtime = Runtime()
    session = ProductionHeldOutSession(
        condition=ConditionName.A_FIXED_SELECT,
        runtime=runtime,  # type: ignore[arg-type]
    )
    snapshot = session.finalize_unstarted_call(call, result)
    assert snapshot == 928
    assert runtime.state.remaining_registered_p95_seconds == 928
    assert runtime.state.completed_result_hashes == {call.call_id: result.content_hash}
    assert runtime.state.active_call_id is None


def test_bundle_preserves_healthy_resume_and_fails_safe_before_closing_ledger() -> None:
    class Runtime:
        def __init__(self, *, detach_fails: bool = False, shutdown_fails: bool = False):
            self._services = {ConditionName.C2_LLM_QUERY: object()}
            self._identities = {ConditionName.C2_LLM_QUERY: object()}
            self.detach_fails = detach_fails
            self.shutdown_fails = shutdown_fails
            self.detach_calls = 0
            self.shutdown_calls = 0

        def detach_live_services_for_resume(self) -> None:
            self.detach_calls += 1
            if self.detach_fails:
                raise RuntimeError("TEST-ONLY detach failure")
            self._services.clear()

        def shutdown_condition(self, _identity) -> None:
            self.shutdown_calls += 1
            if self.shutdown_fails:
                raise RuntimeError("TEST-ONLY shutdown failure")
            self._services.clear()

    closed: list[str] = []
    healthy_runtime = Runtime()
    healthy = ProductionHeldOutBundle(
        cpu=object(),
        sessions={},
        runtime=healthy_runtime,  # type: ignore[arg-type]
        close_callback=lambda: closed.append("healthy"),
    )
    healthy.preserve_for_resume()
    assert healthy_runtime.detach_calls == 1
    assert healthy_runtime.shutdown_calls == 0
    assert closed == ["healthy"]

    failed_runtime = Runtime(detach_fails=True, shutdown_fails=True)
    failed = ProductionHeldOutBundle(
        cpu=object(),
        sessions={},
        runtime=failed_runtime,  # type: ignore[arg-type]
        close_callback=lambda: closed.append("unsafe"),
    )
    with pytest.raises(RuntimeError, match="shutdown failure"):
        failed.preserve_for_resume()
    assert failed_runtime.detach_calls == 1
    assert failed_runtime.shutdown_calls == 1
    assert closed == ["healthy"]
