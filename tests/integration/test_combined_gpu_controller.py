from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import story_projection_onto.combined_gpu_production as production
from story_projection_onto.combined_gpu_production import (
    CombinedGpuController,
    CombinedServiceIdentity,
    CombinedServiceShutdownReceipt,
)
from story_projection_onto.contracts import ConditionName
from story_projection_onto.phase5_execution import execute_phase5
from story_projection_onto.phase5_production import Phase5OwnedServiceIdentity
from story_projection_onto.store import (
    ArtifactStore,
    BlobStore,
    Compression,
    GpuEventKind,
    Ledger,
)
from tests.unit.test_combined_gpu_block import combined_fixture, digest
from tests.unit.test_phase5_execution import _Adapter, _inputs, _Verifier


def _artifacts(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(
        BlobStore(tmp_path / "cas", compression=Compression.GZIP),
        Ledger(tmp_path / "ledger.sqlite3"),
    )


class _Tick:
    def __init__(self, value):
        self.value = value

    def __call__(self):
        result = self.value
        self.value += timedelta(seconds=2)
        return result


class _Service:
    def __init__(self, identity, artifacts):
        self._identity = identity
        self._artifacts = artifacts

    def identity(self):
        return self._identity

    @property
    def actual_allocated_service_seconds(self):
        return self._artifacts.ledger.gpu_summary().total_allocated_seconds

    def execute_combined(self, *_args, **_kwargs):
        raise AssertionError("ordinary dispatch is replaced by the orchestration probe")

    def recover_combined(self, _request_hash, _repair_authority, _remaining):
        return None

    def execute_phase5_owned(self, *_args, **_kwargs):
        raise AssertionError("Phase 5 dispatch is replaced by the orchestration probe")

    def recover_phase5_owned(self, _request, _repair_authority, _remaining):
        return None


class _Lifecycle:
    def __init__(self, *, artifacts, runtime, clock):
        self.artifacts = artifacts
        self.runtime = runtime
        self.clock = clock
        self.activations = 0
        self.active_recoveries = 0
        self.shutdowns = 0
        self.shutdown_recoveries = 0
        self.service = None

    def activate(self, slot):
        self.activations += 1
        started = self.clock()
        ended = started + timedelta(seconds=1)
        event = self.artifacts.ledger.record_gpu_event(
            event_id="test-combined-only-model-load",
            event_kind=GpuEventKind.MODEL_LOAD,
            allocated_seconds=1,
            started_at=started,
            ended_at=ended,
            succeeded=True,
        )
        event_hash = production._record_hash(event, production._GPU_EVENT_FIELDS)
        phase5 = Phase5OwnedServiceIdentity(
            service_id="test-combined-service",
            global_accounting_id=slot.global_accounting_id,
            model_manifest_hash=self.runtime.model_manifest_hash,
            decoding_manifest_hash=self.runtime.decoding_manifest_hashes[
                ConditionName.C2_LLM_QUERY
            ],
            model_load_event_id=event.event_id,
            model_load_event_record_hash=event_hash,
            cumulative_gpu_seconds_at_handoff=(
                self.artifacts.ledger.gpu_summary().total_allocated_seconds
            ),
            handed_off_at=ended,
        )
        identity = CombinedServiceIdentity(
            service_id=phase5.service_id,
            global_accounting_id=slot.global_accounting_id,
            activation_slot_hash=slot.content_hash,
            runtime_binding_hash=self.runtime.content_hash,
            model_manifest_hash=self.runtime.model_manifest_hash,
            decoding_manifest_hashes=self.runtime.decoding_manifest_hashes,
            model_load_event_id=event.event_id,
            model_load_event_record_hash=event_hash,
            service_pid=4321,
            service_start_ticks=8765,
            gpu_seconds_before_load=slot.gpu_seconds_before,
            cumulative_gpu_seconds_after_load=(
                self.artifacts.ledger.gpu_summary().total_allocated_seconds
            ),
            activated_at=ended,
            phase5_identity=phase5,
        )
        self.service = _Service(identity, self.artifacts)
        return self.service

    def recover_active(self, _activation_slot_hash):
        self.active_recoveries += 1
        return self.service

    def shutdown(self, service, requested_at):
        self.shutdowns += 1
        assert service is self.service
        identity = service.identity()
        return CombinedServiceShutdownReceipt(
            service_identity_hash=identity.content_hash,
            service_pid=identity.service_pid,
            service_start_ticks=identity.service_start_ticks,
            shutdown_started_at=requested_at,
            stopped_at=requested_at,
            cumulative_gpu_seconds_after_shutdown=(
                self.artifacts.ledger.gpu_summary().total_allocated_seconds
            ),
        )

    def recover_shutdown(self, _service_identity_hash):
        self.shutdown_recoveries += 1
        return None


def _prebuilt_phase5_index(tmp_path: Path):
    protocol, gate, inputs = _inputs()
    index = execute_phase5(
        inputs=inputs,
        protocol=protocol,
        prerequisites=gate,
        adapter=_Adapter(),
        ledger_verifier=_Verifier(),
        output_root=tmp_path / "phase5-prebuilt",
        completed_at=inputs.frozen_at + timedelta(hours=1),
    )
    return protocol, gate, inputs, index


def _controller(tmp_path: Path):
    configuration, _selections, _sources, runtime, gate, manifest = combined_fixture()
    protocol, phase5_gate, phase5_inputs, phase5_index = _prebuilt_phase5_index(tmp_path)
    restricted_root = tmp_path / "artifacts" / "restricted"
    artifacts = _artifacts(restricted_root / "global")
    predecessor_start = manifest.created_at - timedelta(seconds=1002)
    artifacts.ledger.record_gpu_event(
        event_id="test-predecessor-accounting",
        event_kind=GpuEventKind.WARM_UP,
        allocated_seconds=gate.actual_allocated_gpu_seconds_before_block,
        started_at=predecessor_start,
        ended_at=predecessor_start + timedelta(seconds=1000),
        succeeded=True,
    )
    clock = _Tick(manifest.created_at + timedelta(seconds=10))
    lifecycle = _Lifecycle(artifacts=artifacts, runtime=runtime, clock=clock)
    controller = CombinedGpuController(
        run_id="test-combined-run",
        configuration=configuration,
        manifest=manifest,
        runtime=runtime,
        upstream_gate=gate,
        phase5_inputs=phase5_inputs,
        phase5_protocol=protocol,
        phase5_prerequisites=phase5_gate,
        provider=SimpleNamespace(),
        lifecycle_owner=lifecycle,
        artifacts=artifacts,
        output_root=restricted_root / "combined_gpu_block",
        clock=clock,
    )
    return controller, lifecycle, phase5_index


def test_combined_owner_dispatches_exact_order_with_one_load_and_one_shutdown(
    tmp_path: Path,
    monkeypatch,
) -> None:
    controller, lifecycle, phase5_index = _controller(tmp_path)
    dispatch_order: list[int] = []
    phase5_calls = tuple(controller.manifest.calls[12:21])
    monkeypatch.setattr(controller, "_validate_gate", lambda: phase5_calls)

    def ordinary(**kwargs):
        dispatch_order.append(kwargs["call"].ordinal)
        return SimpleNamespace(content_hash=digest(f"itt-{kwargs['call'].ordinal}"))

    def phase5(**kwargs):
        view = kwargs["adapter"].service
        assert view._service is lifecycle.service
        assert not hasattr(view, "shutdown")
        dispatch_order.extend(item.ordinal for item in view._calls.values())
        return phase5_index

    monkeypatch.setattr(production, "_execute_or_recover_ordinary", ordinary)
    monkeypatch.setattr(production, "execute_phase5", phase5)
    monkeypatch.setattr(production, "_validate_phase5_claims", lambda **_kwargs: ())
    monkeypatch.setattr(
        controller,
        "_validate_completed_work",
        lambda **_kwargs: ((), controller.manifest.created_at + timedelta(seconds=1)),
    )
    completed = object()
    monkeypatch.setattr(controller, "_complete", lambda **_kwargs: completed)

    assert controller.run() is completed
    assert dispatch_order == list(range(1, 50))
    assert lifecycle.activations == 1
    assert lifecycle.shutdowns == 1
    assert lifecycle.active_recoveries == 0
    assert lifecycle.shutdown_recoveries == 1
    loads = [
        item
        for item in controller.artifacts.ledger.gpu_events()
        if item.event_kind is GpuEventKind.MODEL_LOAD
    ]
    assert len(loads) == 1


def test_resume_after_physical_shutdown_finalizes_without_service_recovery_or_resend(
    tmp_path: Path,
    monkeypatch,
) -> None:
    controller, lifecycle, phase5_index = _controller(tmp_path)
    phase5_calls = tuple(controller.manifest.calls[12:21])
    monkeypatch.setattr(controller, "_validate_gate", lambda: phase5_calls)
    controller.journal.initialize()
    controller.journal.append(Path("manifest.json"), controller.manifest)
    slot, fresh = controller._activation_slot()
    service, identity = controller._obtain_service(slot, fresh)
    controller.journal.append(Path("phase5_index.json"), phase5_index)
    stopped_at = controller.clock()
    shutdown = CombinedServiceShutdownReceipt(
        service_identity_hash=identity.content_hash,
        service_pid=identity.service_pid,
        service_start_ticks=identity.service_start_ticks,
        shutdown_started_at=stopped_at,
        stopped_at=stopped_at,
        cumulative_gpu_seconds_after_shutdown=(
            controller.artifacts.ledger.gpu_summary().total_allocated_seconds
        ),
    )
    controller.journal.append(Path("shutdown_receipt.json"), shutdown)
    assert service is lifecycle.service
    before_counts = (
        lifecycle.activations,
        lifecycle.active_recoveries,
        lifecycle.shutdowns,
        lifecycle.shutdown_recoveries,
    )

    fake_ordinary = tuple(
        SimpleNamespace(
            content_hash=digest(f"recovered-ordinary-{index}"),
            outcome="failed",
        )
        for index in range(40)
    )
    monkeypatch.setattr(production, "_all_required_work_exists", lambda *_args: True)
    monkeypatch.setattr(
        production,
        "_load_verified_ordinary_records",
        lambda **_kwargs: fake_ordinary,
    )
    monkeypatch.setattr(production, "_validate_phase5_claims", lambda **_kwargs: ())
    monkeypatch.setattr(
        controller,
        "_validate_completed_work",
        lambda **_kwargs: ((), stopped_at - timedelta(seconds=1)),
    )
    index = controller.run()

    assert index.vllm_service_stopped is True
    assert index.base_call_count == 49
    assert lifecycle.activations == before_counts[0]
    assert lifecycle.active_recoveries == before_counts[1]
    assert lifecycle.shutdowns == before_counts[2]
    assert lifecycle.shutdown_recoveries == before_counts[3]
    assert controller.journal.exists(Path("execution_index.json"))


def test_activation_replay_is_bound_to_run_id(tmp_path: Path) -> None:
    controller, _lifecycle, _phase5_index = _controller(tmp_path)
    controller.journal.initialize()
    _slot, fresh = controller._activation_slot()
    assert fresh is True
    controller.run_id = "different-combined-run"
    try:
        controller._activation_slot()
    except production.CombinedProductionError as error:
        assert "activation intent changed" in str(error)
    else:
        raise AssertionError("combined activation replay accepted another run_id")


def test_resume_after_unconsumed_activation_slot_performs_first_load_once(
    tmp_path: Path,
) -> None:
    controller, lifecycle, _phase5_index = _controller(tmp_path)
    controller.journal.initialize()
    controller.journal.append(Path("manifest.json"), controller.manifest)
    slot, fresh = controller._activation_slot()
    assert fresh is True

    # Simulate loss of the first controller before lifecycle.activate().  The
    # second invocation sees the durable slot but no physical-lifecycle trace.
    replayed, replay_is_fresh = controller._activation_slot()
    assert replayed == slot
    assert replay_is_fresh is False
    service, identity = controller._obtain_service(replayed, replay_is_fresh)

    assert service is lifecycle.service
    assert identity == service.identity()
    assert lifecycle.active_recoveries == 1
    assert lifecycle.activations == 1
    assert controller.journal.exists(Path("service_identity.json"))
    assert len(
        [
            item
            for item in controller.artifacts.ledger.gpu_events()
            if item.event_kind is GpuEventKind.MODEL_LOAD
        ]
    ) == 1


def test_post_acquisition_controller_failure_terminalizes_service_and_receipts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    controller, lifecycle, _phase5_index = _controller(tmp_path)
    phase5_calls = tuple(controller.manifest.calls[12:21])
    monkeypatch.setattr(controller, "_validate_gate", lambda: phase5_calls)
    monkeypatch.setattr(
        production,
        "_execute_or_recover_ordinary",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("injected controller crash")),
    )

    try:
        controller.run()
    except RuntimeError as error:
        assert "injected controller crash" in str(error)
    else:
        raise AssertionError("injected post-acquisition crash was swallowed")

    assert lifecycle.activations == 1
    assert lifecycle.shutdowns == 1
    assert controller.journal.exists(Path("shutdown_receipt.json"))
    receipt = controller.journal.load(
        Path("shutdown_receipt.json"),
        CombinedServiceShutdownReceipt,
    )
    assert receipt.service_identity_hash == lifecycle.service.identity().content_hash


def test_identity_append_crash_after_activation_still_terminalizes_exact_service(
    tmp_path: Path,
    monkeypatch,
) -> None:
    controller, lifecycle, _phase5_index = _controller(tmp_path)
    phase5_calls = tuple(controller.manifest.calls[12:21])
    monkeypatch.setattr(controller, "_validate_gate", lambda: phase5_calls)
    original_append = controller.journal.append
    failed_once = False

    def crash_once(relative, value):
        nonlocal failed_once
        if relative == Path("service_identity.json") and not failed_once:
            failed_once = True
            raise RuntimeError("injected identity publication crash")
        return original_append(relative, value)

    monkeypatch.setattr(controller.journal, "append", crash_once)
    try:
        controller.run()
    except RuntimeError as error:
        assert "identity publication crash" in str(error)
    else:
        raise AssertionError("injected identity publication crash was swallowed")

    assert lifecycle.activations == 1
    assert lifecycle.active_recoveries == 1
    assert lifecycle.shutdowns == 1
    assert controller.journal.exists(Path("service_identity.json"))
    assert controller.journal.exists(Path("shutdown_receipt.json"))
