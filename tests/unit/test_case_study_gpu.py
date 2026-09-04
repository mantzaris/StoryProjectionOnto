from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from story_projection_onto.case_study_execution import (
    CASE_SERVICE_START_WATCHDOG_SECONDS,
    CaseArtifactReference,
    CaseExecutionAdmissionReceipt,
    CaseExecutionRepository,
    CaseGpuCallAuditReceipt,
    CaseStudyAdmissionError,
    CaseStudyProductionController,
    _parse_record,
)
from story_projection_onto.case_study_gpu import (
    CaseGpuShutdownReceipt,
    ProductionCaseStudyGpuAdapter,
    build_production_case_study_gpu_adapter,
)
from story_projection_onto.case_study_runtime import (
    CaseC1RequestEnvelope,
    CaseGpuCallSlot,
    CaseModelRuntimeBinding,
    CaseStudyExecutionPlan,
    load_attested_restricted_case_study,
)
from story_projection_onto.contracts import (
    AbstractionLevel,
    ConditionName,
    RunOutcome,
    canonical_json,
    canonical_sha256,
)
from story_projection_onto.development_adapter import (
    DevelopmentConstructionConfiguration,
)
from story_projection_onto.gpu_runtime import (
    GenerationResult,
    GuidedJSONRequest,
    ServiceState,
    TokenizerManifest,
)
from story_projection_onto.llm import CapabilityManifest, base_condition_output_schema
from story_projection_onto.manifest import build_source_manifest
from story_projection_onto.store import (
    ArtifactStore,
    BlobStore,
    Compression,
    GpuEventKind,
    Ledger,
    ReleaseClass,
)

ROOT = Path(__file__).resolve().parents[2]
T0 = datetime(2026, 9, 4, 12, 0, tzinfo=UTC)


def _fixture_module(path: str, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _Clock:
    def __init__(self, value: datetime) -> None:
        self.value = value

    def __call__(self) -> datetime:
        self.value += timedelta(microseconds=10)
        return self.value

    def advance_past(self, threshold: datetime) -> None:
        if self.value <= threshold:
            self.value = threshold + timedelta(microseconds=10)


class _Tokenizer:
    def apply_chat_template(self, conversation: object, **kwargs: object) -> list[int]:
        del kwargs
        return list(range(len(canonical_json(conversation).encode("utf-8")) // 8 + 1))

    def encode(self, text: str, **kwargs: object) -> list[int]:
        del kwargs
        return list(range(len(text.encode("utf-8")) // 8))


def _tokenizer_manifest(revision: str) -> TokenizerManifest:
    return TokenizerManifest(
        schema_version="1.0.0",
        repository="Qwen/Qwen3-8B-AWQ",
        revision=revision,
        tokenizer_class="ArtificialTokenizer",
        tokenizer_revision=revision,
        tokenizer_file_sha256=(("tokenizer.json", "1" * 64),),
        eos_token_id=1,
        end_of_turn_token_ids=(2,),
        stop_token_ids=(1, 2),
        chat_template_sha256="2" * 64,
        nonthinking_probe_sha256="3" * 64,
        nonthinking_probe_token_count=1,
        local_files_only=True,
        trust_remote_code=False,
        enable_thinking=False,
    )


def _coherent_plan(
    original: CaseStudyExecutionPlan,
    construction: DevelopmentConstructionConfiguration,
    tokenizer_manifest: TokenizerManifest,
    *,
    configuration_hash: str,
    model_manifest_hash: str,
) -> CaseStudyExecutionPlan:
    runtime_values = original.model_runtime.model_dump(mode="python", exclude={"content_hash"})
    runtime_values.update(
        {
            "selected_launcher_configuration_hash": configuration_hash,
            "selected_snapshot_manifest_hash": model_manifest_hash,
            "selected_tokenizer_manifest_hash": tokenizer_manifest.manifest_sha256,
            "upper_ontology_hash": construction.upper_ontology.content_hash,
            "c1_capability_manifest_hash": CapabilityManifest.for_condition(
                ConditionName.C1_LLM_PRE
            ).content_hash,
            "c2_capability_manifest_hash": CapabilityManifest.for_condition(
                ConditionName.C2_LLM_QUERY
            ).content_hash,
            "output_schema_hash": canonical_sha256(
                base_condition_output_schema(ConditionName.C1_LLM_PRE)
            ),
        }
    )
    runtime = CaseModelRuntimeBinding.model_validate(runtime_values)
    slots = []
    for original_slot in original.gpu_call_slots:
        values = original_slot.model_dump(mode="python", exclude={"content_hash"})
        values["runtime_binding_hash"] = runtime.content_hash
        slots.append(CaseGpuCallSlot.model_validate(values))
    values = original.model_dump(mode="python", exclude={"content_hash"})
    values.update({"model_runtime": runtime, "gpu_call_slots": tuple(slots)})
    return CaseStudyExecutionPlan.model_validate(values)


def _draft(request: GuidedJSONRequest, *, invalid_parent: bool) -> dict[str, object]:
    user = json.loads(request.messages[-1].content)
    evidence_rows = user.get("evidence_snapshot")
    if evidence_rows is None:
        evidence_rows = user["evidence_packet"]["evidence"]
    evidence_id = evidence_rows[0][0]
    parent = "not-a-frozen-upper-type" if invalid_parent else "entity"
    return {
        "contextual_interpretation": "Artificial fixture interpretation.",
        "local_schema": {
            "schema_id": "case-schema",
            "contextual_types": [
                {
                    "type_id": "case-type",
                    "label": "Artificial type",
                    "definition": "A type grounded in the artificial fixture.",
                    "parent_upper_type": parent,
                    "abstraction": AbstractionLevel.EVENT_ROLE.value,
                    "evidence_ids": [evidence_id],
                }
            ],
            "predicates": [],
            "abstraction": AbstractionLevel.EVENT_ROLE.value,
        },
        "instance_graph": {
            "entities": [],
            "events": [],
            "proposition_contents": [],
            "assertions": [],
        },
        "decisions": [
            {
                "decision_id": "case-decision",
                "operator": "contextual_type",
                "evidence_ids": [evidence_id],
                "rationale": "Create the evidence-grounded artificial type.",
                "decided_at": "2026-09-04T12:00:00Z",
                "input_object_ids": [],
                "created_object_ids": ["case-type"],
                "removed_object_ids": [],
            }
        ],
        "omissions": [],
        "uncertainty_and_abstentions": [],
        "budget_accounting": {
            "nodes_used": 0,
            "assertions_used": 0,
            "display_nodes_used": 0,
            "display_assertions_used": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        },
    }


@dataclass
class _FakeOwnedService:
    ledger: Ledger
    clock: _Clock
    configuration_hash: str
    invalid_once_call_id: str | None = None
    state: ServiceState = ServiceState.STOPPED
    pid: int = 4242
    generate_count: int = 0
    start_count: int = 0
    shutdown_count: int = 0
    seen_base: set[str] = field(default_factory=set)
    start_watchdogs: list[float] = field(default_factory=list)
    lease_live: bool = False
    terminal_session: object | None = None
    recovered_identity: object | None = None
    detach_count: int = 0

    @property
    def configuration(self) -> object:
        return SimpleNamespace(configuration_hash=self.configuration_hash)

    @property
    def actual_allocated_service_seconds(self) -> float:
        return sum(item.allocated_seconds for item in self.ledger.gpu_events())

    def start(
        self,
        *,
        session_id: str,
        event_id: str,
        watchdog_seconds: float,
        remaining_required_seconds: float = 0,
    ) -> None:
        del session_id, remaining_required_seconds
        self.start_watchdogs.append(watchdog_seconds)
        self.start_count += 1
        started = self.clock()
        ended = self.clock()
        self.ledger.record_gpu_event(
            event_id=event_id,
            event_kind=GpuEventKind.GPU_SESSION_START,
            allocated_seconds=0.1,
            started_at=started,
            ended_at=ended,
            succeeded=True,
            details={"artificial_fixture": True},
        )
        self.state = ServiceState.READY

    def resume_from_checkpoint(self, path: Path) -> bool:
        if not path.is_file():
            return False
        self.state = ServiceState.READY
        return True

    def resume_live_service_lease(
        self,
        *,
        expected_session_id: str,
        expected_event_id: str,
        watchdog_seconds: float,
    ) -> bool:
        del expected_session_id, expected_event_id, watchdog_seconds
        if self.lease_live:
            self.state = ServiceState.READY
            return True
        return False

    def recover_stale_service_lease(self) -> object | None:
        return self.terminal_session

    @property
    def last_recovered_process_identity(self) -> object | None:
        return self.recovered_identity

    def write_resume_checkpoint(self, path: Path) -> None:
        path.write_text("artificial live checkpoint\n", encoding="utf-8")

    def detach_for_controller_restart(self, path: Path) -> None:
        self.write_resume_checkpoint(path)
        self.detach_count += 1
        self.state = ServiceState.STOPPED

    def generate(
        self,
        request: GuidedJSONRequest,
        *,
        event_id: str,
        watchdog_seconds: float,
        repair: bool = False,
        job_id: str | None = None,
        attempt_id: str | None = None,
        remaining_required_seconds: float = 0,
        accounting_details: dict[str, object] | None = None,
    ) -> GenerationResult:
        del watchdog_seconds, remaining_required_seconds
        assert self.state is ServiceState.READY
        self.generate_count += 1
        invalid = (
            not repair
            and request.request_id == self.invalid_once_call_id
            and request.request_id not in self.seen_base
        )
        self.seen_base.add(request.request_id)
        parsed = _draft(request, invalid_parent=invalid)
        raw = canonical_json(parsed).encode("utf-8")
        started = self.clock()
        ended = self.clock()
        self.ledger.record_gpu_event(
            event_id=event_id,
            event_kind=GpuEventKind.REPAIR if repair else GpuEventKind.INFERENCE,
            allocated_seconds=0.2,
            started_at=started,
            ended_at=ended,
            succeeded=True,
            job_id=job_id,
            attempt_id=attempt_id,
            details=accounting_details,
        )
        return GenerationResult(
            request_id=request.request_id,
            request_hash=request.request_hash,
            response_sha256=hashlib.sha256(raw).hexdigest(),
            parsed_object=parsed,
            raw_response=raw,
            prompt_tokens=request.rendered_input_token_count,
            completion_tokens=50,
            finish_reason="stop",
        )

    def shutdown(self) -> None:
        self.shutdown_count += 1
        self.state = ServiceState.STOPPED


def _setup(tmp_path: Path) -> tuple[Any, ...]:
    runtime_module = _fixture_module(
        "tests/unit/test_case_study_runtime.py", "_case_gpu_runtime_fixture"
    )
    execution_module = _fixture_module(
        "tests/unit/test_case_study_execution.py", "_case_gpu_execution_fixture"
    )
    fixture = runtime_module.case_fixture.__wrapped__(tmp_path)
    loaded = load_attested_restricted_case_study(
        restricted_root=fixture.restricted_root,
        index_path=fixture.index_path,
        manifest_path=fixture.manifest_path,
        preregistration_path=fixture.preregistration_path,
        attestation_path=fixture.attestation_path,
    )
    ledger = Ledger(fixture.restricted_root / "cumulative.sqlite3")
    ledger.record_gpu_event(
        event_id="verified-predecessor-gpu-seconds",
        event_kind=GpuEventKind.INFERENCE,
        allocated_seconds=212.281778,
        started_at=T0,
        ended_at=T0 + timedelta(seconds=212.281778),
        succeeded=True,
        details={"predecessor": True},
    )
    artifacts = ArtifactStore(
        BlobStore(fixture.restricted_root / "blobs", compression=Compression.GZIP),
        ledger,
    )
    construction = DevelopmentConstructionConfiguration.load(
        ROOT / "configs/study/development_construction.json"
    )
    tokenizer_manifest = _tokenizer_manifest(fixture.plan.model_runtime.model_revision)
    configuration_hash = "4" * 64
    model_manifest_hash = "5" * 64
    plan = _coherent_plan(
        fixture.plan,
        construction,
        tokenizer_manifest,
        configuration_hash=configuration_hash,
        model_manifest_hash=model_manifest_hash,
    )
    source = CaseArtifactReference(
        logical_content_hash="6" * 64,
        artifact_hash="6" * 64,
        object_kind="case_source_manifest",
    )
    admission = CaseExecutionAdmissionReceipt(
        receipt_id="artificial-case-execution-admission",
        execution_plan_hash=plan.content_hash,
        admission_attestation_hash=fixture.admission.content_hash,
        admission_evidence_bundle=source,
        source_manifest=source,
        source_revision="artificial-test-source",
        construction_configuration_file_sha256=construction.source_file_sha256,
        construction_configuration_hash=construction.content_hash,
        predecessor_ledger_sha256="7" * 64,
        prior_gpu_event_inventory_hash="8" * 64,
        allocated_gpu_seconds_before_case=212.281778,
        admitted_at=T0 + timedelta(minutes=5),
    )
    clock = _Clock(T0 + timedelta(minutes=6))
    repository = CaseExecutionRepository(
        plan=plan,
        artifacts=artifacts,
        restricted_root=fixture.restricted_root,
        resume_directory=fixture.restricted_root / "resume",
        state_pointer_path=fixture.restricted_root / "state/current.json",
        clock=clock,
    )
    service = _FakeOwnedService(
        ledger,
        clock,
        configuration_hash,
        invalid_once_call_id=plan.gpu_call_slots[4].call_id,
    )
    adapter = ProductionCaseStudyGpuAdapter(
        root=ROOT,
        plan=plan,
        admission=admission,
        construction=construction,
        tokenizer=_Tokenizer(),
        tokenizer_manifest=tokenizer_manifest,
        service=service,
        artifacts=artifacts,
        repository=repository,
        state_pointer_path=fixture.restricted_root / "state/gpu.json",
        model_manifest_hash=model_manifest_hash,
        process_start_ticks=lambda pid: 77 if pid == 4242 else 0,
        clock=clock,
    )
    return (
        fixture,
        loaded,
        plan,
        admission,
        ledger,
        artifacts,
        repository,
        service,
        adapter,
        execution_module,
        clock,
    )


def _admission_reference(
    admission: CaseExecutionAdmissionReceipt,
    artifacts: ArtifactStore,
    clock: _Clock,
) -> CaseArtifactReference:
    artifact = artifacts.put_bytes(
        admission.to_canonical_json().encode("utf-8"),
        media_type="application/json",
        release_class=ReleaseClass.RESTRICTED,
        created_at=clock(),
    )
    return CaseArtifactReference(
        logical_content_hash=admission.content_hash,
        artifact_hash=artifact.content_hash,
        object_kind="case_execution_admission",
    )


def test_production_adapter_runs_real_guided_4_8_1_and_one_repair(
    tmp_path: Path,
) -> None:
    (
        fixture,
        loaded,
        plan,
        admission,
        ledger,
        artifacts,
        repository,
        service,
        adapter,
        execution_module,
        clock,
    ) = _setup(tmp_path)
    ledger.register_study(
        study_id=plan.execution_id,
        protocol_hash=plan.content_hash,
        code_manifest_hash=admission.source_manifest.logical_content_hash,
        configuration_hash=admission.construction_configuration_hash,
        release_class=ReleaseClass.RESTRICTED,
        created_at=clock(),
    )
    admission_reference = _admission_reference(admission, artifacts, clock)
    classical = execution_module._FixtureClassicalAdapter(plan)
    controller = CaseStudyProductionController(
        root=ROOT,
        plan=plan,
        loaded=loaded,
        admission=admission,
        repository=repository,
        ledger=ledger,
        artifacts=artifacts,
        classical=classical,
        gpu=adapter,
        token_counter=lambda value: len(value.split()),
        service_checkpoint_path=fixture.restricted_root / "state/service.json",
        reveal_waiter=clock.advance_past,
        clock=clock,
    )

    result = controller.run(admission_reference)

    assert result.status.complete
    assert result.status.completed_output_count == 25
    terminal_state = repository.load_state()
    assert terminal_state is not None
    terminal_resume = repository.load_resume(terminal_state.resume_manifest)
    non_succeeded = tuple(
        (item.projection_job_id, item.terminal_outcome, item.failure_lineage_hash)
        for item in terminal_resume.output_receipts
        if item.terminal_outcome is not RunOutcome.SUCCEEDED
    )
    assert non_succeeded == ()
    assert service.generate_count == 14
    assert service.start_count == 1
    assert service.start_watchdogs == [CASE_SERVICE_START_WATCHDOG_SECONDS]
    assert service.shutdown_count == 1
    state = adapter._state()
    assert len(state.completed_receipts) == 13
    assert len(state.call_audits) == 13
    assert len(state.c1_preparations) == 4
    assert len(state.c1_projection_receipts) == 8
    assert state.model_service_start_count == 1
    assert state.model_load_count == 1
    assert state.model_service_shutdown_count == 1
    assert state.lifecycle_receipt is not None
    assert state.shutdown_receipt is not None
    shutdown = _parse_record(artifacts, state.shutdown_receipt, CaseGpuShutdownReceipt)
    assert isinstance(shutdown, CaseGpuShutdownReceipt)
    assert shutdown.cumulative_allocated_gpu_seconds == pytest.approx(
        sum(item.allocated_seconds for item in ledger.gpu_events())
    )
    assert state.consumed_repair_reservation_ids == (
        f"{plan.execution_id}:{plan.gpu_call_slots[4].call_id}:repair",
    )
    repair_events = tuple(
        item for item in ledger.gpu_events() if item.event_kind is GpuEventKind.REPAIR
    )
    assert len(repair_events) == 1
    assert json.loads(repair_events[0].details_json)["reserve_call_class"] == ("reserve_standard")
    assert (
        json.loads(repair_events[0].details_json)["reserve_reservation_id"]
        == (state.consumed_repair_reservation_ids[0])
    )
    repaired_audit = _parse_record(
        artifacts,
        state.call_audits[plan.gpu_call_slots[4].call_id],
        CaseGpuCallAuditReceipt,
    )
    assert isinstance(repaired_audit, CaseGpuCallAuditReceipt)
    assert repaired_audit.service_identity == state.service_identity
    assert repaired_audit.query_access_receipt is not None
    assert repaired_audit.prequery_barrier is not None
    assert repaired_audit.repair_semantic_request_hash is not None
    assert repaired_audit.repair_rendered_request_hash is not None
    assert (
        ledger.get_model_call(repaired_audit.base_model_call_id).request_hash
        == repaired_audit.rendered_request_hash
    )
    assert repaired_audit.repair_model_call_id is not None
    assert (
        ledger.get_model_call(repaired_audit.repair_model_call_id).request_hash
        == repaired_audit.repair_rendered_request_hash
    )
    c1_audit = _parse_record(
        artifacts,
        state.call_audits[plan.gpu_call_slots[0].call_id],
        CaseGpuCallAuditReceipt,
    )
    assert isinstance(c1_audit, CaseGpuCallAuditReceipt)
    assert c1_audit.query_access_receipt is None
    assert c1_audit.prequery_barrier is None

    source_sentence = (
        b"An invented amber courier crosses a painted bridge in this artificial fixture."
    )
    for path in (fixture.restricted_root / "blobs").rglob("*.jsonl.gz"):
        assert source_sentence not in gzip.decompress(path.read_bytes())

    replayed = controller.run(admission_reference)
    assert replayed.status.complete
    assert replayed.terminal_state_hash == result.terminal_state_hash
    assert replayed.content_hash == result.content_hash
    assert service.generate_count == 14
    assert service.shutdown_count == 1


def test_terminal_active_call_is_reconstructed_without_second_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        fixture,
        loaded,
        plan,
        admission,
        ledger,
        artifacts,
        repository,
        service,
        adapter,
        execution_module,
        clock,
    ) = _setup(tmp_path)
    admission_reference = _admission_reference(admission, artifacts, clock)
    ledger.register_study(
        study_id=plan.execution_id,
        protocol_hash=plan.content_hash,
        code_manifest_hash=admission.source_manifest.logical_content_hash,
        configuration_hash=admission.construction_configuration_hash,
        release_class=ReleaseClass.RESTRICTED,
        created_at=clock(),
    )
    controller = CaseStudyProductionController(
        root=ROOT,
        plan=plan,
        loaded=loaded,
        admission=admission,
        repository=repository,
        ledger=ledger,
        artifacts=artifacts,
        classical=execution_module._FixtureClassicalAdapter(plan),
        gpu=adapter,
        token_counter=lambda value: len(value.split()),
        service_checkpoint_path=fixture.restricted_root / "state/service.json",
        reveal_waiter=clock.advance_past,
        clock=clock,
    )
    controller_state = repository.initialize(admission_reference)
    controller_state = controller._materialize_bounded_packets(controller_state)
    first_preflight = adapter.preflight(plan=plan, bounded_packets=controller._bounded)
    clock()
    second_preflight = adapter.preflight(plan=plan, bounded_packets=controller._bounded)
    assert second_preflight == first_preflight
    adapter.start_once(
        execution_id=plan.execution_id,
        remaining_required_seconds=(
            sum(item.watchdog_seconds for item in plan.gpu_call_slots) + 240
        ),
    )
    adapter.checkpoint(fixture.restricted_root / "state/service.json")
    adopted_state = controller._start_or_resume_service(controller_state)
    assert adopted_state.model_service_start_count == 1
    assert adopted_state.model_load_count == 1
    assert service.start_count == 1
    call = plan.gpu_call_slots[0]
    window = plan.windows[0]
    protected_packet = controller._bounded[window.window_id]
    envelope = CaseC1RequestEnvelope(
        execution_plan_hash=plan.content_hash,
        call_slot_hash=call.content_hash,
        evidence_binding_hash=window.evidence_binding_hash,
        snapshot_hash=protected_packet.snapshot_assembly.snapshot.content_hash,
        packet_hash=protected_packet.packet.content_hash,
    )

    original_persist = ProductionCaseStudyGpuAdapter._persist_gpu_audit

    def interrupt_after_terminal_model_call(
        _adapter: ProductionCaseStudyGpuAdapter,
        **_values: object,
    ) -> None:
        raise OSError("artificial pointer interruption")

    monkeypatch.setattr(
        ProductionCaseStudyGpuAdapter,
        "_persist_gpu_audit",
        interrupt_after_terminal_model_call,
    )
    with pytest.raises(OSError, match="artificial pointer interruption"):
        adapter.preconstruct_c1(
            call=call,
            envelope=envelope,
            protected_packet=protected_packet,
        )
    assert service.generate_count == 1
    assert adapter._state().active_call_id == call.call_id

    monkeypatch.setattr(
        ProductionCaseStudyGpuAdapter,
        "_persist_gpu_audit",
        original_persist,
    )
    recovered = adapter.preconstruct_c1(
        call=call,
        envelope=envelope,
        protected_packet=protected_packet,
    )

    assert recovered.terminal_outcome is RunOutcome.SUCCEEDED
    assert service.generate_count == 1
    assert adapter._state().active_call_id is None
    adapter.shutdown()
    assert service.shutdown_count == 1


def test_activation_intent_recovers_live_lease_before_identity_or_checkpoint(
    tmp_path: Path,
) -> None:
    (
        fixture,
        _loaded,
        plan,
        _admission,
        _ledger,
        _artifacts,
        _repository,
        service,
        adapter,
        _execution_module,
        _clock,
    ) = _setup(tmp_path)
    # Reuse the real bounded materialization path to satisfy the lossless packing gate.
    admission_reference = _admission_reference(adapter.admission, adapter.artifacts, service.clock)
    controller = CaseStudyProductionController(
        root=ROOT,
        plan=plan,
        loaded=_loaded,
        admission=adapter.admission,
        repository=adapter.repository,
        ledger=adapter.artifacts.ledger,
        artifacts=adapter.artifacts,
        classical=_execution_module._FixtureClassicalAdapter(plan),
        gpu=adapter,
        token_counter=lambda value: len(value.split()),
        service_checkpoint_path=fixture.restricted_root / "state/service.json",
        reveal_waiter=service.clock.advance_past,
        clock=service.clock,
    )
    controller_state = adapter.repository.initialize(admission_reference)
    controller._materialize_bounded_packets(controller_state)
    adapter.preflight(plan=plan, bounded_packets=controller._bounded)
    reserve = float(sum(item.watchdog_seconds for item in plan.gpu_call_slots) + 240)
    state, intent = adapter._ensure_activation_intent(
        adapter._state(),
        remaining_required_seconds=reserve,
    )
    assert state.activation_intent is not None
    assert state.model_service_start_count == 0

    # The physical start happened, but power failed before identity/state/checkpoint.
    service.start(
        session_id=intent.session_id,
        event_id=intent.load_event_id,
        watchdog_seconds=CASE_SERVICE_START_WATCHDOG_SECONDS,
        remaining_required_seconds=reserve,
    )
    service.state = ServiceState.STOPPED
    service.lease_live = True

    assert adapter.resume_live(fixture.restricted_root / "state/missing-checkpoint.json")
    recovered = adapter._state()
    assert recovered.model_service_start_count == 1
    assert recovered.model_load_count == 1
    assert service.start_count == 1
    adapter.shutdown()


def test_terminal_journal_reconstructs_shutdown_after_physical_stop(
    tmp_path: Path,
) -> None:
    (
        fixture,
        loaded,
        plan,
        admission,
        _ledger,
        artifacts,
        repository,
        service,
        adapter,
        execution_module,
        clock,
    ) = _setup(tmp_path)
    admission_reference = _admission_reference(admission, artifacts, clock)
    controller = CaseStudyProductionController(
        root=ROOT,
        plan=plan,
        loaded=loaded,
        admission=admission,
        repository=repository,
        ledger=artifacts.ledger,
        artifacts=artifacts,
        classical=execution_module._FixtureClassicalAdapter(plan),
        gpu=adapter,
        token_counter=lambda value: len(value.split()),
        service_checkpoint_path=fixture.restricted_root / "state/service.json",
        reveal_waiter=clock.advance_past,
        clock=clock,
    )
    controller._materialize_bounded_packets(repository.initialize(admission_reference))
    adapter.preflight(plan=plan, bounded_packets=controller._bounded)
    adapter.start_once(
        execution_id=plan.execution_id,
        remaining_required_seconds=float(
            sum(item.watchdog_seconds for item in plan.gpu_call_slots) + 240
        ),
    )
    gpu_state, _intent = adapter._ensure_shutdown_intent(adapter._state())
    assert gpu_state.shutdown_intent is not None
    service.shutdown()
    service.terminal_session = SimpleNamespace(
        service_session_id=f"{plan.execution_id}-model-load",
        ended_at=clock().isoformat(),
    )

    assert not adapter.resume_live(fixture.restricted_root / "state/missing.json")
    terminal = adapter._state()
    assert terminal.model_service_shutdown_count == 1
    assert terminal.shutdown_receipt is not None
    adapter.shutdown()
    assert service.shutdown_count == 1


def test_controller_and_gpu_state_recover_from_immutable_pointer_history(
    tmp_path: Path,
) -> None:
    (
        _fixture,
        _loaded,
        plan,
        admission,
        _ledger,
        artifacts,
        repository,
        _service,
        adapter,
        _execution_module,
        clock,
    ) = _setup(tmp_path)
    admission_reference = _admission_reference(admission, artifacts, clock)
    controller_state = repository.initialize(admission_reference)
    repository.state_pointer_path.write_text("interrupted pointer", encoding="utf-8")
    assert repository.load_state() == controller_state

    initial_gpu_state = adapter._persist_initial_if_needed()
    adapter.state_pointer_path.write_text("interrupted pointer", encoding="utf-8")
    assert adapter._state() == initial_gpu_state
    assert plan.content_hash == controller_state.execution_plan_hash


def test_routine_controller_io_error_handoffs_healthy_service_for_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        fixture,
        loaded,
        plan,
        admission,
        _ledger,
        artifacts,
        repository,
        service,
        adapter,
        execution_module,
        clock,
    ) = _setup(tmp_path)
    admission_reference = _admission_reference(admission, artifacts, clock)
    controller = CaseStudyProductionController(
        root=ROOT,
        plan=plan,
        loaded=loaded,
        admission=admission,
        repository=repository,
        ledger=artifacts.ledger,
        artifacts=artifacts,
        classical=execution_module._FixtureClassicalAdapter(plan),
        gpu=adapter,
        token_counter=lambda value: len(value.split()),
        service_checkpoint_path=fixture.restricted_root / "state/service.json",
        reveal_waiter=clock.advance_past,
        clock=clock,
    )
    original_prepare = CaseStudyProductionController._prepare_all

    def interrupted_prepare(
        _controller: CaseStudyProductionController,
        _state: object,
    ) -> object:
        raise OSError("artificial durable-storage interruption")

    monkeypatch.setattr(
        CaseStudyProductionController,
        "_prepare_all",
        interrupted_prepare,
    )
    with pytest.raises(OSError, match="durable-storage interruption"):
        controller.run(admission_reference)
    assert controller.preserved_live_service_for_resume
    assert service.start_count == 1
    assert service.detach_count == 1
    assert service.shutdown_count == 0

    monkeypatch.setattr(
        CaseStudyProductionController,
        "_prepare_all",
        original_prepare,
    )
    result = controller.run(admission_reference)
    assert result.status.complete
    assert service.start_count == 1
    assert service.shutdown_count == 1


def test_factory_rehashes_source_tree_and_rejects_drift(tmp_path: Path) -> None:
    (
        _fixture,
        loaded,
        plan,
        admission,
        _ledger,
        artifacts,
        repository,
        service,
        _adapter,
        _execution_module,
        clock,
    ) = _setup(tmp_path)
    source_root = tmp_path / "source"
    (source_root / "src").mkdir(parents=True)
    source_file = source_root / "src/runtime.py"
    source_file.write_text("FROZEN = True\n", encoding="utf-8")
    revision = "artificial-case-source-v1"
    source_manifest = build_source_manifest(source_root, revision)
    source_payload = canonical_json(source_manifest.to_dict()).encode("utf-8")
    source_artifact = artifacts.put_bytes(
        source_payload,
        media_type="application/json",
        release_class=ReleaseClass.RESTRICTED,
        created_at=clock(),
    )
    source_reference = CaseArtifactReference(
        logical_content_hash=canonical_sha256(source_manifest.to_dict()),
        artifact_hash=source_artifact.content_hash,
        object_kind="case_source_manifest",
    )
    admission_values = admission.model_dump(mode="python", exclude={"content_hash"})
    admission_values.update({"source_manifest": source_reference, "source_revision": revision})
    bound_admission = CaseExecutionAdmissionReceipt.model_validate(admission_values)
    construction = DevelopmentConstructionConfiguration.load(
        ROOT / "configs/study/development_construction.json"
    )
    tokenizer_manifest = _tokenizer_manifest(plan.model_runtime.model_revision)

    built = build_production_case_study_gpu_adapter(
        root=source_root,
        plan=plan,
        loaded=loaded,
        admission=bound_admission,
        construction=construction,
        tokenizer=_Tokenizer(),
        tokenizer_manifest=tokenizer_manifest,
        service=service,
        artifacts=artifacts,
        repository=repository,
        state_pointer_path=tmp_path / "restricted/state/factory-gpu.json",
        model_manifest_hash="5" * 64,
        process_start_ticks=lambda _pid: 77,
        clock=clock,
    )
    assert isinstance(built, ProductionCaseStudyGpuAdapter)

    source_file.write_text("FROZEN = False\n", encoding="utf-8")
    with pytest.raises(CaseStudyAdmissionError, match="source association changed"):
        build_production_case_study_gpu_adapter(
            root=source_root,
            plan=plan,
            loaded=loaded,
            admission=bound_admission,
            construction=construction,
            tokenizer=_Tokenizer(),
            tokenizer_manifest=tokenizer_manifest,
            service=service,
            artifacts=artifacts,
            repository=repository,
            state_pointer_path=tmp_path / "restricted/state/factory-gpu-2.json",
            model_manifest_hash="5" * 64,
            process_start_ticks=lambda _pid: 77,
            clock=clock,
        )
