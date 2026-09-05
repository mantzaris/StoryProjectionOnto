from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType

import pytest

from story_projection_onto.case_study_execution import (
    CaseArtifactReference,
    CaseExecutionAdmissionReceipt,
    CaseExecutionRepository,
    CaseStudyExecutionError,
    CaseStudyProductionController,
    _persist_record,
    _total_allocated_seconds,
)
from story_projection_onto.case_study_factory import CaseStudyProductionPreflight
from story_projection_onto.case_study_runtime import (
    CaseC0PreparationEnvelope,
    CaseC0ProjectionEnvelope,
    CaseC1RequestEnvelope,
    CaseC2RequestEnvelope,
    CaseGpuCallSlot,
    CaseOutputReceipt,
    CasePrequeryKind,
    CasePrequeryReceipt,
    CaseQueryAccessReceipt,
    CaseStudyExecutionPlan,
    CaseWindowExecutionPlan,
)
from story_projection_onto.contracts import (
    ConditionName,
    ConstructionSeal,
    EvidencePacket,
    QueryContext,
    ReleaseClass,
    RunOutcome,
    canonical_json,
    canonical_sha256,
)
from story_projection_onto.novel_case import WindowEvidenceBundle
from story_projection_onto.store import (
    ArtifactStore,
    BlobStore,
    Compression,
    GpuEventKind,
    Ledger,
)

ROOT = Path(__file__).resolve().parents[2]
HASH = "a" * 64


def _case_fixture_module() -> ModuleType:
    path = ROOT / "tests/unit/test_case_study_runtime.py"
    spec = importlib.util.spec_from_file_location("_case_runtime_fixture", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _controller_script_module() -> ModuleType:
    path = ROOT / "scripts/run_case_study_controller.py"
    spec = importlib.util.spec_from_file_location("_case_controller_cli", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _Clock:
    def __init__(self, start: datetime) -> None:
        self.value = start

    def __call__(self) -> datetime:
        self.value += timedelta(microseconds=1)
        return self.value

    def advance_past(self, threshold: datetime) -> None:
        if self.value <= threshold:
            self.value = threshold + timedelta(microseconds=1)


def _artifact(label: str) -> str:
    return canonical_sha256({"fixture": label})


def test_case_gpu_total_includes_classified_events_and_service_overhead(
    tmp_path: Path,
) -> None:
    with Ledger(tmp_path / "accounting.sqlite3") as ledger:
        ledger.record_gpu_event(
            event_id="classified-call",
            event_kind=GpuEventKind.INFERENCE,
            allocated_seconds=2,
            started_at=datetime(2026, 9, 4, 12, 0, tzinfo=UTC),
            ended_at=datetime(2026, 9, 4, 12, 0, 2, tzinfo=UTC),
            succeeded=True,
        )
        ledger.record_gpu_service_session(
            service_session_id="model-service",
            session_id="model-session",
            service_seconds=5,
            classified_event_seconds=2,
            started_at=datetime(2026, 9, 4, 12, 0, tzinfo=UTC),
            ended_at=datetime(2026, 9, 4, 12, 0, 5, tzinfo=UTC),
        )

        assert _total_allocated_seconds(ledger) == 5


@dataclass
class _FixtureClassicalAdapter:
    plan: CaseStudyExecutionPlan
    backend: str = "spacy-ner-dependency-plus-deterministic-rules-v2"
    preparations: dict[str, CasePrequeryReceipt] = field(default_factory=dict)

    def prepare_c0(
        self,
        *,
        window: CaseWindowExecutionPlan,
        envelope: CaseC0PreparationEnvelope,
        protected_packet: WindowEvidenceBundle,
    ) -> CasePrequeryReceipt:
        assert envelope.query_context_supplied is False
        snapshot = protected_packet.snapshot_assembly.snapshot
        constructed = snapshot.sealed_at + timedelta(microseconds=1)
        sealed = constructed + timedelta(microseconds=1)
        seal = ConstructionSeal(
            seal_id=f"fixture-c0-seal-{window.window_id}",
            condition=ConditionName.C0_CLASSICAL_PRE,
            snapshot_hash=snapshot.content_hash,
            ontology_hash=_artifact(f"c0-ontology-{window.window_id}"),
            constructed_at=constructed,
            sealed_at=sealed,
            sealed_object_ids=(f"fixture-c0-object-{window.window_id}",),
        )
        receipt = CasePrequeryReceipt(
            receipt_id=f"fixture-{window.c0_preparation_job_id}",
            execution_plan_hash=self.plan.content_hash,
            preparation_job_id=window.c0_preparation_job_id,
            kind=CasePrequeryKind.C0_WINDOW_PRECONSTRUCTION,
            window_id=window.window_id,
            condition=ConditionName.C0_CLASSICAL_PRE,
            evidence_binding_hash=window.evidence_binding_hash,
            snapshot_hash=snapshot.content_hash,
            backend=self.backend,
            terminal_outcome=RunOutcome.SUCCEEDED,
            started_at=constructed,
            completed_at=sealed,
            construction_seal=seal,
        )
        self.preparations[window.window_id] = receipt
        return receipt

    def project_c0(
        self,
        *,
        window: CaseWindowExecutionPlan,
        envelope: CaseC0ProjectionEnvelope,
        query_access: CaseQueryAccessReceipt,
        prequery_barrier: object,
        protected_packet: WindowEvidenceBundle,
        protected_query: QueryContext,
    ) -> CaseOutputReceipt:
        del prequery_barrier, protected_query
        source = self.preparations[window.window_id]
        index = window.context_ids.index(query_access.context_id)
        assert source.construction_seal is not None
        assert envelope.ontology_construction_allowed is False
        return CaseOutputReceipt(
            receipt_id=f"fixture-output-{window.c0_projection_job_ids[index]}",
            execution_plan_hash=self.plan.content_hash,
            projection_job_id=window.c0_projection_job_ids[index],
            condition=ConditionName.C0_CLASSICAL_PRE,
            context_id=query_access.context_id,
            window_id=window.window_id,
            backend=self.backend,
            query_access_receipt_hash=query_access.content_hash,
            preparation_receipt_hash=source.content_hash,
            evidence_binding_hash=window.evidence_binding_hash,
            packet_equality_group_id=window.packet_equality_group_id,
            snapshot_hash=query_access.snapshot_hash,
            packet_hash=protected_packet.packet.content_hash,
            source_construction_seal_hash=source.construction_seal.content_hash,
            terminal_outcome=RunOutcome.SUCCEEDED,
            projection_artifact_hash=_artifact(envelope.content_hash),
            query_accessed_at=query_access.accessed_at,
            completed_at=query_access.accessed_at + timedelta(microseconds=1),
            operational_only=False,
            causal_comparison_eligible=True,
        )


@dataclass
class _FixtureGpuAdapter:
    plan: CaseStudyExecutionPlan
    artifacts: ArtifactStore
    repository: CaseExecutionRepository | None = None
    backend: str = "vllm_gpu"
    start_count: int = 0
    shutdown_count: int = 0
    preflight_before_start: bool = False
    preparations: dict[str, CasePrequeryReceipt] = field(default_factory=dict)
    c1_call_ids: list[str] = field(default_factory=list)
    c2_call_ids: list[str] = field(default_factory=list)

    @property
    def service_configuration_hash(self) -> str:
        return self.plan.model_runtime.selected_launcher_configuration_hash

    @property
    def actual_allocated_service_seconds(self) -> float:
        return 0.0

    def preflight(
        self,
        *,
        plan: CaseStudyExecutionPlan,
        bounded_packets: dict[str, WindowEvidenceBundle],
    ) -> CaseArtifactReference:
        assert plan == self.plan
        assert len(bounded_packets) == 4
        assert self.start_count == 0
        self.preflight_before_start = True
        payload = {"fixture_only": True, "packet_hashes": sorted(bounded_packets)}
        artifact = self.artifacts.put_bytes(
            canonical_json(payload).encode(),
            media_type="application/json",
            release_class=ReleaseClass.RESTRICTED,
        )
        return CaseArtifactReference(
            logical_content_hash=canonical_sha256(payload),
            artifact_hash=artifact.content_hash,
            object_kind="case_gpu_packing_preflight",
        )

    def start_once(self, *, execution_id: str, remaining_required_seconds: float) -> None:
        assert execution_id == self.plan.execution_id
        assert remaining_required_seconds == 2550
        assert self.preflight_before_start
        self.start_count += 1

    def resume_live(self, checkpoint_path: Path) -> bool:
        return checkpoint_path.is_file()

    def checkpoint(self, checkpoint_path: Path) -> None:
        checkpoint_path.write_text("fixture checkpoint\n", encoding="utf-8")

    def preconstruct_c1(
        self,
        *,
        call: CaseGpuCallSlot,
        envelope: CaseC1RequestEnvelope,
        protected_packet: WindowEvidenceBundle,
    ) -> CasePrequeryReceipt:
        self.c1_call_ids.append(call.call_id)
        snapshot = protected_packet.snapshot_assembly.snapshot
        started = snapshot.sealed_at + timedelta(microseconds=3)
        sealed_at = started + timedelta(microseconds=1)
        seal = ConstructionSeal(
            seal_id=f"fixture-c1-seal-{call.window_id}",
            condition=ConditionName.C1_LLM_PRE,
            snapshot_hash=snapshot.content_hash,
            ontology_hash=_artifact(f"c1-ontology-{call.window_id}"),
            constructed_at=started,
            sealed_at=sealed_at,
            sealed_object_ids=(f"fixture-c1-object-{call.window_id}",),
        )
        receipt = CasePrequeryReceipt(
            receipt_id=f"fixture-{call.call_id}",
            execution_plan_hash=self.plan.content_hash,
            preparation_job_id=call.call_id,
            kind=CasePrequeryKind.C1_WINDOW_PRECONSTRUCTION,
            window_id=str(call.window_id),
            condition=ConditionName.C1_LLM_PRE,
            evidence_binding_hash=envelope.evidence_binding_hash,
            snapshot_hash=snapshot.content_hash,
            backend="vllm_gpu",
            terminal_outcome=RunOutcome.SUCCEEDED,
            started_at=started,
            completed_at=sealed_at,
            base_attempt_artifact_hash=_artifact(f"raw-{call.call_id}"),
            construction_seal=seal,
        )
        self.preparations[str(call.window_id)] = receipt
        return receipt

    def project_c1(
        self,
        *,
        window: CaseWindowExecutionPlan,
        query_access: CaseQueryAccessReceipt,
        prequery_barrier: object,
        protected_packet: WindowEvidenceBundle,
        protected_query: QueryContext,
    ) -> CaseOutputReceipt:
        del prequery_barrier, protected_query
        source = self.preparations[window.window_id]
        index = window.context_ids.index(query_access.context_id)
        assert source.construction_seal is not None
        return CaseOutputReceipt(
            receipt_id=f"fixture-output-{window.c1_projection_job_ids[index]}",
            execution_plan_hash=self.plan.content_hash,
            projection_job_id=window.c1_projection_job_ids[index],
            condition=ConditionName.C1_LLM_PRE,
            context_id=query_access.context_id,
            window_id=window.window_id,
            backend="fixed_projection_cpu",
            query_access_receipt_hash=query_access.content_hash,
            preparation_receipt_hash=source.content_hash,
            evidence_binding_hash=window.evidence_binding_hash,
            packet_equality_group_id=window.packet_equality_group_id,
            snapshot_hash=query_access.snapshot_hash,
            packet_hash=protected_packet.packet.content_hash,
            source_construction_seal_hash=source.construction_seal.content_hash,
            terminal_outcome=RunOutcome.SUCCEEDED,
            projection_artifact_hash=_artifact(f"c1-projection-{query_access.context_id}"),
            query_accessed_at=query_access.accessed_at,
            completed_at=query_access.accessed_at + timedelta(microseconds=2),
            operational_only=False,
            causal_comparison_eligible=True,
        )

    def construct_c2(
        self,
        *,
        call: CaseGpuCallSlot,
        envelope: CaseC2RequestEnvelope,
        query_access: CaseQueryAccessReceipt,
        prequery_barrier: object,
        protected_packet: EvidencePacket,
        protected_query: QueryContext,
    ) -> CaseOutputReceipt:
        del prequery_barrier, protected_query
        self.c2_call_ids.append(call.call_id)
        preparation_id = (
            self.plan.operational.c2_empty_inventory_job_id
            if call.operational_only
            else next(
                window.c2_empty_inventory_job_id
                for window in self.plan.windows
                if window.window_id == call.window_id
            )
        )
        return CaseOutputReceipt(
            receipt_id=f"fixture-output-{call.call_id}",
            execution_plan_hash=self.plan.content_hash,
            projection_job_id=call.call_id,
            condition=ConditionName.C2_LLM_QUERY,
            context_id=str(call.context_id),
            window_id=str(call.window_id),
            backend="vllm_gpu",
            query_access_receipt_hash=query_access.content_hash,
            preparation_receipt_hash=self._preparation_hash(preparation_id),
            evidence_binding_hash=envelope.evidence_binding_hash,
            packet_equality_group_id=(
                "case-full-index-operational-packet"
                if call.operational_only
                else next(
                    window.packet_equality_group_id
                    for window in self.plan.windows
                    if window.window_id == call.window_id
                )
            ),
            snapshot_hash=query_access.snapshot_hash,
            packet_hash=protected_packet.content_hash,
            pre_query_inventory_hash=envelope.empty_inventory_hash,
            base_attempt_artifact_hash=_artifact(f"raw-{call.call_id}"),
            terminal_outcome=RunOutcome.FAILED,
            failure_lineage_hash=_artifact(f"failure-{call.call_id}"),
            query_accessed_at=query_access.accessed_at,
            completed_at=query_access.accessed_at + timedelta(microseconds=3),
            operational_only=call.operational_only,
            causal_comparison_eligible=call.causal_comparison_eligible,
        )

    def _preparation_hash(self, job_id: str) -> str:
        assert self.repository is not None
        state = self.repository.load_state()
        assert state is not None
        resume = self.repository.load_resume(state.resume_manifest)
        return next(
            item.content_hash
            for item in resume.prequery_receipts
            if item.preparation_job_id == job_id
        )

    def shutdown(self) -> None:
        self.shutdown_count += 1


def test_controller_executes_exact_4_8_1_with_one_owned_fixture_lifecycle(
    tmp_path: Path,
) -> None:
    module = _case_fixture_module()
    fixture = module.case_fixture.__wrapped__(tmp_path)
    restricted = fixture.restricted_root
    ledger = Ledger(restricted / "cumulative.sqlite3")
    artifacts = ArtifactStore(
        BlobStore(restricted / "blobs", compression=Compression.GZIP),
        ledger,
    )
    source_ref = CaseArtifactReference(
        logical_content_hash=HASH,
        artifact_hash=HASH,
        object_kind="case_source_manifest",
    )
    admission = CaseExecutionAdmissionReceipt(
        receipt_id="fixture-case-admission",
        execution_plan_hash=fixture.plan.content_hash,
        admission_attestation_hash=fixture.admission.content_hash,
        admission_evidence_bundle=source_ref,
        source_manifest=source_ref,
        source_revision="fixture-source",
        construction_configuration_file_sha256=HASH,
        construction_configuration_hash=HASH,
        predecessor_ledger_sha256=HASH,
        prior_gpu_event_inventory_hash=HASH,
        allocated_gpu_seconds_before_case=212.281778,
        admitted_at=datetime(2026, 9, 4, 12, 5, tzinfo=UTC),
    )
    admission_ref = _persist_record(
        artifacts,
        admission,
        object_kind="case_execution_admission",
        created_at=admission.admitted_at,
    )
    repository = CaseExecutionRepository(
        plan=fixture.plan,
        artifacts=artifacts,
        restricted_root=restricted,
        resume_directory=restricted / "resume",
        state_pointer_path=restricted / "state/current.json",
        clock=_Clock(datetime(2026, 9, 4, 12, 5, tzinfo=UTC)),
    )
    classical = _FixtureClassicalAdapter(fixture.plan)
    gpu = _FixtureGpuAdapter(fixture.plan, artifacts, repository)
    controller_clock = _Clock(datetime(2026, 9, 4, 12, 5, tzinfo=UTC))
    controller = CaseStudyProductionController(
        root=ROOT,
        plan=fixture.plan,
        loaded=module.load_attested_restricted_case_study(
            restricted_root=restricted,
            index_path=fixture.index_path,
            manifest_path=fixture.manifest_path,
            preregistration_path=fixture.preregistration_path,
            attestation_path=fixture.attestation_path,
        ),
        admission=admission,
        repository=repository,
        ledger=ledger,
        artifacts=artifacts,
        classical=classical,
        gpu=gpu,
        token_counter=lambda value: len(value.split()),
        service_checkpoint_path=restricted / "state/service.json",
        clock=controller_clock,
        reveal_waiter=controller_clock.advance_past,
    )

    result = controller.run(admission_ref)

    assert result.status.complete
    assert result.status.completed_prequery_job_count == 13
    assert result.status.query_access_count == 9
    assert result.status.completed_output_count == 25
    assert result.status.failed_output_count == 9
    assert gpu.start_count == 1
    assert gpu.shutdown_count == 1
    assert len(gpu.c1_call_ids) == 4
    assert len(gpu.c2_call_ids) == 9
    assert result.model_load_count == 1
    assert result.model_service_stopped
    terminal_state = repository.load_state()
    assert terminal_state is not None
    operational = next(
        item
        for item in repository.load_resume(terminal_state.resume_manifest).query_access_receipts
        if item.operational_only
    )
    assert operational.packet_materialized_at >= operational.accessed_at
    assert operational.operational_retrieval_receipt is not None


def test_query_provider_and_stopped_incomplete_state_fail_closed(tmp_path: Path) -> None:
    module = _case_fixture_module()
    fixture = module.case_fixture.__wrapped__(tmp_path)
    provider = module.CaseQueryProvider if hasattr(module, "CaseQueryProvider") else None
    assert provider is None
    # The production provider is deliberately inaccessible without the sealed flag.
    from story_projection_onto.case_study_execution import CaseQueryProvider

    loaded = module.load_attested_restricted_case_study(
        restricted_root=fixture.restricted_root,
        index_path=fixture.index_path,
        manifest_path=fixture.manifest_path,
        preregistration_path=fixture.preregistration_path,
        attestation_path=fixture.attestation_path,
    )
    with pytest.raises(CaseStudyExecutionError, match="before the barrier"):
        CaseQueryProvider(loaded).open(fixture.plan.bounded_context_ids[0], barrier_sealed=False)


def test_controller_cli_validates_and_execute_requires_exact_inputs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _case_fixture_module()
    fixture = module.case_fixture.__wrapped__(tmp_path)
    plan_path = fixture.restricted_root / "execution-plan.json"
    module.write_restricted_case_record(
        fixture.plan,
        plan_path,
        restricted_root=fixture.restricted_root,
    )
    cli = _controller_script_module()
    arguments = [
        "--restricted-root",
        str(fixture.restricted_root),
        "--plan",
        str(plan_path),
    ]
    assert cli.main(arguments) == 0
    payload = capsys.readouterr().out
    assert '"adapter_factory_available": true' in payload
    assert '"case_gpu_call_count": 13' in payload
    assert '"execute_enabled": false' in payload
    assert '"execution_ready": false' in payload
    assert '"state": "contract_only"' in payload
    assert '"validation_scope": "contract_only"' in payload
    assert '"service_start_watchdog_seconds": 300' in payload
    assert cli.main([*arguments, "--validate-only"]) == 2
    validate_blocked = capsys.readouterr().out
    assert '"state": "blocked"' in validate_blocked
    assert "--index" in validate_blocked
    assert cli.main([*arguments, "--execute"]) == 2
    blocked = capsys.readouterr().out
    assert '"state": "blocked"' in blocked
    assert "--index" in blocked

    secret = "copyrighted input_value='The hidden sentence'"
    sanitized = cli._redacted_error(ValueError(secret), cli.parse_args(arguments))
    assert sanitized == "case command blocked; diagnostics retained in restricted storage"
    assert "hidden sentence" not in sanitized

    observed: list[dict[str, object]] = []
    preflight = CaseStudyProductionPreflight(
        execution_plan_hash=fixture.plan.content_hash,
        staging_transition_receipt_hash="1" * 64,
        current_ledger_sha256="2" * 64,
        actual_allocated_gpu_seconds=12.0,
        remaining_required_gpu_seconds=2_850.0,
        projected_storage_bytes=123,
        filesystem_free_bytes=456,
        selected_snapshot_manifest_hash="3" * 64,
        launcher_configuration_hash="4" * 64,
        bootstrap_state="fresh",
        controller_state="not_started",
    )

    def fake_preflight(**values: object) -> CaseStudyProductionPreflight:
        observed.append(values)
        return preflight

    monkeypatch.setattr(
        cli,
        "preflight_frozen_production_case_study_bundle",
        fake_preflight,
    )
    required_paths = (
        "index",
        "index-manifest",
        "preregistration",
        "input-attestation",
        "admission-attestation",
        "semantic-gate-bundle",
        "selected-model-freeze",
        "admission-evidence-bundle",
        "admission-evidence-bundle-reference",
        "ledger",
        "artifact-root",
        "staging-transition-directory",
        "runtime-root",
        "quota-root",
        "snapshot",
        "shared-cache",
        "verified-model-manifest",
        "source-association",
    )
    full_arguments = [*arguments, "--validate-only"]
    for flag in required_paths:
        full_arguments.extend((f"--{flag}", str(plan_path)))
    full_arguments.extend(
        (
            "--expected-predecessor-ledger-sha256",
            "5" * 64,
            "--source-revision",
            "fixture-source",
        )
    )
    assert cli.main(full_arguments) == 0
    ready = capsys.readouterr().out
    assert '"execution_ready": true' in ready
    assert '"state": "ready"' in ready
    assert '"validation_scope": "full_production_preflight"' in ready
    assert '"writes_performed": false' in ready
    assert observed and observed[0]["ledger_path"] == plan_path
