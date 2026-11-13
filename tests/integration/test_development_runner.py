from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from story_projection_onto.contracts import (
    ConditionName,
    OutputBudgets,
    PrequeryPreparationBinding,
    QueryAccessEvent,
    RunOutcome,
)
from story_projection_onto.development_runtime import (
    CallExecutionEnvelope,
    DevelopmentCallKind,
    DevelopmentCallManifest,
    DevelopmentCallSpec,
    DevelopmentPhase,
    DevelopmentPrequeryInputs,
    DevelopmentRunner,
    DevelopmentScientificAssessment,
    FixedSchemaDerivationPlan,
    FixedSchemaDerivationReceipt,
    ForecastControl,
    LiveServiceIdentity,
    ServiceCallResult,
    StageReference,
    UnitPrequeryBinding,
    load_development_call_manifest,
)

ROOT = Path(__file__).resolve().parents[2]
START = datetime(2026, 9, 3, 12, 0, tzinfo=UTC)


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


class TickingClock:
    def __init__(self, start: datetime = START) -> None:
        self.value = start

    def __call__(self) -> datetime:
        self.value += timedelta(milliseconds=1)
        return self.value


def _preexisting_conditions(unit_id: str) -> tuple[ConditionName, ...]:
    return {
        "dev-unit-01": (
            ConditionName.C0_CLASSICAL_PRE,
            ConditionName.C2_LLM_QUERY,
            ConditionName.A_NO_CONTEXT,
        ),
        "dev-unit-02": (
            ConditionName.C0_CLASSICAL_PRE,
            ConditionName.C2_LLM_QUERY,
            ConditionName.A_NO_TEMPORAL_EPISTEMIC,
        ),
        "dev-unit-03": (
            ConditionName.C0_CLASSICAL_PRE,
            ConditionName.C2_LLM_QUERY,
            ConditionName.A_NO_RARE_GUARD,
        ),
        "dev-unit-04": (
            ConditionName.C0_CLASSICAL_PRE,
            ConditionName.C2_LLM_QUERY,
        ),
    }[unit_id]


def development_inputs(manifest: DevelopmentCallManifest) -> DevelopmentPrequeryInputs:
    bindings = []
    for unit_id in ("dev-unit-01", "dev-unit-02", "dev-unit-03", "dev-unit-04"):
        prequery = next(call.prequery_stage for call in manifest.calls if call.unit_id == unit_id)
        neutral = next(item for item in manifest.neutral_evidence_stages if item.unit_id == unit_id)
        evidence_path = ROOT / prequery.relative_path / "evidence.json"
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        snapshot = evidence["snapshot"]
        preparations = tuple(
            PrequeryPreparationBinding(
                unit_id=snapshot["world_or_window_id"],
                condition=condition,
                seed_block=(
                    None if condition is ConditionName.C0_CLASSICAL_PRE else manifest.seed_block
                ),
                snapshot_hash=snapshot["content_hash"],
                preparation_hash=digest(f"{unit_id}:{condition.value}:preparation"),
                lineage_artifact_hash=digest(f"{unit_id}:{condition.value}:lineage"),
                completed_at=START - timedelta(minutes=2),
            )
            for condition in _preexisting_conditions(unit_id)
        )
        bindings.append(
            UnitPrequeryBinding(
                unit_id=unit_id,
                runtime_unit_id=snapshot["world_or_window_id"],
                snapshot_hash=snapshot["content_hash"],
                staged_model_visible_evidence_hash=prequery.evidence_artifact_hash,
                neutral_full_evidence_artifact_hash=(neutral.neutral_evidence_artifact_hash),
                evidence_equivalence_certificate_hash=(neutral.equivalence_certificate_hash),
                preexisting_preparation_bindings=preparations,
                completed_at=START - timedelta(minutes=1),
            )
        )
    config_hashes = tuple(
        None
        if call.kind is DevelopmentCallKind.FIXED_SELECTION
        else digest(f"run-condition-config:{call.call_id}")
        for call in manifest.calls
    )
    plans = tuple(
        FixedSchemaDerivationPlan(
            ordinal=call.ordinal,
            call_id=call.call_id,
            unit_id=call.unit_id,
            source_c1_call_id=call.source_c1_call_id,
            budgets=OutputBudgets(
                node_budget=10,
                assertion_budget=20,
                display_node_budget=10,
                display_assertion_budget=20,
            ),
            model_stack_hash=digest("model-stack"),
            decoding_family_hash=digest("decoding-family"),
            seed_manifest_hash=digest("seed-manifest"),
            prompt_hash=digest("fixed-prompt"),
            base_output_schema_hash=digest("fixed-base-schema"),
            scored_schema_hash=digest("scored-schema"),
            capability_manifest_hash=digest("fixed-capabilities"),
            validator_hash=digest("development-validator"),
            upper_ontology_hash=digest("upper-ontology"),
            prequery_evidence_artifact_hash=call.prequery_stage.evidence_artifact_hash,
            query_stage_manifest_hash=call.query_stage.staging_manifest_hash,
        )
        for call in manifest.calls
        if call.kind is DevelopmentCallKind.FIXED_SELECTION
    )
    return DevelopmentPrequeryInputs(
        source_tree_hash=digest("source-tree"),
        selected_model_freeze_hash=digest("selected-model-freeze"),
        upper_ontology_hash=digest("upper-ontology"),
        prompt_family_hash=digest("development-prompt-family"),
        output_schema_hash=digest("development-output-schema"),
        validator_hash=digest("development-validator"),
        run_condition_config_hashes=config_hashes,
        fixed_schema_derivation_plans=plans,
        unit_bindings=tuple(bindings),
    )


def passing_assessment(
    _manifest: DevelopmentCallManifest,
    _rows: tuple[object, ...],
) -> DevelopmentScientificAssessment:
    return DevelopmentScientificAssessment(
        c0_explicit_family_coverage=1.0,
        c0_direct_assertion_precision=0.90,
        c0_direct_assertion_recall=0.80,
        c0_valid_evidence_reference_rate=1.0,
        c1_schema_valid=True,
        c1_all_construction_operators_exercised=True,
        c1_valid_evidence_id_rate=1.0,
        c1_grounding_precision=0.97,
        c1_union_gold_recall_after_seal=0.80,
        c2_construction_operator_present=True,
        programmed_horizon_leak_count=0,
        evidence_packets_equal=True,
        c1_query_blindness_verified=True,
        c2_empty_prequery_inventories_verified=True,
        fixed_complete_graph_packing_verified=True,
        fixed_constructive_operations_rejected=True,
        ablation_one_switch_verified=True,
        gold_firewall_verified=True,
    )


class ControllerCrash(BaseException):
    pass


class MockBorrowedService:
    def __init__(
        self,
        inputs: DevelopmentPrequeryInputs,
        clock: TickingClock,
        *,
        allocated_at_start: float = 100.0,
        invalid_calls: frozenset[str] = frozenset(),
        crash_after_execution: int | None = None,
        drop_result_on_crash: bool = False,
    ) -> None:
        self.inputs = inputs
        self.clock = clock
        self.allocated = allocated_at_start
        self.invalid_calls = invalid_calls
        self.crash_after_execution = crash_after_execution
        self.drop_result_on_crash = drop_result_on_crash
        self.crashed = False
        self.executions: list[str] = []
        self.recoveries: list[str] = []
        self.opened_stages: list[str] = []
        self.envelopes: dict[str, CallExecutionEnvelope] = {}
        self.results: dict[str, ServiceCallResult] = {}
        self.query_events: dict[str, QueryAccessEvent] = {}
        self._identity = LiveServiceIdentity(
            owner_run_id="fallback-acceptance-and-development",
            service_pid=4242,
            service_start_ticks=9001,
            gpu_session_event_id="gpu-session-03",
            launcher_configuration_hash=digest("launcher-configuration"),
            model_snapshot_hash=digest("model-snapshot"),
            selected_model_freeze_hash=inputs.selected_model_freeze_hash,
            source_execution_hash=inputs.source_tree_hash,
        )

    def identity(self) -> LiveServiceIdentity:
        return self._identity

    def allocated_gpu_seconds(self) -> float:
        return self.allocated

    def open_query(
        self,
        stage: StageReference,
        barrier: object,
    ) -> QueryAccessEvent:
        assert hasattr(barrier, "sealed_at")
        existing = self.query_events.get(stage.staging_manifest_hash)
        if existing is not None:
            return existing
        self.opened_stages.append(stage.staging_manifest_hash)
        unit = next(
            item
            for item in self.inputs.unit_bindings
            if item.staged_model_visible_evidence_hash == stage.evidence_artifact_hash
        )
        accessed_at = self.clock()
        event = QueryAccessEvent(
            access_event_id=f"access-{stage.stage_id}",
            execution_id="development-execution-01",
            query_context_hash=digest(f"context:{stage.stage_id}"),
            model_visible_query_hash=digest(f"visible-context:{stage.stage_id}"),
            snapshot_hash=unit.snapshot_hash,
            stage_manifest_hash=stage.staging_manifest_hash,
            query_artifact_hash=stage.query_artifact_hash,
            prequery_barrier_hash=barrier.content_hash,
            registered_revealed_at=accessed_at,
            accessed_at=accessed_at,
        )
        self.query_events[stage.staging_manifest_hash] = event
        return event

    def _successful_result(
        self,
        call: DevelopmentCallSpec,
        envelope: CallExecutionEnvelope,
    ) -> ServiceCallResult:
        preparation_bindings: tuple[PrequeryPreparationBinding, ...] = ()
        preparation_hash = None
        attempt_hash = None
        seal_hash = None
        if call.kind is DevelopmentCallKind.C1_PRECONSTRUCTION:
            unit = self.inputs.binding_for(call.unit_id)
            c1_completed_at = self.clock()
            preparation_hash = digest(f"{call.call_id}:c1-preparation")
            seal_hash = digest(f"{call.call_id}:construction-seal")
            fixed_preparation_hash = digest(f"{call.call_id}:fixed-preparation")
            preparation_bindings = (
                PrequeryPreparationBinding(
                    unit_id=unit.runtime_unit_id,
                    condition=ConditionName.C1_LLM_PRE,
                    seed_block=call.seed_block,
                    snapshot_hash=unit.snapshot_hash,
                    preparation_hash=preparation_hash,
                    lineage_artifact_hash=seal_hash,
                    completed_at=c1_completed_at,
                ),
                PrequeryPreparationBinding(
                    unit_id=unit.runtime_unit_id,
                    condition=ConditionName.A_FIXED_SELECT,
                    seed_block=call.seed_block,
                    snapshot_hash=unit.snapshot_hash,
                    preparation_hash=fixed_preparation_hash,
                    lineage_artifact_hash=digest(f"{call.call_id}:fixed-lineage"),
                    completed_at=self.clock(),
                ),
            )
        else:
            attempt_hash = digest(f"{call.call_id}:condition-attempt")
        fixed_derivation = self._fixed_derivation(call, envelope)
        return ServiceCallResult(
            call_id=call.call_id,
            outcome=RunOutcome.SUCCEEDED,
            request_started=True,
            request_hash=digest(f"{call.call_id}:request"),
            response_artifact_hash=digest(f"{call.call_id}:raw-output"),
            validated_generation_hash=digest(f"{call.call_id}:validated-generation"),
            validation_record_hash=digest(f"{call.call_id}:validation"),
            condition_attempt_hash=attempt_hash,
            condition_preparation_hash=preparation_hash,
            ledger_receipt_hash=digest(f"{call.call_id}:ledger"),
            gpu_event_id=f"gpu-{call.call_id}",
            service_identity_hash=self._identity.content_hash,
            run_condition_config_hash=(
                fixed_derivation.exact_run_condition_config_hash
                if fixed_derivation is not None
                else envelope.expected_run_condition_config_hash
            ),
            fixed_schema_derivation=fixed_derivation,
            query_access_event_hash=(
                None
                if envelope.query_access_event is None
                else envelope.query_access_event.content_hash
            ),
            construction_seal_hash=seal_hash,
            prequery_preparation_bindings=preparation_bindings,
            repair_parent_raw_output_hash=(
                envelope.parent_output_artifact_hash
                if call.kind is DevelopmentCallKind.REPAIR_PROBE
                else None
            ),
            allocated_gpu_seconds=1.0,
            prompt_tokens=100,
            completion_tokens=50,
            completed_at=self.clock(),
        )

    def _invalid_result(
        self,
        call: DevelopmentCallSpec,
        envelope: CallExecutionEnvelope,
    ) -> ServiceCallResult:
        fixed_derivation = self._fixed_derivation(call, envelope)
        return ServiceCallResult(
            call_id=call.call_id,
            outcome=RunOutcome.INVALID,
            request_started=True,
            request_hash=digest(f"{call.call_id}:request"),
            response_artifact_hash=digest(f"{call.call_id}:raw-invalid-output"),
            validation_record_hash=digest(f"{call.call_id}:failed-validation"),
            ledger_receipt_hash=digest(f"{call.call_id}:ledger"),
            gpu_event_id=f"gpu-{call.call_id}",
            service_identity_hash=self._identity.content_hash,
            run_condition_config_hash=(
                fixed_derivation.exact_run_condition_config_hash
                if fixed_derivation is not None
                else envelope.expected_run_condition_config_hash
            ),
            fixed_schema_derivation=fixed_derivation,
            query_access_event_hash=(
                None
                if envelope.query_access_event is None
                else envelope.query_access_event.content_hash
            ),
            repair_parent_raw_output_hash=(
                envelope.parent_output_artifact_hash
                if call.kind is DevelopmentCallKind.REPAIR_PROBE
                else None
            ),
            allocated_gpu_seconds=1.0,
            prompt_tokens=100,
            completion_tokens=10,
            failure_code="schema_invalid",
            completed_at=self.clock(),
        )

    def _fixed_derivation(
        self,
        call: DevelopmentCallSpec,
        envelope: CallExecutionEnvelope,
    ) -> FixedSchemaDerivationReceipt | None:
        if call.kind is not DevelopmentCallKind.FIXED_SELECTION:
            return None
        plan = self.inputs.fixed_schema_plan_for(call.ordinal)
        return FixedSchemaDerivationReceipt(
            plan_hash=plan.content_hash,
            call_id=call.call_id,
            unit_id=call.unit_id,
            source_c1_construction_seal_hash=envelope.source_c1_seal_hash,
            source_c1_preparation_hash=digest(f"{call.unit_id}:c1-preparation"),
            fixed_ontology_hash=digest(f"{call.unit_id}:fixed-ontology"),
            evidence_alias_bijection_hash=digest(f"{call.unit_id}:aliases"),
            derived_output_schema_hash=digest(f"{call.unit_id}:fixed-schema"),
            derived_decoding_manifest_hash=digest(f"{call.unit_id}:fixed-decoding"),
            exact_run_condition_config_hash=digest(
                f"run-condition-config:{call.call_id}:derived"
            ),
            exact_run_condition_config_artifact_hash=digest(
                f"run-condition-config:{call.call_id}:artifact"
            ),
            derived_at=START - timedelta(seconds=1),
        )

    def execute_call(
        self,
        call: DevelopmentCallSpec,
        envelope: CallExecutionEnvelope,
    ) -> ServiceCallResult:
        self.executions.append(call.call_id)
        self.envelopes[call.call_id] = envelope
        self.allocated += 1.0
        result = (
            self._invalid_result(call, envelope)
            if call.call_id in self.invalid_calls
            else self._successful_result(call, envelope)
        )
        self.results[call.call_id] = result
        if self.crash_after_execution == len(self.executions) and not self.crashed:
            self.crashed = True
            if self.drop_result_on_crash:
                self.results.pop(call.call_id)
            raise ControllerCrash(call.call_id)
        return result

    def recover_call(
        self,
        call: DevelopmentCallSpec,
        envelope: CallExecutionEnvelope,
    ) -> ServiceCallResult | None:
        self.recoveries.append(call.call_id)
        assert self.envelopes[call.call_id] == envelope
        return self.results.get(call.call_id)


def make_runner(
    checkpoint: Path,
    service: MockBorrowedService,
    manifest: DevelopmentCallManifest,
    inputs: DevelopmentPrequeryInputs,
    clock: TickingClock,
    *,
    post_development_seconds: float = 1_000.0,
) -> DevelopmentRunner:
    return DevelopmentRunner(
        execution_id="development-execution-01",
        manifest=manifest,
        prequery_inputs=inputs,
        service=service,
        checkpoint_path=checkpoint,
        forecast_control=ForecastControl(
            forecast_receipt_hash=digest("fallback-continuation-forecast"),
            gpu_call_inventory_file_sha256=(manifest.gpu_call_inventory_file_sha256),
            post_development_mandatory_forecast_seconds=post_development_seconds,
        ),
        assessment_provider=passing_assessment,
        clock=clock,
    )


@pytest.mark.integration
def test_exact_24_calls_share_one_borrowed_service_and_respect_query_barrier(
    tmp_path: Path,
) -> None:
    manifest = load_development_call_manifest(ROOT)
    inputs = development_inputs(manifest)
    clock = TickingClock()
    service = MockBorrowedService(inputs, clock)

    result = make_runner(
        tmp_path / "development-checkpoint.json", service, manifest, inputs, clock
    ).run()

    assert result.phase is DevelopmentPhase.COMPLETED
    assert result.gate.passed is True
    assert len(result.itt_records) == 24
    assert Counter(row.call_class for row in result.itt_records) == Counter(
        {
            "development_c1": 4,
            "development_c2": 12,
            "development_fixed_select": 4,
            "development_ablation": 3,
            "development_repair": 1,
        }
    )
    assert service.executions == [call.call_id for call in manifest.calls]
    assert len(service.opened_stages) == len(set(service.opened_stages)) == 12
    assert len(result.query_access_events) == 12
    assert result.forecast.development_allocated_seconds == 24.0
    assert result.service_start_count_by_runner == 0
    assert result.model_load_count_by_runner == 0
    assert result.service_shutdown_by_runner is False
    assert result.service_returned_live_to_owner is True
    assert result.prequery_barrier_hash is not None
    first_query = service.envelopes[manifest.calls[4].call_id]
    assert first_query.prequery_barrier_hash == result.prequery_barrier_hash
    assert first_query.query_access_event is not None
    for call in manifest.calls[:4]:
        assert service.envelopes[call.call_id].prequery_barrier_hash is None
        assert service.envelopes[call.call_id].query_access_event is None


@pytest.mark.integration
def test_controller_resume_recovers_durable_active_call_without_duplicate_request(
    tmp_path: Path,
) -> None:
    manifest = load_development_call_manifest(ROOT)
    inputs = development_inputs(manifest)
    clock = TickingClock()
    service = MockBorrowedService(inputs, clock, crash_after_execution=7)
    checkpoint = tmp_path / "development-checkpoint.json"

    with pytest.raises(ControllerCrash):
        make_runner(checkpoint, service, manifest, inputs, clock).run()
    result = make_runner(checkpoint, service, manifest, inputs, clock).run()

    assert result.gate.passed is True
    assert len(service.executions) == 24
    assert Counter(service.executions) == Counter(call.call_id for call in manifest.calls)
    assert service.recoveries == [call.call_id for call in manifest.calls[:7]]
    assert result.forecast.development_allocated_seconds == 24.0


@pytest.mark.integration
def test_resume_without_durable_receipt_records_unknown_started_call_as_itt(
    tmp_path: Path,
) -> None:
    manifest = load_development_call_manifest(ROOT)
    inputs = development_inputs(manifest)
    clock = TickingClock()
    service = MockBorrowedService(
        inputs,
        clock,
        crash_after_execution=7,
        drop_result_on_crash=True,
    )
    checkpoint = tmp_path / "development-checkpoint.json"

    with pytest.raises(ControllerCrash):
        make_runner(checkpoint, service, manifest, inputs, clock).run()
    result = make_runner(checkpoint, service, manifest, inputs, clock).run()

    interrupted = result.itt_records[6]
    assert interrupted.call_id == manifest.calls[6].call_id
    assert interrupted.outcome is RunOutcome.INTERRUPTED
    assert interrupted.request_start_state.value == "unknown_after_interruption"
    assert interrupted.failure_code == "controller_interrupted_without_durable_call_receipt"
    assert service.executions.count(interrupted.call_id) == 1
    assert result.forecast.development_allocated_seconds == 24.0
    assert result.gate.passed is False


@pytest.mark.integration
def test_failed_c1_is_itt_and_mechanically_blocks_only_same_unit_fixed_select(
    tmp_path: Path,
) -> None:
    manifest = load_development_call_manifest(ROOT)
    inputs = development_inputs(manifest)
    clock = TickingClock()
    service = MockBorrowedService(
        inputs,
        clock,
        invalid_calls=frozenset({"dev-c1-u01"}),
    )

    result = make_runner(
        tmp_path / "development-checkpoint.json", service, manifest, inputs, clock
    ).run()

    blocked = next(row for row in result.itt_records if row.call_id == "dev-fixed-u01-q03")
    assert blocked.outcome is RunOutcome.FAILED
    assert blocked.failure_code == "same_unit_c1_seal_unavailable"
    assert blocked.allocated_gpu_seconds == 0.0
    assert len(service.executions) == 23
    assert len(result.itt_records) == 24
    assert len(result.query_access_events) == 12
    assert result.gate.passed is False


@pytest.mark.integration
def test_schedule_rejection_emits_24_unstarted_itt_rows_without_model_requests(
    tmp_path: Path,
) -> None:
    manifest = load_development_call_manifest(ROOT)
    inputs = development_inputs(manifest)
    clock = TickingClock()
    service = MockBorrowedService(inputs, clock, allocated_at_start=30_000.0)

    result = make_runner(
        tmp_path / "development-checkpoint.json",
        service,
        manifest,
        inputs,
        clock,
        post_development_seconds=1_000.0,
    ).run()

    assert service.executions == []
    assert service.opened_stages == []
    assert len(result.itt_records) == 24
    assert all(row.allocated_gpu_seconds == 0.0 for row in result.itt_records)
    assert result.admission_failure_code == "mandatory_manifest_forecast_not_admitted"
    assert result.gate.forecast_admitted is False
    assert result.gate.passed is False
