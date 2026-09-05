from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import story_projection_onto.held_out_controller as held_out_controller
import story_projection_onto.held_out_primary as held_out
from story_projection_onto.conditions.base import ConditionPreparation
from story_projection_onto.contracts import (
    ConditionName,
    ConstructionCertificate,
    ConstructionOperator,
    ConstructionSeal,
    OntologyDecision,
    PreQueryInventory,
    PrequeryPreparationBinding,
    QueryAccessEvent,
    canonical_sha256,
)
from story_projection_onto.development_runtime import RunOutcome
from story_projection_onto.held_out_controller import (
    C0ConstructionReceipt,
    HeldOutExecutionError,
    InterruptedCallRecoveryRequired,
    PreconstructedProjectionReceipt,
    authorize_scorer_bridge,
    execute_reviewed_held_out_manifest,
)
from story_projection_onto.held_out_primary import (
    FixedSelectCapabilityAudit,
    GlobalGpuScheduleSnapshot,
    HeldOutAblationPrequeryReceipt,
    HeldOutC2PrequeryReceipt,
    HeldOutCallArtifactReceipt,
    HeldOutCallEnvelope,
    HeldOutCASReference,
    HeldOutQueryOpening,
    HeldOutServiceResult,
    HeldOutServiceShutdownReceipt,
    HeldOutSessionIdentity,
    RepairReservePoolSnapshot,
    ReviewedHeldOutPlan,
    load_held_out_control_configuration,
)
from story_projection_onto.held_out_results import close_held_out_primary_results

ROOT = Path(__file__).resolve().parents[2]
NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _digest(value: object) -> str:
    return canonical_sha256({"TEST_ONLY": value})


def _ref(call_id: str, object_kind: str, logical_hash: str) -> HeldOutCASReference:
    return HeldOutCASReference(
        artifact_hash=_digest((call_id, object_kind, "physical")),
        logical_content_hash=logical_hash,
        object_kind=object_kind,
        media_type=f"application/vnd.story-projection.{object_kind.replace('_', '-')}+json",
        release_class="restricted",
    )


class _Cpu:
    def __init__(self, manifest):
        self.build_calls = 0
        self.projection_calls = 0
        self.query_stages = {
            query.staging_manifest_hash: query
            for unit in manifest.units
            for query in unit.query_stages
        }

    def build_c0(self, unit):
        assert not hasattr(unit, "query_stages")
        self.build_calls += 1
        return C0ConstructionReceipt(
            unit_id=unit.unit_id,
            prequery_stage_hash=unit.prequery_stage.staging_manifest_hash,
            outcome=RunOutcome.SUCCEEDED,
            construction_seal_hash=_digest((unit.unit_id, "c0-seal")),
            complete_graph_hash=_digest((unit.unit_id, "c0-graph")),
            completed_at=NOW,
        )

    def project_preconstructed(
        self,
        *,
        unit,
        query_stage_hash,
        query_opening,
        condition,
        seed_block,
        construction_seal_hash,
        complete_graph_hash,
    ):
        self.projection_calls += 1
        query = query_opening.opened_stage
        packet_hash = query_opening.evidence_packet_hash
        horizon_hash = query.horizon_hash
        budget_hash = query.budget_hash
        return PreconstructedProjectionReceipt(
            unit_id=unit.unit_id,
            query_stage_hash=query_stage_hash,
            condition=condition,
            seed_block=seed_block,
            source_construction_seal_hash=construction_seal_hash,
            source_complete_graph_hash=complete_graph_hash,
            outcome=RunOutcome.SUCCEEDED,
            projection_artifact_hash=_digest(
                (unit.unit_id, query_stage_hash, condition, seed_block)
            ),
            evidence_packet_hash=packet_hash,
            horizon_hash=horizon_hash,
            budget_hash=budget_hash,
            completed_at=NOW + timedelta(minutes=4),
        )

    def project_unavailable(self, **_kwargs):
        raise AssertionError("all TEST-ONLY preconstructions succeed")


class _SharedGpu:
    actual = 1000.0
    remaining = 20000.0
    calls = 0
    events = 0


class _Session:
    def __init__(self, condition: ConditionName, shared: _SharedGpu):
        self.condition = condition
        self.shared = shared
        self._identity = None

    def identity(self):
        if self._identity is not None:
            return self._identity
        self.shared.events += 1
        self.shared.actual += 0.001
        self.shared.remaining -= 180
        number = {
            ConditionName.C1_LLM_PRE: 1,
            ConditionName.C2_LLM_QUERY: 2,
            ConditionName.A_FIXED_SELECT: 3,
        }[self.condition]
        self._identity = HeldOutSessionIdentity(
            session_id=f"TEST-ONLY-session-{self.condition.value}",
            condition=self.condition,
            service_identity_hash=_digest((self.condition, "service")),
            service_pid=1000 + number,
            service_start_ticks=2000 + number,
            global_accounting_id="TEST-ONLY-global-accounting",
            global_ledger_chain_hash=_digest(("ledger", self.shared.events)),
            allocation_event_id=f"TEST-ONLY-allocation-{number}",
            model_load_event_id=f"TEST-ONLY-load-{number}",
            model_load_event_hash=_digest((self.condition, "load")),
        )
        return self._identity

    def schedule_snapshot(self):
        return GlobalGpuScheduleSnapshot(
            global_accounting_id="TEST-ONLY-global-accounting",
            gpu_call_inventory_file_sha256=(
                "b9a63e12c05321969b8c8edee03575ea11f678062d28ce62e34d577ec2603ea0"
            ),
            development_execution_result_hash="PENDING",
            development_predecessor_allocated_gpu_seconds=1000.0,
            ledger_chain_hash=_digest(("ledger", self.shared.events)),
            actual_allocated_gpu_seconds=self.shared.actual,
            remaining_registered_p95_seconds=self.shared.remaining,
            repair_reserves=(
                RepairReservePoolSnapshot(
                    reserve_class="reserve_long",
                    total_slots=4,
                    consumed_slots=0,
                    watchdog_seconds=240,
                ),
                RepairReservePoolSnapshot(
                    reserve_class="reserve_standard",
                    total_slots=8,
                    consumed_slots=0,
                    watchdog_seconds=150,
                ),
                RepairReservePoolSnapshot(
                    reserve_class="reserve_short",
                    total_slots=4,
                    consumed_slots=0,
                    watchdog_seconds=90,
                ),
            ),
            captured_at=NOW + timedelta(seconds=self.shared.events),
        )

    def execute_call(self, call, envelope: HeldOutCallEnvelope):
        self.shared.calls += 1
        self.shared.events += 1
        # The exact live-service interval includes controller/validation time
        # beyond the classified 0.01-second request event.
        self.shared.actual += 0.02
        self.shared.remaining -= call.p95_seconds
        values = (None, None, None)
        if envelope.query_stage is not None:
            assert envelope.query_opening is not None
            values = (
                envelope.query_opening.evidence_packet_hash,
                envelope.query_stage.horizon_hash,
                envelope.query_stage.budget_hash,
            )
        completed_at = NOW + timedelta(seconds=self.shared.events)
        if envelope.query_opening is not None:
            completed_at = max(
                completed_at,
                envelope.query_opening.opened_at + timedelta(seconds=1),
            )
        kwargs = {}
        if call.condition is ConditionName.C1_LLM_PRE:
            graph_hash = _digest((call.call_id, "graph"))
            seal = ConstructionSeal(
                seal_id=f"TEST-ONLY-seal-{call.ordinal}",
                condition=ConditionName.C1_LLM_PRE,
                snapshot_hash=envelope.prequery_stage.snapshot_hash,
                ontology_hash=graph_hash,
                constructed_at=NOW + timedelta(seconds=self.shared.calls - 0.75),
                sealed_at=NOW + timedelta(seconds=self.shared.calls - 0.5),
                sealed_object_ids=(f"TEST-ONLY-object-{call.ordinal}",),
            )
            kwargs.update(
                construction_seal_hash=seal.content_hash,
                complete_c1_graph_hash=graph_hash,
                construction_seal=seal,
            )
        elif call.condition is ConditionName.C2_LLM_QUERY:
            assert envelope.query_stage is not None
            assert envelope.query_opening is not None
            assert envelope.c2_prequery_receipt is not None
            reveal_time = envelope.query_opening.opened_at
            inventory = envelope.c2_prequery_receipt.inventory
            decision = OntologyDecision(
                decision_id=f"TEST-ONLY-c2-decision-{call.ordinal}",
                operator=ConstructionOperator.CONTEXTUAL_TYPE,
                evidence_ids=("TEST-ONLY-evidence",),
                rationale="TEST-ONLY contextual construction",
                decided_at=reveal_time + timedelta(seconds=0.1),
                created_object_ids=(f"TEST-ONLY-created-{call.ordinal}",),
            )
            certificate = ConstructionCertificate(
                certificate_id=f"TEST-ONLY-c2-certificate-{call.ordinal}",
                condition=ConditionName.C2_LLM_QUERY,
                snapshot_hash=envelope.prequery_stage.snapshot_hash,
                packet_hash=values[0],
                query_context_hash=envelope.query_stage.query_context_hash,
                query_access_event_hash=envelope.query_opening.query_access_event.content_hash,
                stage_manifest_hash=envelope.query_stage.staging_manifest_hash,
                prequery_barrier_hash=envelope.prequery_barrier.content_hash,
                generation_lineage_hash=_digest((call.call_id, "generation")),
                raw_output_artifact_hash=_digest((call.call_id, "raw")),
                normalized_draft_hash=_digest((call.call_id, "draft")),
                validation_bundle_hash=_digest((call.call_id, "validation")),
                query_revealed_at=reveal_time,
                completed_at=completed_at,
                decisions=(decision,),
                pre_query_inventory_hash=inventory.content_hash,
            )
            kwargs.update(
                empty_prequery_inventory_hash=inventory.content_hash,
                construction_certificate_hash=certificate.content_hash,
                pre_query_inventory=inventory,
                construction_certificate=certificate,
                query_revealed_at=reveal_time,
            )
        else:
            assert envelope.query_stage is not None
            assert envelope.query_opening is not None
            reveal_time = envelope.query_opening.opened_at
            decision = OntologyDecision(
                decision_id=f"TEST-ONLY-fixed-decision-{call.ordinal}",
                operator=ConstructionOperator.SELECTION,
                evidence_ids=("TEST-ONLY-evidence",),
                rationale="TEST-ONLY sealed-object selection",
                decided_at=reveal_time + timedelta(seconds=0.1),
                input_object_ids=(f"TEST-ONLY-source-{call.ordinal}",),
            )
            certificate = ConstructionCertificate(
                certificate_id=f"TEST-ONLY-fixed-certificate-{call.ordinal}",
                condition=ConditionName.A_FIXED_SELECT,
                snapshot_hash=envelope.prequery_stage.snapshot_hash,
                packet_hash=values[0],
                query_context_hash=envelope.query_stage.query_context_hash,
                query_access_event_hash=envelope.query_opening.query_access_event.content_hash,
                stage_manifest_hash=envelope.query_stage.staging_manifest_hash,
                prequery_barrier_hash=envelope.prequery_barrier.content_hash,
                generation_lineage_hash=_digest((call.call_id, "generation")),
                raw_output_artifact_hash=_digest((call.call_id, "raw")),
                normalized_draft_hash=_digest((call.call_id, "draft")),
                validation_bundle_hash=_digest((call.call_id, "validation")),
                query_revealed_at=reveal_time,
                completed_at=completed_at,
                decisions=(decision,),
                inherited_construction_seal_hash=envelope.source_c1_seal_hash,
            )
            kwargs["fixed_select_capability_audit"] = FixedSelectCapabilityAudit(
                complete_c1_graph_hash=envelope.complete_c1_graph_hash,
                source_c1_seal_hash=envelope.source_c1_seal_hash,
                constructive_operator_attempt_count=0,
                mechanically_rejected_operator_count=0,
            )
            kwargs.update(
                construction_certificate_hash=certificate.content_hash,
                construction_certificate=certificate,
                query_revealed_at=reveal_time,
            )
        request_hash = _digest((call.call_id, "request"))
        output_hash = _digest((call.call_id, "output"))
        validation_hash = _digest((call.call_id, "validation"))
        packing_hash = _digest((call.call_id, "packing"))
        artifact_receipt = HeldOutCallArtifactReceipt(
            call_id=call.call_id,
            call_spec_hash=call.content_hash,
            condition=call.condition,
            semantic_request=_ref(
                call.call_id,
                (
                    "preconstruction_request"
                    if call.condition is ConditionName.C1_LLM_PRE
                    else "construction_request"
                ),
                request_hash,
            ),
            packing_report=_ref(call.call_id, "packing_report", packing_hash),
            output=_ref(
                call.call_id,
                (
                    "condition_preparation"
                    if call.condition is ConditionName.C1_LLM_PRE
                    else "condition_attempt"
                ),
                output_hash,
            ),
            validation=_ref(call.call_id, "validated_generation", validation_hash),
            raw_response=_ref(
                call.call_id,
                "raw_model_response",
                _digest((call.call_id, "raw")),
            ),
            gpu_event_id=f"TEST-ONLY-gpu-{call.ordinal}",
            gpu_event_hash=_digest((call.call_id, "gpu")),
            model_call_id=f"TEST-ONLY-model-{call.ordinal}",
            model_call_record_hash=_digest((call.call_id, "model")),
            ledger_chain_hash=_digest(("ledger", self.shared.events)),
            completed_at=completed_at,
        )
        if call.condition is ConditionName.C1_LLM_PRE:
            fixed_preparation_hash = _digest((call.call_id, "fixed-preparation"))
            fixed_lineage_hash = _digest((call.call_id, "fixed-lineage"))
            kwargs["prequery_preparation_bindings"] = (
                PrequeryPreparationBinding(
                    unit_id=call.unit_id,
                    condition=ConditionName.C1_LLM_PRE,
                    seed_block=call.seed_block,
                    snapshot_hash=envelope.prequery_stage.snapshot_hash,
                    preparation_hash=output_hash,
                    lineage_artifact_hash=kwargs["construction_seal_hash"],
                    completed_at=kwargs["construction_seal"].sealed_at,
                ),
                PrequeryPreparationBinding(
                    unit_id=call.unit_id,
                    condition=ConditionName.A_FIXED_SELECT,
                    seed_block=call.seed_block,
                    snapshot_hash=envelope.prequery_stage.snapshot_hash,
                    preparation_hash=fixed_preparation_hash,
                    lineage_artifact_hash=fixed_lineage_hash,
                    completed_at=kwargs["construction_seal"].sealed_at,
                ),
            )
        return HeldOutServiceResult(
            call_id=call.call_id,
            condition=call.condition,
            outcome=RunOutcome.SUCCEEDED,
            request_started=True,
            request_hash=request_hash,
            output_artifact_hash=output_hash,
            validation_artifact_hash=validation_hash,
            ledger_receipt_hash=artifact_receipt.content_hash,
            ledger_receipt_artifact_hash=_digest((call.call_id, "receipt-artifact")),
            packing_report_hash=packing_hash,
            artifact_receipt=artifact_receipt,
            global_ledger_chain_hash=_digest(("ledger", self.shared.events)),
            evidence_packet_hash=values[0],
            horizon_hash=values[1],
            budget_hash=values[2],
            allocated_gpu_seconds=0.01,
            repair_attempts=0,
            completed_at=completed_at,
            **kwargs,
        )

    def recover_call(self, _call, _envelope):
        raise AssertionError("completed TEST-ONLY records must resume without service recovery")

    def finalize_unstarted_call(self, call, result):
        assert result.request_started is False
        assert result.call_id == call.call_id
        self.shared.remaining -= call.p95_seconds
        return self.schedule_snapshot()


class _Runtime:
    def __init__(self, shared: _SharedGpu):
        self.shared = shared
        self.shutdowns = {}
        self.validations = 0

    def prepare_c2_empty_inventory(self, unit, *, seed_block):
        inventory = PreQueryInventory(
            inventory_id=f"TEST-ONLY-empty-{unit.unit_id}-s{seed_block}",
            snapshot_hash=unit.prequery_stage.snapshot_hash,
            recorded_at=NOW,
        )
        preparation = ConditionPreparation(
            preparation_id=f"TEST-ONLY-preparation-{unit.unit_id}-s{seed_block}",
            condition=ConditionName.C2_LLM_QUERY,
            snapshot_hash=unit.prequery_stage.snapshot_hash,
            completed_at=NOW,
            empty_inventory=inventory,
        )
        return HeldOutC2PrequeryReceipt(
            unit_id=unit.unit_id,
            seed_block=seed_block,
            prequery_stage_hash=unit.prequery_stage.staging_manifest_hash,
            inventory=inventory,
            inventory_artifact=_ref(
                f"{unit.unit_id}-s{seed_block}",
                "pre_query_inventory",
                inventory.content_hash,
            ),
            preparation=preparation,
            preparation_artifact=_ref(
                f"{unit.unit_id}-s{seed_block}",
                "condition_preparation",
                preparation.content_hash,
            ),
            completed_at=NOW,
        )

    def validate_c2_prequery_receipt(self, _receipt):
        return None

    def prepare_ablation_empty_inventory(self, unit, *, condition):
        inventory = PreQueryInventory(
            inventory_id=f"TEST-ONLY-empty-{condition.value}-{unit.unit_id}-s1",
            condition=condition,
            snapshot_hash=unit.prequery_stage.snapshot_hash,
            recorded_at=NOW,
        )
        preparation = ConditionPreparation(
            preparation_id=f"TEST-ONLY-preparation-{condition.value}-{unit.unit_id}-s1",
            condition=condition,
            snapshot_hash=unit.prequery_stage.snapshot_hash,
            completed_at=NOW,
            empty_inventory=inventory,
        )
        return HeldOutAblationPrequeryReceipt(
            unit_id=unit.unit_id,
            condition=condition,
            prequery_stage_hash=unit.prequery_stage.staging_manifest_hash,
            preparation=preparation,
            inventory_artifact=_ref(
                f"{unit.unit_id}-{condition.value}",
                "pre_query_inventory",
                inventory.content_hash,
            ),
            preparation_artifact=_ref(
                f"{unit.unit_id}-{condition.value}",
                "condition_preparation",
                preparation.content_hash,
            ),
            completed_at=NOW,
        )

    def validate_ablation_prequery_receipt(self, _receipt):
        return None

    def persist_prequery_barrier(self, barrier):
        self.barrier = barrier

    def open_or_recover_query(self, stage, *, barrier, persisted):
        if persisted is not None:
            return persisted
        context_hash = _digest((stage.staging_manifest_hash, "context"))
        horizon_hash = _digest((stage.staging_manifest_hash, "horizon"))
        budget_hash = _digest((stage.staging_manifest_hash, "budget"))
        opened = stage.__class__(
            **stage.model_dump(
                mode="python",
                exclude={
                    "content_hash",
                    "query_context_hash",
                    "horizon_hash",
                    "budget_hash",
                },
            ),
            query_context_hash=context_hash,
            horizon_hash=horizon_hash,
            budget_hash=budget_hash,
        )
        accessed_at = barrier.sealed_at + timedelta(seconds=1)
        packet_hash = _digest((stage.staging_manifest_hash, "packet"))
        event = QueryAccessEvent(
            access_event_id=f"TEST-ONLY-access-{stage.stage_id}",
            execution_id=barrier.execution_id,
            query_context_hash=context_hash,
            model_visible_query_hash=_digest((stage.staging_manifest_hash, "visible-query")),
            snapshot_hash=stage.snapshot_hash,
            stage_manifest_hash=stage.staging_manifest_hash,
            query_artifact_hash=stage.query_artifact_hash,
            prequery_barrier_hash=barrier.content_hash,
            packet_hash=packet_hash,
            registered_revealed_at=NOW,
            accessed_at=accessed_at,
        )
        return HeldOutQueryOpening(
            sealed_stage_hash=stage.staging_manifest_hash,
            opened_stage=opened,
            query_access_event=event,
            query_access_artifact=_ref(stage.stage_id, "query_access_event", event.content_hash),
            evidence_packet_hash=packet_hash,
            evidence_packet_artifact=_ref(stage.stage_id, "evidence_packet", packet_hash),
            opened_at=accessed_at,
        )

    def validate_result_artifacts(self, _call, _envelope, _result):
        self.validations += 1

    def shutdown_condition(self, identity):
        prior = self.shutdowns.get(identity.condition)
        if prior is None:
            self.shared.events += 1
            self.shared.actual += 0.001
            snapshot = _Session(identity.condition, self.shared).schedule_snapshot()
            prior = HeldOutServiceShutdownReceipt(
                condition=identity.condition,
                session_identity_hash=identity.content_hash,
                allocation_event_id=identity.allocation_event_id,
                model_load_event_id=identity.model_load_event_id,
                model_load_event_hash=identity.model_load_event_hash,
                global_accounting_id=identity.global_accounting_id,
                final_ledger_chain_hash=snapshot.ledger_chain_hash,
                schedule_after_shutdown=snapshot,
                stopped_at=snapshot.captured_at,
            )
            self.shutdowns[identity.condition] = (prior, snapshot)
            return prior, snapshot
        return prior


class _CrashOnceSession(_Session):
    def __init__(self, condition: ConditionName, shared: _SharedGpu):
        super().__init__(condition, shared)
        self.crash_after_execute = True
        self.pending_result = None

    def execute_call(self, call, envelope):
        result = super().execute_call(call, envelope)
        self.pending_result = result
        return result

    def schedule_snapshot(self):
        if self.pending_result is not None and self.crash_after_execute:
            self.crash_after_execute = False
            raise RuntimeError("TEST-ONLY crash after durable service receipt")
        return super().schedule_snapshot()

    def recover_call(self, _call, _envelope):
        result = self.pending_result
        self.pending_result = None
        return result


class _TimeoutFirstSession(_Session):
    def __init__(self, condition: ConditionName, shared: _SharedGpu):
        super().__init__(condition, shared)
        self.timed_out = False

    def execute_call(self, call, envelope):
        result = super().execute_call(call, envelope)
        if self.timed_out:
            return result
        self.timed_out = True
        payload = result.model_dump(mode="python", exclude={"content_hash"})
        payload.update(
            outcome=RunOutcome.TIMED_OUT,
            failure_code="held_out_timeout",
        )
        return HeldOutServiceResult.model_validate(payload)


class _PreIntentInterruptSession(_Session):
    def __init__(self, condition: ConditionName, shared: _SharedGpu):
        super().__init__(condition, shared)
        self.interrupt_once = True

    def execute_call(self, call, envelope):
        if self.interrupt_once:
            self.interrupt_once = False
            raise KeyboardInterrupt("TEST-ONLY interruption before semantic intent")
        return super().execute_call(call, envelope)

    def recovery_pending(self, _call, _envelope):
        return False

    def recover_call(self, call, envelope):
        return self.execute_call(call, envelope)


def test_complete_control_plane_inventory_and_exact_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configuration = load_held_out_control_configuration(ROOT)
    manifest = held_out._derive_call_manifest(ROOT, configuration)
    reviewed_plan = ReviewedHeldOutPlan(
        call_manifest=manifest,
        review_completion_manifest_hash=_digest("review-completion"),
        review_draft_seal_hash=_digest("review-draft"),
        final_reviewed_seal_hash=_digest("reviewed-seal"),
    )
    monkeypatch.setattr(
        held_out_controller,
        "open_reviewed_held_out_plan",
        lambda **_kwargs: reviewed_plan,
    )
    cpu = _Cpu(manifest)
    shared = _SharedGpu()
    sessions = {
        condition: _Session(condition, shared)
        for condition in (
            ConditionName.C1_LLM_PRE,
            ConditionName.C2_LLM_QUERY,
            ConditionName.A_FIXED_SELECT,
        )
    }
    runtime = _Runtime(shared)
    output = tmp_path / "TEST-ONLY-held-out"
    completed = NOW + timedelta(hours=1)
    execution = execute_reviewed_held_out_manifest(
        reviewed_plan=reviewed_plan,
        configuration=configuration,
        repository=ROOT,
        review_completion_root=tmp_path / "TEST-ONLY-review",
        configuration_path=held_out.DEFAULT_HELD_OUT_CONTROL_PATH,
        cpu=cpu,
        sessions=sessions,
        runtime=runtime,
        output_root=output,
        completion_clock=lambda: completed,
    )
    assert len(execution.itt_records) == 168
    assert len(execution.c2_prequery_receipts) == 24
    assert len(execution.ablation_prequery_receipts) == 36
    assert len(execution.prequery_barrier.preparation_bindings) == 120
    assert len(execution.preconstructed_projections) == 108
    assert execution.completed_at == completed
    assert cpu.build_calls == 12
    assert cpu.projection_calls == 108
    assert shared.calls == 168
    assert tuple(item.condition for item in execution.service_shutdown_receipts) == (
        ConditionName.C1_LLM_PRE,
        ConditionName.C2_LLM_QUERY,
        ConditionName.A_FIXED_SELECT,
    )
    assert len({item.allocation_event_id for item in execution.session_identities}) == 3
    assert len({item.model_load_event_id for item in execution.session_identities}) == 3
    assert all(
        item.result.fixed_select_capability_audit is None
        or item.result.fixed_select_capability_audit.accepted_constructive_operator_count == 0
        for item in execution.itt_records
    )
    bridge = authorize_scorer_bridge(
        execution,
        call_manifest=manifest,
        output_root=output,
        runtime=runtime,
        authorized_at=completed + timedelta(seconds=1),
    )
    assert bridge.runtime_namespace_closed and not bridge.model_input_open
    results_root = tmp_path / "TEST-ONLY-held-out-results"
    closed_bridge, primary_gate = close_held_out_primary_results(
        reviewed_plan=reviewed_plan,
        call_manifest=manifest,
        execution=execution,
        held_out_output_root=output,
        results_root=results_root,
        runtime=runtime,
        clock=lambda: completed + timedelta(seconds=1),
    )
    assert closed_bridge == bridge
    assert primary_gate.held_out_execution_manifest_hash == execution.content_hash
    assert primary_gate.scorer_bridge_hash == bridge.content_hash
    assert primary_gate.result_artifact_hashes == bridge.output_artifact_hashes
    assert (results_root / "held_out_execution_manifest.json").read_bytes() == (
        output / "execution_manifest.json"
    ).read_bytes()
    assert {
        item.name for item in results_root.iterdir()
    } == {
        "held_out_execution_manifest.json",
        "primary_results_gate.json",
        "scorer_bridge.json",
    }
    interrupted_result_temporary = results_root / (
        ".primary_results_gate.json.TEST-ONLY-power-loss.tmp"
    )
    interrupted_result_temporary.write_bytes(
        (results_root / "primary_results_gate.json").read_bytes()
    )
    replayed_bridge, replayed_gate = close_held_out_primary_results(
        reviewed_plan=reviewed_plan,
        call_manifest=manifest,
        execution=execution,
        held_out_output_root=output,
        results_root=results_root,
        runtime=runtime,
        clock=lambda: completed + timedelta(days=1),
    )
    assert replayed_bridge == bridge
    assert replayed_gate == primary_gate
    assert interrupted_result_temporary.exists()

    first_itt_path = next((output / "itt").glob("*.json"))
    interrupted_temporary = first_itt_path.with_name(
        f".{first_itt_path.name}.power-loss.tmp"
    )
    interrupted_temporary.write_bytes(first_itt_path.read_bytes())
    resumed = execute_reviewed_held_out_manifest(
        reviewed_plan=reviewed_plan,
        configuration=configuration,
        repository=ROOT,
        review_completion_root=tmp_path / "TEST-ONLY-review",
        configuration_path=held_out.DEFAULT_HELD_OUT_CONTROL_PATH,
        cpu=cpu,
        sessions=sessions,
        runtime=runtime,
        output_root=output,
        completion_clock=lambda: completed + timedelta(days=1),
    )
    assert resumed.content_hash == execution.content_hash
    assert resumed.completed_at == completed
    assert interrupted_temporary.exists()
    assert shared.calls == 168
    assert cpu.build_calls == 12
    assert cpu.projection_calls == 108


def test_finalized_journal_rejects_extra_and_partial_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configuration = load_held_out_control_configuration(ROOT)
    manifest = held_out._derive_call_manifest(ROOT, configuration)
    reviewed_plan = ReviewedHeldOutPlan(
        call_manifest=manifest,
        review_completion_manifest_hash=_digest("review-completion"),
        review_draft_seal_hash=_digest("review-draft"),
        final_reviewed_seal_hash=_digest("reviewed-seal"),
    )
    monkeypatch.setattr(
        held_out_controller,
        "open_reviewed_held_out_plan",
        lambda **_kwargs: reviewed_plan,
    )
    cpu = _Cpu(manifest)
    shared = _SharedGpu()
    shared.actual = 1000.0
    shared.remaining = 20000.0
    shared.calls = 0
    sessions = {
        condition: _Session(condition, shared)
        for condition in (
            ConditionName.C1_LLM_PRE,
            ConditionName.C2_LLM_QUERY,
            ConditionName.A_FIXED_SELECT,
        )
    }
    runtime = _Runtime(shared)
    output = tmp_path / "TEST-ONLY-held-out"
    execution = execute_reviewed_held_out_manifest(
        reviewed_plan=reviewed_plan,
        configuration=configuration,
        repository=ROOT,
        review_completion_root=tmp_path / "TEST-ONLY-review",
        configuration_path=held_out.DEFAULT_HELD_OUT_CONTROL_PATH,
        cpu=cpu,
        sessions=sessions,
        runtime=runtime,
        output_root=output,
        completion_clock=lambda: NOW + timedelta(hours=1),
    )
    tampered_itt = next((output / "itt").glob("*.json"))
    original_itt = tampered_itt.read_bytes()
    tampered_itt.write_text("{}")
    with pytest.raises(HeldOutExecutionError, match="journal record"):
        authorize_scorer_bridge(
            execution,
            call_manifest=manifest,
            output_root=output,
            runtime=runtime,
            authorized_at=execution.completed_at + timedelta(seconds=1),
        )
    tampered_itt.write_bytes(original_itt)
    canary = output / "itt" / "unexpected.json"
    canary.write_text("TEST-ONLY")
    with pytest.raises(HeldOutExecutionError, match="unexpected held-out journal file"):
        execute_reviewed_held_out_manifest(
            reviewed_plan=reviewed_plan,
            configuration=configuration,
            repository=ROOT,
            review_completion_root=tmp_path / "TEST-ONLY-review",
            configuration_path=held_out.DEFAULT_HELD_OUT_CONTROL_PATH,
            cpu=cpu,
            sessions=sessions,
            runtime=runtime,
            output_root=output,
            completion_clock=lambda: NOW + timedelta(hours=1),
        )
    canary.unlink()
    next((output / "projections").glob("*.json")).unlink()
    with pytest.raises(HeldOutExecutionError, match="finalized held-out journal is partial"):
        execute_reviewed_held_out_manifest(
            reviewed_plan=reviewed_plan,
            configuration=configuration,
            repository=ROOT,
            review_completion_root=tmp_path / "TEST-ONLY-review",
            configuration_path=held_out.DEFAULT_HELD_OUT_CONTROL_PATH,
            cpu=cpu,
            sessions=sessions,
            runtime=runtime,
            output_root=output,
            completion_clock=lambda: NOW + timedelta(hours=1),
        )


def test_executor_cannot_bypass_materialized_review_gate(tmp_path: Path) -> None:
    configuration = load_held_out_control_configuration(ROOT)
    manifest = held_out._derive_call_manifest(ROOT, configuration)
    reviewed_plan = ReviewedHeldOutPlan(
        call_manifest=manifest,
        review_completion_manifest_hash=_digest("forged-review-completion"),
        review_draft_seal_hash=_digest("forged-review-draft"),
        final_reviewed_seal_hash=_digest("forged-review-seal"),
    )
    with pytest.raises(held_out.HeldOutReviewGateError, match="independent review"):
        execute_reviewed_held_out_manifest(
            reviewed_plan=reviewed_plan,
            configuration=configuration,
            repository=ROOT,
            review_completion_root=tmp_path / "missing-review",
            configuration_path=held_out.DEFAULT_HELD_OUT_CONTROL_PATH,
            cpu=_Cpu(manifest),
            sessions={},
            runtime=_Runtime(_SharedGpu()),
            output_root=tmp_path / "must-not-exist",
            completion_clock=lambda: NOW + timedelta(hours=1),
        )
    assert not (tmp_path / "must-not-exist").exists()


def test_interrupted_call_recovers_against_original_snapshot_without_resend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configuration = load_held_out_control_configuration(ROOT)
    manifest = held_out._derive_call_manifest(ROOT, configuration)
    reviewed_plan = ReviewedHeldOutPlan(
        call_manifest=manifest,
        review_completion_manifest_hash=_digest("review-completion-recovery"),
        review_draft_seal_hash=_digest("review-draft-recovery"),
        final_reviewed_seal_hash=_digest("reviewed-seal-recovery"),
    )
    monkeypatch.setattr(
        held_out_controller,
        "open_reviewed_held_out_plan",
        lambda **_kwargs: reviewed_plan,
    )
    shared = _SharedGpu()
    shared.actual = 1000.0
    shared.remaining = 20000.0
    shared.calls = 0
    c1 = _CrashOnceSession(ConditionName.C1_LLM_PRE, shared)
    sessions = {
        ConditionName.C1_LLM_PRE: c1,
        ConditionName.C2_LLM_QUERY: _Session(ConditionName.C2_LLM_QUERY, shared),
        ConditionName.A_FIXED_SELECT: _Session(ConditionName.A_FIXED_SELECT, shared),
    }
    runtime = _Runtime(shared)
    output = tmp_path / "TEST-ONLY-recovery"
    arguments = dict(
        reviewed_plan=reviewed_plan,
        configuration=configuration,
        repository=ROOT,
        review_completion_root=tmp_path / "TEST-ONLY-review",
        configuration_path=held_out.DEFAULT_HELD_OUT_CONTROL_PATH,
        cpu=_Cpu(manifest),
        sessions=sessions,
        runtime=runtime,
        output_root=output,
        completion_clock=lambda: NOW + timedelta(hours=1),
    )
    with pytest.raises(
        held_out_controller.InterruptedCallRecoveryRequired,
        match="requires recovery",
    ):
        execute_reviewed_held_out_manifest(**arguments)
    assert shared.calls == 1
    assert len(tuple((output / "call_slots").glob("*.json"))) == 1
    assert not (output / "itt" / f"001-{manifest.calls[0].call_id}.json").exists()

    execution = execute_reviewed_held_out_manifest(**arguments)
    assert len(execution.itt_records) == 168
    assert shared.calls == 168


def test_timeout_stops_only_its_condition_block_and_records_deterministic_itt_skips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configuration = load_held_out_control_configuration(ROOT)
    manifest = held_out._derive_call_manifest(ROOT, configuration)
    reviewed_plan = ReviewedHeldOutPlan(
        call_manifest=manifest,
        review_completion_manifest_hash=_digest("review-completion-timeout"),
        review_draft_seal_hash=_digest("review-draft-timeout"),
        final_reviewed_seal_hash=_digest("reviewed-seal-timeout"),
    )
    monkeypatch.setattr(
        held_out_controller,
        "open_reviewed_held_out_plan",
        lambda **_kwargs: reviewed_plan,
    )
    shared = _SharedGpu()
    shared.actual = 1000.0
    shared.remaining = 20000.0
    shared.calls = 0
    shared.events = 0
    sessions = {
        ConditionName.C1_LLM_PRE: _Session(ConditionName.C1_LLM_PRE, shared),
        ConditionName.C2_LLM_QUERY: _TimeoutFirstSession(
            ConditionName.C2_LLM_QUERY,
            shared,
        ),
        ConditionName.A_FIXED_SELECT: _Session(
            ConditionName.A_FIXED_SELECT,
            shared,
        ),
    }
    execution = execute_reviewed_held_out_manifest(
        reviewed_plan=reviewed_plan,
        configuration=configuration,
        repository=ROOT,
        review_completion_root=tmp_path / "TEST-ONLY-review",
        configuration_path=held_out.DEFAULT_HELD_OUT_CONTROL_PATH,
        cpu=_Cpu(manifest),
        sessions=sessions,
        runtime=_Runtime(shared),
        output_root=tmp_path / "TEST-ONLY-timeout",
        completion_clock=lambda: NOW + timedelta(hours=1),
    )
    c2_records = tuple(
        item
        for item in execution.itt_records
        if item.result.condition is ConditionName.C2_LLM_QUERY
    )
    fixed_records = tuple(
        item
        for item in execution.itt_records
        if item.result.condition is ConditionName.A_FIXED_SELECT
    )
    assert len(c2_records) == 72
    assert c2_records[0].result.outcome is RunOutcome.TIMED_OUT
    assert c2_records[0].result.request_started
    assert all(
        not item.result.request_started
        and item.result.failure_code == "condition_service_unavailable_after_timeout"
        for item in c2_records[1:]
    )
    assert all(
        item.schedule_after.actual_allocated_gpu_seconds
        == item.schedule_before.actual_allocated_gpu_seconds
        and item.schedule_after.remaining_registered_p95_seconds
        == item.schedule_before.remaining_registered_p95_seconds - 95
        for item in c2_records[1:]
    )
    assert all(item.result.request_started for item in fixed_records)
    assert shared.calls == 24 + 1 + 72


def test_pre_intent_interruption_resumes_exact_slot_without_duplicate_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configuration = load_held_out_control_configuration(ROOT)
    manifest = held_out._derive_call_manifest(ROOT, configuration)
    reviewed_plan = ReviewedHeldOutPlan(
        call_manifest=manifest,
        review_completion_manifest_hash=_digest("review-completion-pre-intent"),
        review_draft_seal_hash=_digest("review-draft-pre-intent"),
        final_reviewed_seal_hash=_digest("reviewed-seal-pre-intent"),
    )
    monkeypatch.setattr(
        held_out_controller,
        "open_reviewed_held_out_plan",
        lambda **_kwargs: reviewed_plan,
    )
    shared = _SharedGpu()
    shared.actual = 1000.0
    shared.remaining = 20000.0
    shared.calls = 0
    shared.events = 0
    c1 = _PreIntentInterruptSession(ConditionName.C1_LLM_PRE, shared)
    arguments = dict(
        reviewed_plan=reviewed_plan,
        configuration=configuration,
        repository=ROOT,
        review_completion_root=tmp_path / "TEST-ONLY-review",
        configuration_path=held_out.DEFAULT_HELD_OUT_CONTROL_PATH,
        cpu=_Cpu(manifest),
        sessions={
            ConditionName.C1_LLM_PRE: c1,
            ConditionName.C2_LLM_QUERY: _Session(ConditionName.C2_LLM_QUERY, shared),
            ConditionName.A_FIXED_SELECT: _Session(
                ConditionName.A_FIXED_SELECT,
                shared,
            ),
        },
        runtime=_Runtime(shared),
        output_root=tmp_path / "TEST-ONLY-pre-intent",
        completion_clock=lambda: NOW + timedelta(hours=1),
    )
    with pytest.raises(InterruptedCallRecoveryRequired, match="requires recovery"):
        execute_reviewed_held_out_manifest(**arguments)
    assert shared.calls == 0
    assert len(tuple((arguments["output_root"] / "call_slots").glob("*.json"))) == 1

    execution = execute_reviewed_held_out_manifest(**arguments)
    assert len(execution.itt_records) == 168
    assert shared.calls == 168
