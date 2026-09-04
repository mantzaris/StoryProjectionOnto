"""Trusted, gold-free compiler for the combined 49-call production block.

This module consumes only already-frozen runtime artifacts and the small scorer-
only *selection* manifests used to choose registered subsets.  It never reads a
gold projection and emits no scoring fields.  Its output is safe to hand to the
combined production coordinator; the model sees only each call's separately
allowlisted request payload.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

from story_projection_onto.combined_gpu_block import (
    ACTIVE_ABLATION_CONDITIONS,
    AblationPrequeryLineage,
    CombinedBlockError,
    CombinedQuerySource,
    CombinedUpstreamGate,
    PrimaryC2Reference,
    PublicEvidencePacketPointer,
    RestrictedArtifactPointer,
)
from story_projection_onto.contracts import (
    ConditionName,
    QueryContext,
    RunOutcome,
    canonical_sha256,
)
from story_projection_onto.development_runtime import (
    DevelopmentExecutionResult,
    DevelopmentPhase,
)
from story_projection_onto.held_out_controller import (
    HeldOutExecutionManifest,
    ScorerBridgeAuthorization,
)
from story_projection_onto.held_out_primary import (
    GlobalGpuScheduleSnapshot,
    HeldOutCallManifest,
    HeldOutCASReference,
)
from story_projection_onto.phase5_execution import Phase5ExecutionInputManifest


def _restricted(reference: HeldOutCASReference) -> RestrictedArtifactPointer:
    if reference.release_class != "restricted":
        raise CombinedBlockError("combined predecessor CAS object must remain restricted")
    return RestrictedArtifactPointer(
        artifact_hash=reference.artifact_hash,
        logical_content_hash=reference.logical_content_hash,
        object_kind=reference.object_kind,
        media_type=reference.media_type,
    )


def _public_packet(reference: HeldOutCASReference) -> PublicEvidencePacketPointer:
    if (
        reference.release_class != "public"
        or reference.object_kind != "evidence_packet"
        or reference.media_type
        != "application/vnd.story-projection.evidence-packet+json"
    ):
        raise CombinedBlockError(
            "combined evidence predecessor must be the exact public packet artifact"
        )
    return PublicEvidencePacketPointer(
        artifact_hash=reference.artifact_hash,
        logical_content_hash=reference.logical_content_hash,
    )


def replay_combined_upstream_gate(
    *,
    development: DevelopmentExecutionResult,
    held_out_calls: HeldOutCallManifest,
    held_out_execution: HeldOutExecutionManifest,
    scorer_bridge: ScorerBridgeAuthorization,
    final_schedule: GlobalGpuScheduleSnapshot,
    phase5_inputs: Phase5ExecutionInputManifest,
    verified_at: datetime,
) -> CombinedUpstreamGate:
    """Reproduce every blocking predecessor instead of trusting status booleans."""

    if verified_at.tzinfo is None or verified_at.utcoffset() is None:
        raise CombinedBlockError("combined upstream verification time must be aware")
    if (
        development.phase
        not in {DevelopmentPhase.COMPLETED, DevelopmentPhase.COMPLETED_WITH_FAILURES}
        or not development.gate.passed
        or development.admission_failure_code is not None
        or held_out_calls.development_execution_result_hash != development.content_hash
        or held_out_execution.call_manifest_hash != held_out_calls.content_hash
        or scorer_bridge.execution_manifest_hash != held_out_execution.content_hash
        or scorer_bridge.call_manifest_hash != held_out_calls.content_hash
        or scorer_bridge.final_reviewed_seal_hash != held_out_execution.final_reviewed_seal_hash
        or not scorer_bridge.runtime_namespace_closed
        or scorer_bridge.model_input_open
        or tuple(item.content_hash for item in held_out_execution.itt_records)
        != scorer_bridge.itt_record_hashes
        or held_out_execution.final_schedule_snapshot_hash != final_schedule.content_hash
        or phase5_inputs.held_out_execution_manifest_hash != held_out_execution.content_hash
        or phase5_inputs.held_out_call_manifest_hash != held_out_calls.content_hash
        or phase5_inputs.final_reviewed_seal_hash != held_out_execution.final_reviewed_seal_hash
        or phase5_inputs.frozen_at <= scorer_bridge.authorized_at
        or verified_at < phase5_inputs.frozen_at
    ):
        raise CombinedBlockError(
            "combined block requires accepted development, complete primary ITT, and frozen Phase 5"
        )
    if final_schedule.actual_allocated_gpu_seconds + 1e-6 < (
        held_out_execution.itt_records[-1].schedule_after.actual_allocated_gpu_seconds
    ):
        raise CombinedBlockError("combined predecessor cumulative GPU schedule changed")
    short = next(
        item for item in final_schedule.repair_reserves if item.reserve_class == "reserve_short"
    )
    registry_hash = canonical_sha256(
        tuple(item.content_hash for item in held_out_execution.ablation_prequery_receipts)
    )
    return CombinedUpstreamGate(
        development_execution_result_hash=development.content_hash,
        development_gate_passed=True,
        held_out_call_manifest_hash=held_out_calls.content_hash,
        held_out_execution_manifest_hash=held_out_execution.content_hash,
        held_out_scorer_bridge_hash=scorer_bridge.content_hash,
        held_out_itt_record_hashes=scorer_bridge.itt_record_hashes,
        held_out_final_schedule_snapshot_hash=final_schedule.content_hash,
        held_out_prequery_barrier_hash=held_out_execution.prequery_barrier.content_hash,
        ablation_prequery_registry_hash=registry_hash,
        phase5_input_manifest_hash=phase5_inputs.content_hash,
        final_reviewed_seal_hash=held_out_execution.final_reviewed_seal_hash,
        global_accounting_id=final_schedule.global_accounting_id,
        actual_allocated_gpu_seconds_before_block=(final_schedule.actual_allocated_gpu_seconds),
        remaining_registered_p95_seconds_before_block=(
            final_schedule.remaining_registered_p95_seconds
        ),
        consumed_short_reserve_slots_before_block=short.consumed_slots,
        verified_at=verified_at,
    )


def compile_combined_query_sources(
    *,
    held_out_calls: HeldOutCallManifest,
    held_out_execution: HeldOutExecutionManifest,
    scorer_bridge: ScorerBridgeAuthorization,
    contexts_by_query_stage_hash: Mapping[str, QueryContext],
) -> Mapping[str, CombinedQuerySource]:
    """Compile all 36 gold-free held-out sources for later registered selection."""

    if (
        scorer_bridge.execution_manifest_hash != held_out_execution.content_hash
        or scorer_bridge.call_manifest_hash != held_out_calls.content_hash
        or scorer_bridge.prequery_barrier_hash != held_out_execution.prequery_barrier.content_hash
        or tuple(item.content_hash for item in held_out_execution.query_openings)
        != scorer_bridge.query_opening_hashes
    ):
        raise CombinedBlockError("combined sources do not share the closed held-out runtime")
    stages = {
        stage.staging_manifest_hash: (unit, stage)
        for unit in held_out_calls.units
        for stage in unit.query_stages
    }
    if set(contexts_by_query_stage_hash) != set(stages):
        raise CombinedBlockError("combined source contexts must cover exactly 36 query stages")
    openings = {item.sealed_stage_hash: item for item in held_out_execution.query_openings}
    c2_receipts = {
        (item.unit_id, item.seed_block): item for item in held_out_execution.c2_prequery_receipts
    }
    ablation_receipts = {
        (item.unit_id, item.condition): item
        for item in held_out_execution.ablation_prequery_receipts
    }
    c2_calls = {
        item.query_stage_hash: (item, record)
        for item, record in zip(
            held_out_calls.calls,
            held_out_execution.itt_records,
            strict=True,
        )
        if item.call_class == "test_c2" and item.seed_block == 1
    }
    if len(c2_calls) != 36:
        raise CombinedBlockError("combined sources require every seed-1 primary C2 call")
    ablation_registry_hash = canonical_sha256(
        tuple(item.content_hash for item in held_out_execution.ablation_prequery_receipts)
    )

    sources: dict[str, CombinedQuerySource] = {}
    for stage_hash, context in contexts_by_query_stage_hash.items():
        unit, stage = stages[stage_hash]
        opening = openings.get(stage_hash)
        primary_pair = c2_calls.get(stage_hash)
        c2 = c2_receipts.get((unit.unit_id, 1))
        if opening is None or primary_pair is None or c2 is None:
            raise CombinedBlockError("combined source lacks opening, C2 call, or empty inventory")
        call, itt = primary_pair
        if (
            stage.query_context_hash != context.content_hash
            or opening.opened_stage != stage
            or opening.query_access_event.query_context_hash != context.content_hash
            or opening.evidence_packet_hash != opening.evidence_packet_artifact.logical_content_hash
            or c2.prequery_stage_hash != unit.prequery_stage.staging_manifest_hash
            or itt.call_spec_hash != call.content_hash
            or itt.result.call_id != call.call_id
            or itt.result.condition is not ConditionName.C2_LLM_QUERY
            or itt.result.empty_prequery_inventory_hash != c2.inventory.content_hash
        ):
            raise CombinedBlockError("combined source C2/evidence/query lineage changed")
        ablation_lineage = []
        for condition in ACTIVE_ABLATION_CONDITIONS:
            receipt = ablation_receipts.get((unit.unit_id, condition))
            if (
                receipt is None
                or receipt.prequery_stage_hash != unit.prequery_stage.staging_manifest_hash
                or receipt.preparation.empty_inventory is None
            ):
                raise CombinedBlockError("combined source lacks a condition-matching ablation seal")
            inventory = receipt.preparation.empty_inventory
            ablation_lineage.append(
                AblationPrequeryLineage(
                    condition=condition,
                    preparation_hash=receipt.preparation.content_hash,
                    inventory_hash=inventory.content_hash,
                    receipt_hash=receipt.content_hash,
                    preparation_artifact_hash=receipt.preparation_artifact.artifact_hash,
                    inventory_artifact_hash=receipt.inventory_artifact.artifact_hash,
                    completed_at=receipt.completed_at,
                )
            )
        result = itt.result
        output_pointer = None
        attempt_hash = None
        if result.outcome is RunOutcome.SUCCEEDED:
            if result.artifact_receipt is None or result.artifact_receipt.output is None:
                raise CombinedBlockError("successful seed-1 C2 result lacks typed output CAS")
            output_pointer = _restricted(result.artifact_receipt.output)
            attempt_hash = result.artifact_receipt.output.logical_content_hash
        source = CombinedQuerySource(
            execution_id=held_out_execution.execution_id,
            held_out_call_manifest_hash=held_out_calls.content_hash,
            held_out_execution_manifest_hash=held_out_execution.content_hash,
            held_out_scorer_bridge_hash=scorer_bridge.content_hash,
            final_reviewed_seal_hash=held_out_execution.final_reviewed_seal_hash,
            ablation_prequery_registry_hash=ablation_registry_hash,
            unit_id=unit.unit_id,
            context_id=context.context_id,
            context=context,
            prequery_stage=unit.prequery_stage,
            query_stage=stage,
            prequery_barrier_hash=held_out_execution.prequery_barrier.content_hash,
            prequery_barrier_sealed_at=held_out_execution.prequery_barrier.sealed_at,
            query_access_event_hash=opening.query_access_event.content_hash,
            query_accessed_at=opening.query_access_event.accessed_at,
            packet_hash=opening.evidence_packet_hash,
            packet_artifact=_public_packet(opening.evidence_packet_artifact),
            c2_seed1_preparation_hash=c2.preparation.content_hash,
            c2_seed1_inventory_hash=c2.inventory.content_hash,
            c2_seed1_receipt_hash=c2.content_hash,
            c2_seed1_preparation_artifact_hash=c2.preparation_artifact.artifact_hash,
            c2_seed1_inventory_artifact_hash=c2.inventory_artifact.artifact_hash,
            c2_seed1_completed_at=c2.completed_at,
            ablation_prequery_lineage=tuple(ablation_lineage),
            primary_c2=PrimaryC2Reference(
                call_id=call.call_id,
                call_spec_hash=call.content_hash,
                itt_record_hash=itt.content_hash,
                outcome=result.outcome,
                output_artifact=output_pointer,
                condition_attempt_hash=attempt_hash,
            ),
        )
        if source.context_id in sources:
            raise CombinedBlockError("combined query context IDs must be unique")
        sources[source.context_id] = source
    if len(sources) != 36:
        raise CombinedBlockError("combined source compiler did not emit 36 contexts")
    return sources


__all__ = [
    "compile_combined_query_sources",
    "replay_combined_upstream_gate",
]
