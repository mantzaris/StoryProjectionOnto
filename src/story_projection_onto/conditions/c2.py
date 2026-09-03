"""Active post-query ``C2 LLMQuery`` request and completion boundary.

The functions are deliberately transport-free.  They create an empty pre-query
inventory, assemble a model request only after reveal, then certify a parsed GPU
draft without adding missing semantics on CPU.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import datetime

from story_projection_onto.conditions.base import (
    ConditionAttemptRecord,
    ConditionIntegrityError,
    ConditionPreparation,
    ProduceInputs,
)
from story_projection_onto.contracts import (
    ConditionName,
    ConstructionCapabilities,
    ConstructionCertificate,
    ConstructionRequest,
    EvidenceSnapshot,
    OntologyDraft,
    OntologyProjection,
    PreQueryInventory,
    RunOutcome,
    RuntimeIdentifiers,
    ValidationRecord,
    to_model_visible_packet,
    to_model_visible_query,
)


def _identifier(prefix: str, *parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(payload).hexdigest()[:20]}"


def prepare_empty_c2_inventory(
    snapshot: EvidenceSnapshot,
    *,
    recorded_at: datetime,
    condition: ConditionName = ConditionName.C2_LLM_QUERY,
) -> ConditionPreparation:
    """Prove that active conditions own no ontology before a query is revealed."""

    if condition not in {
        ConditionName.C2_LLM_QUERY,
        ConditionName.A_NO_CONTEXT,
        ConditionName.A_NO_TEMPORAL_EPISTEMIC,
        ConditionName.A_NO_RARE_GUARD,
    }:
        raise ConditionIntegrityError("empty inventory is only valid for active construction")
    if recorded_at < snapshot.sealed_at:
        raise ConditionIntegrityError("C2 inventory cannot predate the sealed evidence snapshot")
    inventory = PreQueryInventory(
        inventory_id=_identifier("c2-empty-inventory", condition.value, snapshot.content_hash),
        condition=condition,
        snapshot_hash=snapshot.content_hash,
        recorded_at=recorded_at,
    )
    return ConditionPreparation(
        preparation_id=_identifier("c2-preparation", inventory.content_hash),
        condition=condition,
        snapshot_hash=snapshot.content_hash,
        completed_at=recorded_at,
        empty_inventory=inventory,
    )


def build_c2_construction_request(
    inputs: ProduceInputs,
    *,
    runtime: RuntimeIdentifiers,
    requested_at: datetime,
) -> ConstructionRequest:
    """Create the active request from packet/context only, never a hidden graph."""

    condition = inputs.run_config.condition
    if condition not in {
        ConditionName.C2_LLM_QUERY,
        ConditionName.A_NO_CONTEXT,
        ConditionName.A_NO_TEMPORAL_EPISTEMIC,
        ConditionName.A_NO_RARE_GUARD,
    }:
        raise ConditionIntegrityError("C2 request helper received a non-active condition")
    if requested_at < inputs.context.revealed_at:
        raise ConditionIntegrityError("C2 construction request predates query reveal")
    if inputs.preparation.empty_inventory is None:
        raise ConditionIntegrityError("C2 request lacks its empty pre-query inventory")
    return ConstructionRequest(
        request_id=_identifier(
            "c2-request", condition.value, inputs.context.content_hash, inputs.run_config.seed_block
        ),
        condition=condition,
        snapshot_hash=inputs.snapshot.content_hash,
        packet=to_model_visible_packet(inputs.packet),
        context=to_model_visible_query(inputs.context),
        upper_ontology=inputs.upper_ontology,
        budgets=inputs.context.budgets,
        capabilities=ConstructionCapabilities.active_construction(),
        runtime=runtime,
        requested_at=requested_at,
        model_visible_revisions=inputs.revisions,
    )


def finalize_c2_draft(
    inputs: ProduceInputs,
    *,
    request: ConstructionRequest,
    draft: OntologyDraft,
    validation_records: Sequence[ValidationRecord],
    completed_at: datetime,
    raw_output_hash: str,
) -> ConditionAttemptRecord:
    """Certify the LLM's semantics verbatim after validation; CPU adds no facts."""

    condition = inputs.run_config.condition
    if request.condition is not condition:
        raise ConditionIntegrityError("C2 request and run condition differ")
    if request.requested_at < inputs.context.revealed_at or completed_at < request.requested_at:
        raise ConditionIntegrityError("C2 request/completion timing violates query reveal")
    if request.fixed_ontology is not None:
        raise ConditionIntegrityError("C2 cannot receive a prebuilt ontology")
    draft.budget_accounting.validate_against(inputs.context.budgets)
    inventory = inputs.preparation.empty_inventory
    if inventory is None:
        raise ConditionIntegrityError("C2 completion lacks empty-inventory lineage")
    certificate = ConstructionCertificate(
        certificate_id=_identifier("c2-certificate", request.content_hash, raw_output_hash),
        condition=condition,
        snapshot_hash=inputs.snapshot.content_hash,
        packet_hash=inputs.packet.content_hash,
        query_context_hash=inputs.context.content_hash,
        query_revealed_at=inputs.context.revealed_at,
        completed_at=completed_at,
        decisions=draft.decisions,
        pre_query_inventory_hash=inventory.content_hash,
    )
    projection_id = _identifier("c2-projection", certificate.content_hash)
    projection = OntologyProjection(
        projection_id=projection_id,
        condition=condition,
        snapshot_hash=inputs.snapshot.content_hash,
        packet_hash=inputs.packet.content_hash,
        context_hash=inputs.context.content_hash,
        upper_ontology=inputs.upper_ontology,
        local_schema=draft.local_schema,
        instance_graph=draft.instance_graph,
        decisions=draft.decisions,
        omissions=draft.omissions,
        validation_records=tuple(validation_records),
        budget_accounting=draft.budget_accounting,
        budgets=inputs.context.budgets,
        pre_query_inventory=inventory,
        construction_certificate=certificate,
        run_id=_identifier("c2-run", inputs.run_config.content_hash, projection_id),
        release_class=inputs.packet.release_class,
    )
    return ConditionAttemptRecord(
        attempt_id=_identifier("c2-attempt", request.content_hash, raw_output_hash),
        condition=condition,
        unit_id=inputs.context.context_id,
        seed_block=inputs.run_config.seed_block,
        outcome=RunOutcome.SUCCEEDED,
        projection=projection,
        raw_output_hash=raw_output_hash,
        release_class=inputs.packet.release_class,
    )


def failed_c2_attempt(
    inputs: ProduceInputs,
    *,
    outcome: RunOutcome,
    failure_code: str,
    raw_output_hash: str | None = None,
) -> ConditionAttemptRecord:
    """Retain a failed/invalid/time-out attempt instead of dropping the unit."""

    if outcome not in {
        RunOutcome.INVALID,
        RunOutcome.FAILED,
        RunOutcome.TIMED_OUT,
        RunOutcome.INTERRUPTED,
    }:
        raise ValueError("failed C2 attempt requires a terminal nonsuccess outcome")
    return ConditionAttemptRecord(
        attempt_id=_identifier(
            "c2-attempt-failed",
            inputs.context.content_hash,
            inputs.run_config.seed_block,
            failure_code,
        ),
        condition=inputs.run_config.condition,
        unit_id=inputs.context.context_id,
        seed_block=inputs.run_config.seed_block,
        outcome=outcome,
        raw_output_hash=raw_output_hash,
        failure_code=failure_code,
        release_class=inputs.packet.release_class,
    )


class LLMQueryCondition:
    """Named condition facade; GPU transport calls the pure functions above."""

    condition = ConditionName.C2_LLM_QUERY

    def prepare(
        self,
        snapshot: EvidenceSnapshot,
        *,
        recorded_at: datetime,
    ) -> ConditionPreparation:
        return prepare_empty_c2_inventory(snapshot, recorded_at=recorded_at)
