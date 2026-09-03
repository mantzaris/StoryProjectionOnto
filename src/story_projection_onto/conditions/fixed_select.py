"""Mechanistic ``A-FixedSelect`` condition with construction mechanically forbidden."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import datetime

from story_projection_onto.conditions.base import (
    ConditionAttemptRecord,
    ConditionIntegrityError,
    ConditionPreparation,
    FixedSelectionPreparation,
    ProduceInputs,
)
from story_projection_onto.contracts import (
    ConditionName,
    ConstructionCapabilities,
    ConstructionCertificate,
    ConstructionRequest,
    OntologyDraft,
    OntologyProjection,
    RunOutcome,
    RuntimeIdentifiers,
    ValidationRecord,
    to_model_visible_packet,
    to_model_visible_query,
)
from story_projection_onto.llm import (
    enforce_fixed_select_draft,
    sealed_inventory_from_fixed_ontology,
)


def _identifier(prefix: str, *parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(payload).hexdigest()[:20]}"


def prepare_fixed_selection(
    c1_preparation: ConditionPreparation,
    *,
    seed_block: int,
    prepared_at: datetime,
) -> ConditionPreparation:
    """Bind FixedSelect to one complete, already sealed same-seed C1 ontology."""

    source = c1_preparation.sealed_preontology
    if c1_preparation.condition is not ConditionName.C1_LLM_PRE or source is None:
        raise ConditionIntegrityError("A-FixedSelect preparation requires C1 preconstruction")
    fixed = FixedSelectionPreparation(
        preparation_id=_identifier("fixed-source", source.content_hash, seed_block),
        source_c1_preontology=source,
        seed_block=seed_block,
        prepared_at=prepared_at,
    )
    return ConditionPreparation(
        preparation_id=_identifier("fixed-preparation", fixed.content_hash),
        condition=ConditionName.A_FIXED_SELECT,
        snapshot_hash=source.snapshot_hash,
        completed_at=prepared_at,
        fixed_selection=fixed,
    )


def build_fixed_select_request(
    inputs: ProduceInputs,
    *,
    runtime: RuntimeIdentifiers,
    requested_at: datetime,
) -> ConstructionRequest:
    """Pack the complete inherited graph under the selection-only capability record."""

    if inputs.run_config.condition is not ConditionName.A_FIXED_SELECT:
        raise ConditionIntegrityError("fixed request helper received another condition")
    if requested_at < inputs.context.revealed_at:
        raise ConditionIntegrityError("FixedSelect request predates query reveal")
    preparation = inputs.preparation.fixed_selection
    if preparation is None:
        raise ConditionIntegrityError("FixedSelect lacks inherited C1 preparation")
    source = preparation.source_c1_preontology
    if source.seed_block != inputs.run_config.seed_block:
        raise ConditionIntegrityError("FixedSelect and inherited C1 use different seeds")
    return ConstructionRequest(
        request_id=_identifier(
            "fixed-request", source.content_hash, inputs.context.content_hash, source.seed_block
        ),
        condition=ConditionName.A_FIXED_SELECT,
        snapshot_hash=inputs.snapshot.content_hash,
        packet=to_model_visible_packet(inputs.packet),
        context=to_model_visible_query(inputs.context),
        upper_ontology=inputs.upper_ontology,
        budgets=inputs.context.budgets,
        capabilities=ConstructionCapabilities.fixed_selection(),
        runtime=runtime,
        requested_at=requested_at,
        fixed_ontology=source.as_fixed_ontology(),
        model_visible_revisions=inputs.revisions,
    )


def finalize_fixed_select_draft(
    inputs: ProduceInputs,
    *,
    request: ConstructionRequest,
    draft: OntologyDraft,
    validation_records: Sequence[ValidationRecord],
    completed_at: datetime,
    raw_output_hash: str,
) -> ConditionAttemptRecord:
    """Reject any new/mutated semantics before constructing a projection record."""

    preparation = inputs.preparation.fixed_selection
    if preparation is None or request.fixed_ontology is None:
        raise ConditionIntegrityError("FixedSelect completion lacks its complete C1 graph")
    if completed_at < request.requested_at:
        raise ConditionIntegrityError("FixedSelect completion predates its request")
    source = preparation.source_c1_preontology
    inventory = sealed_inventory_from_fixed_ontology(
        request.fixed_ontology,
        seed_block=preparation.seed_block,
        source_draft=source.draft,
    )
    enforce_fixed_select_draft(
        draft,
        sealed=inventory,
        seed_block=inputs.run_config.seed_block or 0,
    )
    draft.budget_accounting.validate_against(inputs.context.budgets)
    certificate = ConstructionCertificate(
        certificate_id=_identifier("fixed-certificate", request.content_hash, raw_output_hash),
        condition=ConditionName.A_FIXED_SELECT,
        snapshot_hash=inputs.snapshot.content_hash,
        packet_hash=inputs.packet.content_hash,
        query_context_hash=inputs.context.content_hash,
        query_revealed_at=inputs.context.revealed_at,
        completed_at=completed_at,
        decisions=draft.decisions,
        inherited_construction_seal_hash=source.construction_seal.content_hash,
    )
    projection_id = _identifier("fixed-projection", certificate.content_hash)
    projection = OntologyProjection(
        projection_id=projection_id,
        condition=ConditionName.A_FIXED_SELECT,
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
        construction_seal=source.construction_seal,
        construction_certificate=certificate,
        run_id=_identifier("fixed-run", inputs.run_config.content_hash, projection_id),
        release_class=inputs.packet.release_class,
    )
    return ConditionAttemptRecord(
        attempt_id=_identifier("fixed-attempt", request.content_hash, raw_output_hash),
        condition=ConditionName.A_FIXED_SELECT,
        unit_id=inputs.context.context_id,
        seed_block=inputs.run_config.seed_block,
        outcome=RunOutcome.SUCCEEDED,
        projection=projection,
        raw_output_hash=raw_output_hash,
        release_class=inputs.packet.release_class,
    )


class FixedSelectCondition:
    """Named facade for selection-only request/finalization orchestration."""

    condition = ConditionName.A_FIXED_SELECT

    def prepare(
        self,
        c1_preparation: ConditionPreparation,
        *,
        seed_block: int,
        prepared_at: datetime,
    ) -> ConditionPreparation:
        return prepare_fixed_selection(
            c1_preparation,
            seed_block=seed_block,
            prepared_at=prepared_at,
        )
