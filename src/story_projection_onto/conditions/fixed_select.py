"""Mechanistic ``A-FixedSelect`` condition with construction mechanically forbidden."""

from __future__ import annotations

import hashlib
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
    OntologyProjection,
    RunOutcome,
    RuntimeIdentifiers,
    ValidatedGeneration,
    canonical_sha256,
    to_model_visible_packet,
    to_model_visible_query,
)
from story_projection_onto.llm import (
    enforce_fixed_select_draft,
    sealed_inventory_from_fixed_ontology,
)
from story_projection_onto.validate import validate_draft_structure, validate_projection_lineage


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
    if requested_at < inputs.query_processing_started_at:
        raise ConditionIntegrityError("FixedSelect request predates query processing")
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
    generation: ValidatedGeneration,
) -> ConditionAttemptRecord:
    """Reject any new/mutated semantics before constructing a projection record."""

    preparation = inputs.preparation.fixed_selection
    if preparation is None or request.fixed_ontology is None:
        raise ConditionIntegrityError("FixedSelect completion lacks its complete C1 graph")
    if generation.condition is not ConditionName.A_FIXED_SELECT:
        raise ConditionIntegrityError("FixedSelect finalizer received another condition")
    if generation.request_hash != request.content_hash:
        raise ConditionIntegrityError("FixedSelect generation cites a different request")
    if generation.query_access_event_hash != inputs.query_access.content_hash:
        raise ConditionIntegrityError("FixedSelect generation cites a different query access")
    if generation.prequery_barrier_hash != inputs.prequery_barrier.content_hash:
        raise ConditionIntegrityError("FixedSelect generation cites a different prequery barrier")
    if generation.stage_manifest_hash != inputs.query_access.stage_manifest_hash:
        raise ConditionIntegrityError("FixedSelect generation cites a different query stage")
    config = inputs.run_config
    runtime = request.runtime
    runtime_bindings = (
        (generation.model_stack_hash, config.model_stack_hash, "model stack"),
        (generation.decoding_manifest_hash, config.decoding_manifest_hash, "decoding"),
        (generation.seed_manifest_hash, config.seed_manifest_hash, "seed manifest"),
        (generation.seed, config.resolved_seed, "resolved seed"),
        (generation.prompt_hash, config.prompt_hash, "prompt"),
        (generation.output_schema_hash, config.output_schema_hash, "output schema"),
        (
            generation.capability_manifest_hash,
            config.capability_manifest_hash,
            "capability manifest",
        ),
        (generation.validator_hash, config.validator_hash, "validator"),
        (runtime.prompt_hash, config.prompt_hash, "request prompt"),
        (runtime.output_schema_hash, config.output_schema_hash, "request output schema"),
        (runtime.decoding_config_hash, config.decoding_manifest_hash, "request decoding"),
    )
    for observed, expected, label in runtime_bindings:
        if observed != expected:
            raise ConditionIntegrityError(
                f"FixedSelect {label} binding differs from run configuration"
            )
    if request.snapshot_hash != inputs.snapshot.content_hash:
        raise ConditionIntegrityError("FixedSelect request cites a different snapshot")
    if request.packet != to_model_visible_packet(inputs.packet):
        raise ConditionIntegrityError("FixedSelect request packet differs from ProduceInputs")
    if request.context != to_model_visible_query(inputs.context):
        raise ConditionIntegrityError("FixedSelect request context differs from ProduceInputs")
    if request.upper_ontology != inputs.upper_ontology:
        raise ConditionIntegrityError("FixedSelect upper ontology differs from ProduceInputs")
    if (
        request.budgets != inputs.context.budgets
        or request.model_visible_revisions != inputs.revisions
    ):
        raise ConditionIntegrityError("FixedSelect budgets or revisions differ from ProduceInputs")
    if request.requested_at < inputs.query_processing_started_at:
        raise ConditionIntegrityError("FixedSelect request predates query processing")
    if generation.generation_started_at < request.requested_at:
        raise ConditionIntegrityError("FixedSelect completion predates its request")
    source = preparation.source_c1_preontology
    if request.fixed_ontology != source.as_fixed_ontology():
        raise ConditionIntegrityError("FixedSelect request does not contain the exact C1 seal")
    draft = generation.draft
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
    structure = validate_draft_structure(
        draft=draft,
        upper_ontology=request.upper_ontology,
        evidence=request.packet.evidence,
        budgets=request.budgets,
        capabilities=request.capabilities,
    )
    if not structure.accepted:
        raise ConditionIntegrityError("FixedSelect draft failed deterministic boundary validation")
    if structure.content_hash not in generation.validator_report_hashes:
        raise ConditionIntegrityError("FixedSelect generation does not bind its boundary report")
    certificate = ConstructionCertificate(
        certificate_id=_identifier(
            "fixed-certificate", request.content_hash, generation.raw_output_artifact_hash
        ),
        condition=ConditionName.A_FIXED_SELECT,
        snapshot_hash=inputs.snapshot.content_hash,
        packet_hash=inputs.packet.content_hash,
        query_context_hash=inputs.context.content_hash,
        query_access_event_hash=inputs.query_access.content_hash,
        stage_manifest_hash=generation.stage_manifest_hash,
        prequery_barrier_hash=inputs.prequery_barrier.content_hash,
        generation_lineage_hash=generation.content_hash,
        raw_output_artifact_hash=generation.raw_output_artifact_hash,
        normalized_draft_hash=generation.normalized_draft_hash,
        validation_bundle_hash=canonical_sha256(generation.validation_records),
        query_revealed_at=inputs.query_access.accessed_at,
        completed_at=generation.validated_at,
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
        query_access_event_hash=inputs.query_access.content_hash,
        generation_lineage_hash=generation.content_hash,
        raw_output_artifact_hash=generation.raw_output_artifact_hash,
        normalized_draft_hash=generation.normalized_draft_hash,
        validation_bundle_hash=canonical_sha256(generation.validation_records),
        upper_ontology=inputs.upper_ontology,
        local_schema=draft.local_schema,
        instance_graph=draft.instance_graph,
        decisions=draft.decisions,
        omissions=draft.omissions,
        validation_records=generation.validation_records,
        budget_accounting=draft.budget_accounting,
        budgets=inputs.context.budgets,
        construction_seal=source.construction_seal,
        construction_certificate=certificate,
        run_id=_identifier("fixed-run", inputs.run_config.content_hash, projection_id),
        release_class=inputs.packet.release_class,
    )
    lineage = validate_projection_lineage(
        projection,
        inputs.context,
        query_access=inputs.query_access,
        packet_materialization=inputs.packet_materialization,
        request_created_at=request.requested_at,
        seed_block=inputs.run_config.seed_block or 0,
        sealed_seed_block=source.seed_block,
    )
    if not lineage.accepted:
        raise ConditionIntegrityError("A-FixedSelect projection failed timing/lineage validation")
    return ConditionAttemptRecord(
        attempt_id=_identifier(
            "fixed-attempt", request.content_hash, generation.raw_output_artifact_hash
        ),
        condition=ConditionName.A_FIXED_SELECT,
        unit_id=inputs.context.context_id,
        seed_block=inputs.run_config.seed_block,
        outcome=RunOutcome.SUCCEEDED,
        projection=projection,
        raw_output_hash=generation.raw_output_artifact_hash,
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

    def produce(
        self,
        inputs: ProduceInputs,
        *,
        request: ConstructionRequest,
        generation: ValidatedGeneration,
    ) -> ConditionAttemptRecord:
        return finalize_fixed_select_draft(inputs, request=request, generation=generation)
