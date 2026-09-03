"""Deterministic boundary validation for LLM-generated ontology artifacts.

The checks here reject unsupported or temporally inadmissible material.  They never
fill missing semantics, infer a causal claim, invent a qualification, or rewrite a
draft.  A rejected report can be shown to the model for the single bounded repair
defined in :mod:`story_projection_onto.llm`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from enum import StrEnum
from typing import Literal, Self

from pydantic import AwareDatetime, Field, field_validator, model_validator

from .contracts import (
    CONSTRUCTIVE_OPERATORS,
    ConditionName,
    EvidencePacket,
    EvidenceSnapshot,
    OntologyDraft,
    OntologyProjection,
    QueryContext,
)
from .llm import (
    ConstructionCapability,
    FixedSelectCapabilityError,
    FixedSelectOutputAudit,
    LLMCondition,
    RepairLineageMetadata,
    RuntimeManifest,
    SealedOntologyInventory,
    enforce_fixed_select_output,
)


class ValidationSeverity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


class ValidationCode(StrEnum):
    SNAPSHOT_HASH_MISMATCH = "snapshot_hash_mismatch"
    HORIZON_MISMATCH = "horizon_mismatch"
    CONTEXT_LINEAGE_MISMATCH = "context_lineage_mismatch"
    LINEAGE_HASH_MISMATCH = "lineage_hash_mismatch"
    DUPLICATE_EVIDENCE_ID = "duplicate_evidence_id"
    PACKET_RECORD_MISMATCH = "packet_record_mismatch"
    EVIDENCE_OUTSIDE_SNAPSHOT = "evidence_outside_snapshot"
    EVIDENCE_OUTSIDE_PACKET = "evidence_outside_packet"
    HORIZON_REJECTED_EVIDENCE = "horizon_rejected_evidence"
    DISCOURSE_HORIZON_LEAK = "discourse_horizon_leak"
    REVELATION_HORIZON_LEAK = "revelation_horizon_leak"
    MISSING_GROUNDING = "missing_grounding"
    UNSUPPORTED_GROUNDING = "unsupported_grounding"
    UNKNOWN_GROUNDING = "unknown_grounding"
    UNGROUNDED_FACTUAL_CLAUSE = "ungrounded_factual_clause"
    MISSING_PREQUERY_SEAL = "missing_prequery_seal"
    PREQUERY_SEAL_AFTER_REVEAL = "prequery_seal_after_reveal"
    PREQUERY_DECISION_AFTER_REVEAL = "prequery_decision_after_reveal"
    PREQUERY_DECISION_AFTER_SEAL = "prequery_decision_after_seal"
    MISSING_PREQUERY_INVENTORY = "missing_prequery_inventory"
    NONEMPTY_C2_PREQUERY_INVENTORY = "nonempty_c2_prequery_inventory"
    QUERY_REQUEST_BEFORE_REVEAL = "query_request_before_reveal"
    MISSING_CONSTRUCTION_DECISION = "missing_construction_decision"
    QUERY_DECISION_BEFORE_REVEAL = "query_decision_before_reveal"
    DECISION_AFTER_COMPLETION = "decision_after_completion"
    UNSEALED_LINEAGE_ID = "unsealed_lineage_id"
    CROSS_SEED_LINEAGE = "cross_seed_lineage"
    FIXED_SELECT_CAPABILITY = "fixed_select_capability"
    MULTIPLE_REPAIRS = "multiple_repairs"
    INVALID_REPAIR_PARENT = "invalid_repair_parent"


class ValidationDiagnostic(RuntimeManifest):
    """A fact-free diagnostic: identify the defect without suggesting semantics."""

    code: ValidationCode
    severity: ValidationSeverity = ValidationSeverity.ERROR
    path: str = Field(min_length=1)
    message: str = Field(min_length=1)
    related_ids: tuple[str, ...] = ()

    @field_validator("related_ids")
    @classmethod
    def related_ids_are_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("related diagnostic IDs must be unique")
        return value


class BoundaryValidationReport(RuntimeManifest):
    """Immutable output of one deterministic validation pass."""

    validation_status: Literal["accepted", "rejected", "invalid"]
    diagnostics: tuple[ValidationDiagnostic, ...] = ()

    @model_validator(mode="after")
    def status_matches_diagnostics(self) -> Self:
        has_error = any(
            diagnostic.severity is ValidationSeverity.ERROR for diagnostic in self.diagnostics
        )
        if self.validation_status == "accepted" and has_error:
            raise ValueError("an accepted report cannot contain error diagnostics")
        if self.validation_status == "rejected" and not has_error:
            raise ValueError("a rejected report must contain an error diagnostic")
        return self

    @property
    def accepted(self) -> bool:
        return self.validation_status == "accepted"

    def raise_for_errors(self) -> None:
        if not self.accepted:
            raise BoundaryValidationError(self)


class BoundaryValidationError(ValueError):
    def __init__(self, report: BoundaryValidationReport) -> None:
        self.report = report
        message = "; ".join(diagnostic.message for diagnostic in report.diagnostics)
        super().__init__(message or report.validation_status)


def _report(diagnostics: Iterable[ValidationDiagnostic]) -> BoundaryValidationReport:
    frozen = tuple(diagnostics)
    has_error = any(item.severity is ValidationSeverity.ERROR for item in frozen)
    status: Literal["accepted", "rejected"] = "rejected" if has_error else "accepted"
    return BoundaryValidationReport(validation_status=status, diagnostics=frozen)


class SpoilerHorizonBoundary(RuntimeManifest):
    """Registered visibility bound; deliberately contains no story-time field."""

    maximum_discourse_position: int | tuple[int, int, int]
    maximum_revelation_position: int | None = Field(default=None, ge=0)

    @property
    def effective_revelation_position(self) -> int | None:
        # Absence means revelation is not numerically bounded here.  Discourse and
        # revelation coordinates are intentionally not interchangeable.
        return self.maximum_revelation_position

    @field_validator("maximum_discourse_position")
    @classmethod
    def discourse_position_is_nonnegative(
        cls, value: int | tuple[int, int, int]
    ) -> int | tuple[int, int, int]:
        values = (value,) if isinstance(value, int) else value
        if any(item < 0 for item in values):
            raise ValueError("discourse position coordinates must be nonnegative")
        return value


class EvidenceBoundaryRecord(RuntimeManifest):
    """Only the coordinates needed to audit packet and horizon membership."""

    evidence_id: str = Field(min_length=1)
    discourse_position: int | tuple[int, int, int]
    revelation_position: int | None = Field(default=None, ge=0)

    @field_validator("discourse_position")
    @classmethod
    def discourse_position_is_nonnegative(
        cls, value: int | tuple[int, int, int]
    ) -> int | tuple[int, int, int]:
        values = (value,) if isinstance(value, int) else value
        if any(item < 0 for item in values):
            raise ValueError("discourse position coordinates must be nonnegative")
        return value


def _discourse_key(value: int | tuple[int, int, int]) -> tuple[int, int, int]:
    if isinstance(value, int):
        return (value, 0, 0)
    return value


class GroundingSupportStatus(StrEnum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


class EvidenceCitationAssessment(RuntimeManifest):
    """Independent validation assessment for one model-emitted citation."""

    evidence_id: str = Field(min_length=1)
    support_status: GroundingSupportStatus


class FactualClauseGrounding(RuntimeManifest):
    """Evidence mapping for one factual clause in a label or ``why_matters``."""

    clause_id: str = Field(min_length=1)
    evidence_ids: tuple[str, ...]

    @field_validator("evidence_ids")
    @classmethod
    def evidence_ids_are_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("clause evidence IDs must be unique")
        return value


class GroundedFactualRecord(RuntimeManifest):
    """Grounding material for an assertion, node description, or event description."""

    record_id: str = Field(min_length=1)
    citations: tuple[EvidenceCitationAssessment, ...]
    factual_clauses: tuple[FactualClauseGrounding, ...] = ()
    proposition_revelation_position: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def citation_ids_are_unique(self) -> Self:
        ids = tuple(citation.evidence_id for citation in self.citations)
        if len(ids) != len(set(ids)):
            raise ValueError("record citations must use unique evidence IDs")
        clause_ids = tuple(clause.clause_id for clause in self.factual_clauses)
        if len(clause_ids) != len(set(clause_ids)):
            raise ValueError("factual clause IDs must be unique within a record")
        return self


def validate_evidence_grounding(
    *,
    snapshot_eligible_evidence_ids: Iterable[str],
    packet_ordered_evidence_ids: Sequence[str],
    packet_records: Sequence[EvidenceBoundaryRecord],
    horizon_rejected_evidence_ids: Iterable[str],
    horizon: SpoilerHorizonBoundary,
    factual_records: Sequence[GroundedFactualRecord],
) -> BoundaryValidationReport:
    """Validate evidence equality, packet membership, horizon, and factual grounding.

    ``support_status`` is supplied by a separate deterministic or reviewed grounding
    assessment.  Mere citation is never silently treated as semantic support.
    """

    diagnostics: list[ValidationDiagnostic] = []
    snapshot_ids = set(snapshot_eligible_evidence_ids)
    ordered_packet_ids = tuple(packet_ordered_evidence_ids)
    packet_ids = set(ordered_packet_ids)
    record_ids = tuple(record.evidence_id for record in packet_records)
    rejected_ids = set(horizon_rejected_evidence_ids)

    if len(ordered_packet_ids) != len(packet_ids):
        diagnostics.append(
            ValidationDiagnostic(
                code=ValidationCode.DUPLICATE_EVIDENCE_ID,
                path="packet.ordered_evidence_ids",
                message="packet evidence IDs must be unique",
            )
        )
    if len(record_ids) != len(set(record_ids)):
        diagnostics.append(
            ValidationDiagnostic(
                code=ValidationCode.DUPLICATE_EVIDENCE_ID,
                path="packet.records",
                message="packet evidence records must have unique IDs",
            )
        )
    if tuple(record_ids) != ordered_packet_ids:
        diagnostics.append(
            ValidationDiagnostic(
                code=ValidationCode.PACKET_RECORD_MISMATCH,
                path="packet.records",
                message="packet record order must exactly match ordered_evidence_ids",
                related_ids=tuple(sorted(set(record_ids) ^ packet_ids)),
            )
        )

    outside_snapshot = packet_ids - snapshot_ids
    if outside_snapshot:
        diagnostics.append(
            ValidationDiagnostic(
                code=ValidationCode.EVIDENCE_OUTSIDE_SNAPSHOT,
                path="packet.ordered_evidence_ids",
                message="packet contains evidence outside the sealed snapshot",
                related_ids=tuple(sorted(outside_snapshot)),
            )
        )
    rejected_in_packet = packet_ids & rejected_ids
    if rejected_in_packet:
        diagnostics.append(
            ValidationDiagnostic(
                code=ValidationCode.HORIZON_REJECTED_EVIDENCE,
                path="packet.ordered_evidence_ids",
                message="packet includes evidence recorded as rejected by the horizon",
                related_ids=tuple(sorted(rejected_in_packet)),
            )
        )

    for index, evidence in enumerate(packet_records):
        if _discourse_key(evidence.discourse_position) > _discourse_key(
            horizon.maximum_discourse_position
        ):
            diagnostics.append(
                ValidationDiagnostic(
                    code=ValidationCode.DISCOURSE_HORIZON_LEAK,
                    path=f"packet.records.{index}.discourse_position",
                    message="evidence discourse position exceeds the spoiler horizon",
                    related_ids=(evidence.evidence_id,),
                )
            )
        if (
            evidence.revelation_position is not None
            and horizon.effective_revelation_position is not None
            and evidence.revelation_position > horizon.effective_revelation_position
        ):
            diagnostics.append(
                ValidationDiagnostic(
                    code=ValidationCode.REVELATION_HORIZON_LEAK,
                    path=f"packet.records.{index}.revelation_position",
                    message="evidence revelation position exceeds the registered horizon",
                    related_ids=(evidence.evidence_id,),
                )
            )

    for index, record in enumerate(factual_records):
        record_path = f"factual_records.{index}"
        if not record.citations:
            diagnostics.append(
                ValidationDiagnostic(
                    code=ValidationCode.MISSING_GROUNDING,
                    path=f"{record_path}.citations",
                    message="factual record has no evidence citation",
                    related_ids=(record.record_id,),
                )
            )
        cited_ids = {citation.evidence_id for citation in record.citations}
        outside_snapshot_citations = cited_ids - snapshot_ids
        if outside_snapshot_citations:
            diagnostics.append(
                ValidationDiagnostic(
                    code=ValidationCode.EVIDENCE_OUTSIDE_SNAPSHOT,
                    path=f"{record_path}.citations",
                    message="factual record cites evidence outside the sealed snapshot",
                    related_ids=tuple(sorted(outside_snapshot_citations)),
                )
            )
        outside_packet = cited_ids - packet_ids
        if outside_packet:
            diagnostics.append(
                ValidationDiagnostic(
                    code=ValidationCode.EVIDENCE_OUTSIDE_PACKET,
                    path=f"{record_path}.citations",
                    message="factual record cites evidence outside the frozen packet",
                    related_ids=tuple(sorted(outside_packet)),
                )
            )
        cited_rejected = cited_ids & rejected_ids
        if cited_rejected:
            diagnostics.append(
                ValidationDiagnostic(
                    code=ValidationCode.HORIZON_REJECTED_EVIDENCE,
                    path=f"{record_path}.citations",
                    message="factual record cites horizon-rejected evidence",
                    related_ids=tuple(sorted(cited_rejected)),
                )
            )
        unsupported = tuple(
            citation.evidence_id
            for citation in record.citations
            if citation.support_status is GroundingSupportStatus.UNSUPPORTED
        )
        if unsupported:
            diagnostics.append(
                ValidationDiagnostic(
                    code=ValidationCode.UNSUPPORTED_GROUNDING,
                    path=f"{record_path}.citations",
                    message="citation assessment does not support the factual record",
                    related_ids=unsupported,
                )
            )
        unknown = tuple(
            citation.evidence_id
            for citation in record.citations
            if citation.support_status is GroundingSupportStatus.UNKNOWN
        )
        if unknown:
            diagnostics.append(
                ValidationDiagnostic(
                    code=ValidationCode.UNKNOWN_GROUNDING,
                    path=f"{record_path}.citations",
                    message="citation support is unknown and cannot be accepted as grounded",
                    related_ids=unknown,
                )
            )

        for clause_index, clause in enumerate(record.factual_clauses):
            if not clause.evidence_ids or not set(clause.evidence_ids).issubset(cited_ids):
                diagnostics.append(
                    ValidationDiagnostic(
                        code=ValidationCode.UNGROUNDED_FACTUAL_CLAUSE,
                        path=f"{record_path}.factual_clauses.{clause_index}",
                        message="every factual clause must map to cited evidence",
                        related_ids=(record.record_id, clause.clause_id),
                    )
                )
        if (
            record.proposition_revelation_position is not None
            and horizon.effective_revelation_position is not None
            and record.proposition_revelation_position > horizon.effective_revelation_position
        ):
            diagnostics.append(
                ValidationDiagnostic(
                    code=ValidationCode.REVELATION_HORIZON_LEAK,
                    path=f"{record_path}.proposition_revelation_position",
                    message="asserted proposition is revealed after the registered horizon",
                    related_ids=(record.record_id,),
                )
            )

    return _report(diagnostics)


def validate_draft_evidence_grounding(
    *,
    snapshot: EvidenceSnapshot,
    packet: EvidencePacket,
    context: QueryContext,
    draft: OntologyDraft,
    support_assessments: Mapping[tuple[str, str], GroundingSupportStatus | str],
) -> BoundaryValidationReport:
    """Apply the evidence gate directly to the common ontology contracts.

    The caller must supply an independent support assessment for every cited
    ``(record_id, evidence_id)`` pair. Missing assessments become ``unknown`` rather
    than being optimistically inferred from citation presence.
    """

    diagnostics: list[ValidationDiagnostic] = []
    if packet.snapshot_hash != snapshot.content_hash:
        diagnostics.append(
            ValidationDiagnostic(
                code=ValidationCode.SNAPSHOT_HASH_MISMATCH,
                path="packet.snapshot_hash",
                message="evidence packet does not cite the supplied sealed snapshot",
            )
        )
    if context.spoiler_horizon != snapshot.horizon:
        diagnostics.append(
            ValidationDiagnostic(
                code=ValidationCode.HORIZON_MISMATCH,
                path="context.spoiler_horizon",
                message="query and sealed evidence snapshot use different spoiler horizons",
            )
        )

    def citations(
        record_id: str, evidence_ids: Sequence[str]
    ) -> tuple[EvidenceCitationAssessment, ...]:
        return tuple(
            EvidenceCitationAssessment(
                evidence_id=evidence_id,
                support_status=GroundingSupportStatus(
                    support_assessments.get(
                        (record_id, evidence_id), GroundingSupportStatus.UNKNOWN
                    )
                ),
            )
            for evidence_id in evidence_ids
        )

    records: list[GroundedFactualRecord] = []

    def add_record(
        record_id: str,
        evidence_ids: Sequence[str],
        *,
        clauses: Sequence[FactualClauseGrounding] = (),
        revelation_position: int | None = None,
    ) -> None:
        records.append(
            GroundedFactualRecord(
                record_id=record_id,
                citations=citations(record_id, evidence_ids),
                factual_clauses=tuple(clauses),
                proposition_revelation_position=revelation_position,
            )
        )

    for contextual_type in draft.local_schema.contextual_types:
        add_record(contextual_type.type_id, contextual_type.evidence_ids)
    for predicate in draft.local_schema.predicates:
        add_record(predicate.predicate_id, predicate.evidence_ids)
    for entity in draft.instance_graph.entities:
        add_record(entity.entity_id, entity.evidence_ids)
    for event in draft.instance_graph.events:
        add_record(event.event_id, event.evidence_ids)
    for proposition in draft.instance_graph.proposition_contents:
        add_record(proposition.proposition_content_id, proposition.evidence_ids)
    for assertion in draft.instance_graph.assertions:
        add_record(
            assertion.assertion_id,
            assertion.evidence_ids,
            clauses=(
                FactualClauseGrounding(
                    clause_id=f"{assertion.assertion_id}:why_matters",
                    evidence_ids=assertion.why_matters_evidence_ids,
                ),
            ),
            revelation_position=assertion.temporal_scope.revelation_position.revelation_order,
        )
    for decision in draft.decisions:
        add_record(decision.decision_id, decision.evidence_ids)

    max_revelation = context.spoiler_horizon.max_revelation_position
    report = validate_evidence_grounding(
        snapshot_eligible_evidence_ids=snapshot.eligible_evidence_ids,
        packet_ordered_evidence_ids=packet.ordered_evidence_ids,
        packet_records=tuple(
            EvidenceBoundaryRecord(
                evidence_id=evidence.evidence_id,
                discourse_position=evidence.discourse_position.ordering_key,
            )
            for evidence in packet.evidence
        ),
        horizon_rejected_evidence_ids=packet.horizon_rejections,
        horizon=SpoilerHorizonBoundary(
            maximum_discourse_position=(
                context.spoiler_horizon.max_discourse_position.ordering_key
            ),
            maximum_revelation_position=(
                max_revelation.revelation_order if max_revelation is not None else None
            ),
        ),
        factual_records=tuple(records),
    )
    return _report((*diagnostics, *report.diagnostics))


class ConstructionLineageAudit(RuntimeManifest):
    """Condition-neutral values needed to prove construction timing and ancestry."""

    condition: LLMCondition
    query_revealed_at: AwareDatetime
    request_created_at: AwareDatetime | None = None
    completed_at: AwareDatetime
    prequery_seal_completed_at: AwareDatetime | None = None
    prequery_inventory_recorded_at: AwareDatetime | None = None
    prequery_inventory_ids: tuple[str, ...] = ()
    construction_decision_timestamps: tuple[AwareDatetime, ...] = ()
    selection_decision_timestamps: tuple[AwareDatetime, ...] = ()
    sealed_semantic_ids: tuple[str, ...] = ()
    output_semantic_ids: tuple[str, ...] = ()
    seed_block: int = Field(ge=0)
    sealed_seed_block: int | None = Field(default=None, ge=0)

    @field_validator("prequery_inventory_ids", "sealed_semantic_ids", "output_semantic_ids")
    @classmethod
    def lineage_ids_are_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("lineage semantic IDs must be unique")
        return value

    @model_validator(mode="after")
    def completion_follows_request(self) -> Self:
        if self.request_created_at is not None and self.completed_at < self.request_created_at:
            raise ValueError("completed_at cannot precede request_created_at")
        return self


def validate_construction_lineage(
    audit: ConstructionLineageAudit,
) -> BoundaryValidationReport:
    """Validate pre-query sealing, empty C2 inventory, and post-reveal decisions."""

    diagnostics: list[ValidationDiagnostic] = []

    if audit.condition is LLMCondition.C1_LLM_PRE:
        if audit.prequery_seal_completed_at is None:
            diagnostics.append(
                ValidationDiagnostic(
                    code=ValidationCode.MISSING_PREQUERY_SEAL,
                    path="prequery_seal_completed_at",
                    message="C1 requires a construction seal completed before query reveal",
                )
            )
        elif audit.prequery_seal_completed_at >= audit.query_revealed_at:
            diagnostics.append(
                ValidationDiagnostic(
                    code=ValidationCode.PREQUERY_SEAL_AFTER_REVEAL,
                    path="prequery_seal_completed_at",
                    message="C1 construction seal is not strictly before query reveal",
                )
            )
        late_prequery_decisions = tuple(
            timestamp
            for timestamp in audit.construction_decision_timestamps
            if timestamp >= audit.query_revealed_at
        )
        if late_prequery_decisions:
            diagnostics.append(
                ValidationDiagnostic(
                    code=ValidationCode.PREQUERY_DECISION_AFTER_REVEAL,
                    path="construction_decision_timestamps",
                    message="C1 contains a construction decision at or after query reveal",
                )
            )
        if audit.prequery_seal_completed_at is not None and any(
            timestamp > audit.prequery_seal_completed_at
            for timestamp in audit.construction_decision_timestamps
        ):
            diagnostics.append(
                ValidationDiagnostic(
                    code=ValidationCode.PREQUERY_DECISION_AFTER_SEAL,
                    path="construction_decision_timestamps",
                    message="C1 construction decision follows its immutable seal",
                )
            )
        unsealed = set(audit.output_semantic_ids) - set(audit.sealed_semantic_ids)
        if unsealed:
            diagnostics.append(
                ValidationDiagnostic(
                    code=ValidationCode.UNSEALED_LINEAGE_ID,
                    path="output_semantic_ids",
                    message="C1 projection output contains an ID absent from its pre-query seal",
                    related_ids=tuple(sorted(unsealed)),
                )
            )

    elif audit.condition is LLMCondition.C2_LLM_QUERY:
        if audit.prequery_inventory_recorded_at is None:
            diagnostics.append(
                ValidationDiagnostic(
                    code=ValidationCode.MISSING_PREQUERY_INVENTORY,
                    path="prequery_inventory_recorded_at",
                    message="C2 requires a pre-query inventory audit",
                )
            )
        elif audit.prequery_inventory_recorded_at >= audit.query_revealed_at:
            diagnostics.append(
                ValidationDiagnostic(
                    code=ValidationCode.MISSING_PREQUERY_INVENTORY,
                    path="prequery_inventory_recorded_at",
                    message="C2 inventory was not recorded before query reveal",
                )
            )
        if audit.prequery_inventory_ids:
            diagnostics.append(
                ValidationDiagnostic(
                    code=ValidationCode.NONEMPTY_C2_PREQUERY_INVENTORY,
                    path="prequery_inventory_ids",
                    message="C2 pre-query inventory contains constructed ontology objects",
                    related_ids=audit.prequery_inventory_ids,
                )
            )
        if audit.prequery_seal_completed_at is not None or audit.sealed_semantic_ids:
            diagnostics.append(
                ValidationDiagnostic(
                    code=ValidationCode.NONEMPTY_C2_PREQUERY_INVENTORY,
                    path="sealed_semantic_ids",
                    message="C2 cannot inherit a prebuilt ontology or construction seal",
                    related_ids=audit.sealed_semantic_ids,
                )
            )
        _append_query_time_diagnostics(audit, diagnostics, construction_required=True)

    elif audit.condition is LLMCondition.A_FIXED_SELECT:
        if audit.prequery_seal_completed_at is None:
            diagnostics.append(
                ValidationDiagnostic(
                    code=ValidationCode.MISSING_PREQUERY_SEAL,
                    path="prequery_seal_completed_at",
                    message="A-FixedSelect requires the inherited C1 construction seal",
                )
            )
        elif audit.prequery_seal_completed_at >= audit.query_revealed_at:
            diagnostics.append(
                ValidationDiagnostic(
                    code=ValidationCode.PREQUERY_SEAL_AFTER_REVEAL,
                    path="prequery_seal_completed_at",
                    message="inherited C1 seal is not strictly before query reveal",
                )
            )
        if audit.construction_decision_timestamps:
            diagnostics.append(
                ValidationDiagnostic(
                    code=ValidationCode.FIXED_SELECT_CAPABILITY,
                    path="construction_decision_timestamps",
                    message="A-FixedSelect cannot contain construction decisions",
                )
            )
        _append_query_time_diagnostics(audit, diagnostics, construction_required=False)
        unsealed = set(audit.output_semantic_ids) - set(audit.sealed_semantic_ids)
        if unsealed:
            diagnostics.append(
                ValidationDiagnostic(
                    code=ValidationCode.UNSEALED_LINEAGE_ID,
                    path="output_semantic_ids",
                    message="A-FixedSelect output contains a novel semantic ID",
                    related_ids=tuple(sorted(unsealed)),
                )
            )
        if audit.sealed_seed_block is None or audit.seed_block != audit.sealed_seed_block:
            diagnostics.append(
                ValidationDiagnostic(
                    code=ValidationCode.CROSS_SEED_LINEAGE,
                    path="seed_block",
                    message="A-FixedSelect must use the same seed block as sealed C1",
                )
            )

    return _report(diagnostics)


def _projection_semantic_ids(projection: OntologyProjection) -> tuple[str, ...]:
    schema = projection.local_schema
    graph = projection.instance_graph
    return tuple(
        dict.fromkeys(
            (
                schema.schema_id,
                *(item.type_id for item in schema.contextual_types),
                *(item.predicate_id for item in schema.predicates),
                *(item.entity_id for item in graph.entities),
                *(item.event_id for item in graph.events),
                *(item.proposition_content_id for item in graph.proposition_contents),
                *(item.assertion_id for item in graph.assertions),
            )
        )
    )


def validate_projection_lineage(
    projection: OntologyProjection,
    context: QueryContext,
    *,
    request_created_at: AwareDatetime | None,
    seed_block: int,
    sealed_seed_block: int | None = None,
) -> BoundaryValidationReport:
    """Adapt a full projection's seals/certificates to the timing-lineage gate."""

    if projection.condition not in {
        ConditionName.C1_LLM_PRE,
        ConditionName.C2_LLM_QUERY,
        ConditionName.A_FIXED_SELECT,
    }:
        raise ValueError("GPU lineage adapter accepts only C1, C2, and A-FixedSelect")

    diagnostics: list[ValidationDiagnostic] = []
    if projection.context_hash != context.content_hash:
        diagnostics.append(
            ValidationDiagnostic(
                code=ValidationCode.CONTEXT_LINEAGE_MISMATCH,
                path="projection.context_hash",
                message="projection does not cite the supplied query context",
            )
        )

    seal = projection.construction_seal
    certificate = projection.construction_certificate
    inventory = projection.pre_query_inventory
    if certificate is not None and certificate.query_context_hash != context.content_hash:
        diagnostics.append(
            ValidationDiagnostic(
                code=ValidationCode.CONTEXT_LINEAGE_MISMATCH,
                path="construction_certificate.query_context_hash",
                message="construction certificate cites a different query context",
            )
        )

    lineage_hashes = (
        ("construction_seal.snapshot_hash", seal.snapshot_hash) if seal is not None else None,
        ("pre_query_inventory.snapshot_hash", inventory.snapshot_hash)
        if inventory is not None
        else None,
        ("construction_certificate.snapshot_hash", certificate.snapshot_hash)
        if certificate is not None
        else None,
    )
    for lineage_item in lineage_hashes:
        if lineage_item is not None and lineage_item[1] != projection.snapshot_hash:
            diagnostics.append(
                ValidationDiagnostic(
                    code=ValidationCode.LINEAGE_HASH_MISMATCH,
                    path=lineage_item[0],
                    message="lineage snapshot hash differs from the projection snapshot hash",
                )
            )
    if certificate is not None and certificate.packet_hash != projection.packet_hash:
        diagnostics.append(
            ValidationDiagnostic(
                code=ValidationCode.LINEAGE_HASH_MISMATCH,
                path="construction_certificate.packet_hash",
                message="certificate packet hash differs from the projection packet hash",
            )
        )
    if certificate is not None and certificate.query_revealed_at != context.revealed_at:
        diagnostics.append(
            ValidationDiagnostic(
                code=ValidationCode.CONTEXT_LINEAGE_MISMATCH,
                path="construction_certificate.query_revealed_at",
                message="certificate and context query-reveal timestamps differ",
            )
        )

    construction_times = tuple(
        decision.decided_at
        for decision in projection.decisions
        if (
            projection.condition is not ConditionName.C1_LLM_PRE
            or decision.operator in CONSTRUCTIVE_OPERATORS
        )
    )
    selection_times = tuple(
        decision.decided_at
        for decision in projection.decisions
        if decision.operator not in CONSTRUCTIVE_OPERATORS
    )
    prequery_inventory_ids: tuple[str, ...] = ()
    if inventory is not None:
        prequery_inventory_ids = tuple(
            item
            for values in (
                inventory.ontology_refs,
                inventory.local_schema_ids,
                inventory.entity_ids,
                inventory.event_ids,
                inventory.assertion_ids,
                inventory.finalized_predicate_ids,
            )
            for item in values
        )

    completed_at = (
        certificate.completed_at
        if certificate is not None
        else seal.sealed_at
        if seal is not None
        else context.revealed_at
    )
    audit = ConstructionLineageAudit(
        condition=projection.condition,
        query_revealed_at=context.revealed_at,
        request_created_at=request_created_at,
        completed_at=completed_at,
        prequery_seal_completed_at=seal.sealed_at if seal is not None else None,
        prequery_inventory_recorded_at=(inventory.recorded_at if inventory is not None else None),
        prequery_inventory_ids=prequery_inventory_ids,
        construction_decision_timestamps=construction_times,
        selection_decision_timestamps=selection_times,
        sealed_semantic_ids=seal.sealed_object_ids if seal is not None else (),
        output_semantic_ids=_projection_semantic_ids(projection),
        seed_block=seed_block,
        sealed_seed_block=sealed_seed_block,
    )
    report = validate_construction_lineage(audit)
    return _report((*diagnostics, *report.diagnostics))


def _append_query_time_diagnostics(
    audit: ConstructionLineageAudit,
    diagnostics: list[ValidationDiagnostic],
    *,
    construction_required: bool,
) -> None:
    if audit.request_created_at is None or audit.request_created_at < audit.query_revealed_at:
        diagnostics.append(
            ValidationDiagnostic(
                code=ValidationCode.QUERY_REQUEST_BEFORE_REVEAL,
                path="request_created_at",
                message="query-time request is absent or precedes query reveal",
            )
        )
    timestamps = (
        audit.construction_decision_timestamps
        if construction_required
        else audit.selection_decision_timestamps
    )
    path = (
        "construction_decision_timestamps"
        if construction_required
        else "selection_decision_timestamps"
    )
    if construction_required and not timestamps:
        diagnostics.append(
            ValidationDiagnostic(
                code=ValidationCode.MISSING_CONSTRUCTION_DECISION,
                path=path,
                message="C2 certificate must contain a post-query construction decision",
            )
        )
    for timestamp in timestamps:
        if timestamp <= audit.query_revealed_at:
            diagnostics.append(
                ValidationDiagnostic(
                    code=ValidationCode.QUERY_DECISION_BEFORE_REVEAL,
                    path=path,
                    message="query-time decision does not strictly follow query reveal",
                )
            )
            break
        if timestamp > audit.completed_at:
            diagnostics.append(
                ValidationDiagnostic(
                    code=ValidationCode.DECISION_AFTER_COMPLETION,
                    path=path,
                    message="decision timestamp follows construction completion",
                )
            )
            break


def validate_fixed_select(
    output: FixedSelectOutputAudit,
    sealed: SealedOntologyInventory,
) -> BoundaryValidationReport:
    """Return a standard validation report for the mechanical capability check."""

    try:
        enforce_fixed_select_output(output, sealed)
    except FixedSelectCapabilityError as error:
        diagnostics = tuple(
            ValidationDiagnostic(
                code=ValidationCode.FIXED_SELECT_CAPABILITY,
                path="fixed_select_output",
                message=message,
            )
            for message in error.violations
        )
        return _report(diagnostics)
    return _report(())


def validate_single_repair_lineage(
    repairs: Sequence[RepairLineageMetadata],
    *,
    known_attempt_ids: Iterable[str] = (),
) -> BoundaryValidationReport:
    """Reject a second repair or a repair whose direct base attempt is unknown."""

    diagnostics: list[ValidationDiagnostic] = []
    if len(repairs) > 1:
        diagnostics.append(
            ValidationDiagnostic(
                code=ValidationCode.MULTIPLE_REPAIRS,
                path="repairs",
                message="at most one repair is permitted for an artifact",
                related_ids=tuple(repair.repair_attempt_id for repair in repairs),
            )
        )
    known = set(known_attempt_ids)
    for index, repair in enumerate(repairs):
        if known and repair.base_attempt_id not in known:
            diagnostics.append(
                ValidationDiagnostic(
                    code=ValidationCode.INVALID_REPAIR_PARENT,
                    path=f"repairs.{index}.base_attempt_id",
                    message="repair does not reference a known base attempt",
                    related_ids=(repair.base_attempt_id,),
                )
            )
    return _report(diagnostics)


def fixed_select_forbidden_capabilities() -> frozenset[ConstructionCapability]:
    """Expose the fail-closed set for grammar/capability regression tests."""

    from .llm import CONSTRUCTIVE_CAPABILITIES

    return CONSTRUCTIVE_CAPABILITIES
