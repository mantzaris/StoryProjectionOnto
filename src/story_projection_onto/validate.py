"""Deterministic boundary validation for LLM-generated ontology artifacts.

Structural checks here reject malformed references, out-of-bound citations, invalid
temporal shapes, and capability violations.  Semantic evidence support is accepted
only when a separate scorer or reviewer supplies an assessment; citation presence is
never treated as support by itself.  These validators never fill missing semantics,
infer a causal claim, invent a qualification, or rewrite a draft.  A rejected
structural report can be shown to the model for the single bounded repair defined in
:mod:`story_projection_onto.llm`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, Self

from pydantic import AwareDatetime, Field, field_validator, model_validator

from .contracts import (
    ACTIVE_QUERY_CONSTRUCTION_CONDITIONS,
    CONSTRUCTIVE_OPERATORS,
    AllenRelation,
    ConditionName,
    ConstructionCapabilities,
    ConstructionOperator,
    DiscoursePosition,
    EvidencePacket,
    EvidenceRecord,
    EvidenceSnapshot,
    HolderRelativeTime,
    InstanceGraph,
    ModelVisibleEvidenceInput,
    ModelVisibleEvidenceRecord,
    NarrativeCommitment,
    OntologyDraft,
    OntologyProjection,
    OutputBudgets,
    PacketMaterializationEvent,
    QueryAccessEvent,
    QueryContext,
    RevelationPosition,
    RoleBinding,
    SpoilerHorizon,
    StoryTime,
    TemporalKind,
    UpperOntology,
    ValidityTime,
    to_model_visible_query,
)
from .display_selection import DisplaySelectionError, compile_registered_display_selection
from .llm import (
    ABLATION_QUALIFICATION_REASON,
    ConstructionCapability,
    FixedSelectCapabilityError,
    FixedSelectOutputAudit,
    LLMCondition,
    RepairLineageMetadata,
    RuntimeManifest,
    SealedOntologyInventory,
    enforce_fixed_select_output,
)
from .temporal import (
    TemporalDiagnosticCode,
    TemporalValidationResult,
    validate_assertion_temporality,
    validate_temporal_extent,
    validate_temporal_scope,
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
    ABLATION_QUALIFICATION_PRESENT = "ablation_qualification_present"
    MULTIPLE_REPAIRS = "multiple_repairs"
    INVALID_REPAIR_PARENT = "invalid_repair_parent"
    BUDGET_ACCOUNTING_MISMATCH = "budget_accounting_mismatch"
    UNKNOWN_REFERENCE = "unknown_reference"
    SCHEMA_PARENT_MISMATCH = "schema_parent_mismatch"
    PREDICATE_SIGNATURE_MISMATCH = "predicate_signature_mismatch"
    TEMPORAL_INTERVAL_INVALID = "temporal_interval_invalid"
    TEMPORAL_VALUE_INVALID = "temporal_value_invalid"
    TEMPORAL_ORDER_CYCLE = "temporal_order_cycle"
    DECISION_DELTA_INVALID = "decision_delta_invalid"
    DESCRIPTION_SUPPORT_INVALID = "description_support_invalid"
    MENTION_SUPPORT_INVALID = "mention_support_invalid"
    PROVENANCE_MISMATCH = "provenance_mismatch"
    PROPOSITION_ASSERTION_MISMATCH = "proposition_assertion_mismatch"
    REPAIR_MUTATION_OUTSIDE_DIAGNOSTIC = "repair_mutation_outside_diagnostic"


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


def _source_bound_draft_provenance_diagnostics(
    *,
    draft: OntologyDraft,
    evidence: Sequence[EvidenceRecord | ModelVisibleEvidenceInput],
) -> tuple[ValidationDiagnostic, ...]:
    """Bind emitted assertion provenance to exact indexed source identity.

    Historical request fixtures without this lineage remain parseable, but this
    production validator rejects them.  Assertion-specific provenance identifiers and
    extraction methods may differ; the immutable locator/source hash may not, and an
    emitted provenance confidence cannot exceed either indexed confidence bound.
    Nothing is synthesized on the model's behalf.
    """

    diagnostics: list[ValidationDiagnostic] = []
    exact_evidence: dict[str, EvidenceRecord | ModelVisibleEvidenceRecord] = {}
    for index, item in enumerate(evidence):
        if (
            isinstance(item, (EvidenceRecord, ModelVisibleEvidenceRecord))
            and item.provenance.source_artifact_hash is not None
        ):
            exact_evidence[item.evidence_id] = item
        else:
            diagnostics.append(
                ValidationDiagnostic(
                    code=ValidationCode.PROVENANCE_MISMATCH,
                    path=f"evidence.{index}",
                    message=(
                        "production evidence lacks passage, text-hash, provenance, or "
                        "record-confidence lineage"
                    ),
                    related_ids=(item.evidence_id,),
                )
            )

    for assertion in draft.instance_graph.assertions:
        path = f"instance_graph.assertions.{assertion.assertion_id}.provenance"
        observed_by_id = {item.evidence_id: item for item in assertion.provenance}
        cited_ids = set(assertion.evidence_ids)
        observed_ids = tuple(item.evidence_id for item in assertion.provenance)
        provenance_ids = tuple(item.provenance_id for item in assertion.provenance)
        mismatched_ids: set[str] = set()
        for evidence_id in cited_ids & set(observed_by_id) & set(exact_evidence):
            observed = observed_by_id[evidence_id]
            indexed = exact_evidence[evidence_id]
            confidence_ceiling = min(indexed.confidence, indexed.provenance.confidence)
            if (
                observed.locator != indexed.provenance.locator
                or observed.source_artifact_hash
                != indexed.provenance.source_artifact_hash
                or observed.confidence > confidence_ceiling
            ):
                mismatched_ids.add(evidence_id)
        missing_exact_ids = cited_ids - set(exact_evidence)
        if (
            set(observed_ids) != cited_ids
            or len(observed_ids) != len(set(observed_ids))
            or len(provenance_ids) != len(set(provenance_ids))
            or mismatched_ids
            or missing_exact_ids
        ):
            diagnostics.append(
                ValidationDiagnostic(
                    code=ValidationCode.PROVENANCE_MISMATCH,
                    path=path,
                    message=(
                        "assertion provenance must exactly preserve each cited source "
                        "locator/hash and stay within indexed confidence bounds"
                    ),
                    related_ids=tuple(
                        sorted(
                            (set(observed_ids) ^ cited_ids)
                            | mismatched_ids
                            | missing_exact_ids
                        )
                    ),
                )
            )
    return tuple(diagnostics)


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

    diagnostics.extend(
        _source_bound_draft_provenance_diagnostics(
            draft=draft,
            evidence=packet.evidence,
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


def _record_identifier(value: Mapping[str, object]) -> str | None:
    for name in (
        "schema_id",
        "type_id",
        "predicate_id",
        "entity_id",
        "event_id",
        "proposition_content_id",
        "assertion_id",
        "decision_id",
        "evidence_id",
        "provenance_id",
        "candidate_id",
        "clue_id",
    ):
        identifier = value.get(name)
        if isinstance(identifier, str):
            return identifier
    return None


def _canonical_diff_paths(
    before: object,
    after: object,
    *,
    path: str = "",
) -> tuple[str, ...]:
    """Return stable semantic paths, keying record arrays by their IDs."""

    if isinstance(before, Mapping) and isinstance(after, Mapping):
        paths: list[str] = []
        keys = sorted((set(before) | set(after)) - {"content_hash"})
        for key in keys:
            child_path = f"{path}.{key}" if path else str(key)
            if key not in before or key not in after:
                paths.append(child_path)
            else:
                paths.extend(_canonical_diff_paths(before[key], after[key], path=child_path))
        return tuple(paths)
    if (
        isinstance(before, Sequence)
        and isinstance(after, Sequence)
        and not isinstance(before, (str, bytes, bytearray))
        and not isinstance(after, (str, bytes, bytearray))
    ):
        before_values = list(before)
        after_values = list(after)
        before_records = all(
            isinstance(item, Mapping) and _record_identifier(item) for item in before_values
        )
        after_records = all(
            isinstance(item, Mapping) and _record_identifier(item) for item in after_values
        )
        if before_records and after_records:
            before_by_id = {
                _record_identifier(item): item
                for item in before_values
                if isinstance(item, Mapping)
            }
            after_by_id = {
                _record_identifier(item): item for item in after_values if isinstance(item, Mapping)
            }
            paths = []
            for identifier in sorted(set(before_by_id) | set(after_by_id)):
                child_path = f"{path}.{identifier}" if path else str(identifier)
                if identifier not in before_by_id or identifier not in after_by_id:
                    paths.append(child_path)
                else:
                    paths.extend(
                        _canonical_diff_paths(
                            before_by_id[identifier],
                            after_by_id[identifier],
                            path=child_path,
                        )
                    )
            return tuple(paths)
        if len(before_values) != len(after_values):
            return (path,)
        paths = []
        for index, (before_item, after_item) in enumerate(
            zip(before_values, after_values, strict=True)
        ):
            paths.extend(
                _canonical_diff_paths(
                    before_item,
                    after_item,
                    path=f"{path}.{index}" if path else str(index),
                )
            )
        return tuple(paths)
    return () if before == after else (path,)


def validate_repair_preservation(
    *,
    base_draft: Mapping[str, object],
    repaired_draft: Mapping[str, object],
    diagnosed_paths: Sequence[str],
) -> BoundaryValidationReport:
    """Reject any repair mutation outside an explicitly diagnosed semantic path."""

    if not diagnosed_paths or any(not path for path in diagnosed_paths):
        raise ValueError("repair preservation requires nonempty diagnosed paths")
    changes = _canonical_diff_paths(base_draft, repaired_draft)
    unauthorized = tuple(
        change
        for change in changes
        if not any(
            change == allowed
            or change.startswith(f"{allowed}.")
            or allowed.startswith(f"{change}.")
            for allowed in diagnosed_paths
        )
    )
    if not unauthorized:
        return _report(())
    return _report(
        (
            ValidationDiagnostic(
                code=ValidationCode.REPAIR_MUTATION_OUTSIDE_DIAGNOSTIC,
                path=unauthorized[0],
                message="repair changed content outside the diagnosed field paths",
                related_ids=unauthorized[:32],
            ),
        )
    )


@dataclass(frozen=True)
class ProjectionDependencyClosure:
    """Existing graph objects required to keep selected assertions well formed.

    This is reachability over an already constructed, sealed graph. It does not
    create, rewrite, or rank semantics; query-time C0/C1 selection uses it only to
    avoid detaching retained objects from dependencies authored before query reveal.
    """

    assertion_ids: frozenset[str]
    node_ids: frozenset[str]
    proposition_content_ids: frozenset[str]


def _temporal_dependency_ids(
    extent: StoryTime | ValidityTime | HolderRelativeTime,
) -> frozenset[str]:
    dependencies: set[str] = set()
    if extent.kind is TemporalKind.RELATIVE and extent.anchor_id is not None:
        dependencies.add(extent.anchor_id)
    if extent.kind is TemporalKind.PARTIAL_ORDER:
        for constraint in extent.partial_order:
            dependencies.add(constraint.left_id)
            dependencies.add(constraint.right_id)
    return frozenset(dependencies)


def close_projection_dependencies(
    graph: InstanceGraph,
    *,
    seed_assertion_ids: Iterable[str],
) -> ProjectionDependencyClosure:
    """Return the recursive sealed-object closure for assertion selection.

    The closure includes binary/n-ary endpoints, node description assertions,
    epistemic holders, proposition contents and their endpoints, and every object
    named by relative or partial-order temporal extents. Unknown source references
    remain unresolved so final structural validation rejects them rather than this
    helper inventing a repair.
    """

    entities = {item.entity_id: item for item in graph.entities}
    events = {item.event_id: item for item in graph.events}
    propositions = {
        item.proposition_content_id: item for item in graph.proposition_contents
    }
    assertions = {item.assertion_id: item for item in graph.assertions}
    known_ids = set(entities) | set(events) | set(propositions) | set(assertions)
    seeds = set(seed_assertion_ids)
    unknown_seeds = seeds - set(assertions)
    if unknown_seeds:
        raise ValueError(
            "projection dependency closure received unknown assertion IDs: "
            + ", ".join(sorted(unknown_seeds))
        )

    selected_assertions: set[str] = set()
    selected_nodes: set[str] = set()
    selected_propositions: set[str] = set()
    visited: set[str] = set()
    pending = set(seeds)

    def temporal_dependencies(
        *extents: StoryTime | ValidityTime | HolderRelativeTime,
    ) -> set[str]:
        return {
            identifier
            for extent in extents
            for identifier in _temporal_dependency_ids(extent)
        }

    while pending:
        identifier = min(pending)
        pending.remove(identifier)
        if identifier in visited:
            continue
        visited.add(identifier)
        dependencies: set[str] = set()

        assertion = assertions.get(identifier)
        if assertion is not None:
            selected_assertions.add(identifier)
            dependencies.update(
                value
                for value in (assertion.subject_id, assertion.object_id)
                if value is not None
            )
            dependencies.update(role.object_id for role in assertion.roles)
            if assertion.proposition_content_id is not None:
                dependencies.add(assertion.proposition_content_id)
            dependencies.update(
                temporal_dependencies(
                    assertion.temporal_scope.story_time,
                    assertion.temporal_scope.validity_time,
                )
            )
            if assertion.epistemic_scope is not None:
                dependencies.add(assertion.epistemic_scope.holder_id)
                dependencies.update(
                    temporal_dependencies(
                        assertion.epistemic_scope.holder_relative_time
                    )
                )
        elif identifier in entities:
            entity = entities[identifier]
            selected_nodes.add(identifier)
            dependencies.update(entity.description_assertion_ids)
            dependencies.update(temporal_dependencies(entity.temporal_state))
        elif identifier in events:
            event = events[identifier]
            selected_nodes.add(identifier)
            dependencies.update(event.description_assertion_ids)
            dependencies.update(temporal_dependencies(event.occurrence_time))
        elif identifier in propositions:
            proposition = propositions[identifier]
            selected_propositions.add(identifier)
            dependencies.update(
                value
                for value in (proposition.subject_id, proposition.object_id)
                if value is not None
            )
            dependencies.update(role.object_id for role in proposition.roles)
            dependencies.update(
                temporal_dependencies(
                    proposition.temporal_content.story_time,
                    proposition.temporal_content.validity_time,
                )
            )

        pending.update((dependencies & known_ids) - visited)

    return ProjectionDependencyClosure(
        assertion_ids=frozenset(selected_assertions),
        node_ids=frozenset(selected_nodes),
        proposition_content_ids=frozenset(selected_propositions),
    )


def validate_draft_structure(
    *,
    draft: OntologyDraft,
    upper_ontology: UpperOntology,
    evidence: Sequence[ModelVisibleEvidenceInput],
    horizon: SpoilerHorizon,
    budgets: OutputBudgets,
    capabilities: ConstructionCapabilities,
) -> BoundaryValidationReport:
    """Validate references and declared construction without supplying semantics.

    The routine only rejects.  It never creates an ID, relation, temporal value,
    qualification, or repair suggestion for the model.
    """

    diagnostics: list[ValidationDiagnostic] = []

    def add(code: ValidationCode, path: str, message: str, *ids: str) -> None:
        diagnostics.append(
            ValidationDiagnostic(
                code=code,
                path=path,
                message=message,
                related_ids=tuple(dict.fromkeys(ids)),
            )
        )

    diagnostics.extend(
        _source_bound_draft_provenance_diagnostics(draft=draft, evidence=evidence)
    )

    graph = draft.instance_graph
    type_by_id = {item.type_id: item for item in draft.local_schema.contextual_types}
    predicate_by_id = {item.predicate_id: item for item in draft.local_schema.predicates}
    entity_by_id = {item.entity_id: item for item in graph.entities}
    event_by_id = {item.event_id: item for item in graph.events}
    proposition_by_id = {item.proposition_content_id: item for item in graph.proposition_contents}
    assertion_by_id = {item.assertion_id: item for item in graph.assertions}
    referent_ids = set(entity_by_id) | set(event_by_id) | set(proposition_by_id)
    graph_ids = referent_ids | set(assertion_by_id)
    schema_ids = {draft.local_schema.schema_id} | set(type_by_id) | set(predicate_by_id)
    targetable_ids = graph_ids | schema_ids

    qualification_ablation = (
        capabilities == ConstructionCapabilities.active_without_temporal_epistemic()
    )
    if qualification_ablation:
        if graph.proposition_contents:
            add(
                ValidationCode.ABLATION_QUALIFICATION_PRESENT,
                "instance_graph.proposition_contents",
                "A-NoTemporalEpistemic cannot emit epistemic proposition content",
                *(item.proposition_content_id for item in graph.proposition_contents),
            )

        def is_ablated_extent(extent: StoryTime | ValidityTime) -> bool:
            return (
                extent.kind is TemporalKind.UNKNOWN
                and extent.reason == ABLATION_QUALIFICATION_REASON
            )

        for entity in graph.entities:
            if not is_ablated_extent(entity.temporal_state):
                add(
                    ValidationCode.ABLATION_QUALIFICATION_PRESENT,
                    f"instance_graph.entities.{entity.entity_id}.temporal_state",
                    "A-NoTemporalEpistemic cannot supply entity temporal state",
                    entity.entity_id,
                )
        for event in graph.events:
            if not is_ablated_extent(event.occurrence_time):
                add(
                    ValidationCode.ABLATION_QUALIFICATION_PRESENT,
                    f"instance_graph.events.{event.event_id}.occurrence_time",
                    "A-NoTemporalEpistemic cannot supply event occurrence time",
                    event.event_id,
                )
        for assertion in graph.assertions:
            scope = assertion.temporal_scope
            temporal_absent = (
                is_ablated_extent(scope.story_time)
                and is_ablated_extent(scope.validity_time)
                and scope.discourse_position == DiscoursePosition(passage_order=0)
                and scope.revelation_position
                == RevelationPosition(
                    revelation_order=0,
                    label="qualification-ablated",
                )
            )
            epistemic_absent = (
                assertion.proposition_content_id is None
                and assertion.epistemic_scope is None
                and assertion.narrative_commitment is NarrativeCommitment.UNKNOWN
            )
            if not temporal_absent or not epistemic_absent:
                add(
                    ValidationCode.ABLATION_QUALIFICATION_PRESENT,
                    f"instance_graph.assertions.{assertion.assertion_id}",
                    "A-NoTemporalEpistemic cannot supply temporal or epistemic qualification",
                    assertion.assertion_id,
                )

    evidence_by_id = {item.evidence_id: item for item in evidence}
    mention_by_id = {
        candidate.candidate_id: candidate
        for record in evidence
        for candidate in record.mention_candidates
    }
    event_candidate_by_id = {
        candidate.candidate_id: candidate
        for record in evidence
        for candidate in record.event_candidates
    }
    relation_candidate_ids = {
        candidate.candidate_id
        for record in evidence
        for candidate in record.relation_phrase_candidates
    }
    temporal_clue_ids = {clue.clue_id for record in evidence for clue in record.temporal_clues}
    candidate_ids = (
        set(mention_by_id) | set(event_candidate_by_id) | relation_candidate_ids | temporal_clue_ids
    )

    accounting = draft.budget_accounting
    if capabilities == ConstructionCapabilities.prequery_construction():
        # C1's sealed preontology is intentionally more comprehensive than any one
        # contextual projection.  It is never a registered display object; only the
        # later fixed-selection projection receives the common final display gate.
        # Preserve the original prebuild accounting checks exactly so renderer
        # feasibility cannot silently reduce C1's construction capacity.
        actual_nodes = len(graph.entities) + len(graph.events)
        actual_assertions = len(graph.assertions)
        if accounting.nodes_used != actual_nodes:
            add(
                ValidationCode.BUDGET_ACCOUNTING_MISMATCH,
                "budget_accounting.nodes_used",
                "declared node use differs from the entity/event graph count",
            )
        if accounting.assertions_used != actual_assertions:
            add(
                ValidationCode.BUDGET_ACCOUNTING_MISMATCH,
                "budget_accounting.assertions_used",
                "declared assertion use differs from the qualified-assertion count",
            )
        if accounting.display_nodes_used > actual_nodes:
            add(
                ValidationCode.BUDGET_ACCOUNTING_MISMATCH,
                "budget_accounting.display_nodes_used",
                "display-node use exceeds materialized graph nodes",
            )
        if accounting.display_assertions_used > actual_assertions:
            add(
                ValidationCode.BUDGET_ACCOUNTING_MISMATCH,
                "budget_accounting.display_assertions_used",
                "display-assertion use exceeds materialized assertions",
            )
        try:
            accounting.validate_against(budgets)
        except ValueError as exc:
            add(
                ValidationCode.BUDGET_ACCOUNTING_MISMATCH,
                "budget_accounting",
                str(exc),
            )
    else:
        try:
            compile_registered_display_selection(graph, accounting, budgets)
        except DisplaySelectionError as exc:
            for path in exc.repair_paths:
                add(
                    ValidationCode.BUDGET_ACCOUNTING_MISMATCH,
                    path,
                    str(exc),
                )

    upper_types = set(upper_ontology.primitive_types)
    upper_relations = set(upper_ontology.primitive_relations)
    for contextual_type in type_by_id.values():
        if not contextual_type.evidence_ids:
            add(
                ValidationCode.MISSING_GROUNDING,
                f"local_schema.contextual_types.{contextual_type.type_id}.evidence_ids",
                "contextual type requires nonempty evidence grounding",
                contextual_type.type_id,
            )
        if contextual_type.parent_upper_type not in upper_types:
            add(
                ValidationCode.SCHEMA_PARENT_MISMATCH,
                f"local_schema.contextual_types.{contextual_type.type_id}.parent_upper_type",
                "contextual type has no declared upper-ontology parent",
                contextual_type.type_id,
                contextual_type.parent_upper_type,
            )
    for predicate in predicate_by_id.values():
        if not predicate.evidence_ids:
            add(
                ValidationCode.MISSING_GROUNDING,
                f"local_schema.predicates.{predicate.predicate_id}.evidence_ids",
                "local predicate requires nonempty evidence grounding",
                predicate.predicate_id,
            )
        if predicate.parent_upper_relation not in upper_relations:
            add(
                ValidationCode.SCHEMA_PARENT_MISMATCH,
                f"local_schema.predicates.{predicate.predicate_id}.parent_upper_relation",
                "local predicate has no declared upper-ontology parent",
                predicate.predicate_id,
                predicate.parent_upper_relation,
            )
        unknown_types = (set(predicate.domain_type_ids) | set(predicate.range_type_ids)) - set(
            type_by_id
        )
        if unknown_types:
            add(
                ValidationCode.UNKNOWN_REFERENCE,
                f"local_schema.predicates.{predicate.predicate_id}",
                "predicate signature refers to an unknown contextual type",
                predicate.predicate_id,
                *sorted(unknown_types),
            )
        if len(predicate.role_names) != len(set(predicate.role_names)):
            add(
                ValidationCode.PREDICATE_SIGNATURE_MISMATCH,
                f"local_schema.predicates.{predicate.predicate_id}.role_names",
                "predicate role names must be unique",
                predicate.predicate_id,
            )

    for kind, records in (("entities", graph.entities), ("events", graph.events)):
        for record in records:
            record_id = record.entity_id if kind == "entities" else record.event_id
            if record.contextual_type_id not in type_by_id:
                add(
                    ValidationCode.UNKNOWN_REFERENCE,
                    f"instance_graph.{kind}.{record_id}.contextual_type_id",
                    "graph node refers to an unknown contextual type",
                    record_id,
                    record.contextual_type_id,
                )
            unknown_descriptions = set(record.description_assertion_ids) - set(assertion_by_id)
            if unknown_descriptions:
                add(
                    ValidationCode.DESCRIPTION_SUPPORT_INVALID,
                    f"instance_graph.{kind}.{record_id}.description_assertion_ids",
                    "node description refers to an unknown assertion",
                    record_id,
                    *sorted(unknown_descriptions),
                )

    for entity in graph.entities:
        unknown_mentions = set(entity.supported_mention_candidate_ids) - set(mention_by_id)
        if unknown_mentions:
            add(
                ValidationCode.MENTION_SUPPORT_INVALID,
                f"instance_graph.entities.{entity.entity_id}.supported_mention_candidate_ids",
                "entity refers to a mention absent from the frozen evidence",
                entity.entity_id,
                *sorted(unknown_mentions),
            )
        misplaced = tuple(
            candidate_id
            for candidate_id in entity.supported_mention_candidate_ids
            if candidate_id in mention_by_id
            and mention_by_id[candidate_id].evidence_id not in entity.evidence_ids
        )
        if misplaced:
            add(
                ValidationCode.MENTION_SUPPORT_INVALID,
                f"instance_graph.entities.{entity.entity_id}.evidence_ids",
                "entity evidence omits the source of a supported mention",
                entity.entity_id,
                *misplaced,
            )
    for event in graph.events:
        if not any(
            candidate.evidence_id in event.evidence_ids
            for candidate in event_candidate_by_id.values()
        ):
            add(
                ValidationCode.MISSING_GROUNDING,
                f"instance_graph.events.{event.event_id}.evidence_ids",
                "reified event has no event-candidate anchor in cited evidence",
                event.event_id,
            )

    def role_references(
        *,
        record_id: str,
        roles: Sequence[RoleBinding],
        evidence_ids: Sequence[str],
        path: str,
    ) -> tuple[str, ...]:
        object_ids: list[str] = []
        for index, role in enumerate(roles):
            object_id = role.object_id
            object_ids.append(object_id)
            if object_id not in referent_ids:
                add(
                    ValidationCode.UNKNOWN_REFERENCE,
                    f"{path}.roles.{index}.object_id",
                    "role binding refers to an unknown graph object",
                    record_id,
                    object_id,
                )
            role_evidence = set(role.evidence_ids)
            if not role_evidence or not role_evidence.issubset(evidence_ids):
                add(
                    ValidationCode.MISSING_GROUNDING,
                    f"{path}.roles.{index}.evidence_ids",
                    "role evidence must be nonempty and contained in parent evidence",
                    record_id,
                )
        return tuple(object_ids)

    for proposition in graph.proposition_contents:
        predicate = predicate_by_id.get(proposition.predicate_id)
        path = f"instance_graph.proposition_contents.{proposition.proposition_content_id}"
        if predicate is None:
            add(
                ValidationCode.UNKNOWN_REFERENCE,
                f"{path}.predicate_id",
                "proposition refers to an unknown local predicate",
                proposition.proposition_content_id,
                proposition.predicate_id,
            )
        if proposition.subject_id is not None:
            unknown = {
                proposition.subject_id,
                proposition.object_id,
            } - referent_ids
            unknown.discard(None)
            if unknown:
                add(
                    ValidationCode.UNKNOWN_REFERENCE,
                    path,
                    "proposition endpoint refers to an unknown graph object",
                    proposition.proposition_content_id,
                    *sorted(unknown),
                )
            if predicate is not None and predicate.arity != 2:
                add(
                    ValidationCode.PREDICATE_SIGNATURE_MISMATCH,
                    path,
                    "binary proposition uses a non-binary local predicate",
                    proposition.proposition_content_id,
                    predicate.predicate_id,
                )
        else:
            object_ids = role_references(
                record_id=proposition.proposition_content_id,
                roles=proposition.roles,
                evidence_ids=proposition.evidence_ids,
                path=path,
            )
            if predicate is not None and (
                len(object_ids) != predicate.arity
                or tuple(role.role for role in proposition.roles) != predicate.role_names
            ):
                add(
                    ValidationCode.PREDICATE_SIGNATURE_MISMATCH,
                    path,
                    "proposition roles do not match the local predicate signature",
                    proposition.proposition_content_id,
                    predicate.predicate_id,
                )

    assertion_footprints: dict[str, set[str]] = {}
    for assertion in graph.assertions:
        path = f"instance_graph.assertions.{assertion.assertion_id}"
        predicate = predicate_by_id.get(assertion.predicate_id)
        footprint: set[str] = set()
        if predicate is None:
            add(
                ValidationCode.UNKNOWN_REFERENCE,
                f"{path}.predicate_id",
                "assertion refers to an unknown local predicate",
                assertion.assertion_id,
                assertion.predicate_id,
            )
        if assertion.subject_id is not None:
            endpoints = {assertion.subject_id, assertion.object_id}
            unknown = endpoints - referent_ids
            unknown.discard(None)
            footprint.update(item for item in endpoints if item is not None)
            if unknown:
                add(
                    ValidationCode.UNKNOWN_REFERENCE,
                    path,
                    "assertion endpoint refers to an unknown graph object",
                    assertion.assertion_id,
                    *sorted(unknown),
                )
            if predicate is not None and predicate.arity != 2:
                add(
                    ValidationCode.PREDICATE_SIGNATURE_MISMATCH,
                    path,
                    "binary assertion uses a non-binary local predicate",
                    assertion.assertion_id,
                    predicate.predicate_id,
                )
        else:
            role_ids = role_references(
                record_id=assertion.assertion_id,
                roles=assertion.roles,
                evidence_ids=assertion.evidence_ids,
                path=path,
            )
            footprint.update(role_ids)
            if predicate is not None and (
                len(role_ids) != predicate.arity
                or tuple(role.role for role in assertion.roles) != predicate.role_names
            ):
                add(
                    ValidationCode.PREDICATE_SIGNATURE_MISMATCH,
                    path,
                    "assertion roles do not match the local predicate signature",
                    assertion.assertion_id,
                    predicate.predicate_id,
                )
        if assertion.proposition_content_id is not None:
            proposition = proposition_by_id.get(assertion.proposition_content_id)
            if proposition is None:
                add(
                    ValidationCode.UNKNOWN_REFERENCE,
                    f"{path}.proposition_content_id",
                    "epistemic assertion refers to unknown proposition content",
                    assertion.assertion_id,
                    assertion.proposition_content_id,
                )
            else:
                footprint.update(
                    item
                    for item in (proposition.subject_id, proposition.object_id)
                    if item is not None
                )
                footprint.update(role.object_id for role in proposition.roles)
                assertion_shape = (
                    assertion.predicate_id,
                    assertion.subject_id,
                    assertion.object_id,
                    tuple(
                        (role.role, role.object_id, role.evidence_ids) for role in assertion.roles
                    ),
                )
                proposition_shape = (
                    proposition.predicate_id,
                    proposition.subject_id,
                    proposition.object_id,
                    tuple(
                        (role.role, role.object_id, role.evidence_ids) for role in proposition.roles
                    ),
                )
                if assertion_shape != proposition_shape or not set(
                    proposition.evidence_ids
                ).issubset(assertion.evidence_ids):
                    add(
                        ValidationCode.PROPOSITION_ASSERTION_MISMATCH,
                        f"{path}.proposition_content_id",
                        "attributed assertion does not preserve its proposition content",
                        assertion.assertion_id,
                        proposition.proposition_content_id,
                    )
        if assertion.epistemic_scope is not None:
            if assertion.epistemic_scope.holder_id not in entity_by_id:
                add(
                    ValidationCode.UNKNOWN_REFERENCE,
                    f"{path}.epistemic_scope.holder_id",
                    "epistemic holder is not a materialized entity",
                    assertion.assertion_id,
                    assertion.epistemic_scope.holder_id,
                )
            footprint.add(assertion.epistemic_scope.holder_id)
        assertion_footprints[assertion.assertion_id] = footprint

    for kind, records in (("entities", graph.entities), ("events", graph.events)):
        for record in records:
            record_id = record.entity_id if kind == "entities" else record.event_id
            for assertion_id in record.description_assertion_ids:
                if (
                    assertion_id in assertion_footprints
                    and record_id not in assertion_footprints[assertion_id]
                ):
                    add(
                        ValidationCode.DESCRIPTION_SUPPORT_INVALID,
                        f"instance_graph.{kind}.{record_id}.description_assertion_ids",
                        "description support assertion does not involve the described node",
                        record_id,
                        assertion_id,
                    )

    cited_ids = set()
    grounded_records = (
        *draft.local_schema.contextual_types,
        *draft.local_schema.predicates,
        *graph.entities,
        *graph.events,
        *graph.proposition_contents,
        *graph.assertions,
        *draft.decisions,
    )
    for record in grounded_records:
        record_evidence = set(record.evidence_ids)
        cited_ids.update(record_evidence)
        unknown = record_evidence - set(evidence_by_id)
        if unknown:
            add(
                ValidationCode.EVIDENCE_OUTSIDE_PACKET,
                f"records.{_record_identifier(record.model_dump(mode='python'))}.evidence_ids",
                "record cites evidence outside the complete frozen input",
                *sorted(unknown),
            )

    all_input_ids = targetable_ids | candidate_ids
    allowed_operators = {
        operator
        for operator, enabled in (
            (ConstructionOperator.MERGE, capabilities.merge_split),
            (ConstructionOperator.SPLIT, capabilities.merge_split),
            (ConstructionOperator.CONTEXTUAL_TYPE, capabilities.create_entities),
            (ConstructionOperator.SCHEMA_RELATION, capabilities.create_schema_predicates),
            (ConstructionOperator.EVENT_REIFICATION, capabilities.reify_events),
            (ConstructionOperator.ABSTRACTION, capabilities.change_abstraction),
            (
                ConstructionOperator.TEMPORAL_QUALIFICATION,
                capabilities.add_temporal_qualification,
            ),
            (
                ConstructionOperator.EPISTEMIC_QUALIFICATION,
                capabilities.add_epistemic_qualification,
            ),
        )
        if enabled
    }
    allowed_operators.update(
        {
            ConstructionOperator.INCLUDE_EXCLUDE,
            ConstructionOperator.RARE_PRESERVATION,
        }
    )
    if capabilities.select_existing:
        allowed_operators.add(ConstructionOperator.SELECTION)
    if capabilities.compress_existing:
        allowed_operators.add(ConstructionOperator.COMPRESSION)
    if capabilities.write_supported_descriptions:
        allowed_operators.add(ConstructionOperator.SUPPORTED_DESCRIPTION)

    operator_targets = {
        ConstructionOperator.MERGE: set(entity_by_id),
        ConstructionOperator.SPLIT: set(entity_by_id),
        ConstructionOperator.CONTEXTUAL_TYPE: set(type_by_id),
        ConstructionOperator.SCHEMA_RELATION: {draft.local_schema.schema_id} | set(predicate_by_id),
        ConstructionOperator.EVENT_REIFICATION: set(event_by_id),
        ConstructionOperator.ABSTRACTION: schema_ids | set(entity_by_id) | set(event_by_id),
        ConstructionOperator.TEMPORAL_QUALIFICATION: set(assertion_by_id),
        ConstructionOperator.EPISTEMIC_QUALIFICATION: set(proposition_by_id) | set(assertion_by_id),
        ConstructionOperator.RARE_PRESERVATION: targetable_ids,
    }
    for decision in draft.decisions:
        path = f"decisions.{decision.decision_id}"
        if decision.operator not in allowed_operators:
            add(
                ValidationCode.DECISION_DELTA_INVALID,
                f"{path}.operator",
                "decision operator is disabled by the supplied capability manifest",
                decision.decision_id,
            )
        unknown_inputs = set(decision.input_object_ids) - all_input_ids
        unknown_created = set(decision.created_object_ids) - targetable_ids
        unknown_removed = set(decision.removed_object_ids) - all_input_ids
        if unknown_inputs or unknown_created or unknown_removed:
            add(
                ValidationCode.DECISION_DELTA_INVALID,
                path,
                "decision refers to an object absent from evidence or the generated graph",
                decision.decision_id,
                *sorted(unknown_inputs | unknown_created | unknown_removed),
            )
        if set(decision.removed_object_ids) & targetable_ids:
            add(
                ValidationCode.DECISION_DELTA_INVALID,
                f"{path}.removed_object_ids",
                "a reportedly removed object remains in the generated ontology",
                decision.decision_id,
            )
        expected_targets = operator_targets.get(decision.operator)
        if expected_targets is not None and (
            not decision.created_object_ids
            or not set(decision.created_object_ids).issubset(expected_targets)
        ):
            add(
                ValidationCode.DECISION_DELTA_INVALID,
                f"{path}.created_object_ids",
                "constructive decision does not create an object of its declared kind",
                decision.decision_id,
                *decision.created_object_ids,
            )
        if decision.operator in {ConstructionOperator.MERGE, ConstructionOperator.SPLIT}:
            if len(set(decision.input_object_ids)) < 2:
                add(
                    ValidationCode.DECISION_DELTA_INVALID,
                    f"{path}.input_object_ids",
                    "merge/split requires at least two distinct input objects",
                    decision.decision_id,
                )
            for created_id in decision.created_object_ids:
                entity = entity_by_id.get(created_id)
                if entity is not None and not (
                    set(entity.supported_mention_candidate_ids) & set(decision.input_object_ids)
                ):
                    add(
                        ValidationCode.DECISION_DELTA_INVALID,
                        path,
                        "merge/split output is not anchored to any declared input mention",
                        decision.decision_id,
                        created_id,
                    )

    temporal_owners = referent_ids | set(assertion_by_id)
    order_edges: dict[str, set[str]] = {identifier: set() for identifier in temporal_owners}

    def add_registered_temporal_diagnostics(
        result: TemporalValidationResult,
        owner_id: str,
    ) -> None:
        code_map = {
            TemporalDiagnosticCode.INVALID_INTERVAL_BOUNDS: (
                ValidationCode.TEMPORAL_INTERVAL_INVALID
            ),
            TemporalDiagnosticCode.INVALID_EXPLICIT_TIME: (
                ValidationCode.TEMPORAL_VALUE_INVALID
            ),
            TemporalDiagnosticCode.EVIDENCE_AFTER_SPOILER_HORIZON: (
                ValidationCode.DISCOURSE_HORIZON_LEAK
            ),
            TemporalDiagnosticCode.REVELATION_AFTER_SPOILER_HORIZON: (
                ValidationCode.REVELATION_HORIZON_LEAK
            ),
        }
        for diagnostic in result.diagnostics:
            code = code_map.get(diagnostic.code)
            if code is None:
                continue
            add(
                code,
                diagnostic.field_path,
                diagnostic.message,
                owner_id,
                *diagnostic.involved_ids,
            )

    def validate_temporal(
        owner_id: str,
        field_name: str,
        extent: StoryTime | ValidityTime | HolderRelativeTime,
    ) -> None:
        kind = extent.kind
        path = f"temporal.{owner_id}.{field_name}"
        if kind is TemporalKind.RELATIVE:
            anchor = extent.anchor_id
            relation = extent.relation
            if anchor not in temporal_owners:
                add(
                    ValidationCode.UNKNOWN_REFERENCE,
                    f"{path}.anchor_id",
                    "relative temporal anchor is absent from the graph",
                    owner_id,
                    anchor,
                )
            elif relation in {AllenRelation.BEFORE, AllenRelation.MEETS}:
                order_edges[owner_id].add(anchor)
            elif relation in {AllenRelation.AFTER, AllenRelation.MET_BY}:
                order_edges[anchor].add(owner_id)
        if kind is TemporalKind.PARTIAL_ORDER:
            for index, constraint in enumerate(extent.partial_order):
                if (
                    constraint.left_id not in temporal_owners
                    or constraint.right_id not in temporal_owners
                ):
                    add(
                        ValidationCode.UNKNOWN_REFERENCE,
                        f"{path}.partial_order.{index}",
                        "partial-order constraint names an absent graph object",
                        constraint.left_id,
                        constraint.right_id,
                    )
                elif constraint.relation in {AllenRelation.BEFORE, AllenRelation.MEETS}:
                    order_edges[constraint.left_id].add(constraint.right_id)
                elif constraint.relation in {AllenRelation.AFTER, AllenRelation.MET_BY}:
                    order_edges[constraint.right_id].add(constraint.left_id)

    for entity in graph.entities:
        validate_temporal(entity.entity_id, "temporal_state", entity.temporal_state)
        add_registered_temporal_diagnostics(
            validate_temporal_extent(
                entity.temporal_state,
                known_anchor_ids=temporal_owners,
                field_path=f"entities.{entity.entity_id}.temporal_state",
            ),
            entity.entity_id,
        )
    for event in graph.events:
        validate_temporal(event.event_id, "occurrence_time", event.occurrence_time)
        add_registered_temporal_diagnostics(
            validate_temporal_extent(
                event.occurrence_time,
                known_anchor_ids=temporal_owners,
                field_path=f"events.{event.event_id}.occurrence_time",
            ),
            event.event_id,
        )
    for proposition in graph.proposition_contents:
        validate_temporal(
            proposition.proposition_content_id,
            "story_time",
            proposition.temporal_content.story_time,
        )
        validate_temporal(
            proposition.proposition_content_id,
            "validity_time",
            proposition.temporal_content.validity_time,
        )
        add_registered_temporal_diagnostics(
            validate_temporal_scope(
                proposition.temporal_content,
                horizon=horizon,
                known_anchor_ids=temporal_owners,
                field_path=(
                    "proposition_contents."
                    f"{proposition.proposition_content_id}.temporal_content"
                ),
            ),
            proposition.proposition_content_id,
        )
    for assertion in graph.assertions:
        validate_temporal(assertion.assertion_id, "story_time", assertion.temporal_scope.story_time)
        validate_temporal(
            assertion.assertion_id, "validity_time", assertion.temporal_scope.validity_time
        )
        if assertion.epistemic_scope is not None:
            validate_temporal(
                assertion.assertion_id,
                "holder_relative_time",
                assertion.epistemic_scope.holder_relative_time,
            )
        add_registered_temporal_diagnostics(
            validate_assertion_temporality(
                assertion,
                horizon=horizon,
                known_anchor_ids=temporal_owners,
            ),
            assertion.assertion_id,
        )

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(identifier: str) -> bool:
        if identifier in visiting:
            return True
        if identifier in visited:
            return False
        visiting.add(identifier)
        if any(visit(next_id) for next_id in order_edges[identifier]):
            return True
        visiting.remove(identifier)
        visited.add(identifier)
        return False

    if any(visit(identifier) for identifier in tuple(order_edges)):
        add(
            ValidationCode.TEMPORAL_ORDER_CYCLE,
            "instance_graph",
            "strict before/meets constraints contain a cycle",
        )

    return _report(diagnostics)


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

    elif audit.condition in ACTIVE_QUERY_CONSTRUCTION_CONDITIONS:
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
    query_access: QueryAccessEvent,
    packet_materialization: PacketMaterializationEvent | None = None,
    request_created_at: AwareDatetime | None,
    seed_block: int,
    sealed_seed_block: int | None = None,
) -> BoundaryValidationReport:
    """Adapt a full projection's seals/certificates to the timing-lineage gate."""

    if projection.condition not in (
        {
            ConditionName.C1_LLM_PRE,
            ConditionName.A_FIXED_SELECT,
        }
        | ACTIVE_QUERY_CONSTRUCTION_CONDITIONS
    ):
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

    if query_access.query_context_hash != context.content_hash:
        diagnostics.append(
            ValidationDiagnostic(
                code=ValidationCode.CONTEXT_LINEAGE_MISMATCH,
                path="query_access.query_context_hash",
                message="query-access event cites a different query context",
            )
        )
    if query_access.model_visible_query_hash != to_model_visible_query(context).content_hash:
        diagnostics.append(
            ValidationDiagnostic(
                code=ValidationCode.CONTEXT_LINEAGE_MISMATCH,
                path="query_access.model_visible_query_hash",
                message="query-access event cites different model-visible query semantics",
            )
        )
    if projection.query_access_event_hash != query_access.content_hash:
        diagnostics.append(
            ValidationDiagnostic(
                code=ValidationCode.LINEAGE_HASH_MISMATCH,
                path="projection.query_access_event_hash",
                message="projection does not cite the supplied query-access event",
            )
        )
    if query_access.snapshot_hash != projection.snapshot_hash:
        diagnostics.append(
            ValidationDiagnostic(
                code=ValidationCode.LINEAGE_HASH_MISMATCH,
                path="query_access.snapshot_hash",
                message="query-access event cites a different evidence snapshot",
            )
        )
    bound_packet_hash = query_access.packet_hash
    if bound_packet_hash is None and packet_materialization is not None:
        if (
            packet_materialization.execution_id != query_access.execution_id
            or packet_materialization.query_access_event_hash != query_access.content_hash
            or packet_materialization.snapshot_hash != query_access.snapshot_hash
            or packet_materialization.started_at < query_access.accessed_at
        ):
            diagnostics.append(
                ValidationDiagnostic(
                    code=ValidationCode.LINEAGE_HASH_MISMATCH,
                    path="packet_materialization",
                    message="packet materialization does not descend from query access",
                )
            )
        else:
            bound_packet_hash = packet_materialization.packet_hash
    if bound_packet_hash != projection.packet_hash:
        diagnostics.append(
            ValidationDiagnostic(
                code=ValidationCode.LINEAGE_HASH_MISMATCH,
                path="query_access.packet_hash",
                message="query access/materialization cites a different evidence packet",
            )
        )
    if query_access.registered_revealed_at != context.revealed_at:
        diagnostics.append(
            ValidationDiagnostic(
                code=ValidationCode.CONTEXT_LINEAGE_MISMATCH,
                path="query_access.registered_revealed_at",
                message="query-access event and registered reveal timestamp differ",
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
    if certificate is not None and (
        certificate.query_revealed_at != query_access.accessed_at
        or certificate.query_access_event_hash != query_access.content_hash
        or certificate.stage_manifest_hash != query_access.stage_manifest_hash
        or certificate.prequery_barrier_hash != query_access.prequery_barrier_hash
    ):
        diagnostics.append(
            ValidationDiagnostic(
                code=ValidationCode.CONTEXT_LINEAGE_MISMATCH,
                path="construction_certificate.query_revealed_at",
                message="certificate and physical query-access lineage differ",
            )
        )

    construction_times = tuple(
        decision.decided_at
        for decision in projection.decisions
        if decision.operator in CONSTRUCTIVE_OPERATORS
    )
    selection_times = tuple(
        decision.decided_at
        for decision in projection.decisions
        if decision.operator not in CONSTRUCTIVE_OPERATORS
    )
    if projection.condition is ConditionName.C1_LLM_PRE and any(
        timestamp <= query_access.accessed_at for timestamp in selection_times
    ):
        diagnostics.append(
            ValidationDiagnostic(
                code=ValidationCode.QUERY_DECISION_BEFORE_REVEAL,
                path="projection.decisions",
                message="C1 query-time projection decision did not follow physical query access",
            )
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
        else query_access.accessed_at
    )
    audit = ConstructionLineageAudit(
        condition=projection.condition,
        query_revealed_at=query_access.accessed_at,
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
