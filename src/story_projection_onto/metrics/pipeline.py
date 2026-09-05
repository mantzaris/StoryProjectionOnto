"""Ledger-facing scoring pipeline over the frozen projection adapters."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import Field, model_validator

from story_projection_onto.contracts import (
    ConditionName,
    EvidencePacket,
    GoldContextualProjection,
    GoldContrastInvariant,
    ImmutableRecord,
    OntologyProjection,
    QueryContext,
    ReleaseClass,
    canonical_json,
    canonical_sha256,
)
from story_projection_onto.metrics.adapters import (
    ProjectionMetricAdapter,
    adapt_projection_for_metrics,
    projection_is_content_bearing,
    projection_is_structurally_valid,
    verify_normalized_decisions,
    verify_projection_decisions,
)
from story_projection_onto.metrics.alignment import (
    AlignmentPlan,
    AlignmentResult,
    GroundingStatus,
    NormalizedDecision,
    OntologyDecisionScore,
    prediction_records,
    score_alignment,
    score_ontology_decisions,
)
from story_projection_onto.metrics.clutter import (
    ClutterProfile,
    LabelRectangle,
    Point,
    clutter_profile,
)
from story_projection_onto.metrics.community import (
    ConductanceResult,
    CrossSeedStability,
    LeidenConfiguration,
    LeidenPartition,
    PartitionAgreement,
    conductance,
    cross_seed_stability,
    run_leiden_cpm,
    score_partition,
)
from story_projection_onto.metrics.config import StudyMetricConfiguration
from story_projection_onto.metrics.contrastive import (
    ContrastInvariantScore,
    ContrastiveScore,
    SignedDecisionChange,
    derive_signed_changes,
    score_contrast_invariants,
    score_contrastive_changes,
)
from story_projection_onto.metrics.decision_states import (
    DecisionStateCompilation,
    InvariantCompilationIssue,
    compile_and_verify_decision_states,
    compile_gold_contrast_invariants,
    compile_gold_decision_targets,
    compile_projection_assertion_semantics,
)
from story_projection_onto.metrics.entropy import EntropyProfile, entropy_profile
from story_projection_onto.metrics.rare import (
    RarePivotalAnnotation,
    RarePivotalScore,
    score_rare_pivotal,
)


class PipelineMetricStatus(StrEnum):
    VALUE = "value"
    NOT_APPLICABLE = "not_applicable"
    UNDEFINED = "undefined"
    INVALID = "invalid"


class MetricResultRow(ImmutableRecord):
    projection_id: str | None = Field(default=None, min_length=1)
    unit_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    metric_name: str = Field(min_length=1)
    metric_version_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: PipelineMetricStatus
    value: float | int | None = None
    numerator: float | int | None = None
    denominator: float | int | None = None

    @model_validator(mode="after")
    def value_agrees_with_status(self) -> Self:
        if self.status is PipelineMetricStatus.VALUE and self.value is None:
            raise ValueError("value status requires a numeric metric value")
        if self.status is not PipelineMetricStatus.VALUE and self.value is not None:
            raise ValueError("unavailable metric status cannot carry a value")
        return self


class StructuralMetricPanel(ImmutableRecord):
    valid_content_bearing: bool
    failure_reason: str | None = None
    entropy: EntropyProfile | None = None
    leiden_base: LeidenPartition | None = None
    leiden_half: LeidenPartition | None = None
    leiden_double: LeidenPartition | None = None

    @model_validator(mode="after")
    def invalid_panels_are_closed(self) -> Self:
        structural = (self.entropy, self.leiden_base, self.leiden_half, self.leiden_double)
        if not self.valid_content_bearing and any(item is not None for item in structural):
            raise ValueError("invalid or empty projections cannot expose structural metrics")
        if self.valid_content_bearing and self.failure_reason is not None:
            raise ValueError("valid structural panels cannot carry a failure reason")
        return self


class ProjectionScoreBundle(ImmutableRecord):
    scoring_input_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    adapter: ProjectionMetricAdapter
    alignment: AlignmentResult
    ontology_decisions: OntologyDecisionScore
    decision_compilation: DecisionStateCompilation | None = None
    structural: StructuralMetricPanel
    metric_rows: tuple[MetricResultRow, ...]


class ScorerMetricPlan(ImmutableRecord):
    """Immutable scorer-only inputs bound to one contextual gold projection."""

    scorer_namespace: Literal["scorer_only"] = "scorer_only"
    plan_id: str = Field(min_length=1)
    source_gold_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    gold_projection: GoldContextualProjection | None = None
    alignment_plan: AlignmentPlan
    gold_decisions: tuple[NormalizedDecision, ...]
    rare_annotations: tuple[RarePivotalAnnotation, ...]
    gold_community_assignments: tuple[tuple[str, str], ...] = ()
    community_eligible: bool = False
    valid_evidence_ids: tuple[str, ...]
    valid_evidence_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    predicate_aliases: tuple[tuple[str, str], ...] = ()

    @model_validator(mode="after")
    def scorer_inputs_are_bound_and_canonical(self) -> Self:
        if self.source_gold_hash != self.alignment_plan.source_gold_hash:
            raise ValueError("metric plan and alignment plan name different gold projections")
        if (
            self.gold_projection is not None
            and self.gold_projection.content_hash != self.source_gold_hash
        ):
            raise ValueError("metric plan embeds a different gold projection")
        if self.gold_projection is None and any(
            len(item.semantic_signature) != 64
            or any(
                character not in "0123456789abcdef"
                for character in item.semantic_signature
            )
            for item in self.gold_decisions
        ):
            raise ValueError(
                "noncanonical gold decision signatures require the source gold projection"
            )
        decision_slots = tuple(item.slot_key for item in self.gold_decisions)
        if len(decision_slots) != len(set(decision_slots)):
            raise ValueError("gold decision slots must be unique")
        if self.gold_decisions != tuple(
            sorted(self.gold_decisions, key=lambda item: item.slot_key)
        ):
            raise ValueError("gold decisions must be canonically ordered by slot")
        if self.valid_evidence_ids != tuple(sorted(set(self.valid_evidence_ids))):
            raise ValueError("valid evidence IDs must be sorted and unique")
        if self.valid_evidence_manifest_hash != canonical_sha256(self.valid_evidence_ids):
            raise ValueError("valid evidence manifest hash does not match its IDs")
        if self.predicate_aliases != tuple(sorted(set(self.predicate_aliases))):
            raise ValueError("predicate aliases must be canonical and source-unique")
        if len(dict(self.predicate_aliases)) != len(self.predicate_aliases):
            raise ValueError("predicate aliases require unique source labels")
        community_anchors = tuple(item[0] for item in self.gold_community_assignments)
        if self.gold_community_assignments != tuple(sorted(self.gold_community_assignments)):
            raise ValueError("gold community assignments must be canonically ordered")
        if len(community_anchors) != len(set(community_anchors)):
            raise ValueError("gold community anchors must be unique")
        if self.community_eligible != bool(self.gold_community_assignments):
            raise ValueError("community eligibility must agree with reviewed gold assignments")
        node_targets = {item.target_id for item in self.alignment_plan.node_targets}
        if not set(community_anchors).issubset(node_targets):
            raise ValueError("gold community assignments reference unknown node targets")
        assertion_targets = {
            item.target_id for item in self.alignment_plan.assertion_targets
        }
        rare_ids = tuple(item.assertion_target_id for item in self.rare_annotations)
        if self.rare_annotations != tuple(
            sorted(self.rare_annotations, key=lambda item: item.assertion_target_id)
        ):
            raise ValueError("rare annotations must be canonically ordered")
        if len(rare_ids) != len(set(rare_ids)):
            raise ValueError("rare annotations require unique assertion targets")
        if set(rare_ids) != assertion_targets:
            raise ValueError("rare annotations must cover every and only assertion target")
        return self


class GroundingAuditInput(ImmutableRecord):
    """Output-specific, evidence-manifest-bound grounding verdicts."""

    projection_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    valid_evidence_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    assertion_statuses: tuple[tuple[str, GroundingStatus], ...]
    audit_artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def statuses_are_canonical(self) -> Self:
        assertion_ids = tuple(item[0] for item in self.assertion_statuses)
        if self.assertion_statuses != tuple(sorted(self.assertion_statuses)):
            raise ValueError("grounding statuses must be canonically ordered")
        if len(assertion_ids) != len(set(assertion_ids)):
            raise ValueError("grounding statuses require unique assertion IDs")
        return self


class GeometryMetricInput(ImmutableRecord):
    """Frozen visualization geometry, cryptographically bound to one projection."""

    projection_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    visualization_state_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    materialization_source_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_result_hashes: tuple[str, ...] = Field(min_length=1)
    layout_config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    style_config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    font_config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    viewport_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    renderer_runtime_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    positions: tuple[tuple[str, Point], ...]
    renderer_only_positions: tuple[tuple[str, Point], ...]
    label_rectangles: tuple[LabelRectangle, ...]
    label_semantic_ids: tuple[tuple[str, str], ...]
    visible_semantic_ids: tuple[str, ...]
    irrelevant_semantic_ids: tuple[str, ...]
    rare_pivotal_discoverability: tuple[tuple[str, int], ...]

    @model_validator(mode="after")
    def geometry_is_canonical(self) -> Self:
        if self.source_result_hashes != tuple(sorted(set(self.source_result_hashes))):
            raise ValueError("geometry source-result hashes must be sorted and unique")
        if any(
            len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
            for value in self.source_result_hashes
        ):
            raise ValueError("geometry source-result hashes must be SHA-256 digests")
        position_ids = tuple(item[0] for item in self.positions)
        if self.positions != tuple(sorted(self.positions, key=lambda item: item[0])):
            raise ValueError("geometry positions must be canonically ordered")
        if len(position_ids) != len(set(position_ids)):
            raise ValueError("geometry positions require unique node IDs")
        renderer_only_ids = tuple(item[0] for item in self.renderer_only_positions)
        if self.renderer_only_positions != tuple(
            sorted(self.renderer_only_positions, key=lambda item: item[0])
        ):
            raise ValueError("renderer-only positions must be canonically ordered")
        if len(renderer_only_ids) != len(set(renderer_only_ids)):
            raise ValueError("renderer-only positions require unique IDs")
        if set(renderer_only_ids) & set(position_ids):
            raise ValueError("semantic and renderer-only position IDs must be disjoint")
        rectangle_ids = tuple(item.label_id for item in self.label_rectangles)
        if self.label_rectangles != tuple(
            sorted(self.label_rectangles, key=lambda item: item.label_id)
        ):
            raise ValueError("label rectangles must be canonically ordered")
        if len(rectangle_ids) != len(set(rectangle_ids)):
            raise ValueError("label rectangles require unique label IDs")
        label_component_ids = tuple(item[0] for item in self.label_semantic_ids)
        if self.label_semantic_ids != tuple(sorted(self.label_semantic_ids)):
            raise ValueError("label-to-semantic bindings must be canonically ordered")
        if len(label_component_ids) != len(set(label_component_ids)):
            raise ValueError("label-to-semantic bindings require unique label IDs")
        if set(label_component_ids) != set(rectangle_ids):
            raise ValueError("every label rectangle requires one semantic binding")
        for values, name in (
            (self.visible_semantic_ids, "visible semantic IDs"),
            (self.irrelevant_semantic_ids, "irrelevant semantic IDs"),
        ):
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{name} must be sorted and unique")
        discoverability_ids = tuple(item[0] for item in self.rare_pivotal_discoverability)
        if self.rare_pivotal_discoverability != tuple(
            sorted(self.rare_pivotal_discoverability)
        ) or len(discoverability_ids) != len(set(discoverability_ids)):
            raise ValueError("rare-pivotal discoverability must be canonical and unique")
        if not set(self.irrelevant_semantic_ids).issubset(self.visible_semantic_ids):
            raise ValueError("irrelevant visible IDs must be a subset of visible IDs")
        if not {item[1] for item in self.label_semantic_ids}.issubset(
            self.visible_semantic_ids
        ):
            raise ValueError("label bindings must reference visible semantics")
        return self


class CommunityMetricPanel(ImmutableRecord):
    agreement_base: PartitionAgreement
    agreement_half: PartitionAgreement
    agreement_double: PartitionAgreement
    conductance_base: ConductanceResult
    conductance_half: ConductanceResult
    conductance_double: ConductanceResult


class CrossSeedCommunityScore(ImmutableRecord):
    """Mention-aligned stability for exactly one paired construction-seed cell."""

    scorer_plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    condition: ConditionName
    world_id: str = Field(min_length=1)
    context_id: str = Field(min_length=1)
    seed_blocks: tuple[int, int]
    first_projection_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    second_projection_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    stability: CrossSeedStability | None = None
    metric_rows: tuple[MetricResultRow, ...]

    @model_validator(mode="after")
    def seed_pair_and_validity_are_exact(self) -> Self:
        if self.seed_blocks != tuple(sorted(set(self.seed_blocks))):
            raise ValueError("cross-seed stability requires two distinct ordered seeds")
        if (self.first_projection_hash is None) != (self.second_projection_hash is None):
            raise ValueError("cross-seed projection hashes must be jointly present or absent")
        if self.stability is not None and self.first_projection_hash is None:
            raise ValueError("defined cross-seed stability requires two valid projections")
        return self


class ContrastMetricPlan(ImmutableRecord):
    """Frozen scorer-only A-to-B delta and unchanged-fact targets."""

    scorer_namespace: Literal["scorer_only"] = "scorer_only"
    plan_id: str = Field(min_length=1)
    world_id: str = Field(min_length=1)
    before_context_id: str = Field(min_length=1)
    after_context_id: str = Field(min_length=1)
    before_scorer_plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    after_scorer_plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    gold_changes: tuple[SignedDecisionChange, ...]
    invariants: tuple[GoldContrastInvariant, ...]

    @model_validator(mode="after")
    def targets_are_canonical(self) -> Self:
        if self.before_context_id == self.after_context_id:
            raise ValueError("contrast plan requires two distinct contexts")
        if self.gold_changes != tuple(
            sorted(self.gold_changes, key=lambda item: canonical_json(item))
        ):
            raise ValueError("contrast changes must be canonically ordered")
        invariant_ids = tuple(item.invariant_id for item in self.invariants)
        if self.invariants != tuple(
            sorted(self.invariants, key=lambda item: item.invariant_id)
        ) or len(invariant_ids) != len(set(invariant_ids)):
            raise ValueError("contrast invariants must be canonically ordered and unique")
        return self


class ContrastPairScore(ImmutableRecord):
    plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    condition: ConditionName
    seed_block: int | None = Field(default=None, gt=0)
    before_projection_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    after_projection_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    changes: ContrastiveScore
    invariants: ContrastInvariantScore
    invariant_compilation_issues: tuple[InvariantCompilationIssue, ...] = ()
    metric_rows: tuple[MetricResultRow, ...]

    @model_validator(mode="after")
    def projections_are_joint(self) -> Self:
        if (self.before_projection_hash is None) != (self.after_projection_hash is None):
            raise ValueError("contrast projection hashes must be jointly present or absent")
        return self


class ScorerOnlyProjectionReplayMaterial(ImmutableRecord):
    """Gold-free semantic source retained only inside the restricted scorer bundle.

    Phase 4's metric adapter deliberately discards reviewer-readable labels and
    qualifications.  Retaining the *exact* already-scored projection together with
    its condition-neutral query and evidence inputs lets later blinded review replay
    that adapter and render assessable panels without consulting an unbound side
    channel.  This record is scorer-only even when its synthetic children would each
    otherwise be publishable.
    """

    scorer_namespace: Literal["scorer_only"] = "scorer_only"
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED
    projection: OntologyProjection
    context: QueryContext
    evidence_packet: EvidencePacket

    @model_validator(mode="after")
    def semantic_inputs_are_exactly_bound(self) -> Self:
        bindings = (
            ("context", self.projection.context_hash, self.context.content_hash),
            ("packet", self.projection.packet_hash, self.evidence_packet.content_hash),
            (
                "snapshot",
                self.projection.snapshot_hash,
                self.evidence_packet.snapshot_hash,
            ),
        )
        for name, projection_hash, source_hash in bindings:
            if projection_hash != source_hash:
                raise ValueError(
                    f"scorer-only replay {name} differs from the scored projection"
                )
        if self.projection.budgets != self.context.budgets:
            raise ValueError("scorer-only replay context and projection budgets differ")
        packet_evidence_ids = set(self.evidence_packet.ordered_evidence_ids)
        emitted_evidence_ids = {
            evidence_id
            for item in (
                *self.projection.local_schema.contextual_types,
                *self.projection.local_schema.predicates,
                *self.projection.instance_graph.entities,
                *self.projection.instance_graph.events,
                *self.projection.instance_graph.proposition_contents,
                *self.projection.instance_graph.assertions,
                *self.projection.decisions,
            )
            for evidence_id in item.evidence_ids
        }
        if not emitted_evidence_ids.issubset(packet_evidence_ids):
            raise ValueError("scorer-only replay projection cites evidence outside its packet")
        return self


class CompleteProjectionScoreBundle(ImmutableRecord):
    projection: ProjectionScoreBundle
    scorer_only_replay: ScorerOnlyProjectionReplayMaterial
    rare_pivotal: RarePivotalScore
    clutter: ClutterProfile | None = None
    community: CommunityMetricPanel | None = None
    metric_rows: tuple[MetricResultRow, ...]

    @model_validator(mode="after")
    def replay_material_names_the_scored_projection(self) -> Self:
        replay_projection = self.scorer_only_replay.projection
        if replay_projection.content_hash != self.projection.adapter.projection_hash:
            raise ValueError("scorer-only replay projection differs from the metric adapter")
        if replay_projection.projection_id != self.projection.adapter.projection_id:
            raise ValueError("scorer-only replay projection ID differs from the metric adapter")
        return self


class IntendedMetricUnit(ImmutableRecord):
    """One preregistered cell that must receive a result even when generation fails."""

    unit_id: str = Field(min_length=1)
    job_id: str = Field(min_length=1)
    condition: ConditionName
    world_id: str = Field(min_length=1)
    context_id: str = Field(min_length=1)
    seed_block: int | None = Field(default=None, gt=0)
    snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    packet_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    context_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    scorer_plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    relevant_node_gold_count: int = Field(ge=0)
    strict_assertion_gold_count: int = Field(ge=0)
    ontology_decision_gold_count: int = Field(ge=0)
    rare_pivotal_gold_count: int = Field(ge=0)

    @model_validator(mode="after")
    def seed_shape_matches_condition(self) -> Self:
        if self.condition is ConditionName.C0_CLASSICAL_PRE:
            if self.seed_block is not None:
                raise ValueError("deterministic C0 intended units must be unseeded")
        elif self.seed_block is None:
            raise ValueError("LLM-condition intended units require a seed block")
        return self


class IntendedMetricManifest(ImmutableRecord):
    """Frozen denominator of intended cells; output existence cannot change membership."""

    manifest_id: str = Field(min_length=1)
    units: tuple[IntendedMetricUnit, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def cells_are_canonical_and_unique(self) -> Self:
        if self.units != tuple(sorted(self.units, key=lambda item: item.unit_id)):
            raise ValueError("intended metric units must be canonically ordered")
        unit_ids = tuple(item.unit_id for item in self.units)
        job_ids = tuple(item.job_id for item in self.units)
        cells = tuple(
            (item.condition, item.world_id, item.context_id, item.seed_block)
            for item in self.units
        )
        if len(unit_ids) != len(set(unit_ids)):
            raise ValueError("intended metric manifest contains duplicate unit IDs")
        if len(job_ids) != len(set(job_ids)):
            raise ValueError("intended metric manifest contains duplicate job IDs")
        if len(cells) != len(set(cells)):
            raise ValueError("intended metric manifest contains duplicate condition cells")
        return self


class OutputFailureKind(StrEnum):
    MISSING = "missing"
    TIMEOUT = "timeout"
    REFUSED = "refused"
    PARSE_INVALID = "parse_invalid"
    VALIDATION_INVALID = "validation_invalid"
    UNREPAIRED = "unrepaired"
    EXECUTION_FAILURE = "execution_failure"


class FailedMetricOutput(ImmutableRecord):
    intended_unit: IntendedMetricUnit
    failure_kind: OutputFailureKind
    failure_artifact_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    allocated_gpu_seconds: float = Field(default=0.0, ge=0.0)
    output_valid: Literal[False] = False


class IntendedUnitScore(ImmutableRecord):
    intended_unit: IntendedMetricUnit
    output_valid: bool
    projection_id: str | None = None
    projection_bundle_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    failure_kind: OutputFailureKind | None = None
    failure_artifact_hash: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    allocated_gpu_seconds: float = Field(default=0.0, ge=0.0)
    scoring_diagnostics: tuple[str, ...] = ()
    rows: tuple[MetricResultRow, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validity_and_lineage_are_joint(self) -> Self:
        if self.output_valid:
            if self.projection_id is None or self.projection_bundle_hash is None:
                raise ValueError("valid intended-unit scores require projection lineage")
            if self.failure_kind is not None:
                raise ValueError("valid intended-unit scores cannot carry a failure kind")
            if self.failure_artifact_hash is not None or self.allocated_gpu_seconds != 0.0:
                raise ValueError("valid intended-unit scores cannot carry failed-output lineage")
        elif self.failure_kind is None:
            raise ValueError("invalid intended-unit scores require a failure kind")
        if any(row.projection_id != self.projection_id for row in self.rows):
            raise ValueError("metric rows must agree with intended-unit projection lineage")
        metric_names = tuple(row.metric_name for row in self.rows)
        if len(metric_names) != len(set(metric_names)):
            raise ValueError("intended-unit metric rows must have unique metric names")
        if len({row.metric_version_hash for row in self.rows}) != 1:
            raise ValueError("intended-unit metric rows must use one metric version")
        return self


def _require_metric_version(
    score: IntendedUnitScore,
    configuration: StudyMetricConfiguration,
) -> None:
    expected = configuration.metric_version_hash
    if any(row.metric_version_hash != expected for row in score.rows):
        raise ValueError(
            f"intended-unit score {score.intended_unit.unit_id!r} does not use the "
            "active metric formula/configuration version"
        )


def _status_and_value(value: float | int | None) -> tuple[PipelineMetricStatus, float | int | None]:
    if value is None:
        return PipelineMetricStatus.NOT_APPLICABLE, None
    return PipelineMetricStatus.VALUE, value


def _metric_row(
    *,
    adapter: ProjectionMetricAdapter,
    configuration: StudyMetricConfiguration,
    name: str,
    value: float | int | None,
    numerator: float | int | None = None,
    denominator: float | int | None = None,
    force_status: PipelineMetricStatus | None = None,
    unit_hash: str | None = None,
) -> MetricResultRow:
    status, materialized = _status_and_value(value)
    if force_status is not None:
        status = force_status
        materialized = value if force_status is PipelineMetricStatus.VALUE else None
    return MetricResultRow(
        projection_id=adapter.projection_id,
        unit_hash=unit_hash or adapter.projection_hash,
        metric_name=name,
        metric_version_hash=configuration.metric_version_hash,
        status=status,
        value=materialized,
        numerator=numerator,
        denominator=denominator,
    )


def score_projection(
    projection: OntologyProjection,
    *,
    configuration: StudyMetricConfiguration,
    alignment_plan: AlignmentPlan,
    gold_decisions: Sequence[NormalizedDecision],
    valid_evidence_ids: frozenset[str],
    grounding_by_assertion_id: Mapping[str, GroundingStatus],
    predicate_aliases: Mapping[str, str] | None = None,
    gold_projection: GoldContextualProjection | None = None,
    compute_community: bool = True,
) -> ProjectionScoreBundle:
    """Score one projection and materialize rows ready for ``Ledger.record_metric``.

    Structural metrics fail closed whenever the projection lacks an all-accepted
    validation lineage. Semantic metrics retain the registered intention-to-treat
    behavior by receiving ``invalid_semantic_output=True``.
    """

    adapter = adapt_projection_for_metrics(projection, configuration)
    decision_compilation = (
        compile_and_verify_decision_states(
            gold=gold_projection,
            projection=projection,
            emitted=adapter.normalized_decisions,
            gold_targets=gold_decisions,
            predicate_aliases=predicate_aliases,
        )
        if gold_projection is not None
        else None
    )
    scoring_unit_hash = canonical_sha256(
        {
            "projection_hash": projection.content_hash,
            "alignment_plan_hash": alignment_plan.content_hash,
            "gold_decisions": tuple(gold_decisions),
            "gold_projection_hash": (
                gold_projection.content_hash if gold_projection is not None else None
            ),
            "decision_compilation_hash": (
                decision_compilation.content_hash
                if decision_compilation is not None
                else None
            ),
            "valid_evidence_ids": tuple(sorted(valid_evidence_ids)),
            "grounding_by_assertion_id": tuple(
                sorted((key, value.value) for key, value in grounding_by_assertion_id.items())
            ),
            "predicate_aliases": tuple(sorted((predicate_aliases or {}).items())),
            "metric_configuration_hash": configuration.content_hash,
            "metric_formula_revision": configuration.metric_formula_revision,
        }
    )
    predicted_nodes, predicted_assertions, _ = prediction_records(
        projection,
        predicate_aliases=predicate_aliases,
        valid_evidence_ids=valid_evidence_ids,
        grounding_by_assertion_id=grounding_by_assertion_id,
        plan=alignment_plan,
    )
    semantic_output_valid = (
        adapter.structurally_valid and projection_is_content_bearing(projection)
    )
    alignment = score_alignment(
        plan=alignment_plan,
        predicted_nodes=predicted_nodes,
        predicted_assertions=predicted_assertions,
        invalid_semantic_output=not semantic_output_valid,
    )
    scored_gold_decisions = (
        decision_compilation.gold_decisions
        if decision_compilation is not None
        else tuple(gold_decisions)
    )
    scored_predicted_decisions = (
        decision_compilation.predicted_decisions
        if decision_compilation is not None
        else verify_projection_decisions(projection, gold_targets=gold_decisions)
    )
    decision_score = score_ontology_decisions(
        gold=scored_gold_decisions,
        predicted=scored_predicted_decisions,
        invalid_semantic_output=not semantic_output_valid,
    )
    if not adapter.structurally_valid:
        structural = StructuralMetricPanel(
            valid_content_bearing=False,
            failure_reason="projection validation lineage is absent or contains a rejection",
        )
    elif not adapter.node_ids:
        structural = StructuralMetricPanel(
            valid_content_bearing=False,
            failure_reason="projection contains no entity/event node",
        )
    else:
        entropy = entropy_profile(
            adapter.node_ids,
            adapter.edges,
            canonical_vocabulary=configuration.canonical_relation_vocabulary,
            assertion_relations=adapter.assertion_relations,
        )
        if compute_community:
            base = run_leiden_cpm(
                adapter.node_ids,
                adapter.edges,
                LeidenConfiguration(
                    resolution=configuration.leiden_base_resolution,
                    seed=configuration.leiden_seed,
                ),
            )
            half = run_leiden_cpm(
                adapter.node_ids,
                adapter.edges,
                LeidenConfiguration(
                    resolution=configuration.leiden_half_resolution,
                    seed=configuration.leiden_seed,
                ),
            )
            double = run_leiden_cpm(
                adapter.node_ids,
                adapter.edges,
                LeidenConfiguration(
                    resolution=configuration.leiden_double_resolution,
                    seed=configuration.leiden_seed,
                ),
            )
        else:
            base = half = double = None
        structural = StructuralMetricPanel(
            valid_content_bearing=True,
            entropy=entropy,
            leiden_base=base,
            leiden_half=half,
            leiden_double=double,
        )

    rows: list[MetricResultRow] = []
    for prefix, score in (
        ("contextual_node", alignment.node_score),
        ("strict_qualified_assertion", alignment.strict_assertion_score),
    ):
        for suffix, value in (
            ("precision", score.precision),
            ("recall", score.recall),
            ("f1", score.f1),
        ):
            rows.append(
                _metric_row(
                    adapter=adapter,
                    configuration=configuration,
                    name=f"{prefix}_{suffix}",
                    value=value,
                    numerator=score.true_positive_count,
                    denominator=(
                        score.precision_denominator
                        if suffix == "precision"
                        else score.recall_denominator
                        if suffix == "recall"
                        else None
                    ),
                    unit_hash=scoring_unit_hash,
                )
            )
    for name, result in (
        ("essential_temporal_qualification_accuracy", alignment.essential_temporal_accuracy),
        ("evidence_citation_validity", alignment.evidence_citation_validity),
        ("grounding_precision", alignment.grounding_precision),
        ("unsupported_assertion_rate", alignment.unsupported_assertion_rate),
    ):
        rows.append(
            _metric_row(
                adapter=adapter,
                configuration=configuration,
                name=name,
                value=result.value,
                numerator=result.numerator,
                denominator=result.denominator,
                unit_hash=scoring_unit_hash,
            )
        )
    rows.append(
        _metric_row(
            adapter=adapter,
            configuration=configuration,
            name="ontology_decision_macro_f1",
            value=decision_score.macro_f1,
            denominator=decision_score.active_family_count,
            unit_hash=scoring_unit_hash,
        )
    )
    for name, value in (
        (
            "ontology_decision_unrepresentable_gold_count",
            decision_compilation.unrepresentable_gold_count
            if decision_compilation is not None
            else 0,
        ),
        (
            "ontology_decision_unrepresentable_prediction_count",
            decision_compilation.unrepresentable_prediction_count
            if decision_compilation is not None
            else 0,
        ),
    ):
        rows.append(
            _metric_row(
                adapter=adapter,
                configuration=configuration,
                name=name,
                value=value,
                unit_hash=scoring_unit_hash,
            )
        )
    for component in decision_score.components:
        for suffix, value, denominator in (
            ("precision", component.score.precision, component.score.precision_denominator),
            ("recall", component.score.recall, component.score.recall_denominator),
            ("f1", component.score.f1, None),
        ):
            rows.append(
                _metric_row(
                    adapter=adapter,
                    configuration=configuration,
                    name=f"ontology_decision_{component.family.value}_{suffix}",
                    value=value,
                    numerator=component.score.true_positive_count,
                    denominator=denominator,
                    unit_hash=scoring_unit_hash,
                )
            )
    if structural.entropy is None:
        for name in (
            "degree_histogram_entropy",
            "degree_histogram_entropy_raw",
            "degree_histogram_support_size",
            "occupied_degree_bin_count",
            "degree_histogram_denominator",
            "degree_mass_entropy",
            "degree_mass_entropy_raw",
            "degree_mass_support_size",
            "degree_mass_denominator",
            "local_relation_neighborhood_entropy",
            "local_relation_neighborhood_entropy_raw",
            "local_relation_neighborhood_support_size",
            "local_relation_neighborhood_denominator",
            "native_schema_relation_entropy",
            "native_schema_relation_entropy_raw",
            "native_schema_relation_support_size",
            "active_native_relation_count",
            "native_schema_relation_denominator",
            "canonical_mapped_relation_entropy",
            "canonical_mapped_relation_entropy_raw",
            "canonical_mapped_relation_support_size",
            "canonical_mapped_relation_denominator",
            "canonical_relation_vocabulary_size",
            "canonical_other_count",
            "canonical_other_rate",
            "isolate_count",
            "positive_degree_node_count",
            "node_count",
            "assertion_edge_count",
            "topology_edge_count",
            "leiden_cluster_count_half",
            "leiden_cluster_count_base",
            "leiden_cluster_count_double",
            "leiden_modularity_half",
            "leiden_modularity_base",
            "leiden_modularity_double",
        ):
            rows.append(
                _metric_row(
                    adapter=adapter,
                    configuration=configuration,
                    name=name,
                    value=None,
                    force_status=(
                        PipelineMetricStatus.INVALID
                        if not adapter.structurally_valid
                        else PipelineMetricStatus.UNDEFINED
                    ),
                    unit_hash=scoring_unit_hash,
                )
            )
    else:
        entropy = structural.entropy
        undefined_entropy_names = {
            "degree_histogram_entropy",
            "degree_histogram_entropy_raw",
            "degree_histogram_denominator",
            "degree_mass_entropy",
            "degree_mass_entropy_raw",
            "degree_mass_denominator",
            "local_relation_neighborhood_entropy",
            "local_relation_neighborhood_entropy_raw",
            "local_relation_neighborhood_denominator",
            "native_schema_relation_entropy",
            "native_schema_relation_entropy_raw",
            "native_schema_relation_denominator",
            "canonical_mapped_relation_entropy",
            "canonical_mapped_relation_entropy_raw",
            "canonical_mapped_relation_denominator",
            "canonical_other_rate",
            "leiden_modularity_half",
            "leiden_modularity_base",
            "leiden_modularity_double",
        }
        for name, value in (
            ("degree_histogram_entropy", entropy.degree_histogram.normalized),
            ("degree_histogram_entropy_raw", entropy.degree_histogram.raw),
            ("degree_histogram_support_size", entropy.degree_histogram.support_size),
            ("occupied_degree_bin_count", entropy.occupied_degree_bin_count),
            ("degree_histogram_denominator", entropy.degree_histogram.denominator),
            ("degree_mass_entropy", entropy.degree_mass.normalized),
            ("degree_mass_entropy_raw", entropy.degree_mass.raw),
            ("degree_mass_support_size", entropy.degree_mass.support_size),
            ("degree_mass_denominator", entropy.degree_mass.denominator),
            (
                "local_relation_neighborhood_entropy",
                entropy.mean_local_relation_neighborhood.normalized,
            ),
            (
                "local_relation_neighborhood_entropy_raw",
                entropy.mean_local_relation_neighborhood.raw,
            ),
            (
                "local_relation_neighborhood_support_size",
                entropy.mean_local_relation_neighborhood.support_size,
            ),
            (
                "local_relation_neighborhood_denominator",
                entropy.mean_local_relation_neighborhood.denominator,
            ),
            ("native_schema_relation_entropy", entropy.native_schema_relation.normalized),
            ("native_schema_relation_entropy_raw", entropy.native_schema_relation.raw),
            ("native_schema_relation_support_size", entropy.native_schema_relation.support_size),
            ("active_native_relation_count", entropy.active_native_relation_count),
            ("native_schema_relation_denominator", entropy.native_schema_relation.denominator),
            ("canonical_mapped_relation_entropy", entropy.canonical_mapped_relation.normalized),
            ("canonical_mapped_relation_entropy_raw", entropy.canonical_mapped_relation.raw),
            (
                "canonical_mapped_relation_support_size",
                entropy.canonical_mapped_relation.support_size,
            ),
            (
                "canonical_mapped_relation_denominator",
                entropy.canonical_mapped_relation.denominator,
            ),
            ("canonical_relation_vocabulary_size", entropy.canonical_vocabulary_size),
            ("canonical_other_count", entropy.canonical_other_count),
            ("canonical_other_rate", entropy.canonical_other_rate),
            ("isolate_count", entropy.isolate_count),
            ("positive_degree_node_count", entropy.positive_degree_node_count),
            ("node_count", entropy.node_count),
            ("assertion_edge_count", entropy.assertion_edge_count),
            ("topology_edge_count", entropy.topology_edge_count),
            (
                "leiden_cluster_count_half",
                structural.leiden_half.cluster_count if structural.leiden_half else None,
            ),
            (
                "leiden_cluster_count_base",
                structural.leiden_base.cluster_count if structural.leiden_base else None,
            ),
            (
                "leiden_cluster_count_double",
                structural.leiden_double.cluster_count if structural.leiden_double else None,
            ),
            (
                "leiden_modularity_half",
                structural.leiden_half.modularity if structural.leiden_half else None,
            ),
            (
                "leiden_modularity_base",
                structural.leiden_base.modularity if structural.leiden_base else None,
            ),
            (
                "leiden_modularity_double",
                structural.leiden_double.modularity if structural.leiden_double else None,
            ),
        ):
            rows.append(
                _metric_row(
                    adapter=adapter,
                    configuration=configuration,
                    name=name,
                    value=value,
                    force_status=(
                        PipelineMetricStatus.UNDEFINED
                        if value is None and name in undefined_entropy_names
                        else None
                    ),
                    unit_hash=scoring_unit_hash,
                )
            )
    return ProjectionScoreBundle(
        scoring_input_hash=scoring_unit_hash,
        adapter=adapter,
        alignment=alignment,
        ontology_decisions=decision_score,
        decision_compilation=decision_compilation,
        structural=structural,
        metric_rows=tuple(rows),
    )


def _partition_on_gold_anchors(
    partition: LeidenPartition,
    alignment: AlignmentResult,
) -> dict[str, str]:
    """Map local Leiden memberships through condition-blind node alignment."""

    local_assignments = partition.by_node
    result: dict[str, str] = {}
    for match in alignment.all_anchor_node_matches:
        cluster = local_assignments.get(match.prediction_id)
        if cluster is None:
            continue
        previous = result.setdefault(match.target_id, cluster)
        if previous != cluster:
            raise ValueError("one gold anchor mapped to conflicting predicted communities")
    return result


def score_complete_projection(
    projection: OntologyProjection,
    *,
    context: QueryContext,
    evidence_packet: EvidencePacket,
    configuration: StudyMetricConfiguration,
    scorer_plan: ScorerMetricPlan,
    grounding_audit: GroundingAuditInput,
    geometry: GeometryMetricInput | None,
) -> CompleteProjectionScoreBundle:
    """Run and materialize the complete registered per-projection metric panel.

    Pair-level contrast/paraphrase and cross-seed stability are intentionally handled
    by their paired orchestrators; every metric that is defined on one projection is
    produced here from immutable, content-addressed inputs.
    """

    if not (
        projection_is_structurally_valid(projection)
        and projection_is_content_bearing(projection)
    ):
        raise ValueError(
            "complete projection scoring requires structurally valid, content-bearing output"
        )

    if grounding_audit.projection_hash != projection.content_hash:
        raise ValueError("grounding audit names a different projection")
    if (
        grounding_audit.valid_evidence_manifest_hash
        != scorer_plan.valid_evidence_manifest_hash
    ):
        raise ValueError("grounding audit and scorer plan use different evidence manifests")
    assertion_ids = tuple(
        sorted(item.assertion_id for item in projection.instance_graph.assertions)
    )
    status_ids = tuple(item[0] for item in grounding_audit.assertion_statuses)
    if status_ids != assertion_ids:
        raise ValueError("grounding audit must cover every and only emitted assertion")
    if geometry is not None:
        if geometry.projection_hash != projection.content_hash:
            raise ValueError("geometry names a different projection")
        renderer = configuration.renderer
        expected_hashes = (
            renderer.layout_config_hash,
            renderer.style_config_hash,
            renderer.font_config_hash,
            renderer.viewport_hash,
        )
        actual_hashes = (
            geometry.layout_config_hash,
            geometry.style_config_hash,
            geometry.font_config_hash,
            geometry.viewport_hash,
        )
        if actual_hashes != expected_hashes:
            raise ValueError("geometry does not use the frozen renderer configuration")

    base = score_projection(
        projection,
        configuration=configuration,
        alignment_plan=scorer_plan.alignment_plan,
        gold_decisions=scorer_plan.gold_decisions,
        valid_evidence_ids=frozenset(scorer_plan.valid_evidence_ids),
        grounding_by_assertion_id=dict(grounding_audit.assertion_statuses),
        predicate_aliases=dict(scorer_plan.predicate_aliases),
        gold_projection=scorer_plan.gold_projection,
    )
    matched_assertions = frozenset(
        item.target_id for item in base.alignment.strict_assertion_matches
    )
    rare = score_rare_pivotal(
        annotations=scorer_plan.rare_annotations,
        strictly_matched_assertion_target_ids=matched_assertions,
        invalid_semantic_output=not base.adapter.structurally_valid,
    )
    complete_input_hash = canonical_sha256(
        {
            "base_scoring_input_hash": base.scoring_input_hash,
            "scorer_plan_hash": scorer_plan.content_hash,
            "grounding_audit_hash": grounding_audit.content_hash,
            "geometry_hash": geometry.content_hash if geometry is not None else None,
        }
    )
    rows = list(base.metric_rows)

    def append(
        name: str,
        value: float | int | None,
        *,
        numerator: float | int | None = None,
        denominator: float | int | None = None,
        status: PipelineMetricStatus | None = None,
    ) -> None:
        rows.append(
            _metric_row(
                adapter=base.adapter,
                configuration=configuration,
                name=name,
                value=value,
                numerator=numerator,
                denominator=denominator,
                force_status=status,
                unit_hash=complete_input_hash,
            )
        )

    for name, result in (
        ("rare_pivotal_qualified_assertion_recall", rare.qualified_assertion_recall),
        ("rare_pivotal_support_path_survival", rare.complete_support_path_survival),
    ):
        append(
            name,
            result.value,
            numerator=result.numerator,
            denominator=result.denominator,
        )
    for stratum in rare.stratified_qualified_assertion_recall:
        result = stratum.qualified_assertion_recall
        append(
            f"qualified_assertion_recall_{stratum.stratum.value}",
            result.value,
            numerator=result.numerator,
            denominator=result.denominator,
        )

    clutter: ClutterProfile | None = None
    clutter_metric_names = (
        "density",
        "component_count",
        "label_overlap_count",
        "label_overlap_area",
        "irrelevant_visible_load",
        "crossing_opportunity_indicator",
        "crossing_opportunity_count",
        "crossing_count",
        "conditional_crossing_rate",
        "rare_pivotal_discoverability_mean_interactions",
        "rare_pivotal_discoverability_max_interactions",
        "rare_pivotal_discovered_count",
        "rare_pivotal_omitted_count",
    )
    if geometry is None:
        for name in clutter_metric_names:
            append(name, None, status=PipelineMetricStatus.NOT_APPLICABLE)
    else:
        rare_targets = tuple(
            item.assertion_target_id
            for item in scorer_plan.rare_annotations
            if item.is_rare and item.is_pivotal
        )
        clutter = clutter_profile(
            base.adapter.node_ids,
            base.adapter.edges,
            positions=dict(geometry.positions),
            label_rectangles=geometry.label_rectangles,
            label_semantic_ids=dict(geometry.label_semantic_ids),
            visible_semantic_ids=geometry.visible_semantic_ids,
            irrelevant_semantic_ids=frozenset(geometry.irrelevant_semantic_ids),
            rare_pivotal_discoverability=dict(geometry.rare_pivotal_discoverability),
            rare_pivotal_target_ids=rare_targets,
            structurally_valid=base.adapter.structurally_valid,
            assertion_record_count=len(base.adapter.assertion_relations),
            assertion_semantic_ids=tuple(
                item.assertion_id for item in base.adapter.assertion_relations
            ),
        )
        clutter_values: tuple[
            tuple[str, float | int | None, float | int | None, float | int | None], ...
        ] = (
            ("density", clutter.density, None, None),
            ("component_count", clutter.component_count, None, None),
            ("label_overlap_count", clutter.label_overlap_count, None, None),
            ("label_overlap_area", clutter.label_overlap_area, None, None),
            (
                "irrelevant_visible_load",
                clutter.irrelevant_visible_fraction,
                clutter.irrelevant_visible_count,
                clutter.visible_semantic_count,
            ),
            (
                "crossing_opportunity_indicator",
                clutter.crossings.opportunity_indicator,
                None,
                None,
            ),
            (
                "crossing_opportunity_count",
                clutter.crossings.opportunity_count,
                None,
                None,
            ),
            ("crossing_count", clutter.crossings.crossing_count, None, None),
            (
                "conditional_crossing_rate",
                clutter.crossings.conditional_crossing_rate,
                clutter.crossings.crossing_count,
                clutter.crossings.opportunity_count,
            ),
            (
                "rare_pivotal_discoverability_mean_interactions",
                clutter.rare_pivotal_discoverability_mean_interactions,
                None,
                clutter.rare_pivotal_discoverability_denominator,
            ),
            (
                "rare_pivotal_discoverability_max_interactions",
                clutter.rare_pivotal_discoverability_max_interactions,
                None,
                clutter.rare_pivotal_discoverability_denominator,
            ),
            (
                "rare_pivotal_discovered_count",
                clutter.rare_pivotal_discovered_count,
                clutter.rare_pivotal_discovered_count,
                clutter.rare_pivotal_discoverability_denominator,
            ),
            (
                "rare_pivotal_omitted_count",
                clutter.rare_pivotal_omitted_count,
                clutter.rare_pivotal_omitted_count,
                clutter.rare_pivotal_discoverability_denominator,
            ),
        )
        structural_status = (
            None
            if clutter.valid_content_bearing
            else PipelineMetricStatus.INVALID
            if not base.adapter.structurally_valid
            else PipelineMetricStatus.UNDEFINED
        )
        for name, value, numerator, denominator in clutter_values:
            append(
                name,
                value,
                numerator=numerator,
                denominator=denominator,
                status=structural_status,
            )

    community_panel: CommunityMetricPanel | None = None
    base_community_names = (
        "community_adjusted_mutual_information",
        "community_purity",
        "community_mean_conductance",
        "community_conductance_defined_cluster_count",
        "community_conductance_undefined_cluster_count",
        "community_fragmentation_error",
        "community_merging_error",
        "community_omitted_anchor_count",
    )
    community_names = (
        *base_community_names,
        *(f"{name}_half" for name in base_community_names),
        *(f"{name}_double" for name in base_community_names),
    )
    if (
        scorer_plan.community_eligible
        and base.structural.valid_content_bearing
        and base.structural.leiden_base is not None
        and base.structural.leiden_half is not None
        and base.structural.leiden_double is not None
    ):
        gold_assignments = dict(scorer_plan.gold_community_assignments)
        anchors = tuple(gold_assignments)
        partitions = {
            "half": base.structural.leiden_half,
            "base": base.structural.leiden_base,
            "double": base.structural.leiden_double,
        }
        agreements = {
            key: score_partition(
                gold_assignments=gold_assignments,
                predicted_assignments=_partition_on_gold_anchors(partition, base.alignment),
                anchors=anchors,
            )
            for key, partition in partitions.items()
        }
        conductance_results = {
            key: conductance(
                base.adapter.node_ids,
                base.adapter.edges,
                partition.by_node,
            )
            for key, partition in partitions.items()
        }
        community_panel = CommunityMetricPanel(
            agreement_base=agreements["base"],
            agreement_half=agreements["half"],
            agreement_double=agreements["double"],
            conductance_base=conductance_results["base"],
            conductance_half=conductance_results["half"],
            conductance_double=conductance_results["double"],
        )
        for resolution in ("half", "base", "double"):
            agreement = agreements[resolution]
            conductance_result = conductance_results[resolution]
            suffix = "" if resolution == "base" else f"_{resolution}"
            for name, value, numerator, denominator in (
                (
                    f"community_adjusted_mutual_information{suffix}",
                    agreement.adjusted_mutual_information,
                    None,
                    agreement.anchor_count,
                ),
                (
                    f"community_purity{suffix}",
                    agreement.purity,
                    None,
                    agreement.anchor_count,
                ),
                (
                    f"community_mean_conductance{suffix}",
                    conductance_result.unweighted_mean,
                    None,
                    conductance_result.defined_cluster_count,
                ),
                (
                    f"community_conductance_defined_cluster_count{suffix}",
                    conductance_result.defined_cluster_count,
                    conductance_result.defined_cluster_count,
                    (
                        conductance_result.defined_cluster_count
                        + conductance_result.undefined_cluster_count
                    ),
                ),
                (
                    f"community_conductance_undefined_cluster_count{suffix}",
                    conductance_result.undefined_cluster_count,
                    conductance_result.undefined_cluster_count,
                    (
                        conductance_result.defined_cluster_count
                        + conductance_result.undefined_cluster_count
                    ),
                ),
                (
                    f"community_fragmentation_error{suffix}",
                    agreement.fragmentation_error,
                    None,
                    agreement.fragmentation_pair_denominator,
                ),
                (
                    f"community_merging_error{suffix}",
                    agreement.merging_error,
                    None,
                    agreement.merging_pair_denominator,
                ),
                (
                    f"community_omitted_anchor_count{suffix}",
                    agreement.omitted_anchor_count,
                    agreement.omitted_anchor_count,
                    agreement.anchor_count,
                ),
            ):
                append(
                    name,
                    value,
                    numerator=numerator,
                    denominator=denominator,
                    status=(
                        PipelineMetricStatus.UNDEFINED if value is None else None
                    ),
                )
    else:
        status = (
            PipelineMetricStatus.NOT_APPLICABLE
            if not scorer_plan.community_eligible
            else PipelineMetricStatus.INVALID
            if not base.adapter.structurally_valid
            else PipelineMetricStatus.UNDEFINED
        )
        for name in community_names:
            append(name, None, status=status)

    return CompleteProjectionScoreBundle(
        projection=base,
        scorer_only_replay=ScorerOnlyProjectionReplayMaterial(
            projection=projection,
            context=context,
            evidence_packet=evidence_packet,
        ),
        rare_pivotal=rare,
        clutter=clutter,
        community=community_panel,
        metric_rows=tuple(rows),
    )


def _validate_intended_unit_against_plan(
    intended: IntendedMetricUnit,
    scorer_plan: ScorerMetricPlan,
) -> None:
    if intended.scorer_plan_hash != scorer_plan.content_hash:
        raise ValueError("intended unit names a different scorer plan")
    expected_counts = (
        sum(item.is_contextually_relevant for item in scorer_plan.alignment_plan.node_targets),
        len(scorer_plan.alignment_plan.assertion_targets),
        len(scorer_plan.gold_decisions),
        sum(item.is_rare and item.is_pivotal for item in scorer_plan.rare_annotations),
    )
    actual_counts = (
        intended.relevant_node_gold_count,
        intended.strict_assertion_gold_count,
        intended.ontology_decision_gold_count,
        intended.rare_pivotal_gold_count,
    )
    if actual_counts != expected_counts:
        raise ValueError("intended-unit gold denominators differ from its scorer plan")


def score_intended_projection(
    intended: IntendedMetricUnit,
    projection: OntologyProjection,
    *,
    context: QueryContext,
    evidence_packet: EvidencePacket,
    configuration: StudyMetricConfiguration,
    scorer_plan: ScorerMetricPlan,
    grounding_audit: GroundingAuditInput,
    geometry: GeometryMetricInput | None,
) -> IntendedUnitScore:
    """Score one successful intended cell after checking all lineage joins."""

    _validate_intended_unit_against_plan(intended, scorer_plan)
    if projection.condition is not intended.condition:
        raise ValueError("projection condition differs from intended unit")
    for name, actual, expected in (
        ("snapshot", projection.snapshot_hash, intended.snapshot_hash),
        ("packet", projection.packet_hash, intended.packet_hash),
        ("context", projection.context_hash, intended.context_hash),
    ):
        if actual != expected:
            raise ValueError(f"projection {name} hash differs from intended unit")
    if not (
        projection_is_structurally_valid(projection)
        and projection_is_content_bearing(projection)
    ):
        return score_failed_output(
            FailedMetricOutput(
                intended_unit=intended,
                failure_kind=OutputFailureKind.VALIDATION_INVALID,
                failure_artifact_hash=projection.content_hash,
            ),
            configuration=configuration,
            scorer_plan=scorer_plan,
        )
    bundle = score_complete_projection(
        projection,
        context=context,
        evidence_packet=evidence_packet,
        configuration=configuration,
        scorer_plan=scorer_plan,
        grounding_audit=grounding_audit,
        geometry=geometry,
    )
    return IntendedUnitScore(
        intended_unit=intended,
        output_valid=True,
        projection_id=projection.projection_id,
        projection_bundle_hash=bundle.content_hash,
        scoring_diagnostics=tuple(
            f"decision_compilation:{item.side}:{item.slot_key}:{item.reason}"
            for item in (
                bundle.projection.decision_compilation.issues
                if bundle.projection.decision_compilation is not None
                else ()
            )
        ),
        rows=bundle.metric_rows,
    )


_FAILED_STRUCTURAL_METRICS = (
    "degree_histogram_entropy",
    "degree_histogram_entropy_raw",
    "degree_histogram_support_size",
    "occupied_degree_bin_count",
    "degree_histogram_denominator",
    "degree_mass_entropy",
    "degree_mass_entropy_raw",
    "degree_mass_support_size",
    "degree_mass_denominator",
    "local_relation_neighborhood_entropy",
    "local_relation_neighborhood_entropy_raw",
    "local_relation_neighborhood_support_size",
    "local_relation_neighborhood_denominator",
    "native_schema_relation_entropy",
    "native_schema_relation_entropy_raw",
    "native_schema_relation_support_size",
    "active_native_relation_count",
    "native_schema_relation_denominator",
    "canonical_mapped_relation_entropy",
    "canonical_mapped_relation_entropy_raw",
    "canonical_mapped_relation_support_size",
    "canonical_mapped_relation_denominator",
    "canonical_relation_vocabulary_size",
    "canonical_other_count",
    "canonical_other_rate",
    "isolate_count",
    "positive_degree_node_count",
    "node_count",
    "assertion_edge_count",
    "topology_edge_count",
    "leiden_cluster_count_half",
    "density",
    "component_count",
    "label_overlap_count",
    "label_overlap_area",
    "irrelevant_visible_load",
    "crossing_opportunity_indicator",
    "crossing_opportunity_count",
    "crossing_count",
    "conditional_crossing_rate",
    "rare_pivotal_discoverability_mean_interactions",
    "rare_pivotal_discoverability_max_interactions",
    "rare_pivotal_discovered_count",
    "rare_pivotal_omitted_count",
    "leiden_cluster_count_base",
    "leiden_cluster_count_double",
    "leiden_modularity_half",
    "leiden_modularity_base",
    "leiden_modularity_double",
    "community_adjusted_mutual_information",
    "community_purity",
    "community_mean_conductance",
    "community_conductance_defined_cluster_count",
    "community_conductance_undefined_cluster_count",
    "community_fragmentation_error",
    "community_merging_error",
    "community_omitted_anchor_count",
    "community_adjusted_mutual_information_half",
    "community_purity_half",
    "community_mean_conductance_half",
    "community_conductance_defined_cluster_count_half",
    "community_conductance_undefined_cluster_count_half",
    "community_fragmentation_error_half",
    "community_merging_error_half",
    "community_omitted_anchor_count_half",
    "community_adjusted_mutual_information_double",
    "community_purity_double",
    "community_mean_conductance_double",
    "community_conductance_defined_cluster_count_double",
    "community_conductance_undefined_cluster_count_double",
    "community_fragmentation_error_double",
    "community_merging_error_double",
    "community_omitted_anchor_count_double",
)


def score_failed_output(
    failure: FailedMetricOutput,
    *,
    configuration: StudyMetricConfiguration,
    scorer_plan: ScorerMetricPlan,
) -> IntendedUnitScore:
    """Materialize ITT semantic zeros and structural NA for a missing projection."""

    intended = failure.intended_unit
    _validate_intended_unit_against_plan(intended, scorer_plan)
    unit_hash = canonical_sha256(
        {
            "intended_unit_hash": intended.content_hash,
            "failure_hash": failure.content_hash,
            "scorer_plan_hash": scorer_plan.content_hash,
            "metric_configuration_hash": configuration.content_hash,
            "metric_formula_revision": configuration.metric_formula_revision,
        }
    )

    def row(
        name: str,
        *,
        value: float | None,
        denominator: int | None,
        status: PipelineMetricStatus,
    ) -> MetricResultRow:
        return MetricResultRow(
            projection_id=None,
            unit_hash=unit_hash,
            metric_name=name,
            metric_version_hash=configuration.metric_version_hash,
            status=status,
            value=value,
            numerator=0 if denominator is not None else None,
            denominator=denominator,
        )

    rows: list[MetricResultRow] = []
    for prefix, gold_count in (
        ("contextual_node", intended.relevant_node_gold_count),
        ("strict_qualified_assertion", intended.strict_assertion_gold_count),
    ):
        status = (
            PipelineMetricStatus.VALUE
            if gold_count > 0
            else PipelineMetricStatus.NOT_APPLICABLE
        )
        for suffix in ("precision", "recall", "f1"):
            denominator = 0 if suffix == "precision" else gold_count if suffix == "recall" else None
            rows.append(
                row(
                    f"{prefix}_{suffix}",
                    value=0.0 if gold_count > 0 else None,
                    denominator=denominator,
                    status=status,
                )
            )
    compiled_failed_gold, failed_gold_issues = (
        compile_gold_decision_targets(
            scorer_plan.gold_projection,
            scorer_plan.gold_decisions,
            predicate_aliases=dict(scorer_plan.predicate_aliases),
        )
        if scorer_plan.gold_projection is not None
        else (scorer_plan.gold_decisions, ())
    )
    decision_score = score_ontology_decisions(
        gold=compiled_failed_gold,
        predicted=(),
        invalid_semantic_output=True,
    )
    rows.append(
        row(
            "ontology_decision_macro_f1",
            value=decision_score.macro_f1,
            denominator=decision_score.active_family_count,
            status=(
                PipelineMetricStatus.VALUE
                if decision_score.macro_f1 is not None
                else PipelineMetricStatus.NOT_APPLICABLE
            ),
        )
    )
    rows.extend(
        (
            row(
                "ontology_decision_unrepresentable_gold_count",
                value=len(failed_gold_issues),
                denominator=None,
                status=PipelineMetricStatus.VALUE,
            ),
            row(
                "ontology_decision_unrepresentable_prediction_count",
                value=0,
                denominator=None,
                status=PipelineMetricStatus.VALUE,
            ),
        )
    )
    for component in decision_score.components:
        for suffix, value, denominator in (
            ("precision", component.score.precision, component.score.precision_denominator),
            ("recall", component.score.recall, component.score.recall_denominator),
            ("f1", component.score.f1, None),
        ):
            rows.append(
                row(
                    f"ontology_decision_{component.family.value}_{suffix}",
                    value=value,
                    denominator=denominator,
                    status=(
                        PipelineMetricStatus.VALUE
                        if value is not None
                        else PipelineMetricStatus.NOT_APPLICABLE
                    ),
                )
            )
    rare = score_rare_pivotal(
        annotations=scorer_plan.rare_annotations,
        strictly_matched_assertion_target_ids=frozenset(),
        invalid_semantic_output=True,
    )
    for name, result in (
        ("rare_pivotal_qualified_assertion_recall", rare.qualified_assertion_recall),
        ("rare_pivotal_support_path_survival", rare.complete_support_path_survival),
    ):
        rows.append(
            row(
                name,
                value=result.value,
                denominator=result.denominator,
                status=(
                    PipelineMetricStatus.VALUE
                    if result.value is not None
                    else PipelineMetricStatus.NOT_APPLICABLE
                ),
            )
        )
    for stratum in rare.stratified_qualified_assertion_recall:
        result = stratum.qualified_assertion_recall
        rows.append(
            row(
                f"qualified_assertion_recall_{stratum.stratum.value}",
                value=result.value,
                denominator=result.denominator,
                status=(
                    PipelineMetricStatus.VALUE
                    if result.value is not None
                    else PipelineMetricStatus.NOT_APPLICABLE
                ),
            )
        )
    essential_temporal_count = sum(
        target.essential_temporal for target in scorer_plan.alignment_plan.assertion_targets
    )
    rows.append(
        row(
            "essential_temporal_qualification_accuracy",
            value=0.0 if essential_temporal_count else None,
            denominator=essential_temporal_count,
            status=(
                PipelineMetricStatus.VALUE
                if essential_temporal_count
                else PipelineMetricStatus.NOT_APPLICABLE
            ),
        )
    )
    for name in (
        "evidence_citation_validity",
        "grounding_precision",
        "unsupported_assertion_rate",
    ):
        rows.append(
            row(
                name,
                value=None,
                denominator=None,
                status=PipelineMetricStatus.NOT_APPLICABLE,
            )
        )
    rows.extend(
        row(
            name,
            value=None,
            denominator=None,
            status=PipelineMetricStatus.INVALID,
        )
        for name in _FAILED_STRUCTURAL_METRICS
    )
    return IntendedUnitScore(
        intended_unit=intended,
        output_valid=False,
        failure_kind=failure.failure_kind,
        failure_artifact_hash=failure.failure_artifact_hash,
        allocated_gpu_seconds=failure.allocated_gpu_seconds,
        scoring_diagnostics=tuple(
            f"decision_compilation:gold:{item.slot_key}:{item.reason}"
            for item in failed_gold_issues
        ),
        rows=tuple(rows),
    )


def score_cross_seed_community(
    first_score: IntendedUnitScore,
    second_score: IntendedUnitScore,
    *,
    first_bundle: CompleteProjectionScoreBundle | None,
    second_bundle: CompleteProjectionScoreBundle | None,
    configuration: StudyMetricConfiguration,
    scorer_plan: ScorerMetricPlan,
) -> CrossSeedCommunityScore:
    """Score the registered construction-seed pair on one fixed gold-anchor universe."""

    first_unit = first_score.intended_unit
    second_unit = second_score.intended_unit
    _require_metric_version(first_score, configuration)
    _require_metric_version(second_score, configuration)
    _validate_intended_unit_against_plan(first_unit, scorer_plan)
    _validate_intended_unit_against_plan(second_unit, scorer_plan)
    identities = (
        first_unit.condition,
        first_unit.world_id,
        first_unit.context_id,
        first_unit.scorer_plan_hash,
    )
    if identities != (
        second_unit.condition,
        second_unit.world_id,
        second_unit.context_id,
        second_unit.scorer_plan_hash,
    ):
        raise ValueError("cross-seed inputs must name one condition/world/context/scorer plan")
    seed_blocks = (first_unit.seed_block, second_unit.seed_block)
    if None in seed_blocks or tuple(sorted(seed_blocks)) != (1, 2):
        raise ValueError("cross-seed stability requires registered seed blocks 1 and 2")
    ordered = sorted(
        (
            (first_unit.seed_block, first_score, first_bundle),
            (second_unit.seed_block, second_score, second_bundle),
        ),
        key=lambda item: int(item[0] or 0),
    )
    for _, score, bundle in ordered:
        if score.output_valid != (bundle is not None):
            raise ValueError("valid cross-seed scores require their complete score bundle")
        if bundle is not None and score.projection_bundle_hash != bundle.content_hash:
            raise ValueError("cross-seed complete bundle hash differs from intended-unit score")

    pair_hash = canonical_sha256(
        {
            "first_score_hash": ordered[0][1].content_hash,
            "second_score_hash": ordered[1][1].content_hash,
            "first_bundle_hash": ordered[0][2].content_hash if ordered[0][2] else None,
            "second_bundle_hash": ordered[1][2].content_hash if ordered[1][2] else None,
            "scorer_plan_hash": scorer_plan.content_hash,
            "metric_configuration_hash": configuration.content_hash,
            "metric_formula_revision": configuration.metric_formula_revision,
        }
    )
    first_complete = ordered[0][2]
    second_complete = ordered[1][2]
    stability: CrossSeedStability | None = None
    status = PipelineMetricStatus.NOT_APPLICABLE
    if scorer_plan.community_eligible:
        status = PipelineMetricStatus.INVALID
        if first_complete is not None and second_complete is not None:
            first_partition = first_complete.projection.structural.leiden_base
            second_partition = second_complete.projection.structural.leiden_base
            if first_partition is not None and second_partition is not None:
                anchors = tuple(anchor for anchor, _ in scorer_plan.gold_community_assignments)
                stability = cross_seed_stability(
                    seed_a_assignments=_partition_on_gold_anchors(
                        first_partition,
                        first_complete.projection.alignment,
                    ),
                    seed_b_assignments=_partition_on_gold_anchors(
                        second_partition,
                        second_complete.projection.alignment,
                    ),
                    anchors=anchors,
                )
                status = PipelineMetricStatus.VALUE

    def pair_row(name: str, value: float | int | None, denominator: int | None) -> MetricResultRow:
        return MetricResultRow(
            projection_id=None,
            unit_hash=pair_hash,
            metric_name=name,
            metric_version_hash=configuration.metric_version_hash,
            status=status,
            value=value if status is PipelineMetricStatus.VALUE else None,
            numerator=None,
            denominator=denominator,
        )

    anchor_count = stability.anchor_count if stability is not None else None
    rows = (
        pair_row(
            "cross_seed_community_adjusted_mutual_information",
            stability.adjusted_mutual_information if stability is not None else None,
            anchor_count,
        ),
        pair_row(
            "cross_seed_community_variation_of_information",
            stability.variation_of_information if stability is not None else None,
            anchor_count,
        ),
        pair_row(
            "cross_seed_community_omitted_seed_1_count",
            stability.omitted_seed_a_count if stability is not None else None,
            anchor_count,
        ),
        pair_row(
            "cross_seed_community_omitted_seed_2_count",
            stability.omitted_seed_b_count if stability is not None else None,
            anchor_count,
        ),
    )
    return CrossSeedCommunityScore(
        scorer_plan_hash=scorer_plan.content_hash,
        condition=first_unit.condition,
        world_id=first_unit.world_id,
        context_id=first_unit.context_id,
        seed_blocks=(1, 2),
        first_projection_hash=(
            first_complete.projection.adapter.projection_hash
            if first_complete is not None and second_complete is not None
            else None
        ),
        second_projection_hash=(
            second_complete.projection.adapter.projection_hash
            if first_complete is not None and second_complete is not None
            else None
        ),
        stability=stability,
        metric_rows=rows,
    )


def score_contrast_pair(
    before_score: IntendedUnitScore,
    after_score: IntendedUnitScore,
    *,
    before_bundle: CompleteProjectionScoreBundle | None,
    after_bundle: CompleteProjectionScoreBundle | None,
    before_projection: OntologyProjection | None,
    after_projection: OntologyProjection | None,
    before_scorer_plan: ScorerMetricPlan,
    after_scorer_plan: ScorerMetricPlan,
    contrast_plan: ContrastMetricPlan,
    configuration: StudyMetricConfiguration,
) -> ContrastPairScore:
    """Score one same-evidence contrast using only exact canonical final-state changes."""

    before_unit = before_score.intended_unit
    after_unit = after_score.intended_unit
    _require_metric_version(before_score, configuration)
    _require_metric_version(after_score, configuration)
    _validate_intended_unit_against_plan(before_unit, before_scorer_plan)
    _validate_intended_unit_against_plan(after_unit, after_scorer_plan)
    if contrast_plan.before_scorer_plan_hash != before_scorer_plan.content_hash or (
        contrast_plan.after_scorer_plan_hash != after_scorer_plan.content_hash
    ):
        raise ValueError("contrast plan names different per-context scorer plans")
    if (before_unit.world_id, before_unit.context_id) != (
        contrast_plan.world_id,
        contrast_plan.before_context_id,
    ) or (after_unit.world_id, after_unit.context_id) != (
        contrast_plan.world_id,
        contrast_plan.after_context_id,
    ):
        raise ValueError("contrast intended units differ from the frozen contrast plan")
    if before_unit.condition is not after_unit.condition or (
        before_unit.seed_block != after_unit.seed_block
    ):
        raise ValueError("contrast outputs must share condition and paired seed block")
    if before_unit.snapshot_hash != after_unit.snapshot_hash or (
        before_unit.packet_hash != after_unit.packet_hash
    ):
        raise ValueError("contrast outputs must use the identical snapshot and evidence packet")
    before_gold_decisions = (
        compile_gold_decision_targets(
            before_scorer_plan.gold_projection,
            before_scorer_plan.gold_decisions,
            predicate_aliases=dict(before_scorer_plan.predicate_aliases),
        )[0]
        if before_scorer_plan.gold_projection is not None
        else before_scorer_plan.gold_decisions
    )
    after_gold_decisions = (
        compile_gold_decision_targets(
            after_scorer_plan.gold_projection,
            after_scorer_plan.gold_decisions,
            predicate_aliases=dict(after_scorer_plan.predicate_aliases),
        )[0]
        if after_scorer_plan.gold_projection is not None
        else after_scorer_plan.gold_decisions
    )
    expected_gold_changes = derive_signed_changes(before_gold_decisions, after_gold_decisions)
    if tuple(sorted(expected_gold_changes, key=canonical_json)) != contrast_plan.gold_changes:
        raise ValueError("contrast delta does not equal the two frozen final-state plans")
    invariant_issues: tuple[InvariantCompilationIssue, ...] = ()
    compiled_invariants = contrast_plan.invariants
    if before_scorer_plan.gold_projection is not None and (
        after_scorer_plan.gold_projection is not None
    ):
        before_invariants, before_issues = compile_gold_contrast_invariants(
            before_scorer_plan.gold_projection,
            contrast_plan.invariants,
            predicate_aliases=dict(before_scorer_plan.predicate_aliases),
        )
        after_invariants, after_issues = compile_gold_contrast_invariants(
            after_scorer_plan.gold_projection,
            contrast_plan.invariants,
            predicate_aliases=dict(after_scorer_plan.predicate_aliases),
        )
        before_signatures = {
            item.invariant_id: item.expected_signature for item in before_invariants
        }
        after_signatures = {
            item.invariant_id: item.expected_signature for item in after_invariants
        }
        if before_signatures != after_signatures:
            raise ValueError(
                "registered contrast invariant is not unchanged in the two gold graphs"
            )
        compiled_invariants = before_invariants
        invariant_issues = tuple(
            sorted(
                {
                    (item.invariant_id, item.reason): item
                    for item in (*before_issues, *after_issues)
                }.values(),
                key=lambda item: (item.invariant_id, item.reason),
            )
        )
    for score, bundle, projection in (
        (before_score, before_bundle, before_projection),
        (after_score, after_bundle, after_projection),
    ):
        if score.output_valid != (bundle is not None and projection is not None):
            raise ValueError("valid contrast score requires its projection and complete bundle")
        if bundle is not None and score.projection_bundle_hash != bundle.content_hash:
            raise ValueError("contrast complete bundle hash differs from intended-unit score")
        if projection is not None and bundle is not None and (
            projection.content_hash != bundle.projection.adapter.projection_hash
        ):
            raise ValueError("contrast projection hash differs from its metric adapter")

    pair_invalid = not before_score.output_valid or not after_score.output_valid
    if pair_invalid:
        predicted_changes: tuple[SignedDecisionChange, ...] = ()
        before_assertions = ()
        after_assertions = ()
    else:
        assert before_bundle is not None and after_bundle is not None
        assert before_projection is not None and after_projection is not None
        before_verified = (
            before_bundle.projection.decision_compilation.predicted_decisions
            if before_bundle.projection.decision_compilation is not None
            else verify_normalized_decisions(
                before_bundle.projection.adapter.normalized_decisions,
                gold_targets=before_scorer_plan.gold_decisions,
            )
        )
        after_verified = (
            after_bundle.projection.decision_compilation.predicted_decisions
            if after_bundle.projection.decision_compilation is not None
            else verify_normalized_decisions(
                after_bundle.projection.adapter.normalized_decisions,
                gold_targets=after_scorer_plan.gold_decisions,
            )
        )
        predicted_changes = derive_signed_changes(before_verified, after_verified)
        if before_scorer_plan.gold_projection is not None and (
            after_scorer_plan.gold_projection is not None
        ):
            before_assertions = compile_projection_assertion_semantics(
                before_scorer_plan.gold_projection,
                before_projection,
                predicate_aliases=dict(before_scorer_plan.predicate_aliases),
            )
            after_assertions = compile_projection_assertion_semantics(
                after_scorer_plan.gold_projection,
                after_projection,
                predicate_aliases=dict(after_scorer_plan.predicate_aliases),
            )
        else:
            before_assertions = before_bundle.projection.adapter.assertion_semantics
            after_assertions = after_bundle.projection.adapter.assertion_semantics
    change_score = score_contrastive_changes(
        gold=contrast_plan.gold_changes,
        predicted=predicted_changes,
        invalid_semantic_output=pair_invalid,
    )
    invariant_score = score_contrast_invariants(
        invariants=compiled_invariants,
        before_assertions=before_assertions,
        after_assertions=after_assertions,
        before_invalid=not before_score.output_valid,
        after_invalid=not after_score.output_valid,
    )
    pair_hash = canonical_sha256(
        {
            "before_score_hash": before_score.content_hash,
            "after_score_hash": after_score.content_hash,
            "before_bundle_hash": before_bundle.content_hash if before_bundle else None,
            "after_bundle_hash": after_bundle.content_hash if after_bundle else None,
            "before_projection_hash": (
                before_projection.content_hash if before_projection else None
            ),
            "after_projection_hash": (
                after_projection.content_hash if after_projection else None
            ),
            "contrast_plan_hash": contrast_plan.content_hash,
            "metric_configuration_hash": configuration.content_hash,
            "metric_formula_revision": configuration.metric_formula_revision,
        }
    )

    def pair_row(
        name: str,
        value: float | int | None,
        *,
        numerator: int | None = None,
        denominator: int | None = None,
    ) -> MetricResultRow:
        return MetricResultRow(
            projection_id=None,
            unit_hash=pair_hash,
            metric_name=name,
            metric_version_hash=configuration.metric_version_hash,
            status=(
                PipelineMetricStatus.VALUE
                if value is not None
                else PipelineMetricStatus.NOT_APPLICABLE
            ),
            value=value,
            numerator=numerator,
            denominator=denominator,
        )

    change = change_score.decision_change_score
    collapse_value = (
        change_score.collapse_rate_contribution
        if change_score.gold_nonselection_change_count > 0
        else None
    )
    rows = (
        pair_row(
            "contrastive_decision_change_precision",
            change.precision,
            numerator=change.true_positive_count,
            denominator=change.precision_denominator,
        ),
        pair_row(
            "contrastive_decision_change_recall",
            change.recall,
            numerator=change.true_positive_count,
            denominator=change.recall_denominator,
        ),
        pair_row("contrastive_decision_change_f1", change.f1),
        pair_row(
            "contrastive_collapse",
            collapse_value,
            numerator=int(change_score.ontological_collapse),
            denominator=(1 if collapse_value is not None else 0),
        ),
        pair_row(
            "contrast_invariant_preservation_rate",
            invariant_score.preservation_rate.value,
            numerator=invariant_score.preserved_in_both_count,
            denominator=invariant_score.invariant_count,
        ),
        pair_row(
            "contrast_invariant_unrepresentable_gold_count",
            len(invariant_issues),
        ),
    )
    return ContrastPairScore(
        plan_hash=contrast_plan.content_hash,
        condition=before_unit.condition,
        seed_block=before_unit.seed_block,
        before_projection_hash=(
            before_bundle.projection.adapter.projection_hash if before_bundle else None
        ),
        after_projection_hash=(
            after_bundle.projection.adapter.projection_hash if after_bundle else None
        ),
        changes=change_score,
        invariants=invariant_score,
        invariant_compilation_issues=invariant_issues,
        metric_rows=rows,
    )


def validate_intended_unit_manifest(units: Sequence[IntendedMetricUnit]) -> None:
    """Reject duplicate cells/jobs before any output-dependent scoring begins."""

    unit_ids = tuple(item.unit_id for item in units)
    job_ids = tuple(item.job_id for item in units)
    cells = tuple(
        (item.condition, item.world_id, item.context_id, item.seed_block) for item in units
    )
    if len(unit_ids) != len(set(unit_ids)):
        raise ValueError("intended metric manifest contains duplicate unit IDs")
    if len(job_ids) != len(set(job_ids)):
        raise ValueError("intended metric manifest contains duplicate job IDs")
    if len(cells) != len(set(cells)):
        raise ValueError("intended metric manifest contains duplicate condition cells")


def materialize_intention_to_treat_scores(
    intended_manifest: IntendedMetricManifest,
    *,
    successful_scores: Sequence[IntendedUnitScore],
    recorded_failures: Sequence[FailedMetricOutput],
    scorer_plans_by_hash: Mapping[str, ScorerMetricPlan],
    configuration: StudyMetricConfiguration,
) -> tuple[IntendedUnitScore, ...]:
    """Reconcile persisted successes/failures and synthesize explicit missing-output rows."""

    successes: dict[str, IntendedUnitScore] = {}
    for score in successful_scores:
        unit_hash = score.intended_unit.content_hash
        if not score.output_valid:
            raise ValueError("successful score collection contains an invalid output")
        _require_metric_version(score, configuration)
        if unit_hash in successes:
            raise ValueError("duplicate successful intended-unit score")
        successes[unit_hash] = score
    failures: dict[str, FailedMetricOutput] = {}
    for failure in recorded_failures:
        unit_hash = failure.intended_unit.content_hash
        if unit_hash in failures:
            raise ValueError("duplicate failed intended-unit output")
        failures[unit_hash] = failure
    intended_hashes = {item.content_hash for item in intended_manifest.units}
    extras = (set(successes) | set(failures)) - intended_hashes
    if extras:
        raise ValueError("success/failure records contain units outside the intended manifest")
    overlap = set(successes) & set(failures)
    if overlap:
        raise ValueError("an intended unit cannot be both successful and failed")

    scores = []
    for intended in intended_manifest.units:
        unit_hash = intended.content_hash
        if unit_hash in successes:
            scores.append(successes[unit_hash])
            continue
        failure = failures.get(unit_hash) or FailedMetricOutput(
            intended_unit=intended,
            failure_kind=OutputFailureKind.MISSING,
        )
        try:
            scorer_plan = scorer_plans_by_hash[intended.scorer_plan_hash]
        except KeyError as error:
            raise ValueError("intended unit lacks its immutable scorer plan") from error
        scores.append(
            score_failed_output(
                failure,
                configuration=configuration,
                scorer_plan=scorer_plan,
            )
        )
    return tuple(scores)


def analysis_observations_from_scores(
    scores: Sequence[IntendedUnitScore],
    *,
    intended_manifest: IntendedMetricManifest,
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Create registered analysis inputs from complete intended-unit score rows.

    This function is also the ledger-read boundary: callers deserialize persisted
    ``IntendedUnitScore`` artifacts, rather than hand-authoring validity flags or
    semantic values after seeing results.
    """

    from story_projection_onto.analysis import (
        MetricObservation,
        RarePivotalCountObservation,
    )

    validate_intended_unit_manifest(tuple(item.intended_unit for item in scores))
    score_by_unit_hash: dict[str, IntendedUnitScore] = {}
    for score in scores:
        key = score.intended_unit.content_hash
        if key in score_by_unit_hash:
            raise ValueError(f"duplicate score for intended unit {score.intended_unit.unit_id!r}")
        score_by_unit_hash[key] = score
    intended_by_hash = {item.content_hash: item for item in intended_manifest.units}
    if set(score_by_unit_hash) != set(intended_by_hash):
        missing = sorted(
            intended_by_hash[key].unit_id for key in set(intended_by_hash) - set(score_by_unit_hash)
        )
        extra = sorted(
            score_by_unit_hash[key].intended_unit.unit_id
            for key in set(score_by_unit_hash) - set(intended_by_hash)
        )
        raise ValueError(
            f"score set differs from intended-unit manifest; missing={missing}, extra={extra}"
        )
    metric_versions = {
        row.metric_version_hash for score in scores for row in score.rows
    }
    if len(metric_versions) != 1:
        raise ValueError("analysis score artifacts must use one metric version")
    metric_observations = []
    rare_observations = []
    for intended in intended_manifest.units:
        score = score_by_unit_hash[intended.content_hash]
        intended = score.intended_unit
        rows_by_name: dict[str, MetricResultRow] = {}
        for row_value in score.rows:
            if row_value.metric_name in rows_by_name:
                raise ValueError(
                    f"intended unit {intended.unit_id!r} repeats metric "
                    f"{row_value.metric_name!r}"
                )
            rows_by_name[row_value.metric_name] = row_value
        for metric_name, gold_count in (
            ("strict_qualified_assertion_f1", intended.strict_assertion_gold_count),
            ("ontology_decision_macro_f1", intended.ontology_decision_gold_count),
        ):
            try:
                metric_row = rows_by_name[metric_name]
            except KeyError as error:
                raise ValueError(
                    f"intended unit {intended.unit_id!r} lacks {metric_name!r}"
                ) from error
            metric_observations.append(
                MetricObservation(
                    condition=intended.condition.value,
                    world_id=intended.world_id,
                    context_id=intended.context_id,
                    seed_block=intended.seed_block,
                    metric_name=metric_name,
                    value=metric_row.value,
                    output_valid=score.output_valid,
                    gold_nonempty=gold_count > 0,
                    gold_count=gold_count,
                    scorer_plan_hash=intended.scorer_plan_hash,
                )
            )
            if gold_count > 0 and metric_row.value is None:
                raise ValueError(
                    f"intended unit {intended.unit_id!r} has NA {metric_name!r} "
                    "despite a positive gold denominator"
                )
            if not score.output_valid and gold_count > 0 and metric_row.value != 0.0:
                raise ValueError(
                    f"invalid intended unit {intended.unit_id!r} must contribute zero "
                    f"to {metric_name!r}"
                )
        try:
            rare_row = rows_by_name["rare_pivotal_qualified_assertion_recall"]
        except KeyError as error:
            raise ValueError("intended unit lacks rare-pivotal ITT row") from error
        if rare_row.denominator != intended.rare_pivotal_gold_count:
            raise ValueError("rare-pivotal row denominator differs from intended manifest")
        if intended.rare_pivotal_gold_count > 0 and rare_row.value is None:
            raise ValueError("positive rare-pivotal denominator requires a numeric ITT value")
        if (
            not score.output_valid
            and intended.rare_pivotal_gold_count > 0
            and rare_row.value != 0.0
        ):
            raise ValueError("invalid output must contribute zero rare-pivotal recall")
        rare_numerator = rare_row.numerator or 0
        if int(rare_numerator) != rare_numerator:
            raise ValueError("rare-pivotal numerator must be an integer count")
        rare_observations.append(
            RarePivotalCountObservation(
                condition=intended.condition.value,
                world_id=intended.world_id,
                context_id=intended.context_id,
                seed_block=intended.seed_block,
                true_positive_count=int(rare_numerator) if score.output_valid else 0,
                gold_count=intended.rare_pivotal_gold_count,
                output_valid=score.output_valid,
                scorer_plan_hash=intended.scorer_plan_hash,
            )
        )
    return tuple(metric_observations), tuple(rare_observations)


def persist_metric_rows(
    ledger: Any,
    *,
    study_id: str,
    job_id: str,
    rows: Sequence[MetricResultRow],
    result_artifact_hash: str | None = None,
) -> None:
    """Persist typed results without importing store machinery into pure metric code."""

    from story_projection_onto.store import MetricStatus

    if not rows:
        raise ValueError("metric persistence requires a nonempty score inventory")
    semantic_projection_ids = {row.projection_id for row in rows}
    if len(semantic_projection_ids) != 1:
        raise ValueError("one intended score cannot mix projection identities")
    semantic_projection_id = next(iter(semantic_projection_ids), None)
    ledger_projections = tuple(ledger.projections_for_job(job_id))
    if semantic_projection_id is None:
        if ledger_projections:
            raise ValueError(
                "failed ITT metric rows conflict with a durable job projection"
            )
        ledger_projection_id = None
    else:
        if len(ledger_projections) != 1:
            raise ValueError(
                "scored projection does not resolve exactly one job-bound ledger row"
            )
        ledger_projection_id = ledger_projections[0].projection_id

    for row in rows:
        ledger.record_metric(
            metric_id=(
                f"metric-{canonical_sha256((study_id, job_id, row, result_artifact_hash))[:24]}"
            ),
            study_id=study_id,
            job_id=job_id,
            projection_id=ledger_projection_id,
            unit_hash=row.unit_hash,
            metric_name=row.metric_name,
            metric_version_hash=row.metric_version_hash,
            status=MetricStatus(row.status.value),
            value=row.value,
            numerator=row.numerator,
            denominator=row.denominator,
            result_artifact_hash=result_artifact_hash,
        )


def persist_intended_unit_score(
    artifact_store: Any,
    *,
    study_id: str,
    score: IntendedUnitScore,
    release_class: ReleaseClass = ReleaseClass.PUBLIC,
) -> str:
    """Persist one canonical score artifact, lineage link, and metric rows."""

    payload = canonical_json(score).encode("utf-8") + b"\n"
    artifact = artifact_store.put_bytes(
        payload,
        media_type="application/vnd.story-projection-onto.intended-score+jsonl",
        release_class=release_class,
    )
    artifact_store.ledger.link_artifact(
        job_id=score.intended_unit.job_id,
        content_hash=artifact.content_hash,
        role="metric-intended-unit-score",
    )
    persist_metric_rows(
        artifact_store.ledger,
        study_id=study_id,
        job_id=score.intended_unit.job_id,
        rows=score.rows,
        result_artifact_hash=artifact.content_hash,
    )
    return artifact.content_hash


def load_intended_unit_scores(
    artifact_store: Any,
    *,
    artifact_hashes: Sequence[str],
    intended_manifest: IntendedMetricManifest,
    allow_restricted: bool = False,
) -> tuple[IntendedUnitScore, ...]:
    """Load verified CAS score artifacts and require exact intended-manifest coverage."""

    if len(artifact_hashes) != len(set(artifact_hashes)):
        raise ValueError("intended score artifact hashes must be unique")
    loaded: list[IntendedUnitScore] = []
    for artifact_hash in artifact_hashes:
        record = artifact_store.ledger.get_artifact(artifact_hash)
        payload = artifact_store.blobs.read_bytes(
            record,
            allow_restricted=allow_restricted,
        )
        lines = tuple(line for line in payload.splitlines() if line.strip())
        if len(lines) != 1:
            raise ValueError("each intended-unit score artifact must contain exactly one JSON row")
        try:
            raw = json.loads(lines[0])
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("intended-unit score artifact is not valid UTF-8 JSON") from error
        loaded.append(IntendedUnitScore.model_validate(raw))
    # The observation conversion owns the fail-closed exact-coverage check. Reuse it
    # here so loading cannot silently omit a missing/timed-out intended cell.
    analysis_observations_from_scores(loaded, intended_manifest=intended_manifest)
    return tuple(
        sorted(loaded, key=lambda item: item.intended_unit.unit_id)
    )


def load_intended_unit_scores_from_metric_records(
    artifact_store: Any,
    *,
    metric_records: Sequence[Any],
    intended_manifest: IntendedMetricManifest,
    metric_version_hash: str,
    allow_restricted: bool = False,
) -> tuple[IntendedUnitScore, ...]:
    """Resolve score artifacts from immutable ledger metric rows, then verify coverage."""

    if any(item.metric_version_hash != metric_version_hash for item in metric_records):
        raise ValueError("ledger metric rows mix formula/configuration versions")
    artifact_hashes = {
        item.result_artifact_hash
        for item in metric_records
        if item.result_artifact_hash is not None
    }
    if any(item.result_artifact_hash is None for item in metric_records):
        raise ValueError("ledger metric row lacks its result artifact provenance")
    scores = load_intended_unit_scores(
        artifact_store,
        artifact_hashes=tuple(sorted(artifact_hashes)),
        intended_manifest=intended_manifest,
        allow_restricted=allow_restricted,
    )
    if any(
        row.metric_version_hash != metric_version_hash
        for score in scores
        for row in score.rows
    ):
        raise ValueError("score artifacts differ from the requested ledger metric version")
    return scores


__all__ = [
    "CompleteProjectionScoreBundle",
    "ContrastMetricPlan",
    "ContrastPairScore",
    "CrossSeedCommunityScore",
    "FailedMetricOutput",
    "GeometryMetricInput",
    "GroundingAuditInput",
    "IntendedMetricManifest",
    "IntendedMetricUnit",
    "IntendedUnitScore",
    "MetricResultRow",
    "OutputFailureKind",
    "PipelineMetricStatus",
    "ProjectionScoreBundle",
    "ScorerMetricPlan",
    "ScorerOnlyProjectionReplayMaterial",
    "StructuralMetricPanel",
    "analysis_observations_from_scores",
    "load_intended_unit_scores",
    "load_intended_unit_scores_from_metric_records",
    "materialize_intention_to_treat_scores",
    "persist_intended_unit_score",
    "persist_metric_rows",
    "score_complete_projection",
    "score_contrast_pair",
    "score_cross_seed_community",
    "score_failed_output",
    "score_intended_projection",
    "score_projection",
    "validate_intended_unit_manifest",
]
