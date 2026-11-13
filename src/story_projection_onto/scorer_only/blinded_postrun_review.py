"""Condition-blind, scorer-only post-run review materialization.

The held-out error taxonomy and the community-coherence rubric are human review
instruments, not model inputs and not additional experimental units.  This module
creates opaque review packages, keeps the condition/world rejoin maps restricted,
validates externally authored completions, and emits canonical one-row-per-output
tables whose declared independent unit remains the world.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import shutil
import stat
import tempfile
import unicodedata
from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Annotated, Any, Literal, Self, cast

from pydantic import AwareDatetime, Field, StringConstraints, model_validator

from story_projection_onto.contracts import (
    ConditionName,
    Identifier,
    ImmutableRecord,
    ReleaseClass,
    RunOutcome,
    Sha256Digest,
    canonical_sha256,
)
from story_projection_onto.held_out_controller import HeldOutExecutionManifest
from story_projection_onto.held_out_primary import HeldOutCallManifest
from story_projection_onto.metrics.adapters import adapt_projection_for_metrics
from story_projection_onto.metrics.community import (
    LeidenConfiguration,
    LeidenPartition,
    run_leiden_cpm,
)
from story_projection_onto.metrics.config import (
    CommunityReviewEntry,
    CommunityReviewTemplate,
    StudyMetricConfiguration,
)
from story_projection_onto.metrics.pipeline import (
    CompleteProjectionScoreBundle,
    IntendedMetricManifest,
    IntendedUnitScore,
    MetricResultRow,
    OutputFailureKind,
    PipelineMetricStatus,
    ScorerMetricPlan,
)
from story_projection_onto.scorer_only.phase4_analysis import (
    REGISTERED_PRIMARY_SCORE_COUNT,
    Phase4AnalysisConfiguration,
    Phase4AnalysisError,
    Phase4AnalysisIndex,
)
from story_projection_onto.scorer_only.phase4_replay import replay_phase4_analysis_outputs

RelativePath = Annotated[str, StringConstraints(min_length=1, max_length=500)]
_MAX_REVIEW_FILE_BYTES = 128 * 1024 * 1024
_PRIMARY_CONDITIONS = frozenset(
    {
        ConditionName.C0_CLASSICAL_PRE,
        ConditionName.C1_LLM_PRE,
        ConditionName.C2_LLM_QUERY,
        ConditionName.A_FIXED_SELECT,
    }
)
_PRIMARY_CONDITION_ORDER = (
    ConditionName.C0_CLASSICAL_PRE,
    ConditionName.C1_LLM_PRE,
    ConditionName.C2_LLM_QUERY,
    ConditionName.A_FIXED_SELECT,
)
_COMMUNITY_RESOLUTION_ROLES = ("half", "base", "double")
FROZEN_COMMUNITY_REVIEW_SELECTION_RULE = MappingProxyType(
    {
        "rule_id": "preoutput-hash-first-paired-context-v2",
        "candidate_source": "primary-intended-contexts-with-reviewed-community-gold",
        "rank_key": "sha256-rule-domain-and-condition-independent-intended-context-lineage",
        "missing_output_policy": "fail-without-substitution",
        "selected_shared_context_count": 1,
        "llm_seed_block": 1,
        "primary_conditions": tuple(item.value for item in _PRIMARY_CONDITION_ORDER),
        "resolution_roles": _COMMUNITY_RESOLUTION_ROLES,
        "quality_metric_selection_forbidden": True,
    }
)
FROZEN_COMMUNITY_REVIEW_SELECTION_RULE_HASH = canonical_sha256(
    FROZEN_COMMUNITY_REVIEW_SELECTION_RULE
)
_PHASE4_SOURCE_ROLES = frozenset(
    {
        "combined_call_manifest",
        "combined_execution",
        "held_out_call_manifest",
        "held_out_execution",
        "held_out_scorer_bridge",
        "review_completion_manifest",
    }
)


class FailureSignalDirection(StrEnum):
    """Mechanical direction for one frozen, cell-level failure signal."""

    BELOW = "below"
    ABOVE = "above"


class RegisteredFailureSignalGate(ImmutableRecord):
    """One predeclared signal used to identify successful outputs for review.

    These are error-analysis inclusion gates, not confirmatory hypothesis tests.
    Fidelity and support measures fail below perfection; unsupported content and
    renderer omission fail above zero.  Entropy, cluster count, and general
    clutter are intentionally absent because the registered plan assigns them no
    inherently favorable direction.
    """

    gate_id: Identifier
    metric_name: Identifier
    direction: FailureSignalDirection
    threshold: float


FROZEN_REGISTERED_FAILURE_SIGNAL_GATES = tuple(
    sorted(
        (
            RegisteredFailureSignalGate(
                gate_id="contextual-node-f1-below-1",
                metric_name="contextual_node_f1",
                direction=FailureSignalDirection.BELOW,
                threshold=1.0,
            ),
            RegisteredFailureSignalGate(
                gate_id="essential-temporal-accuracy-below-1",
                metric_name="essential_temporal_qualification_accuracy",
                direction=FailureSignalDirection.BELOW,
                threshold=1.0,
            ),
            RegisteredFailureSignalGate(
                gate_id="evidence-citation-validity-below-1",
                metric_name="evidence_citation_validity",
                direction=FailureSignalDirection.BELOW,
                threshold=1.0,
            ),
            RegisteredFailureSignalGate(
                gate_id="grounding-precision-below-1",
                metric_name="grounding_precision",
                direction=FailureSignalDirection.BELOW,
                threshold=1.0,
            ),
            RegisteredFailureSignalGate(
                gate_id="ontology-decision-macro-f1-below-1",
                metric_name="ontology_decision_macro_f1",
                direction=FailureSignalDirection.BELOW,
                threshold=1.0,
            ),
            RegisteredFailureSignalGate(
                gate_id="rare-pivotal-recall-below-1",
                metric_name="rare_pivotal_qualified_assertion_recall",
                direction=FailureSignalDirection.BELOW,
                threshold=1.0,
            ),
            RegisteredFailureSignalGate(
                gate_id="rare-pivotal-renderer-omission-above-0",
                metric_name="rare_pivotal_omitted_count",
                direction=FailureSignalDirection.ABOVE,
                threshold=0.0,
            ),
            RegisteredFailureSignalGate(
                gate_id="rare-pivotal-support-path-below-1",
                metric_name="rare_pivotal_support_path_survival",
                direction=FailureSignalDirection.BELOW,
                threshold=1.0,
            ),
            RegisteredFailureSignalGate(
                gate_id="strict-qualified-assertion-f1-below-1",
                metric_name="strict_qualified_assertion_f1",
                direction=FailureSignalDirection.BELOW,
                threshold=1.0,
            ),
            RegisteredFailureSignalGate(
                gate_id="unsupported-assertion-rate-above-0",
                metric_name="unsupported_assertion_rate",
                direction=FailureSignalDirection.ABOVE,
                threshold=0.0,
            ),
        ),
        key=lambda item: item.gate_id,
    )
)
FROZEN_REGISTERED_FAILURE_SIGNAL_POLICY_HASH = canonical_sha256(
    FROZEN_REGISTERED_FAILURE_SIGNAL_GATES
)


class BlindedReviewError(RuntimeError):
    """A blind package or externally authored completion violated its contract."""


class HeldOutErrorCode(StrEnum):
    EVIDENCE_RETRIEVAL_OMISSION = "evidence_retrieval_omission"
    OVER_MERGE = "over_merge"
    UNDER_MERGE = "under_merge"
    WRONG_SCHEMA_RELATION = "wrong_schema_relation"
    EVENT_BOUNDARY_ROLE_ERROR = "event_boundary_role_error"
    STORY_DISCOURSE_REVELATION_CONFUSION = "story_discourse_revelation_confusion"
    EPISTEMIC_HOLDER_COMMITMENT_ERROR = "epistemic_holder_commitment_error"
    CAUSAL_OVERCLAIM = "causal_overclaim"
    UNSUPPORTED_DESCRIPTION = "unsupported_description"
    WRONG_ABSTRACTION = "wrong_abstraction"
    RARE_PIVOTAL_LOSS = "rare_pivotal_loss"
    CONTRASTIVE_COLLAPSE = "contrastive_collapse"
    PARAPHRASE_INSTABILITY = "paraphrase_instability"
    INVALID_STRUCTURE = "invalid_structure"
    RENDERER_HIDING = "renderer_hiding"


FROZEN_ERROR_CODES = tuple(HeldOutErrorCode)


class HeldOutErrorTaxonomy(ImmutableRecord):
    taxonomy_id: Literal["held-out-error-taxonomy-v1"]
    codes: tuple[HeldOutErrorCode, ...]

    @model_validator(mode="after")
    def exact_frozen_inventory(self) -> Self:
        if self.codes != FROZEN_ERROR_CODES:
            raise ValueError("held-out error taxonomy differs from the frozen inventory")
        return self


class ReviewPanelSource(ImmutableRecord):
    relative_path: RelativePath
    file_sha256: Sha256Digest
    media_type: Literal["image/png", "application/json"]


class NeutralFailurePanelMetric(ImmutableRecord):
    """Condition-free view of one registered signal metric."""

    metric_name: Identifier
    status: PipelineMetricStatus
    value: float | int | None = None
    numerator: float | int | None = None
    denominator: float | int | None = None

    @model_validator(mode="after")
    def value_matches_status(self) -> Self:
        if (self.status is PipelineMetricStatus.VALUE) != (self.value is not None):
            raise ValueError("neutral panel metric value differs from its status")
        return self


class NeutralFailurePanelEdge(ImmutableRecord):
    edge_id: Identifier
    source_id: Identifier
    target_id: Identifier
    native_relation_id: Identifier
    canonical_relation_id: Identifier | None = None


class NeutralFailurePanelDecision(ImmutableRecord):
    family: Identifier
    operator: Identifier
    anchor_count: int = Field(gt=0)


class NeutralFailureReviewPanel(ImmutableRecord):
    """Reviewer-facing structure with every experimental identity removed."""

    panel_schema: Literal["condition-blind-held-out-failure-panel-v1"] = (
        "condition-blind-held-out-failure-panel-v1"
    )
    condition_blind: Literal[True] = True
    source_identifiers_visible: Literal[False] = False
    output_status: Literal["succeeded", "failed", "timed_out", "invalid", "interrupted"]
    failed_gate_ids: tuple[Identifier, ...]
    metrics: tuple[NeutralFailurePanelMetric, ...]
    structurally_valid: bool | None = None
    nodes: tuple[Identifier, ...] = ()
    edges: tuple[NeutralFailurePanelEdge, ...] = ()
    assertion_relation_ids: tuple[Identifier, ...] = ()
    decisions: tuple[NeutralFailurePanelDecision, ...] = ()

    @model_validator(mode="after")
    def panel_is_canonical_and_status_consistent(self) -> Self:
        if self.failed_gate_ids != tuple(sorted(set(self.failed_gate_ids))):
            raise ValueError("neutral review-panel gates must be canonical")
        if self.metrics != tuple(sorted(self.metrics, key=lambda item: item.metric_name)):
            raise ValueError("neutral review-panel metrics must be sorted")
        if self.nodes != tuple(sorted(set(self.nodes))):
            raise ValueError("neutral review-panel nodes must be sorted and unique")
        if self.edges != tuple(sorted(self.edges, key=lambda item: item.edge_id)):
            raise ValueError("neutral review-panel edges must be sorted")
        if self.output_status == "succeeded" and self.structurally_valid is not True:
            raise ValueError("successful neutral panels require valid structure")
        if self.output_status != "succeeded" and any(
            (self.nodes, self.edges, self.assertion_relation_ids, self.decisions)
        ):
            raise ValueError("non-success panels cannot imply a retained valid projection")
        return self


class HeldOutFailureSource(ImmutableRecord):
    source_failure_id: Identifier
    world_id: Identifier
    context_id: Identifier
    condition: ConditionName
    seed_block: Literal[1, 2] | None
    result_artifact_hash: Sha256Digest
    projection_hash: Sha256Digest | None = None
    output_status: Literal["succeeded", "failed", "timed_out", "invalid", "interrupted"]
    failed_gate_ids: tuple[Identifier, ...]
    review_panel: ReviewPanelSource

    @model_validator(mode="after")
    def source_is_a_primary_held_out_failure(self) -> Self:
        if self.condition not in _PRIMARY_CONDITIONS:
            raise ValueError("held-out error review accepts only primary study conditions")
        if not self.failed_gate_ids or self.failed_gate_ids != tuple(
            sorted(set(self.failed_gate_ids))
        ):
            raise ValueError("each error-review source must have canonical failed gates")
        if self.output_status == "succeeded" and self.projection_hash is None:
            raise ValueError("a successful failure candidate must bind its projection")
        if (self.condition is ConditionName.C0_CLASSICAL_PRE) != (self.seed_block is None):
            raise ValueError("only deterministic C0 failure sources are unseeded")
        registered_gate_ids = {
            item.gate_id for item in FROZEN_REGISTERED_FAILURE_SIGNAL_GATES
        }
        if self.output_status == "succeeded":
            if not set(self.failed_gate_ids).issubset(registered_gate_ids):
                raise ValueError("successful failure source names an unregistered metric gate")
        elif self.failed_gate_ids != (f"output-status-{self.output_status}",):
            raise ValueError("non-success failure source must name its exact output status")
        return self


class HeldOutFailureSourceManifest(ImmutableRecord):
    manifest_id: Identifier
    split: Literal["held_out"] = "held_out"
    analysis_artifact_hash: Sha256Digest
    analysis_index_file_sha256: Sha256Digest
    phase4_replay_receipt_hash: Sha256Digest
    phase4_output_inventory_hash: Sha256Digest
    primary_intended_manifest_hash: Sha256Digest
    primary_scores_file_sha256: Sha256Digest
    primary_scores_logical_hash: Sha256Digest
    held_out_call_manifest_hash: Sha256Digest
    held_out_call_manifest_file_sha256: Sha256Digest
    held_out_execution_hash: Sha256Digest
    held_out_execution_file_sha256: Sha256Digest
    primary_receipt_inventory_hash: Sha256Digest
    scored_cell_eligibility_hash: Sha256Digest
    failure_signal_policy_id: Literal["registered-cell-failure-signals-v1"] = (
        "registered-cell-failure-signals-v1"
    )
    failure_signal_policy_hash: Sha256Digest
    failure_selection_rule: Literal["non_success_or_registered_failure_signal"] = (
        "non_success_or_registered_failure_signal"
    )
    held_out_output_count: Literal[288] = 288
    scored_primary_cell_count: Literal[252] = 252
    c0_construction_receipt_count: Literal[12] = 12
    preconstructed_projection_receipt_count: Literal[108] = 108
    model_call_itt_receipt_count: Literal[168] = 168
    eligible_failure_count: int = Field(gt=0)
    items: tuple[HeldOutFailureSource, ...]
    selected_at: AwareDatetime
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED

    @model_validator(mode="after")
    def inventory_is_unique(self) -> Self:
        identifiers = [item.source_failure_id for item in self.items]
        cells = [
            (item.world_id, item.context_id, item.condition, item.seed_block)
            for item in self.items
        ]
        if not identifiers:
            raise ValueError("held-out error review requires at least one failure")
        if len(self.items) != self.eligible_failure_count:
            raise ValueError("error source must enumerate every eligible held-out failure")
        if self.eligible_failure_count > self.scored_primary_cell_count:
            raise ValueError("failure count cannot exceed the scored held-out cell inventory")
        if (
            self.c0_construction_receipt_count
            + self.preconstructed_projection_receipt_count
            + self.model_call_itt_receipt_count
            != self.held_out_output_count
        ):
            raise ValueError("held-out failure source does not bind exactly 288 receipts")
        if (
            self.failure_signal_policy_hash
            != FROZEN_REGISTERED_FAILURE_SIGNAL_POLICY_HASH
        ):
            raise ValueError("held-out failure selection policy differs from its frozen gates")
        if len(identifiers) != len(set(identifiers)) or len(cells) != len(set(cells)):
            raise ValueError("held-out failure source inventory contains duplicates")
        if self.items != tuple(sorted(self.items, key=lambda item: item.source_failure_id)):
            raise ValueError("held-out failure sources must be sorted by opaque ID")
        return self


class BlindedErrorItem(ImmutableRecord):
    blind_item_id: Identifier
    panel_file: RelativePath
    panel_sha256: Sha256Digest
    media_type: Literal["image/png", "application/json"]


class BlindedErrorReviewPackage(ImmutableRecord):
    package_id: Identifier
    condition_blind: Literal[True] = True
    taxonomy_id: Literal["held-out-error-taxonomy-v1"]
    taxonomy_hash: Sha256Digest
    source_manifest_hash: Sha256Digest
    items: tuple[BlindedErrorItem, ...]
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED

    @model_validator(mode="after")
    def blind_inventory_is_canonical(self) -> Self:
        identifiers = [item.blind_item_id for item in self.items]
        if not identifiers or identifiers != sorted(set(identifiers)):
            raise ValueError("blinded error items must be nonempty, sorted, and unique")
        return self


class ErrorRejoinEntry(ImmutableRecord):
    blind_item_id: Identifier
    source_failure_id: Identifier
    world_id: Identifier
    context_id: Identifier
    condition: ConditionName
    seed_block: Literal[1, 2] | None
    result_artifact_hash: Sha256Digest
    projection_hash: Sha256Digest | None
    output_status: Literal["succeeded", "failed", "timed_out", "invalid", "interrupted"]
    failed_gate_ids: tuple[Identifier, ...]


class ErrorReviewRejoinMap(ImmutableRecord):
    map_id: Identifier
    package_hash: Sha256Digest
    source_manifest_hash: Sha256Digest
    entries: tuple[ErrorRejoinEntry, ...]
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED


class ErrorReviewJudgment(ImmutableRecord):
    blind_item_id: Identifier
    error_codes: tuple[HeldOutErrorCode, ...]
    reviewer_note: str = Field(default="", max_length=4000)

    @model_validator(mode="after")
    def at_least_one_canonical_code(self) -> Self:
        values = tuple(code.value for code in self.error_codes)
        if not values or values != tuple(sorted(set(values))):
            raise ValueError("every failure requires sorted unique taxonomy codes")
        return self


class ErrorReviewCompletion(ImmutableRecord):
    completion_id: Identifier
    package_hash: Sha256Digest
    reviewer_id: Identifier
    completed_at: AwareDatetime
    judgments: tuple[ErrorReviewJudgment, ...]
    condition_blind: Literal[True] = True
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED


class ErrorAdjudicationEntry(ImmutableRecord):
    blind_item_id: Identifier
    disposition: Literal["accept", "amend"]
    final_error_codes: tuple[HeldOutErrorCode, ...]
    rationale: str = Field(min_length=1, max_length=4000)

    @model_validator(mode="after")
    def final_codes_are_canonical(self) -> Self:
        values = tuple(code.value for code in self.final_error_codes)
        if not values or values != tuple(sorted(set(values))):
            raise ValueError("adjudication must retain sorted unique error codes")
        return self


class ErrorReviewAdjudication(ImmutableRecord):
    adjudication_id: Identifier
    package_hash: Sha256Digest
    completion_hash: Sha256Digest
    adjudicator_id: Identifier
    adjudicated_at: AwareDatetime
    entries: tuple[ErrorAdjudicationEntry, ...]
    condition_blind: Literal[True] = True
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED


class ErrorReviewFinalization(ImmutableRecord):
    finalization_id: Identifier
    package_hash: Sha256Digest
    rejoin_map_hash: Sha256Digest
    completion_hash: Sha256Digest
    adjudication_hash: Sha256Digest
    canonical_table_file: Literal["held_out_error_review.csv"] = "held_out_error_review.csv"
    canonical_table_sha256: Sha256Digest
    reviewed_failure_count: int = Field(gt=0)
    independent_unit: Literal["world"] = "world"
    complete: Literal[True] = True
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED


class NeutralCommunityTemporalConstraint(ImmutableRecord):
    left_reference_id: Identifier
    relation: Identifier
    right_reference_id: Identifier


class NeutralCommunityTemporalDisplay(ImmutableRecord):
    """Temporal value with every source anchor replaced by a panel-local alias."""

    kind: Identifier
    point: int | None = None
    start: int | None = None
    end: int | None = None
    label: str | None = None
    anchor_reference_id: Identifier | None = None
    relation: Identifier | None = None
    partial_order: tuple[NeutralCommunityTemporalConstraint, ...] = ()
    reason: str | None = None


class NeutralCommunityContextDisplay(ImmutableRecord):
    wording: str = Field(min_length=1)
    lens: str = Field(min_length=1)
    target: str = Field(min_length=1)
    story_scope: NeutralCommunityTemporalDisplay
    spoiler_max_passage_order: int = Field(ge=0)
    spoiler_max_sentence_order: int = Field(ge=0)
    spoiler_max_token_order: int = Field(ge=0)
    spoiler_max_revelation_order: int | None = Field(default=None, ge=0)
    viewpoint_holder_id: Identifier | None = None
    viewpoint_attitude: Identifier | None = None
    abstraction: Identifier


class NeutralCommunityEvidenceRecord(ImmutableRecord):
    """Condition-neutral packet evidence with all join identifiers pseudonymized."""

    evidence_id: Identifier
    text: str = Field(min_length=1)
    passage_order: int = Field(ge=0)
    sentence_order: int = Field(ge=0)
    token_order: int = Field(ge=0)
    mention_surfaces: tuple[str, ...]
    event_trigger_surfaces: tuple[str, ...]
    relation_phrases: tuple[str, ...]
    temporal_clues: tuple[str, ...]
    provenance_method: str = Field(min_length=1)
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]


class NeutralCommunityPanelNode(ImmutableRecord):
    """One pseudonymized node with reviewer-readable, supported semantics."""

    node_id: Identifier
    cluster_id: Identifier
    node_kind: Literal["entity", "event"]
    label: str = Field(min_length=1)
    contextual_type_label: str = Field(min_length=1)
    contextual_type_definition: str = Field(min_length=1)
    contextual_role: str | None = None
    reification_reason: str | None = None
    abstraction: Identifier
    temporal_state: NeutralCommunityTemporalDisplay
    uncertainty: Identifier
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    description: str = Field(min_length=1)
    description_assertion_ids: tuple[Identifier, ...]
    evidence_ids: tuple[Identifier, ...]
    support_anchor_ids: tuple[Identifier, ...]
    supported_mention_surfaces: tuple[str, ...] = ()

    @model_validator(mode="after")
    def semantic_support_is_canonical(self) -> Self:
        for values in (
            self.description_assertion_ids,
            self.evidence_ids,
            self.support_anchor_ids,
        ):
            if not values or values != tuple(sorted(set(values))):
                raise ValueError("neutral community node support must be nonempty and canonical")
        if (self.node_kind == "entity") != (self.contextual_role is not None):
            raise ValueError("only neutral entity nodes carry a contextual role")
        if (self.node_kind == "event") != (self.reification_reason is not None):
            raise ValueError("only neutral event nodes carry a reification reason")
        return self


class NeutralCommunityPanelEdge(ImmutableRecord):
    edge_id: Identifier
    source_id: Identifier
    target_id: Identifier
    native_relation_id: Identifier
    predicate_label: str = Field(min_length=1)
    predicate_definition: str = Field(min_length=1)
    canonical_relation_category: Identifier | None = None


class NeutralCommunityRoleBinding(ImmutableRecord):
    role_label: str = Field(min_length=1)
    object_id: Identifier
    evidence_ids: tuple[Identifier, ...]


class NeutralCommunityPanelAssertionSupport(ImmutableRecord):
    """Readable qualified assertion; no gold target or source identifier is exposed."""

    assertion_id: Identifier
    predicate_label: str = Field(min_length=1)
    predicate_definition: str = Field(min_length=1)
    canonical_relation_category: Identifier | None = None
    subject_id: Identifier | None = None
    object_id: Identifier | None = None
    roles: tuple[NeutralCommunityRoleBinding, ...] = ()
    direction: Literal["forward", "inverse"]
    story_time: NeutralCommunityTemporalDisplay
    validity_time: NeutralCommunityTemporalDisplay
    discourse_passage_order: int = Field(ge=0)
    discourse_sentence_order: int = Field(ge=0)
    discourse_token_order: int = Field(ge=0)
    revelation_order: int = Field(ge=0)
    revelation_label: str | None = None
    epistemic_holder_id: Identifier | None = None
    epistemic_attitude: Identifier | None = None
    holder_relative_time: NeutralCommunityTemporalDisplay | None = None
    narrative_commitment: Identifier
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    contextual_relevance: Annotated[float, Field(ge=0.0, le=1.0)]
    evidence_ids: tuple[Identifier, ...]
    provenance_methods: tuple[str, ...]
    why_matters: str = Field(min_length=1)
    why_matters_evidence_ids: tuple[Identifier, ...]

    @model_validator(mode="after")
    def assertion_shape_and_support_are_complete(self) -> Self:
        binary = self.subject_id is not None and self.object_id is not None
        partial_binary = (self.subject_id is None) != (self.object_id is None)
        if partial_binary or binary == bool(self.roles):
            raise ValueError("neutral assertion requires exactly one binary or n-ary shape")
        for values in (self.evidence_ids, self.why_matters_evidence_ids):
            if not values or values != tuple(sorted(set(values))):
                raise ValueError("neutral assertion evidence must be nonempty and canonical")
        if not set(self.why_matters_evidence_ids).issubset(self.evidence_ids):
            raise ValueError("neutral why-matters support must be assertion evidence")
        epistemic_fields = (
            self.epistemic_holder_id,
            self.epistemic_attitude,
            self.holder_relative_time,
        )
        if any(value is None for value in epistemic_fields) and any(
            value is not None for value in epistemic_fields
        ):
            raise ValueError("neutral epistemic display must be wholly present or absent")
        return self


class NeutralCommunityReviewPanel(ImmutableRecord):
    """Reviewer-facing partition with experimental identities removed.

    Query/evidence text and supported projection semantics are retained so the three
    registered rubric dimensions are observable.  Native predicate, node, cluster,
    assertion, mention, evidence, temporal-anchor, and experimental identifiers are
    replaced with panel-local aliases.  Gold targets and expected effects are absent.
    """

    panel_schema: Literal["condition-blind-community-partition-panel-v2"] = (
        "condition-blind-community-partition-panel-v2"
    )
    condition_blind: Literal[True] = True
    source_identifiers_visible: Literal[False] = False
    context: NeutralCommunityContextDisplay
    evidence: tuple[NeutralCommunityEvidenceRecord, ...]
    nodes: tuple[NeutralCommunityPanelNode, ...]
    edges: tuple[NeutralCommunityPanelEdge, ...]
    assertion_support: tuple[NeutralCommunityPanelAssertionSupport, ...]
    cluster_count: int = Field(gt=0)

    @model_validator(mode="after")
    def neutral_inventory_is_canonical(self) -> Self:
        node_ids = tuple(item.node_id for item in self.nodes)
        edge_ids = tuple(item.edge_id for item in self.edges)
        assertion_ids = tuple(item.assertion_id for item in self.assertion_support)
        evidence_ids = tuple(item.evidence_id for item in self.evidence)
        if not node_ids or node_ids != tuple(sorted(set(node_ids))):
            raise ValueError("neutral community panel nodes must be nonempty and canonical")
        if edge_ids != tuple(sorted(set(edge_ids))):
            raise ValueError("neutral community panel edges must be canonical")
        if assertion_ids != tuple(sorted(set(assertion_ids))):
            raise ValueError("neutral community assertions must be canonical")
        if not evidence_ids or evidence_ids != tuple(sorted(set(evidence_ids))):
            raise ValueError("neutral community evidence must be nonempty and canonical")
        if any(
            endpoint not in set(node_ids)
            for edge in self.edges
            for endpoint in (edge.source_id, edge.target_id)
        ):
            raise ValueError("neutral community edge names an unknown node")
        if self.cluster_count != len({item.cluster_id for item in self.nodes}):
            raise ValueError("neutral community cluster count differs from node assignments")
        evidence_set = set(evidence_ids)
        if any(not set(item.evidence_ids).issubset(evidence_set) for item in self.nodes):
            raise ValueError("neutral community node cites absent panel evidence")
        if any(
            not set(item.evidence_ids).issubset(evidence_set)
            or not set(item.why_matters_evidence_ids).issubset(evidence_set)
            for item in self.assertion_support
        ):
            raise ValueError("neutral community assertion cites absent panel evidence")
        return self


class CommunityPartitionSource(ImmutableRecord):
    source_partition_id: Identifier
    world_id: Identifier
    context_id: Identifier
    condition: ConditionName
    seed_block: Literal[1, 2] | None
    resolution: Annotated[float, Field(gt=0.0)]
    projection_hash: Sha256Digest
    partition_hash: Sha256Digest
    node_count: int = Field(gt=0)
    cluster_count: int = Field(gt=0)
    review_panel: ReviewPanelSource

    @model_validator(mode="after")
    def primary_condition_only(self) -> Self:
        if self.condition not in _PRIMARY_CONDITIONS:
            raise ValueError("community review accepts only primary study conditions")
        if (self.condition is ConditionName.C0_CLASSICAL_PRE) != (self.seed_block is None):
            raise ValueError("only deterministic C0 community sources are unseeded")
        if self.cluster_count > self.node_count:
            raise ValueError("community cluster count cannot exceed its node count")
        return self


class CommunityReviewSourceManifest(ImmutableRecord):
    manifest_id: Identifier
    analysis_artifact_hash: Sha256Digest
    analysis_index_file_sha256: Sha256Digest
    phase4_replay_receipt_hash: Sha256Digest
    phase4_output_inventory_hash: Sha256Digest
    phase4_source_binding_inventory_hash: Sha256Digest
    analysis_configuration_hash: Sha256Digest
    metric_configuration_hash: Sha256Digest
    metric_version_hash: Sha256Digest
    primary_intended_manifest_hash: Sha256Digest
    primary_scores_file_sha256: Sha256Digest
    primary_scores_logical_hash: Sha256Digest
    complete_bundles_file_sha256: Sha256Digest
    complete_bundles_logical_hash: Sha256Digest
    scorer_plans_file_sha256: Sha256Digest
    scorer_plans_logical_hash: Sha256Digest
    selection_rule_id: Literal["preoutput-hash-first-paired-context-v2"] = (
        "preoutput-hash-first-paired-context-v2"
    )
    selection_rule_hash: Sha256Digest
    preoutput_candidate_inventory_hash: Sha256Digest
    preoutput_eligible_context_count: int = Field(gt=0, le=36)
    missing_output_policy: Literal["fail-without-substitution"] = (
        "fail-without-substitution"
    )
    selected_cell_lineage_hash: Sha256Digest
    registered_primary_conditions: tuple[
        Literal[ConditionName.C0_CLASSICAL_PRE],
        Literal[ConditionName.C1_LLM_PRE],
        Literal[ConditionName.C2_LLM_QUERY],
        Literal[ConditionName.A_FIXED_SELECT],
    ] = _PRIMARY_CONDITION_ORDER
    registered_resolution_values: tuple[float, float, float]
    selected_cell_count: Literal[4] = 4
    partitions_per_selected_cell: Literal[3] = 3
    eligible_partition_count: Literal[12] = 12
    items: tuple[CommunityPartitionSource, ...]
    selected_at: AwareDatetime
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED

    @model_validator(mode="after")
    def inventory_is_unique(self) -> Self:
        identifiers = [item.source_partition_id for item in self.items]
        cells = [
            (item.world_id, item.context_id, item.condition, item.seed_block, item.resolution)
            for item in self.items
        ]
        if not identifiers or len(identifiers) != len(set(identifiers)):
            raise ValueError("community source IDs must be nonempty and unique")
        if self.selection_rule_hash != FROZEN_COMMUNITY_REVIEW_SELECTION_RULE_HASH:
            raise ValueError("community review selection rule differs from its frozen policy")
        if self.registered_primary_conditions != _PRIMARY_CONDITION_ORDER:
            raise ValueError("community source primary-condition order changed")
        half, base, double = self.registered_resolution_values
        if not (
            math.isclose(half, base * 0.5, abs_tol=1e-15)
            and math.isclose(double, base * 2.0, abs_tol=1e-15)
        ):
            raise ValueError("community source resolutions are not registered half/base/double")
        if len(self.items) != self.eligible_partition_count:
            raise ValueError("community source must enumerate every rubric-eligible partition")
        if len(cells) != len(set(cells)):
            raise ValueError("community source inventory repeats a scored cell")
        if self.items != tuple(sorted(self.items, key=lambda item: item.source_partition_id)):
            raise ValueError("community partition sources must be sorted by opaque ID")
        by_condition: dict[ConditionName, list[CommunityPartitionSource]] = {}
        for item in self.items:
            by_condition.setdefault(item.condition, []).append(item)
        if set(by_condition) != set(_PRIMARY_CONDITION_ORDER):
            raise ValueError("community source does not cover every primary condition")
        registered_resolutions = set(self.registered_resolution_values)
        for condition, condition_items in by_condition.items():
            selected_cells = {
                (
                    item.world_id,
                    item.context_id,
                    item.seed_block,
                    item.projection_hash,
                )
                for item in condition_items
            }
            if (
                len(selected_cells) != 1
                or len(condition_items) != self.partitions_per_selected_cell
                or {item.resolution for item in condition_items} != registered_resolutions
            ):
                raise ValueError(
                    f"community source must retain one cell at all resolutions for {condition}"
                )
        selected_world_contexts = {
            (item.world_id, item.context_id) for item in self.items
        }
        if len(selected_world_contexts) != 1:
            raise ValueError("community review cells must form one paired world-context quartet")
        if any(
            item.condition is not ConditionName.C0_CLASSICAL_PRE and item.seed_block != 1
            for item in self.items
        ):
            raise ValueError("community review fixes the LLM comparison quartet to seed block 1")
        return self


class BlindedCommunityItem(ImmutableRecord):
    blinded_output_id: Identifier
    panel_file: RelativePath
    panel_sha256: Sha256Digest
    media_type: Literal["image/png", "application/json"]


class BlindedCommunityReviewPackage(ImmutableRecord):
    package_id: Identifier
    condition_blind: Literal[True] = True
    template_id: Literal["condition-blind-community-review-v1"]
    rubric_revision: Literal["community-coherence-rubric-v1"]
    rubric_template_hash: Sha256Digest
    source_manifest_hash: Sha256Digest
    items: tuple[BlindedCommunityItem, ...]
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED


class CommunityRejoinEntry(ImmutableRecord):
    blinded_output_id: Identifier
    source_partition_id: Identifier
    world_id: Identifier
    context_id: Identifier
    condition: ConditionName
    seed_block: Literal[1, 2] | None
    resolution: float
    projection_hash: Sha256Digest
    partition_hash: Sha256Digest
    node_count: int
    cluster_count: int

    @model_validator(mode="after")
    def seed_shape_matches_condition(self) -> Self:
        if (self.condition is ConditionName.C0_CLASSICAL_PRE) != (
            self.seed_block is None
        ):
            raise ValueError("only deterministic C0 community rejoin entries are unseeded")
        return self


class CommunityReviewRejoinMap(ImmutableRecord):
    map_id: Identifier
    package_hash: Sha256Digest
    source_manifest_hash: Sha256Digest
    entries: tuple[CommunityRejoinEntry, ...]
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED


class CommunityReviewCompletion(ImmutableRecord):
    completion_id: Identifier
    package_hash: Sha256Digest
    reviewer_id: Identifier
    completed_at: AwareDatetime
    reviews: tuple[CommunityReviewEntry, ...]
    condition_blind: Literal[True] = True
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED


class CommunityReviewFinalization(ImmutableRecord):
    finalization_id: Identifier
    package_hash: Sha256Digest
    rejoin_map_hash: Sha256Digest
    completion_hash: Sha256Digest
    canonical_table_file: Literal["community_blind_review.csv"] = (
        "community_blind_review.csv"
    )
    canonical_table_sha256: Sha256Digest
    reviewed_partition_count: int = Field(gt=0)
    independent_unit: Literal["world"] = "world"
    complete: Literal[True] = True
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED


def _file_sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _model_bytes(value: ImmutableRecord) -> bytes:
    return (value.to_canonical_json() + "\n").encode("utf-8")


def _assert_no_symlink_chain(path: Path) -> None:
    current = path.absolute()
    while True:
        if current.is_symlink():
            raise BlindedReviewError(f"symlinked review path is forbidden: {path}")
        if current.parent == current:
            return
        current = current.parent


def _restricted_root(path: Path) -> Path:
    _assert_no_symlink_chain(path)
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise BlindedReviewError("restricted review root does not exist") from error
    if not resolved.is_dir():
        raise BlindedReviewError("restricted review root is not a directory")
    return resolved


def _contained_path(
    restricted_root: Path,
    path: Path,
    *,
    must_exist: bool,
    regular_file: bool = False,
) -> Path:
    root = _restricted_root(restricted_root)
    candidate = path.absolute()
    if not candidate.is_relative_to(restricted_root.absolute()):
        raise BlindedReviewError("review path is outside the explicit restricted root")
    _assert_no_symlink_chain(candidate)
    if must_exist:
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as error:
            raise BlindedReviewError("required review input is missing") from error
        if not resolved.is_relative_to(root):
            raise BlindedReviewError("review input escapes its restricted root")
        if regular_file and not resolved.is_file():
            raise BlindedReviewError("review input is not a regular file")
        return resolved
    parent = candidate
    while not parent.exists():
        if parent.parent == parent:
            raise BlindedReviewError("review output has no existing ancestor")
        parent = parent.parent
    resolved_parent = parent.resolve(strict=True)
    if not resolved_parent.is_relative_to(root):
        raise BlindedReviewError("review output escapes its restricted root")
    return candidate


def _safe_config_file(path: Path) -> Path:
    _assert_no_symlink_chain(path)
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise BlindedReviewError(f"missing frozen review configuration: {path.name}") from error
    if not resolved.is_file():
        raise BlindedReviewError("frozen review configuration is not a regular file")
    return resolved


def _relative_path(value: str) -> PurePosixPath:
    if "\\" in value:
        raise BlindedReviewError("backslashes are forbidden in review panel paths")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise BlindedReviewError("unsafe review panel path")
    return path


def _read_limited(path: Path) -> bytes:
    size = path.stat().st_size
    if size <= 0 or size > _MAX_REVIEW_FILE_BYTES:
        raise BlindedReviewError("review file is empty or exceeds 128 MiB")
    return path.read_bytes()


def _read_model(
    restricted_root: Path,
    path: Path,
    model_type: type[ImmutableRecord],
) -> ImmutableRecord:
    source = _contained_path(
        restricted_root,
        path,
        must_exist=True,
        regular_file=True,
    )
    try:
        return model_type.model_validate_json(_read_limited(source))
    except BlindedReviewError:
        raise
    except Exception as error:
        raise BlindedReviewError(f"invalid {model_type.__name__}: {error}") from error


def _panel_bytes(
    restricted_root: Path,
    source_manifest_path: Path,
    panel: ReviewPanelSource,
) -> bytes:
    relative = _relative_path(panel.relative_path)
    source = _contained_path(
        restricted_root,
        source_manifest_path.parent.joinpath(*relative.parts),
        must_exist=True,
        regular_file=True,
    )
    payload = _read_limited(source)
    if _file_sha256_bytes(payload) != panel.file_sha256:
        raise BlindedReviewError("review panel hash mismatch")
    return payload


def _blind_id(prefix: str, source_hash: str, source_id: str) -> str:
    digest = hashlib.sha256(
        f"story-projection-onto/{prefix}/{source_hash}/{source_id}".encode()
    ).hexdigest()
    return f"{prefix}-{digest[:24]}"


def _verify_existing_bundle(path: Path, files: Mapping[str, bytes]) -> None:
    if not path.is_dir() or path.is_symlink():
        raise BlindedReviewError("existing append-only review target is not a directory")
    actual: set[str] = set()
    for item in path.rglob("*"):
        if item.is_symlink():
            raise BlindedReviewError(
                "existing append-only review bundle contains a symlink"
            )
        try:
            mode = item.lstat().st_mode
        except OSError as error:
            raise BlindedReviewError(
                "existing append-only review bundle inventory is unreadable"
            ) from error
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise BlindedReviewError(
                "existing append-only review bundle contains a special file"
            )
        actual.add(item.relative_to(path).as_posix())
    if actual != set(files):
        raise BlindedReviewError("existing review bundle inventory differs")
    for relative, expected in files.items():
        candidate = path.joinpath(*PurePosixPath(relative).parts)
        _assert_no_symlink_chain(candidate)
        if candidate.read_bytes() != expected:
            raise BlindedReviewError("existing append-only review bundle differs")


def _materialize_bundle(
    *,
    restricted_root: Path,
    output_root: Path,
    identity_hash: str,
    files: Mapping[str, bytes],
) -> tuple[Path, Literal["created", "verified"]]:
    target_parent = _contained_path(
        restricted_root,
        output_root,
        must_exist=False,
    )
    target_parent.mkdir(parents=True, exist_ok=True)
    _assert_no_symlink_chain(target_parent)
    target = target_parent / identity_hash
    if target.name != identity_hash:
        raise BlindedReviewError("content-addressed review target name changed")
    if target.exists() or target.is_symlink():
        _verify_existing_bundle(target, files)
        return target, "verified"
    stage = Path(tempfile.mkdtemp(prefix=f".{identity_hash[:16]}.", dir=target_parent))
    try:
        for relative, payload in sorted(files.items()):
            logical = _relative_path(relative)
            destination = stage.joinpath(*logical.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        os.rename(stage, target)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return target, "created"


class _PrimaryScoreLineage(ImmutableRecord):
    """Restricted join between one Phase 4 score and its terminal runtime receipt."""

    intended_unit_id: Identifier
    condition: ConditionName
    seed_block: Literal[1, 2] | None
    context_id: Identifier
    snapshot_hash: Sha256Digest
    packet_hash: Sha256Digest
    context_hash: Sha256Digest
    result_receipt_hash: Sha256Digest
    runtime_outcome: RunOutcome


def _canonical_external_model(
    path: Path,
    model_type: type[ImmutableRecord],
    *,
    label: str,
) -> tuple[ImmutableRecord, bytes]:
    _assert_no_symlink_chain(path)
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise BlindedReviewError(f"missing {label}") from error
    if not resolved.is_file():
        raise BlindedReviewError(f"{label} is not a regular file")
    payload = _read_limited(resolved)
    try:
        value = model_type.model_validate_json(payload)
    except Exception as error:
        raise BlindedReviewError(f"invalid {label}: {error}") from error
    assert isinstance(value, ImmutableRecord)
    if payload != _model_bytes(value):
        raise BlindedReviewError(f"{label} is not canonically serialized")
    return value, payload


def _canonical_jsonl_models(
    path: Path,
    model_type: type[ImmutableRecord],
    *,
    label: str,
    allow_empty: bool = False,
) -> tuple[tuple[ImmutableRecord, ...], bytes]:
    _assert_no_symlink_chain(path)
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise BlindedReviewError(f"missing {label}") from error
    if not resolved.is_file():
        raise BlindedReviewError(f"{label} is not a regular file")
    size = resolved.stat().st_size
    if size > _MAX_REVIEW_FILE_BYTES or (size == 0 and not allow_empty):
        raise BlindedReviewError(f"{label} is empty or exceeds 128 MiB")
    payload = resolved.read_bytes()
    if not payload:
        return (), payload
    if payload.startswith(b"\xef\xbb\xbf") or b"\r" in payload or not payload.endswith(b"\n"):
        raise BlindedReviewError(f"{label} is not canonical JSONL")
    values: list[ImmutableRecord] = []
    for line in payload.splitlines():
        if not line:
            raise BlindedReviewError(f"{label} contains an empty JSONL row")
        try:
            raw = json.loads(line)
            value = model_type.model_validate(raw)
        except Exception as error:
            raise BlindedReviewError(f"invalid {label} row: {error}") from error
        assert isinstance(value, ImmutableRecord)
        if line != value.to_canonical_json().encode("utf-8"):
            raise BlindedReviewError(f"{label} row is not canonically serialized")
        values.append(value)
    return tuple(values), payload


def _phase4_binding(
    index: Phase4AnalysisIndex,
    role: str,
    value: ImmutableRecord,
    payload: bytes,
) -> None:
    bindings = {item.role: item for item in index.source_bindings}
    if len(bindings) != len(index.source_bindings) or set(bindings) != _PHASE4_SOURCE_ROLES:
        raise BlindedReviewError("Phase 4 source bindings lack the exact frozen inventory")
    binding = bindings[role]
    if (
        binding.file_sha256 != _file_sha256_bytes(payload)
        or binding.logical_content_hash != value.content_hash
    ):
        raise BlindedReviewError(f"Phase 4 {role} binding differs from its source file")


def _runtime_failure_kind(outcome: RunOutcome) -> OutputFailureKind:
    return {
        RunOutcome.TIMED_OUT: OutputFailureKind.TIMEOUT,
        RunOutcome.INVALID: OutputFailureKind.VALIDATION_INVALID,
        RunOutcome.INTERRUPTED: OutputFailureKind.EXECUTION_FAILURE,
        RunOutcome.FAILED: OutputFailureKind.EXECUTION_FAILURE,
    }.get(outcome, OutputFailureKind.EXECUTION_FAILURE)


def _primary_score_lineage(
    *,
    call_manifest: HeldOutCallManifest,
    execution: HeldOutExecutionManifest,
) -> tuple[Mapping[str, _PrimaryScoreLineage], str]:
    """Reconstruct all 252 score joins and hash all 288 retained primary receipts."""

    if execution.call_manifest_hash != call_manifest.content_hash:
        raise BlindedReviewError("held-out execution names another call manifest")
    unit_ids = {item.unit_id for item in call_manifest.units}
    if (
        len(execution.c0_constructions) != 12
        or len(unit_ids) != 12
        or {item.unit_id for item in execution.c0_constructions} != unit_ids
    ):
        raise BlindedReviewError("C0 construction receipts do not cover all held-out units")

    stage_by_hash = {
        stage.staging_manifest_hash: (unit.unit_id, stage)
        for unit in call_manifest.units
        for stage in unit.query_stages
    }
    if len(stage_by_hash) != 36:
        raise BlindedReviewError("held-out call manifest lacks 36 unique query stages")
    opening_by_hash = {item.sealed_stage_hash: item for item in execution.query_openings}
    if (
        len(opening_by_hash) != len(execution.query_openings)
        or set(opening_by_hash) != set(stage_by_hash)
    ):
        raise BlindedReviewError("held-out query openings differ from the frozen query stages")

    expected_projection_keys = {
        (unit.unit_id, stage.staging_manifest_hash, condition, seed)
        for unit in call_manifest.units
        for stage in unit.query_stages
        for condition, seeds in (
            (ConditionName.C0_CLASSICAL_PRE, (None,)),
            (ConditionName.C1_LLM_PRE, (1, 2)),
        )
        for seed in seeds
    }
    projection_by_key = {
        (item.unit_id, item.query_stage_hash, item.condition, item.seed_block): item
        for item in execution.preconstructed_projections
    }
    if (
        len(projection_by_key) != len(execution.preconstructed_projections)
        or set(projection_by_key) != expected_projection_keys
    ):
        raise BlindedReviewError(
            "preconstructed projection receipts lack the exact 108-cell inventory"
        )

    score_lineage: dict[str, _PrimaryScoreLineage] = {}
    for key, receipt in projection_by_key.items():
        unit_id, stage_hash, condition, seed = key
        expected_unit_id, stage = stage_by_hash[stage_hash]
        opening = opening_by_hash[stage_hash]
        if expected_unit_id != unit_id or opening.query_context is None:
            raise BlindedReviewError("preconstructed receipt has invalid query-stage lineage")
        intended_unit_id = (
            f"primary:{unit_id}:{stage.stage_id}:{condition.value}:s{seed or 0}"
        )
        score_lineage[intended_unit_id] = _PrimaryScoreLineage(
            intended_unit_id=intended_unit_id,
            condition=condition,
            seed_block=seed,
            context_id=opening.query_context.context_id,
            snapshot_hash=stage.snapshot_hash,
            packet_hash=opening.evidence_packet_hash,
            context_hash=opening.query_context.content_hash,
            result_receipt_hash=receipt.content_hash,
            runtime_outcome=receipt.outcome,
        )

    if len(call_manifest.calls) != 168 or len(execution.itt_records) != 168:
        raise BlindedReviewError("held-out model-call ITT inventory must contain 168 records")
    prequery_c1_count = 0
    for call, record in zip(call_manifest.calls, execution.itt_records, strict=True):
        if (
            record.ordinal != call.ordinal
            or record.call_spec_hash != call.content_hash
            or record.result.call_id != call.call_id
            or record.result.condition is not call.condition
        ):
            raise BlindedReviewError("held-out ITT receipt differs from its call specification")
        if call.query_stage_hash is None:
            if call.condition is not ConditionName.C1_LLM_PRE:
                raise BlindedReviewError("only C1 preconstruction calls may be query-blind")
            prequery_c1_count += 1
            continue
        try:
            expected_unit_id, stage = stage_by_hash[call.query_stage_hash]
            opening = opening_by_hash[call.query_stage_hash]
        except KeyError as error:
            raise BlindedReviewError("query-time ITT receipt names an unknown stage") from error
        if (
            expected_unit_id != call.unit_id
            or opening.query_context is None
            or call.condition
            not in {ConditionName.C2_LLM_QUERY, ConditionName.A_FIXED_SELECT}
        ):
            raise BlindedReviewError("query-time ITT receipt has invalid stage lineage")
        intended_unit_id = (
            f"primary:{call.unit_id}:{stage.stage_id}:{call.condition.value}:"
            f"s{call.seed_block}"
        )
        if intended_unit_id in score_lineage:
            raise BlindedReviewError("primary score lineage contains a duplicate cell")
        score_lineage[intended_unit_id] = _PrimaryScoreLineage(
            intended_unit_id=intended_unit_id,
            condition=call.condition,
            seed_block=call.seed_block,
            context_id=opening.query_context.context_id,
            snapshot_hash=stage.snapshot_hash,
            packet_hash=opening.evidence_packet_hash,
            context_hash=opening.query_context.content_hash,
            result_receipt_hash=record.content_hash,
            runtime_outcome=record.result.outcome,
        )

    if prequery_c1_count != 24 or len(score_lineage) != REGISTERED_PRIMARY_SCORE_COUNT:
        raise BlindedReviewError(
            "primary receipt lineage does not reproduce 24 C1 preconstructions and 252 scores"
        )
    receipt_inventory = {
        "c0_construction_receipts": tuple(
            item.content_hash for item in execution.c0_constructions
        ),
        "preconstructed_projection_receipts": tuple(
            item.content_hash for item in execution.preconstructed_projections
        ),
        "model_call_itt_receipts": tuple(item.content_hash for item in execution.itt_records),
    }
    receipt_hashes = tuple(
        item for values in receipt_inventory.values() for item in values
    )
    if len(receipt_hashes) != 288:
        raise AssertionError("internal 288-receipt primary contract changed")
    if len(set(receipt_hashes)) != 288:
        raise BlindedReviewError("primary execution repeats a retained result receipt")
    return score_lineage, canonical_sha256(receipt_inventory)


def _score_failure_gate_ids(
    score: IntendedUnitScore,
    *,
    output_status: str,
) -> tuple[str, ...]:
    if not score.output_valid:
        return (f"output-status-{output_status}",)
    rows = {item.metric_name: item for item in score.rows}
    if len(rows) != len(score.rows):
        raise BlindedReviewError("Phase 4 intended score repeats a metric row")
    failed: list[str] = []
    for gate in FROZEN_REGISTERED_FAILURE_SIGNAL_GATES:
        try:
            row = rows[gate.metric_name]
        except KeyError as error:
            raise BlindedReviewError(
                f"Phase 4 intended score lacks registered failure signal {gate.metric_name}"
            ) from error
        if row.status is not PipelineMetricStatus.VALUE:
            continue
        assert row.value is not None
        if (
            gate.direction is FailureSignalDirection.BELOW
            and row.value < gate.threshold
        ) or (
            gate.direction is FailureSignalDirection.ABOVE
            and row.value > gate.threshold
        ):
            failed.append(gate.gate_id)
    return tuple(sorted(failed))


def _neutral_panel_metric(row: MetricResultRow) -> NeutralFailurePanelMetric:
    return NeutralFailurePanelMetric(
        metric_name=row.metric_name,
        status=row.status,
        value=row.value,
        numerator=row.numerator,
        denominator=row.denominator,
    )


def _neutral_failure_panel(
    *,
    score: IntendedUnitScore,
    bundle: CompleteProjectionScoreBundle | None,
    output_status: Literal["succeeded", "failed", "timed_out", "invalid", "interrupted"],
    failed_gate_ids: tuple[str, ...],
) -> NeutralFailureReviewPanel:
    rows = {item.metric_name: item for item in score.rows}
    metrics = tuple(
        sorted(
            (
                _neutral_panel_metric(rows[gate.metric_name])
                for gate in FROZEN_REGISTERED_FAILURE_SIGNAL_GATES
            ),
            key=lambda item: item.metric_name,
        )
    )
    if bundle is None:
        return NeutralFailureReviewPanel(
            output_status=output_status,
            failed_gate_ids=failed_gate_ids,
            metrics=metrics,
            structurally_valid=False if output_status == "invalid" else None,
        )

    adapter = bundle.projection.adapter
    node_alias = {
        value: f"node-{index:03d}"
        for index, value in enumerate(sorted(adapter.node_ids), 1)
    }
    relations = sorted(
        {
            item
            for edge in adapter.edges
            for item in (edge.native_predicate_id, edge.canonical_predicate_id)
            if item is not None
        }
        | {
            item
            for relation in adapter.assertion_relations
            for item in (
                relation.native_predicate_id,
                relation.canonical_predicate_id,
            )
            if item is not None
        }
    )
    relation_alias = {
        value: f"relation-{index:03d}" for index, value in enumerate(relations, 1)
    }
    edges = tuple(
        NeutralFailurePanelEdge(
            edge_id=f"edge-{index:03d}",
            source_id=node_alias[edge.source_id],
            target_id=node_alias[edge.target_id],
            native_relation_id=relation_alias[edge.native_predicate_id],
            canonical_relation_id=(
                None
                if edge.canonical_predicate_id is None
                else relation_alias[edge.canonical_predicate_id]
            ),
        )
        for index, edge in enumerate(
            sorted(
                adapter.edges,
                key=lambda item: (
                    node_alias[item.source_id],
                    node_alias[item.target_id],
                    relation_alias[item.native_predicate_id],
                    item.edge_id,
                ),
            ),
            1,
        )
    )
    assertion_relations = tuple(
        sorted(
            relation_alias[item.native_predicate_id]
            for item in adapter.assertion_relations
        )
    )
    decisions = tuple(
        sorted(
            (
                NeutralFailurePanelDecision(
                    family=item.family.value,
                    operator=item.operator.value,
                    anchor_count=len(item.anchor_ids),
                )
                for item in adapter.normalized_decisions
            ),
            key=lambda item: (item.family, item.operator, item.anchor_count),
        )
    )
    return NeutralFailureReviewPanel(
        output_status=output_status,
        failed_gate_ids=failed_gate_ids,
        metrics=metrics,
        structurally_valid=adapter.structurally_valid,
        nodes=tuple(sorted(node_alias.values())),
        edges=edges,
        assertion_relation_ids=assertion_relations,
        decisions=decisions,
    )


def _output_status(
    score: IntendedUnitScore,
    runtime_outcome: RunOutcome,
) -> Literal["succeeded", "failed", "timed_out", "invalid", "interrupted"]:
    if score.output_valid:
        return "succeeded"
    if runtime_outcome is RunOutcome.SUCCEEDED:
        return "invalid"
    if runtime_outcome in {
        RunOutcome.FAILED,
        RunOutcome.TIMED_OUT,
        RunOutcome.INVALID,
        RunOutcome.INTERRUPTED,
    }:
        return runtime_outcome.value
    raise BlindedReviewError("primary score binds a nonterminal runtime outcome")


def _assert_panel_hides_source_identity(
    payload: bytes,
    *,
    score: IntendedUnitScore,
    lineage: _PrimaryScoreLineage,
    projection_hash: str | None,
) -> None:
    forbidden = {
        score.intended_unit.unit_id,
        score.intended_unit.job_id,
        score.intended_unit.world_id,
        score.intended_unit.context_id,
        score.intended_unit.condition.value,
        lineage.result_receipt_hash,
        *(value for value in (projection_hash, score.projection_id) if value is not None),
    }
    for value in forbidden:
        encoded = json.dumps(value, ensure_ascii=False).encode("utf-8")
        if encoded in payload:
            raise BlindedReviewError("condition-blind panel leaked a source identity")


def prepare_held_out_failure_source(
    *,
    repository: Path,
    phase4_configuration_path: Path,
    phase4_output_root: Path,
    held_out_root: Path,
    selected_at: datetime,
) -> tuple[HeldOutFailureSourceManifest, dict[str, bytes]]:
    """Derive the exhaustive Section-19 failure inventory without discretion.

    The complete Phase 4 bundle is replayed before selection.  The 252 intended
    score cells are then joined to the exact 12 C0 construction, 108 projection,
    and 168 model-call ITT receipts retained by the primary execution (288 total
    unique receipt records).  A cell is eligible iff it is not a valid terminal
    success or at least one frozen metric gate fails.
    """

    if selected_at.tzinfo is None or selected_at.utcoffset() is None:
        raise BlindedReviewError("failure-source selection time must be timezone-aware")
    try:
        replay = replay_phase4_analysis_outputs(
            repository=repository,
            configuration_path=phase4_configuration_path,
            output_root=phase4_output_root,
        )
    except (OSError, Phase4AnalysisError, ValueError) as error:
        raise BlindedReviewError(f"complete Phase 4 replay failed: {error}") from error

    index_value, index_payload = _canonical_external_model(
        phase4_output_root / "analysis_index.json",
        Phase4AnalysisIndex,
        label="Phase 4 analysis index",
    )
    intended_value, _ = _canonical_external_model(
        phase4_output_root / "inputs/primary_intended_manifest.json",
        IntendedMetricManifest,
        label="Phase 4 primary intended manifest",
    )
    score_values, score_payload = _canonical_jsonl_models(
        phase4_output_root / "scores/primary.jsonl",
        IntendedUnitScore,
        label="Phase 4 primary scores",
    )
    bundle_values, _ = _canonical_jsonl_models(
        phase4_output_root / "scores/complete_bundles.jsonl",
        CompleteProjectionScoreBundle,
        label="Phase 4 complete projection bundles",
        allow_empty=True,
    )
    call_value, call_payload = _canonical_external_model(
        held_out_root / "call_manifest.json",
        HeldOutCallManifest,
        label="held-out call manifest",
    )
    execution_value, execution_payload = _canonical_external_model(
        held_out_root / "execution_manifest.json",
        HeldOutExecutionManifest,
        label="held-out execution manifest",
    )
    assert isinstance(index_value, Phase4AnalysisIndex)
    assert isinstance(intended_value, IntendedMetricManifest)
    assert isinstance(call_value, HeldOutCallManifest)
    assert isinstance(execution_value, HeldOutExecutionManifest)
    scores = tuple(value for value in score_values if isinstance(value, IntendedUnitScore))
    bundles = tuple(
        value for value in bundle_values if isinstance(value, CompleteProjectionScoreBundle)
    )
    if len(scores) != len(score_values) or len(bundles) != len(bundle_values):
        raise AssertionError("typed Phase 4 loader returned another record class")

    _phase4_binding(index_value, "held_out_call_manifest", call_value, call_payload)
    _phase4_binding(index_value, "held_out_execution", execution_value, execution_payload)
    if selected_at < index_value.completed_at:
        raise BlindedReviewError("failure-source selection predates completed Phase 4 analysis")
    if (
        replay.analysis_index_hash != index_value.content_hash
        or len(scores) != REGISTERED_PRIMARY_SCORE_COUNT
        or intended_value.units != tuple(item.intended_unit for item in scores)
        or intended_value.content_hash != index_value.primary_intended_manifest_hash
    ):
        raise BlindedReviewError("Phase 4 primary score denominator or replay binding changed")
    primary_output = next(
        (
            item
            for item in index_value.output_files
            if item.relative_path == "scores/primary.jsonl"
        ),
        None,
    )
    if (
        primary_output is None
        or primary_output.file_sha256 != _file_sha256_bytes(score_payload)
        or primary_output.logical_content_hash
        != canonical_sha256(tuple(json.loads(line) for line in score_payload.splitlines()))
        or primary_output.row_count != REGISTERED_PRIMARY_SCORE_COUNT
    ):
        raise BlindedReviewError("Phase 4 primary score file differs from its exact index entry")

    lineage_by_unit, receipt_inventory_hash = _primary_score_lineage(
        call_manifest=call_value,
        execution=execution_value,
    )
    score_by_unit = {item.intended_unit.unit_id: item for item in scores}
    if len(score_by_unit) != len(scores) or set(score_by_unit) != set(lineage_by_unit):
        raise BlindedReviewError("Phase 4 scores do not cover the exact receipt-derived cells")
    bundle_by_hash = {item.content_hash: item for item in bundles}
    if len(bundle_by_hash) != len(bundles):
        raise BlindedReviewError("Phase 4 complete bundle inventory contains duplicates")

    items: list[HeldOutFailureSource] = []
    panel_files: dict[str, bytes] = {}
    eligibility_rows: list[tuple[str, str, str, tuple[str, ...]]] = []
    for score in scores:
        intended = score.intended_unit
        lineage = lineage_by_unit[intended.unit_id]
        if (
            intended.condition is not lineage.condition
            or intended.seed_block != lineage.seed_block
            or intended.context_id != lineage.context_id
            or intended.snapshot_hash != lineage.snapshot_hash
            or intended.packet_hash != lineage.packet_hash
            or intended.context_hash != lineage.context_hash
        ):
            raise BlindedReviewError("Phase 4 intended cell differs from its runtime receipt")
        if score.output_valid and lineage.runtime_outcome is not RunOutcome.SUCCEEDED:
            raise BlindedReviewError("valid Phase 4 score binds a non-success runtime receipt")
        expected_failure_kind = (
            OutputFailureKind.VALIDATION_INVALID
            if lineage.runtime_outcome is RunOutcome.SUCCEEDED and not score.output_valid
            else _runtime_failure_kind(lineage.runtime_outcome)
        )
        if not score.output_valid and score.failure_kind is not expected_failure_kind:
            raise BlindedReviewError("Phase 4 failure kind differs from its runtime outcome")
        status = _output_status(score, lineage.runtime_outcome)
        failed_gate_ids = _score_failure_gate_ids(score, output_status=status)
        eligibility_rows.append(
            (
                intended.content_hash,
                lineage.result_receipt_hash,
                status,
                failed_gate_ids,
            )
        )
        if not failed_gate_ids:
            continue
        bundle = (
            None
            if score.projection_bundle_hash is None
            else bundle_by_hash.get(score.projection_bundle_hash)
        )
        if score.output_valid and bundle is None:
            raise BlindedReviewError("valid Phase 4 failure candidate lacks its complete bundle")
        if bundle is not None and bundle.metric_rows != score.rows:
            raise BlindedReviewError("Phase 4 score rows differ from their complete bundle")
        projection_hash = (
            None if bundle is None else bundle.projection.adapter.projection_hash
        )
        panel = _neutral_failure_panel(
            score=score,
            bundle=bundle,
            output_status=status,
            failed_gate_ids=failed_gate_ids,
        )
        panel_payload = _model_bytes(panel)
        _assert_panel_hides_source_identity(
            panel_payload,
            score=score,
            lineage=lineage,
            projection_hash=projection_hash,
        )
        source_failure_id = (
            "failure-"
            + hashlib.sha256(
                (
                    "story-projection-onto/held-out-failure/v1/"
                    f"{index_value.content_hash}/{intended.content_hash}"
                ).encode()
            ).hexdigest()[:24]
        )
        relative_panel = f"panels/{source_failure_id}.json"
        if relative_panel in panel_files:
            raise BlindedReviewError("opaque failure-panel identity collision")
        panel_files[relative_panel] = panel_payload
        items.append(
            HeldOutFailureSource(
                source_failure_id=source_failure_id,
                world_id=intended.world_id,
                context_id=intended.context_id,
                condition=intended.condition,
                seed_block=intended.seed_block,
                result_artifact_hash=lineage.result_receipt_hash,
                projection_hash=projection_hash,
                output_status=status,
                failed_gate_ids=failed_gate_ids,
                review_panel=ReviewPanelSource(
                    relative_path=relative_panel,
                    file_sha256=_file_sha256_bytes(panel_payload),
                    media_type="application/json",
                ),
            )
        )

    if not items:
        raise BlindedReviewError(
            "the frozen selection rule found no held-out failures to review"
        )
    try:
        closing_replay = replay_phase4_analysis_outputs(
            repository=repository,
            configuration_path=phase4_configuration_path,
            output_root=phase4_output_root,
        )
    except (OSError, Phase4AnalysisError, ValueError) as error:
        raise BlindedReviewError(f"closing Phase 4 replay failed: {error}") from error
    closing_call, closing_call_payload = _canonical_external_model(
        held_out_root / "call_manifest.json",
        HeldOutCallManifest,
        label="closing held-out call manifest",
    )
    closing_execution, closing_execution_payload = _canonical_external_model(
        held_out_root / "execution_manifest.json",
        HeldOutExecutionManifest,
        label="closing held-out execution manifest",
    )
    if (
        closing_replay != replay
        or closing_call != call_value
        or closing_call_payload != call_payload
        or closing_execution != execution_value
        or closing_execution_payload != execution_payload
    ):
        raise BlindedReviewError("failure-source inputs changed during deterministic selection")
    ordered_items = tuple(sorted(items, key=lambda item: item.source_failure_id))
    manifest = HeldOutFailureSourceManifest(
        manifest_id=f"heldout-failures-{index_value.content_hash[:20]}",
        analysis_artifact_hash=index_value.content_hash,
        analysis_index_file_sha256=_file_sha256_bytes(index_payload),
        phase4_replay_receipt_hash=replay.content_hash,
        phase4_output_inventory_hash=replay.output_inventory_hash,
        primary_intended_manifest_hash=intended_value.content_hash,
        primary_scores_file_sha256=_file_sha256_bytes(score_payload),
        primary_scores_logical_hash=canonical_sha256(
            tuple(json.loads(line) for line in score_payload.splitlines())
        ),
        held_out_call_manifest_hash=call_value.content_hash,
        held_out_call_manifest_file_sha256=_file_sha256_bytes(call_payload),
        held_out_execution_hash=execution_value.content_hash,
        held_out_execution_file_sha256=_file_sha256_bytes(execution_payload),
        primary_receipt_inventory_hash=receipt_inventory_hash,
        scored_cell_eligibility_hash=canonical_sha256(tuple(eligibility_rows)),
        failure_signal_policy_hash=FROZEN_REGISTERED_FAILURE_SIGNAL_POLICY_HASH,
        eligible_failure_count=len(ordered_items),
        items=ordered_items,
        selected_at=selected_at,
    )
    files = {"source_manifest.json": _model_bytes(manifest), **panel_files}
    return manifest, files


def materialize_held_out_failure_source(
    *,
    repository: Path,
    phase4_configuration_path: Path,
    phase4_output_root: Path,
    held_out_root: Path,
    selected_at: datetime,
    restricted_root: Path,
    output_root: Path,
) -> tuple[Path, Literal["created", "verified"], HeldOutFailureSourceManifest]:
    manifest, files = prepare_held_out_failure_source(
        repository=repository,
        phase4_configuration_path=phase4_configuration_path,
        phase4_output_root=phase4_output_root,
        held_out_root=held_out_root,
        selected_at=selected_at,
    )
    path, state = _materialize_bundle(
        restricted_root=restricted_root,
        output_root=output_root,
        identity_hash=manifest.content_hash,
        files=files,
    )
    return path, state, manifest


def _require_phase4_output_binding(
    index: Phase4AnalysisIndex,
    *,
    relative_path: str,
    payload: bytes,
    logical_hash: str,
    row_count: int,
) -> None:
    output = next(
        (item for item in index.output_files if item.relative_path == relative_path),
        None,
    )
    if (
        output is None
        or output.file_sha256 != _file_sha256_bytes(payload)
        or output.logical_content_hash != logical_hash
        or output.row_count != row_count
    ):
        raise BlindedReviewError(
            f"Phase 4 {relative_path} differs from its exact analysis-index entry"
        )


def _community_selection_rank(context_lineage_hash: str) -> str:
    return hashlib.sha256(
        (
            "story-projection-onto/community-review-selection/v2/"
            f"{context_lineage_hash}"
        ).encode()
    ).hexdigest()


def _neutral_community_panel(
    *,
    bundle: CompleteProjectionScoreBundle,
    partition: LeidenPartition,
    canonical_relation_vocabulary: tuple[str, ...],
) -> NeutralCommunityReviewPanel:
    adapter = bundle.projection.adapter
    replay = bundle.scorer_only_replay
    projection = replay.projection
    context = replay.context
    packet = replay.evidence_packet
    node_alias = {
        value: f"node-{index:03d}"
        for index, value in enumerate(sorted(adapter.node_ids), 1)
    }
    evidence_alias = {
        value: f"source-{index:03d}"
        for index, value in enumerate(sorted(packet.ordered_evidence_ids), 1)
    }
    assertion_alias = {
        item.assertion_id: f"claim-{index:03d}"
        for index, item in enumerate(
            sorted(projection.instance_graph.assertions, key=lambda value: value.assertion_id),
            1,
        )
    }

    temporal_values = [
        context.story_scope,
        *(item.temporal_state for item in projection.instance_graph.entities),
        *(item.occurrence_time for item in projection.instance_graph.events),
        *(
            temporal
            for item in projection.instance_graph.assertions
            for temporal in (
                item.temporal_scope.story_time,
                item.temporal_scope.validity_time,
            )
        ),
        *(
            item.epistemic_scope.holder_relative_time
            for item in projection.instance_graph.assertions
            if item.epistemic_scope is not None
        ),
    ]
    reference_values = {
        reference
        for temporal in temporal_values
        for reference in (
            temporal.anchor_id,
            *(
                endpoint
                for constraint in temporal.partial_order
                for endpoint in (constraint.left_id, constraint.right_id)
            ),
        )
        if reference is not None and reference not in node_alias
    }
    reference_values.update(
        value
        for value in (
            context.viewpoint.holder_id if context.viewpoint is not None else None,
            *(
                endpoint
                for item in projection.instance_graph.assertions
                for endpoint in (
                    item.subject_id,
                    item.object_id,
                    *(role.object_id for role in item.roles),
                )
            ),
            *(
                item.epistemic_scope.holder_id
                for item in projection.instance_graph.assertions
                if item.epistemic_scope is not None
            ),
        )
        if value is not None and value not in node_alias
    )
    reference_alias = {
        value: f"referent-{index:03d}"
        for index, value in enumerate(sorted(reference_values), 1)
    }

    def neutral_reference(value: str) -> str:
        if value in node_alias:
            return node_alias[value]
        try:
            return reference_alias[value]
        except KeyError as error:
            raise BlindedReviewError(
                "community semantic display encountered an unbound reference"
            ) from error

    def neutral_temporal(value: Any) -> NeutralCommunityTemporalDisplay:
        return NeutralCommunityTemporalDisplay(
            kind=value.kind.value,
            point=value.point,
            start=value.start,
            end=value.end,
            label=value.label,
            anchor_reference_id=(
                neutral_reference(value.anchor_id)
                if value.anchor_id is not None
                else None
            ),
            relation=value.relation.value if value.relation is not None else None,
            partial_order=tuple(
                sorted(
                    (
                        NeutralCommunityTemporalConstraint(
                            left_reference_id=neutral_reference(item.left_id),
                            relation=item.relation.value,
                            right_reference_id=neutral_reference(item.right_id),
                        )
                        for item in value.partial_order
                    ),
                    key=lambda item: (
                        item.left_reference_id,
                        item.relation,
                        item.right_reference_id,
                    ),
                )
            ),
            reason=value.reason,
        )

    support_values = {
        support
        for object_id, supports in adapter.object_anchors
        if object_id in node_alias
        for support in supports
    } | {
        support
        for assertion in adapter.assertion_semantics
        for support in (*assertion.anchor_ids, *assertion.evidence_ids)
    }
    support_alias = {
        value: f"anchor-{index:03d}"
        for index, value in enumerate(sorted(support_values), 1)
    }
    native_relations = sorted({item.native_predicate_id for item in adapter.edges})
    native_relation_alias = {
        value: f"predicate-slot-{index:03d}"
        for index, value in enumerate(native_relations, 1)
    }
    cluster_members: dict[str, tuple[str, ...]] = {}
    for cluster_id in {cluster for _, cluster in partition.assignments}:
        cluster_members[cluster_id] = tuple(
            sorted(
                node_alias[node_id]
                for node_id, assigned_cluster in partition.assignments
                if assigned_cluster == cluster_id
            )
        )
    cluster_alias = {
        cluster_id: f"group-{index:03d}"
        for index, cluster_id in enumerate(
            sorted(cluster_members, key=lambda item: cluster_members[item]),
            1,
        )
    }
    assignment_by_node = dict(partition.assignments)
    anchors_by_object = dict(adapter.object_anchors)
    local_types = {
        item.type_id: item for item in projection.local_schema.contextual_types
    }
    local_predicates = {
        item.predicate_id: item for item in projection.local_schema.predicates
    }
    mention_surfaces = {
        item.candidate_id: item.surface
        for evidence in packet.evidence
        for item in evidence.mention_candidates
    }
    evidence = tuple(
        NeutralCommunityEvidenceRecord(
            evidence_id=evidence_alias[item.evidence_id],
            text=item.text,
            passage_order=item.discourse_position.passage_order,
            sentence_order=item.discourse_position.sentence_order,
            token_order=item.discourse_position.token_order,
            mention_surfaces=tuple(
                sorted({candidate.surface for candidate in item.mention_candidates})
            ),
            event_trigger_surfaces=tuple(
                sorted({candidate.trigger_surface for candidate in item.event_candidates})
            ),
            relation_phrases=tuple(
                sorted(
                    {candidate.surface_phrase for candidate in item.relation_phrase_candidates}
                )
            ),
            temporal_clues=tuple(
                sorted(
                    {
                        (
                            candidate.normalized_expression
                            if candidate.relation is None
                            else (
                                f"{candidate.normalized_expression} "
                                f"[{candidate.relation.value}]"
                            )
                        )
                        for candidate in item.temporal_clues
                    }
                )
            ),
            provenance_method=item.provenance.extraction_method,
            confidence=item.confidence,
        )
        for item in sorted(packet.evidence, key=lambda value: evidence_alias[value.evidence_id])
    )

    def node_values(node_id: str) -> tuple[str, Any]:
        entity = next(
            (
                item
                for item in projection.instance_graph.entities
                if item.entity_id == node_id
            ),
            None,
        )
        if entity is not None:
            return "entity", entity
        event = next(
            (item for item in projection.instance_graph.events if item.event_id == node_id),
            None,
        )
        if event is None:
            raise BlindedReviewError("metric adapter names a node absent from its projection")
        return "event", event

    nodes = tuple(
        NeutralCommunityPanelNode(
            node_id=node_alias[node_id],
            cluster_id=cluster_alias[assignment_by_node[node_id]],
            node_kind=node_kind,
            label=node.label,
            contextual_type_label=local_types[node.contextual_type_id].label,
            contextual_type_definition=local_types[node.contextual_type_id].definition,
            contextual_role=(node.contextual_role if node_kind == "entity" else None),
            reification_reason=(node.reification_reason if node_kind == "event" else None),
            abstraction=(
                node.abstraction.value
                if node_kind == "entity"
                else local_types[node.contextual_type_id].abstraction.value
            ),
            temporal_state=neutral_temporal(
                node.temporal_state if node_kind == "entity" else node.occurrence_time
            ),
            uncertainty=node.uncertainty.value,
            confidence=node.confidence,
            description=node.description,
            description_assertion_ids=tuple(
                sorted(assertion_alias[value] for value in node.description_assertion_ids)
            ),
            evidence_ids=tuple(
                sorted(evidence_alias[value] for value in node.evidence_ids)
            ),
            support_anchor_ids=tuple(
                sorted(
                    support_alias[value]
                    for value in anchors_by_object.get(node_id, ())
                )
            ),
            supported_mention_surfaces=(
                tuple(
                    sorted(
                        mention_surfaces[value]
                        for value in node.supported_mention_candidate_ids
                    )
                )
                if node_kind == "entity"
                else ()
            ),
        )
        for node_id in sorted(adapter.node_ids)
        for node_kind, node in (node_values(node_id),)
    )
    edges = tuple(
        NeutralCommunityPanelEdge(
            edge_id=f"edge-{index:03d}",
            source_id=node_alias[edge.source_id],
            target_id=node_alias[edge.target_id],
            native_relation_id=native_relation_alias[edge.native_predicate_id],
            predicate_label=local_predicates[edge.native_predicate_id].label,
            predicate_definition=local_predicates[edge.native_predicate_id].definition,
            canonical_relation_category=(
                edge.canonical_predicate_id
                if edge.canonical_predicate_id in canonical_relation_vocabulary
                else None
            ),
        )
        for index, edge in enumerate(
            sorted(
                adapter.edges,
                key=lambda item: (
                    node_alias[item.source_id],
                    node_alias[item.target_id],
                    native_relation_alias[item.native_predicate_id],
                    item.edge_id,
                ),
            ),
            1,
        )
    )
    relation_by_assertion = {
        item.assertion_id: item.canonical_predicate_id
        for item in adapter.assertion_relations
    }
    assertion_support = tuple(
        NeutralCommunityPanelAssertionSupport(
            assertion_id=assertion_alias[item.assertion_id],
            predicate_label=local_predicates[item.predicate_id].label,
            predicate_definition=local_predicates[item.predicate_id].definition,
            canonical_relation_category=(
                relation_by_assertion.get(item.assertion_id)
                if relation_by_assertion.get(item.assertion_id)
                in canonical_relation_vocabulary
                else None
            ),
            subject_id=(
                neutral_reference(item.subject_id)
                if item.subject_id is not None
                else None
            ),
            object_id=(
                neutral_reference(item.object_id)
                if item.object_id is not None
                else None
            ),
            roles=tuple(
                NeutralCommunityRoleBinding(
                    role_label=role.role,
                    object_id=neutral_reference(role.object_id),
                    evidence_ids=tuple(
                        sorted(evidence_alias[value] for value in role.evidence_ids)
                    ),
                )
                for role in item.roles
            ),
            direction=item.direction,
            story_time=neutral_temporal(item.temporal_scope.story_time),
            validity_time=neutral_temporal(item.temporal_scope.validity_time),
            discourse_passage_order=item.temporal_scope.discourse_position.passage_order,
            discourse_sentence_order=item.temporal_scope.discourse_position.sentence_order,
            discourse_token_order=item.temporal_scope.discourse_position.token_order,
            revelation_order=item.temporal_scope.revelation_position.revelation_order,
            revelation_label=item.temporal_scope.revelation_position.label,
            epistemic_holder_id=(
                neutral_reference(item.epistemic_scope.holder_id)
                if item.epistemic_scope is not None
                else None
            ),
            epistemic_attitude=(
                item.epistemic_scope.attitude.value
                if item.epistemic_scope is not None
                else None
            ),
            holder_relative_time=(
                neutral_temporal(item.epistemic_scope.holder_relative_time)
                if item.epistemic_scope is not None
                else None
            ),
            narrative_commitment=item.narrative_commitment.value,
            confidence=item.confidence,
            contextual_relevance=item.contextual_relevance,
            evidence_ids=tuple(
                sorted(evidence_alias[value] for value in item.evidence_ids)
            ),
            provenance_methods=tuple(
                sorted({value.extraction_method for value in item.provenance})
            ),
            why_matters=item.why_matters,
            why_matters_evidence_ids=tuple(
                sorted(evidence_alias[value] for value in item.why_matters_evidence_ids)
            ),
        )
        for item in sorted(
            projection.instance_graph.assertions,
            key=lambda value: value.assertion_id,
        )
    )
    context_display = NeutralCommunityContextDisplay(
        wording=context.wording,
        lens=context.lens,
        target=context.target,
        story_scope=neutral_temporal(context.story_scope),
        spoiler_max_passage_order=(
            context.spoiler_horizon.max_discourse_position.passage_order
        ),
        spoiler_max_sentence_order=(
            context.spoiler_horizon.max_discourse_position.sentence_order
        ),
        spoiler_max_token_order=context.spoiler_horizon.max_discourse_position.token_order,
        spoiler_max_revelation_order=(
            context.spoiler_horizon.max_revelation_position.revelation_order
            if context.spoiler_horizon.max_revelation_position is not None
            else None
        ),
        viewpoint_holder_id=(
            neutral_reference(context.viewpoint.holder_id)
            if context.viewpoint is not None
            else None
        ),
        viewpoint_attitude=(
            context.viewpoint.attitude_scope.value
            if context.viewpoint is not None
            and context.viewpoint.attitude_scope is not None
            else None
        ),
        abstraction=context.abstraction.value,
    )
    return NeutralCommunityReviewPanel(
        context=context_display,
        evidence=evidence,
        nodes=nodes,
        edges=edges,
        assertion_support=assertion_support,
        cluster_count=partition.cluster_count,
    )


def _normalized_blinding_text(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def _decoded_string_leaves(value: object) -> tuple[str, ...]:
    result: list[str] = []

    def visit(current: object) -> None:
        if isinstance(current, str):
            result.append(current)
        elif isinstance(current, Mapping):
            for key, child in current.items():
                result.append(str(key))
                visit(child)
        elif isinstance(current, list):
            for child in current:
                visit(child)

    visit(value)
    return tuple(result)


def _decoded_mapping_keys(value: object) -> tuple[str, ...]:
    result: list[str] = []

    def visit(current: object) -> None:
        if isinstance(current, Mapping):
            for key, child in current.items():
                result.append(str(key))
                visit(child)
        elif isinstance(current, list):
            for child in current:
                visit(child)

    visit(value)
    return tuple(result)


def _contains_normalized_identity(text: str, identity: str) -> bool:
    """Use substring matching, with token boundaries only for very short aliases."""

    normalized_text = _normalized_blinding_text(text)
    normalized_identity = _normalized_blinding_text(identity)
    if not normalized_identity:
        return False
    if len(normalized_identity) >= 4:
        return normalized_identity in normalized_text
    start = 0
    while True:
        index = normalized_text.find(normalized_identity, start)
        if index < 0:
            return False
        before = normalized_text[index - 1] if index else ""
        after_index = index + len(normalized_identity)
        after = normalized_text[after_index] if after_index < len(normalized_text) else ""
        if (not before or not before.isalnum()) and (not after or not after.isalnum()):
            return True
        start = index + 1


def _evidence_candidate_identifier(candidate: object) -> str:
    """Return one known candidate identifier and reject missing/ambiguous schemas."""

    values = {
        value
        for field in (
            "candidate_id",
            "mention_id",
            "event_id",
            "relation_phrase_id",
        )
        for value in (getattr(candidate, field, None),)
        if isinstance(value, str) and value
    }
    if len(values) != 1:
        raise BlindedReviewError(
            "condition-blind community panel encountered an unbound evidence candidate"
        )
    return next(iter(values))


def _temporal_clue_identifier(clue: object) -> str:
    """Return one known temporal-clue ID and reject missing/ambiguous schemas."""

    values = {
        value
        for field in ("clue_id", "temporal_clue_id")
        for value in (getattr(clue, field, None),)
        if isinstance(value, str) and value
    }
    if len(values) != 1:
        raise BlindedReviewError(
            "condition-blind community panel encountered an unbound temporal clue"
        )
    return next(iter(values))


def _assert_community_panel_hides_source_identity(
    payload: bytes,
    *,
    score: IntendedUnitScore,
    bundle: CompleteProjectionScoreBundle,
    partition: LeidenPartition,
    canonical_relation_vocabulary: tuple[str, ...],
) -> None:
    intended = score.intended_unit
    adapter = bundle.projection.adapter
    replay = bundle.scorer_only_replay
    projection = replay.projection
    context = replay.context
    packet = replay.evidence_packet
    forbidden = {
        intended.unit_id,
        intended.job_id,
        intended.world_id,
        intended.context_id,
        intended.condition.value,
        intended.snapshot_hash,
        intended.packet_hash,
        intended.context_hash,
        intended.scorer_plan_hash,
        score.content_hash,
        score.projection_id,
        score.projection_bundle_hash,
        bundle.content_hash,
        replay.content_hash,
        adapter.projection_id,
        adapter.projection_hash,
        projection.projection_id,
        projection.snapshot_hash,
        projection.packet_hash,
        projection.context_hash,
        projection.query_access_event_hash,
        projection.generation_lineage_hash,
        projection.raw_output_artifact_hash,
        projection.normalized_draft_hash,
        projection.validation_bundle_hash,
        projection.local_schema.schema_id,
        projection.upper_ontology.ontology_id,
        projection.run_id,
        context.context_id,
        context.spoiler_horizon.horizon_id,
        packet.packet_id,
        packet.snapshot_hash,
        *(item.value for item in _PRIMARY_CONDITION_ORDER),
        *(item.type_id for item in projection.local_schema.contextual_types),
        *(item.predicate_id for item in projection.local_schema.predicates),
        *(item.entity_id for item in projection.instance_graph.entities),
        *(item.event_id for item in projection.instance_graph.events),
        *(
            item.proposition_content_id
            for item in projection.instance_graph.proposition_contents
        ),
        *(item.assertion_id for item in projection.instance_graph.assertions),
        *(item.decision_id for item in projection.decisions),
        *(item.evidence_id for item in packet.evidence),
        *(item.passage_id for item in packet.evidence),
        *(item.text_hash for item in packet.evidence),
        *(item.provenance.provenance_id for item in packet.evidence),
        *(item.provenance.locator for item in packet.evidence),
        *(item.provenance.source_artifact_hash for item in packet.evidence),
        *(
            _evidence_candidate_identifier(candidate)
            for item in packet.evidence
            for candidate in (
                *item.mention_candidates,
                *item.event_candidates,
                *item.relation_phrase_candidates,
            )
        ),
        *(
            _temporal_clue_identifier(candidate)
            for item in packet.evidence
            for candidate in item.temporal_clues
        ),
        *(item for item in adapter.node_ids),
        *(item.edge_id for item in adapter.edges),
        *(item.native_predicate_id for item in adapter.edges),
        *(item.assertion_id for item in adapter.assertion_semantics),
        *(item.semantic_signature for item in adapter.assertion_semantics),
        *(value for _, values in adapter.object_anchors for value in values),
        *(value for item in adapter.assertion_semantics for value in item.anchor_ids),
        *(value for item in adapter.assertion_semantics for value in item.evidence_ids),
        *(cluster for _, cluster in partition.assignments),
    }
    forbidden.discard(None)
    # A native predicate may legitimately equal a frozen, condition-independent
    # canonical bin.  That shared category is intentionally reviewer-visible.
    forbidden.difference_update(canonical_relation_vocabulary)
    try:
        decoded = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BlindedReviewError("condition-blind community panel is invalid JSON") from error
    leaves = _decoded_string_leaves(decoded)
    for identity in forbidden:
        if any(_contains_normalized_identity(value, identity) for value in leaves):
            raise BlindedReviewError(
                "condition-blind community panel leaked a source identity"
            )
    forbidden_keys = {
            "condition",
            "context_id",
            "gold",
            "gold_hash",
            "expected_effect",
            "partition_hash",
            "projection_hash",
            "resolution",
            "seed_block",
            "source_partition_id",
            "world_id",
    }
    normalized_keys = {
        _normalized_blinding_text(value) for value in _decoded_mapping_keys(decoded)
    }
    if any(
        key in forbidden_keys
        or key.startswith("gold_")
        or "expected_effect" in key
        for key in normalized_keys
    ):
        raise BlindedReviewError("condition-blind community panel exposed a rejoin field")


def _community_partition_is_available(
    *,
    score: IntendedUnitScore,
    bundle_by_hash: Mapping[str, CompleteProjectionScoreBundle],
    plan: ScorerMetricPlan,
    metric_configuration: StudyMetricConfiguration,
) -> CompleteProjectionScoreBundle | None:
    if not score.output_valid:
        return None
    if score.projection_bundle_hash is None:
        raise BlindedReviewError("valid community candidate lacks projection-bundle lineage")
    bundle = bundle_by_hash.get(score.projection_bundle_hash)
    if bundle is None:
        raise BlindedReviewError("valid community candidate lacks its complete Phase 4 bundle")
    if bundle.metric_rows != score.rows:
        raise BlindedReviewError("community candidate score rows differ from its complete bundle")
    replay = bundle.scorer_only_replay
    intended = score.intended_unit
    replay_projection = replay.projection
    if (
        replay_projection.condition is not intended.condition
        or replay_projection.snapshot_hash != intended.snapshot_hash
        or replay_projection.packet_hash != intended.packet_hash
        or replay_projection.context_hash != intended.context_hash
        or replay.context.content_hash != intended.context_hash
        or replay.evidence_packet.content_hash != intended.packet_hash
    ):
        raise BlindedReviewError(
            "community candidate scorer-only projection/input lineage differs"
        )
    recomputed_adapter = adapt_projection_for_metrics(
        replay_projection,
        metric_configuration,
    )
    if recomputed_adapter != bundle.projection.adapter:
        raise BlindedReviewError(
            "community candidate adapter differs from exact projection recomputation"
        )
    structural = bundle.projection.structural
    if not structural.valid_content_bearing:
        return None
    partitions = (
        structural.leiden_half,
        structural.leiden_base,
        structural.leiden_double,
    )
    if not plan.community_eligible:
        if bundle.community is not None:
            raise BlindedReviewError("community-ineligible scorer plan produced a rubric partition")
        return None
    if bundle.community is None or any(item is None for item in partitions):
        raise BlindedReviewError(
            "eligible content-bearing community cell lacks all registered partitions"
        )
    expected_resolutions = (
        metric_configuration.leiden_half_resolution,
        metric_configuration.leiden_base_resolution,
        metric_configuration.leiden_double_resolution,
    )
    for partition, expected_resolution in zip(partitions, expected_resolutions, strict=True):
        assert partition is not None
        recomputed_partition = run_leiden_cpm(
            recomputed_adapter.node_ids,
            recomputed_adapter.edges,
            LeidenConfiguration(
                resolution=expected_resolution,
                seed=metric_configuration.leiden_seed,
            ),
        )
        if (
            not math.isclose(partition.resolution, expected_resolution, abs_tol=1e-15)
            or partition.seed != metric_configuration.leiden_seed
            or tuple(node_id for node_id, _ in partition.assignments)
            != tuple(sorted(recomputed_adapter.node_ids))
            or partition.cluster_count <= 0
            or partition != recomputed_partition
        ):
            raise BlindedReviewError(
                "community partition differs from exact registered Leiden recomputation"
            )
    return bundle


def prepare_community_review_source(
    *,
    repository: Path,
    phase4_configuration_path: Path,
    phase4_output_root: Path,
    selected_at: datetime,
) -> tuple[CommunityReviewSourceManifest, dict[str, bytes]]:
    """Select a small, balanced rubric panel from immutable Phase 4 outputs.

    Candidate membership is fixed by the pre-output intended manifest and reviewed
    community eligibility.  The first same-world/context primary quartet under a
    condition-independent hash rank is selected before output validity is inspected,
    with seed block 1 frozen for LLM conditions and deterministic C0 unseeded.  If
    that frozen quartet is incomplete, the review fails without substituting a later
    context.  All three registered resolutions are retained, yielding twelve panels.
    """

    if selected_at.tzinfo is None or selected_at.utcoffset() is None:
        raise BlindedReviewError("community-source selection time must be timezone-aware")
    try:
        replay = replay_phase4_analysis_outputs(
            repository=repository,
            configuration_path=phase4_configuration_path,
            output_root=phase4_output_root,
        )
        analysis_configuration = Phase4AnalysisConfiguration.load(
            repository.resolve(strict=True), phase4_configuration_path
        )
        metric_configuration = StudyMetricConfiguration.load(
            repository / analysis_configuration.metric_configuration_path,
            seed_manifest_path=repository / analysis_configuration.seed_manifest_path,
        )
    except (OSError, Phase4AnalysisError, ValueError) as error:
        raise BlindedReviewError(f"complete Phase 4 replay failed: {error}") from error

    index_value, index_payload = _canonical_external_model(
        phase4_output_root / "analysis_index.json",
        Phase4AnalysisIndex,
        label="Phase 4 analysis index",
    )
    intended_value, intended_payload = _canonical_external_model(
        phase4_output_root / "inputs/primary_intended_manifest.json",
        IntendedMetricManifest,
        label="Phase 4 primary intended manifest",
    )
    score_values, score_payload = _canonical_jsonl_models(
        phase4_output_root / "scores/primary.jsonl",
        IntendedUnitScore,
        label="Phase 4 primary scores",
    )
    bundle_values, bundle_payload = _canonical_jsonl_models(
        phase4_output_root / "scores/complete_bundles.jsonl",
        CompleteProjectionScoreBundle,
        label="Phase 4 complete projection bundles",
        allow_empty=True,
    )
    plan_values, plan_payload = _canonical_jsonl_models(
        phase4_output_root / "inputs/scorer_plans.jsonl",
        ScorerMetricPlan,
        label="Phase 4 scorer plans",
    )
    assert isinstance(index_value, Phase4AnalysisIndex)
    assert isinstance(intended_value, IntendedMetricManifest)
    # The canonical loaders validate every row against these exact model classes.
    scores = cast(tuple[IntendedUnitScore, ...], tuple(score_values))
    bundles = cast(tuple[CompleteProjectionScoreBundle, ...], tuple(bundle_values))
    plans = cast(tuple[ScorerMetricPlan, ...], tuple(plan_values))

    if selected_at < index_value.completed_at:
        raise BlindedReviewError("community selection predates completed Phase 4 analysis")
    if (
        replay.analysis_index_hash != index_value.content_hash
        or replay.analysis_configuration_hash != analysis_configuration.content_hash
        or replay.metric_configuration_hash != metric_configuration.content_hash
        or replay.metric_version_hash != metric_configuration.metric_version_hash
        or index_value.analysis_configuration_hash != analysis_configuration.content_hash
        or index_value.metric_configuration_hash != metric_configuration.content_hash
        or index_value.metric_version_hash != metric_configuration.metric_version_hash
        or len(scores) != REGISTERED_PRIMARY_SCORE_COUNT
        or intended_value.units != tuple(item.intended_unit for item in scores)
        or intended_value.content_hash != index_value.primary_intended_manifest_hash
    ):
        raise BlindedReviewError("Phase 4 community source denominator or configuration changed")

    payloads = {
        "inputs/primary_intended_manifest.json": (
            intended_payload,
            intended_value.content_hash,
            1,
        ),
        "scores/primary.jsonl": (
            score_payload,
            canonical_sha256(tuple(json.loads(line) for line in score_payload.splitlines())),
            len(scores),
        ),
        "scores/complete_bundles.jsonl": (
            bundle_payload,
            canonical_sha256(tuple(json.loads(line) for line in bundle_payload.splitlines())),
            len(bundles),
        ),
        "inputs/scorer_plans.jsonl": (
            plan_payload,
            canonical_sha256(tuple(json.loads(line) for line in plan_payload.splitlines())),
            len(plans),
        ),
    }
    for relative_path, (payload, logical_hash, row_count) in payloads.items():
        _require_phase4_output_binding(
            index_value,
            relative_path=relative_path,
            payload=payload,
            logical_hash=logical_hash,
            row_count=row_count,
        )

    expected_primary_counts = {
        ConditionName.C0_CLASSICAL_PRE: 36,
        ConditionName.C1_LLM_PRE: 72,
        ConditionName.C2_LLM_QUERY: 72,
        ConditionName.A_FIXED_SELECT: 72,
    }
    observed_primary_counts = {
        condition: sum(item.condition is condition for item in intended_value.units)
        for condition in _PRIMARY_CONDITION_ORDER
    }
    if observed_primary_counts != expected_primary_counts or any(
        item.condition not in _PRIMARY_CONDITIONS for item in intended_value.units
    ):
        raise BlindedReviewError("community source lacks the exact 252 primary-condition cells")

    plan_by_hash = {item.content_hash: item for item in plans}
    bundle_by_hash = {item.content_hash: item for item in bundles}
    if len(plan_by_hash) != len(plans) or len(bundle_by_hash) != len(bundles):
        raise BlindedReviewError("Phase 4 scorer-plan or complete-bundle inventory repeats hashes")
    if any(item.scorer_plan_hash not in plan_by_hash for item in intended_value.units):
        raise BlindedReviewError("primary intended cell lacks its exact scorer plan")

    score_by_cell = {
        (
            item.intended_unit.world_id,
            item.intended_unit.context_id,
            item.intended_unit.condition,
            item.intended_unit.seed_block,
        ): item
        for item in scores
    }
    if len(score_by_cell) != len(scores):
        raise BlindedReviewError("primary score inventory repeats a condition cell")
    quartet_conditions = (
        (ConditionName.C0_CLASSICAL_PRE, None),
        (ConditionName.C1_LLM_PRE, 1),
        (ConditionName.C2_LLM_QUERY, 1),
        (ConditionName.A_FIXED_SELECT, 1),
    )
    preoutput_candidates: list[tuple[str, str, tuple[str, ...]]] = []
    ranked_quartets: list[
        tuple[str, str, tuple[IntendedUnitScore, ...], ScorerMetricPlan]
    ] = []
    c0_scores = tuple(
        item
        for item in scores
        if item.intended_unit.condition is ConditionName.C0_CLASSICAL_PRE
    )
    for c0_score in c0_scores:
        c0_intended = c0_score.intended_unit
        try:
            quartet = tuple(
                score_by_cell[
                    (
                        c0_intended.world_id,
                        c0_intended.context_id,
                        condition,
                        seed,
                    )
                ]
                for condition, seed in quartet_conditions
            )
        except KeyError as error:
            raise BlindedReviewError(
                "primary intended cells lack a fixed-seed paired comparison quartet"
            ) from error
        lineage_values = {
            (
                item.intended_unit.snapshot_hash,
                item.intended_unit.packet_hash,
                item.intended_unit.context_hash,
                item.intended_unit.scorer_plan_hash,
            )
            for item in quartet
        }
        if len(lineage_values) != 1:
            raise BlindedReviewError(
                "paired community candidates do not share evidence/context/scorer lineage"
            )
        snapshot_hash, packet_hash, context_hash, scorer_plan_hash = next(
            iter(lineage_values)
        )
        plan = plan_by_hash[scorer_plan_hash]
        if not plan.community_eligible:
            continue
        context_lineage_hash = canonical_sha256(
            {
                "world_id": c0_intended.world_id,
                "context_id": c0_intended.context_id,
                "snapshot_hash": snapshot_hash,
                "packet_hash": packet_hash,
                "context_hash": context_hash,
                "scorer_plan_hash": scorer_plan_hash,
            }
        )
        rank = _community_selection_rank(context_lineage_hash)
        preoutput_candidates.append(
            (
                context_lineage_hash,
                rank,
                tuple(item.intended_unit.content_hash for item in quartet),
            )
        )
        ranked_quartets.append((rank, context_lineage_hash, quartet, plan))

    if not preoutput_candidates:
        raise BlindedReviewError("no pre-output context is eligible for community review")
    selected_rank, _, frozen_quartet, selected_plan = min(
        ranked_quartets,
        key=lambda item: (item[0], item[1]),
    )
    selected_quartet: list[
        tuple[IntendedUnitScore, CompleteProjectionScoreBundle]
    ] = []
    for score in frozen_quartet:
        candidate = _community_partition_is_available(
            score=score,
            bundle_by_hash=bundle_by_hash,
            plan=selected_plan,
            metric_configuration=metric_configuration,
        )
        if candidate is None:
            raise BlindedReviewError(
                "pre-output-selected community quartet is unavailable; "
                "post-output substitution is forbidden"
            )
        selected_quartet.append((score, candidate))
    if len(selected_quartet) != len(quartet_conditions):
        raise BlindedReviewError(
            "pre-output-selected community quartet is incomplete"
        )
    selected = tuple(
        (selected_rank, score, bundle) for score, bundle in selected_quartet
    )

    resolution_values = (
        metric_configuration.leiden_half_resolution,
        metric_configuration.leiden_base_resolution,
        metric_configuration.leiden_double_resolution,
    )
    items: list[CommunityPartitionSource] = []
    panel_files: dict[str, bytes] = {}
    selected_lineage: list[tuple[str, str, str, str]] = []
    for rank, score, bundle in selected:
        intended = score.intended_unit
        selected_lineage.append(
            (intended.content_hash, rank, score.content_hash, bundle.content_hash)
        )
        structural = bundle.projection.structural
        partitions = (
            structural.leiden_half,
            structural.leiden_base,
            structural.leiden_double,
        )
        for resolution_role, partition in zip(
            _COMMUNITY_RESOLUTION_ROLES, partitions, strict=True
        ):
            assert partition is not None
            source_partition_id = (
                "community-partition-"
                + hashlib.sha256(
                    (
                        "story-projection-onto/community-partition/v1/"
                        f"{index_value.content_hash}/{intended.content_hash}/{resolution_role}"
                    ).encode()
                ).hexdigest()[:24]
            )
            panel = _neutral_community_panel(
                bundle=bundle,
                partition=partition,
                canonical_relation_vocabulary=metric_configuration.canonical_relation_vocabulary,
            )
            panel_payload = _model_bytes(panel)
            _assert_community_panel_hides_source_identity(
                panel_payload,
                score=score,
                bundle=bundle,
                partition=partition,
                canonical_relation_vocabulary=metric_configuration.canonical_relation_vocabulary,
            )
            relative_panel = f"panels/{source_partition_id}.json"
            if relative_panel in panel_files:
                raise BlindedReviewError("opaque community-panel identity collision")
            panel_files[relative_panel] = panel_payload
            items.append(
                CommunityPartitionSource(
                    source_partition_id=source_partition_id,
                    world_id=intended.world_id,
                    context_id=intended.context_id,
                    condition=intended.condition,
                    seed_block=intended.seed_block,
                    resolution=partition.resolution,
                    projection_hash=bundle.projection.adapter.projection_hash,
                    partition_hash=partition.content_hash,
                    node_count=len(partition.assignments),
                    cluster_count=partition.cluster_count,
                    review_panel=ReviewPanelSource(
                        relative_path=relative_panel,
                        file_sha256=_file_sha256_bytes(panel_payload),
                        media_type="application/json",
                    ),
                )
            )

    try:
        closing_replay = replay_phase4_analysis_outputs(
            repository=repository,
            configuration_path=phase4_configuration_path,
            output_root=phase4_output_root,
        )
    except (OSError, Phase4AnalysisError, ValueError) as error:
        raise BlindedReviewError(f"closing Phase 4 replay failed: {error}") from error
    closing_files = (
        (phase4_output_root / "analysis_index.json", index_payload),
        (phase4_output_root / "inputs/primary_intended_manifest.json", intended_payload),
        (phase4_output_root / "scores/primary.jsonl", score_payload),
        (phase4_output_root / "scores/complete_bundles.jsonl", bundle_payload),
        (phase4_output_root / "inputs/scorer_plans.jsonl", plan_payload),
    )
    if closing_replay != replay or any(
        path.read_bytes() != expected for path, expected in closing_files
    ):
        raise BlindedReviewError("community-source inputs changed during deterministic selection")

    ordered_items = tuple(sorted(items, key=lambda item: item.source_partition_id))
    manifest = CommunityReviewSourceManifest(
        manifest_id=f"community-source-{index_value.content_hash[:20]}",
        analysis_artifact_hash=index_value.content_hash,
        analysis_index_file_sha256=_file_sha256_bytes(index_payload),
        phase4_replay_receipt_hash=replay.content_hash,
        phase4_output_inventory_hash=replay.output_inventory_hash,
        phase4_source_binding_inventory_hash=canonical_sha256(index_value.source_bindings),
        analysis_configuration_hash=analysis_configuration.content_hash,
        metric_configuration_hash=metric_configuration.content_hash,
        metric_version_hash=metric_configuration.metric_version_hash,
        primary_intended_manifest_hash=intended_value.content_hash,
        primary_scores_file_sha256=_file_sha256_bytes(score_payload),
        primary_scores_logical_hash=payloads["scores/primary.jsonl"][1],
        complete_bundles_file_sha256=_file_sha256_bytes(bundle_payload),
        complete_bundles_logical_hash=payloads["scores/complete_bundles.jsonl"][1],
        scorer_plans_file_sha256=_file_sha256_bytes(plan_payload),
        scorer_plans_logical_hash=payloads["inputs/scorer_plans.jsonl"][1],
        selection_rule_hash=FROZEN_COMMUNITY_REVIEW_SELECTION_RULE_HASH,
        preoutput_candidate_inventory_hash=canonical_sha256(
            tuple(sorted(preoutput_candidates))
        ),
        preoutput_eligible_context_count=len(preoutput_candidates),
        selected_cell_lineage_hash=canonical_sha256(tuple(selected_lineage)),
        registered_resolution_values=resolution_values,
        items=ordered_items,
        selected_at=selected_at,
    )
    return manifest, {"source_manifest.json": _model_bytes(manifest), **panel_files}


def materialize_community_review_source(
    *,
    repository: Path,
    phase4_configuration_path: Path,
    phase4_output_root: Path,
    selected_at: datetime,
    restricted_root: Path,
    output_root: Path,
) -> tuple[Path, Literal["created", "verified"], CommunityReviewSourceManifest]:
    manifest, files = prepare_community_review_source(
        repository=repository,
        phase4_configuration_path=phase4_configuration_path,
        phase4_output_root=phase4_output_root,
        selected_at=selected_at,
    )
    path, state = _materialize_bundle(
        restricted_root=restricted_root,
        output_root=output_root,
        identity_hash=manifest.content_hash,
        files=files,
    )
    return path, state, manifest


def load_error_taxonomy(path: Path) -> HeldOutErrorTaxonomy:
    try:
        return HeldOutErrorTaxonomy.model_validate_json(_safe_config_file(path).read_bytes())
    except BlindedReviewError:
        raise
    except Exception as error:
        raise BlindedReviewError(f"invalid frozen error taxonomy: {error}") from error


def prepare_error_review_package(
    *,
    restricted_root: Path,
    source_manifest_path: Path,
    taxonomy_path: Path,
) -> tuple[BlindedErrorReviewPackage, ErrorReviewRejoinMap, dict[str, bytes]]:
    source = _read_model(
        restricted_root,
        source_manifest_path,
        HeldOutFailureSourceManifest,
    )
    assert isinstance(source, HeldOutFailureSourceManifest)
    taxonomy = load_error_taxonomy(taxonomy_path)
    blind_rows: list[tuple[str, HeldOutFailureSource, bytes, str]] = []
    for item in source.items:
        blind = _blind_id("error", source.content_hash, item.source_failure_id)
        payload = _panel_bytes(restricted_root, source_manifest_path, item.review_panel)
        suffix = ".png" if item.review_panel.media_type == "image/png" else ".json"
        blind_rows.append((blind, item, payload, suffix))
    blind_rows.sort(key=lambda row: row[0])
    package = BlindedErrorReviewPackage(
        package_id=f"error-review-{source.content_hash[:20]}",
        taxonomy_id=taxonomy.taxonomy_id,
        taxonomy_hash=taxonomy.content_hash,
        source_manifest_hash=source.content_hash,
        items=tuple(
            BlindedErrorItem(
                blind_item_id=blind,
                panel_file=f"panels/{blind}{suffix}",
                panel_sha256=item.review_panel.file_sha256,
                media_type=item.review_panel.media_type,
            )
            for blind, item, _, suffix in blind_rows
        ),
    )
    rejoin = ErrorReviewRejoinMap(
        map_id=f"error-rejoin-{source.content_hash[:20]}",
        package_hash=package.content_hash,
        source_manifest_hash=source.content_hash,
        entries=tuple(
            ErrorRejoinEntry(
                blind_item_id=blind,
                source_failure_id=item.source_failure_id,
                world_id=item.world_id,
                context_id=item.context_id,
                condition=item.condition,
                seed_block=item.seed_block,
                result_artifact_hash=item.result_artifact_hash,
                projection_hash=item.projection_hash,
                output_status=item.output_status,
                failed_gate_ids=item.failed_gate_ids,
            )
            for blind, item, _, _ in blind_rows
        ),
    )
    files = {
        "reviewer/review_package.json": _model_bytes(package),
        "scorer_only/rejoin_map.json": _model_bytes(rejoin),
        **{
            f"reviewer/panels/{blind}{suffix}": payload
            for blind, _, payload, suffix in blind_rows
        },
    }
    return package, rejoin, files


def materialize_error_review_package(
    *,
    restricted_root: Path,
    source_manifest_path: Path,
    taxonomy_path: Path,
    output_root: Path,
) -> tuple[Path, Literal["created", "verified"], BlindedErrorReviewPackage]:
    package, _, files = prepare_error_review_package(
        restricted_root=restricted_root,
        source_manifest_path=source_manifest_path,
        taxonomy_path=taxonomy_path,
    )
    path, state = _materialize_bundle(
        restricted_root=restricted_root,
        output_root=output_root,
        identity_hash=package.content_hash,
        files=files,
    )
    return path, state, package


def _load_error_package(
    restricted_root: Path,
    package_root: Path,
) -> tuple[BlindedErrorReviewPackage, ErrorReviewRejoinMap]:
    package = _read_model(
        restricted_root,
        package_root / "reviewer/review_package.json",
        BlindedErrorReviewPackage,
    )
    rejoin = _read_model(
        restricted_root,
        package_root / "scorer_only/rejoin_map.json",
        ErrorReviewRejoinMap,
    )
    assert isinstance(package, BlindedErrorReviewPackage)
    assert isinstance(rejoin, ErrorReviewRejoinMap)
    if rejoin.package_hash != package.content_hash:
        raise BlindedReviewError("error-review rejoin map names another package")
    package_ids = [item.blind_item_id for item in package.items]
    map_ids = [item.blind_item_id for item in rejoin.entries]
    if sorted(package_ids) != sorted(map_ids):
        raise BlindedReviewError("error-review rejoin inventory differs from package")
    for item in package.items:
        panel = _contained_path(
            restricted_root,
            package_root / "reviewer" / Path(*_relative_path(item.panel_file).parts),
            must_exist=True,
            regular_file=True,
        )
        if _file_sha256_bytes(_read_limited(panel)) != item.panel_sha256:
            raise BlindedReviewError("blinded error-review panel changed")
    return package, rejoin


def _canonical_error_table(
    rejoin: ErrorReviewRejoinMap,
    adjudication: ErrorReviewAdjudication,
) -> bytes:
    columns = (
        "world_id",
        "context_id",
        "condition",
        "seed_block",
        "source_failure_id",
        "blind_item_id",
        "output_status",
        "projection_hash",
        "result_artifact_hash",
        "failed_gate_ids",
        "error_codes",
        "independent_unit",
    )
    final = {item.blind_item_id: item for item in adjudication.entries}
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for item in sorted(
        rejoin.entries,
        key=lambda row: (row.world_id, row.context_id, row.condition.value, row.seed_block),
    ):
        judgment = final[item.blind_item_id]
        writer.writerow(
            {
                "world_id": item.world_id,
                "context_id": item.context_id,
                "condition": item.condition.value,
                "seed_block": item.seed_block,
                "source_failure_id": item.source_failure_id,
                "blind_item_id": item.blind_item_id,
                "output_status": item.output_status,
                "projection_hash": item.projection_hash or "",
                "result_artifact_hash": item.result_artifact_hash,
                "failed_gate_ids": "|".join(item.failed_gate_ids),
                "error_codes": "|".join(code.value for code in judgment.final_error_codes),
                "independent_unit": "world",
            }
        )
    return stream.getvalue().encode("utf-8")


def prepare_error_review_finalization(
    *,
    restricted_root: Path,
    package_root: Path,
    completion_path: Path,
    adjudication_path: Path,
) -> tuple[ErrorReviewFinalization, dict[str, bytes]]:
    package, rejoin = _load_error_package(restricted_root, package_root)
    completion = _read_model(
        restricted_root,
        completion_path,
        ErrorReviewCompletion,
    )
    adjudication = _read_model(
        restricted_root,
        adjudication_path,
        ErrorReviewAdjudication,
    )
    assert isinstance(completion, ErrorReviewCompletion)
    assert isinstance(adjudication, ErrorReviewAdjudication)
    if completion.package_hash != package.content_hash:
        raise BlindedReviewError("error completion names another package")
    if (
        adjudication.package_hash != package.content_hash
        or adjudication.completion_hash != completion.content_hash
    ):
        raise BlindedReviewError("error adjudication lineage mismatch")
    expected = {item.blind_item_id for item in package.items}
    reviewed = [item.blind_item_id for item in completion.judgments]
    final = [item.blind_item_id for item in adjudication.entries]
    if len(reviewed) != len(set(reviewed)) or set(reviewed) != expected:
        raise BlindedReviewError("error completion must cover every blinded failure exactly once")
    if len(final) != len(set(final)) or set(final) != expected:
        raise BlindedReviewError("error adjudication must cover every blinded failure exactly once")
    reviewed_by_id = {item.blind_item_id: item for item in completion.judgments}
    for item in adjudication.entries:
        reviewed_codes = reviewed_by_id[item.blind_item_id].error_codes
        if item.disposition == "accept" and item.final_error_codes != reviewed_codes:
            raise BlindedReviewError("accepted adjudication changed reviewer codes")
    table = _canonical_error_table(rejoin, adjudication)
    finalization = ErrorReviewFinalization(
        finalization_id=f"error-final-{adjudication.content_hash[:20]}",
        package_hash=package.content_hash,
        rejoin_map_hash=rejoin.content_hash,
        completion_hash=completion.content_hash,
        adjudication_hash=adjudication.content_hash,
        canonical_table_sha256=_file_sha256_bytes(table),
        reviewed_failure_count=len(expected),
    )
    files = {
        "completion.json": _model_bytes(completion),
        "adjudication.json": _model_bytes(adjudication),
        "held_out_error_review.csv": table,
        "finalization.json": _model_bytes(finalization),
    }
    return finalization, files


def materialize_error_review_finalization(
    *,
    restricted_root: Path,
    package_root: Path,
    completion_path: Path,
    adjudication_path: Path,
    output_root: Path,
) -> tuple[Path, Literal["created", "verified"], ErrorReviewFinalization]:
    finalization, files = prepare_error_review_finalization(
        restricted_root=restricted_root,
        package_root=package_root,
        completion_path=completion_path,
        adjudication_path=adjudication_path,
    )
    path, state = _materialize_bundle(
        restricted_root=restricted_root,
        output_root=output_root,
        identity_hash=finalization.content_hash,
        files=files,
    )
    return path, state, finalization


def prepare_community_review_package(
    *,
    restricted_root: Path,
    source_manifest_path: Path,
    rubric_template_path: Path,
) -> tuple[
    BlindedCommunityReviewPackage,
    CommunityReviewRejoinMap,
    dict[str, bytes],
]:
    source = _read_model(
        restricted_root,
        source_manifest_path,
        CommunityReviewSourceManifest,
    )
    assert isinstance(source, CommunityReviewSourceManifest)
    try:
        template = CommunityReviewTemplate.load(_safe_config_file(rubric_template_path))
    except Exception as error:
        raise BlindedReviewError(f"invalid frozen community rubric: {error}") from error
    if template.reviews:
        raise BlindedReviewError("frozen community rubric template must not contain reviews")
    template_hash = canonical_sha256(template)
    blind_rows: list[tuple[str, CommunityPartitionSource, bytes, str]] = []
    for item in source.items:
        blind = _blind_id("community", source.content_hash, item.source_partition_id)
        payload = _panel_bytes(restricted_root, source_manifest_path, item.review_panel)
        suffix = ".png" if item.review_panel.media_type == "image/png" else ".json"
        blind_rows.append((blind, item, payload, suffix))
    blind_rows.sort(key=lambda row: row[0])
    package = BlindedCommunityReviewPackage(
        package_id=f"community-review-{source.content_hash[:20]}",
        template_id=template.template_id,
        rubric_revision=template.rubric_revision,
        rubric_template_hash=template_hash,
        source_manifest_hash=source.content_hash,
        items=tuple(
            BlindedCommunityItem(
                blinded_output_id=blind,
                panel_file=f"panels/{blind}{suffix}",
                panel_sha256=item.review_panel.file_sha256,
                media_type=item.review_panel.media_type,
            )
            for blind, item, _, suffix in blind_rows
        ),
    )
    rejoin = CommunityReviewRejoinMap(
        map_id=f"community-rejoin-{source.content_hash[:20]}",
        package_hash=package.content_hash,
        source_manifest_hash=source.content_hash,
        entries=tuple(
            CommunityRejoinEntry(
                blinded_output_id=blind,
                source_partition_id=item.source_partition_id,
                world_id=item.world_id,
                context_id=item.context_id,
                condition=item.condition,
                seed_block=item.seed_block,
                resolution=item.resolution,
                projection_hash=item.projection_hash,
                partition_hash=item.partition_hash,
                node_count=item.node_count,
                cluster_count=item.cluster_count,
            )
            for blind, item, _, _ in blind_rows
        ),
    )
    files = {
        "reviewer/review_package.json": _model_bytes(package),
        "scorer_only/rejoin_map.json": _model_bytes(rejoin),
        **{
            f"reviewer/panels/{blind}{suffix}": payload
            for blind, _, payload, suffix in blind_rows
        },
    }
    return package, rejoin, files


def materialize_community_review_package(
    *,
    restricted_root: Path,
    source_manifest_path: Path,
    rubric_template_path: Path,
    output_root: Path,
) -> tuple[Path, Literal["created", "verified"], BlindedCommunityReviewPackage]:
    package, _, files = prepare_community_review_package(
        restricted_root=restricted_root,
        source_manifest_path=source_manifest_path,
        rubric_template_path=rubric_template_path,
    )
    path, state = _materialize_bundle(
        restricted_root=restricted_root,
        output_root=output_root,
        identity_hash=package.content_hash,
        files=files,
    )
    return path, state, package


def _load_community_package(
    restricted_root: Path,
    package_root: Path,
) -> tuple[BlindedCommunityReviewPackage, CommunityReviewRejoinMap]:
    package = _read_model(
        restricted_root,
        package_root / "reviewer/review_package.json",
        BlindedCommunityReviewPackage,
    )
    rejoin = _read_model(
        restricted_root,
        package_root / "scorer_only/rejoin_map.json",
        CommunityReviewRejoinMap,
    )
    assert isinstance(package, BlindedCommunityReviewPackage)
    assert isinstance(rejoin, CommunityReviewRejoinMap)
    if rejoin.package_hash != package.content_hash:
        raise BlindedReviewError("community rejoin map names another package")
    package_ids = [item.blinded_output_id for item in package.items]
    map_ids = [item.blinded_output_id for item in rejoin.entries]
    if len(package_ids) != len(set(package_ids)) or sorted(package_ids) != sorted(map_ids):
        raise BlindedReviewError("community rejoin inventory differs from package")
    for item in package.items:
        panel = _contained_path(
            restricted_root,
            package_root / "reviewer" / Path(*_relative_path(item.panel_file).parts),
            must_exist=True,
            regular_file=True,
        )
        if _file_sha256_bytes(_read_limited(panel)) != item.panel_sha256:
            raise BlindedReviewError("blinded community panel changed")
    return package, rejoin


def _canonical_community_table(
    rejoin: CommunityReviewRejoinMap,
    completion: CommunityReviewCompletion,
) -> bytes:
    columns = (
        "world_id",
        "context_id",
        "condition",
        "seed_block",
        "resolution",
        "source_partition_id",
        "blinded_output_id",
        "projection_hash",
        "partition_hash",
        "node_count",
        "cluster_count",
        "semantic_coherence",
        "interpretability",
        "evidence_support",
        "rubric_mean",
        "independent_unit",
    )
    reviews = {item.blinded_output_id: item for item in completion.reviews}
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for item in sorted(
        rejoin.entries,
        key=lambda row: (
            row.world_id,
            row.context_id,
            row.condition.value,
            row.seed_block,
            row.resolution,
        ),
    ):
        review = reviews[item.blinded_output_id]
        mean = (
            review.semantic_coherence + review.interpretability + review.evidence_support
        ) / 3.0
        writer.writerow(
            {
                "world_id": item.world_id,
                "context_id": item.context_id,
                "condition": item.condition.value,
                "seed_block": item.seed_block,
                "resolution": format(item.resolution, ".17g"),
                "source_partition_id": item.source_partition_id,
                "blinded_output_id": item.blinded_output_id,
                "projection_hash": item.projection_hash,
                "partition_hash": item.partition_hash,
                "node_count": item.node_count,
                "cluster_count": item.cluster_count,
                "semantic_coherence": review.semantic_coherence,
                "interpretability": review.interpretability,
                "evidence_support": review.evidence_support,
                "rubric_mean": format(mean, ".17g"),
                "independent_unit": "world",
            }
        )
    return stream.getvalue().encode("utf-8")


def prepare_community_review_finalization(
    *,
    restricted_root: Path,
    package_root: Path,
    completion_path: Path,
) -> tuple[CommunityReviewFinalization, dict[str, bytes]]:
    package, rejoin = _load_community_package(restricted_root, package_root)
    completion = _read_model(
        restricted_root,
        completion_path,
        CommunityReviewCompletion,
    )
    assert isinstance(completion, CommunityReviewCompletion)
    if completion.package_hash != package.content_hash:
        raise BlindedReviewError("community completion names another package")
    expected = {item.blinded_output_id for item in package.items}
    received = [item.blinded_output_id for item in completion.reviews]
    if len(received) != len(set(received)) or set(received) != expected:
        raise BlindedReviewError("community completion must cover each blind item exactly once")
    table = _canonical_community_table(rejoin, completion)
    finalization = CommunityReviewFinalization(
        finalization_id=f"community-final-{completion.content_hash[:20]}",
        package_hash=package.content_hash,
        rejoin_map_hash=rejoin.content_hash,
        completion_hash=completion.content_hash,
        canonical_table_sha256=_file_sha256_bytes(table),
        reviewed_partition_count=len(expected),
    )
    files = {
        "completion.json": _model_bytes(completion),
        "community_blind_review.csv": table,
        "finalization.json": _model_bytes(finalization),
    }
    return finalization, files


def materialize_community_review_finalization(
    *,
    restricted_root: Path,
    package_root: Path,
    completion_path: Path,
    output_root: Path,
) -> tuple[Path, Literal["created", "verified"], CommunityReviewFinalization]:
    finalization, files = prepare_community_review_finalization(
        restricted_root=restricted_root,
        package_root=package_root,
        completion_path=completion_path,
    )
    path, state = _materialize_bundle(
        restricted_root=restricted_root,
        output_root=output_root,
        identity_hash=finalization.content_hash,
        files=files,
    )
    return path, state, finalization


__all__ = [
    "FROZEN_COMMUNITY_REVIEW_SELECTION_RULE",
    "FROZEN_COMMUNITY_REVIEW_SELECTION_RULE_HASH",
    "FROZEN_ERROR_CODES",
    "FROZEN_REGISTERED_FAILURE_SIGNAL_GATES",
    "FROZEN_REGISTERED_FAILURE_SIGNAL_POLICY_HASH",
    "BlindedCommunityReviewPackage",
    "BlindedErrorReviewPackage",
    "BlindedReviewError",
    "CommunityPartitionSource",
    "CommunityReviewCompletion",
    "CommunityReviewFinalization",
    "CommunityReviewSourceManifest",
    "ErrorAdjudicationEntry",
    "ErrorReviewAdjudication",
    "ErrorReviewCompletion",
    "ErrorReviewFinalization",
    "ErrorReviewJudgment",
    "FailureSignalDirection",
    "HeldOutErrorCode",
    "HeldOutErrorTaxonomy",
    "HeldOutFailureSource",
    "HeldOutFailureSourceManifest",
    "NeutralCommunityContextDisplay",
    "NeutralCommunityEvidenceRecord",
    "NeutralCommunityPanelAssertionSupport",
    "NeutralCommunityPanelEdge",
    "NeutralCommunityPanelNode",
    "NeutralCommunityReviewPanel",
    "NeutralCommunityRoleBinding",
    "NeutralCommunityTemporalConstraint",
    "NeutralCommunityTemporalDisplay",
    "NeutralFailurePanelDecision",
    "NeutralFailurePanelEdge",
    "NeutralFailurePanelMetric",
    "NeutralFailureReviewPanel",
    "RegisteredFailureSignalGate",
    "ReviewPanelSource",
    "load_error_taxonomy",
    "materialize_community_review_finalization",
    "materialize_community_review_package",
    "materialize_community_review_source",
    "materialize_error_review_finalization",
    "materialize_error_review_package",
    "materialize_held_out_failure_source",
    "prepare_community_review_finalization",
    "prepare_community_review_package",
    "prepare_community_review_source",
    "prepare_error_review_finalization",
    "prepare_error_review_package",
    "prepare_held_out_failure_source",
]
