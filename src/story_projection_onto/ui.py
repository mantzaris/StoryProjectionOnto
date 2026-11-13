"""Local visualization and typed feedback support for the bounded Phase 5 study.

The renderer in this module is deliberately downstream of a validated
``OntologyProjection``.  It may choose visibility and geometry, but it never creates a
semantic object or changes the projection hash.  Feedback anchors remain evidence- and
mention-based; projection-local identifiers are resolved separately for each receiving
condition.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal, Protocol, Self

from pydantic import AwareDatetime, Field, model_validator

from story_projection_onto.contracts import (
    ACTIVE_QUERY_CONSTRUCTION_CONDITIONS,
    ConditionName,
    ConstructionOperator,
    DiscoursePosition,
    EvidenceBadge,
    EvidencePacket,
    ExplicitValueState,
    FeedbackAction,
    FeedbackAnchor,
    FeedbackResolution,
    FeedbackResolutionStatus,
    HolderRelativeTime,
    Identifier,
    ImmutableRecord,
    ModelVisibleRevision,
    NarrativeCommitment,
    NodePosition,
    OntologyProjection,
    QueryContext,
    ReleaseClass,
    RoleBinding,
    Sha256Digest,
    SpoilerHorizon,
    StoryTime,
    TemporalKind,
    UserRevision,
    Viewport,
    VisualizationAssertion,
    VisualizationNode,
    VisualizationState,
    VisualizationTemporalFilter,
    canonical_json,
    canonical_sha256,
)
from story_projection_onto.display_selection import (
    DISPLAY_SELECTION_RULE_REVISION,
    DisplaySelectionError,
    compile_registered_display_selection,
)
from story_projection_onto.display_selection import (
    assertion_endpoint_ids as _projection_assertion_endpoint_ids,
)
from story_projection_onto.feedback_provenance import ResearcherTraceSubmissionReceipt
from story_projection_onto.temporal import (
    discourse_is_within_horizon,
    revelation_is_within_horizon,
)


class VisualizationCompilationError(ValueError):
    """A projection cannot be rendered without violating scientific contracts."""


class VisualizationObjectKind(StrEnum):
    ENTITY = "entity"
    EVENT = "event"


class SemanticDisplayMode(StrEnum):
    """Whether the renderer retains all output or exposes only supported assertions."""

    ALL_STRUCTURAL = "all_structural"
    SUPPORTED_ONLY = "supported_only"


class VisualizationContentScope(StrEnum):
    """Pre-overlay content scope used by a visualization."""

    FULL_STRUCTURAL = "full_structural"
    REGISTERED_DISPLAY = "registered_display"


NODE_DESCRIPTION_WITHHELD = (
    "Projection-authored description withheld because post-hoc description support "
    "was not verified."
)
WHY_MATTERS_WITHHELD = (
    "Projection-authored why-it-matters text withheld because post-hoc description "
    "support was not verified."
)


class SemanticSupportStatus(StrEnum):
    """Post-hoc support verdict; runtime construction never creates these values."""

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    INSUFFICIENT = "insufficient"


class SemanticAssessmentSourceKind(StrEnum):
    SCORER = "scorer"
    REVIEWER = "reviewer"


class VisualizationAssertionSemanticAssessment(ImmutableRecord):
    """One scorer/reviewer verdict for an emitted projection assertion.

    Assertion support and description support are deliberately separate.  A strict
    assertion match does not establish that generated ``why_matters`` or dependent
    node-description prose is supported.
    """

    projection_assertion_id: Identifier
    projection_assertion_hash: Sha256Digest
    assertion_support_status: SemanticSupportStatus
    description_support_status: SemanticSupportStatus
    supported_description_evidence_ids: tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def supported_description_has_evidence(self) -> Self:
        evidence_ids = self.supported_description_evidence_ids
        if evidence_ids != tuple(sorted(evidence_ids)) or len(evidence_ids) != len(
            set(evidence_ids)
        ):
            raise ValueError("supported-description evidence IDs must be unique and sorted")
        if self.description_support_status is SemanticSupportStatus.SUPPORTED:
            if self.assertion_support_status is not SemanticSupportStatus.SUPPORTED:
                raise ValueError("description support requires a supported assertion")
            if not evidence_ids:
                raise ValueError("supported descriptions require verified evidence")
        elif evidence_ids:
            raise ValueError("unverified descriptions cannot carry supported evidence")
        return self


class VisualizationSemanticOverlay(ImmutableRecord):
    """Immutable post-hoc semantic assessment for one exact projection input cell.

    This DTO is scorer/reviewer output only.  It is never a construction request input,
    never mutates the projection, and intentionally contains no gold target or expected
    answer.  ``source_artifact_hash`` binds the upstream scorer/reviewer artifact from
    which these presentation-safe verdicts were materialized.
    """

    overlay_id: Identifier
    projection_hash: Sha256Digest
    snapshot_hash: Sha256Digest
    packet_hash: Sha256Digest
    context_hash: Sha256Digest
    source_kind: SemanticAssessmentSourceKind
    source_artifact_hash: Sha256Digest
    assessment_revision: str = Field(min_length=1)
    assertion_assessments: tuple[VisualizationAssertionSemanticAssessment, ...]
    assessed_at: AwareDatetime
    release_class: ReleaseClass

    @model_validator(mode="after")
    def assessment_inventory_is_canonical(self) -> Self:
        assertion_ids = tuple(
            item.projection_assertion_id for item in self.assertion_assessments
        )
        if assertion_ids != tuple(sorted(assertion_ids)):
            raise ValueError("semantic overlay assessments must be sorted by assertion ID")
        if len(assertion_ids) != len(set(assertion_ids)):
            raise ValueError("semantic overlay assessments require unique assertion IDs")
        return self


class VisualizationDisplaySelection(ImmutableRecord):
    """Frozen, condition-neutral pre-overlay visibility selection.

    The registered display scope is the only source for direct visual-clutter
    geometry.  The full scope remains available for raw intention-to-treat inspection.
    """

    projection_hash: Sha256Digest
    content_scope: VisualizationContentScope
    rule_revision: Literal[
        "context-relevance-qualified-dependency-closure-v2"
    ] = (
        DISPLAY_SELECTION_RULE_REVISION
    )
    budget_accounting_hash: Sha256Digest
    output_budgets_hash: Sha256Digest
    projection_node_ids: tuple[Identifier, ...]
    projection_assertion_ids: tuple[Identifier, ...]

    @model_validator(mode="after")
    def selected_ids_are_canonical(self) -> Self:
        for label, values in (
            ("node", self.projection_node_ids),
            ("assertion", self.projection_assertion_ids),
        ):
            if values != tuple(sorted(values)) or len(values) != len(set(values)):
                raise ValueError(f"selected {label} IDs must be unique and sorted")
        return self


DEFAULT_VISUALIZATION_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "study" / "visualization.json"
)


class VisualizationConfiguration(ImmutableRecord):
    """Frozen renderer configuration shared by every study condition.

    The layout is deliberately deterministic and downstream of ontology construction.
    Its seed is bound to the synthetic benchmark's root-derived layout seed so geometry
    cannot be selected separately for a condition or after inspecting results.
    """

    configuration_id: Identifier
    layout_name: Literal["preset-anchor-hash-v2"] = "preset-anchor-hash-v2"
    coordinate_rule: Literal["sha256-layout-seed-anchor-grid-v2"] = (
        "sha256-layout-seed-anchor-grid-v2"
    )
    layout_seed: int = Field(ge=0)
    seed_manifest_hash: Sha256Digest
    layout_seed_entry_hash: Sha256Digest
    viewport: Viewport
    style_name: Literal["phase5-study-v2"] = "phase5-study-v2"
    progressive_disclosure: Literal[True] = True
    font_family: str = Field(min_length=1)
    font_base_px: int = Field(gt=0)
    max_retained_screenshots: Literal[12] = 12


def load_visualization_configuration(
    path: Path = DEFAULT_VISUALIZATION_CONFIG_PATH,
) -> VisualizationConfiguration:
    """Load and hash the one tracked visualization configuration."""

    return VisualizationConfiguration.model_validate_json(path.read_text(encoding="utf-8"))


class VisualizationNodeDetail(ImmutableRecord):
    """Progressively disclosed metadata for one semantic projection object."""

    visualization_node_id: Identifier
    stable_anchor_id: Identifier
    projection_object_id: Identifier
    object_kind: VisualizationObjectKind
    mention_candidate_ids: tuple[Identifier, ...] = ()
    evidence_ids: tuple[Identifier, ...]
    aliases: tuple[str, ...] = ()
    contextual_type_label: str = Field(min_length=1)
    contextual_type_definition: str = Field(min_length=1)
    contextual_role: str = Field(min_length=1)
    description_support_status: Literal[
        "pending", "supported", "unsupported", "insufficient"
    ] = "pending"

    @model_validator(mode="after")
    def require_condition_independent_anchor_material(self) -> Self:
        if not self.evidence_ids and not self.mention_candidate_ids:
            raise ValueError("visual nodes require evidence or mention anchor material")
        return self


class VisualizationAssertionDetail(ImmutableRecord):
    """Metadata hidden from the overview until an assertion is selected."""

    visualization_assertion_id: Identifier
    stable_anchor_id: Identifier
    projection_assertion_id: Identifier
    predicate_label: str = Field(min_length=1)
    predicate_definition: str = Field(min_length=1)
    contextual_relevance: float = Field(ge=0.0, le=1.0)
    narrative_commitment: NarrativeCommitment
    role_labels: tuple[Identifier, ...] = ()
    evidence_ids: tuple[Identifier, ...]
    why_matters_evidence_ids: tuple[Identifier, ...]
    assertion_support_status: Literal[
        "pending", "supported", "unsupported", "insufficient"
    ] = "pending"
    description_support_status: Literal[
        "pending", "supported", "unsupported", "insufficient"
    ] = "pending"
    proposition_content_id: Identifier | None = None
    holder_relative_time: HolderRelativeTime | None = None

    @model_validator(mode="after")
    def epistemic_detail_is_complete(self) -> Self:
        if (self.proposition_content_id is None) != (self.holder_relative_time is None):
            raise ValueError(
                "holder-relative time and proposition content must be present or absent together"
            )
        if not self.evidence_ids:
            raise ValueError("visual assertion details require assertion evidence")
        if not set(self.why_matters_evidence_ids).issubset(self.evidence_ids):
            raise ValueError("visual why support must be assertion evidence")
        if self.description_support_status == "supported" and not self.why_matters_evidence_ids:
            raise ValueError("supported visual descriptions require verified evidence")
        if self.description_support_status in {"unsupported", "insufficient"} and (
            self.why_matters_evidence_ids
        ):
            raise ValueError("unverified visual descriptions cannot claim supported evidence")
        return self


class EvidenceMetadata(ImmutableRecord):
    """Authorized evidence detail without requiring protected source text."""

    evidence_id: Identifier
    passage_id: Identifier
    discourse_position: DiscoursePosition
    locator: str = Field(min_length=1)
    extraction_method: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    release_class: ReleaseClass
    public_text: str | None = None

    @model_validator(mode="after")
    def protected_text_is_never_embedded(self) -> Self:
        if self.release_class is ReleaseClass.RESTRICTED and self.public_text is not None:
            raise ValueError("restricted evidence metadata cannot embed source text")
        return self


class VisualizationBundle(ImmutableRecord):
    """One immutable projection view plus condition-neutral anchor metadata."""

    bundle_id: Identifier
    projection_id: Identifier
    projection_hash: Sha256Digest
    condition: ConditionName
    context: QueryContext
    context_hash: Sha256Digest
    snapshot_hash: Sha256Digest
    packet_hash: Sha256Digest
    semantic_assessment_status: Literal[
        "pending_scorer_or_reviewer", "verified_scorer_or_reviewer"
    ] = "pending_scorer_or_reviewer"
    semantic_display_mode: SemanticDisplayMode = SemanticDisplayMode.ALL_STRUCTURAL
    semantic_overlay: VisualizationSemanticOverlay | None = None
    display_selection: VisualizationDisplaySelection
    state: VisualizationState
    node_details: tuple[VisualizationNodeDetail, ...]
    assertion_details: tuple[VisualizationAssertionDetail, ...]
    evidence_metadata: tuple[EvidenceMetadata, ...]
    release_class: ReleaseClass

    @model_validator(mode="after")
    def bundle_references_are_complete(self) -> Self:
        if self.state.projection_hash != self.projection_hash:
            raise ValueError("visualization bundle and state projection hashes differ")
        if self.display_selection.projection_hash != self.projection_hash:
            raise ValueError("display selection is bound to another projection")
        if self.context.content_hash != self.context_hash:
            raise ValueError("visualization bundle does not hash its query context")
        displayed_assertion_ids = {
            item.projection_assertion_id for item in self.state.assertions
        }
        if self.semantic_display_mode is SemanticDisplayMode.ALL_STRUCTURAL and (
            displayed_assertion_ids
            != set(self.display_selection.projection_assertion_ids)
        ):
            raise ValueError("displayed assertions do not match the pre-overlay selection")
        if self.semantic_overlay is None:
            if self.semantic_assessment_status != "pending_scorer_or_reviewer":
                raise ValueError("verified semantic status requires a bound overlay")
            if self.semantic_display_mode is not SemanticDisplayMode.ALL_STRUCTURAL:
                raise ValueError("supported-only display requires a bound semantic overlay")
        else:
            overlay = self.semantic_overlay
            if self.semantic_assessment_status != "verified_scorer_or_reviewer":
                raise ValueError("a bound semantic overlay must be disclosed as verified")
            if (
                overlay.projection_hash != self.projection_hash
                or overlay.snapshot_hash != self.snapshot_hash
                or overlay.packet_hash != self.packet_hash
                or overlay.context_hash != self.context_hash
            ):
                raise ValueError("semantic overlay is bound to another visualization input")
            if self.semantic_display_mode is SemanticDisplayMode.SUPPORTED_ONLY:
                expected_assertion_ids = {
                    item.projection_assertion_id
                    for item in overlay.assertion_assessments
                    if item.assertion_support_status is SemanticSupportStatus.SUPPORTED
                }.intersection(self.display_selection.projection_assertion_ids)
                if displayed_assertion_ids != expected_assertion_ids:
                    raise ValueError(
                        "displayed assertions do not match the semantic display mode"
                    )
        node_ids = {item.visualization_node_id for item in self.state.nodes}
        detail_node_ids = {item.visualization_node_id for item in self.node_details}
        if node_ids != detail_node_ids:
            raise ValueError("visual node details must cover the state exactly")
        assertion_ids = {item.visualization_assertion_id for item in self.state.assertions}
        detail_assertion_ids = {item.visualization_assertion_id for item in self.assertion_details}
        if assertion_ids != detail_assertion_ids:
            raise ValueError("visual assertion details must cover the state exactly")
        evidence_ids = [item.evidence_id for item in self.evidence_metadata]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("visual evidence metadata IDs must be unique")
        available = set(evidence_ids)
        badges = [item.evidence_badge for item in self.state.nodes]
        badges.extend(item.evidence_badge for item in self.state.assertions)
        if any(not set(item.evidence_ids).issubset(available) for item in badges):
            raise ValueError("every visual evidence badge must resolve in bundle metadata")
        nodes = {item.visualization_node_id: item for item in self.state.nodes}
        displayed_node_ids = {item.projection_object_id for item in nodes.values()}
        if self.semantic_display_mode is SemanticDisplayMode.ALL_STRUCTURAL and (
            displayed_node_ids != set(self.display_selection.projection_node_ids)
        ):
            raise ValueError("displayed nodes do not match the pre-overlay display selection")
        if self.semantic_display_mode is SemanticDisplayMode.SUPPORTED_ONLY:
            required_node_ids: set[str] = set()
            for assertion in self.state.assertions:
                for visual_id in _visual_assertion_endpoint_ids(assertion):
                    required_node_ids.add(nodes[visual_id].projection_object_id)
            if displayed_node_ids != required_node_ids:
                raise ValueError(
                    "supported-only display must contain exactly supported assertion endpoints"
                )
        assessment_by_id = (
            None
            if self.semantic_overlay is None
            else {
                item.projection_assertion_id: item
                for item in self.semantic_overlay.assertion_assessments
            }
        )
        for detail in self.node_details:
            node = nodes[detail.visualization_node_id]
            if detail.stable_anchor_id != detail.visualization_node_id:
                raise ValueError("visual node stable anchors must equal their renderer IDs")
            if detail.projection_object_id != node.projection_object_id:
                raise ValueError("visual node detail refers to another projection object")
            if detail.contextual_role != node.contextual_role:
                raise ValueError("visual node detail and overview roles differ")
            if detail.evidence_ids != node.evidence_badge.evidence_ids:
                raise ValueError("visual node detail and overview evidence differ")
            expected_description_status = _description_support_status(
                node.description_assertion_ids, assessment_by_id
            )
            if detail.description_support_status != expected_description_status:
                raise ValueError("visual node description support status differs from overlay")
            if (
                self.semantic_display_mode is SemanticDisplayMode.SUPPORTED_ONLY
                and expected_description_status != "supported"
                and node.description != NODE_DESCRIPTION_WITHHELD
            ):
                raise ValueError("unsupported node description was not withheld")
        assertions = {item.visualization_assertion_id: item for item in self.state.assertions}
        for detail in self.assertion_details:
            assertion = assertions[detail.visualization_assertion_id]
            if detail.stable_anchor_id != detail.visualization_assertion_id:
                raise ValueError("visual assertion stable anchors must equal renderer IDs")
            if detail.projection_assertion_id != assertion.projection_assertion_id:
                raise ValueError("visual assertion detail refers to another projection assertion")
            if detail.evidence_ids != assertion.evidence_badge.evidence_ids:
                raise ValueError("visual assertion detail and overview evidence differ")
            if detail.predicate_label != assertion.contextual_label:
                raise ValueError("visual assertion detail and overview labels differ")
            if detail.why_matters_evidence_ids and not set(
                detail.why_matters_evidence_ids
            ).issubset(assertion.evidence_badge.evidence_ids):
                raise ValueError("visual why support is absent from the evidence badge")
            if assessment_by_id is None:
                if (
                    detail.assertion_support_status != "pending"
                    or detail.description_support_status != "pending"
                    or not detail.why_matters_evidence_ids
                ):
                    raise ValueError("unassessed visual assertion must remain pending")
            else:
                assessment = assessment_by_id[detail.projection_assertion_id]
                if (
                    detail.assertion_support_status
                    != assessment.assertion_support_status.value
                    or detail.description_support_status
                    != assessment.description_support_status.value
                    or detail.why_matters_evidence_ids
                    != assessment.supported_description_evidence_ids
                ):
                    raise ValueError("visual assertion support differs from its overlay")
                if (
                    self.semantic_display_mode is SemanticDisplayMode.SUPPORTED_ONLY
                    and assessment.description_support_status
                    is not SemanticSupportStatus.SUPPORTED
                    and assertion.why_matters != WHY_MATTERS_WITHHELD
                ):
                    raise ValueError("unsupported why_matters text was not withheld")
            has_epistemic_overview = assertion.epistemic_holder_id is not None
            if has_epistemic_overview != (detail.holder_relative_time is not None):
                raise ValueError("visual overview and detail epistemic status differ")
        if self.release_class is ReleaseClass.PUBLIC and any(
            item.release_class is ReleaseClass.RESTRICTED for item in self.evidence_metadata
        ):
            raise ValueError("a public visualization bundle cannot expose restricted metadata")
        if (
            self.release_class is ReleaseClass.PUBLIC
            and self.semantic_overlay is not None
            and self.semantic_overlay.release_class is ReleaseClass.RESTRICTED
        ):
            raise ValueError("a public visualization bundle cannot expose a restricted overlay")
        return self


def _identifier_digest(namespace: str, payload: object, *, length: int = 24) -> str:
    return f"{namespace}-{canonical_sha256(payload)[:length]}"


def _stable_node_bases(projection: OntologyProjection) -> dict[str, str]:
    """Derive condition-independent bases from evidence/mention anchors.

    Event evidence can occasionally anchor more than one event.  Such collisions receive a
    deterministic ordinal below; the collision is visible in the node details and never
    silently merges semantic objects.
    """

    bases: dict[str, str] = {}
    for entity in projection.instance_graph.entities:
        bases[entity.entity_id] = _identifier_digest(
            "entity-anchor",
            {"mention_candidate_ids": sorted(entity.supported_mention_candidate_ids)},
        )
    for event in projection.instance_graph.events:
        bases[event.event_id] = _identifier_digest(
            "event-anchor",
            {"evidence_ids": sorted(event.evidence_ids)},
        )
    return bases


def _deduplicate_anchor_ids(
    projection: OntologyProjection,
    bases: Mapping[str, str],
) -> dict[str, str]:
    objects = {item.entity_id: item for item in projection.instance_graph.entities} | {
        item.event_id: item for item in projection.instance_graph.events
    }
    grouped: dict[str, list[str]] = {}
    for object_id, base in bases.items():
        grouped.setdefault(base, []).append(object_id)
    result: dict[str, str] = {}
    for base, object_ids in grouped.items():
        ordered = sorted(object_ids, key=lambda item: (objects[item].content_hash, item))
        for ordinal, object_id in enumerate(ordered, 1):
            result[object_id] = base if len(ordered) == 1 else f"{base}-{ordinal}"
    return result


def _position_for_anchor(anchor_id: str, layout_seed: int) -> tuple[float, float]:
    """Frozen anchor-hash geometry keeps matching nodes fixed across conditions."""

    digest = hashlib.sha256(f"{layout_seed}:{anchor_id}".encode()).digest()
    x = float(int.from_bytes(digest[:4], "big") % 961 - 480)
    y = float(int.from_bytes(digest[4:8], "big") % 641 - 320)
    return x, y


def _assertion_uncertainty(assertion) -> ExplicitValueState:
    kinds = {
        assertion.temporal_scope.story_time.kind,
        assertion.temporal_scope.validity_time.kind,
    }
    if TemporalKind.INVALID in kinds:
        return ExplicitValueState.INVALID
    if TemporalKind.HORIZON_WITHHELD in kinds:
        return ExplicitValueState.HORIZON_WITHHELD
    if (
        TemporalKind.UNKNOWN in kinds
        or assertion.narrative_commitment is NarrativeCommitment.UNKNOWN
    ):
        return ExplicitValueState.UNKNOWN
    return ExplicitValueState.KNOWN


def _structurally_accepted_assertion_ids(projection: OntologyProjection) -> frozenset[str]:
    """Return every emitted assertion from a structurally accepted projection.

    ``OntologyProjection`` enforces its runtime structural-validation envelope.  A
    scorer or reviewer may later assess semantic evidence support, but that result must
    not alter the graph supplied to direct clutter or intention-to-treat measurements.
    """

    return frozenset(
        assertion.assertion_id for assertion in projection.instance_graph.assertions
    )


def _evidence_metadata(
    packet: EvidencePacket,
    *,
    include_public_evidence_text: bool,
) -> tuple[EvidenceMetadata, ...]:
    return tuple(
        EvidenceMetadata(
            evidence_id=record.evidence_id,
            passage_id=record.passage_id,
            discourse_position=record.discourse_position,
            locator=record.provenance.locator,
            extraction_method=record.provenance.extraction_method,
            confidence=record.confidence,
            release_class=record.release_class,
            public_text=(
                record.text
                if include_public_evidence_text and record.release_class is ReleaseClass.PUBLIC
                else None
            ),
        )
        for record in packet.evidence
    )


def _compile_display_selection(
    projection: OntologyProjection,
    content_scope: VisualizationContentScope,
) -> VisualizationDisplaySelection:
    """Apply the one frozen, condition-neutral display-budget rule."""

    node_records = (*projection.instance_graph.entities, *projection.instance_graph.events)
    assertion_records = projection.instance_graph.assertions
    node_ids = tuple(
        getattr(item, "entity_id", getattr(item, "event_id", None)) for item in node_records
    )
    assertion_ids = tuple(item.assertion_id for item in assertion_records)
    accounting = projection.budget_accounting
    if accounting.nodes_used != len(node_records) or accounting.assertions_used != len(
        assertion_records
    ):
        raise VisualizationCompilationError(
            "projection object counts differ from authoritative budget accounting"
        )

    if content_scope is VisualizationContentScope.FULL_STRUCTURAL:
        selected_node_ids = tuple(sorted(node_ids))
        selected_assertion_ids = tuple(sorted(assertion_ids))
    else:
        try:
            selection = compile_registered_display_selection(
                projection.instance_graph,
                accounting,
                projection.budgets,
            )
        except DisplaySelectionError as error:
            raise VisualizationCompilationError(str(error)) from error
        selected_node_ids = selection.node_ids
        selected_assertion_ids = selection.assertion_ids

    return VisualizationDisplaySelection(
        projection_hash=projection.content_hash,
        content_scope=content_scope,
        budget_accounting_hash=accounting.content_hash,
        output_budgets_hash=projection.budgets.content_hash,
        projection_node_ids=selected_node_ids,
        projection_assertion_ids=selected_assertion_ids,
    )


def _verify_semantic_overlay(
    projection: OntologyProjection,
    context: QueryContext,
    packet: EvidencePacket,
    overlay: VisualizationSemanticOverlay,
) -> dict[str, VisualizationAssertionSemanticAssessment]:
    expected_bindings = (
        (overlay.projection_hash, projection.content_hash, "projection"),
        (overlay.snapshot_hash, projection.snapshot_hash, "snapshot"),
        (overlay.snapshot_hash, packet.snapshot_hash, "packet snapshot"),
        (overlay.packet_hash, packet.content_hash, "packet"),
        (overlay.context_hash, context.content_hash, "context"),
    )
    for observed, expected, label in expected_bindings:
        if observed != expected:
            raise VisualizationCompilationError(
                f"semantic overlay is bound to another {label}"
            )
    assertions = {
        item.assertion_id: item for item in projection.instance_graph.assertions
    }
    assessments = {
        item.projection_assertion_id: item for item in overlay.assertion_assessments
    }
    if set(assessments) != set(assertions):
        raise VisualizationCompilationError(
            "semantic overlay must assess every and only emitted assertion"
        )
    packet_evidence_ids = {item.evidence_id for item in packet.evidence}
    for assertion_id, assertion in assertions.items():
        assessment = assessments[assertion_id]
        if assessment.projection_assertion_hash != assertion.content_hash:
            raise VisualizationCompilationError(
                f"semantic overlay assertion hash differs for {assertion_id!r}"
            )
        supported_description_evidence = set(
            assessment.supported_description_evidence_ids
        )
        if not supported_description_evidence.issubset(packet_evidence_ids):
            raise VisualizationCompilationError(
                "semantic overlay cites description evidence outside the packet"
            )
        if not supported_description_evidence.issubset(assertion.evidence_ids):
            raise VisualizationCompilationError(
                "semantic overlay cites evidence outside its projection assertion"
            )
        if not supported_description_evidence.issubset(
            assertion.why_matters_evidence_ids
        ):
            raise VisualizationCompilationError(
                "semantic overlay cites evidence absent from why_matters lineage"
            )
    return assessments


def _description_support_status(
    assertion_ids: Sequence[str],
    assessments: Mapping[str, VisualizationAssertionSemanticAssessment] | None,
) -> Literal["pending", "supported", "unsupported", "insufficient"]:
    if assessments is None:
        return "pending"
    rows = tuple(assessments[item] for item in assertion_ids)
    if all(
        item.assertion_support_status is SemanticSupportStatus.SUPPORTED
        and item.description_support_status is SemanticSupportStatus.SUPPORTED
        for item in rows
    ):
        return "supported"
    if any(
        item.assertion_support_status is SemanticSupportStatus.UNSUPPORTED
        or item.description_support_status is SemanticSupportStatus.UNSUPPORTED
        for item in rows
    ):
        return "unsupported"
    return "insufficient"


def build_visualization_bundle(
    projection: OntologyProjection,
    context: QueryContext,
    packet: EvidencePacket,
    *,
    include_public_evidence_text: bool = False,
    visualization_config: VisualizationConfiguration | None = None,
    content_scope: VisualizationContentScope = VisualizationContentScope.FULL_STRUCTURAL,
    semantic_overlay: VisualizationSemanticOverlay | None = None,
    semantic_display_mode: SemanticDisplayMode = SemanticDisplayMode.ALL_STRUCTURAL,
) -> VisualizationBundle:
    """Compile a structurally accepted projection into a renderer-only DTO.

    The default deliberately retains the full raw intention-to-treat output and marks
    semantic assessment pending.  Registered clutter capture must explicitly request
    ``REGISTERED_DISPLAY``; that scope applies the frozen common display-budget rule.
    Post-hoc semantic filtering is available only with an exact scorer/reviewer overlay
    and never changes the source projection.
    """

    config = visualization_config or load_visualization_configuration()
    try:
        content_scope = VisualizationContentScope(content_scope)
        semantic_display_mode = SemanticDisplayMode(semantic_display_mode)
    except ValueError as error:
        raise VisualizationCompilationError(str(error)) from error
    if projection.context_hash != context.content_hash:
        raise VisualizationCompilationError("projection and visualization context differ")
    if projection.packet_hash != packet.content_hash:
        raise VisualizationCompilationError("projection and visualization packet differ")
    if projection.snapshot_hash != packet.snapshot_hash:
        raise VisualizationCompilationError("projection and visualization snapshots differ")
    accepted = _structurally_accepted_assertion_ids(projection)
    display_selection = _compile_display_selection(projection, content_scope)
    assessments = (
        None
        if semantic_overlay is None
        else _verify_semantic_overlay(projection, context, packet, semantic_overlay)
    )
    if semantic_display_mode is SemanticDisplayMode.SUPPORTED_ONLY and assessments is None:
        raise VisualizationCompilationError(
            "supported-only display requires a verified scorer/reviewer overlay"
        )
    selected_assertion_ids = set(display_selection.projection_assertion_ids)
    if semantic_display_mode is SemanticDisplayMode.SUPPORTED_ONLY:
        assert assessments is not None
        selected_assertion_ids.intersection_update(
            assertion_id
            for assertion_id, assessment in assessments.items()
            if assessment.assertion_support_status is SemanticSupportStatus.SUPPORTED
        )
    selected_node_ids = set(display_selection.projection_node_ids)
    if semantic_display_mode is SemanticDisplayMode.SUPPORTED_ONLY:
        selected_node_ids = set()
        for assertion in projection.instance_graph.assertions:
            if assertion.assertion_id in selected_assertion_ids:
                selected_node_ids.update(_projection_assertion_endpoint_ids(assertion))
    type_definitions = {item.type_id: item for item in projection.local_schema.contextual_types}
    predicate_definitions = {item.predicate_id: item for item in projection.local_schema.predicates}
    bases = _stable_node_bases(projection)
    visual_ids = _deduplicate_anchor_ids(projection, bases)

    nodes: list[VisualizationNode] = []
    details: list[VisualizationNodeDetail] = []
    positions: list[NodePosition] = []
    object_to_visual: dict[str, str] = {}

    entity_rows = [
        (
            item.entity_id,
            VisualizationObjectKind.ENTITY,
            item.label,
            item.contextual_type_id,
            item.contextual_role,
            item.abstraction,
            item.temporal_state,
            item.uncertainty,
            item.confidence,
            item.evidence_ids,
            item.description,
            item.description_assertion_ids,
            item.supported_mention_candidate_ids,
            item.aliases,
        )
        for item in projection.instance_graph.entities
    ]
    event_rows = [
        (
            item.event_id,
            VisualizationObjectKind.EVENT,
            item.label,
            item.contextual_type_id,
            item.reification_reason,
            projection.local_schema.abstraction,
            item.occurrence_time,
            item.uncertainty,
            item.confidence,
            item.evidence_ids,
            item.description,
            item.description_assertion_ids,
            (),
            (),
        )
        for item in projection.instance_graph.events
    ]
    for (
        object_id,
        object_kind,
        label,
        contextual_type_id,
        contextual_role,
        abstraction,
        temporal_state,
        uncertainty,
        confidence,
        evidence_ids,
        description,
        description_assertion_ids,
        mention_ids,
        aliases,
    ) in (*entity_rows, *event_rows):
        if object_id not in selected_node_ids:
            continue
        if not set(description_assertion_ids).issubset(accepted):
            continue
        description_support_status = _description_support_status(
            description_assertion_ids, assessments
        )
        displayed_description = description
        if (
            semantic_display_mode is SemanticDisplayMode.SUPPORTED_ONLY
            and description_support_status != "supported"
        ):
            displayed_description = NODE_DESCRIPTION_WITHHELD
        type_definition = type_definitions.get(contextual_type_id)
        if type_definition is None:
            raise VisualizationCompilationError(
                f"visual object {object_id!r} uses undefined contextual type"
            )
        visual_id = visual_ids[object_id]
        object_to_visual[object_id] = visual_id
        node = VisualizationNode(
            visualization_node_id=visual_id,
            projection_object_id=object_id,
            projection_object_hash=next(
                item.content_hash
                for item in (*projection.instance_graph.entities, *projection.instance_graph.events)
                if getattr(item, "entity_id", getattr(item, "event_id", None)) == object_id
            ),
            contextual_label=label,
            contextual_type_id=contextual_type_id,
            contextual_role=contextual_role,
            abstraction=abstraction,
            temporal_state=temporal_state,
            uncertainty=uncertainty,
            confidence=confidence,
            evidence_badge=EvidenceBadge(
                available=bool(evidence_ids),
                evidence_ids=evidence_ids,
                count=len(evidence_ids),
            ),
            description=displayed_description,
            description_assertion_ids=description_assertion_ids,
            release_class=projection.release_class,
        )
        nodes.append(node)
        details.append(
            VisualizationNodeDetail(
                visualization_node_id=visual_id,
                stable_anchor_id=visual_id,
                projection_object_id=object_id,
                object_kind=object_kind,
                mention_candidate_ids=mention_ids,
                evidence_ids=evidence_ids,
                aliases=aliases,
                contextual_type_label=type_definition.label,
                contextual_type_definition=type_definition.definition,
                contextual_role=contextual_role,
                description_support_status=description_support_status,
            )
        )
        x, y = _position_for_anchor(visual_id, config.layout_seed)
        positions.append(NodePosition(visualization_node_id=visual_id, x=x, y=y))

    visual_assertions: list[VisualizationAssertion] = []
    assertion_details: list[VisualizationAssertionDetail] = []
    structural_collisions: dict[str, int] = {}
    for assertion in sorted(
        projection.instance_graph.assertions,
        key=lambda item: (item.content_hash, item.assertion_id),
    ):
        if (
            assertion.assertion_id not in accepted
            or assertion.assertion_id not in selected_assertion_ids
        ):
            continue
        predicate = predicate_definitions.get(assertion.predicate_id)
        if predicate is None:
            raise VisualizationCompilationError(
                f"visual assertion {assertion.assertion_id!r} uses undefined predicate"
            )
        if assertion.subject_id is not None:
            if (
                assertion.subject_id not in object_to_visual
                or assertion.object_id not in object_to_visual
            ):
                raise VisualizationCompilationError(
                    "display selection omitted an endpoint required by a selected assertion"
                )
            source_id = object_to_visual[assertion.subject_id]
            target_id = object_to_visual[assertion.object_id]
            visual_roles: tuple[RoleBinding, ...] = ()
            endpoint_signature: object = {
                "source": source_id,
                "target": target_id,
                "direction": assertion.direction,
            }
        else:
            if any(role.object_id not in object_to_visual for role in assertion.roles):
                raise VisualizationCompilationError(
                    "display selection omitted an n-ary role required by a selected assertion"
                )
            source_id = None
            target_id = None
            visual_roles = tuple(
                RoleBinding(
                    role=role.role,
                    object_id=object_to_visual[role.object_id],
                    evidence_ids=role.evidence_ids,
                )
                for role in assertion.roles
            )
            endpoint_signature = {
                "roles": sorted((role.role, role.object_id) for role in visual_roles)
            }
        structural_payload = {
            "predicate_label": predicate.label.casefold(),
            "parent_upper_relation": predicate.parent_upper_relation,
            "endpoints": endpoint_signature,
            "evidence_ids": sorted(assertion.evidence_ids),
        }
        stable_base = _identifier_digest("assertion-anchor", structural_payload)
        ordinal = structural_collisions.get(stable_base, 0) + 1
        structural_collisions[stable_base] = ordinal
        visual_assertion_id = stable_base if ordinal == 1 else f"{stable_base}-{ordinal}"
        holder = None
        if assertion.epistemic_scope is not None:
            holder = object_to_visual.get(assertion.epistemic_scope.holder_id)
            if holder is None:
                raise VisualizationCompilationError(
                    "display selection omitted the holder required by an epistemic assertion"
                )
        attitude = (
            assertion.epistemic_scope.attitude if assertion.epistemic_scope is not None else None
        )
        assessment = (
            None if assessments is None else assessments[assertion.assertion_id]
        )
        assertion_support_status = (
            "pending" if assessment is None else assessment.assertion_support_status.value
        )
        description_support_status = (
            "pending" if assessment is None else assessment.description_support_status.value
        )
        displayed_why_matters = assertion.why_matters
        if (
            semantic_display_mode is SemanticDisplayMode.SUPPORTED_ONLY
            and description_support_status != "supported"
        ):
            displayed_why_matters = WHY_MATTERS_WITHHELD
        why_matters_evidence_ids = (
            assertion.why_matters_evidence_ids
            if assessment is None
            else assessment.supported_description_evidence_ids
        )
        visual_assertions.append(
            VisualizationAssertion(
                visualization_assertion_id=visual_assertion_id,
                projection_assertion_id=assertion.assertion_id,
                projection_assertion_hash=assertion.content_hash,
                source_visualization_node_id=source_id,
                target_visualization_node_id=target_id,
                roles=visual_roles,
                contextual_label=predicate.label,
                predicate_id=assertion.predicate_id,
                direction=assertion.direction,
                temporal_scope=assertion.temporal_scope,
                uncertainty=_assertion_uncertainty(assertion),
                epistemic_holder_id=holder,
                epistemic_attitude=attitude,
                confidence=assertion.confidence,
                evidence_badge=EvidenceBadge(
                    available=True,
                    evidence_ids=assertion.evidence_ids,
                    count=len(assertion.evidence_ids),
                ),
                provenance=assertion.provenance,
                why_matters=displayed_why_matters,
                why_matters_assertion_id=assertion.assertion_id,
                release_class=projection.release_class,
            )
        )
        assertion_details.append(
            VisualizationAssertionDetail(
                visualization_assertion_id=visual_assertion_id,
                stable_anchor_id=visual_assertion_id,
                projection_assertion_id=assertion.assertion_id,
                predicate_label=predicate.label,
                predicate_definition=predicate.definition,
                contextual_relevance=assertion.contextual_relevance,
                narrative_commitment=assertion.narrative_commitment,
                role_labels=tuple(role.role for role in visual_roles),
                evidence_ids=assertion.evidence_ids,
                why_matters_evidence_ids=why_matters_evidence_ids,
                assertion_support_status=assertion_support_status,
                description_support_status=description_support_status,
                proposition_content_id=assertion.proposition_content_id,
                holder_relative_time=(
                    None
                    if assertion.epistemic_scope is None
                    else assertion.epistemic_scope.holder_relative_time
                ),
            )
        )

    nodes.sort(key=lambda item: item.visualization_node_id)
    details.sort(key=lambda item: item.visualization_node_id)
    positions.sort(key=lambda item: item.visualization_node_id)
    visual_assertions.sort(key=lambda item: item.visualization_assertion_id)
    assertion_details.sort(key=lambda item: item.visualization_assertion_id)
    projection_hash = projection.content_hash
    layout_payload = {
        "algorithm": "preset",
        "coordinate_rule": config.coordinate_rule,
        "seed": config.layout_seed,
        "seed_manifest_hash": config.seed_manifest_hash,
        "layout_seed_entry_hash": config.layout_seed_entry_hash,
    }
    style_payload = {
        "style": config.style_name,
        "progressive_disclosure": config.progressive_disclosure,
    }
    font_payload = {"family": config.font_family, "base_px": config.font_base_px}
    default_full_pending = (
        content_scope is VisualizationContentScope.FULL_STRUCTURAL
        and semantic_overlay is None
        and semantic_display_mode is SemanticDisplayMode.ALL_STRUCTURAL
    )
    state_identity = (
        {"projection_hash": projection_hash, "filter": "none"}
        if default_full_pending
        else {
            "projection_hash": projection_hash,
            "filter": "none",
            "display_selection_hash": display_selection.content_hash,
            "semantic_overlay_hash": (
                None if semantic_overlay is None else semantic_overlay.content_hash
            ),
            "semantic_display_mode": semantic_display_mode.value,
        }
    )
    state = VisualizationState(
        visualization_state_id=_identifier_digest("view", state_identity),
        projection_hash=projection_hash,
        semantic_hash=projection_hash,
        layout_name=config.layout_name,
        layout_config_hash=canonical_sha256(layout_payload),
        style_config_hash=canonical_sha256(style_payload),
        font_config_hash=canonical_sha256(font_payload),
        layout_seed=config.layout_seed,
        viewport=config.viewport,
        nodes=tuple(nodes),
        assertions=tuple(visual_assertions),
        positions=tuple(positions),
        visible_node_ids=tuple(item.visualization_node_id for item in nodes),
        visible_assertion_ids=tuple(item.visualization_assertion_id for item in visual_assertions),
        labels_visible=True,
        temporal_filter=VisualizationTemporalFilter(),
        release_class=projection.release_class,
    )
    evidence_metadata = _evidence_metadata(
        packet,
        include_public_evidence_text=include_public_evidence_text,
    )
    bundle_release_class = (
        ReleaseClass.RESTRICTED
        if projection.release_class is ReleaseClass.RESTRICTED
        or (
            semantic_overlay is not None
            and semantic_overlay.release_class is ReleaseClass.RESTRICTED
        )
        or any(item.release_class is ReleaseClass.RESTRICTED for item in evidence_metadata)
        else ReleaseClass.PUBLIC
    )
    bundle_identity = (
        {"projection_hash": projection_hash}
        if default_full_pending
        else {
            "projection_hash": projection_hash,
            "display_selection_hash": display_selection.content_hash,
            "semantic_overlay_hash": (
                None if semantic_overlay is None else semantic_overlay.content_hash
            ),
            "semantic_display_mode": semantic_display_mode.value,
        }
    )
    return VisualizationBundle(
        bundle_id=_identifier_digest("bundle", bundle_identity),
        projection_id=projection.projection_id,
        projection_hash=projection_hash,
        condition=projection.condition,
        context=context,
        context_hash=context.content_hash,
        snapshot_hash=projection.snapshot_hash,
        packet_hash=packet.content_hash,
        semantic_assessment_status=(
            "pending_scorer_or_reviewer"
            if semantic_overlay is None
            else "verified_scorer_or_reviewer"
        ),
        semantic_display_mode=semantic_display_mode,
        semantic_overlay=semantic_overlay,
        display_selection=display_selection,
        state=state,
        node_details=tuple(details),
        assertion_details=tuple(assertion_details),
        evidence_metadata=evidence_metadata,
        release_class=bundle_release_class,
    )


def _numeric_bounds(value: StoryTime) -> tuple[float, float] | None:
    if value.kind is TemporalKind.POINT:
        assert value.point is not None
        return float(value.point), float(value.point)
    if value.kind is TemporalKind.INTERVAL:
        return (
            float("-inf") if value.start is None else float(value.start),
            float("inf") if value.end is None else float(value.end),
        )
    return None


def _passes_story_filter(value: StoryTime, requested: StoryTime | None) -> bool:
    if value.kind in {TemporalKind.HORIZON_WITHHELD, TemporalKind.INVALID}:
        return False
    if requested is None:
        return True
    value_bounds = _numeric_bounds(value)
    requested_bounds = _numeric_bounds(requested)
    # Unknown, relative, and partial-order time is retained rather than falsely asserted
    # outside a numeric interval.
    if value_bounds is None or requested_bounds is None:
        return True
    return value_bounds[0] <= requested_bounds[1] and requested_bounds[0] <= value_bounds[1]


def _assertion_passes_temporal_filter(
    assertion: VisualizationAssertion,
    temporal_filter: VisualizationTemporalFilter,
) -> bool:
    from story_projection_onto.temporal import query_time_visibility

    if assertion.temporal_scope.story_time.kind in {
        TemporalKind.HORIZON_WITHHELD,
        TemporalKind.INVALID,
    }:
        return False
    if (
        temporal_filter.story_scope is not None
        and query_time_visibility(
            assertion.temporal_scope.story_time,
            assertion.temporal_scope.validity_time,
            temporal_filter.story_scope,
        )
        is False
    ):
        return False
    horizon = temporal_filter.spoiler_horizon
    if horizon is not None and (
        not discourse_is_within_horizon(
            assertion.temporal_scope.discourse_position,
            horizon,
        )
        or not revelation_is_within_horizon(
            assertion.temporal_scope.revelation_position,
            horizon,
        )
    ):
        return False
    return not (
        temporal_filter.epistemic_holder_id is not None
        and assertion.epistemic_holder_id != temporal_filter.epistemic_holder_id
    )


def _visual_assertion_endpoint_ids(assertion: VisualizationAssertion) -> frozenset[str]:
    endpoint_ids = {
        item
        for item in (
            assertion.source_visualization_node_id,
            assertion.target_visualization_node_id,
        )
        if item is not None
    }
    endpoint_ids.update(role.object_id for role in assertion.roles)
    if assertion.epistemic_holder_id is not None:
        endpoint_ids.add(assertion.epistemic_holder_id)
    return frozenset(endpoint_ids)


def filter_visualization_bundle(
    bundle: VisualizationBundle,
    temporal_filter: VisualizationTemporalFilter,
) -> VisualizationBundle:
    """Apply story/spoiler/holder visibility without mutating semantic content."""

    evidence_by_id = {item.evidence_id: item for item in bundle.evidence_metadata}
    assertions_by_projection_id = {
        item.projection_assertion_id: item for item in bundle.state.assertions
    }
    visible_nodes: set[str] = set()
    for node in bundle.state.nodes:
        if not _passes_story_filter(node.temporal_state, temporal_filter.story_scope):
            continue
        if temporal_filter.spoiler_horizon is not None:
            eligible = [
                evidence_by_id[evidence_id]
                for evidence_id in node.evidence_badge.evidence_ids
                if evidence_id in evidence_by_id
            ]
            # A lower temporary horizon must not reveal a label, alias, or entity merge
            # that depends partly on later evidence.  Requiring every cited item to be in
            # scope is conservative and keeps rendering downstream of evidence access.
            if eligible and not all(
                discourse_is_within_horizon(
                    item.discourse_position,
                    temporal_filter.spoiler_horizon,
                )
                for item in eligible
            ):
                continue
            supporting_assertions = [
                assertions_by_projection_id[assertion_id]
                for assertion_id in node.description_assertion_ids
                if assertion_id in assertions_by_projection_id
            ]
            # Node labels/descriptions can leak a later revelation even when the node's
            # direct evidence badge itself is early. Hide the node if any assertion on
            # which its supported description depends is beyond either horizon axis.
            if supporting_assertions and not all(
                _assertion_passes_temporal_filter(item, temporal_filter)
                for item in supporting_assertions
            ):
                continue
        visible_nodes.add(node.visualization_node_id)

    assertions_by_id = {item.visualization_assertion_id: item for item in bundle.state.assertions}
    temporally_eligible = {
        item.visualization_assertion_id
        for item in bundle.state.assertions
        if _assertion_passes_temporal_filter(item, temporal_filter)
    }
    visible_assertions = sorted(
        assertion_id
        for assertion_id in temporally_eligible
        if _visual_assertion_endpoint_ids(assertions_by_id[assertion_id]).issubset(visible_nodes)
    )

    payload = bundle.state.model_dump(mode="python", exclude={"content_hash"})
    payload.update(
        {
            "visualization_state_id": _identifier_digest(
                "view",
                {
                    "projection_hash": bundle.projection_hash,
                    "filter_hash": temporal_filter.content_hash,
                },
            ),
            "visible_node_ids": tuple(sorted(visible_nodes)),
            "visible_assertion_ids": tuple(sorted(visible_assertions)),
            "temporal_filter": temporal_filter,
        }
    )
    state = VisualizationState(**payload)
    bundle_payload = bundle.model_dump(mode="python", exclude={"content_hash"})
    bundle_payload.update(
        {
            "bundle_id": _identifier_digest(
                "bundle-filtered",
                {
                    "source_bundle_hash": bundle.content_hash,
                    "filter_hash": temporal_filter.content_hash,
                },
            ),
            "state": state,
        }
    )
    return VisualizationBundle(**bundle_payload)


class VisualizationChangeKind(StrEnum):
    ADDED = "added"
    REMOVED = "removed"
    MERGED = "merged"
    SPLIT = "split"
    REQUALIFIED = "requalified"


class VisualizationChange(ImmutableRecord):
    change_id: Identifier
    kind: VisualizationChangeKind
    object_kind: Literal["node", "assertion"]
    before_visualization_ids: tuple[Identifier, ...] = ()
    after_visualization_ids: tuple[Identifier, ...] = ()
    anchor_ids: tuple[Identifier, ...]

    @model_validator(mode="after")
    def validate_change_shape(self) -> Self:
        if not self.before_visualization_ids and not self.after_visualization_ids:
            raise ValueError("visualization changes require a before or after object")
        if not self.anchor_ids:
            raise ValueError("visualization changes require condition-independent anchors")
        if self.kind is VisualizationChangeKind.ADDED and self.before_visualization_ids:
            raise ValueError("added changes cannot have before IDs")
        if self.kind is VisualizationChangeKind.REMOVED and self.after_visualization_ids:
            raise ValueError("removed changes cannot have after IDs")
        if self.kind is VisualizationChangeKind.MERGED and (
            len(self.before_visualization_ids) < 2 or len(self.after_visualization_ids) != 1
        ):
            raise ValueError("merge diffs require many before nodes and one after node")
        if self.kind is VisualizationChangeKind.SPLIT and (
            len(self.before_visualization_ids) != 1 or len(self.after_visualization_ids) < 2
        ):
            raise ValueError("split diffs require one before node and many after nodes")
        return self


class VisualizationDiff(ImmutableRecord):
    diff_id: Identifier
    before_projection_id: Identifier
    before_projection_hash: Sha256Digest
    after_projection_id: Identifier
    after_projection_hash: Sha256Digest
    changes: tuple[VisualizationChange, ...]
    shared_fixed_anchor_ids: tuple[Identifier, ...]
    fixed_anchor_positions_preserved: bool


def _detail_anchor_sets(
    details: Sequence[VisualizationNodeDetail],
) -> dict[str, frozenset[str]]:
    return {
        item.visualization_node_id: frozenset(item.mention_candidate_ids or item.evidence_ids)
        for item in details
    }


def _semantic_view_hash(value, *, excluded: frozenset[str]) -> str:
    payload = value.model_dump(mode="python", exclude={"content_hash", *excluded})
    return canonical_sha256(payload)


def compare_visualizations(
    before: VisualizationBundle,
    after: VisualizationBundle,
) -> VisualizationDiff:
    """Create a deterministic anchor-aligned before/after graph diff."""

    before_nodes = {item.visualization_node_id: item for item in before.state.nodes}
    after_nodes = {item.visualization_node_id: item for item in after.state.nodes}
    before_anchors = _detail_anchor_sets(before.node_details)
    after_anchors = _detail_anchor_sets(after.node_details)
    before_kinds = {item.visualization_node_id: item.object_kind for item in before.node_details}
    after_kinds = {item.visualization_node_id: item.object_kind for item in after.node_details}
    changes: list[VisualizationChange] = []
    consumed_before: set[str] = set()
    consumed_after: set[str] = set()

    for after_id, after_set in sorted(after_anchors.items()):
        if after_kinds[after_id] is not VisualizationObjectKind.ENTITY:
            continue
        overlapping = tuple(
            sorted(
                before_id
                for before_id, before_set in before_anchors.items()
                if before_kinds[before_id] is VisualizationObjectKind.ENTITY
                and before_set.intersection(after_set)
            )
        )
        overlapping_union = frozenset().union(*(before_anchors[item] for item in overlapping))
        overlaps_another_after_entity = any(
            other_id != after_id
            and after_kinds[other_id] is VisualizationObjectKind.ENTITY
            and bool(other_set.intersection(after_set))
            for other_id, other_set in after_anchors.items()
        )
        if (
            len(overlapping) >= 2
            and overlapping_union == after_set
            and not overlaps_another_after_entity
        ):
            anchors = tuple(
                sorted(after_set.union(*(before_anchors[item] for item in overlapping)))
            )
            changes.append(
                VisualizationChange(
                    change_id=_identifier_digest(
                        "change", {"kind": "merged", "before": overlapping, "after": after_id}
                    ),
                    kind=VisualizationChangeKind.MERGED,
                    object_kind="node",
                    before_visualization_ids=overlapping,
                    after_visualization_ids=(after_id,),
                    anchor_ids=anchors,
                )
            )
            consumed_before.update(overlapping)
            consumed_after.add(after_id)

    for before_id, before_set in sorted(before_anchors.items()):
        if before_kinds[before_id] is not VisualizationObjectKind.ENTITY:
            continue
        overlapping = tuple(
            sorted(
                after_id
                for after_id, after_set in after_anchors.items()
                if after_kinds[after_id] is VisualizationObjectKind.ENTITY
                and before_set.intersection(after_set)
            )
        )
        overlapping_union = frozenset().union(*(after_anchors[item] for item in overlapping))
        overlaps_another_before_entity = any(
            other_id != before_id
            and before_kinds[other_id] is VisualizationObjectKind.ENTITY
            and bool(other_set.intersection(before_set))
            for other_id, other_set in before_anchors.items()
        )
        if (
            len(overlapping) >= 2
            and before_id not in consumed_before
            and overlapping_union == before_set
            and not overlaps_another_before_entity
        ):
            anchors = tuple(
                sorted(before_set.union(*(after_anchors[item] for item in overlapping)))
            )
            changes.append(
                VisualizationChange(
                    change_id=_identifier_digest(
                        "change", {"kind": "split", "before": before_id, "after": overlapping}
                    ),
                    kind=VisualizationChangeKind.SPLIT,
                    object_kind="node",
                    before_visualization_ids=(before_id,),
                    after_visualization_ids=overlapping,
                    anchor_ids=anchors,
                )
            )
            consumed_before.add(before_id)
            consumed_after.update(overlapping)

    common_nodes = set(before_nodes).intersection(after_nodes)
    before_node_details = {item.visualization_node_id: item for item in before.node_details}
    after_node_details = {item.visualization_node_id: item for item in after.node_details}
    for visual_id in sorted(common_nodes):
        if visual_id in consumed_before or visual_id in consumed_after:
            continue
        if _semantic_view_hash(
            before_nodes[visual_id],
            excluded=frozenset({"projection_object_id", "projection_object_hash"}),
        ) != _semantic_view_hash(
            after_nodes[visual_id],
            excluded=frozenset({"projection_object_id", "projection_object_hash"}),
        ) or _semantic_view_hash(
            before_node_details[visual_id],
            excluded=frozenset({"projection_object_id"}),
        ) != _semantic_view_hash(
            after_node_details[visual_id],
            excluded=frozenset({"projection_object_id"}),
        ):
            changes.append(
                VisualizationChange(
                    change_id=_identifier_digest(
                        "change", {"kind": "node-requalified", "id": visual_id}
                    ),
                    kind=VisualizationChangeKind.REQUALIFIED,
                    object_kind="node",
                    before_visualization_ids=(visual_id,),
                    after_visualization_ids=(visual_id,),
                    anchor_ids=tuple(sorted(before_anchors[visual_id] | after_anchors[visual_id])),
                )
            )

    for visual_id in sorted(set(before_nodes) - set(after_nodes) - consumed_before):
        changes.append(
            VisualizationChange(
                change_id=_identifier_digest("change", {"kind": "node-removed", "id": visual_id}),
                kind=VisualizationChangeKind.REMOVED,
                object_kind="node",
                before_visualization_ids=(visual_id,),
                anchor_ids=tuple(sorted(before_anchors[visual_id])),
            )
        )
    for visual_id in sorted(set(after_nodes) - set(before_nodes) - consumed_after):
        changes.append(
            VisualizationChange(
                change_id=_identifier_digest("change", {"kind": "node-added", "id": visual_id}),
                kind=VisualizationChangeKind.ADDED,
                object_kind="node",
                after_visualization_ids=(visual_id,),
                anchor_ids=tuple(sorted(after_anchors[visual_id])),
            )
        )

    before_assertions = {item.visualization_assertion_id: item for item in before.state.assertions}
    after_assertions = {item.visualization_assertion_id: item for item in after.state.assertions}
    before_assertion_details = {
        item.visualization_assertion_id: item for item in before.assertion_details
    }
    after_assertion_details = {
        item.visualization_assertion_id: item for item in after.assertion_details
    }
    common_assertions = set(before_assertions).intersection(after_assertions)
    for visual_id in sorted(common_assertions):
        if _semantic_view_hash(
            before_assertions[visual_id],
            excluded=frozenset({"projection_assertion_id", "projection_assertion_hash"}),
        ) != _semantic_view_hash(
            after_assertions[visual_id],
            excluded=frozenset({"projection_assertion_id", "projection_assertion_hash"}),
        ) or _semantic_view_hash(
            before_assertion_details[visual_id],
            excluded=frozenset({"projection_assertion_id"}),
        ) != _semantic_view_hash(
            after_assertion_details[visual_id],
            excluded=frozenset({"projection_assertion_id"}),
        ):
            anchor_ids = tuple(
                sorted(
                    set(before_assertion_details[visual_id].evidence_ids)
                    | set(after_assertion_details[visual_id].evidence_ids)
                )
            )
            changes.append(
                VisualizationChange(
                    change_id=_identifier_digest(
                        "change", {"kind": "assertion-requalified", "id": visual_id}
                    ),
                    kind=VisualizationChangeKind.REQUALIFIED,
                    object_kind="assertion",
                    before_visualization_ids=(visual_id,),
                    after_visualization_ids=(visual_id,),
                    anchor_ids=anchor_ids,
                )
            )
    for visual_id in sorted(set(before_assertions) - set(after_assertions)):
        changes.append(
            VisualizationChange(
                change_id=_identifier_digest(
                    "change", {"kind": "assertion-removed", "id": visual_id}
                ),
                kind=VisualizationChangeKind.REMOVED,
                object_kind="assertion",
                before_visualization_ids=(visual_id,),
                anchor_ids=before_assertion_details[visual_id].evidence_ids,
            )
        )
    for visual_id in sorted(set(after_assertions) - set(before_assertions)):
        changes.append(
            VisualizationChange(
                change_id=_identifier_digest(
                    "change", {"kind": "assertion-added", "id": visual_id}
                ),
                kind=VisualizationChangeKind.ADDED,
                object_kind="assertion",
                after_visualization_ids=(visual_id,),
                anchor_ids=after_assertion_details[visual_id].evidence_ids,
            )
        )

    before_positions = {
        item.visualization_node_id: (item.x, item.y) for item in before.state.positions
    }
    after_positions = {
        item.visualization_node_id: (item.x, item.y) for item in after.state.positions
    }
    shared = tuple(sorted(set(before_positions).intersection(after_positions)))
    positions_preserved = all(before_positions[item] == after_positions[item] for item in shared)
    changes.sort(key=lambda item: (item.object_kind, item.kind.value, item.change_id))
    return VisualizationDiff(
        diff_id=_identifier_digest(
            "diff",
            {
                "before": before.projection_hash,
                "after": after.projection_hash,
                "changes": [item.content_hash for item in changes],
            },
        ),
        before_projection_id=before.projection_id,
        before_projection_hash=before.projection_hash,
        after_projection_id=after.projection_id,
        after_projection_hash=after.projection_hash,
        changes=tuple(changes),
        shared_fixed_anchor_ids=shared,
        fixed_anchor_positions_preserved=positions_preserved,
    )


class MergeSplitOperation(StrEnum):
    MERGE = "merge"
    SPLIT = "split"


class RefineContextIntent(ImmutableRecord):
    after_context_id: Identifier
    after_revealed_at: AwareDatetime
    lens: str | None = Field(default=None, min_length=1)
    story_scope: StoryTime | None = None
    spoiler_horizon: SpoilerHorizon | None = None

    @model_validator(mode="after")
    def refinement_changes_registered_context_fields(self) -> Self:
        if self.lens is None and self.story_scope is None and self.spoiler_horizon is None:
            raise ValueError("REFINE_CONTEXT must change lens, story scope, or spoiler horizon")
        return self


class MergeSplitIntent(ImmutableRecord):
    operation: MergeSplitOperation
    grouped_mention_candidate_ids: tuple[tuple[Identifier, ...], ...]

    @model_validator(mode="after")
    def groups_form_a_real_identity_request(self) -> Self:
        if len(self.grouped_mention_candidate_ids) < 2:
            raise ValueError("merge/split requests require at least two mention groups")
        flattened: list[str] = []
        for group in self.grouped_mention_candidate_ids:
            if not group:
                raise ValueError("merge/split mention groups cannot be empty")
            if len(group) != len(set(group)):
                raise ValueError("mention IDs must be unique within a merge/split group")
            flattened.extend(group)
        if len(flattened) != len(set(flattened)):
            raise ValueError("mention IDs cannot occur in more than one merge/split group")
        return self


class RevisionInstruction(ImmutableRecord):
    """Runner-only typed intent that compiles to the shared ``UserRevision``."""

    revision: UserRevision
    before_context: QueryContext
    after_context: QueryContext
    refine_context_intent: RefineContextIntent | None = None
    merge_split_intent: MergeSplitIntent | None = None

    @model_validator(mode="after")
    def typed_intent_matches_shared_revision(self) -> Self:
        if not self.revision.anchors:
            raise ValueError("typed revisions require condition-independent anchors")
        if self.revision.before_context_hash != self.before_context.content_hash:
            raise ValueError("revision does not hash its before context")
        if self.revision.after_context_hash != self.after_context.content_hash:
            raise ValueError("revision does not hash its after context")
        populated = sum(
            item is not None for item in (self.refine_context_intent, self.merge_split_intent)
        )
        if populated != 1:
            raise ValueError("revision instruction requires exactly one typed intent")
        intent = self.refine_context_intent or self.merge_split_intent
        expected_action = (
            FeedbackAction.REFINE_CONTEXT
            if self.refine_context_intent is not None
            else FeedbackAction.REQUEST_MERGE_SPLIT
        )
        if self.revision.action is not expected_action:
            raise ValueError("typed intent and UserRevision action differ")
        expected_change = canonical_json(intent, include_content_hash=False)
        if self.revision.requested_change != expected_change:
            raise ValueError("UserRevision requested_change does not serialize its typed intent")
        if self.revision.created_at < self.before_context.revealed_at:
            raise ValueError("a revision cannot predate its original query reveal")
        if (
            self.refine_context_intent is not None
            and self.after_context.revealed_at < self.revision.created_at
        ):
            raise ValueError("the revised context cannot be revealed before the revision")
        if self.refine_context_intent is not None:
            replayed_context = apply_context_refinement(
                self.before_context,
                self.refine_context_intent,
            )
            if replayed_context != self.after_context:
                raise ValueError("revised context does not replay from its typed intent")
            if replayed_context.content_hash == self.before_context.content_hash:
                raise ValueError("REFINE_CONTEXT must make a semantic context change")
        if self.merge_split_intent is not None:
            if self.before_context != self.after_context:
                raise ValueError("merge/split feedback cannot silently change query context")
            requested_mentions = {
                item
                for group in self.merge_split_intent.grouped_mention_candidate_ids
                for item in group
            }
            anchored_mentions = {
                item for anchor in self.revision.anchors for item in anchor.mention_candidate_ids
            }
            if requested_mentions != anchored_mentions:
                raise ValueError("merge/split groups must exactly equal anchored mention IDs")
        return self

    def model_visible(
        self,
        own_resolution: FeedbackResolution | None = None,
    ) -> ModelVisibleRevision:
        return ModelVisibleRevision(
            revision=self.revision,
            own_resolution=own_resolution,
        )


def apply_context_refinement(
    before: QueryContext,
    intent: RefineContextIntent,
) -> QueryContext:
    payload = before.model_dump(mode="python", exclude={"content_hash"})
    payload["context_id"] = intent.after_context_id
    payload["revealed_at"] = intent.after_revealed_at
    if intent.lens is not None:
        payload["lens"] = intent.lens
    if intent.story_scope is not None:
        payload["story_scope"] = intent.story_scope
    if intent.spoiler_horizon is not None:
        payload["spoiler_horizon"] = intent.spoiler_horizon
    return QueryContext(**payload)


def build_refine_context_instruction(
    *,
    revision_id: str,
    before_context: QueryContext,
    intent: RefineContextIntent,
    anchors: Sequence[FeedbackAnchor],
    rationale: str,
    sequence: int,
    created_at: datetime,
) -> RevisionInstruction:
    after_context = apply_context_refinement(before_context, intent)
    revision = UserRevision(
        revision_id=revision_id,
        action=FeedbackAction.REFINE_CONTEXT,
        anchors=tuple(anchors),
        requested_change=canonical_json(intent, include_content_hash=False),
        rationale=rationale,
        sequence=sequence,
        before_context_hash=before_context.content_hash,
        after_context_hash=after_context.content_hash,
        created_at=created_at,
    )
    return RevisionInstruction(
        revision=revision,
        before_context=before_context,
        after_context=after_context,
        refine_context_intent=intent,
    )


def build_merge_split_instruction(
    *,
    revision_id: str,
    context: QueryContext,
    intent: MergeSplitIntent,
    anchors: Sequence[FeedbackAnchor],
    rationale: str,
    sequence: int,
    created_at: datetime,
) -> RevisionInstruction:
    revision = UserRevision(
        revision_id=revision_id,
        action=FeedbackAction.REQUEST_MERGE_SPLIT,
        anchors=tuple(anchors),
        requested_change=canonical_json(intent, include_content_hash=False),
        rationale=rationale,
        sequence=sequence,
        before_context_hash=context.content_hash,
        after_context_hash=context.content_hash,
        created_at=created_at,
    )
    return RevisionInstruction(
        revision=revision,
        before_context=context,
        after_context=context,
        merge_split_intent=intent,
    )


def assert_revision_anchors_resolve_in_packet(
    instruction: RevisionInstruction,
    packet: EvidencePacket,
) -> None:
    evidence_ids = {item.evidence_id for item in packet.evidence}
    mention_to_evidence = {
        mention.candidate_id: record.evidence_id
        for record in packet.evidence
        for mention in record.mention_candidates
    }
    for anchor in instruction.revision.anchors:
        if not set(anchor.evidence_ids).issubset(evidence_ids):
            raise ValueError("feedback anchor references evidence outside the frozen packet")
        if not set(anchor.mention_candidate_ids).issubset(mention_to_evidence):
            raise ValueError("feedback anchor references mentions outside the frozen packet")
        if anchor.evidence_ids and any(
            mention_to_evidence[item] not in anchor.evidence_ids
            for item in anchor.mention_candidate_ids
        ):
            raise ValueError("feedback mention anchors must belong to their named evidence")


def assert_revision_has_no_projection_local_anchors(
    instruction: RevisionInstruction,
    projections: Sequence[OntologyProjection],
) -> None:
    local_ids: set[str] = set()
    for projection in projections:
        local_ids.add(projection.local_schema.schema_id)
        local_ids.update(item.type_id for item in projection.local_schema.contextual_types)
        local_ids.update(item.predicate_id for item in projection.local_schema.predicates)
        local_ids.update(item.entity_id for item in projection.instance_graph.entities)
        local_ids.update(item.event_id for item in projection.instance_graph.events)
        local_ids.update(item.assertion_id for item in projection.instance_graph.assertions)
    anchor_ids = {
        item
        for anchor in instruction.revision.anchors
        for item in (*anchor.evidence_ids, *anchor.mention_candidate_ids)
    }
    anchor_ids.update(
        anchor.role_constraint
        for anchor in instruction.revision.anchors
        if anchor.role_constraint is not None
    )
    anchor_ids.update(
        anchor.time_constraint.anchor_id
        for anchor in instruction.revision.anchors
        if anchor.time_constraint is not None and anchor.time_constraint.anchor_id is not None
    )
    semantic_tokens = {
        token
        for anchor in instruction.revision.anchors
        for token in re.findall(
            r"[A-Za-z0-9][A-Za-z0-9_.:/-]*",
            anchor.requested_semantic_signature,
        )
    }
    semantic_tokens.update(
        re.findall(
            r"[A-Za-z0-9][A-Za-z0-9_.:/-]*",
            instruction.revision.rationale,
        )
    )
    anchor_ids.update(semantic_tokens)
    overlap = sorted(anchor_ids.intersection(local_ids))
    if overlap:
        raise ValueError("feedback anchors contain projection-local IDs: " + ", ".join(overlap))


def _entity_ids_for_groups(
    projection: OntologyProjection,
    groups: Sequence[Sequence[str]],
) -> tuple[str, ...] | None:
    resolved: list[str] = []
    for group in groups:
        matches = [
            entity.entity_id
            for entity in projection.instance_graph.entities
            if set(group).issubset(entity.supported_mention_candidate_ids)
        ]
        if len(matches) != 1:
            return None
        resolved.append(matches[0])
    return tuple(resolved)


def _merge_split_result_satisfies(
    intent: MergeSplitIntent,
    before: OntologyProjection,
    after: OntologyProjection,
) -> tuple[bool, tuple[str, ...]]:
    before_ids = _entity_ids_for_groups(
        before,
        intent.grouped_mention_candidate_ids,
    )
    after_ids = _entity_ids_for_groups(
        after,
        intent.grouped_mention_candidate_ids,
    )
    if before_ids is None or after_ids is None:
        return False, ()
    if intent.operation is MergeSplitOperation.MERGE:
        valid = len(set(before_ids)) >= 2 and len(set(after_ids)) == 1
        required_operator = ConstructionOperator.MERGE
    else:
        valid = len(set(before_ids)) == 1 and len(set(after_ids)) >= 2
        required_operator = ConstructionOperator.SPLIT
    before_targets = set(before_ids)
    after_targets = set(after_ids)
    certificate_records_operation = any(
        decision.operator is required_operator
        and before_targets.issubset(
            set(decision.input_object_ids) | set(decision.removed_object_ids)
        )
        and after_targets.issubset(decision.created_object_ids)
        for decision in after.decisions
    )
    return valid and certificate_records_operation, tuple(sorted(set(before_ids)))


def resolve_feedback_for_condition(
    *,
    instruction: RevisionInstruction,
    packet: EvidencePacket,
    receiving_condition: ConditionName,
    before_projection: OntologyProjection,
    after_projection: OntologyProjection | None,
    after_packet: EvidencePacket | None = None,
    resolver_hash: str,
    seed: int,
    resolved_at: datetime,
) -> FeedbackResolution:
    """Resolve one shared revision independently for one condition.

    This function validates decisions; it never constructs the requested semantics.  C2's
    ``after_projection`` must therefore come from its separately metered regeneration.
    """

    if before_projection.condition is not receiving_condition:
        raise ValueError("feedback resolution condition differs from before projection")
    if before_projection.context_hash != instruction.revision.before_context_hash:
        raise ValueError("feedback before projection belongs to a different context")
    if before_projection.packet_hash != packet.content_hash:
        raise ValueError("feedback resolution packet differs from before projection")
    if resolved_at < instruction.revision.created_at:
        raise ValueError("feedback resolution cannot predate the recorded revision")
    if (
        receiving_condition in ACTIVE_QUERY_CONSTRUCTION_CONDITIONS
        and before_projection.construction_certificate is not None
        and before_projection.construction_certificate.completed_at
        > instruction.revision.created_at
    ):
        raise ValueError("feedback revision cannot predate its C2 parent projection")
    assert_revision_anchors_resolve_in_packet(instruction, packet)
    projections = (
        (before_projection,) if after_projection is None else (before_projection, after_projection)
    )
    assert_revision_has_no_projection_local_anchors(instruction, projections)

    resolved_ids: tuple[str, ...] = ()
    status: FeedbackResolutionStatus
    if instruction.merge_split_intent is not None:
        before_groups = _entity_ids_for_groups(
            before_projection,
            instruction.merge_split_intent.grouped_mention_candidate_ids,
        )
        resolved_ids = () if before_groups is None else tuple(sorted(set(before_groups)))
        if receiving_condition not in ACTIVE_QUERY_CONSTRUCTION_CONDITIONS:
            status = FeedbackResolutionStatus.CAPABILITY_LIMITED
            after_projection = None
        elif after_projection is None:
            status = FeedbackResolutionStatus.INVALID
        else:
            valid, resolved_ids = _merge_split_result_satisfies(
                instruction.merge_split_intent,
                before_projection,
                after_projection,
            )
            status = (
                FeedbackResolutionStatus.RESOLVED if valid else FeedbackResolutionStatus.INVALID
            )
    elif after_projection is None:
        status = (
            FeedbackResolutionStatus.CAPABILITY_LIMITED
            if receiving_condition
            in {
                ConditionName.C0_CLASSICAL_PRE,
                ConditionName.C1_LLM_PRE,
                ConditionName.A_FIXED_SELECT,
            }
            else FeedbackResolutionStatus.INVALID
        )
    else:
        status = FeedbackResolutionStatus.RESOLVED

    if after_projection is not None:
        if after_projection.condition is not receiving_condition:
            raise ValueError("feedback before/after conditions differ")
        if after_projection.context_hash != instruction.revision.after_context_hash:
            raise ValueError("feedback after projection belongs to a different context")
        spoiler_changed = (
            instruction.before_context.spoiler_horizon != instruction.after_context.spoiler_horizon
        )
        if spoiler_changed:
            if after_packet is None:
                raise ValueError("spoiler-scope reconstruction requires its new frozen packet")
            if after_projection.snapshot_hash == before_projection.snapshot_hash:
                raise ValueError("spoiler-scope reconstruction requires a newly sealed snapshot")
            if after_packet.snapshot_hash != after_projection.snapshot_hash:
                raise ValueError("new spoiler-scope packet and projection snapshots differ")
            if after_projection.packet_hash != after_packet.content_hash:
                raise ValueError("feedback after projection and new packet differ")
            assert_revision_anchors_resolve_in_packet(instruction, after_packet)
        elif (
            after_projection.snapshot_hash != before_projection.snapshot_hash
            or after_projection.packet_hash != before_projection.packet_hash
        ):
            raise ValueError(
                "lens/story refinements and merge/split must reuse snapshot and packet"
            )
        if receiving_condition in {
            ConditionName.C0_CLASSICAL_PRE,
            ConditionName.C1_LLM_PRE,
            ConditionName.A_FIXED_SELECT,
        } and (
            before_projection.construction_seal is None
            or after_projection.construction_seal is None
            or before_projection.construction_seal.content_hash
            != after_projection.construction_seal.content_hash
        ):
            raise ValueError("preconstructed feedback may only reproject the same seal")
        if receiving_condition in ACTIVE_QUERY_CONSTRUCTION_CONDITIONS:
            certificate = after_projection.construction_certificate
            if certificate is None:  # guarded by OntologyProjection; explicit for clarity
                raise ValueError("C2 feedback requires a construction certificate")
            if certificate.completed_at <= instruction.revision.created_at:
                raise ValueError("C2 feedback output was not reconstructed after the revision")
            if certificate.completed_at > resolved_at:
                raise ValueError("feedback cannot resolve before C2 reconstruction completes")
            if any(
                decision.decided_at <= instruction.revision.created_at
                for decision in after_projection.decisions
            ):
                raise ValueError("C2 feedback decisions must be made after the revision")

    return FeedbackResolution(
        resolution_id=_identifier_digest(
            "resolution",
            {
                "revision_hash": instruction.revision.content_hash,
                "condition": receiving_condition,
                "before_projection_hash": before_projection.content_hash,
                "after_projection_hash": (
                    None if after_projection is None else after_projection.content_hash
                ),
                "status": status,
            },
        ),
        revision_hash=instruction.revision.content_hash,
        receiving_condition=receiving_condition,
        resolved_object_ids=resolved_ids,
        status=status,
        before_projection_id=before_projection.projection_id,
        before_projection_hash=before_projection.content_hash,
        after_projection_id=(
            after_projection.projection_id if status is FeedbackResolutionStatus.RESOLVED else None
        ),
        after_projection_hash=(
            after_projection.content_hash if status is FeedbackResolutionStatus.RESOLVED else None
        ),
        resolver_hash=resolver_hash,
        seed=seed,
        resolved_at=resolved_at,
    )


class FeedbackReplayExpectation(ImmutableRecord):
    instruction_hash: Sha256Digest
    resolution_hash: Sha256Digest
    before_bundle_hash: Sha256Digest
    after_bundle_hash: Sha256Digest
    diff_hash: Sha256Digest


class FeedbackReplayResult(ImmutableRecord):
    expectation_hash: Sha256Digest
    observed_instruction_hash: Sha256Digest
    observed_resolution_hash: Sha256Digest
    observed_before_bundle_hash: Sha256Digest
    observed_after_bundle_hash: Sha256Digest
    observed_diff_hash: Sha256Digest
    replay_hash_success: bool
    diagnostics: tuple[str, ...]
    checked_at: AwareDatetime


def verify_feedback_replay(
    *,
    expectation: FeedbackReplayExpectation,
    instruction: RevisionInstruction,
    resolution: FeedbackResolution,
    before_bundle: VisualizationBundle,
    after_bundle: VisualizationBundle,
    diff: VisualizationDiff,
    checked_at: datetime,
) -> FeedbackReplayResult:
    recomputed_diff = compare_visualizations(before_bundle, after_bundle)
    observed = {
        "instruction_hash": instruction.content_hash,
        "resolution_hash": resolution.content_hash,
        "before_bundle_hash": before_bundle.content_hash,
        "after_bundle_hash": after_bundle.content_hash,
        "diff_hash": recomputed_diff.content_hash,
    }
    expected = expectation.model_dump(mode="python", exclude={"schema_version", "content_hash"})
    diagnostics = tuple(
        f"{name}: expected {expected[name]}, observed {observed[name]}"
        for name in sorted(observed)
        if observed[name] != expected[name]
    )
    if diff.content_hash != recomputed_diff.content_hash:
        diagnostics = (*diagnostics, "supplied diff does not reproduce from the two bundles")
    return FeedbackReplayResult(
        expectation_hash=expectation.content_hash,
        observed_instruction_hash=observed["instruction_hash"],
        observed_resolution_hash=observed["resolution_hash"],
        observed_before_bundle_hash=observed["before_bundle_hash"],
        observed_after_bundle_hash=observed["after_bundle_hash"],
        observed_diff_hash=observed["diff_hash"],
        replay_hash_success=not diagnostics,
        diagnostics=diagnostics,
        checked_at=checked_at,
    )


class FeedbackEpisodeKind(StrEnum):
    SCRIPTED_KNOWN_ANSWER = "scripted_known_answer"
    RESEARCHER_TRACE = "researcher_trace"


class FeedbackEpisodePlan(ImmutableRecord):
    scorer_namespace: Literal["scorer_only"] = "scorer_only"
    episode_id: Identifier
    kind: FeedbackEpisodeKind
    action: FeedbackAction
    context_id: Identifier
    seed: int = Field(ge=0)
    anchor_hashes: tuple[Sha256Digest, ...]
    known_answer_target_ids: tuple[Identifier, ...] | None = None

    @model_validator(mode="after")
    def gold_targets_are_script_only(self) -> Self:
        if not self.anchor_hashes:
            raise ValueError("feedback episodes require frozen condition-independent anchors")
        if self.kind is FeedbackEpisodeKind.SCRIPTED_KNOWN_ANSWER:
            if not self.known_answer_target_ids:
                raise ValueError("known-answer scripts require scorer-only target IDs")
        elif self.known_answer_target_ids is not None:
            raise ValueError("researcher traces must keep gold-dependent targets NA")
        return self


class FeedbackStudyPlan(ImmutableRecord):
    episodes: tuple[FeedbackEpisodePlan, ...]

    @model_validator(mode="after")
    def enforce_reduced_feedback_inventory(self) -> Self:
        if len(self.episodes) != 9:
            raise ValueError("Phase 5 requires exactly six scripts and three traces")
        if len({item.episode_id for item in self.episodes}) != len(self.episodes):
            raise ValueError("feedback episode IDs must be unique")
        scripts = [
            item for item in self.episodes if item.kind is FeedbackEpisodeKind.SCRIPTED_KNOWN_ANSWER
        ]
        traces = [
            item for item in self.episodes if item.kind is FeedbackEpisodeKind.RESEARCHER_TRACE
        ]
        if len(scripts) != 6 or len(traces) != 3:
            raise ValueError("Phase 5 requires six known-answer scripts and three traces")
        counts = {
            action: sum(item.action is action for item in scripts) for action in FeedbackAction
        }
        if counts != {
            FeedbackAction.REFINE_CONTEXT: 3,
            FeedbackAction.REQUEST_MERGE_SPLIT: 3,
        }:
            raise ValueError("known-answer scripts require three episodes of each action")
        return self


class RevisionCallKind(StrEnum):
    CPU_REPROJECT = "cpu_reproject"
    GPU_RECONSTRUCT = "gpu_reconstruct"


class RevisionCallRecord(ImmutableRecord):
    """Auditable call lineage returned by the injected local revision runner."""

    call_artifact_hash: Sha256Digest
    call_id: Identifier
    condition: ConditionName
    kind: RevisionCallKind
    instruction_hash: Sha256Digest
    before_projection_hash: Sha256Digest
    after_projection_hash: Sha256Digest | None = None
    seed: int = Field(ge=0)
    allocated_gpu_seconds: float = Field(ge=0.0)
    started_at: AwareDatetime
    completed_at: AwareDatetime

    @model_validator(mode="after")
    def call_kind_matches_condition(self) -> Self:
        if self.completed_at < self.started_at:
            raise ValueError("revision call completion predates its start")
        if self.condition in ACTIVE_QUERY_CONSTRUCTION_CONDITIONS:
            if (
                self.kind is not RevisionCallKind.GPU_RECONSTRUCT
                or self.allocated_gpu_seconds <= 0.0
            ):
                raise ValueError("C2 revision calls require metered GPU reconstruction")
        elif self.condition in {
            ConditionName.C0_CLASSICAL_PRE,
            ConditionName.C1_LLM_PRE,
        }:
            if self.kind is not RevisionCallKind.CPU_REPROJECT or self.allocated_gpu_seconds != 0.0:
                raise ValueError("C0/C1 revision calls are CPU sealed-ontology reprojections")
        else:
            raise ValueError("the Phase 5 interface supports C0, C1, and C2 feedback only")
        return self


class RevisionExecutionResult(ImmutableRecord):
    instruction: RevisionInstruction
    resolution: FeedbackResolution
    before_bundle_hash: Sha256Digest
    call: RevisionCallRecord | None = None
    after_bundle: VisualizationBundle | None = None
    diff: VisualizationDiff | None = None
    latency_seconds: float = Field(ge=0.0)
    replay: FeedbackReplayResult | None = None

    @model_validator(mode="after")
    def execution_lineage_is_complete(self) -> Self:
        if self.resolution.revision_hash != self.instruction.revision.content_hash:
            raise ValueError("execution resolution does not reference its instruction")
        after_fields = (self.after_bundle, self.diff, self.replay)
        if any(item is None for item in after_fields) and any(
            item is not None for item in after_fields
        ):
            raise ValueError("after bundle, diff, and replay must be present or absent together")
        if (
            self.resolution.status is FeedbackResolutionStatus.RESOLVED
            and self.after_bundle is None
        ):
            raise ValueError("resolved revision execution requires its output and diff")
        if (
            self.resolution.status is FeedbackResolutionStatus.CAPABILITY_LIMITED
            and self.after_bundle is not None
        ):
            raise ValueError("capability-limited revision cannot claim an output")
        if self.after_bundle is not None:
            assert self.diff is not None
            assert self.replay is not None
            if self.after_bundle.condition is not self.resolution.receiving_condition:
                raise ValueError("execution output belongs to another condition")
            if self.resolution.after_projection_hash != self.after_bundle.projection_hash:
                raise ValueError("execution resolution and after bundle differ")
            if self.diff.after_projection_hash != self.after_bundle.projection_hash:
                raise ValueError("execution diff and after bundle differ")
            if self.diff.before_projection_hash != self.resolution.before_projection_hash:
                raise ValueError("execution diff and resolution before projections differ")
            if self.replay.observed_instruction_hash != self.instruction.content_hash:
                raise ValueError("execution replay and instruction differ")
            if self.replay.observed_resolution_hash != self.resolution.content_hash:
                raise ValueError("execution replay and resolution differ")
            if self.replay.observed_before_bundle_hash != self.before_bundle_hash:
                raise ValueError("execution replay and before bundle differ")
            if self.replay.observed_after_bundle_hash != self.after_bundle.content_hash:
                raise ValueError("execution replay and after bundle differ")
            if self.replay.observed_diff_hash != self.diff.content_hash:
                raise ValueError("execution replay and diff differ")
        condition = self.resolution.receiving_condition
        requires_call = self.resolution.status is not FeedbackResolutionStatus.CAPABILITY_LIMITED
        if requires_call and self.call is None:
            raise ValueError("executed revision requires an auditable condition call")
        if self.call is not None:
            if self.call.condition is not condition:
                raise ValueError("execution call belongs to another condition")
            if self.call.instruction_hash != self.instruction.content_hash:
                raise ValueError("execution call does not bind the instruction")
            if self.call.before_projection_hash != self.resolution.before_projection_hash:
                raise ValueError("execution call does not bind the before projection")
            observed_after_hash = (
                None if self.after_bundle is None else self.after_bundle.projection_hash
            )
            if self.call.after_projection_hash != observed_after_hash:
                raise ValueError("execution call does not bind the retained output")
            if self.call.seed != self.resolution.seed:
                raise ValueError("execution call and resolution seeds differ")
            if self.call.started_at <= self.instruction.revision.created_at:
                raise ValueError("revision execution must start after the request is recorded")
            if self.call.completed_at > self.resolution.resolved_at:
                raise ValueError("revision cannot resolve before its condition call completes")
            call_wall_seconds = (self.call.completed_at - self.call.started_at).total_seconds()
            if self.latency_seconds + 1e-9 < call_wall_seconds:
                raise ValueError("reported revision latency is shorter than its condition call")
        return self


class RevisionDraftSubmission(ImmutableRecord):
    before_projection_id: Identifier
    action: FeedbackAction
    anchors: tuple[FeedbackAnchor, ...]
    rationale: str = Field(min_length=1)
    sequence: int = Field(gt=0)
    seed: int = Field(ge=0)
    lens: str | None = Field(default=None, min_length=1)
    story_scope: StoryTime | None = None
    spoiler_horizon: SpoilerHorizon | None = None
    merge_split_operation: MergeSplitOperation | None = None
    grouped_mention_candidate_ids: tuple[tuple[Identifier, ...], ...] = ()

    @model_validator(mode="after")
    def submission_matches_action(self) -> Self:
        if not self.anchors:
            raise ValueError("revision submissions require feedback anchors")
        has_refinement = any(
            item is not None for item in (self.lens, self.story_scope, self.spoiler_horizon)
        )
        has_merge_split = self.merge_split_operation is not None or bool(
            self.grouped_mention_candidate_ids
        )
        if self.action is FeedbackAction.REFINE_CONTEXT:
            if not has_refinement or has_merge_split:
                raise ValueError("REFINE_CONTEXT requires only registered context fields")
        elif not has_merge_split or has_refinement:
            raise ValueError("REQUEST_MERGE_SPLIT requires only identity groups")
        if self.action is FeedbackAction.REQUEST_MERGE_SPLIT and (
            self.merge_split_operation is None or not self.grouped_mention_candidate_ids
        ):
            raise ValueError("merge/split operation and groups are jointly required")
        return self


class FilterSubmission(ImmutableRecord):
    temporal_filter: VisualizationTemporalFilter


class RevisionRunner(Protocol):
    def __call__(
        self,
        instruction: RevisionInstruction,
        before_bundle: VisualizationBundle,
        seed: int,
    ) -> RevisionExecutionResult: ...


class LocalUiRepository:
    """Small single-process store for immutable local demonstration artifacts."""

    def __init__(self, bundles: Sequence[VisualizationBundle] = ()) -> None:
        self._bundles: dict[str, VisualizationBundle] = {}
        self._evidence: dict[str, EvidenceMetadata] = {}
        for bundle in bundles:
            self.add_bundle(bundle)

    def add_bundle(self, bundle: VisualizationBundle) -> None:
        previous = self._bundles.get(bundle.projection_id)
        if previous is not None and previous.content_hash != bundle.content_hash:
            raise ValueError("a projection ID cannot be rebound to another visualization")
        self._bundles[bundle.projection_id] = bundle
        for item in bundle.evidence_metadata:
            existing = self._evidence.get(item.evidence_id)
            if existing is not None and existing.content_hash != item.content_hash:
                raise ValueError("an evidence ID cannot be rebound to different metadata")
            self._evidence[item.evidence_id] = item

    def bundle(self, projection_id: str) -> VisualizationBundle:
        try:
            return self._bundles[projection_id]
        except KeyError as error:
            raise KeyError(f"unknown projection {projection_id!r}") from error

    def projections(self) -> tuple[VisualizationBundle, ...]:
        return tuple(self._bundles[key] for key in sorted(self._bundles))

    def evidence(self, evidence_id: str, *, allow_restricted: bool) -> EvidenceMetadata:
        try:
            item = self._evidence[evidence_id]
        except KeyError as error:
            raise KeyError(f"unknown evidence {evidence_id!r}") from error
        if item.release_class is ReleaseClass.RESTRICTED and not allow_restricted:
            raise PermissionError("restricted evidence metadata is not authorized")
        return item


def _compile_submission(
    submission: RevisionDraftSubmission,
    before: VisualizationBundle,
    *,
    created_at: datetime,
) -> RevisionInstruction:
    revision_id = _identifier_digest(
        "revision",
        {
            "before_projection_hash": before.projection_hash,
            "action": submission.action,
            "sequence": submission.sequence,
            "created_at": created_at,
        },
    )
    if submission.action is FeedbackAction.REFINE_CONTEXT:
        intent = RefineContextIntent(
            after_context_id=f"{before.context.context_id}.revision-{submission.sequence}",
            after_revealed_at=created_at,
            lens=submission.lens,
            story_scope=submission.story_scope,
            spoiler_horizon=submission.spoiler_horizon,
        )
        return build_refine_context_instruction(
            revision_id=revision_id,
            before_context=before.context,
            intent=intent,
            anchors=submission.anchors,
            rationale=submission.rationale,
            sequence=submission.sequence,
            created_at=created_at,
        )
    assert submission.merge_split_operation is not None
    intent = MergeSplitIntent(
        operation=submission.merge_split_operation,
        grouped_mention_candidate_ids=submission.grouped_mention_candidate_ids,
    )
    return build_merge_split_instruction(
        revision_id=revision_id,
        context=before.context,
        intent=intent,
        anchors=submission.anchors,
        rationale=submission.rationale,
        sequence=submission.sequence,
        created_at=created_at,
    )


def _asset_status(static_directory: Path) -> dict[str, object]:
    manifest_path = static_directory / "cytoscape.vendor.json"
    asset_path = static_directory / "cytoscape.min.js"
    lock_path = static_directory / "cytoscape.lock.json"
    license_path = static_directory / "CYTOSCAPE_LICENSE"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    status: dict[str, object] = {
        "package": manifest["package"],
        "version": manifest["version"],
        "vendored": False,
        "verified": False,
    }
    if not asset_path.is_file() or not lock_path.is_file() or not license_path.is_file():
        return status
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    asset = asset_path.read_bytes()
    actual = hashlib.sha256(asset).hexdigest()
    byte_count = len(asset)
    license_bytes = license_path.read_bytes()
    license_sha256 = hashlib.sha256(license_bytes).hexdigest()
    status["vendored"] = True
    status["verified"] = (
        lock.get("package") == manifest["package"]
        and lock.get("version") == manifest["version"]
        and lock.get("source") == manifest["source"]
        and lock.get("sha256") == actual
        and manifest.get("sha256") == actual
        and lock.get("bytes") == byte_count
        and manifest.get("bytes") == byte_count
        and lock.get("license_file") == manifest.get("license_file") == license_path.name
        and lock.get("license_sha256") == license_sha256
        and manifest.get("license_sha256") == license_sha256
        and lock.get("license_bytes") == len(license_bytes)
        and manifest.get("license_bytes") == len(license_bytes)
    )
    status["sha256"] = actual
    status["bytes"] = byte_count
    return status


def create_app(
    repository: LocalUiRepository,
    *,
    revision_runner: RevisionRunner | None = None,
    revision_seed: int | None = None,
    static_directory: Path | None = None,
    allow_restricted_evidence_metadata: bool = False,
    clock: Callable[[], datetime] | None = None,
    researcher_trace_episode_by_projection_id: Mapping[str, str] | None = None,
    researcher_trace_capture_sink: (
        Callable[[ImmutableRecord, ResearcherTraceSubmissionReceipt], None] | None
    ) = None,
):
    """Create the local single-user FastAPI application.

    A runner is injected explicitly.  Without one, ordinary revision submission returns
    503 rather than pretending that a C2 regeneration occurred.  A projection explicitly
    registered as one of the three researcher traces instead takes the capture-only path:
    the exact instruction and endpoint receipt are durably stored and no runner is called.
    """

    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import FileResponse
        from fastapi.staticfiles import StaticFiles
    except ImportError as error:  # pragma: no cover - exercised only in minimal installs
        raise RuntimeError("the optional 'study' dependencies are required for the UI") from error

    if revision_seed is not None and revision_seed < 0:
        raise ValueError("the configured revision seed must be nonnegative")
    trace_episodes = dict(researcher_trace_episode_by_projection_id or {})
    if bool(trace_episodes) != (researcher_trace_capture_sink is not None):
        raise ValueError(
            "researcher-trace projection mapping and receipt sink must be configured together"
        )
    if len(trace_episodes.values()) != len(set(trace_episodes.values())):
        raise ValueError("each researcher-trace episode must bind one before projection")
    ui_directory = static_directory or Path(__file__).resolve().parents[2] / "ui"
    now = clock or (lambda: datetime.now(UTC))
    app = FastAPI(
        title="StoryProjectionOnto local study interface",
        docs_url=None,
        redoc_url=None,
    )
    app.mount("/static", StaticFiles(directory=ui_directory), name="static")

    @app.get("/")
    def index():
        return FileResponse(ui_directory / "index.html")

    @app.get("/api/health")
    def health():
        return {
            "status": "ok",
            "scope": "local_single_user_study_demonstration",
            "usability_claims": False,
            "revision_actions": [item.value for item in FeedbackAction],
            # Decimal text avoids losing the registered 63-bit seed in JavaScript.
            "revision_seed_decimal": (None if revision_seed is None else str(revision_seed)),
            "cytoscape": _asset_status(ui_directory),
        }

    @app.get("/api/projections")
    def projections():
        return [
            {
                "projection_id": item.projection_id,
                "condition": item.condition.value,
                "context_id": item.context.context_id,
                "projection_hash": item.projection_hash,
            }
            for item in repository.projections()
            if allow_restricted_evidence_metadata
            or item.release_class is not ReleaseClass.RESTRICTED
        ]

    @app.get("/api/projections/{projection_id}")
    def projection(projection_id: str):
        try:
            bundle = repository.bundle(projection_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        if (
            bundle.release_class is ReleaseClass.RESTRICTED
            and not allow_restricted_evidence_metadata
        ):
            raise HTTPException(
                status_code=403,
                detail="restricted projection metadata is not authorized",
            )
        return bundle

    @app.post("/api/projections/{projection_id}/filter")
    def filter_projection(projection_id: str, submission: FilterSubmission):
        try:
            bundle = repository.bundle(projection_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        if (
            bundle.release_class is ReleaseClass.RESTRICTED
            and not allow_restricted_evidence_metadata
        ):
            raise HTTPException(
                status_code=403,
                detail="restricted projection metadata is not authorized",
            )
        return filter_visualization_bundle(bundle, submission.temporal_filter)

    @app.get("/api/evidence/{evidence_id}")
    def evidence(evidence_id: str):
        try:
            return repository.evidence(
                evidence_id,
                allow_restricted=allow_restricted_evidence_metadata,
            )
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except PermissionError as error:
            raise HTTPException(status_code=403, detail=str(error)) from error

    @app.get("/api/diffs/{before_projection_id}/{after_projection_id}")
    def diff(before_projection_id: str, after_projection_id: str):
        try:
            before = repository.bundle(before_projection_id)
            after = repository.bundle(after_projection_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        if not allow_restricted_evidence_metadata and any(
            item.release_class is ReleaseClass.RESTRICTED for item in (before, after)
        ):
            raise HTTPException(
                status_code=403,
                detail="restricted projection metadata is not authorized",
            )
        return compare_visualizations(before, after)

    @app.post("/api/revisions")
    def submit_revision(submission: RevisionDraftSubmission):
        try:
            before = repository.bundle(submission.before_projection_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        if (
            before.release_class is ReleaseClass.RESTRICTED
            and not allow_restricted_evidence_metadata
        ):
            raise HTTPException(
                status_code=403,
                detail="restricted projection metadata is not authorized",
            )
        requested_at = now()
        instruction = _compile_submission(
            submission,
            before,
            created_at=requested_at,
        )
        if revision_seed is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "no frozen feedback seed is configured; no regeneration was started "
                    "and no regeneration was claimed"
                ),
            )
        if submission.seed != revision_seed:
            raise HTTPException(
                status_code=422,
                detail="submitted seed differs from the frozen feedback seed",
            )
        trace_episode_id = trace_episodes.get(submission.before_projection_id)
        if trace_episode_id is not None:
            if before.condition is not ConditionName.C2_LLM_QUERY:
                raise HTTPException(
                    status_code=422,
                    detail="researcher traces require a C2 before projection",
                )
            assert researcher_trace_capture_sink is not None
            instruction_bytes = (instruction.to_canonical_json() + "\n").encode("utf-8")
            submission_json = submission.to_canonical_json()
            submission_bytes = (submission_json + "\n").encode("utf-8")
            receipt = ResearcherTraceSubmissionReceipt(
                receipt_id=(
                    "trace-receipt-"
                    + canonical_sha256(
                        {
                            "episode_id": trace_episode_id,
                            "submission_hash": submission.content_hash,
                            "instruction_hash": instruction.content_hash,
                        }
                    )[:24]
                ),
                episode_id=trace_episode_id,
                action=submission.action,
                requested_at=requested_at,
                before_projection_id=before.projection_id,
                before_projection_hash=before.projection_hash,
                before_bundle_hash=before.content_hash,
                submission_hash=submission.content_hash,
                submission_canonical_json=submission_json,
                submission_file_sha256=hashlib.sha256(submission_bytes).hexdigest(),
                instruction_hash=instruction.content_hash,
                instruction_file_sha256=hashlib.sha256(instruction_bytes).hexdigest(),
                recorded_at=requested_at,
            )
            try:
                researcher_trace_capture_sink(instruction, receipt)
            except (OSError, ValueError) as error:
                raise HTTPException(
                    status_code=500,
                    detail=f"researcher-trace receipt was not durably recorded: {error}",
                ) from error
            return {
                "capture_only": True,
                "episode_id": trace_episode_id,
                "instruction": instruction,
                "submission_receipt": receipt,
                "regeneration_started": False,
            }
        if revision_runner is None:
            raise HTTPException(
                status_code=503,
                detail="no metered condition runner is configured; no regeneration was claimed",
            )
        result = revision_runner(instruction, before, submission.seed)
        if result.instruction.content_hash != instruction.content_hash:
            raise HTTPException(
                status_code=500,
                detail="revision runner returned a different instruction",
            )
        if result.before_bundle_hash != before.content_hash:
            raise HTTPException(
                status_code=500,
                detail="revision runner returned lineage for a different before projection",
            )
        if result.resolution.receiving_condition is not before.condition:
            raise HTTPException(
                status_code=500,
                detail="revision runner returned another condition's resolution",
            )
        if result.resolution.before_projection_hash != before.projection_hash:
            raise HTTPException(
                status_code=500,
                detail="revision runner resolved a different before projection",
            )
        if (
            result.after_bundle is not None
            and result.after_bundle.condition is not before.condition
        ):
            raise HTTPException(
                status_code=500,
                detail="revision runner returned another condition's projection",
            )
        if result.resolution.seed != submission.seed:
            raise HTTPException(
                status_code=500,
                detail="revision runner changed the submitted seed",
            )
        if (
            result.after_bundle is not None
            and result.after_bundle.release_class is ReleaseClass.RESTRICTED
            and not allow_restricted_evidence_metadata
        ):
            raise HTTPException(
                status_code=403,
                detail="revision runner returned restricted metadata without authorization",
            )
        if result.after_bundle is not None:
            repository.add_bundle(result.after_bundle)
        return result

    return app


__all__ = [
    "DISPLAY_SELECTION_RULE_REVISION",
    "EvidenceMetadata",
    "FeedbackEpisodeKind",
    "FeedbackEpisodePlan",
    "FeedbackReplayExpectation",
    "FeedbackReplayResult",
    "FeedbackStudyPlan",
    "FilterSubmission",
    "LocalUiRepository",
    "MergeSplitIntent",
    "MergeSplitOperation",
    "RefineContextIntent",
    "RevisionCallKind",
    "RevisionCallRecord",
    "RevisionDraftSubmission",
    "RevisionExecutionResult",
    "RevisionInstruction",
    "SemanticAssessmentSourceKind",
    "SemanticDisplayMode",
    "SemanticSupportStatus",
    "VisualizationAssertionDetail",
    "VisualizationAssertionSemanticAssessment",
    "VisualizationBundle",
    "VisualizationChange",
    "VisualizationChangeKind",
    "VisualizationCompilationError",
    "VisualizationConfiguration",
    "VisualizationContentScope",
    "VisualizationDisplaySelection",
    "VisualizationNodeDetail",
    "VisualizationObjectKind",
    "VisualizationSemanticOverlay",
    "apply_context_refinement",
    "assert_revision_anchors_resolve_in_packet",
    "assert_revision_has_no_projection_local_anchors",
    "build_merge_split_instruction",
    "build_refine_context_instruction",
    "build_visualization_bundle",
    "compare_visualizations",
    "create_app",
    "filter_visualization_bundle",
    "load_visualization_configuration",
    "resolve_feedback_for_condition",
    "verify_feedback_replay",
]
