"""Scorer-only orchestration for the registered synthetic Phase 4 analysis.

This module is the sole bridge from frozen, gold-free execution journals to the
scorer namespace.  It never participates in request construction.  Every intended
primary cell is materialized before output availability is inspected, terminal
failures receive the registered intention-to-treat rows, and inference is performed
only after the 12-world aggregation.

Geometry is deliberately an input, not something guessed by the scorer.  Successful
content-bearing projections require a :class:`GeometryMetricInput` captured with the
frozen renderer and named by projection content hash.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Self

from pydantic import AwareDatetime, Field, model_validator

from story_projection_onto.analysis import (
    CrossingObservation,
    analysis_to_json,
    analyze_paired_worlds,
    compare_crossing_profiles,
    run_registered_analysis_from_observations,
)
from story_projection_onto.combined_gpu_block import (
    CombinedCallClass,
    CombinedCallManifest,
)
from story_projection_onto.combined_gpu_production import (
    CombinedExecutionIndex,
    CombinedITTRecord,
    CombinedServiceCallResult,
)
from story_projection_onto.conditions.base import ConditionAttemptRecord
from story_projection_onto.contracts import (
    BenchmarkSplit,
    ConditionName,
    EvidencePacket,
    ImmutableRecord,
    OntologyProjection,
    ReleaseClass,
    RunOutcome,
    Sha256Digest,
    canonical_json,
    canonical_sha256,
)
from story_projection_onto.fallback_acceptance import validate_source_association
from story_projection_onto.held_out_controller import (
    HeldOutExecutionManifest,
    ScorerBridgeAuthorization,
)
from story_projection_onto.held_out_primary import HeldOutCallManifest
from story_projection_onto.independent_review_runtime import load_completed_review
from story_projection_onto.metrics.alignment import (
    GroundingStatus,
    build_alignment_plan,
    prediction_records,
    score_alignment,
)
from story_projection_onto.metrics.config import StudyMetricConfiguration
from story_projection_onto.metrics.contrastive import (
    ParaphraseStabilityScore,
    derive_signed_changes,
    score_paraphrase_stability,
)
from story_projection_onto.metrics.decision_states import compile_gold_decision_targets
from story_projection_onto.metrics.pipeline import (
    CompleteProjectionScoreBundle,
    ContrastMetricPlan,
    ContrastPairScore,
    CrossSeedCommunityScore,
    FailedMetricOutput,
    GeometryMetricInput,
    GroundingAuditInput,
    IntendedMetricManifest,
    IntendedMetricUnit,
    IntendedUnitScore,
    MetricResultRow,
    OutputFailureKind,
    ScorerMetricPlan,
    analysis_observations_from_scores,
    score_contrast_pair,
    score_cross_seed_community,
    score_failed_output,
    score_intended_projection,
)
from story_projection_onto.metrics.rare import annotations_from_gold
from story_projection_onto.store import ArtifactStore, BlobStore, Ledger, MetricStatus
from story_projection_onto.synthetic_benchmark import (
    ScorerWorldArtifact,
    compile_alignment_alternatives,
    semantic_atom_to_normalized_decision,
)

REGISTERED_PRIMARY_SCORE_COUNT = 252
REGISTERED_COMBINED_ORDINARY_SCORE_COUNT = 40
REGISTERED_CONTRAST_SCORE_COUNT = 84
REGISTERED_CROSS_SEED_COMMUNITY_COUNT = 108
REGISTERED_PARAPHRASE_PAIR_COUNT = 12
REGISTERED_ABLATION_PAIR_COUNT = 28
REGISTERED_WORLD_COUNT = 12

_PRIMARY_CONDITIONS = (
    ConditionName.C0_CLASSICAL_PRE,
    ConditionName.C1_LLM_PRE,
    ConditionName.C2_LLM_QUERY,
    ConditionName.A_FIXED_SELECT,
)
_ABLATION_CONDITIONS = (
    ConditionName.A_NO_CONTEXT,
    ConditionName.A_NO_TEMPORAL_EPISTEMIC,
    ConditionName.A_NO_RARE_GUARD,
)
_NON_FEEDBACK_COMBINED_CLASSES = frozenset(
    {
        CombinedCallClass.PARAPHRASE_C2,
        CombinedCallClass.ABLATION_NO_CONTEXT,
        CombinedCallClass.ABLATION_NO_TEMPORAL_EPISTEMIC,
        CombinedCallClass.ABLATION_NO_RARE_GUARD,
    }
)


class Phase4AnalysisError(RuntimeError):
    """A scorer input or immutable output failed a registered integrity check."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_file(path: Path, *, maximum_bytes: int = 100 * 1024 * 1024) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise Phase4AnalysisError(f"missing or symlinked scorer input: {path}")
    size = path.stat().st_size
    if size > maximum_bytes:
        raise Phase4AnalysisError(f"scorer input exceeds {maximum_bytes} bytes: {path}")
    return path.read_bytes()


def _load_model(path: Path, model: type[ImmutableRecord]):
    try:
        return model.model_validate_json(_regular_file(path))
    except Phase4AnalysisError:
        raise
    except Exception as error:
        raise Phase4AnalysisError(f"invalid {model.__name__} at {path}: {error}") from error


class Phase4AnalysisConfiguration(ImmutableRecord):
    configuration_id: Literal["conference-phase4-analysis-v1"]
    metric_configuration_path: str
    metric_configuration_file_sha256: Sha256Digest
    methodological_plan_path: str
    methodological_plan_file_sha256: Sha256Digest
    implementation_plan_path: str
    implementation_plan_file_sha256: Sha256Digest
    seed_manifest_path: str
    seed_manifest_file_sha256: Sha256Digest
    seed_manifest_hash: Sha256Digest
    bootstrap_seed_entry_hash: Sha256Digest
    bootstrap_root_seed: int = Field(ge=0, lt=2**63)
    geometry_filename_rule: Literal["projection-content-hash.json"]
    require_geometry_for_valid_content: Literal[True] = True
    primary_score_count: Literal[252] = REGISTERED_PRIMARY_SCORE_COUNT
    combined_ordinary_score_count: Literal[40] = REGISTERED_COMBINED_ORDINARY_SCORE_COUNT
    contrast_score_count: Literal[84] = REGISTERED_CONTRAST_SCORE_COUNT
    cross_seed_community_count: Literal[108] = REGISTERED_CROSS_SEED_COMMUNITY_COUNT
    paraphrase_pair_count: Literal[12] = REGISTERED_PARAPHRASE_PAIR_COUNT
    ablation_pair_count: Literal[28] = REGISTERED_ABLATION_PAIR_COUNT
    independent_unit: Literal["world"] = "world"
    registered_world_count: Literal[12] = REGISTERED_WORLD_COUNT
    registered_contexts_per_world: Literal[3] = 3
    registered_llm_seeds: tuple[Literal[1], Literal[2]] = (1, 2)
    registered_sign_flip_assignments: Literal[4096] = 4096
    registered_bootstrap_resamples: Literal[10000] = 10_000
    rare_pivotal_noninferiority_margin: Literal[-0.05] = -0.05

    @model_validator(mode="after")
    def paths_are_portable(self) -> Self:
        for value in (
            self.metric_configuration_path,
            self.methodological_plan_path,
            self.implementation_plan_path,
            self.seed_manifest_path,
        ):
            path = PurePosixPath(value)
            if path.is_absolute() or ".." in path.parts or "\\" in value:
                raise ValueError("Phase 4 configuration paths must be bounded and portable")
        return self

    @classmethod
    def load(cls, repository: Path, path: Path) -> Phase4AnalysisConfiguration:
        repository = repository.resolve(strict=True)
        configuration = cls.model_validate_json(_regular_file(path))
        for relative, expected in (
            (
                configuration.metric_configuration_path,
                configuration.metric_configuration_file_sha256,
            ),
            (configuration.methodological_plan_path, configuration.methodological_plan_file_sha256),
            (configuration.implementation_plan_path, configuration.implementation_plan_file_sha256),
            (configuration.seed_manifest_path, configuration.seed_manifest_file_sha256),
        ):
            observed = _sha256_file(repository / relative)
            if observed != expected:
                raise Phase4AnalysisError(f"frozen Phase 4 input changed: {relative}")
        seed_manifest = json.loads(
            _regular_file(repository / configuration.seed_manifest_path)
        )
        if canonical_sha256(seed_manifest) != configuration.seed_manifest_hash:
            raise Phase4AnalysisError("seed manifest logical hash changed")
        bootstrap = tuple(
            item for item in seed_manifest.get("entries", ()) if item.get("purpose") == "bootstrap"
        )
        if len(bootstrap) != 1 or (
            bootstrap[0].get("content_hash") != configuration.bootstrap_seed_entry_hash
            or canonical_sha256(bootstrap[0]) != configuration.bootstrap_seed_entry_hash
            or bootstrap[0].get("seed") != configuration.bootstrap_root_seed
        ):
            raise Phase4AnalysisError("Phase 4 bootstrap seed binding changed")
        return configuration


class Phase4SourceBinding(ImmutableRecord):
    role: str = Field(min_length=1)
    file_sha256: Sha256Digest
    logical_content_hash: Sha256Digest


class ScoredMetricObservation(ImmutableRecord):
    source_block: Literal["primary", "combined"]
    unit_id: str
    condition: ConditionName
    world_id: str
    context_id: str
    seed_block: int | None
    output_valid: bool
    metric: MetricResultRow


class ContrastScoreObservation(ImmutableRecord):
    world_id: str
    before_context_id: str
    after_context_id: str
    condition: ConditionName
    seed_block: int | None = None
    score: ContrastPairScore

    @model_validator(mode="after")
    def identity_matches_score(self) -> Self:
        if self.condition is not self.score.condition or self.seed_block != self.score.seed_block:
            raise ValueError("contrast observation identity differs from its score")
        return self


class CrossSeedScoreObservation(ImmutableRecord):
    world_id: str
    context_id: str
    condition: ConditionName
    score: CrossSeedCommunityScore

    @model_validator(mode="after")
    def identity_matches_score(self) -> Self:
        if (
            self.world_id != self.score.world_id
            or self.context_id != self.score.context_id
            or self.condition is not self.score.condition
        ):
            raise ValueError("cross-seed observation identity differs from its score")
        return self


class WorldMetricRow(ImmutableRecord):
    condition: ConditionName
    world_id: str
    metric_name: str
    value: float | None
    context_count: int
    row_count: int
    undefined_context_count: int


class ParaphrasePairResult(ImmutableRecord):
    call_id: str
    world_id: str
    context_id: str
    base_score_hash: Sha256Digest
    paraphrase_score_hash: Sha256Digest
    base_projection_hash: Sha256Digest | None = None
    paraphrase_projection_hash: Sha256Digest | None = None
    status: Literal["value", "invalid"]
    score: ParaphraseStabilityScore | None = None

    @model_validator(mode="after")
    def score_matches_status(self) -> Self:
        if (self.status == "value") != (self.score is not None):
            raise ValueError("paraphrase score and status differ")
        return self


class AblationPairResult(ImmutableRecord):
    call_id: str
    condition: Literal[
        ConditionName.A_NO_CONTEXT,
        ConditionName.A_NO_TEMPORAL_EPISTEMIC,
        ConditionName.A_NO_RARE_GUARD,
    ]
    world_id: str
    context_id: str
    parent_score_hash: Sha256Digest
    ablation_score_hash: Sha256Digest
    principal_metric: str
    parent_value: float | None
    ablation_value: float | None
    difference: float | None

    @model_validator(mode="after")
    def difference_is_exact(self) -> Self:
        expected = (
            None
            if self.parent_value is None or self.ablation_value is None
            else self.ablation_value - self.parent_value
        )
        if self.difference != expected:
            raise ValueError("ablation difference is not ablation minus parent C2")
        return self


class ExploratoryComparison(ImmutableRecord):
    metric_name: str
    comparison: str
    status: Literal["estimated", "not_estimated_incomplete_world_panel"]
    retained_world_count: int = Field(ge=0, le=12)
    estimate: Mapping[str, Any] | None = None

    @model_validator(mode="after")
    def estimate_matches_status(self) -> Self:
        if (self.status == "estimated") != (self.estimate is not None):
            raise ValueError("exploratory estimate and status differ")
        return self


class SecondaryMultiplicityResult(ImmutableRecord):
    family: Literal["entropy-two-sided"] = "entropy-two-sided"
    metric_name: str
    comparison: str
    family_planned_size: Literal[10] = 10
    status: Literal["adjusted", "not_estimated_incomplete_world_panel"]
    raw_two_sided_p_value: float | None = Field(default=None, ge=0.0, le=1.0)
    benjamini_hochberg_adjusted_p_value: float | None = Field(
        default=None, ge=0.0, le=1.0
    )

    @model_validator(mode="after")
    def p_values_match_status(self) -> Self:
        present = (
            self.raw_two_sided_p_value is not None
            and self.benjamini_hochberg_adjusted_p_value is not None
        )
        if (self.status == "adjusted") != present:
            raise ValueError("secondary multiplicity values differ from status")
        return self


class EligibleWorldComparison(ImmutableRecord):
    panel: Literal["gold_aligned_community", "cross_seed_community"]
    metric_name: str
    comparison: str
    eligible_world_ids: tuple[str, ...]
    status: Literal["estimated", "not_estimated_fewer_than_two_worlds"]
    estimate: Mapping[str, Any] | None = None

    @model_validator(mode="after")
    def eligible_panel_is_explicit(self) -> Self:
        if self.eligible_world_ids != tuple(sorted(set(self.eligible_world_ids))):
            raise ValueError("eligible world IDs must be sorted and unique")
        if (self.status == "estimated") != (self.estimate is not None):
            raise ValueError("eligible-world estimate differs from status")
        if self.estimate is not None and len(self.eligible_world_ids) < 2:
            raise ValueError("eligible-world inference requires at least two worlds")
        return self


class CanonicalComparisonRow(ImmutableRecord):
    comparison: str
    metric: str
    estimate: float | None = None
    paired_median: float | None = None
    standardized_effect: float | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None
    p_value: float | None = Field(default=None, ge=0.0, le=1.0)
    adjusted_p_value: float | None = Field(default=None, ge=0.0, le=1.0)
    sign_flip_p_value: float | None = Field(default=None, ge=0.0, le=1.0)
    bootstrap_lower: float | None = None
    bootstrap_upper: float | None = None
    paired_world_count: int = Field(ge=0, le=12)
    status: str
    analysis_family: str
    p_value_kind: str
    multiplicity_method: str
    independent_unit: Literal["world"] = "world"


class Phase4ReportGateStatus(ImmutableRecord):
    gate_status_id: str
    registered_analysis_hash: Sha256Digest
    alpha: Literal[0.05] = 0.05
    rare_safeguard_margin: Literal[-0.05] = -0.05
    organization_adjusted_p_value: float = Field(ge=0.0, le=1.0)
    mechanism_one_sided_p_value: float | None = Field(default=None, ge=0.0, le=1.0)
    c2_vs_c1_rare_lower_bound: float
    c2_vs_fixed_rare_lower_bound: float | None = None
    c2_vs_fixed_rare_p_value: float | None = Field(default=None, ge=0.0, le=1.0)
    organization_gate_passed: bool
    construction_freedom_gate_passed: bool
    c2_vs_c1_rare_safeguard_gate_passed: bool
    c2_vs_fixed_rare_safeguard_status: Literal["passed", "failed", "not_tested"]
    c2_vs_fixed_rare_safeguard_gate_passed: bool

    @model_validator(mode="after")
    def gates_are_mechanical(self) -> Self:
        if self.organization_gate_passed != (
            self.organization_adjusted_p_value <= self.alpha
        ):
            raise ValueError("organization gate must derive from its Holm-adjusted p-value")
        if not self.organization_gate_passed:
            if any(
                value is not None
                for value in (
                    self.mechanism_one_sided_p_value,
                    self.c2_vs_fixed_rare_lower_bound,
                    self.c2_vs_fixed_rare_p_value,
                )
            ):
                raise ValueError("gated mechanism values cannot exist before organization passes")
        elif any(
            value is None
            for value in (
                self.mechanism_one_sided_p_value,
                self.c2_vs_fixed_rare_lower_bound,
                self.c2_vs_fixed_rare_p_value,
            )
        ):
            raise ValueError("organization gate requires complete mechanism safeguards")
        expected_construction = bool(
            self.organization_gate_passed
            and self.mechanism_one_sided_p_value is not None
            and self.mechanism_one_sided_p_value <= self.alpha
        )
        if self.construction_freedom_gate_passed != expected_construction:
            raise ValueError("construction-freedom gate must derive from the mechanism test")
        c2_c1_passes = self.c2_vs_c1_rare_lower_bound > self.rare_safeguard_margin and not (
            math.isclose(
                self.c2_vs_c1_rare_lower_bound,
                self.rare_safeguard_margin,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
        )
        if self.c2_vs_c1_rare_safeguard_gate_passed != c2_c1_passes:
            raise ValueError("C2-C1 rare safeguard must derive from the registered margin")
        fixed_lower_bound = self.c2_vs_fixed_rare_lower_bound
        fixed_tested = fixed_lower_bound is not None
        expected_fixed_pass = bool(
            fixed_lower_bound is not None
            and fixed_lower_bound > self.rare_safeguard_margin
            and not math.isclose(
                fixed_lower_bound,
                self.rare_safeguard_margin,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
        )
        expected_status = (
            "passed" if expected_fixed_pass else ("failed" if fixed_tested else "not_tested")
        )
        if (
            self.c2_vs_fixed_rare_safeguard_gate_passed != expected_fixed_pass
            or self.c2_vs_fixed_rare_safeguard_status != expected_status
            or fixed_tested != (self.c2_vs_fixed_rare_p_value is not None)
        ):
            raise ValueError("C2-Fixed rare safeguard must derive from its registered test")
        return self


class Phase4OutputFile(ImmutableRecord):
    relative_path: str
    file_sha256: Sha256Digest
    logical_content_hash: Sha256Digest | None = None
    row_count: int | None = Field(default=None, ge=0)
    release_class: Literal[ReleaseClass.PUBLIC, ReleaseClass.RESTRICTED]

    @model_validator(mode="after")
    def relative_path_is_bounded(self) -> Self:
        path = PurePosixPath(self.relative_path)
        if path.is_absolute() or ".." in path.parts or "\\" in self.relative_path:
            raise ValueError("Phase 4 output path must be bounded and portable")
        return self


class Phase4TableEntry(ImmutableRecord):
    table_id: str = Field(min_length=1)
    relative_path: str
    columns: tuple[str, ...] = Field(min_length=1)
    row_count: int = Field(ge=0)
    file_sha256: Sha256Digest
    logical_content_hash: Sha256Digest
    release_class: Literal[ReleaseClass.PUBLIC, ReleaseClass.RESTRICTED]
    independent_unit: Literal[
        "intended_cell",
        "metric_observation",
        "world",
        "contrast_pair",
        "cross_seed_pair",
        "paraphrase_pair",
        "ablation_pair",
        "analysis",
    ]
    selection_ready: bool = False

    @model_validator(mode="after")
    def table_contract_is_canonical(self) -> Self:
        path = PurePosixPath(self.relative_path)
        if path.is_absolute() or ".." in path.parts or "\\" in self.relative_path:
            raise ValueError("Phase 4 table path must be bounded and portable")
        if len(self.columns) != len(set(self.columns)):
            raise ValueError("Phase 4 table columns must be unique")
        return self


class Phase4TableManifest(ImmutableRecord):
    manifest_id: str = Field(min_length=1)
    analysis_configuration_hash: Sha256Digest
    metric_version_hash: Sha256Digest
    independent_confirmatory_unit: Literal["world"] = "world"
    tables: tuple[Phase4TableEntry, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def tables_are_canonical(self) -> Self:
        if self.tables != tuple(sorted(self.tables, key=lambda item: item.table_id)):
            raise ValueError("Phase 4 table entries must be sorted by table ID")
        identifiers = tuple(item.table_id for item in self.tables)
        paths = tuple(item.relative_path for item in self.tables)
        if len(identifiers) != len(set(identifiers)) or len(paths) != len(set(paths)):
            raise ValueError("Phase 4 table manifest contains duplicate IDs or paths")
        return self


class Phase4AnalysisIndex(ImmutableRecord):
    run_id: str
    analysis_configuration_hash: Sha256Digest
    metric_configuration_hash: Sha256Digest
    metric_version_hash: Sha256Digest
    source_tree_association_file_sha256: Sha256Digest
    source_tree_association_hash: Sha256Digest
    source_bindings: tuple[Phase4SourceBinding, ...]
    review_completion_manifest_hash: Sha256Digest
    final_reviewed_seal_hash: Sha256Digest
    primary_intended_manifest_hash: Sha256Digest
    combined_intended_manifest_hash: Sha256Digest
    table_manifest_file_sha256: Sha256Digest
    table_manifest_hash: Sha256Digest
    registered_analysis_file_sha256: Sha256Digest
    registered_analysis_hash: Sha256Digest
    report_gate_status_file_sha256: Sha256Digest
    report_gate_status_hash: Sha256Digest
    output_files: tuple[Phase4OutputFile, ...]
    completed_at: AwareDatetime
    primary_score_count: Literal[252] = 252
    combined_ordinary_score_count: Literal[40] = 40
    contrast_score_count: Literal[84] = 84
    cross_seed_community_count: Literal[108] = 108
    paraphrase_pair_count: Literal[12] = 12
    ablation_pair_count: Literal[28] = 28
    registered_world_count: Literal[12] = 12
    registered_sign_flip_assignments: Literal[4096] = 4096
    registered_bootstrap_resamples: Literal[10000] = 10_000
    all_failures_included_in_intention_to_treat: Literal[True] = True
    scorer_namespace: Literal["scorer_only"] = "scorer_only"
    model_runtime_reopened: Literal[False] = False

    @model_validator(mode="after")
    def inventory_is_canonical(self) -> Self:
        if self.source_bindings != tuple(sorted(self.source_bindings, key=lambda item: item.role)):
            raise ValueError("source bindings must be sorted")
        if self.output_files != tuple(
            sorted(self.output_files, key=lambda item: item.relative_path)
        ):
            raise ValueError("output files must be sorted")
        paths = tuple(item.relative_path for item in self.output_files)
        if len(paths) != len(set(paths)):
            raise ValueError("output file inventory contains duplicates")
        return self


@dataclass(frozen=True, slots=True)
class _ScorerContext:
    world: ScorerWorldArtifact
    gold: Any
    alternatives: Any
    plan: ScorerMetricPlan


@dataclass(frozen=True, slots=True)
class _Cell:
    source_block: Literal["primary", "combined"]
    intended: IntendedMetricUnit
    scorer_plan: ScorerMetricPlan
    outcome: RunOutcome
    projection: OntologyProjection | None
    failure_artifact_hash: str | None


@dataclass(frozen=True, slots=True)
class _ScoreCollection:
    scores: tuple[IntendedUnitScore, ...]
    bundles: Mapping[str, CompleteProjectionScoreBundle]
    projections: Mapping[str, OntologyProjection]
    grounding_audits: Mapping[str, GroundingAuditInput]
    geometries: Mapping[str, GeometryMetricInput]
    observations: tuple[ScoredMetricObservation, ...]


def _failure_kind(outcome: RunOutcome) -> OutputFailureKind:
    return {
        RunOutcome.TIMED_OUT: OutputFailureKind.TIMEOUT,
        RunOutcome.INVALID: OutputFailureKind.VALIDATION_INVALID,
        RunOutcome.INTERRUPTED: OutputFailureKind.EXECUTION_FAILURE,
        RunOutcome.FAILED: OutputFailureKind.EXECUTION_FAILURE,
    }.get(outcome, OutputFailureKind.EXECUTION_FAILURE)


def _artifact_model(
    artifacts: ArtifactStore,
    *,
    artifact_hash: str,
    logical_hash: str | None,
    model: type[ImmutableRecord],
):
    try:
        record = artifacts.ledger.get_artifact(artifact_hash)
        payload = artifacts.blobs.read_bytes(record, allow_restricted=True)
        value = model.model_validate_json(payload)
    except Exception as error:
        raise Phase4AnalysisError(
            f"cannot resolve {model.__name__} artifact {artifact_hash}: {error}"
        ) from error
    if logical_hash is not None and value.content_hash != logical_hash:
        raise Phase4AnalysisError(
            f"{model.__name__} logical hash differs from its immutable pointer"
        )
    return value


def _scorer_plan(
    *,
    world: ScorerWorldArtifact,
    gold: Any,
    alternatives: Any,
    valid_evidence_ids: tuple[str, ...],
    configuration: StudyMetricConfiguration,
) -> ScorerMetricPlan:
    if valid_evidence_ids != tuple(sorted(set(valid_evidence_ids))):
        raise Phase4AnalysisError("valid evidence IDs are not canonical")
    try:
        alignment = build_alignment_plan(
            gold,
            alternatives,
            compiled_alternatives=compile_alignment_alternatives(gold, alternatives),
        )
        target_decisions = tuple(
            sorted(
                (
                    semantic_atom_to_normalized_decision(item)
                    for item in world.semantic_atoms_by_query[gold.query_id]
                ),
                key=lambda item: item.slot_key,
            )
        )
        return ScorerMetricPlan(
            plan_id=f"phase4-{world.world_spec.world_id}-{gold.query_id}",
            source_gold_hash=gold.content_hash,
            gold_projection=gold,
            alignment_plan=alignment,
            gold_decisions=target_decisions,
            rare_annotations=tuple(
                sorted(
                    annotations_from_gold(gold),
                    key=lambda item: item.assertion_target_id,
                )
            ),
            gold_community_assignments=tuple(
                sorted((item.anchor_id, item.community_id) for item in gold.communities)
            ),
            community_eligible=bool(gold.communities),
            valid_evidence_ids=valid_evidence_ids,
            valid_evidence_manifest_hash=canonical_sha256(valid_evidence_ids),
            predicate_aliases=tuple(sorted(configuration.upper_relation_mapping)),
        )
    except Exception as error:
        raise Phase4AnalysisError(
            f"cannot compile scorer plan for {world.world_spec.world_id}/{gold.query_id}: {error}"
        ) from error


def _scorer_sources(
    *,
    benchmark_root: Path,
    review_root: Path,
) -> tuple[dict[str, ScorerWorldArtifact], Mapping[tuple[str, str], tuple[Any, Any]], Any]:
    review = load_completed_review(benchmark_root=benchmark_root, output_root=review_root)
    worlds: dict[str, ScorerWorldArtifact] = {}
    held_out_root = benchmark_root / "scorer_only" / "held_out"
    for path in sorted(held_out_root.glob("syn-test-*.json")):
        world = _load_model(path, ScorerWorldArtifact)
        if world.world_spec.split is not BenchmarkSplit.HELD_OUT:
            raise Phase4AnalysisError(f"non-held-out scorer artifact in held-out tree: {path}")
        if world.world_spec.world_id in worlds:
            raise Phase4AnalysisError("duplicate held-out scorer world")
        worlds[world.world_spec.world_id] = world
    if len(worlds) != REGISTERED_WORLD_COUNT:
        raise Phase4AnalysisError("Phase 4 requires exactly twelve held-out scorer worlds")
    reviewed: dict[tuple[str, str], tuple[Any, Any]] = {}
    for artifact in review.reviewed_artifacts:
        key = (artifact.gold_projection.world_id, artifact.gold_projection.query_id)
        if key in reviewed:
            raise Phase4AnalysisError("review completion repeats a scorer context")
        reviewed[key] = (artifact.gold_projection, artifact.alternatives)
    return worlds, reviewed, review


def _gold_pair(
    world: ScorerWorldArtifact,
    query_id: str,
    reviewed: Mapping[tuple[str, str], tuple[Any, Any]],
) -> tuple[Any, Any]:
    source_gold = next((item for item in world.gold_projections if item.query_id == query_id), None)
    if source_gold is None:
        raise Phase4AnalysisError(f"unknown scorer query {world.world_spec.world_id}/{query_id}")
    source_alternatives = next(
        (
            item
            for item in world.alternatives
            if item.gold_projection_id == source_gold.gold_projection_id
        ),
        None,
    )
    if source_alternatives is None:
        raise Phase4AnalysisError("scorer context lacks its permissible alternatives")
    return reviewed.get((world.world_spec.world_id, query_id), (source_gold, source_alternatives))


def _world_by_query(worlds: Mapping[str, ScorerWorldArtifact]) -> dict[str, ScorerWorldArtifact]:
    result: dict[str, ScorerWorldArtifact] = {}
    for world in worlds.values():
        for gold in world.gold_projections:
            if gold.query_id in result:
                raise Phase4AnalysisError("held-out scorer query IDs are not globally unique")
            result[gold.query_id] = world
    if len(result) != 36:
        raise Phase4AnalysisError("held-out scorer catalog requires exactly 36 contexts")
    return result


def _intended(
    *,
    unit_id: str,
    job_id: str,
    condition: ConditionName,
    world_id: str,
    context_id: str,
    seed_block: int | None,
    snapshot_hash: str,
    packet_hash: str,
    context_hash: str,
    plan: ScorerMetricPlan,
) -> IntendedMetricUnit:
    return IntendedMetricUnit(
        unit_id=unit_id,
        job_id=job_id,
        condition=condition,
        world_id=world_id,
        context_id=context_id,
        seed_block=seed_block,
        snapshot_hash=snapshot_hash,
        packet_hash=packet_hash,
        context_hash=context_hash,
        scorer_plan_hash=plan.content_hash,
        relevant_node_gold_count=sum(
            item.is_contextually_relevant for item in plan.alignment_plan.node_targets
        ),
        strict_assertion_gold_count=len(plan.alignment_plan.assertion_targets),
        ontology_decision_gold_count=len(plan.gold_decisions),
        rare_pivotal_gold_count=sum(
            item.is_rare and item.is_pivotal for item in plan.rare_annotations
        ),
    )


def _verify_scorer_bridge(
    *,
    execution: HeldOutExecutionManifest,
    call_manifest: HeldOutCallManifest,
    bridge: ScorerBridgeAuthorization,
) -> None:
    if (
        execution.call_manifest_hash != call_manifest.content_hash
        or bridge.execution_manifest_hash != execution.content_hash
        or bridge.call_manifest_hash != call_manifest.content_hash
        or bridge.final_reviewed_seal_hash != execution.final_reviewed_seal_hash
        or bridge.itt_record_hashes
        != tuple(item.content_hash for item in execution.itt_records)
        or bridge.projection_receipt_hashes
        != tuple(item.content_hash for item in execution.preconstructed_projections)
        or bridge.runtime_namespace_closed is not True
        or bridge.model_input_open is not False
    ):
        raise Phase4AnalysisError("held-out scorer bridge does not reproduce the frozen execution")
    expected_outputs = tuple(
        item.result.output_artifact_hash
        for item in execution.itt_records
        if item.result.output_artifact_hash is not None
    ) + tuple(
        item.projection_artifact_hash
        for item in execution.preconstructed_projections
        if item.projection_artifact_hash is not None
    )
    if bridge.output_artifact_hashes != expected_outputs:
        raise Phase4AnalysisError("held-out scorer bridge output inventory changed")


def _primary_cells(
    *,
    call_manifest: HeldOutCallManifest,
    execution: HeldOutExecutionManifest,
    artifacts: ArtifactStore,
    worlds: Mapping[str, ScorerWorldArtifact],
    reviewed: Mapping[tuple[str, str], tuple[Any, Any]],
    metric_configuration: StudyMetricConfiguration,
) -> tuple[tuple[_Cell, ...], Mapping[str, _ScorerContext]]:
    query_world = _world_by_query(worlds)
    openings = {item.sealed_stage_hash: item for item in execution.query_openings}
    if len(openings) != 36:
        raise Phase4AnalysisError("held-out execution lacks 36 unique query openings")
    projection_receipts = {
        (item.unit_id, item.query_stage_hash, item.condition, item.seed_block): item
        for item in execution.preconstructed_projections
    }
    if len(projection_receipts) != len(execution.preconstructed_projections):
        raise Phase4AnalysisError("preconstructed projection receipts repeat a scorer cell")
    call_records = {
        (call.unit_id, call.query_stage_hash, call.condition, call.seed_block): record
        for call, record in zip(call_manifest.calls, execution.itt_records, strict=True)
        if call.query_stage_hash is not None
    }
    query_time_call_count = sum(
        call.query_stage_hash is not None for call in call_manifest.calls
    )
    if len(call_records) != query_time_call_count:
        raise Phase4AnalysisError("query-time ITT records repeat a scorer cell")
    contexts: dict[str, _ScorerContext] = {}
    cells: list[_Cell] = []
    for unit in call_manifest.units:
        for stage in unit.query_stages:
            opening = openings.get(stage.staging_manifest_hash)
            if opening is None or opening.query_context is None:
                raise Phase4AnalysisError("held-out query opening lacks its typed context")
            context = opening.query_context
            world = query_world.get(context.context_id)
            if world is None:
                raise Phase4AnalysisError("held-out context is absent from scorer gold")
            packet = _artifact_model(
                artifacts,
                artifact_hash=opening.evidence_packet_artifact.artifact_hash,
                logical_hash=opening.evidence_packet_hash,
                model=EvidencePacket,
            )
            valid_evidence_ids = tuple(sorted(packet.ordered_evidence_ids))
            gold, alternatives = _gold_pair(world, context.context_id, reviewed)
            plan = _scorer_plan(
                world=world,
                gold=gold,
                alternatives=alternatives,
                valid_evidence_ids=valid_evidence_ids,
                configuration=metric_configuration,
            )
            prior = contexts.setdefault(
                context.context_id,
                _ScorerContext(world=world, gold=gold, alternatives=alternatives, plan=plan),
            )
            if prior.plan != plan:
                raise Phase4AnalysisError("one held-out context compiled multiple scorer plans")
            for condition, seeds in (
                (ConditionName.C0_CLASSICAL_PRE, (None,)),
                (ConditionName.C1_LLM_PRE, (1, 2)),
                (ConditionName.C2_LLM_QUERY, (1, 2)),
                (ConditionName.A_FIXED_SELECT, (1, 2)),
            ):
                for seed in seeds:
                    key = (unit.unit_id, stage.staging_manifest_hash, condition, seed)
                    projection = None
                    failure_hash = None
                    if condition in {ConditionName.C0_CLASSICAL_PRE, ConditionName.C1_LLM_PRE}:
                        receipt = projection_receipts.get(key)
                        if receipt is None:
                            raise Phase4AnalysisError("preconstructed projection receipt is absent")
                        outcome = receipt.outcome
                        failure_hash = receipt.failure_artifact_hash
                        if outcome is RunOutcome.SUCCEEDED:
                            assert receipt.projection_artifact_hash is not None
                            projection = _artifact_model(
                                artifacts,
                                artifact_hash=receipt.projection_artifact_hash,
                                logical_hash=None,
                                model=OntologyProjection,
                            )
                    else:
                        record = call_records.get(key)
                        if record is None:
                            raise Phase4AnalysisError("query-time ITT record is absent")
                        result = record.result
                        outcome = result.outcome
                        failure_hash = (
                            result.artifact_receipt.raw_response.artifact_hash
                            if result.artifact_receipt is not None
                            and result.artifact_receipt.raw_response is not None
                            else None
                        )
                        if outcome is RunOutcome.SUCCEEDED:
                            reference = (
                                None
                                if result.artifact_receipt is None
                                else result.artifact_receipt.output
                            )
                            if reference is None:
                                raise Phase4AnalysisError("successful query-time output lacks CAS")
                            attempt = _artifact_model(
                                artifacts,
                                artifact_hash=reference.artifact_hash,
                                logical_hash=reference.logical_content_hash,
                                model=ConditionAttemptRecord,
                            )
                            projection = attempt.projection
                            if attempt.outcome is not outcome or projection is None:
                                raise Phase4AnalysisError(
                                    "query-time attempt differs from ITT result"
                                )
                    intended = _intended(
                        unit_id=(
                            f"primary:{unit.unit_id}:{stage.stage_id}:{condition.value}:"
                            f"s{seed if seed is not None else 0}"
                        ),
                        job_id=(
                            f"phase4-primary-{unit.unit_id}-{stage.stage_id}-"
                            f"{condition.value}-s{seed if seed is not None else 0}"
                        ),
                        condition=condition,
                        world_id=world.world_spec.world_id,
                        context_id=context.context_id,
                        seed_block=seed,
                        snapshot_hash=stage.snapshot_hash,
                        packet_hash=opening.evidence_packet_hash,
                        context_hash=context.content_hash,
                        plan=plan,
                    )
                    cells.append(
                        _Cell(
                            source_block="primary",
                            intended=intended,
                            scorer_plan=plan,
                            outcome=outcome,
                            projection=projection,
                            failure_artifact_hash=failure_hash,
                        )
                    )
    cells.sort(key=lambda item: item.intended.unit_id)
    if len(cells) != REGISTERED_PRIMARY_SCORE_COUNT:
        raise Phase4AnalysisError("primary intended metric inventory does not contain 252 cells")
    return tuple(cells), contexts


def _combined_cells(
    *,
    combined_root: Path,
    manifest: CombinedCallManifest,
    execution_index: CombinedExecutionIndex,
    artifacts: ArtifactStore,
    scorer_contexts: Mapping[str, _ScorerContext],
    metric_configuration: StudyMetricConfiguration,
) -> tuple[_Cell, ...]:
    if (
        execution_index.manifest_hash != manifest.content_hash
        or execution_index.base_call_count != 49
        or execution_index.ordinary_call_count != 40
        or not execution_index.vllm_service_stopped
    ):
        raise Phase4AnalysisError("combined execution index differs from its frozen manifest")
    ordinary_calls = tuple(
        item for item in manifest.calls if item.call_class in _NON_FEEDBACK_COMBINED_CLASSES
    )
    if len(ordinary_calls) != REGISTERED_COMBINED_ORDINARY_SCORE_COUNT:
        raise Phase4AnalysisError("combined manifest lacks its forty scorer-facing calls")
    cells: list[_Cell] = []
    observed_itt_hashes = []
    for call in ordinary_calls:
        prefix = f"{call.ordinal:03d}-{call.call_id}"
        result = _load_model(
            combined_root / "calls" / f"{prefix}.result.json",
            CombinedServiceCallResult,
        )
        itt = _load_model(
            combined_root / "calls" / f"{prefix}.itt.json",
            CombinedITTRecord,
        )
        if (
            result.call_id != call.call_id
            or result.call_spec_hash != call.content_hash
            or result.condition is not call.condition
            or itt.call_id != call.call_id
            or itt.call_spec_hash != call.content_hash
            or itt.service_result_hash != result.content_hash
            or itt.condition is not call.condition
            or itt.outcome is not result.outcome
            or itt.condition_attempt_hash
            != result.condition_attempt_artifact.logical_content_hash
            or not itt.included_in_intention_to_treat
        ):
            raise Phase4AnalysisError(f"combined result lineage changed for {call.call_id}")
        observed_itt_hashes.append(itt.content_hash)
        scorer = scorer_contexts.get(call.context_id)
        if scorer is None:
            raise Phase4AnalysisError("combined call context is absent from primary scorer plans")
        packet_pointer = call.source.packet_artifact
        packet = _artifact_model(
            artifacts,
            artifact_hash=packet_pointer.artifact_hash,
            logical_hash=packet_pointer.logical_content_hash,
            model=EvidencePacket,
        )
        plan = _scorer_plan(
            world=scorer.world,
            gold=scorer.gold,
            alternatives=scorer.alternatives,
            valid_evidence_ids=tuple(sorted(packet.ordered_evidence_ids)),
            configuration=metric_configuration,
        )
        if plan != scorer.plan:
            raise Phase4AnalysisError("combined and primary packets compile different scorer plans")
        projection = None
        failure_hash = None
        if result.outcome is RunOutcome.SUCCEEDED:
            pointer = result.condition_attempt_artifact
            attempt = _artifact_model(
                artifacts,
                artifact_hash=pointer.artifact_hash,
                logical_hash=pointer.logical_content_hash,
                model=ConditionAttemptRecord,
            )
            if (
                attempt.outcome is not result.outcome
                or attempt.condition is not call.condition
                or attempt.projection is None
            ):
                raise Phase4AnalysisError("combined condition attempt differs from service result")
            projection = attempt.projection
        else:
            failure_hash = result.condition_attempt_artifact.artifact_hash
        effective_context = call.paraphrase_context or call.source.context
        intended = _intended(
            unit_id=f"combined:{call.call_id}",
            job_id=f"phase4-combined-{call.call_id}",
            condition=call.condition,
            world_id=scorer.world.world_spec.world_id,
            context_id=call.context_id,
            seed_block=call.seed_block,
            snapshot_hash=call.source.prequery_stage.snapshot_hash,
            packet_hash=call.source.packet_hash,
            context_hash=effective_context.content_hash,
            plan=plan,
        )
        cells.append(
            _Cell(
                source_block="combined",
                intended=intended,
                scorer_plan=plan,
                outcome=result.outcome,
                projection=projection,
                failure_artifact_hash=failure_hash,
            )
        )
    if tuple(observed_itt_hashes) != execution_index.ordinary_itt_record_hashes:
        raise Phase4AnalysisError("combined ordinary ITT inventory differs from execution index")
    cells.sort(key=lambda item: item.intended.unit_id)
    return tuple(cells)


def _grounding_audit(
    projection: OntologyProjection,
    plan: ScorerMetricPlan,
) -> GroundingAuditInput:
    """Classify exact gold-supported assertions without trusting model support flags.

    A provisional all-supported pass identifies assertions whose aligned participants,
    relation, qualification, and cited evidence match a permissible gold target.  Only
    those predictions receive ``SUPPORTED``; other emitted factual assertions receive
    ``UNSUPPORTED``.  This is scorer-side known-answer auditing, never prompt material.
    """

    valid_evidence = frozenset(plan.valid_evidence_ids)
    provisional = {
        item.assertion_id: (
            GroundingStatus.SUPPORTED
            if valid_evidence.intersection(item.evidence_ids)
            else GroundingStatus.UNSUPPORTED
        )
        for item in projection.instance_graph.assertions
    }
    predicted_nodes, predicted_assertions, _ = prediction_records(
        projection,
        predicate_aliases=dict(plan.predicate_aliases),
        valid_evidence_ids=valid_evidence,
        grounding_by_assertion_id=provisional,
        plan=plan.alignment_plan,
    )
    preliminary = score_alignment(
        plan=plan.alignment_plan,
        predicted_nodes=predicted_nodes,
        predicted_assertions=predicted_assertions,
    )
    supported = {
        item.prediction_id for item in preliminary.structurally_aligned_assertion_matches
    }
    statuses = tuple(
        sorted(
            (
                assertion.assertion_id,
                (
                    GroundingStatus.SUPPORTED
                    if assertion.assertion_id in supported
                    else GroundingStatus.UNSUPPORTED
                ),
            )
            for assertion in projection.instance_graph.assertions
        )
    )
    audit_payload = {
        "audit_revision": "known-answer-structural-support-v1",
        "projection_hash": projection.content_hash,
        "valid_evidence_manifest_hash": plan.valid_evidence_manifest_hash,
        "assertion_statuses": tuple((key, value.value) for key, value in statuses),
    }
    return GroundingAuditInput(
        projection_hash=projection.content_hash,
        valid_evidence_manifest_hash=plan.valid_evidence_manifest_hash,
        assertion_statuses=statuses,
        audit_artifact_hash=canonical_sha256(audit_payload),
    )


def _geometry(
    projection: OntologyProjection,
    *,
    geometry_root: Path,
    required: bool,
) -> GeometryMetricInput | None:
    path = geometry_root / f"{projection.content_hash}.json"
    if not path.is_file():
        if required:
            raise Phase4AnalysisError(
                "missing frozen renderer geometry for valid projection: "
                f"{projection.content_hash} (expected {path})"
            )
        return None
    geometry = _load_model(path, GeometryMetricInput)
    if geometry.projection_hash != projection.content_hash:
        raise Phase4AnalysisError("renderer geometry names another projection")
    return geometry


def _score_cells(
    cells: Sequence[_Cell],
    *,
    configuration: StudyMetricConfiguration,
    geometry_root: Path,
    require_geometry: bool,
) -> _ScoreCollection:
    scores = []
    bundles: dict[str, CompleteProjectionScoreBundle] = {}
    projections: dict[str, OntologyProjection] = {}
    grounding_audits: dict[str, GroundingAuditInput] = {}
    geometries: dict[str, GeometryMetricInput] = {}
    observations = []
    for cell in cells:
        intended = cell.intended
        if cell.outcome is RunOutcome.SUCCEEDED:
            if cell.projection is None:
                raise Phase4AnalysisError("successful intended cell lacks a projection")
            projection = cell.projection
            structurally_valid = bool(projection.validation_records) and all(
                item.validation_status.value == "accepted"
                for item in projection.validation_records
            )
            geometry = _geometry(
                projection,
                geometry_root=geometry_root,
                required=require_geometry and structurally_valid,
            )
            audit = _grounding_audit(projection, cell.scorer_plan)
            previous_audit = grounding_audits.setdefault(projection.content_hash, audit)
            if previous_audit != audit:
                raise Phase4AnalysisError("one projection produced multiple grounding audits")
            if geometry is not None:
                previous_geometry = geometries.setdefault(projection.content_hash, geometry)
                if previous_geometry != geometry:
                    raise Phase4AnalysisError("one projection resolved multiple geometry inputs")
            score = score_intended_projection(
                intended,
                projection,
                configuration=configuration,
                scorer_plan=cell.scorer_plan,
                grounding_audit=audit,
                geometry=geometry,
            )
            if score.output_valid:
                # Recompute once through the public complete-panel function so paired
                # metrics retain exact partition/alignment objects, not reconstructed rows.
                from story_projection_onto.metrics.pipeline import score_complete_projection

                bundle = score_complete_projection(
                    projection,
                    configuration=configuration,
                    scorer_plan=cell.scorer_plan,
                    grounding_audit=audit,
                    geometry=geometry,
                )
                if score.projection_bundle_hash != bundle.content_hash:
                    raise Phase4AnalysisError("score and complete metric bundle hashes differ")
                bundles[intended.content_hash] = bundle
                projections[intended.content_hash] = projection
        else:
            if cell.projection is not None:
                raise Phase4AnalysisError("failed intended cell retained a projection")
            score = score_failed_output(
                FailedMetricOutput(
                    intended_unit=intended,
                    failure_kind=_failure_kind(cell.outcome),
                    failure_artifact_hash=cell.failure_artifact_hash,
                ),
                configuration=configuration,
                scorer_plan=cell.scorer_plan,
            )
        scores.append(score)
        observations.extend(
            ScoredMetricObservation(
                source_block=cell.source_block,
                unit_id=intended.unit_id,
                condition=intended.condition,
                world_id=intended.world_id,
                context_id=intended.context_id,
                seed_block=intended.seed_block,
                output_valid=score.output_valid,
                metric=row,
            )
            for row in score.rows
        )
    return _ScoreCollection(
        scores=tuple(scores),
        bundles=bundles,
        projections=projections,
        grounding_audits=grounding_audits,
        geometries=geometries,
        observations=tuple(observations),
    )


def _row_value(score: IntendedUnitScore, metric_name: str) -> float | None:
    rows = tuple(item for item in score.rows if item.metric_name == metric_name)
    if len(rows) != 1:
        raise Phase4AnalysisError(
            f"score {score.intended_unit.unit_id} has {len(rows)} rows for {metric_name}"
        )
    value = rows[0].value
    return None if value is None else float(value)


def _contrast_plan(
    before: _ScorerContext,
    after: _ScorerContext,
) -> ContrastMetricPlan:
    before_decisions, _ = compile_gold_decision_targets(
        before.gold,
        before.plan.gold_decisions,
        predicate_aliases=dict(before.plan.predicate_aliases),
    )
    after_decisions, _ = compile_gold_decision_targets(
        after.gold,
        after.plan.gold_decisions,
        predicate_aliases=dict(after.plan.predicate_aliases),
    )
    changes = tuple(
        sorted(
            derive_signed_changes(before_decisions, after_decisions),
            key=canonical_json,
        )
    )
    return ContrastMetricPlan(
        plan_id=f"contrast-{before.world.world_spec.world_id}",
        world_id=before.world.world_spec.world_id,
        before_context_id=before.gold.query_id,
        after_context_id=after.gold.query_id,
        before_scorer_plan_hash=before.plan.content_hash,
        after_scorer_plan_hash=after.plan.content_hash,
        gold_changes=changes,
        invariants=tuple(
            sorted(before.gold.contrast_invariants, key=lambda item: item.invariant_id)
        ),
    )


def _contrast_scores(
    *,
    primary: _ScoreCollection,
    scorer_contexts: Mapping[str, _ScorerContext],
    configuration: StudyMetricConfiguration,
) -> tuple[ContrastScoreObservation, ...]:
    by_cell = {
        (
            item.intended_unit.world_id,
            item.intended_unit.context_id,
            item.intended_unit.condition,
            item.intended_unit.seed_block,
        ): item
        for item in primary.scores
    }
    results = []
    worlds = {item.world.world_spec.world_id: item.world for item in scorer_contexts.values()}
    for world_id, world in sorted(worlds.items()):
        before_id = world.gold_projections[0].query_id
        after_id = world.gold_projections[1].query_id
        before = scorer_contexts[before_id]
        after = scorer_contexts[after_id]
        plan = _contrast_plan(before, after)
        for condition, seeds in (
            (ConditionName.C0_CLASSICAL_PRE, (None,)),
            (ConditionName.C1_LLM_PRE, (1, 2)),
            (ConditionName.C2_LLM_QUERY, (1, 2)),
            (ConditionName.A_FIXED_SELECT, (1, 2)),
        ):
            for seed in seeds:
                first = by_cell[(world_id, before_id, condition, seed)]
                second = by_cell[(world_id, after_id, condition, seed)]
                first_hash = first.intended_unit.content_hash
                second_hash = second.intended_unit.content_hash
                score = score_contrast_pair(
                    first,
                    second,
                    before_bundle=primary.bundles.get(first_hash),
                    after_bundle=primary.bundles.get(second_hash),
                    before_projection=primary.projections.get(first_hash),
                    after_projection=primary.projections.get(second_hash),
                    before_scorer_plan=before.plan,
                    after_scorer_plan=after.plan,
                    contrast_plan=plan,
                    configuration=configuration,
                )
                results.append(
                    ContrastScoreObservation(
                        world_id=world_id,
                        before_context_id=before_id,
                        after_context_id=after_id,
                        condition=condition,
                        seed_block=seed,
                        score=score,
                    )
                )
    if len(results) != REGISTERED_CONTRAST_SCORE_COUNT:
        raise Phase4AnalysisError("contrast inventory does not contain 84 paired scores")
    return tuple(results)


def _cross_seed_scores(
    *,
    primary: _ScoreCollection,
    scorer_contexts: Mapping[str, _ScorerContext],
    configuration: StudyMetricConfiguration,
) -> tuple[CrossSeedScoreObservation, ...]:
    by_cell = {
        (
            item.intended_unit.context_id,
            item.intended_unit.condition,
            item.intended_unit.seed_block,
        ): item
        for item in primary.scores
    }
    results = []
    for context_id, scorer in sorted(scorer_contexts.items()):
        for condition in (
            ConditionName.C1_LLM_PRE,
            ConditionName.C2_LLM_QUERY,
            ConditionName.A_FIXED_SELECT,
        ):
            first = by_cell[(context_id, condition, 1)]
            second = by_cell[(context_id, condition, 2)]
            first_hash = first.intended_unit.content_hash
            second_hash = second.intended_unit.content_hash
            score = score_cross_seed_community(
                first,
                second,
                first_bundle=primary.bundles.get(first_hash),
                second_bundle=primary.bundles.get(second_hash),
                configuration=configuration,
                scorer_plan=scorer.plan,
            )
            results.append(
                CrossSeedScoreObservation(
                    world_id=score.world_id,
                    context_id=score.context_id,
                    condition=score.condition,
                    score=score,
                )
            )
    if len(results) != REGISTERED_CROSS_SEED_COMMUNITY_COUNT:
        raise Phase4AnalysisError("cross-seed community inventory does not contain 108 scores")
    return tuple(results)


def _assertion_signature(bundle: CompleteProjectionScoreBundle | None) -> frozenset[str]:
    if bundle is None:
        return frozenset()
    return frozenset(
        item.semantic_signature for item in bundle.projection.adapter.assertion_semantics
    )


def _paraphrase_results(
    *,
    primary: _ScoreCollection,
    combined: _ScoreCollection,
    combined_manifest: CombinedCallManifest,
) -> tuple[ParaphrasePairResult, ...]:
    primary_by_context = {
        item.intended_unit.context_id: item
        for item in primary.scores
        if item.intended_unit.condition is ConditionName.C2_LLM_QUERY
        and item.intended_unit.seed_block == 1
    }
    combined_by_call = {
        item.intended_unit.unit_id.removeprefix("combined:"): item
        for item in combined.scores
    }
    results = []
    for call in combined_manifest.calls:
        if call.call_class is not CombinedCallClass.PARAPHRASE_C2:
            continue
        base = primary_by_context.get(call.context_id)
        paraphrase = combined_by_call.get(call.call_id)
        if base is None or paraphrase is None:
            raise Phase4AnalysisError("paraphrase pair lacks its primary or rewritten score")
        base_hash = base.intended_unit.content_hash
        paraphrase_hash = paraphrase.intended_unit.content_hash
        score = None
        status: Literal["value", "invalid"] = "invalid"
        if base.output_valid and paraphrase.output_valid:
            base_f1 = _row_value(base, "strict_qualified_assertion_f1")
            paraphrase_f1 = _row_value(paraphrase, "strict_qualified_assertion_f1")
            if base_f1 is None or paraphrase_f1 is None:
                raise Phase4AnalysisError("valid paraphrase pair has undefined strict F1")
            score = score_paraphrase_stability(
                base_signature=_assertion_signature(primary.bundles.get(base_hash)),
                paraphrase_signature=_assertion_signature(
                    combined.bundles.get(paraphrase_hash)
                ),
                base_strict_f1=base_f1,
                paraphrase_strict_f1=paraphrase_f1,
            )
            status = "value"
        results.append(
            ParaphrasePairResult(
                call_id=call.call_id,
                world_id=base.intended_unit.world_id,
                context_id=call.context_id,
                base_score_hash=base.content_hash,
                paraphrase_score_hash=paraphrase.content_hash,
                base_projection_hash=(
                    primary.projections[base_hash].content_hash
                    if base_hash in primary.projections
                    else None
                ),
                paraphrase_projection_hash=(
                    combined.projections[paraphrase_hash].content_hash
                    if paraphrase_hash in combined.projections
                    else None
                ),
                status=status,
                score=score,
            )
        )
    if len(results) != REGISTERED_PARAPHRASE_PAIR_COUNT:
        raise Phase4AnalysisError("paraphrase inventory does not contain twelve pairs")
    return tuple(results)


def _ablation_results(
    *,
    primary: _ScoreCollection,
    combined: _ScoreCollection,
    combined_manifest: CombinedCallManifest,
) -> tuple[AblationPairResult, ...]:
    primary_by_context = {
        item.intended_unit.context_id: item
        for item in primary.scores
        if item.intended_unit.condition is ConditionName.C2_LLM_QUERY
        and item.intended_unit.seed_block == 1
    }
    combined_by_call = {
        item.intended_unit.unit_id.removeprefix("combined:"): item
        for item in combined.scores
    }
    principal = {
        ConditionName.A_NO_CONTEXT: "ontology_decision_macro_f1",
        ConditionName.A_NO_TEMPORAL_EPISTEMIC: (
            "essential_temporal_qualification_accuracy"
        ),
        ConditionName.A_NO_RARE_GUARD: "rare_pivotal_qualified_assertion_recall",
    }
    results = []
    for call in combined_manifest.calls:
        if call.condition not in _ABLATION_CONDITIONS:
            continue
        parent = primary_by_context.get(call.context_id)
        ablation = combined_by_call.get(call.call_id)
        if parent is None or ablation is None:
            raise Phase4AnalysisError("ablation pair lacks its seed-1 C2 parent")
        metric = principal[call.condition]
        parent_value = _row_value(parent, metric)
        ablation_value = _row_value(ablation, metric)
        results.append(
            AblationPairResult(
                call_id=call.call_id,
                condition=call.condition,
                world_id=parent.intended_unit.world_id,
                context_id=call.context_id,
                parent_score_hash=parent.content_hash,
                ablation_score_hash=ablation.content_hash,
                principal_metric=metric,
                parent_value=parent_value,
                ablation_value=ablation_value,
                difference=(
                    None
                    if parent_value is None or ablation_value is None
                    else ablation_value - parent_value
                ),
            )
        )
    if len(results) != REGISTERED_ABLATION_PAIR_COUNT:
        raise Phase4AnalysisError("ablation inventory does not contain 28 pairs")
    if Counter(item.condition for item in results) != Counter(
        {
            ConditionName.A_NO_CONTEXT: 12,
            ConditionName.A_NO_TEMPORAL_EPISTEMIC: 8,
            ConditionName.A_NO_RARE_GUARD: 8,
        }
    ):
        raise Phase4AnalysisError("ablation condition counts changed")
    return tuple(results)


def _world_metric_rows(primary: _ScoreCollection) -> tuple[WorldMetricRow, ...]:
    """Two-stage aggregation for every primary metric, retaining structural NA."""

    metric_names = tuple(
        sorted({row.metric.metric_name for row in primary.observations})
    )
    by_key: dict[
        tuple[str, ConditionName, str], list[ScoredMetricObservation]
    ] = defaultdict(list)
    for row in primary.observations:
        by_key[(row.metric.metric_name, row.condition, row.world_id)].append(row)
    results = []
    for metric_name in metric_names:
        for condition in _PRIMARY_CONDITIONS:
            worlds = sorted(
                world_id
                for name, candidate, world_id in by_key
                if name == metric_name and candidate is condition
            )
            if len(worlds) != REGISTERED_WORLD_COUNT:
                raise Phase4AnalysisError(
                    f"{metric_name}/{condition.value} lacks twelve world panels"
                )
            for world_id in worlds:
                rows = by_key[(metric_name, condition, world_id)]
                by_context: dict[str, list[ScoredMetricObservation]] = defaultdict(list)
                for row in rows:
                    by_context[row.context_id].append(row)
                if len(by_context) != 3:
                    raise Phase4AnalysisError(
                        f"{metric_name}/{condition.value}/{world_id} lacks three contexts"
                    )
                context_values = []
                undefined = 0
                eligible_context_count = 0
                for context_id, context_rows in sorted(by_context.items()):
                    seeds = tuple(
                        sorted(
                            item.seed_block
                            for item in context_rows
                            if item.seed_block is not None
                        )
                    )
                    if condition is ConditionName.C0_CLASSICAL_PRE:
                        valid_shape = len(context_rows) == 1 and seeds == ()
                    else:
                        valid_shape = len(context_rows) == 2 and seeds == (1, 2)
                    if not valid_shape:
                        raise Phase4AnalysisError(
                            f"{metric_name}/{condition.value}/{world_id}/{context_id} "
                            "has an invalid seed panel"
                        )
                    if metric_name.startswith("community_"):
                        not_applicable = tuple(
                            item.metric.status.value == "not_applicable"
                            for item in context_rows
                        )
                        if any(not_applicable) and not all(not_applicable):
                            raise Phase4AnalysisError(
                                f"{metric_name}/{condition.value}/{world_id}/{context_id} "
                                "mixes eligible and ineligible community rows"
                            )
                        if all(not_applicable):
                            continue
                    eligible_context_count += 1
                    values = [item.metric.value for item in context_rows]
                    if any(value is None for value in values):
                        undefined += 1
                    else:
                        context_values.append(sum(float(value) for value in values) / len(values))
                value = (
                    None
                    if undefined or not eligible_context_count
                    else sum(context_values) / len(context_values)
                )
                results.append(
                    WorldMetricRow(
                        condition=condition,
                        world_id=world_id,
                        metric_name=metric_name,
                        value=value,
                        context_count=eligible_context_count,
                        row_count=len(rows),
                        undefined_context_count=undefined,
                    )
                )
    return tuple(
        sorted(
            results,
            key=lambda item: (item.metric_name, item.condition.value, item.world_id),
        )
    )


def _contrast_world_metric_rows(
    observations: Sequence[ContrastScoreObservation],
) -> tuple[WorldMetricRow, ...]:
    grouped: dict[tuple[str, ConditionName, str], list[MetricResultRow]] = defaultdict(list)
    for observation in observations:
        for row in observation.score.metric_rows:
            grouped[(row.metric_name, observation.condition, observation.world_id)].append(row)
    results = []
    for (metric_name, condition, world_id), rows in sorted(
        grouped.items(), key=lambda item: (item[0][0], item[0][1].value, item[0][2])
    ):
        expected = 1 if condition is ConditionName.C0_CLASSICAL_PRE else 2
        if len(rows) != expected:
            raise Phase4AnalysisError(
                f"contrast {metric_name}/{condition.value}/{world_id} has {len(rows)} "
                f"rows; expected {expected}"
            )
        eligible = tuple(row for row in rows if row.status.value != "not_applicable")
        values = tuple(row.value for row in eligible)
        value = (
            None
            if not eligible or any(item is None for item in values)
            else sum(float(item) for item in values) / len(values)
        )
        results.append(
            WorldMetricRow(
                condition=condition,
                world_id=world_id,
                metric_name=metric_name,
                value=value,
                context_count=1,
                row_count=len(rows),
                undefined_context_count=int(value is None),
            )
        )
    for metric_name in {item.metric_name for item in results}:
        for condition in _PRIMARY_CONDITIONS:
            if sum(
                item.metric_name == metric_name and item.condition is condition
                for item in results
            ) != REGISTERED_WORLD_COUNT:
                raise Phase4AnalysisError(
                    f"contrast {metric_name}/{condition.value} lacks twelve world rows"
                )
    return tuple(results)


def _cross_seed_world_metric_rows(
    observations: Sequence[CrossSeedScoreObservation],
) -> tuple[WorldMetricRow, ...]:
    grouped: dict[tuple[str, ConditionName, str], list[MetricResultRow]] = defaultdict(list)
    for observation in observations:
        for row in observation.score.metric_rows:
            grouped[(row.metric_name, observation.condition, observation.world_id)].append(row)
    results = []
    for (metric_name, condition, world_id), rows in sorted(
        grouped.items(), key=lambda item: (item[0][0], item[0][1].value, item[0][2])
    ):
        if len(rows) != 3:
            raise Phase4AnalysisError(
                f"cross-seed {metric_name}/{condition.value}/{world_id} must have "
                "three context rows"
            )
        eligible = tuple(row for row in rows if row.status.value != "not_applicable")
        values = tuple(row.value for row in eligible)
        value = (
            None
            if not eligible or any(item is None for item in values)
            else sum(float(item) for item in values) / len(values)
        )
        results.append(
            WorldMetricRow(
                condition=condition,
                world_id=world_id,
                metric_name=metric_name,
                value=value,
                context_count=len(eligible),
                row_count=3,
                undefined_context_count=sum(item is None for item in values),
            )
        )
    expected_conditions = (
        ConditionName.C1_LLM_PRE,
        ConditionName.C2_LLM_QUERY,
        ConditionName.A_FIXED_SELECT,
    )
    for metric_name in {item.metric_name for item in results}:
        for condition in expected_conditions:
            if sum(
                item.metric_name == metric_name and item.condition is condition
                for item in results
            ) != REGISTERED_WORLD_COUNT:
                raise Phase4AnalysisError(
                    f"cross-seed {metric_name}/{condition.value} lacks twelve world rows"
                )
    return tuple(results)


_EXPLORATORY_METRICS = (
    "contrastive_decision_change_precision",
    "contrastive_decision_change_recall",
    "contrastive_decision_change_f1",
    "contrastive_collapse",
    "contrast_invariant_preservation_rate",
    "contextual_node_f1",
    "essential_temporal_qualification_accuracy",
    "evidence_citation_validity",
    "grounding_precision",
    "unsupported_assertion_rate",
    "rare_pivotal_support_path_survival",
    "degree_histogram_entropy",
    "degree_mass_entropy",
    "local_relation_neighborhood_entropy",
    "native_schema_relation_entropy",
    "canonical_mapped_relation_entropy",
    "node_count",
    "assertion_edge_count",
    "density",
    "component_count",
    "label_overlap_count",
    "label_overlap_area",
    "irrelevant_visible_load",
    "leiden_cluster_count_base",
    "leiden_modularity_base",
    "community_adjusted_mutual_information",
    "community_purity",
    "community_mean_conductance",
    "community_fragmentation_error",
    "community_merging_error",
)

_ENTROPY_MULTIPLICITY_METRICS = (
    "degree_histogram_entropy",
    "degree_mass_entropy",
    "local_relation_neighborhood_entropy",
    "native_schema_relation_entropy",
    "canonical_mapped_relation_entropy",
)

_GOLD_ALIGNED_COMMUNITY_METRICS = tuple(
    f"{base}{suffix}"
    for base in (
        "community_adjusted_mutual_information",
        "community_purity",
        "community_mean_conductance",
        "community_fragmentation_error",
        "community_merging_error",
    )
    for suffix in ("", "_half", "_double")
)

_CROSS_SEED_COMMUNITY_METRICS = (
    "cross_seed_community_adjusted_mutual_information",
    "cross_seed_community_variation_of_information",
    "cross_seed_community_omitted_seed_1_count",
    "cross_seed_community_omitted_seed_2_count",
)


def _bootstrap_seed(root: int, label: str) -> int:
    payload = f"story-projection-onto/analysis/bootstrap/{root}/{label}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _exploratory_comparisons(
    world_rows: Sequence[WorldMetricRow],
    *,
    bootstrap_root_seed: int,
) -> tuple[ExploratoryComparison, ...]:
    lookup = {
        (item.metric_name, item.condition, item.world_id): item.value
        for item in world_rows
    }
    results = []
    worlds = tuple(
        sorted(
            {
                item.world_id
                for item in world_rows
                if item.condition is ConditionName.C2_LLM_QUERY
            }
        )
    )
    for metric_name in _EXPLORATORY_METRICS:
        for comparator in (ConditionName.C1_LLM_PRE, ConditionName.C0_CLASSICAL_PRE):
            treatment = {
                world: lookup.get((metric_name, ConditionName.C2_LLM_QUERY, world))
                for world in worlds
            }
            control = {
                world: lookup.get((metric_name, comparator, world)) for world in worlds
            }
            retained = tuple(
                world
                for world in worlds
                if treatment[world] is not None and control[world] is not None
            )
            comparison = f"C2-{comparator.value}"
            estimate = None
            status: Literal[
                "estimated", "not_estimated_incomplete_world_panel"
            ] = "not_estimated_incomplete_world_panel"
            if len(retained) == REGISTERED_WORLD_COUNT:
                value = analyze_paired_worlds(
                    {world: float(treatment[world]) for world in retained},
                    {world: float(control[world]) for world in retained},
                    comparison=comparison,
                    metric_name=metric_name,
                    bootstrap_seed=_bootstrap_seed(
                        bootstrap_root_seed,
                        f"exploratory/{metric_name}/{comparison}",
                    ),
                )
                estimate = asdict(value)
                status = "estimated"
            results.append(
                ExploratoryComparison(
                    metric_name=metric_name,
                    comparison=comparison,
                    status=status,
                    retained_world_count=len(retained),
                    estimate=estimate,
                )
            )
    return tuple(results)


def _entropy_multiplicity(
    comparisons: Sequence[ExploratoryComparison],
) -> tuple[SecondaryMultiplicityResult, ...]:
    selected = tuple(
        item
        for item in comparisons
        if item.metric_name in _ENTROPY_MULTIPLICITY_METRICS
        and item.comparison in {"C2-C1", "C2-C0"}
    )
    if len(selected) != 10:
        raise Phase4AnalysisError("entropy multiplicity family must contain ten planned tests")
    estimated = []
    for item in selected:
        if item.estimate is None:
            continue
        raw = item.estimate.get("two_sided_p_value")
        if not isinstance(raw, int | float) or isinstance(raw, bool):
            raise Phase4AnalysisError("estimated entropy comparison lacks a two-sided p-value")
        estimated.append((item, float(raw)))
    ordered = sorted(estimated, key=lambda pair: (pair[1], pair[0].metric_name, pair[0].comparison))
    adjusted_by_key: dict[tuple[str, str], float] = {}
    running = 1.0
    for rank, (item, raw) in reversed(tuple(enumerate(ordered, 1))):
        running = min(running, raw * 10 / rank, 1.0)
        adjusted_by_key[(item.metric_name, item.comparison)] = running
    results = []
    for item in sorted(selected, key=lambda value: (value.metric_name, value.comparison)):
        key = (item.metric_name, item.comparison)
        if item.estimate is None:
            results.append(
                SecondaryMultiplicityResult(
                    metric_name=item.metric_name,
                    comparison=item.comparison,
                    status="not_estimated_incomplete_world_panel",
                )
            )
            continue
        raw = float(item.estimate["two_sided_p_value"])
        results.append(
            SecondaryMultiplicityResult(
                metric_name=item.metric_name,
                comparison=item.comparison,
                status="adjusted",
                raw_two_sided_p_value=raw,
                benjamini_hochberg_adjusted_p_value=adjusted_by_key[key],
            )
        )
    return tuple(results)


def _eligible_world_comparisons(
    rows: Sequence[WorldMetricRow],
    *,
    panel: Literal["gold_aligned_community", "cross_seed_community"],
    metric_names: Sequence[str],
    comparator_conditions: Sequence[ConditionName],
    bootstrap_root_seed: int,
) -> tuple[EligibleWorldComparison, ...]:
    lookup = {
        (item.metric_name, item.condition, item.world_id): item
        for item in rows
    }
    worlds = tuple(
        sorted(
            {
                item.world_id
                for item in rows
                if item.condition is ConditionName.C2_LLM_QUERY
            }
        )
    )
    if len(worlds) != REGISTERED_WORLD_COUNT:
        raise Phase4AnalysisError(f"{panel} does not expose all twelve world identities")
    results = []
    for metric_name in metric_names:
        for comparator in comparator_conditions:
            pairs = []
            for world_id in worlds:
                treatment = lookup.get(
                    (metric_name, ConditionName.C2_LLM_QUERY, world_id)
                )
                control = lookup.get((metric_name, comparator, world_id))
                if treatment is None or control is None:
                    raise Phase4AnalysisError(
                        f"{panel}/{metric_name} lacks a declared world row"
                    )
                if treatment.context_count != control.context_count:
                    raise Phase4AnalysisError(
                        f"{panel}/{metric_name}/{world_id} eligibility differs by condition"
                    )
                if treatment.value is not None and control.value is not None:
                    pairs.append((world_id, float(treatment.value), float(control.value)))
            retained = tuple(item[0] for item in pairs)
            comparison = f"C2-{comparator.value}"
            estimate = None
            status: Literal[
                "estimated", "not_estimated_fewer_than_two_worlds"
            ] = "not_estimated_fewer_than_two_worlds"
            if len(pairs) >= 2:
                estimate = asdict(
                    analyze_paired_worlds(
                        {world: treatment for world, treatment, _ in pairs},
                        {world: control for world, _, control in pairs},
                        comparison=comparison,
                        metric_name=metric_name,
                        bootstrap_seed=_bootstrap_seed(
                            bootstrap_root_seed,
                            f"eligible/{panel}/{metric_name}/{comparison}",
                        ),
                        require_world_count=len(pairs),
                    )
                )
                status = "estimated"
            results.append(
                EligibleWorldComparison(
                    panel=panel,
                    metric_name=metric_name,
                    comparison=comparison,
                    eligible_world_ids=retained,
                    status=status,
                    estimate=estimate,
                )
            )
    return tuple(results)


_COMPARISON_COLUMNS = (
    "comparison",
    "metric",
    "estimate",
    "paired_median",
    "standardized_effect",
    "ci_lower",
    "ci_upper",
    "p_value",
    "adjusted_p_value",
    "sign_flip_p_value",
    "bootstrap_lower",
    "bootstrap_upper",
    "paired_world_count",
    "status",
    "analysis_family",
    "p_value_kind",
    "multiplicity_method",
    "independent_unit",
)


def _paired_comparison_row(
    value: Any | None,
    *,
    comparison: str,
    metric: str,
    family: str,
    p_value_kind: Literal["one_sided_superiority", "two_sided"],
    adjusted_p_value: float | None = None,
    multiplicity_method: str = "none",
    unavailable_status: str = "not_estimated",
    unavailable_world_count: int = 0,
) -> CanonicalComparisonRow:
    if value is None:
        return CanonicalComparisonRow(
            comparison=comparison,
            metric=metric,
            paired_world_count=unavailable_world_count,
            status=unavailable_status,
            analysis_family=family,
            p_value_kind=p_value_kind,
            multiplicity_method=multiplicity_method,
        )
    raw = asdict(value) if not isinstance(value, Mapping) else value
    p_key = (
        "one_sided_superiority_p_value"
        if p_value_kind == "one_sided_superiority"
        else "two_sided_p_value"
    )
    sign_key = (
        "one_sided_p_value"
        if p_value_kind == "one_sided_superiority"
        else "two_sided_p_value"
    )
    sign_flip = raw.get("sign_flip")
    bootstrap = raw.get("bootstrap")
    return CanonicalComparisonRow(
        comparison=comparison,
        metric=metric,
        estimate=float(raw["mean_difference"]),
        paired_median=float(raw["median_difference"]),
        standardized_effect=(
            None
            if raw["small_sample_corrected_effect_gz"] is None
            else float(raw["small_sample_corrected_effect_gz"])
        ),
        ci_lower=float(raw["two_sided_ci_lower"]),
        ci_upper=float(raw["two_sided_ci_upper"]),
        p_value=float(raw[p_key]),
        adjusted_p_value=adjusted_p_value,
        sign_flip_p_value=(
            float(sign_flip[sign_key]) if isinstance(sign_flip, Mapping) else None
        ),
        bootstrap_lower=(
            float(bootstrap["lower"]) if isinstance(bootstrap, Mapping) else None
        ),
        bootstrap_upper=(
            float(bootstrap["upper"]) if isinstance(bootstrap, Mapping) else None
        ),
        paired_world_count=int(raw["world_count"]),
        status="estimated",
        analysis_family=family,
        p_value_kind=p_value_kind,
        multiplicity_method=multiplicity_method,
    )


def _noninferiority_comparison_row(
    value: Any | None,
    *,
    comparison: str,
    family: str,
    unavailable_status: str = "not_estimated",
) -> CanonicalComparisonRow:
    metric = "rare_pivotal_qualified_assertion_recall"
    if value is None:
        return CanonicalComparisonRow(
            comparison=comparison,
            metric=metric,
            paired_world_count=0,
            status=unavailable_status,
            analysis_family=family,
            p_value_kind="one_sided_noninferiority",
            multiplicity_method="separate_claim_safeguard",
        )
    raw = asdict(value) if not isinstance(value, Mapping) else value
    bootstrap = raw.get("bootstrap")
    return CanonicalComparisonRow(
        comparison=comparison,
        metric=metric,
        estimate=float(raw["mean_difference"]),
        ci_lower=float(raw["one_sided_95_lower_bound"]),
        p_value=float(raw["one_sided_noninferiority_p_value"]),
        bootstrap_lower=(
            float(bootstrap["lower"]) if isinstance(bootstrap, Mapping) else None
        ),
        bootstrap_upper=(
            float(bootstrap["upper"]) if isinstance(bootstrap, Mapping) else None
        ),
        paired_world_count=int(raw["world_count"]),
        status="passes" if bool(raw["passes"]) else "fails",
        analysis_family=family,
        p_value_kind="one_sided_noninferiority",
        multiplicity_method="separate_claim_safeguard",
    )


def _canonical_comparison_rows(
    *,
    registered: Any,
    exploratory: Sequence[ExploratoryComparison],
    entropy_multiplicity: Sequence[SecondaryMultiplicityResult],
    eligible: Sequence[EligibleWorldComparison],
    crossings: Sequence[Mapping[str, Any]],
) -> tuple[CanonicalComparisonRow, ...]:
    holm = {item.hypothesis: item.adjusted_p_value for item in registered.primary_holm}
    rows = [
        _paired_comparison_row(
            registered.semantic.primary_c2_vs_c1,
            comparison="C2-C1",
            metric="strict_qualified_assertion_f1",
            family="registered_primary",
            p_value_kind="one_sided_superiority",
            adjusted_p_value=holm["H1-semantic"],
            multiplicity_method="Holm_two_primary_hypotheses",
        ),
        _paired_comparison_row(
            registered.organization.primary_c2_vs_c1,
            comparison="C2-C1",
            metric="ontology_decision_macro_f1",
            family="registered_primary",
            p_value_kind="one_sided_superiority",
            adjusted_p_value=holm["H1-organization"],
            multiplicity_method="Holm_two_primary_hypotheses",
        ),
        _paired_comparison_row(
            registered.semantic.secondary_c2_vs_c0,
            comparison="C2-C0",
            metric="strict_qualified_assertion_f1",
            family="required_secondary",
            p_value_kind="two_sided",
        ),
        _paired_comparison_row(
            registered.organization.secondary_c2_vs_c0,
            comparison="C2-C0",
            metric="ontology_decision_macro_f1",
            family="required_secondary",
            p_value_kind="two_sided",
        ),
        _noninferiority_comparison_row(
            registered.rare_c2_vs_c1,
            comparison="C2-C1",
            family="rare_pivotal_safeguard",
        ),
        _paired_comparison_row(
            registered.rare_secondary_c2_vs_c0,
            comparison="C2-C0",
            metric="rare_pivotal_qualified_assertion_recall",
            family="required_secondary",
            p_value_kind="two_sided",
        ),
        _paired_comparison_row(
            registered.mechanism_c2_vs_fixed_select,
            comparison="C2-A-FixedSelect",
            metric="ontology_decision_macro_f1",
            family="gated_mechanism",
            p_value_kind="one_sided_superiority",
            unavailable_status=registered.mechanism_status,
        ),
        _noninferiority_comparison_row(
            registered.rare_c2_vs_fixed_select,
            comparison="C2-A-FixedSelect",
            family="gated_mechanism_safeguard",
            unavailable_status=registered.mechanism_status,
        ),
    ]
    entropy_adjustments = {
        (item.metric_name, item.comparison): item.benjamini_hochberg_adjusted_p_value
        for item in entropy_multiplicity
    }
    for item in exploratory:
        rows.append(
            _paired_comparison_row(
                item.estimate,
                comparison=item.comparison,
                metric=item.metric_name,
                family="secondary_complete_world_panel",
                p_value_kind="two_sided",
                adjusted_p_value=entropy_adjustments.get(
                    (item.metric_name, item.comparison)
                ),
                multiplicity_method=(
                    "Benjamini-Hochberg_entropy_family"
                    if item.metric_name in _ENTROPY_MULTIPLICITY_METRICS
                    else "none"
                ),
                unavailable_status=item.status,
                unavailable_world_count=item.retained_world_count,
            )
        )
    for item in eligible:
        rows.append(
            _paired_comparison_row(
                item.estimate,
                comparison=item.comparison,
                metric=item.metric_name,
                family=item.panel,
                p_value_kind="two_sided",
                unavailable_status=item.status,
                unavailable_world_count=len(item.eligible_world_ids),
            )
        )
    for comparison in crossings:
        comparison_name = str(comparison["comparison"])
        for key, metric in (
            ("opportunity_incidence", "crossing_opportunity_incidence"),
            ("opportunity_count", "crossing_opportunity_count"),
            ("conditional_rate", "conditional_crossing_rate"),
        ):
            rows.append(
                _paired_comparison_row(
                    comparison.get(key),
                    comparison=comparison_name,
                    metric=metric,
                    family="crossing_two_part",
                    p_value_kind="two_sided",
                    unavailable_status="not_estimated_no_retained_world_panel",
                )
            )
        pooled = comparison.get("pooled_rate_difference")
        rows.append(
            CanonicalComparisonRow(
                comparison=comparison_name,
                metric="exposure_weighted_crossing_rate",
                estimate=None if pooled is None else float(pooled),
                paired_world_count=REGISTERED_WORLD_COUNT,
                status="descriptive_only" if pooled is not None else "not_applicable",
                analysis_family="crossing_two_part",
                p_value_kind="not_tested",
                multiplicity_method="none",
            )
        )
    ordered = tuple(
        sorted(rows, key=lambda item: (item.analysis_family, item.metric, item.comparison))
    )
    identities = tuple(
        (item.analysis_family, item.metric, item.comparison) for item in ordered
    )
    if len(identities) != len(set(identities)):
        raise Phase4AnalysisError("canonical comparison table contains duplicate rows")
    return ordered


def _comparison_csv(values: Sequence[CanonicalComparisonRow]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=_COMPARISON_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for value in values:
        raw = value.model_dump(mode="json")
        writer.writerow({column: raw.get(column) for column in _COMPARISON_COLUMNS})
    return stream.getvalue().encode("utf-8")


def _mapping_csv(
    columns: tuple[str, ...],
    values: Sequence[Mapping[str, Any]],
) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    expected = set(columns)
    for value in values:
        if set(value) != expected:
            raise Phase4AnalysisError("CSV row differs from its declared column contract")
        writer.writerow(value)
    return stream.getvalue().encode("utf-8")


_UNIT_METRIC_CSV_COLUMNS = (
    "source_block",
    "unit_id",
    "condition",
    "world_id",
    "context_id",
    "seed_block",
    "output_valid",
    "projection_id",
    "unit_hash",
    "metric_name",
    "metric_status",
    "value",
    "numerator",
    "denominator",
    "metric_version_hash",
)

_WORLD_METRIC_CSV_COLUMNS = (
    "condition",
    "world_id",
    "metric_name",
    "value",
    "context_count",
    "row_count",
    "undefined_context_count",
)

_RARE_CSV_COLUMNS = (
    "condition",
    "world_id",
    "context_id",
    "seed_block",
    "true_positive_count",
    "gold_count",
    "output_valid",
    "scorer_plan_hash",
)

_PARAPHRASE_CSV_COLUMNS = (
    "call_id",
    "world_id",
    "context_id",
    "status",
    "signature_divergence",
    "strict_f1_change",
    "base_strict_f1",
    "paraphrase_strict_f1",
    "base_score_hash",
    "paraphrase_score_hash",
    "base_projection_hash",
    "paraphrase_projection_hash",
)

_ABLATION_CSV_COLUMNS = (
    "call_id",
    "condition",
    "world_id",
    "context_id",
    "principal_metric",
    "parent_value",
    "ablation_value",
    "difference",
    "parent_score_hash",
    "ablation_score_hash",
)

_CONTRAST_METRIC_CSV_COLUMNS = (
    "world_id",
    "before_context_id",
    "after_context_id",
    "condition",
    "seed_block",
    "unit_hash",
    "metric_name",
    "metric_status",
    "value",
    "numerator",
    "denominator",
    "metric_version_hash",
)

_CROSS_SEED_METRIC_CSV_COLUMNS = (
    "world_id",
    "context_id",
    "condition",
    "seed_blocks",
    "unit_hash",
    "metric_name",
    "metric_status",
    "value",
    "numerator",
    "denominator",
    "metric_version_hash",
)

_REPORT_GATE_CSV_COLUMNS = (
    "gate_status_hash",
    "registered_analysis_hash",
    "alpha",
    "rare_safeguard_margin",
    "organization_adjusted_p_value",
    "mechanism_one_sided_p_value",
    "c2_vs_c1_rare_lower_bound",
    "c2_vs_fixed_rare_lower_bound",
    "c2_vs_fixed_rare_p_value",
    "organization_gate_passed",
    "construction_freedom_gate_passed",
    "c2_vs_c1_rare_safeguard_gate_passed",
    "c2_vs_fixed_rare_safeguard_status",
    "c2_vs_fixed_rare_safeguard_gate_passed",
)


def _unit_metric_csv_rows(
    observations: Sequence[ScoredMetricObservation],
) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        {
            "source_block": item.source_block,
            "unit_id": item.unit_id,
            "condition": item.condition.value,
            "world_id": item.world_id,
            "context_id": item.context_id,
            "seed_block": item.seed_block,
            "output_valid": item.output_valid,
            "projection_id": item.metric.projection_id,
            "unit_hash": item.metric.unit_hash,
            "metric_name": item.metric.metric_name,
            "metric_status": item.metric.status.value,
            "value": item.metric.value,
            "numerator": item.metric.numerator,
            "denominator": item.metric.denominator,
            "metric_version_hash": item.metric.metric_version_hash,
        }
        for item in observations
    )


def _world_metric_csv_rows(
    rows: Sequence[WorldMetricRow],
) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        {
            "condition": item.condition.value,
            "world_id": item.world_id,
            "metric_name": item.metric_name,
            "value": item.value,
            "context_count": item.context_count,
            "row_count": item.row_count,
            "undefined_context_count": item.undefined_context_count,
        }
        for item in rows
    )


def _rare_csv_rows(values: Sequence[Any]) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        {
            "condition": item.condition,
            "world_id": item.world_id,
            "context_id": item.context_id,
            "seed_block": item.seed_block,
            "true_positive_count": item.true_positive_count,
            "gold_count": item.gold_count,
            "output_valid": item.output_valid,
            "scorer_plan_hash": item.scorer_plan_hash,
        }
        for item in values
    )


def _paraphrase_csv_rows(
    values: Sequence[ParaphrasePairResult],
) -> tuple[Mapping[str, Any], ...]:
    rows = []
    for item in values:
        score = item.score
        rows.append(
            {
                "call_id": item.call_id,
                "world_id": item.world_id,
                "context_id": item.context_id,
                "status": item.status,
                "signature_divergence": None if score is None else score.signature_divergence,
                "strict_f1_change": None if score is None else score.strict_f1_change,
                "base_strict_f1": None if score is None else score.base_strict_f1,
                "paraphrase_strict_f1": (
                    None if score is None else score.paraphrase_strict_f1
                ),
                "base_score_hash": item.base_score_hash,
                "paraphrase_score_hash": item.paraphrase_score_hash,
                "base_projection_hash": item.base_projection_hash,
                "paraphrase_projection_hash": item.paraphrase_projection_hash,
            }
        )
    return tuple(rows)


def _ablation_csv_rows(
    values: Sequence[AblationPairResult],
) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        {
            "call_id": item.call_id,
            "condition": item.condition.value,
            "world_id": item.world_id,
            "context_id": item.context_id,
            "principal_metric": item.principal_metric,
            "parent_value": item.parent_value,
            "ablation_value": item.ablation_value,
            "difference": item.difference,
            "parent_score_hash": item.parent_score_hash,
            "ablation_score_hash": item.ablation_score_hash,
        }
        for item in values
    )


def _contrast_metric_csv_rows(
    values: Sequence[ContrastScoreObservation],
) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        {
            "world_id": item.world_id,
            "before_context_id": item.before_context_id,
            "after_context_id": item.after_context_id,
            "condition": item.condition.value,
            "seed_block": item.seed_block,
            "unit_hash": row.unit_hash,
            "metric_name": row.metric_name,
            "metric_status": row.status.value,
            "value": row.value,
            "numerator": row.numerator,
            "denominator": row.denominator,
            "metric_version_hash": row.metric_version_hash,
        }
        for item in values
        for row in item.score.metric_rows
    )


def _cross_seed_metric_csv_rows(
    values: Sequence[CrossSeedScoreObservation],
) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        {
            "world_id": item.world_id,
            "context_id": item.context_id,
            "condition": item.condition.value,
            "seed_blocks": "|".join(str(seed) for seed in item.score.seed_blocks),
            "unit_hash": row.unit_hash,
            "metric_name": row.metric_name,
            "metric_status": row.status.value,
            "value": row.value,
            "numerator": row.numerator,
            "denominator": row.denominator,
            "metric_version_hash": row.metric_version_hash,
        }
        for item in values
        for row in item.score.metric_rows
    )


def _report_gate_csv_row(value: Phase4ReportGateStatus) -> Mapping[str, Any]:
    return {
        "gate_status_hash": value.content_hash,
        "registered_analysis_hash": value.registered_analysis_hash,
        "alpha": value.alpha,
        "rare_safeguard_margin": value.rare_safeguard_margin,
        "organization_adjusted_p_value": value.organization_adjusted_p_value,
        "mechanism_one_sided_p_value": value.mechanism_one_sided_p_value,
        "c2_vs_c1_rare_lower_bound": value.c2_vs_c1_rare_lower_bound,
        "c2_vs_fixed_rare_lower_bound": value.c2_vs_fixed_rare_lower_bound,
        "c2_vs_fixed_rare_p_value": value.c2_vs_fixed_rare_p_value,
        "organization_gate_passed": value.organization_gate_passed,
        "construction_freedom_gate_passed": value.construction_freedom_gate_passed,
        "c2_vs_c1_rare_safeguard_gate_passed": (
            value.c2_vs_c1_rare_safeguard_gate_passed
        ),
        "c2_vs_fixed_rare_safeguard_status": value.c2_vs_fixed_rare_safeguard_status,
        "c2_vs_fixed_rare_safeguard_gate_passed": (
            value.c2_vs_fixed_rare_safeguard_gate_passed
        ),
    }


def _crossing_observations(primary: _ScoreCollection) -> tuple[CrossingObservation, ...]:
    observations = []
    for score in primary.scores:
        crossing = _row_value(score, "crossing_count")
        opportunities = _row_value(score, "crossing_opportunity_count")
        observations.append(
            CrossingObservation(
                condition=score.intended_unit.condition.value,
                world_id=score.intended_unit.world_id,
                context_id=score.intended_unit.context_id,
                seed_block=score.intended_unit.seed_block,
                crossing_count=None if crossing is None else int(crossing),
                opportunity_count=None if opportunities is None else int(opportunities),
            )
        )
    return tuple(observations)


def _crossing_comparisons(
    primary: _ScoreCollection,
    *,
    bootstrap_root_seed: int,
) -> tuple[Mapping[str, Any], ...]:
    observations = _crossing_observations(primary)
    results = tuple(
        compare_crossing_profiles(
            observations,
            treatment_condition=ConditionName.C2_LLM_QUERY.value,
            comparator_condition=comparator.value,
            comparison=f"C2-{comparator.value}",
            bootstrap_root_seed=bootstrap_root_seed,
        )
        for comparator in (ConditionName.C1_LLM_PRE, ConditionName.C0_CLASSICAL_PRE)
    )
    return tuple(asdict(item) for item in results)


@dataclass(frozen=True, slots=True)
class Phase4Preflight:
    analysis_configuration_hash: str
    metric_configuration_hash: str
    held_out_execution_hash: str
    held_out_bridge_hash: str
    combined_execution_hash: str
    review_completion_hash: str
    primary_cell_count: int
    combined_cell_count: int
    successful_projection_count: int
    missing_geometry_hashes: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return not self.missing_geometry_hashes


@dataclass(frozen=True, slots=True)
class _PreparedInputs:
    analysis_configuration: Phase4AnalysisConfiguration
    metric_configuration: StudyMetricConfiguration
    call_manifest: HeldOutCallManifest
    held_out_execution: HeldOutExecutionManifest
    scorer_bridge: ScorerBridgeAuthorization
    combined_manifest: CombinedCallManifest
    combined_execution: CombinedExecutionIndex
    primary_cells: tuple[_Cell, ...]
    combined_cells: tuple[_Cell, ...]
    scorer_contexts: Mapping[str, _ScorerContext]
    review: Any
    source_bindings: tuple[Phase4SourceBinding, ...]
    source_association_file_sha256: str
    source_association_hash: str
    source_tree_hash: str


def _source_binding(role: str, path: Path, value: ImmutableRecord) -> Phase4SourceBinding:
    return Phase4SourceBinding(
        role=role,
        file_sha256=_sha256_file(path),
        logical_content_hash=value.content_hash,
    )


def _prepare_inputs(
    *,
    repository: Path,
    configuration_path: Path,
    held_out_root: Path,
    scorer_bridge_path: Path,
    combined_root: Path,
    review_root: Path,
    benchmark_root: Path,
    ledger_path: Path,
    artifact_root: Path,
    source_association_path: Path,
) -> tuple[_PreparedInputs, Ledger]:
    repository = repository.resolve(strict=True)
    configuration = Phase4AnalysisConfiguration.load(repository, configuration_path)
    metric_configuration = StudyMetricConfiguration.load(
        repository / configuration.metric_configuration_path,
        seed_manifest_path=repository / configuration.seed_manifest_path,
    )
    call_manifest_path = held_out_root / "call_manifest.json"
    execution_path = held_out_root / "execution_manifest.json"
    combined_manifest_path = combined_root / "manifest.json"
    combined_execution_path = combined_root / "execution_index.json"
    call_manifest = _load_model(call_manifest_path, HeldOutCallManifest)
    execution = _load_model(execution_path, HeldOutExecutionManifest)
    bridge = _load_model(scorer_bridge_path, ScorerBridgeAuthorization)
    combined_manifest = _load_model(combined_manifest_path, CombinedCallManifest)
    combined_execution = _load_model(combined_execution_path, CombinedExecutionIndex)
    _verify_scorer_bridge(
        execution=execution,
        call_manifest=call_manifest,
        bridge=bridge,
    )
    if tuple(item.content_hash for item in combined_manifest.calls) != (
        combined_execution.all_call_spec_hashes_in_order
    ):
        raise Phase4AnalysisError("combined execution call inventory changed")
    for call in combined_manifest.calls:
        source = call.source
        if (
            source.held_out_call_manifest_hash != call_manifest.content_hash
            or source.held_out_execution_manifest_hash != execution.content_hash
            or source.held_out_scorer_bridge_hash != bridge.content_hash
            or source.final_reviewed_seal_hash != execution.final_reviewed_seal_hash
        ):
            raise Phase4AnalysisError("combined call source names another primary execution")

    worlds, reviewed, review = _scorer_sources(
        benchmark_root=benchmark_root,
        review_root=review_root,
    )
    if (
        review.final_seal.content_hash != execution.final_reviewed_seal_hash
        or review.manifest.content_hash != execution.review_completion_manifest_hash
    ):
        raise Phase4AnalysisError("execution and reproduced independent review gate differ")
    ledger = Ledger(ledger_path)
    artifacts = ArtifactStore(BlobStore(artifact_root), ledger)
    try:
        primary_cells, scorer_contexts = _primary_cells(
            call_manifest=call_manifest,
            execution=execution,
            artifacts=artifacts,
            worlds=worlds,
            reviewed=reviewed,
            metric_configuration=metric_configuration,
        )
        combined_cells = _combined_cells(
            combined_root=combined_root,
            manifest=combined_manifest,
            execution_index=combined_execution,
            artifacts=artifacts,
            scorer_contexts=scorer_contexts,
            metric_configuration=metric_configuration,
        )
    except BaseException:
        ledger.close()
        raise
    try:
        source_raw = validate_source_association(
            source_association_path,
            source_root=repository,
        )
    except Exception as error:
        raise Phase4AnalysisError(f"invalid Phase 4 source association: {error}") from error
    source_association_hash = source_raw.get("manifest_sha256")
    if not isinstance(source_association_hash, str):
        raise Phase4AnalysisError("source association lacks its manifest SHA-256")
    source_tree_hash = source_raw.get("local_tree_sha256")
    if not isinstance(source_tree_hash, str):
        raise Phase4AnalysisError("source association lacks its local tree SHA-256")
    source_bindings = tuple(
        sorted(
            (
                _source_binding("held_out_call_manifest", call_manifest_path, call_manifest),
                _source_binding("held_out_execution", execution_path, execution),
                _source_binding("held_out_scorer_bridge", scorer_bridge_path, bridge),
                _source_binding(
                    "combined_call_manifest", combined_manifest_path, combined_manifest
                ),
                _source_binding("combined_execution", combined_execution_path, combined_execution),
                Phase4SourceBinding(
                    role="review_completion_manifest",
                    file_sha256=_sha256_file(review_root / "completion_manifest.json"),
                    logical_content_hash=review.manifest.content_hash,
                ),
            ),
            key=lambda item: item.role,
        )
    )
    return (
        _PreparedInputs(
            analysis_configuration=configuration,
            metric_configuration=metric_configuration,
            call_manifest=call_manifest,
            held_out_execution=execution,
            scorer_bridge=bridge,
            combined_manifest=combined_manifest,
            combined_execution=combined_execution,
            primary_cells=primary_cells,
            combined_cells=combined_cells,
            scorer_contexts=scorer_contexts,
            review=review,
            source_bindings=source_bindings,
            source_association_file_sha256=_sha256_file(source_association_path),
            source_association_hash=source_association_hash,
            source_tree_hash=source_tree_hash,
        ),
        ledger,
    )


def preflight_phase4_analysis(
    *,
    repository: Path,
    configuration_path: Path,
    held_out_root: Path,
    scorer_bridge_path: Path,
    combined_root: Path,
    review_root: Path,
    benchmark_root: Path,
    ledger_path: Path,
    artifact_root: Path,
    source_association_path: Path,
    geometry_root: Path,
) -> Phase4Preflight:
    prepared, ledger = _prepare_inputs(
        repository=repository,
        configuration_path=configuration_path,
        held_out_root=held_out_root,
        scorer_bridge_path=scorer_bridge_path,
        combined_root=combined_root,
        review_root=review_root,
        benchmark_root=benchmark_root,
        ledger_path=ledger_path,
        artifact_root=artifact_root,
        source_association_path=source_association_path,
    )
    try:
        successful = tuple(
            cell.projection
            for cell in (*prepared.primary_cells, *prepared.combined_cells)
            if cell.outcome is RunOutcome.SUCCEEDED and cell.projection is not None
        )
        missing = tuple(
            sorted(
                {
                    projection.content_hash
                    for projection in successful
                    if bool(projection.validation_records)
                    and all(
                        item.validation_status.value == "accepted"
                        for item in projection.validation_records
                    )
                    and (
                        (geometry_root / f"{projection.content_hash}.json").is_symlink()
                        or not (
                            geometry_root / f"{projection.content_hash}.json"
                        ).is_file()
                    )
                }
            )
        )
        missing_set = set(missing)
        renderer = prepared.metric_configuration.renderer
        expected_renderer_hashes = (
            renderer.layout_config_hash,
            renderer.style_config_hash,
            renderer.font_config_hash,
        )
        for projection in successful:
            structurally_valid = bool(projection.validation_records) and all(
                item.validation_status.value == "accepted"
                for item in projection.validation_records
            )
            if not structurally_valid or projection.content_hash in missing_set:
                continue
            geometry = _geometry(projection, geometry_root=geometry_root, required=True)
            assert geometry is not None
            if (
                geometry.layout_config_hash,
                geometry.style_config_hash,
                geometry.font_config_hash,
            ) != expected_renderer_hashes:
                raise Phase4AnalysisError(
                    "renderer geometry differs from the frozen Phase 4 metric configuration"
                )
        return Phase4Preflight(
            analysis_configuration_hash=prepared.analysis_configuration.content_hash,
            metric_configuration_hash=prepared.metric_configuration.content_hash,
            held_out_execution_hash=prepared.held_out_execution.content_hash,
            held_out_bridge_hash=prepared.scorer_bridge.content_hash,
            combined_execution_hash=prepared.combined_execution.content_hash,
            review_completion_hash=prepared.review.manifest.content_hash,
            primary_cell_count=len(prepared.primary_cells),
            combined_cell_count=len(prepared.combined_cells),
            successful_projection_count=len(successful),
            missing_geometry_hashes=missing,
        )
    finally:
        ledger.close()


def _append_exact(path: Path, payload: bytes, *, restricted: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise Phase4AnalysisError(f"append-only Phase 4 output drift: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600 if restricted else 0o644)
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
                raise Phase4AnalysisError(
                    f"concurrent Phase 4 output drift: {path}"
                ) from error
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _jsonl(values: Iterable[ImmutableRecord]) -> bytes:
    return b"".join((item.to_canonical_json() + "\n").encode("utf-8") for item in values)


def _json_payload(value: Any) -> bytes:
    if isinstance(value, ImmutableRecord):
        return (value.to_canonical_json() + "\n").encode("utf-8")
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class _OutputPayload:
    relative_path: str
    payload: bytes
    logical_hash: str | None
    row_count: int | None
    release_class: ReleaseClass
    metric_rows: tuple[MetricResultRow, ...] = ()
    table_id: str | None = None
    columns: tuple[str, ...] = ()
    independent_unit: Literal[
        "intended_cell",
        "metric_observation",
        "world",
        "contrast_pair",
        "cross_seed_pair",
        "paraphrase_pair",
        "ablation_pair",
        "analysis",
    ] | None = None
    selection_ready: bool = False

    def __post_init__(self) -> None:
        table_fields_present = (
            self.table_id is not None,
            bool(self.columns),
            self.independent_unit is not None,
        )
        if any(table_fields_present) and not (
            all(table_fields_present)
            and self.row_count is not None
            and self.logical_hash is not None
        ):
            raise ValueError("table output metadata must be complete or entirely absent")
        if self.selection_ready and self.table_id is None:
            raise ValueError("only declared tables can be selection-ready")


def _record_metric_rows(
    *,
    ledger: Ledger,
    study_id: str,
    artifact_hash: str,
    rows: Sequence[MetricResultRow],
) -> None:
    for ordinal, row in enumerate(rows, 1):
        metric_id = (
            "phase4-metric-"
            + canonical_sha256(
                {
                    "study_id": study_id,
                    "artifact_hash": artifact_hash,
                    "ordinal": ordinal,
                    "row": row,
                }
            )[:32]
        )
        ledger.record_metric(
            metric_id=metric_id,
            study_id=study_id,
            projection_id=None,
            unit_hash=row.unit_hash,
            metric_name=row.metric_name,
            metric_version_hash=row.metric_version_hash,
            status=MetricStatus(row.status.value),
            value=row.value,
            numerator=row.numerator,
            denominator=row.denominator,
            result_artifact_hash=artifact_hash,
        )


def _dataclass_jsonl(values: Sequence[Any]) -> bytes:
    return b"".join((analysis_to_json(item) + "\n").encode("utf-8") for item in values)


def _dataclass_logical_hash(values: Sequence[Any]) -> str:
    return canonical_sha256(tuple(asdict(item) for item in values))


def _metric_rows(scores: Sequence[IntendedUnitScore]) -> tuple[MetricResultRow, ...]:
    return tuple(row for score in scores for row in score.rows)


def _verify_registered_analysis(
    value: Any,
    *,
    configuration: Phase4AnalysisConfiguration,
) -> None:
    raw = asdict(value)
    assignment_counts: list[int] = []
    bootstrap_counts: list[int] = []

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                if key == "assignment_count":
                    assignment_counts.append(int(child))
                elif key == "resamples":
                    bootstrap_counts.append(int(child))
                visit(child)
        elif isinstance(item, tuple | list):
            for child in item:
                visit(child)

    visit(raw)
    if len(assignment_counts) not in {5, 6} or any(
        count != configuration.registered_sign_flip_assignments
        for count in assignment_counts
    ):
        raise Phase4AnalysisError("registered result did not use every 4,096 sign assignment")
    if len(bootstrap_counts) not in {6, 8} or any(
        count != configuration.registered_bootstrap_resamples
        for count in bootstrap_counts
    ):
        raise Phase4AnalysisError("registered result did not use 10,000 world bootstraps")
    paired = (
        value.semantic.primary_c2_vs_c1,
        value.semantic.secondary_c2_vs_c0,
        value.organization.primary_c2_vs_c1,
        value.organization.secondary_c2_vs_c0,
        value.rare_secondary_c2_vs_c0,
        *((value.mechanism_c2_vs_fixed_select,) if value.mechanism_c2_vs_fixed_select else ()),
    )
    if any(
        item.world_count != REGISTERED_WORLD_COUNT
        or len(item.world_ids) != REGISTERED_WORLD_COUNT
        or item.degrees_of_freedom != REGISTERED_WORLD_COUNT - 1
        for item in paired
    ):
        raise Phase4AnalysisError("registered paired analysis was not performed on 12 worlds")
    if value.rare_c2_vs_c1.world_count != REGISTERED_WORLD_COUNT:
        raise Phase4AnalysisError("rare-pivotal noninferiority did not use twelve worlds")


def _report_gate_status(registered: Any) -> Phase4ReportGateStatus:
    holm = {item.hypothesis: item for item in registered.primary_holm}
    organization = holm["H1-organization"]
    mechanism = registered.mechanism_c2_vs_fixed_select
    fixed_rare = registered.rare_c2_vs_fixed_select
    value = Phase4ReportGateStatus(
        gate_status_id="phase4-report-gates-" + canonical_sha256(asdict(registered))[:24],
        registered_analysis_hash=canonical_sha256(asdict(registered)),
        organization_adjusted_p_value=organization.adjusted_p_value,
        mechanism_one_sided_p_value=(
            None if mechanism is None else mechanism.one_sided_superiority_p_value
        ),
        c2_vs_c1_rare_lower_bound=registered.rare_c2_vs_c1.one_sided_95_lower_bound,
        c2_vs_fixed_rare_lower_bound=(
            None if fixed_rare is None else fixed_rare.one_sided_95_lower_bound
        ),
        c2_vs_fixed_rare_p_value=(
            None if fixed_rare is None else fixed_rare.one_sided_noninferiority_p_value
        ),
        organization_gate_passed=organization.rejected,
        construction_freedom_gate_passed=bool(
            mechanism is not None
            and mechanism.one_sided_superiority_p_value <= registered.alpha
        ),
        c2_vs_c1_rare_safeguard_gate_passed=registered.rare_c2_vs_c1.passes,
        c2_vs_fixed_rare_safeguard_status=(
            "not_tested"
            if fixed_rare is None
            else ("passed" if fixed_rare.passes else "failed")
        ),
        c2_vs_fixed_rare_safeguard_gate_passed=bool(
            fixed_rare is not None and fixed_rare.passes
        ),
    )
    expected_overall = bool(
        value.organization_gate_passed
        and value.construction_freedom_gate_passed
        and value.c2_vs_c1_rare_safeguard_gate_passed
        and value.c2_vs_fixed_rare_safeguard_gate_passed
    )
    if expected_overall != registered.mechanism_statistical_rare_gate_passes:
        raise Phase4AnalysisError("report gate status differs from registered analysis flags")
    return value


def _make_payload(
    *,
    relative_path: str,
    values: Sequence[ImmutableRecord],
    release_class: ReleaseClass,
    metric_rows: tuple[MetricResultRow, ...] = (),
    table_id: str | None = None,
    columns: tuple[str, ...] = (),
    independent_unit: Literal[
        "intended_cell",
        "metric_observation",
        "world",
        "contrast_pair",
        "cross_seed_pair",
        "paraphrase_pair",
        "ablation_pair",
        "analysis",
    ] | None = None,
    selection_ready: bool = False,
) -> _OutputPayload:
    return _OutputPayload(
        relative_path=relative_path,
        payload=_jsonl(values),
        logical_hash=canonical_sha256(tuple(values)),
        row_count=len(values),
        release_class=release_class,
        metric_rows=metric_rows,
        table_id=table_id,
        columns=columns,
        independent_unit=independent_unit,
        selection_ready=selection_ready,
    )


def run_phase4_analysis(
    *,
    repository: Path,
    configuration_path: Path,
    held_out_root: Path,
    scorer_bridge_path: Path,
    combined_root: Path,
    review_root: Path,
    benchmark_root: Path,
    ledger_path: Path,
    artifact_root: Path,
    source_association_path: Path,
    geometry_root: Path,
    output_root: Path,
    completed_at: AwareDatetime,
) -> Phase4AnalysisIndex:
    """Score the immutable synthetic execution and write one append-only analysis bundle."""

    if completed_at.tzinfo is None or completed_at.utcoffset() is None:
        raise Phase4AnalysisError("Phase 4 completion timestamp must be timezone-aware")
    if output_root.is_symlink() or (output_root.exists() and not output_root.is_dir()):
        raise Phase4AnalysisError("Phase 4 output root must be a non-symlink directory")
    output_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    prepared, ledger = _prepare_inputs(
        repository=repository,
        configuration_path=configuration_path,
        held_out_root=held_out_root,
        scorer_bridge_path=scorer_bridge_path,
        combined_root=combined_root,
        review_root=review_root,
        benchmark_root=benchmark_root,
        ledger_path=ledger_path,
        artifact_root=artifact_root,
        source_association_path=source_association_path,
    )
    artifacts = ArtifactStore(BlobStore(artifact_root), ledger)
    configuration = prepared.analysis_configuration
    metric_configuration = prepared.metric_configuration
    try:
        primary = _score_cells(
            prepared.primary_cells,
            configuration=metric_configuration,
            geometry_root=geometry_root,
            require_geometry=configuration.require_geometry_for_valid_content,
        )
        combined = _score_cells(
            prepared.combined_cells,
            configuration=metric_configuration,
            geometry_root=geometry_root,
            require_geometry=configuration.require_geometry_for_valid_content,
        )
        if len(primary.scores) != REGISTERED_PRIMARY_SCORE_COUNT:
            raise Phase4AnalysisError("primary scoring did not retain all 252 ITT cells")
        if len(combined.scores) != REGISTERED_COMBINED_ORDINARY_SCORE_COUNT:
            raise Phase4AnalysisError("combined scoring did not retain all forty ITT cells")

        primary_manifest = IntendedMetricManifest(
            manifest_id=f"phase4-primary-{prepared.held_out_execution.content_hash[:24]}",
            units=tuple(item.intended for item in prepared.primary_cells),
        )
        combined_manifest = IntendedMetricManifest(
            manifest_id=f"phase4-combined-{prepared.combined_execution.content_hash[:24]}",
            units=tuple(item.intended for item in prepared.combined_cells),
        )
        metric_observations, rare_observations = analysis_observations_from_scores(
            primary.scores,
            intended_manifest=primary_manifest,
        )
        # This second conversion is an explicit ITT completeness check for the
        # paraphrase/ablation block; it is not fed into confirmatory inference.
        analysis_observations_from_scores(
            combined.scores,
            intended_manifest=combined_manifest,
        )
        registered = run_registered_analysis_from_observations(
            metric_observations,
            rare_observations,
            bootstrap_root_seed=configuration.bootstrap_root_seed,
        )
        _verify_registered_analysis(registered, configuration=configuration)
        report_gate_status = _report_gate_status(registered)

        contrast = _contrast_scores(
            primary=primary,
            scorer_contexts=prepared.scorer_contexts,
            configuration=metric_configuration,
        )
        cross_seed = _cross_seed_scores(
            primary=primary,
            scorer_contexts=prepared.scorer_contexts,
            configuration=metric_configuration,
        )
        paraphrase = _paraphrase_results(
            primary=primary,
            combined=combined,
            combined_manifest=prepared.combined_manifest,
        )
        ablations = _ablation_results(
            primary=primary,
            combined=combined,
            combined_manifest=prepared.combined_manifest,
        )
        world_metrics = _world_metric_rows(primary)
        contrast_world_metrics = _contrast_world_metric_rows(contrast)
        cross_seed_world_metrics = _cross_seed_world_metric_rows(cross_seed)
        exploratory = _exploratory_comparisons(
            (*world_metrics, *contrast_world_metrics),
            bootstrap_root_seed=configuration.bootstrap_root_seed,
        )
        entropy_multiplicity = _entropy_multiplicity(exploratory)
        crossings = _crossing_comparisons(
            primary,
            bootstrap_root_seed=configuration.bootstrap_root_seed,
        )
        eligible_comparisons = (
            *_eligible_world_comparisons(
                world_metrics,
                panel="gold_aligned_community",
                metric_names=_GOLD_ALIGNED_COMMUNITY_METRICS,
                comparator_conditions=(
                    ConditionName.C1_LLM_PRE,
                    ConditionName.C0_CLASSICAL_PRE,
                ),
                bootstrap_root_seed=configuration.bootstrap_root_seed,
            ),
            *_eligible_world_comparisons(
                cross_seed_world_metrics,
                panel="cross_seed_community",
                metric_names=_CROSS_SEED_COMMUNITY_METRICS,
                comparator_conditions=(
                    ConditionName.C1_LLM_PRE,
                    ConditionName.A_FIXED_SELECT,
                ),
                bootstrap_root_seed=configuration.bootstrap_root_seed,
            ),
        )
        comparison_rows = _canonical_comparison_rows(
            registered=registered,
            exploratory=exploratory,
            entropy_multiplicity=entropy_multiplicity,
            eligible=eligible_comparisons,
            crossings=crossings,
        )

        scorer_plans = tuple(
            sorted(
                {
                    value.plan.content_hash: value.plan
                    for value in prepared.scorer_contexts.values()
                }.values(),
                key=lambda item: item.plan_id,
            )
        )
        audits = tuple(
            sorted(
                {
                    item.content_hash: item
                    for item in (
                        *primary.grounding_audits.values(),
                        *combined.grounding_audits.values(),
                    )
                }.values(),
                key=lambda item: (item.projection_hash, item.content_hash),
            )
        )
        geometries = tuple(
            sorted(
                {
                    item.content_hash: item
                    for item in (*primary.geometries.values(), *combined.geometries.values())
                }.values(),
                key=lambda item: (item.projection_hash, item.content_hash),
            )
        )
        bundles = tuple(
            sorted(
                {
                    item.content_hash: item
                    for item in (*primary.bundles.values(), *combined.bundles.values())
                }.values(),
                key=lambda item: item.content_hash,
            )
        )
        unit_observations = tuple((*primary.observations, *combined.observations))
        contrast_rows = tuple(row for item in contrast for row in item.score.metric_rows)
        cross_seed_rows = tuple(row for item in cross_seed for row in item.score.metric_rows)
        unit_metric_csv_rows = _unit_metric_csv_rows(unit_observations)
        world_metric_csv_rows = _world_metric_csv_rows(world_metrics)
        contrast_metric_csv_rows = _contrast_metric_csv_rows(contrast)
        contrast_world_csv_rows = _world_metric_csv_rows(contrast_world_metrics)
        cross_seed_metric_csv_rows = _cross_seed_metric_csv_rows(cross_seed)
        cross_seed_world_csv_rows = _world_metric_csv_rows(cross_seed_world_metrics)
        rare_csv_rows = _rare_csv_rows(rare_observations)
        paraphrase_csv_rows = _paraphrase_csv_rows(paraphrase)
        ablation_csv_rows = _ablation_csv_rows(ablations)

        output_specs: list[_OutputPayload] = [
            _OutputPayload(
                relative_path="inputs/primary_intended_manifest.json",
                payload=_json_payload(primary_manifest),
                logical_hash=primary_manifest.content_hash,
                row_count=1,
                release_class=ReleaseClass.RESTRICTED,
            ),
            _OutputPayload(
                relative_path="analysis/report_gate_status.json",
                payload=_json_payload(report_gate_status),
                logical_hash=report_gate_status.content_hash,
                row_count=1,
                release_class=ReleaseClass.PUBLIC,
            ),
            _OutputPayload(
                relative_path="inputs/combined_intended_manifest.json",
                payload=_json_payload(combined_manifest),
                logical_hash=combined_manifest.content_hash,
                row_count=1,
                release_class=ReleaseClass.RESTRICTED,
            ),
            _make_payload(
                relative_path="inputs/scorer_plans.jsonl",
                values=scorer_plans,
                release_class=ReleaseClass.RESTRICTED,
            ),
            _make_payload(
                relative_path="inputs/grounding_audits.jsonl",
                values=audits,
                release_class=ReleaseClass.RESTRICTED,
            ),
            _make_payload(
                relative_path="inputs/geometry_inputs.jsonl",
                values=geometries,
                release_class=ReleaseClass.PUBLIC,
            ),
            _make_payload(
                relative_path="scores/primary.jsonl",
                values=primary.scores,
                release_class=ReleaseClass.RESTRICTED,
                metric_rows=_metric_rows(primary.scores),
            ),
            _make_payload(
                relative_path="scores/combined.jsonl",
                values=combined.scores,
                release_class=ReleaseClass.RESTRICTED,
                metric_rows=_metric_rows(combined.scores),
            ),
            _make_payload(
                relative_path="scores/complete_bundles.jsonl",
                values=bundles,
                release_class=ReleaseClass.RESTRICTED,
            ),
            _make_payload(
                relative_path="tables/unit_metrics.jsonl",
                values=unit_observations,
                release_class=ReleaseClass.PUBLIC,
                table_id="unit_metric_observations",
                columns=(
                    "content_hash",
                    "source_block",
                    "unit_id",
                    "condition",
                    "world_id",
                    "context_id",
                    "seed_block",
                    "output_valid",
                    "metric",
                ),
                independent_unit="metric_observation",
                selection_ready=True,
            ),
            _make_payload(
                relative_path="tables/world_metrics.jsonl",
                values=world_metrics,
                release_class=ReleaseClass.PUBLIC,
                table_id="primary_world_metrics",
                columns=(
                    "content_hash",
                    "condition",
                    "world_id",
                    "metric_name",
                    "value",
                    "context_count",
                    "row_count",
                    "undefined_context_count",
                ),
                independent_unit="world",
                selection_ready=True,
            ),
            _make_payload(
                relative_path="tables/contrast_scores.jsonl",
                values=contrast,
                release_class=ReleaseClass.PUBLIC,
                metric_rows=contrast_rows,
                table_id="contrast_pair_scores",
                columns=(
                    "content_hash",
                    "world_id",
                    "before_context_id",
                    "after_context_id",
                    "condition",
                    "seed_block",
                    "score",
                ),
                independent_unit="contrast_pair",
                selection_ready=True,
            ),
            _make_payload(
                relative_path="tables/contrast_world_metrics.jsonl",
                values=contrast_world_metrics,
                release_class=ReleaseClass.PUBLIC,
                table_id="contrast_world_metrics",
                columns=(
                    "content_hash",
                    "condition",
                    "world_id",
                    "metric_name",
                    "value",
                    "context_count",
                    "row_count",
                    "undefined_context_count",
                ),
                independent_unit="world",
            ),
            _make_payload(
                relative_path="tables/cross_seed_scores.jsonl",
                values=cross_seed,
                release_class=ReleaseClass.PUBLIC,
                metric_rows=cross_seed_rows,
                table_id="cross_seed_community_scores",
                columns=("content_hash", "world_id", "context_id", "condition", "score"),
                independent_unit="cross_seed_pair",
            ),
            _make_payload(
                relative_path="tables/cross_seed_world_metrics.jsonl",
                values=cross_seed_world_metrics,
                release_class=ReleaseClass.PUBLIC,
                table_id="cross_seed_world_metrics",
                columns=(
                    "content_hash",
                    "condition",
                    "world_id",
                    "metric_name",
                    "value",
                    "context_count",
                    "row_count",
                    "undefined_context_count",
                ),
                independent_unit="world",
            ),
            _make_payload(
                relative_path="tables/paraphrase.jsonl",
                values=paraphrase,
                release_class=ReleaseClass.PUBLIC,
                table_id="paraphrase_pairs",
                columns=(
                    "content_hash",
                    "call_id",
                    "world_id",
                    "context_id",
                    "base_score_hash",
                    "paraphrase_score_hash",
                    "base_projection_hash",
                    "paraphrase_projection_hash",
                    "status",
                    "score",
                ),
                independent_unit="paraphrase_pair",
                selection_ready=True,
            ),
            _make_payload(
                relative_path="tables/ablations.jsonl",
                values=ablations,
                release_class=ReleaseClass.PUBLIC,
                table_id="ablation_pairs",
                columns=(
                    "content_hash",
                    "call_id",
                    "condition",
                    "world_id",
                    "context_id",
                    "parent_score_hash",
                    "ablation_score_hash",
                    "principal_metric",
                    "parent_value",
                    "ablation_value",
                    "difference",
                ),
                independent_unit="ablation_pair",
                selection_ready=True,
            ),
        ]
        output_specs.extend(
            (
                _OutputPayload(
                    relative_path="tables/report_gate_status.csv",
                    payload=_mapping_csv(
                        _REPORT_GATE_CSV_COLUMNS,
                        (_report_gate_csv_row(report_gate_status),),
                    ),
                    logical_hash=canonical_sha256(
                        (_report_gate_csv_row(report_gate_status),)
                    ),
                    row_count=1,
                    release_class=ReleaseClass.PUBLIC,
                    table_id="report_gate_status_csv",
                    columns=_REPORT_GATE_CSV_COLUMNS,
                    independent_unit="analysis",
                ),
                _OutputPayload(
                    relative_path="tables/unit_metrics.csv",
                    payload=_mapping_csv(_UNIT_METRIC_CSV_COLUMNS, unit_metric_csv_rows),
                    logical_hash=canonical_sha256(unit_metric_csv_rows),
                    row_count=len(unit_metric_csv_rows),
                    release_class=ReleaseClass.PUBLIC,
                    table_id="report_unit_metrics_csv",
                    columns=_UNIT_METRIC_CSV_COLUMNS,
                    independent_unit="metric_observation",
                    selection_ready=True,
                ),
                _OutputPayload(
                    relative_path="tables/world_metrics.csv",
                    payload=_mapping_csv(_WORLD_METRIC_CSV_COLUMNS, world_metric_csv_rows),
                    logical_hash=canonical_sha256(world_metric_csv_rows),
                    row_count=len(world_metric_csv_rows),
                    release_class=ReleaseClass.PUBLIC,
                    table_id="report_world_metrics_csv",
                    columns=_WORLD_METRIC_CSV_COLUMNS,
                    independent_unit="world",
                    selection_ready=True,
                ),
                _OutputPayload(
                    relative_path="tables/contrast_metrics.csv",
                    payload=_mapping_csv(
                        _CONTRAST_METRIC_CSV_COLUMNS,
                        contrast_metric_csv_rows,
                    ),
                    logical_hash=canonical_sha256(contrast_metric_csv_rows),
                    row_count=len(contrast_metric_csv_rows),
                    release_class=ReleaseClass.PUBLIC,
                    table_id="report_contrast_metrics_csv",
                    columns=_CONTRAST_METRIC_CSV_COLUMNS,
                    independent_unit="contrast_pair",
                    selection_ready=True,
                ),
                _OutputPayload(
                    relative_path="tables/contrast_world_metrics.csv",
                    payload=_mapping_csv(
                        _WORLD_METRIC_CSV_COLUMNS,
                        contrast_world_csv_rows,
                    ),
                    logical_hash=canonical_sha256(contrast_world_csv_rows),
                    row_count=len(contrast_world_csv_rows),
                    release_class=ReleaseClass.PUBLIC,
                    table_id="report_contrast_world_metrics_csv",
                    columns=_WORLD_METRIC_CSV_COLUMNS,
                    independent_unit="world",
                ),
                _OutputPayload(
                    relative_path="tables/cross_seed_metrics.csv",
                    payload=_mapping_csv(
                        _CROSS_SEED_METRIC_CSV_COLUMNS,
                        cross_seed_metric_csv_rows,
                    ),
                    logical_hash=canonical_sha256(cross_seed_metric_csv_rows),
                    row_count=len(cross_seed_metric_csv_rows),
                    release_class=ReleaseClass.PUBLIC,
                    table_id="report_cross_seed_metrics_csv",
                    columns=_CROSS_SEED_METRIC_CSV_COLUMNS,
                    independent_unit="cross_seed_pair",
                ),
                _OutputPayload(
                    relative_path="tables/cross_seed_world_metrics.csv",
                    payload=_mapping_csv(
                        _WORLD_METRIC_CSV_COLUMNS,
                        cross_seed_world_csv_rows,
                    ),
                    logical_hash=canonical_sha256(cross_seed_world_csv_rows),
                    row_count=len(cross_seed_world_csv_rows),
                    release_class=ReleaseClass.PUBLIC,
                    table_id="report_cross_seed_world_metrics_csv",
                    columns=_WORLD_METRIC_CSV_COLUMNS,
                    independent_unit="world",
                ),
                _OutputPayload(
                    relative_path="tables/rare_pivotal_observations.csv",
                    payload=_mapping_csv(_RARE_CSV_COLUMNS, rare_csv_rows),
                    logical_hash=canonical_sha256(rare_csv_rows),
                    row_count=len(rare_csv_rows),
                    release_class=ReleaseClass.PUBLIC,
                    table_id="report_rare_pivotal_csv",
                    columns=_RARE_CSV_COLUMNS,
                    independent_unit="metric_observation",
                ),
                _OutputPayload(
                    relative_path="tables/paraphrase.csv",
                    payload=_mapping_csv(_PARAPHRASE_CSV_COLUMNS, paraphrase_csv_rows),
                    logical_hash=canonical_sha256(paraphrase_csv_rows),
                    row_count=len(paraphrase_csv_rows),
                    release_class=ReleaseClass.PUBLIC,
                    table_id="report_paraphrase_csv",
                    columns=_PARAPHRASE_CSV_COLUMNS,
                    independent_unit="paraphrase_pair",
                    selection_ready=True,
                ),
                _OutputPayload(
                    relative_path="tables/ablations.csv",
                    payload=_mapping_csv(_ABLATION_CSV_COLUMNS, ablation_csv_rows),
                    logical_hash=canonical_sha256(ablation_csv_rows),
                    row_count=len(ablation_csv_rows),
                    release_class=ReleaseClass.PUBLIC,
                    table_id="report_ablations_csv",
                    columns=_ABLATION_CSV_COLUMNS,
                    independent_unit="ablation_pair",
                    selection_ready=True,
                ),
            )
        )

        metric_payload = _dataclass_jsonl(metric_observations)
        rare_payload = _dataclass_jsonl(rare_observations)
        exploratory_payload = _jsonl(exploratory)
        entropy_payload = _jsonl(entropy_multiplicity)
        eligible_payload = _jsonl(eligible_comparisons)
        crossing_payload = _json_payload(crossings)
        registered_payload = (analysis_to_json(registered) + "\n").encode("utf-8")
        output_specs.extend(
            (
                _OutputPayload(
                    relative_path="tables/registered_metric_observations.jsonl",
                    payload=metric_payload,
                    logical_hash=_dataclass_logical_hash(metric_observations),
                    row_count=len(metric_observations),
                    release_class=ReleaseClass.PUBLIC,
                    table_id="registered_metric_observations",
                    columns=(
                        "condition",
                        "world_id",
                        "context_id",
                        "seed_block",
                        "metric_name",
                        "value",
                        "output_valid",
                        "gold_nonempty",
                        "gold_count",
                        "scorer_plan_hash",
                    ),
                    independent_unit="metric_observation",
                ),
                _OutputPayload(
                    relative_path="tables/rare_pivotal_observations.jsonl",
                    payload=rare_payload,
                    logical_hash=_dataclass_logical_hash(rare_observations),
                    row_count=len(rare_observations),
                    release_class=ReleaseClass.PUBLIC,
                    table_id="rare_pivotal_observations",
                    columns=(
                        "condition",
                        "world_id",
                        "context_id",
                        "seed_block",
                        "true_positive_count",
                        "gold_count",
                        "output_valid",
                        "scorer_plan_hash",
                    ),
                    independent_unit="metric_observation",
                ),
                _OutputPayload(
                    relative_path="analysis/exploratory.jsonl",
                    payload=exploratory_payload,
                    logical_hash=canonical_sha256(exploratory),
                    row_count=len(exploratory),
                    release_class=ReleaseClass.PUBLIC,
                    table_id="secondary_paired_comparisons",
                    columns=(
                        "content_hash",
                        "metric_name",
                        "comparison",
                        "status",
                        "retained_world_count",
                        "estimate",
                    ),
                    independent_unit="analysis",
                ),
                _OutputPayload(
                    relative_path="analysis/eligible_world_comparisons.jsonl",
                    payload=eligible_payload,
                    logical_hash=canonical_sha256(eligible_comparisons),
                    row_count=len(eligible_comparisons),
                    release_class=ReleaseClass.PUBLIC,
                    table_id="eligible_world_comparisons",
                    columns=(
                        "content_hash",
                        "panel",
                        "metric_name",
                        "comparison",
                        "eligible_world_ids",
                        "status",
                        "estimate",
                    ),
                    independent_unit="world",
                ),
                _OutputPayload(
                    relative_path="analysis/entropy_multiplicity.jsonl",
                    payload=entropy_payload,
                    logical_hash=canonical_sha256(entropy_multiplicity),
                    row_count=len(entropy_multiplicity),
                    release_class=ReleaseClass.PUBLIC,
                    table_id="entropy_benjamini_hochberg",
                    columns=(
                        "content_hash",
                        "family",
                        "metric_name",
                        "comparison",
                        "family_planned_size",
                        "status",
                        "raw_two_sided_p_value",
                        "benjamini_hochberg_adjusted_p_value",
                    ),
                    independent_unit="analysis",
                ),
                _OutputPayload(
                    relative_path="tables/comparisons.csv",
                    payload=_comparison_csv(comparison_rows),
                    logical_hash=canonical_sha256(comparison_rows),
                    row_count=len(comparison_rows),
                    release_class=ReleaseClass.PUBLIC,
                    table_id="canonical_comparisons",
                    columns=_COMPARISON_COLUMNS,
                    independent_unit="world",
                ),
                _OutputPayload(
                    relative_path="analysis/crossing_profiles.json",
                    payload=crossing_payload,
                    logical_hash=canonical_sha256(crossings),
                    row_count=len(crossings),
                    release_class=ReleaseClass.PUBLIC,
                    table_id="crossing_profile_comparisons",
                    columns=("comparison_result",),
                    independent_unit="analysis",
                ),
                _OutputPayload(
                    relative_path="analysis/registered_analysis.json",
                    payload=registered_payload,
                    logical_hash=canonical_sha256(asdict(registered)),
                    row_count=1,
                    release_class=ReleaseClass.PUBLIC,
                    table_id="registered_world_level_analysis",
                    columns=("registered_analysis_result",),
                    independent_unit="analysis",
                ),
            )
        )

        protocol_hash = canonical_sha256(
            {
                "methodological_plan": configuration.methodological_plan_file_sha256,
                "implementation_plan": configuration.implementation_plan_file_sha256,
            }
        )
        study_id = "phase4-" + canonical_sha256(
            {
                "analysis_configuration": configuration.content_hash,
                "held_out_execution": prepared.held_out_execution.content_hash,
                "combined_execution": prepared.combined_execution.content_hash,
                "source_tree": prepared.source_tree_hash,
            }
        )[:24]
        ledger.register_study(
            study_id=study_id,
            protocol_hash=protocol_hash,
            code_manifest_hash=prepared.source_tree_hash,
            configuration_hash=configuration.content_hash,
            release_class=ReleaseClass.RESTRICTED,
            created_at=completed_at,
        )

        output_files: list[Phase4OutputFile] = []
        table_entries: list[Phase4TableEntry] = []
        artifact_hash_by_path: dict[str, str] = {}
        for spec in sorted(output_specs, key=lambda item: item.relative_path):
            path = output_root / spec.relative_path
            _append_exact(
                path,
                spec.payload,
                restricted=spec.release_class is ReleaseClass.RESTRICTED,
            )
            record = artifacts.put_bytes(
                spec.payload,
                media_type=(
                    "application/x-ndjson"
                    if spec.relative_path.endswith(".jsonl")
                    else (
                        "text/csv"
                        if spec.relative_path.endswith(".csv")
                        else "application/json"
                    )
                ),
                release_class=spec.release_class,
                created_at=completed_at,
            )
            artifact_hash_by_path[spec.relative_path] = record.content_hash
            output_files.append(
                Phase4OutputFile(
                    relative_path=spec.relative_path,
                    file_sha256=record.content_hash,
                    logical_content_hash=spec.logical_hash,
                    row_count=spec.row_count,
                    release_class=spec.release_class,
                )
            )
            if spec.metric_rows:
                _record_metric_rows(
                    ledger=ledger,
                    study_id=study_id,
                    artifact_hash=record.content_hash,
                    rows=spec.metric_rows,
                )
            if spec.table_id is not None:
                assert spec.logical_hash is not None
                assert spec.row_count is not None
                assert spec.independent_unit is not None
                table_entries.append(
                    Phase4TableEntry(
                        table_id=spec.table_id,
                        relative_path=spec.relative_path,
                        columns=spec.columns,
                        row_count=spec.row_count,
                        file_sha256=record.content_hash,
                        logical_content_hash=spec.logical_hash,
                        release_class=spec.release_class,
                        independent_unit=spec.independent_unit,
                        selection_ready=spec.selection_ready,
                    )
                )

        table_manifest = Phase4TableManifest(
            manifest_id=f"phase4-tables-{study_id.removeprefix('phase4-')}",
            analysis_configuration_hash=configuration.content_hash,
            metric_version_hash=metric_configuration.metric_version_hash,
            tables=tuple(sorted(table_entries, key=lambda item: item.table_id)),
        )
        table_manifest_payload = _json_payload(table_manifest)
        table_manifest_relative = "table_manifest.json"
        _append_exact(
            output_root / table_manifest_relative,
            table_manifest_payload,
            restricted=False,
        )
        table_manifest_artifact = artifacts.put_bytes(
            table_manifest_payload,
            media_type="application/json",
            release_class=ReleaseClass.PUBLIC,
            created_at=completed_at,
        )
        output_files.append(
            Phase4OutputFile(
                relative_path=table_manifest_relative,
                file_sha256=table_manifest_artifact.content_hash,
                logical_content_hash=table_manifest.content_hash,
                row_count=1,
                release_class=ReleaseClass.PUBLIC,
            )
        )
        registered_path = "analysis/registered_analysis.json"
        report_gate_path = "analysis/report_gate_status.json"
        index = Phase4AnalysisIndex(
            run_id=study_id,
            analysis_configuration_hash=configuration.content_hash,
            metric_configuration_hash=metric_configuration.content_hash,
            metric_version_hash=metric_configuration.metric_version_hash,
            source_tree_association_file_sha256=prepared.source_association_file_sha256,
            source_tree_association_hash=prepared.source_association_hash,
            source_bindings=prepared.source_bindings,
            review_completion_manifest_hash=prepared.review.manifest.content_hash,
            final_reviewed_seal_hash=prepared.review.final_seal.content_hash,
            primary_intended_manifest_hash=primary_manifest.content_hash,
            combined_intended_manifest_hash=combined_manifest.content_hash,
            table_manifest_file_sha256=table_manifest_artifact.content_hash,
            table_manifest_hash=table_manifest.content_hash,
            registered_analysis_file_sha256=artifact_hash_by_path[registered_path],
            registered_analysis_hash=canonical_sha256(asdict(registered)),
            report_gate_status_file_sha256=artifact_hash_by_path[report_gate_path],
            report_gate_status_hash=report_gate_status.content_hash,
            output_files=tuple(sorted(output_files, key=lambda item: item.relative_path)),
            completed_at=completed_at,
        )
        index_payload = _json_payload(index)
        _append_exact(output_root / "analysis_index.json", index_payload, restricted=True)
        artifacts.put_bytes(
            index_payload,
            media_type="application/json",
            release_class=ReleaseClass.RESTRICTED,
            created_at=completed_at,
        )
        reproduced = Phase4AnalysisIndex.model_validate_json(
            _regular_file(output_root / "analysis_index.json")
        )
        if reproduced != index:
            raise Phase4AnalysisError("Phase 4 analysis index failed exact readback")
        return index
    finally:
        ledger.close()


__all__ = [
    "REGISTERED_ABLATION_PAIR_COUNT",
    "REGISTERED_COMBINED_ORDINARY_SCORE_COUNT",
    "REGISTERED_CONTRAST_SCORE_COUNT",
    "REGISTERED_CROSS_SEED_COMMUNITY_COUNT",
    "REGISTERED_PARAPHRASE_PAIR_COUNT",
    "REGISTERED_PRIMARY_SCORE_COUNT",
    "REGISTERED_WORLD_COUNT",
    "AblationPairResult",
    "CanonicalComparisonRow",
    "ContrastScoreObservation",
    "CrossSeedScoreObservation",
    "EligibleWorldComparison",
    "ExploratoryComparison",
    "ParaphrasePairResult",
    "Phase4AnalysisConfiguration",
    "Phase4AnalysisError",
    "Phase4AnalysisIndex",
    "Phase4OutputFile",
    "Phase4Preflight",
    "Phase4ReportGateStatus",
    "Phase4SourceBinding",
    "Phase4TableEntry",
    "Phase4TableManifest",
    "ScoredMetricObservation",
    "SecondaryMultiplicityResult",
    "WorldMetricRow",
    "preflight_phase4_analysis",
    "run_phase4_analysis",
]
