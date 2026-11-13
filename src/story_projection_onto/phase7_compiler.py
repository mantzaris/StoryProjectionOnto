"""Deterministic compiler from immutable study outputs to conference artifacts.

The compiler is intentionally not a statistics implementation.  Phase 4, feedback,
resource, and case-study producers publish canonical result tables and register their
bytes in a self-hashed source registry.  This module verifies that closed inventory,
applies only the six frozen qualitative-selection rules, canonicalizes table order,
and builds the report/release chain without manually transcribing a scientific value.

Incomplete registries are valid inputs.  They produce an explicitly incomplete
report and selection receipt; they never produce placeholder efficacy rows.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StringConstraints, model_validator

from story_projection_onto.case_study_analysis import (
    NOVEL_CASE_COLUMNS,
    CaseStudyNarrativeAnalysisReceipt,
)
from story_projection_onto.case_study_runtime import CaseReviewDimension
from story_projection_onto.contracts import (
    ConditionName,
    FeedbackAction,
)
from story_projection_onto.contracts import (
    canonical_sha256 as canonical_record_sha256,
)
from story_projection_onto.final_accounting import (
    FAILURE_ACCOUNTING_COLUMNS,
    FINAL_ACCOUNTING_SCHEMA_VERSION,
    RESOURCE_ACCOUNTING_COLUMNS,
    FinalAccountingError,
    compile_final_accounting,
)
from story_projection_onto.independent_review_runtime import (
    ReviewCompletionError,
    load_completed_review,
    load_public_reviewed_gold_publication,
)
from story_projection_onto.report_ingestion import (
    FINAL_TABLE_DEPENDENCIES,
    IngestedTableSpec,
    PredecessorState,
    ReportingIngestionManifest,
    SourceArtifactSpec,
    SourceFamily,
    compile_ingestion_receipt,
)
from story_projection_onto.reporting import (
    REGISTERED_COMPLETE_TABLE_IDS,
    REGISTERED_FIGURE_IDS,
    REGISTERED_SECTION_IDS,
    SHA256_PATTERN,
    FigureSpec,
    PhaseStatus,
    ReportStatus,
    ResultManifest,
    ReviewSupplementSpec,
    SectionSpec,
    SupplementalMetricCount,
    SupplementalMetricSourceSpec,
    TableSpec,
    build_results_report,
    canonical_manifest_payload,
    canonical_sha256,
    load_reporting_policy,
    section_contract_sha256,
    verify_report_build,
)
from story_projection_onto.scorer_only.blinded_postrun_review import (
    FROZEN_ERROR_CODES,
    BlindedCommunityReviewPackage,
    BlindedErrorReviewPackage,
    BlindedReviewError,
    CommunityReviewCompletion,
    CommunityReviewFinalization,
    CommunityReviewRejoinMap,
    CommunityReviewSourceManifest,
    ErrorReviewAdjudication,
    ErrorReviewCompletion,
    ErrorReviewFinalization,
    ErrorReviewRejoinMap,
    HeldOutFailureSourceManifest,
    load_error_taxonomy,
    prepare_community_review_finalization,
    prepare_community_review_package,
    prepare_error_review_finalization,
    prepare_error_review_package,
)
from story_projection_onto.scorer_only.phase5_report import (
    FEEDBACK_TABLE_COLUMNS,
    SCRIPTED_METRICS,
    TRACE_METRICS,
    Phase5FeedbackTableReceipt,
    Phase5FeedbackTableRow,
)
from story_projection_onto.synthetic_benchmark import IndependentReviewGateError

PHASE7_COMPILER_SCHEMA_VERSION = "1.0.0"
Sha256Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Identifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")]
RelativePath = Annotated[str, StringConstraints(min_length=1, max_length=500)]

_EXAMPLE_IDS = (
    "development_tutorial",
    "rare_pivotal",
    "temporal_epistemic",
    "held_out_illustration",
    "counterexample",
    "narrative_illustration",
)
_ERROR_REVIEW_COLUMNS = (
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
_ERROR_REVIEW_SUMMARY_COLUMNS = (
    "error_code",
    "failure_count",
    "world_count",
    "reviewed_failure_count",
    "independent_unit",
)
_COMMUNITY_REVIEW_COLUMNS = (
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
_COMMUNITY_REVIEW_SUMMARY_COLUMNS = (
    "condition",
    "resolution",
    "semantic_coherence",
    "interpretability",
    "evidence_support",
    "rubric_mean",
    "reviewed_partition_count",
    "independent_unit",
)
_EXAMPLE_ORDER = {value: index for index, value in enumerate(_EXAMPLE_IDS)}
_CONDITION_ORDER = {
    "c0_classical_pre": 0,
    "c1_llm_pre": 1,
    "c2_llm_query": 2,
    "a_fixed_select": 3,
}
_SUPPLEMENTAL_METRIC_TABLE_IDS = frozenset(
    {
        "community",
        "entropy_clutter",
        "mechanism_c2_vs_fixed",
        "primary_c2_vs_c1",
        "rare_pivotal",
    }
)


class Phase7CompilationError(RuntimeError):
    """A production-report input or generated artifact violated its contract."""


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TableProducer(StrEnum):
    CANONICAL_CSV = "canonical_csv"
    STUDY_STATUS = "study_status"
    QUALITATIVE_SELECTION = "qualitative_selection"


class PublicSourceCategory(StrEnum):
    """Required, non-overlapping parts of the public reproduction source tree."""

    SCIENCE_CODE = "science_code"
    STUDY_CONFIGURATION = "study_configuration"
    JSON_SCHEMAS = "json_schemas"
    PROMPTS = "prompts"
    SYNTHETIC_BENCHMARK = "synthetic_benchmark"
    SYNTHETIC_GOLD = "synthetic_gold"
    INTERACTIVE_INTERFACE = "interactive_interface"
    REPORT_ASSETS = "report_assets"
    PROJECT_METADATA = "project_metadata"
    VERIFICATION_TESTS = "verification_tests"


_PUBLIC_SOURCE_ROOTS: Mapping[PublicSourceCategory, tuple[str, ...]] = {
    PublicSourceCategory.INTERACTIVE_INTERFACE: ("ui",),
    PublicSourceCategory.JSON_SCHEMAS: ("schemas/jsonschema",),
    PublicSourceCategory.PROJECT_METADATA: (),
    PublicSourceCategory.PROMPTS: ("prompts",),
    PublicSourceCategory.REPORT_ASSETS: ("scripts",),
    PublicSourceCategory.SCIENCE_CODE: ("src/story_projection_onto",),
    PublicSourceCategory.STUDY_CONFIGURATION: ("configs/case_study", "configs/study"),
    PublicSourceCategory.SYNTHETIC_BENCHMARK: (
        "data/synthetic/condition_inputs",
        "data/synthetic/manifests",
        "data/synthetic/model_visible",
    ),
    PublicSourceCategory.SYNTHETIC_GOLD: ("data/synthetic/scorer_only",),
    PublicSourceCategory.VERIFICATION_TESTS: ("tests",),
}
_PUBLIC_SOURCE_SUFFIXES: Mapping[PublicSourceCategory, tuple[str, ...]] = {
    PublicSourceCategory.INTERACTIVE_INTERFACE: (
        ".css",
        ".html",
        ".js",
        ".json",
        ".md",
        ".sh",
    ),
    PublicSourceCategory.JSON_SCHEMAS: (".json",),
    PublicSourceCategory.PROJECT_METADATA: (),
    PublicSourceCategory.PROMPTS: (".md",),
    PublicSourceCategory.REPORT_ASSETS: (".mjs", ".py"),
    PublicSourceCategory.SCIENCE_CODE: (".py",),
    PublicSourceCategory.STUDY_CONFIGURATION: (".json", ".md"),
    PublicSourceCategory.SYNTHETIC_BENCHMARK: (".json",),
    PublicSourceCategory.SYNTHETIC_GOLD: (".json",),
    PublicSourceCategory.VERIFICATION_TESTS: (".json", ".mjs", ".py"),
}
_PUBLIC_SOURCE_EXTENSIONLESS_NAMES: Mapping[PublicSourceCategory, tuple[str, ...]] = {
    category: () for category in PublicSourceCategory
}
_PUBLIC_SOURCE_EXTENSIONLESS_NAMES = {
    **_PUBLIC_SOURCE_EXTENSIONLESS_NAMES,
    PublicSourceCategory.INTERACTIVE_INTERFACE: ("CYTOSCAPE_LICENSE",),
}
_PUBLIC_SOURCE_EXPLICIT_PATHS: Mapping[PublicSourceCategory, tuple[str, ...]] = {
    category: () for category in PublicSourceCategory
}
_PUBLIC_SOURCE_EXPLICIT_PATHS = {
    **_PUBLIC_SOURCE_EXPLICIT_PATHS,
    PublicSourceCategory.PROJECT_METADATA: (
        "LICENSE",
        "README.md",
        "pyproject.toml",
        "uv.lock",
    ),
    PublicSourceCategory.REPORT_ASSETS: (
        "docs/BLINDED_POSTRUN_AND_QUALITATIVE.md",
        "docs/HELD_OUT_PRIMARY_EXECUTION.md",
        "docs/INDEPENDENT_REVIEW_HANDOFF.md",
        "docs/PHASE4_SCORING_AND_ANALYSIS.md",
        "docs/PHASE5_EXECUTION.md",
        "docs/PHASE5_SCRIPT_COMMITMENT.md",
        "docs/PHASE5_SOURCE_PRODUCTION.md",
        "docs/PHASE6_EXECUTION.md",
        "docs/PHASE7_REPORTING.md",
        "docs/SYNTHETIC_BENCHMARK.md",
        "plan_notes/IMPLEMENTATION_PLAN_QUERY_DEPENDENT_TEMPORAL_ONTOLOGY.md",
        "plan_notes/METHODOLOGICAL_PLAN_QUERY_DEPENDENT_TEMPORAL_ONTOLOGY.md",
    ),
}
_PUBLIC_SOURCE_REQUIRED_PATHS: Mapping[PublicSourceCategory, tuple[str, ...]] = {
    PublicSourceCategory.INTERACTIVE_INTERFACE: (
        "ui/CYTOSCAPE_LICENSE",
        "ui/app.js",
        "ui/cytoscape.min.js",
        "ui/index.html",
        "ui/style.css",
    ),
    PublicSourceCategory.JSON_SCHEMAS: (
        "schemas/jsonschema/evidence_packet.schema.json",
        "schemas/jsonschema/fallback_control_plane_incident.schema.json",
        "schemas/jsonschema/ontology_projection.schema.json",
        "schemas/jsonschema/qualified_assertion.schema.json",
        "schemas/jsonschema/schema_manifest.json",
    ),
    PublicSourceCategory.PROJECT_METADATA: (
        "LICENSE",
        "README.md",
        "pyproject.toml",
        "uv.lock",
    ),
    PublicSourceCategory.PROMPTS: (
        "prompts/ablations/no_temporal_epistemic_v1.md",
        "prompts/c1_pre/prompt_v1.md",
        "prompts/c2_query/prompt_v1.md",
        "prompts/fixed_select/prompt_v1.md",
        "prompts/repair/prompt_v1.md",
    ),
    PublicSourceCategory.REPORT_ASSETS: (
        "docs/PHASE4_SCORING_AND_ANALYSIS.md",
        "docs/PHASE5_EXECUTION.md",
        "docs/PHASE6_EXECUTION.md",
        "docs/PHASE7_REPORTING.md",
        "plan_notes/IMPLEMENTATION_PLAN_QUERY_DEPENDENT_TEMPORAL_ONTOLOGY.md",
        "plan_notes/METHODOLOGICAL_PLAN_QUERY_DEPENDENT_TEMPORAL_ONTOLOGY.md",
        "scripts/build_fallback_control_plane_incident.py",
        "scripts/build_final_accounting_source_recipe.py",
        "scripts/build_public_bundle.py",
        "scripts/build_results_report.py",
        "scripts/compile_final_accounting.py",
        "scripts/compile_phase7_results.py",
        "scripts/run_phase4_analysis.py",
        "scripts/run_phase5_feedback.py",
    ),
    PublicSourceCategory.SCIENCE_CODE: (
        "src/story_projection_onto/case_study_analysis.py",
        "src/story_projection_onto/conditions/c0.py",
        "src/story_projection_onto/conditions/c1.py",
        "src/story_projection_onto/conditions/c2.py",
        "src/story_projection_onto/conditions/fixed_select.py",
        "src/story_projection_onto/contracts.py",
        "src/story_projection_onto/fallback_control_plane_incident.py",
        "src/story_projection_onto/final_accounting.py",
        "src/story_projection_onto/metrics/pipeline.py",
        "src/story_projection_onto/phase7_compiler.py",
        "src/story_projection_onto/public_release.py",
        "src/story_projection_onto/scorer_only/phase4_analysis.py",
        "src/story_projection_onto/scorer_only/phase5_report.py",
        "src/story_projection_onto/synthetic_benchmark.py",
    ),
    PublicSourceCategory.STUDY_CONFIGURATION: (
        "configs/study/authority.json",
        "configs/study/feedback.json",
        "configs/study/metrics.json",
        "configs/study/phase4_analysis.json",
        "configs/study/phase5_execution.json",
        "configs/study/phase7_compiler.json",
        "configs/study/public_reproduction_sources.json",
        "configs/study/reporting.json",
        "configs/study/resource_limits.json",
        "configs/study/synthetic_benchmark.json",
        "configs/study/visualization.json",
    ),
    PublicSourceCategory.SYNTHETIC_BENCHMARK: (
        "data/synthetic/manifests/benchmark_manifest.json",
        "data/synthetic/manifests/seed_manifest.json",
    ),
    PublicSourceCategory.SYNTHETIC_GOLD: (
        "data/synthetic/scorer_only/development/syn-dev-01.json",
        "data/synthetic/scorer_only/held_out/draft_seal.json",
        "data/synthetic/scorer_only/held_out/syn-test-01.json",
        "data/synthetic/scorer_only/held_out/syn-test-12.json",
        "data/synthetic/scorer_only/sampling/a_no_context_selection.json",
        "data/synthetic/scorer_only/sampling/paraphrase_selection.json",
    ),
    PublicSourceCategory.VERIFICATION_TESTS: (
        "tests/conftest.py",
        "tests/fixtures/phase1/c1_pre_output.json",
        "tests/fixtures/phase4/metric_regression_cases.json",
        "tests/integration/test_condition_pathways.py",
        "tests/integration/test_system_browser_smoke.py",
        "tests/integration/test_xgrammar_decoder_compatibility.py",
        "tests/integration/ui_browser_smoke.mjs",
        "tests/property/test_benchmark_properties.py",
        "tests/property/test_metric_structure_properties.py",
        "tests/unit/test_fallback_control_plane_incident.py",
        "tests/unit/test_final_accounting.py",
        "tests/unit/test_gpu_runtime.py",
        "tests/unit/test_phase7_compiler.py",
        "tests/unit/test_public_release.py",
    ),
}


class MetricSourceFilter(FrozenModel):
    """One exact predeclared filter over a supplemental Phase 4 metric table."""

    column: Annotated[str, StringConstraints(min_length=1, max_length=100)]
    equals: Annotated[str, StringConstraints(max_length=500)]


class MetricRowRequirement(FrozenModel):
    """Expected row inventory for one registered metric in one source table."""

    metric_name: Identifier
    expected_row_count: int = Field(ge=1)


class MetricPartitionRequirement(FrozenModel):
    """Expected per-metric cardinality for one exhaustive source partition."""

    column: Annotated[str, StringConstraints(min_length=1, max_length=100)]
    value: Annotated[str, StringConstraints(max_length=500)]
    expected_rows_per_metric: int = Field(ge=1)


class SupplementalMetricSourceContract(FrozenModel):
    """Frozen producer and row-level coverage required beside a compact paper table.

    Phase 7 does not recompute these scientific values. It verifies and publishes the
    canonical Phase 4 rows that carry the registered diagnostics which are intentionally
    not all promoted to paired-comparison rows.
    """

    source_role: Identifier
    producer_table_id: Identifier
    description: Annotated[str, StringConstraints(min_length=1, max_length=500)]
    metric_column: Literal["metric_name"] = "metric_name"
    required_columns: tuple[Annotated[str, StringConstraints(min_length=1)], ...]
    key_columns: tuple[Annotated[str, StringConstraints(min_length=1)], ...]
    row_filters: tuple[MetricSourceFilter, ...] = ()
    metrics: tuple[MetricRowRequirement, ...]
    partitions: tuple[MetricPartitionRequirement, ...]

    @model_validator(mode="after")
    def inventory_is_exhaustive(self) -> Self:
        for values, label in (
            (self.required_columns, "required columns"),
            (self.key_columns, "key columns"),
            (tuple(item.metric_name for item in self.metrics), "metric names"),
            (
                tuple((item.column, item.value) for item in self.partitions),
                "metric partitions",
            ),
        ):
            if not values or len(values) != len(set(values)):
                raise ValueError(f"supplemental metric {label} must be nonempty and unique")
        filter_columns = tuple(item.column for item in self.row_filters)
        if len(filter_columns) != len(set(filter_columns)):
            raise ValueError("supplemental metric filter columns must be unique")
        columns = set(self.required_columns)
        referenced = (
            {self.metric_column}
            | set(self.key_columns)
            | {item.column for item in self.row_filters}
            | {item.column for item in self.partitions}
        )
        if not referenced.issubset(columns):
            raise ValueError("supplemental metric contract references an undeclared column")
        partition_columns = {item.column for item in self.partitions}
        if len(partition_columns) != 1:
            raise ValueError("supplemental metric partitions require one exhaustive column")
        expected_rows = {item.expected_row_count for item in self.metrics}
        if len(expected_rows) != 1 or sum(
            item.expected_rows_per_metric for item in self.partitions
        ) != next(iter(expected_rows)):
            raise ValueError("supplemental metric partitions must exhaust every metric row")
        return self


class SupplementalMetricRowCount(FrozenModel):
    metric_name: Identifier
    row_count: int = Field(ge=0)


class SupplementalMetricSourceBinding(FrozenModel):
    """Materialized, hash-bound evidence for one supplemental metric source."""

    source_role: Identifier
    source_artifact_id: Identifier
    table_manifest_artifact_id: Identifier
    source_row_count: int = Field(ge=1)
    selected_row_count: int = Field(ge=1)
    metric_row_counts: tuple[SupplementalMetricRowCount, ...]

    @model_validator(mode="after")
    def metric_counts_are_canonical(self) -> Self:
        names = tuple(item.metric_name for item in self.metric_row_counts)
        if names != tuple(sorted(set(names))):
            raise ValueError("supplemental metric row counts must be sorted and unique")
        if sum(item.row_count for item in self.metric_row_counts) > self.selected_row_count:
            raise ValueError("supplemental metric counts exceed selected source rows")
        return self


class ExactRowFilter(FrozenModel):
    column: Annotated[str, StringConstraints(min_length=1, max_length=100)]
    equals: Annotated[str, StringConstraints(max_length=500)] | None = None
    one_of: tuple[Annotated[str, StringConstraints(max_length=500)], ...] = ()

    @model_validator(mode="after")
    def exactly_one_value_form(self) -> Self:
        if (self.equals is None) == (not self.one_of):
            raise ValueError("row filter requires exactly one of equals or one_of")
        if self.one_of != tuple(sorted(set(self.one_of))):
            raise ValueError("row-filter one_of values must be sorted and unique")
        return self

    def accepts(self, value: str) -> bool:
        return value == self.equals if self.equals is not None else value in self.one_of


class ExactRowIdentityGroup(FrozenModel):
    """Compact Cartesian specification of exact rows selected from a producer table."""

    fixed: Mapping[Annotated[str, StringConstraints(min_length=1, max_length=100)], str]
    varying: Mapping[
        Annotated[str, StringConstraints(min_length=1, max_length=100)],
        tuple[Annotated[str, StringConstraints(max_length=500)], ...],
    ] = Field(default_factory=dict)

    @model_validator(mode="after")
    def identity_axes_are_canonical(self) -> Self:
        if not self.fixed and not self.varying:
            raise ValueError("row-identity group must constrain at least one column")
        if set(self.fixed) & set(self.varying):
            raise ValueError("row-identity column cannot be both fixed and varying")
        for values in self.varying.values():
            if not values or values != tuple(sorted(set(values))):
                raise ValueError("row-identity varying values must be sorted and unique")
        return self


class TableContract(FrozenModel):
    table_id: Identifier
    producer: TableProducer
    required_columns: tuple[Annotated[str, StringConstraints(min_length=1)], ...]
    sort_columns: tuple[Annotated[str, StringConstraints(min_length=1)], ...]
    display_columns: tuple[Annotated[str, StringConstraints(min_length=1)], ...] = ()
    source_producer_table_id: Identifier | None = None
    registered_row_filters: tuple[ExactRowFilter, ...] = ()
    expected_output_row_count: int | None = Field(default=None, ge=1)
    row_identity_columns: tuple[
        Annotated[str, StringConstraints(min_length=1, max_length=100)], ...
    ] = ()
    exact_row_identity_groups: tuple[ExactRowIdentityGroup, ...] = ()
    supplemental_metric_sources: tuple[SupplementalMetricSourceContract, ...] = ()
    description: Annotated[str, StringConstraints(min_length=1, max_length=500)]

    @model_validator(mode="after")
    def valid_contract(self) -> Self:
        if len(self.required_columns) != len(set(self.required_columns)):
            raise ValueError("table required columns must be unique")
        if not set(self.sort_columns).issubset(self.required_columns):
            raise ValueError("table sort columns must be required columns")
        if not self.display_columns:
            raise ValueError("every report table requires an explicit PDF display-column set")
        if len(self.display_columns) != len(set(self.display_columns)):
            raise ValueError("table display columns must be unique")
        if not set(self.display_columns).issubset(self.required_columns):
            raise ValueError("table display columns must be required columns")
        if len(self.display_columns) > 8:
            raise ValueError("PDF display tables are limited to eight readable columns")
        source_roles = tuple(item.source_role for item in self.supplemental_metric_sources)
        if source_roles != tuple(sorted(set(source_roles))):
            raise ValueError("supplemental metric source roles must be sorted and unique")
        if self.supplemental_metric_sources and self.source_producer_table_id is None:
            raise ValueError(
                "supplemental metric panels must bind their compact source producer"
            )
        selection_fields = (
            self.registered_row_filters,
            self.expected_output_row_count,
            self.row_identity_columns,
            self.exact_row_identity_groups,
        )
        if (self.source_producer_table_id is None) != (not any(selection_fields)):
            raise ValueError(
                "a Phase 4 producer requires one complete frozen row-selection contract"
            )
        if self.source_producer_table_id is not None:
            filter_columns = tuple(item.column for item in self.registered_row_filters)
            if filter_columns != tuple(sorted(set(filter_columns))):
                raise ValueError("registered row filters must be sorted and unique")
            if not set(filter_columns).issubset(self.required_columns):
                raise ValueError("registered row filter references an undeclared column")
            if not self.row_identity_columns or self.row_identity_columns != tuple(
                sorted(set(self.row_identity_columns))
            ):
                raise ValueError("row identity columns must be sorted, nonempty, and unique")
            if not set(self.row_identity_columns).issubset(self.required_columns):
                raise ValueError("row identity references an undeclared column")
            expected_identities = self.expected_row_identities()
            if self.exact_row_identity_groups and len(expected_identities) != (
                self.expected_output_row_count
            ):
                raise ValueError(
                    "exact row identities must exhaust the expected output row count"
                )
            if not self.exact_row_identity_groups and self.registered_row_filters:
                raise ValueError(
                    "a filtered producer table requires exact expected row identities"
                )
        expected = {
            "study_status": TableProducer.STUDY_STATUS,
            "qualitative_examples": TableProducer.QUALITATIVE_SELECTION,
        }.get(self.table_id, TableProducer.CANONICAL_CSV)
        if self.producer is not expected:
            raise ValueError(f"table {self.table_id} has the wrong production route")
        return self

    def expected_row_identities(self) -> frozenset[tuple[str, ...]]:
        """Expand the configured identity groups without consulting result values."""

        if not self.exact_row_identity_groups:
            return frozenset()
        identities: set[tuple[str, ...]] = set()
        required_columns = set(self.row_identity_columns)
        for group in self.exact_row_identity_groups:
            if set(group.fixed) | set(group.varying) != required_columns:
                raise ValueError(
                    "each exact row-identity group must cover every identity column"
                )
            partial: list[dict[str, str]] = [dict(group.fixed)]
            for column in sorted(group.varying):
                partial = [
                    {**row, column: value}
                    for row in partial
                    for value in group.varying[column]
                ]
            for row in partial:
                identity = tuple(row[column] for column in self.row_identity_columns)
                if identity in identities:
                    raise ValueError("exact row-identity groups overlap")
                identities.add(identity)
        return frozenset(identities)


class PhaseLabel(FrozenModel):
    phase_id: Annotated[str, StringConstraints(pattern=r"^phase_[1-7]$")]
    label: Annotated[str, StringConstraints(min_length=1, max_length=160)]


class PrimaryFigureColumns(FrozenModel):
    label: str
    estimate: str
    lower: str
    upper: str


class PublicSourceInventoryGroup(FrozenModel):
    """One reviewed source subtree with a frozen relative-path inventory."""

    category: PublicSourceCategory
    roots: tuple[RelativePath, ...] = ()
    explicit_paths: tuple[RelativePath, ...] = ()
    include_suffixes: tuple[
        Annotated[str, StringConstraints(pattern=r"^\.[A-Za-z0-9]+$")], ...
    ] = ()
    include_extensionless_names: tuple[Identifier, ...] = ()
    required_paths: tuple[RelativePath, ...]
    expected_path_count: int = Field(ge=1)
    path_inventory_sha256: Sha256Digest

    @model_validator(mode="after")
    def inventory_specification_is_canonical(self) -> Self:
        sequences = (
            self.roots,
            self.explicit_paths,
            self.include_suffixes,
            self.include_extensionless_names,
            self.required_paths,
        )
        if any(values != tuple(sorted(set(values))) for values in sequences):
            raise ValueError("public-source inventory fields must be sorted and unique")
        if not self.roots and not self.explicit_paths:
            raise ValueError("public-source inventory group is empty")
        if self.roots and not (
            self.include_suffixes or self.include_extensionless_names
        ):
            raise ValueError("public-source inventory root lacks a file allowlist")
        if not self.required_paths:
            raise ValueError("public-source inventory group lacks required sentinel paths")
        expected = (
            _PUBLIC_SOURCE_ROOTS[self.category],
            _PUBLIC_SOURCE_EXPLICIT_PATHS[self.category],
            _PUBLIC_SOURCE_SUFFIXES[self.category],
            _PUBLIC_SOURCE_EXTENSIONLESS_NAMES[self.category],
            _PUBLIC_SOURCE_REQUIRED_PATHS[self.category],
        )
        observed = (
            self.roots,
            self.explicit_paths,
            self.include_suffixes,
            self.include_extensionless_names,
            self.required_paths,
        )
        if observed != expected:
            raise ValueError(
                f"public-source {self.category.value} policy differs from its reviewed allowlist"
            )
        return self


class PublicReproductionSourceManifest(FrozenModel):
    """Self-hashed, reviewed allowlist for source material shipped publicly."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    manifest_id: Literal["conference-public-reproduction-v1"]
    groups: tuple[PublicSourceInventoryGroup, ...]
    manifest_sha256: Sha256Digest

    @model_validator(mode="after")
    def exact_category_inventory(self) -> Self:
        categories = tuple(item.category for item in self.groups)
        if categories != tuple(sorted(PublicSourceCategory, key=lambda item: item.value)):
            raise ValueError(
                "public-source manifest requires every category exactly once in order"
            )
        roots = tuple(root for item in self.groups for root in item.roots)
        if len(roots) != len(set(roots)):
            raise ValueError("public-source manifest repeats a source root")
        for index, left in enumerate(roots):
            left_parts = PurePosixPath(left).parts
            for right in roots[index + 1 :]:
                right_parts = PurePosixPath(right).parts
                prefix_length = min(len(left_parts), len(right_parts))
                if left_parts[:prefix_length] == right_parts[:prefix_length]:
                    raise ValueError("public-source manifest contains overlapping roots")
        return self


class Phase7CompilerConfiguration(FrozenModel):
    schema_version: Literal["1.0.0"] = PHASE7_COMPILER_SCHEMA_VERSION
    configuration_id: Literal["conference-phase7-production-v1"]
    reporting_policy_path: RelativePath
    reporting_policy_file_sha256: Sha256Digest
    reporting_policy_sha256: Sha256Digest
    error_review_taxonomy_path: RelativePath
    error_review_taxonomy_file_sha256: Sha256Digest
    community_review_template_path: RelativePath
    community_review_template_file_sha256: Sha256Digest
    phase_labels: tuple[PhaseLabel, ...]
    table_contracts: tuple[TableContract, ...]
    sections: tuple[SectionSpec, ...]
    primary_figure_columns: PrimaryFigureColumns
    public_source_manifest_path: RelativePath
    public_source_manifest_file_sha256: Sha256Digest
    public_source_manifest_sha256: Sha256Digest
    maximum_qualitative_figures: Literal[6] = 6
    public_bundle_limit_bytes: Literal[2_000_000_000] = 2_000_000_000
    configuration_sha256: Sha256Digest

    @model_validator(mode="after")
    def exact_inventory(self) -> Self:
        phase_ids = [item.phase_id for item in self.phase_labels]
        if len(phase_ids) != len(set(phase_ids)) or set(phase_ids) != {
            f"phase_{number}" for number in range(1, 8)
        }:
            raise ValueError("compiler configuration requires seven unique phase labels")
        table_ids = [item.table_id for item in self.table_contracts]
        if len(table_ids) != len(set(table_ids)) or set(table_ids) != set(
            REGISTERED_COMPLETE_TABLE_IDS
        ):
            raise ValueError("compiler configuration requires all registered report tables")
        section_ids = [item.section_id for item in self.sections]
        if len(section_ids) != len(set(section_ids)) or set(section_ids) != set(
            REGISTERED_SECTION_IDS
        ):
            raise ValueError("compiler configuration requires all registered sections")
        referenced = {
            table_id for section in self.sections for table_id in section.table_ids
        }
        if referenced != set(REGISTERED_COMPLETE_TABLE_IDS):
            raise ValueError("every registered table must be attached to a report section")
        supplemental = {
            item.table_id for item in self.table_contracts if item.supplemental_metric_sources
        }
        if supplemental != _SUPPLEMENTAL_METRIC_TABLE_IDS:
            raise ValueError(
                "compiler configuration must bind every registered supplemental metric panel"
            )
        return self

    @classmethod
    def load(cls, path: Path, *, source_root: Path) -> Phase7CompilerConfiguration:
        payload = _load_self_hashed_json(path, "configuration_sha256", "Phase 7 configuration")
        value = cls.model_validate(payload)
        policy_path = _safe_source(source_root, value.reporting_policy_path)
        if _file_sha256(policy_path) != value.reporting_policy_file_sha256:
            raise Phase7CompilationError("frozen reporting policy file changed")
        policy = load_reporting_policy(policy_path)
        if policy.policy_sha256 != value.reporting_policy_sha256:
            raise Phase7CompilationError("compiler configuration names another reporting policy")
        for relative_path, expected_hash, label in (
            (
                value.error_review_taxonomy_path,
                value.error_review_taxonomy_file_sha256,
                "error-review taxonomy",
            ),
            (
                value.community_review_template_path,
                value.community_review_template_file_sha256,
                "community-review template",
            ),
        ):
            if _file_sha256(_safe_source(source_root, relative_path)) != expected_hash:
                raise Phase7CompilationError(f"frozen {label} file changed")
        if section_contract_sha256(value.sections) != policy.section_contract_sha256:
            raise Phase7CompilationError("configured section prose differs from reporting policy")
        public_manifest_path = _safe_source(
            source_root, value.public_source_manifest_path
        )
        if _file_sha256(public_manifest_path) != value.public_source_manifest_file_sha256:
            raise Phase7CompilationError("frozen public-source manifest file changed")
        public_manifest = load_public_reproduction_source_manifest(public_manifest_path)
        if public_manifest.manifest_sha256 != value.public_source_manifest_sha256:
            raise Phase7CompilationError(
                "compiler configuration names another public-source manifest"
            )
        return value


class PhaseInputState(FrozenModel):
    phase_id: Annotated[str, StringConstraints(pattern=r"^phase_[1-7]$")]
    status: ReportStatus
    reason: Annotated[str, StringConstraints(min_length=1, max_length=1000)]
    source_artifact_ids: tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def completed_has_lineage(self) -> Self:
        if self.status is ReportStatus.COMPLETE and not self.source_artifact_ids:
            raise ValueError("a complete phase requires immutable source artifacts")
        if self.status is ReportStatus.NOT_APPLICABLE:
            raise ValueError("all seven conference phases are applicable")
        if len(self.source_artifact_ids) != len(set(self.source_artifact_ids)):
            raise ValueError("phase source artifact IDs must be unique")
        return self


_MECHANISM_TABLE_FILTERS = (
    ExactRowFilter(
        column="analysis_family",
        one_of=(
            "gated_mechanism",
            "gated_mechanism_safeguard",
            "mechanism_support",
        ),
    ),
    ExactRowFilter(column="comparison", equals="C2-A-FixedSelect"),
)
_MECHANISM_TABLE_ROW_IDENTITIES = frozenset(
    {
        (
            "gated_mechanism",
            "C2-A-FixedSelect",
            "ontology_decision_macro_f1",
        ),
        (
            "gated_mechanism_safeguard",
            "C2-A-FixedSelect",
            "rare_pivotal_qualified_assertion_recall",
        ),
        (
            "mechanism_support",
            "C2-A-FixedSelect",
            "contrastive_decision_change_f1",
        ),
        (
            "mechanism_support",
            "C2-A-FixedSelect",
            "contrastive_collapse",
        ),
    }
)


def _validate_mechanism_table_rows(rows: Sequence[Mapping[str, str]]) -> None:
    identities = {
        (item["analysis_family"], item["comparison"], item["metric"])
        for item in rows
    }
    if len(rows) != len(identities) or identities != _MECHANISM_TABLE_ROW_IDENTITIES:
        raise Phase7CompilationError(
            "mechanism table lacks its exact endpoint, safeguard, change-F1, or collapse rows"
        )


def _validate_registered_table_rows(
    contract: TableContract,
    rows: Sequence[Mapping[str, str]],
) -> None:
    """Require the exact frozen selection from one immutable Phase 4 producer table."""

    if contract.source_producer_table_id is None:
        return
    if len(rows) != contract.expected_output_row_count:
        raise Phase7CompilationError(
            f"registered output row count changed for {contract.table_id}"
        )
    identities = tuple(
        tuple(row[column] for column in contract.row_identity_columns) for row in rows
    )
    if len(identities) != len(set(identities)):
        raise Phase7CompilationError(
            f"registered row identities are not unique for {contract.table_id}"
        )
    expected = contract.expected_row_identities()
    if expected and set(identities) != expected:
        raise Phase7CompilationError(
            f"registered row identities changed for {contract.table_id}"
        )

    if contract.table_id == "paraphrase_contrastive":
        world_ids = tuple(row["world_id"] for row in rows)
        context_ids = tuple(row["context_id"] for row in rows)
        call_ids = tuple(row["call_id"] for row in rows)
        if any(
            len(values) != 12 or len(values) != len(set(values))
            for values in (world_ids, context_ids, call_ids)
        ):
            raise Phase7CompilationError(
                "paraphrase table must contain one exact call/context from each of 12 worlds"
            )
    elif contract.table_id == "ablations":
        condition_counts = Counter(row["condition"] for row in rows)
        expected_counts = Counter(
            {
                ConditionName.A_NO_CONTEXT.value: 12,
                ConditionName.A_NO_TEMPORAL_EPISTEMIC.value: 8,
                ConditionName.A_NO_RARE_GUARD.value: 8,
            }
        )
        principal = {
            ConditionName.A_NO_CONTEXT.value: "ontology_decision_macro_f1",
            ConditionName.A_NO_TEMPORAL_EPISTEMIC.value: (
                "essential_temporal_qualification_accuracy"
            ),
            ConditionName.A_NO_RARE_GUARD.value: (
                "rare_pivotal_qualified_assertion_recall"
            ),
        }
        if condition_counts != expected_counts or any(
            row["principal_metric"] != principal.get(row["condition"])
            for row in rows
        ):
            raise Phase7CompilationError(
                "ablation table lacks its exact 12/8/8 condition and principal-metric inventory"
            )
        worlds_by_condition = {
            condition: {row["world_id"] for row in rows if row["condition"] == condition}
            for condition in expected_counts
        }
        if (
            len(worlds_by_condition[ConditionName.A_NO_CONTEXT.value]) != 12
            or len(worlds_by_condition[ConditionName.A_NO_RARE_GUARD.value]) != 8
        ):
            raise Phase7CompilationError(
                "ablation table violates the registered distinct-world subsets"
            )


def _validate_registered_table_binding(
    contract: TableContract,
    binding: Phase7TableBinding,
) -> None:
    """Reject operator-defined filtering or partial Phase 4 producer selections."""

    expected_scope = "interim" if binding.table_id == "study_status" else "final"
    if binding.scope != expected_scope:
        raise Phase7CompilationError(
            f"table {binding.table_id} differs from its registered {expected_scope} scope"
        )
    if contract.source_producer_table_id is not None:
        if (
            binding.row_filters != contract.registered_row_filters
            or binding.output_row_count != contract.expected_output_row_count
        ):
            raise Phase7CompilationError(
                f"table {binding.table_id} differs from its frozen row-selection contract"
            )
        if binding.producer_manifest_artifact_id is None:
            raise Phase7CompilationError(
                f"table {binding.table_id} lacks its Phase 4 producer manifest"
            )
    elif binding.producer_manifest_artifact_id is not None:
        raise Phase7CompilationError(
            f"unexpected producer manifest binding: {binding.table_id}"
        )


def _validate_final_accounting_table_selection(
    *,
    table_id: str,
    row_filters: Sequence[ExactRowFilter],
    producer_manifest_artifact_id: str | None,
    singleton_join_artifact_id: str | None,
    singleton_join_row_count: int | None,
    supplemental_metric_sources: Sequence[object],
) -> None:
    """Reject every report-side transform of a final-accounting producer table."""

    if table_id not in {"failure_accounting", "resource_accounting"}:
        return
    if (
        row_filters
        or producer_manifest_artifact_id is not None
        or singleton_join_artifact_id is not None
        or singleton_join_row_count is not None
        or supplemental_metric_sources
    ):
        raise Phase7CompilationError(
            "final accounting tables prohibit filters, joins, and alternate producers"
        )


class Phase7TableBinding(FrozenModel):
    table_id: Identifier
    status: ReportStatus
    reason: Annotated[str, StringConstraints(min_length=1, max_length=1000)]
    scope: Literal["interim", "final"]
    source_artifact_ids: tuple[Identifier, ...] = ()
    source_table_artifact_id: Identifier | None = None
    producer_manifest_artifact_id: Identifier | None = None
    source_receipt_artifact_id: Identifier | None = None
    source_row_count: int | None = Field(default=None, ge=0)
    output_row_count: int | None = Field(default=None, ge=0)
    row_filters: tuple[ExactRowFilter, ...] = ()
    singleton_join_artifact_id: Identifier | None = None
    singleton_join_row_count: Literal[1] | None = None
    supplemental_metric_sources: tuple[SupplementalMetricSourceBinding, ...] = ()

    @model_validator(mode="after")
    def source_agrees_with_status(self) -> Self:
        available = self.status is ReportStatus.COMPLETE
        generated = self.table_id in {"study_status", "qualitative_examples"}
        if self.status is ReportStatus.NOT_APPLICABLE:
            raise ValueError("all registered tables are applicable")
        if not available and any(
            (
                self.source_artifact_ids,
                self.source_table_artifact_id,
                self.producer_manifest_artifact_id,
                self.source_receipt_artifact_id,
                self.source_row_count,
                self.output_row_count,
                self.row_filters,
                self.singleton_join_artifact_id,
                self.singleton_join_row_count,
                self.supplemental_metric_sources,
            )
        ):
            raise ValueError("an unavailable table cannot claim source material")
        if available and self.table_id != "study_status" and not self.source_artifact_ids:
            raise ValueError("an available scientific table requires source lineage")
        if available and not generated and (
            self.source_table_artifact_id is None
            or self.source_row_count is None
            or self.output_row_count is None
        ):
            raise ValueError("an available canonical table requires its source CSV and row count")
        if generated and any(
            (
                self.source_table_artifact_id is not None,
                self.producer_manifest_artifact_id is not None,
                self.source_receipt_artifact_id is not None,
                self.source_row_count is not None,
                self.output_row_count is not None,
                self.row_filters,
                self.singleton_join_artifact_id,
                self.singleton_join_row_count,
            )
        ):
            raise ValueError("generated report tables cannot claim an upstream CSV")
        if self.source_table_artifact_id is not None and (
            self.source_table_artifact_id not in self.source_artifact_ids
        ):
            raise ValueError("the source CSV must be part of table lineage")
        if self.producer_manifest_artifact_id is not None and (
            self.producer_manifest_artifact_id not in self.source_artifact_ids
        ):
            raise ValueError("the producer manifest must be part of table lineage")
        if self.source_receipt_artifact_id is not None and (
            self.source_receipt_artifact_id not in self.source_artifact_ids
        ):
            raise ValueError("the source receipt must be part of table lineage")
        if (self.singleton_join_artifact_id is None) != (
            self.singleton_join_row_count is None
        ):
            raise ValueError("singleton join artifact and row count must be supplied together")
        if self.singleton_join_artifact_id is not None and (
            self.singleton_join_artifact_id not in self.source_artifact_ids
        ):
            raise ValueError("the singleton join CSV must be part of table lineage")
        if len(self.source_artifact_ids) != len(set(self.source_artifact_ids)):
            raise ValueError("table source artifact IDs must be unique")
        source_roles = tuple(item.source_role for item in self.supplemental_metric_sources)
        if source_roles != tuple(sorted(set(source_roles))):
            raise ValueError("supplemental metric source bindings must be sorted and unique")
        supplemental_artifacts = {
            artifact_id
            for item in self.supplemental_metric_sources
            for artifact_id in (item.source_artifact_id, item.table_manifest_artifact_id)
        }
        if not supplemental_artifacts.issubset(self.source_artifact_ids):
            raise ValueError("supplemental metric artifacts must be part of table lineage")
        columns = [item.column for item in self.row_filters]
        if len(columns) != len(set(columns)):
            raise ValueError("table row filters cannot repeat a column")
        return self


class QualitativeProducerArtifactBinding(FrozenModel):
    source_manifest_artifact_id: Identifier
    candidate_set_artifact_id: Identifier
    composite_manifest_artifact_id: Identifier
    materialization_receipt_artifact_id: Identifier

    def artifact_ids(self) -> tuple[str, ...]:
        return (
            self.source_manifest_artifact_id,
            self.candidate_set_artifact_id,
            self.composite_manifest_artifact_id,
            self.materialization_receipt_artifact_id,
        )

    @model_validator(mode="after")
    def roles_are_distinct(self) -> QualitativeProducerArtifactBinding:
        if len(set(self.artifact_ids())) != len(self.artifact_ids()):
            raise ValueError("qualitative producer roles must name distinct artifacts")
        return self


class ErrorReviewArtifactBinding(FrozenModel):
    source_manifest_artifact_id: Identifier
    package_artifact_id: Identifier
    rejoin_map_artifact_id: Identifier
    completion_artifact_id: Identifier
    adjudication_artifact_id: Identifier
    finalization_artifact_id: Identifier
    canonical_table_artifact_id: Identifier

    def artifact_ids(self) -> tuple[str, ...]:
        return (
            self.source_manifest_artifact_id,
            self.package_artifact_id,
            self.rejoin_map_artifact_id,
            self.completion_artifact_id,
            self.adjudication_artifact_id,
            self.finalization_artifact_id,
            self.canonical_table_artifact_id,
        )

    @model_validator(mode="after")
    def roles_are_distinct(self) -> ErrorReviewArtifactBinding:
        if len(set(self.artifact_ids())) != len(self.artifact_ids()):
            raise ValueError("error-review roles must name distinct artifacts")
        return self


class CommunityReviewArtifactBinding(FrozenModel):
    source_manifest_artifact_id: Identifier
    package_artifact_id: Identifier
    rejoin_map_artifact_id: Identifier
    completion_artifact_id: Identifier
    finalization_artifact_id: Identifier
    canonical_table_artifact_id: Identifier

    def artifact_ids(self) -> tuple[str, ...]:
        return (
            self.source_manifest_artifact_id,
            self.package_artifact_id,
            self.rejoin_map_artifact_id,
            self.completion_artifact_id,
            self.finalization_artifact_id,
            self.canonical_table_artifact_id,
        )

    @model_validator(mode="after")
    def roles_are_distinct(self) -> CommunityReviewArtifactBinding:
        if len(set(self.artifact_ids())) != len(self.artifact_ids()):
            raise ValueError("community-review roles must name distinct artifacts")
        return self


class _FinalAccountingOutputRecord(FrozenModel):
    table_id: Literal["failure_accounting", "resource_accounting"]
    relative_path: RelativePath
    file_sha256: Sha256Digest
    row_count: int = Field(ge=1)
    columns: tuple[Identifier, ...]

    @model_validator(mode="after")
    def exact_columns(self) -> Self:
        expected = {
            "failure_accounting": FAILURE_ACCOUNTING_COLUMNS,
            "resource_accounting": RESOURCE_ACCOUNTING_COLUMNS,
        }[self.table_id]
        if self.columns != expected:
            raise ValueError(f"{self.table_id} receipt columns changed")
        if PurePosixPath(self.relative_path).name != self.relative_path:
            raise ValueError("final-accounting receipt outputs must use leaf filenames")
        return self


class _FinalAccountingCompilationReceipt(FrozenModel):
    schema_version: Literal["1.0.0"] = FINAL_ACCOUNTING_SCHEMA_VERSION
    kind: Literal["final_phase7_accounting_compilation_receipt"]
    accounting_id: Annotated[
        str,
        StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$"),
    ]
    compiled_at_utc: AwareDatetime
    source_recipe_manifest_sha256: Sha256Digest
    source_recipe_file_sha256: Sha256Digest
    source_materialization_recipe_sha256: Sha256Digest
    native_source_file_sha256: tuple[Sha256Digest, ...]
    base_call_inventory_file_sha256: Sha256Digest
    base_accounting_events: int = Field(ge=1)
    base_inference_attempts: int = Field(ge=1)
    effective_accounting_events: int = Field(ge=1)
    effective_service_start_events: int = Field(ge=1)
    authorized_amendment_file_sha256: tuple[Sha256Digest, ...]
    ledger_snapshots: tuple[Mapping[str, Any], ...]
    receipt_file_sha256: tuple[Sha256Digest, ...]
    reconciliation: Mapping[str, Any]
    outputs: tuple[_FinalAccountingOutputRecord, ...]
    manifest_sha256: Sha256Digest

    @model_validator(mode="after")
    def exact_output_inventory(self) -> Self:
        if tuple(item.table_id for item in self.outputs) != (
            "failure_accounting",
            "resource_accounting",
        ):
            raise ValueError(
                "final-accounting receipt must contain its exact two ordered outputs"
            )
        if not self.native_source_file_sha256 or not self.ledger_snapshots:
            raise ValueError("final-accounting receipt lacks immutable source evidence")
        return self


class Phase7SourceRegistry(FrozenModel):
    schema_version: Literal["1.0.0"] = PHASE7_COMPILER_SCHEMA_VERSION
    registry_id: Identifier
    compiled_at_utc: AwareDatetime
    study_id: Annotated[str, StringConstraints(min_length=1, max_length=160)]
    status_reason: Annotated[str, StringConstraints(min_length=1, max_length=1000)]
    code_revision: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
    dirty_worktree: bool
    model_repository: Annotated[str, StringConstraints(min_length=1, max_length=200)] | None
    model_revision: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")] | None
    predecessors: tuple[PredecessorState, ...]
    phases: tuple[PhaseInputState, ...]
    tables: tuple[Phase7TableBinding, ...]
    qualitative_candidate_set_artifact_id: Identifier | None = None
    qualitative_producer_artifacts: QualitativeProducerArtifactBinding | None = None
    error_review_artifacts: ErrorReviewArtifactBinding | None = None
    community_review_artifacts: CommunityReviewArtifactBinding | None = None
    independent_review_completion_manifest_artifact_id: Identifier | None = None
    public_reviewed_gold_manifest_artifact_id: Identifier | None = None
    public_artifact_ids: tuple[Identifier, ...] = ()
    manifest_sha256: Sha256Digest

    @model_validator(mode="after")
    def exact_inventory(self) -> Self:
        families = [item.family for item in self.predecessors]
        if len(families) != len(set(families)) or set(families) != set(SourceFamily):
            raise ValueError("source registry requires each predecessor family exactly once")
        phases = [item.phase_id for item in self.phases]
        if len(phases) != len(set(phases)) or set(phases) != {
            f"phase_{number}" for number in range(1, 8)
        }:
            raise ValueError("source registry requires all seven phases")
        table_ids = [item.table_id for item in self.tables]
        if len(table_ids) != len(set(table_ids)) or set(table_ids) != set(
            REGISTERED_COMPLETE_TABLE_IDS
        ):
            raise ValueError("source registry requires every registered table")
        artifacts = [item for state in self.predecessors for item in state.artifacts]
        artifact_ids = [item.artifact_id for item in artifacts]
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("source artifact IDs must be globally unique")
        known = set(artifact_ids)
        references = {
            artifact_id for phase in self.phases for artifact_id in phase.source_artifact_ids
        } | {
            artifact_id for table in self.tables for artifact_id in table.source_artifact_ids
        } | set(self.public_artifact_ids)
        if self.qualitative_candidate_set_artifact_id is not None:
            references.add(self.qualitative_candidate_set_artifact_id)
        for binding in (
            self.qualitative_producer_artifacts,
            self.error_review_artifacts,
            self.community_review_artifacts,
        ):
            if binding is not None:
                references.update(binding.artifact_ids())
        for artifact_id in (
            self.independent_review_completion_manifest_artifact_id,
            self.public_reviewed_gold_manifest_artifact_id,
        ):
            if artifact_id is not None:
                references.add(artifact_id)
        if not references.issubset(known):
            raise ValueError("source registry names an unknown artifact")
        by_id = {item.artifact_id: item for item in artifacts}
        if any(by_id[item].release_class != "public" for item in self.public_artifact_ids):
            raise ValueError("public artifact allowlist contains a restricted artifact")
        if len(self.public_artifact_ids) != len(set(self.public_artifact_ids)):
            raise ValueError("public artifact IDs must be unique")
        if (
            self.qualitative_producer_artifacts is not None
            and self.qualitative_candidate_set_artifact_id
            != self.qualitative_producer_artifacts.candidate_set_artifact_id
        ):
            raise ValueError("qualitative candidate alias differs from its producer binding")
        return self


class QualitativeExample(StrEnum):
    DEVELOPMENT_TUTORIAL = "development_tutorial"
    RARE_PIVOTAL = "rare_pivotal"
    TEMPORAL_EPISTEMIC = "temporal_epistemic"
    HELD_OUT_ILLUSTRATION = "held_out_illustration"
    COUNTEREXAMPLE = "counterexample"
    NARRATIVE_ILLUSTRATION = "narrative_illustration"


class C2IntentionToTreatScore(FrozenModel):
    seed_block: Literal[1, 2]
    outcome: Literal["succeeded", "failed", "timed_out", "invalid", "interrupted"]
    strict_qualified_assertion_f1: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def failure_is_registered_zero(self) -> Self:
        if self.outcome != "succeeded" and self.strict_qualified_assertion_f1 != 0.0:
            raise ValueError("non-success C2 outcomes must retain the registered ITT zero")
        return self


class QualitativeDisplayRow(FrozenModel):
    context_id: str = Field(min_length=1, max_length=160)
    condition: Literal[
        "c0_classical_pre",
        "c1_llm_pre",
        "c2_llm_query",
        "a_fixed_select",
    ]
    node_summary: str = Field(min_length=1, max_length=1000)
    assertion_summary: str = Field(min_length=1, max_length=1500)
    temporal_sequence: str = Field(min_length=1, max_length=1000)
    why_matters: str = Field(min_length=1, max_length=1000)
    opaque_evidence_ids: tuple[str, ...]
    projection_hash: Sha256Digest | None = None

    @model_validator(mode="after")
    def evidence_is_canonical(self) -> Self:
        if not self.opaque_evidence_ids:
            raise ValueError("qualitative display rows require grounded evidence IDs")
        if self.opaque_evidence_ids != tuple(sorted(set(self.opaque_evidence_ids))):
            raise ValueError("qualitative evidence IDs must be sorted and unique")
        return self


class QualitativeCandidate(FrozenModel):
    candidate_id: Identifier
    eligible_examples: tuple[QualitativeExample, ...]
    evidence_split: Literal["development", "held_out", "case_study"]
    world_id: str = Field(min_length=1, max_length=160)
    context_ids: tuple[str, ...]
    query_ordinal: Literal["A", "B", "A+B", "not_applicable"]
    stratum: Literal["easy", "medium", "hard", "case_study"]
    window_ordinal: int | None = Field(default=None, ge=1, le=4)
    rare_pivotal_denominator: int = Field(default=0, ge=0)
    complete_support_path: bool = False
    temporal_epistemic_eligible: bool = False
    holder_attributed: bool = False
    c2_itt_scores: tuple[C2IntentionToTreatScore, ...] = ()
    reviewed_failure_id: Identifier | None = None
    display_rows: tuple[QualitativeDisplayRow, ...]
    composite_figure_artifact_id: Identifier
    source_artifact_ids: tuple[Identifier, ...]
    paraphrase_only: bool
    opaque_evidence_only: bool
    contains_verbatim_copyrighted_text: bool

    @model_validator(mode="after")
    def candidate_is_coherent(self) -> Self:
        if not self.eligible_examples or len(self.eligible_examples) != len(
            set(self.eligible_examples)
        ):
            raise ValueError("candidate eligibility must be nonempty and unique")
        if not self.context_ids or self.context_ids != tuple(sorted(set(self.context_ids))):
            raise ValueError("candidate contexts must be sorted and unique")
        if {item.context_id for item in self.display_rows} != set(self.context_ids):
            raise ValueError("display rows must cover exactly the candidate contexts")
        keys = [(item.context_id, item.condition) for item in self.display_rows]
        if len(keys) != len(set(keys)):
            raise ValueError("candidate repeats a context/condition display")
        if not self.source_artifact_ids or len(self.source_artifact_ids) != len(
            set(self.source_artifact_ids)
        ):
            raise ValueError("candidate source artifact IDs must be nonempty and unique")
        if self.composite_figure_artifact_id not in self.source_artifact_ids:
            raise ValueError("candidate composite figure must be part of source lineage")
        if self.evidence_split == "case_study":
            if self.window_ordinal is None or self.stratum != "case_study":
                raise ValueError("narrative candidates require a registered window ordinal")
            if (
                not self.paraphrase_only
                or not self.opaque_evidence_only
                or self.contains_verbatim_copyrighted_text
            ):
                raise ValueError("narrative candidates fail the public copyright boundary")
        elif self.window_ordinal is not None or self.stratum == "case_study":
            raise ValueError("synthetic candidates cannot claim a narrative window")
        if QualitativeExample.COUNTEREXAMPLE in self.eligible_examples:
            if (
                self.stratum != "hard"
                or {item.seed_block for item in self.c2_itt_scores} != {1, 2}
                or self.reviewed_failure_id is None
            ):
                raise ValueError("counterexample candidates require both hard-stratum C2 ITT seeds")
        elif self.c2_itt_scores or self.reviewed_failure_id is not None:
            raise ValueError(
                "outcome-dependent scores/review IDs are permitted only for counterexamples"
            )
        return self


class QualitativeCandidateSet(FrozenModel):
    schema_version: Literal["1.0.0"] = PHASE7_COMPILER_SCHEMA_VERSION
    candidate_set_id: Identifier
    frozen_reporting_policy_sha256: Sha256Digest
    candidates: tuple[QualitativeCandidate, ...]
    copyright_release_attestation_hash: Sha256Digest
    content_hash: Sha256Digest

    @model_validator(mode="after")
    def candidates_are_unique(self) -> Self:
        identifiers = [item.candidate_id for item in self.candidates]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("qualitative candidate IDs must be unique")
        return self


class CompiledPhase7Artifacts(FrozenModel):
    registry_sha256: Sha256Digest
    configuration_sha256: Sha256Digest
    build_token: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{16}$")]
    study_status: ReportStatus
    output_hashes: Mapping[str, Sha256Digest]
    current_pointer_sha256: Sha256Digest


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_relative_path(value: str) -> PurePosixPath:
    if "\\" in value:
        raise Phase7CompilationError(f"backslash is prohibited in Phase 7 path: {value}")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise Phase7CompilationError(f"unsafe Phase 7 path: {value}")
    return path


def _assert_no_symlink_chain(path: Path) -> None:
    current = path.absolute()
    while True:
        if current.is_symlink():
            raise Phase7CompilationError(f"symlink is prohibited in Phase 7 inputs: {path}")
        if current.parent == current:
            break
        current = current.parent


def _safe_source(root: Path, relative_path: str) -> Path:
    relative = _validate_relative_path(relative_path)
    candidate = root.joinpath(*relative.parts)
    _assert_no_symlink_chain(candidate)
    try:
        resolved = candidate.resolve(strict=True)
        resolved_root = root.resolve(strict=True)
    except OSError as error:
        raise Phase7CompilationError(f"missing Phase 7 source: {relative_path}") from error
    if not resolved.is_relative_to(resolved_root) or not resolved.is_file():
        raise Phase7CompilationError(f"Phase 7 source escapes its root: {relative_path}")
    return resolved


def _load_self_hashed_json(path: Path, field: str, label: str) -> dict[str, Any]:
    _assert_no_symlink_chain(path)
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise Phase7CompilationError(f"missing {label}: {path}") from error
    if raw.startswith(b"\xef\xbb\xbf"):
        raise Phase7CompilationError(f"{label} must be UTF-8 without BOM")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Phase7CompilationError(f"invalid {label}: {error}") from error
    if not isinstance(payload, dict):
        raise Phase7CompilationError(f"{label} must be one JSON object")
    supplied = payload.get(field)
    immutable = {key: value for key, value in payload.items() if key != field}
    if supplied != canonical_sha256(immutable):
        raise Phase7CompilationError(f"{label} canonical self-hash mismatch")
    return payload


def load_public_reproduction_source_manifest(
    path: Path,
) -> PublicReproductionSourceManifest:
    payload = _load_self_hashed_json(
        path,
        "manifest_sha256",
        "public reproduction source manifest",
    )
    return PublicReproductionSourceManifest.model_validate(payload)


def _safe_source_directory(root: Path, relative_path: str) -> Path:
    relative = _validate_relative_path(relative_path)
    candidate = root.joinpath(*relative.parts)
    _assert_no_symlink_chain(candidate)
    try:
        resolved = candidate.resolve(strict=True)
        resolved_root = root.resolve(strict=True)
    except OSError as error:
        raise Phase7CompilationError(
            f"missing public reproduction source root: {relative_path}"
        ) from error
    if not resolved.is_relative_to(resolved_root) or not resolved.is_dir():
        raise Phase7CompilationError(
            f"public reproduction source root escapes its tree: {relative_path}"
        )
    return resolved


def expand_public_reproduction_source_paths(
    manifest: PublicReproductionSourceManifest,
    *,
    source_root: Path,
) -> tuple[str, ...]:
    """Expand and verify the exact reviewed static-source inventory."""

    source_root = source_root.resolve(strict=True)
    paths_by_category: dict[PublicSourceCategory, tuple[str, ...]] = {}
    owner_by_path: dict[str, PublicSourceCategory] = {}
    for group in manifest.groups:
        selected = set(group.explicit_paths)
        for relative in group.explicit_paths:
            _safe_source(source_root, relative)
        for relative_root in group.roots:
            directory = _safe_source_directory(source_root, relative_root)
            for current, directory_names, file_names in os.walk(
                directory, followlinks=False
            ):
                current_path = Path(current)
                for name in (*directory_names, *file_names):
                    if (current_path / name).is_symlink():
                        raise Phase7CompilationError(
                            "symlink is prohibited in public reproduction source root: "
                            f"{(current_path / name).relative_to(source_root).as_posix()}"
                        )
                for name in file_names:
                    candidate = current_path / name
                    suffix = candidate.suffix.lower()
                    if (
                        suffix not in group.include_suffixes
                        and name not in group.include_extensionless_names
                    ):
                        continue
                    relative = candidate.relative_to(source_root).as_posix()
                    _safe_source(source_root, relative)
                    selected.add(relative)
        ordered = tuple(sorted(selected))
        missing_required = set(group.required_paths) - set(ordered)
        if missing_required:
            raise Phase7CompilationError(
                f"public-source {group.category.value} inventory lacks required paths: "
                + ", ".join(sorted(missing_required))
            )
        observed_hash = canonical_sha256(
            {"category": group.category.value, "paths": ordered}
        )
        if (
            len(ordered) != group.expected_path_count
            or observed_hash != group.path_inventory_sha256
        ):
            raise Phase7CompilationError(
                f"public-source {group.category.value} path inventory changed"
            )
        for relative in ordered:
            prior = owner_by_path.get(relative)
            if prior is not None:
                raise Phase7CompilationError(
                    f"public source belongs to both {prior.value} and "
                    f"{group.category.value}: {relative}"
                )
            owner_by_path[relative] = group.category
        paths_by_category[group.category] = ordered
    if set(paths_by_category) != set(PublicSourceCategory):
        raise Phase7CompilationError("public reproduction source categories are incomplete")
    return tuple(sorted(owner_by_path))


def load_phase7_source_registry(path: Path) -> Phase7SourceRegistry:
    payload = _load_self_hashed_json(path, "manifest_sha256", "Phase 7 source registry")
    return Phase7SourceRegistry.model_validate(payload)


def load_qualitative_candidate_set(path: Path) -> QualitativeCandidateSet:
    payload = _load_self_hashed_json(path, "content_hash", "qualitative candidate set")
    return QualitativeCandidateSet.model_validate(payload)


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _model_json_bytes(value: BaseModel) -> bytes:
    return _json_bytes(value.model_dump(mode="json"))


def _csv_bytes(columns: Sequence[str], rows: Sequence[Mapping[str, str]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _canonical_source_csv(
    path: Path,
    *,
    expected_hash: str,
    expected_rows: int,
    expected_output_rows: int,
    row_filters: Sequence[ExactRowFilter],
    contract: TableContract,
) -> tuple[tuple[str, ...], tuple[dict[str, str], ...], bytes]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_hash:
        raise Phase7CompilationError(f"source table hash mismatch: {contract.table_id}")
    if raw.startswith(b"\xef\xbb\xbf") or b"\r" in raw:
        raise Phase7CompilationError(f"source table is not UTF-8/LF CSV: {contract.table_id}")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise Phase7CompilationError(f"source table is not UTF-8: {contract.table_id}") from error
    reader = csv.DictReader(text.splitlines())
    columns = tuple(reader.fieldnames or ())
    if not columns or len(columns) != len(set(columns)):
        raise Phase7CompilationError(
            f"source table has missing/duplicate columns: {contract.table_id}"
        )
    if not set(contract.required_columns).issubset(columns):
        missing = sorted(set(contract.required_columns) - set(columns))
        raise Phase7CompilationError(
            f"source table {contract.table_id} lacks columns: {', '.join(missing)}"
        )
    rows = tuple(dict(row) for row in reader)
    if len(rows) != expected_rows or any(
        None in row or any(value is None for value in row.values()) for row in rows
    ):
        raise Phase7CompilationError(f"source table row contract failed: {contract.table_id}")
    if _csv_bytes(columns, rows) != raw:
        raise Phase7CompilationError(
            f"source table serialization is not canonical: {contract.table_id}"
        )
    unknown_filters = {item.column for item in row_filters} - set(columns)
    if unknown_filters:
        raise Phase7CompilationError(
            f"source table {contract.table_id} lacks filter columns: "
            + ", ".join(sorted(unknown_filters))
        )
    rows = tuple(
        row
        for row in rows
        if all(item.accepts(row[item.column]) for item in row_filters)
    )
    if len(rows) != expected_output_rows:
        raise Phase7CompilationError(
            f"source table filter count changed for {contract.table_id}: "
            f"{len(rows)} != {expected_output_rows}"
        )
    ordered = tuple(
        sorted(rows, key=lambda row: tuple(row[item] for item in contract.sort_columns))
    )
    return columns, ordered, _csv_bytes(columns, ordered)


def _canonical_singleton_csv(
    path: Path,
    *,
    expected_hash: str,
    table_id: str,
) -> tuple[tuple[str, ...], dict[str, str]]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_hash:
        raise Phase7CompilationError(f"singleton table hash mismatch: {table_id}")
    if raw.startswith(b"\xef\xbb\xbf") or b"\r" in raw:
        raise Phase7CompilationError(f"singleton table is not canonical UTF-8/LF CSV: {table_id}")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise Phase7CompilationError(f"singleton table is not UTF-8: {table_id}") from error
    reader = csv.DictReader(text.splitlines())
    columns = tuple(reader.fieldnames or ())
    rows = tuple(dict(row) for row in reader)
    if not columns or len(columns) != len(set(columns)) or len(rows) != 1:
        raise Phase7CompilationError(f"singleton table must contain exactly one row: {table_id}")
    if _csv_bytes(columns, rows) != raw:
        raise Phase7CompilationError(f"singleton table serialization is not canonical: {table_id}")
    return columns, rows[0]


def _canonical_csv_counts_for_manifest(
    path: Path,
    *,
    expected_hash: str,
) -> tuple[int, tuple[str, ...]]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_hash:
        raise Phase7CompilationError("Phase 4 manifest-bound CSV hash mismatch")
    if raw.startswith(b"\xef\xbb\xbf") or b"\r" in raw:
        raise Phase7CompilationError("Phase 4 manifest-bound CSV is not UTF-8/LF")
    try:
        reader = csv.DictReader(raw.decode("utf-8").splitlines())
    except UnicodeDecodeError as error:
        raise Phase7CompilationError("Phase 4 manifest-bound CSV is not UTF-8") from error
    columns = tuple(reader.fieldnames or ())
    rows = tuple(dict(row) for row in reader)
    if (
        not columns
        or len(columns) != len(set(columns))
        or any(None in row or any(value is None for value in row.values()) for row in rows)
        or _csv_bytes(columns, rows) != raw
    ):
        raise Phase7CompilationError("Phase 4 manifest-bound CSV is not canonical")
    return len(rows), columns


def _inspect_supplemental_metric_source_csv(
    path: Path,
    *,
    expected_hash: str,
    contract: SupplementalMetricSourceContract,
) -> tuple[
    int,
    int,
    tuple[SupplementalMetricRowCount, ...],
    tuple[str, ...],
    str | None,
]:
    """Verify one frozen Phase 4 metric CSV without aggregating or editing values."""

    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_hash:
        raise Phase7CompilationError(
            f"supplemental metric source hash mismatch: {contract.source_role}"
        )
    if raw.startswith(b"\xef\xbb\xbf") or b"\r" in raw:
        raise Phase7CompilationError(
            f"supplemental metric source is not UTF-8/LF CSV: {contract.source_role}"
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise Phase7CompilationError(
            f"supplemental metric source is not UTF-8: {contract.source_role}"
        ) from error
    reader = csv.DictReader(text.splitlines())
    columns = tuple(reader.fieldnames or ())
    rows = tuple(dict(row) for row in reader)
    if not columns or len(columns) != len(set(columns)):
        raise Phase7CompilationError(
            f"supplemental metric source has missing/duplicate columns: {contract.source_role}"
        )
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise Phase7CompilationError(
            f"supplemental metric source has a ragged row: {contract.source_role}"
        )
    if _csv_bytes(columns, rows) != raw:
        raise Phase7CompilationError(
            f"supplemental metric source is not canonical: {contract.source_role}"
        )
    missing = set(contract.required_columns) - set(columns)
    if missing:
        raise Phase7CompilationError(
            f"supplemental metric source {contract.source_role} lacks columns: "
            + ", ".join(sorted(missing))
        )
    selected = tuple(
        row
        for row in rows
        if all(row[item.column] == item.equals for item in contract.row_filters)
    )
    requirements = {item.metric_name: item.expected_row_count for item in contract.metrics}
    metric_rows = tuple(
        row for row in selected if row[contract.metric_column] in requirements
    )
    identities = tuple(
        (row[contract.metric_column], *(row[column] for column in contract.key_columns))
        for row in metric_rows
    )
    if len(identities) != len(set(identities)):
        raise Phase7CompilationError(
            f"supplemental metric source repeats a registered key: {contract.source_role}"
        )
    counts = tuple(
        SupplementalMetricRowCount(
            metric_name=metric_name,
            row_count=sum(row[contract.metric_column] == metric_name for row in metric_rows),
        )
        for metric_name in sorted(requirements)
    )
    observed = {item.metric_name: item.row_count for item in counts}
    if observed != requirements:
        changed = {
            name: (observed.get(name, 0), expected)
            for name, expected in requirements.items()
            if observed.get(name, 0) != expected
        }
        raise Phase7CompilationError(
            f"supplemental metric row inventory changed for {contract.source_role}: {changed}"
        )
    partition_column = contract.partitions[0].column
    expected_partitions = {
        item.value: item.expected_rows_per_metric for item in contract.partitions
    }
    for metric_name in sorted(requirements):
        observed_partitions = {
            value: sum(
                row[contract.metric_column] == metric_name
                and row[partition_column] == value
                for row in metric_rows
            )
            for value in expected_partitions
        }
        if observed_partitions != expected_partitions:
            raise Phase7CompilationError(
                "supplemental metric partition inventory changed for "
                f"{contract.source_role}/{metric_name}: {observed_partitions}"
            )
    version_hashes = {
        row["metric_version_hash"]
        for row in selected
        if "metric_version_hash" in contract.required_columns
    }
    if version_hashes and (
        len(version_hashes) != 1
        or any(SHA256_PATTERN.fullmatch(item) is None for item in version_hashes)
    ):
        raise Phase7CompilationError(
            f"supplemental metric source mixes metric revisions: {contract.source_role}"
        )
    metric_version_hash = next(iter(version_hashes)) if version_hashes else None
    return len(rows), len(selected), counts, columns, metric_version_hash


def _verify_phase4_table_manifest_binding(
    manifest_path: Path,
    *,
    expected_manifest_hash: str,
    source: SourceArtifactSpec,
    source_columns: Sequence[str],
    source_row_count: int,
    producer_table_id: str,
    metric_version_hash: str | None = None,
) -> None:
    """Prove that a supplemental CSV is one entry in the same Phase 4 table manifest."""

    if _file_sha256(manifest_path) != expected_manifest_hash:
        raise Phase7CompilationError("Phase 4 table-manifest physical hash mismatch")
    raw_manifest = manifest_path.read_bytes()
    payload = _load_self_hashed_json(
        manifest_path,
        "content_hash",
        "Phase 4 table manifest",
    )
    canonical_manifest = (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    if canonical_manifest != raw_manifest:
        raise Phase7CompilationError("Phase 4 table manifest is not canonical JSON")
    if (
        payload.get("schema_version") != "1.0.0"
        or payload.get("independent_confirmatory_unit") != "world"
        or SHA256_PATTERN.fullmatch(str(payload.get("analysis_configuration_hash"))) is None
        or SHA256_PATTERN.fullmatch(str(payload.get("metric_version_hash"))) is None
    ):
        raise Phase7CompilationError("Phase 4 table manifest has an invalid protocol header")
    if metric_version_hash is not None and payload["metric_version_hash"] != metric_version_hash:
        raise Phase7CompilationError(
            f"Phase 4 metric revision mismatch for {producer_table_id}"
        )
    tables = payload.get("tables")
    if not isinstance(tables, list) or any(not isinstance(item, dict) for item in tables):
        raise Phase7CompilationError("Phase 4 table manifest lacks its table inventory")
    table_ids = tuple(str(item.get("table_id")) for item in tables)
    if table_ids != tuple(sorted(set(table_ids))):
        raise Phase7CompilationError("Phase 4 table manifest inventory is not canonical")
    matches = [item for item in tables if item.get("table_id") == producer_table_id]
    if len(matches) != 1:
        raise Phase7CompilationError(
            f"Phase 4 table manifest does not uniquely bind {producer_table_id}"
        )
    entry = matches[0]
    relative = entry.get("relative_path")
    source_relative = PurePosixPath(source.relative_path).as_posix()
    manifest_relative = PurePosixPath(relative) if isinstance(relative, str) else None
    if (
        manifest_relative is None
        or manifest_relative.is_absolute()
        or ".." in manifest_relative.parts
        or "\\" in relative
        or not (
            source_relative == relative or source_relative.endswith(f"/{relative}")
        )
    ):
        raise Phase7CompilationError(
            f"Phase 4 manifest path mismatch for {producer_table_id}"
        )
    if (
        entry.get("file_sha256") != source.file_sha256
        or entry.get("row_count") != source_row_count
        or tuple(entry.get("columns", ())) != tuple(source_columns)
        or entry.get("release_class") != "public"
    ):
        raise Phase7CompilationError(
            f"Phase 4 manifest metadata mismatch for {producer_table_id}"
        )


def _artifact_inventory(registry: Phase7SourceRegistry) -> dict[str, SourceArtifactSpec]:
    return {
        artifact.artifact_id: artifact
        for predecessor in registry.predecessors
        for artifact in predecessor.artifacts
    }


def _artifact_families(
    artifact_ids: Sequence[str], artifacts: Mapping[str, SourceArtifactSpec]
) -> frozenset[SourceFamily]:
    return frozenset(artifacts[item].family for item in artifact_ids)


def _study_status(registry: Phase7SourceRegistry) -> ReportStatus:
    statuses = {item.status for item in registry.predecessors} | {
        item.status for item in registry.phases
    } | {item.status for item in registry.tables}
    if statuses == {ReportStatus.COMPLETE} and not registry.dirty_worktree:
        if registry.model_repository is None or registry.model_revision is None:
            raise Phase7CompilationError("a complete registry requires the frozen model revision")
        return ReportStatus.COMPLETE
    return ReportStatus.BLOCKED if ReportStatus.BLOCKED in statuses else ReportStatus.INCOMPLETE


def _study_status_source_artifact_ids(
    phases: Sequence[PhaseInputState],
) -> tuple[str, ...]:
    """Return the immutable lineage of the generated phase-status rows."""

    return tuple(
        sorted(
            {
                artifact_id
                for phase in phases
                for artifact_id in phase.source_artifact_ids
            }
        )
    )


def _phase5_feedback_csv_bytes(rows: Sequence[Phase5FeedbackTableRow]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=FEEDBACK_TABLE_COLUMNS,
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        payload = row.model_dump(mode="json", exclude={"schema_version", "content_hash"})
        writer.writerow({column: payload[column] for column in FEEDBACK_TABLE_COLUMNS})
    return stream.getvalue().encode("utf-8")


def _verify_phase5_feedback_table_binding(
    binding: Phase7TableBinding,
    *,
    artifacts: Mapping[str, SourceArtifactSpec],
    public_artifact_ids: set[str],
    source_root: Path,
) -> None:
    """Bind the complete public feedback CSV to its typed producer receipt."""

    if binding.source_table_artifact_id is None:
        raise Phase7CompilationError("complete feedback table lacks its source CSV")
    if binding.source_receipt_artifact_id is None:
        raise Phase7CompilationError("complete feedback table lacks its producer receipt")
    source = artifacts[binding.source_table_artifact_id]
    receipt_artifact = artifacts[binding.source_receipt_artifact_id]
    if (
        source.family is not SourceFamily.FEEDBACK
        or receipt_artifact.family is not SourceFamily.FEEDBACK
        or source.media_type != "text/csv"
        or receipt_artifact.media_type != "application/json"
        or source.release_class != "public"
        or receipt_artifact.release_class != "public"
    ):
        raise Phase7CompilationError(
            "feedback table and producer receipt must be public feedback-family artifacts"
        )
    if not {
        binding.source_table_artifact_id,
        binding.source_receipt_artifact_id,
    }.issubset(public_artifact_ids):
        raise Phase7CompilationError(
            "feedback table and producer receipt must both enter the public bundle"
        )
    if (
        receipt_artifact.logical_hash_field != "content_hash"
        or receipt_artifact.logical_hash_mode != "canonical_without_field"
        or receipt_artifact.logical_hash is None
    ):
        raise Phase7CompilationError(
            "feedback producer receipt lacks its canonical content-hash contract"
        )
    if binding.row_filters:
        raise Phase7CompilationError("the feedback producer table cannot be filtered")

    receipt_path = _safe_source(source_root, receipt_artifact.relative_path)
    receipt_bytes = receipt_path.read_bytes()
    if hashlib.sha256(receipt_bytes).hexdigest() != receipt_artifact.file_sha256:
        raise Phase7CompilationError("feedback producer receipt physical hash mismatch")
    try:
        receipt = Phase5FeedbackTableReceipt.model_validate_json(receipt_bytes)
    except ValueError as error:
        raise Phase7CompilationError("invalid Phase 5 feedback producer receipt") from error
    if receipt_bytes != (receipt.to_canonical_json() + "\n").encode("utf-8"):
        raise Phase7CompilationError("Phase 5 feedback producer receipt is noncanonical")
    if receipt.content_hash != receipt_artifact.logical_hash:
        raise Phase7CompilationError("feedback producer receipt logical hash mismatch")

    source_path = _safe_source(source_root, source.relative_path)
    source_bytes = source_path.read_bytes()
    if (
        hashlib.sha256(source_bytes).hexdigest() != source.file_sha256
        or receipt.table_file_sha256 != source.file_sha256
    ):
        raise Phase7CompilationError("feedback CSV differs from its producer receipt")
    try:
        decoded = source_bytes.decode("utf-8")
        reader = csv.DictReader(decoded.splitlines())
        columns = tuple(reader.fieldnames or ())
        raw_rows = tuple(dict(row) for row in reader)
    except (UnicodeDecodeError, csv.Error) as error:
        raise Phase7CompilationError("feedback CSV is not valid UTF-8 CSV") from error
    if columns != FEEDBACK_TABLE_COLUMNS or columns != receipt.table_columns:
        raise Phase7CompilationError("feedback CSV columns differ from its producer receipt")
    if (
        binding.source_row_count != receipt.table_row_count
        or binding.output_row_count != receipt.table_row_count
        or len(raw_rows) != receipt.table_row_count
    ):
        raise Phase7CompilationError("feedback CSV row count differs from its producer receipt")
    try:
        rows = tuple(
            Phase5FeedbackTableRow.model_validate(
                {
                    column: (
                        None
                        if column in {"value", "numerator", "denominator"} and value == ""
                        else value
                    )
                    for column, value in raw.items()
                }
            )
            for raw in raw_rows
        )
    except ValueError as error:
        raise Phase7CompilationError("feedback CSV contains an invalid typed row") from error
    if source_bytes != _phase5_feedback_csv_bytes(rows):
        raise Phase7CompilationError("feedback CSV is not in producer-canonical form")
    if canonical_record_sha256(rows) != receipt.table_logical_hash:
        raise Phase7CompilationError("feedback CSV logical hash differs from its receipt")

    keys = tuple(
        (item.episode_class.value, item.condition.value, item.action.value, item.metric)
        for item in rows
    )
    if keys != tuple(sorted(set(keys))):
        raise Phase7CompilationError("feedback CSV row keys are not canonical and unique")
    scripted_rows = tuple(
        item for item in rows if item.episode_class.value == "scripted_known_answer"
    )
    trace_rows = tuple(
        item for item in rows if item.episode_class.value == "researcher_trace"
    )
    expected_scripted_keys = {
        (condition.value, action.value, metric)
        for condition in (
            ConditionName.C0_CLASSICAL_PRE,
            ConditionName.C1_LLM_PRE,
            ConditionName.C2_LLM_QUERY,
        )
        for action in (FeedbackAction.REFINE_CONTEXT, FeedbackAction.REQUEST_MERGE_SPLIT)
        for metric in SCRIPTED_METRICS
    }
    if (
        len(scripted_rows) != receipt.scripted_table_row_count
        or {
            (item.condition.value, item.action.value, item.metric)
            for item in scripted_rows
        }
        != expected_scripted_keys
        or any(item.episode_count != 3 for item in scripted_rows)
    ):
        raise Phase7CompilationError("feedback CSV lacks its exact 6-by-3 scripted inventory")
    trace_groups = {(item.condition.value, item.action.value) for item in trace_rows}
    expected_trace_keys = {
        (*group, metric) for group in trace_groups for metric in TRACE_METRICS
    }
    if (
        len(trace_groups) != receipt.trace_action_group_count
        or len(trace_rows) != receipt.researcher_trace_table_row_count
        or {
            (item.condition.value, item.action.value, item.metric) for item in trace_rows
        }
        != expected_trace_keys
        or any(
            sum(item.episode_count for item in trace_rows if item.metric == metric) != 3
            for metric in TRACE_METRICS
        )
    ):
        raise Phase7CompilationError("feedback CSV lacks exactly three gold-NA trace scores")


def _validate_novel_case_rows(rows: Sequence[Mapping[str, str]]) -> None:
    identities = tuple(
        tuple(row[column] for column in NOVEL_CASE_COLUMNS[:5]) for row in rows
    )
    if len(identities) != len(set(identities)):
        raise Phase7CompilationError("novel-case report rows are not uniquely keyed")
    bounded = tuple(row for row in rows if row["scope"] == "bounded_same_evidence")
    operational = tuple(
        row for row in rows if row["scope"] == "full_index_operational_noncausal"
    )
    if len(bounded) + len(operational) != len(rows):
        raise Phase7CompilationError("novel-case report contains an unregistered scope")

    context_pairs = {(row["window_id"], row["context_id"]) for row in bounded}
    window_contexts: dict[str, set[str]] = {}
    for window_id, context_id in context_pairs:
        window_contexts.setdefault(window_id, set()).add(context_id)
    if len(context_pairs) != 8 or len(window_contexts) != 4 or any(
        len(context_ids) != 2 for context_ids in window_contexts.values()
    ):
        raise Phase7CompilationError(
            "novel-case bounded rows lack four windows and eight context units"
        )

    conditions = tuple(
        item.value
        for item in (
            ConditionName.C0_CLASSICAL_PRE,
            ConditionName.C1_LLM_PRE,
            ConditionName.C2_LLM_QUERY,
        )
    )
    review_metrics = tuple(
        f"review_{dimension.value}" for dimension in CaseReviewDimension
    )
    expected_review = {
        (window_id, context_id, condition, metric)
        for window_id, context_id in context_pairs
        for condition in conditions
        for metric in review_metrics
    }
    observed_review = {
        (row["window_id"], row["context_id"], row["condition"], row["metric"])
        for row in bounded
        if row["metric"] in review_metrics
    }
    if observed_review != expected_review:
        raise Phase7CompilationError(
            "novel-case report lacks the exact 8-by-3-by-5 primary review inventory"
        )

    detailed_metrics = {
        f"detailed_{subject}_{measure}"
        for subject in ("assertion", "event")
        for measure in ("f1", "precision", "recall")
    }
    detailed = tuple(row for row in bounded if row["metric"] in detailed_metrics)
    detailed_pairs = {(row["window_id"], row["context_id"]) for row in detailed}
    expected_detailed = {
        (window_id, context_id, condition, metric)
        for window_id, context_id in detailed_pairs
        for condition in conditions
        for metric in detailed_metrics
    }
    observed_detailed = {
        (row["window_id"], row["context_id"], row["condition"], row["metric"])
        for row in detailed
    }
    if len(detailed_pairs) != 4 or observed_detailed != expected_detailed:
        raise Phase7CompilationError(
            "novel-case report lacks detailed matching for exactly four context units"
        )

    agreement_metrics = {
        f"second_reader_agreement_{dimension.value}"
        for dimension in CaseReviewDimension
    }
    allowed_bounded_metrics = set(review_metrics) | detailed_metrics | agreement_metrics
    if any(row["metric"] not in allowed_bounded_metrics for row in bounded):
        raise Phase7CompilationError("novel-case report contains an unregistered review metric")

    operational_metrics = {
        "horizon_rejected_match_count",
        "omitted_by_cap_count",
        "output_succeeded",
        "retrieved_evidence_count",
    }
    if (
        len(operational) != 4
        or len({row["window_id"] for row in operational}) != 1
        or len({row["context_id"] for row in operational}) != 1
        or {row["condition"] for row in operational} != {ConditionName.C2_LLM_QUERY.value}
        or {row["metric"] for row in operational} != operational_metrics
    ):
        raise Phase7CompilationError(
            "novel-case report lacks its four-row full-index operational inventory"
        )


def _verify_novel_case_table_binding(
    binding: Phase7TableBinding,
    *,
    contract: TableContract,
    artifacts: Mapping[str, SourceArtifactSpec],
    public_artifact_ids: set[str],
    source_root: Path,
) -> None:
    """Bind complete narrative rows to the restricted typed analysis receipt."""

    if binding.source_table_artifact_id is None:
        raise Phase7CompilationError("complete novel-case table lacks its source CSV")
    if binding.source_receipt_artifact_id is None:
        raise Phase7CompilationError("complete novel-case table lacks its producer receipt")
    source = artifacts[binding.source_table_artifact_id]
    receipt_artifact = artifacts[binding.source_receipt_artifact_id]
    if (
        source.family is not SourceFamily.CASE_STUDY
        or receipt_artifact.family is not SourceFamily.CASE_STUDY
        or source.media_type != "text/csv"
        or receipt_artifact.media_type != "application/json"
        or source.release_class != "public"
        or receipt_artifact.release_class != "restricted"
        or binding.source_table_artifact_id not in public_artifact_ids
        or binding.source_receipt_artifact_id in public_artifact_ids
    ):
        raise Phase7CompilationError(
            "novel-case CSV must be public and its same-family receipt restricted"
        )
    if (
        receipt_artifact.logical_hash_field != "content_hash"
        or receipt_artifact.logical_hash_mode != "canonical_without_field"
        or receipt_artifact.logical_hash is None
    ):
        raise Phase7CompilationError(
            "novel-case producer receipt lacks its canonical content-hash contract"
        )
    if binding.row_filters:
        raise Phase7CompilationError("novel-case producer table cannot be filtered")

    receipt_path = _safe_source(source_root, receipt_artifact.relative_path)
    receipt_bytes = receipt_path.read_bytes()
    if hashlib.sha256(receipt_bytes).hexdigest() != receipt_artifact.file_sha256:
        raise Phase7CompilationError("novel-case producer receipt physical hash mismatch")
    try:
        receipt = CaseStudyNarrativeAnalysisReceipt.model_validate_json(receipt_bytes)
    except ValueError as error:
        raise Phase7CompilationError("invalid novel-case producer receipt") from error
    if receipt_bytes != (receipt.to_canonical_json() + "\n").encode("utf-8"):
        raise Phase7CompilationError("novel-case producer receipt is noncanonical")
    if receipt.content_hash != receipt_artifact.logical_hash:
        raise Phase7CompilationError("novel-case producer receipt logical hash mismatch")
    if receipt.publication_status != "public_scan_passed":
        raise Phase7CompilationError("novel-case producer receipt has not passed release scan")

    assert binding.source_row_count is not None
    assert binding.output_row_count is not None
    columns, rows, _payload = _canonical_source_csv(
        _safe_source(source_root, source.relative_path),
        expected_hash=source.file_sha256,
        expected_rows=binding.source_row_count,
        expected_output_rows=binding.output_row_count,
        row_filters=binding.row_filters,
        contract=contract,
    )
    if columns != NOVEL_CASE_COLUMNS:
        raise Phase7CompilationError("novel-case CSV columns differ from its producer")
    if (
        receipt.public_table_file_sha256 != source.file_sha256
        or receipt.restricted_table_file_sha256 != source.file_sha256
        or receipt.public_row_count != len(rows)
        or binding.source_row_count != receipt.public_row_count
        or binding.output_row_count != receipt.public_row_count
    ):
        raise Phase7CompilationError("novel-case CSV differs from its producer receipt")
    inventory_hash = canonical_record_sha256(
        sorted(
            rows,
            key=lambda row: tuple(row[column] for column in NOVEL_CASE_COLUMNS),
        )
    )
    if inventory_hash != receipt.public_row_inventory_hash:
        raise Phase7CompilationError("novel-case CSV row inventory differs from its receipt")
    _validate_novel_case_rows(rows)


def _verify_final_accounting_table_bindings(
    registry: Phase7SourceRegistry,
    *,
    artifacts: Mapping[str, SourceArtifactSpec],
    public_artifact_ids: set[str],
    source_root: Path,
) -> None:
    """Bind both public accounting CSVs to one exact public runtime receipt."""

    table_ids = ("failure_accounting", "resource_accounting")
    bindings = {
        item.table_id: item for item in registry.tables if item.table_id in table_ids
    }
    claimed = any(
        item.status is ReportStatus.COMPLETE
        or item.source_receipt_artifact_id is not None
        for item in bindings.values()
    )
    if not claimed:
        return
    if any(bindings[table_id].status is not ReportStatus.COMPLETE for table_id in table_ids):
        raise Phase7CompilationError(
            "final failure/resource accounting tables must become complete together"
        )
    receipt_ids = {
        bindings[table_id].source_receipt_artifact_id for table_id in table_ids
    }
    if None in receipt_ids or len(receipt_ids) != 1:
        raise Phase7CompilationError(
            "final accounting tables must share one compilation receipt"
        )
    receipt_id = next(iter(receipt_ids))
    assert receipt_id is not None
    receipt_artifact = artifacts[receipt_id]
    if (
        receipt_artifact.family is not SourceFamily.RUNTIME
        or receipt_artifact.media_type != "application/json"
        or receipt_artifact.release_class != "public"
        or receipt_artifact.logical_hash_field != "manifest_sha256"
        or receipt_artifact.logical_hash_mode != "canonical_without_field"
        or receipt_artifact.logical_hash is None
        or receipt_id not in public_artifact_ids
    ):
        raise Phase7CompilationError(
            "final accounting receipt must be a self-hashed public runtime artifact"
        )
    receipt_path = _safe_source(source_root, receipt_artifact.relative_path)
    try:
        receipt_payload = _load_self_hashed_json(
            receipt_path,
            "manifest_sha256",
            "final accounting compilation receipt",
        )
        receipt = _FinalAccountingCompilationReceipt.model_validate(receipt_payload)
    except ValueError as error:
        raise Phase7CompilationError(
            f"invalid final accounting compilation receipt: {error}"
        ) from error
    if (
        _file_sha256(receipt_path) != receipt_artifact.file_sha256
        or receipt.manifest_sha256 != receipt_artifact.logical_hash
        or receipt_path.name
        != f"final_accounting_receipt.{receipt.manifest_sha256[:16]}.json"
    ):
        raise Phase7CompilationError("final accounting receipt identity changed")

    recipe_matches = tuple(
        (artifact_id, artifact)
        for artifact_id, artifact in artifacts.items()
        if artifact.family is SourceFamily.RUNTIME
        and artifact.media_type == "application/json"
        and artifact.release_class == "restricted"
        and artifact.logical_hash_field == "manifest_sha256"
        and artifact.logical_hash_mode == "canonical_without_field"
        and artifact.logical_hash == receipt.source_recipe_manifest_sha256
        and artifact.file_sha256 == receipt.source_recipe_file_sha256
    )
    if len(recipe_matches) != 1:
        raise Phase7CompilationError(
            "final accounting receipt lacks one exact restricted runtime recipe"
        )
    recipe_id, recipe_artifact = recipe_matches[0]
    if recipe_id in public_artifact_ids or any(
        recipe_id not in bindings[table_id].source_artifact_ids
        for table_id in table_ids
    ):
        raise Phase7CompilationError(
            "final accounting tables do not bind their restricted compiler recipe"
        )
    recipe_path = _safe_source(source_root, recipe_artifact.relative_path)
    if recipe_path.name != (
        f"final_accounting_recipe.{receipt.source_recipe_manifest_sha256[:16]}.json"
    ):
        raise Phase7CompilationError("final accounting recipe identity changed")

    outputs = {item.table_id: item for item in receipt.outputs}
    source_paths: dict[str, Path] = {}
    for table_id in table_ids:
        binding = bindings[table_id]
        _validate_final_accounting_table_selection(
            table_id=table_id,
            row_filters=binding.row_filters,
            producer_manifest_artifact_id=binding.producer_manifest_artifact_id,
            singleton_join_artifact_id=binding.singleton_join_artifact_id,
            singleton_join_row_count=binding.singleton_join_row_count,
            supplemental_metric_sources=binding.supplemental_metric_sources,
        )
        source_id = binding.source_table_artifact_id
        if source_id is None:
            raise Phase7CompilationError(
                f"complete {table_id} table lacks its producer CSV"
            )
        source = artifacts[source_id]
        if (
            source.family is not SourceFamily.RUNTIME
            or source.media_type != "text/csv"
            or source.release_class != "public"
            or source.logical_hash is not None
            or source_id not in public_artifact_ids
        ):
            raise Phase7CompilationError(
                f"{table_id} must be a public runtime CSV in the public bundle"
            )
        output = outputs[table_id]
        expected_columns = {
            "failure_accounting": FAILURE_ACCOUNTING_COLUMNS,
            "resource_accounting": RESOURCE_ACCOUNTING_COLUMNS,
        }[table_id]
        source_path = _safe_source(source_root, source.relative_path)
        source_paths[table_id] = source_path
        rows = _read_canonical_review_csv(
            source_root=source_root,
            artifact=source,
            columns=expected_columns,
            label=table_id,
        )
        if (
            source_path.parent != receipt_path.parent
            or source_path.name != output.relative_path
            or source_path.name != f"{table_id}.{source.file_sha256[:16]}.csv"
            or output.file_sha256 != source.file_sha256
            or output.row_count != len(rows)
            or binding.source_row_count != len(rows)
            or binding.output_row_count != len(rows)
        ):
            raise Phase7CompilationError(
                f"{table_id} CSV differs from the final accounting receipt"
            )
    try:
        replay = compile_final_accounting(
            recipe_path=recipe_path,
            source_root=source_root,
            output_root=receipt_path.parent,
            verify_only=True,
        )
    except FinalAccountingError as error:
        raise Phase7CompilationError(
            f"final accounting producer replay failed: {error}"
        ) from error
    if (
        replay.failure_table_path != source_paths["failure_accounting"]
        or replay.resource_table_path != source_paths["resource_accounting"]
        or replay.receipt_path != receipt_path
        or replay.receipt != receipt_payload
    ):
        raise Phase7CompilationError(
            "final accounting receipt/tables differ from compiler replay"
        )


def _verify_public_reviewed_gold_binding(
    registry: Phase7SourceRegistry,
    *,
    artifacts: Mapping[str, SourceArtifactSpec],
    public_artifact_ids: set[str],
    source_root: Path,
) -> None:
    """Reproduce and bind the public final gold to the restricted review completion."""

    completion_id = registry.independent_review_completion_manifest_artifact_id
    publication_id = registry.public_reviewed_gold_manifest_artifact_id
    if completion_id is None or publication_id is None:
        raise Phase7CompilationError(
            "complete Phase 7 input lacks reviewed-gold completion/publication manifests"
        )
    completion_spec = artifacts[completion_id]
    publication_spec = artifacts[publication_id]
    if (
        completion_spec.family is not SourceFamily.HELD_OUT
        or completion_spec.media_type != "application/json"
        or completion_spec.release_class != "restricted"
        or completion_spec.logical_hash_field != "content_hash"
        or completion_spec.logical_hash_mode != "immutable_record"
        or completion_spec.logical_hash is None
    ):
        raise Phase7CompilationError(
            "review completion manifest must be a canonical restricted held-out artifact"
        )
    if (
        publication_spec.family is not SourceFamily.HELD_OUT
        or publication_spec.media_type != "application/json"
        or publication_spec.release_class != "public"
        or publication_spec.logical_hash_field != "content_hash"
        or publication_spec.logical_hash_mode != "immutable_record"
        or publication_spec.logical_hash is None
        or publication_id not in public_artifact_ids
    ):
        raise Phase7CompilationError(
            "reviewed-gold publication manifest must be canonical, public, and bundled"
        )
    expected_completion_path = (
        "artifacts/restricted/scorer_only/independent_review/completion_manifest.json"
    )
    expected_publication_path = (
        "artifacts/public/scorer_only/final_reviewed_gold/publication_manifest.json"
    )
    if (
        completion_spec.relative_path != expected_completion_path
        or publication_spec.relative_path != expected_publication_path
    ):
        raise Phase7CompilationError(
            "reviewed-gold manifests differ from their fixed restricted/public paths"
        )
    completion_path = _safe_source(source_root, completion_spec.relative_path)
    publication_path = _safe_source(source_root, publication_spec.relative_path)
    try:
        completion = load_completed_review(
            benchmark_root=_safe_source_directory(source_root, "data/synthetic"),
            output_root=completion_path.parent,
        )
        publication = load_public_reviewed_gold_publication(
            completion=completion,
            output_root=publication_path.parent,
        )
    except (IndependentReviewGateError, ReviewCompletionError) as error:
        raise Phase7CompilationError(
            f"reviewed-gold publication does not reproduce: {error}"
        ) from error
    if (
        completion.manifest.content_hash != completion_spec.logical_hash
        or hashlib.sha256(completion_path.read_bytes()).hexdigest()
        != completion_spec.file_sha256
        or publication.manifest.content_hash != publication_spec.logical_hash
        or hashlib.sha256(publication_path.read_bytes()).hexdigest()
        != publication_spec.file_sha256
        or publication.manifest.source_completion_manifest_hash
        != completion.manifest.content_hash
        or publication.manifest.source_completion_manifest_file_sha256
        != completion_spec.file_sha256
        or publication.manifest.final_seal_hash != completion.final_seal.content_hash
    ):
        raise Phase7CompilationError(
            "reviewed-gold public manifest differs from its restricted completion"
        )

    records_by_relative_path: dict[str, BaseModel] = {
        expected_publication_path: publication.manifest,
        (
            "artifacts/public/scorer_only/final_reviewed_gold/final_seal.json"
        ): publication.final_seal,
    }
    artifacts_by_blind = {
        item.blind_projection_id: item for item in publication.reviewed_artifacts
    }
    for entry in publication.manifest.reviewed_artifacts:
        records_by_relative_path[
            f"artifacts/public/scorer_only/final_reviewed_gold/{entry.artifact_file}"
        ] = artifacts_by_blind[entry.blind_projection_id]
    artifact_specs_by_path: dict[str, list[tuple[str, SourceArtifactSpec]]] = {}
    for artifact_id, artifact in artifacts.items():
        artifact_specs_by_path.setdefault(artifact.relative_path, []).append(
            (artifact_id, artifact)
        )
    for relative_path, record in records_by_relative_path.items():
        matches = artifact_specs_by_path.get(relative_path, [])
        if len(matches) != 1:
            raise Phase7CompilationError(
                "each finalized reviewed-gold public file must be registered exactly once"
            )
        artifact_id, artifact = matches[0]
        if (
            artifact.family is not SourceFamily.HELD_OUT
            or artifact.media_type != "application/json"
            or artifact.release_class != "public"
            or artifact.logical_hash_field != "content_hash"
            or artifact.logical_hash_mode != "immutable_record"
            or artifact.logical_hash != record.content_hash
            or artifact_id not in public_artifact_ids
            or _file_sha256(_safe_source(source_root, relative_path))
            != artifact.file_sha256
        ):
            raise Phase7CompilationError(
                "finalized reviewed-gold file lacks exact public registry lineage"
            )
    phase_2 = next(item for item in registry.phases if item.phase_id == "phase_2")
    if completion_id not in phase_2.source_artifact_ids:
        raise Phase7CompilationError(
            "Phase 2 status omits independent-review completion lineage"
        )
    phase_7 = next(item for item in registry.phases if item.phase_id == "phase_7")
    if publication_id not in phase_7.source_artifact_ids:
        raise Phase7CompilationError(
            "Phase 7 status omits public reviewed-gold publication lineage"
        )


def _require_review_artifact(
    artifact_id: str,
    *,
    artifacts: Mapping[str, SourceArtifactSpec],
    media_type: Literal["application/json", "text/csv"],
    release_class: Literal["public", "restricted"],
    label: str,
) -> SourceArtifactSpec:
    artifact = artifacts[artifact_id]
    if (
        artifact.family is not SourceFamily.HELD_OUT
        or artifact.media_type != media_type
        or artifact.release_class != release_class
    ):
        raise Phase7CompilationError(
            f"{label} must be a {release_class} held-out {media_type} artifact"
        )
    if media_type == "application/json" and (
        artifact.logical_hash_field != "content_hash"
        or artifact.logical_hash_mode != "immutable_record"
        or artifact.logical_hash is None
    ):
        raise Phase7CompilationError(
            f"{label} must use the producer's immutable-record content hash"
        )
    if media_type == "text/csv" and artifact.logical_hash is not None:
        raise Phase7CompilationError(f"{label} CSV must use only its physical file hash")
    return artifact


def _load_bound_review_record(
    *,
    source_root: Path,
    artifact: SourceArtifactSpec,
    model_type: type[BaseModel],
    label: str,
) -> BaseModel:
    path = _safe_source(source_root, artifact.relative_path)
    if _file_sha256(path) != artifact.file_sha256:
        raise Phase7CompilationError(f"{label} physical hash changed")
    try:
        record = model_type.model_validate_json(path.read_bytes())
    except Exception as error:
        raise Phase7CompilationError(f"invalid {label}: {error}") from error
    if getattr(record, "content_hash", None) != artifact.logical_hash:
        raise Phase7CompilationError(f"{label} logical hash changed")
    return record


def _read_canonical_review_csv(
    *,
    source_root: Path,
    artifact: SourceArtifactSpec,
    columns: tuple[str, ...],
    label: str,
) -> tuple[dict[str, str], ...]:
    path = _safe_source(source_root, artifact.relative_path)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != artifact.file_sha256:
        raise Phase7CompilationError(f"{label} physical hash changed")
    if raw.startswith(b"\xef\xbb\xbf") or b"\r" in raw:
        raise Phase7CompilationError(f"{label} is not canonical UTF-8/LF CSV")
    try:
        reader = csv.DictReader(raw.decode("utf-8").splitlines())
    except UnicodeDecodeError as error:
        raise Phase7CompilationError(f"{label} is not UTF-8") from error
    if tuple(reader.fieldnames or ()) != columns:
        raise Phase7CompilationError(f"{label} columns differ from its producer")
    rows = tuple(dict(row) for row in reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise Phase7CompilationError(f"{label} has a ragged row")
    regenerated = _csv_bytes(columns, rows)
    if regenerated != raw:
        raise Phase7CompilationError(f"{label} serialization differs from its producer")
    return rows


def _verify_error_review_binding(
    registry: Phase7SourceRegistry,
    configuration: Phase7CompilerConfiguration,
    *,
    artifacts: Mapping[str, SourceArtifactSpec],
    source_root: Path,
) -> tuple[ErrorReviewFinalization, tuple[dict[str, str], ...]]:
    binding = registry.error_review_artifacts
    if binding is None:
        raise Phase7CompilationError("complete reporting lacks the blinded error-review chain")
    json_roles = (
        (binding.source_manifest_artifact_id, HeldOutFailureSourceManifest, "error source"),
        (binding.package_artifact_id, BlindedErrorReviewPackage, "error package"),
        (binding.rejoin_map_artifact_id, ErrorReviewRejoinMap, "error rejoin map"),
        (binding.completion_artifact_id, ErrorReviewCompletion, "error completion"),
        (binding.adjudication_artifact_id, ErrorReviewAdjudication, "error adjudication"),
        (binding.finalization_artifact_id, ErrorReviewFinalization, "error finalization"),
    )
    records: dict[str, BaseModel] = {}
    specs: dict[str, SourceArtifactSpec] = {}
    for artifact_id, model_type, label in json_roles:
        spec = _require_review_artifact(
            artifact_id,
            artifacts=artifacts,
            media_type="application/json",
            release_class="restricted",
            label=label,
        )
        specs[label] = spec
        records[label] = _load_bound_review_record(
            source_root=source_root,
            artifact=spec,
            model_type=model_type,
            label=label,
        )
    table_spec = _require_review_artifact(
        binding.canonical_table_artifact_id,
        artifacts=artifacts,
        media_type="text/csv",
        release_class="restricted",
        label="error-review table",
    )
    source = records["error source"]
    package = records["error package"]
    rejoin = records["error rejoin map"]
    finalization = records["error finalization"]
    assert isinstance(source, HeldOutFailureSourceManifest)
    assert isinstance(package, BlindedErrorReviewPackage)
    assert isinstance(rejoin, ErrorReviewRejoinMap)
    assert isinstance(finalization, ErrorReviewFinalization)
    if (
        package.source_manifest_hash != source.content_hash
        or rejoin.source_manifest_hash != source.content_hash
        or rejoin.package_hash != package.content_hash
    ):
        raise Phase7CompilationError("error-review source/package/rejoin lineage changed")
    package_path = _safe_source(source_root, specs["error package"].relative_path)
    package_root = package_path.parent.parent
    if (
        package_path.name != "review_package.json"
        or package_path.parent.name != "reviewer"
        or _safe_source(source_root, specs["error rejoin map"].relative_path)
        != package_root / "scorer_only/rejoin_map.json"
    ):
        raise Phase7CompilationError("error-review package paths differ from its producer")
    try:
        expected_package, expected_rejoin, _ = prepare_error_review_package(
            restricted_root=source_root,
            source_manifest_path=_safe_source(
                source_root, specs["error source"].relative_path
            ),
            taxonomy_path=_safe_source(
                source_root, configuration.error_review_taxonomy_path
            ),
        )
    except BlindedReviewError as error:
        raise Phase7CompilationError(f"error-review package does not reproduce: {error}") from error
    if expected_package != package or expected_rejoin != rejoin:
        raise Phase7CompilationError("error-review package/rejoin differs from producer replay")
    taxonomy = load_error_taxonomy(
        _safe_source(source_root, configuration.error_review_taxonomy_path)
    )
    if package.taxonomy_hash != taxonomy.content_hash:
        raise Phase7CompilationError("error-review package names another frozen taxonomy")
    final_path = _safe_source(source_root, specs["error finalization"].relative_path)
    final_root = final_path.parent
    expected_final_paths = {
        "error completion": final_root / "completion.json",
        "error adjudication": final_root / "adjudication.json",
        "error finalization": final_root / "finalization.json",
    }
    if any(
        _safe_source(source_root, specs[label].relative_path) != expected
        for label, expected in expected_final_paths.items()
    ) or _safe_source(source_root, table_spec.relative_path) != final_root / (
        finalization.canonical_table_file
    ):
        raise Phase7CompilationError("error-review final paths differ from its producer")
    try:
        expected_finalization, files = prepare_error_review_finalization(
            restricted_root=source_root,
            package_root=package_root,
            completion_path=final_root / "completion.json",
            adjudication_path=final_root / "adjudication.json",
        )
    except BlindedReviewError as error:
        raise Phase7CompilationError(
            f"error-review finalization does not reproduce: {error}"
        ) from error
    table_path = _safe_source(source_root, table_spec.relative_path)
    if (
        expected_finalization != finalization
        or files[finalization.canonical_table_file] != table_path.read_bytes()
        or finalization.canonical_table_sha256 != table_spec.file_sha256
    ):
        raise Phase7CompilationError("error-review finalization/table differs from producer replay")
    rows = _read_canonical_review_csv(
        source_root=source_root,
        artifact=table_spec,
        columns=_ERROR_REVIEW_COLUMNS,
        label="error-review table",
    )
    if (
        len(rows) != finalization.reviewed_failure_count
        or len({row["source_failure_id"] for row in rows}) != len(rows)
        or len({row["blind_item_id"] for row in rows}) != len(rows)
        or any(row["independent_unit"] != "world" for row in rows)
    ):
        raise Phase7CompilationError("error-review table inventory is incomplete or duplicated")
    valid_codes = {item.value for item in FROZEN_ERROR_CODES}
    for row in rows:
        codes = tuple(row["error_codes"].split("|"))
        if not codes or codes != tuple(sorted(set(codes))) or not set(codes) <= valid_codes:
            raise Phase7CompilationError("error-review table contains invalid taxonomy codes")
    return finalization, rows


def _verify_community_review_binding(
    registry: Phase7SourceRegistry,
    configuration: Phase7CompilerConfiguration,
    *,
    artifacts: Mapping[str, SourceArtifactSpec],
    source_root: Path,
) -> tuple[CommunityReviewFinalization, tuple[dict[str, str], ...]]:
    binding = registry.community_review_artifacts
    if binding is None:
        raise Phase7CompilationError(
            "complete reporting lacks the blinded community-review chain"
        )
    json_roles = (
        (binding.source_manifest_artifact_id, CommunityReviewSourceManifest, "community source"),
        (binding.package_artifact_id, BlindedCommunityReviewPackage, "community package"),
        (binding.rejoin_map_artifact_id, CommunityReviewRejoinMap, "community rejoin map"),
        (binding.completion_artifact_id, CommunityReviewCompletion, "community completion"),
        (binding.finalization_artifact_id, CommunityReviewFinalization, "community finalization"),
    )
    records: dict[str, BaseModel] = {}
    specs: dict[str, SourceArtifactSpec] = {}
    for artifact_id, model_type, label in json_roles:
        spec = _require_review_artifact(
            artifact_id,
            artifacts=artifacts,
            media_type="application/json",
            release_class="restricted",
            label=label,
        )
        specs[label] = spec
        records[label] = _load_bound_review_record(
            source_root=source_root,
            artifact=spec,
            model_type=model_type,
            label=label,
        )
    table_spec = _require_review_artifact(
        binding.canonical_table_artifact_id,
        artifacts=artifacts,
        media_type="text/csv",
        release_class="restricted",
        label="community-review table",
    )
    source = records["community source"]
    package = records["community package"]
    rejoin = records["community rejoin map"]
    finalization = records["community finalization"]
    assert isinstance(source, CommunityReviewSourceManifest)
    assert isinstance(package, BlindedCommunityReviewPackage)
    assert isinstance(rejoin, CommunityReviewRejoinMap)
    assert isinstance(finalization, CommunityReviewFinalization)
    if (
        package.source_manifest_hash != source.content_hash
        or rejoin.source_manifest_hash != source.content_hash
        or rejoin.package_hash != package.content_hash
    ):
        raise Phase7CompilationError("community-review source/package/rejoin lineage changed")
    package_path = _safe_source(source_root, specs["community package"].relative_path)
    package_root = package_path.parent.parent
    if (
        package_path.name != "review_package.json"
        or package_path.parent.name != "reviewer"
        or _safe_source(source_root, specs["community rejoin map"].relative_path)
        != package_root / "scorer_only/rejoin_map.json"
    ):
        raise Phase7CompilationError("community-review package paths differ from its producer")
    try:
        expected_package, expected_rejoin, _ = prepare_community_review_package(
            restricted_root=source_root,
            source_manifest_path=_safe_source(
                source_root, specs["community source"].relative_path
            ),
            rubric_template_path=_safe_source(
                source_root, configuration.community_review_template_path
            ),
        )
    except BlindedReviewError as error:
        raise Phase7CompilationError(
            f"community-review package does not reproduce: {error}"
        ) from error
    if expected_package != package or expected_rejoin != rejoin:
        raise Phase7CompilationError(
            "community-review package/rejoin differs from producer replay"
        )
    final_path = _safe_source(source_root, specs["community finalization"].relative_path)
    final_root = final_path.parent
    expected_final_paths = {
        "community completion": final_root / "completion.json",
        "community finalization": final_root / "finalization.json",
    }
    if any(
        _safe_source(source_root, specs[label].relative_path) != expected
        for label, expected in expected_final_paths.items()
    ) or _safe_source(source_root, table_spec.relative_path) != final_root / (
        finalization.canonical_table_file
    ):
        raise Phase7CompilationError("community-review final paths differ from its producer")
    try:
        expected_finalization, files = prepare_community_review_finalization(
            restricted_root=source_root,
            package_root=package_root,
            completion_path=final_root / "completion.json",
        )
    except BlindedReviewError as error:
        raise Phase7CompilationError(
            f"community-review finalization does not reproduce: {error}"
        ) from error
    table_path = _safe_source(source_root, table_spec.relative_path)
    if (
        expected_finalization != finalization
        or files[finalization.canonical_table_file] != table_path.read_bytes()
        or finalization.canonical_table_sha256 != table_spec.file_sha256
    ):
        raise Phase7CompilationError(
            "community-review finalization/table differs from producer replay"
        )
    rows = _read_canonical_review_csv(
        source_root=source_root,
        artifact=table_spec,
        columns=_COMMUNITY_REVIEW_COLUMNS,
        label="community-review table",
    )
    identities = {
        (row["condition"], row["seed_block"], row["resolution"]) for row in rows
    }
    if (
        len(rows) != finalization.reviewed_partition_count
        or finalization.reviewed_partition_count != 12
        or len(identities) != 12
        or len({(row["world_id"], row["context_id"]) for row in rows}) != 1
        or any(row["independent_unit"] != "world" for row in rows)
    ):
        raise Phase7CompilationError("community-review table is not the registered 12-row panel")
    for row in rows:
        try:
            scores = tuple(
                int(row[column])
                for column in (
                    "semantic_coherence",
                    "interpretability",
                    "evidence_support",
                )
            )
            mean = float(row["rubric_mean"])
        except ValueError as error:
            raise Phase7CompilationError("community-review rubric score is invalid") from error
        if any(value < 1 or value > 5 for value in scores) or not math.isclose(
            mean, sum(scores) / 3.0, abs_tol=1e-15
        ):
            raise Phase7CompilationError("community-review rubric mean is invalid")
    return finalization, rows


def _verify_qualitative_producer_binding(
    registry: Phase7SourceRegistry,
    configuration: Phase7CompilerConfiguration,
    *,
    artifacts: Mapping[str, SourceArtifactSpec],
    public_artifact_ids: set[str],
    source_root: Path,
    error_rows: tuple[dict[str, str], ...],
) -> None:
    binding = registry.qualitative_producer_artifacts
    if binding is None:
        raise Phase7CompilationError(
            "complete qualitative reporting lacks its materialization producer chain"
        )
    source_spec = artifacts[binding.source_manifest_artifact_id]
    candidate_spec = artifacts[binding.candidate_set_artifact_id]
    composite_spec = artifacts[binding.composite_manifest_artifact_id]
    receipt_spec = artifacts[binding.materialization_receipt_artifact_id]
    if (
        source_spec.media_type != "application/json"
        or source_spec.release_class != "restricted"
        or source_spec.logical_hash_field != "content_hash"
        or source_spec.logical_hash_mode != "immutable_record"
        or source_spec.logical_hash is None
    ):
        raise Phase7CompilationError(
            "qualitative source manifest must be a restricted immutable record"
        )
    if (
        candidate_spec.media_type != "application/json"
        or candidate_spec.release_class != "public"
        or candidate_spec.logical_hash_field != "content_hash"
        or candidate_spec.logical_hash_mode != "canonical_without_field"
        or candidate_spec.logical_hash is None
    ):
        raise Phase7CompilationError(
            "qualitative candidate set must use its canonical public producer hash"
        )
    for spec, label in (
        (composite_spec, "qualitative composite manifest"),
        (receipt_spec, "qualitative materialization receipt"),
    ):
        if (
            spec.media_type != "application/json"
            or spec.release_class != "public"
            or spec.logical_hash_field != "content_hash"
            or spec.logical_hash_mode != "immutable_record"
            or spec.logical_hash is None
        ):
            raise Phase7CompilationError(
                f"{label} must be a public immutable producer record"
            )
    output_ids = {
        binding.candidate_set_artifact_id,
        binding.composite_manifest_artifact_id,
        binding.materialization_receipt_artifact_id,
    }
    if not output_ids.issubset(public_artifact_ids):
        raise Phase7CompilationError(
            "qualitative producer manifests are absent from the public bundle"
        )
    source_path = _safe_source(source_root, source_spec.relative_path)
    candidate_path = _safe_source(source_root, candidate_spec.relative_path)
    output_root = candidate_path.parent
    if (
        candidate_path.name != "qualitative_candidate_set.json"
        or _safe_source(source_root, composite_spec.relative_path)
        != output_root / "qualitative_composite_manifest.json"
        or _safe_source(source_root, receipt_spec.relative_path)
        != output_root / "materialization_receipt.json"
    ):
        raise Phase7CompilationError(
            "qualitative producer outputs differ from their canonical sibling paths"
        )
    try:
        from story_projection_onto.scorer_only.qualitative_materializer import (
            QualitativeCompositeManifest,
            QualitativeMaterializationError,
            QualitativeMaterializationReceipt,
            QualitativeMaterializationSource,
            prepare_qualitative_materialization,
        )

        source = QualitativeMaterializationSource.model_validate_json(
            source_path.read_bytes()
        )
        candidate_set = load_qualitative_candidate_set(candidate_path)
        composite = QualitativeCompositeManifest.model_validate_json(
            _safe_source(source_root, composite_spec.relative_path).read_bytes()
        )
        receipt = QualitativeMaterializationReceipt.model_validate_json(
            _safe_source(source_root, receipt_spec.relative_path).read_bytes()
        )
        expected_candidate_set, expected_composite, expected_receipt, files = (
            prepare_qualitative_materialization(
                restricted_root=source_root,
                source_manifest_path=source_path,
                reporting_policy_path=_safe_source(
                    source_root, configuration.reporting_policy_path
                ),
            )
        )
    except (QualitativeMaterializationError, ValueError) as error:
        raise Phase7CompilationError(
            f"qualitative producer chain does not reproduce: {error}"
        ) from error
    if (
        source.content_hash != source_spec.logical_hash
        or candidate_set.content_hash != candidate_spec.logical_hash
        or composite.content_hash != composite_spec.logical_hash
        or receipt.content_hash != receipt_spec.logical_hash
        or candidate_set != expected_candidate_set
        or composite != expected_composite
        or receipt != expected_receipt
    ):
        raise Phase7CompilationError("qualitative producer records differ from replay")
    artifact_specs_by_path: dict[str, list[tuple[str, SourceArtifactSpec]]] = {}
    for artifact_id, artifact in artifacts.items():
        artifact_specs_by_path.setdefault(artifact.relative_path, []).append(
            (artifact_id, artifact)
        )
    output_prefix = PurePosixPath(candidate_spec.relative_path).parent
    for relative_path, payload in files.items():
        full_relative = (output_prefix / relative_path).as_posix()
        matches = artifact_specs_by_path.get(full_relative, [])
        if len(matches) != 1:
            raise Phase7CompilationError(
                "each qualitative producer output must be registered exactly once"
            )
        artifact_id, artifact = matches[0]
        if (
            artifact.release_class != "public"
            or artifact_id not in public_artifact_ids
            or artifact.file_sha256 != hashlib.sha256(payload).hexdigest()
            or _safe_source(source_root, full_relative).read_bytes() != payload
        ):
            raise Phase7CompilationError(
                "qualitative producer output lacks exact public registry lineage"
            )
    if set(files) != {
        "qualitative_candidate_set.json",
        "qualitative_composite_manifest.json",
        "materialization_receipt.json",
        *(item.relative_path for item in composite.figures),
    }:
        raise Phase7CompilationError("qualitative producer output inventory changed")
    source_parent = PurePosixPath(source_spec.relative_path).parent
    for panel in source.panels:
        matches = artifact_specs_by_path.get(
            (source_parent / panel.relative_path).as_posix(), []
        )
        if len(matches) != 1:
            raise Phase7CompilationError(
                "each qualitative source panel must be registered exactly once"
            )
        artifact_id, artifact = matches[0]
        if (
            artifact_id != panel.panel_artifact_id
            or artifact.media_type != "image/png"
            or artifact.file_sha256 != panel.file_sha256
        ):
            raise Phase7CompilationError("qualitative panel registry identity changed")
    for candidate in candidate_set.candidates:
        if not set(candidate.source_artifact_ids).issubset(artifacts):
            raise Phase7CompilationError(
                f"qualitative candidate {candidate.candidate_id} has unregistered lineage"
            )
    selected = _choose_qualitative_candidates(
        candidate_set,
        policy_path=_safe_source(source_root, configuration.reporting_policy_path),
    )
    selected_ids = tuple(candidate.candidate_id for _, candidate, _ in selected)
    selected_rules = tuple(rule_hash for _, _, rule_hash in selected)
    if (
        receipt.selected_candidate_ids != selected_ids
        or receipt.selected_example_ids != tuple(example.value for example, _, _ in selected)
        or receipt.selection_rule_hashes != selected_rules
        or composite.selected_candidate_ids != selected_ids
    ):
        raise Phase7CompilationError("qualitative selection receipt differs from frozen rules")
    counterexample = next(
        candidate
        for example, candidate, _ in selected
        if example is QualitativeExample.COUNTEREXAMPLE
    )
    assert counterexample.reviewed_failure_id is not None
    reviewed_matches = tuple(
        row
        for row in error_rows
        if row["source_failure_id"] == counterexample.reviewed_failure_id
    )
    if len(reviewed_matches) != 1:
        raise Phase7CompilationError(
            "qualitative counterexample does not identify exactly one reviewed failure row"
        )
    reviewed = reviewed_matches[0]
    if (
        len(counterexample.context_ids) != 1
        or reviewed["world_id"] != counterexample.world_id
        or reviewed["context_id"] != counterexample.context_ids[0]
        or reviewed["condition"] != "C2"
        or reviewed["seed_block"] not in {"1", "2"}
    ):
        raise Phase7CompilationError(
            "qualitative counterexample reviewed-failure identity changed"
        )
    error_binding = registry.error_review_artifacts
    assert error_binding is not None
    qualitative_table = next(
        item for item in registry.tables if item.table_id == "qualitative_examples"
    )
    required_lineage = set(binding.artifact_ids()) | set(error_binding.artifact_ids())
    if not required_lineage.issubset(qualitative_table.source_artifact_ids) or not set(
        error_binding.artifact_ids()
    ).issubset(counterexample.source_artifact_ids):
        raise Phase7CompilationError(
            "qualitative counterexample omits its producer/error-review lineage"
        )
    primary_table = next(
        item for item in registry.tables if item.table_id == "primary_c2_vs_c1"
    )
    metric_binding = next(
        (
            item
            for item in primary_table.supplemental_metric_sources
            if item.source_role == "primary_unit_temporal_epistemic"
        ),
        None,
    )
    if metric_binding is None:
        raise Phase7CompilationError(
            "qualitative counterexample lacks the complete primary ITT metric source"
        )
    metric_spec = artifacts[metric_binding.source_artifact_id]
    metric_path = _safe_source(source_root, metric_spec.relative_path)
    metric_reader = csv.DictReader(metric_path.read_text(encoding="utf-8").splitlines())
    score_rows = tuple(
        row
        for row in metric_reader
        if row["source_block"] == "primary"
        and row["condition"] == "C2"
        and row["metric_name"] == "strict_qualified_assertion_f1"
    )
    hard_worlds: set[str] = set()
    for number in range(1, 13):
        scorer_path = _safe_source(
            source_root,
            f"data/synthetic/scorer_only/held_out/syn-test-{number:02d}.json",
        )
        payload = json.loads(scorer_path.read_text(encoding="utf-8"))
        if payload["world_spec"]["difficulty"] == "hard":
            hard_worlds.add(payload["world_spec"]["world_id"])
    hard_rows = tuple(row for row in score_rows if row["world_id"] in hard_worlds)
    by_context: dict[tuple[str, str], dict[int, float]] = {}
    for row in hard_rows:
        if row["metric_status"] != "value" or row["seed_block"] not in {"1", "2"}:
            raise Phase7CompilationError("hard-stratum C2 ITT metric row is incomplete")
        key = (row["world_id"], row["context_id"])
        seed = int(row["seed_block"])
        if seed in by_context.setdefault(key, {}):
            raise Phase7CompilationError("hard-stratum C2 ITT metric row is duplicated")
        by_context[key][seed] = float(row["value"])
    if len(hard_worlds) != 4 or len(by_context) != 12 or any(
        set(values) != {1, 2} for values in by_context.values()
    ):
        raise Phase7CompilationError(
            "hard-stratum counterexample denominator is not 4 worlds by 3 contexts by 2 seeds"
        )
    selected_key = min(
        by_context,
        key=lambda key: (
            sum(by_context[key].values()) / 2.0,
            key[0],
            key[1],
        ),
    )
    candidate_scores = {
        item.seed_block: item.strict_qualified_assertion_f1
        for item in counterexample.c2_itt_scores
    }
    reviewed_score = next(
        item
        for item in counterexample.c2_itt_scores
        if item.seed_block == int(reviewed["seed_block"])
    )
    if (
        selected_key != (counterexample.world_id, counterexample.context_ids[0])
        or candidate_scores != by_context[selected_key]
        or reviewed_score.outcome != reviewed["output_status"]
        or not {
            metric_binding.source_artifact_id,
            metric_binding.table_manifest_artifact_id,
        }.issubset(counterexample.source_artifact_ids)
    ):
        raise Phase7CompilationError(
            "qualitative counterexample differs from the complete ITT/error sources"
        )


def _verify_registry_contracts(
    registry: Phase7SourceRegistry,
    configuration: Phase7CompilerConfiguration,
    *,
    source_root: Path,
) -> None:
    artifacts = _artifact_inventory(registry)
    contracts = {item.table_id: item for item in configuration.table_contracts}
    predecessor_status = {item.family: item.status for item in registry.predecessors}
    public_artifact_ids = set(registry.public_artifact_ids)
    phase4_table_manifest_hashes: set[str] = set()
    for binding in registry.tables:
        contract = contracts[binding.table_id]
        expected_producer = {
            "study_status": TableProducer.STUDY_STATUS,
            "qualitative_examples": TableProducer.QUALITATIVE_SELECTION,
        }.get(binding.table_id, TableProducer.CANONICAL_CSV)
        if contract.producer is not expected_producer:
            raise Phase7CompilationError(f"unexpected table producer: {binding.table_id}")
        if binding.status is not ReportStatus.COMPLETE:
            continue
        _validate_registered_table_binding(contract, binding)
        if binding.table_id == "mechanism_c2_vs_fixed" and (
            binding.row_filters != _MECHANISM_TABLE_FILTERS
            or binding.output_row_count != len(_MECHANISM_TABLE_ROW_IDENTITIES)
        ):
            raise Phase7CompilationError(
                "mechanism table binding lacks its frozen four-row inventory"
            )
        expected_metric_sources = {
            item.source_role: item for item in contract.supplemental_metric_sources
        }
        bound_metric_sources = {
            item.source_role: item for item in binding.supplemental_metric_sources
        }
        if set(bound_metric_sources) != set(expected_metric_sources):
            raise Phase7CompilationError(
                f"table {binding.table_id} does not bind its exact supplemental metric sources"
            )
        if binding.scope == "final":
            incomplete = {
                family
                for family in FINAL_TABLE_DEPENDENCIES[binding.table_id]
                if predecessor_status[family] is not ReportStatus.COMPLETE
            }
            if incomplete:
                raise Phase7CompilationError(
                    f"final table {binding.table_id} has incomplete predecessors: "
                    + ", ".join(sorted(item.value for item in incomplete))
                )
            missing = FINAL_TABLE_DEPENDENCIES[binding.table_id] - _artifact_families(
                binding.source_artifact_ids, artifacts
            )
            if binding.table_id != "study_status" and missing:
                raise Phase7CompilationError(
                    f"final table {binding.table_id} lacks source lineage for: "
                    + ", ".join(sorted(item.value for item in missing))
                )
        if binding.source_table_artifact_id is not None:
            source = artifacts[binding.source_table_artifact_id]
            if source.media_type != "text/csv":
                raise Phase7CompilationError(
                    f"source table artifact is not CSV: {binding.table_id}"
                )
        if binding.table_id == "feedback":
            _verify_phase5_feedback_table_binding(
                binding,
                artifacts=artifacts,
                public_artifact_ids=public_artifact_ids,
                source_root=source_root,
            )
        elif binding.table_id == "novel_case":
            _verify_novel_case_table_binding(
                binding,
                contract=contract,
                artifacts=artifacts,
                public_artifact_ids=public_artifact_ids,
                source_root=source_root,
            )
        elif (
            binding.source_receipt_artifact_id is not None
            and binding.table_id
            not in {"failure_accounting", "resource_accounting"}
        ):
            raise Phase7CompilationError(
                f"unexpected source receipt binding: {binding.table_id}"
            )
        if binding.singleton_join_artifact_id is not None:
            singleton = artifacts[binding.singleton_join_artifact_id]
            if singleton.media_type != "text/csv":
                raise Phase7CompilationError(
                    f"singleton join artifact is not CSV: {binding.table_id}"
                )
        manifest_ids: set[str] = set()
        if binding.producer_manifest_artifact_id is not None:
            producer_manifest = artifacts[binding.producer_manifest_artifact_id]
            assert binding.source_table_artifact_id is not None
            producer_source = artifacts[binding.source_table_artifact_id]
            expected_source_families = FINAL_TABLE_DEPENDENCIES[binding.table_id]
            if (
                producer_manifest.media_type != "application/json"
                or producer_manifest.release_class != "public"
                or producer_source.release_class != "public"
                or producer_manifest.family is not producer_source.family
                or producer_source.family not in expected_source_families
                or not {
                    binding.producer_manifest_artifact_id,
                    binding.source_table_artifact_id,
                }.issubset(public_artifact_ids)
            ):
                raise Phase7CompilationError(
                    f"table {binding.table_id} producer CSV/manifest must be public, "
                    "same-family, and bundled"
                )
            manifest_ids.add(binding.producer_manifest_artifact_id)
        for source_role, metric_binding in bound_metric_sources.items():
            metric_contract = expected_metric_sources[source_role]
            source = artifacts[metric_binding.source_artifact_id]
            manifest = artifacts[metric_binding.table_manifest_artifact_id]
            if (
                source.media_type != "text/csv"
                or source.release_class != "public"
                or manifest.media_type != "application/json"
                or manifest.release_class != "public"
            ):
                raise Phase7CompilationError(
                    f"supplemental metric source {source_role} must be public CSV plus manifest"
                )
            if not {
                metric_binding.source_artifact_id,
                metric_binding.table_manifest_artifact_id,
            }.issubset(public_artifact_ids):
                raise Phase7CompilationError(
                    f"supplemental metric source {source_role} is absent from the public bundle"
                )
            source_rows, selected_rows, metric_counts, columns, metric_version_hash = (
                _inspect_supplemental_metric_source_csv(
                    _safe_source(source_root, source.relative_path),
                    expected_hash=source.file_sha256,
                    contract=metric_contract,
                )
            )
            observed_binding = SupplementalMetricSourceBinding(
                source_role=source_role,
                source_artifact_id=metric_binding.source_artifact_id,
                table_manifest_artifact_id=metric_binding.table_manifest_artifact_id,
                source_row_count=source_rows,
                selected_row_count=selected_rows,
                metric_row_counts=metric_counts,
            )
            if observed_binding != metric_binding:
                raise Phase7CompilationError(
                    f"supplemental metric source evidence changed: {source_role}"
                )
            _verify_phase4_table_manifest_binding(
                _safe_source(source_root, manifest.relative_path),
                expected_manifest_hash=manifest.file_sha256,
                source=source,
                source_columns=columns,
                source_row_count=source_rows,
                producer_table_id=metric_contract.producer_table_id,
                metric_version_hash=metric_version_hash,
            )
            manifest_ids.add(metric_binding.table_manifest_artifact_id)
            phase4_table_manifest_hashes.add(manifest.file_sha256)
        if len(manifest_ids) > 1:
            raise Phase7CompilationError(
                f"table {binding.table_id} mixes Phase 4 table manifests"
            )
        if contract.source_producer_table_id is not None:
            assert binding.source_table_artifact_id is not None
            if not manifest_ids:
                raise Phase7CompilationError(
                    f"table {binding.table_id} lacks its Phase 4 table manifest"
                )
            manifest_id = next(iter(manifest_ids))
            source = artifacts[binding.source_table_artifact_id]
            manifest = artifacts[manifest_id]
            source_rows, source_columns = _canonical_csv_counts_for_manifest(
                _safe_source(source_root, source.relative_path),
                expected_hash=source.file_sha256,
            )
            _verify_phase4_table_manifest_binding(
                _safe_source(source_root, manifest.relative_path),
                expected_manifest_hash=manifest.file_sha256,
                source=source,
                source_columns=source_columns,
                source_row_count=source_rows,
                producer_table_id=contract.source_producer_table_id,
            )
    if len(phase4_table_manifest_hashes) > 1:
        raise Phase7CompilationError(
            "supplemental metric panels mix Phase 4 table manifests"
        )
    _verify_final_accounting_table_bindings(
        registry,
        artifacts=artifacts,
        public_artifact_ids=public_artifact_ids,
        source_root=source_root,
    )
    known = set(artifacts)
    for phase in registry.phases:
        if not set(phase.source_artifact_ids).issubset(known):
            raise Phase7CompilationError(f"phase {phase.phase_id} has unknown source lineage")
    review_binding_claimed = any(
        item is not None
        for item in (
            registry.independent_review_completion_manifest_artifact_id,
            registry.public_reviewed_gold_manifest_artifact_id,
        )
    )
    if _study_status(registry) is ReportStatus.COMPLETE or review_binding_claimed:
        _verify_public_reviewed_gold_binding(
            registry,
            artifacts=artifacts,
            public_artifact_ids=public_artifact_ids,
            source_root=source_root,
        )
    table_statuses = {item.table_id: item.status for item in registry.tables}
    study_complete = _study_status(registry) is ReportStatus.COMPLETE
    error_required = (
        study_complete
        or table_statuses["qualitative_examples"] is ReportStatus.COMPLETE
        or registry.error_review_artifacts is not None
        or registry.qualitative_producer_artifacts is not None
    )
    community_required = (
        study_complete
        or table_statuses["community"] is ReportStatus.COMPLETE
        or registry.community_review_artifacts is not None
    )
    qualitative_required = (
        study_complete
        or table_statuses["qualitative_examples"] is ReportStatus.COMPLETE
        or registry.qualitative_producer_artifacts is not None
    )
    error_rows: tuple[dict[str, str], ...] = ()
    if error_required:
        _, error_rows = _verify_error_review_binding(
            registry,
            configuration,
            artifacts=artifacts,
            source_root=source_root,
        )
    if community_required:
        _verify_community_review_binding(
            registry,
            configuration,
            artifacts=artifacts,
            source_root=source_root,
        )
        community_binding = registry.community_review_artifacts
        assert community_binding is not None
        community_table = next(
            item for item in registry.tables if item.table_id == "community"
        )
        if (
            community_table.status is ReportStatus.COMPLETE
            and not set(community_binding.artifact_ids()).issubset(
                community_table.source_artifact_ids
            )
        ):
            raise Phase7CompilationError(
                "community report table omits its blinded-rubric producer chain"
            )
    if qualitative_required:
        _verify_qualitative_producer_binding(
            registry,
            configuration,
            artifacts=artifacts,
            public_artifact_ids=public_artifact_ids,
            source_root=source_root,
            error_rows=error_rows,
        )


def _required_condition_set(candidate: QualitativeCandidate) -> set[str]:
    if candidate.evidence_split == "case_study":
        return {"c0_classical_pre", "c1_llm_pre", "c2_llm_query"}
    return set(_CONDITION_ORDER)


def _candidate_mean(candidate: QualitativeCandidate) -> float:
    if {item.seed_block for item in candidate.c2_itt_scores} != {1, 2}:
        raise Phase7CompilationError("counterexample candidate lacks both registered C2 seeds")
    return sum(item.strict_qualified_assertion_f1 for item in candidate.c2_itt_scores) / 2.0


def _choose_qualitative_candidates(
    candidates: QualitativeCandidateSet,
    *,
    policy_path: Path,
) -> tuple[tuple[QualitativeExample, QualitativeCandidate, str], ...]:
    policy = load_reporting_policy(policy_path)
    if candidates.frozen_reporting_policy_sha256 != policy.policy_sha256:
        raise Phase7CompilationError("qualitative candidates name another reporting policy")
    available = candidates.candidates

    def eligible(kind: QualitativeExample) -> list[QualitativeCandidate]:
        return [item for item in available if kind in item.eligible_examples]

    tutorial = [
        item
        for item in eligible(QualitativeExample.DEVELOPMENT_TUTORIAL)
        if item.evidence_split == "development"
        and item.world_id == "syn-dev-01"
        and item.query_ordinal == "A+B"
        and len(item.context_ids) >= 2
    ]
    rare = [
        item
        for item in eligible(QualitativeExample.RARE_PIVOTAL)
        if item.evidence_split == "development"
        and item.rare_pivotal_denominator > 0
        and item.complete_support_path
    ]
    temporal = [
        item
        for item in eligible(QualitativeExample.TEMPORAL_EPISTEMIC)
        if item.evidence_split == "development" and item.temporal_epistemic_eligible
    ]
    held_out = [
        item
        for item in eligible(QualitativeExample.HELD_OUT_ILLUSTRATION)
        if item.evidence_split == "held_out"
        and item.world_id == "syn-test-03"
        and "ctx_556b0577875afcad7419" in item.context_ids
        and item.query_ordinal == "A"
    ]
    counterexamples = [
        item
        for item in eligible(QualitativeExample.COUNTEREXAMPLE)
        if item.evidence_split == "held_out" and item.stratum == "hard"
    ]
    narrative = [
        item
        for item in eligible(QualitativeExample.NARRATIVE_ILLUSTRATION)
        if item.evidence_split == "case_study" and len(item.context_ids) == 2
    ]
    groups = (tutorial, rare, temporal, held_out, counterexamples, narrative)
    missing = [
        _EXAMPLE_IDS[index] for index, values in enumerate(groups) if not values
    ]
    if missing:
        raise Phase7CompilationError(
            "qualitative candidate set cannot satisfy frozen rules: " + ", ".join(missing)
        )
    chosen = (
        min(tutorial, key=lambda item: item.candidate_id),
        min(rare, key=lambda item: (item.world_id, item.context_ids, item.candidate_id)),
        min(
            temporal,
            key=lambda item: (
                not item.holder_attributed,
                item.world_id,
                item.context_ids,
                item.candidate_id,
            ),
        ),
        min(held_out, key=lambda item: item.candidate_id),
        min(
            counterexamples,
            key=lambda item: (
                _candidate_mean(item),
                item.world_id,
                item.context_ids,
                item.candidate_id,
            ),
        ),
        min(
            narrative,
            key=lambda item: (
                item.window_ordinal if item.window_ordinal is not None else 99,
                item.context_ids,
                item.candidate_id,
            ),
        ),
    )
    results = []
    by_id = {item.example_id: item for item in policy.qualitative_selection_rules}
    for raw_id, candidate in zip(_EXAMPLE_IDS, chosen, strict=True):
        example_id = QualitativeExample(raw_id)
        required_conditions = _required_condition_set(candidate)
        for context_id in candidate.context_ids:
            conditions = {
                row.condition
                for row in candidate.display_rows
                if row.context_id == context_id
            }
            if not required_conditions.issubset(conditions):
                missing = ", ".join(sorted(required_conditions - conditions))
                raise Phase7CompilationError(
                    "selected qualitative candidate lacks condition displays for "
                    f"{candidate.candidate_id}/{context_id}: {missing}"
                )
        rule = by_id[raw_id]
        rule_hash = canonical_sha256(rule.model_dump(mode="json"))
        results.append((example_id, candidate, rule_hash))
    return tuple(results)


def select_qualitative_candidates(
    candidates: QualitativeCandidateSet,
    *,
    policy_path: Path,
) -> tuple[tuple[QualitativeExample, QualitativeCandidate, str], ...]:
    """Apply the six frozen reporting rules without rendering or recomputing scores."""

    return _choose_qualitative_candidates(candidates, policy_path=policy_path)


def _qualitative_outputs(
    *,
    candidates: QualitativeCandidateSet,
    selected: Sequence[tuple[QualitativeExample, QualitativeCandidate, str]],
    registry: Phase7SourceRegistry,
    policy_path: Path,
    artifacts: Mapping[str, SourceArtifactSpec],
    source_root: Path,
    stage: Path,
    token: str,
) -> tuple[bytes, tuple[FigureSpec, ...], tuple[str, ...], tuple[str, ...]]:
    policy = load_reporting_policy(policy_path)
    rule_by_id = {item.example_id: item for item in policy.qualitative_selection_rules}
    selection_records: list[dict[str, Any]] = []
    example_records: list[dict[str, Any]] = []
    table_rows: list[dict[str, str]] = []
    figures: list[FigureSpec] = []
    used_source_ids: set[str] = set()
    generated_paths: list[str] = []
    for example, candidate, rule_hash in selected:
        source_ids = tuple(sorted(set(candidate.source_artifact_ids)))
        if not set(source_ids).issubset(artifacts):
            raise Phase7CompilationError(
                f"qualitative candidate {candidate.candidate_id} names an unknown artifact"
            )
        used_source_ids.update(source_ids)
        figure_source = artifacts[candidate.composite_figure_artifact_id]
        if figure_source.media_type != "image/png" or figure_source.release_class != "public":
            raise Phase7CompilationError(
                f"qualitative figure {candidate.composite_figure_artifact_id} is not public PNG"
            )
        selection_basis = (
            f"mean_c2_strict_f1={_candidate_mean(candidate):.17g} over ITT seeds 1 and 2"
            if example is QualitativeExample.COUNTEREXAMPLE
            else "frozen pre-output ordering and tie-break"
        )
        selection_records.append(
            {
                "example_id": example.value,
                "candidate_id": candidate.candidate_id,
                "report_label": rule_by_id[example.value].report_label,
                "selection_rule_sha256": rule_hash,
                "selection_basis": selection_basis,
                "source_artifact_ids": list(source_ids),
                "source_artifact_hashes": [artifacts[item].file_sha256 for item in source_ids],
            }
        )
        displays = tuple(
            sorted(
                candidate.display_rows,
                key=lambda item: (item.context_id, _CONDITION_ORDER[item.condition]),
            )
        )
        for display in displays:
            table_rows.append(
                {
                    "example_id": example.value,
                    "report_label": rule_by_id[example.value].report_label,
                    "candidate_id": candidate.candidate_id,
                    "source_split": candidate.evidence_split,
                    "world_id": candidate.world_id,
                    "context_id": display.context_id,
                    "condition": display.condition,
                    "node_summary": display.node_summary,
                    "assertion_summary": display.assertion_summary,
                    "temporal_sequence": display.temporal_sequence,
                    "why_matters": display.why_matters,
                    "opaque_evidence_ids": ";".join(display.opaque_evidence_ids),
                    "projection_hash": display.projection_hash or "NA",
                    "selection_rule_sha256": rule_hash,
                    "illustration_only": "true",
                }
            )
        example_records.append(
            {
                "example_id": example.value,
                "candidate_id": candidate.candidate_id,
                "evidence_split": candidate.evidence_split,
                "world_id": candidate.world_id,
                "context_ids": list(candidate.context_ids),
                "query_ordinal": candidate.query_ordinal,
                "stratum": candidate.stratum,
                "window_ordinal": candidate.window_ordinal,
                "display_rows": [item.model_dump(mode="json") for item in displays],
                "selection_rule_sha256": rule_hash,
                "paraphrase_only": candidate.paraphrase_only,
                "opaque_evidence_only": candidate.opaque_evidence_only,
                "contains_verbatim_copyrighted_text": (
                    candidate.contains_verbatim_copyrighted_text
                ),
            }
        )
        figure_relative = f"figures/qualitative_{example.value}.{token}.png"
        source_path = _safe_source(source_root, figure_source.relative_path)
        target = stage / figure_relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source_path.read_bytes())
        figure_hash = _file_sha256(target)
        if figure_hash != figure_source.file_sha256:
            raise Phase7CompilationError("qualitative figure changed while being copied")
        figures.append(
            FigureSpec(
                figure_id=f"qualitative_{example.value}",
                kind="artifact_png",
                title=f"Qualitative illustration: {example.value.replace('_', ' ')}",
                relative_path=figure_relative,
                section_id="qualitative_examples",
                caption=(
                    f"{rule_by_id[example.value].report_label.capitalize()}: "
                    f"{candidate.world_id} ({', '.join(candidate.context_ids)})."
                ),
                sha256=figure_hash,
                source_artifact_hashes=(figure_source.file_sha256,),
            )
        )
        generated_paths.append(figure_relative)
    selection_payload = canonical_manifest_payload(
        {
            "schema_version": PHASE7_COMPILER_SCHEMA_VERSION,
            "kind": "qualitative_selection_manifest",
            "source_candidate_set_hash": candidates.content_hash,
            "source_registry_sha256": registry.manifest_sha256,
            "reporting_policy_sha256": policy.policy_sha256,
            "selection_count": len(selection_records),
            "selections": selection_records,
            "selection_uses_condition_outputs_only_for_declared_counterexample": True,
        }
    )
    examples_payload = canonical_manifest_payload(
        {
            "schema_version": PHASE7_COMPILER_SCHEMA_VERSION,
            "kind": "qualitative_example_manifest",
            "source_selection_manifest_sha256": selection_payload["manifest_sha256"],
            "copyright_release_attestation_hash": (
                candidates.copyright_release_attestation_hash
            ),
            "examples": example_records,
            "illustrations_are_not_confirmatory_evidence": True,
        }
    )
    selection_relative = f"qualitative/selection_manifest.{token}.json"
    examples_relative = f"qualitative/example_manifest.{token}.json"
    (stage / selection_relative).parent.mkdir(parents=True, exist_ok=True)
    (stage / selection_relative).write_bytes(_json_bytes(selection_payload))
    (stage / examples_relative).write_bytes(_json_bytes(examples_payload))
    generated_paths.extend((selection_relative, examples_relative))
    columns = (
        "example_id",
        "report_label",
        "candidate_id",
        "source_split",
        "world_id",
        "context_id",
        "condition",
        "node_summary",
        "assertion_summary",
        "temporal_sequence",
        "why_matters",
        "opaque_evidence_ids",
        "projection_hash",
        "selection_rule_sha256",
        "illustration_only",
    )
    table_rows.sort(
        key=lambda row: (
            _EXAMPLE_ORDER[row["example_id"]],
            row["context_id"],
            _CONDITION_ORDER[row["condition"]],
        )
    )
    return (
        _csv_bytes(columns, table_rows),
        tuple(figures),
        tuple(sorted(used_source_ids)),
        tuple(generated_paths),
    )


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _make_ingestion_manifest(
    *,
    registry: Phase7SourceRegistry,
    table_specs: Sequence[IngestedTableSpec],
) -> ReportingIngestionManifest:
    payload: dict[str, Any] = {
        "schema_version": "1.0.0",
        "ingestion_id": f"phase7-{registry.registry_id}",
        "compiled_at_utc": registry.compiled_at_utc.isoformat().replace("+00:00", "Z"),
        "predecessors": [item.model_dump(mode="json") for item in registry.predecessors],
        "tables": [item.model_dump(mode="json") for item in table_specs],
    }
    payload["manifest_sha256"] = canonical_sha256(payload)
    return ReportingIngestionManifest.model_validate(payload)


def _phase_records(
    registry: Phase7SourceRegistry,
    configuration: Phase7CompilerConfiguration,
    artifacts: Mapping[str, SourceArtifactSpec],
) -> tuple[PhaseStatus, ...]:
    states = {item.phase_id: item for item in registry.phases}
    return tuple(
        PhaseStatus(
            phase_id=label.phase_id,
            label=label.label,
            status=states[label.phase_id].status,
            reason=states[label.phase_id].reason,
            source_artifact_hashes=tuple(
                artifacts[item].file_sha256
                for item in states[label.phase_id].source_artifact_ids
            ),
        )
        for label in configuration.phase_labels
    )


def _build_public_input_manifest(
    *,
    source_root: Path,
    logical_output_root: Path,
    stage: Path,
    registry: Phase7SourceRegistry,
    configuration: Phase7CompilerConfiguration,
    generated_relative_paths: Sequence[str],
    token: str,
    status: ReportStatus,
) -> tuple[str, bytes]:
    try:
        output_prefix = logical_output_root.resolve().relative_to(source_root.resolve()).as_posix()
    except ValueError as error:
        raise Phase7CompilationError(
            "Phase 7 output root must be inside the source root to compile a public allowlist"
        ) from error
    artifacts = _artifact_inventory(registry)
    public_source_manifest = load_public_reproduction_source_manifest(
        _safe_source(source_root, configuration.public_source_manifest_path)
    )
    public_source_paths = expand_public_reproduction_source_paths(
        public_source_manifest,
        source_root=source_root,
    )
    entries: dict[str, dict[str, str]] = {}

    def add_source(source_relative: str, bundle_relative: str, sha256: str) -> None:
        prior = entries.get(bundle_relative)
        value = {
            "source_relative_path": source_relative,
            "bundle_relative_path": bundle_relative,
            "sha256": sha256,
            "release_class": "public",
        }
        if prior is not None and prior != value:
            raise Phase7CompilationError(f"duplicate public bundle target: {bundle_relative}")
        entries[bundle_relative] = value

    for artifact_id in registry.public_artifact_ids:
        artifact = artifacts[artifact_id]
        add_source(artifact.relative_path, artifact.relative_path, artifact.file_sha256)
    for relative in public_source_paths:
        source = _safe_source(source_root, relative)
        add_source(relative, relative, _file_sha256(source))
    for relative in sorted(set(generated_relative_paths)):
        path = stage / relative
        if not path.is_file():
            raise Phase7CompilationError(f"public generated output is absent: {relative}")
        final_source = f"{output_prefix}/{relative}" if output_prefix != "." else relative
        bundle_relative = (
            f"reports/{Path(relative).name}"
            if "/" not in relative
            else f"reports/{relative}"
        )
        add_source(final_source, bundle_relative, _file_sha256(path))
    total = sum((stage / item).stat().st_size for item in set(generated_relative_paths))
    total += sum(
        _safe_source(source_root, artifacts[item].relative_path).stat().st_size
        for item in registry.public_artifact_ids
    )
    total += sum(
        _safe_source(source_root, item).stat().st_size
        for item in public_source_paths
    )
    if total > configuration.public_bundle_limit_bytes:
        raise Phase7CompilationError("Phase 7 public allowlist exceeds the 2 GB limit")
    payload = canonical_manifest_payload(
        {
            "schema_version": "1.0.0",
            "bundle_status": status.value,
            "status_reason": registry.status_reason,
            "source_registry_sha256": registry.manifest_sha256,
            "entries": [entries[key] for key in sorted(entries)],
            "intentional_exclusions": [
                "model weights and caches",
                "restricted/copyrighted novel text and derived FTS indexes",
                "raw prompts or model outputs containing protected prose",
                "detailed/reconstructive case-study offsets",
                "SSH material and machine-specific paths",
                "restricted predecessor artifacts and the source registry",
            ],
        }
    )
    relative = f"public_bundle_inputs.{token}.json"
    return relative, _json_bytes(payload)


def _incomplete_qualitative_outputs(
    *,
    registry: Phase7SourceRegistry,
    policy_path: Path,
    token: str,
    reason: str,
) -> tuple[str, bytes, str, bytes]:
    policy = load_reporting_policy(policy_path)
    rules = {item.example_id: item for item in policy.qualitative_selection_rules}
    selections = [
        {
            "example_id": example_id,
            "status": "blocked" if "lawful" in reason.casefold() else "incomplete",
            "reason": reason,
            "selection_rule_sha256": canonical_sha256(
                rules[example_id].model_dump(mode="json")
            ),
            "candidate_id": None,
        }
        for example_id in _EXAMPLE_IDS
    ]
    selection = canonical_manifest_payload(
        {
            "schema_version": PHASE7_COMPILER_SCHEMA_VERSION,
            "kind": "qualitative_selection_manifest",
            "source_registry_sha256": registry.manifest_sha256,
            "reporting_policy_sha256": policy.policy_sha256,
            "selection_count": 0,
            "selections": selections,
            "selection_uses_condition_outputs_only_for_declared_counterexample": True,
        }
    )
    examples = canonical_manifest_payload(
        {
            "schema_version": PHASE7_COMPILER_SCHEMA_VERSION,
            "kind": "qualitative_example_manifest",
            "source_selection_manifest_sha256": selection["manifest_sha256"],
            "status": "incomplete",
            "reason": reason,
            "examples": [],
            "illustrations_are_not_confirmatory_evidence": True,
        }
    )
    return (
        f"qualitative/selection_manifest.{token}.json",
        _json_bytes(selection),
        f"qualitative/example_manifest.{token}.json",
        _json_bytes(examples),
    )


def _error_review_summary(
    rows: Sequence[Mapping[str, str]],
    *,
    reviewed_failure_count: int,
) -> bytes:
    output_rows = []
    for code in FROZEN_ERROR_CODES:
        matches = [
            row for row in rows if code.value in row["error_codes"].split("|")
        ]
        output_rows.append(
            {
                "error_code": code.value,
                "failure_count": str(len(matches)),
                "world_count": str(len({row["world_id"] for row in matches})),
                "reviewed_failure_count": str(reviewed_failure_count),
                "independent_unit": "world",
            }
        )
    return _csv_bytes(_ERROR_REVIEW_SUMMARY_COLUMNS, output_rows)


def _community_review_summary(
    rows: Sequence[Mapping[str, str]],
    *,
    reviewed_partition_count: int,
) -> bytes:
    condition_order = {"C0": 0, "C1": 1, "C2": 2, "A-FixedSelect": 3}
    ordered = sorted(
        rows,
        key=lambda row: (condition_order[row["condition"]], float(row["resolution"])),
    )
    return _csv_bytes(
        _COMMUNITY_REVIEW_SUMMARY_COLUMNS,
        tuple(
            {
                "condition": row["condition"],
                "resolution": row["resolution"],
                "semantic_coherence": row["semantic_coherence"],
                "interpretability": row["interpretability"],
                "evidence_support": row["evidence_support"],
                "rubric_mean": row["rubric_mean"],
                "reviewed_partition_count": str(reviewed_partition_count),
                "independent_unit": "world",
            }
            for row in ordered
        ),
    )


def _materialize_review_supplements(
    *,
    source_root: Path,
    stage: Path,
    token: str,
    registry: Phase7SourceRegistry,
    configuration: Phase7CompilerConfiguration,
    artifacts: Mapping[str, SourceArtifactSpec],
) -> tuple[tuple[ReviewSupplementSpec, ...], tuple[str, ...]]:
    specs: list[ReviewSupplementSpec] = []
    paths: list[str] = []
    if registry.error_review_artifacts is not None:
        finalization, rows = _verify_error_review_binding(
            registry,
            configuration,
            artifacts=artifacts,
            source_root=source_root,
        )
        payload = _error_review_summary(
            rows,
            reviewed_failure_count=finalization.reviewed_failure_count,
        )
        relative_path = f"tables/held_out_error_review_summary.{token}.csv"
        _write(stage / relative_path, payload)
        binding = registry.error_review_artifacts
        specs.append(
            ReviewSupplementSpec(
                supplement_id="held_out_error_review",
                section_id="error_limitations",
                relative_path=relative_path,
                sha256=hashlib.sha256(payload).hexdigest(),
                row_count=len(FROZEN_ERROR_CODES),
                required_columns=_ERROR_REVIEW_SUMMARY_COLUMNS,
                display_columns=(
                    "error_code",
                    "failure_count",
                    "world_count",
                    "reviewed_failure_count",
                ),
                description=(
                    "Condition-blind held-out error taxonomy counts; zero-count registered "
                    "categories are retained."
                ),
                source_artifact_hashes=tuple(
                    artifacts[item].file_sha256 for item in binding.artifact_ids()
                ),
            )
        )
        paths.append(relative_path)
    if registry.community_review_artifacts is not None:
        finalization, rows = _verify_community_review_binding(
            registry,
            configuration,
            artifacts=artifacts,
            source_root=source_root,
        )
        payload = _community_review_summary(
            rows,
            reviewed_partition_count=finalization.reviewed_partition_count,
        )
        relative_path = f"tables/community_blind_review_summary.{token}.csv"
        _write(stage / relative_path, payload)
        binding = registry.community_review_artifacts
        specs.append(
            ReviewSupplementSpec(
                supplement_id="community_blind_review",
                section_id="community",
                relative_path=relative_path,
                sha256=hashlib.sha256(payload).hexdigest(),
                row_count=12,
                required_columns=_COMMUNITY_REVIEW_SUMMARY_COLUMNS,
                display_columns=(
                    "condition",
                    "resolution",
                    "semantic_coherence",
                    "interpretability",
                    "evidence_support",
                    "rubric_mean",
                ),
                description=(
                    "Condition-blind community-coherence rubric supplement for the frozen "
                    "paired context and three registered Leiden resolutions."
                ),
                source_artifact_hashes=tuple(
                    artifacts[item].file_sha256 for item in binding.artifact_ids()
                ),
            )
        )
        paths.append(relative_path)
    return tuple(sorted(specs, key=lambda item: item.supplement_id)), tuple(paths)


def _assemble_phase7_tree(
    *,
    source_root: Path,
    logical_output_root: Path,
    stage: Path,
    configuration: Phase7CompilerConfiguration,
    registry: Phase7SourceRegistry,
) -> tuple[CompiledPhase7Artifacts, tuple[str, ...]]:
    """Build a complete deterministic output tree in a fresh staging directory."""

    stage.mkdir(parents=True, exist_ok=False)
    policy_path = _safe_source(source_root, configuration.reporting_policy_path)
    artifacts = _artifact_inventory(registry)
    _verify_registry_contracts(registry, configuration, source_root=source_root)
    review_supplements, review_supplement_paths = _materialize_review_supplements(
        source_root=source_root,
        stage=stage,
        token=registry.manifest_sha256[:16],
        registry=registry,
        configuration=configuration,
        artifacts=artifacts,
    )
    contracts = {item.table_id: item for item in configuration.table_contracts}
    bindings = {item.table_id: item for item in registry.tables}
    token = registry.manifest_sha256[:16]
    status = _study_status(registry)
    generated_paths: list[str] = []
    generated_paths.extend(review_supplement_paths)
    qualitative_figures: tuple[FigureSpec, ...] = ()
    qualitative_bytes: bytes | None = None

    qualitative_binding = bindings["qualitative_examples"]
    if qualitative_binding.status is ReportStatus.COMPLETE:
        candidate_id = registry.qualitative_candidate_set_artifact_id
        if candidate_id is None or candidate_id not in artifacts:
            raise Phase7CompilationError(
                "complete qualitative table requires the registered candidate set"
            )
        candidate_artifact = artifacts[candidate_id]
        if candidate_artifact.media_type != "application/json":
            raise Phase7CompilationError("qualitative candidate set is not JSON")
        candidate_path = _safe_source(source_root, candidate_artifact.relative_path)
        if _file_sha256(candidate_path) != candidate_artifact.file_sha256:
            raise Phase7CompilationError("qualitative candidate-set file changed")
        candidate_set = load_qualitative_candidate_set(candidate_path)
        if (
            candidate_artifact.logical_hash is not None
            and candidate_artifact.logical_hash != candidate_set.content_hash
        ):
            raise Phase7CompilationError("qualitative candidate-set logical hash changed")
        selected = _choose_qualitative_candidates(candidate_set, policy_path=policy_path)
        if len(selected) > configuration.maximum_qualitative_figures:
            raise Phase7CompilationError(
                "qualitative selection exceeds the configured publication-figure limit"
            )
        (
            qualitative_bytes,
            qualitative_figures,
            used_source_ids,
            qualitative_paths,
        ) = _qualitative_outputs(
            candidates=candidate_set,
            selected=selected,
            registry=registry,
            policy_path=policy_path,
            artifacts=artifacts,
            source_root=source_root,
            stage=stage,
            token=token,
        )
        if not set(used_source_ids).issubset(qualitative_binding.source_artifact_ids):
            raise Phase7CompilationError(
                "qualitative table lineage omits a selected candidate source artifact"
            )
        generated_paths.extend(qualitative_paths)
    else:
        incomplete = _incomplete_qualitative_outputs(
            registry=registry,
            policy_path=policy_path,
            token=token,
            reason=qualitative_binding.reason,
        )
        for relative, payload in ((incomplete[0], incomplete[1]), (incomplete[2], incomplete[3])):
            _write(stage / relative, payload)
            generated_paths.append(relative)

    phase_states = {item.phase_id: item for item in registry.phases}
    table_payloads: dict[str, tuple[tuple[str, ...], bytes, int, tuple[str, ...]]] = {}
    for table_id in sorted(REGISTERED_COMPLETE_TABLE_IDS):
        binding = bindings[table_id]
        if binding.status is not ReportStatus.COMPLETE:
            continue
        contract = contracts[table_id]
        if contract.producer is TableProducer.STUDY_STATUS:
            rows = tuple(
                {
                    "phase_id": phase.phase_id,
                    "status": phase_states[phase.phase_id].status.value,
                    "reason": phase_states[phase.phase_id].reason,
                }
                for phase in configuration.phase_labels
            )
            columns = contract.required_columns
            payload = _csv_bytes(columns, rows)
            # The table is generated from the seven attested phase states rather
            # than copied from an upstream CSV.  Its immutable provenance is
            # therefore the union of the exact artifacts that support those
            # states.  In an interim registry this union may legitimately be
            # empty; an all-complete registry cannot be empty because
            # ``PhaseInputState`` requires lineage for every complete phase.
            source_ids = _study_status_source_artifact_ids(registry.phases)
        elif contract.producer is TableProducer.QUALITATIVE_SELECTION:
            if qualitative_bytes is None:
                raise Phase7CompilationError("qualitative table is marked complete without rows")
            columns = contract.required_columns
            payload = qualitative_bytes
            reader = csv.DictReader(payload.decode("utf-8").splitlines())
            rows = tuple(reader)
            source_ids = binding.source_artifact_ids
        else:
            assert binding.source_table_artifact_id is not None
            assert binding.source_row_count is not None
            assert binding.output_row_count is not None
            source = artifacts[binding.source_table_artifact_id]
            source_path = _safe_source(source_root, source.relative_path)
            singleton_columns: tuple[str, ...] = ()
            singleton_row: dict[str, str] = {}
            source_contract = contract
            if binding.singleton_join_artifact_id is not None:
                singleton_source = artifacts[binding.singleton_join_artifact_id]
                singleton_columns, singleton_row = _canonical_singleton_csv(
                    _safe_source(source_root, singleton_source.relative_path),
                    expected_hash=singleton_source.file_sha256,
                    table_id=table_id,
                )
                source_contract = contract.model_copy(
                    update={
                        "required_columns": tuple(
                            item
                            for item in contract.required_columns
                            if item not in singleton_columns
                        )
                    }
                )
            columns, rows, payload = _canonical_source_csv(
                source_path,
                expected_hash=source.file_sha256,
                expected_rows=binding.source_row_count,
                expected_output_rows=binding.output_row_count,
                row_filters=binding.row_filters,
                contract=source_contract,
            )
            if singleton_columns:
                overlap = set(columns) & set(singleton_columns)
                if overlap:
                    raise Phase7CompilationError(
                        f"singleton join repeats columns for {table_id}: "
                        + ", ".join(sorted(overlap))
                    )
                columns = (*columns, *singleton_columns)
                rows = tuple({**row, **singleton_row} for row in rows)
                payload = _csv_bytes(columns, rows)
            if not set(contract.required_columns).issubset(columns):
                raise Phase7CompilationError(
                    f"joined source table lacks required columns: {table_id}"
                )
            _validate_registered_table_rows(contract, rows)
            if table_id == "mechanism_c2_vs_fixed":
                _validate_mechanism_table_rows(rows)
            source_ids = binding.source_artifact_ids
        relative = f"tables/{table_id}.{token}.csv"
        _write(stage / relative, payload)
        generated_paths.append(relative)
        table_payloads[table_id] = (tuple(columns), payload, len(rows), tuple(source_ids))

    if "study_status" not in table_payloads:
        raise Phase7CompilationError("the compiler always requires a generated study-status table")

    ingested_specs: list[IngestedTableSpec] = []
    for table_id in sorted(REGISTERED_COMPLETE_TABLE_IDS):
        binding = bindings[table_id]
        contract = contracts[table_id]
        if binding.status is ReportStatus.COMPLETE:
            columns, payload, row_count, source_ids = table_payloads[table_id]
            ingested_specs.append(
                IngestedTableSpec(
                    table_id=table_id,
                    status=binding.status,
                    reason=binding.reason,
                    scope=binding.scope,
                    relative_path=f"tables/{table_id}.{token}.csv",
                    file_sha256=hashlib.sha256(payload).hexdigest(),
                    row_count=row_count,
                    required_columns=contract.required_columns,
                    source_artifact_ids=source_ids,
                )
            )
        else:
            ingested_specs.append(
                IngestedTableSpec(
                    table_id=table_id,
                    status=binding.status,
                    reason=binding.reason,
                    scope=binding.scope,
                )
            )
    ingestion = _make_ingestion_manifest(registry=registry, table_specs=ingested_specs)
    ingestion_relative = f"report_ingestion_manifest.{token}.json"
    ingestion_payload = _model_json_bytes(ingestion)
    _write(stage / ingestion_relative, ingestion_payload)
    generated_paths.append(ingestion_relative)
    receipt = compile_ingestion_receipt(
        ingestion,
        source_root=source_root,
        table_root=stage,
        manifest_file_sha256=hashlib.sha256(ingestion_payload).hexdigest(),
    )
    receipt_relative = f"report_ingestion_receipt.{token}.json"
    receipt_payload = _model_json_bytes(receipt)
    _write(stage / receipt_relative, receipt_payload)
    generated_paths.append(receipt_relative)

    receipt_tables = {item.table_id: item for item in receipt.tables}
    result_tables_list: list[TableSpec] = []
    for table_id in sorted(table_payloads):
        contract = contracts[table_id]
        binding = bindings[table_id]
        source_contracts = {
            item.source_role: item for item in contract.supplemental_metric_sources
        }
        supplemental_sources = tuple(
            SupplementalMetricSourceSpec(
                source_role=item.source_role,
                source_relative_path=artifacts[item.source_artifact_id].relative_path,
                source_file_sha256=artifacts[item.source_artifact_id].file_sha256,
                table_manifest_relative_path=(
                    artifacts[item.table_manifest_artifact_id].relative_path
                ),
                table_manifest_file_sha256=(
                    artifacts[item.table_manifest_artifact_id].file_sha256
                ),
                source_row_count=item.source_row_count,
                selected_row_count=item.selected_row_count,
                metric_row_counts=tuple(
                    SupplementalMetricCount(
                        metric_name=count.metric_name,
                        row_count=count.row_count,
                    )
                    for count in item.metric_row_counts
                ),
                description=source_contracts[item.source_role].description,
            )
            for item in binding.supplemental_metric_sources
        )
        result_tables_list.append(
            TableSpec(
                table_id=table_id,
                relative_path=receipt_tables[table_id].relative_path,
                sha256=receipt_tables[table_id].file_sha256,
                row_count=receipt_tables[table_id].row_count,
                required_columns=contract.required_columns,
                display_columns=contract.display_columns,
                status=ReportStatus.COMPLETE,
                description=contract.description,
                source_artifact_hashes=receipt_tables[table_id].source_artifact_hashes,
                supplemental_metric_sources=supplemental_sources,
            )
        )
    result_tables = tuple(result_tables_list)
    figure_specs: list[FigureSpec] = [
        FigureSpec(
            figure_id="phase_status",
            kind="phase_status",
            title="Registered study phase status",
            relative_path=f"figures/study_status.{token}.pdf",
        )
    ]
    if bindings["primary_c2_vs_c1"].status is ReportStatus.COMPLETE:
        primary_columns = configuration.primary_figure_columns
        available_columns = set(table_payloads["primary_c2_vs_c1"][0])
        required = {
            primary_columns.label,
            primary_columns.estimate,
            primary_columns.lower,
            primary_columns.upper,
        }
        if not required.issubset(available_columns):
            raise Phase7CompilationError("primary table cannot drive the registered forest figure")
        figure_specs.append(
            FigureSpec(
                figure_id="primary_effects",
                kind="forest",
                title="Primary paired C2 minus C1 effects across twelve worlds",
                relative_path=f"figures/primary_effects.{token}.pdf",
                table_id="primary_c2_vs_c1",
                label_column=primary_columns.label,
                estimate_column=primary_columns.estimate,
                lower_column=primary_columns.lower,
                upper_column=primary_columns.upper,
            )
        )
    figure_specs.extend(qualitative_figures)
    if status is ReportStatus.COMPLETE and not set(REGISTERED_FIGURE_IDS).issubset(
        {item.figure_id for item in figure_specs}
    ):
        raise Phase7CompilationError("complete Phase 7 output lacks a registered figure")

    result_payload: dict[str, Any] = {
        "schema_version": "1.0.0",
        "study_id": registry.study_id,
        "study_status": status.value,
        "status_reason": registry.status_reason,
        "generated_at_utc": registry.compiled_at_utc.isoformat().replace("+00:00", "Z"),
        "code_revision": registry.code_revision,
        "dirty_worktree": registry.dirty_worktree,
        "model_repository": registry.model_repository,
        "model_revision": registry.model_revision,
        "phases": [
            item.model_dump(mode="json")
            for item in _phase_records(registry, configuration, artifacts)
        ],
        "tables": [item.model_dump(mode="json") for item in result_tables],
        "review_supplements": [
            item.model_dump(mode="json") for item in review_supplements
        ],
        "figures": [item.model_dump(mode="json") for item in figure_specs],
        "sections": [
            item.model_copy(
                update={
                    "table_ids": tuple(
                        table_id for table_id in item.table_ids if table_id in table_payloads
                    )
                }
            ).model_dump(mode="json")
            for item in configuration.sections
        ],
        "source_artifact_hashes": [item.file_sha256 for item in receipt.artifacts],
        "ingestion_receipt_relative_path": receipt_relative,
        "ingestion_receipt_file_sha256": hashlib.sha256(receipt_payload).hexdigest(),
        "ingestion_receipt_sha256": receipt.receipt_sha256,
        "reporting_policy_sha256": configuration.reporting_policy_sha256,
    }
    result_payload["manifest_sha256"] = canonical_sha256(result_payload)
    result = ResultManifest.model_validate(result_payload)
    result_relative = f"results_manifest.{token}.json"
    result_bytes = _model_json_bytes(result)
    _write(stage / result_relative, result_bytes)
    _write(stage / "results_manifest.json", result_bytes)
    generated_paths.extend((result_relative, "results_manifest.json"))

    build_results_report(stage / result_relative, policy_path, stage)
    verify_report_build(stage / result_relative, policy_path, stage)
    aliases_and_unique = (
        ("RESULTS_REPORT.md", f"RESULTS_REPORT.{token}.md"),
        ("RESULTS_REPORT.pdf", f"RESULTS_REPORT.{token}.pdf"),
        ("REPRODUCIBILITY.md", f"REPRODUCIBILITY.{token}.md"),
        ("result_figure_manifest.json", f"result_figure_manifest.{token}.json"),
    )
    for alias, unique in aliases_and_unique:
        _write(stage / unique, (stage / alias).read_bytes())
        generated_paths.extend((alias, unique))
    generated_paths.extend(item.relative_path for item in figure_specs)

    public_generated = [
        item
        for item in generated_paths
        if not item.startswith("report_ingestion_manifest.")
        and item not in {
            "results_manifest.json",
            "RESULTS_REPORT.md",
            "RESULTS_REPORT.pdf",
            "REPRODUCIBILITY.md",
            "result_figure_manifest.json",
        }
    ]
    public_relative, public_payload = _build_public_input_manifest(
        source_root=source_root,
        logical_output_root=logical_output_root,
        stage=stage,
        registry=registry,
        configuration=configuration,
        generated_relative_paths=public_generated,
        token=token,
        status=status,
    )
    _write(stage / public_relative, public_payload)
    _write(stage / "public_bundle_inputs.json", public_payload)
    generated_paths.extend((public_relative, "public_bundle_inputs.json"))

    immutable_outputs = tuple(
        sorted(
            set(generated_paths)
            - {
                "results_manifest.json",
                "RESULTS_REPORT.md",
                "RESULTS_REPORT.pdf",
                "REPRODUCIBILITY.md",
                "result_figure_manifest.json",
                "public_bundle_inputs.json",
            }
        )
    )
    output_hashes = {relative: _file_sha256(stage / relative) for relative in immutable_outputs}
    compilation = canonical_manifest_payload(
        {
            "schema_version": PHASE7_COMPILER_SCHEMA_VERSION,
            "kind": "phase7_compilation_manifest",
            "source_registry_sha256": registry.manifest_sha256,
            "compiler_configuration_sha256": configuration.configuration_sha256,
            "study_status": status.value,
            "build_token": token,
            "immutable_outputs": [
                {"relative_path": relative, "sha256": output_hashes[relative]}
                for relative in immutable_outputs
            ],
            "aliases": {
                "results_manifest.json": result_relative,
                "RESULTS_REPORT.md": f"RESULTS_REPORT.{token}.md",
                "RESULTS_REPORT.pdf": f"RESULTS_REPORT.{token}.pdf",
                "REPRODUCIBILITY.md": f"REPRODUCIBILITY.{token}.md",
                "result_figure_manifest.json": f"result_figure_manifest.{token}.json",
                "public_bundle_inputs.json": public_relative,
            },
            "manual_numeric_transcription_permitted": False,
        }
    )
    compilation_relative = f"phase7_compilation.{token}.json"
    compilation_bytes = _json_bytes(compilation)
    _write(stage / compilation_relative, compilation_bytes)
    output_hashes[compilation_relative] = hashlib.sha256(compilation_bytes).hexdigest()
    pointer = canonical_manifest_payload(
        {
            "schema_version": PHASE7_COMPILER_SCHEMA_VERSION,
            "kind": "phase7_current_pointer",
            "source_registry_sha256": registry.manifest_sha256,
            "compiler_configuration_sha256": configuration.configuration_sha256,
            "build_token": token,
            "study_status": status.value,
            "compilation_manifest_relative_path": compilation_relative,
            "compilation_manifest_file_sha256": output_hashes[compilation_relative],
            "result_manifest_relative_path": result_relative,
            "result_manifest_file_sha256": output_hashes[result_relative],
            "report_pdf_relative_path": f"RESULTS_REPORT.{token}.pdf",
            "report_pdf_file_sha256": output_hashes[f"RESULTS_REPORT.{token}.pdf"],
            "public_bundle_input_relative_path": public_relative,
            "public_bundle_input_file_sha256": hashlib.sha256(public_payload).hexdigest(),
        }
    )
    pointer_bytes = _json_bytes(pointer)
    _write(stage / f"phase7_current.{token}.json", pointer_bytes)
    _write(stage / "phase7_current.json", pointer_bytes)
    generated_paths.extend(
        (compilation_relative, f"phase7_current.{token}.json", "phase7_current.json")
    )
    output_hashes[f"phase7_current.{token}.json"] = hashlib.sha256(pointer_bytes).hexdigest()
    return (
        CompiledPhase7Artifacts(
            registry_sha256=registry.manifest_sha256,
            configuration_sha256=configuration.configuration_sha256,
            build_token=token,
            study_status=status,
            output_hashes=dict(sorted(output_hashes.items())),
            current_pointer_sha256=pointer["manifest_sha256"],
        ),
        tuple(sorted(set(generated_paths))),
    )


_REPLACEABLE_ALIASES = frozenset(
    {
        "results_manifest.json",
        "RESULTS_REPORT.md",
        "RESULTS_REPORT.pdf",
        "REPRODUCIBILITY.md",
        "result_figure_manifest.json",
        "public_bundle_inputs.json",
        "phase7_current.json",
    }
)


def _atomic_install(source: Path, target: Path, *, replaceable: bool) -> None:
    _assert_no_symlink_chain(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = source.read_bytes()
    if target.exists() and not target.is_file():
        raise Phase7CompilationError(f"Phase 7 output target is not a file: {target}")
    if target.exists() and target.read_bytes() == payload:
        return
    if target.exists() and not replaceable:
        raise Phase7CompilationError(f"append-only Phase 7 output drift: {target}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".partial", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o644)
        if replaceable:
            os.replace(temporary, target)
        else:
            try:
                os.link(temporary, target)
            except FileExistsError:
                if target.read_bytes() != payload:
                    raise Phase7CompilationError(
                        f"concurrent Phase 7 output drift: {target}"
                    ) from None
        directory = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_stage(stage: Path, output_root: Path, generated_paths: Sequence[str]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    unique = [item for item in generated_paths if item not in _REPLACEABLE_ALIASES]
    aliases = [item for item in generated_paths if item in _REPLACEABLE_ALIASES]
    for relative in (*sorted(unique), *sorted(aliases)):
        _atomic_install(
            stage / relative,
            output_root / relative,
            replaceable=relative in _REPLACEABLE_ALIASES,
        )


def compile_phase7_results(
    *,
    source_root: Path,
    configuration_path: Path,
    registry_path: Path,
    output_root: Path,
) -> CompiledPhase7Artifacts:
    """Compile and atomically publish one content-addressed Phase 7 result set."""

    source_root = source_root.resolve(strict=True)
    configuration = Phase7CompilerConfiguration.load(
        configuration_path,
        source_root=source_root,
    )
    registry = load_phase7_source_registry(registry_path)
    output_root = output_root.resolve()
    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".phase7-build-", dir=output_root.parent
    ) as temporary:
        stage = Path(temporary) / "tree"
        result, paths = _assemble_phase7_tree(
            source_root=source_root,
            logical_output_root=output_root,
            stage=stage,
            configuration=configuration,
            registry=registry,
        )
        _publish_stage(stage, output_root, paths)
    verify_phase7_results(
        source_root=source_root,
        configuration_path=configuration_path,
        registry_path=registry_path,
        output_root=output_root,
    )
    from story_projection_onto.public_release import load_public_entries, scan_public_entries

    public_manifest = output_root / f"public_bundle_inputs.{result.build_token}.json"
    scan_public_entries(source_root, load_public_entries(public_manifest))
    return result


def verify_phase7_results(
    *,
    source_root: Path,
    configuration_path: Path,
    registry_path: Path,
    output_root: Path,
) -> CompiledPhase7Artifacts:
    """Regenerate in isolation and byte-compare every current Phase 7 output."""

    source_root = source_root.resolve(strict=True)
    output_root = output_root.resolve(strict=True)
    configuration = Phase7CompilerConfiguration.load(
        configuration_path,
        source_root=source_root,
    )
    registry = load_phase7_source_registry(registry_path)
    with tempfile.TemporaryDirectory(prefix="story-projection-phase7-verify-") as temporary:
        stage = Path(temporary) / "tree"
        result, paths = _assemble_phase7_tree(
            source_root=source_root,
            logical_output_root=output_root,
            stage=stage,
            configuration=configuration,
            registry=registry,
        )
        for relative in paths:
            observed = output_root / relative
            if not observed.is_file() or observed.is_symlink():
                raise Phase7CompilationError(f"missing or symlinked Phase 7 output: {relative}")
            if observed.read_bytes() != (stage / relative).read_bytes():
                raise Phase7CompilationError(f"Phase 7 output does not reproduce: {relative}")
    return result


__all__ = [
    "C2IntentionToTreatScore",
    "CommunityReviewArtifactBinding",
    "CompiledPhase7Artifacts",
    "ErrorReviewArtifactBinding",
    "ExactRowFilter",
    "Phase7CompilationError",
    "Phase7CompilerConfiguration",
    "Phase7SourceRegistry",
    "Phase7TableBinding",
    "PublicReproductionSourceManifest",
    "PublicSourceCategory",
    "PublicSourceInventoryGroup",
    "QualitativeCandidate",
    "QualitativeCandidateSet",
    "QualitativeDisplayRow",
    "QualitativeExample",
    "QualitativeProducerArtifactBinding",
    "TableContract",
    "TableProducer",
    "compile_phase7_results",
    "expand_public_reproduction_source_paths",
    "load_phase7_source_registry",
    "load_public_reproduction_source_manifest",
    "load_qualitative_candidate_set",
    "select_qualitative_candidates",
    "verify_phase7_results",
]
