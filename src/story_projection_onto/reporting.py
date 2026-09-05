"""Reproducible, status-aware conference report generation.

The report builder never discovers result files or computes scientific statistics.
It consumes an explicit manifest of immutable canonical CSV tables, verifies every
byte and row count, and renders the same in-memory document to Markdown and PDF.
Missing gates are rendered as incomplete sections rather than inferred outcomes.
"""

from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import math
import re
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from itertools import chain
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StringConstraints, model_validator

REPORT_SCHEMA_VERSION = "1.0.0"
REPORT_TITLE = "StoryProjectionOnto: Conference-Study Results Report"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
Sha256Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]

REGISTERED_COMPLETE_TABLE_IDS = frozenset(
    {
        "ablations",
        "community",
        "entropy_clutter",
        "failure_accounting",
        "feedback",
        "novel_case",
        "paraphrase_contrastive",
        "primary_c2_vs_c1",
        "qualitative_examples",
        "rare_pivotal",
        "resource_accounting",
        "secondary_c2_vs_c0",
        "mechanism_c2_vs_fixed",
        "study_status",
    }
)
REGISTERED_SECTION_IDS = frozenset(
    {
        "research_questions",
        "conditions",
        "temporal_representation",
        "synthetic_benchmark",
        "resource_controls",
        "primary_results",
        "secondary_results",
        "mechanism_results",
        "rare_pivotal",
        "entropy_clutter",
        "community",
        "paraphrase_contrastive",
        "ablations",
        "feedback_interface",
        "novel_case",
        "runtime_failures",
        "error_limitations",
        "claim_boundaries",
        "qualitative_examples",
    }
)
REGISTERED_FIGURE_IDS = frozenset({"phase_status", "primary_effects"})
REGISTERED_SUPPLEMENTAL_METRIC_TABLE_IDS = frozenset(
    {"primary_c2_vs_c1", "rare_pivotal", "entropy_clutter", "community"}
)
REGISTERED_REVIEW_SUPPLEMENT_IDS = frozenset(
    {"community_blind_review", "held_out_error_review"}
)


class ReportingError(RuntimeError):
    """Base error for report input, rendering, or consistency failures."""


class ReportStatus(StrEnum):
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    BLOCKED = "blocked"
    NOT_APPLICABLE = "not_applicable"


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PhaseStatus(FrozenModel):
    phase_id: Annotated[str, StringConstraints(pattern=r"^phase_[1-7]$")]
    label: Annotated[str, StringConstraints(min_length=1, max_length=160)]
    status: ReportStatus
    reason: Annotated[str, StringConstraints(min_length=1, max_length=1000)]
    source_artifact_hashes: tuple[Sha256Digest, ...] = ()


class SupplementalMetricCount(FrozenModel):
    metric_name: Annotated[str, StringConstraints(min_length=1, max_length=160)]
    row_count: int = Field(ge=1)


class SupplementalMetricSourceSpec(FrozenModel):
    """Public, immutable numeric rows that accompany a compact result table."""

    source_role: Annotated[str, StringConstraints(min_length=1, max_length=128)]
    source_relative_path: Annotated[str, StringConstraints(min_length=1, max_length=500)]
    source_file_sha256: Sha256Digest
    table_manifest_relative_path: Annotated[
        str, StringConstraints(min_length=1, max_length=500)
    ]
    table_manifest_file_sha256: Sha256Digest
    source_row_count: int = Field(ge=1)
    selected_row_count: int = Field(ge=1)
    metric_row_counts: tuple[SupplementalMetricCount, ...]
    description: Annotated[str, StringConstraints(min_length=1, max_length=500)]

    @model_validator(mode="after")
    def inventory_is_canonical(self) -> SupplementalMetricSourceSpec:
        for value in (self.source_relative_path, self.table_manifest_relative_path):
            path = Path(value)
            if path.is_absolute() or ".." in path.parts or "\\" in value:
                raise ValueError("supplemental metric paths must be bounded and portable")
        names = tuple(item.metric_name for item in self.metric_row_counts)
        if names != tuple(sorted(set(names))):
            raise ValueError("supplemental metric counts must be sorted and unique")
        if sum(item.row_count for item in self.metric_row_counts) > self.selected_row_count:
            raise ValueError("supplemental metric counts exceed selected source rows")
        return self


class TableSpec(FrozenModel):
    table_id: Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]{1,63}$")]
    relative_path: Annotated[
        str,
        StringConstraints(pattern=r"^tables/[A-Za-z0-9_.-]+\.csv$"),
    ]
    sha256: Sha256Digest
    row_count: int = Field(ge=0)
    required_columns: tuple[Annotated[str, StringConstraints(min_length=1)], ...]
    display_columns: tuple[Annotated[str, StringConstraints(min_length=1)], ...] = ()
    status: ReportStatus
    description: Annotated[str, StringConstraints(min_length=1, max_length=500)]
    source_artifact_hashes: tuple[Sha256Digest, ...] = ()
    supplemental_metric_sources: tuple[SupplementalMetricSourceSpec, ...] = ()

    @model_validator(mode="after")
    def columns_are_unique(self) -> TableSpec:
        if not self.required_columns:
            raise ValueError("required_columns must not be empty")
        if len(set(self.required_columns)) != len(self.required_columns):
            raise ValueError("required_columns must be unique")
        if len(set(self.display_columns)) != len(self.display_columns):
            raise ValueError("display_columns must be unique")
        if self.display_columns and not set(self.display_columns).issubset(
            self.required_columns
        ):
            raise ValueError("display_columns must be required canonical-table columns")
        if len(self.display_columns) > 8:
            raise ValueError("PDF display tables are limited to eight readable columns")
        roles = tuple(item.source_role for item in self.supplemental_metric_sources)
        if roles != tuple(sorted(set(roles))):
            raise ValueError("supplemental metric source roles must be sorted and unique")
        expected = self.table_id in REGISTERED_SUPPLEMENTAL_METRIC_TABLE_IDS
        if self.status is ReportStatus.COMPLETE and expected != bool(
            self.supplemental_metric_sources
        ):
            raise ValueError(
                "complete table does not carry its registered supplemental metric sources"
            )
        hashes = {
            digest
            for source in self.supplemental_metric_sources
            for digest in (
                source.source_file_sha256,
                source.table_manifest_file_sha256,
            )
        }
        if not hashes.issubset(self.source_artifact_hashes):
            raise ValueError("supplemental metric sources are absent from table lineage")
        return self


class ReviewSupplementSpec(FrozenModel):
    """Sanitized, mechanically derived table from a restricted blinded review."""

    supplement_id: Literal["community_blind_review", "held_out_error_review"]
    section_id: Literal["community", "error_limitations"]
    relative_path: Annotated[
        str,
        StringConstraints(pattern=r"^tables/[A-Za-z0-9_.-]+\.csv$"),
    ]
    sha256: Sha256Digest
    row_count: int = Field(gt=0)
    required_columns: tuple[Annotated[str, StringConstraints(min_length=1)], ...]
    display_columns: tuple[Annotated[str, StringConstraints(min_length=1)], ...]
    description: Annotated[str, StringConstraints(min_length=1, max_length=500)]
    source_artifact_hashes: tuple[Sha256Digest, ...]

    @model_validator(mode="after")
    def exact_public_shape(self) -> ReviewSupplementSpec:
        expected_section = {
            "community_blind_review": "community",
            "held_out_error_review": "error_limitations",
        }[self.supplement_id]
        if self.section_id != expected_section:
            raise ValueError("review supplement is attached to the wrong report section")
        if (
            not self.required_columns
            or len(self.required_columns) != len(set(self.required_columns))
            or not self.display_columns
            or len(self.display_columns) != len(set(self.display_columns))
            or not set(self.display_columns).issubset(self.required_columns)
        ):
            raise ValueError("review supplement columns are empty, duplicated, or inconsistent")
        if len(self.display_columns) > 8:
            raise ValueError("PDF display tables are limited to eight readable columns")
        if not self.source_artifact_hashes:
            raise ValueError("review supplement requires immutable source hashes")
        return self


class FigureSpec(FrozenModel):
    figure_id: Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]{1,63}$")]
    kind: Literal["phase_status", "forest", "artifact_png"]
    title: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    relative_path: Annotated[
        str,
        StringConstraints(pattern=r"^figures/[A-Za-z0-9_.-]+\.(?:pdf|png)$"),
    ]
    table_id: str | None = None
    label_column: str | None = None
    estimate_column: str | None = None
    lower_column: str | None = None
    upper_column: str | None = None
    section_id: str | None = None
    caption: Annotated[str, StringConstraints(min_length=1, max_length=500)] | None = None
    sha256: Sha256Digest | None = None
    source_artifact_hashes: tuple[Sha256Digest, ...] = ()

    @model_validator(mode="after")
    def forest_has_columns(self) -> FigureSpec:
        if self.kind == "forest" and not all(
            (
                self.table_id,
                self.label_column,
                self.estimate_column,
                self.lower_column,
                self.upper_column,
            )
        ):
            raise ValueError("forest figures require a table and four column names")
        if self.kind == "phase_status" and self.table_id is not None:
            raise ValueError("phase_status figures do not consume a table")
        if self.kind == "artifact_png":
            if not self.relative_path.endswith(".png"):
                raise ValueError("artifact figures must be PNG files")
            if not all((self.section_id, self.caption, self.sha256)):
                raise ValueError("artifact figures require section, caption, and SHA-256")
            if self.table_id is not None or any(
                (self.label_column, self.estimate_column, self.lower_column, self.upper_column)
            ):
                raise ValueError("artifact figures cannot claim a generated-table mapping")
            if not self.source_artifact_hashes:
                raise ValueError("artifact figures require immutable source lineage")
        elif any((self.section_id, self.caption, self.sha256, self.source_artifact_hashes)):
            raise ValueError("only artifact figures carry external-image metadata")
        if self.kind != "artifact_png" and not self.relative_path.endswith(".pdf"):
            raise ValueError("generated figures must be PDF files")
        return self


class SectionSpec(FrozenModel):
    section_id: Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]{1,63}$")]
    title: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    required_phases: tuple[Annotated[str, StringConstraints(pattern=r"^phase_[1-7]$")], ...]
    table_ids: tuple[str, ...] = ()
    fixed_text: tuple[Annotated[str, StringConstraints(min_length=1)], ...] = ()


class QualitativeSelectionRule(FrozenModel):
    example_id: Literal[
        "development_tutorial",
        "rare_pivotal",
        "temporal_epistemic",
        "held_out_illustration",
        "counterexample",
        "narrative_illustration",
    ]
    evidence_split: Literal["development", "held_out", "case_study"]
    selection_time: Literal[
        "before_condition_outputs",
        "rule_frozen_before_outputs_applied_after_itt_scoring",
    ]
    rule: Annotated[str, StringConstraints(min_length=1, max_length=1000)]
    tie_break: Annotated[str, StringConstraints(min_length=1, max_length=500)]
    report_label: Literal[
        "pedagogical development example",
        "qualitative held-out illustration",
        "qualitative failure illustration",
        "descriptive narrative illustration",
    ]

    @model_validator(mode="after")
    def counterexample_timing_is_honest(self) -> QualitativeSelectionRule:
        outcome_applied = (
            self.selection_time == "rule_frozen_before_outputs_applied_after_itt_scoring"
        )
        if outcome_applied != (self.example_id == "counterexample"):
            raise ValueError("only the declared ITT counterexample rule is applied after scoring")
        return self


class ReportingPolicy(FrozenModel):
    schema_version: Literal["1.0.0"] = REPORT_SCHEMA_VERSION
    policy_id: Annotated[str, StringConstraints(min_length=1, max_length=160)]
    required_complete_phase_ids: tuple[str, ...]
    required_complete_table_ids: tuple[str, ...]
    required_section_ids: tuple[str, ...]
    required_figure_ids: tuple[str, ...]
    qualitative_selection_rules: tuple[QualitativeSelectionRule, ...]
    claim_boundaries: tuple[Annotated[str, StringConstraints(min_length=1)], ...]
    section_contract_sha256: Sha256Digest
    policy_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_policy(self) -> ReportingPolicy:
        if set(self.required_complete_phase_ids) != {f"phase_{item}" for item in range(1, 8)}:
            raise ValueError("report policy must require all seven phases")
        if set(self.required_complete_table_ids) != REGISTERED_COMPLETE_TABLE_IDS:
            raise ValueError("report policy omits a registered complete-result table")
        if set(self.required_section_ids) != REGISTERED_SECTION_IDS:
            raise ValueError("report policy omits a registered report section")
        if set(self.required_figure_ids) != REGISTERED_FIGURE_IDS:
            raise ValueError("report policy omits a registered figure")
        example_ids = [item.example_id for item in self.qualitative_selection_rules]
        if len(example_ids) != len(set(example_ids)) or len(example_ids) != 6:
            raise ValueError("report policy must define each qualitative selection exactly once")
        return self


class ResultManifest(FrozenModel):
    schema_version: Literal["1.0.0"] = REPORT_SCHEMA_VERSION
    study_id: Annotated[str, StringConstraints(min_length=1, max_length=160)]
    study_status: ReportStatus
    status_reason: Annotated[str, StringConstraints(min_length=1, max_length=1000)]
    generated_at_utc: AwareDatetime
    code_revision: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
    dirty_worktree: bool
    model_repository: Annotated[str, StringConstraints(min_length=1, max_length=200)] | None
    model_revision: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")] | None
    phases: tuple[PhaseStatus, ...]
    tables: tuple[TableSpec, ...]
    review_supplements: tuple[ReviewSupplementSpec, ...] = ()
    figures: tuple[FigureSpec, ...]
    sections: tuple[SectionSpec, ...]
    source_artifact_hashes: tuple[Sha256Digest, ...]
    ingestion_receipt_relative_path: Annotated[
        str,
        StringConstraints(
            pattern=r"^report_ingestion_receipt(?:\.[0-9a-f]{12,64})?\.json$"
        ),
    ]
    ingestion_receipt_file_sha256: Sha256Digest
    ingestion_receipt_sha256: Sha256Digest
    reporting_policy_sha256: Sha256Digest
    manifest_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_manifest(self) -> ResultManifest:
        for values, label in (
            (self.phases, "phase_id"),
            (self.tables, "table_id"),
            (self.review_supplements, "supplement_id"),
            (self.figures, "figure_id"),
            (self.sections, "section_id"),
        ):
            keys = [getattr(item, label) for item in values]
            if len(keys) != len(set(keys)):
                raise ValueError(f"duplicate {label}")
        phase_ids = {item.phase_id for item in self.phases}
        table_ids = {item.table_id for item in self.tables}
        for section in self.sections:
            if not set(section.required_phases).issubset(phase_ids):
                raise ValueError(f"unknown phase in section {section.section_id}")
            if not set(section.table_ids).issubset(table_ids):
                raise ValueError(f"unknown table in section {section.section_id}")
        for figure in self.figures:
            if figure.table_id is not None and figure.table_id not in table_ids:
                raise ValueError(f"unknown table in figure {figure.figure_id}")
        return self


@dataclass(frozen=True, slots=True)
class LoadedTable:
    spec: TableSpec
    columns: tuple[str, ...]
    rows: tuple[Mapping[str, str], ...]


@dataclass(frozen=True, slots=True)
class LoadedReviewSupplement:
    spec: ReviewSupplementSpec
    columns: tuple[str, ...]
    rows: tuple[Mapping[str, str], ...]


@dataclass(frozen=True, slots=True)
class ReportSection:
    spec: SectionSpec
    status: ReportStatus
    reason: str
    tables: tuple[LoadedTable, ...]
    review_supplements: tuple[LoadedReviewSupplement, ...]


@dataclass(frozen=True, slots=True)
class ReportDocument:
    manifest: ResultManifest
    policy: ReportingPolicy
    sections: tuple[ReportSection, ...]
    tables: Mapping[str, LoadedTable]
    review_supplements: Mapping[str, LoadedReviewSupplement]
    ingestion_receipt: Any


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_manifest_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a manifest payload carrying its canonical self-hash."""

    result = dict(payload)
    result.pop("manifest_sha256", None)
    result["manifest_sha256"] = canonical_sha256(result)
    return result


def section_contract_sha256(sections: Iterable[SectionSpec]) -> str:
    """Hash pre-result prose and gates without freezing future table attachment."""

    return canonical_sha256(
        [
            {
                "section_id": item.section_id,
                "title": item.title,
                "required_phases": list(item.required_phases),
                "fixed_text": list(item.fixed_text),
            }
            for item in sections
        ]
    )


def _assert_no_symlink_chain(path: Path) -> None:
    current = path.absolute()
    while True:
        if current.is_symlink():
            raise ReportingError(f"symlink is prohibited in immutable inputs: {path}")
        if current.parent == current:
            return
        current = current.parent


def _safe_under(root: Path, relative_path: str) -> Path:
    if "\\" in relative_path:
        raise ReportingError(f"backslash is prohibited in report path: {relative_path}")
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ReportingError(f"unsafe report path: {relative_path}")
    candidate = root / relative
    _assert_no_symlink_chain(candidate)
    resolved_root = root.resolve()
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(resolved_root):
        raise ReportingError(f"report input escapes root: {relative_path}")
    return resolved


def load_result_manifest(path: Path) -> ResultManifest:
    _assert_no_symlink_chain(path)
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ReportingError("result manifest must be UTF-8 without BOM")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReportingError(f"invalid result manifest: {exc}") from exc
    supplied_hash = payload.get("manifest_sha256")
    immutable = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    if supplied_hash != canonical_sha256(immutable):
        raise ReportingError("manifest_sha256 does not match canonical manifest content")
    return ResultManifest.model_validate(payload)


def load_reporting_policy(path: Path) -> ReportingPolicy:
    _assert_no_symlink_chain(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReportingError(f"invalid reporting policy: {exc}") from exc
    supplied_hash = payload.get("policy_sha256")
    immutable = {key: value for key, value in payload.items() if key != "policy_sha256"}
    if supplied_hash != canonical_sha256(immutable):
        raise ReportingError("policy_sha256 does not match canonical policy content")
    return ReportingPolicy.model_validate(payload)


def _validate_complete_gate(manifest: ResultManifest, policy: ReportingPolicy) -> None:
    if manifest.reporting_policy_sha256 != policy.policy_sha256:
        raise ReportingError("result manifest is not bound to the supplied reporting policy")
    if {item.section_id for item in manifest.sections} == REGISTERED_SECTION_IDS and (
        section_contract_sha256(manifest.sections) != policy.section_contract_sha256
    ):
        raise ReportingError("report section prose differs from the frozen pre-result contract")
    if manifest.study_status is not ReportStatus.COMPLETE:
        return
    if manifest.dirty_worktree:
        raise ReportingError("a dirty-worktree report cannot be labeled complete")
    if not manifest.model_repository or not manifest.model_revision:
        raise ReportingError("a complete report requires the frozen model and immutable revision")
    if not manifest.source_artifact_hashes:
        raise ReportingError("a complete report requires source artifact hashes")
    phases = {item.phase_id: item for item in manifest.phases}
    missing_phases = set(policy.required_complete_phase_ids) - set(phases)
    incomplete_phases = {
        phase_id
        for phase_id in policy.required_complete_phase_ids
        if phase_id in phases and phases[phase_id].status is not ReportStatus.COMPLETE
    }
    if missing_phases or incomplete_phases:
        raise ReportingError(
            "complete report has missing/incomplete phases: "
            + ", ".join(sorted(missing_phases | incomplete_phases))
        )
    phases_without_sources = [
        phase_id
        for phase_id in policy.required_complete_phase_ids
        if not phases[phase_id].source_artifact_hashes
    ]
    if phases_without_sources:
        raise ReportingError(
            "complete phase status lacks immutable source hashes: "
            + ", ".join(phases_without_sources)
        )
    tables = {item.table_id: item for item in manifest.tables}
    missing_tables = set(policy.required_complete_table_ids) - set(tables)
    incomplete_tables = {
        table_id
        for table_id in policy.required_complete_table_ids
        if table_id in tables and tables[table_id].status is not ReportStatus.COMPLETE
    }
    if missing_tables or incomplete_tables:
        raise ReportingError(
            "complete report has missing/incomplete tables: "
            + ", ".join(sorted(missing_tables | incomplete_tables))
        )
    tables_without_sources = [
        table_id
        for table_id in policy.required_complete_table_ids
        if not tables[table_id].source_artifact_hashes
    ]
    if tables_without_sources:
        raise ReportingError(
            "complete table lacks upstream artifact hashes: " + ", ".join(tables_without_sources)
        )
    empty_tables = [
        table_id
        for table_id in policy.required_complete_table_ids
        if tables[table_id].row_count == 0
    ]
    if empty_tables:
        raise ReportingError(
            "complete report has empty registered tables: " + ", ".join(empty_tables)
        )
    if set(policy.required_section_ids) - {item.section_id for item in manifest.sections}:
        raise ReportingError("complete report omits one or more registered sections")
    supplement_ids = {item.supplement_id for item in manifest.review_supplements}
    if supplement_ids != REGISTERED_REVIEW_SUPPLEMENT_IDS:
        raise ReportingError(
            "complete report omits a blinded-review supplement: "
            + ", ".join(sorted(REGISTERED_REVIEW_SUPPLEMENT_IDS - supplement_ids))
        )
    referenced_tables = set(chain.from_iterable(item.table_ids for item in manifest.sections))
    unreported_tables = set(policy.required_complete_table_ids) - referenced_tables
    if unreported_tables:
        raise ReportingError(
            "complete report does not render registered tables: "
            + ", ".join(sorted(unreported_tables))
        )
    if set(policy.required_figure_ids) - {item.figure_id for item in manifest.figures}:
        raise ReportingError("complete report omits one or more registered figures")


def load_canonical_table(root: Path, spec: TableSpec) -> LoadedTable:
    path = _safe_under(root, spec.relative_path)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != spec.sha256:
        raise ReportingError(f"table hash mismatch: {spec.table_id}")
    if raw.startswith(b"\xef\xbb\xbf") or b"\r" in raw:
        raise ReportingError(f"table is not canonical UTF-8/LF CSV: {spec.table_id}")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ReportingError(f"table is not UTF-8: {spec.table_id}") from exc
    reader = csv.DictReader(text.splitlines())
    columns = tuple(reader.fieldnames or ())
    if len(columns) != len(set(columns)):
        raise ReportingError(f"duplicate CSV columns: {spec.table_id}")
    missing = set(spec.required_columns) - set(columns)
    if missing:
        raise ReportingError(
            f"table {spec.table_id} lacks required columns: {', '.join(sorted(missing))}"
        )
    rows = tuple(dict(row) for row in reader)
    if len(rows) != spec.row_count:
        raise ReportingError(
            f"table {spec.table_id} row count {len(rows)} != manifest {spec.row_count}"
        )
    return LoadedTable(spec=spec, columns=columns, rows=rows)


def load_review_supplement(
    root: Path, spec: ReviewSupplementSpec
) -> LoadedReviewSupplement:
    path = _safe_under(root, spec.relative_path)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != spec.sha256:
        raise ReportingError(f"review supplement hash mismatch: {spec.supplement_id}")
    if raw.startswith(b"\xef\xbb\xbf") or b"\r" in raw:
        raise ReportingError(
            f"review supplement is not canonical UTF-8/LF CSV: {spec.supplement_id}"
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ReportingError(f"review supplement is not UTF-8: {spec.supplement_id}") from exc
    reader = csv.DictReader(text.splitlines())
    columns = tuple(reader.fieldnames or ())
    if not columns or len(columns) != len(set(columns)):
        raise ReportingError(f"review supplement has invalid columns: {spec.supplement_id}")
    if not set(spec.required_columns).issubset(columns):
        raise ReportingError(f"review supplement lacks columns: {spec.supplement_id}")
    rows = tuple(dict(row) for row in reader)
    if len(rows) != spec.row_count:
        raise ReportingError(f"review supplement row count changed: {spec.supplement_id}")
    regenerated = io.StringIO(newline="")
    writer = csv.DictWriter(regenerated, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    if regenerated.getvalue().encode("utf-8") != raw:
        raise ReportingError(
            f"review supplement CSV serialization is not canonical: {spec.supplement_id}"
        )
    return LoadedReviewSupplement(spec=spec, columns=columns, rows=rows)


def build_document(manifest_path: Path, policy_path: Path) -> ReportDocument:
    manifest = load_result_manifest(manifest_path)
    policy = load_reporting_policy(policy_path)
    _validate_complete_gate(manifest, policy)
    root = manifest_path.parent
    from story_projection_onto.report_ingestion import load_ingestion_receipt

    receipt_path = _safe_under(root, manifest.ingestion_receipt_relative_path)
    if _file_sha256(receipt_path) != manifest.ingestion_receipt_file_sha256:
        raise ReportingError("reporting ingestion receipt file hash changed")
    ingestion_receipt = load_ingestion_receipt(receipt_path)
    if ingestion_receipt.receipt_sha256 != manifest.ingestion_receipt_sha256:
        raise ReportingError("result manifest binds another reporting ingestion receipt")
    tables = {item.table_id: load_canonical_table(root, item) for item in manifest.tables}
    review_supplements = {
        item.supplement_id: load_review_supplement(root, item)
        for item in manifest.review_supplements
    }
    receipt_tables = {item.table_id: item for item in ingestion_receipt.tables}
    available_receipt_tables = {
        table_id
        for table_id, item in receipt_tables.items()
        if item.status is ReportStatus.COMPLETE
    }
    if set(tables) != available_receipt_tables:
        raise ReportingError("result table inventory differs from the ingestion receipt")
    for table_id, table in tables.items():
        ingested = receipt_tables[table_id]
        if (
            ingested.relative_path != table.spec.relative_path
            or ingested.file_sha256 != table.spec.sha256
            or ingested.row_count != table.spec.row_count
            or ingested.status is not table.spec.status
            or not set(table.spec.required_columns).issubset(ingested.columns)
            or ingested.source_artifact_hashes != table.spec.source_artifact_hashes
        ):
            raise ReportingError(f"table {table_id} differs from its ingestion receipt")
    expected_source_hashes = tuple(item.file_sha256 for item in ingestion_receipt.artifacts)
    if manifest.source_artifact_hashes != expected_source_hashes:
        raise ReportingError("result-manifest sources differ from verified ingestion artifacts")
    verified_source_hashes = set(expected_source_hashes)
    for supplement in review_supplements.values():
        if not set(supplement.spec.source_artifact_hashes).issubset(
            verified_source_hashes
        ):
            raise ReportingError(
                f"review supplement {supplement.spec.supplement_id} cites an "
                "unverified predecessor artifact"
            )
    for phase in manifest.phases:
        if not set(phase.source_artifact_hashes).issubset(verified_source_hashes):
            raise ReportingError(f"phase {phase.phase_id} cites an unverified predecessor artifact")
    section_ids = {item.section_id for item in manifest.sections}
    for figure in manifest.figures:
        if figure.kind != "artifact_png":
            continue
        if figure.section_id not in section_ids:
            raise ReportingError(f"artifact figure {figure.figure_id} names an unknown section")
        if not set(figure.source_artifact_hashes).issubset(verified_source_hashes):
            raise ReportingError(
                f"artifact figure {figure.figure_id} cites an unverified predecessor artifact"
            )
        figure_path = _safe_under(root, figure.relative_path)
        if _file_sha256(figure_path) != figure.sha256:
            raise ReportingError(f"artifact figure hash mismatch: {figure.figure_id}")
    if manifest.study_status is ReportStatus.COMPLETE:
        incomplete_predecessors = [
            item.family.value
            for item in ingestion_receipt.predecessors
            if item.status is not ReportStatus.COMPLETE
        ]
        nonfinal_tables = [
            item.table_id
            for item in ingestion_receipt.tables
            if item.status is not ReportStatus.COMPLETE or item.scope != "final"
        ]
        if incomplete_predecessors or nonfinal_tables:
            raise ReportingError(
                "complete report has incomplete ingestion gates: "
                + ", ".join(sorted((*incomplete_predecessors, *nonfinal_tables)))
            )
    status_table = tables.get("study_status")
    if status_table is not None:
        expected_status = {
            item.phase_id: (item.status.value, item.reason) for item in manifest.phases
        }
        observed_status = {
            row["phase_id"]: (row["status"], row["reason"]) for row in status_table.rows
        }
        if observed_status != expected_status:
            raise ReportingError("study_status table does not match result-manifest phase gates")
    phases = {item.phase_id: item for item in manifest.phases}
    sections: list[ReportSection] = []
    for spec in manifest.sections:
        phase_failures = [
            phases[phase_id]
            for phase_id in spec.required_phases
            if phases[phase_id].status is not ReportStatus.COMPLETE
        ]
        table_failures = [
            tables[table_id].spec
            for table_id in spec.table_ids
            if tables[table_id].spec.status is not ReportStatus.COMPLETE
        ]
        if phase_failures or table_failures:
            failures = [f"{item.label}: {item.reason}" for item in phase_failures]
            failures.extend(f"{item.table_id}: {item.description}" for item in table_failures)
            statuses = {item.status for item in (*phase_failures, *table_failures)}
            status = (
                ReportStatus.BLOCKED
                if ReportStatus.BLOCKED in statuses
                else ReportStatus.INCOMPLETE
            )
            reason = "; ".join(failures) or "Required source artifacts are incomplete."
        else:
            status = ReportStatus.COMPLETE
            reason = "All registered source gates for this section are complete."
        sections.append(
            ReportSection(
                spec=spec,
                status=status,
                reason=reason,
                tables=tuple(tables[table_id] for table_id in spec.table_ids),
                review_supplements=tuple(
                    supplement
                    for supplement in review_supplements.values()
                    if supplement.spec.section_id == spec.section_id
                ),
            )
        )
    return ReportDocument(
        manifest=manifest,
        policy=policy,
        sections=tuple(sections),
        tables=tables,
        review_supplements=review_supplements,
        ingestion_receipt=ingestion_receipt,
    )


def _markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _report_table_id(table: LoadedTable | LoadedReviewSupplement) -> str:
    return (
        table.spec.table_id
        if isinstance(table, LoadedTable)
        else table.spec.supplement_id
    )


def _markdown_table(table: LoadedTable | LoadedReviewSupplement) -> str:
    lines = [
        (
            f"<!-- table:{_report_table_id(table)} sha256:{table.spec.sha256} "
            f"rows:{table.spec.row_count} -->"
        ),
        "| " + " | ".join(_markdown_cell(item) for item in table.columns) + " |",
        "| " + " | ".join("---" for _ in table.columns) + " |",
    ]
    for row in table.rows:
        lines.append("| " + " | ".join(_markdown_cell(row[item]) for item in table.columns) + " |")
    return "\n".join(lines)


def _markdown_supplemental_metric_sources(table: LoadedTable) -> str:
    if not table.spec.supplemental_metric_sources:
        return ""
    lines = [
        "#### Complete registered metric rows",
        "",
        (
            "The compact comparison above is accompanied by the following immutable public "
            "Phase 4 CSV rows. Values, metric statuses, numerators, and denominators are copied "
            "from these sources; Phase 7 does not recompute them."
        ),
        "",
    ]
    for source in table.spec.supplemental_metric_sources:
        counts = ", ".join(
            f"`{item.metric_name}` ({item.row_count})" for item in source.metric_row_counts
        )
        lines.extend(
            (
                f"- **{source.source_role}:** {source.description}",
                (
                    f"  Source: `{source.source_relative_path}`; SHA-256 "
                    f"`{source.source_file_sha256}`; {source.source_row_count} total rows, "
                    f"{source.selected_row_count} rows after the frozen source filter."
                ),
                f"  Registered metrics: {counts}.",
                (
                    f"  Producer manifest: `{source.table_manifest_relative_path}`; SHA-256 "
                    f"`{source.table_manifest_file_sha256}`."
                ),
                "",
            )
        )
    return "\n".join(lines)


def _figure_section_id(spec: FigureSpec) -> str | None:
    """Return the frozen report section that presents a figure.

    Generated figures predate the external-artifact ``section_id`` field, so their
    placement is fixed here rather than inferred from result values.  Artifact PNGs
    continue to carry an explicit section binding in the immutable result manifest.
    """

    if spec.kind == "phase_status":
        return "resource_controls"
    if spec.kind == "forest":
        return "primary_results"
    return spec.section_id


def _pdf_columns(table: LoadedTable | LoadedReviewSupplement) -> tuple[str, ...]:
    """Columns selected before outcomes for the compact PDF presentation.

    The Markdown source and canonical CSV retain every column.  The PDF uses the
    explicit presentation subset from the result manifest so dense provenance fields
    do not collapse into unreadable sub-centimetre columns.
    """

    return table.spec.display_columns or table.columns


def render_markdown(document: ReportDocument) -> str:
    manifest = document.manifest
    model = (
        f"`{manifest.model_repository}` at `{manifest.model_revision}`"
        if manifest.model_repository and manifest.model_revision
        else "not yet accepted/frozen"
    )
    lines = [
        f"# {REPORT_TITLE}",
        "",
        f"**Study status:** `{manifest.study_status.value}` — {manifest.status_reason}",
        "",
        f"**Frozen input manifest:** `{manifest.manifest_sha256}`",
        "",
        (
            f"**Code revision:** `{manifest.code_revision}` "
            f"(dirty worktree: `{str(manifest.dirty_worktree).lower()}`)"
        ),
        "",
        f"**Model:** {model}",
        "",
        "> This document is generated from hash-verified canonical tables. An incomplete or",
        "> blocked section is not evidence of a null result and contains no imputed scientific",
        "> value.",
        "",
        "## Study overview",
        "",
        (
            "Narrative evidence does not determine one universally useful ontology. This study "
            "tests whether revealing a user's context before ontology construction changes the "
            "semantic fidelity and organization of the resulting evidence-grounded graph."
        ),
        "",
        (
            "- **RQ1:** Does query-dependent LLM construction improve qualified-assertion and "
            "ontology-decision fidelity relative to query-blind LLM and classical construction?"
        ),
        "",
        (
            "- **RQ2:** How does construction timing change entropy and directly measured visual "
            "clutter when semantic and rare-pivotal safeguards remain visible?"
        ),
        "",
        (
            "- **RQ3:** How does construction timing change gold-aligned community structure and "
            "cross-seed cluster stability?"
        ),
        "",
        (
            "The query-blind evidence index stores only source-grounded evidence and retrieval "
            "metadata. It is not a hidden ontology. C2 must form contextual entities, events, "
            "schema, relations, abstractions, and qualified assertions after query reveal."
        ),
        "",
    ]
    for index, section in enumerate(document.sections, start=1):
        lines.extend((f"## {index}. {section.spec.title}", ""))
        if section.status is not ReportStatus.COMPLETE:
            lines.extend(
                (
                    f"**STATUS: {section.status.value.upper()}** — {section.reason}",
                    "",
                    "No confirmatory or descriptive outcome is available for this section.",
                    "",
                )
            )
        for paragraph in section.spec.fixed_text:
            lines.extend((paragraph, ""))
        if section.spec.section_id == "qualitative_examples":
            lines.extend(("### Pre-output selection rules", ""))
            for rule in document.policy.qualitative_selection_rules:
                lines.extend(
                    (
                        f"- **{rule.example_id} — {rule.report_label}:** {rule.rule} "
                        f"Tie-break: {rule.tie_break}",
                        "",
                    )
                )
        if section.spec.section_id == "claim_boundaries":
            lines.extend(("### Exact boundaries", ""))
            for boundary in document.policy.claim_boundaries:
                lines.append(f"- {boundary}")
            lines.append("")
        for table in section.tables:
            if table.spec.status is ReportStatus.COMPLETE:
                if section.status is not ReportStatus.COMPLETE:
                    lines.extend(
                        (
                            "_Interim verified table. It is not a completed section outcome._",
                            "",
                        )
                    )
                lines.extend((f"### {table.spec.description}", "", _markdown_table(table), ""))
                supplemental = _markdown_supplemental_metric_sources(table)
                if supplemental:
                    lines.extend((supplemental, ""))
        for supplement in section.review_supplements:
            lines.extend(
                (
                    f"### {supplement.spec.description}",
                    "",
                    _markdown_table(supplement),
                    "",
                )
            )
        for figure in document.manifest.figures:
            if _figure_section_id(figure) == section.spec.section_id:
                if figure.kind == "artifact_png":
                    rendered = (
                        f"![{_markdown_cell(figure.caption or figure.title)}]"
                        f"({figure.relative_path})"
                    )
                else:
                    rendered = (
                        f"[{_markdown_cell(figure.title)}]({figure.relative_path})"
                    )
                lines.extend(
                    (
                        f"### {figure.caption or figure.title}",
                        "",
                        rendered,
                        "",
                        (
                            f"<!-- figure:{figure.figure_id} "
                            f"sha256:{figure.sha256 or 'generated-from-canonical-table'} -->"
                        ),
                        "",
                    )
                )
    lines.extend(
        (
            "## Machine-readable provenance",
            "",
            f"Document source manifest SHA-256: `{manifest.manifest_sha256}`.",
            "",
            f"Verified ingestion receipt SHA-256: `{manifest.ingestion_receipt_sha256}`.",
            "",
            "Every displayed table carries its immutable CSV SHA-256 and row count in an HTML",
            "comment.",
            "Regenerate and verify this report with `scripts/build_results_report.py --verify`.",
            "",
        )
    )
    return "\n".join(lines)


def _reportlab() -> tuple[Any, ...]:
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            Image,
            LongTable,
            PageBreak,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            TableStyle,
        )
    except ImportError as exc:  # pragma: no cover - production dependency gate
        raise ReportingError(
            "PDF generation requires the pinned study dependency reportlab>=4.2,<5"
        ) from exc
    return (
        colors,
        TA_CENTER,
        letter,
        ParagraphStyle,
        getSampleStyleSheet,
        inch,
        Image,
        LongTable,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        TableStyle,
    )


def _generated_figure_flowable(document: ReportDocument, spec: FigureSpec) -> Any:
    """Create the vector figure that is embedded in the report PDF.

    Standalone figure PDFs are still emitted for publication reuse.  This flowable
    uses the same immutable phase/table values, closing the prior gap where generated
    figures were manifested but absent from the actual report.
    """

    try:
        from reportlab.graphics.shapes import Circle, Drawing, Line, Rect, String
        from reportlab.lib import colors
    except ImportError as exc:  # pragma: no cover - production dependency gate
        raise ReportingError("embedded figure generation requires reportlab") from exc

    width = 520.0
    if spec.kind == "phase_status":
        height = 42.0 + 38.0 * len(document.manifest.phases)
        drawing = Drawing(width, height)
        for index, phase in enumerate(document.manifest.phases):
            y = height - 34.0 - index * 38.0
            red, green, blue = _status_color(phase.status)
            drawing.add(
                Rect(
                    4,
                    y - 8,
                    92,
                    20,
                    rx=3,
                    ry=3,
                    fillColor=colors.Color(red, green, blue),
                    strokeColor=None,
                )
            )
            drawing.add(
                String(
                    50,
                    y - 2,
                    phase.status.value.upper(),
                    fontName="Helvetica-Bold",
                    fontSize=7.5,
                    textAnchor="middle",
                    fillColor=colors.white,
                )
            )
            drawing.add(
                String(
                    108,
                    y + 3,
                    phase.label[:70],
                    fontName="Helvetica-Bold",
                    fontSize=8.5,
                    fillColor=colors.HexColor("#102A43"),
                )
            )
            reason = phase.reason if len(phase.reason) <= 92 else phase.reason[:89] + "..."
            drawing.add(
                String(
                    108,
                    y - 8,
                    reason,
                    fontName="Helvetica",
                    fontSize=6.5,
                    fillColor=colors.HexColor("#486581"),
                )
            )
        return drawing

    if spec.kind != "forest":
        raise ReportingError(f"unsupported generated figure kind: {spec.kind}")
    table = document.tables[spec.table_id or ""]
    label = spec.label_column or ""
    estimate = spec.estimate_column or ""
    lower = spec.lower_column or ""
    upper = spec.upper_column or ""
    required = {label, estimate, lower, upper}
    if not required.issubset(table.columns):
        raise ReportingError(f"forest figure {spec.figure_id} columns absent from table")
    values: list[tuple[str, float, float, float]] = []
    for row in table.rows:
        try:
            triple = (float(row[estimate]), float(row[lower]), float(row[upper]))
        except ValueError as exc:
            raise ReportingError(
                f"forest figure {spec.figure_id} contains nonnumeric values"
            ) from exc
        if not all(math.isfinite(item) for item in triple):
            raise ReportingError(f"forest figure {spec.figure_id} contains nonfinite values")
        if not triple[1] <= triple[0] <= triple[2]:
            raise ReportingError(f"forest figure {spec.figure_id} has unordered interval")
        values.append((row[label], *triple))
    if not values:
        raise ReportingError(f"forest figure {spec.figure_id} has no rows")
    height = max(120.0, 58.0 + 28.0 * len(values))
    drawing = Drawing(width, height)
    domain_low = min(0.0, *(item[2] for item in values))
    domain_high = max(0.0, *(item[3] for item in values))
    span = domain_high - domain_low or 1.0
    x0, x1 = 190.0, width - 16.0

    def position(value: float) -> float:
        return x0 + (value - domain_low) / span * (x1 - x0)

    drawing.add(
        Line(
            position(0.0),
            30,
            position(0.0),
            height - 20,
            strokeColor=colors.HexColor("#9FB3C8"),
            strokeWidth=0.8,
        )
    )
    for index, (row_label, point, low, high) in enumerate(values):
        y = height - 30.0 - index * 28.0
        drawing.add(
            String(
                x0 - 10,
                y - 3,
                row_label[:44],
                fontName="Helvetica",
                fontSize=8,
                textAnchor="end",
                fillColor=colors.HexColor("#102A43"),
            )
        )
        drawing.add(
            Line(
                position(low),
                y,
                position(high),
                y,
                strokeColor=colors.HexColor("#26547C"),
                strokeWidth=2,
            )
        )
        drawing.add(
            Circle(
                position(point),
                y,
                3.5,
                fillColor=colors.HexColor("#B22D43"),
                strokeColor=None,
            )
        )
    drawing.add(
        String(
            x0,
            12,
            f"{domain_low:.3g}",
            fontName="Helvetica",
            fontSize=7,
            fillColor=colors.HexColor("#102A43"),
        )
    )
    drawing.add(
        String(
            x1,
            12,
            f"{domain_high:.3g}",
            fontName="Helvetica",
            fontSize=7,
            textAnchor="end",
            fillColor=colors.HexColor("#102A43"),
        )
    )
    return drawing


def render_pdf(document: ReportDocument, output_path: Path) -> None:
    (
        colors,
        center,
        letter,
        paragraph_style,
        get_styles,
        inch,
        image,
        long_table,
        page_break,
        paragraph,
        document_template,
        spacer,
        table_style,
    ) = _reportlab()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    styles = get_styles()
    styles["Heading1"].keepWithNext = True
    styles["Heading2"].keepWithNext = True
    styles.add(paragraph_style(name="ReportTitle", parent=styles["Title"], alignment=center))
    styles.add(
        paragraph_style(
            name="ReportTable",
            parent=styles["BodyText"],
            fontSize=6.7,
            leading=8.0,
            wordWrap="CJK",
        )
    )
    styles.add(
        paragraph_style(
            name="ReportTableHeader",
            parent=styles["ReportTable"],
            textColor=colors.white,
            fontName="Helvetica-Bold",
        )
    )
    styles.add(
        paragraph_style(
            name="Status",
            parent=styles["BodyText"],
            borderColor=colors.HexColor("#8B1E3F"),
            borderWidth=0.8,
            borderPadding=8,
            backColor=colors.HexColor("#F8E9EE"),
            spaceAfter=10,
        )
    )
    doc = document_template(
        str(output_path),
        pagesize=letter,
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
        title=REPORT_TITLE,
        author="StoryProjectionOnto reproducible reporting pipeline",
        subject=f"source manifest {document.manifest.manifest_sha256}",
        pageCompression=0,
        invariant=1,
    )
    story: list[Any] = [
        paragraph(REPORT_TITLE, styles["ReportTitle"]),
        paragraph(
            f"Study status: <b>{document.manifest.study_status.value.upper()}</b> — "
            + html.escape(document.manifest.status_reason),
            styles["Status"],
        ),
        paragraph(
            "This report is generated from hash-verified canonical tables. Missing results are "
            "marked incomplete; no values are imputed.",
            styles["BodyText"],
        ),
        spacer(1, 10),
        paragraph("Study overview", styles["Heading1"]),
        paragraph(
            "Narrative evidence does not determine one universally useful ontology. This study "
            "tests whether revealing a user's context before ontology construction changes the "
            "semantic fidelity and organization of the resulting evidence-grounded graph.",
            styles["BodyText"],
        ),
        paragraph(
            "<b>RQ1:</b> Does query-dependent LLM construction improve qualified-assertion and "
            "ontology-decision fidelity relative to query-blind LLM and classical construction?",
            styles["BodyText"],
        ),
        paragraph(
            "<b>RQ2:</b> How does construction timing change entropy and directly measured visual "
            "clutter when semantic and rare-pivotal safeguards remain visible?",
            styles["BodyText"],
        ),
        paragraph(
            "<b>RQ3:</b> How does construction timing change gold-aligned community structure and "
            "cross-seed cluster stability?",
            styles["BodyText"],
        ),
        paragraph(
            "The query-blind evidence index stores only source-grounded evidence and retrieval "
            "metadata; it is not a hidden ontology. C2 forms contextual entities, events, schema, "
            "relations, abstractions, and qualified assertions after query reveal.",
            styles["BodyText"],
        ),
        spacer(1, 10),
    ]
    for index, section in enumerate(document.sections, start=1):
        story.append(paragraph(f"{index}. {html.escape(section.spec.title)}", styles["Heading1"]))
        if section.status is not ReportStatus.COMPLETE:
            story.append(
                paragraph(
                    f"STATUS: {section.status.value.upper()} — {html.escape(section.reason)}",
                    styles["Status"],
                )
            )
            story.append(
                paragraph(
                    "No confirmatory or descriptive outcome is available for this section.",
                    styles["BodyText"],
                )
            )
        for text in section.spec.fixed_text:
            story.append(paragraph(html.escape(text), styles["BodyText"]))
            story.append(spacer(1, 5))
        if section.spec.section_id == "qualitative_examples":
            story.append(paragraph("Pre-output selection rules", styles["Heading2"]))
            for rule in document.policy.qualitative_selection_rules:
                story.append(
                    paragraph(
                        f"<b>{html.escape(rule.example_id)} — "
                        f"{html.escape(rule.report_label)}:</b> {html.escape(rule.rule)} "
                        f"Tie-break: {html.escape(rule.tie_break)}",
                        styles["BodyText"],
                    )
                )
                story.append(spacer(1, 4))
        if section.spec.section_id == "claim_boundaries":
            story.append(paragraph("Exact boundaries", styles["Heading2"]))
            for boundary in document.policy.claim_boundaries:
                story.append(paragraph("• " + html.escape(boundary), styles["BodyText"]))
        for table in section.tables:
            if table.spec.status is ReportStatus.COMPLETE:
                if section.status is not ReportStatus.COMPLETE:
                    story.append(
                        paragraph(
                            "Interim verified table. It is not a completed section outcome.",
                            styles["BodyText"],
                        )
                    )
                story.append(paragraph(html.escape(table.spec.description), styles["Heading2"]))
                display_columns = _pdf_columns(table)
                matrix = [
                    [
                        paragraph(html.escape(col), styles["ReportTableHeader"])
                        for col in display_columns
                    ]
                ]
                matrix.extend(
                    [
                        paragraph(html.escape(row[col]), styles["ReportTable"])
                        for col in display_columns
                    ]
                    for row in table.rows
                )
                weights = []
                for column in display_columns:
                    longest = max(
                        (len(column), *(min(len(row[column]), 48) for row in table.rows)),
                    )
                    weights.append(max(6, min(longest, 32)))
                total_weight = sum(weights) or 1
                widths = [7.3 * inch * weight / total_weight for weight in weights]
                rendered = long_table(matrix, colWidths=widths, repeatRows=1, splitByRow=True)
                rendered.setStyle(
                    table_style(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#243B53")),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#9FB3C8")),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("LEFTPADDING", (0, 0), (-1, -1), 3),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                        ]
                    )
                )
                story.extend((rendered, spacer(1, 8)))
                if table.spec.supplemental_metric_sources:
                    story.append(
                        paragraph("Complete registered metric rows", styles["Heading2"])
                    )
                    story.append(
                        paragraph(
                            "The compact table is accompanied by immutable public Phase 4 CSV "
                            "rows. Phase 7 verifies and reports their lineage without recomputing "
                            "the scientific values.",
                            styles["BodyText"],
                        )
                    )
                    for source in table.spec.supplemental_metric_sources:
                        metrics = ", ".join(
                            f"{item.metric_name} ({item.row_count})"
                            for item in source.metric_row_counts
                        )
                        story.append(
                            paragraph(
                                f"<b>{html.escape(source.source_role)}:</b> "
                                f"{html.escape(source.description)} Source "
                                f"<font name='Courier'>{html.escape(source.source_relative_path)}"
                                "</font>; SHA-256 "
                                f"<font name='Courier'>{source.source_file_sha256}</font>; "
                                f"{source.source_row_count} total and "
                                f"{source.selected_row_count} filter-selected rows.",
                                styles["BodyText"],
                            )
                        )
                        story.append(
                            paragraph(
                                "Registered metrics: " + html.escape(metrics) + ".",
                                styles["BodyText"],
                            )
                        )
                        story.append(spacer(1, 5))
        for supplement in section.review_supplements:
            story.append(
                paragraph(html.escape(supplement.spec.description), styles["Heading2"])
            )
            display_columns = _pdf_columns(supplement)
            matrix = [
                [
                    paragraph(html.escape(column), styles["ReportTableHeader"])
                    for column in display_columns
                ]
            ]
            matrix.extend(
                [
                    paragraph(html.escape(row[column]), styles["ReportTable"])
                    for column in display_columns
                ]
                for row in supplement.rows
            )
            weights = [
                max(
                    6,
                    min(
                        max(
                            len(column),
                            *(min(len(row[column]), 48) for row in supplement.rows),
                        ),
                        32,
                    ),
                )
                for column in display_columns
            ]
            total_weight = sum(weights) or 1
            widths = [7.3 * inch * weight / total_weight for weight in weights]
            rendered = long_table(
                matrix,
                colWidths=widths,
                repeatRows=1,
                splitByRow=True,
            )
            rendered.setStyle(
                table_style(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#243B53")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#9FB3C8")),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 3),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                    ]
                )
            )
            story.extend((rendered, spacer(1, 8)))
        for figure in document.manifest.figures:
            if _figure_section_id(figure) != section.spec.section_id:
                continue
            story.append(paragraph(html.escape(figure.caption or figure.title), styles["Heading2"]))
            if figure.kind == "artifact_png":
                figure_path = _safe_under(output_path.parent, figure.relative_path)
                rendered_image = image(str(figure_path))
                available_width = 7.3 * inch
                available_height = 6.0 * inch
                scale = min(
                    available_width / rendered_image.imageWidth,
                    available_height / rendered_image.imageHeight,
                    1.0,
                )
                rendered_image.drawWidth = rendered_image.imageWidth * scale
                rendered_image.drawHeight = rendered_image.imageHeight * scale
                story.extend((rendered_image, spacer(1, 8)))
            else:
                story.extend((_generated_figure_flowable(document, figure), spacer(1, 8)))
        if index in {5, 10, 15}:
            story.append(page_break())
    story.extend(
        (
            paragraph("Machine-readable provenance", styles["Heading1"]),
            paragraph(
                "Source manifest SHA-256: " + document.manifest.manifest_sha256,
                styles["BodyText"],
            ),
        )
    )
    doc.build(story)


def _status_color(status: ReportStatus) -> tuple[float, float, float]:
    return {
        ReportStatus.COMPLETE: (0.18, 0.55, 0.34),
        ReportStatus.INCOMPLETE: (0.84, 0.55, 0.12),
        ReportStatus.BLOCKED: (0.70, 0.18, 0.25),
        ReportStatus.NOT_APPLICABLE: (0.45, 0.49, 0.53),
    }[status]


def _draw_figure(document: ReportDocument, spec: FigureSpec, output_path: Path) -> None:
    try:
        from reportlab.pdfgen import canvas
    except ImportError as exc:  # pragma: no cover - production dependency gate
        raise ReportingError("figure generation requires reportlab") from exc
    output_path.parent.mkdir(parents=True, exist_ok=True)
    page = canvas.Canvas(
        str(output_path),
        pagesize=(720, 432),
        pageCompression=0,
        invariant=1,
    )
    page.setTitle(spec.title)
    page.setAuthor("StoryProjectionOnto reproducible reporting pipeline")
    page.setSubject(f"source manifest {document.manifest.manifest_sha256}")
    page.setFont("Helvetica-Bold", 16)
    page.drawString(42, 395, spec.title)
    page.setFont("Helvetica", 9)
    page.drawRightString(678, 397, f"manifest {document.manifest.manifest_sha256[:12]}")
    if spec.kind == "phase_status":
        for index, phase in enumerate(document.manifest.phases):
            y = 355 - index * 43
            red, green, blue = _status_color(phase.status)
            page.setFillColorRGB(red, green, blue)
            page.roundRect(42, y - 12, 120, 24, 4, fill=1, stroke=0)
            page.setFillColorRGB(1, 1, 1)
            page.setFont("Helvetica-Bold", 9)
            page.drawCentredString(102, y - 3, phase.status.value.upper())
            page.setFillColorRGB(0.1, 0.15, 0.2)
            page.setFont("Helvetica-Bold", 10)
            page.drawString(178, y + 2, phase.label)
            page.setFont("Helvetica", 8)
            reason = phase.reason if len(phase.reason) <= 90 else phase.reason[:87] + "..."
            page.drawString(178, y - 10, reason)
    else:
        table = document.tables[spec.table_id or ""]
        label = spec.label_column or ""
        estimate = spec.estimate_column or ""
        lower = spec.lower_column or ""
        upper = spec.upper_column or ""
        required = {label, estimate, lower, upper}
        if not required.issubset(table.columns):
            raise ReportingError(f"forest figure {spec.figure_id} columns absent from table")
        values: list[tuple[str, float, float, float]] = []
        for row in table.rows:
            try:
                estimate_value = float(row[estimate])
                lower_value = float(row[lower])
                upper_value = float(row[upper])
            except ValueError as exc:
                raise ReportingError(
                    f"forest figure {spec.figure_id} contains nonnumeric values"
                ) from exc
            if not all(math.isfinite(item) for item in (estimate_value, lower_value, upper_value)):
                raise ReportingError(f"forest figure {spec.figure_id} contains nonfinite values")
            if lower_value > estimate_value or estimate_value > upper_value:
                raise ReportingError(f"forest figure {spec.figure_id} has unordered interval")
            values.append((row[label], estimate_value, lower_value, upper_value))
        if not values:
            raise ReportingError(f"forest figure {spec.figure_id} has no rows")
        domain_low = min(0.0, *(item[2] for item in values))
        domain_high = max(0.0, *(item[3] for item in values))
        span = domain_high - domain_low or 1.0
        x0, x1 = 245.0, 670.0

        def position(value: float) -> float:
            return x0 + (value - domain_low) / span * (x1 - x0)

        page.setStrokeColorRGB(0.65, 0.68, 0.72)
        page.line(position(0.0), 65, position(0.0), 360)
        for index, (row_label, point, low, high) in enumerate(values[:10]):
            y = 345 - index * 27
            page.setFillColorRGB(0.1, 0.15, 0.2)
            page.setFont("Helvetica", 9)
            page.drawRightString(230, y - 3, row_label[:38])
            page.setStrokeColorRGB(0.15, 0.34, 0.52)
            page.setLineWidth(2)
            page.line(position(low), y, position(high), y)
            page.setFillColorRGB(0.70, 0.18, 0.25)
            page.circle(position(point), y, 4, fill=1, stroke=0)
        page.setFillColorRGB(0.1, 0.15, 0.2)
        page.setFont("Helvetica", 8)
        page.drawString(x0, 45, f"{domain_low:.3g}")
        page.drawRightString(x1, 45, f"{domain_high:.3g}")
        page.drawCentredString((x0 + x1) / 2, 28, "paired estimate with two-sided 95% interval")
    page.showPage()
    page.save()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def build_results_report(
    manifest_path: Path,
    policy_path: Path,
    output_root: Path | None = None,
) -> dict[str, Any]:
    document = build_document(manifest_path, policy_path)
    root = output_root or manifest_path.parent
    markdown_path = root / "RESULTS_REPORT.md"
    pdf_path = root / "RESULTS_REPORT.pdf"
    markdown = render_markdown(document)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown, encoding="utf-8", newline="\n")
    render_pdf(document, pdf_path)
    figure_records: list[dict[str, Any]] = []
    for spec in document.manifest.figures:
        figure_path = root / spec.relative_path
        if spec.kind == "artifact_png":
            if _file_sha256(figure_path) != spec.sha256:
                raise ReportingError(f"artifact figure hash mismatch: {spec.figure_id}")
        else:
            _draw_figure(document, spec, figure_path)
        figure_records.append(
            {
                "figure_id": spec.figure_id,
                "kind": spec.kind,
                "relative_path": spec.relative_path,
                "sha256": _file_sha256(figure_path),
                "report_section_id": _figure_section_id(spec),
                "embedded_in_report": True,
                "source_table_sha256": (
                    document.tables[spec.table_id].spec.sha256 if spec.table_id else None
                ),
                "source_artifact_hashes": list(spec.source_artifact_hashes),
            }
        )
    table_records = [
        {
            "table_id": table.spec.table_id,
            "relative_path": table.spec.relative_path,
            "sha256": table.spec.sha256,
            "row_count": table.spec.row_count,
            "display_columns": list(_pdf_columns(table)),
            "source_artifact_hashes": list(table.spec.source_artifact_hashes),
        }
        for table in document.tables.values()
    ]
    review_supplement_records = [
        {
            "supplement_id": supplement.spec.supplement_id,
            "section_id": supplement.spec.section_id,
            "relative_path": supplement.spec.relative_path,
            "sha256": supplement.spec.sha256,
            "row_count": supplement.spec.row_count,
            "display_columns": list(_pdf_columns(supplement)),
            "source_artifact_hashes": list(
                supplement.spec.source_artifact_hashes
            ),
        }
        for supplement in document.review_supplements.values()
    ]
    payload: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "kind": "result_and_figure_manifest",
        "source_result_manifest_sha256": document.manifest.manifest_sha256,
        "source_ingestion_receipt_sha256": (document.manifest.ingestion_receipt_sha256),
        "source_ingestion_receipt_file_sha256": (document.manifest.ingestion_receipt_file_sha256),
        "source_artifact_hashes": list(document.manifest.source_artifact_hashes),
        "study_status": document.manifest.study_status.value,
        "reports": [
            {"relative_path": "RESULTS_REPORT.md", "sha256": _file_sha256(markdown_path)},
            {"relative_path": "RESULTS_REPORT.pdf", "sha256": _file_sha256(pdf_path)},
        ],
        "tables": table_records,
        "review_supplements": review_supplement_records,
        "figures": figure_records,
    }
    write_json(root / "result_figure_manifest.json", canonical_manifest_payload(payload))
    (root / "REPRODUCIBILITY.md").write_text(
        render_reproducibility_report(document),
        encoding="utf-8",
        newline="\n",
    )
    verify_report_build(
        manifest_path,
        policy_path,
        root,
        require_reproducibility_report=False,
    )
    return payload


def verify_report_build(
    manifest_path: Path,
    policy_path: Path,
    output_root: Path | None = None,
    *,
    require_reproducibility_report: bool = True,
) -> None:
    document = build_document(manifest_path, policy_path)
    root = output_root or manifest_path.parent
    markdown_path = root / "RESULTS_REPORT.md"
    pdf_path = root / "RESULTS_REPORT.pdf"
    expected = render_markdown(document)
    if markdown_path.read_text(encoding="utf-8") != expected:
        raise ReportingError("RESULTS_REPORT.md is stale or contains manually transcribed values")
    if not pdf_path.is_file() or not pdf_path.read_bytes().startswith(b"%PDF-"):
        raise ReportingError("RESULTS_REPORT.pdf is missing or invalid")
    build_manifest_path = root / "result_figure_manifest.json"
    payload = json.loads(build_manifest_path.read_text(encoding="utf-8"))
    supplied_hash = payload.pop("manifest_sha256", None)
    if supplied_hash != canonical_sha256(payload):
        raise ReportingError("result/figure manifest self-hash mismatch")
    if payload.get("source_result_manifest_sha256") != document.manifest.manifest_sha256:
        raise ReportingError("result/figure manifest points to another result manifest")
    if (
        payload.get("source_ingestion_receipt_sha256") != document.manifest.ingestion_receipt_sha256
        or payload.get("source_ingestion_receipt_file_sha256")
        != document.manifest.ingestion_receipt_file_sha256
    ):
        raise ReportingError("result/figure manifest ingestion lineage is stale")
    if payload.get("source_artifact_hashes") != list(document.manifest.source_artifact_hashes):
        raise ReportingError("result/figure manifest source-artifact lineage is stale")
    if payload.get("study_status") != document.manifest.study_status.value:
        raise ReportingError("result/figure manifest study status is stale")
    report_items = payload.get("reports", [])
    report_hashes = {item["relative_path"]: item["sha256"] for item in report_items}
    if len(report_hashes) != len(report_items) or set(report_hashes) != {
        "RESULTS_REPORT.md",
        "RESULTS_REPORT.pdf",
    }:
        raise ReportingError("result/figure manifest report inventory is not exact")
    for relative_path in ("RESULTS_REPORT.md", "RESULTS_REPORT.pdf"):
        if report_hashes.get(relative_path) != _file_sha256(root / relative_path):
            raise ReportingError(f"stale report hash: {relative_path}")
    table_items = payload.get("tables", [])
    table_ids = [item["table_id"] for item in table_items]
    if len(table_ids) != len(set(table_ids)) or set(table_ids) != set(document.tables):
        raise ReportingError("result/figure manifest table inventory is not exact")
    for item in table_items:
        table = document.tables.get(item["table_id"])
        if table is None or any(
            (
                item.get("relative_path") != table.spec.relative_path,
                item.get("sha256") != table.spec.sha256,
                item.get("row_count") != table.spec.row_count,
                item.get("display_columns") != list(_pdf_columns(table)),
                item.get("source_artifact_hashes") != list(table.spec.source_artifact_hashes),
            )
        ):
            raise ReportingError(f"stale table provenance: {item['table_id']}")
    supplement_items = payload.get("review_supplements", [])
    supplement_ids = [item["supplement_id"] for item in supplement_items]
    if (
        len(supplement_ids) != len(set(supplement_ids))
        or set(supplement_ids) != set(document.review_supplements)
    ):
        raise ReportingError("result/figure manifest review-supplement inventory is not exact")
    for item in supplement_items:
        supplement = document.review_supplements.get(item["supplement_id"])
        if supplement is None or any(
            (
                item.get("section_id") != supplement.spec.section_id,
                item.get("relative_path") != supplement.spec.relative_path,
                item.get("sha256") != supplement.spec.sha256,
                item.get("row_count") != supplement.spec.row_count,
                item.get("display_columns") != list(_pdf_columns(supplement)),
                item.get("source_artifact_hashes")
                != list(supplement.spec.source_artifact_hashes),
            )
        ):
            raise ReportingError(
                f"stale review-supplement provenance: {item['supplement_id']}"
            )
    figure_items = payload.get("figures", [])
    figure_ids = [item["figure_id"] for item in figure_items]
    expected_figures = {item.figure_id: item for item in document.manifest.figures}
    if len(figure_ids) != len(set(figure_ids)) or set(figure_ids) != set(expected_figures):
        raise ReportingError("result/figure manifest figure inventory is not exact")
    for item in figure_items:
        spec = expected_figures[item["figure_id"]]
        expected_source = document.tables[spec.table_id].spec.sha256 if spec.table_id else None
        if (
            item.get("kind") != spec.kind
            or item.get("relative_path") != spec.relative_path
            or item.get("report_section_id") != _figure_section_id(spec)
            or item.get("embedded_in_report") is not True
            or item.get("source_table_sha256") != expected_source
            or item.get("source_artifact_hashes", []) != list(spec.source_artifact_hashes)
        ):
            raise ReportingError(f"stale figure provenance: {item['figure_id']}")
        path = _safe_under(root, item["relative_path"])
        if item["sha256"] != _file_sha256(path):
            raise ReportingError(f"stale figure hash: {item['figure_id']}")

    with tempfile.TemporaryDirectory(prefix="story-projection-report-verify-") as temporary:
        regeneration_root = Path(temporary)
        regenerated_pdf = regeneration_root / "RESULTS_REPORT.pdf"
        render_pdf(document, regenerated_pdf)
        if regenerated_pdf.read_bytes() != pdf_path.read_bytes():
            raise ReportingError("RESULTS_REPORT.pdf does not reproduce from immutable tables")
        for spec in document.manifest.figures:
            if spec.kind == "artifact_png":
                continue
            regenerated_figure = regeneration_root / spec.relative_path
            _draw_figure(document, spec, regenerated_figure)
            checked_figure = _safe_under(root, spec.relative_path)
            if regenerated_figure.read_bytes() != checked_figure.read_bytes():
                raise ReportingError(
                    f"figure does not reproduce from immutable tables: {spec.figure_id}"
                )

    reproducibility_path = root / "REPRODUCIBILITY.md"
    if reproducibility_path.is_file():
        expected_reproducibility = render_reproducibility_report(document)
        if reproducibility_path.read_text(encoding="utf-8") != expected_reproducibility:
            raise ReportingError("REPRODUCIBILITY.md is stale")
    elif require_reproducibility_report and document.manifest.study_status is ReportStatus.COMPLETE:
        raise ReportingError("a complete report requires REPRODUCIBILITY.md")


def render_reproducibility_report(document: ReportDocument) -> str:
    """Render the editable reproducibility handoff from the same verified manifest."""

    lines = [
        "# Reproducibility report",
        "",
        f"Study status: `{document.manifest.study_status.value}`.",
        "",
        f"Result manifest: `{document.manifest.manifest_sha256}`.",
        "",
        f"Code revision: `{document.manifest.code_revision}`; dirty: "
        f"`{str(document.manifest.dirty_worktree).lower()}`.",
        "",
        "The report builder verifies UTF-8/LF CSV bytes, SHA-256 values, row counts, required",
        "columns, predecessor ingestion, phase gates, PDF presence, figure hashes, and exact",
        "Markdown regeneration.",
        "Scientific statistics are computed upstream; this layer never recomputes or imputes them.",
        "",
        "## Inputs",
        "",
        f"Ingestion receipt: `{document.manifest.ingestion_receipt_sha256}`.",
        "",
    ]
    for table in document.tables.values():
        lines.append(
            f"- `{table.spec.table_id}`: `{table.spec.sha256}`, {table.spec.row_count} rows, "
            f"status `{table.spec.status.value}`."
        )
        for source in table.spec.supplemental_metric_sources:
            lines.append(
                f"  - supplemental `{source.source_role}`: `{source.source_file_sha256}`, "
                f"{source.selected_row_count} selected rows in "
                f"`{source.source_relative_path}`."
            )
    for supplement in document.review_supplements.values():
        lines.append(
            f"- blinded-review supplement `{supplement.spec.supplement_id}`: "
            f"`{supplement.spec.sha256}`, {supplement.spec.row_count} sanitized rows."
        )
    lines.extend(
        (
            "",
            "## Regeneration",
            "",
            "Run `python scripts/build_results_report.py --manifest reports/results_manifest.json`",
            "from the repository root in the pinned study environment. Add `--verify` for a",
            "read-only consistency check.",
            "",
            "The public bundle is a separate allowlist-only build. Restricted novel text, detailed",
            "offsets, FTS indexes, model weights, caches, raw prompts containing protected prose,",
            "and private paths are never public-bundle inputs.",
            "",
        )
    )
    return "\n".join(lines)


def write_reproducibility_report(
    manifest_path: Path,
    policy_path: Path,
    output_root: Path | None = None,
) -> Path:
    document = build_document(manifest_path, policy_path)
    root = output_root or manifest_path.parent
    output = root / "REPRODUCIBILITY.md"
    output.write_text(
        render_reproducibility_report(document),
        encoding="utf-8",
        newline="\n",
    )
    return output


def iter_report_values(document: ReportDocument) -> Iterable[str]:
    """Expose table cells used by both renderers for independent consistency tests."""

    for section in document.sections:
        if section.status is ReportStatus.COMPLETE:
            for table in section.tables:
                yield from table.columns
                for row in table.rows:
                    yield from (row[column] for column in table.columns)
                for source in table.spec.supplemental_metric_sources:
                    yield source.source_role
                    yield source.description
                    yield source.source_relative_path
                    yield source.source_file_sha256
                    yield source.table_manifest_relative_path
                    yield source.table_manifest_file_sha256
                    yield str(source.source_row_count)
                    yield str(source.selected_row_count)
                    for metric in source.metric_row_counts:
                        yield metric.metric_name
                        yield str(metric.row_count)
            for supplement in section.review_supplements:
                yield from supplement.columns
                for row in supplement.rows:
                    yield from (row[column] for column in supplement.columns)
