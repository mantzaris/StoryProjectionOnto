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
import os
import tempfile
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StringConstraints, model_validator

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
    FigureSpec,
    PhaseStatus,
    ReportStatus,
    ResultManifest,
    SectionSpec,
    TableSpec,
    build_results_report,
    canonical_manifest_payload,
    canonical_sha256,
    load_reporting_policy,
    section_contract_sha256,
    verify_report_build,
)

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
_EXAMPLE_ORDER = {value: index for index, value in enumerate(_EXAMPLE_IDS)}
_CONDITION_ORDER = {
    "c0_classical_pre": 0,
    "c1_llm_pre": 1,
    "c2_llm_query": 2,
    "a_fixed_select": 3,
}


class Phase7CompilationError(RuntimeError):
    """A production-report input or generated artifact violated its contract."""


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TableProducer(StrEnum):
    CANONICAL_CSV = "canonical_csv"
    STUDY_STATUS = "study_status"
    QUALITATIVE_SELECTION = "qualitative_selection"


class TableContract(FrozenModel):
    table_id: Identifier
    producer: TableProducer
    required_columns: tuple[Annotated[str, StringConstraints(min_length=1)], ...]
    sort_columns: tuple[Annotated[str, StringConstraints(min_length=1)], ...]
    description: Annotated[str, StringConstraints(min_length=1, max_length=500)]

    @model_validator(mode="after")
    def valid_contract(self) -> Self:
        if len(self.required_columns) != len(set(self.required_columns)):
            raise ValueError("table required columns must be unique")
        if not set(self.sort_columns).issubset(self.required_columns):
            raise ValueError("table sort columns must be required columns")
        expected = {
            "study_status": TableProducer.STUDY_STATUS,
            "qualitative_examples": TableProducer.QUALITATIVE_SELECTION,
        }.get(self.table_id, TableProducer.CANONICAL_CSV)
        if self.producer is not expected:
            raise ValueError(f"table {self.table_id} has the wrong production route")
        return self


class PhaseLabel(FrozenModel):
    phase_id: Annotated[str, StringConstraints(pattern=r"^phase_[1-7]$")]
    label: Annotated[str, StringConstraints(min_length=1, max_length=160)]


class PrimaryFigureColumns(FrozenModel):
    label: str
    estimate: str
    lower: str
    upper: str


class Phase7CompilerConfiguration(FrozenModel):
    schema_version: Literal["1.0.0"] = PHASE7_COMPILER_SCHEMA_VERSION
    configuration_id: Literal["conference-phase7-production-v1"]
    reporting_policy_path: RelativePath
    reporting_policy_file_sha256: Sha256Digest
    reporting_policy_sha256: Sha256Digest
    phase_labels: tuple[PhaseLabel, ...]
    table_contracts: tuple[TableContract, ...]
    sections: tuple[SectionSpec, ...]
    primary_figure_columns: PrimaryFigureColumns
    public_static_paths: tuple[RelativePath, ...]
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
        if len(self.public_static_paths) != len(set(self.public_static_paths)):
            raise ValueError("public static paths must be unique")
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
        if section_contract_sha256(value.sections) != policy.section_contract_sha256:
            raise Phase7CompilationError("configured section prose differs from reporting policy")
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


class Phase7TableBinding(FrozenModel):
    table_id: Identifier
    status: ReportStatus
    reason: Annotated[str, StringConstraints(min_length=1, max_length=1000)]
    scope: Literal["interim", "final"]
    source_artifact_ids: tuple[Identifier, ...] = ()
    source_table_artifact_id: Identifier | None = None
    source_row_count: int | None = Field(default=None, ge=0)
    output_row_count: int | None = Field(default=None, ge=0)
    row_filters: tuple[ExactRowFilter, ...] = ()
    singleton_join_artifact_id: Identifier | None = None
    singleton_join_row_count: Literal[1] | None = None

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
                self.source_row_count,
                self.output_row_count,
                self.row_filters,
                self.singleton_join_artifact_id,
                self.singleton_join_row_count,
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
        columns = [item.column for item in self.row_filters]
        if len(columns) != len(set(columns)):
            raise ValueError("table row filters cannot repeat a column")
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
        if not references.issubset(known):
            raise ValueError("source registry names an unknown artifact")
        by_id = {item.artifact_id: item for item in artifacts}
        if any(by_id[item].release_class != "public" for item in self.public_artifact_ids):
            raise ValueError("public artifact allowlist contains a restricted artifact")
        if len(self.public_artifact_ids) != len(set(self.public_artifact_ids)):
            raise ValueError("public artifact IDs must be unique")
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
            if self.stratum != "hard" or {item.seed_block for item in self.c2_itt_scores} != {1, 2}:
                raise ValueError("counterexample candidates require both hard-stratum C2 ITT seeds")
        elif self.c2_itt_scores:
            raise ValueError("outcome-dependent scores are permitted only for counterexamples")
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


def _verify_registry_contracts(
    registry: Phase7SourceRegistry,
    configuration: Phase7CompilerConfiguration,
) -> None:
    artifacts = _artifact_inventory(registry)
    contracts = {item.table_id: item for item in configuration.table_contracts}
    predecessor_status = {item.family: item.status for item in registry.predecessors}
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
        if binding.singleton_join_artifact_id is not None:
            singleton = artifacts[binding.singleton_join_artifact_id]
            if singleton.media_type != "text/csv":
                raise Phase7CompilationError(
                    f"singleton join artifact is not CSV: {binding.table_id}"
                )
    known = set(artifacts)
    for phase in registry.phases:
        if not set(phase.source_artifact_ids).issubset(known):
            raise Phase7CompilationError(f"phase {phase.phase_id} has unknown source lineage")


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
    for relative in configuration.public_static_paths:
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
        for item in configuration.public_static_paths
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
    _verify_registry_contracts(registry, configuration)
    contracts = {item.table_id: item for item in configuration.table_contracts}
    bindings = {item.table_id: item for item in registry.tables}
    token = registry.manifest_sha256[:16]
    status = _study_status(registry)
    generated_paths: list[str] = []
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
            source_ids: tuple[str, ...] = ()
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
    result_tables = tuple(
        TableSpec(
            table_id=table_id,
            relative_path=receipt_tables[table_id].relative_path,
            sha256=receipt_tables[table_id].file_sha256,
            row_count=receipt_tables[table_id].row_count,
            required_columns=contracts[table_id].required_columns,
            status=ReportStatus.COMPLETE,
            description=contracts[table_id].description,
            source_artifact_hashes=receipt_tables[table_id].source_artifact_hashes,
        )
        for table_id in sorted(table_payloads)
    )
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
    "CompiledPhase7Artifacts",
    "ExactRowFilter",
    "Phase7CompilationError",
    "Phase7CompilerConfiguration",
    "Phase7SourceRegistry",
    "Phase7TableBinding",
    "QualitativeCandidate",
    "QualitativeCandidateSet",
    "QualitativeDisplayRow",
    "QualitativeExample",
    "TableContract",
    "TableProducer",
    "compile_phase7_results",
    "load_phase7_source_registry",
    "load_qualitative_candidate_set",
    "verify_phase7_results",
]
