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


class TableSpec(FrozenModel):
    table_id: Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]{1,63}$")]
    relative_path: Annotated[
        str,
        StringConstraints(pattern=r"^tables/[A-Za-z0-9_.-]+\.csv$"),
    ]
    sha256: Sha256Digest
    row_count: int = Field(ge=0)
    required_columns: tuple[Annotated[str, StringConstraints(min_length=1)], ...]
    status: ReportStatus
    description: Annotated[str, StringConstraints(min_length=1, max_length=500)]
    source_artifact_hashes: tuple[Sha256Digest, ...] = ()

    @model_validator(mode="after")
    def columns_are_unique(self) -> TableSpec:
        if not self.required_columns:
            raise ValueError("required_columns must not be empty")
        if len(set(self.required_columns)) != len(self.required_columns):
            raise ValueError("required_columns must be unique")
        return self


class FigureSpec(FrozenModel):
    figure_id: Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]{1,63}$")]
    kind: Literal["phase_status", "forest"]
    title: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    relative_path: Annotated[
        str,
        StringConstraints(pattern=r"^figures/[A-Za-z0-9_.-]+\.pdf$"),
    ]
    table_id: str | None = None
    label_column: str | None = None
    estimate_column: str | None = None
    lower_column: str | None = None
    upper_column: str | None = None

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
            self.selection_time
            == "rule_frozen_before_outputs_applied_after_itt_scoring"
        )
        if outcome_applied != (self.example_id == "counterexample"):
            raise ValueError(
                "only the declared ITT counterexample rule is applied after scoring"
            )
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
    figures: tuple[FigureSpec, ...]
    sections: tuple[SectionSpec, ...]
    source_artifact_hashes: tuple[Sha256Digest, ...]
    reporting_policy_sha256: Sha256Digest
    manifest_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_manifest(self) -> ResultManifest:
        for values, label in (
            (self.phases, "phase_id"),
            (self.tables, "table_id"),
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
class ReportSection:
    spec: SectionSpec
    status: ReportStatus
    reason: str
    tables: tuple[LoadedTable, ...]


@dataclass(frozen=True, slots=True)
class ReportDocument:
    manifest: ResultManifest
    policy: ReportingPolicy
    sections: tuple[ReportSection, ...]
    tables: Mapping[str, LoadedTable]


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


def _safe_under(root: Path, relative_path: str) -> Path:
    if "\\" in relative_path:
        raise ReportingError(f"backslash is prohibited in report path: {relative_path}")
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ReportingError(f"unsafe report path: {relative_path}")
    candidate = root / relative
    if candidate.is_symlink():
        raise ReportingError(f"symlink is prohibited in immutable inputs: {relative_path}")
    resolved_root = root.resolve()
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(resolved_root):
        raise ReportingError(f"report input escapes root: {relative_path}")
    return resolved


def load_result_manifest(path: Path) -> ResultManifest:
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
            "complete table lacks upstream artifact hashes: "
            + ", ".join(tables_without_sources)
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


def build_document(manifest_path: Path, policy_path: Path) -> ReportDocument:
    manifest = load_result_manifest(manifest_path)
    policy = load_reporting_policy(policy_path)
    _validate_complete_gate(manifest, policy)
    root = manifest_path.parent
    tables = {item.table_id: load_canonical_table(root, item) for item in manifest.tables}
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
            )
        )
    return ReportDocument(
        manifest=manifest,
        policy=policy,
        sections=tuple(sections),
        tables=tables,
    )


def _markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _markdown_table(table: LoadedTable) -> str:
    lines = [
        (
            f"<!-- table:{table.spec.table_id} sha256:{table.spec.sha256} "
            f"rows:{table.spec.row_count} -->"
        ),
        "| " + " | ".join(_markdown_cell(item) for item in table.columns) + " |",
        "| " + " | ".join("---" for _ in table.columns) + " |",
    ]
    for row in table.rows:
        lines.append("| " + " | ".join(_markdown_cell(row[item]) for item in table.columns) + " |")
    return "\n".join(lines)


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
    lines.extend(
        (
            "## Machine-readable provenance",
            "",
            f"Document source manifest SHA-256: `{manifest.manifest_sha256}`.",
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
        LongTable,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        TableStyle,
    )


def render_pdf(document: ReportDocument, output_path: Path) -> None:
    (
        colors,
        center,
        letter,
        paragraph_style,
        get_styles,
        inch,
        long_table,
        page_break,
        paragraph,
        document_template,
        spacer,
        table_style,
    ) = _reportlab()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    styles = get_styles()
    styles.add(paragraph_style(name="ReportTitle", parent=styles["Title"], alignment=center))
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
                matrix = [
                    [paragraph(html.escape(col), styles["BodyText"]) for col in table.columns]
                ]
                matrix.extend(
                    [paragraph(html.escape(row[col]), styles["BodyText"]) for col in table.columns]
                    for row in table.rows
                )
                widths = [7.3 * inch / max(1, len(table.columns))] * len(table.columns)
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
        _draw_figure(document, spec, figure_path)
        figure_records.append(
            {
                "figure_id": spec.figure_id,
                "relative_path": spec.relative_path,
                "sha256": _file_sha256(figure_path),
                "source_table_sha256": (
                    document.tables[spec.table_id].spec.sha256 if spec.table_id else None
                ),
            }
        )
    table_records = [
        {
            "table_id": table.spec.table_id,
            "relative_path": table.spec.relative_path,
            "sha256": table.spec.sha256,
            "row_count": table.spec.row_count,
            "source_artifact_hashes": list(table.spec.source_artifact_hashes),
        }
        for table in document.tables.values()
    ]
    payload: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "kind": "result_and_figure_manifest",
        "source_result_manifest_sha256": document.manifest.manifest_sha256,
        "source_artifact_hashes": list(document.manifest.source_artifact_hashes),
        "study_status": document.manifest.study_status.value,
        "reports": [
            {"relative_path": "RESULTS_REPORT.md", "sha256": _file_sha256(markdown_path)},
            {"relative_path": "RESULTS_REPORT.pdf", "sha256": _file_sha256(pdf_path)},
        ],
        "tables": table_records,
        "figures": figure_records,
    }
    write_json(root / "result_figure_manifest.json", canonical_manifest_payload(payload))
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
                item.get("source_artifact_hashes")
                != list(table.spec.source_artifact_hashes),
            )
        ):
            raise ReportingError(f"stale table provenance: {item['table_id']}")
    figure_items = payload.get("figures", [])
    figure_ids = [item["figure_id"] for item in figure_items]
    expected_figures = {item.figure_id: item for item in document.manifest.figures}
    if len(figure_ids) != len(set(figure_ids)) or set(figure_ids) != set(expected_figures):
        raise ReportingError("result/figure manifest figure inventory is not exact")
    for item in figure_items:
        spec = expected_figures[item["figure_id"]]
        expected_source = (
            document.tables[spec.table_id].spec.sha256 if spec.table_id else None
        )
        if (
            item.get("relative_path") != spec.relative_path
            or item.get("source_table_sha256") != expected_source
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
        "columns, phase gates, PDF presence, figure hashes, and exact Markdown regeneration.",
        "Scientific statistics are computed upstream; this layer never recomputes or imputes them.",
        "",
        "## Inputs",
        "",
    ]
    for table in document.tables.values():
        lines.append(
            f"- `{table.spec.table_id}`: `{table.spec.sha256}`, {table.spec.row_count} rows, "
            f"status `{table.spec.status.value}`."
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
