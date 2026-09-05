"""Fail-closed ingestion boundary between execution artifacts and paper tables.

The report renderer must never discover scientific outputs or accept a table merely
because it exists under ``reports/``.  This module verifies one explicit registry of
predecessor artifacts and canonical CSV tables, then emits a sanitized receipt.  The
receipt contains hashes and status only, so it can be released without exposing a
restricted path or copyrighted payload.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StringConstraints, model_validator

from story_projection_onto.contracts import (
    canonical_sha256 as canonical_record_sha256,
)
from story_projection_onto.reporting import (
    REGISTERED_COMPLETE_TABLE_IDS,
    ReportingError,
    ReportStatus,
    canonical_sha256,
)

INGESTION_SCHEMA_VERSION = "1.0.0"
Sha256Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Identifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")]


class ReportingIngestionError(ReportingError):
    """A predecessor, table, or ingestion receipt failed verification."""


class SourceFamily(StrEnum):
    FALLBACK = "fallback"
    DEVELOPMENT = "development"
    HELD_OUT = "heldout"
    ABLATIONS = "ablations"
    FEEDBACK = "feedback"
    CASE_STUDY = "case_study"
    RUNTIME = "runtime"
    STORAGE = "storage"


class MeasurementDomain(StrEnum):
    SCIENTIFIC = "scientific"
    GPU_SERVICE_TIME = "gpu_service_time"
    RUNPOD_WALL_TIME = "runpod_wall_time"
    PROCESS_MEMORY = "process_memory"
    GPU_MEMORY = "gpu_memory"
    PROJECT_STORAGE = "project_storage"


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SourceArtifactSpec(FrozenModel):
    artifact_id: Identifier
    family: SourceFamily
    relative_path: Annotated[str, StringConstraints(min_length=1, max_length=500)]
    file_sha256: Sha256Digest
    media_type: Literal[
        "application/json",
        "application/jsonl",
        "application/x-sqlite3",
        "application/pdf",
        "image/png",
        "text/csv",
    ]
    release_class: Literal["public", "restricted"]
    logical_hash: Sha256Digest | None = None
    logical_hash_field: Literal[
        "none",
        "content_hash",
        "manifest_sha256",
        "policy_sha256",
    ] = "none"
    logical_hash_mode: Literal[
        "none", "declared", "canonical_without_field", "immutable_record"
    ] = "none"
    required_json_fields: tuple[Identifier, ...] = ()
    measurement_domains: tuple[MeasurementDomain, ...] = ()

    @model_validator(mode="after")
    def logical_hash_contract(self) -> Self:
        no_logical_hash = self.logical_hash_field == "none"
        if no_logical_hash != (self.logical_hash is None):
            raise ValueError("logical hash field/value must either both be absent or both present")
        if no_logical_hash != (self.logical_hash_mode == "none"):
            raise ValueError("logical hash mode must agree with the logical hash field")
        if (
            self.logical_hash_mode == "immutable_record"
            and self.logical_hash_field != "content_hash"
        ):
            raise ValueError("immutable-record hashing requires a content_hash field")
        if self.media_type != "application/json" and (
            self.logical_hash is not None or self.required_json_fields
        ):
            raise ValueError("only JSON predecessor artifacts have JSON logical contracts")
        if len(set(self.required_json_fields)) != len(self.required_json_fields):
            raise ValueError("required JSON fields must be unique")
        if len(set(self.measurement_domains)) != len(self.measurement_domains):
            raise ValueError("measurement domains must be unique")
        if {
            MeasurementDomain.GPU_SERVICE_TIME,
            MeasurementDomain.RUNPOD_WALL_TIME,
        }.issubset(self.measurement_domains):
            raise ValueError("GPU-service time and pod wall time require distinct artifacts")
        return self


class PredecessorState(FrozenModel):
    family: SourceFamily
    status: ReportStatus
    reason: Annotated[str, StringConstraints(min_length=1, max_length=1000)]
    artifacts: tuple[SourceArtifactSpec, ...] = ()

    @model_validator(mode="after")
    def family_and_completion(self) -> Self:
        if any(item.family is not self.family for item in self.artifacts):
            raise ValueError("predecessor contains an artifact from another family")
        if self.status is ReportStatus.COMPLETE and not self.artifacts:
            raise ValueError("a complete predecessor requires at least one immutable artifact")
        if self.status is ReportStatus.NOT_APPLICABLE:
            raise ValueError("all eight registered predecessor families are applicable")
        return self


class IngestedTableSpec(FrozenModel):
    table_id: Identifier
    status: ReportStatus
    reason: Annotated[str, StringConstraints(min_length=1, max_length=1000)]
    scope: Literal["interim", "final"]
    relative_path: Annotated[str, StringConstraints(min_length=1, max_length=500)] | None = None
    file_sha256: Sha256Digest | None = None
    row_count: int | None = Field(default=None, ge=0)
    required_columns: tuple[Annotated[str, StringConstraints(min_length=1)], ...] = ()
    source_artifact_ids: tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def availability_contract(self) -> Self:
        available = self.status is ReportStatus.COMPLETE
        supplied = (
            self.relative_path is not None,
            self.file_sha256 is not None,
            self.row_count is not None,
            bool(self.required_columns),
        )
        if available and not all(supplied):
            raise ValueError("an available table requires path, hash, rows, and columns")
        if not available and any(supplied):
            raise ValueError("an incomplete table cannot point at a result table")
        if not available and self.source_artifact_ids:
            raise ValueError("an incomplete table cannot claim source artifacts")
        if len(set(self.required_columns)) != len(self.required_columns):
            raise ValueError("table required columns must be unique")
        if len(set(self.source_artifact_ids)) != len(self.source_artifact_ids):
            raise ValueError("table source artifact IDs must be unique")
        if available and self.table_id != "study_status" and not self.source_artifact_ids:
            raise ValueError("an available result table requires immutable source artifacts")
        return self


class ReportingIngestionManifest(FrozenModel):
    schema_version: Literal["1.0.0"] = INGESTION_SCHEMA_VERSION
    ingestion_id: Identifier
    compiled_at_utc: AwareDatetime
    predecessors: tuple[PredecessorState, ...]
    tables: tuple[IngestedTableSpec, ...]
    manifest_sha256: Sha256Digest

    @model_validator(mode="after")
    def exact_registered_inventory(self) -> Self:
        families = [item.family for item in self.predecessors]
        if len(families) != len(set(families)) or set(families) != set(SourceFamily):
            raise ValueError("ingestion manifest requires each predecessor family exactly once")
        table_ids = [item.table_id for item in self.tables]
        if len(table_ids) != len(set(table_ids)) or set(table_ids) != set(
            REGISTERED_COMPLETE_TABLE_IDS
        ):
            raise ValueError("ingestion manifest requires every registered report table")
        artifacts = [item for state in self.predecessors for item in state.artifacts]
        artifact_ids = [item.artifact_id for item in artifacts]
        paths = [item.relative_path for item in artifacts]
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("predecessor artifact IDs must be globally unique")
        if len(paths) != len(set(paths)):
            raise ValueError("one predecessor file cannot be registered twice")
        known = set(artifact_ids)
        if any(not set(table.source_artifact_ids).issubset(known) for table in self.tables):
            raise ValueError("a table names an unknown predecessor artifact")
        return self


class VerifiedArtifact(FrozenModel):
    artifact_id: Identifier
    family: SourceFamily
    file_sha256: Sha256Digest
    logical_hash: Sha256Digest | None
    media_type: str
    measurement_domains: tuple[MeasurementDomain, ...]


class VerifiedPredecessor(FrozenModel):
    family: SourceFamily
    status: ReportStatus
    reason: str
    artifact_ids: tuple[Identifier, ...]


class VerifiedTable(FrozenModel):
    table_id: Identifier
    status: ReportStatus
    reason: str
    scope: Literal["interim", "final"]
    relative_path: str | None
    file_sha256: Sha256Digest | None
    row_count: int | None
    columns: tuple[str, ...]
    source_artifact_hashes: tuple[Sha256Digest, ...]


class ReportingIngestionReceipt(FrozenModel):
    schema_version: Literal["1.0.0"] = INGESTION_SCHEMA_VERSION
    source_manifest_sha256: Sha256Digest
    source_manifest_file_sha256: Sha256Digest
    predecessors: tuple[VerifiedPredecessor, ...]
    artifacts: tuple[VerifiedArtifact, ...]
    tables: tuple[VerifiedTable, ...]
    compiled_at_utc: AwareDatetime
    gpu_service_time_artifact_ids: tuple[Identifier, ...]
    runpod_wall_time_artifact_ids: tuple[Identifier, ...]
    receipt_sha256: Sha256Digest

    @model_validator(mode="after")
    def receipt_inventory(self) -> Self:
        if {item.family for item in self.predecessors} != set(SourceFamily):
            raise ValueError("ingestion receipt lacks a predecessor family")
        if {item.table_id for item in self.tables} != set(REGISTERED_COMPLETE_TABLE_IDS):
            raise ValueError("ingestion receipt lacks a registered table")
        artifact_ids = {item.artifact_id for item in self.artifacts}
        gpu_ids = set(self.gpu_service_time_artifact_ids)
        pod_ids = set(self.runpod_wall_time_artifact_ids)
        if not gpu_ids.issubset(artifact_ids) or not pod_ids.issubset(artifact_ids):
            raise ValueError("runtime measurement IDs do not resolve to verified artifacts")
        if gpu_ids & pod_ids:
            raise ValueError("GPU-service time and pod wall time share an artifact")
        return self


FINAL_TABLE_DEPENDENCIES: Mapping[str, frozenset[SourceFamily]] = {
    "study_status": frozenset(SourceFamily),
    "primary_c2_vs_c1": frozenset({SourceFamily.HELD_OUT}),
    "secondary_c2_vs_c0": frozenset({SourceFamily.HELD_OUT}),
    "mechanism_c2_vs_fixed": frozenset({SourceFamily.HELD_OUT}),
    "rare_pivotal": frozenset({SourceFamily.HELD_OUT}),
    "entropy_clutter": frozenset({SourceFamily.HELD_OUT}),
    "community": frozenset({SourceFamily.HELD_OUT}),
    "paraphrase_contrastive": frozenset({SourceFamily.HELD_OUT}),
    "ablations": frozenset({SourceFamily.ABLATIONS}),
    "feedback": frozenset({SourceFamily.FEEDBACK}),
    "novel_case": frozenset({SourceFamily.CASE_STUDY}),
    "failure_accounting": frozenset(
        {
            SourceFamily.FALLBACK,
            SourceFamily.DEVELOPMENT,
            SourceFamily.HELD_OUT,
            SourceFamily.ABLATIONS,
            SourceFamily.FEEDBACK,
            SourceFamily.CASE_STUDY,
            SourceFamily.RUNTIME,
        }
    ),
    "resource_accounting": frozenset(
        {
            SourceFamily.FALLBACK,
            SourceFamily.DEVELOPMENT,
            SourceFamily.HELD_OUT,
            SourceFamily.ABLATIONS,
            SourceFamily.FEEDBACK,
            SourceFamily.CASE_STUDY,
            SourceFamily.RUNTIME,
            SourceFamily.STORAGE,
        }
    ),
    "qualitative_examples": frozenset(
        {SourceFamily.DEVELOPMENT, SourceFamily.HELD_OUT, SourceFamily.CASE_STUDY}
    ),
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_no_symlink_chain(path: Path) -> None:
    current = path.absolute()
    while True:
        if current.is_symlink():
            raise ReportingIngestionError(f"symlinked reporting input is forbidden: {path}")
        if current.parent == current:
            return
        current = current.parent


def _safe_file(root: Path, relative_path: str) -> Path:
    if "\\" in relative_path:
        raise ReportingIngestionError(f"backslash is forbidden in source path: {relative_path}")
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ReportingIngestionError(f"unsafe reporting source path: {relative_path}")
    candidate = root / relative
    _assert_no_symlink_chain(candidate)
    try:
        resolved = candidate.resolve(strict=True)
        root_resolved = root.resolve(strict=True)
    except OSError as error:
        raise ReportingIngestionError(
            f"reporting source artifact is absent: {relative_path}"
        ) from error
    if not resolved.is_relative_to(root_resolved) or not resolved.is_file():
        raise ReportingIngestionError(
            f"reporting source is not a regular in-root file: {relative_path}"
        )
    return resolved


def _load_self_hashed_json(path: Path, hash_field: str, label: str) -> dict[str, Any]:
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ReportingIngestionError(f"{label} must be UTF-8 without BOM")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReportingIngestionError(f"invalid {label}: {error}") from error
    if not isinstance(payload, dict):
        raise ReportingIngestionError(f"{label} must be one JSON object")
    supplied = payload.get(hash_field)
    immutable = {key: value for key, value in payload.items() if key != hash_field}
    if supplied != canonical_sha256(immutable):
        raise ReportingIngestionError(f"{label} canonical self-hash mismatch")
    return payload


def load_ingestion_manifest(path: Path) -> ReportingIngestionManifest:
    _assert_no_symlink_chain(path)
    if not path.is_file():
        raise ReportingIngestionError("reporting ingestion manifest is absent")
    payload = _load_self_hashed_json(path, "manifest_sha256", "ingestion manifest")
    try:
        return ReportingIngestionManifest.model_validate(payload)
    except ValueError as error:
        raise ReportingIngestionError(f"invalid ingestion manifest: {error}") from error


def load_ingestion_receipt(path: Path) -> ReportingIngestionReceipt:
    _assert_no_symlink_chain(path)
    if not path.is_file():
        raise ReportingIngestionError("reporting ingestion receipt is absent")
    payload = _load_self_hashed_json(path, "receipt_sha256", "ingestion receipt")
    try:
        return ReportingIngestionReceipt.model_validate(payload)
    except ValueError as error:
        raise ReportingIngestionError(f"invalid ingestion receipt: {error}") from error


def _verify_artifact(root: Path, spec: SourceArtifactSpec) -> VerifiedArtifact:
    path = _safe_file(root, spec.relative_path)
    if _file_sha256(path) != spec.file_sha256:
        raise ReportingIngestionError(f"predecessor file hash changed: {spec.artifact_id}")
    if spec.media_type == "application/json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ReportingIngestionError(
                f"invalid predecessor JSON: {spec.artifact_id}"
            ) from error
        if not isinstance(payload, dict):
            raise ReportingIngestionError(f"predecessor JSON is not an object: {spec.artifact_id}")
        missing = set(spec.required_json_fields) - set(payload)
        if missing:
            raise ReportingIngestionError(
                f"predecessor {spec.artifact_id} lacks fields: {', '.join(sorted(missing))}"
            )
        if spec.logical_hash_field != "none":
            supplied = payload.get(spec.logical_hash_field)
            if supplied != spec.logical_hash:
                raise ReportingIngestionError(
                    f"predecessor logical hash changed: {spec.artifact_id}"
                )
            if spec.logical_hash_mode == "canonical_without_field":
                immutable = {
                    key: value for key, value in payload.items() if key != spec.logical_hash_field
                }
                if canonical_sha256(immutable) != supplied:
                    raise ReportingIngestionError(
                        f"predecessor canonical hash is invalid: {spec.artifact_id}"
                    )
            elif (
                spec.logical_hash_mode == "immutable_record"
                and canonical_record_sha256(payload) != supplied
            ):
                raise ReportingIngestionError(
                    f"predecessor immutable-record hash is invalid: {spec.artifact_id}"
                )
    return VerifiedArtifact(
        artifact_id=spec.artifact_id,
        family=spec.family,
        file_sha256=spec.file_sha256,
        logical_hash=spec.logical_hash,
        media_type=spec.media_type,
        measurement_domains=spec.measurement_domains,
    )


def _read_canonical_csv(
    root: Path,
    table: IngestedTableSpec,
) -> tuple[tuple[str, ...], tuple[dict[str, str], ...]]:
    assert table.relative_path is not None
    assert table.file_sha256 is not None
    assert table.row_count is not None
    path = _safe_file(root, table.relative_path)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != table.file_sha256:
        raise ReportingIngestionError(f"ingested table hash changed: {table.table_id}")
    if raw.startswith(b"\xef\xbb\xbf") or b"\r" in raw:
        raise ReportingIngestionError(f"table is not canonical UTF-8/LF CSV: {table.table_id}")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ReportingIngestionError(f"table is not UTF-8: {table.table_id}") from error
    reader = csv.DictReader(text.splitlines())
    columns = tuple(reader.fieldnames or ())
    if not columns or len(columns) != len(set(columns)):
        raise ReportingIngestionError(f"table has missing/duplicate columns: {table.table_id}")
    missing = set(table.required_columns) - set(columns)
    if missing:
        raise ReportingIngestionError(
            f"table {table.table_id} lacks columns: {', '.join(sorted(missing))}"
        )
    rows = tuple(dict(row) for row in reader)
    if len(rows) != table.row_count:
        raise ReportingIngestionError(f"table row count changed: {table.table_id}")
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ReportingIngestionError(f"table has a ragged CSV row: {table.table_id}")
    regenerated = io.StringIO(newline="")
    writer = csv.DictWriter(regenerated, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    if regenerated.getvalue().encode("utf-8") != raw:
        raise ReportingIngestionError(f"table CSV serialization is not canonical: {table.table_id}")
    return columns, rows


def _verify_runtime_measurement_separation(
    table: IngestedTableSpec,
    rows: tuple[dict[str, str], ...],
    artifacts: Mapping[str, VerifiedArtifact],
) -> None:
    if table.table_id != "resource_accounting":
        return
    source_domains = {
        domain
        for artifact_id in table.source_artifact_ids
        for domain in artifacts[artifact_id].measurement_domains
    }
    for row in rows:
        metric = row.get("metric", "").casefold()
        status = row.get("status", "").casefold()
        value = row.get("value", "").strip()
        if (
            "gpu" in metric
            and "time" in metric
            and status == "observed"
            and MeasurementDomain.GPU_SERVICE_TIME not in source_domains
        ):
            raise ReportingIngestionError(
                "observed GPU-service time lacks a dedicated metering artifact"
            )
        if "runpod" in metric and "wall_time" in metric:
            if status == "observed" and MeasurementDomain.RUNPOD_WALL_TIME not in source_domains:
                raise ReportingIngestionError(
                    "observed RunPod wall time lacks a distinct wall-time artifact"
                )
            if status != "observed" and value:
                raise ReportingIngestionError(
                    "incomplete RunPod wall time cannot contain an imputed value"
                )


def compile_ingestion_receipt(
    manifest: ReportingIngestionManifest,
    *,
    source_root: Path,
    table_root: Path,
    manifest_file_sha256: str,
) -> ReportingIngestionReceipt:
    """Verify all registered sources/tables and return a sanitized immutable receipt."""

    artifacts = tuple(
        _verify_artifact(source_root, artifact)
        for state in manifest.predecessors
        for artifact in state.artifacts
    )
    artifacts_by_id = {item.artifact_id: item for item in artifacts}
    statuses = {item.family: item.status for item in manifest.predecessors}
    verified_tables: list[VerifiedTable] = []
    for table in manifest.tables:
        if table.status is not ReportStatus.COMPLETE:
            verified_tables.append(
                VerifiedTable(
                    table_id=table.table_id,
                    status=table.status,
                    reason=table.reason,
                    scope=table.scope,
                    relative_path=None,
                    file_sha256=None,
                    row_count=None,
                    columns=(),
                    source_artifact_hashes=(),
                )
            )
            continue
        if table.scope == "final":
            incomplete = {
                family
                for family in FINAL_TABLE_DEPENDENCIES[table.table_id]
                if statuses[family] is not ReportStatus.COMPLETE
            }
            if incomplete:
                raise ReportingIngestionError(
                    f"final table {table.table_id} has incomplete predecessors: "
                    + ", ".join(sorted(item.value for item in incomplete))
                )
            source_families = {
                artifacts_by_id[artifact_id].family for artifact_id in table.source_artifact_ids
            }
            missing_lineage = FINAL_TABLE_DEPENDENCIES[table.table_id] - source_families
            if missing_lineage:
                raise ReportingIngestionError(
                    f"final table {table.table_id} lacks source lineage for: "
                    + ", ".join(sorted(item.value for item in missing_lineage))
                )
        columns, rows = _read_canonical_csv(table_root, table)
        _verify_runtime_measurement_separation(table, rows, artifacts_by_id)
        source_hashes = (
            (manifest.manifest_sha256,)
            if table.table_id == "study_status"
            else tuple(
                artifacts_by_id[artifact_id].file_sha256
                for artifact_id in table.source_artifact_ids
            )
        )
        verified_tables.append(
            VerifiedTable(
                table_id=table.table_id,
                status=table.status,
                reason=table.reason,
                scope=table.scope,
                relative_path=table.relative_path,
                file_sha256=table.file_sha256,
                row_count=table.row_count,
                columns=columns,
                source_artifact_hashes=source_hashes,
            )
        )
    gpu_ids = tuple(
        item.artifact_id
        for item in artifacts
        if MeasurementDomain.GPU_SERVICE_TIME in item.measurement_domains
    )
    wall_ids = tuple(
        item.artifact_id
        for item in artifacts
        if MeasurementDomain.RUNPOD_WALL_TIME in item.measurement_domains
    )
    payload: dict[str, Any] = {
        "schema_version": INGESTION_SCHEMA_VERSION,
        "source_manifest_sha256": manifest.manifest_sha256,
        "source_manifest_file_sha256": manifest_file_sha256,
        "predecessors": [
            {
                "family": item.family.value,
                "status": item.status.value,
                "reason": item.reason,
                "artifact_ids": [artifact.artifact_id for artifact in item.artifacts],
            }
            for item in manifest.predecessors
        ],
        "artifacts": [item.model_dump(mode="json") for item in artifacts],
        "tables": [item.model_dump(mode="json") for item in verified_tables],
        "compiled_at_utc": manifest.compiled_at_utc.isoformat().replace("+00:00", "Z"),
        "gpu_service_time_artifact_ids": list(gpu_ids),
        "runpod_wall_time_artifact_ids": list(wall_ids),
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return ReportingIngestionReceipt.model_validate(payload)


def verify_ingestion_from_files(
    manifest_path: Path,
    receipt_path: Path,
    *,
    source_root: Path,
    table_root: Path,
) -> ReportingIngestionReceipt:
    manifest = load_ingestion_manifest(manifest_path)
    expected = compile_ingestion_receipt(
        manifest,
        source_root=source_root,
        table_root=table_root,
        manifest_file_sha256=_file_sha256(manifest_path),
    )
    observed = load_ingestion_receipt(receipt_path)
    if observed != expected:
        raise ReportingIngestionError(
            "reporting ingestion receipt does not reproduce from immutable predecessors"
        )
    return observed


def write_ingestion_receipt(
    manifest_path: Path,
    output_path: Path,
    *,
    source_root: Path,
    table_root: Path,
) -> ReportingIngestionReceipt:
    manifest = load_ingestion_manifest(manifest_path)
    receipt = compile_ingestion_receipt(
        manifest,
        source_root=source_root,
        table_root=table_root,
        manifest_file_sha256=_file_sha256(manifest_path),
    )
    payload = (receipt.model_dump_json(indent=2) + "\n").encode("utf-8")
    _assert_no_symlink_chain(output_path)
    if output_path.exists():
        if not output_path.is_file() or output_path.read_bytes() != payload:
            raise ReportingIngestionError("append-only reporting ingestion receipt changed")
        return receipt
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _assert_no_symlink_chain(output_path.parent)
    with output_path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    directory = os.open(output_path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return receipt


def ingestion_file_sha256(path: Path) -> str:
    return _file_sha256(path)


__all__ = [
    "FINAL_TABLE_DEPENDENCIES",
    "INGESTION_SCHEMA_VERSION",
    "IngestedTableSpec",
    "MeasurementDomain",
    "PredecessorState",
    "ReportingIngestionError",
    "ReportingIngestionManifest",
    "ReportingIngestionReceipt",
    "SourceArtifactSpec",
    "SourceFamily",
    "VerifiedArtifact",
    "VerifiedPredecessor",
    "VerifiedTable",
    "compile_ingestion_receipt",
    "ingestion_file_sha256",
    "load_ingestion_manifest",
    "load_ingestion_receipt",
    "verify_ingestion_from_files",
    "write_ingestion_receipt",
]
