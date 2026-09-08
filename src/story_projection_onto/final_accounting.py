"""Deterministic, fail-closed final failure and resource accounting.

The Phase 7 report consumes two deceptively small CSVs.  This module makes
those CSVs a projection of frozen evidence rather than a hand-maintained
summary.  A self-hashed recipe binds the registered call inventory, every
cumulative ledger snapshot and CAS, authorized inventory amendments, and
typed terminal receipts.  The compiler then proves that:

* cumulative snapshots are append-only;
* every registered inference slot is present exactly once;
* every GPU event, service session, VLLM call, and GPU-linked attempt belongs
  to exactly one accounting row;
* repair ancestry and intention-to-treat membership agree with independent
  receipts; and
* the sum of row allocations is exactly ``Ledger.gpu_summary`` semantics,
  including the unclassified service-overhead complement.

The recipe may describe unfinished mandatory calls, which remain explicit
``incomplete`` rows.  It cannot describe an unregistered call or omit an
observed allocation.  All outputs are canonical, content-addressed, and safe
to register as public Phase 7 table sources; input paths never enter them.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import secrets
import sqlite3
import stat
from collections import defaultdict
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal, Self
from urllib.parse import quote

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

from story_projection_onto.contracts import canonical_json, canonical_sha256
from story_projection_onto.experiment import (
    REGISTERED_ACCOUNTING_EVENTS,
    REGISTERED_CALL_CLASSES,
    REGISTERED_HARD_SECONDS,
    REGISTERED_INFERENCE_ATTEMPTS,
    REGISTERED_SCHEDULED_SECONDS,
    REGISTERED_SESSION_STARTS,
    SESSION_START_CLASS,
    GPUCallInventory,
)
from story_projection_onto.ledger_verify import (
    LedgerVerificationReport,
    derive_call_status,
    verify_ledger,
)
from story_projection_onto.store import Ledger

FINAL_ACCOUNTING_SCHEMA_VERSION = "1.0.0"
FAILURE_ACCOUNTING_COLUMNS = (
    "phase",
    "run_id",
    "call_id",
    "condition",
    "outcome",
    "outcome_scope",
    "scientific_status",
    "attempt_class",
    "repair_parent_call_id",
    "allocated_gpu_seconds",
    "included_in_itt",
    "failure_type",
)
RESOURCE_ACCOUNTING_COLUMNS = (
    "scope",
    "metric",
    "value",
    "unit",
    "status",
    "source_note",
)

Sha256Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Identifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$")]
Token = Annotated[str, StringConstraints(min_length=1, max_length=500)]
RelativePath = Annotated[str, StringConstraints(min_length=1, max_length=700)]


class FinalAccountingError(RuntimeError):
    """Frozen accounting evidence failed validation or reconciliation."""


class FrozenModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
    )


class ReceiptKind(StrEnum):
    EXECUTION = "final_execution_inventory_receipt"
    RESULT = "final_result_inventory_receipt"
    REPAIR = "final_repair_inventory_receipt"
    SERVICE = "final_service_inventory_receipt"
    TOKEN = "final_token_accounting_receipt"
    STORAGE = "final_storage_accounting_receipt"
    WALL_TIME = "runpod_container_wall_time_sample"
    ATTEMPT_EXCLUSION = "final_attempt_exclusion_receipt"


class SelfHashField(StrEnum):
    MANIFEST = "manifest_sha256"
    CONTENT = "content_hash"


class NativeSourceRole(StrEnum):
    """Closed producer contracts permitted to supply native call identity."""

    PHASE1_ACCEPTANCE_RESULT = "phase1_acceptance_result"
    DEVELOPMENT_CALL_MANIFEST = "development_call_manifest"
    HELD_OUT_CALL_MANIFEST = "held_out_call_manifest"
    HELD_OUT_EXECUTION_MANIFEST = "held_out_execution_manifest"
    COMBINED_CALL_MANIFEST = "combined_call_manifest"
    FEEDBACK_EXECUTION_MANIFEST = "feedback_execution_manifest"
    CASE_EXECUTION_PLAN = "case_execution_plan"
    CASE_EXECUTION_RESULT = "case_execution_result"


class FrozenFileReference(FrozenModel):
    artifact_id: Identifier
    relative_path: RelativePath
    file_sha256: Sha256Digest
    self_hash_field: SelfHashField | None = None


class ReceiptReference(FrozenFileReference):
    receipt_kind: ReceiptKind


class SourceFileRoute(FrozenModel):
    """A path-only route resolved and physically hashed by the materializer."""

    artifact_id: Identifier
    relative_path: RelativePath
    self_hash_field: SelfHashField | None = None


class NativeSourceRoute(SourceFileRoute):
    """Route to one artifact emitted by a recognized study producer."""

    producer_role: NativeSourceRole

    @model_validator(mode="after")
    def native_sources_are_logically_sealed(self) -> Self:
        if self.self_hash_field is None:
            raise ValueError("native phase sources must carry canonical self-hashes")
        return self


class LedgerSourceRoute(FrozenModel):
    """Declarative lineage metadata for one frozen ledger/CAS pair."""

    ledger_id: Identifier
    lineage_id: Identifier
    sequence: int = Field(ge=0, strict=True)
    parent_ledger_id: Identifier | None = None
    ledger_relative_path: RelativePath
    cas_relative_path: RelativePath


class FinalAccountingSourceRecipe(FrozenModel):
    """Routing-only input from which the complete accounting recipe is derived."""

    schema_version: Literal[FINAL_ACCOUNTING_SCHEMA_VERSION] = FINAL_ACCOUNTING_SCHEMA_VERSION
    kind: Literal["final_phase7_accounting_source_recipe"] = "final_phase7_accounting_source_recipe"
    accounting_id: Identifier
    compiled_at_utc: AwareDatetime
    base_call_inventory: SourceFileRoute
    native_source_artifacts: tuple[NativeSourceRoute, ...]
    authorization_amendments: tuple[SourceFileRoute, ...] = ()
    ledger_sources: tuple[LedgerSourceRoute, ...]
    wall_time_receipts: tuple[SourceFileRoute, ...]
    manifest_sha256: Sha256Digest

    @model_validator(mode="after")
    def closed_routes(self) -> Self:
        if not self.native_source_artifacts:
            raise ValueError("materialization requires native call/result source artifacts")
        if not self.ledger_sources:
            raise ValueError("materialization requires at least one ledger source")
        if not self.wall_time_receipts:
            raise ValueError("materialization requires at least one wall-time receipt")
        artifact_ids = [self.base_call_inventory.artifact_id]
        artifact_ids.extend(item.artifact_id for item in self.native_source_artifacts)
        artifact_ids.extend(item.artifact_id for item in self.authorization_amendments)
        artifact_ids.extend(item.artifact_id for item in self.wall_time_receipts)
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("source-recipe artifact IDs must be globally unique")
        paths = [self.base_call_inventory.relative_path]
        paths.extend(item.relative_path for item in self.native_source_artifacts)
        paths.extend(item.relative_path for item in self.authorization_amendments)
        paths.extend(item.relative_path for item in self.wall_time_receipts)
        if len(paths) != len(set(paths)):
            raise ValueError("source-recipe file-route paths must be globally unique")
        ledger_ids = [item.ledger_id for item in self.ledger_sources]
        if len(ledger_ids) != len(set(ledger_ids)):
            raise ValueError("source-recipe ledger IDs must be unique")
        immutable = self.model_dump(mode="json", exclude={"manifest_sha256"})
        if self.manifest_sha256 != canonical_sha256(immutable):
            raise ValueError("source recipe canonical self-hash mismatch")
        return self


class LedgerSnapshotSpec(FrozenModel):
    ledger_id: Identifier
    lineage_id: Identifier
    sequence: int = Field(ge=0, strict=True)
    parent_ledger_id: Identifier | None = None
    ledger_relative_path: RelativePath
    ledger_file_sha256: Sha256Digest
    cas_relative_path: RelativePath
    cas_inventory_sha256: Sha256Digest


class RegisteredCallSlot(FrozenModel):
    slot_id: Identifier
    call_class: Token
    ordinal: int = Field(ge=1, strict=True)
    phase: Annotated[str, StringConstraints(pattern=r"^phase_[1-7]$")]
    run_id: Token
    call_id: Token
    condition: Token
    included_in_itt: bool
    ledger_id: Identifier | None = None
    job_id: Token | None = None
    attempt_id: Token | None = None
    model_call_id: Token | None = None
    gpu_event_ids: tuple[Token, ...] = ()

    @model_validator(mode="after")
    def linkage_is_all_or_none(self) -> Self:
        base = (self.ledger_id, self.job_id, self.attempt_id)
        if any(item is not None for item in base) != all(item is not None for item in base):
            raise ValueError("executed call linkage requires ledger, job, and attempt IDs")
        if self.ledger_id is None and (self.model_call_id is not None or self.gpu_event_ids):
            raise ValueError("unexecuted call slot cannot claim model or GPU rows")
        if len(self.gpu_event_ids) != len(set(self.gpu_event_ids)):
            raise ValueError("call GPU-event IDs must be unique")
        return self

    @property
    def executed(self) -> bool:
        return self.ledger_id is not None


class RegisteredServiceSlot(FrozenModel):
    slot_id: Identifier
    source: Literal["base_inventory", "authorized_amendment"]
    ordinal: int = Field(ge=1, strict=True)
    phase: Annotated[str, StringConstraints(pattern=r"^phase_[1-7]$")]
    run_id: Token
    call_id: Token
    amendment_artifact_id: Identifier | None = None
    ledger_id: Identifier | None = None
    service_session_id: Token | None = None
    gpu_event_ids: tuple[Token, ...] = ()

    @model_validator(mode="after")
    def linkage_is_coherent(self) -> Self:
        if (self.source == "authorized_amendment") != (self.amendment_artifact_id is not None):
            raise ValueError("only amendment service slots name an authorization artifact")
        linked = self.ledger_id is not None
        if linked != (self.service_session_id is not None):
            raise ValueError("executed service slot requires ledger and service-session IDs")
        if not linked and self.gpu_event_ids:
            raise ValueError("unexecuted service slot cannot claim GPU events")
        if len(self.gpu_event_ids) != len(set(self.gpu_event_ids)):
            raise ValueError("service GPU-event IDs must be unique")
        return self

    @property
    def executed(self) -> bool:
        return self.ledger_id is not None


class AttemptExclusion(FrozenModel):
    ledger_id: Identifier
    attempt_id: Token
    reason: Literal["hand_authored_fixture", "cpu_only_non_gpu", "non_study_diagnostic"]


class FinalAccountingRecipe(FrozenModel):
    schema_version: Literal[FINAL_ACCOUNTING_SCHEMA_VERSION] = FINAL_ACCOUNTING_SCHEMA_VERSION
    kind: Literal["final_phase7_accounting_recipe"] = "final_phase7_accounting_recipe"
    accounting_id: Identifier
    compiled_at_utc: AwareDatetime
    source_materialization_recipe_sha256: Sha256Digest
    base_call_inventory: FrozenFileReference
    native_source_artifacts: tuple[FrozenFileReference, ...]
    authorization_amendments: tuple[FrozenFileReference, ...] = ()
    ledger_snapshots: tuple[LedgerSnapshotSpec, ...]
    receipts: tuple[ReceiptReference, ...]
    call_slots: tuple[RegisteredCallSlot, ...]
    service_slots: tuple[RegisteredServiceSlot, ...]
    attempt_exclusions: tuple[AttemptExclusion, ...] = ()
    manifest_sha256: Sha256Digest

    @model_validator(mode="after")
    def closed_inventory(self) -> Self:
        if not self.ledger_snapshots:
            raise ValueError("final accounting requires at least one ledger snapshot")
        if not self.native_source_artifacts:
            raise ValueError("final accounting requires native phase source artifacts")
        for values, label in (
            (tuple(item.ledger_id for item in self.ledger_snapshots), "ledger IDs"),
            (
                tuple(item.artifact_id for item in self.native_source_artifacts),
                "native source artifact IDs",
            ),
            (tuple(item.artifact_id for item in self.authorization_amendments), "amendment IDs"),
            (tuple(item.artifact_id for item in self.receipts), "receipt IDs"),
            (tuple(item.slot_id for item in self.call_slots), "call-slot IDs"),
            (tuple(item.slot_id for item in self.service_slots), "service-slot IDs"),
            (
                tuple((item.ledger_id, item.attempt_id) for item in self.attempt_exclusions),
                "attempt exclusions",
            ),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{label} must be unique")
        artifact_ids = (
            {self.base_call_inventory.artifact_id}
            | {item.artifact_id for item in self.native_source_artifacts}
            | {item.artifact_id for item in self.authorization_amendments}
            | {item.artifact_id for item in self.receipts}
        )
        if len(artifact_ids) != (
            1
            + len(self.native_source_artifacts)
            + len(self.authorization_amendments)
            + len(self.receipts)
        ):
            raise ValueError("all input artifact IDs must be globally unique")
        expected_calls = {
            (name, ordinal)
            for name, (count, _seconds) in REGISTERED_CALL_CLASSES.items()
            if name != SESSION_START_CLASS
            for ordinal in range(1, count + 1)
        }
        actual_calls = {(item.call_class, item.ordinal) for item in self.call_slots}
        if actual_calls != expected_calls or len(self.call_slots) != REGISTERED_INFERENCE_ATTEMPTS:
            raise ValueError("call slots differ from the exact registered 278-attempt inventory")
        base_services = [item for item in self.service_slots if item.source == "base_inventory"]
        if len(base_services) != REGISTERED_SESSION_STARTS or {
            item.ordinal for item in base_services
        } != set(range(1, REGISTERED_SESSION_STARTS + 1)):
            raise ValueError("base service slots differ from the registered eight starts")
        amendment_ids = {item.artifact_id for item in self.authorization_amendments}
        if any(
            item.amendment_artifact_id not in amendment_ids
            for item in self.service_slots
            if item.source == "authorized_amendment"
        ):
            raise ValueError("amendment service slot names an unregistered authorization")
        ledger_ids = {item.ledger_id for item in self.ledger_snapshots}
        if any(
            item.ledger_id is not None and item.ledger_id not in ledger_ids
            for item in (*self.call_slots, *self.service_slots)
        ) or any(item.ledger_id not in ledger_ids for item in self.attempt_exclusions):
            raise ValueError("accounting linkage names an unknown ledger")
        table_keys = {(item.phase, item.run_id, item.call_id) for item in self.call_slots}
        table_keys.update((item.phase, item.run_id, item.call_id) for item in self.service_slots)
        if len(table_keys) != len(self.call_slots) + len(self.service_slots):
            raise ValueError("failure-table phase/run/call keys must be unique")
        receipt_kinds = {item.receipt_kind for item in self.receipts}
        required_receipt_kinds = {
            ReceiptKind.EXECUTION,
            ReceiptKind.RESULT,
            ReceiptKind.REPAIR,
            ReceiptKind.SERVICE,
            ReceiptKind.TOKEN,
            ReceiptKind.STORAGE,
            ReceiptKind.WALL_TIME,
            ReceiptKind.ATTEMPT_EXCLUSION,
        }
        if not required_receipt_kinds.issubset(receipt_kinds):
            raise ValueError("final accounting recipe lacks one or more required receipt kinds")
        immutable = self.model_dump(mode="json", exclude={"manifest_sha256"})
        if self.manifest_sha256 != canonical_sha256(immutable):
            raise ValueError("final accounting recipe canonical self-hash mismatch")
        return self


@dataclass(frozen=True)
class FinalAccountingOutputs:
    failure_table_path: Path
    resource_table_path: Path
    receipt_path: Path
    receipt: Mapping[str, Any]


@dataclass(frozen=True)
class FinalAccountingMaterializationOutputs:
    recipe_path: Path
    receipt_paths: tuple[Path, ...]
    recipe: FinalAccountingRecipe


@dataclass(frozen=True)
class FinalAccountingSourceBuildOutputs:
    recipe_path: Path
    recipe: FinalAccountingSourceRecipe


@dataclass(frozen=True)
class _LedgerSnapshot:
    spec: LedgerSnapshotSpec
    report: LedgerVerificationReport
    row_sets: Mapping[str, frozenset[str]]
    row_hashes: Mapping[str, str]
    rows: Mapping[str, tuple[Mapping[str, Any], ...]]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_real_file(path: Path, *, label: str) -> None:
    current = Path(os.path.abspath(path))
    while True:
        if current.is_symlink():
            raise FinalAccountingError(f"{label} has a symbolic-link ancestor")
        if current.parent == current:
            break
        current = current.parent
    try:
        metadata = path.stat()
    except OSError as exc:
        raise FinalAccountingError(f"missing {label}: {path}") from exc
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise FinalAccountingError(f"{label} must be one singly-linked regular file")


def _real_root(path: Path, *, label: str) -> Path:
    lexical = Path(os.path.abspath(path))
    current = lexical
    while True:
        if current.is_symlink():
            raise FinalAccountingError(f"{label} has a symbolic-link ancestor")
        if current.parent == current:
            break
        current = current.parent
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as exc:
        raise FinalAccountingError(f"missing {label}: {path}") from exc
    if resolved != lexical or not resolved.is_dir():
        raise FinalAccountingError(f"{label} must be one real directory")
    return resolved


def _safe_input(root: Path, relative_path: str, *, label: str, directory: bool = False) -> Path:
    if "\\" in relative_path:
        raise FinalAccountingError(f"{label} path contains a backslash")
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise FinalAccountingError(f"unsafe {label} path")
    candidate = root.joinpath(*relative.parts)
    lexical = Path(os.path.abspath(candidate))
    try:
        lexical.relative_to(root)
    except ValueError as exc:
        raise FinalAccountingError(f"{label} escapes the source root") from exc
    probe = root
    for component in relative.parts:
        probe /= component
        if probe.is_symlink():
            raise FinalAccountingError(f"{label} path contains a symbolic link")
    try:
        resolved = lexical.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise FinalAccountingError(f"missing or escaping {label}") from exc
    if directory:
        if resolved != lexical or not resolved.is_dir():
            raise FinalAccountingError(f"{label} must be one real directory")
    else:
        _assert_real_file(lexical, label=label)
    return lexical


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def _load_json_file(path: Path, *, label: str) -> Mapping[str, Any]:
    _assert_real_file(path, label=label)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise FinalAccountingError(f"cannot read {label}") from exc
    if raw.startswith(b"\xef\xbb\xbf"):
        raise FinalAccountingError(f"{label} must be UTF-8 without BOM")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FinalAccountingError(f"invalid {label}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise FinalAccountingError(f"{label} must be one JSON object")
    return value


def _verify_self_hash(payload: Mapping[str, Any], field: SelfHashField, *, label: str) -> None:
    supplied = payload.get(field.value)
    immutable = {key: value for key, value in payload.items() if key != field.value}
    if supplied != canonical_sha256(immutable):
        raise FinalAccountingError(f"{label} canonical self-hash mismatch")


def load_final_accounting_recipe(path: Path) -> FinalAccountingRecipe:
    payload = _load_json_file(path, label="final accounting recipe")
    try:
        return FinalAccountingRecipe.model_validate(payload)
    except Exception as exc:
        raise FinalAccountingError(f"invalid final accounting recipe: {exc}") from exc


def load_final_accounting_source_recipe(path: Path) -> FinalAccountingSourceRecipe:
    payload = _load_json_file(path, label="final accounting source recipe")
    try:
        return FinalAccountingSourceRecipe.model_validate(payload)
    except Exception as exc:
        raise FinalAccountingError(f"invalid final accounting source recipe: {exc}") from exc


def _load_reference(
    root: Path,
    reference: FrozenFileReference,
    *,
    label: str,
) -> tuple[Path, Mapping[str, Any]]:
    path = _safe_input(root, reference.relative_path, label=label)
    before = path.stat()
    digest = _file_sha256(path)
    if digest != reference.file_sha256:
        raise FinalAccountingError(f"{label} physical SHA-256 mismatch")
    payload = _load_json_file(path, label=label)
    if reference.self_hash_field is not None:
        _verify_self_hash(payload, reference.self_hash_field, label=label)
    after = path.stat()
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        or not stat.S_ISREG(after.st_mode)
        or after.st_nlink != 1
        or _file_sha256(path) != digest
    ):
        raise FinalAccountingError(f"{label} changed while it was read")
    return path, payload


def cas_inventory_sha256(root: Path) -> str:
    """Hash every directory entry in a frozen CAS and reject aliasable entries."""

    real = _real_root(root, label="CAS root")
    rows: list[dict[str, object]] = []
    for directory, names, files in os.walk(real, topdown=True, followlinks=False):
        current = Path(directory)
        for name in sorted(names):
            child = current / name
            metadata = child.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise FinalAccountingError("CAS contains a symlink or special directory entry")
        for name in sorted(files):
            child = current / name
            metadata = child.lstat()
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise FinalAccountingError("CAS contains a symlink, special file, or hardlink")
            rows.append(
                {
                    "relative_path": child.relative_to(real).as_posix(),
                    "size_bytes": metadata.st_size,
                    "file_sha256": _file_sha256(child),
                }
            )
    rows.sort(key=lambda row: str(row["relative_path"]))
    return canonical_sha256(rows)


def _readonly_connection(path: Path) -> sqlite3.Connection:
    uri = f"file:{quote(str(path), safe='/')}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True, isolation_level=None, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _snapshot_rows(
    path: Path,
) -> tuple[
    dict[str, tuple[Mapping[str, Any], ...]],
    dict[str, frozenset[str]],
    dict[str, str],
]:
    rows_by_table: dict[str, tuple[Mapping[str, Any], ...]] = {}
    sets: dict[str, frozenset[str]] = {}
    hashes: dict[str, str] = {}
    connection = _readonly_connection(path)
    try:
        tables = {
            row["name"]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        for table in Ledger._APPEND_ONLY_TABLES:
            if table not in tables:
                raise FinalAccountingError(f"ledger is missing append-only table {table!r}")
            columns = tuple(
                row["name"] for row in connection.execute(f"PRAGMA table_info({table})")
            )
            raw_rows = tuple(
                {column: row[column] for column in columns}
                for row in connection.execute(f"SELECT * FROM {table}")
            )
            encoded = tuple(sorted(canonical_json(row) for row in raw_rows))
            rows_by_table[table] = raw_rows
            sets[table] = frozenset(encoded)
            hashes[table] = hashlib.sha256("\n".join(encoded).encode("utf-8")).hexdigest()
    finally:
        connection.close()
    return rows_by_table, sets, hashes


def _load_ledger_snapshot(root: Path, spec: LedgerSnapshotSpec) -> _LedgerSnapshot:
    ledger = _safe_input(root, spec.ledger_relative_path, label=f"ledger {spec.ledger_id}")
    cas = _safe_input(
        root,
        spec.cas_relative_path,
        label=f"CAS {spec.ledger_id}",
        directory=True,
    )
    for suffix in ("-wal", "-journal"):
        sidecar = Path(str(ledger) + suffix)
        if sidecar.is_symlink() or (sidecar.exists() and sidecar.stat().st_size):
            raise FinalAccountingError(f"ledger {spec.ledger_id} is not closed/checkpointed")
    before = ledger.stat()
    if _file_sha256(ledger) != spec.ledger_file_sha256:
        raise FinalAccountingError(f"ledger {spec.ledger_id} file SHA-256 mismatch")
    if cas_inventory_sha256(cas) != spec.cas_inventory_sha256:
        raise FinalAccountingError(f"CAS {spec.ledger_id} inventory SHA-256 mismatch")
    try:
        report = verify_ledger(ledger, cas)
    except Exception as exc:
        raise FinalAccountingError(f"ledger/CAS audit failed for {spec.ledger_id}: {exc}") from exc
    rows, row_sets, row_hashes = _snapshot_rows(ledger)
    after = ledger.stat()
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        or not stat.S_ISREG(after.st_mode)
        or after.st_nlink != 1
        or _file_sha256(ledger) != spec.ledger_file_sha256
    ):
        raise FinalAccountingError(f"ledger {spec.ledger_id} changed during verification")
    if cas_inventory_sha256(cas) != spec.cas_inventory_sha256:
        raise FinalAccountingError(f"CAS {spec.ledger_id} changed during verification")
    return _LedgerSnapshot(spec, report, row_sets, row_hashes, rows)


def _terminal_snapshots(
    snapshots: Sequence[_LedgerSnapshot],
) -> tuple[_LedgerSnapshot, ...]:
    by_lineage: dict[str, list[_LedgerSnapshot]] = defaultdict(list)
    by_id = {item.spec.ledger_id: item for item in snapshots}
    for snapshot in snapshots:
        by_lineage[snapshot.spec.lineage_id].append(snapshot)
    terminals: list[_LedgerSnapshot] = []
    for lineage, values in sorted(by_lineage.items()):
        values.sort(key=lambda item: item.spec.sequence)
        if [item.spec.sequence for item in values] != list(range(len(values))):
            raise FinalAccountingError(f"ledger lineage {lineage} has a sequence gap")
        for index, current in enumerate(values):
            expected_parent = None if index == 0 else values[index - 1].spec.ledger_id
            if current.spec.parent_ledger_id != expected_parent:
                raise FinalAccountingError(f"ledger lineage {lineage} parent chain changed")
            if index:
                parent = by_id[expected_parent]  # type: ignore[index]
                for table in Ledger._APPEND_ONLY_TABLES:
                    missing = parent.row_sets[table] - current.row_sets[table]
                    if missing:
                        raise FinalAccountingError(
                            f"ledger lineage {lineage} is not append-only in table {table}"
                        )
                if (
                    current.report.gpu_total_allocated_microseconds
                    < parent.report.gpu_total_allocated_microseconds
                ):
                    raise FinalAccountingError(f"ledger lineage {lineage} GPU total regressed")
        terminals.append(values[-1])
    return tuple(terminals)


def _require_disjoint_terminal_ledgers(terminals: Sequence[_LedgerSnapshot]) -> None:
    """Prevent one cumulative ledger from being counted as another lineage."""

    keys = {
        "attempts": "attempt_id",
        "gpu_events": "event_id",
        "gpu_service_sessions": "service_session_id",
        "model_calls": "model_call_id",
    }
    for table, key in keys.items():
        owners: dict[str, str] = {}
        for snapshot in terminals:
            for row in snapshot.rows[table]:
                identifier = str(row[key])
                prior = owners.setdefault(identifier, snapshot.spec.ledger_id)
                if prior != snapshot.spec.ledger_id:
                    raise FinalAccountingError(
                        f"terminal ledgers duplicate {table} identifier {identifier!r}"
                    )


def _short_presampler_failure(
    snapshot: _LedgerSnapshot,
    session: Mapping[str, Any],
) -> bool:
    """Recognize a launch failure that ended before the 1-second sampler cadence."""

    started = _parse_time(session["started_at"])
    ended = _parse_time(session["ended_at"])
    if (ended - started).total_seconds() >= 1.0:
        return False
    matching = [
        event
        for event in snapshot.rows["gpu_events"]
        if event["event_id"] == session["service_session_id"]
    ]
    if len(matching) != 1:
        return False
    event = matching[0]
    details = _json_object(event["details_json"], label="short service failure details")
    return (
        not bool(event["succeeded"])
        and event["event_kind"] in {"failure", "rejected"}
        and details.get("intended_event_kind") == SESSION_START_CLASS
        and details.get("session_id") == session["session_id"]
        and int(event["allocated_microseconds"]) < 1_000_000
        and _parse_time(event["started_at"]) <= ended
        and _parse_time(event["ended_at"]) >= started
    )


def _require_accounting_samples(
    terminals: Sequence[_LedgerSnapshot],
) -> Mapping[str, tuple[str, ...]]:
    uncovered_short_failures: dict[str, tuple[str, ...]] = {}
    for snapshot in terminals:
        storage = snapshot.rows["storage_samples"]
        resources = snapshot.rows["resource_samples"]
        if not storage:
            raise FinalAccountingError(
                f"terminal ledger {snapshot.spec.ledger_id} has no project-wide "
                "storage sample evidence"
            )
        if snapshot.report.gpu_total_allocated_microseconds <= 0:
            continue
        if not resources:
            raise FinalAccountingError(
                f"terminal ledger {snapshot.spec.ledger_id} has GPU allocation "
                "without any resource sample evidence"
            )
        storage_coverage = {(str(row["phase"]), _parse_time(row["sampled_at"])) for row in storage}
        for resource in resources:
            identity = (
                f"resource_sample:{resource['sample_id']}",
                _parse_time(resource["sampled_at"]),
            )
            if identity not in storage_coverage:
                raise FinalAccountingError(
                    f"resource sample {resource['sample_id']} lacks its paired storage sample"
                )
        short_failures: list[str] = []
        for session in snapshot.rows["gpu_service_sessions"]:
            started = _parse_time(session["started_at"])
            ended = _parse_time(session["ended_at"])
            if any(started <= _parse_time(row["sampled_at"]) <= ended for row in resources):
                continue
            if _short_presampler_failure(snapshot, session):
                short_failures.append(str(session["service_session_id"]))
                continue
            raise FinalAccountingError(
                f"GPU service {session['service_session_id']} lacks resource-sample coverage"
            )
        uncovered_short_failures[snapshot.spec.ledger_id] = tuple(sorted(short_failures))
    return uncovered_short_failures


def _require_source_chronology(
    source_recipe: FinalAccountingSourceRecipe,
    terminals: Sequence[_LedgerSnapshot],
) -> None:
    timestamps = _terminal_evidence_timestamps(terminals)
    if timestamps and source_recipe.compiled_at_utc < max(timestamps):
        raise FinalAccountingError(
            "materialization source recipe predates terminal ledger evidence"
        )


def _terminal_evidence_timestamps(
    terminals: Sequence[_LedgerSnapshot],
) -> tuple[datetime, ...]:
    return tuple(
        _parse_time(value)
        for snapshot in terminals
        for table, rows in snapshot.rows.items()
        if table != "schema_metadata"
        for row in rows
        for field, value in row.items()
        if field.endswith("_at") and value is not None
    )


def _amendment_additional_services(payload: Mapping[str, Any], inventory_hash: str) -> int:
    kind = payload.get("kind")
    if kind not in {
        "phase1_fallback_service_retry_amendment",
        "phase1_fallback_second_recovery_overlay",
    }:
        raise FinalAccountingError("authorization amendment has an unrecognized producer kind")
    if kind == "phase1_fallback_second_recovery_overlay":
        from story_projection_onto.fallback_acceptance import (
            SecondFallbackRecoveryOverlay,
        )

        try:
            SecondFallbackRecoveryOverlay.model_validate(payload)
        except Exception as exc:
            raise FinalAccountingError(
                "second recovery amendment violates its exact typed contract"
            ) from exc
    if payload.get("base_gpu_call_inventory_file_sha256") != inventory_hash:
        raise FinalAccountingError("authorization amendment changed the base call inventory")
    authorization = payload.get("authorization_status")
    if authorization is None and isinstance(payload.get("authorization"), Mapping):
        authorization = payload["authorization"].get("status")
    if authorization != "authorized":
        raise FinalAccountingError("inventory amendment is not explicitly authorized")
    delta = payload.get("amendment")
    if not isinstance(delta, Mapping):
        raise FinalAccountingError("inventory amendment lacks its typed delta")
    additional = delta.get("additional_fallback_service_loads")
    if isinstance(additional, bool) or not isinstance(additional, int) or additional <= 0:
        raise FinalAccountingError("inventory amendment service delta is invalid")
    extra_inference = delta.get(
        "additional_unreserved_inference_attempts",
        delta.get("additional_inference_attempts"),
    )
    if extra_inference != 0:
        raise FinalAccountingError("inventory amendment adds unregistered inference attempts")
    maximum = delta.get(
        "amended_maximum_inference_attempts",
        delta.get("maximum_inference_attempts_unchanged"),
    )
    if maximum != REGISTERED_INFERENCE_ATTEMPTS:
        raise FinalAccountingError("inventory amendment changed the 278-attempt maximum")
    return additional


def _receipt_payloads(
    root: Path,
    references: Sequence[ReceiptReference],
    *,
    source_recipe_hash: str,
    native_file_hashes: Sequence[str],
    ledger_file_hashes: Sequence[str],
    amendment_file_hashes: Sequence[str],
    inventory_file_hash: str,
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    expected_native_hashes = tuple(sorted(set(native_file_hashes)))
    expected_ledger_hashes = tuple(sorted(set(ledger_file_hashes)))
    expected_amendment_hashes = tuple(sorted(set(amendment_file_hashes)))
    for reference in references:
        if reference.self_hash_field is None:
            raise FinalAccountingError(
                f"receipt {reference.artifact_id} must carry a canonical self-hash"
            )
        _path, payload = _load_reference(
            root,
            reference,
            label=f"{reference.receipt_kind.value} {reference.artifact_id}",
        )
        if payload.get("kind") != reference.receipt_kind.value:
            raise FinalAccountingError(f"receipt {reference.artifact_id} has the wrong kind")
        if reference.receipt_kind is not ReceiptKind.WALL_TIME and (
            payload.get("generated_by") != "story_projection_onto.final_accounting.materializer/v1"
            or payload.get("source_materialization_recipe_sha256") != source_recipe_hash
            or _string_list(
                payload,
                "source_native_file_sha256",
                label=reference.artifact_id,
            )
            != expected_native_hashes
            or _string_list(
                payload,
                "source_ledger_file_sha256",
                label=reference.artifact_id,
            )
            != expected_ledger_hashes
            or _string_list(
                payload,
                "source_authorization_file_sha256",
                label=reference.artifact_id,
            )
            != expected_amendment_hashes
            or payload.get("base_call_inventory_file_sha256") != inventory_file_hash
        ):
            raise FinalAccountingError(
                f"receipt {reference.artifact_id} lacks exact materializer/source lineage"
            )
        result[reference.artifact_id] = payload
    return result


def _list_of_mappings(
    payload: Mapping[str, Any], field: str, *, label: str
) -> tuple[Mapping[str, Any], ...]:
    value = payload.get(field)
    if not isinstance(value, list) or not all(isinstance(item, Mapping) for item in value):
        raise FinalAccountingError(f"{label} requires a {field} object array")
    return tuple(value)


def _string_list(payload: Mapping[str, Any], field: str, *, label: str) -> tuple[str, ...]:
    value = payload.get(field)
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise FinalAccountingError(f"{label} requires a {field} string array")
    if len(value) != len(set(value)):
        raise FinalAccountingError(f"{label} {field} contains duplicates")
    return tuple(value)


def _indexed_rows(snapshot: _LedgerSnapshot, table: str, key: str) -> dict[str, Mapping[str, Any]]:
    return {str(row[key]): row for row in snapshot.rows[table]}


def _attempt_exclusions(
    recipe: FinalAccountingRecipe,
    payloads: Mapping[str, Mapping[str, Any]],
) -> set[tuple[str, str]]:
    declared = {
        (item.ledger_id, item.attempt_id, item.reason) for item in recipe.attempt_exclusions
    }
    receipted: set[tuple[str, str, str]] = set()
    for reference in recipe.receipts:
        if reference.receipt_kind is not ReceiptKind.ATTEMPT_EXCLUSION:
            continue
        for item in _list_of_mappings(
            payloads[reference.artifact_id], "exclusions", label=reference.artifact_id
        ):
            triple = (item.get("ledger_id"), item.get("attempt_id"), item.get("reason"))
            if not all(isinstance(value, str) for value in triple):
                raise FinalAccountingError("attempt-exclusion receipt has an invalid binding")
            receipted.add(triple)  # type: ignore[arg-type]
    if declared != receipted:
        raise FinalAccountingError("attempt exclusions differ from their immutable receipts")
    return {(ledger, attempt) for ledger, attempt, _reason in declared}


def _validate_result_receipts(
    recipe: FinalAccountingRecipe,
    payloads: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    registered: set[str] = set()
    itt: set[str] = set()
    non_itt: set[str] = set()
    outcomes: dict[str, str] = {}
    allowed_outcomes = {
        "success",
        "repaired",
        "failed",
        "invalid",
        "timed_out",
        "interrupted",
        "incomplete",
        "not_used",
    }
    for reference in recipe.receipts:
        if reference.receipt_kind is not ReceiptKind.RESULT:
            continue
        payload = payloads[reference.artifact_id]
        current = set(_string_list(payload, "registered_slot_ids", label=reference.artifact_id))
        current_itt = set(_string_list(payload, "itt_slot_ids", label=reference.artifact_id))
        current_non = set(_string_list(payload, "non_itt_slot_ids", label=reference.artifact_id))
        if current_itt & current_non or current_itt | current_non != current:
            raise FinalAccountingError("result receipt does not partition ITT membership")
        if registered & current:
            raise FinalAccountingError("result receipts duplicate a registered call slot")
        registered.update(current)
        itt.update(current_itt)
        non_itt.update(current_non)
        for binding in _list_of_mappings(payload, "terminal_outcomes", label=reference.artifact_id):
            if set(binding) != {"slot_id", "outcome"}:
                raise FinalAccountingError("result outcome binding has unexpected fields")
            slot_id, outcome = binding.get("slot_id"), binding.get("outcome")
            if (
                not isinstance(slot_id, str)
                or not isinstance(outcome, str)
                or outcome not in allowed_outcomes
                or slot_id in outcomes
            ):
                raise FinalAccountingError("result receipt has an invalid/duplicate outcome")
            outcomes[slot_id] = outcome
    expected = {item.slot_id for item in recipe.call_slots}
    if registered != expected:
        raise FinalAccountingError("result receipts do not cover the exact 278 call slots")
    if itt != {item.slot_id for item in recipe.call_slots if item.included_in_itt}:
        raise FinalAccountingError("recipe ITT membership differs from result receipts")
    if non_itt != expected - itt:
        raise FinalAccountingError("non-ITT result receipt inventory is incomplete")
    if set(outcomes) != expected:
        raise FinalAccountingError("result receipts do not cover every terminal call outcome")
    return outcomes


def _validate_execution_receipts(
    recipe: FinalAccountingRecipe,
    payloads: Mapping[str, Mapping[str, Any]],
) -> None:
    observed: dict[str, Mapping[str, Any]] = {}
    for reference in recipe.receipts:
        if reference.receipt_kind is not ReceiptKind.EXECUTION:
            continue
        for binding in _list_of_mappings(
            payloads[reference.artifact_id], "call_bindings", label=reference.artifact_id
        ):
            slot_id = binding.get("slot_id")
            if not isinstance(slot_id, str) or slot_id in observed:
                raise FinalAccountingError("execution receipts contain a duplicate/invalid slot")
            observed[slot_id] = binding
    expected_slots = {item.slot_id: item for item in recipe.call_slots if item.executed}
    if set(observed) != set(expected_slots):
        raise FinalAccountingError("execution receipts differ from executed call slots")
    for slot_id, slot in expected_slots.items():
        expected = {
            "slot_id": slot.slot_id,
            "ledger_id": slot.ledger_id,
            "job_id": slot.job_id,
            "attempt_id": slot.attempt_id,
            "model_call_id": slot.model_call_id,
            "gpu_event_ids": list(slot.gpu_event_ids),
        }
        if dict(observed[slot_id]) != expected:
            raise FinalAccountingError(f"execution receipt changed linkage for {slot_id}")


def _validate_repair_receipts(
    actual: Mapping[str, str],
    recipe: FinalAccountingRecipe,
    payloads: Mapping[str, Mapping[str, Any]],
) -> None:
    observed: dict[str, str] = {}
    for reference in recipe.receipts:
        if reference.receipt_kind is not ReceiptKind.REPAIR:
            continue
        for binding in _list_of_mappings(
            payloads[reference.artifact_id], "repair_bindings", label=reference.artifact_id
        ):
            slot_id, parent = binding.get("slot_id"), binding.get("parent_slot_id")
            if not isinstance(slot_id, str) or not isinstance(parent, str) or slot_id in observed:
                raise FinalAccountingError("repair receipt contains an invalid/duplicate binding")
            observed[slot_id] = parent
    if observed != dict(actual):
        raise FinalAccountingError("repair receipts differ from ledger attempt ancestry")


def _validate_service_receipts(
    recipe: FinalAccountingRecipe,
    payloads: Mapping[str, Mapping[str, Any]],
    *,
    all_stopped: bool,
) -> None:
    registered: set[str] = set()
    bindings: dict[str, Mapping[str, Any]] = {}
    stopped_claims: list[bool] = []
    for reference in recipe.receipts:
        if reference.receipt_kind is not ReceiptKind.SERVICE:
            continue
        payload = payloads[reference.artifact_id]
        current = set(
            _string_list(payload, "registered_service_slot_ids", label=reference.artifact_id)
        )
        if registered & current:
            raise FinalAccountingError("service receipts duplicate registered slots")
        registered.update(current)
        stopped = payload.get("all_services_stopped")
        if not isinstance(stopped, bool):
            raise FinalAccountingError("service receipt lacks a boolean stopped state")
        stopped_claims.append(stopped)
        for binding in _list_of_mappings(payload, "service_bindings", label=reference.artifact_id):
            slot_id = binding.get("slot_id")
            if not isinstance(slot_id, str) or slot_id in bindings:
                raise FinalAccountingError("service receipts contain a duplicate binding")
            bindings[slot_id] = binding
    if registered != {item.slot_id for item in recipe.service_slots}:
        raise FinalAccountingError("service receipts do not cover the effective slot inventory")
    expected = {item.slot_id: item for item in recipe.service_slots if item.executed}
    if set(bindings) != set(expected):
        raise FinalAccountingError("service receipts differ from executed service slots")
    for slot_id, slot in expected.items():
        wanted = {
            "slot_id": slot.slot_id,
            "ledger_id": slot.ledger_id,
            "service_session_id": slot.service_session_id,
            "gpu_event_ids": list(slot.gpu_event_ids),
        }
        if dict(bindings[slot_id]) != wanted:
            raise FinalAccountingError(f"service receipt changed linkage for {slot_id}")
    if not stopped_claims or any(claim != all_stopped for claim in stopped_claims):
        raise FinalAccountingError("service stopped receipt disagrees with terminal ledgers")


def _validate_token_receipts(
    terminals: Sequence[_LedgerSnapshot],
    recipe: FinalAccountingRecipe,
    payloads: Mapping[str, Mapping[str, Any]],
) -> None:
    by_ledger: dict[str, Mapping[str, Any]] = {}
    for reference in recipe.receipts:
        if reference.receipt_kind is not ReceiptKind.TOKEN:
            continue
        payload = payloads[reference.artifact_id]
        ledger_id = payload.get("ledger_id")
        if not isinstance(ledger_id, str) or ledger_id in by_ledger:
            raise FinalAccountingError("token receipts require one unique terminal ledger ID")
        by_ledger[ledger_id] = payload
    if set(by_ledger) != {item.spec.ledger_id for item in terminals}:
        raise FinalAccountingError("token receipts do not cover every terminal ledger")
    for snapshot in terminals:
        calls = [row for row in snapshot.rows["model_calls"] if row["backend"] == "vllm_gpu"]
        expected_ids = sorted(str(row["model_call_id"]) for row in calls)
        payload = by_ledger[snapshot.spec.ledger_id]
        if _string_list(payload, "model_call_ids", label="token receipt") != tuple(expected_ids):
            raise FinalAccountingError("token receipt model-call inventory mismatch")
        prompt = sum(int(row["prompt_tokens"]) for row in calls)
        completion = sum(int(row["completion_tokens"]) for row in calls)
        if (
            payload.get("prompt_tokens") != prompt
            or payload.get("completion_tokens") != completion
            or payload.get("total_tokens") != prompt + completion
        ):
            raise FinalAccountingError("token receipt totals differ from the ledger")


def _validate_storage_receipts(
    terminals: Sequence[_LedgerSnapshot],
    recipe: FinalAccountingRecipe,
    payloads: Mapping[str, Mapping[str, Any]],
    resource_coverage_gaps: Mapping[str, tuple[str, ...]],
) -> None:
    by_ledger: dict[str, Mapping[str, Any]] = {}
    for reference in recipe.receipts:
        if reference.receipt_kind is not ReceiptKind.STORAGE:
            continue
        payload = payloads[reference.artifact_id]
        ledger_id = payload.get("ledger_id")
        if not isinstance(ledger_id, str) or ledger_id in by_ledger:
            raise FinalAccountingError("storage receipts require one unique terminal ledger ID")
        by_ledger[ledger_id] = payload
    if set(by_ledger) != {item.spec.ledger_id for item in terminals}:
        raise FinalAccountingError("storage receipts do not cover every terminal ledger")
    for snapshot in terminals:
        storage = snapshot.rows["storage_samples"]
        resources = snapshot.rows["resource_samples"]
        payload = by_ledger[snapshot.spec.ledger_id]
        expected_ids = tuple(sorted(str(row["sample_id"]) for row in storage))
        if _string_list(payload, "storage_sample_ids", label="storage receipt") != expected_ids:
            raise FinalAccountingError("storage receipt sample inventory mismatch")
        expected_resource_ids = tuple(sorted(str(row["sample_id"]) for row in resources))
        if (
            _string_list(payload, "resource_sample_ids", label="storage receipt")
            != expected_resource_ids
        ):
            raise FinalAccountingError("storage receipt resource inventory mismatch")
        expected_gaps = resource_coverage_gaps.get(snapshot.spec.ledger_id, ())
        if _string_list(
            payload,
            "uncovered_short_failed_service_session_ids",
            label="storage receipt",
        ) != expected_gaps or payload.get("resource_sampling_coverage_complete") is not (
            not expected_gaps
        ):
            raise FinalAccountingError("storage receipt resource coverage differs from ledger")
        peak_actual = max(
            [int(row["current_occupied_bytes"]) for row in storage]
            + [int(row["project_storage_bytes"]) for row in resources]
            + [0]
        )
        peak_projected = max([int(row["projected_occupied_bytes"]) for row in storage] + [0])
        minimum_headroom = min(
            [int(row["effective_projected_headroom_bytes"]) for row in storage] or [0]
        )
        all_allowed = all(bool(row["allowed"]) for row in storage)
        if (
            payload.get("peak_project_storage_bytes") != peak_actual
            or payload.get("peak_projected_storage_bytes") != peak_projected
            or payload.get("minimum_effective_headroom_bytes") != minimum_headroom
            or payload.get("all_storage_samples_allowed") is not all_allowed
        ):
            raise FinalAccountingError("storage receipt summary differs from the ledger")


def _wall_time_microseconds(
    recipe: FinalAccountingRecipe,
    payloads: Mapping[str, Mapping[str, Any]],
    terminals: Sequence[_LedgerSnapshot],
) -> int:
    samples: dict[str, int] = {}
    sampled_times: list[datetime] = []
    count = 0
    for reference in recipe.receipts:
        if reference.receipt_kind is not ReceiptKind.WALL_TIME:
            continue
        payload = payloads[reference.artifact_id]
        count += 1
        start_time, sampled_time, elapsed = _validated_wall_time_sample(payload)
        if sampled_time > recipe.compiled_at_utc:
            raise FinalAccountingError("wall-time receipt postdates the accounting source freeze")
        sampled_times.append(sampled_time)
        key = start_time.astimezone(UTC).isoformat()
        samples[key] = max(samples.get(key, 0), elapsed)
    if count == 0:
        raise FinalAccountingError("final accounting requires at least one wall-time receipt")
    ledger_times = _terminal_evidence_timestamps(terminals)
    if ledger_times and max(sampled_times) < max(ledger_times):
        raise FinalAccountingError("wall-time evidence predates terminal ledger evidence")
    return sum(samples.values())


def _validated_wall_time_sample(
    payload: Mapping[str, Any],
) -> tuple[datetime, datetime, int]:
    if (
        payload.get("schema_version") != FINAL_ACCOUNTING_SCHEMA_VERSION
        or payload.get("kind") != ReceiptKind.WALL_TIME.value
        or payload.get("scientific_gpu_accounting") is not False
        or payload.get("billing_time_equivalence_claimed") is not False
        or payload.get("container_pid") != 1
        or payload.get("scope") != "current_container_pid_1_lifetime"
        or payload.get("measurement_method") != "linux_proc_boot_epoch_plus_pid1_start_ticks"
    ):
        raise FinalAccountingError("wall-time receipt makes an unsupported accounting claim")
    elapsed = payload.get("elapsed_microseconds")
    started = payload.get("container_pid_1_started_at")
    sampled = payload.get("sampled_at")
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, int)
        or elapsed < 0
        or not isinstance(started, str)
        or not isinstance(sampled, str)
    ):
        raise FinalAccountingError("wall-time receipt has invalid measurement fields")
    try:
        start_time = datetime.fromisoformat(started.replace("Z", "+00:00"))
        sampled_time = datetime.fromisoformat(sampled.replace("Z", "+00:00"))
    except ValueError as exc:
        raise FinalAccountingError("wall-time receipt timestamp is invalid") from exc
    if start_time.tzinfo is None or sampled_time.tzinfo is None or sampled_time < start_time:
        raise FinalAccountingError("wall-time receipt chronology is invalid")
    delta = sampled_time - start_time
    expected_elapsed = delta.days * 86_400_000_000 + delta.seconds * 1_000_000 + delta.microseconds
    if elapsed != expected_elapsed:
        raise FinalAccountingError("wall-time receipt elapsed duration is inconsistent")
    return start_time, sampled_time, elapsed


def _parse_time(value: object) -> datetime:
    if not isinstance(value, str):
        raise FinalAccountingError("ledger timestamp is not text")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise FinalAccountingError("ledger timestamp is invalid") from exc
    if parsed.tzinfo is None:
        raise FinalAccountingError("ledger timestamp lacks an offset")
    return parsed


def _failure_outcome(failure_kind: str | None) -> str:
    if failure_kind == "timeout":
        return "timed_out"
    if failure_kind in {"invalid_output", "validation"}:
        return "invalid"
    if failure_kind == "interrupted":
        return "interrupted"
    return "failed"


def _seconds(microseconds: int) -> str:
    return f"{Decimal(microseconds) / Decimal(1_000_000):.6f}"


def _bool(value: bool) -> str:
    return "true" if value else "false"


def _compile_failure_rows(
    recipe: FinalAccountingRecipe,
    terminals: Sequence[_LedgerSnapshot],
    exclusions: set[tuple[str, str]],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    terminal_by_id = {item.spec.ledger_id: item for item in terminals}
    slot_by_attempt: dict[tuple[str, str], RegisteredCallSlot] = {}
    claimed_model_calls: set[tuple[str, str]] = set()
    claimed_events: set[tuple[str, str]] = set()
    claimed_sessions: set[tuple[str, str]] = set()
    direct_outcomes: dict[str, str] = {}
    scientific_statuses: dict[str, str] = {}
    repair_parent_slots: dict[str, str] = {}
    allocation_by_slot: dict[str, int] = {}
    failure_by_slot: dict[str, str] = {}
    prompt_by_slot: dict[str, int] = {}
    completion_by_slot: dict[str, int] = {}
    contexts_by_slot: dict[str, int] = {}
    attempt_class_by_slot: dict[str, str] = {}

    for slot in recipe.call_slots:
        if not slot.executed:
            scientific_statuses[slot.slot_id] = "not_assessed"
            direct_outcomes[slot.slot_id] = (
                "not_used" if slot.call_class.startswith("reserve_") else "incomplete"
            )
            allocation_by_slot[slot.slot_id] = 0
            prompt_by_slot[slot.slot_id] = 0
            completion_by_slot[slot.slot_id] = 0
            contexts_by_slot[slot.slot_id] = 0
            attempt_class_by_slot[slot.slot_id] = "not_started"
            failure_by_slot[slot.slot_id] = ""
            continue
        assert (
            slot.ledger_id is not None and slot.job_id is not None and slot.attempt_id is not None
        )
        snapshot = terminal_by_id.get(slot.ledger_id)
        if snapshot is None:
            raise FinalAccountingError("executed call is not bound to a terminal ledger")
        attempts = _indexed_rows(snapshot, "attempts", "attempt_id")
        events = _indexed_rows(snapshot, "gpu_events", "event_id")
        model_calls = _indexed_rows(snapshot, "model_calls", "model_call_id")
        failures = {str(row["attempt_id"]): row for row in snapshot.rows["failures"]}
        attempt = attempts.get(slot.attempt_id)
        if attempt is None or attempt["job_id"] != slot.job_id:
            raise FinalAccountingError(f"call slot {slot.slot_id} changed job/attempt linkage")
        attempt_key = (slot.ledger_id, slot.attempt_id)
        if attempt_key in slot_by_attempt:
            raise FinalAccountingError("two registered slots claim the same attempt")
        slot_by_attempt[attempt_key] = slot
        actual_event_ids = {
            str(row["event_id"])
            for row in snapshot.rows["gpu_events"]
            if row["attempt_id"] == slot.attempt_id
        }
        if actual_event_ids != set(slot.gpu_event_ids):
            raise FinalAccountingError(f"call slot {slot.slot_id} GPU-event inventory mismatch")
        micros = 0
        for event_id in slot.gpu_event_ids:
            event = events.get(event_id)
            if (
                event is None
                or event["job_id"] != slot.job_id
                or event["attempt_id"] != slot.attempt_id
            ):
                raise FinalAccountingError(f"call slot {slot.slot_id} has cross-lineage GPU event")
            event_key = (slot.ledger_id, event_id)
            if event_key in claimed_events:
                raise FinalAccountingError("one GPU event is claimed by multiple slots")
            claimed_events.add(event_key)
            micros += int(event["allocated_microseconds"])
        call = None
        actual_calls = [
            row for row in snapshot.rows["model_calls"] if row["attempt_id"] == slot.attempt_id
        ]
        prompt_by_slot[slot.slot_id] = 0
        completion_by_slot[slot.slot_id] = 0
        contexts_by_slot[slot.slot_id] = 0
        if slot.model_call_id is None:
            if actual_calls:
                raise FinalAccountingError(f"call slot {slot.slot_id} omits its model-call row")
        else:
            call = model_calls.get(slot.model_call_id)
            if (
                call is None
                or len(actual_calls) != 1
                or call["attempt_id"] != slot.attempt_id
                or call["job_id"] != slot.job_id
                or call["backend"] != "vllm_gpu"
            ):
                raise FinalAccountingError(f"call slot {slot.slot_id} model-call linkage mismatch")
            key = (slot.ledger_id, slot.model_call_id)
            if key in claimed_model_calls:
                raise FinalAccountingError("one model call is claimed by multiple slots")
            claimed_model_calls.add(key)
            if call["gpu_event_id"] not in slot.gpu_event_ids:
                raise FinalAccountingError("model-call event is absent from its call slot")
            prompt_by_slot[slot.slot_id] = int(call["prompt_tokens"])
            completion_by_slot[slot.slot_id] = int(call["completion_tokens"])
            contexts_by_slot[slot.slot_id] = int(call["served_context_count"])
        failure = failures.get(slot.attempt_id)
        failure_kind = None if failure is None else str(failure["failure_kind"])
        event_failed = any(events[event_id]["succeeded"] == 0 for event_id in slot.gpu_event_ids)
        if failure is not None:
            # A transport-complete request may still have an invalid output.
            # The failure takes precedence; do not rewrite either source row.
            outcome = _failure_outcome(failure_kind)
        elif call is not None and bool(call["successful"]):
            if event_failed:
                raise FinalAccountingError(
                    "successful model call has contradictory failure evidence"
                )
            outcome = "success"
        elif call is not None or event_failed:
            raise FinalAccountingError("unsuccessful call lacks an immutable failure record")
        else:
            outcome = "incomplete"
        scientific_statuses[slot.slot_id] = (
            str(
                derive_call_status(
                    call,
                    events.get(call["gpu_event_id"]),
                    snapshot.rows["failures"],
                    snapshot.rows["validations"],
                    diagnostic_only=False,
                )["scientific_status"]
            )
            if call is not None
            else "not_assessed"
        )
        direct_outcomes[slot.slot_id] = outcome
        allocation_by_slot[slot.slot_id] = micros
        failure_by_slot[slot.slot_id] = failure_kind or ""
        attempt_class_by_slot[slot.slot_id] = str(attempt["attempt_kind"])

    for slot in recipe.call_slots:
        if not slot.executed:
            continue
        assert slot.ledger_id is not None and slot.attempt_id is not None
        snapshot = terminal_by_id[slot.ledger_id]
        attempt = _indexed_rows(snapshot, "attempts", "attempt_id")[slot.attempt_id]
        parent_id = attempt["parent_attempt_id"]
        if parent_id is None:
            if attempt["attempt_kind"] != "base":
                raise FinalAccountingError("non-base registered attempt lacks its parent")
            continue
        parent_slot = slot_by_attempt.get((slot.ledger_id, str(parent_id)))
        if parent_slot is None:
            if (slot.ledger_id, str(parent_id)) not in exclusions:
                raise FinalAccountingError("registered repair/retry parent is not accounted")
            repair_parent_slots[slot.slot_id] = f"excluded-attempt:{slot.ledger_id}:{parent_id}"
            continue
        repair_parent_slots[slot.slot_id] = parent_slot.slot_id
        if (
            direct_outcomes[slot.slot_id] == "success"
            and direct_outcomes[parent_slot.slot_id] != "success"
        ):
            direct_outcomes[parent_slot.slot_id] = "repaired"

    for snapshot in terminals:
        ledger_id = snapshot.spec.ledger_id
        attempts = _indexed_rows(snapshot, "attempts", "attempt_id")
        calls = _indexed_rows(snapshot, "model_calls", "model_call_id")
        events = _indexed_rows(snapshot, "gpu_events", "event_id")
        relevant_attempts = set(attempts)
        claimed_attempts = {
            attempt_id
            for (claimed_ledger, attempt_id) in slot_by_attempt
            if claimed_ledger == ledger_id
        }
        excluded_attempts = {
            attempt_id
            for (excluded_ledger, attempt_id) in exclusions
            if excluded_ledger == ledger_id
        }
        if (
            claimed_attempts & excluded_attempts
            or relevant_attempts != claimed_attempts | excluded_attempts
        ):
            raise FinalAccountingError(f"ledger {ledger_id} attempt inventory is not exact")
        for attempt_id in excluded_attempts:
            attempt = attempts.get(attempt_id)
            if attempt is None:
                raise FinalAccountingError("attempt exclusion names an unknown attempt")
            linked_events = [row for row in events.values() if row["attempt_id"] == attempt_id]
            linked_calls = [row for row in calls.values() if row["attempt_id"] == attempt_id]
            if linked_events or any(row["backend"] == "vllm_gpu" for row in linked_calls):
                raise FinalAccountingError("GPU/VLLM attempt cannot be excluded from accounting")
        expected_calls = {
            (ledger_id, str(row["model_call_id"]))
            for row in calls.values()
            if row["backend"] == "vllm_gpu"
        }
        if {item for item in claimed_model_calls if item[0] == ledger_id} != expected_calls:
            raise FinalAccountingError(f"ledger {ledger_id} VLLM call inventory is not exact")

    service_outcomes: dict[str, str] = {}
    service_failure_types: dict[str, str] = {}
    for slot in recipe.service_slots:
        if not slot.executed:
            service_outcomes[slot.slot_id] = (
                "incomplete" if slot.source == "base_inventory" else "not_used"
            )
            service_failure_types[slot.slot_id] = ""
            allocation_by_slot[slot.slot_id] = 0
            continue
        assert slot.ledger_id is not None and slot.service_session_id is not None
        snapshot = terminal_by_id.get(slot.ledger_id)
        if snapshot is None:
            raise FinalAccountingError("executed service slot is not on a terminal ledger")
        events = _indexed_rows(snapshot, "gpu_events", "event_id")
        sessions = _indexed_rows(snapshot, "gpu_service_sessions", "service_session_id")
        session = sessions.get(slot.service_session_id)
        if session is None:
            raise FinalAccountingError("service slot names an unknown session")
        session_key = (slot.ledger_id, slot.service_session_id)
        if session_key in claimed_sessions:
            raise FinalAccountingError("one service session is claimed by multiple slots")
        claimed_sessions.add(session_key)
        micros = int(session["overhead_microseconds"])
        kinds: list[str] = []
        for event_id in slot.gpu_event_ids:
            event = events.get(event_id)
            if event is None:
                raise FinalAccountingError("service slot names an unknown GPU event")
            event_key = (slot.ledger_id, event_id)
            if event_key in claimed_events:
                raise FinalAccountingError("one GPU event is claimed by multiple accounting rows")
            if not _service_event_belongs_to_session(session, event):
                raise FinalAccountingError("service-slot event falls outside its service session")
            claimed_events.add(event_key)
            micros += int(event["allocated_microseconds"])
            if event["succeeded"] == 0:
                kinds.append(str(event["event_kind"]))
        allocation_by_slot[slot.slot_id] = micros
        if "timeout" in kinds:
            service_outcomes[slot.slot_id] = "timed_out"
            service_failure_types[slot.slot_id] = "timeout"
        elif kinds:
            service_outcomes[slot.slot_id] = "failed"
            service_failure_types[slot.slot_id] = kinds[0]
        else:
            service_outcomes[slot.slot_id] = "success"
            service_failure_types[slot.slot_id] = ""

    total_summary_micros = 0
    for snapshot in terminals:
        ledger_id = snapshot.spec.ledger_id
        events = snapshot.rows["gpu_events"]
        sessions = snapshot.rows["gpu_service_sessions"]
        if {item for item in claimed_events if item[0] == ledger_id} != {
            (ledger_id, str(row["event_id"])) for row in events
        }:
            raise FinalAccountingError(f"ledger {ledger_id} GPU-event inventory is not exact")
        if {item for item in claimed_sessions if item[0] == ledger_id} != {
            (ledger_id, str(row["service_session_id"])) for row in sessions
        }:
            raise FinalAccountingError(f"ledger {ledger_id} service-session inventory is not exact")
        for session in sessions:
            contained = [row for row in events if _service_event_belongs_to_session(session, row)]
            if sum(int(row["allocated_microseconds"]) for row in contained) != int(
                session["classified_event_microseconds"]
            ):
                raise FinalAccountingError("service classified-event subtotal is not chronological")
        for event in events:
            containing_sessions = [
                session for session in sessions if _service_event_belongs_to_session(session, event)
            ]
            if len(containing_sessions) != 1:
                raise FinalAccountingError(
                    "every GPU event must belong to exactly one non-overlapping service session"
                )
        total_summary_micros += snapshot.report.gpu_total_allocated_microseconds
    if sum(allocation_by_slot.values()) != total_summary_micros:
        raise FinalAccountingError("accounting rows do not sum to ledger GPU-summary semantics")

    rows: list[dict[str, str]] = []
    for slot in recipe.call_slots:
        parent_slot_id = repair_parent_slots.get(slot.slot_id)
        parent_call = ""
        if parent_slot_id is not None:
            if parent_slot_id.startswith("excluded-attempt:"):
                parent_call = parent_slot_id
            else:
                parent_call = next(
                    item.call_id for item in recipe.call_slots if item.slot_id == parent_slot_id
                )
        rows.append(
            {
                "phase": slot.phase,
                "run_id": slot.run_id,
                "call_id": slot.call_id,
                "condition": slot.condition,
                "outcome": direct_outcomes[slot.slot_id],
                "outcome_scope": "execution_and_recorded_failure_not_scientific_acceptance",
                "scientific_status": scientific_statuses[slot.slot_id],
                "attempt_class": attempt_class_by_slot[slot.slot_id],
                "repair_parent_call_id": parent_call,
                "allocated_gpu_seconds": _seconds(allocation_by_slot[slot.slot_id]),
                "included_in_itt": _bool(slot.included_in_itt),
                "failure_type": failure_by_slot[slot.slot_id],
            }
        )
    for slot in recipe.service_slots:
        rows.append(
            {
                "phase": slot.phase,
                "run_id": slot.run_id,
                "call_id": slot.call_id,
                "condition": "gpu_service",
                "outcome": service_outcomes[slot.slot_id],
                "outcome_scope": "service_execution_only",
                "scientific_status": "not_applicable",
                "attempt_class": "service",
                "repair_parent_call_id": "",
                "allocated_gpu_seconds": _seconds(allocation_by_slot[slot.slot_id]),
                "included_in_itt": "false",
                "failure_type": service_failure_types[slot.slot_id],
            }
        )
    rows.sort(key=lambda row: (row["phase"], row["run_id"], row["call_id"]))
    facts = {
        "allocation_by_slot": allocation_by_slot,
        "prompt_by_slot": prompt_by_slot,
        "completion_by_slot": completion_by_slot,
        "contexts_by_slot": contexts_by_slot,
        "outcome_by_slot": direct_outcomes,
        "repair_parent_slots": repair_parent_slots,
        "total_gpu_microseconds": total_summary_micros,
    }
    return rows, facts


def _resource_row(
    scope: str,
    metric: str,
    value: object,
    unit: str,
    note: str,
    *,
    status: str = "observed",
) -> dict[str, str]:
    rendered = _bool(value) if isinstance(value, bool) else str(value)
    return {
        "scope": scope,
        "metric": metric,
        "value": rendered,
        "unit": unit,
        "status": status,
        "source_note": note,
    }


def _compile_resource_rows(
    recipe: FinalAccountingRecipe,
    terminals: Sequence[_LedgerSnapshot],
    facts: Mapping[str, Any],
    wall_microseconds: int,
    resource_coverage_gaps: Mapping[str, tuple[str, ...]],
) -> list[dict[str, str]]:
    ledger_note = "terminal immutable ledgers: " + ",".join(
        item.spec.ledger_id for item in terminals
    )
    total_micros = int(facts["total_gpu_microseconds"])
    all_resources = [row for item in terminals for row in item.rows["resource_samples"]]
    all_storage = [row for item in terminals for row in item.rows["storage_samples"]]
    all_calls = [
        row
        for item in terminals
        for row in item.rows["model_calls"]
        if row["backend"] == "vllm_gpu"
    ]
    by_kind: dict[str, int] = defaultdict(int)
    for item in terminals:
        for kind, micros in item.report.gpu_by_kind_microseconds:
            by_kind[kind] += micros
    prompt = sum(int(row["prompt_tokens"]) for row in all_calls)
    completion = sum(int(row["completion_tokens"]) for row in all_calls)
    peak_ram = max([int(row["process_ram_bytes"]) for row in all_resources] + [0])
    peak_vram = max([int(row["gpu_vram_bytes"]) for row in all_resources] + [0])
    peak_storage = max(
        [int(row["project_storage_bytes"]) for row in all_resources]
        + [int(row["current_occupied_bytes"]) for row in all_storage]
        + [0]
    )
    peak_projected = max([int(row["projected_occupied_bytes"]) for row in all_storage] + [0])
    minimum_headroom = min(
        [int(row["effective_projected_headroom_bytes"]) for row in all_storage] or [0]
    )
    maximum_workers = max([int(row["cpu_worker_count"]) for row in all_resources] + [0])
    all_stopped = all(
        item.report.unresolved_gpu_allocation_count == 0
        and item.report.unresolved_gpu_service_count == 0
        for item in terminals
    )
    outcomes = dict(facts["outcome_by_slot"])
    required_slots = [
        item for item in recipe.call_slots if not item.call_class.startswith("reserve_")
    ]
    uncovered_services = sorted(
        f"{ledger_id}:{service_id}"
        for ledger_id, service_ids in resource_coverage_gaps.items()
        for service_id in service_ids
    )
    resource_peak_status = "observed" if not uncovered_services else "observed_lower_bound"
    resource_peak_note = (
        "maximum immutable resource sample"
        if not uncovered_services
        else "maximum immutable resource sample; short pre-sampler launch failure is unsampled"
    )
    rows = [
        _resource_row(
            "study",
            "actual_allocated_gpu_time",
            _seconds(total_micros),
            "seconds",
            ledger_note,
        ),
        _resource_row(
            "study",
            "actual_allocated_gpu_time",
            f"{Decimal(total_micros) / Decimal(3_600_000_000):.9f}",
            "hours",
            ledger_note,
        ),
        _resource_row(
            "study",
            "registered_inference_slots",
            len(recipe.call_slots),
            "count",
            "frozen GPU call inventory",
        ),
        _resource_row(
            "study",
            "effective_service_slots",
            len(recipe.service_slots),
            "count",
            "base inventory plus authorized amendments",
        ),
        _resource_row(
            "study",
            "started_inference_slots",
            sum(item.executed for item in recipe.call_slots),
            "count",
            ledger_note,
        ),
        _resource_row(
            "study",
            "successful_inference_slots",
            sum(value == "success" for value in outcomes.values()),
            "count",
            ledger_note,
        ),
        _resource_row(
            "study",
            "repaired_inference_slots",
            sum(value == "repaired" for value in outcomes.values()),
            "count",
            ledger_note,
        ),
        _resource_row(
            "study",
            "failed_or_invalid_inference_slots",
            sum(
                value in {"failed", "invalid", "timed_out", "interrupted"}
                for value in outcomes.values()
            ),
            "count",
            ledger_note,
        ),
        _resource_row(
            "study",
            "incomplete_mandatory_inference_slots",
            sum(outcomes[item.slot_id] == "incomplete" for item in required_slots),
            "count",
            "exact non-reserve slot inventory",
        ),
        _resource_row(
            "study",
            "itt_registered_slots",
            sum(item.included_in_itt for item in recipe.call_slots),
            "count",
            "immutable result inventory receipts",
        ),
        _resource_row(
            "study",
            "prompt_tokens",
            prompt,
            "tokens",
            "terminal ledger model-call rows plus token receipts",
        ),
        _resource_row(
            "study",
            "completion_tokens",
            completion,
            "tokens",
            "terminal ledger model-call rows plus token receipts",
        ),
        _resource_row(
            "study",
            "total_tokens",
            prompt + completion,
            "tokens",
            "terminal ledger model-call rows plus token receipts",
        ),
        _resource_row(
            "study",
            "peak_gpu_vram",
            peak_vram,
            "bytes",
            resource_peak_note,
            status=resource_peak_status,
        ),
        _resource_row(
            "study",
            "peak_process_ram",
            peak_ram,
            "bytes",
            resource_peak_note,
            status=resource_peak_status,
        ),
        _resource_row(
            "study",
            "peak_project_storage",
            peak_storage,
            "bytes",
            "resource and storage receipts",
        ),
        _resource_row(
            "study",
            "peak_projected_storage",
            peak_projected,
            "bytes",
            "maximum immutable storage preflight projection",
        ),
        _resource_row(
            "study",
            "minimum_effective_storage_headroom",
            minimum_headroom,
            "bytes",
            "minimum immutable storage preflight headroom",
        ),
        _resource_row(
            "study",
            "maximum_cpu_study_workers",
            maximum_workers,
            "count",
            resource_peak_note,
            status=resource_peak_status,
        ),
        _resource_row(
            "study",
            "resource_sampling_coverage_complete",
            not uncovered_services,
            "boolean",
            "one sample inside every allocated service except enumerated sub-second "
            "pre-sampler failures",
            status="compliant" if not uncovered_services else "incomplete_evidence",
        ),
        _resource_row(
            "study",
            "uncovered_short_failed_gpu_services",
            len(uncovered_services),
            "count",
            "hash-bound service IDs are enumerated in the final compilation receipt",
            status="observed" if not uncovered_services else "incomplete_evidence",
        ),
        _resource_row(
            "study",
            "all_storage_samples_allowed",
            all(bool(row["allowed"]) for row in all_storage),
            "boolean",
            "immutable storage receipts",
        ),
        _resource_row(
            "study",
            "all_gpu_services_stopped",
            all_stopped,
            "boolean",
            "terminal allocation and service journals plus service receipts",
        ),
        _resource_row(
            "study",
            "scheduled_gpu_ceiling_compliant",
            total_micros <= REGISTERED_SCHEDULED_SECONDS * 1_000_000,
            "boolean",
            "registered nine-hour scheduled ceiling",
        ),
        _resource_row(
            "study",
            "hard_gpu_stop_compliant",
            total_micros < REGISTERED_HARD_SECONDS * 1_000_000,
            "boolean",
            "strict ten-hour hard stop",
        ),
        _resource_row(
            "study",
            "runpod_container_wall_time_lower_bound",
            _seconds(wall_microseconds),
            "seconds",
            "sum of latest PID-1 lifetime samples per distinct container; not billing time",
            status="observed_lower_bound",
        ),
        _resource_row(
            "study",
            "runpod_container_wall_time_lower_bound",
            f"{Decimal(wall_microseconds) / Decimal(3_600_000_000):.9f}",
            "hours",
            "separate from scientific GPU allocation",
            status="observed_lower_bound",
        ),
    ]
    for kind, micros in sorted(by_kind.items()):
        rows.append(
            _resource_row(
                "gpu_event_kind",
                kind,
                _seconds(micros),
                "seconds",
                "ledger GPU summary; service_overhead is session duration minus classified events",
            )
        )
    allocation = dict(facts["allocation_by_slot"])
    prompts = dict(facts["prompt_by_slot"])
    completions = dict(facts["completion_by_slot"])
    contexts = dict(facts["contexts_by_slot"])
    groups: dict[tuple[str, str], list[RegisteredCallSlot]] = defaultdict(list)
    for slot in recipe.call_slots:
        for group in (("phase", slot.phase), ("condition", slot.condition), ("run", slot.run_id)):
            groups[group].append(slot)
    for (kind, name), slots in sorted(groups.items()):
        scope = f"{kind}:{name}"
        note = "sum over exact registered slots in this scope"
        rows.extend(
            [
                _resource_row(
                    scope,
                    "allocated_gpu_time",
                    _seconds(sum(allocation[item.slot_id] for item in slots)),
                    "seconds",
                    note,
                ),
                _resource_row(
                    scope,
                    "prompt_tokens",
                    sum(prompts[item.slot_id] for item in slots),
                    "tokens",
                    note,
                ),
                _resource_row(
                    scope,
                    "completion_tokens",
                    sum(completions[item.slot_id] for item in slots),
                    "tokens",
                    note,
                ),
                _resource_row(
                    scope,
                    "served_contexts",
                    sum(contexts[item.slot_id] for item in slots),
                    "count",
                    note,
                ),
                _resource_row(scope, "registered_slots", len(slots), "count", note),
            ]
        )
    rows.sort(key=lambda row: (row["scope"], row["metric"], row["unit"]))
    keys = [(row["scope"], row["metric"], row["unit"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise FinalAccountingError("resource table contains duplicate scope/metric/unit rows")
    return rows


def _csv_bytes(columns: Sequence[str], rows: Sequence[Mapping[str, str]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _recover_publish_partial(path: Path) -> None:
    """Remove only an unambiguous same-inode partial left after link publication."""

    if path.is_symlink() or not path.exists():
        return
    metadata = path.stat()
    if metadata.st_nlink == 1:
        return
    prefix = f".{path.name}."
    candidates = []
    for candidate in path.parent.iterdir():
        name = candidate.name
        token = (
            name[len(prefix) : -len(".partial")]
            if (name.startswith(prefix) and name.endswith(".partial"))
            else ""
        )
        if len(token) != 32 or any(character not in "0123456789abcdef" for character in token):
            continue
        try:
            candidate_metadata = candidate.lstat()
        except OSError:
            continue
        if (
            stat.S_ISREG(candidate_metadata.st_mode)
            and candidate_metadata.st_dev == metadata.st_dev
            and candidate_metadata.st_ino == metadata.st_ino
        ):
            candidates.append(candidate)
    if len(candidates) != 1 or metadata.st_nlink != 2:
        raise FinalAccountingError("immutable output has unexplained hardlink aliases")
    candidates[0].unlink()


def _publish_no_replace(path: Path, payload: bytes, *, verify_only: bool) -> None:
    if path.exists() or path.is_symlink():
        if not verify_only:
            _recover_publish_partial(path)
        _assert_real_file(path, label="existing final accounting output")
        if path.read_bytes() != payload:
            raise FinalAccountingError(f"immutable output collision: {path.name}")
        return
    if verify_only:
        raise FinalAccountingError(f"missing immutable output: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
    temporary = path.parent / f".{path.name}.{secrets.token_hex(16)}.partial"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fchmod(stream.fileno(), 0o644)
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as exc:
            _recover_publish_partial(path)
            _assert_real_file(path, label="existing final accounting output")
            if path.read_bytes() != payload:
                raise FinalAccountingError(f"immutable output collision: {path.name}") from exc
    except BaseException:
        with suppress(OSError):
            temporary.unlink()
        raise
    finally:
        os.close(descriptor)
        with suppress(OSError):
            temporary.unlink()
    directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    _assert_real_file(path, label="published final accounting output")
    if path.read_bytes() != payload:
        raise FinalAccountingError("published final accounting output changed")


@dataclass(frozen=True)
class _MaterializedPlan:
    call_slots: tuple[RegisteredCallSlot, ...]
    service_slots: tuple[RegisteredServiceSlot, ...]


@dataclass(frozen=True)
class _ObservedAttempt:
    ledger_id: str
    job_id: str
    attempt_id: str
    model_call_id: str | None
    gpu_event_ids: tuple[str, ...]
    call_class: str
    condition: str
    phase: str
    run_id: str
    call_id: str
    created_at: datetime
    parent_attempt_id: str | None


def _source_route_reference(
    root: Path,
    route: SourceFileRoute,
    *,
    label: str,
) -> tuple[FrozenFileReference, Mapping[str, Any]]:
    path = _safe_input(root, route.relative_path, label=label)
    reference = FrozenFileReference(
        artifact_id=route.artifact_id,
        relative_path=route.relative_path,
        file_sha256=_file_sha256(path),
        self_hash_field=route.self_hash_field,
    )
    _path, payload = _load_reference(root, reference, label=label)
    return reference, payload


def _json_object(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, str):
        raise FinalAccountingError(f"{label} is not encoded JSON")
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise FinalAccountingError(f"{label} is invalid JSON") from exc
    if not isinstance(payload, Mapping):
        raise FinalAccountingError(f"{label} is not a JSON object")
    return payload


def _walk_json(value: object) -> Sequence[object]:
    pending = [value]
    seen: list[object] = []
    while pending:
        current = pending.pop()
        seen.append(current)
        if isinstance(current, Mapping):
            pending.extend(current.values())
        elif isinstance(current, (list, tuple)):
            pending.extend(current)
    return seen


_FALLBACK_CALL_PLAN: tuple[tuple[str, str, str, str, int], ...] = (
    ("fallback-c1-01", "acceptance_c1", "c1_llm_pre", "reserve_long", 240),
    ("fallback-c2-01", "acceptance_c2", "c2_llm_query", "reserve_standard", 150),
    ("fallback-c2-02", "acceptance_c2", "c2_llm_query", "reserve_standard", 150),
    (
        "fallback-fixed-01",
        "acceptance_fixed_select",
        "a_fixed_select",
        "reserve_short",
        90,
    ),
)
_FALLBACK_CALL_BY_ID = {row[0]: row for row in _FALLBACK_CALL_PLAN}
_FALLBACK_CALL_STATUSES = frozenset({"completed", "failed", "resumed"})


def _fallback_base_call_id(call_id: str) -> str | None:
    if call_id in _FALLBACK_CALL_BY_ID:
        return call_id
    suffix = "-repair-01"
    if call_id.endswith(suffix):
        candidate = call_id[: -len(suffix)]
        if candidate in _FALLBACK_CALL_BY_ID:
            return candidate
    return None


def _validate_fallback_result_payload(payload: Mapping[str, Any]) -> None:
    """Validate the exact four-call fallback plan and its optional repair."""

    calls = payload.get("calls")
    reserve = payload.get("reserve_consumption")
    if not isinstance(calls, list) or not calls or not isinstance(reserve, list):
        raise FinalAccountingError("fallback Phase 1 result lacks its call inventory")

    expected_base_ids = tuple(row[0] for row in _FALLBACK_CALL_PLAN)
    seen_base_ids: list[str] = []
    seen_call_ids: set[str] = set()
    repair_ids: set[str] = set()
    for item in calls:
        if not isinstance(item, Mapping):
            raise FinalAccountingError("fallback Phase 1 call result is not an object")
        call_id = item.get("call_id")
        status = item.get("status")
        if not isinstance(call_id, str) or status not in _FALLBACK_CALL_STATUSES:
            raise FinalAccountingError("fallback Phase 1 call result has invalid identity/status")
        if call_id in seen_call_ids:
            raise FinalAccountingError("fallback Phase 1 call result identity is duplicated")
        seen_call_ids.add(call_id)
        base_call_id = _fallback_base_call_id(call_id)
        if base_call_id is None:
            raise FinalAccountingError("fallback Phase 1 result names an unregistered call")
        if call_id == base_call_id:
            expected_index = len(seen_base_ids)
            if (
                expected_index >= len(expected_base_ids)
                or call_id != expected_base_ids[expected_index]
            ):
                raise FinalAccountingError("fallback Phase 1 base-call order changed")
            seen_base_ids.append(call_id)
        else:
            if not seen_base_ids or base_call_id != seen_base_ids[-1] or repair_ids:
                raise FinalAccountingError("fallback Phase 1 repair lineage is invalid")
            repair_ids.add(call_id)

    if payload.get("base_call_count") not in {None, len(_FALLBACK_CALL_PLAN)}:
        raise FinalAccountingError("fallback Phase 1 base-call count changed")
    completed_count = payload.get("completed_base_call_count")
    if completed_count is not None and (
        isinstance(completed_count, bool)
        or not isinstance(completed_count, int)
        or not 0 <= completed_count <= len(_FALLBACK_CALL_PLAN)
    ):
        raise FinalAccountingError("fallback Phase 1 completed-call count is invalid")
    if isinstance(completed_count, int) and completed_count > len(seen_base_ids):
        raise FinalAccountingError("fallback Phase 1 completed-call count exceeds its results")

    seen_reservation_ids: set[str] = set()
    seen_reserve_call_ids: set[str] = set()
    for item in reserve:
        if not isinstance(item, Mapping):
            raise FinalAccountingError("fallback Phase 1 reserve receipt is not an object")
        call_id = item.get("call_id")
        reservation_id = item.get("reservation_id")
        if not isinstance(call_id, str) or not isinstance(reservation_id, str):
            raise FinalAccountingError("fallback Phase 1 reserve receipt lacks identity")
        base_call_id = _fallback_base_call_id(call_id)
        if base_call_id is None:
            raise FinalAccountingError("fallback Phase 1 reserve names an unregistered call")
        base_plan = _FALLBACK_CALL_BY_ID[base_call_id]
        expected_reserve = "reserve_short" if call_id != base_call_id else base_plan[3]
        expected_watchdog = 90 if call_id != base_call_id else base_plan[4]
        if (
            item.get("reserve_call_class") != expected_reserve
            or item.get("watchdog_seconds") != expected_watchdog
        ):
            raise FinalAccountingError("fallback Phase 1 reserve class/watchdog changed")
        if reservation_id in seen_reservation_ids or call_id in seen_reserve_call_ids:
            raise FinalAccountingError("fallback Phase 1 reserve receipt is duplicated")
        seen_reservation_ids.add(reservation_id)
        seen_reserve_call_ids.add(call_id)

    if not set(seen_base_ids).issubset(seen_reserve_call_ids):
        raise FinalAccountingError("fallback Phase 1 call lacks its reserve receipt")
    if any(
        call_id.endswith("-repair-01") and call_id.removesuffix("-repair-01") not in seen_base_ids
        for call_id in seen_reserve_call_ids
    ):
        raise FinalAccountingError("fallback Phase 1 reserve has no parent call")
    repair_attempt_count = payload.get("repair_attempt_count")
    reserve_repair_count = sum(call_id.endswith("-repair-01") for call_id in seen_reserve_call_ids)
    if repair_attempt_count is not None and repair_attempt_count != reserve_repair_count:
        raise FinalAccountingError("fallback Phase 1 repair count differs from reserves")

    if payload["gate_passed"] is True:
        if tuple(seen_base_ids) != expected_base_ids:
            raise FinalAccountingError("passed fallback Phase 1 result lacks all four base calls")
        if completed_count not in {None, len(_FALLBACK_CALL_PLAN)}:
            raise FinalAccountingError("passed fallback Phase 1 result is not complete")
        if not isinstance(payload.get("development_execution_result"), Mapping):
            raise FinalAccountingError(
                "passed fallback Phase 1 result lacks its development execution result"
            )


def _fallback_index_payload(payload: Mapping[str, Any]) -> list[Mapping[str, object]]:
    """Return only producer-validated identities and registered semantics."""

    indexed: list[Mapping[str, object]] = [{"run_id": payload["run_id"]}]
    call_rows = {
        str(item["call_id"]): item for item in payload["calls"] if isinstance(item, Mapping)
    }
    reserve_rows = {
        str(item["call_id"]): item
        for item in payload["reserve_consumption"]
        if isinstance(item, Mapping)
    }
    all_call_ids = set(call_rows) | set(reserve_rows)
    for call_id in sorted(all_call_ids):
        base_call_id = _fallback_base_call_id(call_id)
        assert base_call_id is not None
        _base_id, call_class, condition, base_reserve, _watchdog = _FALLBACK_CALL_BY_ID[
            base_call_id
        ]
        source = {**reserve_rows.get(call_id, {}), **call_rows.get(call_id, {})}
        identity: dict[str, object] = {
            "call_id": call_id,
            "call_class": call_class,
            "condition": condition,
        }
        for field in (
            "attempt_id",
            "event_id",
            "gpu_event_id",
            "job_id",
            "model_call_id",
            "request_id",
            "reservation_id",
            "reserve_reservation_id",
        ):
            value = source.get(field)
            if isinstance(value, str) and value:
                identity[field] = value
        if call_id in reserve_rows:
            identity["repair_reserve_class"] = (
                "reserve_short" if call_id != base_call_id else base_reserve
            )
        indexed.append(identity)
    return indexed


def _validate_native_source_payload(
    route: NativeSourceRoute,
    payload: Mapping[str, Any],
) -> None:
    """Reject operator-authored generic JSON as native scientific provenance."""

    role = route.producer_role
    if role is NativeSourceRole.PHASE1_ACCEPTANCE_RESULT:
        if route.self_hash_field is not SelfHashField.MANIFEST:
            raise FinalAccountingError("Phase 1 acceptance results require manifest_sha256")
        kind = payload.get("kind")
        if kind not in {
            "phase1_gpu_acceptance_result",
            "phase1_fallback_micro_pilot_result",
        }:
            raise FinalAccountingError("Phase 1 native source has an unrecognized producer kind")
        if (
            payload.get("schema_version") != FINAL_ACCOUNTING_SCHEMA_VERSION
            or not isinstance(payload.get("run_id"), str)
            or not isinstance(payload.get("gate_passed"), bool)
        ):
            raise FinalAccountingError("Phase 1 native result violates its producer contract")
        if kind == "phase1_fallback_micro_pilot_result":
            _validate_fallback_result_payload(payload)
        return

    if route.self_hash_field is not SelfHashField.CONTENT:
        raise FinalAccountingError(f"native producer {role.value} requires content_hash")
    # Imports remain local so accounting of an incomplete Phase 1 run does not load
    # the later-phase controller modules merely to validate its sources.
    if role is NativeSourceRole.DEVELOPMENT_CALL_MANIFEST:
        from story_projection_onto.development_runtime import DevelopmentCallManifest

        model: type[BaseModel] = DevelopmentCallManifest
    elif role is NativeSourceRole.HELD_OUT_CALL_MANIFEST:
        from story_projection_onto.held_out_primary import HeldOutCallManifest

        model = HeldOutCallManifest
    elif role is NativeSourceRole.HELD_OUT_EXECUTION_MANIFEST:
        from story_projection_onto.held_out_controller import HeldOutExecutionManifest

        model = HeldOutExecutionManifest
    elif role is NativeSourceRole.COMBINED_CALL_MANIFEST:
        from story_projection_onto.combined_gpu_block import CombinedCallManifest

        model = CombinedCallManifest
    elif role is NativeSourceRole.FEEDBACK_EXECUTION_MANIFEST:
        from story_projection_onto.feedback_runtime import (
            FeedbackStudyExecutionManifest,
        )

        model = FeedbackStudyExecutionManifest
    elif role is NativeSourceRole.CASE_EXECUTION_PLAN:
        from story_projection_onto.case_study_runtime import CaseStudyExecutionPlan

        model = CaseStudyExecutionPlan
    else:
        from story_projection_onto.case_study_execution import CaseStudyExecutionResult

        model = CaseStudyExecutionResult
    try:
        model.model_validate(payload)
    except Exception as exc:
        raise FinalAccountingError(
            f"native producer {role.value} violates its exact typed contract"
        ) from exc


def _required_schedule_role(call_class: str) -> NativeSourceRole | None:
    if call_class.startswith("acceptance_"):
        return NativeSourceRole.PHASE1_ACCEPTANCE_RESULT
    if call_class.startswith("development_"):
        return NativeSourceRole.DEVELOPMENT_CALL_MANIFEST
    if call_class.startswith("test_"):
        return NativeSourceRole.HELD_OUT_CALL_MANIFEST
    if call_class.startswith(("paraphrase_", "scripted_", "researcher_", "ablation_")):
        return NativeSourceRole.COMBINED_CALL_MANIFEST
    if call_class.startswith("case_"):
        return NativeSourceRole.CASE_EXECUTION_PLAN
    return None


def _association_roles(call_class: str, phase: str) -> frozenset[NativeSourceRole]:
    schedule = _required_schedule_role(call_class)
    if schedule is not None:
        return frozenset({schedule})
    if not call_class.startswith("reserve_"):
        return frozenset()
    return {
        "phase_1": frozenset(
            {
                NativeSourceRole.PHASE1_ACCEPTANCE_RESULT,
                NativeSourceRole.DEVELOPMENT_CALL_MANIFEST,
            }
        ),
        "phase_3": frozenset(
            {
                NativeSourceRole.HELD_OUT_CALL_MANIFEST,
                NativeSourceRole.HELD_OUT_EXECUTION_MANIFEST,
                NativeSourceRole.COMBINED_CALL_MANIFEST,
            }
        ),
        "phase_5": frozenset(
            {
                NativeSourceRole.COMBINED_CALL_MANIFEST,
                NativeSourceRole.FEEDBACK_EXECUTION_MANIFEST,
            }
        ),
        "phase_6": frozenset(
            {
                NativeSourceRole.CASE_EXECUTION_PLAN,
                NativeSourceRole.CASE_EXECUTION_RESULT,
            }
        ),
    }.get(phase, frozenset())


def _native_source_index(
    sources: Sequence[tuple[NativeSourceRoute, Mapping[str, Any]]],
) -> tuple[
    Mapping[NativeSourceRole, frozenset[str]],
    Mapping[str, frozenset[str]],
    Mapping[str, frozenset[str]],
    Mapping[str, frozenset[str]],
]:
    strings_by_role: dict[NativeSourceRole, set[str]] = defaultdict(set)
    class_by_identifier: dict[str, set[str]] = defaultdict(set)
    repair_reserve_by_identifier: dict[str, set[str]] = defaultdict(set)
    condition_by_identifier: dict[str, set[str]] = defaultdict(set)
    identifier_fields = {
        "attempt_id",
        "call_id",
        "event_id",
        "gpu_event_id",
        "job_id",
        "model_call_id",
        "request_id",
        "reservation_id",
        "reserve_reservation_id",
    }
    registered = set(REGISTERED_CALL_CLASSES) - {SESSION_START_CLASS}
    for route, payload in sources:
        _validate_native_source_payload(route, payload)
        indexed_payload: object = payload
        if route.producer_role is NativeSourceRole.PHASE1_ACCEPTANCE_RESULT:
            if payload.get("kind") == "phase1_fallback_micro_pilot_result":
                indexed_payload = _fallback_index_payload(payload)
            else:
                indexed_payload = {"run_id": payload["run_id"]}
        for value in _walk_json(indexed_payload):
            if isinstance(value, str):
                strings_by_role[route.producer_role].add(value)
                continue
            if not isinstance(value, Mapping):
                continue
            raw_class = value.get("reserve_call_class", value.get("call_class"))
            repair_reserve = value.get("repair_reserve_class")
            condition_candidates: set[str] = set()
            explicit_condition = value.get("condition")
            if isinstance(explicit_condition, str):
                normalized = _normalize_condition(explicit_condition)
                if normalized is not None:
                    condition_candidates.add(normalized)
            for field in ("call_class", "forecast_proxy_call_class"):
                underlying_class = value.get(field)
                if (
                    isinstance(underlying_class, str)
                    and underlying_class in registered
                    and not underlying_class.startswith("reserve_")
                ):
                    condition_candidates.add(_condition_for_call(underlying_class))
            for field in identifier_fields:
                identifier = value.get(field)
                if isinstance(identifier, str) and identifier:
                    if isinstance(raw_class, str) and raw_class in registered:
                        class_by_identifier[identifier].add(raw_class)
                    if (
                        isinstance(repair_reserve, str)
                        and repair_reserve.startswith("reserve_")
                        and repair_reserve in registered
                    ):
                        repair_reserve_by_identifier[identifier].add(repair_reserve)
                    condition_by_identifier[identifier].update(condition_candidates)
    return (
        {role: frozenset(strings) for role, strings in strings_by_role.items()},
        {identifier: frozenset(classes) for identifier, classes in class_by_identifier.items()},
        {
            identifier: frozenset(classes)
            for identifier, classes in repair_reserve_by_identifier.items()
        },
        {
            identifier: frozenset(conditions)
            for identifier, conditions in condition_by_identifier.items()
        },
    )


def _attempt_identifiers(
    *,
    attempt: Mapping[str, Any],
    identity: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    model_calls: Sequence[Mapping[str, Any]],
) -> frozenset[str]:
    values: set[str] = {
        str(attempt["attempt_id"]),
        str(attempt["job_id"]),
    }
    for field in ("call_id", "run", "run_id", "execution_id", "study_id", "manifest_id"):
        value = identity.get(field)
        if isinstance(value, str) and value:
            values.add(value)
    for row in events:
        values.add(str(row["event_id"]))
        details = _json_object(row["details_json"], label="GPU-event details")
        for field in (
            "call_id",
            "request_id",
            "reservation_id",
            "reserve_reservation_id",
        ):
            value = details.get(field)
            if isinstance(value, str) and value:
                values.add(value)
    for row in model_calls:
        values.add(str(row["model_call_id"]))
        for field in ("request_hash", "response_artifact_hash"):
            value = row[field]
            if isinstance(value, str) and value:
                values.add(value)
    return frozenset(values)


def _attempt_call_class(
    *,
    attempt_id: str,
    events: Sequence[Mapping[str, Any]],
    identifiers: frozenset[str],
    attempt_kind: str,
    native_classes: Mapping[str, frozenset[str]],
    native_repair_reserves: Mapping[str, frozenset[str]],
) -> str | None:
    registered = set(REGISTERED_CALL_CLASSES) - {SESSION_START_CLASS}
    direct: set[str] = set()
    direct_reserves: set[str] = set()
    for event in events:
        details = _json_object(event["details_json"], label=f"event for {attempt_id}")
        for field in ("reserve_call_class", "call_class"):
            value = details.get(field)
            if isinstance(value, str):
                if value not in registered:
                    raise FinalAccountingError(
                        f"attempt {attempt_id} names unregistered call class {value!r}"
                    )
                if field == "reserve_call_class":
                    direct_reserves.add(value)
                else:
                    direct.add(value)
    if len(direct_reserves) > 1:
        raise FinalAccountingError(f"attempt {attempt_id} has conflicting reserve classes")
    if direct_reserves:
        return next(iter(direct_reserves))
    if attempt_kind != "base":
        repair_reserves: set[str] = set()
        for identifier in identifiers:
            repair_reserves.update(native_repair_reserves.get(identifier, ()))
        if len(repair_reserves) > 1:
            raise FinalAccountingError(
                f"attempt {attempt_id} has ambiguous repair-reserve association"
            )
        if repair_reserves:
            return next(iter(repair_reserves))
    if len(direct) > 1:
        raise FinalAccountingError(f"attempt {attempt_id} has conflicting event call classes")
    if direct:
        return next(iter(direct))
    inferred: set[str] = set()
    for identifier in identifiers:
        inferred.update(native_classes.get(identifier, ()))
    if len(inferred) > 1:
        raise FinalAccountingError(
            f"attempt {attempt_id} has ambiguous native call-class association"
        )
    return next(iter(inferred)) if inferred else None


def _phase_for_call(
    call_class: str,
    event_details: Sequence[Mapping[str, Any]],
    *,
    identifiers: frozenset[str] = frozenset(),
    native_classes: Mapping[str, frozenset[str]] | None = None,
) -> str:
    """Map calls to phases using registered classes, never free-text substrings."""

    if call_class.startswith(("acceptance_", "development_")):
        return "phase_1"
    if call_class.startswith(("scripted_", "researcher_")):
        return "phase_5"
    if call_class.startswith("case_"):
        return "phase_6"
    if not call_class.startswith("reserve_"):
        return "phase_3"

    # A reserve attempt inherits the phase of its exact underlying registered
    # call.  Runtime event fields take precedence only by contributing another
    # exact candidate; the native producer association is independently bound
    # by identifier.  A label such as ``fallback_test`` is deliberately inert.
    underlying_classes: set[str] = set()
    explicit_phases: set[str] = set()
    for details in event_details:
        phase = details.get("phase")
        if isinstance(phase, str):
            if phase not in {f"phase_{ordinal}" for ordinal in range(1, 8)}:
                raise FinalAccountingError(f"reserve call names invalid phase {phase!r}")
            explicit_phases.add(phase)
        for field in ("call_class", "forecast_proxy_call_class"):
            value = details.get(field)
            if not isinstance(value, str):
                continue
            if value not in REGISTERED_CALL_CLASSES or value == SESSION_START_CLASS:
                raise FinalAccountingError(
                    f"reserve call names unregistered underlying class {value!r}"
                )
            if not value.startswith("reserve_"):
                underlying_classes.add(value)
    for identifier in identifiers:
        for value in (native_classes or {}).get(identifier, ()):
            if not value.startswith("reserve_"):
                underlying_classes.add(value)
    inferred_phases = {_phase_for_call(value, ()) for value in underlying_classes}
    candidates = explicit_phases | inferred_phases
    if len(candidates) > 1:
        raise FinalAccountingError("reserve call has conflicting exact phase associations")
    if candidates:
        return next(iter(candidates))
    if event_details or identifiers:
        raise FinalAccountingError("reserve call lacks an exact underlying phase association")
    # Unstarted registered reserve slots are retained in the Phase-1 planning
    # inventory. Executed attempts always take the fail-closed branch above.
    return "phase_1"


def _condition_for_call(call_class: str) -> str:
    if "fixed_select" in call_class:
        return "a_fixed_select"
    if "c1" in call_class:
        return "c1_llm_pre"
    if "c2" in call_class:
        return "c2_llm_query"
    if call_class == "ablation_no_context":
        return "a_no_context"
    if call_class == "ablation_no_temporal_epistemic":
        return "a_no_temporal_epistemic"
    if call_class == "ablation_no_rare_guard":
        return "a_no_rare_guard"
    if call_class.startswith("reserve_"):
        return "reserve_repair"
    return "study_control"


def _normalize_condition(value: str) -> str | None:
    normalized = value.casefold().replace("-", "_").replace(" ", "_")
    return {
        "c0": "c0_classical_pre",
        "c0_classical_pre": "c0_classical_pre",
        "c1": "c1_llm_pre",
        "c1_llm_pre": "c1_llm_pre",
        "c2": "c2_llm_query",
        "c2_llm_query": "c2_llm_query",
        "a_fixedselect": "a_fixed_select",
        "a_fixed_select": "a_fixed_select",
        "a_nocontext": "a_no_context",
        "a_no_context": "a_no_context",
        "a_notemporalepistemic": "a_no_temporal_epistemic",
        "a_no_temporal_epistemic": "a_no_temporal_epistemic",
        "a_norareguard": "a_no_rare_guard",
        "a_no_rare_guard": "a_no_rare_guard",
    }.get(normalized)


def _attempt_condition(
    *,
    attempt_id: str,
    call_class: str,
    events: Sequence[Mapping[str, Any]],
    identifiers: frozenset[str],
    native_conditions: Mapping[str, frozenset[str]],
) -> str:
    if not call_class.startswith("reserve_"):
        return _condition_for_call(call_class)
    candidates: set[str] = set()
    for event in events:
        details = _json_object(event["details_json"], label=f"event for {attempt_id}")
        condition = details.get("condition")
        if isinstance(condition, str):
            normalized = _normalize_condition(condition)
            if normalized is not None:
                candidates.add(normalized)
        underlying_class = details.get("call_class")
        if (
            isinstance(underlying_class, str)
            and underlying_class in REGISTERED_CALL_CLASSES
            and not underlying_class.startswith("reserve_")
            and underlying_class != SESSION_START_CLASS
        ):
            candidates.add(_condition_for_call(underlying_class))
    for identifier in identifiers:
        candidates.update(native_conditions.get(identifier, ()))
    if len(candidates) != 1:
        state = "absent" if not candidates else "ambiguous"
        raise FinalAccountingError(f"reserve attempt {attempt_id} has {state} underlying condition")
    return next(iter(candidates))


def _included_in_itt(call_class: str) -> bool:
    return call_class.startswith(("test_", "paraphrase_", "ablation_"))


def _first_text(mapping: Mapping[str, Any], fields: Sequence[str]) -> str | None:
    for field in fields:
        value = mapping.get(field)
        if isinstance(value, str) and value:
            return value
    return None


def _observed_attempts(
    terminals: Sequence[_LedgerSnapshot],
    *,
    native_strings_by_role: Mapping[NativeSourceRole, frozenset[str]],
    native_classes: Mapping[str, frozenset[str]],
    native_repair_reserves: Mapping[str, frozenset[str]],
    native_conditions: Mapping[str, frozenset[str]],
) -> tuple[tuple[_ObservedAttempt, ...], tuple[AttemptExclusion, ...]]:
    observed: list[_ObservedAttempt] = []
    exclusions: list[AttemptExclusion] = []
    for snapshot in terminals:
        ledger_id = snapshot.spec.ledger_id
        jobs = _indexed_rows(snapshot, "jobs", "job_id")
        attempts = sorted(
            snapshot.rows["attempts"],
            key=lambda row: (_parse_time(row["created_at"]), str(row["attempt_id"])),
        )
        for attempt in attempts:
            attempt_id = str(attempt["attempt_id"])
            job_id = str(attempt["job_id"])
            job = jobs.get(job_id)
            if job is None:
                raise FinalAccountingError(f"attempt {attempt_id} lost its job")
            identity = _json_object(job["identity_json"], label=f"job {job_id} identity")
            events = tuple(
                row for row in snapshot.rows["gpu_events"] if row["attempt_id"] == attempt_id
            )
            model_calls = tuple(
                row for row in snapshot.rows["model_calls"] if row["attempt_id"] == attempt_id
            )
            vllm_calls = tuple(row for row in model_calls if row["backend"] == "vllm_gpu")
            fixture_calls = tuple(
                row for row in model_calls if row["backend"] == "hand_authored_fixture"
            )
            if fixture_calls:
                if events or vllm_calls or len(fixture_calls) != len(model_calls):
                    raise FinalAccountingError(
                        f"hand-authored attempt {attempt_id} is mixed with GPU evidence"
                    )
                exclusions.append(
                    AttemptExclusion(
                        ledger_id=ledger_id,
                        attempt_id=attempt_id,
                        reason="hand_authored_fixture",
                    )
                )
                continue
            if len(vllm_calls) > 1:
                raise FinalAccountingError(
                    f"attempt {attempt_id} has more than one VLLM model-call row"
                )
            identifiers = _attempt_identifiers(
                attempt=attempt,
                identity=identity,
                events=events,
                model_calls=model_calls,
            )
            call_class = _attempt_call_class(
                attempt_id=attempt_id,
                events=events,
                identifiers=identifiers,
                attempt_kind=str(attempt["attempt_kind"]),
                native_classes=native_classes,
                native_repair_reserves=native_repair_reserves,
            )
            if call_class is None:
                if events or vllm_calls or identity.get("kind") != "cpu_projection":
                    raise FinalAccountingError(
                        f"attempt {attempt_id} has no registered call-class source"
                    )
                exclusions.append(
                    AttemptExclusion(
                        ledger_id=ledger_id,
                        attempt_id=attempt_id,
                        reason="cpu_only_non_gpu",
                    )
                )
                continue
            details = tuple(
                _json_object(row["details_json"], label=f"event for {attempt_id}") for row in events
            )
            condition = _attempt_condition(
                attempt_id=attempt_id,
                call_class=call_class,
                events=events,
                identifiers=identifiers,
                native_conditions=native_conditions,
            )
            phase = _phase_for_call(
                call_class,
                details,
                identifiers=identifiers,
                native_classes=native_classes,
            )
            permitted_roles = _association_roles(call_class, phase)
            if not permitted_roles or not any(
                identifiers & native_strings_by_role.get(role, frozenset())
                for role in permitted_roles
            ):
                raise FinalAccountingError(
                    f"attempt {attempt_id} is absent from its exact native producer family"
                )
            run_id = (
                _first_text(
                    identity,
                    ("execution_id", "run_id", "run", "study_id", "manifest_id"),
                )
                or ledger_id
            )
            event_identity = next(
                (
                    candidate
                    for row in details
                    if (
                        candidate := _first_text(
                            row,
                            (
                                "reserve_reservation_id",
                                "reservation_id",
                                "call_id",
                                "request_id",
                            ),
                        )
                    )
                ),
                None,
            )
            call_id = event_identity or _first_text(identity, ("call_id",)) or attempt_id
            observed.append(
                _ObservedAttempt(
                    ledger_id=ledger_id,
                    job_id=job_id,
                    attempt_id=attempt_id,
                    model_call_id=(str(vllm_calls[0]["model_call_id"]) if vllm_calls else None),
                    gpu_event_ids=tuple(sorted(str(row["event_id"]) for row in events)),
                    call_class=call_class,
                    condition=condition,
                    phase=phase,
                    run_id=run_id,
                    call_id=call_id,
                    created_at=_parse_time(attempt["created_at"]),
                    parent_attempt_id=(
                        None
                        if attempt["parent_attempt_id"] is None
                        else str(attempt["parent_attempt_id"])
                    ),
                )
            )
    return (
        tuple(observed),
        tuple(sorted(exclusions, key=lambda item: (item.ledger_id, item.attempt_id))),
    )


def _registered_call_slots(
    observed: Sequence[_ObservedAttempt],
) -> tuple[RegisteredCallSlot, ...]:
    by_class: dict[str, list[_ObservedAttempt]] = defaultdict(list)
    for item in observed:
        by_class[item.call_class].append(item)
    slots: list[RegisteredCallSlot] = []
    used_table_keys: set[tuple[str, str, str]] = set()
    for call_class, (count, _watchdog) in REGISTERED_CALL_CLASSES.items():
        if call_class == SESSION_START_CLASS:
            continue
        actual = sorted(
            by_class.pop(call_class, ()),
            key=lambda item: (
                item.created_at,
                item.ledger_id,
                item.attempt_id,
            ),
        )
        if len(actual) > count:
            raise FinalAccountingError(
                f"observed {len(actual)} {call_class} attempts exceeds registered {count}"
            )
        for ordinal in range(1, count + 1):
            slot_id = f"{call_class}-{ordinal:03d}"
            item = actual[ordinal - 1] if ordinal <= len(actual) else None
            if item is None:
                phase = _phase_for_call(call_class, ())
                run_id = f"planned-{call_class}"
                call_id = slot_id
                linkage: dict[str, Any] = {}
            else:
                phase = item.phase
                run_id = item.run_id
                call_id = item.call_id
                key = (phase, run_id, call_id)
                if key in used_table_keys:
                    call_id = f"{call_id}--{item.attempt_id}"
                linkage = {
                    "ledger_id": item.ledger_id,
                    "job_id": item.job_id,
                    "attempt_id": item.attempt_id,
                    "model_call_id": item.model_call_id,
                    "gpu_event_ids": item.gpu_event_ids,
                }
            used_table_keys.add((phase, run_id, call_id))
            slots.append(
                RegisteredCallSlot(
                    slot_id=slot_id,
                    call_class=call_class,
                    ordinal=ordinal,
                    phase=phase,
                    run_id=run_id,
                    call_id=call_id,
                    condition=(_condition_for_call(call_class) if item is None else item.condition),
                    included_in_itt=_included_in_itt(call_class),
                    **linkage,
                )
            )
    if by_class:
        raise FinalAccountingError("observed attempts include unknown call-class groups")
    return tuple(slots)


def _amendment_run_id(payload: Mapping[str, Any]) -> str:
    value = payload.get("authorized_recovery_run_id")
    if not isinstance(value, str) or not value:
        raise FinalAccountingError("authorization amendment lacks recovery run identity")
    return value


def _base_service_phase(ordinal: int) -> str:
    """Return the phase fixed by the registered eight-service allocation."""

    if ordinal <= 3:
        return "phase_1"
    if ordinal <= 7:
        return "phase_3"
    if ordinal == 8:
        return "phase_6"
    raise FinalAccountingError("base service ordinal exceeds the registered inventory")


def _service_amendment_claims(
    session: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    authorized_run_ids: frozenset[str],
) -> frozenset[str]:
    claims: set[str] = set()
    session_id = session.get("session_id")
    if isinstance(session_id, str) and session_id in authorized_run_ids:
        claims.add(session_id)
    for row in (session, *events):
        details = _json_object(
            row.get("details_json"),
            label="GPU service amendment details",
        )
        if "authorized_recovery_run_id" not in details:
            continue
        value = details["authorized_recovery_run_id"]
        if not isinstance(value, str) or not value:
            raise FinalAccountingError("service amendment identity is not text")
        if value not in authorized_run_ids:
            raise FinalAccountingError(
                "service session claims an unauthorized recovery run identity"
            )
        claims.add(value)
    return frozenset(claims)


def _service_event_belongs_to_session(
    session: Mapping[str, Any],
    event: Mapping[str, Any],
) -> bool:
    if _parse_time(session["started_at"]) <= _parse_time(event["started_at"]) and _parse_time(
        event["ended_at"]
    ) <= _parse_time(session["ended_at"]):
        return True
    if event.get("event_id") != session.get("service_session_id"):
        return False
    details = _json_object(event.get("details_json"), label="service lifecycle event details")
    return (
        event.get("attempt_id") is None
        and details.get("intended_event_kind") == SESSION_START_CLASS
        and details.get("session_id") == session.get("session_id")
        and event.get("event_kind") in {SESSION_START_CLASS, "failure", "rejected", "timeout"}
        and _parse_time(event["started_at"]) <= _parse_time(session["ended_at"])
        and _parse_time(event["ended_at"]) >= _parse_time(session["started_at"])
    )


def _registered_service_slots(
    terminals: Sequence[_LedgerSnapshot],
    *,
    amendment_payloads: Sequence[tuple[FrozenFileReference, Mapping[str, Any]]],
    inventory_hash: str,
) -> tuple[RegisteredServiceSlot, ...]:
    sessions: list[tuple[_LedgerSnapshot, Mapping[str, Any], tuple[Mapping[str, Any], ...]]] = []
    for snapshot in terminals:
        for session in snapshot.rows["gpu_service_sessions"]:
            contained = tuple(
                event
                for event in snapshot.rows["gpu_events"]
                if _service_event_belongs_to_session(session, event)
            )
            sessions.append((snapshot, session, contained))
    sessions.sort(
        key=lambda item: (
            _parse_time(item[1]["started_at"]),
            item[0].spec.ledger_id,
            str(item[1]["service_session_id"]),
        )
    )
    amendment_capacities: list[tuple[FrozenFileReference, Mapping[str, Any], int]] = []
    for reference, payload in amendment_payloads:
        amendment_capacities.append(
            (
                reference,
                payload,
                _amendment_additional_services(payload, inventory_hash),
            )
        )
    amendment_run_ids = [_amendment_run_id(item[1]) for item in amendment_capacities]
    if len(amendment_run_ids) != len(set(amendment_run_ids)):
        raise FinalAccountingError("authorization amendments reuse one recovery run identity")
    matched_by_amendment: dict[str, list[Any]] = {
        item[0].artifact_id: [] for item in amendment_capacities
    }
    base_sessions: list[Any] = []
    authorized_run_ids = frozenset(amendment_run_ids)
    for observed in sessions:
        _snapshot, session, events = observed
        matches = sorted(_service_amendment_claims(session, events, authorized_run_ids))
        if len(matches) > 1:
            raise FinalAccountingError(
                "one service session matches multiple authorization amendments"
            )
        if not matches:
            base_sessions.append(observed)
            continue
        matched_index = amendment_run_ids.index(matches[0])
        matched_reference = amendment_capacities[matched_index][0]
        matched_by_amendment[matched_reference.artifact_id].append(observed)
    if len(base_sessions) > REGISTERED_SESSION_STARTS:
        raise FinalAccountingError("observed base service sessions exceed registered eight")
    for reference, _payload, additional in amendment_capacities:
        if len(matched_by_amendment[reference.artifact_id]) > additional:
            raise FinalAccountingError(
                f"service sessions exceed authorization {reference.artifact_id}"
            )

    slots: list[RegisteredServiceSlot] = []
    for ordinal in range(1, REGISTERED_SESSION_STARTS + 1):
        observed = base_sessions[ordinal - 1] if ordinal <= len(base_sessions) else None
        if observed is None:
            slots.append(
                RegisteredServiceSlot(
                    slot_id=f"service-base-{ordinal:03d}",
                    source="base_inventory",
                    ordinal=ordinal,
                    phase="phase_1",
                    run_id=f"planned-service-base-{ordinal:03d}",
                    call_id=f"service-base-{ordinal:03d}",
                )
            )
            continue
        snapshot, session, events = observed
        service_id = str(session["service_session_id"])
        slots.append(
            RegisteredServiceSlot(
                slot_id=f"service-base-{ordinal:03d}",
                source="base_inventory",
                ordinal=ordinal,
                phase=_base_service_phase(ordinal),
                run_id=str(session["session_id"]),
                call_id=service_id,
                ledger_id=snapshot.spec.ledger_id,
                service_session_id=service_id,
                gpu_event_ids=tuple(
                    sorted(str(row["event_id"]) for row in events if row["attempt_id"] is None)
                ),
            )
        )

    for reference, payload, additional in amendment_capacities:
        run_id = _amendment_run_id(payload)
        amendment_sessions = matched_by_amendment[reference.artifact_id]
        for ordinal in range(1, additional + 1):
            observed = (
                amendment_sessions[ordinal - 1] if ordinal <= len(amendment_sessions) else None
            )
            amendment_token = canonical_sha256(reference.artifact_id)[:16]
            slot_id = f"service-amendment-{amendment_token}-{ordinal:03d}"
            if observed is None:
                slots.append(
                    RegisteredServiceSlot(
                        slot_id=slot_id,
                        source="authorized_amendment",
                        ordinal=ordinal,
                        phase="phase_1",
                        run_id=run_id,
                        call_id=slot_id,
                        amendment_artifact_id=reference.artifact_id,
                    )
                )
                continue
            snapshot, session, events = observed
            service_id = str(session["service_session_id"])
            slots.append(
                RegisteredServiceSlot(
                    slot_id=slot_id,
                    source="authorized_amendment",
                    ordinal=ordinal,
                    phase="phase_1",
                    run_id=run_id,
                    call_id=service_id,
                    amendment_artifact_id=reference.artifact_id,
                    ledger_id=snapshot.spec.ledger_id,
                    service_session_id=service_id,
                    gpu_event_ids=tuple(
                        sorted(str(row["event_id"]) for row in events if row["attempt_id"] is None)
                    ),
                )
            )
    return tuple(slots)


def _materialization_destination(root: Path, output_root: Path, *, verify_only: bool) -> Path:
    destination = Path(os.path.abspath(output_root))
    try:
        relative = destination.relative_to(root)
    except ValueError as exc:
        raise FinalAccountingError(
            "materialized accounting must remain inside source root"
        ) from exc
    if relative == Path("."):
        raise FinalAccountingError("materialized accounting requires a child output directory")
    probe = root
    for component in relative.parts:
        probe /= component
        if probe.is_symlink():
            raise FinalAccountingError("materialization output path contains a symbolic link")
    if verify_only:
        return _real_root(destination, label="accounting materialization output root")
    existing = destination
    while not existing.exists() and existing.parent != existing:
        existing = existing.parent
    _real_root(existing, label="accounting materialization output ancestor")
    destination.mkdir(parents=True, exist_ok=True)
    return _real_root(destination, label="accounting materialization output root")


def _adapter_payload(
    payload: Mapping[str, Any],
    *,
    source_recipe_hash: str,
    native_hashes: Sequence[str],
    ledger_hashes: Sequence[str],
    amendment_hashes: Sequence[str],
    inventory_hash: str,
) -> dict[str, Any]:
    result = {
        "schema_version": FINAL_ACCOUNTING_SCHEMA_VERSION,
        "generated_by": "story_projection_onto.final_accounting.materializer/v1",
        "source_materialization_recipe_sha256": source_recipe_hash,
        "source_native_file_sha256": sorted(set(native_hashes)),
        "source_ledger_file_sha256": sorted(set(ledger_hashes)),
        "source_authorization_file_sha256": sorted(set(amendment_hashes)),
        "base_call_inventory_file_sha256": inventory_hash,
        **payload,
    }
    result["manifest_sha256"] = canonical_sha256(result)
    return result


def _adapter_receipt_bytes(
    *,
    artifact_id: str,
    receipt_kind: ReceiptKind,
    payload: Mapping[str, Any],
    destination: Path,
    root: Path,
) -> tuple[ReceiptReference, Path, bytes]:
    encoded = _json_bytes(payload)
    manifest_hash = str(payload["manifest_sha256"])
    path = destination / f"{artifact_id}.{manifest_hash[:16]}.json"
    reference = ReceiptReference(
        artifact_id=artifact_id,
        relative_path=path.relative_to(root).as_posix(),
        file_sha256=hashlib.sha256(encoded).hexdigest(),
        self_hash_field=SelfHashField.MANIFEST,
        receipt_kind=receipt_kind,
    )
    return reference, path, encoded


def build_final_accounting_source_recipe(
    *,
    accounting_id: str,
    source_root: Path,
    output_root: Path,
    base_call_inventory: SourceFileRoute,
    native_source_artifacts: Sequence[NativeSourceRoute],
    authorization_amendments: Sequence[SourceFileRoute],
    ledger_sources: Sequence[LedgerSourceRoute],
    wall_time_receipts: Sequence[SourceFileRoute],
    verify_only: bool = False,
) -> FinalAccountingSourceBuildOutputs:
    """Build a routing recipe solely from explicitly named immutable sources.

    The freeze time is the latest supplied wall-time sample.  Consequently a stale
    sample cannot be papered over with an operator-entered timestamp: it fails if it
    predates any terminal ledger evidence.
    """

    root = _real_root(source_root, label="final accounting source root")
    inventory_reference, _inventory_payload = _source_route_reference(
        root,
        base_call_inventory,
        label="registered GPU call inventory",
    )
    try:
        GPUCallInventory.load(
            _safe_input(
                root,
                inventory_reference.relative_path,
                label="registered GPU call inventory",
            ),
            enforce_registered_plan=True,
        )
    except Exception as exc:
        raise FinalAccountingError(f"registered GPU call inventory is invalid: {exc}") from exc

    native_payloads: list[tuple[NativeSourceRoute, Mapping[str, Any]]] = []
    for route in native_source_artifacts:
        _reference, payload = _source_route_reference(
            root,
            route,
            label=f"native phase source {route.artifact_id}",
        )
        native_payloads.append((route, payload))
    (
        native_strings_by_role,
        native_classes,
        native_repair_reserves,
        native_conditions,
    ) = _native_source_index(native_payloads)

    amendment_payloads: list[tuple[FrozenFileReference, Mapping[str, Any]]] = []
    for route in authorization_amendments:
        if route.self_hash_field is None:
            raise FinalAccountingError("authorization amendments must be self-hashed")
        reference, payload = _source_route_reference(
            root,
            route,
            label=f"authorization amendment {route.artifact_id}",
        )
        _amendment_additional_services(payload, inventory_reference.file_sha256)
        amendment_payloads.append((reference, payload))

    wall_samples: list[datetime] = []
    for route in wall_time_receipts:
        if route.self_hash_field is None:
            raise FinalAccountingError("wall-time receipts must be self-hashed")
        _reference, payload = _source_route_reference(
            root,
            route,
            label=f"wall-time receipt {route.artifact_id}",
        )
        _started, sampled, _elapsed = _validated_wall_time_sample(payload)
        wall_samples.append(sampled)
    if not wall_samples:
        raise FinalAccountingError("source recipe requires at least one wall-time receipt")

    snapshots: list[_LedgerSnapshot] = []
    for route in ledger_sources:
        ledger_path = _safe_input(
            root,
            route.ledger_relative_path,
            label=f"ledger {route.ledger_id}",
        )
        cas_path = _safe_input(
            root,
            route.cas_relative_path,
            label=f"CAS {route.ledger_id}",
            directory=True,
        )
        spec = LedgerSnapshotSpec(
            ledger_id=route.ledger_id,
            lineage_id=route.lineage_id,
            sequence=route.sequence,
            parent_ledger_id=route.parent_ledger_id,
            ledger_relative_path=route.ledger_relative_path,
            ledger_file_sha256=_file_sha256(ledger_path),
            cas_relative_path=route.cas_relative_path,
            cas_inventory_sha256=cas_inventory_sha256(cas_path),
        )
        snapshots.append(_load_ledger_snapshot(root, spec))
    terminals = _terminal_snapshots(snapshots)
    _require_disjoint_terminal_ledgers(terminals)
    _require_accounting_samples(terminals)
    ledger_times = _terminal_evidence_timestamps(terminals)
    freeze_time = max(wall_samples)
    if ledger_times and freeze_time < max(ledger_times):
        raise FinalAccountingError(
            "latest wall-time receipt predates terminal scientific ledger evidence"
        )

    observed, _exclusions = _observed_attempts(
        terminals,
        native_strings_by_role=native_strings_by_role,
        native_classes=native_classes,
        native_repair_reserves=native_repair_reserves,
        native_conditions=native_conditions,
    )
    _registered_call_slots(observed)
    _registered_service_slots(
        terminals,
        amendment_payloads=amendment_payloads,
        inventory_hash=inventory_reference.file_sha256,
    )

    payload: dict[str, Any] = {
        "schema_version": FINAL_ACCOUNTING_SCHEMA_VERSION,
        "kind": "final_phase7_accounting_source_recipe",
        "accounting_id": accounting_id,
        "compiled_at_utc": freeze_time.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "base_call_inventory": base_call_inventory.model_dump(mode="json"),
        "native_source_artifacts": [
            item.model_dump(mode="json") for item in native_source_artifacts
        ],
        "authorization_amendments": [
            item.model_dump(mode="json") for item in authorization_amendments
        ],
        "ledger_sources": [item.model_dump(mode="json") for item in ledger_sources],
        "wall_time_receipts": [item.model_dump(mode="json") for item in wall_time_receipts],
    }
    payload["manifest_sha256"] = canonical_sha256(payload)
    try:
        recipe = FinalAccountingSourceRecipe.model_validate(payload)
    except Exception as exc:
        raise FinalAccountingError(f"cannot build final accounting source recipe: {exc}") from exc
    encoded = _json_bytes(recipe.model_dump(mode="json"))
    destination = _materialization_destination(root, output_root, verify_only=verify_only)
    recipe_path = destination / (f"final_accounting_sources.{recipe.manifest_sha256[:16]}.json")
    _publish_no_replace(recipe_path, encoded, verify_only=verify_only)
    return FinalAccountingSourceBuildOutputs(recipe_path=recipe_path, recipe=recipe)


def materialize_final_accounting_recipe(
    *,
    source_recipe_path: Path,
    source_root: Path,
    output_root: Path,
    verify_only: bool = False,
) -> FinalAccountingMaterializationOutputs:
    """Derive all normalized receipts and the compiler recipe from frozen artifacts."""

    root = _real_root(source_root, label="final accounting source root")
    source_path = Path(os.path.abspath(source_recipe_path))
    try:
        source_path.relative_to(root)
    except ValueError as exc:
        raise FinalAccountingError("source recipe must remain inside the source root") from exc
    _assert_real_file(source_path, label="final accounting source recipe")
    source_recipe = load_final_accounting_source_recipe(source_path)
    source_reference = FrozenFileReference(
        artifact_id="fa-materialization-source-recipe",
        relative_path=source_path.relative_to(root).as_posix(),
        file_sha256=_file_sha256(source_path),
        self_hash_field=SelfHashField.MANIFEST,
    )
    _load_reference(root, source_reference, label="final accounting source recipe")

    inventory_reference, _inventory_payload = _source_route_reference(
        root,
        source_recipe.base_call_inventory,
        label="registered GPU call inventory",
    )
    try:
        GPUCallInventory.load(
            _safe_input(
                root,
                inventory_reference.relative_path,
                label="registered GPU call inventory",
            ),
            enforce_registered_plan=True,
        )
    except Exception as exc:
        raise FinalAccountingError(f"registered GPU call inventory is invalid: {exc}") from exc

    native_references: list[FrozenFileReference] = [source_reference]
    native_payloads: list[tuple[NativeSourceRoute, Mapping[str, Any]]] = []
    for route in source_recipe.native_source_artifacts:
        reference, payload = _source_route_reference(
            root, route, label=f"native phase source {route.artifact_id}"
        )
        native_references.append(reference)
        native_payloads.append((route, payload))
    (
        native_strings_by_role,
        native_classes,
        native_repair_reserves,
        native_conditions,
    ) = _native_source_index(native_payloads)

    amendment_inputs: list[tuple[FrozenFileReference, Mapping[str, Any]]] = []
    for route in source_recipe.authorization_amendments:
        if route.self_hash_field is None:
            raise FinalAccountingError("authorization amendments must be self-hashed")
        reference, payload = _source_route_reference(
            root, route, label=f"authorization amendment {route.artifact_id}"
        )
        amendment_inputs.append((reference, payload))

    wall_references: list[ReceiptReference] = []
    wall_payloads: dict[str, Mapping[str, Any]] = {}
    for route in source_recipe.wall_time_receipts:
        if route.self_hash_field is None:
            raise FinalAccountingError("wall-time receipts must be self-hashed")
        reference, payload = _source_route_reference(
            root, route, label=f"wall-time receipt {route.artifact_id}"
        )
        if payload.get("kind") != ReceiptKind.WALL_TIME.value:
            raise FinalAccountingError("wall-time route names the wrong artifact kind")
        wall_payloads[route.artifact_id] = payload
        wall_references.append(
            ReceiptReference(
                **reference.model_dump(),
                receipt_kind=ReceiptKind.WALL_TIME,
            )
        )

    ledger_specs: list[LedgerSnapshotSpec] = []
    snapshots: list[_LedgerSnapshot] = []
    for route in source_recipe.ledger_sources:
        ledger_path = _safe_input(
            root, route.ledger_relative_path, label=f"ledger {route.ledger_id}"
        )
        cas_path = _safe_input(
            root,
            route.cas_relative_path,
            label=f"CAS {route.ledger_id}",
            directory=True,
        )
        spec = LedgerSnapshotSpec(
            ledger_id=route.ledger_id,
            lineage_id=route.lineage_id,
            sequence=route.sequence,
            parent_ledger_id=route.parent_ledger_id,
            ledger_relative_path=route.ledger_relative_path,
            ledger_file_sha256=_file_sha256(ledger_path),
            cas_relative_path=route.cas_relative_path,
            cas_inventory_sha256=cas_inventory_sha256(cas_path),
        )
        ledger_specs.append(spec)
        snapshots.append(_load_ledger_snapshot(root, spec))
    terminals = _terminal_snapshots(snapshots)
    _require_disjoint_terminal_ledgers(terminals)
    resource_coverage_gaps = _require_accounting_samples(terminals)
    _require_source_chronology(source_recipe, terminals)

    observed, exclusions = _observed_attempts(
        terminals,
        native_strings_by_role=native_strings_by_role,
        native_classes=native_classes,
        native_repair_reserves=native_repair_reserves,
        native_conditions=native_conditions,
    )
    call_slots = _registered_call_slots(observed)
    service_slots = _registered_service_slots(
        terminals,
        amendment_payloads=amendment_inputs,
        inventory_hash=inventory_reference.file_sha256,
    )
    plan = _MaterializedPlan(call_slots, service_slots)
    _failure_rows, facts = _compile_failure_rows(
        plan,  # type: ignore[arg-type]
        terminals,
        {(item.ledger_id, item.attempt_id) for item in exclusions},
    )
    all_stopped = all(
        item.report.unresolved_gpu_allocation_count == 0
        and item.report.unresolved_gpu_service_count == 0
        for item in terminals
    )

    destination = _materialization_destination(root, output_root, verify_only=verify_only)
    native_hashes = [item.file_sha256 for item in native_references]
    ledger_hashes = [item.ledger_file_sha256 for item in ledger_specs]
    amendment_hashes = [item[0].file_sha256 for item in amendment_inputs]
    shared = {
        "source_recipe_hash": source_recipe.manifest_sha256,
        "native_hashes": native_hashes,
        "ledger_hashes": ledger_hashes,
        "amendment_hashes": amendment_hashes,
        "inventory_hash": inventory_reference.file_sha256,
    }
    executed_calls = [item for item in call_slots if item.executed]
    generated_payloads: list[tuple[str, ReceiptKind, dict[str, Any]]] = [
        (
            "fa-execution",
            ReceiptKind.EXECUTION,
            _adapter_payload(
                {
                    "kind": ReceiptKind.EXECUTION.value,
                    "call_bindings": [
                        {
                            "slot_id": item.slot_id,
                            "ledger_id": item.ledger_id,
                            "job_id": item.job_id,
                            "attempt_id": item.attempt_id,
                            "model_call_id": item.model_call_id,
                            "gpu_event_ids": list(item.gpu_event_ids),
                        }
                        for item in executed_calls
                    ],
                },
                **shared,
            ),
        ),
        (
            "fa-result",
            ReceiptKind.RESULT,
            _adapter_payload(
                {
                    "kind": ReceiptKind.RESULT.value,
                    "registered_slot_ids": [item.slot_id for item in call_slots],
                    "itt_slot_ids": [item.slot_id for item in call_slots if item.included_in_itt],
                    "non_itt_slot_ids": [
                        item.slot_id for item in call_slots if not item.included_in_itt
                    ],
                    "terminal_outcomes": [
                        {
                            "slot_id": item.slot_id,
                            "outcome": facts["outcome_by_slot"][item.slot_id],
                        }
                        for item in call_slots
                    ],
                },
                **shared,
            ),
        ),
        (
            "fa-repair",
            ReceiptKind.REPAIR,
            _adapter_payload(
                {
                    "kind": ReceiptKind.REPAIR.value,
                    "repair_bindings": [
                        {"slot_id": slot_id, "parent_slot_id": parent_id}
                        for slot_id, parent_id in sorted(facts["repair_parent_slots"].items())
                    ],
                },
                **shared,
            ),
        ),
        (
            "fa-service",
            ReceiptKind.SERVICE,
            _adapter_payload(
                {
                    "kind": ReceiptKind.SERVICE.value,
                    "registered_service_slot_ids": [item.slot_id for item in service_slots],
                    "service_bindings": [
                        {
                            "slot_id": item.slot_id,
                            "ledger_id": item.ledger_id,
                            "service_session_id": item.service_session_id,
                            "gpu_event_ids": list(item.gpu_event_ids),
                        }
                        for item in service_slots
                        if item.executed
                    ],
                    "all_services_stopped": all_stopped,
                },
                **shared,
            ),
        ),
        (
            "fa-attempt-exclusion",
            ReceiptKind.ATTEMPT_EXCLUSION,
            _adapter_payload(
                {
                    "kind": ReceiptKind.ATTEMPT_EXCLUSION.value,
                    "exclusions": [item.model_dump(mode="json") for item in exclusions],
                },
                **shared,
            ),
        ),
    ]
    for snapshot in terminals:
        vllm_calls = sorted(
            (row for row in snapshot.rows["model_calls"] if row["backend"] == "vllm_gpu"),
            key=lambda row: str(row["model_call_id"]),
        )
        generated_payloads.append(
            (
                f"fa-token-{canonical_sha256(snapshot.spec.ledger_id)[:16]}",
                ReceiptKind.TOKEN,
                _adapter_payload(
                    {
                        "kind": ReceiptKind.TOKEN.value,
                        "ledger_id": snapshot.spec.ledger_id,
                        "model_call_ids": [str(row["model_call_id"]) for row in vllm_calls],
                        "prompt_tokens": sum(int(row["prompt_tokens"]) for row in vllm_calls),
                        "completion_tokens": sum(
                            int(row["completion_tokens"]) for row in vllm_calls
                        ),
                        "total_tokens": sum(
                            int(row["prompt_tokens"]) + int(row["completion_tokens"])
                            for row in vllm_calls
                        ),
                    },
                    **shared,
                ),
            )
        )
        storage = snapshot.rows["storage_samples"]
        resources = snapshot.rows["resource_samples"]
        generated_payloads.append(
            (
                f"fa-storage-{canonical_sha256(snapshot.spec.ledger_id)[:16]}",
                ReceiptKind.STORAGE,
                _adapter_payload(
                    {
                        "kind": ReceiptKind.STORAGE.value,
                        "ledger_id": snapshot.spec.ledger_id,
                        "storage_sample_ids": sorted(str(row["sample_id"]) for row in storage),
                        "resource_sample_ids": sorted(str(row["sample_id"]) for row in resources),
                        "uncovered_short_failed_service_session_ids": list(
                            resource_coverage_gaps.get(snapshot.spec.ledger_id, ())
                        ),
                        "resource_sampling_coverage_complete": not resource_coverage_gaps.get(
                            snapshot.spec.ledger_id, ()
                        ),
                        "peak_project_storage_bytes": max(
                            [int(row["current_occupied_bytes"]) for row in storage]
                            + [int(row["project_storage_bytes"]) for row in resources]
                            + [0]
                        ),
                        "peak_projected_storage_bytes": max(
                            [int(row["projected_occupied_bytes"]) for row in storage] + [0]
                        ),
                        "minimum_effective_headroom_bytes": min(
                            [int(row["effective_projected_headroom_bytes"]) for row in storage]
                            or [0]
                        ),
                        "all_storage_samples_allowed": all(bool(row["allowed"]) for row in storage),
                    },
                    **shared,
                ),
            )
        )

    receipt_material: list[tuple[ReceiptReference, Path, bytes]] = []
    for artifact_id, kind, payload in generated_payloads:
        receipt_material.append(
            _adapter_receipt_bytes(
                artifact_id=artifact_id,
                receipt_kind=kind,
                payload=payload,
                destination=destination,
                root=root,
            )
        )
    receipt_references = [item[0] for item in receipt_material]
    receipt_references.extend(wall_references)
    recipe_payload: dict[str, Any] = {
        "schema_version": FINAL_ACCOUNTING_SCHEMA_VERSION,
        "kind": "final_phase7_accounting_recipe",
        "accounting_id": source_recipe.accounting_id,
        "compiled_at_utc": source_recipe.model_dump(mode="json")["compiled_at_utc"],
        "source_materialization_recipe_sha256": source_recipe.manifest_sha256,
        "base_call_inventory": inventory_reference.model_dump(mode="json"),
        "native_source_artifacts": [item.model_dump(mode="json") for item in native_references],
        "authorization_amendments": [item[0].model_dump(mode="json") for item in amendment_inputs],
        "ledger_snapshots": [item.model_dump(mode="json") for item in ledger_specs],
        "receipts": [item.model_dump(mode="json") for item in receipt_references],
        "call_slots": [item.model_dump(mode="json") for item in call_slots],
        "service_slots": [item.model_dump(mode="json") for item in service_slots],
        "attempt_exclusions": [item.model_dump(mode="json") for item in exclusions],
    }
    recipe_payload["manifest_sha256"] = canonical_sha256(recipe_payload)
    try:
        final_recipe = FinalAccountingRecipe.model_validate(recipe_payload)
    except Exception as exc:
        raise FinalAccountingError(f"materialized final recipe is invalid: {exc}") from exc
    _wall_time_microseconds(final_recipe, wall_payloads, terminals)
    recipe_bytes = _json_bytes(final_recipe.model_dump(mode="json"))
    recipe_path = destination / (
        f"final_accounting_recipe.{final_recipe.manifest_sha256[:16]}.json"
    )

    for _reference, path, encoded in receipt_material:
        _publish_no_replace(path, encoded, verify_only=verify_only)
    _publish_no_replace(recipe_path, recipe_bytes, verify_only=verify_only)
    return FinalAccountingMaterializationOutputs(
        recipe_path=recipe_path,
        receipt_paths=tuple(item[1] for item in receipt_material),
        recipe=final_recipe,
    )


def compile_final_accounting(
    *,
    recipe_path: Path,
    source_root: Path,
    output_root: Path,
    verify_only: bool = False,
) -> FinalAccountingOutputs:
    """Compile or independently replay the two final canonical accounting tables."""

    root = _real_root(source_root, label="final accounting source root")
    recipe_file = Path(os.path.abspath(recipe_path))
    try:
        recipe_file.relative_to(root)
    except ValueError as exc:
        raise FinalAccountingError("recipe must remain inside the source root") from exc
    _assert_real_file(recipe_file, label="final accounting recipe")
    recipe_file_hash = _file_sha256(recipe_file)
    recipe = load_final_accounting_recipe(recipe_file)

    inventory_path, _payload = _load_reference(
        root, recipe.base_call_inventory, label="registered GPU call inventory"
    )
    try:
        inventory = GPUCallInventory.load(inventory_path, enforce_registered_plan=True)
    except Exception as exc:
        raise FinalAccountingError(f"registered GPU call inventory is invalid: {exc}") from exc
    if (
        inventory.accounting_events != REGISTERED_ACCOUNTING_EVENTS
        or inventory.maximum_inference_attempts != REGISTERED_INFERENCE_ATTEMPTS
    ):
        raise FinalAccountingError("registered GPU call inventory totals changed")

    native_payloads: list[Mapping[str, Any]] = []
    for reference in recipe.native_source_artifacts:
        _path, payload = _load_reference(
            root,
            reference,
            label=f"native source artifact {reference.artifact_id}",
        )
        native_payloads.append(payload)
    source_recipe_matches = [
        (reference, payload)
        for reference, payload in zip(recipe.native_source_artifacts, native_payloads, strict=True)
        if payload.get("kind") == "final_phase7_accounting_source_recipe"
        and payload.get("manifest_sha256") == recipe.source_materialization_recipe_sha256
    ]
    if len(source_recipe_matches) != 1:
        raise FinalAccountingError(
            "native sources must contain the exact unique materialization source recipe"
        )
    try:
        source_recipe = FinalAccountingSourceRecipe.model_validate(source_recipe_matches[0][1])
    except Exception as exc:
        raise FinalAccountingError(f"materialization source recipe is invalid: {exc}") from exc
    payload_by_artifact_id = {
        reference.artifact_id: payload
        for reference, payload in zip(recipe.native_source_artifacts, native_payloads, strict=True)
    }
    for route in source_recipe.native_source_artifacts:
        payload = payload_by_artifact_id.get(route.artifact_id)
        if payload is None:
            raise FinalAccountingError("final recipe omitted a typed native source")
        _validate_native_source_payload(route, payload)
    typed_native_sources = tuple(
        (route, payload_by_artifact_id[route.artifact_id])
        for route in source_recipe.native_source_artifacts
    )
    if (
        source_recipe.accounting_id != recipe.accounting_id
        or source_recipe.compiled_at_utc != recipe.compiled_at_utc
        or source_recipe.base_call_inventory.model_dump()
        != {
            "artifact_id": recipe.base_call_inventory.artifact_id,
            "relative_path": recipe.base_call_inventory.relative_path,
            "self_hash_field": recipe.base_call_inventory.self_hash_field,
        }
    ):
        raise FinalAccountingError("final recipe changed source-recipe identity or inventory route")
    source_reference = source_recipe_matches[0][0]
    final_native_routes = {
        (item.artifact_id, item.relative_path, item.self_hash_field)
        for item in recipe.native_source_artifacts
        if item.artifact_id != source_reference.artifact_id
    }
    if final_native_routes != {
        (item.artifact_id, item.relative_path, item.self_hash_field)
        for item in source_recipe.native_source_artifacts
    }:
        raise FinalAccountingError("final recipe changed native source routes")
    final_ledger_routes = {
        (
            item.ledger_id,
            item.lineage_id,
            item.sequence,
            item.parent_ledger_id,
            item.ledger_relative_path,
            item.cas_relative_path,
        )
        for item in recipe.ledger_snapshots
    }
    if final_ledger_routes != {
        (
            item.ledger_id,
            item.lineage_id,
            item.sequence,
            item.parent_ledger_id,
            item.ledger_relative_path,
            item.cas_relative_path,
        )
        for item in source_recipe.ledger_sources
    }:
        raise FinalAccountingError("final recipe changed ledger source routes")
    final_amendment_routes = {
        (item.artifact_id, item.relative_path, item.self_hash_field)
        for item in recipe.authorization_amendments
    }
    if final_amendment_routes != {
        (item.artifact_id, item.relative_path, item.self_hash_field)
        for item in source_recipe.authorization_amendments
    }:
        raise FinalAccountingError("final recipe changed amendment source routes")
    wall_routes = {
        (item.artifact_id, item.relative_path, item.self_hash_field)
        for item in recipe.receipts
        if item.receipt_kind is ReceiptKind.WALL_TIME
    }
    if wall_routes != {
        (item.artifact_id, item.relative_path, item.self_hash_field)
        for item in source_recipe.wall_time_receipts
    }:
        raise FinalAccountingError("final recipe changed wall-time source routes")

    extra_by_amendment: dict[str, int] = {}
    amendment_payloads: list[tuple[FrozenFileReference, Mapping[str, Any]]] = []
    for reference in recipe.authorization_amendments:
        _path, payload = _load_reference(
            root, reference, label=f"amendment {reference.artifact_id}"
        )
        if reference.self_hash_field is None:
            raise FinalAccountingError("authorization amendments must carry canonical self-hashes")
        extra_by_amendment[reference.artifact_id] = _amendment_additional_services(
            payload, recipe.base_call_inventory.file_sha256
        )
        amendment_payloads.append((reference, payload))
    actual_extra: dict[str, int] = defaultdict(int)
    actual_ordinals: dict[str, set[int]] = defaultdict(set)
    for slot in recipe.service_slots:
        if slot.amendment_artifact_id is not None:
            actual_extra[slot.amendment_artifact_id] += 1
            actual_ordinals[slot.amendment_artifact_id].add(slot.ordinal)
    if dict(actual_extra) != extra_by_amendment:
        raise FinalAccountingError("effective service slots differ from authorized amendments")
    for artifact_id, count in extra_by_amendment.items():
        if actual_ordinals[artifact_id] != set(range(1, count + 1)):
            raise FinalAccountingError(
                "amendment service-slot ordinals differ from the authorized inventory"
            )
    if len(recipe.service_slots) != REGISTERED_SESSION_STARTS + sum(extra_by_amendment.values()):
        raise FinalAccountingError("effective accounting-event inventory does not reconcile")

    snapshots = tuple(_load_ledger_snapshot(root, spec) for spec in recipe.ledger_snapshots)
    terminals = _terminal_snapshots(snapshots)
    _require_disjoint_terminal_ledgers(terminals)
    resource_coverage_gaps = _require_accounting_samples(terminals)
    _require_source_chronology(source_recipe, terminals)
    (
        native_strings_by_role,
        native_classes,
        native_repair_reserves,
        native_conditions,
    ) = _native_source_index(typed_native_sources)
    expected_observed, expected_exclusions = _observed_attempts(
        terminals,
        native_strings_by_role=native_strings_by_role,
        native_classes=native_classes,
        native_repair_reserves=native_repair_reserves,
        native_conditions=native_conditions,
    )
    if _registered_call_slots(expected_observed) != recipe.call_slots:
        raise FinalAccountingError("final recipe call slots differ from native ledger replay")
    if expected_exclusions != recipe.attempt_exclusions:
        raise FinalAccountingError("final recipe attempt exclusions differ from ledger replay")
    if (
        _registered_service_slots(
            terminals,
            amendment_payloads=amendment_payloads,
            inventory_hash=recipe.base_call_inventory.file_sha256,
        )
        != recipe.service_slots
    ):
        raise FinalAccountingError("final recipe service slots differ from ledger replay")
    payloads = _receipt_payloads(
        root,
        recipe.receipts,
        source_recipe_hash=recipe.source_materialization_recipe_sha256,
        native_file_hashes=[item.file_sha256 for item in recipe.native_source_artifacts],
        ledger_file_hashes=[item.ledger_file_sha256 for item in recipe.ledger_snapshots],
        amendment_file_hashes=[item.file_sha256 for item in recipe.authorization_amendments],
        inventory_file_hash=recipe.base_call_inventory.file_sha256,
    )
    exclusions = _attempt_exclusions(recipe, payloads)
    receipted_outcomes = _validate_result_receipts(recipe, payloads)
    _validate_execution_receipts(recipe, payloads)
    failure_rows, facts = _compile_failure_rows(recipe, terminals, exclusions)
    if receipted_outcomes != facts["outcome_by_slot"]:
        raise FinalAccountingError("result receipt outcomes differ from ledger-derived outcomes")
    _validate_repair_receipts(facts["repair_parent_slots"], recipe, payloads)
    all_stopped = all(
        item.report.unresolved_gpu_allocation_count == 0
        and item.report.unresolved_gpu_service_count == 0
        for item in terminals
    )
    _validate_service_receipts(recipe, payloads, all_stopped=all_stopped)
    _validate_token_receipts(terminals, recipe, payloads)
    _validate_storage_receipts(
        terminals,
        recipe,
        payloads,
        resource_coverage_gaps,
    )
    wall_microseconds = _wall_time_microseconds(recipe, payloads, terminals)
    resource_rows = _compile_resource_rows(
        recipe,
        terminals,
        facts,
        wall_microseconds,
        resource_coverage_gaps,
    )

    failure_bytes = _csv_bytes(FAILURE_ACCOUNTING_COLUMNS, failure_rows)
    resource_bytes = _csv_bytes(RESOURCE_ACCOUNTING_COLUMNS, resource_rows)
    failure_hash = hashlib.sha256(failure_bytes).hexdigest()
    resource_hash = hashlib.sha256(resource_bytes).hexdigest()
    failure_name = f"failure_accounting.{failure_hash[:16]}.csv"
    resource_name = f"resource_accounting.{resource_hash[:16]}.csv"
    terminal_ids = {item.spec.ledger_id for item in terminals}
    receipt_payload: dict[str, Any] = {
        "schema_version": FINAL_ACCOUNTING_SCHEMA_VERSION,
        "kind": "final_phase7_accounting_compilation_receipt",
        "accounting_id": recipe.accounting_id,
        "compiled_at_utc": recipe.compiled_at_utc.isoformat().replace("+00:00", "Z"),
        "source_recipe_manifest_sha256": recipe.manifest_sha256,
        "source_recipe_file_sha256": recipe_file_hash,
        "source_materialization_recipe_sha256": (recipe.source_materialization_recipe_sha256),
        "native_source_file_sha256": sorted(
            item.file_sha256 for item in recipe.native_source_artifacts
        ),
        "base_call_inventory_file_sha256": recipe.base_call_inventory.file_sha256,
        "base_accounting_events": REGISTERED_ACCOUNTING_EVENTS,
        "base_inference_attempts": REGISTERED_INFERENCE_ATTEMPTS,
        "effective_accounting_events": len(recipe.call_slots) + len(recipe.service_slots),
        "effective_service_start_events": len(recipe.service_slots),
        "authorized_amendment_file_sha256": sorted(
            item.file_sha256 for item in recipe.authorization_amendments
        ),
        "ledger_snapshots": [
            {
                "ledger_id": item.spec.ledger_id,
                "lineage_id": item.spec.lineage_id,
                "sequence": item.spec.sequence,
                "parent_ledger_id": item.spec.parent_ledger_id,
                "ledger_file_sha256": item.spec.ledger_file_sha256,
                "cas_inventory_sha256": item.spec.cas_inventory_sha256,
                "terminal_for_lineage": item.spec.ledger_id in terminal_ids,
                "row_hashes": dict(sorted(item.row_hashes.items())),
                "gpu_total_allocated_microseconds": item.report.gpu_total_allocated_microseconds,
            }
            for item in sorted(
                snapshots, key=lambda value: (value.spec.lineage_id, value.spec.sequence)
            )
        ],
        "receipt_file_sha256": sorted(item.file_sha256 for item in recipe.receipts),
        "reconciliation": {
            "registered_call_slots": len(recipe.call_slots),
            "registered_service_slots": len(recipe.service_slots),
            "executed_call_slots": sum(item.executed for item in recipe.call_slots),
            "executed_service_slots": sum(item.executed for item in recipe.service_slots),
            "attempt_exclusion_count": len(recipe.attempt_exclusions),
            "itt_slot_count": sum(item.included_in_itt for item in recipe.call_slots),
            "incomplete_mandatory_slot_count": sum(
                facts["outcome_by_slot"][item.slot_id] == "incomplete"
                for item in recipe.call_slots
                if not item.call_class.startswith("reserve_")
            ),
            "total_allocated_gpu_microseconds": facts["total_gpu_microseconds"],
            "service_overhead_microseconds": sum(
                dict(item.report.gpu_by_kind_microseconds).get("service_overhead", 0)
                for item in terminals
            ),
            "all_gpu_services_stopped": all_stopped,
            "resource_sample_count": sum(len(item.rows["resource_samples"]) for item in terminals),
            "storage_sample_count": sum(len(item.rows["storage_samples"]) for item in terminals),
            "resource_covered_service_count": sum(
                len(item.rows["gpu_service_sessions"])
                - len(resource_coverage_gaps.get(item.spec.ledger_id, ()))
                for item in terminals
            ),
            "resource_sampling_coverage_complete": not any(resource_coverage_gaps.values()),
            "uncovered_short_failed_gpu_service_ids": sorted(
                f"{ledger_id}:{service_id}"
                for ledger_id, service_ids in resource_coverage_gaps.items()
                for service_id in service_ids
            ),
            "runpod_container_wall_time_lower_bound_microseconds": wall_microseconds,
        },
        "outputs": [
            {
                "table_id": "failure_accounting",
                "relative_path": failure_name,
                "file_sha256": failure_hash,
                "row_count": len(failure_rows),
                "columns": list(FAILURE_ACCOUNTING_COLUMNS),
            },
            {
                "table_id": "resource_accounting",
                "relative_path": resource_name,
                "file_sha256": resource_hash,
                "row_count": len(resource_rows),
                "columns": list(RESOURCE_ACCOUNTING_COLUMNS),
            },
        ],
    }
    receipt_payload["manifest_sha256"] = canonical_sha256(receipt_payload)
    receipt_bytes = _json_bytes(receipt_payload)
    receipt_name = f"final_accounting_receipt.{receipt_payload['manifest_sha256'][:16]}.json"

    destination = Path(os.path.abspath(output_root))
    if verify_only:
        destination = _real_root(destination, label="final accounting output root")
    else:
        existing = destination
        while not existing.exists() and existing.parent != existing:
            existing = existing.parent
        _real_root(existing, label="final accounting output ancestor")
        destination.mkdir(parents=True, exist_ok=True)
        destination = _real_root(destination, label="final accounting output root")
    failure_path = destination / failure_name
    resource_path = destination / resource_name
    receipt_path = destination / receipt_name
    _publish_no_replace(failure_path, failure_bytes, verify_only=verify_only)
    _publish_no_replace(resource_path, resource_bytes, verify_only=verify_only)
    _publish_no_replace(receipt_path, receipt_bytes, verify_only=verify_only)
    return FinalAccountingOutputs(failure_path, resource_path, receipt_path, receipt_payload)


__all__ = [
    "FAILURE_ACCOUNTING_COLUMNS",
    "FINAL_ACCOUNTING_SCHEMA_VERSION",
    "RESOURCE_ACCOUNTING_COLUMNS",
    "AttemptExclusion",
    "FinalAccountingError",
    "FinalAccountingMaterializationOutputs",
    "FinalAccountingOutputs",
    "FinalAccountingRecipe",
    "FinalAccountingSourceBuildOutputs",
    "FinalAccountingSourceRecipe",
    "FrozenFileReference",
    "LedgerSnapshotSpec",
    "LedgerSourceRoute",
    "NativeSourceRole",
    "NativeSourceRoute",
    "ReceiptKind",
    "ReceiptReference",
    "RegisteredCallSlot",
    "RegisteredServiceSlot",
    "SelfHashField",
    "SourceFileRoute",
    "build_final_accounting_source_recipe",
    "cas_inventory_sha256",
    "compile_final_accounting",
    "load_final_accounting_recipe",
    "load_final_accounting_source_recipe",
    "materialize_final_accounting_recipe",
]
