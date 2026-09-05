"""Cryptographically bind the Phase 6 admission-evidence staging boundary.

The case-study factory must persist nine admission objects before it can create
the runtime bootstrap records.  This module makes that otherwise small mutation
an independently auditable transition: H0 is captured before any staging write,
H1 is captured immediately afterwards, and the receipt proves that the only
ledger/CAS changes were the declared admission objects.  Resume verification
may accept an append-only successor of H1, but never substitutes that successor
for the archived H1 boundary.

Only paths relative to an explicit restricted root enter the records.  SQLite
is opened read-only here; normal ``Ledger`` construction is deliberately avoided
because it enables WAL and prepares writable schema state.
"""

from __future__ import annotations

import base64
import collections
import gzip
import hashlib
import json
import math
import os
import sqlite3
import stat
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal, Self
from urllib.parse import quote

from pydantic import AwareDatetime, Field, StringConstraints, model_validator

from story_projection_onto.contracts import (
    ImmutableRecord,
    Sha256Digest,
    canonical_json,
    canonical_sha256,
)
from story_projection_onto.ledger_verify import audit_ledger

_GATE_ROLES = frozenset(
    {
        "synthetic_run_closure_hash",
        "timing_lineage_audit_hash",
        "gold_firewall_audit_hash",
        "registered_metric_regeneration_hash",
        "blinded_error_review_hash",
        "storage_preflight_hash",
        "gpu_schedule_admission_hash",
        "public_release_scan_hash",
    }
)
_BUNDLE_ROLE = "admission_evidence_bundle"
_EXPECTED_ROLES = _GATE_ROLES | {_BUNDLE_ROLE}
_GATE_MEDIA_TYPE = "application/vnd.story-projection.case-admission-gate+json"
_BUNDLE_MEDIA_TYPE = "application/json"
_GPU_TABLES = (
    "gpu_allocation_journal",
    "gpu_events",
    "gpu_service_journal",
    "gpu_service_sessions",
    "model_calls",
)

StagedPayloadRole = Literal[
    "synthetic_run_closure_hash",
    "timing_lineage_audit_hash",
    "gold_firewall_audit_hash",
    "registered_metric_regeneration_hash",
    "blinded_error_review_hash",
    "storage_preflight_hash",
    "gpu_schedule_admission_hash",
    "public_release_scan_hash",
    "admission_evidence_bundle",
]
CompressionName = Literal["gzip", "zstd"]
RelativePath = Annotated[str, StringConstraints(min_length=1, max_length=512)]


class CaseStudyTransitionError(RuntimeError):
    """The staged admission transition cannot be proven exactly."""


def _validate_relative_path(value: str, *, label: str) -> str:
    if "\\" in value or "\x00" in value:
        raise ValueError(f"{label} is not a portable relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or value in {"", "."} or ".." in path.parts:
        raise ValueError(f"{label} must be a bounded relative path")
    return path.as_posix()


class CaseStagedPayloadDescriptor(ImmutableRecord):
    """One object that is allowed to appear between H0 and H1."""

    role: StagedPayloadRole
    logical_content_hash: Sha256Digest
    artifact_hash: Sha256Digest
    raw_size_bytes: int = Field(ge=0)
    compression: CompressionName
    media_type: Literal[
        "application/vnd.story-projection.case-admission-gate+json",
        "application/json",
    ]
    release_class: Literal["restricted"] = "restricted"

    @model_validator(mode="after")
    def media_type_matches_role(self) -> Self:
        expected = _BUNDLE_MEDIA_TYPE if self.role == _BUNDLE_ROLE else _GATE_MEDIA_TYPE
        if self.media_type != expected:
            raise ValueError(f"{self.role} requires media type {expected}")
        return self

    @property
    def expected_relative_path(self) -> str:
        suffix = ".jsonl.gz" if self.compression == "gzip" else ".jsonl.zst"
        return f"{self.artifact_hash[:2]}/{self.artifact_hash}{suffix}"


class CasePostH1ArtifactDescriptor(ImmutableRecord):
    """One ordered bootstrap artifact allowed after the archived H1 boundary."""

    role: str = Field(pattern=r"^[a-z][a-z0-9_]{0,127}$")
    logical_content_hash: Sha256Digest
    artifact_hash: Sha256Digest
    raw_size_bytes: int = Field(ge=0)
    compression: CompressionName
    media_type: str = Field(min_length=1, max_length=256)
    release_class: Literal["restricted"] = "restricted"

    @property
    def expected_relative_path(self) -> str:
        suffix = ".jsonl.gz" if self.compression == "gzip" else ".jsonl.zst"
        return f"{self.artifact_hash[:2]}/{self.artifact_hash}{suffix}"


class LedgerTableInventory(ImmutableRecord):
    table_name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    row_count: int = Field(ge=0)
    row_inventory_hash: Sha256Digest


class LedgerSnapshotInventory(ImmutableRecord):
    snapshot_relative_path: RelativePath
    file_sha256: Sha256Digest
    schema_inventory_hash: Sha256Digest
    tables: tuple[LedgerTableInventory, ...]
    gpu_inventory_hash: Sha256Digest
    allocated_gpu_microseconds: int = Field(ge=0)

    @model_validator(mode="after")
    def canonical_inventory(self) -> Self:
        _validate_relative_path(self.snapshot_relative_path, label="ledger snapshot")
        names = tuple(table.table_name for table in self.tables)
        if names != tuple(sorted(set(names))):
            raise ValueError("ledger table inventory must be sorted and unique")
        return self


class CasFileInventoryEntry(ImmutableRecord):
    relative_path: RelativePath
    file_sha256: Sha256Digest
    size_bytes: int = Field(ge=0)

    @model_validator(mode="after")
    def portable_path(self) -> Self:
        _validate_relative_path(self.relative_path, label="CAS entry")
        return self


class CaseCasInventoryManifest(ImmutableRecord):
    entries: tuple[CasFileInventoryEntry, ...]
    release_class: Literal["restricted"] = "restricted"

    @model_validator(mode="after")
    def canonical_entries(self) -> Self:
        paths = tuple(entry.relative_path for entry in self.entries)
        if paths != tuple(sorted(set(paths))):
            raise ValueError("CAS inventory entries must be sorted and unique")
        return self


class LedgerArtifactInventoryEntry(ImmutableRecord):
    artifact_hash: Sha256Digest
    compression: CompressionName
    media_type: str = Field(min_length=1, max_length=256)
    raw_size_bytes: int = Field(ge=0)
    stored_size_bytes: int = Field(ge=0)
    relative_path: RelativePath
    release_class: Literal["restricted"]
    created_at: str = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def portable_path(self) -> Self:
        _validate_relative_path(self.relative_path, label="artifact")
        return self


class CaseAdmissionStagingIntent(ImmutableRecord):
    transition_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,191}$")
    execution_plan_hash: Sha256Digest
    admission_attestation_hash: Sha256Digest
    semantic_bundle_hash: Sha256Digest
    h0_ledger: LedgerSnapshotInventory
    h0_cas_manifest_relative_path: RelativePath
    h0_cas_manifest_hash: Sha256Digest
    expected_payloads: tuple[CaseStagedPayloadDescriptor, ...]
    expected_payload_inventory_hash: Sha256Digest
    evidence_bundle_hash: Sha256Digest
    evidence_bundle_artifact_hash: Sha256Digest
    evidence_bundle_reference_hash: Sha256Digest
    captured_at: AwareDatetime
    release_class: Literal["restricted"] = "restricted"

    @model_validator(mode="after")
    def exact_staging_inventory(self) -> Self:
        _validate_relative_path(
            self.h0_cas_manifest_relative_path,
            label="H0 CAS manifest",
        )
        roles = tuple(payload.role for payload in self.expected_payloads)
        if roles != tuple(sorted(_EXPECTED_ROLES)):
            raise ValueError("staging intent requires the nine sorted payload roles exactly once")
        inventory_hash = _hash_tree(
            tuple(json.loads(item.to_canonical_json()) for item in self.expected_payloads)
        )
        if self.expected_payload_inventory_hash != inventory_hash:
            raise ValueError("expected payload inventory hash mismatch")
        bundle = next(item for item in self.expected_payloads if item.role == _BUNDLE_ROLE)
        if (
            bundle.logical_content_hash != self.evidence_bundle_hash
            or bundle.artifact_hash != self.evidence_bundle_artifact_hash
        ):
            raise ValueError("bundle descriptor does not match the declared evidence bundle")
        expected_reference_hash = canonical_sha256(
            {
                "schema_version": "1.0.0",
                "logical_content_hash": self.evidence_bundle_hash,
                "artifact_hash": self.evidence_bundle_artifact_hash,
                "object_kind": "case_admission_evidence_bundle",
            }
        )
        if self.evidence_bundle_reference_hash != expected_reference_hash:
            raise ValueError("evidence bundle reference hash is not reproducible")
        return self


class CaseAdmissionStagingReceipt(ImmutableRecord):
    transition_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,191}$")
    intent_hash: Sha256Digest
    execution_plan_hash: Sha256Digest
    admission_attestation_hash: Sha256Digest
    semantic_bundle_hash: Sha256Digest
    h0_ledger_hash: Sha256Digest
    h0_cas_manifest_hash: Sha256Digest
    h1_ledger: LedgerSnapshotInventory
    h1_cas_manifest_relative_path: RelativePath
    h1_cas_manifest_hash: Sha256Digest
    expected_payload_inventory_hash: Sha256Digest
    added_artifacts: tuple[LedgerArtifactInventoryEntry, ...]
    reused_expected_artifact_hashes: tuple[Sha256Digest, ...]
    added_cas_entries: tuple[CasFileInventoryEntry, ...]
    gpu_inventory_hash: Sha256Digest
    allocated_gpu_microseconds: int = Field(ge=0)
    evidence_bundle_hash: Sha256Digest
    evidence_bundle_artifact_hash: Sha256Digest
    evidence_bundle_reference_hash: Sha256Digest
    completed_at: AwareDatetime
    release_class: Literal["restricted"] = "restricted"

    @model_validator(mode="after")
    def canonical_delta(self) -> Self:
        _validate_relative_path(
            self.h1_cas_manifest_relative_path,
            label="H1 CAS manifest",
        )
        artifact_hashes = tuple(item.artifact_hash for item in self.added_artifacts)
        if artifact_hashes != tuple(sorted(set(artifact_hashes))):
            raise ValueError("added artifacts must be sorted and unique")
        if self.reused_expected_artifact_hashes != tuple(
            sorted(set(self.reused_expected_artifact_hashes))
        ):
            raise ValueError("reused artifact hashes must be sorted and unique")
        cas_paths = tuple(item.relative_path for item in self.added_cas_entries)
        if cas_paths != tuple(sorted(set(cas_paths))):
            raise ValueError("added CAS entries must be sorted and unique")
        return self


@dataclass(frozen=True, slots=True)
class CaseAdmissionTransitionPaths:
    directory: Path
    intent: Path
    h0_ledger: Path
    h0_cas_manifest: Path
    receipt: Path
    h1_ledger: Path
    h1_cas_manifest: Path


@dataclass(frozen=True, slots=True)
class VerifiedCaseAdmissionStagingTransition:
    intent: CaseAdmissionStagingIntent
    receipt: CaseAdmissionStagingReceipt
    archived_h1_ledger_sha256: str
    archived_h1_inventory_hash: str
    archived_h1_gpu_inventory_hash: str
    archived_h1_allocated_gpu_microseconds: int
    current_ledger_sha256: str
    current_ledger_inventory_hash: str
    current_gpu_inventory_hash: str
    current_allocated_gpu_microseconds: int
    current_matches_archived_h1: bool
    current_is_append_only_successor: bool


@dataclass(frozen=True, slots=True)
class VerifiedCasePostH1ArtifactPrefix:
    transition_receipt_hash: str
    preexisting_roles: tuple[str, ...]
    newly_registered_prefix_roles: tuple[str, ...]
    completed_prefix_count: int
    next_orphan_cas_role: str | None
    archived_h1_ledger_sha256: str
    current_ledger_sha256: str
    archived_h1_gpu_inventory_hash: str
    current_gpu_inventory_hash: str
    allocated_gpu_microseconds: int


@dataclass(frozen=True, slots=True)
class _DatabaseState:
    schema_hash: str
    rows: Mapping[str, tuple[str, ...]]
    artifacts: Mapping[str, Mapping[str, Any]]
    gpu_inventory_hash: str
    allocated_gpu_microseconds: int


def case_admission_transition_paths(
    *,
    restricted_root: Path,
    transition_directory: Path,
) -> CaseAdmissionTransitionPaths:
    root = _real_directory(restricted_root, label="restricted root")
    directory = _bounded_path(transition_directory, root=root, label="transition directory")
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    if directory.is_symlink() or directory.resolve(strict=True) != directory:
        raise CaseStudyTransitionError("transition directory must be a stable real directory")
    return CaseAdmissionTransitionPaths(
        directory=directory,
        intent=directory / "staging-intent.json",
        h0_ledger=directory / "h0-ledger.sqlite3",
        h0_cas_manifest=directory / "h0-cas-manifest.json",
        receipt=directory / "staging-receipt.json",
        h1_ledger=directory / "h1-ledger.sqlite3",
        h1_cas_manifest=directory / "h1-cas-manifest.json",
    )


def capture_case_admission_staging_intent(
    *,
    restricted_root: Path,
    ledger_path: Path,
    cas_root: Path,
    transition_directory: Path,
    transition_id: str,
    execution_plan_hash: str,
    admission_attestation_hash: str,
    semantic_bundle_hash: str,
    expected_payloads: Sequence[CaseStagedPayloadDescriptor],
    evidence_bundle_hash: str,
    evidence_bundle_artifact_hash: str,
    evidence_bundle_reference_hash: str,
    captured_at: datetime,
) -> CaseAdmissionStagingIntent:
    """Archive H0 and publish the intent before any admission object is staged.

    If the intent already exists, it is treated as the authority.  This allows
    recovery after staging began without falsely requiring the live ledger to
    still equal H0.
    """

    root, ledger, cas, paths = _resolve_inputs(
        restricted_root=restricted_root,
        ledger_path=ledger_path,
        cas_root=cas_root,
        transition_directory=transition_directory,
    )
    _repair_interrupted_transition_publications(paths)
    payloads = _normalise_payloads(expected_payloads)
    if paths.intent.exists():
        intent = _load_record(paths.intent, CaseAdmissionStagingIntent)
        _require_intent_call_binding(
            intent,
            transition_id=transition_id,
            execution_plan_hash=execution_plan_hash,
            admission_attestation_hash=admission_attestation_hash,
            semantic_bundle_hash=semantic_bundle_hash,
            payloads=payloads,
            evidence_bundle_hash=evidence_bundle_hash,
            evidence_bundle_artifact_hash=evidence_bundle_artifact_hash,
            evidence_bundle_reference_hash=evidence_bundle_reference_hash,
        )
        _verify_h0_archives(root=root, cas=cas, intent=intent)
        return intent

    _require_quiescent_sqlite(ledger)
    _require_valid_ledger(ledger, cas, label="H0")
    live_state = _read_database_state(ledger)
    _publish_copy_no_replace(ledger, paths.h0_ledger)
    archived_state = _read_database_state(paths.h0_ledger)
    if archived_state != live_state:
        raise CaseStudyTransitionError("archived H0 is not the captured logical ledger")
    h0_inventory = _ledger_inventory(
        state=archived_state,
        snapshot_path=paths.h0_ledger,
        root=root,
    )
    h0_cas = _scan_cas(cas)
    _publish_record_no_replace(paths.h0_cas_manifest, h0_cas)
    intent = CaseAdmissionStagingIntent(
        transition_id=transition_id,
        execution_plan_hash=execution_plan_hash,
        admission_attestation_hash=admission_attestation_hash,
        semantic_bundle_hash=semantic_bundle_hash,
        h0_ledger=h0_inventory,
        h0_cas_manifest_relative_path=_relative(paths.h0_cas_manifest, root),
        h0_cas_manifest_hash=h0_cas.content_hash,
        expected_payloads=payloads,
        expected_payload_inventory_hash=_payload_inventory_hash(payloads),
        evidence_bundle_hash=evidence_bundle_hash,
        evidence_bundle_artifact_hash=evidence_bundle_artifact_hash,
        evidence_bundle_reference_hash=evidence_bundle_reference_hash,
        captured_at=captured_at,
    )
    _publish_record_no_replace(paths.intent, intent)
    return intent


def finalize_case_admission_staging_transition(
    *,
    restricted_root: Path,
    ledger_path: Path,
    cas_root: Path,
    transition_directory: Path,
    completed_at: datetime,
) -> CaseAdmissionStagingReceipt:
    """Archive H1 and prove the exact H0-to-H1 admission-staging delta."""

    root, ledger, cas, paths = _resolve_inputs(
        restricted_root=restricted_root,
        ledger_path=ledger_path,
        cas_root=cas_root,
        transition_directory=transition_directory,
    )
    _repair_interrupted_transition_publications(paths)
    intent = _load_record(paths.intent, CaseAdmissionStagingIntent)
    if paths.receipt.exists():
        receipt = _load_record(paths.receipt, CaseAdmissionStagingReceipt)
        _verify_transition_archives(root=root, cas=cas, intent=intent, receipt=receipt)
        _verify_live_successor(
            ledger=ledger,
            cas=cas,
            h1_ledger=_path_from_relative(
                root,
                receipt.h1_ledger.snapshot_relative_path,
            ),
            h1_cas=_load_record(
                _path_from_relative(root, receipt.h1_cas_manifest_relative_path),
                CaseCasInventoryManifest,
            ),
        )
        return receipt

    h1_ledger_exists = paths.h1_ledger.exists()
    h1_cas_manifest_exists = paths.h1_cas_manifest.exists()
    if h1_ledger_exists:
        _require_independent_ledger_archive(
            source=ledger,
            archive=paths.h1_ledger,
            label="H1 ledger",
        )
    # If both H1 archives survived a crash, they—not a later live successor—are
    # the boundary from which the missing receipt must be reconstructed.
    if h1_ledger_exists and h1_cas_manifest_exists:
        h1_source = paths.h1_ledger
        h1_cas = _load_record(paths.h1_cas_manifest, CaseCasInventoryManifest)
    elif h1_ledger_exists:
        # Publication is ledger first, then CAS manifest.  A power loss between
        # those two durable links must be mechanically resumable.  No bootstrap
        # mutation is authorized before the receipt exists, so the live state
        # must still be the exact archived-ledger/CAS boundary.
        _require_quiescent_sqlite(ledger)
        _require_valid_ledger(ledger, cas, label="recoverable H1")
        if _file_sha256(ledger) != _file_sha256(paths.h1_ledger):
            raise CaseStudyTransitionError(
                "live ledger changed after the partial H1 archive"
            )
        h1_source = paths.h1_ledger
        current_cas = _scan_cas(cas)
        h1_cas = _expected_h1_cas_manifest(
            root=root,
            intent=intent,
            current=current_cas,
        )
        if current_cas != h1_cas:
            raise CaseStudyTransitionError("live CAS changed after the partial H1 archive")
        _prove_staging_delta(
            root=root,
            cas=cas,
            intent=intent,
            h1_ledger=h1_source,
            h1_cas=h1_cas,
        )
        _publish_record_no_replace(paths.h1_cas_manifest, h1_cas)
    elif h1_cas_manifest_exists:
        # This is not the normal publication order, but accepting a valid exact
        # counterpart makes recovery symmetric without trusting a later live
        # successor as the missing H1 boundary.
        _require_quiescent_sqlite(ledger)
        _require_valid_ledger(ledger, cas, label="recoverable H1")
        h1_cas = _load_record(paths.h1_cas_manifest, CaseCasInventoryManifest)
        if _scan_cas(cas) != h1_cas:
            raise CaseStudyTransitionError("live CAS changed after the partial H1 archive")
        _prove_staging_delta(
            root=root,
            cas=cas,
            intent=intent,
            h1_ledger=ledger,
            h1_cas=h1_cas,
        )
        _publish_copy_no_replace(ledger, paths.h1_ledger)
        h1_source = paths.h1_ledger
    else:
        _require_quiescent_sqlite(ledger)
        _require_valid_ledger(ledger, cas, label="H1")
        h1_source = ledger
        h1_cas = _scan_cas(cas)

    delta = _prove_staging_delta(
        root=root,
        cas=cas,
        intent=intent,
        h1_ledger=h1_source,
        h1_cas=h1_cas,
    )
    if h1_source == ledger:
        _publish_copy_no_replace(ledger, paths.h1_ledger)
        _publish_record_no_replace(paths.h1_cas_manifest, h1_cas)
    else:
        _verify_live_successor(
            ledger=ledger,
            cas=cas,
            h1_ledger=paths.h1_ledger,
            h1_cas=h1_cas,
        )
    h1_state = _read_database_state(paths.h1_ledger)
    h1_inventory = _ledger_inventory(
        state=h1_state,
        snapshot_path=paths.h1_ledger,
        root=root,
    )
    receipt = CaseAdmissionStagingReceipt(
        transition_id=intent.transition_id,
        intent_hash=intent.content_hash,
        execution_plan_hash=intent.execution_plan_hash,
        admission_attestation_hash=intent.admission_attestation_hash,
        semantic_bundle_hash=intent.semantic_bundle_hash,
        h0_ledger_hash=intent.h0_ledger.content_hash,
        h0_cas_manifest_hash=intent.h0_cas_manifest_hash,
        h1_ledger=h1_inventory,
        h1_cas_manifest_relative_path=_relative(paths.h1_cas_manifest, root),
        h1_cas_manifest_hash=h1_cas.content_hash,
        expected_payload_inventory_hash=intent.expected_payload_inventory_hash,
        added_artifacts=delta[0],
        reused_expected_artifact_hashes=delta[1],
        added_cas_entries=delta[2],
        gpu_inventory_hash=h1_state.gpu_inventory_hash,
        allocated_gpu_microseconds=h1_state.allocated_gpu_microseconds,
        evidence_bundle_hash=intent.evidence_bundle_hash,
        evidence_bundle_artifact_hash=intent.evidence_bundle_artifact_hash,
        evidence_bundle_reference_hash=intent.evidence_bundle_reference_hash,
        completed_at=completed_at,
    )
    _publish_record_no_replace(paths.receipt, receipt)
    _verify_transition_archives(root=root, cas=cas, intent=intent, receipt=receipt)
    return receipt


def verify_case_admission_staging_transition(
    *,
    restricted_root: Path,
    ledger_path: Path,
    cas_root: Path,
    transition_directory: Path,
    execution_plan_hash: str,
    admission_attestation_hash: str,
    semantic_bundle_hash: str,
    evidence_bundle_hash: str,
    evidence_bundle_artifact_hash: str,
    evidence_bundle_reference_hash: str,
    require_current_h1: bool = True,
) -> VerifiedCaseAdmissionStagingTransition:
    """Verify archived H0/H1 and either exact H1 or an append-only successor.

    ``require_current_h1=False`` does not authorize successor rows.  It only
    proves that archived H1 is a prefix of the current ledger/CAS so a caller can
    separately validate bootstrap/runtime lineage for the additional rows.
    """

    root, ledger, cas, paths = _resolve_inputs(
        restricted_root=restricted_root,
        ledger_path=ledger_path,
        cas_root=cas_root,
        transition_directory=transition_directory,
    )
    intent = _load_record(paths.intent, CaseAdmissionStagingIntent)
    receipt = _load_record(paths.receipt, CaseAdmissionStagingReceipt)
    expected_bindings = (
        ("execution plan", intent.execution_plan_hash, execution_plan_hash),
        ("admission attestation", intent.admission_attestation_hash, admission_attestation_hash),
        ("semantic bundle", intent.semantic_bundle_hash, semantic_bundle_hash),
        ("evidence bundle", intent.evidence_bundle_hash, evidence_bundle_hash),
        (
            "evidence bundle artifact",
            intent.evidence_bundle_artifact_hash,
            evidence_bundle_artifact_hash,
        ),
        (
            "evidence bundle reference",
            intent.evidence_bundle_reference_hash,
            evidence_bundle_reference_hash,
        ),
    )
    for label, recorded, expected in expected_bindings:
        if recorded != expected:
            raise CaseStudyTransitionError(f"{label} does not match staging transition")
    _verify_transition_archives(root=root, cas=cas, intent=intent, receipt=receipt)

    _require_quiescent_sqlite(ledger)
    _require_valid_ledger(ledger, cas, label="current successor")
    h1_state = _read_database_state(
        _path_from_relative(root, receipt.h1_ledger.snapshot_relative_path)
    )
    current_state = _read_database_state(ledger)
    current_cas = _scan_cas(cas)
    h1_cas = _load_record(
        _path_from_relative(root, receipt.h1_cas_manifest_relative_path),
        CaseCasInventoryManifest,
    )
    logical_match = current_state == h1_state
    raw_match = _file_sha256(ledger) == receipt.h1_ledger.file_sha256
    cas_match = current_cas == h1_cas
    exact_match = logical_match and raw_match and cas_match
    if require_current_h1 and not exact_match:
        raise CaseStudyTransitionError("current ledger/CAS is not the archived H1 boundary")
    successor = _is_append_only_successor(h1_state, current_state)
    _require_cas_superset(h1_cas, current_cas)
    if not successor:
        raise CaseStudyTransitionError("current ledger is not an append-only successor of H1")
    return VerifiedCaseAdmissionStagingTransition(
        intent=intent,
        receipt=receipt,
        archived_h1_ledger_sha256=receipt.h1_ledger.file_sha256,
        archived_h1_inventory_hash=_ledger_inventory_hash(h1_state),
        archived_h1_gpu_inventory_hash=receipt.h1_ledger.gpu_inventory_hash,
        archived_h1_allocated_gpu_microseconds=receipt.h1_ledger.allocated_gpu_microseconds,
        current_ledger_sha256=_file_sha256(ledger),
        current_ledger_inventory_hash=_ledger_inventory_hash(current_state),
        current_gpu_inventory_hash=current_state.gpu_inventory_hash,
        current_allocated_gpu_microseconds=current_state.allocated_gpu_microseconds,
        current_matches_archived_h1=exact_match,
        current_is_append_only_successor=successor,
    )


def verify_case_post_h1_artifact_prefix(
    *,
    restricted_root: Path,
    ledger_path: Path,
    cas_root: Path,
    transition_directory: Path,
    expected_payloads: Sequence[CasePostH1ArtifactDescriptor],
) -> VerifiedCasePostH1ArtifactPrefix:
    """Prove a crash-safe zero/partial/full artifact-only bootstrap prefix.

    The descriptor order is the write order.  Existing H1 objects are treated
    as content-addressed reuse.  New artifact rows must be exactly a prefix of
    the remaining descriptors; a single CAS-only blob for the next descriptor
    is also accepted because ``BlobStore`` renames before ledger registration.
    No other ledger row, GPU record, or CAS object may change.
    """

    root, ledger, cas, paths = _resolve_inputs(
        restricted_root=restricted_root,
        ledger_path=ledger_path,
        cas_root=cas_root,
        transition_directory=transition_directory,
    )
    descriptors = tuple(expected_payloads)
    roles = tuple(item.role for item in descriptors)
    hashes = tuple(item.artifact_hash for item in descriptors)
    if not descriptors:
        raise CaseStudyTransitionError("post-H1 prefix requires at least one descriptor")
    if len(set(roles)) != len(roles) or len(set(hashes)) != len(hashes):
        raise CaseStudyTransitionError("post-H1 descriptor roles and hashes must be unique")

    intent = _load_record(paths.intent, CaseAdmissionStagingIntent)
    receipt = _load_record(paths.receipt, CaseAdmissionStagingReceipt)
    _verify_transition_archives(root=root, cas=cas, intent=intent, receipt=receipt)
    h1_ledger = _path_from_relative(root, receipt.h1_ledger.snapshot_relative_path)
    h1_cas = _load_record(
        _path_from_relative(root, receipt.h1_cas_manifest_relative_path),
        CaseCasInventoryManifest,
    )
    _require_quiescent_sqlite(ledger)
    _require_valid_ledger(ledger, cas, label="post-H1 artifact prefix")
    h1_state = _read_database_state(h1_ledger)
    current_state = _read_database_state(ledger)
    if h1_state.schema_hash != current_state.schema_hash or set(h1_state.rows) != set(
        current_state.rows
    ):
        raise CaseStudyTransitionError("ledger schema changed after H1")
    if (
        h1_state.gpu_inventory_hash != current_state.gpu_inventory_hash
        or h1_state.allocated_gpu_microseconds != current_state.allocated_gpu_microseconds
    ):
        raise CaseStudyTransitionError("GPU allocation inventory changed after H1")
    for table in sorted(h1_state.rows):
        if table != "artifacts" and h1_state.rows[table] != current_state.rows[table]:
            raise CaseStudyTransitionError(f"unexpected post-H1 ledger delta: {table}")

    h1_hashes = set(h1_state.artifacts)
    current_hashes = set(current_state.artifacts)
    if not h1_hashes <= current_hashes:
        raise CaseStudyTransitionError("an H1 artifact row disappeared")
    for digest in h1_hashes:
        if h1_state.artifacts[digest] != current_state.artifacts[digest]:
            raise CaseStudyTransitionError("an H1 artifact row changed")
    remaining = tuple(item for item in descriptors if item.artifact_hash not in h1_hashes)
    added_hashes = current_hashes - h1_hashes
    prefix_count = next(
        (
            count
            for count in range(len(remaining) + 1)
            if {item.artifact_hash for item in remaining[:count]} == added_hashes
        ),
        -1,
    )
    if prefix_count < 0:
        raise CaseStudyTransitionError("artifact rows are not the declared post-H1 prefix")
    for descriptor in remaining[:prefix_count]:
        _verify_payload_descriptor(
            descriptor,
            row=current_state.artifacts[descriptor.artifact_hash],
            cas=cas,
        )
    for descriptor in descriptors:
        if descriptor.artifact_hash in h1_hashes:
            _verify_payload_descriptor(
                descriptor,
                row=h1_state.artifacts[descriptor.artifact_hash],
                cas=cas,
            )

    current_cas = _scan_cas(cas)
    h1_files = {item.relative_path: item for item in h1_cas.entries}
    current_files = {item.relative_path: item for item in current_cas.entries}
    if not set(h1_files) <= set(current_files):
        raise CaseStudyTransitionError("an H1 CAS file disappeared")
    for path, entry in h1_files.items():
        if current_files[path] != entry:
            raise CaseStudyTransitionError("an H1 CAS file changed")
    added_paths = set(current_files) - set(h1_files)
    required_paths = {
        item.expected_relative_path
        for item in remaining[:prefix_count]
        if item.expected_relative_path not in h1_files
    }
    if not required_paths <= added_paths:
        raise CaseStudyTransitionError("registered post-H1 artifact lacks its CAS object")
    orphan_paths = added_paths - required_paths
    next_descriptor = remaining[prefix_count] if prefix_count < len(remaining) else None
    allowed_orphan_path = (
        next_descriptor.expected_relative_path
        if next_descriptor is not None and next_descriptor.expected_relative_path not in h1_files
        else None
    )
    if orphan_paths not in (set(), {allowed_orphan_path} if allowed_orphan_path else set()):
        raise CaseStudyTransitionError("CAS files are not the declared post-H1 prefix")
    orphan_role = None
    if orphan_paths:
        assert next_descriptor is not None
        _verify_payload_bytes(next_descriptor, cas=cas)
        orphan_role = next_descriptor.role
    return VerifiedCasePostH1ArtifactPrefix(
        transition_receipt_hash=receipt.content_hash,
        preexisting_roles=tuple(
            item.role for item in descriptors if item.artifact_hash in h1_hashes
        ),
        newly_registered_prefix_roles=tuple(item.role for item in remaining[:prefix_count]),
        completed_prefix_count=prefix_count,
        next_orphan_cas_role=orphan_role,
        archived_h1_ledger_sha256=receipt.h1_ledger.file_sha256,
        current_ledger_sha256=_file_sha256(ledger),
        archived_h1_gpu_inventory_hash=h1_state.gpu_inventory_hash,
        current_gpu_inventory_hash=current_state.gpu_inventory_hash,
        allocated_gpu_microseconds=current_state.allocated_gpu_microseconds,
    )


def _resolve_inputs(
    *,
    restricted_root: Path,
    ledger_path: Path,
    cas_root: Path,
    transition_directory: Path,
) -> tuple[Path, Path, Path, CaseAdmissionTransitionPaths]:
    root = _real_directory(restricted_root, label="restricted root")
    ledger = _bounded_existing_file(ledger_path, root=root, label="ledger")
    # Production uses the one repository CAS under ``artifacts/blobs`` while
    # transition records live under the sibling ignored restricted root.  The
    # caller must first apply the repository's shared-CAS allowlist; this layer
    # requires a stable real root and records only paths relative to it.
    cas = _real_directory(cas_root, label="CAS root")
    try:
        ledger.relative_to(cas)
    except ValueError:
        pass
    else:
        raise CaseStudyTransitionError("ledger cannot be stored inside the CAS")
    paths = case_admission_transition_paths(
        restricted_root=root,
        transition_directory=transition_directory,
    )
    if ledger in {paths.h0_ledger, paths.h1_ledger}:
        raise CaseStudyTransitionError("live ledger cannot be a transition archive")
    try:
        paths.directory.relative_to(cas)
    except ValueError:
        pass
    else:
        raise CaseStudyTransitionError("transition archives cannot be stored inside the CAS")
    return root, ledger, cas, paths


def _normalise_payloads(
    payloads: Sequence[CaseStagedPayloadDescriptor],
) -> tuple[CaseStagedPayloadDescriptor, ...]:
    result = tuple(sorted(payloads, key=lambda item: item.role))
    roles = tuple(item.role for item in result)
    if roles != tuple(sorted(_EXPECTED_ROLES)):
        raise CaseStudyTransitionError("exactly nine unique staging payload roles are required")
    return result


def _payload_inventory_hash(payloads: Sequence[CaseStagedPayloadDescriptor]) -> str:
    return _hash_tree(tuple(json.loads(item.to_canonical_json()) for item in payloads))


def _require_intent_call_binding(
    intent: CaseAdmissionStagingIntent,
    *,
    transition_id: str,
    execution_plan_hash: str,
    admission_attestation_hash: str,
    semantic_bundle_hash: str,
    payloads: tuple[CaseStagedPayloadDescriptor, ...],
    evidence_bundle_hash: str,
    evidence_bundle_artifact_hash: str,
    evidence_bundle_reference_hash: str,
) -> None:
    observed = (
        intent.transition_id,
        intent.execution_plan_hash,
        intent.admission_attestation_hash,
        intent.semantic_bundle_hash,
        intent.expected_payloads,
        intent.evidence_bundle_hash,
        intent.evidence_bundle_artifact_hash,
        intent.evidence_bundle_reference_hash,
    )
    expected = (
        transition_id,
        execution_plan_hash,
        admission_attestation_hash,
        semantic_bundle_hash,
        payloads,
        evidence_bundle_hash,
        evidence_bundle_artifact_hash,
        evidence_bundle_reference_hash,
    )
    if observed != expected:
        raise CaseStudyTransitionError("existing staging intent conflicts with this call")


def _verify_h0_archives(
    *, root: Path, cas: Path, intent: CaseAdmissionStagingIntent
) -> tuple[_DatabaseState, CaseCasInventoryManifest]:
    snapshot = _path_from_relative(root, intent.h0_ledger.snapshot_relative_path)
    _require_singly_linked_regular_file(snapshot, label="archived H0 ledger")
    manifest_path = _path_from_relative(root, intent.h0_cas_manifest_relative_path)
    if _file_sha256(snapshot) != intent.h0_ledger.file_sha256:
        raise CaseStudyTransitionError("archived H0 ledger hash changed")
    state = _read_database_state(snapshot)
    if _ledger_inventory_hash(state) != _ledger_inventory_hash_from_record(intent.h0_ledger):
        raise CaseStudyTransitionError("archived H0 logical inventory changed")
    manifest = _load_record(manifest_path, CaseCasInventoryManifest)
    if manifest.content_hash != intent.h0_cas_manifest_hash:
        raise CaseStudyTransitionError("archived H0 CAS manifest hash changed")
    _require_valid_ledger(snapshot, cas, label="archived H0")
    return state, manifest


def _verify_transition_archives(
    *,
    root: Path,
    cas: Path,
    intent: CaseAdmissionStagingIntent,
    receipt: CaseAdmissionStagingReceipt,
) -> None:
    if (
        receipt.transition_id != intent.transition_id
        or receipt.intent_hash != intent.content_hash
        or receipt.execution_plan_hash != intent.execution_plan_hash
        or receipt.admission_attestation_hash != intent.admission_attestation_hash
        or receipt.semantic_bundle_hash != intent.semantic_bundle_hash
        or receipt.h0_ledger_hash != intent.h0_ledger.content_hash
        or receipt.h0_cas_manifest_hash != intent.h0_cas_manifest_hash
        or receipt.expected_payload_inventory_hash != intent.expected_payload_inventory_hash
        or receipt.evidence_bundle_hash != intent.evidence_bundle_hash
        or receipt.evidence_bundle_artifact_hash != intent.evidence_bundle_artifact_hash
        or receipt.evidence_bundle_reference_hash != intent.evidence_bundle_reference_hash
    ):
        raise CaseStudyTransitionError("staging receipt is not bound to its intent")
    if receipt.completed_at < intent.captured_at:
        raise CaseStudyTransitionError("staging completion predates the H0 intent")
    _verify_h0_archives(root=root, cas=cas, intent=intent)
    h1_path = _path_from_relative(root, receipt.h1_ledger.snapshot_relative_path)
    _require_singly_linked_regular_file(h1_path, label="archived H1 ledger")
    if _file_sha256(h1_path) != receipt.h1_ledger.file_sha256:
        raise CaseStudyTransitionError("archived H1 ledger hash changed")
    h1_state = _read_database_state(h1_path)
    if _ledger_inventory_hash(h1_state) != _ledger_inventory_hash_from_record(receipt.h1_ledger):
        raise CaseStudyTransitionError("archived H1 logical inventory changed")
    h1_cas_path = _path_from_relative(root, receipt.h1_cas_manifest_relative_path)
    h1_cas = _load_record(h1_cas_path, CaseCasInventoryManifest)
    if h1_cas.content_hash != receipt.h1_cas_manifest_hash:
        raise CaseStudyTransitionError("archived H1 CAS manifest hash changed")
    _require_valid_ledger(h1_path, cas, label="archived H1")
    delta = _prove_staging_delta(
        root=root,
        cas=cas,
        intent=intent,
        h1_ledger=h1_path,
        h1_cas=h1_cas,
    )
    if delta != (
        receipt.added_artifacts,
        receipt.reused_expected_artifact_hashes,
        receipt.added_cas_entries,
    ):
        raise CaseStudyTransitionError("staging receipt delta does not match archived H0/H1")
    for artifact in receipt.added_artifacts:
        try:
            created_at = datetime.fromisoformat(
                artifact.created_at[:-1] + "+00:00"
                if artifact.created_at.endswith("Z")
                else artifact.created_at
            )
        except ValueError as exc:
            raise CaseStudyTransitionError("staged artifact time is not ISO-8601") from exc
        if created_at.tzinfo is None or created_at.utcoffset() is None:
            raise CaseStudyTransitionError("staged artifact time has no UTC offset")
        if not intent.captured_at <= created_at <= receipt.completed_at:
            raise CaseStudyTransitionError("staged artifact time falls outside H0/H1")
    if (
        receipt.gpu_inventory_hash != h1_state.gpu_inventory_hash
        or receipt.allocated_gpu_microseconds != h1_state.allocated_gpu_microseconds
    ):
        raise CaseStudyTransitionError("staging receipt GPU inventory changed")


def _prove_staging_delta(
    *,
    root: Path,
    cas: Path,
    intent: CaseAdmissionStagingIntent,
    h1_ledger: Path,
    h1_cas: CaseCasInventoryManifest,
) -> tuple[
    tuple[LedgerArtifactInventoryEntry, ...],
    tuple[str, ...],
    tuple[CasFileInventoryEntry, ...],
]:
    h0_state, h0_cas = _verify_h0_archives(root=root, cas=cas, intent=intent)
    h1_state = _read_database_state(h1_ledger)
    if h0_state.schema_hash != h1_state.schema_hash or set(h0_state.rows) != set(h1_state.rows):
        raise CaseStudyTransitionError("ledger schema changed during admission staging")
    if (
        h0_state.gpu_inventory_hash != h1_state.gpu_inventory_hash
        or h0_state.allocated_gpu_microseconds != h1_state.allocated_gpu_microseconds
    ):
        raise CaseStudyTransitionError("GPU allocation inventory changed during staging")
    for table in sorted(h0_state.rows):
        if table != "artifacts" and h0_state.rows[table] != h1_state.rows[table]:
            raise CaseStudyTransitionError(
                f"unexpected non-artifact ledger delta during staging: {table}"
            )

    h0_hashes = set(h0_state.artifacts)
    h1_hashes = set(h1_state.artifacts)
    if not h0_hashes <= h1_hashes:
        raise CaseStudyTransitionError("an H0 artifact row disappeared during staging")
    for digest in h0_hashes:
        if h0_state.artifacts[digest] != h1_state.artifacts[digest]:
            raise CaseStudyTransitionError("an H0 artifact row changed during staging")
    expected_hashes = {item.artifact_hash for item in intent.expected_payloads}
    newly_expected = expected_hashes - h0_hashes
    if h1_hashes - h0_hashes != newly_expected:
        raise CaseStudyTransitionError("artifact-table delta is not the declared staging set")
    if not expected_hashes <= h1_hashes:
        raise CaseStudyTransitionError("one or more declared admission objects are absent")

    descriptors_by_hash: dict[str, list[CaseStagedPayloadDescriptor]] = collections.defaultdict(
        list
    )
    for descriptor in intent.expected_payloads:
        descriptors_by_hash[descriptor.artifact_hash].append(descriptor)
    decoded_by_role: dict[str, Mapping[str, Any]] = {}
    for digest, descriptors in descriptors_by_hash.items():
        row = h1_state.artifacts[digest]
        for descriptor in descriptors:
            decoded_by_role[descriptor.role] = _verify_payload_descriptor(
                descriptor,
                row=row,
                cas=cas,
            )
    _validate_evidence_bundle_payload(intent, decoded_by_role[_BUNDLE_ROLE])

    h0_files = {entry.relative_path: entry for entry in h0_cas.entries}
    h1_files = {entry.relative_path: entry for entry in h1_cas.entries}
    if not set(h0_files) <= set(h1_files):
        raise CaseStudyTransitionError("an H0 CAS file disappeared during staging")
    for path, entry in h0_files.items():
        if h1_files[path] != entry:
            raise CaseStudyTransitionError("an H0 CAS file changed during staging")
    expected_new_paths = {
        descriptor.expected_relative_path
        for descriptor in intent.expected_payloads
        if descriptor.expected_relative_path not in h0_files
    }
    actual_new_paths = set(h1_files) - set(h0_files)
    if actual_new_paths != expected_new_paths:
        raise CaseStudyTransitionError("CAS delta is not the declared staging set")
    current_files = {entry.relative_path: entry for entry in _scan_cas(cas).entries}
    for path, entry in h1_files.items():
        if current_files.get(path) != entry:
            raise CaseStudyTransitionError("an archived H1 CAS object is missing or changed")

    added_artifacts = tuple(
        _artifact_entry(h1_state.artifacts[digest]) for digest in sorted(newly_expected)
    )
    reused = tuple(sorted(expected_hashes & h0_hashes))
    added_cas = tuple(h1_files[path] for path in sorted(actual_new_paths))
    return added_artifacts, reused, added_cas


def _verify_payload_descriptor(
    descriptor: CaseStagedPayloadDescriptor | CasePostH1ArtifactDescriptor,
    *,
    row: Mapping[str, Any],
    cas: Path,
) -> Mapping[str, Any]:
    expected = {
        "content_hash": descriptor.artifact_hash,
        "compression": descriptor.compression,
        "media_type": descriptor.media_type,
        "raw_size_bytes": descriptor.raw_size_bytes,
        "relative_path": descriptor.expected_relative_path,
        "release_class": descriptor.release_class,
    }
    for key, value in expected.items():
        if row.get(key) != value:
            raise CaseStudyTransitionError(
                f"staged payload metadata mismatch for {descriptor.role}: {key}"
            )
    value = _verify_payload_bytes(descriptor, cas=cas)
    return value


def _verify_payload_bytes(
    descriptor: CaseStagedPayloadDescriptor | CasePostH1ArtifactDescriptor,
    *,
    cas: Path,
) -> Mapping[str, Any]:
    encoded = _read_regular_bytes(cas / descriptor.expected_relative_path)
    try:
        if descriptor.compression == "gzip":
            raw = gzip.decompress(encoded)
        else:
            import zstandard  # type: ignore[import-not-found]

            raw = zstandard.ZstdDecompressor().decompress(encoded)
    except Exception as exc:
        raise CaseStudyTransitionError(f"cannot decode staged payload {descriptor.role}") from exc
    if len(raw) != descriptor.raw_size_bytes:
        raise CaseStudyTransitionError(f"staged payload size mismatch: {descriptor.role}")
    if hashlib.sha256(raw).hexdigest() != descriptor.artifact_hash:
        raise CaseStudyTransitionError(f"staged payload hash mismatch: {descriptor.role}")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CaseStudyTransitionError(f"staged payload is not JSON: {descriptor.role}") from exc
    if not isinstance(value, Mapping):
        raise CaseStudyTransitionError(f"staged payload is not a JSON object: {descriptor.role}")
    if _logical_json_hash(value) != descriptor.logical_content_hash:
        raise CaseStudyTransitionError(f"logical staged payload hash mismatch: {descriptor.role}")
    return value


def _validate_evidence_bundle_payload(
    intent: CaseAdmissionStagingIntent,
    value: Mapping[str, Any],
) -> None:
    if value.get("admission_attestation_hash") != intent.admission_attestation_hash:
        raise CaseStudyTransitionError("evidence bundle belongs to another admission attestation")
    raw_evidence = value.get("evidence")
    if not isinstance(raw_evidence, list) or len(raw_evidence) != len(_GATE_ROLES):
        raise CaseStudyTransitionError("evidence bundle does not contain exactly eight gates")
    observed: dict[str, tuple[Any, Any]] = {}
    for item in raw_evidence:
        if not isinstance(item, Mapping):
            raise CaseStudyTransitionError("evidence bundle gate reference is not an object")
        name = item.get("name")
        if not isinstance(name, str) or name in observed:
            raise CaseStudyTransitionError("evidence bundle gate names are invalid or repeated")
        observed[name] = (
            item.get("logical_content_hash"),
            item.get("artifact_hash"),
        )
    expected = {
        descriptor.role: (
            descriptor.logical_content_hash,
            descriptor.artifact_hash,
        )
        for descriptor in intent.expected_payloads
        if descriptor.role in _GATE_ROLES
    }
    if observed != expected:
        raise CaseStudyTransitionError("evidence bundle references differ from staged gates")
    expected_reference_hash = canonical_sha256(
        {
            "schema_version": "1.0.0",
            "logical_content_hash": intent.evidence_bundle_hash,
            "artifact_hash": intent.evidence_bundle_artifact_hash,
            "object_kind": "case_admission_evidence_bundle",
        }
    )
    if intent.evidence_bundle_reference_hash != expected_reference_hash:
        raise CaseStudyTransitionError("evidence bundle reference hash is not reproducible")


def _logical_json_hash(value: Mapping[str, Any]) -> str:
    for hash_field in ("content_hash", "manifest_sha256"):
        advertised = value.get(hash_field)
        if isinstance(advertised, str):
            payload = {key: item for key, item in value.items() if key != hash_field}
            observed = canonical_sha256(payload)
            if advertised != observed:
                raise CaseStudyTransitionError(f"invalid embedded {hash_field}")
            return observed
    return canonical_sha256(value)


def _artifact_entry(row: Mapping[str, Any]) -> LedgerArtifactInventoryEntry:
    return LedgerArtifactInventoryEntry(
        artifact_hash=row["content_hash"],
        compression=row["compression"],
        media_type=row["media_type"],
        raw_size_bytes=row["raw_size_bytes"],
        stored_size_bytes=row["stored_size_bytes"],
        relative_path=row["relative_path"],
        release_class=row["release_class"],
        created_at=row["created_at"],
    )


def _read_database_state(path: Path) -> _DatabaseState:
    database = _bounded_existing_file(path, root=path.parent, label="ledger snapshot")
    uri = f"file:{quote(str(database), safe='/')}?mode=ro&immutable=1"
    try:
        connection = sqlite3.connect(uri, uri=True, isolation_level=None, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("BEGIN")
        integrity = tuple(row[0] for row in connection.execute("PRAGMA integrity_check"))
        if integrity != ("ok",):
            raise CaseStudyTransitionError(f"SQLite integrity check failed: {integrity!r}")
        if tuple(connection.execute("PRAGMA foreign_key_check")):
            raise CaseStudyTransitionError("SQLite foreign-key check failed")
        schema_rows = tuple(
            dict(row)
            for row in connection.execute(
                """SELECT type, name, tbl_name, sql FROM sqlite_master
                   WHERE name NOT LIKE 'sqlite_%'
                   ORDER BY type, name"""
            )
        )
        schema_hash = _hash_tree(schema_rows)
        table_names = tuple(row["name"] for row in schema_rows if row["type"] == "table")
        rows: dict[str, tuple[str, ...]] = {}
        decoded_rows: dict[str, tuple[Mapping[str, Any], ...]] = {}
        for table in sorted(table_names):
            identifier = _sqlite_identifier(table)
            columns = tuple(
                row["name"] for row in connection.execute(f"PRAGMA table_info({identifier})")
            )
            table_rows = tuple(
                {column: _sqlite_value(row[column]) for column in columns}
                for row in connection.execute(f"SELECT * FROM {identifier}")
            )
            encoded = tuple(sorted(canonical_json(row) for row in table_rows))
            rows[table] = encoded
            decoded_rows[table] = tuple(json.loads(item) for item in encoded)
        artifacts = {str(row["content_hash"]): row for row in decoded_rows.get("artifacts", ())}
        gpu_inventory = tuple(
            {
                "table": table,
                "rows": rows.get(table, ()),
            }
            for table in _GPU_TABLES
        )
        event_micros = sum(
            int(row["allocated_microseconds"]) for row in decoded_rows.get("gpu_events", ())
        )
        overhead_micros = sum(
            int(row["overhead_microseconds"])
            for row in decoded_rows.get("gpu_service_sessions", ())
        )
        return _DatabaseState(
            schema_hash=schema_hash,
            rows=rows,
            artifacts=artifacts,
            gpu_inventory_hash=_hash_tree(gpu_inventory),
            allocated_gpu_microseconds=event_micros + overhead_micros,
        )
    except sqlite3.Error as exc:
        raise CaseStudyTransitionError(f"cannot read immutable SQLite ledger: {exc}") from exc
    finally:
        if "connection" in locals():
            connection.close()


def _sqlite_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CaseStudyTransitionError("SQLite inventory contains a non-finite number")
        return value
    if isinstance(value, bytes):
        return {"sqlite_blob_base64": base64.b64encode(value).decode("ascii")}
    raise CaseStudyTransitionError(f"unsupported SQLite value type: {type(value).__name__}")


def _sqlite_identifier(value: Any) -> str:
    if not isinstance(value, str) or "\x00" in value:
        raise CaseStudyTransitionError("SQLite schema contains an invalid identifier")
    return '"' + value.replace('"', '""') + '"'


def _ledger_inventory(
    *, state: _DatabaseState, snapshot_path: Path, root: Path
) -> LedgerSnapshotInventory:
    return LedgerSnapshotInventory(
        snapshot_relative_path=_relative(snapshot_path, root),
        file_sha256=_file_sha256(snapshot_path),
        schema_inventory_hash=state.schema_hash,
        tables=tuple(
            LedgerTableInventory(
                table_name=table,
                row_count=len(rows),
                row_inventory_hash=_hash_tree(rows),
            )
            for table, rows in sorted(state.rows.items())
        ),
        gpu_inventory_hash=state.gpu_inventory_hash,
        allocated_gpu_microseconds=state.allocated_gpu_microseconds,
    )


def _ledger_inventory_hash(state: _DatabaseState) -> str:
    return _hash_tree(
        {
            "schema_inventory_hash": state.schema_hash,
            "tables": tuple(
                {
                    "table_name": table,
                    "row_count": len(rows),
                    "row_inventory_hash": _hash_tree(rows),
                }
                for table, rows in sorted(state.rows.items())
            ),
            "gpu_inventory_hash": state.gpu_inventory_hash,
            "allocated_gpu_microseconds": state.allocated_gpu_microseconds,
        }
    )


def _ledger_inventory_hash_from_record(record: LedgerSnapshotInventory) -> str:
    return _hash_tree(
        {
            "schema_inventory_hash": record.schema_inventory_hash,
            "tables": tuple(
                {
                    "table_name": table.table_name,
                    "row_count": table.row_count,
                    "row_inventory_hash": table.row_inventory_hash,
                }
                for table in record.tables
            ),
            "gpu_inventory_hash": record.gpu_inventory_hash,
            "allocated_gpu_microseconds": record.allocated_gpu_microseconds,
        }
    )


def _scan_cas(root: Path) -> CaseCasInventoryManifest:
    entries: list[CasFileInventoryEntry] = []
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in tuple(directory_names):
            candidate = directory_path / name
            if candidate.is_symlink():
                raise CaseStudyTransitionError("CAS inventory cannot traverse symbolic links")
        for name in file_names:
            candidate = directory_path / name
            info = candidate.lstat()
            if (
                stat.S_ISLNK(info.st_mode)
                or not stat.S_ISREG(info.st_mode)
                or info.st_nlink != 1
            ):
                raise CaseStudyTransitionError(
                    "CAS inventory accepts singly-linked regular files only"
                )
            relative = candidate.relative_to(root).as_posix()
            entries.append(
                CasFileInventoryEntry(
                    relative_path=relative,
                    file_sha256=_file_sha256(candidate),
                    size_bytes=info.st_size,
                )
            )
    entries.sort(key=lambda item: item.relative_path)
    return CaseCasInventoryManifest(entries=tuple(entries))


def _expected_h1_cas_manifest(
    *,
    root: Path,
    intent: CaseAdmissionStagingIntent,
    current: CaseCasInventoryManifest,
) -> CaseCasInventoryManifest:
    """Reconstruct the only valid H1 CAS boundary from H0 and its intent."""

    current_by_path = {entry.relative_path: entry for entry in current.entries}
    expected_paths = {
        item.expected_relative_path for item in intent.expected_payloads
    }
    h0_path = _path_from_relative(root, intent.h0_cas_manifest_relative_path)
    h0 = _load_record(h0_path, CaseCasInventoryManifest)
    expected_paths.update(item.relative_path for item in h0.entries)
    try:
        entries = tuple(current_by_path[path] for path in sorted(expected_paths))
    except KeyError as exc:
        raise CaseStudyTransitionError("partial H1 CAS lacks a declared object") from exc
    return CaseCasInventoryManifest(entries=entries)


def _is_append_only_successor(before: _DatabaseState, after: _DatabaseState) -> bool:
    if before.schema_hash != after.schema_hash or set(before.rows) != set(after.rows):
        return False
    for table in before.rows:
        before_counts = collections.Counter(before.rows[table])
        after_counts = collections.Counter(after.rows[table])
        if any(after_counts[row] < count for row, count in before_counts.items()):
            return False
    return True


def _verify_live_successor(
    *,
    ledger: Path,
    cas: Path,
    h1_ledger: Path,
    h1_cas: CaseCasInventoryManifest,
) -> None:
    _require_quiescent_sqlite(ledger)
    _require_valid_ledger(ledger, cas, label="current successor")
    if not _is_append_only_successor(
        _read_database_state(h1_ledger),
        _read_database_state(ledger),
    ):
        raise CaseStudyTransitionError("current ledger is not an append-only successor of H1")
    _require_cas_superset(h1_cas, _scan_cas(cas))


def _require_cas_superset(
    before: CaseCasInventoryManifest, after: CaseCasInventoryManifest
) -> None:
    after_by_path = {entry.relative_path: entry for entry in after.entries}
    for entry in before.entries:
        if after_by_path.get(entry.relative_path) != entry:
            raise CaseStudyTransitionError("current CAS is not an immutable superset of H1")


def _require_valid_ledger(ledger: Path, cas: Path, *, label: str) -> None:
    report = audit_ledger(ledger, cas)
    if not report.valid:
        codes = ",".join(sorted({issue.code for issue in report.issues}))
        raise CaseStudyTransitionError(f"{label} ledger/CAS audit failed: {codes}")


def _require_quiescent_sqlite(path: Path) -> None:
    for suffix in ("-wal", "-journal"):
        companion = Path(str(path) + suffix)
        if companion.is_symlink():
            raise CaseStudyTransitionError("SQLite companion cannot be a symbolic link")
        if companion.exists() and companion.stat().st_size:
            raise CaseStudyTransitionError("SQLite ledger must be closed and checkpointed")


def _hash_tree(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _real_directory(path: Path, *, label: str) -> Path:
    lexical = Path(os.path.abspath(path))
    if lexical.is_symlink() or not lexical.is_dir():
        raise CaseStudyTransitionError(f"{label} must be an existing real directory")
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as exc:
        raise CaseStudyTransitionError(f"cannot resolve {label}") from exc
    if resolved != lexical:
        raise CaseStudyTransitionError(f"{label} cannot traverse symbolic links")
    return lexical


def _bounded_path(path: Path, *, root: Path, label: str) -> Path:
    lexical = Path(os.path.abspath(path))
    try:
        relative = lexical.relative_to(root)
    except ValueError as exc:
        raise CaseStudyTransitionError(f"{label} must remain under restricted root") from exc
    cursor = root
    for part in relative.parts:
        cursor /= part
        if cursor.exists() and cursor.is_symlink():
            raise CaseStudyTransitionError(f"{label} cannot traverse symbolic links")
    return lexical


def _bounded_existing_file(path: Path, *, root: Path, label: str) -> Path:
    bounded = _bounded_path(path, root=root, label=label)
    try:
        info = bounded.lstat()
    except OSError as exc:
        raise CaseStudyTransitionError(f"{label} is not an existing file") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise CaseStudyTransitionError(f"{label} must be a regular non-symlink file")
    return bounded


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError as exc:
        raise CaseStudyTransitionError("record path escapes restricted root") from exc


def _path_from_relative(root: Path, value: str) -> Path:
    relative = _validate_relative_path(value, label="recorded path")
    return _bounded_existing_file(root / relative, root=root, label="recorded file")


def _read_regular_bytes(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise CaseStudyTransitionError(f"cannot open immutable file {path.name}") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise CaseStudyTransitionError("immutable input is not a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise CaseStudyTransitionError(f"cannot hash immutable file {path.name}") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise CaseStudyTransitionError("immutable input is not a regular file")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _publish_copy_no_replace(source: Path, destination: Path) -> None:
    payload = _read_regular_bytes(source)
    _publish_bytes_no_replace(destination, payload)
    _require_independent_ledger_archive(
        source=source,
        archive=destination,
        label="ledger",
    )


def _require_singly_linked_regular_file(path: Path, *, label: str) -> None:
    try:
        info = path.lstat()
    except OSError as exc:
        raise CaseStudyTransitionError(f"{label} is unavailable") from exc
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise CaseStudyTransitionError(
            f"{label} must be one singly-linked regular file"
        )


def _repair_interrupted_transition_publications(
    paths: CaseAdmissionTransitionPaths,
) -> None:
    """Remove only a publisher-owned hardlink left by an interrupted publish."""

    for path in (
        paths.intent,
        paths.h0_ledger,
        paths.h0_cas_manifest,
        paths.receipt,
        paths.h1_ledger,
        paths.h1_cas_manifest,
    ):
        if not path.exists():
            continue
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink == 1:
            continue
        # ``_publish_bytes_no_replace`` creates exactly this private temporary
        # name and then hardlinks it to the final name. A power loss may persist
        # both links. Never remove arbitrary hardlinks or ambiguous remnants.
        prefix = f".{path.name}."
        suffix = ".partial"
        candidates: list[Path] = []
        for candidate in path.parent.iterdir():
            name = candidate.name
            token = name[len(prefix) : -len(suffix)]
            if (
                not name.startswith(prefix)
                or not name.endswith(suffix)
                or len(token) != 32
                or any(character not in "0123456789abcdef" for character in token)
            ):
                continue
            candidate_info = candidate.lstat()
            if (
                stat.S_ISREG(candidate_info.st_mode)
                and (candidate_info.st_dev, candidate_info.st_ino)
                == (info.st_dev, info.st_ino)
            ):
                candidates.append(candidate)
        if info.st_nlink != 2 or len(candidates) != 1:
            continue
        candidates[0].unlink()
        _fsync_directory(path.parent)


def _require_independent_ledger_archive(
    *, source: Path, archive: Path, label: str
) -> None:
    _require_singly_linked_regular_file(archive, label=f"{label} archive")
    source_info = source.stat(follow_symlinks=False)
    archive_info = archive.stat(follow_symlinks=False)
    if (source_info.st_dev, source_info.st_ino) == (
        archive_info.st_dev,
        archive_info.st_ino,
    ):
        raise CaseStudyTransitionError(f"{label} archive aliases its live source")


def _publish_record_no_replace(path: Path, record: ImmutableRecord) -> None:
    _publish_bytes_no_replace(path, record.to_canonical_json().encode("utf-8") + b"\n")


def _publish_bytes_no_replace(path: Path, payload: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.parent.is_symlink() or path.parent.resolve(strict=True) != path.parent:
        raise CaseStudyTransitionError("immutable output parent cannot traverse symbolic links")
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.partial"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary, flags, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as exc:
            if _read_regular_bytes(path) != payload:
                raise CaseStudyTransitionError(f"immutable output conflicts: {path.name}") from exc
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
        _fsync_directory(path.parent)
    _require_singly_linked_regular_file(path, label="immutable output")


def _load_record(path: Path, record_type: type[ImmutableRecord]) -> Any:
    _require_singly_linked_regular_file(path, label="immutable record")
    try:
        return record_type.model_validate_json(_read_regular_bytes(path))
    except (ValueError, TypeError) as exc:
        raise CaseStudyTransitionError(f"invalid immutable record: {path.name}") from exc


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


__all__ = [
    "CaseAdmissionStagingIntent",
    "CaseAdmissionStagingReceipt",
    "CaseAdmissionTransitionPaths",
    "CaseCasInventoryManifest",
    "CasePostH1ArtifactDescriptor",
    "CaseStagedPayloadDescriptor",
    "CaseStudyTransitionError",
    "VerifiedCaseAdmissionStagingTransition",
    "VerifiedCasePostH1ArtifactPrefix",
    "capture_case_admission_staging_intent",
    "case_admission_transition_paths",
    "finalize_case_admission_staging_transition",
    "verify_case_admission_staging_transition",
    "verify_case_post_h1_artifact_prefix",
]
