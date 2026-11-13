"""Append-only executor for a reviewed held-out primary call manifest.

No implementation in this module can start a model service.  Three injected
sessions, one per GPU condition, own live execution and global GPU metering.
"""

from __future__ import annotations

import math
import os
import tempfile
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, Protocol, Self

from pydantic import AwareDatetime, model_validator

from story_projection_onto.contracts import (
    ConditionName,
    Identifier,
    ImmutableRecord,
    PrequeryBarrier,
    PrequeryPreparationBinding,
    Sha256Digest,
)
from story_projection_onto.development_runtime import RunOutcome
from story_projection_onto.held_out_primary import (
    PRIMARY_SERVICE_START_P95_SECONDS,
    GlobalGpuScheduleSnapshot,
    HeldOutAblationPrequeryReceipt,
    HeldOutC2PrequeryReceipt,
    HeldOutCallEnvelope,
    HeldOutCallManifest,
    HeldOutCallSpec,
    HeldOutControlConfiguration,
    HeldOutControlError,
    HeldOutPrequeryUnit,
    HeldOutQueryOpening,
    HeldOutServiceResult,
    HeldOutServiceShutdownReceipt,
    HeldOutSessionIdentity,
    HeldOutUnitPlan,
    InjectedHeldOutRuntime,
    InjectedHeldOutSession,
    ReviewedHeldOutPlan,
    admit_call,
    load_executable_held_out_configuration,
    load_held_out_control_configuration,
    open_reviewed_held_out_plan,
)


class HeldOutExecutionError(HeldOutControlError):
    """The append-only held-out execution cannot safely continue."""


class InterruptedCallRecoveryRequired(HeldOutExecutionError):
    """A durable call slot exists but its service receipt cannot be recovered."""


class C0ConstructionReceipt(ImmutableRecord):
    unit_id: Identifier
    prequery_stage_hash: Sha256Digest
    outcome: RunOutcome
    construction_seal_hash: Sha256Digest | None = None
    complete_graph_hash: Sha256Digest | None = None
    failure_artifact_hash: Sha256Digest | None = None
    completed_at: AwareDatetime
    deterministic_cpu_only: Literal[True] = True
    allocated_gpu_seconds: Literal[0.0] = 0.0

    @model_validator(mode="after")
    def terminal(self) -> Self:
        if self.outcome not in {
            RunOutcome.SUCCEEDED,
            RunOutcome.INVALID,
            RunOutcome.FAILED,
            RunOutcome.TIMED_OUT,
        }:
            raise ValueError("C0 construction receipt must be terminal")
        succeeded = self.outcome is RunOutcome.SUCCEEDED
        if succeeded != (
            self.construction_seal_hash is not None and self.complete_graph_hash is not None
        ):
            raise ValueError("successful C0 construction requires seal and complete graph")
        if succeeded == (self.failure_artifact_hash is not None):
            raise ValueError("exactly failed C0 construction requires a failure artifact")
        return self


class PreconstructedProjectionReceipt(ImmutableRecord):
    unit_id: Identifier
    query_stage_hash: Sha256Digest
    condition: Literal[ConditionName.C0_CLASSICAL_PRE, ConditionName.C1_LLM_PRE]
    seed_block: Literal[1, 2] | None
    source_construction_seal_hash: Sha256Digest | None
    source_complete_graph_hash: Sha256Digest | None
    outcome: RunOutcome
    projection_artifact_hash: Sha256Digest | None = None
    evidence_packet_hash: Sha256Digest
    horizon_hash: Sha256Digest
    budget_hash: Sha256Digest
    failure_artifact_hash: Sha256Digest | None = None
    completed_at: AwareDatetime
    allocated_gpu_seconds: Literal[0.0] = 0.0

    @model_validator(mode="after")
    def fixed_projection_only(self) -> Self:
        if self.outcome not in {
            RunOutcome.SUCCEEDED,
            RunOutcome.INVALID,
            RunOutcome.FAILED,
            RunOutcome.TIMED_OUT,
        }:
            raise ValueError("preconstructed projection receipt must be terminal")
        if (self.condition is ConditionName.C0_CLASSICAL_PRE) != (self.seed_block is None):
            raise ValueError("only deterministic C0 projection omits an LLM seed")
        succeeded = self.outcome is RunOutcome.SUCCEEDED
        if succeeded != (self.projection_artifact_hash is not None):
            raise ValueError("only successful preconstructed projection has an output")
        if succeeded and (
            self.source_construction_seal_hash is None or self.source_complete_graph_hash is None
        ):
            raise ValueError("successful projection must reuse a sealed complete graph")
        if not succeeded and self.failure_artifact_hash is None:
            raise ValueError("failed CPU projection requires a retained failure artifact")
        return self


class InjectedHeldOutCpu(Protocol):
    """Gold-free deterministic C0 construction and sealed-graph projection."""

    def build_c0(self, unit: HeldOutPrequeryUnit) -> C0ConstructionReceipt: ...

    def project_preconstructed(
        self,
        *,
        unit: HeldOutUnitPlan,
        query_stage_hash: str,
        query_opening: HeldOutQueryOpening,
        condition: ConditionName,
        seed_block: int | None,
        construction_seal_hash: str,
        complete_graph_hash: str,
    ) -> PreconstructedProjectionReceipt: ...

    def project_unavailable(
        self,
        *,
        unit: HeldOutUnitPlan,
        query_stage_hash: str,
        query_opening: HeldOutQueryOpening,
        condition: ConditionName,
        seed_block: int | None,
        source_failure_hash: str,
    ) -> PreconstructedProjectionReceipt: ...


class HeldOutCallSlot(ImmutableRecord):
    """Durable pre-request record used to recover without duplicate GPU work."""

    ordinal: int
    call_spec_hash: Sha256Digest
    envelope: HeldOutCallEnvelope
    session_identity: HeldOutSessionIdentity
    schedule_before: GlobalGpuScheduleSnapshot

    @model_validator(mode="after")
    def hashes_and_route_match(self) -> Self:
        if self.call_spec_hash != self.envelope.call_spec_hash:
            raise ValueError("call slot and execution envelope name different calls")
        if self.schedule_before.global_accounting_id != self.session_identity.global_accounting_id:
            raise ValueError("call slot schedule uses another GPU accounting scope")
        return self


class HeldOutITTRecord(ImmutableRecord):
    ordinal: int
    call_spec_hash: Sha256Digest
    result: HeldOutServiceResult
    session_identity_hash: Sha256Digest
    call_slot_hash: Sha256Digest
    schedule_before: GlobalGpuScheduleSnapshot
    schedule_after: GlobalGpuScheduleSnapshot
    included_in_intention_to_treat: Literal[True] = True

    @model_validator(mode="after")
    def terminal_schedule_is_ordered(self) -> Self:
        if self.schedule_after.captured_at < self.schedule_before.captured_at:
            raise ValueError("ITT schedule cannot move backward in time")
        return self


class ScorerBridgeAuthorization(ImmutableRecord):
    """Hash-only handoff emitted after every primary output is frozen."""

    execution_manifest_hash: Sha256Digest
    call_manifest_hash: Sha256Digest
    final_reviewed_seal_hash: Sha256Digest
    output_artifact_hashes: tuple[Sha256Digest, ...]
    c0_receipt_hashes: tuple[Sha256Digest, ...]
    projection_receipt_hashes: tuple[Sha256Digest, ...]
    c2_prequery_receipt_hashes: tuple[Sha256Digest, ...]
    ablation_prequery_receipt_hashes: tuple[Sha256Digest, ...]
    prequery_barrier_hash: Sha256Digest
    query_opening_hashes: tuple[Sha256Digest, ...]
    service_shutdown_receipt_hashes: tuple[Sha256Digest, ...]
    itt_record_hashes: tuple[Sha256Digest, ...]
    authorized_at: AwareDatetime
    runtime_namespace_closed: Literal[True] = True
    model_input_open: Literal[False] = False
    scorer_namespace: Literal["scorer_only"] = "scorer_only"

    @model_validator(mode="after")
    def complete_itt_inventory(self) -> Self:
        if (
            len(self.c0_receipt_hashes) != 12
            or len(self.projection_receipt_hashes) != 108
            or len(self.c2_prequery_receipt_hashes) != 24
            or len(self.ablation_prequery_receipt_hashes) != 36
            or len(self.query_opening_hashes) != 36
            or len(self.service_shutdown_receipt_hashes) != 3
            or len(self.itt_record_hashes) != 168
        ):
            raise ValueError("scorer bridge requires every frozen success/failure receipt")
        return self


class HeldOutExecutionManifest(ImmutableRecord):
    execution_id: Identifier
    call_manifest_hash: Sha256Digest
    review_completion_manifest_hash: Sha256Digest
    final_reviewed_seal_hash: Sha256Digest
    runtime_binding_hash: Sha256Digest | None = None
    session_identities: tuple[
        HeldOutSessionIdentity, HeldOutSessionIdentity, HeldOutSessionIdentity
    ]
    c0_constructions: tuple[C0ConstructionReceipt, ...]
    c2_prequery_receipts: tuple[HeldOutC2PrequeryReceipt, ...]
    ablation_prequery_receipts: tuple[HeldOutAblationPrequeryReceipt, ...]
    prequery_barrier: PrequeryBarrier
    query_openings: tuple[HeldOutQueryOpening, ...]
    service_shutdown_receipts: tuple[
        HeldOutServiceShutdownReceipt,
        HeldOutServiceShutdownReceipt,
        HeldOutServiceShutdownReceipt,
    ]
    preconstructed_projections: tuple[PreconstructedProjectionReceipt, ...]
    itt_records: tuple[HeldOutITTRecord, ...]
    initial_schedule_snapshot_hash: Sha256Digest
    final_schedule_snapshot_hash: Sha256Digest
    completed_at: AwareDatetime
    runtime_namespace: Literal["gold_free"] = "gold_free"

    @model_validator(mode="after")
    def exact_primary_outputs(self) -> Self:
        if len(self.c0_constructions) != 12:
            raise ValueError("held-out execution requires twelve C0 constructions")
        if len(self.c2_prequery_receipts) != 24:
            raise ValueError("held-out execution requires 24 seeded C2 empty inventories")
        if len({(item.unit_id, item.seed_block) for item in self.c2_prequery_receipts}) != 24:
            raise ValueError("C2 pre-query inventory keys are not unique")
        expected_ablation_keys = {
            (unit.unit_id, condition)
            for unit in self.c0_constructions
            for condition in (
                ConditionName.A_NO_CONTEXT,
                ConditionName.A_NO_TEMPORAL_EPISTEMIC,
                ConditionName.A_NO_RARE_GUARD,
            )
        }
        if (
            len(self.ablation_prequery_receipts) != 36
            or {(item.unit_id, item.condition) for item in self.ablation_prequery_receipts}
            != expected_ablation_keys
        ):
            raise ValueError("held-out execution requires 36 query-blind ablation preparations")
        if (
            len(self.query_openings) != 36
            or len({item.sealed_stage_hash for item in self.query_openings}) != 36
        ):
            raise ValueError("held-out execution requires 36 unique query openings")
        if any(
            item.query_access_event.prequery_barrier_hash != self.prequery_barrier.content_hash
            or item.opened_at <= self.prequery_barrier.sealed_at
            for item in self.query_openings
        ):
            raise ValueError("query opening does not strictly follow the pre-query barrier")
        shutdowns = {item.condition: item for item in self.service_shutdown_receipts}
        if set(shutdowns) != {
            ConditionName.C1_LLM_PRE,
            ConditionName.C2_LLM_QUERY,
            ConditionName.A_FIXED_SELECT,
        }:
            raise ValueError("held-out execution requires three service shutdown receipts")
        if len(self.preconstructed_projections) != 108:
            raise ValueError("held-out execution requires 36 C0 and 72 C1 projections")
        projection_counts = Counter(item.condition for item in self.preconstructed_projections)
        if projection_counts != Counter(
            {ConditionName.C0_CLASSICAL_PRE: 36, ConditionName.C1_LLM_PRE: 72}
        ):
            raise ValueError("preconstructed projection inventory changed")
        if len(self.itt_records) != 168:
            raise ValueError("held-out execution requires 168 ITT model-call records")
        if tuple(item.ordinal for item in self.itt_records) != tuple(range(1, 169)):
            raise ValueError("held-out ITT records are not one complete ordered schedule")
        identities = {item.condition: item for item in self.session_identities}
        if set(identities) != {
            ConditionName.C1_LLM_PRE,
            ConditionName.C2_LLM_QUERY,
            ConditionName.A_FIXED_SELECT,
        }:
            raise ValueError("held-out execution requires three condition-separated sessions")
        if len({item.session_id for item in self.session_identities}) != 3:
            raise ValueError("condition sessions must be distinct")
        if len({item.service_identity_hash for item in self.session_identities}) != 3:
            raise ValueError("conditions cannot share one live-service identity")
        if len({item.allocation_event_id for item in self.session_identities}) != 3:
            raise ValueError("conditions require three distinct allocation events")
        if len({item.model_load_event_hash for item in self.session_identities}) != 3:
            raise ValueError("conditions require three distinct model-load events")
        if (
            len({(item.service_pid, item.service_start_ticks) for item in self.session_identities})
            != 3
        ):
            raise ValueError("conditions require three physically distinct service processes")
        if len({item.global_accounting_id for item in self.session_identities}) != 1:
            raise ValueError("condition sessions must share one global GPU accounting scope")
        identity_hashes = {item.condition: item.content_hash for item in self.session_identities}
        for condition, shutdown in shutdowns.items():
            identity = identities[condition]
            if (
                shutdown.session_identity_hash != identity.content_hash
                or shutdown.allocation_event_id != identity.allocation_event_id
                or shutdown.model_load_event_id != identity.model_load_event_id
                or shutdown.model_load_event_hash != identity.model_load_event_hash
                or shutdown.global_accounting_id != identity.global_accounting_id
            ):
                raise ValueError("service shutdown receipt differs from its allocation")
        if any(
            item.session_identity_hash != identity_hashes.get(item.result.condition)
            for item in self.itt_records
        ):
            raise ValueError("ITT records do not bind their condition's frozen service identity")
        if self.completed_at < max(
            (
                *(item.completed_at for item in self.c0_constructions),
                *(item.completed_at for item in self.c2_prequery_receipts),
                *(item.completed_at for item in self.ablation_prequery_receipts),
                self.prequery_barrier.sealed_at,
                *(item.opened_at for item in self.query_openings),
                *(item.stopped_at for item in self.service_shutdown_receipts),
                *(item.completed_at for item in self.preconstructed_projections),
                *(item.result.completed_at for item in self.itt_records),
            )
        ):
            raise ValueError("execution completion predates a retained primary receipt")
        return self


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _append_exact(path: Path, record: ImmutableRecord) -> bool:
    payload = record.to_canonical_json().encode() + b"\n"
    if path.is_symlink():
        raise HeldOutExecutionError(f"symlinked journal artifact: {path}")
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise HeldOutExecutionError(f"append-only held-out journal drift: {path}")
        return False
    parent_existed = path.parent.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink():
        raise HeldOutExecutionError(f"symlinked journal directory: {path.parent}")
    if not parent_existed:
        # Persist the newly-created journal-directory entry before relying on
        # any record contained by it after a power loss.
        _fsync_directory(path.parent.parent)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    published = False
    try:
        with os.fdopen(descriptor, "wb") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            # A hard link publishes the fully-fsynced inode atomically without
            # replacing a concurrently created append-only destination.
            os.link(temporary, path)
            published = True
        except FileExistsError:
            if not path.is_file() or path.read_bytes() != payload:
                raise HeldOutExecutionError(
                    f"append-only held-out journal drift: {path}"
                ) from None
        # Make the destination durable while the temporary hard link still
        # keeps the inode recoverable if power fails during publication.
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    _fsync_directory(path.parent)
    return published


def _load(path: Path, model_type):
    if path.is_symlink() or not path.is_file():
        raise HeldOutExecutionError(f"missing or symlinked held-out journal record: {path}")
    try:
        return model_type.model_validate_json(path.read_bytes())
    except Exception as error:
        raise HeldOutExecutionError(f"invalid held-out journal record {path}: {error}") from error


def _expected_journal_files(root: Path, manifest: HeldOutCallManifest) -> set[Path]:
    files = {
        root / "call_manifest.json",
        root / "initial_schedule.json",
        root / "prequery_barrier.json",
        root / "final_schedule.json",
        root / "execution_manifest.json",
    }
    files.update(root / "c0" / f"{unit.unit_id}.json" for unit in manifest.units)
    files.update(
        root / "c2_prequery" / f"{unit.unit_id}-s{seed}.json"
        for unit in manifest.units
        for seed in (1, 2)
    )
    files.update(
        root / "ablation_prequery" / f"{unit.unit_id}-{condition.value}.json"
        for unit in manifest.units
        for condition in (
            ConditionName.A_NO_CONTEXT,
            ConditionName.A_NO_TEMPORAL_EPISTEMIC,
            ConditionName.A_NO_RARE_GUARD,
        )
    )
    files.update(
        root / "query_access" / f"{query.stage_id}.json"
        for unit in manifest.units
        for query in unit.query_stages
    )
    files.update(
        root / "shutdown" / f"{condition.value}.json"
        for condition in (
            ConditionName.C1_LLM_PRE,
            ConditionName.C2_LLM_QUERY,
            ConditionName.A_FIXED_SELECT,
        )
    )
    files.update(
        root / "call_slots" / f"{call.ordinal:03d}-{call.call_id}.json" for call in manifest.calls
    )
    files.update(
        root / "itt" / f"{call.ordinal:03d}-{call.call_id}.json" for call in manifest.calls
    )
    files.update(
        root / "projections" / f"{unit.unit_id}-{query.stage_id}-{suffix}.json"
        for unit in manifest.units
        for query in unit.query_stages
        for suffix in ("c0", "c1-s1", "c1-s2")
    )
    return files


def _audit_journal_tree(
    root: Path, manifest: HeldOutCallManifest, *, require_complete: bool
) -> None:
    allowed_dirs = {
        root / "c0",
        root / "c2_prequery",
        root / "ablation_prequery",
        root / "query_access",
        root / "shutdown",
        root / "call_slots",
        root / "itt",
        root / "projections",
    }
    expected_files = _expected_journal_files(root, manifest)
    actual_files: set[Path] = set()
    for path in root.rglob("*"):
        if path.is_symlink() or (path.is_dir() and path not in allowed_dirs):
            raise HeldOutExecutionError(f"unexpected or symlinked held-out journal path: {path}")
        if path.is_file():
            # Same-directory atomic publication can leave only a hidden,
            # non-authoritative temporary inode after sudden power loss. Keep
            # it for audit, but never mistake it for a completed journal record.
            if path.name.startswith(".") and path.name.endswith(".tmp"):
                matching_targets = tuple(
                    candidate
                    for candidate in expected_files
                    if candidate.parent == path.parent
                    and path.name.startswith(f".{candidate.name}.")
                )
                if not matching_targets:
                    raise HeldOutExecutionError(
                        f"unexpected held-out journal temporary: {path}"
                    )
                if path.stat().st_size > 64 * 1024 * 1024:
                    raise HeldOutExecutionError(
                        f"interrupted held-out journal temporary is oversized: {path}"
                    )
                continue
            actual_files.add(path)
            if path not in expected_files:
                raise HeldOutExecutionError(f"unexpected held-out journal file: {path}")
    if require_complete and actual_files != expected_files:
        missing = sorted(str(path.relative_to(root)) for path in expected_files - actual_files)
        raise HeldOutExecutionError(f"finalized held-out journal is partial: {missing[:3]}")


def _replay_complete_journal(
    root: Path,
    manifest: HeldOutCallManifest,
    execution: HeldOutExecutionManifest,
    runtime: InjectedHeldOutRuntime,
) -> None:
    """Resolve every journal record, not merely its expected path, before scoring."""

    persisted_manifest = _load(root / "call_manifest.json", HeldOutCallManifest)
    if persisted_manifest != manifest:
        raise HeldOutExecutionError("persisted held-out call manifest differs from scorer input")
    initial = _load(root / "initial_schedule.json", GlobalGpuScheduleSnapshot)
    final = _load(root / "final_schedule.json", GlobalGpuScheduleSnapshot)
    if (
        initial.content_hash != execution.initial_schedule_snapshot_hash
        or final.content_hash != execution.final_schedule_snapshot_hash
    ):
        raise HeldOutExecutionError("persisted GPU schedule snapshots differ from execution")

    barrier = _load(root / "prequery_barrier.json", PrequeryBarrier)
    if barrier != execution.prequery_barrier:
        raise HeldOutExecutionError("persisted pre-query barrier differs from execution")
    c2_by_key = {(item.unit_id, item.seed_block): item for item in execution.c2_prequery_receipts}
    for unit in manifest.units:
        for seed in (1, 2):
            persisted = _load(
                root / "c2_prequery" / f"{unit.unit_id}-s{seed}.json",
                HeldOutC2PrequeryReceipt,
            )
            if persisted != c2_by_key.get((unit.unit_id, seed)):
                raise HeldOutExecutionError("persisted C2 pre-query receipt changed")
            runtime.validate_c2_prequery_receipt(persisted)
    ablation_by_key = {
        (item.unit_id, item.condition): item for item in execution.ablation_prequery_receipts
    }
    for unit in manifest.units:
        for condition in (
            ConditionName.A_NO_CONTEXT,
            ConditionName.A_NO_TEMPORAL_EPISTEMIC,
            ConditionName.A_NO_RARE_GUARD,
        ):
            persisted = _load(
                root / "ablation_prequery" / f"{unit.unit_id}-{condition.value}.json",
                HeldOutAblationPrequeryReceipt,
            )
            if persisted != ablation_by_key.get((unit.unit_id, condition)):
                raise HeldOutExecutionError("persisted ablation pre-query receipt changed")
            runtime.validate_ablation_prequery_receipt(persisted)
    openings = {item.sealed_stage_hash: item for item in execution.query_openings}
    for unit in manifest.units:
        for stage in unit.query_stages:
            persisted = _load(
                root / "query_access" / f"{stage.stage_id}.json",
                HeldOutQueryOpening,
            )
            if persisted != openings.get(stage.staging_manifest_hash):
                raise HeldOutExecutionError("persisted query opening changed")
    shutdowns = {item.condition: item for item in execution.service_shutdown_receipts}
    for condition, expected in shutdowns.items():
        persisted = _load(
            root / "shutdown" / f"{condition.value}.json",
            HeldOutServiceShutdownReceipt,
        )
        if persisted != expected:
            raise HeldOutExecutionError("persisted service shutdown receipt changed")

    c0_by_unit = {item.unit_id: item for item in execution.c0_constructions}
    expected_units = {item.unit_id for item in manifest.units}
    if len(c0_by_unit) != 12 or set(c0_by_unit) != expected_units:
        raise HeldOutExecutionError("persisted C0 inventory differs from held-out units")
    for unit in manifest.units:
        persisted = _load(
            root / "c0" / f"{unit.unit_id}.json",
            C0ConstructionReceipt,
        )
        if persisted != c0_by_unit[unit.unit_id]:
            raise HeldOutExecutionError("persisted C0 receipt differs from execution")

    projections_by_key = {
        (item.unit_id, item.query_stage_hash, item.condition, item.seed_block): item
        for item in execution.preconstructed_projections
    }
    if len(projections_by_key) != 108:
        raise HeldOutExecutionError("persisted preconstructed projections are not unique")
    expected_projection_keys: set[tuple[str, str, ConditionName, int | None]] = set()
    for unit in manifest.units:
        for query in unit.query_stages:
            for condition, seed, suffix in (
                (ConditionName.C0_CLASSICAL_PRE, None, "c0"),
                (ConditionName.C1_LLM_PRE, 1, "c1-s1"),
                (ConditionName.C1_LLM_PRE, 2, "c1-s2"),
            ):
                key = (unit.unit_id, query.staging_manifest_hash, condition, seed)
                expected_projection_keys.add(key)
                persisted = _load(
                    root / "projections" / f"{unit.unit_id}-{query.stage_id}-{suffix}.json",
                    PreconstructedProjectionReceipt,
                )
                if persisted != projections_by_key.get(key):
                    raise HeldOutExecutionError(
                        "persisted projection receipt differs from execution"
                    )
    if set(projections_by_key) != expected_projection_keys:
        raise HeldOutExecutionError("projection receipt inventory differs from the call plan")

    identities = {item.condition: item for item in execution.session_identities}
    for call, expected_record in zip(manifest.calls, execution.itt_records, strict=True):
        slot = _load(
            root / "call_slots" / f"{call.ordinal:03d}-{call.call_id}.json",
            HeldOutCallSlot,
        )
        persisted_record = _load(
            root / "itt" / f"{call.ordinal:03d}-{call.call_id}.json",
            HeldOutITTRecord,
        )
        if persisted_record != expected_record:
            raise HeldOutExecutionError("persisted ITT receipt differs from execution")
        identity = identities[call.condition]
        if (
            slot.ordinal != call.ordinal
            or slot.call_spec_hash != call.content_hash
            or slot.content_hash != persisted_record.call_slot_hash
            or slot.schedule_before != persisted_record.schedule_before
            or slot.session_identity != identity
        ):
            raise HeldOutExecutionError("persisted call slot differs from its ITT receipt")
        _result_matches_call(call, slot.envelope, persisted_record.result, identity)
        runtime.validate_result_artifacts(call, slot.envelope, persisted_record.result)


def _result_matches_call(
    call: HeldOutCallSpec,
    envelope: HeldOutCallEnvelope,
    result: HeldOutServiceResult,
    identity: HeldOutSessionIdentity,
) -> None:
    if result.call_id != call.call_id or result.condition is not call.condition:
        raise HeldOutExecutionError("service result belongs to another registered call")
    if identity.condition is not call.condition:
        raise HeldOutExecutionError("call was routed through another condition session")
    if result.repair_attempts > call.repair_attempt_budget:
        raise HeldOutExecutionError("service exceeded the registered repair-attempt budget")
    if result.request_started and (
        result.artifact_receipt is None
        or result.artifact_receipt.call_spec_hash != call.content_hash
    ):
        raise HeldOutExecutionError("service CAS receipt does not bind the registered call")
    if envelope.prequery_stage.staging_manifest_hash != call.prequery_stage_hash:
        raise HeldOutExecutionError("call envelope uses another pre-query stage")
    if call.query_stage_hash is None:
        if envelope.query_stage is not None:
            raise HeldOutExecutionError("query-blind C1 envelope contains a query stage")
    elif (
        envelope.query_stage is None
        or envelope.query_stage.staging_manifest_hash != call.query_stage_hash
    ):
        raise HeldOutExecutionError("query-time envelope uses another query stage")
    if (
        result.request_started
        and envelope.query_stage is not None
        and (
            envelope.query_opening is None
            or result.evidence_packet_hash != envelope.query_opening.evidence_packet_hash
            or result.horizon_hash != envelope.query_stage.horizon_hash
            or result.budget_hash != envelope.query_stage.budget_hash
            or result.query_revealed_at != envelope.query_opening.query_access_event.accessed_at
        )
    ):
        raise HeldOutExecutionError(
            "query-time result changed its registered horizon or output budgets"
        )
    if call.call_class == "test_c1" and result.outcome is RunOutcome.SUCCEEDED:
        seal = result.construction_seal
        if seal is None or (
            seal.snapshot_hash != envelope.prequery_stage.snapshot_hash
            or seal.ontology_hash != result.complete_c1_graph_hash
        ):
            raise HeldOutExecutionError("C1 seal does not bind its sealed evidence snapshot")
    if call.call_class == "test_c2" and result.request_started:
        inventory = result.pre_query_inventory
        expected_receipt = envelope.c2_prequery_receipt
        if (
            inventory is None
            or expected_receipt is None
            or inventory != expected_receipt.inventory
            or inventory.snapshot_hash != envelope.prequery_stage.snapshot_hash
        ):
            raise HeldOutExecutionError("C2 empty inventory does not bind its evidence snapshot")
        certificate = result.construction_certificate
        if result.outcome is RunOutcome.SUCCEEDED:
            if certificate is None or envelope.query_stage is None:
                raise HeldOutExecutionError("successful C2 lacks typed construction lineage")
            if (
                certificate.snapshot_hash != envelope.prequery_stage.snapshot_hash
                or certificate.packet_hash != result.evidence_packet_hash
                or certificate.query_context_hash != envelope.query_stage.query_context_hash
                or certificate.stage_manifest_hash != envelope.query_stage.staging_manifest_hash
                or envelope.query_opening is None
                or certificate.query_access_event_hash
                != envelope.query_opening.query_access_event.content_hash
                or envelope.prequery_barrier is None
                or certificate.prequery_barrier_hash != envelope.prequery_barrier.content_hash
                or certificate.pre_query_inventory_hash != inventory.content_hash
                or inventory.recorded_at >= certificate.query_revealed_at
            ):
                raise HeldOutExecutionError("C2 certificate differs from its post-query request")
    if call.call_class == "test_fixed_select" and result.request_started:
        audit = result.fixed_select_capability_audit
        if audit is None or (
            audit.source_c1_seal_hash != envelope.source_c1_seal_hash
            or audit.complete_c1_graph_hash != envelope.complete_c1_graph_hash
            or audit.accepted_constructive_operator_count != 0
        ):
            raise HeldOutExecutionError(
                "FixedSelect result does not bind its complete construction-disabled envelope"
            )
        certificate = result.construction_certificate
        if result.outcome is RunOutcome.SUCCEEDED:
            if certificate is None or envelope.query_stage is None:
                raise HeldOutExecutionError("successful FixedSelect lacks its typed certificate")
            if (
                certificate.snapshot_hash != envelope.prequery_stage.snapshot_hash
                or certificate.packet_hash != result.evidence_packet_hash
                or certificate.query_context_hash != envelope.query_stage.query_context_hash
                or certificate.stage_manifest_hash != envelope.query_stage.staging_manifest_hash
                or envelope.query_opening is None
                or certificate.query_access_event_hash
                != envelope.query_opening.query_access_event.content_hash
                or envelope.prequery_barrier is None
                or certificate.prequery_barrier_hash != envelope.prequery_barrier.content_hash
                or certificate.inherited_construction_seal_hash != envelope.source_c1_seal_hash
            ):
                raise HeldOutExecutionError(
                    "FixedSelect certificate does not bind the inherited same-seed graph"
                )


def _unstarted_failure_result(
    *,
    call: HeldOutCallSpec,
    snapshot: GlobalGpuScheduleSnapshot,
    failure_code: str,
) -> HeldOutServiceResult:
    """Create a zero-request ITT result without pretending inference occurred."""

    return HeldOutServiceResult(
        call_id=call.call_id,
        condition=call.condition,
        outcome=RunOutcome.FAILED,
        request_started=False,
        global_ledger_chain_hash=snapshot.ledger_chain_hash,
        allocated_gpu_seconds=0.0,
        repair_attempts=0,
        failure_code=failure_code,
        completed_at=snapshot.captured_at,
    )


def _assert_accounting_transition(
    *,
    call: HeldOutCallSpec,
    before: GlobalGpuScheduleSnapshot,
    after: GlobalGpuScheduleSnapshot,
    result: HeldOutServiceResult,
    identity: HeldOutSessionIdentity,
    configuration: HeldOutControlConfiguration,
) -> None:
    if (
        before.global_accounting_id != identity.global_accounting_id
        or after.global_accounting_id != identity.global_accounting_id
    ):
        raise HeldOutExecutionError("session snapshots use another global accounting scope")
    if (
        before.gpu_call_inventory_file_sha256 != configuration.gpu_call_inventory_file_sha256
        or after.gpu_call_inventory_file_sha256 != configuration.gpu_call_inventory_file_sha256
    ):
        raise HeldOutExecutionError("GPU transition uses another registered call inventory")
    if (
        before.development_execution_result_hash != after.development_execution_result_hash
        or before.development_predecessor_allocated_gpu_seconds
        != after.development_predecessor_allocated_gpu_seconds
    ):
        raise HeldOutExecutionError("GPU transition changed its development predecessor")
    if after.captured_at < before.captured_at or result.completed_at < before.captured_at:
        raise HeldOutExecutionError("GPU accounting or service completion moved backward in time")
    if result.completed_at > after.captured_at:
        raise HeldOutExecutionError("service result completion follows its after-snapshot")
    before_pools = {item.reserve_class: item.consumed_slots for item in before.repair_reserves}
    after_pools = {item.reserve_class: item.consumed_slots for item in after.repair_reserves}
    deltas = {name: after_pools[name] - before_pools[name] for name in before_pools}
    if any(value < 0 for value in deltas.values()):
        raise HeldOutExecutionError("a protected repair reserve counter regressed")
    if deltas[call.repair_reserve_class] != result.repair_attempts or any(
        value != 0 for name, value in deltas.items() if name != call.repair_reserve_class
    ):
        raise HeldOutExecutionError("call did not consume exactly its registered repair reserve")
    allocation_delta = (
        after.actual_allocated_gpu_seconds - before.actual_allocated_gpu_seconds
    )
    expected_remaining = before.remaining_registered_p95_seconds - (
        call.p95_seconds + result.repair_attempts * call.watchdog_seconds
    )
    if result.request_started:
        if result.allocated_gpu_seconds <= 0:
            raise HeldOutExecutionError("a started GPU request cannot report zero allocation")
        # The continuously allocated service also covers packing, validation,
        # persistence, and inter-snapshot control work. Classified request
        # events are therefore a lower bound on the exact live-service delta.
        if allocation_delta + 1e-6 < result.allocated_gpu_seconds:
            raise HeldOutExecutionError(
                "service allocation delta omits classified request GPU time"
            )
    elif result.allocated_gpu_seconds != 0:
        raise HeldOutExecutionError("an unstarted call cannot consume classified GPU time")
    if (
        allocation_delta < -1e-6
        or not math.isclose(
            after.remaining_registered_p95_seconds,
            expected_remaining,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
        or after.ledger_chain_hash != result.global_ledger_chain_hash
    ):
        raise HeldOutExecutionError(
            "service result does not reproduce exact cumulative GPU accounting"
        )
    if after.actual_allocated_gpu_seconds >= configuration.hard_gpu_seconds_limit:
        raise HeldOutExecutionError("service crossed the hard 10-hour GPU allocation stop")
    if (
        after.actual_allocated_gpu_seconds + after.remaining_registered_p95_seconds
        > configuration.scheduled_gpu_seconds_limit + 1e-6
    ):
        raise HeldOutExecutionError("post-call work no longer fits the 9-hour schedule")


def _assert_query_fairness(
    manifest: HeldOutCallManifest,
    projections: tuple[PreconstructedProjectionReceipt, ...],
    itt_records: tuple[HeldOutITTRecord, ...],
) -> None:
    reference: dict[tuple[str, str], tuple[str, str, str]] = {}
    for item in projections:
        key = (item.unit_id, item.query_stage_hash)
        values = (item.evidence_packet_hash, item.horizon_hash, item.budget_hash)
        prior = reference.setdefault(key, values)
        if prior != values:
            raise HeldOutExecutionError("C0/C1 packet, horizon, or budgets differ")
    call_by_hash = {item.content_hash: item for item in manifest.calls}
    for record in itt_records:
        call = call_by_hash[record.call_spec_hash]
        if call.query_stage_hash is None or not record.result.request_started:
            continue
        values = (
            record.result.evidence_packet_hash,
            record.result.horizon_hash,
            record.result.budget_hash,
        )
        if None in values:
            raise HeldOutExecutionError("started query call lacks fairness hashes")
        if reference.get((call.unit_id, call.query_stage_hash)) != values:
            raise HeldOutExecutionError("condition packet, horizon, or budgets differ")


def _assert_construction_lineage(
    manifest: HeldOutCallManifest,
    c0_receipts: tuple[C0ConstructionReceipt, ...],
    c2_prequery_receipts: tuple[HeldOutC2PrequeryReceipt, ...],
    ablation_prequery_receipts: tuple[HeldOutAblationPrequeryReceipt, ...],
    prequery_barrier: PrequeryBarrier,
    projections: tuple[PreconstructedProjectionReceipt, ...],
    itt_records: tuple[HeldOutITTRecord, ...],
) -> None:
    calls = {item.call_id: item for item in manifest.calls}
    results = {calls_by.result.call_id: calls_by.result for calls_by in itt_records}
    c1_results = [
        item.result for item in itt_records if item.result.condition is ConditionName.C1_LLM_PRE
    ]
    expected_barrier = _build_prequery_barrier(
        manifest=manifest,
        c0_receipts=c0_receipts,
        c2_receipts=c2_prequery_receipts,
        ablation_receipts=ablation_prequery_receipts,
        c1_results=tuple(c1_results),
    )
    if prequery_barrier != expected_barrier:
        raise HeldOutExecutionError(
            "pre-query barrier does not exactly bind all primary and ablation preparations"
        )
    prequery_completed_at = max(
        (
            *(item.completed_at for item in c0_receipts),
            *(item.completed_at for item in c1_results),
            *(item.completed_at for item in c2_prequery_receipts),
            *(item.completed_at for item in ablation_prequery_receipts),
        )
    )
    if prequery_completed_at >= prequery_barrier.sealed_at:
        raise HeldOutExecutionError("pre-query barrier does not follow every preparation")
    for result in c1_results:
        if result.construction_seal is not None and (
            result.construction_seal.sealed_at > result.completed_at
        ):
            raise HeldOutExecutionError("C1 result completed before its construction seal")
    inventories_by_unit_seed = {
        (item.unit_id, item.seed_block): item.inventory.content_hash
        for item in c2_prequery_receipts
    }
    expected_ablation_keys = {
        (unit.unit_id, condition)
        for unit in manifest.units
        for condition in (
            ConditionName.A_NO_CONTEXT,
            ConditionName.A_NO_TEMPORAL_EPISTEMIC,
            ConditionName.A_NO_RARE_GUARD,
        )
    }
    if {
        (item.unit_id, item.condition) for item in ablation_prequery_receipts
    } != expected_ablation_keys:
        raise HeldOutExecutionError("ablation pre-query preparation inventory is incomplete")
    for projection in projections:
        if (
            projection.condition is not ConditionName.C1_LLM_PRE
            or projection.outcome is not RunOutcome.SUCCEEDED
        ):
            continue
        c1_call = next(
            item
            for item in manifest.calls
            if item.call_class == "test_c1"
            and item.unit_id == projection.unit_id
            and item.seed_block == projection.seed_block
        )
        source = results[c1_call.call_id]
        if (
            projection.source_construction_seal_hash != source.construction_seal_hash
            or projection.source_complete_graph_hash != source.complete_c1_graph_hash
        ):
            raise HeldOutExecutionError("C1 projection did not reuse its sealed complete graph")
    for record in itt_records:
        call = calls[record.result.call_id]
        result = record.result
        if (
            result.request_started
            and call.query_stage_hash is not None
            and (
                result.query_revealed_at is None
                or result.query_revealed_at <= prequery_barrier.sealed_at
            )
        ):
            raise HeldOutExecutionError(
                "query-time request was revealed before every C0/C1 preparation completed"
            )
        if (
            call.call_class == "test_c2"
            and result.outcome is RunOutcome.SUCCEEDED
            and (
                not result.empty_prequery_inventory_hash or not result.construction_certificate_hash
            )
        ):
            raise HeldOutExecutionError("C2 lost empty-inventory/certificate lineage")
        if call.call_class == "test_c2" and result.pre_query_inventory is not None:
            expected_inventory = inventories_by_unit_seed.get((call.unit_id, call.seed_block))
            if expected_inventory != result.pre_query_inventory.content_hash:
                raise HeldOutExecutionError("C2 did not reuse its seeded pre-query empty inventory")
        if call.call_class == "test_fixed_select" and result.request_started:
            source = results[call.source_c1_call_id]
            audit = result.fixed_select_capability_audit
            if audit is None or (
                audit.complete_c1_graph_hash != source.complete_c1_graph_hash
                or audit.source_c1_seal_hash != source.construction_seal_hash
                or audit.accepted_constructive_operator_count != 0
            ):
                raise HeldOutExecutionError(
                    "FixedSelect did not receive the complete same-seed C1 graph"
                )


def _build_prequery_barrier(
    *,
    manifest: HeldOutCallManifest,
    c0_receipts: tuple[C0ConstructionReceipt, ...],
    c2_receipts: tuple[HeldOutC2PrequeryReceipt, ...],
    ablation_receipts: tuple[HeldOutAblationPrequeryReceipt, ...],
    c1_results: tuple[HeldOutServiceResult, ...],
) -> PrequeryBarrier:
    """Seal all primary and query-blind ablation preparations before query access."""

    if (
        len(c0_receipts) != 12
        or len(c2_receipts) != 24
        or len(ablation_receipts) != 36
        or len(c1_results) != 24
    ):
        raise HeldOutExecutionError("pre-query barrier inputs are incomplete")
    unit_by_id = {item.unit_id: item for item in manifest.units}
    call_by_id = {item.call_id: item for item in manifest.calls}
    bindings: list[PrequeryPreparationBinding] = []
    for receipt in c0_receipts:
        unit = unit_by_id[receipt.unit_id]
        lineage_hash = receipt.complete_graph_hash or receipt.failure_artifact_hash
        if lineage_hash is None:
            raise HeldOutExecutionError("C0 barrier binding has no terminal lineage")
        bindings.append(
            PrequeryPreparationBinding(
                unit_id=receipt.unit_id,
                condition=ConditionName.C0_CLASSICAL_PRE,
                snapshot_hash=unit.prequery_stage.snapshot_hash,
                preparation_hash=receipt.content_hash,
                lineage_artifact_hash=lineage_hash,
                completed_at=receipt.completed_at,
            )
        )
    for result in c1_results:
        call = call_by_id[result.call_id]
        if call.call_class != "test_c1":
            raise HeldOutExecutionError("non-C1 result entered the pre-query barrier")
        output = None if result.artifact_receipt is None else result.artifact_receipt.output
        if result.outcome is RunOutcome.SUCCEEDED:
            if output is None or len(result.prequery_preparation_bindings) != 2:
                raise HeldOutExecutionError(
                    "successful C1 lacks its C1/Fixed pre-query preparation bindings"
                )
            expected_conditions = {
                ConditionName.C1_LLM_PRE,
                ConditionName.A_FIXED_SELECT,
            }
            if {
                item.condition for item in result.prequery_preparation_bindings
            } != expected_conditions:
                raise HeldOutExecutionError("C1 pre-query binding conditions changed")
            for binding in result.prequery_preparation_bindings:
                if (
                    binding.unit_id != call.unit_id
                    or binding.seed_block != call.seed_block
                    or binding.snapshot_hash
                    != unit_by_id[call.unit_id].prequery_stage.snapshot_hash
                    or binding.completed_at > result.completed_at
                ):
                    raise HeldOutExecutionError("C1 pre-query binding lineage changed")
            c1_binding = next(
                item
                for item in result.prequery_preparation_bindings
                if item.condition is ConditionName.C1_LLM_PRE
            )
            if (
                c1_binding.preparation_hash != output.logical_content_hash
                or c1_binding.lineage_artifact_hash != result.construction_seal_hash
            ):
                raise HeldOutExecutionError("C1 binding differs from its sealed output")
            bindings.extend(result.prequery_preparation_bindings)
        else:
            bindings.append(
                PrequeryPreparationBinding(
                    unit_id=call.unit_id,
                    condition=ConditionName.C1_LLM_PRE,
                    seed_block=call.seed_block,
                    snapshot_hash=unit_by_id[call.unit_id].prequery_stage.snapshot_hash,
                    preparation_hash=result.content_hash,
                    lineage_artifact_hash=(
                        result.ledger_receipt_artifact_hash or result.content_hash
                    ),
                    completed_at=result.completed_at,
                )
            )
    for receipt in c2_receipts:
        bindings.append(
            PrequeryPreparationBinding(
                unit_id=receipt.unit_id,
                condition=ConditionName.C2_LLM_QUERY,
                seed_block=receipt.seed_block,
                snapshot_hash=receipt.inventory.snapshot_hash,
                preparation_hash=receipt.preparation.content_hash,
                lineage_artifact_hash=receipt.preparation_artifact.artifact_hash,
                completed_at=receipt.completed_at,
            )
        )
    for receipt in ablation_receipts:
        inventory = receipt.preparation.empty_inventory
        if inventory is None:
            raise HeldOutExecutionError("ablation pre-query preparation lost its inventory")
        bindings.append(
            PrequeryPreparationBinding(
                unit_id=receipt.unit_id,
                condition=receipt.condition,
                seed_block=receipt.seed_block,
                snapshot_hash=inventory.snapshot_hash,
                preparation_hash=receipt.preparation.content_hash,
                lineage_artifact_hash=receipt.preparation_artifact.artifact_hash,
                completed_at=receipt.completed_at,
            )
        )
    ordered = tuple(
        sorted(
            bindings,
            key=lambda item: (item.unit_id, item.condition.value, item.seed_block or 0),
        )
    )
    sealed_at = max(
        *(item.completed_at for item in ordered),
        *(item.completed_at for item in c1_results),
    ) + timedelta(microseconds=1)
    execution_id = f"held-out-{manifest.content_hash[:20]}"
    return PrequeryBarrier(
        barrier_id=f"prequery-barrier-{execution_id}",
        execution_id=execution_id,
        execution_manifest_hash=manifest.content_hash,
        neutral_evidence_artifact_hashes=tuple(
            sorted(item.prequery_stage.evidence_artifact_hash for item in manifest.units)
        ),
        preparation_bindings=ordered,
        sealed_at=sealed_at,
    )


def execute_reviewed_held_out_manifest(
    *,
    reviewed_plan: ReviewedHeldOutPlan,
    configuration: HeldOutControlConfiguration,
    repository: Path,
    review_completion_root: Path,
    configuration_path: Path,
    runtime_binding_path: Path | None = None,
    cpu: InjectedHeldOutCpu,
    sessions: dict[ConditionName, InjectedHeldOutSession],
    runtime: InjectedHeldOutRuntime,
    output_root: Path,
    completion_clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> HeldOutExecutionManifest:
    """Execute or recover a plan cryptographically bound to the reproduced review gate."""

    verified_plan = open_reviewed_held_out_plan(
        repository=repository,
        review_completion_root=review_completion_root,
        configuration_path=configuration_path,
        runtime_binding_path=runtime_binding_path,
    )
    if runtime_binding_path is None:
        # Direct injection is retained only for fixture-level controller tests;
        # the real opener above cannot pass a PENDING predecessor without a
        # restricted binding.
        verified_configuration = load_held_out_control_configuration(
            repository,
            configuration_path,
        )
    else:
        verified_configuration = load_executable_held_out_configuration(
            repository=repository,
            configuration_path=configuration_path,
            runtime_binding_path=runtime_binding_path,
            expected_plan_hash=verified_plan.call_manifest.content_hash,
        )
    if verified_plan != reviewed_plan:
        raise HeldOutExecutionError(
            "held-out plan differs from the freshly reproduced independent-review gate"
        )
    if verified_configuration != configuration:
        raise HeldOutExecutionError("held-out execution configuration changed after review replay")
    manifest = verified_plan.call_manifest
    if configuration.expected_plan_hash != manifest.content_hash:
        raise HeldOutExecutionError("held-out call manifest is not the frozen configured plan")

    expected_conditions = {
        ConditionName.C1_LLM_PRE,
        ConditionName.C2_LLM_QUERY,
        ConditionName.A_FIXED_SELECT,
    }
    if set(sessions) != expected_conditions:
        raise HeldOutExecutionError("exactly three condition-separated sessions are required")
    ordered_conditions = (
        ConditionName.C1_LLM_PRE,
        ConditionName.C2_LLM_QUERY,
        ConditionName.A_FIXED_SELECT,
    )
    root = output_root.absolute()
    if root.is_symlink():
        raise HeldOutExecutionError("held-out output root cannot be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    execution_path = root / "execution_manifest.json"
    _audit_journal_tree(
        root, manifest, require_complete=execution_path.exists()
    )
    if execution_path.exists():
        persisted_execution = _load(execution_path, HeldOutExecutionManifest)
        if (
            persisted_execution.call_manifest_hash != manifest.content_hash
            or persisted_execution.review_completion_manifest_hash
            != reviewed_plan.review_completion_manifest_hash
            or persisted_execution.final_reviewed_seal_hash
            != reviewed_plan.final_reviewed_seal_hash
            or persisted_execution.runtime_binding_hash
            != reviewed_plan.runtime_binding_hash
        ):
            raise HeldOutExecutionError("finalized execution belongs to another reviewed run")
        _replay_complete_journal(root, manifest, persisted_execution, runtime)
        return persisted_execution
    _append_exact(root / "call_manifest.json", manifest)

    c0_receipts: list[C0ConstructionReceipt] = []
    for unit in manifest.units:
        prequery_unit = HeldOutPrequeryUnit(
            unit_id=unit.unit_id,
            prequery_stage=unit.prequery_stage,
        )
        path = root / "c0" / f"{unit.unit_id}.json"
        if path.exists():
            receipt = _load(path, C0ConstructionReceipt)
        else:
            receipt = cpu.build_c0(prequery_unit)
            _append_exact(path, receipt)
        if (
            receipt.unit_id != unit.unit_id
            or receipt.prequery_stage_hash != unit.prequery_stage.staging_manifest_hash
        ):
            raise HeldOutExecutionError("C0 receipt belongs to another held-out unit")
        c0_receipts.append(receipt)

    c2_prequery_receipts: list[HeldOutC2PrequeryReceipt] = []
    for unit in manifest.units:
        prequery_unit = HeldOutPrequeryUnit(
            unit_id=unit.unit_id,
            prequery_stage=unit.prequery_stage,
        )
        for seed in (1, 2):
            path = root / "c2_prequery" / f"{unit.unit_id}-s{seed}.json"
            if path.exists():
                receipt = _load(path, HeldOutC2PrequeryReceipt)
            else:
                receipt = runtime.prepare_c2_empty_inventory(
                    prequery_unit,
                    seed_block=seed,
                )
                _append_exact(path, receipt)
            if (
                receipt.unit_id != unit.unit_id
                or receipt.seed_block != seed
                or receipt.prequery_stage_hash != unit.prequery_stage.staging_manifest_hash
                or receipt.inventory.snapshot_hash != unit.prequery_stage.snapshot_hash
            ):
                raise HeldOutExecutionError(
                    "C2 empty-inventory receipt belongs to another held-out unit"
                )
            runtime.validate_c2_prequery_receipt(receipt)
            c2_prequery_receipts.append(receipt)

    ablation_prequery_receipts: list[HeldOutAblationPrequeryReceipt] = []
    for unit in manifest.units:
        prequery_unit = HeldOutPrequeryUnit(
            unit_id=unit.unit_id,
            prequery_stage=unit.prequery_stage,
        )
        for condition in (
            ConditionName.A_NO_CONTEXT,
            ConditionName.A_NO_TEMPORAL_EPISTEMIC,
            ConditionName.A_NO_RARE_GUARD,
        ):
            path = root / "ablation_prequery" / f"{unit.unit_id}-{condition.value}.json"
            if path.exists():
                receipt = _load(path, HeldOutAblationPrequeryReceipt)
            else:
                receipt = runtime.prepare_ablation_empty_inventory(
                    prequery_unit,
                    condition=condition,
                )
                _append_exact(path, receipt)
            inventory = receipt.preparation.empty_inventory
            if (
                receipt.unit_id != unit.unit_id
                or receipt.condition is not condition
                or receipt.seed_block != 1
                or receipt.prequery_stage_hash != unit.prequery_stage.staging_manifest_hash
                or inventory is None
                or inventory.snapshot_hash != unit.prequery_stage.snapshot_hash
            ):
                raise HeldOutExecutionError(
                    "ablation empty preparation belongs to another held-out unit"
                )
            runtime.validate_ablation_prequery_receipt(receipt)
            ablation_prequery_receipts.append(receipt)

    initial_path = root / "initial_schedule.json"
    c1_session = sessions[ConditionName.C1_LLM_PRE]
    c1_identity = c1_session.identity()
    if c1_identity.condition is not ConditionName.C1_LLM_PRE:
        raise HeldOutExecutionError("C1 session identity has the wrong condition")
    identity_by_condition = {ConditionName.C1_LLM_PRE: c1_identity}
    if initial_path.exists():
        initial_snapshot = _load(initial_path, GlobalGpuScheduleSnapshot)
    else:
        initial_snapshot = c1_session.schedule_snapshot()
        _append_exact(initial_path, initial_snapshot)
    if initial_snapshot.global_accounting_id != c1_identity.global_accounting_id:
        raise HeldOutExecutionError("initial snapshot uses another global accounting scope")
    if initial_snapshot.ledger_chain_hash != c1_identity.global_ledger_chain_hash:
        raise HeldOutExecutionError("initial snapshot does not continue the C1 ledger chain")
    if initial_snapshot.actual_allocated_gpu_seconds <= 0:
        raise HeldOutExecutionError(
            "held-out execution cannot reset cumulative GPU accounting to an empty ledger"
        )
    if (
        initial_snapshot.development_execution_result_hash
        != manifest.development_execution_result_hash
    ):
        raise HeldOutExecutionError("initial GPU snapshot uses another development result")
    if manifest.allocated_gpu_seconds_before_heldout is not None and not math.isclose(
        initial_snapshot.development_predecessor_allocated_gpu_seconds,
        manifest.allocated_gpu_seconds_before_heldout,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise HeldOutExecutionError(
            "initial GPU snapshot differs from the frozen development predecessor"
        )
    if (
        initial_snapshot.actual_allocated_gpu_seconds
        < initial_snapshot.development_predecessor_allocated_gpu_seconds
    ):
        raise HeldOutExecutionError("cumulative GPU time regressed after development")
    if manifest.post_development_mandatory_forecast_seconds is not None and not math.isclose(
        initial_snapshot.remaining_registered_p95_seconds,
        manifest.post_development_mandatory_forecast_seconds - PRIMARY_SERVICE_START_P95_SECONDS,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise HeldOutExecutionError(
            "initial remaining schedule does not charge the C1 service allocation"
        )
    if initial_snapshot.remaining_registered_p95_seconds < configuration.primary_forecast_seconds:
        raise HeldOutExecutionError("initial global forecast omits held-out primary work")
    admit_call(call=manifest.calls[0], snapshot=initial_snapshot, configuration=configuration)
    previous_allocated = initial_snapshot.actual_allocated_gpu_seconds
    units_by_id = {item.unit_id: item for item in manifest.units}
    query_stages_by_hash = {
        query.staging_manifest_hash: query for unit in manifest.units for query in unit.query_stages
    }
    records: list[HeldOutITTRecord] = []
    completed_results: dict[str, HeldOutServiceResult] = {}
    c2_prequery_by_key = {(item.unit_id, item.seed_block): item for item in c2_prequery_receipts}
    query_openings: dict[str, HeldOutQueryOpening] = {}
    prequery_barrier: PrequeryBarrier | None = None
    active_condition = ConditionName.C1_LLM_PRE
    condition_timeout_sources: dict[ConditionName, str] = {}
    if not getattr(c1_session, "service_available", lambda: True)():
        condition_timeout_sources[ConditionName.C1_LLM_PRE] = c1_identity.content_hash
    shutdown_by_condition: dict[ConditionName, HeldOutServiceShutdownReceipt] = {}
    last_shutdown_snapshot: GlobalGpuScheduleSnapshot | None = None
    for call in manifest.calls:
        session = sessions[call.condition]
        condition_just_started = False
        if call.condition is not active_condition:
            prior_identity = identity_by_condition[active_condition]
            shutdown_path = root / "shutdown" / f"{active_condition.value}.json"
            shutdown_receipt, shutdown_snapshot = runtime.shutdown_condition(prior_identity)
            if shutdown_snapshot != shutdown_receipt.schedule_after_shutdown:
                raise HeldOutExecutionError(
                    "service shutdown did not return its persisted transition snapshot"
                )
            if shutdown_path.exists():
                persisted_shutdown = _load(shutdown_path, HeldOutServiceShutdownReceipt)
                if persisted_shutdown != shutdown_receipt:
                    raise HeldOutExecutionError("service shutdown recovery changed its receipt")
            else:
                _append_exact(shutdown_path, shutdown_receipt)
            if shutdown_snapshot.actual_allocated_gpu_seconds < previous_allocated:
                raise HeldOutExecutionError("GPU allocation regressed during shutdown")
            shutdown_by_condition[active_condition] = shutdown_receipt
            last_shutdown_snapshot = shutdown_snapshot
            previous_allocated = shutdown_snapshot.actual_allocated_gpu_seconds
            active_condition = call.condition
            identity = session.identity()
            if identity.condition is not call.condition:
                raise HeldOutExecutionError("condition session identity has the wrong condition")
            prior_identities = tuple(identity_by_condition.values())
            if any(
                identity.session_id == prior.session_id
                or identity.service_identity_hash == prior.service_identity_hash
                or identity.allocation_event_id == prior.allocation_event_id
                or identity.model_load_event_hash == prior.model_load_event_hash
                or (identity.service_pid, identity.service_start_ticks)
                == (prior.service_pid, prior.service_start_ticks)
                for prior in prior_identities
            ):
                raise HeldOutExecutionError(
                    "condition blocks did not use distinct allocation/load processes"
                )
            if any(
                identity.global_accounting_id != prior.global_accounting_id
                for prior in prior_identities
            ):
                raise HeldOutExecutionError(
                    "condition service did not continue the cumulative GPU ledger"
                )
            identity_by_condition[call.condition] = identity
            if not getattr(session, "service_available", lambda: True)():
                condition_timeout_sources[call.condition] = identity.content_hash
            condition_just_started = True
        else:
            identity = session.identity()
        if identity != identity_by_condition[call.condition]:
            raise HeldOutExecutionError("condition service identity changed during execution")
        source = (
            completed_results.get(call.source_c1_call_id)
            if call.source_c1_call_id is not None
            else None
        )
        source_unavailable = call.source_c1_call_id is not None and (
            source is None
            or source.outcome is not RunOutcome.SUCCEEDED
            or source.construction_seal_hash is None
            or source.complete_c1_graph_hash is None
        )
        condition_stopped_after_timeout = call.condition in condition_timeout_sources
        unit = units_by_id[call.unit_id]
        query_stage = (
            None if call.query_stage_hash is None else query_stages_by_hash[call.query_stage_hash]
        )
        query_opening = None
        if query_stage is not None:
            if prequery_barrier is None:
                c1_results = tuple(
                    completed_results[item.call_id]
                    for item in manifest.calls
                    if item.call_class == "test_c1"
                )
                expected_barrier = _build_prequery_barrier(
                    manifest=manifest,
                    c0_receipts=tuple(c0_receipts),
                    c2_receipts=tuple(c2_prequery_receipts),
                    ablation_receipts=tuple(ablation_prequery_receipts),
                    c1_results=c1_results,
                )
                barrier_path = root / "prequery_barrier.json"
                if barrier_path.exists():
                    prequery_barrier = _load(barrier_path, PrequeryBarrier)
                    if prequery_barrier != expected_barrier:
                        raise HeldOutExecutionError(
                            "persisted pre-query barrier differs from exact preparations"
                        )
                else:
                    runtime.persist_prequery_barrier(expected_barrier)
                    _append_exact(barrier_path, expected_barrier)
                    prequery_barrier = expected_barrier
                runtime.persist_prequery_barrier(prequery_barrier)
            opening_path = root / "query_access" / f"{query_stage.stage_id}.json"
            persisted_opening = (
                _load(opening_path, HeldOutQueryOpening) if opening_path.exists() else None
            )
            query_opening = runtime.open_or_recover_query(
                query_stage,
                barrier=prequery_barrier,
                persisted=persisted_opening,
            )
            if persisted_opening is None:
                _append_exact(opening_path, query_opening)
            elif query_opening != persisted_opening:
                raise HeldOutExecutionError(
                    "audited query recovery differs from its persisted opening"
                )
            query_stage = query_opening.opened_stage
            query_openings[query_opening.sealed_stage_hash] = query_opening
        envelope = HeldOutCallEnvelope(
            call_spec_hash=call.content_hash,
            prequery_stage=unit.prequery_stage,
            query_stage=query_stage,
            source_c1_call_id=call.source_c1_call_id,
            source_c1_seal_hash=(
                None if source_unavailable or source is None else source.construction_seal_hash
            ),
            complete_c1_graph_hash=(
                None if source_unavailable or source is None else source.complete_c1_graph_hash
            ),
            source_c1_output=(
                None
                if source_unavailable or source is None or source.artifact_receipt is None
                else source.artifact_receipt.output
            ),
            dependency_failure_hash=(
                source.content_hash if source_unavailable and source is not None else None
            ),
            prequery_barrier=prequery_barrier,
            query_opening=query_opening,
            c2_prequery_receipt=(
                c2_prequery_by_key[(call.unit_id, call.seed_block)]
                if call.condition is ConditionName.C2_LLM_QUERY
                else None
            ),
            require_empty_prequery_inventory=call.condition is ConditionName.C2_LLM_QUERY,
            construction_operations_permitted=call.construction_operations_permitted,
        )
        slot_path = root / "call_slots" / f"{call.ordinal:03d}-{call.call_id}.json"
        itt_path = root / "itt" / f"{call.ordinal:03d}-{call.call_id}.json"
        if itt_path.exists():
            slot = _load(slot_path, HeldOutCallSlot)
            record = _load(itt_path, HeldOutITTRecord)
            if slot.envelope != envelope:
                raise HeldOutExecutionError(
                    "resumed call slot differs from its registered envelope"
                )
        else:
            recovering = slot_path.exists()
            if recovering:
                slot = _load(slot_path, HeldOutCallSlot)
                if (
                    slot.ordinal != call.ordinal
                    or slot.call_spec_hash != call.content_hash
                    or slot.envelope != envelope
                    or slot.session_identity != identity
                ):
                    raise HeldOutExecutionError("durable call slot differs from the current plan")
                snapshot = slot.schedule_before
                if source_unavailable or condition_stopped_after_timeout:
                    result = _unstarted_failure_result(
                        call=call,
                        snapshot=snapshot,
                        failure_code=(
                            "source_c1_unavailable"
                            if source_unavailable
                            else "condition_service_unavailable_after_timeout"
                        ),
                    )
                    after_snapshot = session.finalize_unstarted_call(call, result)
                else:
                    result = session.recover_call(call, envelope)
                    if result is None:
                        raise InterruptedCallRecoveryRequired(
                            f"durable slot {call.call_id} has no recoverable service receipt"
                        )
                    after_snapshot = session.schedule_snapshot()
            else:
                snapshot = session.schedule_snapshot()
                if snapshot.global_accounting_id != identity.global_accounting_id:
                    raise HeldOutExecutionError(
                        "session snapshot uses another global accounting scope"
                    )
                if snapshot.actual_allocated_gpu_seconds < previous_allocated:
                    raise HeldOutExecutionError(
                        "global GPU allocation counter regressed across sessions"
                    )
                if condition_just_started and (
                    last_shutdown_snapshot is None
                    or not math.isclose(
                        snapshot.remaining_registered_p95_seconds,
                        last_shutdown_snapshot.remaining_registered_p95_seconds
                        - PRIMARY_SERVICE_START_P95_SECONDS,
                        rel_tol=0.0,
                        abs_tol=1e-6,
                    )
                ):
                    raise HeldOutExecutionError(
                        "condition service allocation was not charged exactly once"
                    )
                if (
                    condition_just_started
                    and snapshot.ledger_chain_hash != identity.global_ledger_chain_hash
                ):
                    raise HeldOutExecutionError(
                        "new condition identity does not bind its post-load ledger state"
                    )
                admit_call(call=call, snapshot=snapshot, configuration=configuration)
                slot = HeldOutCallSlot(
                    ordinal=call.ordinal,
                    call_spec_hash=call.content_hash,
                    envelope=envelope,
                    session_identity=identity,
                    schedule_before=snapshot,
                )
                _append_exact(slot_path, slot)
                result = None
            if not recovering and (source_unavailable or condition_stopped_after_timeout):
                result = _unstarted_failure_result(
                    call=call,
                    snapshot=snapshot,
                    failure_code=(
                        "source_c1_unavailable"
                        if source_unavailable
                        else "condition_service_unavailable_after_timeout"
                    ),
                )
                after_snapshot = session.finalize_unstarted_call(call, result)
            elif not recovering:
                try:
                    result = session.execute_call(call, envelope)
                except BaseException as error:
                    if session.recovery_pending(call, envelope) or isinstance(
                        error, KeyboardInterrupt
                    ):
                        raise InterruptedCallRecoveryRequired(
                            f"call {call.call_id} was durably slotted and requires recovery"
                        ) from error
                    raise
                try:
                    after_snapshot = session.schedule_snapshot()
                except BaseException as error:
                    # A session may return only after committing its semantic
                    # receipt, so a later snapshot failure is resumable.
                    raise InterruptedCallRecoveryRequired(
                        f"call {call.call_id} committed a result and requires recovery"
                    ) from error
            if result is None:
                raise InterruptedCallRecoveryRequired(
                    f"durable slot {call.call_id} has no recoverable service receipt"
                )
            _result_matches_call(call, envelope, result, identity)
            runtime.validate_result_artifacts(call, envelope, result)
            _assert_accounting_transition(
                call=call,
                before=snapshot,
                after=after_snapshot,
                result=result,
                identity=identity,
                configuration=configuration,
            )
            record = HeldOutITTRecord(
                ordinal=call.ordinal,
                call_spec_hash=call.content_hash,
                result=result,
                session_identity_hash=identity.content_hash,
                call_slot_hash=slot.content_hash,
                schedule_before=snapshot,
                schedule_after=after_snapshot,
            )
            _append_exact(itt_path, record)
        if record.call_spec_hash != call.content_hash or record.ordinal != call.ordinal:
            raise HeldOutExecutionError("resumed ITT record differs from its registered call")
        if record.session_identity_hash != identity.content_hash:
            raise HeldOutExecutionError("resumed ITT record uses another service identity")
        if record.call_slot_hash != slot.content_hash:
            raise HeldOutExecutionError("resumed ITT record uses another durable call slot")
        _assert_accounting_transition(
            call=call,
            before=record.schedule_before,
            after=record.schedule_after,
            result=record.result,
            identity=identity,
            configuration=configuration,
        )
        if record.schedule_before.actual_allocated_gpu_seconds < previous_allocated:
            raise HeldOutExecutionError("recorded global GPU allocation regressed across calls")
        previous_allocated = record.schedule_after.actual_allocated_gpu_seconds
        _result_matches_call(call, envelope, record.result, identity)
        runtime.validate_result_artifacts(call, envelope, record.result)
        records.append(record)
        completed_results[call.call_id] = record.result
        if record.result.outcome is RunOutcome.TIMED_OUT:
            condition_timeout_sources.setdefault(call.condition, record.result.content_hash)

    final_identity = identity_by_condition[active_condition]
    final_shutdown_path = root / "shutdown" / f"{active_condition.value}.json"
    final_shutdown, final_shutdown_snapshot = runtime.shutdown_condition(final_identity)
    if final_shutdown_snapshot != final_shutdown.schedule_after_shutdown:
        raise HeldOutExecutionError(
            "final shutdown did not return its persisted transition snapshot"
        )
    if final_shutdown_path.exists():
        persisted_shutdown = _load(final_shutdown_path, HeldOutServiceShutdownReceipt)
        if persisted_shutdown != final_shutdown:
            raise HeldOutExecutionError("final service shutdown recovery changed")
    else:
        _append_exact(final_shutdown_path, final_shutdown)
    shutdown_by_condition[active_condition] = final_shutdown
    last_shutdown_snapshot = final_shutdown_snapshot
    previous_allocated = final_shutdown_snapshot.actual_allocated_gpu_seconds

    c1_sources = {
        (call.unit_id, call.seed_block): record.result
        for call, record in zip(manifest.calls, records, strict=True)
        if call.call_class == "test_c1"
    }
    projections: list[PreconstructedProjectionReceipt] = []
    for unit in manifest.units:
        c0 = next(item for item in c0_receipts if item.unit_id == unit.unit_id)
        for sealed_query in unit.query_stages:
            opening = query_openings[sealed_query.staging_manifest_hash]
            query = opening.opened_stage
            projection_jobs = [
                (
                    ConditionName.C0_CLASSICAL_PRE,
                    None,
                    c0.construction_seal_hash,
                    c0.complete_graph_hash,
                )
            ]
            projection_jobs.extend(
                (
                    ConditionName.C1_LLM_PRE,
                    seed,
                    c1_sources[(unit.unit_id, seed)].construction_seal_hash,
                    c1_sources[(unit.unit_id, seed)].complete_c1_graph_hash,
                )
                for seed in (1, 2)
            )
            for condition, seed, seal_hash, graph_hash in projection_jobs:
                suffix = "c0" if seed is None else f"c1-s{seed}"
                path = root / "projections" / f"{unit.unit_id}-{query.stage_id}-{suffix}.json"
                if path.exists():
                    receipt = _load(path, PreconstructedProjectionReceipt)
                elif seal_hash is None or graph_hash is None:
                    source_failure = (
                        c0.content_hash
                        if condition is ConditionName.C0_CLASSICAL_PRE
                        else c1_sources[(unit.unit_id, seed)].content_hash
                    )
                    receipt = cpu.project_unavailable(
                        unit=unit,
                        query_stage_hash=query.staging_manifest_hash,
                        query_opening=opening,
                        condition=condition,
                        seed_block=seed,
                        source_failure_hash=source_failure,
                    )
                    _append_exact(path, receipt)
                else:
                    receipt = cpu.project_preconstructed(
                        unit=unit,
                        query_stage_hash=query.staging_manifest_hash,
                        query_opening=opening,
                        condition=condition,
                        seed_block=seed,
                        construction_seal_hash=seal_hash,
                        complete_graph_hash=graph_hash,
                    )
                    _append_exact(path, receipt)
                if (
                    receipt.unit_id != unit.unit_id
                    or receipt.query_stage_hash != query.staging_manifest_hash
                    or receipt.condition is not condition
                    or receipt.seed_block != seed
                    or receipt.source_construction_seal_hash != seal_hash
                    or receipt.source_complete_graph_hash != graph_hash
                ):
                    raise HeldOutExecutionError(
                        "preconstructed projection receipt differs from its registered source"
                    )
                projections.append(receipt)
    projection_tuple = tuple(projections)
    record_tuple = tuple(records)
    _assert_query_fairness(manifest, projection_tuple, record_tuple)
    if prequery_barrier is None:
        raise HeldOutExecutionError("held-out run completed without a pre-query barrier")
    _assert_construction_lineage(
        manifest,
        tuple(c0_receipts),
        tuple(c2_prequery_receipts),
        tuple(ablation_prequery_receipts),
        prequery_barrier,
        projection_tuple,
        record_tuple,
    )
    final_path = root / "final_schedule.json"
    if final_path.exists():
        final_snapshot = _load(final_path, GlobalGpuScheduleSnapshot)
    else:
        if last_shutdown_snapshot is None:
            raise HeldOutExecutionError("final condition service was not shut down")
        final_snapshot = last_shutdown_snapshot
        _append_exact(final_path, final_snapshot)
    identities = tuple(identity_by_condition[item] for item in ordered_conditions)
    shutdown_receipts = tuple(shutdown_by_condition[item] for item in ordered_conditions)
    if final_snapshot.actual_allocated_gpu_seconds < previous_allocated:
        raise HeldOutExecutionError("final global GPU counter regressed")
    if final_snapshot.global_accounting_id != c1_identity.global_accounting_id:
        raise HeldOutExecutionError("final snapshot uses another global accounting scope")
    if (
        final_snapshot.gpu_call_inventory_file_sha256
        != configuration.gpu_call_inventory_file_sha256
    ):
        raise HeldOutExecutionError("final snapshot uses another GPU call inventory")
    if (
        final_snapshot.development_execution_result_hash
        != manifest.development_execution_result_hash
        or final_snapshot.development_predecessor_allocated_gpu_seconds
        != initial_snapshot.development_predecessor_allocated_gpu_seconds
    ):
        raise HeldOutExecutionError("final snapshot changed its development predecessor")
    if record_tuple and (
        final_snapshot.actual_allocated_gpu_seconds
        < record_tuple[-1].schedule_after.actual_allocated_gpu_seconds
        or not math.isclose(
            final_snapshot.remaining_registered_p95_seconds,
            record_tuple[-1].schedule_after.remaining_registered_p95_seconds,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
    ):
        raise HeldOutExecutionError(
            "post-shutdown GPU schedule does not continue the last ITT transition"
        )
    shutdown_by_condition = {item.condition: item for item in shutdown_receipts}
    if set(shutdown_by_condition) != set(ordered_conditions):
        raise HeldOutExecutionError("shutdown receipts omit a held-out condition service")
    for identity in identities:
        receipt = shutdown_by_condition[identity.condition]
        if (
            receipt.session_identity_hash != identity.content_hash
            or receipt.allocation_event_id != identity.allocation_event_id
            or receipt.model_load_event_id != identity.model_load_event_id
            or receipt.model_load_event_hash != identity.model_load_event_hash
            or receipt.global_accounting_id != identity.global_accounting_id
        ):
            raise HeldOutExecutionError(
                "service shutdown does not close its exact allocation event"
            )
    if (
        shutdown_by_condition[ConditionName.A_FIXED_SELECT].final_ledger_chain_hash
        != final_snapshot.ledger_chain_hash
    ):
        raise HeldOutExecutionError("final shutdown does not close the cumulative ledger")
    if final_snapshot.actual_allocated_gpu_seconds >= configuration.hard_gpu_seconds_limit:
        raise HeldOutExecutionError("final GPU ledger reached the hard 10-hour stop")
    if (
        final_snapshot.actual_allocated_gpu_seconds
        + final_snapshot.remaining_registered_p95_seconds
        > configuration.scheduled_gpu_seconds_limit + 1e-6
    ):
        raise HeldOutExecutionError("final mandatory forecast exceeds the 9-hour schedule")
    completed_at = completion_clock()
    if completed_at.tzinfo is None or completed_at.utcoffset() is None:
        raise HeldOutExecutionError("execution completion clock must be timezone-aware")
    completed_at = completed_at.astimezone(UTC)
    if completed_at < final_snapshot.captured_at:
        raise HeldOutExecutionError("execution completion predates the final GPU snapshot")
    execution = HeldOutExecutionManifest(
        execution_id=f"held-out-{manifest.content_hash[:20]}",
        call_manifest_hash=manifest.content_hash,
        review_completion_manifest_hash=reviewed_plan.review_completion_manifest_hash,
        final_reviewed_seal_hash=reviewed_plan.final_reviewed_seal_hash,
        runtime_binding_hash=reviewed_plan.runtime_binding_hash,
        session_identities=identities,
        c0_constructions=tuple(c0_receipts),
        c2_prequery_receipts=tuple(c2_prequery_receipts),
        ablation_prequery_receipts=tuple(ablation_prequery_receipts),
        prequery_barrier=prequery_barrier,
        query_openings=tuple(
            query_openings[item.staging_manifest_hash]
            for unit in manifest.units
            for item in unit.query_stages
        ),
        service_shutdown_receipts=shutdown_receipts,
        preconstructed_projections=projection_tuple,
        itt_records=record_tuple,
        initial_schedule_snapshot_hash=initial_snapshot.content_hash,
        final_schedule_snapshot_hash=final_snapshot.content_hash,
        completed_at=completed_at,
    )
    _append_exact(execution_path, execution)
    _audit_journal_tree(root, manifest, require_complete=True)
    return execution


def authorize_scorer_bridge(
    execution: HeldOutExecutionManifest,
    *,
    call_manifest: HeldOutCallManifest,
    output_root: Path,
    runtime: InjectedHeldOutRuntime,
    authorized_at: AwareDatetime,
) -> ScorerBridgeAuthorization:
    """Create a hash-only scorer handoff after the gold-free runtime is closed."""

    root = output_root.absolute()
    _audit_journal_tree(root, call_manifest, require_complete=True)
    persisted = _load(root / "execution_manifest.json", HeldOutExecutionManifest)
    if persisted != execution:
        raise HeldOutExecutionError(
            "scorer bridge execution differs from the frozen journal manifest"
        )
    _replay_complete_journal(root, call_manifest, execution, runtime)
    if execution.call_manifest_hash != call_manifest.content_hash:
        raise HeldOutExecutionError("scorer bridge uses another held-out call manifest")
    if authorized_at <= execution.completed_at:
        raise HeldOutExecutionError(
            "scorer bridge authorization must strictly follow runtime closure"
        )
    for call, record in zip(call_manifest.calls, execution.itt_records, strict=True):
        if (
            record.ordinal != call.ordinal
            or record.call_spec_hash != call.content_hash
            or record.result.call_id != call.call_id
            or record.result.condition is not call.condition
        ):
            raise HeldOutExecutionError("scorer bridge ITT inventory differs from the plan")
    _assert_query_fairness(
        call_manifest,
        execution.preconstructed_projections,
        execution.itt_records,
    )
    _assert_construction_lineage(
        call_manifest,
        execution.c0_constructions,
        execution.c2_prequery_receipts,
        execution.ablation_prequery_receipts,
        execution.prequery_barrier,
        execution.preconstructed_projections,
        execution.itt_records,
    )

    outputs = tuple(
        item.result.output_artifact_hash
        for item in execution.itt_records
        if item.result.output_artifact_hash is not None
    ) + tuple(
        item.projection_artifact_hash
        for item in execution.preconstructed_projections
        if item.projection_artifact_hash is not None
    )
    return ScorerBridgeAuthorization(
        execution_manifest_hash=execution.content_hash,
        call_manifest_hash=execution.call_manifest_hash,
        final_reviewed_seal_hash=execution.final_reviewed_seal_hash,
        output_artifact_hashes=outputs,
        c0_receipt_hashes=tuple(item.content_hash for item in execution.c0_constructions),
        projection_receipt_hashes=tuple(
            item.content_hash for item in execution.preconstructed_projections
        ),
        c2_prequery_receipt_hashes=tuple(
            item.content_hash for item in execution.c2_prequery_receipts
        ),
        ablation_prequery_receipt_hashes=tuple(
            item.content_hash for item in execution.ablation_prequery_receipts
        ),
        prequery_barrier_hash=execution.prequery_barrier.content_hash,
        query_opening_hashes=tuple(item.content_hash for item in execution.query_openings),
        service_shutdown_receipt_hashes=tuple(
            item.content_hash for item in execution.service_shutdown_receipts
        ),
        itt_record_hashes=tuple(item.content_hash for item in execution.itt_records),
        authorized_at=authorized_at,
    )


__all__ = [
    "C0ConstructionReceipt",
    "HeldOutCallSlot",
    "HeldOutExecutionError",
    "HeldOutExecutionManifest",
    "HeldOutITTRecord",
    "InjectedHeldOutCpu",
    "InterruptedCallRecoveryRequired",
    "PreconstructedProjectionReceipt",
    "ScorerBridgeAuthorization",
    "authorize_scorer_bridge",
    "execute_reviewed_held_out_manifest",
]
