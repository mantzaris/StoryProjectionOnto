"""Crash-safe closure of held-out primary results for scorer-only consumers.

The held-out execution journal must remain a closed, auditable namespace.  This
module therefore copies its terminal execution manifest into a separate
restricted directory and publishes the scorer bridge and Phase 5 prerequisite
gate beside that copy.  Every file is canonical, append-only, and safe to replay
after an interrupted publication.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

from story_projection_onto.contracts import ConditionName, ImmutableRecord
from story_projection_onto.held_out_controller import (
    HeldOutExecutionError,
    HeldOutExecutionManifest,
    ScorerBridgeAuthorization,
    authorize_scorer_bridge,
)
from story_projection_onto.held_out_primary import (
    HeldOutCallManifest,
    InjectedHeldOutRuntime,
    ReviewedHeldOutPlan,
)
from story_projection_onto.phase5_execution import PrimaryHeldOutResultsGate

DEFAULT_HELD_OUT_RESULTS_ROOT = Path("artifacts/restricted/held_out")
EXECUTION_COPY_NAME = "held_out_execution_manifest.json"
SCORER_BRIDGE_NAME = "scorer_bridge.json"
PRIMARY_RESULTS_GATE_NAME = "primary_results_gate.json"
_EXPECTED_NAMES = frozenset(
    {EXECUTION_COPY_NAME, SCORER_BRIDGE_NAME, PRIMARY_RESULTS_GATE_NAME}
)


class HeldOutResultsClosureError(HeldOutExecutionError):
    """The held-out result handoff cannot be reproduced or published safely."""


def _canonical_bytes(record: ImmutableRecord) -> bytes:
    return (record.to_canonical_json() + "\n").encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _assert_no_symlink_chain(path: Path) -> None:
    current = path.absolute()
    while True:
        if current.is_symlink():
            raise HeldOutResultsClosureError(
                f"symlinked held-out result path is forbidden: {path}"
            )
        if current.parent == current:
            return
        current = current.parent


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _append_exact_bytes(path: Path, payload: bytes) -> bool:
    _assert_no_symlink_chain(path)
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise HeldOutResultsClosureError(
                f"append-only held-out result drift: {path}"
            )
        return False
    parent_existed = path.parent.exists()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    _assert_no_symlink_chain(path.parent)
    if not parent_existed:
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
            os.link(temporary, path)
            published = True
        except FileExistsError:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
                raise HeldOutResultsClosureError(
                    f"concurrent held-out result drift: {path}"
                ) from None
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    _fsync_directory(path.parent)
    return published


def _append_exact(path: Path, record: ImmutableRecord) -> bool:
    return _append_exact_bytes(path, _canonical_bytes(record))


def _load_exact(path: Path, model_type):
    _assert_no_symlink_chain(path)
    if not path.is_file():
        raise HeldOutResultsClosureError(f"held-out result artifact is absent: {path}")
    raw = path.read_bytes()
    try:
        record = model_type.model_validate_json(raw)
    except Exception as error:
        raise HeldOutResultsClosureError(
            f"invalid held-out result artifact {path.name}: {error}"
        ) from error
    if raw != _canonical_bytes(record):
        raise HeldOutResultsClosureError(
            f"held-out result artifact is not canonical: {path.name}"
        )
    return record, raw


def _strictly_after(clock: Callable[[], datetime], predecessor: datetime) -> datetime:
    observed = clock()
    if observed.tzinfo is None or observed.utcoffset() is None:
        raise HeldOutResultsClosureError("held-out result clock must be timezone-aware")
    observed = observed.astimezone(UTC)
    return max(observed, predecessor.astimezone(UTC) + timedelta(microseconds=1))


def _audit_results_directory(root: Path) -> None:
    if not root.exists():
        return
    _assert_no_symlink_chain(root)
    if not root.is_dir():
        raise HeldOutResultsClosureError("held-out result root is not a directory")
    observed: set[str] = set()
    for item in root.iterdir():
        is_interrupted_temporary = any(
            item.name.startswith(f".{expected}.") and item.name.endswith(".tmp")
            for expected in _EXPECTED_NAMES
        )
        if is_interrupted_temporary:
            if item.is_symlink() or not item.is_file():
                raise HeldOutResultsClosureError(
                    f"invalid interrupted held-out result artifact: {item.name}"
                )
            # Preserve an abrupt-power-loss temporary as forensic evidence.  It
            # is not authoritative and cannot shadow the append-only target.
            continue
        observed.add(item.name)
    unexpected = observed - _EXPECTED_NAMES
    if unexpected:
        raise HeldOutResultsClosureError(
            "unexpected held-out result artifact: " + sorted(unexpected)[0]
        )
    allowed_prefixes = (
        frozenset(),
        frozenset({EXECUTION_COPY_NAME}),
        frozenset({EXECUTION_COPY_NAME, SCORER_BRIDGE_NAME}),
        _EXPECTED_NAMES,
    )
    if frozenset(observed) not in allowed_prefixes:
        raise HeldOutResultsClosureError(
            "held-out result publication is incomplete or out of order"
        )


def close_held_out_primary_results(
    *,
    reviewed_plan: ReviewedHeldOutPlan,
    call_manifest: HeldOutCallManifest,
    execution: HeldOutExecutionManifest,
    held_out_output_root: Path,
    results_root: Path,
    runtime: InjectedHeldOutRuntime,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> tuple[ScorerBridgeAuthorization, PrimaryHeldOutResultsGate]:
    """Replay, authorize, and append the scorer bridge and primary-results gate.

    A rerun after any successful append reuses the retained authorization and
    freeze timestamps.  It cannot silently replace an existing result closure.
    """

    source_root = held_out_output_root.absolute()
    target_root = results_root.absolute()
    _assert_no_symlink_chain(source_root)
    _assert_no_symlink_chain(target_root)
    if (
        source_root == target_root
        or source_root in target_root.parents
        or target_root in source_root.parents
    ):
        raise HeldOutResultsClosureError(
            "held-out result gate must use a separate sibling namespace"
        )
    if reviewed_plan.call_manifest != call_manifest:
        raise HeldOutResultsClosureError("reviewed plan and held-out call manifest differ")
    if (
        execution.call_manifest_hash != call_manifest.content_hash
        or execution.review_completion_manifest_hash
        != reviewed_plan.review_completion_manifest_hash
        or execution.final_reviewed_seal_hash != reviewed_plan.final_reviewed_seal_hash
    ):
        raise HeldOutResultsClosureError(
            "held-out execution does not bind the reviewed launch authority"
        )

    source_execution, source_bytes = _load_exact(
        source_root / "execution_manifest.json", HeldOutExecutionManifest
    )
    if source_execution != execution:
        raise HeldOutResultsClosureError(
            "terminal held-out journal differs from the supplied execution"
        )
    _audit_results_directory(target_root)

    execution_copy_path = target_root / EXECUTION_COPY_NAME
    bridge_path = target_root / SCORER_BRIDGE_NAME
    gate_path = target_root / PRIMARY_RESULTS_GATE_NAME

    existing_bridge: ScorerBridgeAuthorization | None = None
    if bridge_path.exists():
        existing_bridge, _ = _load_exact(bridge_path, ScorerBridgeAuthorization)
        authorized_at = existing_bridge.authorized_at
    else:
        authorized_at = _strictly_after(clock, execution.completed_at)
    bridge = authorize_scorer_bridge(
        execution,
        call_manifest=call_manifest,
        output_root=source_root,
        runtime=runtime,
        authorized_at=authorized_at,
    )
    if existing_bridge is not None and existing_bridge != bridge:
        raise HeldOutResultsClosureError(
            "retained scorer bridge differs from replayed authorization"
        )

    _append_exact_bytes(execution_copy_path, source_bytes)
    _append_exact(bridge_path, bridge)

    existing_gate: PrimaryHeldOutResultsGate | None = None
    if gate_path.exists():
        existing_gate, _ = _load_exact(gate_path, PrimaryHeldOutResultsGate)
        frozen_at = existing_gate.frozen_at
    else:
        frozen_at = _strictly_after(clock, bridge.authorized_at)
    gate = PrimaryHeldOutResultsGate(
        gate_id=f"held-out-primary-{execution.content_hash[:20]}",
        lifecycle_state="held_out_primary_results_frozen",
        benchmark_draft_seal_hash=reviewed_plan.review_draft_seal_hash,
        final_reviewed_seal_hash=reviewed_plan.final_reviewed_seal_hash,
        held_out_execution_manifest_relative_path=EXECUTION_COPY_NAME,
        held_out_execution_manifest_hash=execution.content_hash,
        held_out_execution_manifest_file_sha256=_sha256(source_bytes),
        scorer_bridge_relative_path=SCORER_BRIDGE_NAME,
        scorer_bridge_hash=bridge.content_hash,
        scorer_bridge_file_sha256=_sha256(_canonical_bytes(bridge)),
        result_artifact_hashes=bridge.output_artifact_hashes,
        conditions=(
            ConditionName.C0_CLASSICAL_PRE,
            ConditionName.C1_LLM_PRE,
            ConditionName.C2_LLM_QUERY,
            ConditionName.A_FIXED_SELECT,
        ),
        frozen_at=frozen_at,
    )
    if existing_gate is not None and existing_gate != gate:
        raise HeldOutResultsClosureError(
            "retained primary-results gate differs from replayed closure"
        )
    _append_exact(gate_path, gate)
    _audit_results_directory(target_root)

    copied_execution, copied_bytes = _load_exact(
        execution_copy_path, HeldOutExecutionManifest
    )
    persisted_bridge, bridge_bytes = _load_exact(
        bridge_path, ScorerBridgeAuthorization
    )
    persisted_gate, _ = _load_exact(gate_path, PrimaryHeldOutResultsGate)
    if (
        copied_execution != execution
        or copied_bytes != source_bytes
        or persisted_bridge != bridge
        or persisted_gate != gate
        or gate.held_out_execution_manifest_file_sha256 != _sha256(copied_bytes)
        or gate.scorer_bridge_file_sha256 != _sha256(bridge_bytes)
        or gate.result_artifact_hashes != bridge.output_artifact_hashes
        or gate.frozen_at <= bridge.authorized_at
    ):
        raise HeldOutResultsClosureError(
            "persisted primary-results closure failed byte/logical replay"
        )
    return bridge, gate


__all__ = [
    "DEFAULT_HELD_OUT_RESULTS_ROOT",
    "HeldOutResultsClosureError",
    "close_held_out_primary_results",
]
