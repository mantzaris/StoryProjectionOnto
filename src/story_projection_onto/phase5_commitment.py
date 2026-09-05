"""Pre-output commitment for the six registered scripted Phase 5 revisions.

The commitment path deliberately does not author a correction, inspect a condition
output, or open scorer-only data.  It accepts six explicitly authored typed
instructions, verifies them against only the public model-visible query stages, and
freezes their exact hashes in the restricted CAS before the held-out journal starts.

``UserRevision.created_at`` is the precommitted activation time for a scripted
revision.  The separately recorded ``committed_at`` is the observed wall-clock time
at which the instruction became immutable.  Keeping both fields makes the required
pre-output commitment compatible with the later rule that a scripted revision is
applied only after its parent projection has completed, without backdating either
event.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Self

from pydantic import AwareDatetime, model_validator

from story_projection_onto.benchmark_runtime import (
    RuntimeStagingManifest,
    scan_model_payload,
)
from story_projection_onto.contracts import (
    FeedbackAction,
    Identifier,
    ImmutableRecord,
    QueryContext,
    ReleaseClass,
    Sha256Digest,
)
from story_projection_onto.feedback_runtime import (
    FeedbackProtocolConfiguration,
    ScriptedRevisionFreeze,
)
from story_projection_onto.held_out_primary import (
    HeldOutControlConfiguration,
    HeldOutControlError,
    ReviewedHeldOutPlan,
    _derive_call_manifest,
    load_executable_held_out_configuration,
)
from story_projection_onto.independent_review_runtime import (
    IndependentReviewCompletionManifest,
)
from story_projection_onto.phase5_production import (
    Phase5CASReference,
    Phase5MaterializationError,
    persist_phase5_record,
)
from story_projection_onto.store import ArtifactStore
from story_projection_onto.synthetic_benchmark import (
    FinalReviewedSeal,
    load_model_eligible_query,
)
from story_projection_onto.ui import RevisionInstruction

SCRIPT_COUNT = 6
DEFAULT_COMMITMENT_OUTPUT_ROOT = Path("artifacts/restricted/phase5_script_commitment")
REGISTERED_REPAIR_RESERVE_COUNTS = {
    "reserve_long": 4,
    "reserve_standard": 8,
    "reserve_short": 4,
}
HELD_OUT_SERVICE_COUNT = 3

class Phase5CommitmentError(Phase5MaterializationError):
    """The scripted revision inventory cannot be committed without leakage."""


class ScriptedRevisionDraft(ImmutableRecord):
    """One researcher-authored instruction labelled by its registered episode."""

    episode_id: Identifier
    instruction: RevisionInstruction
    condition_output_inspected: Literal[False] = False


class ScriptedRevisionDraftManifest(ImmutableRecord):
    """Exact six-instruction input authored before any condition output is read.

    ``scheduled_activation_at`` is intentionally explicit.  It is not an assertion
    that the revision already occurred; it is the not-before timestamp embedded in
    each future ``UserRevision`` and must follow the actual commitment time.
    """

    draft_id: Identifier
    protocol_hash: Sha256Digest
    benchmark_draft_seal_hash: Sha256Digest
    drafts: tuple[
        ScriptedRevisionDraft,
        ScriptedRevisionDraft,
        ScriptedRevisionDraft,
        ScriptedRevisionDraft,
        ScriptedRevisionDraft,
        ScriptedRevisionDraft,
    ]
    authored_at: AwareDatetime
    scheduled_activation_at: AwareDatetime
    source_namespace: Literal["public_model_visible_query_stages"] = (
        "public_model_visible_query_stages"
    )
    condition_output_inspected: Literal[False] = False
    scorer_gold_read: Literal[False] = False
    model_service_called: Literal[False] = False

    @model_validator(mode="after")
    def exact_scheduled_inventory(self) -> Self:
        episode_ids = tuple(item.episode_id for item in self.drafts)
        if len(set(episode_ids)) != SCRIPT_COUNT:
            raise ValueError("script draft episode IDs must be unique")
        revision_ids = tuple(item.instruction.revision.revision_id for item in self.drafts)
        if len(set(revision_ids)) != SCRIPT_COUNT:
            raise ValueError("script draft revision IDs must be unique")
        if self.authored_at >= self.scheduled_activation_at:
            raise ValueError("script activation must strictly follow draft authorship")
        if any(
            item.instruction.revision.created_at != self.scheduled_activation_at
            for item in self.drafts
        ):
            raise ValueError("every scripted revision must use the declared activation time")
        return self


class Phase5ReviewGateBinding(ImmutableRecord):
    """Hash-only review certificates opened before any selected query payload."""

    review_completion_manifest_hash: Sha256Digest
    review_completion_manifest_file_sha256: Sha256Digest
    final_reviewed_seal_hash: Sha256Digest
    final_reviewed_seal_file_sha256: Sha256Digest
    review_draft_seal_hash: Sha256Digest
    reviewed_projection_count: Literal[9] = 9
    held_out_launch_authorized: Literal[True] = True
    condition_outputs_generated_before_review: Literal[False] = False
    gold_objects_opened: Literal[False] = False


class Phase5ParentReadinessFloor(ImmutableRecord):
    """Registered watchdog floor before end-of-run C0/C1 projections can exist.

    The controller materializes the selected C0/C1 query projections after the
    entire ordered GPU call loop.  The CPU projection tail has no registered
    watchdog, so this is a conservative *floor*, not a claimed wall-time upper
    bound.  Exact chronology is checked again against durable outputs later.
    """

    held_out_call_manifest_hash: Sha256Digest
    held_out_control_configuration_hash: Sha256Digest
    held_out_runtime_binding_hash: Sha256Digest
    selected_c1_parent_call_ordinals: tuple[int, int, int, int, int, int]
    selected_c2_parent_call_ordinals: tuple[int, int, int, int, int, int]
    latest_required_gpu_call_ordinal: Literal[168] = 168
    service_start_count: Literal[3] = 3
    base_call_watchdog_seconds: float
    repair_reserve_watchdog_seconds: float
    service_start_watchdog_seconds: float
    registered_watchdog_floor_seconds: float
    c0_c1_projection_phase: Literal["after_all_gpu_calls_and_final_shutdown"] = (
        "after_all_gpu_calls_and_final_shutdown"
    )
    cpu_projection_tail_has_registered_watchdog: Literal[False] = False
    post_output_chronology_revalidation_required: Literal[True] = True

    @model_validator(mode="after")
    def exact_registered_floor(self) -> Self:
        components = (
            self.base_call_watchdog_seconds
            + self.repair_reserve_watchdog_seconds
            + self.service_start_watchdog_seconds
        )
        if (
            any(item <= 0 for item in self.selected_c1_parent_call_ordinals)
            or any(item <= 0 for item in self.selected_c2_parent_call_ordinals)
            or len(set(self.selected_c1_parent_call_ordinals)) != SCRIPT_COUNT
            or len(set(self.selected_c2_parent_call_ordinals)) != SCRIPT_COUNT
            or any(
                c1 >= c2
                for c1, c2 in zip(
                    self.selected_c1_parent_call_ordinals,
                    self.selected_c2_parent_call_ordinals,
                    strict=True,
                )
            )
            or self.base_call_watchdog_seconds <= 0
            or self.repair_reserve_watchdog_seconds <= 0
            or self.service_start_watchdog_seconds <= 0
            or abs(components - self.registered_watchdog_floor_seconds) > 1e-6
        ):
            raise ValueError("Phase 5 parent-readiness watchdog floor is inconsistent")
        return self


class Phase5ScriptCommitmentEntry(ImmutableRecord):
    """Hash-only lineage for one committed instruction and anchor set."""

    episode_id: Identifier
    context_id: Identifier
    action: FeedbackAction
    query_stage_relative_path: Identifier
    query_stage_manifest_hash: Sha256Digest
    query_stage_manifest_file_sha256: Sha256Digest
    query_artifact_hash: Sha256Digest
    model_visible_evidence_hash: Sha256Digest
    instruction: Phase5CASReference
    freeze: Phase5CASReference
    scheduled_activation_at: AwareDatetime

    @model_validator(mode="after")
    def exact_restricted_references(self) -> Self:
        if self.instruction.object_kind != "revision_instruction":
            raise ValueError("script commitment instruction reference has the wrong kind")
        if self.freeze.object_kind != "scripted_revision_freeze":
            raise ValueError("script commitment freeze reference has the wrong kind")
        if (
            self.instruction.release_class is not ReleaseClass.RESTRICTED
            or self.freeze.release_class is not ReleaseClass.RESTRICTED
        ):
            raise ValueError("script commitment objects must remain restricted")
        relative = Path(self.query_stage_relative_path)
        if relative.is_absolute() or ".." in relative.parts or "\\" in str(relative):
            raise ValueError("script commitment query-stage path is unsafe")
        return self


class Phase5ScriptCommitmentManifest(ImmutableRecord):
    """Immutable proof that the exact scripts preceded every held-out output."""

    commitment_id: Identifier
    protocol_hash: Sha256Digest
    benchmark_draft_seal_hash: Sha256Digest
    review_gate: Phase5ReviewGateBinding
    parent_readiness_floor: Phase5ParentReadinessFloor
    draft_manifest_hash: Sha256Digest
    entries: tuple[
        Phase5ScriptCommitmentEntry,
        Phase5ScriptCommitmentEntry,
        Phase5ScriptCommitmentEntry,
        Phase5ScriptCommitmentEntry,
        Phase5ScriptCommitmentEntry,
        Phase5ScriptCommitmentEntry,
    ]
    committed_at: AwareDatetime
    scheduled_activation_at: AwareDatetime
    activation_delay_seconds: float
    timestamp_policy: Literal[
        "observed_commit_then_precommitted_future_activation_v1"
    ] = "observed_commit_then_precommitted_future_activation_v1"
    source_namespace: Literal["public_model_visible_query_stages"] = (
        "public_model_visible_query_stages"
    )
    held_out_journal_empty: Literal[True] = True
    condition_output_inspected: Literal[False] = False
    held_out_output_read: Literal[False] = False
    scorer_gold_read: Literal[False] = False
    model_service_called: Literal[False] = False

    @model_validator(mode="after")
    def exact_pre_output_inventory(self) -> Self:
        keys = tuple((item.episode_id, item.context_id) for item in self.entries)
        if len(set(keys)) != SCRIPT_COUNT:
            raise ValueError("script commitment requires six distinct episode/context pairs")
        actions = tuple(item.action for item in self.entries)
        if (
            actions.count(FeedbackAction.REFINE_CONTEXT) != 3
            or actions.count(FeedbackAction.REQUEST_MERGE_SPLIT) != 3
        ):
            raise ValueError("script commitment requires three revisions of each action")
        if self.scheduled_activation_at <= self.committed_at:
            raise ValueError("script activation must strictly follow commitment")
        observed_delay = (self.scheduled_activation_at - self.committed_at).total_seconds()
        if (
            observed_delay <= 0
            or abs(observed_delay - self.activation_delay_seconds) > 1e-6
            or observed_delay
            <= self.parent_readiness_floor.registered_watchdog_floor_seconds
        ):
            raise ValueError(
                "script activation does not follow the registered parent-readiness floor"
            )
        if self.benchmark_draft_seal_hash != self.review_gate.review_draft_seal_hash:
            raise ValueError("script commitment review/readiness lineage is inconsistent")
        if any(
            item.scheduled_activation_at != self.scheduled_activation_at
            for item in self.entries
        ):
            raise ValueError("script commitment entries use different activation times")
        return self


@dataclass(frozen=True, slots=True)
class PreparedPhase5ScriptCommitment:
    """In-memory records ready for either side-effect-free validation or persistence."""

    manifest: Phase5ScriptCommitmentManifest
    records: tuple[tuple[RevisionInstruction, ScriptedRevisionFreeze], ...]


def load_scripted_revision_draft(path: Path) -> ScriptedRevisionDraftManifest:
    """Load one bounded, regular, self-hashed draft file without following symlinks."""

    _assert_no_symlink_chain(path)
    if not path.is_file():
        raise Phase5CommitmentError("scripted revision draft is absent")
    if path.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise Phase5CommitmentError("scripted revision draft permissions are not restricted")
    raw = path.read_bytes()
    if len(raw) > 10 * 1024 * 1024:
        raise Phase5CommitmentError("scripted revision draft exceeds 10 MiB")
    try:
        return ScriptedRevisionDraftManifest.model_validate_json(raw)
    except Exception as error:
        raise Phase5CommitmentError(f"invalid scripted revision draft: {error}") from error


def load_phase5_script_commitment(path: Path) -> Phase5ScriptCommitmentManifest:
    """Load one immutable commitment manifest."""

    _assert_no_symlink_chain(path)
    if not path.is_file():
        raise Phase5CommitmentError("Phase 5 script commitment is absent")
    if path.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise Phase5CommitmentError("Phase 5 script commitment permissions are not restricted")
    raw = path.read_bytes()
    if len(raw) > 10 * 1024 * 1024:
        raise Phase5CommitmentError("Phase 5 script commitment exceeds 10 MiB")
    try:
        return Phase5ScriptCommitmentManifest.model_validate_json(raw)
    except Exception as error:
        raise Phase5CommitmentError(f"invalid Phase 5 script commitment: {error}") from error


def _normalise_aware(value: datetime, *, label: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise Phase5CommitmentError(f"{label} must be timezone-aware")
    return value.astimezone(UTC)


def _assert_no_symlink_chain(path: Path) -> None:
    current = path.absolute()
    while True:
        if current.is_symlink():
            raise Phase5CommitmentError("symlinked Phase 5 commitment paths are forbidden")
        if current.parent == current:
            return
        current = current.parent


def _load_restricted_review_record(
    path: Path,
    model_type: type[IndependentReviewCompletionManifest] | type[FinalReviewedSeal],
) -> tuple[IndependentReviewCompletionManifest | FinalReviewedSeal, str]:
    _assert_no_symlink_chain(path)
    if not path.is_file():
        raise Phase5CommitmentError("hash-only independent-review certificate is absent")
    if path.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise Phase5CommitmentError(
            "hash-only independent-review certificate permissions are not restricted"
        )
    raw = path.read_bytes()
    if len(raw) > 10 * 1024 * 1024:
        raise Phase5CommitmentError("hash-only independent-review certificate is oversized")
    try:
        return model_type.model_validate_json(raw), hashlib.sha256(raw).hexdigest()
    except Exception as error:
        raise Phase5CommitmentError(
            "hash-only independent-review certificate is invalid"
        ) from error


def open_phase5_hash_bound_reviewed_plan(
    *,
    repository: Path,
    restricted_root: Path,
    review_completion_root: Path,
    configuration_path: Path,
    runtime_binding_path: Path | None,
) -> tuple[ReviewedHeldOutPlan, HeldOutControlConfiguration, Phase5ReviewGateBinding]:
    """Open only review certificates, then derive the gold-free held-out plan.

    The completion manifest is itself the path-portable proof emitted by the full
    independent-review replay.  This consumer checks it against the final seal but
    deliberately does not open any reviewed projection, scorer binding, or gold
    object.  The ordinary held-out controller still performs its full review replay
    before execution.
    """

    repository = repository.resolve(strict=True)
    restricted_root = restricted_root.resolve(strict=True)
    review_root = _restricted_descendant(
        restricted_root,
        review_completion_root,
        label="independent-review completion root",
    )
    completion, completion_file_sha256 = _load_restricted_review_record(
        review_root / "completion_manifest.json",
        IndependentReviewCompletionManifest,
    )
    final_seal, final_seal_file_sha256 = _load_restricted_review_record(
        review_root / completion.final_seal_file,
        FinalReviewedSeal,
    )
    if (
        completion.final_seal_hash != final_seal.content_hash
        or completion.draft_seal_hash != final_seal.draft_seal_hash
        or completion.package_hash != final_seal.package_hash
        or completion.binding_manifest_hash != final_seal.binding_manifest_hash
        or completion.response_hash != final_seal.response_hash
        or completion.adjudication_hash != final_seal.adjudication_hash
        or tuple(item.artifact_hash for item in completion.reviewed_artifacts)
        != final_seal.reviewed_artifact_hashes
    ):
        raise Phase5CommitmentError(
            "independent-review completion manifest and final seal differ"
        )
    review_gate = Phase5ReviewGateBinding(
        review_completion_manifest_hash=completion.content_hash,
        review_completion_manifest_file_sha256=completion_file_sha256,
        final_reviewed_seal_hash=final_seal.content_hash,
        final_reviewed_seal_file_sha256=final_seal_file_sha256,
        review_draft_seal_hash=final_seal.draft_seal_hash,
    )

    runtime_binding_hash = None
    try:
        if runtime_binding_path is None:
            configuration = load_executable_held_out_configuration(
                repository=repository,
                configuration_path=configuration_path,
            )
        else:
            from story_projection_onto.held_out_binding import (
                load_runtime_bound_held_out_configuration,
            )

            configuration, runtime_binding = load_runtime_bound_held_out_configuration(
                repository=repository,
                configuration_path=configuration_path,
                binding_path=runtime_binding_path,
            )
            runtime_binding_hash = runtime_binding.content_hash
    except HeldOutControlError as error:
        raise Phase5CommitmentError(
            "runtime-bound held-out control cannot be reproduced"
        ) from error

    public_benchmark = repository / configuration.benchmark_manifest_path
    _assert_no_symlink_chain(public_benchmark)
    try:
        benchmark_manifest = json.loads(public_benchmark.read_bytes())
    except (OSError, ValueError) as error:
        raise Phase5CommitmentError("public benchmark manifest is invalid") from error
    if benchmark_manifest.get("draft_seal_hash") != review_gate.review_draft_seal_hash:
        raise Phase5CommitmentError(
            "hash-only review certificate and public benchmark seal differ"
        )

    # Derivation opens only the public stage manifests.  It occurs after the
    # review certificates above have been validated.
    try:
        call_manifest = _derive_call_manifest(repository, configuration)
    except HeldOutControlError as error:
        raise Phase5CommitmentError(
            "runtime-bound held-out call plan cannot be reproduced"
        ) from error
    reviewed_plan = ReviewedHeldOutPlan(
        call_manifest=call_manifest,
        review_completion_manifest_hash=review_gate.review_completion_manifest_hash,
        review_draft_seal_hash=review_gate.review_draft_seal_hash,
        final_reviewed_seal_hash=review_gate.final_reviewed_seal_hash,
        runtime_binding_hash=runtime_binding_hash,
    )
    return reviewed_plan, configuration, review_gate


def _restricted_descendant(restricted_root: Path, path: Path, *, label: str) -> Path:
    _assert_no_symlink_chain(restricted_root)
    _assert_no_symlink_chain(path)
    try:
        relative = path.absolute().relative_to(restricted_root.absolute())
    except ValueError as error:
        raise Phase5CommitmentError(f"{label} must remain under the restricted root") from error
    if not relative.parts:
        raise Phase5CommitmentError(f"{label} must be below the restricted root")
    return path.absolute()


def _assert_unstarted_journal(path: Path) -> None:
    """Require a physically empty/nonexistent held-out journal before commitment."""

    _assert_no_symlink_chain(path)
    if not path.exists():
        return
    if not path.is_dir():
        raise Phase5CommitmentError("held-out journal root is not a directory")
    for entry in path.rglob("*"):
        if entry.is_symlink():
            raise Phase5CommitmentError("held-out journal contains a symbolic link")
        if entry.is_file():
            raise Phase5CommitmentError(
                "held-out journal already contains a record; pre-output commitment is too late"
            )


def validate_phase5_commitment_destinations(
    *,
    repository: Path,
    restricted_root: Path,
    draft_path: Path,
    ledger_path: Path,
    artifact_root: Path,
    output_root: Path,
) -> None:
    """Validate the public/restricted path boundary without opening mutable stores."""

    repository = repository.resolve(strict=True)
    restricted_root = restricted_root.resolve(strict=True)
    if restricted_root != (repository / "artifacts/restricted").resolve(strict=True):
        raise Phase5CommitmentError("restricted root is not the canonical repository root")
    _restricted_descendant(restricted_root, draft_path, label="script draft")
    _restricted_descendant(restricted_root, ledger_path, label="commitment ledger")
    _restricted_descendant(restricted_root, output_root, label="commitment output root")
    if ledger_path.is_symlink() or not ledger_path.is_file():
        raise Phase5CommitmentError("Phase 5 commitment ledger is absent or unsafe")
    _assert_no_symlink_chain(artifact_root)
    if artifact_root.is_symlink() or not artifact_root.is_dir():
        raise Phase5CommitmentError("Phase 5 commitment CAS root is absent or unsafe")
    expected_blob_root = (repository / "artifacts/blobs").absolute()
    try:
        relative = artifact_root.absolute().relative_to(expected_blob_root)
    except ValueError as error:
        raise Phase5CommitmentError(
            "Phase 5 commitment CAS must remain under artifacts/blobs"
        ) from error
    if not relative.parts:
        raise Phase5CommitmentError("Phase 5 commitment requires a bounded CAS namespace")


def _cas_reference(value: ImmutableRecord, object_kind: str) -> Phase5CASReference:
    payload = (value.to_canonical_json() + "\n").encode("utf-8")
    return Phase5CASReference(
        artifact_hash=hashlib.sha256(payload).hexdigest(),
        logical_content_hash=value.content_hash,
        object_kind=object_kind,
        media_type=(
            "application/vnd.story-projection."
            f"{object_kind.replace('_', '-')}+json"
        ),
        release_class=ReleaseClass.RESTRICTED,
    )


def _context_from_public_reveal(context_id: str, reveal) -> QueryContext:
    expected_reveal_id = f"reveal_{context_id.removeprefix('ctx_')}"
    if reveal.reveal_id != expected_reveal_id:
        raise Phase5CommitmentError("public query reveal has another context identity")
    return QueryContext(
        context_id=context_id,
        revealed_at=reveal.revealed_at,
        **reveal.query.model_dump(
            mode="python",
            exclude={"schema_version", "content_hash"},
        ),
    )


def _assert_anchors_resolve(instruction: RevisionInstruction, evidence) -> None:
    evidence_ids = {item.evidence_id for item in evidence}
    mention_to_evidence = {
        mention.candidate_id: item.evidence_id
        for item in evidence
        for mention in item.mention_candidates
    }
    for anchor in instruction.revision.anchors:
        if not set(anchor.evidence_ids).issubset(evidence_ids):
            raise Phase5CommitmentError(
                "scripted revision anchor uses evidence outside its public query stage"
            )
        if not set(anchor.mention_candidate_ids).issubset(mention_to_evidence):
            raise Phase5CommitmentError(
                "scripted revision anchor uses a mention outside its public query stage"
            )
        if anchor.evidence_ids and any(
            mention_to_evidence[mention_id] not in anchor.evidence_ids
            for mention_id in anchor.mention_candidate_ids
        ):
            raise Phase5CommitmentError(
                "scripted revision mention is not supported by its named evidence"
            )


def _parent_readiness_floor(
    *,
    entries: tuple[Phase5ScriptCommitmentEntry, ...],
    reviewed_plan: ReviewedHeldOutPlan,
    configuration: HeldOutControlConfiguration,
) -> Phase5ParentReadinessFloor:
    manifest = reviewed_plan.call_manifest
    if (
        manifest.development_execution_result_hash == "PENDING"
        or reviewed_plan.runtime_binding_hash is None
        or configuration.expected_plan_hash == "PENDING"
        or configuration.expected_plan_hash != manifest.content_hash
        or manifest.gpu_call_inventory_file_sha256
        != configuration.gpu_call_inventory_file_sha256
    ):
        raise Phase5CommitmentError(
            "script commitment requires the runtime-bound post-development held-out plan"
        )
    expected_call_classes = (
        *("test_c1" for _ in range(24)),
        *("test_c2" for _ in range(72)),
        *("test_fixed_select" for _ in range(72)),
    )
    if tuple(item.call_class for item in manifest.calls) != expected_call_classes:
        raise Phase5CommitmentError(
            "held-out call order cannot establish the parent-readiness floor"
        )

    stages = {
        stage.staging_manifest_hash: (unit, stage)
        for unit in manifest.units
        for stage in unit.query_stages
    }
    c1_ordinals: list[int] = []
    c2_ordinals: list[int] = []
    for entry in entries:
        planned = stages.get(entry.query_stage_manifest_hash)
        if planned is None:
            raise Phase5CommitmentError(
                "script commitment query stage is absent from the held-out plan"
            )
        unit, stage = planned
        if (
            stage.relative_path != entry.query_stage_relative_path
            or stage.manifest_file_sha256
            != entry.query_stage_manifest_file_sha256
            or stage.query_artifact_hash != entry.query_artifact_hash
            or stage.evidence_artifact_hash != entry.model_visible_evidence_hash
        ):
            raise Phase5CommitmentError(
                "script commitment query stage differs from the held-out plan"
            )
        c1 = [
            item
            for item in manifest.calls
            if item.call_class == "test_c1"
            and item.unit_id == unit.unit_id
            and item.seed_block == 1
        ]
        c2 = [
            item
            for item in manifest.calls
            if item.call_class == "test_c2"
            and item.unit_id == unit.unit_id
            and item.seed_block == 1
            and item.query_stage_hash == stage.staging_manifest_hash
        ]
        if len(c1) != 1 or len(c2) != 1:
            raise Phase5CommitmentError(
                "script commitment lacks an exact seed-1 C1/C2 parent slot"
            )
        c1_ordinals.append(c1[0].ordinal)
        c2_ordinals.append(c2[0].ordinal)

    watchdog_by_reserve = {
        policy.repair_reserve_class: policy.watchdog_seconds
        for policy in configuration.call_policies
    }
    if set(watchdog_by_reserve) != set(REGISTERED_REPAIR_RESERVE_COUNTS):
        raise Phase5CommitmentError("held-out repair reserve policy changed")
    base_seconds = float(sum(item.watchdog_seconds for item in manifest.calls))
    repair_seconds = float(
        sum(
            REGISTERED_REPAIR_RESERVE_COUNTS[name] * watchdog_by_reserve[name]
            for name in REGISTERED_REPAIR_RESERVE_COUNTS
        )
    )
    service_seconds = float(
        HELD_OUT_SERVICE_COUNT * configuration.service_start_watchdog_seconds
    )
    return Phase5ParentReadinessFloor(
        held_out_call_manifest_hash=manifest.content_hash,
        held_out_control_configuration_hash=configuration.content_hash,
        held_out_runtime_binding_hash=reviewed_plan.runtime_binding_hash,
        selected_c1_parent_call_ordinals=tuple(c1_ordinals),
        selected_c2_parent_call_ordinals=tuple(c2_ordinals),
        base_call_watchdog_seconds=base_seconds,
        repair_reserve_watchdog_seconds=repair_seconds,
        service_start_watchdog_seconds=service_seconds,
        registered_watchdog_floor_seconds=(
            base_seconds + repair_seconds + service_seconds
        ),
    )


def prepare_phase5_script_commitment(
    *,
    draft: ScriptedRevisionDraftManifest,
    protocol: FeedbackProtocolConfiguration,
    reviewed_plan: ReviewedHeldOutPlan,
    held_out_configuration: HeldOutControlConfiguration,
    review_gate: Phase5ReviewGateBinding,
    repository: Path,
    restricted_root: Path,
    held_out_journal_root: Path,
    committed_at: datetime,
) -> PreparedPhase5ScriptCommitment:
    """Validate and assemble the exact six freezes without performing any write.

    The function opens only paths determined by the frozen public protocol under
    ``data/synthetic/model_visible/query_stages``.  It has no parameter through
    which a condition output or scorer object can be supplied.
    """

    repository = repository.resolve(strict=True)
    restricted_root = restricted_root.resolve(strict=True)
    canonical_restricted_root = (repository / "artifacts/restricted").resolve(strict=True)
    if restricted_root != canonical_restricted_root:
        raise Phase5CommitmentError("restricted root is not the canonical repository root")
    if (
        review_gate.review_completion_manifest_hash
        != reviewed_plan.review_completion_manifest_hash
        or review_gate.review_draft_seal_hash
        != reviewed_plan.review_draft_seal_hash
        or review_gate.final_reviewed_seal_hash
        != reviewed_plan.final_reviewed_seal_hash
        or review_gate.review_draft_seal_hash
        != protocol.benchmark_draft_seal_hash
    ):
        raise Phase5CommitmentError(
            "script commitment requires the exact hash-bound independent-review completion"
        )
    journal_root = _restricted_descendant(
        restricted_root,
        held_out_journal_root,
        label="held-out journal root",
    )
    _assert_unstarted_journal(journal_root)

    committed = _normalise_aware(committed_at, label="commitment time")
    authored = _normalise_aware(draft.authored_at, label="draft authorship time")
    scheduled = _normalise_aware(
        draft.scheduled_activation_at,
        label="scheduled revision activation",
    )
    if draft.protocol_hash != protocol.content_hash:
        raise Phase5CommitmentError("script draft uses another feedback protocol")
    if (
        draft.benchmark_draft_seal_hash != protocol.benchmark_draft_seal_hash
        or authored > committed
    ):
        raise Phase5CommitmentError("script draft chronology or benchmark seal changed")
    activation_delay = (scheduled - committed).total_seconds()
    if activation_delay <= 0:
        raise Phase5CommitmentError(
            "scheduled revision activation must strictly follow commitment"
        )

    expected = {item.episode_id: item for item in protocol.scripted_episodes}
    drafts = {item.episode_id: item for item in draft.drafts}
    if set(drafts) != set(expected):
        raise Phase5CommitmentError("script draft IDs differ from the frozen protocol")

    entries: list[Phase5ScriptCommitmentEntry] = []
    records: list[tuple[RevisionInstruction, ScriptedRevisionFreeze]] = []
    for selection in protocol.scripted_episodes:
        item = drafts[selection.episode_id]
        instruction = item.instruction
        if (
            instruction.before_context.context_id != selection.context_id
            or instruction.revision.action is not selection.action
            or instruction.revision.sequence != 1
            or instruction.revision.created_at != scheduled
        ):
            raise Phase5CommitmentError("script instruction differs from its registered slot")
        scan_model_payload(instruction.model_visible().model_dump(mode="json"))

        stage_relative = Path(
            "data/synthetic/model_visible/query_stages"
        ) / f"reveal_{selection.context_id.removeprefix('ctx_')}"
        stage_path = repository / stage_relative
        artifact, reveal = load_model_eligible_query(stage_path, repository / "data/synthetic")
        context = _context_from_public_reveal(selection.context_id, reveal)
        if instruction.before_context != context:
            raise Phase5CommitmentError(
                "script instruction before-context differs from its public query stage"
            )
        _assert_anchors_resolve(instruction, artifact.evidence)

        manifest_path = stage_path / "manifest.json"
        manifest_bytes = manifest_path.read_bytes()
        stage_manifest = RuntimeStagingManifest.model_validate_json(manifest_bytes)
        if (
            stage_manifest.stage_kind.value != "query_revealed"
            or reveal.content_hash not in stage_manifest.artifact_hashes
            or artifact.content_hash not in stage_manifest.artifact_hashes
        ):
            raise Phase5CommitmentError("script public query-stage manifest changed")
        freeze = ScriptedRevisionFreeze(
            episode_id=selection.episode_id,
            context_id=selection.context_id,
            action=selection.action,
            instruction_hash=instruction.content_hash,
            anchor_hashes=tuple(
                anchor.content_hash for anchor in instruction.revision.anchors
            ),
            frozen_at=committed,
        )
        entries.append(
            Phase5ScriptCommitmentEntry(
                episode_id=selection.episode_id,
                context_id=selection.context_id,
                action=selection.action,
                query_stage_relative_path=stage_relative.as_posix(),
                query_stage_manifest_hash=stage_manifest.content_hash,
                query_stage_manifest_file_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                query_artifact_hash=reveal.content_hash,
                model_visible_evidence_hash=artifact.content_hash,
                instruction=_cas_reference(instruction, "revision_instruction"),
                freeze=_cas_reference(freeze, "scripted_revision_freeze"),
                scheduled_activation_at=scheduled,
            )
        )
        records.append((instruction, freeze))

    readiness = _parent_readiness_floor(
        entries=tuple(entries),
        reviewed_plan=reviewed_plan,
        configuration=held_out_configuration,
    )
    if activation_delay <= readiness.registered_watchdog_floor_seconds:
        raise Phase5CommitmentError(
            "scheduled revision activation does not clear the held-out watchdog floor"
        )
    manifest = Phase5ScriptCommitmentManifest(
        commitment_id=f"phase5-script-commitment-{draft.content_hash[:20]}",
        protocol_hash=protocol.content_hash,
        benchmark_draft_seal_hash=protocol.benchmark_draft_seal_hash,
        review_gate=review_gate,
        parent_readiness_floor=readiness,
        draft_manifest_hash=draft.content_hash,
        entries=tuple(entries),
        committed_at=committed,
        scheduled_activation_at=scheduled,
        activation_delay_seconds=activation_delay,
    )
    return PreparedPhase5ScriptCommitment(
        manifest=manifest,
        records=tuple(records),
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _append_bytes_atomically(path: Path, payload: bytes) -> bool:
    """Publish immutable bytes atomically; replay accepts only byte identity."""

    _assert_no_symlink_chain(path)
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise Phase5CommitmentError("append-only script commitment changed")
        return False
    parent_existed = path.parent.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    _assert_no_symlink_chain(path.parent)
    if not parent_existed:
        _fsync_directory(path.parent.parent)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
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
            if not path.is_file() or path.read_bytes() != payload:
                raise Phase5CommitmentError("concurrent script commitment changed") from None
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    _fsync_directory(path.parent)
    return published


def materialize_phase5_script_commitment(
    *,
    prepared: PreparedPhase5ScriptCommitment,
    artifacts: ArtifactStore,
    output_root: Path,
    restricted_root: Path,
) -> tuple[Phase5ScriptCommitmentManifest, bool]:
    """Persist restricted CAS records and one immutable hash-only manifest."""

    root = _restricted_descendant(
        restricted_root.resolve(strict=True),
        output_root,
        label="script commitment output root",
    )
    if root.exists():
        if not root.is_dir() or root.is_symlink():
            raise Phase5CommitmentError("script commitment output root is unsafe")
        unexpected = {
            item.name
            for item in root.iterdir()
            if item.name != "commitment_manifest.json"
        }
        if unexpected:
            raise Phase5CommitmentError("script commitment output root is not exact")

    for entry, (instruction, freeze) in zip(
        prepared.manifest.entries,
        prepared.records,
        strict=True,
    ):
        instruction_reference = persist_phase5_record(
            artifacts,
            instruction,
            object_kind="revision_instruction",
            created_at=prepared.manifest.committed_at,
        )
        freeze_reference = persist_phase5_record(
            artifacts,
            freeze,
            object_kind="scripted_revision_freeze",
            created_at=prepared.manifest.committed_at,
        )
        if (
            instruction_reference != entry.instruction
            or freeze_reference != entry.freeze
        ):
            raise Phase5CommitmentError("persisted script CAS reference changed")

    payload = (prepared.manifest.to_canonical_json() + "\n").encode("utf-8")
    created = _append_bytes_atomically(root / "commitment_manifest.json", payload)
    return prepared.manifest, created


__all__ = [
    "DEFAULT_COMMITMENT_OUTPUT_ROOT",
    "Phase5CommitmentError",
    "Phase5ParentReadinessFloor",
    "Phase5ReviewGateBinding",
    "Phase5ScriptCommitmentEntry",
    "Phase5ScriptCommitmentManifest",
    "PreparedPhase5ScriptCommitment",
    "ScriptedRevisionDraft",
    "ScriptedRevisionDraftManifest",
    "load_phase5_script_commitment",
    "load_scripted_revision_draft",
    "materialize_phase5_script_commitment",
    "open_phase5_hash_bound_reviewed_plan",
    "prepare_phase5_script_commitment",
    "validate_phase5_commitment_destinations",
]
