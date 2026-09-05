"""Frozen Phase 5 feedback inventory and auditable execution records.

This module does not generate ontology semantics.  A condition runner supplies an
``OntologyProjection`` (or an explicit failed attempt); the helpers below validate and
package the per-condition resolution, visualization diff, metered C2 receipt, and replay
lineage.  Scorer-only known answers remain outside these public execution records.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal, Self

from pydantic import AwareDatetime, Field, model_validator

from story_projection_onto.contracts import (
    ConditionName,
    EvidencePacket,
    FeedbackAction,
    FeedbackResolution,
    FeedbackResolutionStatus,
    Identifier,
    ImmutableRecord,
    OntologyProjection,
    Sha256Digest,
)
from story_projection_onto.ui import (
    FeedbackEpisodeKind,
    FeedbackReplayExpectation,
    FeedbackReplayResult,
    RevisionInstruction,
    VisualizationBundle,
    VisualizationConfiguration,
    VisualizationContentScope,
    VisualizationDiff,
    build_visualization_bundle,
    compare_visualizations,
    resolve_feedback_for_condition,
    verify_feedback_replay,
)

DEFAULT_FEEDBACK_PROTOCOL_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "study" / "feedback.json"
)
SCRIPTED_CONDITIONS = (
    ConditionName.C0_CLASSICAL_PRE,
    ConditionName.C1_LLM_PRE,
    ConditionName.C2_LLM_QUERY,
)


class FeedbackDifficulty(StrEnum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class FeedbackScriptSelection(ImmutableRecord):
    """Runner-side selection fixed without inspecting any condition output."""

    episode_id: Identifier
    context_id: Identifier
    difficulty: FeedbackDifficulty
    action: FeedbackAction
    revision_template: Literal["lens_or_story_time", "identity_merge_or_split"]
    scorer_binding_key: Identifier
    condition_output_inspected: Literal[False] = False

    @model_validator(mode="after")
    def template_matches_action(self) -> Self:
        expected = (
            "lens_or_story_time"
            if self.action is FeedbackAction.REFINE_CONTEXT
            else "identity_merge_or_split"
        )
        if self.revision_template != expected:
            raise ValueError("feedback script template and action differ")
        return self


class ResearcherTraceSlot(ImmutableRecord):
    """A context slot whose single action is chosen through the local interface."""

    episode_id: Identifier
    context_id: Identifier
    difficulty: FeedbackDifficulty
    allowed_actions: tuple[FeedbackAction, FeedbackAction]
    gold_dependent_fields: Literal["NA"] = "NA"
    condition: Literal[ConditionName.C2_LLM_QUERY] = ConditionName.C2_LLM_QUERY

    @model_validator(mode="after")
    def exactly_the_two_registered_actions(self) -> Self:
        if self.allowed_actions != (
            FeedbackAction.REFINE_CONTEXT,
            FeedbackAction.REQUEST_MERGE_SPLIT,
        ):
            raise ValueError("researcher traces permit exactly the two registered actions")
        return self


class FeedbackProtocolConfiguration(ImmutableRecord):
    protocol_id: Identifier
    benchmark_draft_seal_hash: Sha256Digest
    selection_rule: str = Field(min_length=1)
    seed_manifest_hash: Sha256Digest
    llm_seed_entry_hash: Sha256Digest
    llm_seed: int = Field(ge=0)
    selected_before_condition_outputs: Literal[True] = True
    allowed_actions: tuple[FeedbackAction, FeedbackAction]
    scripted_episodes: tuple[
        FeedbackScriptSelection,
        FeedbackScriptSelection,
        FeedbackScriptSelection,
        FeedbackScriptSelection,
        FeedbackScriptSelection,
        FeedbackScriptSelection,
    ]
    researcher_trace_slots: tuple[
        ResearcherTraceSlot,
        ResearcherTraceSlot,
        ResearcherTraceSlot,
    ]
    scripted_conditions: tuple[ConditionName, ConditionName, ConditionName]
    llm_seed_block: Literal["llm_block_1"] = "llm_block_1"
    scripted_c2_regeneration_calls: Literal[6] = 6
    researcher_trace_c2_regeneration_calls: Literal[3] = 3
    total_c2_regeneration_calls: Literal[9] = 9
    revisions_per_episode: Literal[1] = 1
    c0_c1_policy: Literal["reproject_same_seal_or_capability_limited"] = (
        "reproject_same_seal_or_capability_limited"
    )
    researcher_trace_gold_policy: Literal["not_applicable"] = "not_applicable"
    scripted_revision_freeze_required: Literal[True] = True
    usability_claims: Literal[False] = False

    @model_validator(mode="after")
    def exact_reduced_inventory(self) -> Self:
        if self.allowed_actions != (
            FeedbackAction.REFINE_CONTEXT,
            FeedbackAction.REQUEST_MERGE_SPLIT,
        ):
            raise ValueError("feedback protocol permits exactly two actions")
        if self.scripted_conditions != SCRIPTED_CONDITIONS:
            raise ValueError("scripted feedback must report C0, C1, and C2")
        episode_ids = [item.episode_id for item in self.scripted_episodes]
        episode_ids.extend(item.episode_id for item in self.researcher_trace_slots)
        if len(episode_ids) != len(set(episode_ids)):
            raise ValueError("feedback episode IDs must be unique")
        context_ids = [item.context_id for item in self.scripted_episodes]
        context_ids.extend(item.context_id for item in self.researcher_trace_slots)
        if len(context_ids) != len(set(context_ids)):
            raise ValueError("feedback contexts must be distinct across the nine episodes")
        actions = Counter(item.action for item in self.scripted_episodes)
        if actions != Counter(
            {
                FeedbackAction.REFINE_CONTEXT: 3,
                FeedbackAction.REQUEST_MERGE_SPLIT: 3,
            }
        ):
            raise ValueError("known-answer scripts require three episodes per action")
        scripted_difficulty = Counter(item.difficulty for item in self.scripted_episodes)
        if scripted_difficulty != Counter({item: 2 for item in FeedbackDifficulty}):
            raise ValueError("known-answer scripts require two cases per difficulty")
        action_difficulty_pairs = Counter(
            (item.action, item.difficulty) for item in self.scripted_episodes
        )
        expected_pairs = Counter(
            (action, difficulty) for action in FeedbackAction for difficulty in FeedbackDifficulty
        )
        if action_difficulty_pairs != expected_pairs:
            raise ValueError("each scripted action requires one case per difficulty")
        trace_difficulty = Counter(item.difficulty for item in self.researcher_trace_slots)
        if trace_difficulty != Counter({item: 1 for item in FeedbackDifficulty}):
            raise ValueError("researcher traces require one context per difficulty")
        return self


def load_feedback_protocol(
    path: Path = DEFAULT_FEEDBACK_PROTOCOL_PATH,
) -> FeedbackProtocolConfiguration:
    return FeedbackProtocolConfiguration.model_validate_json(path.read_text(encoding="utf-8"))


class FeedbackAttemptStatus(StrEnum):
    SUCCEEDED = "succeeded"
    INVALID = "invalid"
    FAILED = "failed"
    TIMED_OUT = "timed_out"


class FeedbackLedgerCallReference(ImmutableRecord):
    """Immutable pointer to one real model-call row and its GPU event.

    A feedback regeneration has one base request and may consume the one bounded
    repair allowed by the shared validation contract.  Keeping the two rows
    explicit avoids reporting nine scientific regeneration *slots* as though
    they necessarily implied exactly nine physical GPU requests.
    """

    model_call_id: Identifier
    model_call_record_hash: Sha256Digest
    gpu_event_id: Identifier
    gpu_event_record_hash: Sha256Digest
    attempt_id: Identifier
    attempt_kind: Literal["base", "repair"]
    call_role: Literal["query_time", "repair"]
    retry_class: Literal["standard", "short"]
    request_hash: Sha256Digest
    parent_model_call_record_hash: Sha256Digest | None = None
    allocated_gpu_seconds: float = Field(gt=0.0)
    started_at: AwareDatetime
    completed_at: AwareDatetime
    successful: bool

    @model_validator(mode="after")
    def exact_attempt_shape(self) -> Self:
        if self.completed_at < self.started_at:
            raise ValueError("feedback GPU ledger call completion predates its start")
        if self.attempt_kind == "base":
            if self.call_role != "query_time" or self.retry_class != "standard":
                raise ValueError("feedback base calls require query-time/standard accounting")
            if self.parent_model_call_record_hash is not None:
                raise ValueError("feedback base calls cannot name a repair parent")
        elif (
            self.call_role != "repair"
            or self.retry_class != "short"
            or self.parent_model_call_record_hash is None
        ):
            raise ValueError("feedback repair calls require short repair-parent lineage")
        return self


class ScriptedRevisionFreeze(ImmutableRecord):
    """Pre-output commitment to a script's exact shared revision and anchors."""

    episode_id: Identifier
    context_id: Identifier
    action: FeedbackAction
    instruction_hash: Sha256Digest
    anchor_hashes: tuple[Sha256Digest, ...]
    frozen_at: AwareDatetime
    condition_output_inspected: Literal[False] = False

    @model_validator(mode="after")
    def require_frozen_anchor_set(self) -> Self:
        if not self.anchor_hashes or len(self.anchor_hashes) != len(set(self.anchor_hashes)):
            raise ValueError("script freezes require unique condition-independent anchors")
        return self


class FeedbackRegenerationReceipt(ImmutableRecord):
    """References to the metered base call and optional bounded repair call."""

    ledger_calls: tuple[FeedbackLedgerCallReference, ...]
    attempt_status: FeedbackAttemptStatus
    instruction_hash: Sha256Digest
    before_projection_hash: Sha256Digest
    after_projection_hash: Sha256Digest | None = None
    failure_artifact_hash: Sha256Digest | None = None
    seed: int = Field(ge=0)
    allocated_gpu_seconds: float = Field(gt=0.0)
    started_at: AwareDatetime
    completed_at: AwareDatetime
    condition: Literal[ConditionName.C2_LLM_QUERY] = ConditionName.C2_LLM_QUERY

    @model_validator(mode="after")
    def receipt_is_complete(self) -> Self:
        if not 1 <= len(self.ledger_calls) <= 2:
            raise ValueError("feedback regeneration requires one base and at most one repair call")
        if self.ledger_calls[0].attempt_kind != "base":
            raise ValueError("feedback regeneration ledger lineage must begin with its base call")
        if len(self.ledger_calls) == 2:
            repair = self.ledger_calls[1]
            if repair.attempt_kind != "repair":
                raise ValueError("the optional second feedback GPU call must be the repair")
            if repair.parent_model_call_record_hash != self.ledger_calls[0].model_call_record_hash:
                raise ValueError("feedback repair does not bind its base model call")
            if repair.started_at < self.ledger_calls[0].completed_at:
                raise ValueError("feedback repair GPU work overlaps or predates its base call")
        model_call_ids = [item.model_call_id for item in self.ledger_calls]
        event_ids = [item.gpu_event_id for item in self.ledger_calls]
        if len(model_call_ids) != len(set(model_call_ids)) or len(event_ids) != len(set(event_ids)):
            raise ValueError("feedback ledger calls and GPU events must be unique")
        if self.started_at != self.ledger_calls[0].started_at:
            raise ValueError("feedback receipt start differs from its base GPU event")
        if self.completed_at != self.ledger_calls[-1].completed_at:
            raise ValueError("feedback receipt completion differs from its final GPU event")
        accounted = sum(item.allocated_gpu_seconds for item in self.ledger_calls)
        if abs(accounted - self.allocated_gpu_seconds) > 1e-6:
            raise ValueError("feedback receipt GPU seconds do not sum its ledger calls")
        if self.completed_at < self.started_at:
            raise ValueError("feedback regeneration completion predates its start")
        if self.attempt_status is FeedbackAttemptStatus.SUCCEEDED:
            if self.after_projection_hash is None or self.failure_artifact_hash is not None:
                raise ValueError("successful feedback calls require only an after projection")
        elif self.failure_artifact_hash is None:
            raise ValueError("unsuccessful feedback calls require a retained failure artifact")
        if (
            self.attempt_status
            in {
                FeedbackAttemptStatus.FAILED,
                FeedbackAttemptStatus.TIMED_OUT,
            }
            and self.after_projection_hash is not None
        ):
            raise ValueError("failed or timed-out feedback calls cannot claim a projection")
        final_call_succeeded = self.ledger_calls[-1].successful
        if self.attempt_status in {
            FeedbackAttemptStatus.SUCCEEDED,
            FeedbackAttemptStatus.INVALID,
        }:
            if not final_call_succeeded:
                raise ValueError("successful/invalid model output requires a completed GPU call")
        elif final_call_succeeded:
            raise ValueError("failed/timed-out feedback cannot cite a successful final GPU call")
        return self


class FeedbackEpisodeExecution(ImmutableRecord):
    """One condition's complete, public-safe feedback execution lineage."""

    episode_id: Identifier
    kind: FeedbackEpisodeKind
    action: FeedbackAction
    condition: ConditionName
    instruction: RevisionInstruction
    resolution: FeedbackResolution
    before_bundle: VisualizationBundle
    after_bundle: VisualizationBundle | None = None
    diff: VisualizationDiff | None = None
    replay: FeedbackReplayResult | None = None
    regeneration_receipt: FeedbackRegenerationReceipt | None = None
    latency_seconds: float = Field(ge=0.0)
    gold_dependent_fields: Literal["applicable", "NA"]

    @model_validator(mode="after")
    def execution_lineage_is_honest(self) -> Self:
        if self.instruction.revision.action is not self.action:
            raise ValueError("feedback execution action differs from its instruction")
        if self.before_bundle.condition is not self.condition:
            raise ValueError("feedback execution before bundle belongs to another condition")
        if self.resolution.receiving_condition is not self.condition:
            raise ValueError("feedback resolution belongs to another condition")
        if self.resolution.revision_hash != self.instruction.revision.content_hash:
            raise ValueError("feedback resolution does not bind the instruction")
        if self.resolution.before_projection_hash != self.before_bundle.projection_hash:
            raise ValueError("feedback resolution does not bind the before projection")
        if self.resolution.resolved_at < self.instruction.revision.created_at:
            raise ValueError("feedback resolution predates its recorded revision")
        after_fields = (self.after_bundle, self.diff, self.replay)
        if any(item is None for item in after_fields) and any(
            item is not None for item in after_fields
        ):
            raise ValueError("after bundle, diff, and replay must be present or absent together")
        if self.after_bundle is not None:
            assert self.diff is not None
            assert self.replay is not None
            if self.after_bundle.condition is not self.condition:
                raise ValueError("feedback before and after conditions differ")
            if self.diff.before_projection_hash != self.before_bundle.projection_hash:
                raise ValueError("feedback diff does not bind the before projection")
            if self.diff.after_projection_hash != self.after_bundle.projection_hash:
                raise ValueError("feedback diff does not bind the after projection")
            if self.replay.observed_instruction_hash != self.instruction.content_hash:
                raise ValueError("feedback replay does not bind the instruction")
            if self.replay.observed_resolution_hash != self.resolution.content_hash:
                raise ValueError("feedback replay does not bind the resolution")
            if self.replay.observed_before_bundle_hash != self.before_bundle.content_hash:
                raise ValueError("feedback replay does not bind the before bundle")
            if self.replay.observed_after_bundle_hash != self.after_bundle.content_hash:
                raise ValueError("feedback replay does not bind the after bundle")
            if self.replay.observed_diff_hash != self.diff.content_hash:
                raise ValueError("feedback replay does not bind the graph diff")
            if self.replay.checked_at < self.resolution.resolved_at:
                raise ValueError("feedback replay predates condition resolution")
            if not self.diff.fixed_anchor_positions_preserved:
                raise ValueError("feedback diff changed a shared frozen anchor position")
        if self.kind is FeedbackEpisodeKind.RESEARCHER_TRACE:
            if self.condition is not ConditionName.C2_LLM_QUERY:
                raise ValueError("researcher-interface traces are C2-only")
            if self.gold_dependent_fields != "NA":
                raise ValueError("researcher traces keep gold-dependent fields NA")
        elif self.gold_dependent_fields != "applicable":
            raise ValueError("known-answer scripts require separately isolated gold scoring")
        if self.condition is ConditionName.C2_LLM_QUERY:
            if self.regeneration_receipt is None:
                raise ValueError("every C2 feedback episode requires a metered receipt")
            receipt = self.regeneration_receipt
            if receipt.instruction_hash != self.instruction.content_hash:
                raise ValueError("feedback GPU receipt does not bind the instruction")
            if receipt.before_projection_hash != self.before_bundle.projection_hash:
                raise ValueError("feedback GPU receipt does not bind the before projection")
            if receipt.seed != self.resolution.seed:
                raise ValueError("feedback GPU receipt and resolution seeds differ")
            if receipt.started_at <= self.instruction.revision.created_at:
                raise ValueError("feedback GPU work must begin after the revision is recorded")
            if receipt.completed_at > self.resolution.resolved_at:
                raise ValueError("feedback cannot resolve before its GPU attempt completes")
            call_wall_seconds = (receipt.completed_at - receipt.started_at).total_seconds()
            if self.latency_seconds + 1e-9 < call_wall_seconds:
                raise ValueError("feedback latency is shorter than its GPU attempt")
            observed_after = (
                None if self.after_bundle is None else self.after_bundle.projection_hash
            )
            if receipt.after_projection_hash != observed_after:
                raise ValueError("feedback GPU receipt does not bind the retained output")
            if (receipt.attempt_status is FeedbackAttemptStatus.SUCCEEDED) != (
                self.resolution.status is FeedbackResolutionStatus.RESOLVED
            ):
                raise ValueError("feedback call and resolution success statuses differ")
        elif self.regeneration_receipt is not None:
            raise ValueError("C0/C1 feedback cannot claim a C2 GPU regeneration")
        if self.resolution.status is FeedbackResolutionStatus.RESOLVED:
            if self.after_bundle is None:
                raise ValueError("resolved feedback requires an after bundle")
            if self.resolution.after_projection_hash != self.after_bundle.projection_hash:
                raise ValueError("resolved feedback does not bind its after projection")
        elif self.resolution.status is FeedbackResolutionStatus.CAPABILITY_LIMITED:
            if self.after_bundle is not None:
                raise ValueError("capability-limited feedback cannot claim an after projection")
        return self


def compile_feedback_execution(
    *,
    episode_id: str,
    kind: FeedbackEpisodeKind,
    instruction: RevisionInstruction,
    packet: EvidencePacket,
    condition: ConditionName,
    before_projection: OntologyProjection,
    after_projection: OntologyProjection | None,
    after_packet: EvidencePacket | None,
    resolver_hash: str,
    seed: int,
    resolved_at: datetime,
    checked_at: datetime,
    latency_seconds: float,
    regeneration_receipt: FeedbackRegenerationReceipt | None,
    visualization_config: VisualizationConfiguration | None = None,
) -> FeedbackEpisodeExecution:
    """Compile an already-run condition result without supplying missing semantics."""

    before_bundle = build_visualization_bundle(
        before_projection,
        instruction.before_context,
        packet,
        visualization_config=visualization_config,
        content_scope=VisualizationContentScope.REGISTERED_DISPLAY,
    )
    resolution = resolve_feedback_for_condition(
        instruction=instruction,
        packet=packet,
        receiving_condition=condition,
        before_projection=before_projection,
        after_projection=after_projection,
        after_packet=after_packet,
        resolver_hash=resolver_hash,
        seed=seed,
        resolved_at=resolved_at,
    )
    after_bundle = None
    diff = None
    replay = None
    if (
        after_projection is not None
        and resolution.status is not FeedbackResolutionStatus.CAPABILITY_LIMITED
    ):
        effective_after_packet = after_packet or packet
        after_bundle = build_visualization_bundle(
            after_projection,
            instruction.after_context,
            effective_after_packet,
            visualization_config=visualization_config,
            content_scope=VisualizationContentScope.REGISTERED_DISPLAY,
        )
        diff = compare_visualizations(before_bundle, after_bundle)
        expectation = FeedbackReplayExpectation(
            instruction_hash=instruction.content_hash,
            resolution_hash=resolution.content_hash,
            before_bundle_hash=before_bundle.content_hash,
            after_bundle_hash=after_bundle.content_hash,
            diff_hash=diff.content_hash,
        )
        replay = verify_feedback_replay(
            expectation=expectation,
            instruction=instruction,
            resolution=resolution,
            before_bundle=before_bundle,
            after_bundle=after_bundle,
            diff=diff,
            checked_at=checked_at,
        )
    return FeedbackEpisodeExecution(
        episode_id=episode_id,
        kind=kind,
        action=instruction.revision.action,
        condition=condition,
        instruction=instruction,
        resolution=resolution,
        before_bundle=before_bundle,
        after_bundle=after_bundle,
        diff=diff,
        replay=replay,
        regeneration_receipt=regeneration_receipt,
        latency_seconds=latency_seconds,
        gold_dependent_fields=(
            "applicable" if kind is FeedbackEpisodeKind.SCRIPTED_KNOWN_ANSWER else "NA"
        ),
    )


class FeedbackStudyExecutionManifest(ImmutableRecord):
    """Complete nine-episode Phase 5 execution, including failed C2 attempts."""

    protocol_hash: Sha256Digest
    scripted_revision_freezes: tuple[
        ScriptedRevisionFreeze,
        ScriptedRevisionFreeze,
        ScriptedRevisionFreeze,
        ScriptedRevisionFreeze,
        ScriptedRevisionFreeze,
        ScriptedRevisionFreeze,
    ]
    executions: tuple[FeedbackEpisodeExecution, ...]
    generated_at: AwareDatetime
    usability_claims: Literal[False] = False

    @model_validator(mode="after")
    def exact_execution_inventory(self) -> Self:
        scripts = [
            item
            for item in self.executions
            if item.kind is FeedbackEpisodeKind.SCRIPTED_KNOWN_ANSWER
        ]
        traces = [
            item for item in self.executions if item.kind is FeedbackEpisodeKind.RESEARCHER_TRACE
        ]
        script_pairs = Counter((item.episode_id, item.condition) for item in scripts)
        if len(scripts) != 18 or any(value != 1 for value in script_pairs.values()):
            raise ValueError("six scripts require one C0, C1, and C2 execution each")
        script_ids = {item.episode_id for item in scripts}
        if len(script_ids) != 6:
            raise ValueError("feedback execution must contain exactly six scripts")
        freezes = {item.episode_id: item for item in self.scripted_revision_freezes}
        if len(freezes) != 6 or set(freezes) != script_ids:
            raise ValueError("six executed scripts require six distinct pre-output freezes")
        for episode_id in script_ids:
            conditions = {item.condition for item in scripts if item.episode_id == episode_id}
            if conditions != set(SCRIPTED_CONDITIONS):
                raise ValueError("each script must report C0, C1, and C2")
            freeze = freezes[episode_id]
            episode_executions = [item for item in scripts if item.episode_id == episode_id]
            instruction = episode_executions[0].instruction
            if freeze.instruction_hash != instruction.content_hash:
                raise ValueError("script execution differs from its pre-output instruction freeze")
            if freeze.action is not instruction.revision.action:
                raise ValueError("script freeze action differs from its revision")
            if freeze.context_id != instruction.before_context.context_id:
                raise ValueError("script freeze context differs from its revision")
            if freeze.anchor_hashes != tuple(
                item.content_hash for item in instruction.revision.anchors
            ):
                raise ValueError("script anchors differ from their pre-output freeze")
            if freeze.frozen_at > instruction.revision.created_at:
                raise ValueError("script revision predates its registered freeze")
        if len(traces) != 3 or len({item.episode_id for item in traces}) != 3:
            raise ValueError("feedback execution must contain three distinct traces")
        c2 = [item for item in self.executions if item.condition is ConditionName.C2_LLM_QUERY]
        if len(c2) != 9:
            raise ValueError("Phase 5 requires exactly nine C2 regeneration attempts")
        receipt_hashes = [
            call.model_call_record_hash
            for item in c2
            if item.regeneration_receipt is not None
            for call in item.regeneration_receipt.ledger_calls
        ]
        if not 9 <= len(receipt_hashes) <= 18 or len(receipt_hashes) != len(set(receipt_hashes)):
            raise ValueError(
                "the nine C2 episodes require unique base calls and at most one repair each"
            )
        scripted_actions = Counter(
            next(item.action for item in scripts if item.episode_id == episode_id)
            for episode_id in script_ids
        )
        if scripted_actions != Counter(
            {
                FeedbackAction.REFINE_CONTEXT: 3,
                FeedbackAction.REQUEST_MERGE_SPLIT: 3,
            }
        ):
            raise ValueError("executed scripts must retain the frozen 3+3 action balance")
        if any(
            len(
                {item.instruction.content_hash for item in scripts if item.episode_id == episode_id}
            )
            != 1
            for episode_id in script_ids
        ):
            raise ValueError("conditions must receive one identical shared revision per script")
        return self


def assert_feedback_manifest_matches_protocol(
    manifest: FeedbackStudyExecutionManifest,
    protocol: FeedbackProtocolConfiguration,
) -> None:
    """Bind completed records to the output-blind frozen episode selections."""

    if manifest.protocol_hash != protocol.content_hash:
        raise ValueError("feedback execution manifest uses another frozen protocol")
    scripts_by_id = {item.episode_id: item for item in protocol.scripted_episodes}
    traces_by_id = {item.episode_id: item for item in protocol.researcher_trace_slots}
    observed_script_ids = {
        item.episode_id
        for item in manifest.executions
        if item.kind is FeedbackEpisodeKind.SCRIPTED_KNOWN_ANSWER
    }
    observed_trace_ids = {
        item.episode_id
        for item in manifest.executions
        if item.kind is FeedbackEpisodeKind.RESEARCHER_TRACE
    }
    if observed_script_ids != set(scripts_by_id):
        raise ValueError("executed script IDs differ from the frozen protocol")
    if observed_trace_ids != set(traces_by_id):
        raise ValueError("executed trace IDs differ from the frozen protocol")
    freezes_by_id = {item.episode_id: item for item in manifest.scripted_revision_freezes}
    if set(freezes_by_id) != set(scripts_by_id):
        raise ValueError("script revision freezes differ from the frozen selections")
    for episode_id, freeze in freezes_by_id.items():
        selection = scripts_by_id[episode_id]
        if freeze.context_id != selection.context_id or freeze.action is not selection.action:
            raise ValueError("script revision freeze differs from the protocol selection")
    for execution in manifest.executions:
        if execution.resolution.seed != protocol.llm_seed:
            raise ValueError("feedback execution seed differs from the frozen protocol")
        if execution.kind is FeedbackEpisodeKind.SCRIPTED_KNOWN_ANSWER:
            selection = scripts_by_id[execution.episode_id]
            if execution.action is not selection.action:
                raise ValueError("executed script action differs from the frozen protocol")
            if execution.instruction.before_context.context_id != selection.context_id:
                raise ValueError("executed script context differs from the frozen protocol")
        else:
            slot = traces_by_id[execution.episode_id]
            if execution.action not in slot.allowed_actions:
                raise ValueError("researcher trace used an unregistered action")
            if execution.instruction.before_context.context_id != slot.context_id:
                raise ValueError("researcher trace context differs from the frozen protocol")


__all__ = [
    "DEFAULT_FEEDBACK_PROTOCOL_PATH",
    "SCRIPTED_CONDITIONS",
    "FeedbackAttemptStatus",
    "FeedbackDifficulty",
    "FeedbackEpisodeExecution",
    "FeedbackLedgerCallReference",
    "FeedbackProtocolConfiguration",
    "FeedbackRegenerationReceipt",
    "FeedbackScriptSelection",
    "FeedbackStudyExecutionManifest",
    "ResearcherTraceSlot",
    "ScriptedRevisionFreeze",
    "assert_feedback_manifest_matches_protocol",
    "compile_feedback_execution",
    "load_feedback_protocol",
]
