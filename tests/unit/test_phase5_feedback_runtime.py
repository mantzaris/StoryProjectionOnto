from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from story_projection_onto.contracts import ConditionName, FeedbackAction
from story_projection_onto.feedback_runtime import (
    FeedbackAttemptStatus,
    FeedbackEpisodeExecution,
    FeedbackLedgerCallReference,
    FeedbackRegenerationReceipt,
    ScriptedRevisionFreeze,
    load_feedback_protocol,
)
from story_projection_onto.ui import (
    FeedbackEpisodeKind,
    MergeSplitIntent,
    MergeSplitOperation,
    VisualizationBundle,
    VisualizationNodeDetail,
    VisualizationObjectKind,
    build_merge_split_instruction,
    build_visualization_bundle,
    compare_visualizations,
    load_visualization_configuration,
)
from tests.unit.test_ui import NOW, context, digest, feedback_anchor, packet, projection

ROOT = Path(__file__).resolve().parents[2]


def test_frozen_visualization_configuration_uses_registered_layout_seed() -> None:
    config = load_visualization_configuration()
    seed_manifest = json.loads(
        (ROOT / "data/synthetic/manifests/seed_manifest.json").read_text(encoding="utf-8")
    )
    entry = next(item for item in seed_manifest["entries"] if item["purpose"] == "layout")
    assert config.layout_seed == entry["seed"]
    assert config.layout_seed_entry_hash == entry["content_hash"]
    assert config.seed_manifest_hash == seed_manifest["content_hash"]
    assert config.max_retained_screenshots == 12


def test_feedback_protocol_freezes_exact_balanced_nine_episode_inventory() -> None:
    protocol = load_feedback_protocol()
    seed_manifest = json.loads(
        (ROOT / "data/synthetic/manifests/seed_manifest.json").read_text(encoding="utf-8")
    )
    seed_entry = next(item for item in seed_manifest["entries"] if item["purpose"] == "llm_block_1")
    assert protocol.seed_manifest_hash == seed_manifest["content_hash"]
    assert protocol.llm_seed_entry_hash == seed_entry["content_hash"]
    assert protocol.llm_seed == seed_entry["seed"]
    assert len(protocol.scripted_episodes) == 6
    assert len(protocol.researcher_trace_slots) == 3
    assert (
        sum(item.action is FeedbackAction.REFINE_CONTEXT for item in protocol.scripted_episodes)
        == 3
    )
    assert (
        sum(
            item.action is FeedbackAction.REQUEST_MERGE_SPLIT for item in protocol.scripted_episodes
        )
        == 3
    )
    assert protocol.total_c2_regeneration_calls == 9
    assert protocol.scripted_revision_freeze_required is True
    assert protocol.usability_claims is False

    registered_contexts: dict[str, tuple[str, str]] = {}
    for path in sorted((ROOT / "data/synthetic/scorer_only/held_out").glob("syn-test-*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        metadata = (payload["world_spec"]["world_id"], payload["world_spec"]["difficulty"])
        registered_contexts.update(
            {item["query_id"]: metadata for item in payload["gold_projections"]}
        )
    selections = (*protocol.scripted_episodes, *protocol.researcher_trace_slots)
    assert {item.context_id for item in selections} <= set(registered_contexts)
    assert len({registered_contexts[item.context_id][0] for item in selections}) == 9
    assert all(
        registered_contexts[item.context_id][1] == item.difficulty.value for item in selections
    )


def test_feedback_protocol_binds_current_benchmark_draft_seal() -> None:
    protocol = load_feedback_protocol()
    draft_seal = json.loads(
        (ROOT / "data/synthetic/scorer_only/held_out/draft_seal.json").read_text(encoding="utf-8")
    )
    assert protocol.benchmark_draft_seal_hash == draft_seal["content_hash"]


def test_event_evidence_overlap_is_not_misreported_as_identity_merge() -> None:
    query_context = context()
    evidence_packet = packet()
    source = build_visualization_bundle(
        projection(ConditionName.C0_CLASSICAL_PRE, query_context, evidence_packet),
        query_context,
        evidence_packet,
    )
    details = []
    rewritten_ids = set()
    for index, item in enumerate(source.node_details):
        payload = item.model_dump(mode="python", exclude={"content_hash"})
        if index < 2:
            payload.update(
                {
                    "object_kind": VisualizationObjectKind.EVENT,
                    "mention_candidate_ids": (),
                    "evidence_ids": ("ev-a",),
                }
            )
            rewritten_ids.add(item.visualization_node_id)
        details.append(VisualizationNodeDetail(**payload))
    payload = source.model_dump(mode="python", exclude={"content_hash", "node_details"})
    state = source.state.model_dump(mode="python", exclude={"content_hash", "nodes"})
    nodes = []
    for item in source.state.nodes:
        node = item.model_dump(mode="python", exclude={"content_hash"})
        if item.visualization_node_id in rewritten_ids:
            node["evidence_badge"] = {
                "available": True,
                "evidence_ids": ("ev-a",),
                "count": 1,
            }
        nodes.append(node)
    state["nodes"] = tuple(nodes)
    payload["state"] = state
    duplicated_event_evidence = VisualizationBundle(**payload, node_details=tuple(details))
    result = compare_visualizations(duplicated_event_evidence, duplicated_event_evidence)
    assert result.changes == ()


def _c2_execution() -> FeedbackEpisodeExecution:
    query_context = context()
    evidence_packet = packet()
    before = projection(ConditionName.C2_LLM_QUERY, query_context, evidence_packet)
    after = projection(
        ConditionName.C2_LLM_QUERY,
        query_context,
        evidence_packet,
        merged=True,
        decision_offset_seconds=65,
    )
    instruction = build_merge_split_instruction(
        revision_id="phase5-runtime-merge",
        context=query_context,
        intent=MergeSplitIntent(
            operation=MergeSplitOperation.MERGE,
            grouped_mention_candidate_ids=(("m-a",), ("m-b",)),
        ),
        anchors=(feedback_anchor("ev-a", "m-a"), feedback_anchor("ev-b", "m-b")),
        rationale="Merge the two evidence-anchored identity groups.",
        sequence=1,
        created_at=NOW + timedelta(minutes=1),
    )
    from story_projection_onto.feedback_runtime import compile_feedback_execution

    receipt = FeedbackRegenerationReceipt(
        ledger_calls=(
            FeedbackLedgerCallReference(
                model_call_id="phase5-model-call",
                model_call_record_hash=digest("phase5-model-call-row"),
                gpu_event_id="phase5-gpu-event",
                gpu_event_record_hash=digest("phase5-gpu-event-row"),
                attempt_id="phase5-attempt-01",
                attempt_kind="base",
                call_role="query_time",
                retry_class="standard",
                request_hash=digest("phase5-request"),
                allocated_gpu_seconds=1.5,
                started_at=NOW + timedelta(minutes=1, seconds=1),
                completed_at=NOW + timedelta(minutes=1, seconds=2),
                successful=True,
            ),
        ),
        attempt_status=FeedbackAttemptStatus.SUCCEEDED,
        instruction_hash=instruction.content_hash,
        before_projection_hash=before.content_hash,
        after_projection_hash=after.content_hash,
        seed=0,
        allocated_gpu_seconds=1.5,
        started_at=NOW + timedelta(minutes=1, seconds=1),
        completed_at=NOW + timedelta(minutes=1, seconds=2),
    )
    return compile_feedback_execution(
        episode_id="feedback-script-identity-easy",
        kind=FeedbackEpisodeKind.SCRIPTED_KNOWN_ANSWER,
        instruction=instruction,
        packet=evidence_packet,
        condition=ConditionName.C2_LLM_QUERY,
        before_projection=before,
        after_projection=after,
        after_packet=None,
        resolver_hash=digest("phase5-resolver"),
        seed=0,
        resolved_at=NOW + timedelta(minutes=1, seconds=7),
        checked_at=NOW + timedelta(minutes=1, seconds=8),
        latency_seconds=1.5,
        regeneration_receipt=receipt,
    )


def test_c2_feedback_execution_binds_metered_call_diff_and_replay() -> None:
    execution = _c2_execution()
    assert execution.resolution.status.value == "resolved"
    assert execution.diff is not None
    assert any(item.kind.value == "merged" for item in execution.diff.changes)
    assert execution.replay is not None
    assert execution.replay.replay_hash_success is True
    assert execution.regeneration_receipt is not None
    assert execution.regeneration_receipt.allocated_gpu_seconds == pytest.approx(1.5)


def test_c2_feedback_execution_cannot_drop_gpu_receipt() -> None:
    execution = _c2_execution()
    payload = execution.model_dump(
        mode="python",
        exclude={"content_hash", "regeneration_receipt"},
    )
    with pytest.raises(ValidationError, match="metered receipt"):
        FeedbackEpisodeExecution(**payload)


def test_script_freeze_binds_exact_shared_instruction_and_anchor_hashes() -> None:
    execution = _c2_execution()
    freeze = ScriptedRevisionFreeze(
        episode_id=execution.episode_id,
        context_id=execution.instruction.before_context.context_id,
        action=execution.action,
        instruction_hash=execution.instruction.content_hash,
        anchor_hashes=tuple(item.content_hash for item in execution.instruction.revision.anchors),
        frozen_at=execution.instruction.revision.created_at - timedelta(seconds=1),
    )
    assert freeze.condition_output_inspected is False
    assert freeze.instruction_hash == execution.instruction.content_hash
