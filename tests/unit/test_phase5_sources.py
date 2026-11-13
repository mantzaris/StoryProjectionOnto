from __future__ import annotations

import hashlib
import os
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

import story_projection_onto.phase5_sources as production
from story_projection_onto.conditions.c0 import ClassicalRuleConfig
from story_projection_onto.contracts import (
    ConditionName,
    ConstructionOperator,
    EvidencePacket,
    EvidenceSnapshot,
    FeedbackAction,
    OntologyDecision,
    OntologyProjection,
    ReleaseClass,
)
from story_projection_onto.development_runtime import RunOutcome
from story_projection_onto.feedback_runtime import load_feedback_protocol
from story_projection_onto.held_out_controller import PreconstructedProjectionReceipt
from story_projection_onto.phase5_production import Phase5CASReference
from story_projection_onto.phase5_sources import (
    Phase5KnownAnswerEntry,
    Phase5KnownAnswerSourceManifest,
    Phase5SelectedParentUnavailable,
    Phase5SourceProductionError,
)
from tests.unit.test_phase5_execution import (
    _inputs,
    _rebind_prequery_validation,
    _without_hashes,
)
from tests.unit.test_ui import NOW, packet


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _known_answers(authored_at: datetime) -> Phase5KnownAnswerSourceManifest:
    protocol = load_feedback_protocol()

    def reference(label: str, object_kind: str, logical_hash: str) -> Phase5CASReference:
        return Phase5CASReference(
            artifact_hash=digest(f"artifact-{label}"),
            logical_content_hash=logical_hash,
            object_kind=object_kind,
            media_type=f"application/vnd.story-projection.{object_kind}+json",
        )

    return Phase5KnownAnswerSourceManifest(
        source_id="TEST-ONLY-phase5-known-answers",
        protocol_hash=protocol.content_hash,
        review_completion_manifest_hash=digest("review-completion"),
        final_reviewed_seal_hash=digest("final-reviewed-seal"),
        answers=tuple(
            Phase5KnownAnswerEntry(
                episode_id=item.episode_id,
                context_id=item.context_id,
                scorer_binding_key=item.scorer_binding_key,
                before_gold_projection_hash=digest(f"before-{item.episode_id}"),
                after_gold_projection_hash=digest(f"after-{item.episode_id}"),
                required_target_change_hash=digest(f"target-{item.episode_id}"),
                before_gold_projection_source=reference(
                    f"before-{item.episode_id}",
                    "feedback_gold_projection_source",
                    digest(f"before-wrapper-{item.episode_id}"),
                ),
                after_gold_projection_source=reference(
                    f"after-{item.episode_id}",
                    "feedback_gold_projection_source",
                    digest(f"after-wrapper-{item.episode_id}"),
                ),
                required_target_change_source=reference(
                    f"target-{item.episode_id}",
                    "feedback_required_target_change",
                    digest(f"target-{item.episode_id}"),
                ),
            )
            for item in protocol.scripted_episodes
        ),
        authored_at=authored_at,
    )


def test_known_answer_source_expands_to_exact_hash_only_condition_bindings() -> None:
    protocol = load_feedback_protocol()
    committed_at = NOW
    source = _known_answers(committed_at - timedelta(seconds=1))
    closed = SimpleNamespace(
        execution=SimpleNamespace(
            review_completion_manifest_hash=digest("review-completion"),
            final_reviewed_seal_hash=digest("final-reviewed-seal"),
        )
    )
    bindings = production._validate_known_answer_source(
        source=source,
        protocol=protocol,
        commitment=SimpleNamespace(committed_at=committed_at),
        closed=closed,
    )

    assert len(bindings) == 18
    assert len({(item.episode_id, item.condition) for item in bindings}) == 18
    assert {item.condition for item in bindings} == {
        ConditionName.C0_CLASSICAL_PRE,
        ConditionName.C1_LLM_PRE,
        ConditionName.C2_LLM_QUERY,
    }
    assert all(item.scorer_namespace == "scorer_only" for item in bindings)


def test_known_answer_source_cannot_be_authored_after_output_blind_commitment() -> None:
    protocol = load_feedback_protocol()
    committed_at = NOW
    with pytest.raises(Phase5SourceProductionError, match="output-blind"):
        production._validate_known_answer_source(
            source=_known_answers(committed_at + timedelta(microseconds=1)),
            protocol=protocol,
            commitment=SimpleNamespace(committed_at=committed_at),
            closed=SimpleNamespace(
                execution=SimpleNamespace(
                    review_completion_manifest_hash=digest("review-completion"),
                    final_reviewed_seal_hash=digest("final-reviewed-seal"),
                )
            ),
        )


def _cpu_source(action: FeedbackAction):
    _protocol, _gate, inputs = _inputs()
    script = next(
        item for item in inputs.scripted_inputs if item.instruction.revision.action is action
    )
    before = next(
        item.before_projection
        for item in script.cpu_inputs
        if item.condition is ConditionName.C0_CLASSICAL_PRE
    )
    preontology = SimpleNamespace(
        construction_seal=before.construction_seal,
        upper_ontology=before.upper_ontology,
    )
    preparation = SimpleNamespace(
        condition=ConditionName.C0_CLASSICAL_PRE,
        content_hash=digest(f"preparation-{action.value}"),
        sealed_preontology=preontology,
    )
    boundary = SimpleNamespace(
        episode_id=script.episode_id,
        stage=SimpleNamespace(
            stage_id="TEST-ONLY-phase5-source-stage",
            staging_manifest_hash=digest("query-stage"),
        ),
        opening=SimpleNamespace(query_context=script.instruction.before_context),
        packet=script.packet,
        snapshot=SimpleNamespace(
            content_hash=before.snapshot_hash,
            horizon=script.instruction.before_context.spoiler_horizon,
        ),
    )
    return script, before, preparation, boundary


def test_merge_split_is_explicitly_capability_limited_without_projection() -> None:
    script, before, preparation, boundary = _cpu_source(
        FeedbackAction.REQUEST_MERGE_SPLIT
    )
    draft = production._build_cpu_draft(
        instruction=script.instruction,
        boundary=boundary,
        condition=ConditionName.C0_CLASSICAL_PRE,
        before_projection=before,
        preparation=preparation,
        prequery_barrier=SimpleNamespace(
            execution_id="TEST-ONLY-held-out",
            content_hash=digest("barrier"),
        ),
        query_accessed_at=script.instruction.revision.created_at + timedelta(seconds=1),
        processing_started_at=script.instruction.revision.created_at
        + timedelta(seconds=2),
        rule_config=ClassicalRuleConfig(),
    )

    assert draft.after_projection is None
    assert draft.instruction_hash == script.instruction.content_hash
    assert draft.source_preparation_hash == preparation.content_hash


def test_refine_context_rejects_constructive_cpu_result(monkeypatch) -> None:
    script, before, preparation, boundary = _cpu_source(FeedbackAction.REFINE_CONTEXT)

    def constructive(_preontology, inputs, *, rule_config):
        payload = _without_hashes(before.model_dump(mode="python"))
        payload.update(
            projection_id="TEST-ONLY-invalid-constructive-cpu-projection",
            context_hash=script.instruction.after_context.content_hash,
            query_access_event_hash=inputs.query_access.content_hash,
            budgets=script.instruction.after_context.budgets,
            decisions=(
                OntologyDecision(
                    decision_id="TEST-ONLY-invalid-cpu-merge",
                    operator=ConstructionOperator.MERGE,
                    evidence_ids=(script.packet.ordered_evidence_ids[0],),
                    rationale="Test-only forbidden post-query construction.",
                    decided_at=inputs.query_processing_started_at,
                    input_object_ids=("m-a", "m-b"),
                    created_object_ids=("entity-ab",),
                    removed_object_ids=("entity-a", "entity-b"),
                ),
            ),
            run_id="TEST-ONLY-invalid-constructive-cpu-run",
        )
        _rebind_prequery_validation(
            payload,
            validated_at=inputs.query_processing_started_at + timedelta(seconds=1),
        )
        return OntologyProjection.model_validate(payload)

    monkeypatch.setattr(production, "project_sealed_c0", constructive)
    with pytest.raises(Phase5SourceProductionError, match="attempted construction"):
        production._build_cpu_draft(
            instruction=script.instruction,
            boundary=boundary,
            condition=ConditionName.C0_CLASSICAL_PRE,
            before_projection=before,
            preparation=preparation,
            prequery_barrier=SimpleNamespace(
                execution_id="TEST-ONLY-held-out",
                content_hash=digest("barrier"),
            ),
            query_accessed_at=script.instruction.revision.created_at
            + timedelta(seconds=1),
            processing_started_at=script.instruction.revision.created_at
            + timedelta(seconds=2),
            rule_config=ClassicalRuleConfig(),
        )


def _packet_and_snapshot() -> tuple[EvidencePacket, EvidenceSnapshot]:
    base = packet()
    snapshot = EvidenceSnapshot(
        snapshot_id="TEST-ONLY-phase5-source-snapshot",
        corpus_id="TEST-ONLY-phase5-source-corpus",
        world_or_window_id="TEST-ONLY-phase5-source-unit",
        horizon=_inputs()[2].scripted_inputs[0].instruction.before_context.spoiler_horizon,
        eligible_evidence_ids=base.ordered_evidence_ids,
        index_config_hash=digest("index"),
        created_at=NOW - timedelta(minutes=10),
        sealed_at=NOW - timedelta(minutes=9),
        release_class=ReleaseClass.PUBLIC,
    )
    values = base.model_dump(mode="python", exclude={"content_hash"})
    values["snapshot_hash"] = snapshot.content_hash
    return EvidencePacket.model_validate(values), snapshot


def test_selected_invalid_parent_fails_closed_with_exact_itt_hash(tmp_path: Path) -> None:
    evidence_packet, snapshot = _packet_and_snapshot()
    context = _inputs()[2].scripted_inputs[0].instruction.before_context
    receipt = PreconstructedProjectionReceipt(
        unit_id=snapshot.world_or_window_id,
        query_stage_hash=digest("query-stage"),
        condition=ConditionName.C0_CLASSICAL_PRE,
        seed_block=None,
        source_construction_seal_hash=None,
        source_complete_graph_hash=None,
        outcome=RunOutcome.INVALID,
        evidence_packet_hash=evidence_packet.content_hash,
        horizon_hash=snapshot.horizon.content_hash,
        budget_hash=context.budgets.content_hash,
        failure_artifact_hash=digest("retained-failure"),
        completed_at=NOW,
    )
    boundary = SimpleNamespace(
        episode_id="TEST-ONLY-failed-parent",
        unit=SimpleNamespace(unit_id=snapshot.world_or_window_id),
        stage=SimpleNamespace(
            stage_id="TEST-ONLY-phase5-source-stage",
            staging_manifest_hash=digest("query-stage"),
        ),
        opening=SimpleNamespace(query_context=context),
        packet=evidence_packet,
        snapshot=snapshot,
    )
    journal = tmp_path / "journal"
    path = journal / production._projection_receipt_path(
        boundary, ConditionName.C0_CLASSICAL_PRE
    )
    path.parent.mkdir(parents=True)
    path.write_text(receipt.to_canonical_json() + "\n", encoding="utf-8")
    os.chmod(path, 0o600)
    with pytest.raises(Phase5SelectedParentUnavailable) as raised:
        production._build_export(
            boundary=boundary,
            condition=ConditionName.C0_CLASSICAL_PRE,
            closed=SimpleNamespace(
                execution=SimpleNamespace(preconstructed_projections=(receipt,))
            ),
            journal_root=journal,
            artifacts=None,
            exported_at=NOW + timedelta(seconds=1),
            expected_frozen_seed=1,
        )

    assert raised.value.outcome is RunOutcome.INVALID
    assert raised.value.source_record_hash == receipt.content_hash


def test_append_exact_is_idempotent_restricted_and_detects_drift(tmp_path: Path) -> None:
    source = _known_answers(NOW)
    path = tmp_path / "restricted" / "known.json"
    assert production._append_exact(path, source) is True
    assert production._append_exact(path, source) is False
    assert path.stat().st_mode & 0o777 == 0o600
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(Phase5SourceProductionError, match="append-only"):
        production._append_exact(path, source)


def test_producer_rejects_noncanonical_restricted_root(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    (repository / "artifacts/restricted").mkdir(parents=True)
    arbitrary = repository / "arbitrary-restricted"
    arbitrary.mkdir()
    with pytest.raises(Phase5SourceProductionError, match="canonical repository"):
        production.prepare_phase5_source_production(
            repository=repository,
            restricted_root=arbitrary,
            output_root=arbitrary / "output",
            protocol=load_feedback_protocol(),
            script_commitment_path=arbitrary / "commitment.json",
            primary_results_gate_path=arbitrary / "gate.json",
            held_out_journal_root=arbitrary / "journal",
            known_answer_source_path=arbitrary / "answers.json",
            trace_instruction_paths={},
            trace_submission_receipt_paths={},
            c0_state_root=arbitrary / "c0",
            c0_rule_config_path=Path("configs/study/c0_rules.json"),
            artifacts=None,
        )
