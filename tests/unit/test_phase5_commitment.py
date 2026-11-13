from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import scripts.freeze_phase5_scripted_revisions as freeze_cli
import story_projection_onto.held_out_primary as held_out
from story_projection_onto.contracts import FeedbackAnchor, QueryContext
from story_projection_onto.feedback_runtime import load_feedback_protocol
from story_projection_onto.held_out_primary import (
    HeldOutCallManifest,
    HeldOutControlConfiguration,
    ReviewedHeldOutPlan,
    load_held_out_control_configuration,
)
from story_projection_onto.phase5_commitment import (
    Phase5CommitmentError,
    Phase5ReviewGateBinding,
    ScriptedRevisionDraft,
    ScriptedRevisionDraftManifest,
    load_phase5_script_commitment,
    materialize_phase5_script_commitment,
    prepare_phase5_script_commitment,
)
from story_projection_onto.phase5_production import RestrictedPhase5CAS
from story_projection_onto.store import ArtifactStore, BlobStore, Compression, Ledger
from story_projection_onto.synthetic_benchmark import load_model_eligible_query
from story_projection_onto.ui import (
    MergeSplitIntent,
    MergeSplitOperation,
    RefineContextIntent,
    build_merge_split_instruction,
    build_refine_context_instruction,
)

ROOT = Path(__file__).resolve().parents[2]
COMMITTED_AT = datetime(2026, 9, 4, 12, 0, tzinfo=UTC)
ACTIVATION_AT = COMMITTED_AT + timedelta(hours=8)


def _public_stage(context_id: str, repository: Path = ROOT):
    stage = (
        repository
        / "data/synthetic/model_visible/query_stages"
        / f"reveal_{context_id.removeprefix('ctx_')}"
    )
    artifact, reveal = load_model_eligible_query(stage, repository / "data/synthetic")
    context = QueryContext(
        context_id=context_id,
        revealed_at=reveal.revealed_at,
        **reveal.query.model_dump(
            mode="python",
            exclude={"schema_version", "content_hash"},
        ),
    )
    return artifact, context


def _anchor(record, mention) -> FeedbackAnchor:
    return FeedbackAnchor(
        evidence_ids=(record.evidence_id,),
        mention_candidate_ids=(mention.candidate_id,),
        requested_semantic_signature=(
            f"evidence:{record.evidence_id}|mention:{mention.candidate_id}"
        ),
    )


def _draft(
    *,
    authored_at: datetime = COMMITTED_AT - timedelta(minutes=1),
    activation_at: datetime = ACTIVATION_AT,
):
    protocol = load_feedback_protocol()
    drafts = []
    for selection in protocol.scripted_episodes:
        artifact, context = _public_stage(selection.context_id)
        records = tuple(item for item in artifact.evidence if item.mention_candidates)
        if selection.action.value == "REFINE_CONTEXT":
            record = records[0]
            mention = record.mention_candidates[0]
            instruction = build_refine_context_instruction(
                revision_id=f"{selection.episode_id}-revision",
                before_context=context,
                intent=RefineContextIntent(
                    after_context_id=f"{selection.context_id}-scripted-refinement",
                    after_revealed_at=activation_at,
                    lens=f"{context.lens}; focus on the evidence-anchored transition",
                ),
                anchors=(_anchor(record, mention),),
                rationale="Apply the preregistered evidence-anchored context refinement.",
                sequence=1,
                created_at=activation_at,
            )
        else:
            first, second = records[:2]
            first_mention = first.mention_candidates[0]
            second_mention = second.mention_candidates[0]
            instruction = build_merge_split_instruction(
                revision_id=f"{selection.episode_id}-revision",
                context=context,
                intent=MergeSplitIntent(
                    operation=MergeSplitOperation.SPLIT,
                    grouped_mention_candidate_ids=(
                        (first_mention.candidate_id,),
                        (second_mention.candidate_id,),
                    ),
                ),
                anchors=(
                    _anchor(first, first_mention),
                    _anchor(second, second_mention),
                ),
                rationale="Keep the two preregistered evidence-anchored mention groups distinct.",
                sequence=1,
                created_at=activation_at,
            )
        drafts.append(
            ScriptedRevisionDraft(
                episode_id=selection.episode_id,
                instruction=instruction,
            )
        )
    return protocol, ScriptedRevisionDraftManifest(
        draft_id="TEST-ONLY-phase5-script-draft",
        protocol_hash=protocol.content_hash,
        benchmark_draft_seal_hash=protocol.benchmark_draft_seal_hash,
        drafts=tuple(drafts),
        authored_at=authored_at,
        scheduled_activation_at=activation_at,
    )


def _mini_repository(tmp_path: Path, protocol) -> Path:
    repository = tmp_path / "repository"
    query_root = repository / "data/synthetic/model_visible/query_stages"
    for selection in protocol.scripted_episodes:
        name = f"reveal_{selection.context_id.removeprefix('ctx_')}"
        shutil.copytree(
            ROOT / "data/synthetic/model_visible/query_stages" / name,
            query_root / name,
        )
    (repository / "artifacts/restricted").mkdir(parents=True)
    return repository


def _runtime_bound_review(protocol):
    configuration = load_held_out_control_configuration(ROOT)
    planned = held_out._derive_call_manifest(ROOT, configuration)
    manifest_payload = planned.model_dump(mode="python", exclude={"content_hash"})
    manifest_payload.update(
        development_execution_result_hash=digest("TEST-ONLY-development-result"),
        allocated_gpu_seconds_before_heldout=1200.0,
        post_development_mandatory_forecast_seconds=18000.0,
    )
    manifest = HeldOutCallManifest.model_validate(manifest_payload)
    configuration_payload = configuration.model_dump(
        mode="python", exclude={"content_hash"}
    )
    configuration_payload["expected_plan_hash"] = manifest.content_hash
    bound_configuration = HeldOutControlConfiguration.model_validate(
        configuration_payload
    )
    completion_hash = digest("TEST-ONLY-independent-review-completion")
    final_seal_hash = digest("TEST-ONLY-final-reviewed-seal")
    reviewed_plan = ReviewedHeldOutPlan(
        call_manifest=manifest,
        review_completion_manifest_hash=completion_hash,
        review_draft_seal_hash=protocol.benchmark_draft_seal_hash,
        final_reviewed_seal_hash=final_seal_hash,
        runtime_binding_hash=digest("TEST-ONLY-held-out-runtime-binding"),
    )
    review_gate = Phase5ReviewGateBinding(
        review_completion_manifest_hash=completion_hash,
        review_completion_manifest_file_sha256=digest(
            "TEST-ONLY-independent-review-completion-file"
        ),
        final_reviewed_seal_hash=final_seal_hash,
        final_reviewed_seal_file_sha256=digest(
            "TEST-ONLY-final-reviewed-seal-file"
        ),
        review_draft_seal_hash=protocol.benchmark_draft_seal_hash,
    )
    return reviewed_plan, bound_configuration, review_gate


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def test_prepare_freezes_exact_public_stage_instructions_without_writes(tmp_path: Path) -> None:
    protocol, draft = _draft()
    reviewed_plan, held_out_configuration, review_gate = _runtime_bound_review(protocol)
    repository = _mini_repository(tmp_path, protocol)
    restricted = repository / "artifacts/restricted"
    journal = restricted / "held-out-journal"

    prepared = prepare_phase5_script_commitment(
        draft=draft,
        protocol=protocol,
        reviewed_plan=reviewed_plan,
        held_out_configuration=held_out_configuration,
        review_gate=review_gate,
        repository=repository,
        restricted_root=restricted,
        held_out_journal_root=journal,
        committed_at=COMMITTED_AT,
    )

    assert len(prepared.manifest.entries) == 6
    assert len(prepared.records) == 6
    assert prepared.manifest.condition_output_inspected is False
    assert prepared.manifest.held_out_output_read is False
    assert prepared.manifest.scorer_gold_read is False
    assert prepared.manifest.model_service_called is False
    assert prepared.manifest.activation_delay_seconds == 8 * 60 * 60
    assert (
        prepared.manifest.parent_readiness_floor.registered_watchdog_floor_seconds
        == 26460
    )
    assert prepared.manifest.parent_readiness_floor.base_call_watchdog_seconds == 23040
    assert prepared.manifest.parent_readiness_floor.repair_reserve_watchdog_seconds == 2520
    assert prepared.manifest.parent_readiness_floor.service_start_watchdog_seconds == 900
    assert prepared.manifest.review_gate == review_gate
    assert not journal.exists()
    assert not (restricted / "phase5-script-commitment").exists()
    for entry, (instruction, freeze) in zip(
        prepared.manifest.entries,
        prepared.records,
        strict=True,
    ):
        assert entry.instruction.logical_content_hash == instruction.content_hash
        assert entry.freeze.logical_content_hash == freeze.content_hash
        assert freeze.frozen_at == COMMITTED_AT
        assert freeze.frozen_at < instruction.revision.created_at


def test_materialize_is_atomic_replayable_and_restricted(tmp_path: Path) -> None:
    protocol, draft = _draft()
    reviewed_plan, held_out_configuration, review_gate = _runtime_bound_review(protocol)
    repository = _mini_repository(tmp_path, protocol)
    restricted = repository / "artifacts/restricted"
    prepared = prepare_phase5_script_commitment(
        draft=draft,
        protocol=protocol,
        reviewed_plan=reviewed_plan,
        held_out_configuration=held_out_configuration,
        review_gate=review_gate,
        repository=repository,
        restricted_root=restricted,
        held_out_journal_root=restricted / "held-out-journal",
        committed_at=COMMITTED_AT,
    )
    artifact_root = repository / "artifacts/blobs/phase5-test"
    ledger = Ledger(restricted / "phase5-test.sqlite")
    try:
        artifacts = ArtifactStore(
            BlobStore(artifact_root, compression=Compression.GZIP),
            ledger,
        )
        output_root = restricted / "phase5-script-commitment"
        manifest, created = materialize_phase5_script_commitment(
            prepared=prepared,
            artifacts=artifacts,
            output_root=output_root,
            restricted_root=restricted,
        )
        assert created is True
        replayed, created_again = materialize_phase5_script_commitment(
            prepared=prepared,
            artifacts=artifacts,
            output_root=output_root,
            restricted_root=restricted,
        )
        assert created_again is False
        assert replayed == manifest
        path = output_root / "commitment_manifest.json"
        assert load_phase5_script_commitment(path) == manifest
        assert path.stat().st_mode & 0o777 == 0o600
        assert not tuple(output_root.glob(".*.tmp"))

        cas = RestrictedPhase5CAS(artifacts)
        for entry, (instruction, freeze) in zip(
            manifest.entries,
            prepared.records,
            strict=True,
        ):
            assert cas.load(
                entry.instruction,
                type(instruction),
                object_kind="revision_instruction",
            ) == instruction
            assert cas.load(
                entry.freeze,
                type(freeze),
                object_kind="scripted_revision_freeze",
            ) == freeze
    finally:
        ledger.close()


def test_commitment_rejects_started_journal_and_late_activation(tmp_path: Path) -> None:
    protocol, draft = _draft()
    reviewed_plan, held_out_configuration, review_gate = _runtime_bound_review(protocol)
    repository = _mini_repository(tmp_path, protocol)
    restricted = repository / "artifacts/restricted"
    journal = restricted / "held-out-journal"
    journal.mkdir()
    (journal / "call_manifest.json").write_text("started", encoding="utf-8")
    with pytest.raises(Phase5CommitmentError, match="too late"):
        prepare_phase5_script_commitment(
            draft=draft,
            protocol=protocol,
            reviewed_plan=reviewed_plan,
            held_out_configuration=held_out_configuration,
            review_gate=review_gate,
            repository=repository,
            restricted_root=restricted,
            held_out_journal_root=journal,
            committed_at=COMMITTED_AT,
        )

    (journal / "call_manifest.json").unlink()
    with pytest.raises(Phase5CommitmentError, match="strictly follow commitment"):
        prepare_phase5_script_commitment(
            draft=draft,
            protocol=protocol,
            reviewed_plan=reviewed_plan,
            held_out_configuration=held_out_configuration,
            review_gate=review_gate,
            repository=repository,
            restricted_root=restricted,
            held_out_journal_root=journal,
            committed_at=ACTIVATION_AT,
        )


def test_commitment_requires_review_binding_and_clears_watchdog_floor(
    tmp_path: Path,
) -> None:
    protocol, draft = _draft(
        activation_at=COMMITTED_AT + timedelta(seconds=26460)
    )
    reviewed_plan, held_out_configuration, review_gate = _runtime_bound_review(protocol)
    repository = _mini_repository(tmp_path, protocol)
    restricted = repository / "artifacts/restricted"
    with pytest.raises(Phase5CommitmentError, match="watchdog floor"):
        prepare_phase5_script_commitment(
            draft=draft,
            protocol=protocol,
            reviewed_plan=reviewed_plan,
            held_out_configuration=held_out_configuration,
            review_gate=review_gate,
            repository=repository,
            restricted_root=restricted,
            held_out_journal_root=restricted / "held-out-journal",
            committed_at=COMMITTED_AT,
        )

    mismatched_review = review_gate.model_copy(
        update={"final_reviewed_seal_hash": digest("wrong-final-review")}
    )
    with pytest.raises(Phase5CommitmentError, match="independent-review completion"):
        prepare_phase5_script_commitment(
            draft=draft,
            protocol=protocol,
            reviewed_plan=reviewed_plan,
            held_out_configuration=held_out_configuration,
            review_gate=mismatched_review,
            repository=repository,
            restricted_root=restricted,
            held_out_journal_root=restricted / "held-out-journal",
            committed_at=COMMITTED_AT,
        )


def test_review_gate_is_rejected_before_any_query_payload_is_opened(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol, draft = _draft()
    reviewed_plan, held_out_configuration, review_gate = _runtime_bound_review(protocol)
    repository = _mini_repository(tmp_path, protocol)
    mismatched_review = review_gate.model_copy(
        update={"final_reviewed_seal_hash": digest("wrong-final-review")}
    )

    def forbid_query_open(*_args, **_kwargs):
        raise AssertionError("query payload opened before review gate")

    monkeypatch.setattr(
        "story_projection_onto.phase5_commitment.load_model_eligible_query",
        forbid_query_open,
    )
    with pytest.raises(Phase5CommitmentError, match="independent-review completion"):
        prepare_phase5_script_commitment(
            draft=draft,
            protocol=protocol,
            reviewed_plan=reviewed_plan,
            held_out_configuration=held_out_configuration,
            review_gate=mismatched_review,
            repository=repository,
            restricted_root=repository / "artifacts/restricted",
            held_out_journal_root=(
                repository / "artifacts/restricted/held-out-journal"
            ),
            committed_at=COMMITTED_AT,
        )


def test_commitment_rejects_a_changed_held_out_condition_block_order(
    tmp_path: Path,
) -> None:
    protocol, draft = _draft()
    reviewed_plan, held_out_configuration, review_gate = _runtime_bound_review(protocol)
    calls = list(reviewed_plan.call_manifest.calls)
    calls[0], calls[24] = calls[24], calls[0]
    changed_manifest = reviewed_plan.call_manifest.model_copy(
        update={"calls": tuple(calls)}
    )
    changed_plan = reviewed_plan.model_copy(update={"call_manifest": changed_manifest})
    repository = _mini_repository(tmp_path, protocol)

    with pytest.raises(Phase5CommitmentError, match="call order"):
        prepare_phase5_script_commitment(
            draft=draft,
            protocol=protocol,
            reviewed_plan=changed_plan,
            held_out_configuration=held_out_configuration,
            review_gate=review_gate,
            repository=repository,
            restricted_root=repository / "artifacts/restricted",
            held_out_journal_root=(
                repository / "artifacts/restricted/held-out-journal"
            ),
            committed_at=COMMITTED_AT,
        )


def test_commitment_rejects_anchor_outside_public_stage(tmp_path: Path) -> None:
    protocol, draft = _draft()
    reviewed_plan, held_out_configuration, review_gate = _runtime_bound_review(protocol)
    repository = _mini_repository(tmp_path, protocol)
    first = draft.drafts[0]
    instruction_payload = first.instruction.model_dump(
        mode="python",
        exclude={"content_hash"},
    )
    revision_payload = first.instruction.revision.model_dump(
        mode="python",
        exclude={"content_hash"},
    )
    anchor_payload = first.instruction.revision.anchors[0].model_dump(
        mode="python",
        exclude={"content_hash"},
    )
    anchor_payload["evidence_ids"] = ("ev_missing",)
    revision_payload["anchors"] = (FeedbackAnchor(**anchor_payload),)
    instruction_payload["revision"] = type(first.instruction.revision)(**revision_payload)
    bad_instruction = type(first.instruction)(**instruction_payload)
    bad_draft = ScriptedRevisionDraftManifest(
        **draft.model_dump(
            mode="python",
            exclude={"content_hash", "drafts"},
        ),
        drafts=(
            ScriptedRevisionDraft(
                episode_id=first.episode_id,
                instruction=bad_instruction,
            ),
            *draft.drafts[1:],
        ),
    )
    with pytest.raises(Phase5CommitmentError, match="outside its public query stage"):
        prepare_phase5_script_commitment(
            draft=bad_draft,
            protocol=protocol,
            reviewed_plan=reviewed_plan,
            held_out_configuration=held_out_configuration,
            review_gate=review_gate,
            repository=repository,
            restricted_root=repository / "artifacts/restricted",
            held_out_journal_root=(
                repository / "artifacts/restricted/held-out-journal"
            ),
            committed_at=COMMITTED_AT,
        )


def test_validate_only_cli_does_not_create_ledger_cas_or_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    observed = datetime.now(UTC)
    activation = observed + timedelta(hours=8)
    protocol, live_draft = _draft(
        authored_at=observed - timedelta(seconds=1),
        activation_at=activation,
    )
    reviewed_plan, held_out_configuration, review_gate = _runtime_bound_review(protocol)
    repository = _mini_repository(tmp_path, protocol)
    restricted = repository / "artifacts/restricted"
    draft_path = restricted / "draft.json"
    draft_path.write_text(live_draft.to_canonical_json() + "\n", encoding="utf-8")
    os.chmod(draft_path, 0o600)
    ledger_path = restricted / "must-not-exist.sqlite"
    artifact_root = repository / "artifacts/blobs/must-not-exist"
    output_root = restricted / "must-not-exist-output"
    ledger_path.touch()
    artifact_root.mkdir(parents=True)
    ledger_bytes = ledger_path.read_bytes()

    monkeypatch.setattr(
        freeze_cli,
        "open_phase5_hash_bound_reviewed_plan",
        lambda **_kwargs: (reviewed_plan, held_out_configuration, review_gate),
    )
    return_code = freeze_cli.main(
        [
            "--repository",
            str(repository),
            "--draft-manifest",
            str(draft_path),
            "--protocol",
            str(ROOT / "configs/study/feedback.json"),
            "--held-out-journal-root",
            str(restricted / "held-out-journal"),
            "--ledger",
            str(ledger_path),
            "--artifact-root",
            str(artifact_root),
            "--restricted-root",
            str(restricted),
            "--output-root",
            str(output_root),
            "--validate-only",
        ]
    )
    output = capsys.readouterr().out
    assert return_code == 0, output
    summary = json.loads(output)
    assert summary["state"] == "validated"
    assert summary["writes_performed"] is False
    assert summary["script_count"] == 6
    assert summary["registered_watchdog_floor_seconds"] == 26460
    assert ledger_path.read_bytes() == ledger_bytes
    assert not tuple(artifact_root.iterdir())
    assert not output_root.exists()
