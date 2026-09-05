from __future__ import annotations

import hashlib
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from story_projection_onto.contracts import ConditionName
from story_projection_onto.metrics.config import CommunityReviewEntry
from story_projection_onto.scorer_only.blinded_postrun_review import (
    FROZEN_COMMUNITY_REVIEW_SELECTION_RULE_HASH,
    FROZEN_REGISTERED_FAILURE_SIGNAL_POLICY_HASH,
    BlindedReviewError,
    CommunityPartitionSource,
    CommunityReviewCompletion,
    CommunityReviewSourceManifest,
    ErrorAdjudicationEntry,
    ErrorReviewAdjudication,
    ErrorReviewCompletion,
    ErrorReviewJudgment,
    HeldOutErrorCode,
    HeldOutFailureSource,
    HeldOutFailureSourceManifest,
    ReviewPanelSource,
    materialize_community_review_finalization,
    materialize_community_review_package,
    materialize_error_review_finalization,
    materialize_error_review_package,
    prepare_error_review_finalization,
    prepare_error_review_package,
)

ROOT = Path(__file__).resolve().parents[2]
TAXONOMY = ROOT / "configs/study/held_out_error_review_taxonomy.json"
RUBRIC = ROOT / "configs/study/community_review_template.json"
NOW = datetime(2026, 1, 2, tzinfo=UTC)


def _write_model(path: Path, model: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(model.to_canonical_json() + "\n")  # type: ignore[attr-defined]


def _panel(directory: Path, name: str, value: str) -> ReviewPanelSource:
    payload = (f'{{"neutral_projection":"{value}"}}\n').encode()
    (directory / name).write_bytes(payload)
    return ReviewPanelSource(
        relative_path=name,
        file_sha256=hashlib.sha256(payload).hexdigest(),
        media_type="application/json",
    )


def _error_source(restricted: Path) -> Path:
    source_directory = restricted / "sources" / "errors"
    source_directory.mkdir(parents=True)
    manifest = HeldOutFailureSourceManifest(
        manifest_id="heldout-failures-v1",
        analysis_artifact_hash="a" * 64,
        analysis_index_file_sha256="b" * 64,
        phase4_replay_receipt_hash="c" * 64,
        phase4_output_inventory_hash="d" * 64,
        primary_intended_manifest_hash="e" * 64,
        primary_scores_file_sha256="f" * 64,
        primary_scores_logical_hash="1" * 64,
        held_out_call_manifest_hash="2" * 64,
        held_out_call_manifest_file_sha256="3" * 64,
        held_out_execution_hash="4" * 64,
        held_out_execution_file_sha256="5" * 64,
        primary_receipt_inventory_hash="6" * 64,
        scored_cell_eligibility_hash="7" * 64,
        failure_signal_policy_hash=FROZEN_REGISTERED_FAILURE_SIGNAL_POLICY_HASH,
        eligible_failure_count=2,
        items=(
            HeldOutFailureSource(
                source_failure_id="failure-01",
                world_id="syn-test-01",
                context_id="context-alpha",
                condition=ConditionName.C0_CLASSICAL_PRE,
                seed_block=None,
                result_artifact_hash="1" * 64,
                projection_hash="2" * 64,
                output_status="succeeded",
                failed_gate_ids=("strict-qualified-assertion-f1-below-1",),
                review_panel=_panel(source_directory, "one.json", "first"),
            ),
            HeldOutFailureSource(
                source_failure_id="failure-02",
                world_id="syn-test-02",
                context_id="context-beta",
                condition=ConditionName.C2_LLM_QUERY,
                seed_block=2,
                result_artifact_hash="3" * 64,
                projection_hash=None,
                output_status="invalid",
                failed_gate_ids=("output-status-invalid",),
                review_panel=_panel(source_directory, "two.json", "second"),
            ),
        ),
        selected_at=NOW,
    )
    path = source_directory / "source.json"
    _write_model(path, manifest)
    return path


def test_error_review_is_condition_blind_complete_append_only_and_world_unit(
    tmp_path: Path,
) -> None:
    restricted = tmp_path / "restricted"
    restricted.mkdir()
    source_path = _error_source(restricted)
    prepared, _, _ = prepare_error_review_package(
        restricted_root=restricted,
        source_manifest_path=source_path,
        taxonomy_path=TAXONOMY,
    )
    visible = prepared.to_canonical_json()
    for secret in (
        "C0",
        "C2",
        "syn-test-01",
        "context-alpha",
        "failure-01",
        "1" * 64,
    ):
        assert secret not in visible

    package_root, state, package = materialize_error_review_package(
        restricted_root=restricted,
        source_manifest_path=source_path,
        taxonomy_path=TAXONOMY,
        output_root=restricted / "error-packages",
    )
    assert state == "created"
    replay_root, replay_state, _ = materialize_error_review_package(
        restricted_root=restricted,
        source_manifest_path=source_path,
        taxonomy_path=TAXONOMY,
        output_root=restricted / "error-packages",
    )
    assert replay_state == "verified"
    assert replay_root == package_root

    judgments = tuple(
        ErrorReviewJudgment(
            blind_item_id=item.blind_item_id,
            error_codes=(HeldOutErrorCode.INVALID_STRUCTURE,),
        )
        for item in package.items
    )
    completion = ErrorReviewCompletion(
        completion_id="external-error-review-v1",
        package_hash=package.content_hash,
        reviewer_id="reviewer-opaque-1",
        completed_at=NOW,
        judgments=judgments,
    )
    completion_path = restricted / "external" / "completion.json"
    _write_model(completion_path, completion)
    adjudication = ErrorReviewAdjudication(
        adjudication_id="error-adjudication-v1",
        package_hash=package.content_hash,
        completion_hash=completion.content_hash,
        adjudicator_id="adjudicator-opaque-1",
        adjudicated_at=NOW,
        entries=tuple(
            ErrorAdjudicationEntry(
                blind_item_id=item.blind_item_id,
                disposition="accept",
                final_error_codes=(HeldOutErrorCode.INVALID_STRUCTURE,),
                rationale="Accepted the blinded review code.",
            )
            for item in package.items
        ),
    )
    adjudication_path = restricted / "external" / "adjudication.json"
    _write_model(adjudication_path, adjudication)
    final_root, final_state, finalization = materialize_error_review_finalization(
        restricted_root=restricted,
        package_root=package_root,
        completion_path=completion_path,
        adjudication_path=adjudication_path,
        output_root=restricted / "error-final",
    )
    assert final_state == "created"
    rows = (final_root / finalization.canonical_table_file).read_text().splitlines()
    assert len(rows) == 3
    assert all(row.endswith(",world") for row in rows[1:])
    assert "C0" in rows[1] or "C0" in rows[2]
    assert "C2" in rows[1] or "C2" in rows[2]


def test_error_review_rejects_incomplete_completion_and_symlink_ancestor(
    tmp_path: Path,
) -> None:
    restricted = tmp_path / "restricted"
    restricted.mkdir()
    source_path = _error_source(restricted)
    package_root, _, package = materialize_error_review_package(
        restricted_root=restricted,
        source_manifest_path=source_path,
        taxonomy_path=TAXONOMY,
        output_root=restricted / "packages",
    )
    completion = ErrorReviewCompletion(
        completion_id="incomplete-review",
        package_hash=package.content_hash,
        reviewer_id="reviewer",
        completed_at=NOW,
        judgments=(
            ErrorReviewJudgment(
                blind_item_id=package.items[0].blind_item_id,
                error_codes=(HeldOutErrorCode.OVER_MERGE,),
            ),
        ),
    )
    completion_path = restricted / "completion.json"
    _write_model(completion_path, completion)
    adjudication = ErrorReviewAdjudication(
        adjudication_id="incomplete-adjudication",
        package_hash=package.content_hash,
        completion_hash=completion.content_hash,
        adjudicator_id="adjudicator",
        adjudicated_at=NOW,
        entries=(
            ErrorAdjudicationEntry(
                blind_item_id=package.items[0].blind_item_id,
                disposition="accept",
                final_error_codes=(HeldOutErrorCode.OVER_MERGE,),
                rationale="Accepted.",
            ),
        ),
    )
    adjudication_path = restricted / "adjudication.json"
    _write_model(adjudication_path, adjudication)
    with pytest.raises(BlindedReviewError, match="every blinded failure"):
        prepare_error_review_finalization(
            restricted_root=restricted,
            package_root=package_root,
            completion_path=completion_path,
            adjudication_path=adjudication_path,
        )

    outside = tmp_path / "outside"
    outside.mkdir()
    (restricted / "linked").symlink_to(source_path.parent, target_is_directory=True)
    with pytest.raises(BlindedReviewError, match="symlinked"):
        prepare_error_review_package(
            restricted_root=restricted,
            source_manifest_path=restricted / "linked" / source_path.name,
            taxonomy_path=TAXONOMY,
        )


def test_existing_reviewer_bundle_rejects_symlink_directory_and_special_file(
    tmp_path: Path,
) -> None:
    restricted = tmp_path / "restricted"
    restricted.mkdir()
    source_path = _error_source(restricted)
    package_root, _, _ = materialize_error_review_package(
        restricted_root=restricted,
        source_manifest_path=source_path,
        taxonomy_path=TAXONOMY,
        output_root=restricted / "packages",
    )
    (package_root / "reviewer" / "leak").symlink_to(
        package_root / "scorer_only",
        target_is_directory=True,
    )
    with pytest.raises(BlindedReviewError, match="contains a symlink"):
        materialize_error_review_package(
            restricted_root=restricted,
            source_manifest_path=source_path,
            taxonomy_path=TAXONOMY,
            output_root=restricted / "packages",
        )

    (package_root / "reviewer" / "leak").unlink()
    fifo = package_root / "reviewer" / "unexpected.fifo"
    try:
        os.mkfifo(fifo)
    except (AttributeError, NotImplementedError, OSError):
        pytest.skip("named pipes are not available on this platform")
    with pytest.raises(BlindedReviewError, match="contains a special file"):
        materialize_error_review_package(
            restricted_root=restricted,
            source_manifest_path=source_path,
            taxonomy_path=TAXONOMY,
            output_root=restricted / "packages",
        )


def test_community_rubric_package_is_blind_and_scores_only_after_rejoin(
    tmp_path: Path,
) -> None:
    restricted = tmp_path / "restricted"
    source_directory = restricted / "sources" / "community"
    source_directory.mkdir(parents=True)
    conditions = (
        (ConditionName.C0_CLASSICAL_PRE, None),
        (ConditionName.C1_LLM_PRE, 1),
        (ConditionName.C2_LLM_QUERY, 1),
        (ConditionName.A_FIXED_SELECT, 1),
    )
    items = tuple(
        CommunityPartitionSource(
            source_partition_id=f"partition-{condition_index}-{resolution_index}",
            world_id="syn-test-01",
            context_id="context-1",
            condition=condition,
            seed_block=seed,
            resolution=resolution,
            projection_hash=hashlib.sha256(
                f"projection-{condition_index}".encode()
            ).hexdigest(),
            partition_hash=hashlib.sha256(
                f"partition-{condition_index}-{resolution_index}".encode()
            ).hexdigest(),
            node_count=8,
            cluster_count=2,
            review_panel=_panel(
                source_directory,
                f"partition-{condition_index}-{resolution_index}.json",
                f"{condition_index}-{resolution_index}",
            ),
        )
        for condition_index, (condition, seed) in enumerate(conditions, 1)
        for resolution_index, resolution in enumerate((0.25, 0.5, 1.0), 1)
    )
    source = CommunityReviewSourceManifest(
        manifest_id="community-source-v1",
        analysis_artifact_hash="a" * 64,
        analysis_index_file_sha256="b" * 64,
        phase4_replay_receipt_hash="c" * 64,
        phase4_output_inventory_hash="d" * 64,
        phase4_source_binding_inventory_hash="e" * 64,
        analysis_configuration_hash="f" * 64,
        metric_configuration_hash="1" * 64,
        metric_version_hash="2" * 64,
        primary_intended_manifest_hash="3" * 64,
        primary_scores_file_sha256="4" * 64,
        primary_scores_logical_hash="5" * 64,
        complete_bundles_file_sha256="6" * 64,
        complete_bundles_logical_hash="7" * 64,
        scorer_plans_file_sha256="8" * 64,
        scorer_plans_logical_hash="9" * 64,
        selection_rule_hash=FROZEN_COMMUNITY_REVIEW_SELECTION_RULE_HASH,
        preoutput_candidate_inventory_hash="a" * 64,
        preoutput_eligible_context_count=1,
        selected_cell_lineage_hash="b" * 64,
        registered_resolution_values=(0.25, 0.5, 1.0),
        items=tuple(sorted(items, key=lambda item: item.source_partition_id)),
        selected_at=NOW,
    )
    source_path = source_directory / "source.json"
    _write_model(source_path, source)
    package_root, _, package = materialize_community_review_package(
        restricted_root=restricted,
        source_manifest_path=source_path,
        rubric_template_path=RUBRIC,
        output_root=restricted / "community-packages",
    )
    visible = (package_root / "reviewer/review_package.json").read_text()
    assert "C0" not in visible and "C2" not in visible and "syn-test" not in visible
    completion = CommunityReviewCompletion(
        completion_id="community-completion-v1",
        package_hash=package.content_hash,
        reviewer_id="reviewer-opaque-2",
        completed_at=NOW,
        reviews=tuple(
            CommunityReviewEntry(
                blinded_output_id=item.blinded_output_id,
                semantic_coherence=5,
                interpretability=4,
                evidence_support=3,
                reviewer_note="Condition-blind assessment.",
            )
            for item in package.items
        ),
    )
    completion_path = restricted / "external" / "community.json"
    _write_model(completion_path, completion)
    final_root, state, finalization = materialize_community_review_finalization(
        restricted_root=restricted,
        package_root=package_root,
        completion_path=completion_path,
        output_root=restricted / "community-final",
    )
    assert state == "created"
    table = (final_root / finalization.canonical_table_file).read_text()
    assert ",4,world\n" in table
    assert "Condition-blind assessment" not in table
    assert finalization.reviewed_partition_count == 12
