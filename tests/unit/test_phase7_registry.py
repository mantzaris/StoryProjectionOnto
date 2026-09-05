from __future__ import annotations

import csv
import hashlib
import io
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest

from story_projection_onto.case_study_analysis import (
    NOVEL_CASE_COLUMNS,
    CaseStudyNarrativeAnalysisReceipt,
)
from story_projection_onto.contracts import (
    ConditionName,
)
from story_projection_onto.contracts import (
    canonical_sha256 as canonical_record_sha256,
)
from story_projection_onto.final_accounting import (
    FAILURE_ACCOUNTING_COLUMNS,
    compile_final_accounting,
    materialize_final_accounting_recipe,
)
from story_projection_onto.independent_review_runtime import (
    load_completed_review,
    materialize_public_reviewed_gold,
    materialize_review_completion,
    prepare_public_reviewed_gold_publication,
    prepare_review_completion,
)
from story_projection_onto.metrics.config import CommunityReviewEntry
from story_projection_onto.phase7_compiler import Phase7CompilationError
from story_projection_onto.phase7_registry import (
    materialize_phase7_source_registry,
    verify_phase7_source_registry,
    write_phase7_source_registry,
)
from story_projection_onto.reporting import canonical_sha256
from story_projection_onto.scorer_only.blinded_postrun_review import (
    FROZEN_COMMUNITY_REVIEW_SELECTION_RULE_HASH,
    CommunityPartitionSource,
    CommunityReviewCompletion,
    CommunityReviewSourceManifest,
    ErrorAdjudicationEntry,
    ErrorReviewAdjudication,
    ErrorReviewCompletion,
    ErrorReviewJudgment,
    HeldOutErrorCode,
    materialize_community_review_finalization,
    materialize_community_review_package,
    materialize_error_review_finalization,
    materialize_error_review_package,
)
from story_projection_onto.scorer_only.phase5_report import (
    FEEDBACK_TABLE_FILE_NAME,
    FEEDBACK_TABLE_RECEIPT_FILE_NAME,
    Phase5FeedbackTableReceipt,
    materialize_phase5_feedback_table,
    prepare_phase5_feedback_table,
)
from tests.unit.test_blinded_postrun_review import _error_source, _panel, _write_model
from tests.unit.test_final_accounting import _fixture as _final_accounting_fixture
from tests.unit.test_independent_review_runtime import _external_records
from tests.unit.test_phase5_report import _complete_sources

ROOT = Path(__file__).resolve().parents[2]


def _source_root(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    for relative in (
        "configs/study/community_review_template.json",
        "configs/study/held_out_error_review_taxonomy.json",
        "configs/study/phase7_compiler.json",
        "configs/study/public_reproduction_sources.json",
        "configs/study/reporting.json",
    ):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    return root


def _recipe_payload() -> dict[str, object]:
    payload = json.loads(
        (ROOT / "configs/study/phase7_source_registry.template.json").read_text(
            encoding="utf-8"
        )
    )
    payload.pop("manifest_sha256")
    for predecessor in payload["predecessors"]:
        for artifact in predecessor["artifacts"]:
            artifact.pop("file_sha256")
            artifact.pop("logical_hash")
    for table in payload["tables"]:
        table.pop("source_row_count")
        table.pop("output_row_count")
        table.pop("singleton_join_row_count", None)
    payload["recipe_sha256"] = canonical_sha256(payload)
    return payload


def _write_recipe(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _novel_case_rows() -> list[dict[str, str]]:
    dimensions = (
        "contextual_relevance",
        "evidence_support",
        "high_level_organization",
        "rare_pivotal_preservation",
        "temporal_integrity",
    )
    conditions = ("C0", "C1", "C2")
    rows: list[dict[str, str]] = []
    for window in range(1, 5):
        for context_suffix in ("a", "b"):
            context_id = f"case-context-{window}-{context_suffix}"
            for condition in conditions:
                for dimension in dimensions:
                    rows.append(
                        {
                            "scope": "bounded_same_evidence",
                            "window_id": f"case-window-{window}",
                            "context_id": context_id,
                            "condition": condition,
                            "metric": f"review_{dimension}",
                            "value": "1",
                            "status": "complete",
                            "interpretation": (
                                "descriptive_human_review_no_population_inference"
                            ),
                        }
                    )
                if context_suffix == "a":
                    for subject in ("assertion", "event"):
                        for measure in ("f1", "precision", "recall"):
                            rows.append(
                                {
                                    "scope": "bounded_same_evidence",
                                    "window_id": f"case-window-{window}",
                                    "context_id": context_id,
                                    "condition": condition,
                                    "metric": f"detailed_{subject}_{measure}",
                                    "value": "1",
                                    "status": "value",
                                    "interpretation": (
                                        "descriptive_review_matching_no_population_inference"
                                    ),
                                }
                            )
    for metric in (
        "horizon_rejected_match_count",
        "omitted_by_cap_count",
        "output_succeeded",
        "retrieved_evidence_count",
    ):
        rows.append(
            {
                "scope": "full_index_operational_noncausal",
                "window_id": "case-window-operational",
                "context_id": "case-context-operational",
                "condition": "C2",
                "metric": metric,
                "value": "1",
                "status": "succeeded",
                "interpretation": (
                    "operational_retrieval_not_same_evidence_causal_comparison"
                ),
            }
        )
    return rows


def _write_novel_case_sources(
    root: Path,
    rows: list[dict[str, str]],
    *,
    receipt_table_hash: str | None = None,
) -> tuple[Path, Path, CaseStudyNarrativeAnalysisReceipt]:
    ordered = sorted(
        rows,
        key=lambda row: tuple(row[column] for column in NOVEL_CASE_COLUMNS[:5]),
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=list(NOVEL_CASE_COLUMNS),
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(ordered)
    table = root / "artifacts/public/case_study/novel_case.csv"
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_bytes(stream.getvalue().encode("utf-8"))
    table_hash = hashlib.sha256(table.read_bytes()).hexdigest()
    receipt = CaseStudyNarrativeAnalysisReceipt(
        analysis_id="test-case-analysis",
        execution_plan_hash="1" * 64,
        restricted_index_manifest_hash="2" * 64,
        corpus_source_sha256="3" * 64,
        terminal_resume_manifest_hash="4" * 64,
        completed_review_hash="5" * 64,
        public_alias_manifest_hash="6" * 64,
        restricted_table_file_sha256=receipt_table_hash or table_hash,
        publication_status="public_scan_passed",
        public_table_file_sha256=receipt_table_hash or table_hash,
        protected_canary_manifest_hash="7" * 64,
        release_scan_receipt_hash="8" * 64,
        public_row_inventory_hash=canonical_sha256(
            sorted(
                rows,
                key=lambda row: tuple(row[column] for column in NOVEL_CASE_COLUMNS),
            )
        ),
        public_row_count=len(rows),
        compiled_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    receipt_path = root / "artifacts/restricted/case_study/analysis_receipt.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(receipt.to_canonical_json() + "\n", encoding="utf-8")
    return table, receipt_path, receipt


def _novel_case_recipe(
    *,
    include_receipt_binding: bool = True,
) -> dict[str, object]:
    payload = _recipe_payload()
    case_study = next(
        item for item in payload["predecessors"] if item["family"] == "case_study"
    )
    case_study.update(
        {
            "status": "complete",
            "reason": "TEST-ONLY bounded narrative analysis is complete.",
            "artifacts": [
                {
                    "artifact_id": "novel-case-table",
                    "family": "case_study",
                    "relative_path": "artifacts/public/case_study/novel_case.csv",
                    "media_type": "text/csv",
                    "release_class": "public",
                },
                {
                    "artifact_id": "novel-case-receipt",
                    "family": "case_study",
                    "relative_path": (
                        "artifacts/restricted/case_study/analysis_receipt.json"
                    ),
                    "media_type": "application/json",
                    "release_class": "restricted",
                    "logical_hash_field": "content_hash",
                    "logical_hash_mode": "canonical_without_field",
                },
            ],
        }
    )
    table = next(
        item for item in payload["tables"] if item["table_id"] == "novel_case"
    )
    table.update(
        {
            "status": "complete",
            "reason": "TEST-ONLY receipt-bound narrative table is complete.",
            "scope": "final",
            "source_artifact_ids": ["novel-case-table", "novel-case-receipt"],
            "source_table_artifact_id": "novel-case-table",
        }
    )
    if include_receipt_binding:
        table["source_receipt_artifact_id"] = "novel-case-receipt"
    payload["public_artifact_ids"] = ["novel-case-table"]
    payload["recipe_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "recipe_sha256"}
    )
    return payload


def _reviewed_gold_recipe_payload(
    *,
    publication_artifact_ids: dict[str, str],
) -> dict[str, object]:
    payload = _recipe_payload()
    completion_id = "independent-review-completion-manifest"
    public_ids = tuple(sorted(publication_artifact_ids.values()))
    heldout = next(
        item for item in payload["predecessors"] if item["family"] == "heldout"
    )
    heldout["artifacts"] = [
        {
            "artifact_id": completion_id,
            "family": "heldout",
            "relative_path": (
                "artifacts/restricted/scorer_only/independent_review/"
                "completion_manifest.json"
            ),
            "media_type": "application/json",
            "release_class": "restricted",
            "logical_hash_field": "content_hash",
            "logical_hash_mode": "immutable_record",
        },
        *[
            {
                "artifact_id": artifact_id,
                "family": "heldout",
                "relative_path": relative_path,
                "media_type": "application/json",
                "release_class": "public",
                "logical_hash_field": "content_hash",
                "logical_hash_mode": "immutable_record",
            }
            for relative_path, artifact_id in sorted(publication_artifact_ids.items())
        ],
    ]
    for phase in payload["phases"]:
        if phase["phase_id"] == "phase_2":
            phase["source_artifact_ids"] = [completion_id]
        elif phase["phase_id"] == "phase_7":
            phase["source_artifact_ids"] = [
                publication_artifact_ids[
                    "artifacts/public/scorer_only/final_reviewed_gold/"
                    "publication_manifest.json"
                ]
            ]
    payload["independent_review_completion_manifest_artifact_id"] = completion_id
    payload["public_reviewed_gold_manifest_artifact_id"] = publication_artifact_ids[
        "artifacts/public/scorer_only/final_reviewed_gold/publication_manifest.json"
    ]
    payload["public_artifact_ids"] = list(public_ids)
    payload["recipe_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "recipe_sha256"}
    )
    return payload


def _write_accounting_csv(
    path: Path,
    columns: tuple[str, ...],
    rows: list[dict[str, str]],
) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    payload = stream.getvalue().encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _final_accounting_recipe_payload(
    root: Path,
) -> tuple[dict[str, object], Path, Path, Path, Path]:
    _final_accounting_fixture(root)
    materialized = materialize_final_accounting_recipe(
        source_recipe_path=root / "source_recipe.json",
        source_root=root,
        output_root=root / "artifacts/restricted/phase7/final-accounting-materialized",
    )
    compiled = compile_final_accounting(
        recipe_path=materialized.recipe_path,
        source_root=root,
        output_root=root / "artifacts/public/runtime/final-accounting",
    )
    failure_path = compiled.failure_table_path
    resource_path = compiled.resource_table_path
    receipt_path = compiled.receipt_path
    final_recipe_path = materialized.recipe_path

    payload = _recipe_payload()
    family_artifact_ids: dict[str, str] = {}
    for predecessor in payload["predecessors"]:
        family = predecessor["family"]
        if family == "runtime":
            artifacts = [
                {
                    "artifact_id": "final-failure-accounting",
                    "family": "runtime",
                    "relative_path": failure_path.relative_to(root).as_posix(),
                    "media_type": "text/csv",
                    "release_class": "public",
                },
                {
                    "artifact_id": "final-resource-accounting",
                    "family": "runtime",
                    "relative_path": resource_path.relative_to(root).as_posix(),
                    "media_type": "text/csv",
                    "release_class": "public",
                },
                {
                    "artifact_id": "final-accounting-receipt",
                    "family": "runtime",
                    "relative_path": receipt_path.relative_to(root).as_posix(),
                    "media_type": "application/json",
                    "release_class": "public",
                    "logical_hash_field": "manifest_sha256",
                    "logical_hash_mode": "canonical_without_field",
                },
                {
                    "artifact_id": "final-accounting-recipe",
                    "family": "runtime",
                    "relative_path": final_recipe_path.relative_to(root).as_posix(),
                    "media_type": "application/json",
                    "release_class": "restricted",
                    "logical_hash_field": "manifest_sha256",
                    "logical_hash_mode": "canonical_without_field",
                },
            ]
            family_artifact_ids[family] = "final-accounting-receipt"
        else:
            artifact_id = f"lineage-{family}"
            lineage_path = root / f"artifacts/public/lineage/{family}.csv"
            _write_accounting_csv(
                lineage_path,
                ("family",),
                [{"family": family}],
            )
            artifacts = [
                {
                    "artifact_id": artifact_id,
                    "family": family,
                    "relative_path": lineage_path.relative_to(root).as_posix(),
                    "media_type": "text/csv",
                    "release_class": "public",
                }
            ]
            family_artifact_ids[family] = artifact_id
        predecessor.update(
            {
                "status": "complete",
                "reason": "TEST-ONLY complete accounting lineage fixture.",
                "artifacts": artifacts,
            }
        )

    for table_id, source_id in (
        ("failure_accounting", "final-failure-accounting"),
        ("resource_accounting", "final-resource-accounting"),
    ):
        dependencies = (
            set(family_artifact_ids)
            if table_id == "resource_accounting"
            else set(family_artifact_ids) - {"storage"}
        )
        table = next(
            item for item in payload["tables"] if item["table_id"] == table_id
        )
        table.update(
            {
                "status": "complete",
                "reason": "TEST-ONLY receipt-bound final accounting is complete.",
                "scope": "final",
                "source_artifact_ids": [
                    source_id,
                    "final-accounting-receipt",
                    "final-accounting-recipe",
                    *sorted(
                        family_artifact_ids[family]
                        for family in dependencies
                        if family != "runtime"
                    ),
                ],
                "source_table_artifact_id": source_id,
                "source_receipt_artifact_id": "final-accounting-receipt",
                "row_filters": [],
            }
        )
    payload["public_artifact_ids"] = [
        "final-accounting-receipt",
        "final-failure-accounting",
        "final-resource-accounting",
    ]
    payload["recipe_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "recipe_sha256"}
    )
    return payload, failure_path, resource_path, receipt_path, final_recipe_path


def _review_artifact_recipe(
    artifact_id: str,
    path: Path,
    *,
    root: Path,
    media_type: str = "application/json",
) -> dict[str, object]:
    payload: dict[str, object] = {
        "artifact_id": artifact_id,
        "family": "heldout",
        "relative_path": path.relative_to(root).as_posix(),
        "media_type": media_type,
        "release_class": "restricted",
    }
    if media_type == "application/json":
        payload.update(
            {
                "logical_hash_field": "content_hash",
                "logical_hash_mode": "immutable_record",
            }
        )
    return payload


def _error_review_recipe_payload(root: Path) -> tuple[dict[str, object], Path]:
    restricted = root / "artifacts/restricted/review-fixture"
    restricted.mkdir(parents=True)
    source_path = _error_source(restricted)
    package_root, _, package = materialize_error_review_package(
        restricted_root=restricted,
        source_manifest_path=source_path,
        taxonomy_path=root / "configs/study/held_out_error_review_taxonomy.json",
        output_root=restricted / "error-packages",
    )
    completion = ErrorReviewCompletion(
        completion_id="phase7-error-review",
        package_hash=package.content_hash,
        reviewer_id="reviewer-opaque",
        completed_at=datetime(2026, 1, 2, tzinfo=UTC),
        judgments=tuple(
            ErrorReviewJudgment(
                blind_item_id=item.blind_item_id,
                error_codes=(HeldOutErrorCode.INVALID_STRUCTURE,),
            )
            for item in package.items
        ),
    )
    external_completion = restricted / "external/error-completion.json"
    _write_model(external_completion, completion)
    adjudication = ErrorReviewAdjudication(
        adjudication_id="phase7-error-adjudication",
        package_hash=package.content_hash,
        completion_hash=completion.content_hash,
        adjudicator_id="adjudicator-opaque",
        adjudicated_at=datetime(2026, 1, 2, tzinfo=UTC),
        entries=tuple(
            ErrorAdjudicationEntry(
                blind_item_id=item.blind_item_id,
                disposition="accept",
                final_error_codes=(HeldOutErrorCode.INVALID_STRUCTURE,),
                rationale="TEST-ONLY accepted blinded code.",
            )
            for item in package.items
        ),
    )
    external_adjudication = restricted / "external/error-adjudication.json"
    _write_model(external_adjudication, adjudication)
    final_root, _, finalization = materialize_error_review_finalization(
        restricted_root=restricted,
        package_root=package_root,
        completion_path=external_completion,
        adjudication_path=external_adjudication,
        output_root=restricted / "error-final",
    )
    paths = {
        "source_manifest_artifact_id": ("error-source", source_path),
        "package_artifact_id": (
            "error-package",
            package_root / "reviewer/review_package.json",
        ),
        "rejoin_map_artifact_id": (
            "error-rejoin",
            package_root / "scorer_only/rejoin_map.json",
        ),
        "completion_artifact_id": (
            "error-completion",
            final_root / "completion.json",
        ),
        "adjudication_artifact_id": (
            "error-adjudication",
            final_root / "adjudication.json",
        ),
        "finalization_artifact_id": (
            "error-finalization",
            final_root / "finalization.json",
        ),
        "canonical_table_artifact_id": (
            "error-table",
            final_root / finalization.canonical_table_file,
        ),
    }
    payload = _recipe_payload()
    heldout = next(
        item for item in payload["predecessors"] if item["family"] == "heldout"
    )
    heldout["artifacts"] = [
        _review_artifact_recipe(
            artifact_id,
            path,
            root=root,
            media_type=(
                "text/csv"
                if role == "canonical_table_artifact_id"
                else "application/json"
            ),
        )
        for role, (artifact_id, path) in paths.items()
    ]
    payload["error_review_artifacts"] = {
        role: artifact_id for role, (artifact_id, _path) in paths.items()
    }
    payload["recipe_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "recipe_sha256"}
    )
    return payload, final_root / finalization.canonical_table_file


def _community_review_recipe_payload(root: Path) -> tuple[dict[str, object], Path]:
    restricted = root / "artifacts/restricted/community-review-fixture"
    source_directory = restricted / "sources/community"
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
        manifest_id="phase7-community-source",
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
        selected_at=datetime(2026, 1, 2, tzinfo=UTC),
    )
    source_path = source_directory / "source.json"
    _write_model(source_path, source)
    package_root, _, package = materialize_community_review_package(
        restricted_root=restricted,
        source_manifest_path=source_path,
        rubric_template_path=root / "configs/study/community_review_template.json",
        output_root=restricted / "community-packages",
    )
    completion = CommunityReviewCompletion(
        completion_id="phase7-community-completion",
        package_hash=package.content_hash,
        reviewer_id="reviewer-opaque",
        completed_at=datetime(2026, 1, 2, tzinfo=UTC),
        reviews=tuple(
            CommunityReviewEntry(
                blinded_output_id=item.blinded_output_id,
                semantic_coherence=5,
                interpretability=4,
                evidence_support=3,
                reviewer_note="TEST-ONLY blinded assessment.",
            )
            for item in package.items
        ),
    )
    external_completion = restricted / "external/community-completion.json"
    _write_model(external_completion, completion)
    final_root, _, finalization = materialize_community_review_finalization(
        restricted_root=restricted,
        package_root=package_root,
        completion_path=external_completion,
        output_root=restricted / "community-final",
    )
    paths = {
        "source_manifest_artifact_id": ("community-source", source_path),
        "package_artifact_id": (
            "community-package",
            package_root / "reviewer/review_package.json",
        ),
        "rejoin_map_artifact_id": (
            "community-rejoin",
            package_root / "scorer_only/rejoin_map.json",
        ),
        "completion_artifact_id": (
            "community-completion",
            final_root / "completion.json",
        ),
        "finalization_artifact_id": (
            "community-finalization",
            final_root / "finalization.json",
        ),
        "canonical_table_artifact_id": (
            "community-table",
            final_root / finalization.canonical_table_file,
        ),
    }
    payload = _recipe_payload()
    heldout = next(
        item for item in payload["predecessors"] if item["family"] == "heldout"
    )
    heldout["artifacts"] = [
        _review_artifact_recipe(
            artifact_id,
            path,
            root=root,
            media_type=(
                "text/csv"
                if role == "canonical_table_artifact_id"
                else "application/json"
            ),
        )
        for role, (artifact_id, path) in paths.items()
    ]
    payload["community_review_artifacts"] = {
        role: artifact_id for role, (artifact_id, _path) in paths.items()
    }
    payload["recipe_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "recipe_sha256"}
    )
    return payload, final_root / finalization.canonical_table_file


def test_template_recipe_materializes_and_replays_without_manual_hashes(
    tmp_path: Path,
) -> None:
    root = _source_root(tmp_path)
    recipe = root / "recipe.json"
    _write_recipe(recipe, _recipe_payload())
    output = root / "artifacts/restricted/phase7/source_registry.json"
    written = write_phase7_source_registry(
        source_root=root,
        configuration_path=root / "configs/study/phase7_compiler.json",
        recipe_path=recipe,
        output_path=output,
    )
    replayed = verify_phase7_source_registry(
        source_root=root,
        configuration_path=root / "configs/study/phase7_compiler.json",
        recipe_path=recipe,
        registry_path=output,
    )
    assert replayed == written
    assert written.tables[0].table_id == "study_status"


def test_registry_binds_public_final_reviewed_gold_to_replayed_completion(
    tmp_path: Path,
) -> None:
    root = _source_root(tmp_path)
    benchmark_root = root / "data/synthetic"
    shutil.copytree(ROOT / "data/synthetic", benchmark_root)
    _package, _response, _adjudication, response_path, adjudication_path = (
        _external_records(tmp_path)
    )
    completion = prepare_review_completion(
        benchmark_root=benchmark_root,
        response_path=response_path,
        adjudication_path=adjudication_path,
    )
    restricted_root = (
        root / "artifacts/restricted/scorer_only/independent_review"
    )
    materialize_review_completion(completion, restricted_root)
    loaded = load_completed_review(
        benchmark_root=benchmark_root,
        output_root=restricted_root,
    )
    publication = prepare_public_reviewed_gold_publication(loaded)
    public_root = root / "artifacts/public/scorer_only/final_reviewed_gold"
    materialize_public_reviewed_gold(publication, public_root)
    completion_payload = json.loads(
        (restricted_root / "completion_manifest.json").read_text(encoding="utf-8")
    )
    assert completion_payload["content_hash"] == canonical_record_sha256(
        completion_payload
    )
    assert completion_payload["content_hash"] != canonical_sha256(
        {
            key: value
            for key, value in completion_payload.items()
            if key != "content_hash"
        }
    )
    publication_artifact_ids: dict[str, str] = {}
    for index, path in enumerate(sorted(public_root.rglob("*.json")), start=1):
        relative = path.relative_to(root).as_posix()
        publication_artifact_ids[relative] = f"final-reviewed-gold-public-{index:02d}"
    assert len(publication_artifact_ids) == 11

    payload = _reviewed_gold_recipe_payload(
        publication_artifact_ids=publication_artifact_ids
    )
    recipe = root / "reviewed-gold-recipe.json"
    legacy_payload = json.loads(json.dumps(payload))
    legacy_heldout = next(
        item
        for item in legacy_payload["predecessors"]
        if item["family"] == "heldout"
    )
    legacy_heldout["artifacts"][0]["logical_hash_mode"] = (
        "canonical_without_field"
    )
    legacy_payload["recipe_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in legacy_payload.items()
            if key != "recipe_sha256"
        }
    )
    _write_recipe(recipe, legacy_payload)
    with pytest.raises(Phase7CompilationError, match="canonical hash is invalid"):
        materialize_phase7_source_registry(
            source_root=root,
            configuration_path=root / "configs/study/phase7_compiler.json",
            recipe_path=recipe,
        )

    _write_recipe(recipe, payload)
    registry = materialize_phase7_source_registry(
        source_root=root,
        configuration_path=root / "configs/study/phase7_compiler.json",
        recipe_path=recipe,
    )
    assert (
        registry.independent_review_completion_manifest_artifact_id
        == "independent-review-completion-manifest"
    )
    assert registry.public_reviewed_gold_manifest_artifact_id == (
        publication_artifact_ids[
            "artifacts/public/scorer_only/final_reviewed_gold/"
            "publication_manifest.json"
        ]
    )
    assert len(registry.public_artifact_ids) == 11

    payload["public_artifact_ids"] = payload["public_artifact_ids"][:-1]
    payload["recipe_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "recipe_sha256"}
    )
    _write_recipe(recipe, payload)
    with pytest.raises(Phase7CompilationError, match="exact public registry lineage"):
        materialize_phase7_source_registry(
            source_root=root,
            configuration_path=root / "configs/study/phase7_compiler.json",
            recipe_path=recipe,
        )

    reviewed_entry = publication.manifest.reviewed_artifacts[0]
    reviewed_path = public_root / reviewed_entry.artifact_file
    stale_scorer_world = (
        benchmark_root
        / "scorer_only/held_out"
        / f"{reviewed_entry.world_id}.json"
    )
    reviewed_path.write_bytes(stale_scorer_world.read_bytes())
    payload = _reviewed_gold_recipe_payload(
        publication_artifact_ids=publication_artifact_ids
    )
    _write_recipe(recipe, payload)
    with pytest.raises(Phase7CompilationError, match="publication drift"):
        materialize_phase7_source_registry(
            source_root=root,
            configuration_path=root / "configs/study/phase7_compiler.json",
            recipe_path=recipe,
        )


def test_registry_replays_error_review_chain_and_rejects_table_tampering(
    tmp_path: Path,
) -> None:
    root = _source_root(tmp_path)
    payload, table_path = _error_review_recipe_payload(root)
    recipe = root / "error-review-recipe.json"
    _write_recipe(recipe, payload)
    registry = materialize_phase7_source_registry(
        source_root=root,
        configuration_path=root / "configs/study/phase7_compiler.json",
        recipe_path=recipe,
    )
    assert registry.error_review_artifacts is not None
    assert (
        registry.error_review_artifacts.canonical_table_artifact_id
        == "error-table"
    )

    original = table_path.read_bytes()
    table_path.write_bytes(original.replace(b"invalid_structure", b"over_merge", 1))
    with pytest.raises(
        Phase7CompilationError,
        match="finalization/table differs from producer replay",
    ):
        materialize_phase7_source_registry(
            source_root=root,
            configuration_path=root / "configs/study/phase7_compiler.json",
            recipe_path=recipe,
        )
    table_path.write_bytes(original)

    payload["error_review_artifacts"]["canonical_table_artifact_id"] = (
        "error-finalization"
    )
    payload["recipe_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "recipe_sha256"}
    )
    _write_recipe(recipe, payload)
    with pytest.raises(ValueError, match="distinct artifacts"):
        materialize_phase7_source_registry(
            source_root=root,
            configuration_path=root / "configs/study/phase7_compiler.json",
            recipe_path=recipe,
        )


def test_registry_replays_community_review_chain_and_rejects_table_tampering(
    tmp_path: Path,
) -> None:
    root = _source_root(tmp_path)
    payload, table_path = _community_review_recipe_payload(root)
    recipe = root / "community-review-recipe.json"
    _write_recipe(recipe, payload)
    registry = materialize_phase7_source_registry(
        source_root=root,
        configuration_path=root / "configs/study/phase7_compiler.json",
        recipe_path=recipe,
    )
    assert registry.community_review_artifacts is not None
    assert (
        registry.community_review_artifacts.canonical_table_artifact_id
        == "community-table"
    )

    original = table_path.read_bytes()
    table_path.write_bytes(original.replace(b",5,4,3,4,world", b",1,4,3,4,world", 1))
    with pytest.raises(
        Phase7CompilationError,
        match="finalization/table differs from producer replay",
    ):
        materialize_phase7_source_registry(
            source_root=root,
            configuration_path=root / "configs/study/phase7_compiler.json",
            recipe_path=recipe,
        )
    table_path.write_bytes(original)

    payload["community_review_artifacts"]["canonical_table_artifact_id"] = (
        "community-finalization"
    )
    payload["recipe_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "recipe_sha256"}
    )
    _write_recipe(recipe, payload)
    with pytest.raises(ValueError, match="distinct artifacts"):
        materialize_phase7_source_registry(
            source_root=root,
            configuration_path=root / "configs/study/phase7_compiler.json",
            recipe_path=recipe,
        )


def test_registry_binds_final_accounting_receipt_and_rejects_tampering(
    tmp_path: Path,
) -> None:
    root = tmp_path / "source"
    payload, failure_path, _resource_path, receipt_path, final_recipe_path = (
        _final_accounting_recipe_payload(root)
    )
    assert _source_root(tmp_path) == root

    def materialize(candidate: dict[str, object], name: str):
        candidate["recipe_sha256"] = canonical_sha256(
            {
                key: value
                for key, value in candidate.items()
                if key != "recipe_sha256"
            }
        )
        recipe = root / name
        _write_recipe(recipe, candidate)
        return materialize_phase7_source_registry(
            source_root=root,
            configuration_path=root / "configs/study/phase7_compiler.json",
            recipe_path=recipe,
        )

    registry = materialize(payload, "final-accounting-recipe.json")
    bindings = {
        item.table_id: item
        for item in registry.tables
        if item.table_id in {"failure_accounting", "resource_accounting"}
    }
    assert {item.source_receipt_artifact_id for item in bindings.values()} == {
        "final-accounting-receipt"
    }
    receipt_record = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt_counts = {
        item["table_id"]: item["row_count"] for item in receipt_record["outputs"]
    }
    assert {
        table_id: binding.source_row_count
        for table_id, binding in bindings.items()
    } == receipt_counts
    assert all(
        "final-accounting-recipe" in item.source_artifact_ids
        for item in bindings.values()
    )
    assert final_recipe_path.name.startswith("final_accounting_recipe.")

    original_table = failure_path.read_bytes()
    tampered_table = original_table.replace(b",success,", b",failed,", 1)
    assert tampered_table != original_table
    failure_path.write_bytes(tampered_table)
    with pytest.raises(Phase7CompilationError, match="final accounting receipt"):
        materialize(payload, "accounting-table-tamper.json")
    failure_path.write_bytes(original_table)

    original_receipt = receipt_path.read_bytes()
    receipt_payload = json.loads(original_receipt)
    receipt_payload["outputs"][0]["columns"] = list(
        reversed(receipt_payload["outputs"][0]["columns"])
    )
    receipt_payload.pop("manifest_sha256")
    receipt_payload["manifest_sha256"] = canonical_sha256(receipt_payload)
    receipt_path.write_text(
        json.dumps(receipt_payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(
        Phase7CompilationError,
        match="invalid final accounting compilation receipt",
    ):
        materialize(payload, "accounting-receipt-tamper.json")
    receipt_path.write_bytes(original_receipt)

    coherent_fake = json.loads(json.dumps(payload))
    reader = csv.DictReader(io.StringIO(original_table.decode("utf-8")))
    fake_rows = [dict(item) for item in reader]
    fake_rows[0]["outcome"] = "failed"
    fake_stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        fake_stream,
        fieldnames=FAILURE_ACCOUNTING_COLUMNS,
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(fake_rows)
    fake_failure_bytes = fake_stream.getvalue().encode("utf-8")
    fake_failure_hash = hashlib.sha256(fake_failure_bytes).hexdigest()
    fake_failure_path = failure_path.parent / (
        f"failure_accounting.{fake_failure_hash[:16]}.csv"
    )
    fake_failure_path.write_bytes(fake_failure_bytes)
    fake_receipt_payload = json.loads(original_receipt)
    fake_failure_output = next(
        item
        for item in fake_receipt_payload["outputs"]
        if item["table_id"] == "failure_accounting"
    )
    fake_failure_output["relative_path"] = fake_failure_path.name
    fake_failure_output["file_sha256"] = fake_failure_hash
    fake_receipt_payload.pop("manifest_sha256")
    fake_receipt_payload["manifest_sha256"] = canonical_sha256(
        fake_receipt_payload
    )
    fake_receipt_path = receipt_path.parent / (
        "final_accounting_receipt."
        f"{fake_receipt_payload['manifest_sha256'][:16]}.json"
    )
    fake_receipt_path.write_text(
        json.dumps(
            fake_receipt_payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    fake_runtime = next(
        item
        for item in coherent_fake["predecessors"]
        if item["family"] == "runtime"
    )
    fake_failure_artifact = next(
        item
        for item in fake_runtime["artifacts"]
        if item["artifact_id"] == "final-failure-accounting"
    )
    fake_failure_artifact["relative_path"] = fake_failure_path.relative_to(
        root
    ).as_posix()
    fake_receipt_artifact = next(
        item
        for item in fake_runtime["artifacts"]
        if item["artifact_id"] == "final-accounting-receipt"
    )
    fake_receipt_artifact["relative_path"] = fake_receipt_path.relative_to(
        root
    ).as_posix()
    with pytest.raises(Phase7CompilationError, match="compiler replay"):
        materialize(coherent_fake, "accounting-coherent-fake.json")

    wrong_family = json.loads(json.dumps(payload))
    runtime = next(
        item for item in wrong_family["predecessors"] if item["family"] == "runtime"
    )
    storage = next(
        item for item in wrong_family["predecessors"] if item["family"] == "storage"
    )
    receipt_recipe = next(
        item
        for item in runtime["artifacts"]
        if item["artifact_id"] == "final-accounting-receipt"
    )
    runtime["artifacts"].remove(receipt_recipe)
    receipt_recipe["family"] = "storage"
    storage["artifacts"].append(receipt_recipe)
    with pytest.raises(Phase7CompilationError, match="public runtime artifact"):
        materialize(wrong_family, "accounting-wrong-family.json")

    wrong_release = json.loads(json.dumps(payload))
    runtime = next(
        item for item in wrong_release["predecessors"] if item["family"] == "runtime"
    )
    receipt_recipe = next(
        item
        for item in runtime["artifacts"]
        if item["artifact_id"] == "final-accounting-receipt"
    )
    receipt_recipe["release_class"] = "restricted"
    wrong_release["public_artifact_ids"].remove("final-accounting-receipt")
    with pytest.raises(Phase7CompilationError, match="public runtime artifact"):
        materialize(wrong_release, "accounting-wrong-release.json")

    omitted_public_id = json.loads(json.dumps(payload))
    omitted_public_id["public_artifact_ids"].remove("final-resource-accounting")
    with pytest.raises(Phase7CompilationError, match="public runtime CSV"):
        materialize(omitted_public_id, "accounting-missing-public-id.json")

    filtered = json.loads(json.dumps(payload))
    failure_binding = next(
        item
        for item in filtered["tables"]
        if item["table_id"] == "failure_accounting"
    )
    failure_binding["row_filters"] = [{"column": "phase", "equals": "phase_1"}]
    with pytest.raises(Phase7CompilationError, match="prohibit filters"):
        materialize(filtered, "accounting-filtered.json")

    joined = json.loads(json.dumps(payload))
    failure_binding = next(
        item
        for item in joined["tables"]
        if item["table_id"] == "failure_accounting"
    )
    failure_binding["source_artifact_ids"].append("final-resource-accounting")
    failure_binding["singleton_join_artifact_id"] = "final-resource-accounting"
    with pytest.raises(Phase7CompilationError, match="prohibit filters"):
        materialize(joined, "accounting-joined.json")


def test_complete_scientific_table_cannot_be_relabeled_interim(
    tmp_path: Path,
) -> None:
    root = tmp_path / "source"
    payload, _failure_path, _resource_path, _receipt_path, _recipe_path = (
        _final_accounting_recipe_payload(root)
    )
    assert _source_root(tmp_path) == root
    failure_binding = next(
        item
        for item in payload["tables"]
        if item["table_id"] == "failure_accounting"
    )
    failure_binding["scope"] = "interim"
    payload["recipe_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "recipe_sha256"}
    )
    recipe = root / "accounting-interim-bypass.json"
    _write_recipe(recipe, payload)
    with pytest.raises(Phase7CompilationError, match="registered final scope"):
        materialize_phase7_source_registry(
            source_root=root,
            configuration_path=root / "configs/study/phase7_compiler.json",
            recipe_path=recipe,
        )


def test_complete_novel_case_binds_restricted_receipt_and_exact_inventory(
    tmp_path: Path,
) -> None:
    root = _source_root(tmp_path)
    rows = _novel_case_rows()
    _write_novel_case_sources(root, rows)
    recipe = root / "novel-case-recipe.json"
    _write_recipe(recipe, _novel_case_recipe())
    registry = materialize_phase7_source_registry(
        source_root=root,
        configuration_path=root / "configs/study/phase7_compiler.json",
        recipe_path=recipe,
    )
    binding = next(item for item in registry.tables if item.table_id == "novel_case")
    assert binding.source_row_count == 196
    assert binding.output_row_count == 196
    assert binding.source_receipt_artifact_id == "novel-case-receipt"
    assert "novel-case-receipt" not in registry.public_artifact_ids

    _write_recipe(recipe, _novel_case_recipe(include_receipt_binding=False))
    with pytest.raises(Phase7CompilationError, match="lacks its producer receipt"):
        materialize_phase7_source_registry(
            source_root=root,
            configuration_path=root / "configs/study/phase7_compiler.json",
            recipe_path=recipe,
        )

    _write_novel_case_sources(root, rows, receipt_table_hash="9" * 64)
    _write_recipe(recipe, _novel_case_recipe())
    with pytest.raises(Phase7CompilationError, match="differs from its producer receipt"):
        materialize_phase7_source_registry(
            source_root=root,
            configuration_path=root / "configs/study/phase7_compiler.json",
            recipe_path=recipe,
        )

    removed = rows[1:]
    _write_novel_case_sources(root, removed)
    with pytest.raises(Phase7CompilationError, match="8-by-3-by-5"):
        materialize_phase7_source_registry(
            source_root=root,
            configuration_path=root / "configs/study/phase7_compiler.json",
            recipe_path=recipe,
        )


def test_complete_feedback_registry_binds_real_phase5_table_receipt_and_tampering(
    tmp_path: Path,
) -> None:
    root = _source_root(tmp_path)
    journal, scoring = _complete_sources(tmp_path)
    _rows, table_bytes, receipt = prepare_phase5_feedback_table(
        feedback_journal_root=journal,
        scoring_root=scoring,
    )
    report_root = root / "artifacts/public/phase5_feedback_report"
    materialize_phase5_feedback_table(
        output_root=report_root,
        table_bytes=table_bytes,
        receipt=receipt,
    )

    payload = _recipe_payload()
    feedback = next(
        item for item in payload["predecessors"] if item["family"] == "feedback"
    )
    feedback.update(
        {
            "status": "complete",
            "reason": "TEST-ONLY typed Phase 5 feedback source is complete.",
            "artifacts": [
                {
                    "artifact_id": "phase5-feedback-table",
                    "family": "feedback",
                    "relative_path": (
                        "artifacts/public/phase5_feedback_report/feedback.csv"
                    ),
                    "media_type": "text/csv",
                    "release_class": "public",
                },
                {
                    "artifact_id": "phase5-feedback-table-receipt",
                    "family": "feedback",
                    "relative_path": (
                        "artifacts/public/phase5_feedback_report/"
                        "feedback_table_receipt.json"
                    ),
                    "media_type": "application/json",
                    "release_class": "public",
                    "logical_hash_field": "content_hash",
                    "logical_hash_mode": "canonical_without_field",
                    "required_json_fields": [
                        "content_hash",
                        "table_file_sha256",
                        "table_logical_hash",
                        "table_columns",
                        "table_row_count",
                        "scripted_condition_score_count",
                        "researcher_trace_count",
                    ],
                },
            ],
        }
    )
    table = next(item for item in payload["tables"] if item["table_id"] == "feedback")
    table.update(
        {
            "status": "complete",
            "reason": "TEST-ONLY feedback table and receipt are complete.",
            "scope": "final",
            "source_artifact_ids": [
                "phase5-feedback-table",
                "phase5-feedback-table-receipt",
            ],
            "source_table_artifact_id": "phase5-feedback-table",
            "source_receipt_artifact_id": "phase5-feedback-table-receipt",
            "row_filters": [],
        }
    )
    payload["public_artifact_ids"] = [
        "phase5-feedback-table",
        "phase5-feedback-table-receipt",
    ]
    payload["recipe_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "recipe_sha256"}
    )
    recipe = root / "feedback-recipe.json"
    _write_recipe(recipe, payload)

    registry = materialize_phase7_source_registry(
        source_root=root,
        configuration_path=root / "configs/study/phase7_compiler.json",
        recipe_path=recipe,
    )
    binding = next(item for item in registry.tables if item.table_id == "feedback")
    assert binding.source_row_count == receipt.table_row_count
    assert binding.output_row_count == receipt.table_row_count
    assert binding.source_receipt_artifact_id == "phase5-feedback-table-receipt"

    table_path = report_root / FEEDBACK_TABLE_FILE_NAME
    original_table_bytes = table_path.read_bytes()
    table_reader = csv.DictReader(io.StringIO(original_table_bytes.decode("utf-8")))
    table_columns = tuple(table_reader.fieldnames or ())
    mutated_rows = [dict(item) for item in table_reader]
    mutated_rows[0]["capability_limited_count"] = (
        "1" if mutated_rows[0]["capability_limited_count"] != "1" else "0"
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=table_columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(mutated_rows)
    table_path.write_text(stream.getvalue(), encoding="utf-8", newline="\n")
    with pytest.raises(Phase7CompilationError, match="differs from its producer receipt"):
        materialize_phase7_source_registry(
            source_root=root,
            configuration_path=root / "configs/study/phase7_compiler.json",
            recipe_path=recipe,
        )
    table_path.write_bytes(original_table_bytes)

    receipt_path = report_root / FEEDBACK_TABLE_RECEIPT_FILE_NAME
    original_receipt_bytes = receipt_path.read_bytes()
    receipt_payload = json.loads(original_receipt_bytes)
    receipt_payload.pop("content_hash")
    receipt_payload["table_file_sha256"] = "f" * 64
    mutated_receipt = Phase5FeedbackTableReceipt.model_validate(receipt_payload)
    receipt_path.write_text(
        mutated_receipt.to_canonical_json() + "\n",
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(Phase7CompilationError, match="differs from its producer receipt"):
        materialize_phase7_source_registry(
            source_root=root,
            configuration_path=root / "configs/study/phase7_compiler.json",
            recipe_path=recipe,
        )
    receipt_path.write_bytes(original_receipt_bytes)

    table.pop("source_receipt_artifact_id")
    payload["recipe_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "recipe_sha256"}
    )
    missing_receipt_recipe = root / "missing-feedback-receipt-recipe.json"
    _write_recipe(missing_receipt_recipe, payload)
    with pytest.raises(Phase7CompilationError, match="lacks its producer receipt"):
        materialize_phase7_source_registry(
            source_root=root,
            configuration_path=root / "configs/study/phase7_compiler.json",
            recipe_path=missing_receipt_recipe,
        )


def test_materializer_rejects_operator_defined_secondary_filter_and_rows(
    tmp_path: Path,
) -> None:
    root = _source_root(tmp_path)
    configuration = json.loads(
        (root / "configs/study/phase7_compiler.json").read_text(encoding="utf-8")
    )
    contract = next(
        item
        for item in configuration["table_contracts"]
        if item["table_id"] == "secondary_c2_vs_c0"
    )
    source = root / "artifacts/public/phase4/comparisons.csv"
    source.parent.mkdir(parents=True)
    with source.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=contract["required_columns"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow(
            {
                column: "primary" if column == "analysis_family" else "test"
                for column in contract["required_columns"]
            }
        )
        writer.writerow(
            {
                column: "secondary" if column == "analysis_family" else "test"
                for column in contract["required_columns"]
            }
        )
    payload = _recipe_payload()
    heldout = next(
        item for item in payload["predecessors"] if item["family"] == "heldout"
    )
    heldout.update(
        {
            "status": "complete",
            "reason": "TEST-ONLY immutable analysis source is complete.",
            "artifacts": [
                {
                    "artifact_id": "heldout-comparisons",
                    "family": "heldout",
                    "relative_path": "artifacts/public/phase4/comparisons.csv",
                    "media_type": "text/csv",
                    "release_class": "public",
                }
            ],
        }
    )
    table = next(
        item for item in payload["tables"] if item["table_id"] == "secondary_c2_vs_c0"
    )
    table.update(
        {
            "status": "complete",
            "reason": "TEST-ONLY secondary table is complete.",
            "scope": "final",
            "source_artifact_ids": ["heldout-comparisons"],
            "source_table_artifact_id": "heldout-comparisons",
            "row_filters": [{"column": "analysis_family", "equals": "secondary"}],
        }
    )
    payload["recipe_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "recipe_sha256"}
    )
    recipe = root / "recipe.json"
    _write_recipe(recipe, payload)
    with pytest.raises(Phase7CompilationError, match="frozen row-selection"):
        materialize_phase7_source_registry(
            source_root=root,
            configuration_path=root / "configs/study/phase7_compiler.json",
            recipe_path=recipe,
        )


def test_materializer_binds_complete_supplemental_metric_inventory_and_manifest(
    tmp_path: Path,
) -> None:
    root = _source_root(tmp_path)
    configuration = json.loads(
        (root / "configs/study/phase7_compiler.json").read_text(encoding="utf-8")
    )
    contract = next(
        item
        for item in configuration["table_contracts"]
        if item["table_id"] == "primary_c2_vs_c1"
    )
    supplemental = contract["supplemental_metric_sources"][0]
    phase4 = root / "artifacts/public/phase4"
    comparisons = phase4 / "tables/comparisons.csv"
    comparisons.parent.mkdir(parents=True)
    with comparisons.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=contract["required_columns"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow(
            {
                column: (
                    "C2-C1"
                    if column == "comparison"
                    else "strict_qualified_assertion_f1"
                    if column == "metric"
                    else "registered_primary"
                    if column == "analysis_family"
                    else "test"
                )
                for column in contract["required_columns"]
            }
        )
        writer.writerow(
            {
                column: (
                    "C2-C1"
                    if column == "comparison"
                    else "ontology_decision_macro_f1"
                    if column == "metric"
                    else "registered_primary"
                    if column == "analysis_family"
                    else "test"
                )
                for column in contract["required_columns"]
            }
        )

    metrics = phase4 / "tables/unit_metrics.csv"
    with metrics.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=supplemental["required_columns"],
            lineterminator="\n",
        )
        writer.writeheader()
        for metric in supplemental["metrics"]:
            for partition in supplemental["partitions"]:
                for index in range(partition["expected_rows_per_metric"]):
                    unit_id = f"{partition['value']}-{index:03d}"
                    writer.writerow(
                        {
                            "source_block": "primary",
                            "unit_id": unit_id,
                            "condition": partition["value"],
                            "world_id": f"world-{index % 12:02d}",
                            "context_id": f"context-{index % 36:02d}",
                            "seed_block": "" if partition["value"] == "C0" else "1",
                            "output_valid": "true",
                            "projection_id": f"projection-{unit_id}",
                            "unit_hash": "a" * 64,
                            "metric_name": metric["metric_name"],
                            "metric_status": "value",
                            "value": "0.5",
                            "numerator": "1",
                            "denominator": "2",
                            "metric_version_hash": "b" * 64,
                        }
                    )

    comparison_hash = hashlib.sha256(comparisons.read_bytes()).hexdigest()
    metrics_hash = hashlib.sha256(metrics.read_bytes()).hexdigest()
    manifest_payload = {
        "schema_version": "1.0.0",
        "manifest_id": "test-phase4-tables",
        "analysis_configuration_hash": "c" * 64,
        "metric_version_hash": "b" * 64,
        "independent_confirmatory_unit": "world",
        "tables": [
            {
                "table_id": "canonical_comparisons",
                "relative_path": "tables/comparisons.csv",
                "columns": contract["required_columns"],
                "row_count": 2,
                "file_sha256": comparison_hash,
                "logical_content_hash": "d" * 64,
                "release_class": "public",
                "independent_unit": "world",
                "selection_ready": False,
            },
            {
                "table_id": supplemental["producer_table_id"],
                "relative_path": "tables/unit_metrics.csv",
                "columns": supplemental["required_columns"],
                "row_count": sum(
                    item["expected_row_count"] for item in supplemental["metrics"]
                ),
                "file_sha256": metrics_hash,
                "logical_content_hash": "e" * 64,
                "release_class": "public",
                "independent_unit": "metric_observation",
                "selection_ready": True,
            },
        ],
    }
    manifest_payload["content_hash"] = canonical_sha256(manifest_payload)
    manifest = phase4 / "table_manifest.json"
    manifest.write_text(
        json.dumps(
            manifest_payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    payload = _recipe_payload()
    heldout = next(
        item for item in payload["predecessors"] if item["family"] == "heldout"
    )
    heldout.update(
        {
            "status": "complete",
            "reason": "TEST-ONLY Phase 4 sources are complete.",
            "artifacts": [
                {
                    "artifact_id": "phase4-comparisons",
                    "family": "heldout",
                    "relative_path": "artifacts/public/phase4/tables/comparisons.csv",
                    "media_type": "text/csv",
                    "release_class": "public",
                },
                {
                    "artifact_id": "phase4-table-manifest",
                    "family": "heldout",
                    "relative_path": "artifacts/public/phase4/table_manifest.json",
                    "media_type": "application/json",
                    "release_class": "public",
                },
                {
                    "artifact_id": "phase4-unit-metrics",
                    "family": "heldout",
                    "relative_path": "artifacts/public/phase4/tables/unit_metrics.csv",
                    "media_type": "text/csv",
                    "release_class": "public",
                },
            ],
        }
    )
    table = next(
        item for item in payload["tables"] if item["table_id"] == "primary_c2_vs_c1"
    )
    table.update(
        {
            "status": "complete",
            "reason": "TEST-ONLY primary and supplemental rows are complete.",
            "scope": "final",
            "source_artifact_ids": [
                "phase4-comparisons",
                "phase4-table-manifest",
                "phase4-unit-metrics",
            ],
            "source_table_artifact_id": "phase4-comparisons",
            "producer_manifest_artifact_id": "phase4-table-manifest",
            "row_filters": contract["registered_row_filters"],
            "supplemental_metric_sources": [
                {
                    "source_role": supplemental["source_role"],
                    "source_artifact_id": "phase4-unit-metrics",
                    "table_manifest_artifact_id": "phase4-table-manifest",
                }
            ],
        }
    )
    payload["public_artifact_ids"] = [
        "phase4-comparisons",
        "phase4-table-manifest",
        "phase4-unit-metrics",
    ]
    payload["recipe_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "recipe_sha256"}
    )
    recipe = root / "recipe.json"
    _write_recipe(recipe, payload)
    registry = materialize_phase7_source_registry(
        source_root=root,
        configuration_path=root / "configs/study/phase7_compiler.json",
        recipe_path=recipe,
    )
    binding = next(
        item for item in registry.tables if item.table_id == "primary_c2_vs_c1"
    )
    evidence = binding.supplemental_metric_sources[0]
    expected_metric_rows = len(supplemental["metrics"]) * 252
    assert evidence.source_row_count == expected_metric_rows
    assert evidence.selected_row_count == expected_metric_rows
    assert {item.metric_name for item in evidence.metric_row_counts} == {
        item["metric_name"] for item in supplemental["metrics"]
    }


def test_registry_recipe_rejects_escape_and_symlinked_ancestry(tmp_path: Path) -> None:
    root = _source_root(tmp_path)
    outside = tmp_path / "outside-recipe.json"
    _write_recipe(outside, _recipe_payload())
    with pytest.raises(Phase7CompilationError, match="inside the registry source root"):
        materialize_phase7_source_registry(
            source_root=root,
            configuration_path=root / "configs/study/phase7_compiler.json",
            recipe_path=outside,
        )

    real_directory = root / "real-recipes"
    real_directory.mkdir()
    linked_directory = root / "linked-recipes"
    linked_directory.symlink_to(real_directory, target_is_directory=True)
    linked_recipe = linked_directory / "recipe.json"
    _write_recipe(real_directory / "recipe.json", _recipe_payload())
    with pytest.raises(Phase7CompilationError, match="symlinked ancestor"):
        materialize_phase7_source_registry(
            source_root=root,
            configuration_path=root / "configs/study/phase7_compiler.json",
            recipe_path=linked_recipe,
        )


def test_registry_output_rejects_escape_and_symlinked_ancestry(tmp_path: Path) -> None:
    root = _source_root(tmp_path)
    recipe = root / "recipe.json"
    _write_recipe(recipe, _recipe_payload())
    with pytest.raises(Phase7CompilationError, match="inside the registry source root"):
        write_phase7_source_registry(
            source_root=root,
            configuration_path=root / "configs/study/phase7_compiler.json",
            recipe_path=recipe,
            output_path=tmp_path / "outside-registry.json",
        )

    real_directory = root / "real-output"
    real_directory.mkdir()
    linked_directory = root / "linked-output"
    linked_directory.symlink_to(real_directory, target_is_directory=True)
    with pytest.raises(Phase7CompilationError, match="symlinked ancestor"):
        write_phase7_source_registry(
            source_root=root,
            configuration_path=root / "configs/study/phase7_compiler.json",
            recipe_path=recipe,
            output_path=linked_directory / "registry.json",
        )


def test_registry_control_paths_reject_lexical_parent_traversal(tmp_path: Path) -> None:
    root = _source_root(tmp_path)
    recipe = root / "recipe.json"
    _write_recipe(recipe, _recipe_payload())
    traversing_recipe = root / "nested" / ".." / "recipe.json"
    with pytest.raises(Phase7CompilationError, match="parent traversal"):
        materialize_phase7_source_registry(
            source_root=root,
            configuration_path=root / "configs/study/phase7_compiler.json",
            recipe_path=traversing_recipe,
        )
    with pytest.raises(Phase7CompilationError, match="parent traversal"):
        write_phase7_source_registry(
            source_root=root,
            configuration_path=root / "configs/study/phase7_compiler.json",
            recipe_path=recipe,
            output_path=root / "artifacts" / ".." / "registry.json",
        )
