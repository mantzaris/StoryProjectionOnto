"""Read-only semantic replay for a completed Phase 4 output bundle.

The Phase 4 producer is intentionally append-only and records its artifacts and metric
rows in the cumulative SQLite ledger.  Calling that producer merely to verify an
existing bundle would therefore touch scientific state.  This module instead replays
the complete *output* contract from immutable files: exact inventory, physical and
logical hashes, typed records, registered world-level analysis, report gates, and every
CSV derived from a JSON/JSONL source.

Valid complete bundles retain the exact restricted projection, context, and evidence
packet needed to recompute their adapter, grounding audit, and registered score panel.
Failed ITT rows are likewise regenerated from their preserved failure lineage.  A
caller that uses this as a phase-transition gate must still compare the analysis
index's source bindings with the native held-out, combined-run, review, and
source-association records; this module does not independently reopen those ledgers.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from story_projection_onto.analysis import (
    MetricObservation,
    RarePivotalCountObservation,
    analysis_to_json,
    run_registered_analysis_from_observations,
)
from story_projection_onto.contracts import (
    ConditionName,
    ImmutableRecord,
    ReleaseClass,
    Sha256Digest,
    canonical_sha256,
)
from story_projection_onto.metrics.config import StudyMetricConfiguration
from story_projection_onto.metrics.pipeline import (
    CompleteProjectionScoreBundle,
    FailedMetricOutput,
    GeometryMetricInput,
    GroundingAuditInput,
    IntendedMetricManifest,
    IntendedUnitScore,
    ScorerMetricPlan,
    analysis_observations_from_scores,
    score_complete_projection,
    score_failed_output,
    score_intended_projection,
)
from story_projection_onto.scorer_only.phase4_analysis import (
    _ABLATION_CSV_COLUMNS,
    _COMPARISON_COLUMNS,
    _CONTRAST_METRIC_CSV_COLUMNS,
    _CROSS_SEED_COMMUNITY_METRICS,
    _CROSS_SEED_METRIC_CSV_COLUMNS,
    _GOLD_ALIGNED_COMMUNITY_METRICS,
    _PARAPHRASE_CSV_COLUMNS,
    _RARE_CSV_COLUMNS,
    _REPORT_GATE_CSV_COLUMNS,
    _UNIT_METRIC_CSV_COLUMNS,
    _WORLD_METRIC_CSV_COLUMNS,
    REGISTERED_ABLATION_PAIR_COUNT,
    REGISTERED_COMBINED_ORDINARY_SCORE_COUNT,
    REGISTERED_CONTRAST_SCORE_COUNT,
    REGISTERED_CROSS_SEED_COMMUNITY_COUNT,
    REGISTERED_PARAPHRASE_PAIR_COUNT,
    REGISTERED_PRIMARY_SCORE_COUNT,
    AblationPairResult,
    CanonicalComparisonRow,
    ContrastScoreObservation,
    CrossSeedScoreObservation,
    EligibleWorldComparison,
    ExploratoryComparison,
    ParaphrasePairResult,
    Phase4AnalysisConfiguration,
    Phase4AnalysisError,
    Phase4AnalysisIndex,
    Phase4ReportGateStatus,
    Phase4TableManifest,
    ScoredMetricObservation,
    SecondaryMultiplicityResult,
    WorldMetricRow,
    _ablation_csv_rows,
    _canonical_comparison_rows,
    _comparison_csv,
    _contrast_metric_csv_rows,
    _contrast_world_metric_rows,
    _cross_seed_metric_csv_rows,
    _cross_seed_world_metric_rows,
    _crossing_comparisons,
    _eligible_world_comparisons,
    _entropy_multiplicity,
    _exploratory_comparisons,
    _grounding_audit,
    _mapping_csv,
    _paraphrase_csv_rows,
    _rare_csv_rows,
    _report_gate_csv_row,
    _report_gate_status,
    _ScoreCollection,
    _unit_metric_csv_rows,
    _verify_registered_analysis,
    _world_metric_csv_rows,
    _world_metric_rows,
)


@dataclass(frozen=True, slots=True)
class _TableContract:
    relative_path: str
    columns: tuple[str, ...]
    release_class: ReleaseClass
    independent_unit: Literal[
        "intended_cell",
        "metric_observation",
        "world",
        "contrast_pair",
        "cross_seed_pair",
        "paraphrase_pair",
        "ablation_pair",
        "analysis",
    ]
    selection_ready: bool = False


_TABLE_CONTRACTS: Mapping[str, _TableContract] = {
    "unit_metric_observations": _TableContract(
        "tables/unit_metrics.jsonl",
        (
            "content_hash",
            "source_block",
            "unit_id",
            "condition",
            "world_id",
            "context_id",
            "seed_block",
            "output_valid",
            "metric",
        ),
        ReleaseClass.PUBLIC,
        "metric_observation",
        True,
    ),
    "primary_world_metrics": _TableContract(
        "tables/world_metrics.jsonl",
        (
            "content_hash",
            "condition",
            "world_id",
            "metric_name",
            "value",
            "context_count",
            "row_count",
            "undefined_context_count",
        ),
        ReleaseClass.PUBLIC,
        "world",
        True,
    ),
    "contrast_pair_scores": _TableContract(
        "tables/contrast_scores.jsonl",
        (
            "content_hash",
            "world_id",
            "before_context_id",
            "after_context_id",
            "condition",
            "seed_block",
            "score",
        ),
        ReleaseClass.PUBLIC,
        "contrast_pair",
        True,
    ),
    "contrast_world_metrics": _TableContract(
        "tables/contrast_world_metrics.jsonl",
        (
            "content_hash",
            "condition",
            "world_id",
            "metric_name",
            "value",
            "context_count",
            "row_count",
            "undefined_context_count",
        ),
        ReleaseClass.PUBLIC,
        "world",
    ),
    "cross_seed_community_scores": _TableContract(
        "tables/cross_seed_scores.jsonl",
        ("content_hash", "world_id", "context_id", "condition", "score"),
        ReleaseClass.PUBLIC,
        "cross_seed_pair",
    ),
    "cross_seed_world_metrics": _TableContract(
        "tables/cross_seed_world_metrics.jsonl",
        (
            "content_hash",
            "condition",
            "world_id",
            "metric_name",
            "value",
            "context_count",
            "row_count",
            "undefined_context_count",
        ),
        ReleaseClass.PUBLIC,
        "world",
    ),
    "paraphrase_pairs": _TableContract(
        "tables/paraphrase.jsonl",
        (
            "content_hash",
            "call_id",
            "world_id",
            "context_id",
            "base_score_hash",
            "paraphrase_score_hash",
            "base_projection_hash",
            "paraphrase_projection_hash",
            "status",
            "score",
        ),
        ReleaseClass.PUBLIC,
        "paraphrase_pair",
        True,
    ),
    "ablation_pairs": _TableContract(
        "tables/ablations.jsonl",
        (
            "content_hash",
            "call_id",
            "condition",
            "world_id",
            "context_id",
            "parent_score_hash",
            "ablation_score_hash",
            "principal_metric",
            "parent_value",
            "ablation_value",
            "difference",
        ),
        ReleaseClass.PUBLIC,
        "ablation_pair",
        True,
    ),
    "report_gate_status_csv": _TableContract(
        "tables/report_gate_status.csv",
        _REPORT_GATE_CSV_COLUMNS,
        ReleaseClass.PUBLIC,
        "analysis",
    ),
    "report_unit_metrics_csv": _TableContract(
        "tables/unit_metrics.csv",
        _UNIT_METRIC_CSV_COLUMNS,
        ReleaseClass.PUBLIC,
        "metric_observation",
        True,
    ),
    "report_world_metrics_csv": _TableContract(
        "tables/world_metrics.csv",
        _WORLD_METRIC_CSV_COLUMNS,
        ReleaseClass.PUBLIC,
        "world",
        True,
    ),
    "report_contrast_metrics_csv": _TableContract(
        "tables/contrast_metrics.csv",
        _CONTRAST_METRIC_CSV_COLUMNS,
        ReleaseClass.PUBLIC,
        "contrast_pair",
        True,
    ),
    "report_contrast_world_metrics_csv": _TableContract(
        "tables/contrast_world_metrics.csv",
        _WORLD_METRIC_CSV_COLUMNS,
        ReleaseClass.PUBLIC,
        "world",
    ),
    "report_cross_seed_metrics_csv": _TableContract(
        "tables/cross_seed_metrics.csv",
        _CROSS_SEED_METRIC_CSV_COLUMNS,
        ReleaseClass.PUBLIC,
        "cross_seed_pair",
    ),
    "report_cross_seed_world_metrics_csv": _TableContract(
        "tables/cross_seed_world_metrics.csv",
        _WORLD_METRIC_CSV_COLUMNS,
        ReleaseClass.PUBLIC,
        "world",
    ),
    "report_rare_pivotal_csv": _TableContract(
        "tables/rare_pivotal_observations.csv",
        _RARE_CSV_COLUMNS,
        ReleaseClass.PUBLIC,
        "metric_observation",
    ),
    "report_paraphrase_csv": _TableContract(
        "tables/paraphrase.csv",
        _PARAPHRASE_CSV_COLUMNS,
        ReleaseClass.PUBLIC,
        "paraphrase_pair",
        True,
    ),
    "report_ablations_csv": _TableContract(
        "tables/ablations.csv",
        _ABLATION_CSV_COLUMNS,
        ReleaseClass.PUBLIC,
        "ablation_pair",
        True,
    ),
    "registered_metric_observations": _TableContract(
        "tables/registered_metric_observations.jsonl",
        (
            "condition",
            "world_id",
            "context_id",
            "seed_block",
            "metric_name",
            "value",
            "output_valid",
            "gold_nonempty",
            "gold_count",
            "scorer_plan_hash",
        ),
        ReleaseClass.PUBLIC,
        "metric_observation",
    ),
    "rare_pivotal_observations": _TableContract(
        "tables/rare_pivotal_observations.jsonl",
        (
            "condition",
            "world_id",
            "context_id",
            "seed_block",
            "true_positive_count",
            "gold_count",
            "output_valid",
            "scorer_plan_hash",
        ),
        ReleaseClass.PUBLIC,
        "metric_observation",
    ),
    "secondary_paired_comparisons": _TableContract(
        "analysis/exploratory.jsonl",
        (
            "content_hash",
            "metric_name",
            "comparison",
            "status",
            "retained_world_count",
            "estimate",
        ),
        ReleaseClass.PUBLIC,
        "analysis",
    ),
    "eligible_world_comparisons": _TableContract(
        "analysis/eligible_world_comparisons.jsonl",
        (
            "content_hash",
            "panel",
            "metric_name",
            "comparison",
            "eligible_world_ids",
            "status",
            "estimate",
        ),
        ReleaseClass.PUBLIC,
        "world",
    ),
    "entropy_benjamini_hochberg": _TableContract(
        "analysis/entropy_multiplicity.jsonl",
        (
            "content_hash",
            "family",
            "metric_name",
            "comparison",
            "family_planned_size",
            "status",
            "raw_two_sided_p_value",
            "benjamini_hochberg_adjusted_p_value",
        ),
        ReleaseClass.PUBLIC,
        "analysis",
    ),
    "canonical_comparisons": _TableContract(
        "tables/comparisons.csv",
        _COMPARISON_COLUMNS,
        ReleaseClass.PUBLIC,
        "world",
    ),
    "crossing_profile_comparisons": _TableContract(
        "analysis/crossing_profiles.json",
        ("comparison_result",),
        ReleaseClass.PUBLIC,
        "analysis",
    ),
    "registered_world_level_analysis": _TableContract(
        "analysis/registered_analysis.json",
        ("registered_analysis_result",),
        ReleaseClass.PUBLIC,
        "analysis",
    ),
}

_NON_TABLE_OUTPUTS: Mapping[str, ReleaseClass] = {
    "analysis/report_gate_status.json": ReleaseClass.PUBLIC,
    "inputs/combined_intended_manifest.json": ReleaseClass.RESTRICTED,
    "inputs/geometry_inputs.jsonl": ReleaseClass.PUBLIC,
    "inputs/grounding_audits.jsonl": ReleaseClass.RESTRICTED,
    "inputs/primary_intended_manifest.json": ReleaseClass.RESTRICTED,
    "inputs/scorer_plans.jsonl": ReleaseClass.RESTRICTED,
    "scores/combined.jsonl": ReleaseClass.RESTRICTED,
    "scores/complete_bundles.jsonl": ReleaseClass.RESTRICTED,
    "scores/primary.jsonl": ReleaseClass.RESTRICTED,
}

_FIXED_ROW_COUNTS: Mapping[str, int] = {
    "analysis/crossing_profiles.json": 2,
    "analysis/registered_analysis.json": 1,
    "analysis/report_gate_status.json": 1,
    "inputs/combined_intended_manifest.json": 1,
    "inputs/primary_intended_manifest.json": 1,
    "scores/combined.jsonl": REGISTERED_COMBINED_ORDINARY_SCORE_COUNT,
    "scores/primary.jsonl": REGISTERED_PRIMARY_SCORE_COUNT,
    "table_manifest.json": 1,
    "tables/ablations.csv": REGISTERED_ABLATION_PAIR_COUNT,
    "tables/ablations.jsonl": REGISTERED_ABLATION_PAIR_COUNT,
    "tables/contrast_scores.jsonl": REGISTERED_CONTRAST_SCORE_COUNT,
    "tables/cross_seed_scores.jsonl": REGISTERED_CROSS_SEED_COMMUNITY_COUNT,
    "tables/paraphrase.csv": REGISTERED_PARAPHRASE_PAIR_COUNT,
    "tables/paraphrase.jsonl": REGISTERED_PARAPHRASE_PAIR_COUNT,
    "tables/report_gate_status.csv": 1,
}


class Phase4OutputReplayReceipt(ImmutableRecord):
    """Deterministic proof returned only after the entire output bundle replays."""

    analysis_index_hash: Sha256Digest
    analysis_configuration_hash: Sha256Digest
    metric_configuration_hash: Sha256Digest
    metric_version_hash: Sha256Digest
    table_manifest_hash: Sha256Digest
    registered_analysis_hash: Sha256Digest
    report_gate_status_hash: Sha256Digest
    output_file_count: Literal[36] = 36
    table_count: Literal[26] = 26
    primary_score_count: Literal[252] = 252
    combined_score_count: Literal[40] = 40
    contrast_score_count: Literal[84] = 84
    cross_seed_score_count: Literal[108] = 108
    paraphrase_pair_count: Literal[12] = 12
    ablation_pair_count: Literal[28] = 28
    output_inventory_hash: Sha256Digest
    all_files_canonical: Literal[True] = True
    registered_analysis_recomputed: Literal[True] = True
    report_gate_recomputed: Literal[True] = True
    intended_cell_scores_recomputed: Literal[True] = True
    valid_grounding_audits_recomputed: Literal[True] = True
    ledger_mutated: Literal[False] = False


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_object(path: Path) -> Mapping[str, Any] | Sequence[Any]:
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf") or b"\r" in raw or not raw.endswith(b"\n"):
        raise Phase4AnalysisError(f"noncanonical Phase 4 JSON bytes: {path}")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Phase4AnalysisError(f"invalid Phase 4 JSON: {path}") from error
    expected = (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    if raw != expected:
        raise Phase4AnalysisError(f"noncanonical Phase 4 JSON serialization: {path}")
    if not isinstance(value, Mapping | Sequence) or isinstance(value, str | bytes):
        raise Phase4AnalysisError(f"Phase 4 JSON has an unsupported top-level value: {path}")
    return value


def _canonical_jsonl(path: Path) -> tuple[Mapping[str, Any], ...]:
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf") or b"\r" in raw:
        raise Phase4AnalysisError(f"noncanonical Phase 4 JSONL bytes: {path}")
    if raw and not raw.endswith(b"\n"):
        raise Phase4AnalysisError(f"Phase 4 JSONL lacks its terminal LF: {path}")
    rows: list[Mapping[str, Any]] = []
    for line in raw.splitlines():
        if not line:
            raise Phase4AnalysisError(f"Phase 4 JSONL contains a blank row: {path}")
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise Phase4AnalysisError(f"invalid Phase 4 JSONL row: {path}") from error
        if not isinstance(value, Mapping):
            raise Phase4AnalysisError(f"Phase 4 JSONL row is not an object: {path}")
        expected = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        if line != expected:
            raise Phase4AnalysisError(f"noncanonical Phase 4 JSONL serialization: {path}")
        rows.append(value)
    return tuple(rows)


def _canonical_csv(path: Path, columns: tuple[str, ...]) -> tuple[Mapping[str, str], ...]:
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf") or b"\r" in raw or not raw.endswith(b"\n"):
        raise Phase4AnalysisError(f"noncanonical Phase 4 CSV bytes: {path}")
    try:
        reader = csv.DictReader(raw.decode("utf-8").splitlines())
    except UnicodeDecodeError as error:
        raise Phase4AnalysisError(f"invalid UTF-8 Phase 4 CSV: {path}") from error
    if tuple(reader.fieldnames or ()) != columns:
        raise Phase4AnalysisError(f"Phase 4 CSV column contract changed: {path}")
    rows = tuple(dict(row) for row in reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise Phase4AnalysisError(f"Phase 4 CSV contains a ragged row: {path}")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    if stream.getvalue().encode("utf-8") != raw:
        raise Phase4AnalysisError(f"Phase 4 CSV is not canonically serialized: {path}")
    return rows


def _typed_rows(
    rows: Sequence[Mapping[str, Any]], model: type[ImmutableRecord]
) -> tuple[ImmutableRecord, ...]:
    try:
        return tuple(model.model_validate(row) for row in rows)
    except Exception as error:
        raise Phase4AnalysisError(f"invalid typed Phase 4 {model.__name__} row") from error


def _assert_payload(path: Path, expected: bytes, label: str) -> None:
    if path.read_bytes() != expected:
        raise Phase4AnalysisError(f"Phase 4 {label} does not reproduce from canonical sources")


def _exact_file_inventory(output_root: Path, expected: set[str]) -> None:
    observed: set[str] = set()
    for path in output_root.rglob("*"):
        if path.is_symlink():
            raise Phase4AnalysisError(f"symlinked Phase 4 output is forbidden: {path}")
        if path.is_file():
            observed.add(path.relative_to(output_root).as_posix())
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise Phase4AnalysisError(
            "Phase 4 output inventory changed "
            f"(missing={missing[:5]!r}, extra={extra[:5]!r})"
        )


def _validate_registered_inventory_metadata(
    *,
    index: Phase4AnalysisIndex,
    table_manifest: Phase4TableManifest,
    configuration: Phase4AnalysisConfiguration,
    metric_configuration: StudyMetricConfiguration,
) -> set[str]:
    """Require the exact registered path/table contract before inspecting payloads."""

    expected_outputs = {
        *(_NON_TABLE_OUTPUTS),
        *(contract.relative_path for contract in _TABLE_CONTRACTS.values()),
        "table_manifest.json",
    }
    if len(expected_outputs) != 36 or len(_TABLE_CONTRACTS) != 26:
        raise AssertionError("internal Phase 4 inventory contract changed without a version bump")
    if (
        index.analysis_configuration_hash != configuration.content_hash
        or table_manifest.analysis_configuration_hash != configuration.content_hash
        or index.metric_configuration_hash != metric_configuration.content_hash
        or index.metric_version_hash != metric_configuration.metric_version_hash
        or table_manifest.metric_version_hash != metric_configuration.metric_version_hash
    ):
        raise Phase4AnalysisError(
            "Phase 4 index/table manifest differs from the frozen analysis or metric configuration"
        )
    output_by_path = {item.relative_path: item for item in index.output_files}
    if set(output_by_path) != expected_outputs:
        raise Phase4AnalysisError("Phase 4 analysis index lacks the exact output contract")
    table_by_id = {item.table_id: item for item in table_manifest.tables}
    if set(table_by_id) != set(_TABLE_CONTRACTS):
        raise Phase4AnalysisError("Phase 4 table manifest lacks the exact table contract")
    for table_id, contract in _TABLE_CONTRACTS.items():
        entry = table_by_id[table_id]
        output = output_by_path[contract.relative_path]
        if (
            entry.relative_path != contract.relative_path
            or entry.columns != contract.columns
            or entry.release_class is not contract.release_class
            or entry.independent_unit != contract.independent_unit
            or entry.selection_ready is not contract.selection_ready
            or entry.file_sha256 != output.file_sha256
            or entry.logical_content_hash != output.logical_content_hash
            or entry.row_count != output.row_count
        ):
            raise Phase4AnalysisError(f"Phase 4 registered table contract changed: {table_id}")
    return expected_outputs


def _recompute_intended_cell_scores(
    *,
    scores: Sequence[IntendedUnitScore],
    complete_bundles: Sequence[CompleteProjectionScoreBundle],
    scorer_plans: Sequence[ScorerMetricPlan],
    grounding_audits: Sequence[GroundingAuditInput],
    geometries: Sequence[GeometryMetricInput],
    metric_configuration: StudyMetricConfiguration,
) -> None:
    """Recompute every ITT cell and every valid complete bundle from typed inputs."""

    plan_by_hash = {item.content_hash: item for item in scorer_plans}
    bundle_by_hash = {item.content_hash: item for item in complete_bundles}
    audit_by_projection = {item.projection_hash: item for item in grounding_audits}
    geometry_by_projection = {item.projection_hash: item for item in geometries}
    if (
        len(plan_by_hash) != len(scorer_plans)
        or len(bundle_by_hash) != len(complete_bundles)
        or len(audit_by_projection) != len(grounding_audits)
        or len(geometry_by_projection) != len(geometries)
    ):
        raise Phase4AnalysisError(
            "Phase 4 replay inputs contain duplicate content/projection identities"
        )

    for persisted_score in scores:
        intended = persisted_score.intended_unit
        plan = plan_by_hash.get(intended.scorer_plan_hash)
        if plan is None:
            raise Phase4AnalysisError("Phase 4 intended cell lacks its scorer plan")
        try:
            if not persisted_score.output_valid:
                assert persisted_score.failure_kind is not None
                recomputed_score = score_failed_output(
                    FailedMetricOutput(
                        intended_unit=intended,
                        failure_kind=persisted_score.failure_kind,
                        failure_artifact_hash=persisted_score.failure_artifact_hash,
                        allocated_gpu_seconds=persisted_score.allocated_gpu_seconds,
                    ),
                    configuration=metric_configuration,
                    scorer_plan=plan,
                )
                if recomputed_score != persisted_score:
                    raise Phase4AnalysisError(
                        "Phase 4 failed-output ITT score failed deterministic recomputation"
                    )
                continue

            assert persisted_score.projection_bundle_hash is not None
            persisted_bundle = bundle_by_hash.get(
                persisted_score.projection_bundle_hash
            )
            if persisted_bundle is None:
                raise Phase4AnalysisError(
                    "Phase 4 valid cell lacks its exact complete score bundle"
                )
            replay = persisted_bundle.scorer_only_replay
            projection = replay.projection
            persisted_audit = audit_by_projection.get(projection.content_hash)
            if persisted_audit is None:
                raise Phase4AnalysisError(
                    "Phase 4 valid replay projection lacks its grounding audit"
                )
            recomputed_audit = _grounding_audit(projection, plan)
            if recomputed_audit != persisted_audit:
                raise Phase4AnalysisError(
                    "Phase 4 grounding audit failed deterministic recomputation"
                )
            geometry = geometry_by_projection.get(projection.content_hash)
            recomputed_bundle = score_complete_projection(
                projection,
                context=replay.context,
                evidence_packet=replay.evidence_packet,
                configuration=metric_configuration,
                scorer_plan=plan,
                grounding_audit=recomputed_audit,
                geometry=geometry,
            )
            recomputed_score = score_intended_projection(
                intended,
                projection,
                context=replay.context,
                evidence_packet=replay.evidence_packet,
                configuration=metric_configuration,
                scorer_plan=plan,
                grounding_audit=recomputed_audit,
                geometry=geometry,
            )
        except Phase4AnalysisError:
            raise
        except Exception as error:
            raise Phase4AnalysisError(
                "Phase 4 intended cell could not be deterministically rescored"
            ) from error
        if recomputed_bundle != persisted_bundle or recomputed_score != persisted_score:
            raise Phase4AnalysisError(
                "Phase 4 valid score/bundle failed deterministic recomputation"
            )


def replay_phase4_analysis_outputs(
    *,
    repository: Path,
    configuration_path: Path,
    output_root: Path,
) -> Phase4OutputReplayReceipt:
    """Read and recompute a complete Phase 4 output bundle without opening its ledger."""

    repository = repository.resolve(strict=True)
    configuration = Phase4AnalysisConfiguration.load(repository, configuration_path)
    metric_configuration = StudyMetricConfiguration.load(
        repository / configuration.metric_configuration_path,
        seed_manifest_path=repository / configuration.seed_manifest_path,
    )
    if output_root.is_symlink():
        raise Phase4AnalysisError("Phase 4 replay root cannot be a symlink")
    output_root = output_root.resolve(strict=True)
    if not output_root.is_dir():
        raise Phase4AnalysisError("Phase 4 replay root must be a directory")

    index_path = output_root / "analysis_index.json"
    table_manifest_path = output_root / "table_manifest.json"
    try:
        index = Phase4AnalysisIndex.model_validate(_canonical_json_object(index_path))
        table_manifest = Phase4TableManifest.model_validate(
            _canonical_json_object(table_manifest_path)
        )
    except Exception as error:
        if isinstance(error, Phase4AnalysisError):
            raise
        raise Phase4AnalysisError("invalid Phase 4 index or table manifest") from error

    expected_outputs = _validate_registered_inventory_metadata(
        index=index,
        table_manifest=table_manifest,
        configuration=configuration,
        metric_configuration=metric_configuration,
    )
    _exact_file_inventory(output_root, {"analysis_index.json", *expected_outputs})
    if (
        index.table_manifest_hash != table_manifest.content_hash
        or index.table_manifest_file_sha256 != _file_sha256(table_manifest_path)
    ):
        raise Phase4AnalysisError("Phase 4 table manifest differs from its analysis index")

    output_by_path = {item.relative_path: item for item in index.output_files}

    release_by_path = {
        **_NON_TABLE_OUTPUTS,
        **{
            contract.relative_path: contract.release_class
            for contract in _TABLE_CONTRACTS.values()
        },
        "table_manifest.json": ReleaseClass.PUBLIC,
    }
    json_rows: dict[str, tuple[Mapping[str, Any], ...]] = {}
    json_objects: dict[str, Mapping[str, Any] | Sequence[Any]] = {}
    for relative_path, output in output_by_path.items():
        path = output_root / relative_path
        if (
            output.release_class is not release_by_path[relative_path]
            or output.logical_content_hash is None
            or _file_sha256(path) != output.file_sha256
        ):
            raise Phase4AnalysisError(f"Phase 4 output metadata changed: {relative_path}")
        if relative_path.endswith(".jsonl"):
            rows = _canonical_jsonl(path)
            json_rows[relative_path] = rows
            logical_hash = canonical_sha256(rows)
            row_count = len(rows)
        elif relative_path.endswith(".json"):
            value = _canonical_json_object(path)
            json_objects[relative_path] = value
            logical_hash = canonical_sha256(value)
            row_count = len(value) if relative_path == "analysis/crossing_profiles.json" else 1
        elif relative_path.endswith(".csv"):
            table = next(
                item for item in table_manifest.tables if item.relative_path == relative_path
            )
            row_count = len(_canonical_csv(path, table.columns))
            logical_hash = output.logical_content_hash
        else:
            raise Phase4AnalysisError(f"unknown Phase 4 output media type: {relative_path}")
        if output.logical_content_hash != logical_hash or output.row_count != row_count:
            raise Phase4AnalysisError(f"Phase 4 output logical hash/count changed: {relative_path}")
        expected_count = _FIXED_ROW_COUNTS.get(relative_path)
        if expected_count is not None and row_count != expected_count:
            raise Phase4AnalysisError(f"Phase 4 registered row count changed: {relative_path}")

    table_manifest_output = output_by_path["table_manifest.json"]
    if (
        table_manifest_output.release_class is not ReleaseClass.PUBLIC
        or table_manifest_output.logical_content_hash != table_manifest.content_hash
        or table_manifest_output.row_count != 1
    ):
        raise Phase4AnalysisError("Phase 4 table-manifest output metadata changed")
    primary_manifest = IntendedMetricManifest.model_validate(
        json_objects["inputs/primary_intended_manifest.json"]
    )
    combined_manifest = IntendedMetricManifest.model_validate(
        json_objects["inputs/combined_intended_manifest.json"]
    )
    primary_scores = _typed_rows(json_rows["scores/primary.jsonl"], IntendedUnitScore)
    combined_scores = _typed_rows(json_rows["scores/combined.jsonl"], IntendedUnitScore)
    scorer_plans = _typed_rows(json_rows["inputs/scorer_plans.jsonl"], ScorerMetricPlan)
    grounding_audits = _typed_rows(
        json_rows["inputs/grounding_audits.jsonl"], GroundingAuditInput
    )
    geometries = _typed_rows(
        json_rows["inputs/geometry_inputs.jsonl"], GeometryMetricInput
    )
    complete_bundles = _typed_rows(
        json_rows["scores/complete_bundles.jsonl"], CompleteProjectionScoreBundle
    )
    if (
        tuple(score.intended_unit for score in primary_scores) != primary_manifest.units
        or tuple(score.intended_unit for score in combined_scores) != combined_manifest.units
        or index.primary_intended_manifest_hash != primary_manifest.content_hash
        or index.combined_intended_manifest_hash != combined_manifest.content_hash
    ):
        raise Phase4AnalysisError("Phase 4 intended-cell denominator differs from scored cells")
    scorer_plan_hashes = {item.content_hash for item in scorer_plans}
    if {
        score.intended_unit.scorer_plan_hash for score in (*primary_scores, *combined_scores)
    } != scorer_plan_hashes:
        raise Phase4AnalysisError("Phase 4 scorer-plan inventory differs from intended cells")
    if any(
        row.metric_version_hash != metric_configuration.metric_version_hash
        for score in (*primary_scores, *combined_scores)
        for row in score.rows
    ):
        raise Phase4AnalysisError("Phase 4 score row uses another metric version")
    expected_bundle_hashes = {
        score.projection_bundle_hash
        for score in (*primary_scores, *combined_scores)
        if score.output_valid
    }
    bundle_by_hash = {item.content_hash: item for item in complete_bundles}
    if expected_bundle_hashes != set(bundle_by_hash):
        raise Phase4AnalysisError("Phase 4 complete score-bundle inventory changed")
    for score in (*primary_scores, *combined_scores):
        if not score.output_valid:
            continue
        assert score.projection_bundle_hash is not None
        bundle = bundle_by_hash[score.projection_bundle_hash]
        if (
            bundle.metric_rows != score.rows
            or bundle.projection.adapter.projection_id != score.projection_id
        ):
            raise Phase4AnalysisError(
                "Phase 4 intended score differs from its complete metric bundle"
            )
    _recompute_intended_cell_scores(
        scores=(*primary_scores, *combined_scores),
        complete_bundles=complete_bundles,
        scorer_plans=scorer_plans,
        grounding_audits=grounding_audits,
        geometries=geometries,
        metric_configuration=metric_configuration,
    )

    unit_observations = _typed_rows(
        json_rows["tables/unit_metrics.jsonl"], ScoredMetricObservation
    )
    expected_unit_observations = tuple(
        ScoredMetricObservation(
            source_block=source_block,
            unit_id=score.intended_unit.unit_id,
            condition=score.intended_unit.condition,
            world_id=score.intended_unit.world_id,
            context_id=score.intended_unit.context_id,
            seed_block=score.intended_unit.seed_block,
            output_valid=score.output_valid,
            metric=row,
        )
        for source_block, scores in (
            ("primary", primary_scores),
            ("combined", combined_scores),
        )
        for score in scores
        for row in score.rows
    )
    if unit_observations != expected_unit_observations:
        raise Phase4AnalysisError("Phase 4 unit-metric table differs from scored cells")

    world_metrics = _typed_rows(json_rows["tables/world_metrics.jsonl"], WorldMetricRow)
    contrast = _typed_rows(
        json_rows["tables/contrast_scores.jsonl"], ContrastScoreObservation
    )
    contrast_world = _typed_rows(
        json_rows["tables/contrast_world_metrics.jsonl"], WorldMetricRow
    )
    cross_seed = _typed_rows(
        json_rows["tables/cross_seed_scores.jsonl"], CrossSeedScoreObservation
    )
    cross_seed_world = _typed_rows(
        json_rows["tables/cross_seed_world_metrics.jsonl"], WorldMetricRow
    )
    paraphrase = _typed_rows(json_rows["tables/paraphrase.jsonl"], ParaphrasePairResult)
    ablations = _typed_rows(json_rows["tables/ablations.jsonl"], AblationPairResult)
    exploratory = _typed_rows(
        json_rows["analysis/exploratory.jsonl"], ExploratoryComparison
    )
    eligible = _typed_rows(
        json_rows["analysis/eligible_world_comparisons.jsonl"], EligibleWorldComparison
    )
    entropy = _typed_rows(
        json_rows["analysis/entropy_multiplicity.jsonl"], SecondaryMultiplicityResult
    )
    primary_collection = _ScoreCollection(
        scores=primary_scores,
        bundles={},
        projections={},
        grounding_audits={},
        geometries={},
        observations=tuple(
            item for item in unit_observations if item.source_block == "primary"
        ),
    )
    recomputed_world_metrics = _world_metric_rows(primary_collection)
    recomputed_contrast_world = _contrast_world_metric_rows(contrast)
    recomputed_cross_seed_world = _cross_seed_world_metric_rows(cross_seed)
    recomputed_exploratory = _exploratory_comparisons(
        (*recomputed_world_metrics, *recomputed_contrast_world),
        bootstrap_root_seed=configuration.bootstrap_root_seed,
    )
    recomputed_entropy = _entropy_multiplicity(recomputed_exploratory)
    recomputed_eligible = (
        *_eligible_world_comparisons(
            recomputed_world_metrics,
            panel="gold_aligned_community",
            metric_names=_GOLD_ALIGNED_COMMUNITY_METRICS,
            comparator_conditions=(
                ConditionName.C1_LLM_PRE,
                ConditionName.C0_CLASSICAL_PRE,
            ),
            bootstrap_root_seed=configuration.bootstrap_root_seed,
        ),
        *_eligible_world_comparisons(
            recomputed_cross_seed_world,
            panel="cross_seed_community",
            metric_names=_CROSS_SEED_COMMUNITY_METRICS,
            comparator_conditions=(
                ConditionName.C1_LLM_PRE,
                ConditionName.A_FIXED_SELECT,
            ),
            bootstrap_root_seed=configuration.bootstrap_root_seed,
        ),
    )
    recomputed_crossings = _crossing_comparisons(
        primary_collection,
        bootstrap_root_seed=configuration.bootstrap_root_seed,
    )
    if (
        world_metrics != recomputed_world_metrics
        or contrast_world != recomputed_contrast_world
        or cross_seed_world != recomputed_cross_seed_world
        or exploratory != recomputed_exploratory
        or entropy != recomputed_entropy
        or eligible != recomputed_eligible
        or tuple(json_objects["analysis/crossing_profiles.json"]) != recomputed_crossings
    ):
        raise Phase4AnalysisError(
            "Phase 4 aggregate, secondary, or crossing analysis failed deterministic replay"
        )

    metric_observation_rows = json_rows["tables/registered_metric_observations.jsonl"]
    rare_observation_rows = json_rows["tables/rare_pivotal_observations.jsonl"]
    try:
        metric_observations = tuple(MetricObservation(**row) for row in metric_observation_rows)
        rare_observations = tuple(
            RarePivotalCountObservation(**row) for row in rare_observation_rows
        )
    except (TypeError, ValueError) as error:
        raise Phase4AnalysisError("invalid registered Phase 4 analysis observations") from error
    expected_metric_observations, expected_rare_observations = (
        analysis_observations_from_scores(
            primary_scores,
            intended_manifest=primary_manifest,
        )
    )
    if (
        metric_observations != expected_metric_observations
        or rare_observations != expected_rare_observations
    ):
        raise Phase4AnalysisError(
            "Phase 4 registered observations differ from the complete ITT scores"
        )
    registered = run_registered_analysis_from_observations(
        metric_observations,
        rare_observations,
        bootstrap_root_seed=configuration.bootstrap_root_seed,
    )
    _verify_registered_analysis(registered, configuration=configuration)
    registered_bytes = (analysis_to_json(registered) + "\n").encode("utf-8")
    _assert_payload(
        output_root / "analysis/registered_analysis.json",
        registered_bytes,
        "registered analysis",
    )
    registered_hash = canonical_sha256(asdict(registered))
    if (
        index.registered_analysis_hash != registered_hash
        or index.registered_analysis_file_sha256 != hashlib.sha256(registered_bytes).hexdigest()
    ):
        raise Phase4AnalysisError("Phase 4 registered-analysis index binding changed")

    report_gate = _report_gate_status(registered)
    gate_bytes = (report_gate.to_canonical_json() + "\n").encode("utf-8")
    _assert_payload(
        output_root / "analysis/report_gate_status.json", gate_bytes, "report gate"
    )
    if (
        Phase4ReportGateStatus.model_validate(
            json_objects["analysis/report_gate_status.json"]
        )
        != report_gate
        or index.report_gate_status_hash != report_gate.content_hash
        or index.report_gate_status_file_sha256 != hashlib.sha256(gate_bytes).hexdigest()
    ):
        raise Phase4AnalysisError("Phase 4 report-gate index binding changed")

    crossings = json_objects["analysis/crossing_profiles.json"]
    if not isinstance(crossings, Sequence) or isinstance(crossings, str | bytes):
        raise Phase4AnalysisError("Phase 4 crossing profile must be a JSON sequence")
    comparison_rows: tuple[CanonicalComparisonRow, ...] = _canonical_comparison_rows(
        registered=registered,
        exploratory=exploratory,
        entropy_multiplicity=entropy,
        eligible=eligible,
        crossings=crossings,
    )

    report_gate_rows = (_report_gate_csv_row(report_gate),)
    unit_csv_rows = _unit_metric_csv_rows(unit_observations)
    world_csv_rows = _world_metric_csv_rows(world_metrics)
    contrast_csv_rows = _contrast_metric_csv_rows(contrast)
    contrast_world_csv_rows = _world_metric_csv_rows(contrast_world)
    cross_seed_csv_rows = _cross_seed_metric_csv_rows(cross_seed)
    cross_seed_world_csv_rows = _world_metric_csv_rows(cross_seed_world)
    rare_csv_rows = _rare_csv_rows(rare_observations)
    paraphrase_csv_rows = _paraphrase_csv_rows(paraphrase)
    ablation_csv_rows = _ablation_csv_rows(ablations)
    csv_replays = {
        "tables/report_gate_status.csv": (
            _mapping_csv(_REPORT_GATE_CSV_COLUMNS, report_gate_rows),
            canonical_sha256(report_gate_rows),
        ),
        "tables/unit_metrics.csv": (
            _mapping_csv(_UNIT_METRIC_CSV_COLUMNS, unit_csv_rows),
            canonical_sha256(unit_csv_rows),
        ),
        "tables/world_metrics.csv": (
            _mapping_csv(_WORLD_METRIC_CSV_COLUMNS, world_csv_rows),
            canonical_sha256(world_csv_rows),
        ),
        "tables/contrast_metrics.csv": (
            _mapping_csv(_CONTRAST_METRIC_CSV_COLUMNS, contrast_csv_rows),
            canonical_sha256(contrast_csv_rows),
        ),
        "tables/contrast_world_metrics.csv": (
            _mapping_csv(_WORLD_METRIC_CSV_COLUMNS, contrast_world_csv_rows),
            canonical_sha256(contrast_world_csv_rows),
        ),
        "tables/cross_seed_metrics.csv": (
            _mapping_csv(_CROSS_SEED_METRIC_CSV_COLUMNS, cross_seed_csv_rows),
            canonical_sha256(cross_seed_csv_rows),
        ),
        "tables/cross_seed_world_metrics.csv": (
            _mapping_csv(_WORLD_METRIC_CSV_COLUMNS, cross_seed_world_csv_rows),
            canonical_sha256(cross_seed_world_csv_rows),
        ),
        "tables/rare_pivotal_observations.csv": (
            _mapping_csv(_RARE_CSV_COLUMNS, rare_csv_rows),
            canonical_sha256(rare_csv_rows),
        ),
        "tables/paraphrase.csv": (
            _mapping_csv(_PARAPHRASE_CSV_COLUMNS, paraphrase_csv_rows),
            canonical_sha256(paraphrase_csv_rows),
        ),
        "tables/ablations.csv": (
            _mapping_csv(_ABLATION_CSV_COLUMNS, ablation_csv_rows),
            canonical_sha256(ablation_csv_rows),
        ),
        "tables/comparisons.csv": (
            _comparison_csv(comparison_rows),
            canonical_sha256(comparison_rows),
        ),
    }
    for relative_path, (payload, logical_hash) in csv_replays.items():
        _assert_payload(output_root / relative_path, payload, relative_path)
        if output_by_path[relative_path].logical_content_hash != logical_hash:
            raise Phase4AnalysisError(
                f"Phase 4 CSV logical hash does not reproduce: {relative_path}"
            )

    inventory_hash = canonical_sha256(
        tuple(
            (
                item.relative_path,
                item.file_sha256,
                item.logical_content_hash,
                item.row_count,
                item.release_class.value,
            )
            for item in index.output_files
        )
    )
    return Phase4OutputReplayReceipt(
        analysis_index_hash=index.content_hash,
        analysis_configuration_hash=configuration.content_hash,
        metric_configuration_hash=metric_configuration.content_hash,
        metric_version_hash=metric_configuration.metric_version_hash,
        table_manifest_hash=table_manifest.content_hash,
        registered_analysis_hash=registered_hash,
        report_gate_status_hash=report_gate.content_hash,
        output_inventory_hash=inventory_hash,
    )


__all__ = ["Phase4OutputReplayReceipt", "replay_phase4_analysis_outputs"]
