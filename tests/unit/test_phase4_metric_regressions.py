from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from story_projection_onto.contracts import ConditionName, canonical_sha256
from story_projection_onto.metrics.alignment import (
    NormalizedDecision,
    score_ontology_decisions,
)
from story_projection_onto.metrics.common import precision_recall_f1
from story_projection_onto.metrics.config import StudyMetricConfiguration
from story_projection_onto.metrics.pipeline import MetricResultRow, PipelineMetricStatus
from story_projection_onto.metrics.rare import RarePivotalAnnotation, score_rare_pivotal

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = ROOT / "tests/fixtures/phase4/metric_regression_cases.json"


def _case_identity(case: dict[str, Any]) -> dict[str, Any]:
    return {
        key: case[key]
        for key in (
            "case_id",
            "condition",
            "fixed_ontology_friendly",
            "output_valid",
            "inputs",
        )
    }


def _row(
    *,
    case_hash: str,
    metric_version_hash: str,
    metric_name: str,
    value: float | None,
    numerator: int | None,
    denominator: int | None,
) -> MetricResultRow:
    return MetricResultRow(
        projection_id=None,
        unit_hash=case_hash,
        metric_name=metric_name,
        metric_version_hash=metric_version_hash,
        status=(
            PipelineMetricStatus.VALUE
            if value is not None
            else PipelineMetricStatus.NOT_APPLICABLE
        ),
        value=value,
        numerator=numerator,
        denominator=denominator,
    )


def _score_case(
    case: dict[str, Any],
    configuration: StudyMetricConfiguration,
) -> tuple[str, tuple[MetricResultRow, ...]]:
    ConditionName(case["condition"])
    inputs = case["inputs"]
    invalid = not case["output_valid"]
    node = precision_recall_f1(
        **inputs["nodes"],
        invalid_semantic_output=invalid,
    )
    assertion = precision_recall_f1(
        **inputs["assertions"],
        invalid_semantic_output=invalid,
    )
    decisions = score_ontology_decisions(
        gold=tuple(
            NormalizedDecision.model_validate(item)
            for item in inputs["gold_decisions"]
        ),
        predicted=tuple(
            NormalizedDecision.model_validate(item)
            for item in inputs["predicted_decisions"]
        ),
        invalid_semantic_output=invalid,
    )
    rare = score_rare_pivotal(
        annotations=tuple(
            RarePivotalAnnotation.model_validate(item)
            for item in inputs["rare_annotations"]
        ),
        strictly_matched_assertion_target_ids=frozenset(
            inputs["strictly_matched_assertion_target_ids"]
        ),
        invalid_semantic_output=invalid,
    )
    case_hash = canonical_sha256(_case_identity(case))
    metric_version_hash = configuration.metric_version_hash
    return case_hash, (
        _row(
            case_hash=case_hash,
            metric_version_hash=metric_version_hash,
            metric_name="contextual_node_f1",
            value=node.f1,
            numerator=node.true_positive_count,
            denominator=None,
        ),
        _row(
            case_hash=case_hash,
            metric_version_hash=metric_version_hash,
            metric_name="strict_qualified_assertion_f1",
            value=assertion.f1,
            numerator=assertion.true_positive_count,
            denominator=None,
        ),
        _row(
            case_hash=case_hash,
            metric_version_hash=metric_version_hash,
            metric_name="ontology_decision_macro_f1",
            value=decisions.macro_f1,
            numerator=None,
            denominator=decisions.active_family_count,
        ),
        _row(
            case_hash=case_hash,
            metric_version_hash=metric_version_hash,
            metric_name="rare_pivotal_qualified_assertion_recall",
            value=rare.qualified_assertion_recall.value,
            numerator=rare.qualified_assertion_recall.numerator,
            denominator=rare.qualified_assertion_recall.denominator,
        ),
        _row(
            case_hash=case_hash,
            metric_version_hash=metric_version_hash,
            metric_name="rare_pivotal_support_path_survival",
            value=rare.complete_support_path_survival.value,
            numerator=rare.complete_support_path_survival.numerator,
            denominator=rare.complete_support_path_survival.denominator,
        ),
    )


def test_six_public_metric_regressions_freeze_hashes_and_rows() -> None:
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    configuration = StudyMetricConfiguration.load(ROOT / "configs/study/metrics.json")
    cases = payload["cases"]
    assert payload["suite_revision"] == "phase4-six-case-regression-v1"
    assert payload["public_synthetic_fixture"] is True
    assert len(cases) == 6
    assert sum(not item["output_valid"] for item in cases) == 1
    assert sum(item["fixed_ontology_friendly"] for item in cases) == 1
    assert {item["condition"] for item in cases} >= {"C0", "C1", "C2", "A-FixedSelect"}

    for case in cases:
        case_hash, rows = _score_case(case, configuration)
        actual_rows = [
            {
                "metric_name": row.metric_name,
                "status": row.status.value,
                "value": row.value,
                "numerator": row.numerator,
                "denominator": row.denominator,
                "content_hash": row.content_hash,
            }
            for row in rows
        ]
        assert case_hash == case["expected_case_hash"], case["case_id"]
        assert actual_rows == case["expected_metric_rows"], case["case_id"]
