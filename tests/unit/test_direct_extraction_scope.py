from pathlib import Path
from types import SimpleNamespace

import pytest

from story_projection_onto.contracts import BenchmarkSplit
from story_projection_onto.scorer_only.direct_extraction import (
    DirectReference,
    aggregate_extraction,
    compile_direct_reference,
    score_direct_preconstruction,
)
from story_projection_onto.synthetic_benchmark import compile_benchmark
from tests.unit.test_metrics_alignment import plan, predicted_assertion, predicted_nodes

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def development_reference():
    build = compile_benchmark()
    spec = next(w for w in build.world_specs if w.split is BenchmarkSplit.DEVELOPMENT)
    scorer = build.scorer_artifacts[spec.world_id]
    evidence = build.narratives[spec.world_id].evidence
    return scorer, evidence, compile_direct_reference(scorer, evidence, ROOT)


def test_complete_direct_witness_map_is_source_bound_and_includes_explicit_order(
    development_reference,
):
    scorer, evidence, ref = development_reference
    assert ref.manifest["references_compiled_without_predictions"]
    assert {w["family"] for w in ref.witness_map.values()} == {"fact", "causal", "precedence"}
    assert {e for w in ref.witness_map.values() for e in w["evidence_ids"]} == {
        e.evidence_id for e in evidence
    }
    assert len(ref.plans) == 3
    assert all(len(p.assertion_targets) == len(ref.witness_map) for p in ref.plans)
    assert compile_direct_reference(scorer, evidence, ROOT).manifest == ref.manifest
    with pytest.raises(ValueError, match="frozen shared evidence"):
        compile_direct_reference(scorer, evidence[:-1], ROOT)
    forbidden = scorer.model_copy(
        update={
            "world_spec": scorer.world_spec.model_copy(update={"split": BenchmarkSplit.HELD_OUT})
        }
    )
    with pytest.raises(ValueError, match="held-out"):
        compile_direct_reference(forbidden, evidence, ROOT)


def test_contextual_relevance_filter_is_unchanged(development_reference):
    from story_projection_onto.scorer_only.development_assessment import (
        DevelopmentScientificAssessmentProvider,
    )

    scorer, _, ref = development_reference
    for ordinal, full in enumerate(ref.plans, 1):
        contextual = DevelopmentScientificAssessmentProvider._alignment_plan(scorer, ordinal)
        relevant = {
            r.target_id for r in scorer.gold_projections[ordinal - 1].relevance if r.is_relevant
        }
        assert {t.target_id for t in contextual.assertion_targets} == {
            t.target_id for t in full.assertion_targets if t.target_id in relevant
        }
        originals = {t.target_id: t for t in full.assertion_targets}
        assert all(t == originals[t.target_id] for t in contextual.assertion_targets)


def fake_draft(predictions):
    scope = SimpleNamespace(
        story_time=SimpleNamespace(kind="point"), validity_time=SimpleNamespace(kind="interval")
    )
    return SimpleNamespace(
        content_hash="a" * 64,
        local_schema=None,
        instance_graph=SimpleNamespace(
            assertions=[
                SimpleNamespace(
                    assertion_id=p.prediction_id,
                    subject_id="s",
                    object_id="o",
                    roles=(),
                    temporal_scope=scope,
                )
                for p in predictions
            ]
        ),
    )


def test_duplicate_predictions_and_three_forms_never_multiply_credit(monkeypatch):
    from story_projection_onto.scorer_only import development_assessment as assessor

    predictions = (
        predicted_assertion(),
        predicted_assertion().model_copy(update={"prediction_id": "duplicate"}),
    )
    ref = DirectReference({"fact:1": {}}, (plan(),) * 3, {"assertion-1": "fact:1"}, {})
    monkeypatch.setattr(
        assessor, "_prediction_bundle", lambda **kw: (predicted_nodes(), predictions, ())
    )
    monkeypatch.setattr(assessor, "_cited_evidence_ids", lambda draft: ("evidence-1",))
    row = score_direct_preconstruction(
        fake_draft(predictions), ref, [SimpleNamespace(evidence_id="evidence-1")]
    )
    assert row["metric"]["true_positive_count"] == 1
    assert row["metric"]["predicted_count"] == 2
    assert row["metric"]["gold_count"] == 1
    assert row["metric"]["precision"] == 0.5
    assert len(row["unmatched_prediction_ids"]) == 1
    assert not aggregate_extraction([row] * 4)["passes"]


def test_wrong_intrinsic_validity_still_fails_strict_extraction(monkeypatch):
    from story_projection_onto.scorer_only import development_assessment as assessor

    assertion = predicted_assertion()
    bad = assertion.model_copy(
        update={
            "signature": assertion.signature.model_copy(
                update={
                    "validity_time": assertion.signature.validity_time.model_copy(update={"end": 9})
                }
            )
        }
    )
    ref = DirectReference({"fact:1": {}}, (plan(),) * 3, {"assertion-1": "fact:1"}, {})
    monkeypatch.setattr(
        assessor, "_prediction_bundle", lambda **kw: (predicted_nodes(), (bad,), ())
    )
    monkeypatch.setattr(assessor, "_cited_evidence_ids", lambda draft: ("evidence-1",))
    row = score_direct_preconstruction(
        fake_draft((bad,)), ref, [SimpleNamespace(evidence_id="evidence-1")]
    )
    assert row["metric"]["true_positive_count"] == 0
    assert row["metric"]["predicted_count"] == 1


def test_integrated_route_uses_four_preconstructions_not_projection_payloads(monkeypatch):
    from story_projection_onto.contracts import ConditionName
    from story_projection_onto.scorer_only import direct_extraction as module
    from story_projection_onto.scorer_only.development_assessment import (
        DevelopmentScientificAssessmentProvider,
    )

    units = [f"unit-{i}" for i in range(4)]
    seen = []
    monkeypatch.setattr(module, "compile_direct_reference", lambda *args: None)
    monkeypatch.setattr(
        module, "score_direct_preconstruction", lambda draft, *args: seen.append(draft)
    )
    monkeypatch.setattr(
        module,
        "aggregate_extraction",
        lambda rows: {
            "explicit_family_coverage": 0.8,
            "metric": {"precision": 0.2, "recall": 0.3},
            "valid_evidence_reference_rate": 1,
        },
    )
    preparations = {
        (u, ConditionName.C0_CLASSICAL_PRE): SimpleNamespace(
            sealed_preontology=SimpleNamespace(draft=u)
        )
        for u in units
    }
    cpu = [
        SimpleNamespace(
            receipt=SimpleNamespace(unit_id=u, condition=ConditionName.C0_CLASSICAL_PRE),
            packet=SimpleNamespace(evidence=()),
        )
        for u in units
        for _ in range(3)
    ]
    result = DevelopmentScientificAssessmentProvider._assess_c0_preconstructions(
        SimpleNamespace(root=ROOT), preparations, cpu, dict.fromkeys(units)
    )
    assert seen == units
    assert result == (0.8, 0.2, 0.3, 1)
