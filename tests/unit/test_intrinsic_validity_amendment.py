"""Development-only semantic amendment regressions, not model experiments."""

from types import SimpleNamespace

import pytest

from story_projection_onto.conditions.c0 import _story_and_validity, _time_compatibility
from story_projection_onto.contracts import (
    BenchmarkSplit,
    ConditionName,
    StoryTime,
    TemporalKind,
    ValidityTime,
    VisualizationTemporalFilter,
)
from story_projection_onto.synthetic_benchmark import (
    CompilerPolicy,
    _evidence_event_occurrence,
    _fact_signature,
    _temporal_scope,
    compile_benchmark,
    compile_gold_projection,
)
from story_projection_onto.temporal import query_time_visibility
from story_projection_onto.ui import _assertion_passes_temporal_filter


@pytest.fixture(scope="module")
def development():
    build = compile_benchmark()
    spec = next(w for w in build.world_specs if w.split is BenchmarkSplit.DEVELOPMENT)
    return build, spec, build.narratives[spec.world_id]


def fact_evidence(narrative, fact):
    identifier = narrative.fact_evidence_ids[fact.fact_id][0]
    return next(e for e in narrative.evidence if e.evidence_id == identifier)


def test_query_window_never_changes_intrinsic_qualification(development):
    build, spec, narrative = development
    context = build.contexts_by_world[spec.world_id][0]
    scopes = (StoryTime(kind="point", point=3), StoryTime(kind="interval", start=2, end=9))
    products = [
        compile_gold_projection(
            spec,
            context.model_copy(update={"story_scope": scope}),
            0,
            narrative,
            build.configuration,
            policy=CompilerPolicy.QUERY_DEPENDENT,
            selected_for_review=False,
        )
        for scope in scopes
    ]
    left, right = products
    assert {a.assertion_id: a.temporal_scope for a in left.projection.qualified_assertions} == {
        a.assertion_id: a.temporal_scope for a in right.projection.qualified_assertions
    }
    assert next(
        a.signature for a in left.semantic_atoms if a.slot_key == "qualification/story-scope"
    ) == next(
        a.signature for a in right.semantic_atoms if a.slot_key == "qualification/story-scope"
    )
    assert left.projection.relevance != right.projection.relevance


def test_latent_endpoints_cannot_override_supplied_evidence(development):
    _, spec, narrative = development
    for fact in spec.facts:
        if not narrative.fact_evidence_ids.get(fact.fact_id):
            continue
        evidence = fact_evidence(narrative, fact)
        changed = fact.model_copy(update={"validity_end": fact.story_position + 100})
        assert _temporal_scope(fact, evidence, 1) == _temporal_scope(changed, evidence, 1)
        assert _fact_signature(fact) == _fact_signature(changed)
        if fact.relation != "holds_office" and fact.event_ref is None:
            validity = _temporal_scope(fact, evidence, 1).validity_time
            assert validity.kind is TemporalKind.UNKNOWN
            assert validity.start is None and validity.end is None


def test_exact_duration_requires_text_and_event_onset_is_not_observation(development):
    _, spec, narrative = development
    office = next(f for f in spec.facts if f.relation == "holds_office")
    evidence = fact_evidence(narrative, office)
    assert _temporal_scope(office, evidence, 1).validity_time.kind is TemporalKind.INTERVAL
    no_duration = evidence.model_copy(
        update={"text": evidence.text.split(" This relation held")[0]}
    )
    assert _temporal_scope(office, no_duration, 1).validity_time.kind is TemporalKind.UNKNOWN
    story, validity = _story_and_validity(no_duration, (), state_like=True)
    assert story.point == office.story_position
    assert validity.kind is TemporalKind.UNKNOWN
    for event in spec.events:
        occurrence = _evidence_event_occurrence(event, narrative.evidence)
        assert occurrence.kind is TemporalKind.INTERVAL
        assert occurrence == _evidence_event_occurrence(
            event.model_copy(update={"duration": event.duration + 10}), narrative.evidence
        )
        assert _evidence_event_occurrence(event, ()).kind is TemporalKind.UNKNOWN


@pytest.mark.parametrize(
    "condition",
    [
        ConditionName.C0_CLASSICAL_PRE,
        ConditionName.C1_LLM_PRE,
        ConditionName.C2_LLM_QUERY,
        ConditionName.A_FIXED_SELECT,
    ],
)
@pytest.mark.parametrize("query_point,visible", [(0, False), (3, True), (5, False)])
def test_all_conditions_share_nonmutating_visibility(condition, query_point, visible, development):
    _, spec, narrative = development
    fact = next(f for f in spec.facts if f.relation == "holds_office" and f.validity_end == 4)
    scope = _temporal_scope(fact, fact_evidence(narrative, fact), 1)
    assertion = SimpleNamespace(temporal_scope=scope, condition=condition)
    before = scope.content_hash
    query = StoryTime(kind="point", point=query_point)
    assert query_time_visibility(scope.story_time, scope.validity_time, query) is visible
    assert (
        _assertion_passes_temporal_filter(assertion, VisualizationTemporalFilter(story_scope=query))
        is visible
    )
    assert _time_compatibility(assertion, query) == (1.0 if visible else -1.75)
    assert scope.content_hash == before


def test_unknown_and_not_applicable_remain_distinct():
    story = StoryTime(kind="point", point=1)
    query = StoryTime(kind="point", point=5)
    assert (
        query_time_visibility(story, ValidityTime(kind="unknown", reason="unstated"), query) is None
    )
    assert query_time_visibility(story, ValidityTime(kind="not_applicable"), query) is False


@pytest.mark.parametrize(
    "wrong",
    [
        {"kind": "interval", "start": 3, "end": 4},  # viewport-clipped onset
        {"kind": "interval", "start": 2, "end": 9},  # invented endpoint
        {"kind": "unknown"},
        {"kind": "not_applicable"},
    ],
)
def test_strict_matching_rejects_wrong_essential_validity(wrong):
    from story_projection_onto.metrics.alignment import TemporalExtentSignature, score_alignment
    from tests.unit.test_metrics_alignment import (
        plan,
        predicted_assertion,
        predicted_nodes,
        world_signature,
    )

    signature = type(world_signature()).model_validate(
        world_signature().model_dump(exclude={"content_hash"})
        | {"validity_time": TemporalExtentSignature(**wrong)}
    )
    result = score_alignment(
        plan=plan(),
        predicted_nodes=predicted_nodes(),
        predicted_assertions=(predicted_assertion(signature=signature),),
    )
    assert result.strict_assertion_score.f1 == 0
    assert result.essential_temporal_accuracy.value == 0
