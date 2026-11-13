from __future__ import annotations

from collections import Counter

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from story_projection_onto.benchmark import (
    CompilerPolicy,
    HorizonBlock,
    NeutralFactRole,
    build_mutation_manifest,
    compile_benchmark,
    compile_gold_projection,
    derive_seed,
    mutate_source_fact,
    semantic_signatures_for_horizon,
)
from story_projection_onto.benchmark_runtime import GoldFirewallError, scan_model_payload

BENCHMARK = compile_benchmark()


def test_mutation_manifest_replays_full_regeneration_and_exact_deltas() -> None:
    development = tuple(item for item in BENCHMARK.world_specs if item.split.value == "development")
    replayed = build_mutation_manifest(
        development,
        BENCHMARK.contexts_by_world,
        BENCHMARK.narratives,
        BENCHMARK.scorer_artifacts,
        BENCHMARK.configuration,
        BENCHMARK.seeds,
    )
    assert replayed.content_hash == BENCHMARK.mutation_manifest.content_hash
    assert all(item.removed_signatures for item in replayed.expectations)
    assert all(item.added_signatures for item in replayed.expectations)
    assert all(
        item.before_regenerated_bundle_hash != item.after_regenerated_bundle_hash
        for item in replayed.rare_deletion_proofs
    )


@settings(max_examples=12, deadline=None)
@given(st.sampled_from(["quietly_delays", "privately_prevents", "silently_enables"]))
def test_single_source_fact_mutation_has_exact_one_for_one_delta(
    replacement_relation: str,
) -> None:
    build = BENCHMARK
    spec = next(item for item in build.world_specs if item.world_id == "syn-dev-01")
    fact = next(item for item in spec.facts if item.role is NeutralFactRole.RARE_PIVOTAL)
    before = semantic_signatures_for_horizon(spec, HorizonBlock.FINAL)
    after = semantic_signatures_for_horizon(
        mutate_source_fact(spec, fact.fact_id, replacement_relation=replacement_relation),
        HorizonBlock.FINAL,
    )
    assert len(before - after) == 1
    assert len(after - before) == 1
    assert len(before & after) == len(before) - 1


@settings(max_examples=20, deadline=None)
@given(
    st.integers(min_value=0, max_value=2**31 - 1),
    st.sampled_from(["world", "query", "paraphrase", "review", "ablation"]),
)
def test_root_seed_derivation_is_deterministic(root_seed: int, namespace: str) -> None:
    assert derive_seed(root_seed, namespace) == derive_seed(root_seed, namespace)
    assert 0 <= derive_seed(root_seed, namespace) < 2**63


@settings(max_examples=12, deadline=None)
@given(st.sampled_from(["syn-test-01", "syn-test-05", "syn-test-09"]))
def test_community_semantics_are_invariant_to_persona_input_order(world_id: str) -> None:
    build = BENCHMARK
    scorer = build.scorer_artifacts[world_id]
    spec = scorer.world_spec
    values = {
        key: value for key, value in spec.model_dump(mode="python").items() if key != "content_hash"
    }
    values["personas"] = list(reversed(values["personas"]))
    reordered_spec = type(spec).model_validate(values)
    context = build.contexts_by_world[world_id][0]
    reordered = compile_gold_projection(
        reordered_spec,
        context,
        0,
        build.narratives[world_id],
        build.configuration,
        policy=CompilerPolicy.QUERY_DEPENDENT,
        selected_for_review=False,
    )
    original_rationales = scorer.community_rationales_by_query[context.context_id]
    assert Counter(
        (item.community_signature, item.semantic_basis) for item in original_rationales
    ) == Counter(
        (item.community_signature, item.semantic_basis) for item in reordered.community_rationales
    )


@settings(max_examples=12, deadline=None)
@given(
    st.sampled_from(
        [
            "syn-test-01",
            "syn_dev_03",
            "held-out",
            "development",
            "scorer_only",
            "gold.expected",
        ]
    )
)
def test_value_level_firewall_rejects_reserved_metadata_in_any_field(value: str) -> None:
    with pytest.raises(GoldFirewallError):
        scan_model_payload({"neutral": {"nested": [value]}})
