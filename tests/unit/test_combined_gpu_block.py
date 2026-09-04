from __future__ import annotations

import json
from collections import Counter
from datetime import timedelta
from pathlib import Path

import pytest

from story_projection_onto.benchmark_runtime import QueryRevealArtifact
from story_projection_onto.combined_gpu_block import (
    COMBINED_CALL_COUNTS,
    AblationPrequeryLineage,
    CombinedBlockError,
    CombinedCallClass,
    CombinedQuerySource,
    CombinedRuntimeBinding,
    CombinedUpstreamGate,
    ConfigurationDelta,
    OneSwitchFingerprint,
    PrimaryC2Reference,
    PublicEvidencePacketPointer,
    assert_one_switch_only,
    compile_combined_call_manifest,
    load_combined_configuration,
    load_registered_combined_selections,
    paraphrase_base_and_variant_are_equivalent,
)
from story_projection_onto.contracts import (
    ConditionName,
    QueryContext,
    RunOutcome,
    canonical_sha256,
)
from story_projection_onto.development_adapter import (
    DEFAULT_DEVELOPMENT_CONSTRUCTION_CONFIG,
    DevelopmentConstructionConfiguration,
)
from story_projection_onto.held_out_primary import PublicStageReference

REPOSITORY = Path(__file__).resolve().parents[2]


def digest(label: str) -> str:
    return canonical_sha256({"test-only": label})


def _contexts() -> dict[str, QueryContext]:
    result = {}
    root = REPOSITORY / "data/synthetic/model_visible/query_stages"
    for path in sorted(root.glob("reveal_*/query.json")):
        reveal = QueryRevealArtifact.model_validate_json(path.read_bytes())
        context = QueryContext(
            context_id=f"ctx_{reveal.reveal_id.removeprefix('reveal_')}",
            revealed_at=reveal.revealed_at,
            **reveal.query.model_dump(
                mode="python",
                exclude={"schema_version", "content_hash"},
            ),
        )
        result[context.context_id] = context
    return result


def combined_fixture():
    configuration = load_combined_configuration(REPOSITORY)
    selections = load_registered_combined_selections(REPOSITORY, configuration)
    required_ids = {
        *(item.base_context_id for item in selections.paraphrases.pairs),
        *(item.query_id for item in selections.no_context.entries),
        *selections.eligibility.temporal_epistemic_context_ids,
        *selections.eligibility.rare_guard_context_ids,
        *(item.context_id for item in selections.feedback.scripted_episodes),
        *(item.context_id for item in selections.feedback.researcher_trace_slots),
    }
    contexts = _contexts()
    sources = {}
    for index, context_id in enumerate(sorted(required_ids), start=1):
        context = contexts[context_id]
        base_time = context.revealed_at - timedelta(seconds=20)
        unit_id = f"test-unit-{index:02d}"
        snapshot_hash = digest(f"snapshot-{context_id}")
        evidence_hash = digest(f"evidence-{context_id}")
        prequery = PublicStageReference(
            relative_path=f"test/prequery/{unit_id}",
            manifest_file_sha256=digest(f"prequery-file-{context_id}"),
            staging_manifest_hash=digest(f"prequery-stage-{context_id}"),
            stage_id=f"prequery-{index:02d}",
            stage_kind="prequery_evidence",
            evidence_artifact_hash=evidence_hash,
            snapshot_hash=snapshot_hash,
        )
        query = PublicStageReference(
            relative_path=f"test/query/{context_id}",
            manifest_file_sha256=digest(f"query-file-{context_id}"),
            staging_manifest_hash=digest(f"query-stage-{context_id}"),
            stage_id=f"query-{index:02d}",
            stage_kind="query_revealed",
            evidence_artifact_hash=evidence_hash,
            snapshot_hash=snapshot_hash,
            query_artifact_hash=digest(f"query-artifact-{context_id}"),
            query_context_hash=context.content_hash,
            horizon_hash=context.spoiler_horizon.content_hash,
            budget_hash=context.budgets.content_hash,
        )
        ablations = tuple(
            AblationPrequeryLineage(
                condition=condition,
                preparation_hash=digest(f"preparation-{condition.value}-{context_id}"),
                inventory_hash=digest(f"inventory-{condition.value}-{context_id}"),
                receipt_hash=digest(f"receipt-{condition.value}-{context_id}"),
                preparation_artifact_hash=digest(
                    f"preparation-artifact-{condition.value}-{context_id}"
                ),
                inventory_artifact_hash=digest(
                    f"inventory-artifact-{condition.value}-{context_id}"
                ),
                completed_at=base_time,
            )
            for condition in (
                ConditionName.A_NO_CONTEXT,
                ConditionName.A_NO_TEMPORAL_EPISTEMIC,
                ConditionName.A_NO_RARE_GUARD,
            )
        )
        packet_hash = digest(f"packet-{context_id}")
        sources[context_id] = CombinedQuerySource(
            execution_id="test-heldout-execution",
            held_out_call_manifest_hash=digest("heldout-calls"),
            held_out_execution_manifest_hash=digest("heldout-execution"),
            held_out_scorer_bridge_hash=digest("scorer-bridge"),
            final_reviewed_seal_hash=digest("review-seal"),
            ablation_prequery_registry_hash=digest("ablation-registry"),
            unit_id=unit_id,
            context_id=context_id,
            context=context,
            prequery_stage=prequery,
            query_stage=query,
            prequery_barrier_hash=digest("barrier"),
            prequery_barrier_sealed_at=base_time + timedelta(seconds=5),
            query_access_event_hash=digest(f"query-access-{context_id}"),
            query_accessed_at=base_time + timedelta(seconds=10),
            packet_hash=packet_hash,
            packet_artifact=PublicEvidencePacketPointer(
                artifact_hash=digest(f"packet-artifact-{context_id}"),
                logical_content_hash=packet_hash,
            ),
            c2_seed1_preparation_hash=digest(f"c2-preparation-{context_id}"),
            c2_seed1_inventory_hash=digest(f"c2-inventory-{context_id}"),
            c2_seed1_receipt_hash=digest(f"c2-receipt-{context_id}"),
            c2_seed1_preparation_artifact_hash=digest(f"c2-preparation-artifact-{context_id}"),
            c2_seed1_inventory_artifact_hash=digest(f"c2-inventory-artifact-{context_id}"),
            c2_seed1_completed_at=base_time,
            ablation_prequery_lineage=ablations,
            primary_c2=PrimaryC2Reference(
                call_id=f"primary-c2-{context_id}",
                call_spec_hash=digest(f"primary-spec-{context_id}"),
                itt_record_hash=digest(f"primary-itt-{context_id}"),
                outcome=RunOutcome.FAILED,
            ),
        )
    prompts = {
        ConditionName.C2_LLM_QUERY: digest("prompt-c2"),
        ConditionName.A_NO_CONTEXT: digest("prompt-c2"),
        ConditionName.A_NO_TEMPORAL_EPISTEMIC: digest("prompt-no-temporal"),
        ConditionName.A_NO_RARE_GUARD: digest("prompt-no-rare"),
    }
    schemas = {
        ConditionName.C2_LLM_QUERY: digest("schema-c2"),
        ConditionName.A_NO_CONTEXT: digest("schema-c2"),
        ConditionName.A_NO_TEMPORAL_EPISTEMIC: digest("schema-no-temporal"),
        ConditionName.A_NO_RARE_GUARD: digest("schema-c2"),
    }
    decodings = {
        ConditionName.C2_LLM_QUERY: digest("decoding-c2"),
        ConditionName.A_NO_CONTEXT: digest("decoding-c2"),
        ConditionName.A_NO_TEMPORAL_EPISTEMIC: digest("decoding-no-temporal"),
        ConditionName.A_NO_RARE_GUARD: digest("decoding-no-rare"),
    }
    runtime = CombinedRuntimeBinding(
        source_tree_association_hash=digest("source-tree"),
        selected_model_freeze_hash=digest("model-freeze"),
        model_manifest_hash=digest("model-manifest"),
        model_repository=configuration.selected_model_repository,
        model_revision=configuration.selected_model_revision,
        served_model_name=configuration.served_model_name,
        tokenizer_manifest_hash=digest("tokenizer"),
        launcher_configuration_hash=digest("launcher"),
        runtime_version="vllm-0.10.2",
        decoding_configuration_file_sha256=configuration.decoding_configuration_file_sha256,
        seed_manifest_hash=configuration.seed_manifest_hash,
        validator_hash=digest("validator"),
        upper_ontology_hash=DevelopmentConstructionConfiguration.load(
            REPOSITORY / DEFAULT_DEVELOPMENT_CONSTRUCTION_CONFIG
        ).upper_ontology.content_hash,
        repair_policy_hash=digest("repair-policy"),
        prompt_hashes=prompts,
        output_schema_hashes=schemas,
        decoding_manifest_hashes=decodings,
        decoding_family_hash=digest("decoding-family"),
        capability_manifest_hashes={
            condition: digest(f"capability-{condition.value}") for condition in prompts
        },
        maximum_input_tokens=10240,
        maximum_output_tokens=2048,
        repair_maximum_input_tokens=10752,
        repair_maximum_output_tokens=1536,
        structured_decoder="guided_json",
    )
    created_at = max(item.query_accessed_at for item in sources.values()) + timedelta(seconds=1)
    gate = CombinedUpstreamGate(
        development_execution_result_hash=digest("development"),
        development_gate_passed=True,
        held_out_call_manifest_hash=digest("heldout-calls"),
        held_out_execution_manifest_hash=digest("heldout-execution"),
        held_out_scorer_bridge_hash=digest("scorer-bridge"),
        held_out_itt_record_hashes=tuple(digest(f"heldout-itt-{index}") for index in range(168)),
        held_out_final_schedule_snapshot_hash=digest("final-schedule"),
        held_out_prequery_barrier_hash=digest("barrier"),
        ablation_prequery_registry_hash=digest("ablation-registry"),
        phase5_input_manifest_hash=digest("phase5-input"),
        final_reviewed_seal_hash=digest("review-seal"),
        global_accounting_id="test-global-accounting",
        actual_allocated_gpu_seconds_before_block=1000.0,
        remaining_registered_p95_seconds_before_block=10000.0,
        consumed_short_reserve_slots_before_block=0,
        verified_at=created_at,
    )
    manifest = compile_combined_call_manifest(
        configuration=configuration,
        selections=selections,
        sources_by_context_id=sources,
        runtime_binding=runtime,
        upstream_gate=gate,
        created_at=created_at,
    )
    return configuration, selections, sources, runtime, gate, manifest


def _fingerprint(**updates) -> OneSwitchFingerprint:
    values = dict(
        model_manifest_hash=digest("model"),
        model_revision="revision",
        tokenizer_hash=digest("tokenizer"),
        packet_hash=digest("packet"),
        ordered_evidence_hash=digest("ordered-evidence"),
        horizon_hash=digest("horizon"),
        upper_ontology_hash=digest("upper"),
        budgets_hash=digest("budgets"),
        seed_manifest_hash=digest("seeds"),
        seed_block=1,
        vllm_seed=1988649846,
        decoding_family_hash=digest("decoding"),
        maximum_input_tokens=10240,
        maximum_output_tokens=2048,
        repair_attempt_budget=1,
        repair_policy_hash=digest("repair"),
        validator_hash=digest("validator"),
    )
    values.update(updates)
    return OneSwitchFingerprint(**values)


def test_registered_config_and_selection_hashes_are_exact() -> None:
    configuration, selections, *_ = combined_fixture()
    assert configuration.base_call_count == 49
    assert len(selections.paraphrases.pairs) == 12
    assert len(selections.no_context.entries) == 12
    assert len(selections.eligibility.temporal_epistemic_context_ids) == 8
    assert len(selections.eligibility.rare_guard_context_ids) == 8


def test_combined_manifest_has_exact_inventory_order_and_forecast() -> None:
    *_, manifest = combined_fixture()
    assert Counter(item.call_class.value for item in manifest.calls) == Counter(
        COMBINED_CALL_COUNTS
    )
    assert tuple(item.ordinal for item in manifest.calls) == tuple(range(1, 50))
    assert manifest.load_inclusive_forecast_seconds == 4835
    assert manifest.maximum_concurrency == 1
    assert [item.call_class for item in manifest.calls[:12]] == [
        CombinedCallClass.PARAPHRASE_C2
    ] * 12
    assert [item.call_class for item in manifest.calls[12:21]] == [
        CombinedCallClass.SCRIPTED_FEEDBACK_C2
    ] * 6 + [CombinedCallClass.RESEARCHER_TRACE_C2] * 3


def test_paraphrases_change_only_wording_and_no_context_hides_structured_fields() -> None:
    *_, manifest = combined_fixture()
    for call in manifest.calls[:12]:
        assert call.paraphrase_context is not None
        assert paraphrase_base_and_variant_are_equivalent(
            call.source.context, call.paraphrase_context
        )
    no_context = next(
        item for item in manifest.calls if item.call_class is CombinedCallClass.ABLATION_NO_CONTEXT
    )
    payload = no_context.model_visible_context().model_dump(mode="json")
    assert set(payload) == {"schema_version", "content_hash", "generic_request", "budgets"}
    assert not {"wording", "lens", "target", "spoiler_horizon"}.intersection(payload)


def test_ablation_preparations_are_condition_matching_and_rare_worlds_distinct() -> None:
    *_, manifest = combined_fixture()
    rare = [
        item
        for item in manifest.calls
        if item.call_class is CombinedCallClass.ABLATION_NO_RARE_GUARD
    ]
    assert len({item.source.prequery_stage.snapshot_hash for item in rare}) == 8
    for call in manifest.calls[21:]:
        lineage = call.source.preparation_for(call.condition)
        assert lineage.condition is call.condition
        assert lineage.completed_at < call.source.prequery_barrier_sealed_at


def test_one_switch_audit_rejects_any_second_delta() -> None:
    baseline = _fingerprint()
    ablated = _fingerprint(context_payload_mode="generic")
    delta = ConfigurationDelta(
        switch_name="context_payload_mode",
        baseline_value="structured",
        ablated_value="generic",
    )
    assert_one_switch_only(baseline, ablated, delta)
    changed_two = _fingerprint(
        context_payload_mode="generic",
        packet_hash=digest("different-packet"),
    )
    with pytest.raises(CombinedBlockError, match="one-switch"):
        assert_one_switch_only(baseline, changed_two, delta)


def test_runtime_binding_requires_distinct_no_temporal_grammar() -> None:
    *_, runtime, _gate, _manifest = combined_fixture()
    payload = runtime.model_dump(mode="python", exclude={"content_hash"})
    payload["output_schema_hashes"][ConditionName.A_NO_TEMPORAL_EPISTEMIC] = payload[
        "output_schema_hashes"
    ][ConditionName.C2_LLM_QUERY]
    with pytest.raises(ValueError, match="distinct output grammar"):
        CombinedRuntimeBinding.model_validate(payload)


def test_tracked_selection_tamper_fails_closed(tmp_path: Path) -> None:
    configuration = load_combined_configuration(REPOSITORY)
    config_payload = json.loads(configuration.to_canonical_json())
    config_payload.pop("content_hash")
    benchmark = REPOSITORY / configuration.benchmark_manifest_path
    copied = tmp_path / configuration.benchmark_manifest_path
    copied.parent.mkdir(parents=True)
    copied.write_bytes(benchmark.read_bytes() + b"\n")
    for binding in configuration.tracked_bindings()[1:]:
        source = REPOSITORY / binding.relative_path
        target = tmp_path / binding.relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    config_path = tmp_path / "configs/study/combined_gpu_block.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config_payload))
    with pytest.raises(CombinedBlockError, match="bytes changed"):
        load_combined_configuration(tmp_path)
