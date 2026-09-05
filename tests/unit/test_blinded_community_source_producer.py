from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from story_projection_onto.contracts import ConditionName, ReleaseClass, canonical_sha256
from story_projection_onto.metrics.community import LeidenPartition
from story_projection_onto.metrics.pipeline import (
    IntendedMetricManifest,
    IntendedMetricUnit,
    IntendedUnitScore,
    MetricResultRow,
    OutputFailureKind,
    PipelineMetricStatus,
)
from story_projection_onto.scorer_only import blinded_postrun_review as review
from story_projection_onto.scorer_only.blinded_postrun_review import (
    FROZEN_COMMUNITY_REVIEW_SELECTION_RULE_HASH,
    BlindedReviewError,
    materialize_community_review_source,
    prepare_community_review_source,
)
from story_projection_onto.scorer_only.phase4_analysis import (
    Phase4AnalysisIndex,
    Phase4OutputFile,
)

ROOT = Path(__file__).resolve().parents[2]
NOW = datetime(2026, 1, 2, tzinfo=UTC)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _jsonl_payload(values: list[dict[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n").encode("utf-8")
        for value in values
    )


def _enum(value: str) -> SimpleNamespace:
    return SimpleNamespace(value=value)


def _time(point: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        kind=_enum("point"),
        point=point,
        start=None,
        end=None,
        label=f"Story step {point}",
        anchor_id=None,
        relation=None,
        partial_order=(),
        reason=None,
    )


def _semantic_replay(
    *,
    intended: IntendedMetricUnit,
    projection_id: str,
    projection_hash: str,
    node_ids: tuple[str, str],
    assertion_id: str,
    evidence_id: str,
    mention_id: str,
    native_relation: str,
) -> SimpleNamespace:
    provenance = SimpleNamespace(
        provenance_id=f"private-provenance-{intended.condition.value}",
        evidence_id=evidence_id,
        extraction_method="deterministic synthetic annotation",
        locator=f"private-locator-{intended.condition.value}",
        source_artifact_hash=_digest(f"private-source-{intended.condition.value}"),
        confidence=1.0,
    )
    mention = SimpleNamespace(
        candidate_id=mention_id,
        surface="Harbor keeper",
    )
    evidence = SimpleNamespace(
        evidence_id=evidence_id,
        passage_id=f"private-passage-{intended.condition.value}",
        text="A harbor keeper opened the gate before the courier crossed.",
        text_hash=_digest("neutral evidence text"),
        discourse_position=SimpleNamespace(
            passage_order=1,
            sentence_order=0,
            token_order=0,
        ),
        mention_candidates=(mention,),
        event_candidates=(),
        relation_phrase_candidates=(
            SimpleNamespace(
                candidate_id=f"private-relation-candidate-{intended.condition.value}",
                surface_phrase="opened the gate for",
            ),
        ),
        temporal_clues=(
            SimpleNamespace(
                clue_id=f"private-temporal-clue-{intended.condition.value}",
                normalized_expression="before crossing",
                relation=_enum("before"),
            ),
        ),
        provenance=provenance,
        confidence=1.0,
    )
    packet = SimpleNamespace(
        packet_id=f"private-packet-{intended.condition.value}",
        snapshot_hash=intended.snapshot_hash,
        content_hash=intended.packet_hash,
        ordered_evidence_ids=(evidence_id,),
        evidence=(evidence,),
    )
    story_time = _time()
    context = SimpleNamespace(
        content_hash=intended.context_hash,
        context_id=intended.context_id,
        wording="Which roles connect the gate opening to the crossing?",
        lens="causal sequence",
        target="the gate opening and crossing",
        story_scope=story_time,
        spoiler_horizon=SimpleNamespace(
            horizon_id="private-horizon",
            max_discourse_position=SimpleNamespace(
                passage_order=3,
                sentence_order=0,
                token_order=0,
            ),
            max_revelation_position=SimpleNamespace(revelation_order=3),
        ),
        viewpoint=None,
        abstraction=_enum("event_role"),
    )
    type_id = f"private-type-{intended.condition.value}"
    local_type = SimpleNamespace(
        type_id=type_id,
        label="Gate participant",
        definition="A participant supported in the gate-crossing sequence.",
        abstraction=_enum("event_role"),
    )
    predicate = SimpleNamespace(
        predicate_id=native_relation,
        label="enabled crossing for",
        definition="The source supplied an evidenced prerequisite for the target.",
    )
    entities = tuple(
        SimpleNamespace(
            entity_id=node_id,
            label=label,
            contextual_type_id=type_id,
            contextual_role=role,
            abstraction=_enum("event_role"),
            temporal_state=story_time,
            uncertainty=_enum("known"),
            confidence=0.95,
            description=description,
            description_assertion_ids=(assertion_id,),
            evidence_ids=(evidence_id,),
            supported_mention_candidate_ids=(mention_id,),
        )
        for node_id, label, role, description in (
            (
                node_ids[0],
                "Harbor keeper",
                "enabler",
                "The keeper enables passage through the gate.",
            ),
            (
                node_ids[1],
                "Courier",
                "beneficiary",
                "The courier crosses after the gate opens.",
            ),
        )
    )
    assertion = SimpleNamespace(
        assertion_id=assertion_id,
        predicate_id=native_relation,
        subject_id=node_ids[0],
        object_id=node_ids[1],
        roles=(),
        direction="forward",
        temporal_scope=SimpleNamespace(
            story_time=story_time,
            validity_time=SimpleNamespace(
                **{**vars(story_time), "label": "Valid during the crossing"}
            ),
            discourse_position=evidence.discourse_position,
            revelation_position=SimpleNamespace(
                revelation_order=1,
                label="Gate opening disclosed",
            ),
        ),
        epistemic_scope=None,
        narrative_commitment=_enum("world_committed"),
        confidence=0.95,
        contextual_relevance=1.0,
        evidence_ids=(evidence_id,),
        provenance=(provenance,),
        why_matters="Opening the gate makes the later crossing possible.",
        why_matters_evidence_ids=(evidence_id,),
    )
    projection = SimpleNamespace(
        content_hash=projection_hash,
        projection_id=projection_id,
        condition=intended.condition,
        snapshot_hash=intended.snapshot_hash,
        packet_hash=intended.packet_hash,
        context_hash=intended.context_hash,
        query_access_event_hash=_digest(f"query-access-{intended.condition.value}"),
        generation_lineage_hash=_digest(f"generation-{intended.condition.value}"),
        raw_output_artifact_hash=_digest(f"raw-{intended.condition.value}"),
        normalized_draft_hash=_digest(f"draft-{intended.condition.value}"),
        validation_bundle_hash=_digest(f"validation-{intended.condition.value}"),
        upper_ontology=SimpleNamespace(ontology_id="private-upper-ontology"),
        local_schema=SimpleNamespace(
            schema_id=f"private-schema-{intended.condition.value}",
            contextual_types=(local_type,),
            predicates=(predicate,),
        ),
        instance_graph=SimpleNamespace(
            entities=entities,
            events=(),
            proposition_contents=(),
            assertions=(assertion,),
        ),
        decisions=(),
        run_id=f"private-run-{intended.condition.value}",
    )
    return SimpleNamespace(
        content_hash=_digest(f"replay-{intended.unit_id}"),
        projection=projection,
        context=context,
        evidence_packet=packet,
    )


def test_candidate_identity_scan_is_strict_across_known_candidate_schemas() -> None:
    assert (
        review._evidence_candidate_identifier(SimpleNamespace(candidate_id="candidate-1"))
        == "candidate-1"
    )
    assert (
        review._evidence_candidate_identifier(SimpleNamespace(mention_id="mention-1"))
        == "mention-1"
    )
    with pytest.raises(BlindedReviewError, match="unbound evidence candidate"):
        review._evidence_candidate_identifier(SimpleNamespace(surface="missing identity"))
    with pytest.raises(BlindedReviewError, match="unbound evidence candidate"):
        review._evidence_candidate_identifier(
            SimpleNamespace(candidate_id="candidate-1", event_id="event-2")
        )
    assert review._temporal_clue_identifier(SimpleNamespace(clue_id="clue-1")) == (
        "clue-1"
    )
    assert review._temporal_clue_identifier(
        SimpleNamespace(temporal_clue_id="clue-2")
    ) == "clue-2"
    with pytest.raises(BlindedReviewError, match="unbound temporal clue"):
        review._temporal_clue_identifier(SimpleNamespace(normalized_expression="before"))
    with pytest.raises(BlindedReviewError, match="unbound temporal clue"):
        review._temporal_clue_identifier(
            SimpleNamespace(clue_id="clue-1", temporal_clue_id="clue-2")
        )


def _install_complete_phase4_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    eligible_positions: frozenset[tuple[int, int]] = frozenset({(1, 1)}),
    make_frozen_candidate_unavailable: bool = False,
) -> tuple[Path, dict[str, Any]]:
    phase4 = tmp_path / "phase4"
    (phase4 / "inputs").mkdir(parents=True)
    (phase4 / "scores").mkdir()

    metric_version_hash = _digest("metric-version")
    units: list[IntendedMetricUnit] = []
    scores_by_unit: dict[str, IntendedUnitScore] = {}
    plans_by_hash: dict[str, SimpleNamespace] = {}
    bundles: list[SimpleNamespace] = []
    adapters_by_projection: dict[str, SimpleNamespace] = {}
    partitions_by_key: dict[tuple[tuple[str, ...], float, int], LeidenPartition] = {}
    valid_keys = {
        (ConditionName.C0_CLASSICAL_PRE, None),
        (ConditionName.C1_LLM_PRE, 1),
        (ConditionName.C2_LLM_QUERY, 1),
        (ConditionName.A_FIXED_SELECT, 1),
    }
    conditions = (
        (ConditionName.C0_CLASSICAL_PRE, (None,)),
        (ConditionName.C1_LLM_PRE, (1, 2)),
        (ConditionName.C2_LLM_QUERY, (1, 2)),
        (ConditionName.A_FIXED_SELECT, (1, 2)),
    )
    ranked_positions = {
        position: review._community_selection_rank(
            canonical_sha256(
                {
                    "world_id": f"secret-world-{position[0]:02d}",
                    "context_id": f"secret-context-{position[0]:02d}-{position[1]}",
                    "snapshot_hash": _digest(
                        f"snapshot-{position[0]}-{position[1]}"
                    ),
                    "packet_hash": _digest(f"packet-{position[0]}-{position[1]}"),
                    "context_hash": _digest(f"context-{position[0]}-{position[1]}"),
                    "scorer_plan_hash": _digest(
                        f"plan-{position[0]}-{position[1]}"
                    ),
                }
            )
        )
        for position in eligible_positions
    }
    frozen_position = min(ranked_positions, key=ranked_positions.__getitem__)
    available_positions = set(eligible_positions)
    if make_frozen_candidate_unavailable:
        available_positions.remove(frozen_position)
    for world in range(1, 13):
        for context in range(1, 4):
            plan_hash = _digest(f"plan-{world}-{context}")
            plans_by_hash[plan_hash] = SimpleNamespace(
                content_hash=plan_hash,
                community_eligible=(world, context) in eligible_positions,
            )
            for condition, seeds in conditions:
                for seed in seeds:
                    unit_id = (
                        f"primary:unit-{world:02d}:stage-{context}:"
                        f"{condition.value}:s{seed or 0}"
                    )
                    intended = IntendedMetricUnit(
                        unit_id=unit_id,
                        job_id=f"job-{world:02d}-{context}-{condition.value}-s{seed or 0}",
                        condition=condition,
                        world_id=f"secret-world-{world:02d}",
                        context_id=f"secret-context-{world:02d}-{context}",
                        seed_block=seed,
                        snapshot_hash=_digest(f"snapshot-{world}-{context}"),
                        packet_hash=_digest(f"packet-{world}-{context}"),
                        context_hash=_digest(f"context-{world}-{context}"),
                        scorer_plan_hash=plan_hash,
                        relevant_node_gold_count=2,
                        strict_assertion_gold_count=1,
                        ontology_decision_gold_count=1,
                        rare_pivotal_gold_count=0,
                    )
                    units.append(intended)
                    is_available = (world, context) in available_positions and (
                        condition,
                        seed,
                    ) in valid_keys
                    projection_id = f"secret-projection-{condition.value}-s{seed or 0}"
                    row = MetricResultRow(
                        projection_id=projection_id if is_available else None,
                        unit_hash=intended.content_hash,
                        metric_name="strict_qualified_assertion_f1",
                        metric_version_hash=metric_version_hash,
                        status=(
                            PipelineMetricStatus.VALUE
                            if is_available
                            else PipelineMetricStatus.INVALID
                        ),
                        value=1.0 if is_available else None,
                    )
                    if not is_available:
                        scores_by_unit[unit_id] = IntendedUnitScore(
                            intended_unit=intended,
                            output_valid=False,
                            failure_kind=OutputFailureKind.VALIDATION_INVALID,
                            rows=(row,),
                        )
                        continue

                    bundle_hash = _digest(f"bundle-{unit_id}")
                    projection_hash = _digest(f"projection-hash-{unit_id}")
                    node_ids = (
                        f"secret-node-{condition.value}-a",
                        f"secret-node-{condition.value}-b",
                    )
                    partitions = tuple(
                        LeidenPartition(
                            resolution=resolution,
                            seed=17,
                            assignments=(
                                (node_ids[0], f"secret-cluster-{role}-a"),
                                (node_ids[1], f"secret-cluster-{role}-b"),
                            ),
                            cluster_count=2,
                            modularity=0.0,
                        )
                        for role, resolution in (
                            ("half", 0.25),
                            ("base", 0.5),
                            ("double", 1.0),
                        )
                    )
                    assertion_id = f"secret-assertion-{condition.value}"
                    evidence_id = f"secret-evidence-{condition.value}"
                    native_relation = f"secret-native-relation-{condition.value}"
                    adapter = SimpleNamespace(
                        projection_id=projection_id,
                        projection_hash=projection_hash,
                        structurally_valid=True,
                        node_ids=node_ids,
                        edges=(
                            SimpleNamespace(
                                edge_id=f"secret-edge-{condition.value}",
                                source_id=node_ids[0],
                                target_id=node_ids[1],
                                native_predicate_id=native_relation,
                                canonical_predicate_id="causal",
                            ),
                        ),
                        assertion_relations=(
                            SimpleNamespace(
                                assertion_id=assertion_id,
                                native_predicate_id=native_relation,
                                canonical_predicate_id="causal",
                            ),
                        ),
                        object_anchors=(
                            (node_ids[0], (evidence_id,)),
                            (node_ids[1], (f"secret-mention-{condition.value}",)),
                        ),
                        assertion_semantics=(
                            SimpleNamespace(
                                assertion_id=assertion_id,
                                anchor_ids=(evidence_id,),
                                evidence_ids=(evidence_id,),
                                semantic_signature=_digest(f"semantics-{condition.value}"),
                            ),
                        ),
                    )
                    structural = SimpleNamespace(
                        valid_content_bearing=True,
                        leiden_half=partitions[0],
                        leiden_base=partitions[1],
                        leiden_double=partitions[2],
                    )
                    replay = _semantic_replay(
                        intended=intended,
                        projection_id=projection_id,
                        projection_hash=projection_hash,
                        node_ids=node_ids,
                        assertion_id=assertion_id,
                        evidence_id=evidence_id,
                        mention_id=f"secret-mention-{condition.value}",
                        native_relation=native_relation,
                    )
                    bundle = SimpleNamespace(
                        content_hash=bundle_hash,
                        metric_rows=(row,),
                        community=SimpleNamespace(),
                        projection=SimpleNamespace(adapter=adapter, structural=structural),
                        scorer_only_replay=replay,
                    )
                    bundles.append(bundle)
                    adapters_by_projection[projection_hash] = adapter
                    for partition in partitions:
                        partitions_by_key[
                            (node_ids, partition.resolution, partition.seed)
                        ] = partition
                    scores_by_unit[unit_id] = IntendedUnitScore(
                        intended_unit=intended,
                        output_valid=True,
                        projection_id=projection_id,
                        projection_bundle_hash=bundle_hash,
                        rows=(row,),
                    )

    ordered_units = tuple(sorted(units, key=lambda item: item.unit_id))
    intended_manifest = IntendedMetricManifest(
        manifest_id="primary-intended-test",
        units=ordered_units,
    )
    scores = tuple(scores_by_unit[item.unit_id] for item in ordered_units)
    score_payload = b"".join(
        (item.to_canonical_json() + "\n").encode("utf-8") for item in scores
    )
    bundle_payload = _jsonl_payload(
        [{"fixture_bundle_hash": item.content_hash} for item in bundles]
    )
    plans = tuple(plans_by_hash[key] for key in sorted(plans_by_hash))
    plan_payload = _jsonl_payload(
        [{"fixture_plan_hash": item.content_hash} for item in plans]
    )
    intended_payload = (intended_manifest.to_canonical_json() + "\n").encode("utf-8")
    output_payloads = {
        "inputs/primary_intended_manifest.json": intended_payload,
        "inputs/scorer_plans.jsonl": plan_payload,
        "scores/complete_bundles.jsonl": bundle_payload,
        "scores/primary.jsonl": score_payload,
    }
    row_counts = {
        "inputs/primary_intended_manifest.json": 1,
        "inputs/scorer_plans.jsonl": len(plans),
        "scores/complete_bundles.jsonl": len(bundles),
        "scores/primary.jsonl": len(scores),
    }
    output_files = tuple(
        Phase4OutputFile(
            relative_path=relative,
            file_sha256=hashlib.sha256(payload).hexdigest(),
            logical_content_hash=(
                intended_manifest.content_hash
                if relative.endswith("primary_intended_manifest.json")
                else canonical_sha256(
                    tuple(json.loads(line) for line in payload.splitlines())
                )
            ),
            row_count=row_counts[relative],
            release_class=ReleaseClass.RESTRICTED,
        )
        for relative, payload in sorted(output_payloads.items())
    )
    analysis_configuration = SimpleNamespace(
        content_hash=_digest("analysis-configuration"),
        metric_configuration_path="configs/study/metrics.json",
        seed_manifest_path="data/synthetic/manifests/seed_manifest.json",
    )
    metric_configuration = SimpleNamespace(
        content_hash=_digest("metric-configuration"),
        metric_version_hash=metric_version_hash,
        leiden_half_resolution=0.25,
        leiden_base_resolution=0.5,
        leiden_double_resolution=1.0,
        leiden_seed=17,
        canonical_relation_vocabulary=("causal", "OTHER"),
    )
    index = Phase4AnalysisIndex(
        run_id="phase4-test",
        analysis_configuration_hash=analysis_configuration.content_hash,
        metric_configuration_hash=metric_configuration.content_hash,
        metric_version_hash=metric_configuration.metric_version_hash,
        source_tree_association_file_sha256=_digest("association-file"),
        source_tree_association_hash=_digest("association"),
        source_bindings=(),
        review_completion_manifest_hash=_digest("review"),
        final_reviewed_seal_hash=_digest("review-seal"),
        primary_intended_manifest_hash=intended_manifest.content_hash,
        combined_intended_manifest_hash=_digest("combined-intended"),
        table_manifest_file_sha256=_digest("table-file"),
        table_manifest_hash=_digest("table"),
        registered_analysis_file_sha256=_digest("registered-file"),
        registered_analysis_hash=_digest("registered"),
        report_gate_status_file_sha256=_digest("gate-file"),
        report_gate_status_hash=_digest("gate"),
        output_files=output_files,
        completed_at=NOW - timedelta(minutes=1),
    )
    index_payload = (index.to_canonical_json() + "\n").encode("utf-8")
    output_payloads["analysis_index.json"] = index_payload
    for relative, payload in output_payloads.items():
        path = phase4 / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    replay_receipt = SimpleNamespace(
        content_hash=_digest("replay-receipt"),
        analysis_index_hash=index.content_hash,
        analysis_configuration_hash=analysis_configuration.content_hash,
        metric_configuration_hash=metric_configuration.content_hash,
        metric_version_hash=metric_configuration.metric_version_hash,
        output_inventory_hash=_digest("output-inventory"),
    )
    state: dict[str, Any] = {
        "index": index,
        "index_payload": index_payload,
        "intended": intended_manifest,
        "intended_payload": intended_payload,
        "scores": scores,
        "score_payload": score_payload,
        "bundles": tuple(bundles),
        "bundle_payload": bundle_payload,
        "plans": plans,
        "plan_payload": plan_payload,
        "replay": replay_receipt,
        "frozen_position": frozen_position,
    }

    monkeypatch.setattr(
        review,
        "replay_phase4_analysis_outputs",
        lambda **_values: state["replay"],
    )
    monkeypatch.setattr(
        review.Phase4AnalysisConfiguration,
        "load",
        classmethod(lambda _cls, *_args, **_kwargs: analysis_configuration),
    )
    monkeypatch.setattr(
        review.StudyMetricConfiguration,
        "load",
        classmethod(lambda _cls, *_args, **_kwargs: metric_configuration),
    )

    def external(path: Path, _model: object, *, label: str) -> tuple[object, bytes]:
        del label
        if path.name == "analysis_index.json":
            return state["index"], state["index_payload"]
        if path.name == "primary_intended_manifest.json":
            return state["intended"], state["intended_payload"]
        raise AssertionError(f"unexpected external model path: {path}")

    def jsonl(
        path: Path,
        _model: object,
        *,
        label: str,
        allow_empty: bool = False,
    ) -> tuple[tuple[object, ...], bytes]:
        del label, allow_empty
        if path.name == "primary.jsonl":
            return state["scores"], state["score_payload"]
        if path.name == "complete_bundles.jsonl":
            return state["bundles"], state["bundle_payload"]
        if path.name == "scorer_plans.jsonl":
            return state["plans"], state["plan_payload"]
        raise AssertionError(f"unexpected JSONL path: {path}")

    monkeypatch.setattr(review, "_canonical_external_model", external)
    monkeypatch.setattr(review, "_canonical_jsonl_models", jsonl)
    monkeypatch.setattr(
        review,
        "adapt_projection_for_metrics",
        lambda projection, _configuration: adapters_by_projection[
            projection.content_hash
        ],
    )
    monkeypatch.setattr(
        review,
        "run_leiden_cpm",
        lambda nodes, _edges, configuration: partitions_by_key[
            (tuple(nodes), configuration.resolution, configuration.seed)
        ],
    )
    state["adapters_by_projection"] = adapters_by_projection
    state["partitions_by_key"] = partitions_by_key
    return phase4, state


def test_community_source_is_balanced_deterministic_and_condition_blind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phase4, _state = _install_complete_phase4_fixture(tmp_path, monkeypatch)

    first, files = prepare_community_review_source(
        repository=ROOT,
        phase4_configuration_path=ROOT / "configs/study/phase4_analysis.json",
        phase4_output_root=phase4,
        selected_at=NOW,
    )
    second, repeated_files = prepare_community_review_source(
        repository=ROOT,
        phase4_configuration_path=ROOT / "configs/study/phase4_analysis.json",
        phase4_output_root=phase4,
        selected_at=NOW,
    )

    assert first == second
    assert files == repeated_files
    assert first.selection_rule_hash == FROZEN_COMMUNITY_REVIEW_SELECTION_RULE_HASH
    assert first.eligible_partition_count == 12
    assert first.preoutput_eligible_context_count == 1
    assert first.missing_output_policy == "fail-without-substitution"
    assert len(files) == 13
    assert len({(item.world_id, item.context_id) for item in first.items}) == 1
    for condition in (
        ConditionName.C0_CLASSICAL_PRE,
        ConditionName.C1_LLM_PRE,
        ConditionName.C2_LLM_QUERY,
        ConditionName.A_FIXED_SELECT,
    ):
        selected = tuple(item for item in first.items if item.condition is condition)
        assert len(selected) == 3
        assert {item.resolution for item in selected} == {0.25, 0.5, 1.0}
        assert all(
            (item.seed_block is None) == (condition is ConditionName.C0_CLASSICAL_PRE)
            for item in selected
        )

    for item in first.items:
        panel = json.loads(files[item.review_panel.relative_path])
        visible = files[item.review_panel.relative_path].decode("utf-8")
        assert panel["condition_blind"] is True
        assert panel["source_identifiers_visible"] is False
        assert panel["cluster_count"] == 2
        assert panel["panel_schema"] == "condition-blind-community-partition-panel-v2"
        assert panel["context"]["wording"].startswith("Which roles")
        assert panel["evidence"][0]["text"].startswith("A harbor keeper")
        assert panel["nodes"][0]["contextual_type_definition"]
        assert panel["nodes"][0]["description"]
        assert panel["assertion_support"][0]["why_matters"]
        assert panel["assertion_support"][0]["provenance_methods"]
        displayed_node_ids = {node["node_id"] for node in panel["nodes"]}
        assert panel["assertion_support"][0]["subject_id"] in displayed_node_ids
        assert panel["assertion_support"][0]["object_id"] in displayed_node_ids
        assert "condition" not in panel
        assert "world_id" not in panel
        assert "context_id" not in panel
        assert "seed_block" not in panel
        for secret in (
            item.world_id,
            item.context_id,
            item.condition.value,
            item.projection_hash,
            item.partition_hash,
            "secret-node",
            "secret-cluster",
            "secret-evidence",
        ):
            assert secret not in visible
        assert '"canonical_relation_category":"causal"' in visible

    restricted = tmp_path / "restricted"
    restricted.mkdir()
    materialized, state, persisted = materialize_community_review_source(
        repository=ROOT,
        phase4_configuration_path=ROOT / "configs/study/phase4_analysis.json",
        phase4_output_root=phase4,
        selected_at=NOW,
        restricted_root=restricted,
        output_root=restricted / "community-sources",
    )
    assert state == "created"
    assert persisted == first
    assert {
        path.relative_to(materialized).as_posix()
        for path in materialized.rglob("*")
        if path.is_file()
    } == set(files)
    replayed, replay_state, _ = materialize_community_review_source(
        repository=ROOT,
        phase4_configuration_path=ROOT / "configs/study/phase4_analysis.json",
        phase4_output_root=phase4,
        selected_at=NOW,
        restricted_root=restricted,
        output_root=restricted / "community-sources",
    )
    assert replayed == materialized
    assert replay_state == "verified"


def test_community_source_rejects_changed_indexed_phase4_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phase4, state = _install_complete_phase4_fixture(tmp_path, monkeypatch)
    state["score_payload"] = b" " + state["score_payload"]

    with pytest.raises(BlindedReviewError, match=r"scores/primary\.jsonl differs"):
        prepare_community_review_source(
            repository=ROOT,
            phase4_configuration_path=ROOT / "configs/study/phase4_analysis.json",
            phase4_output_root=phase4,
            selected_at=NOW,
        )


def test_community_source_recomputes_adapter_and_exact_leiden_partitions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phase4, state = _install_complete_phase4_fixture(tmp_path, monkeypatch)
    original = state["bundles"][0]
    changed_adapter = SimpleNamespace(
        **{
            **vars(original.projection.adapter),
            "projection_id": "coherently-tampered-projection",
        }
    )
    state["bundles"] = (
        SimpleNamespace(
            **{
                **vars(original),
                "projection": SimpleNamespace(
                    adapter=changed_adapter,
                    structural=original.projection.structural,
                ),
            }
        ),
        *state["bundles"][1:],
    )
    with pytest.raises(BlindedReviewError, match="exact projection recomputation"):
        prepare_community_review_source(
            repository=ROOT,
            phase4_configuration_path=ROOT / "configs/study/phase4_analysis.json",
            phase4_output_root=phase4,
            selected_at=NOW,
        )

    phase4, state = _install_complete_phase4_fixture(tmp_path / "partition", monkeypatch)
    original = state["bundles"][0]
    half = original.projection.structural.leiden_half
    tampered_half = LeidenPartition(
        resolution=half.resolution,
        seed=half.seed,
        assignments=tuple(
            (node_id, cluster_id)
            for (node_id, _), (_, cluster_id) in zip(
                half.assignments,
                reversed(half.assignments),
                strict=True,
            )
        ),
        cluster_count=half.cluster_count,
        modularity=half.modularity,
    )
    state["bundles"] = (
        SimpleNamespace(
            **{
                **vars(original),
                "projection": SimpleNamespace(
                    adapter=original.projection.adapter,
                    structural=SimpleNamespace(
                        **{
                            **vars(original.projection.structural),
                            "leiden_half": tampered_half,
                        }
                    ),
                ),
            }
        ),
        *state["bundles"][1:],
    )
    with pytest.raises(BlindedReviewError, match="exact registered Leiden"):
        prepare_community_review_source(
            repository=ROOT,
            phase4_configuration_path=ROOT / "configs/study/phase4_analysis.json",
            phase4_output_root=phase4,
            selected_at=NOW,
        )


def test_community_source_never_substitutes_after_frozen_candidate_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phase4, _ = _install_complete_phase4_fixture(
        tmp_path,
        monkeypatch,
        eligible_positions=frozenset({(1, 1), (1, 2)}),
        make_frozen_candidate_unavailable=True,
    )
    with pytest.raises(BlindedReviewError, match="post-output substitution is forbidden"):
        prepare_community_review_source(
            repository=ROOT,
            phase4_configuration_path=ROOT / "configs/study/phase4_analysis.json",
            phase4_output_root=phase4,
            selected_at=NOW,
        )


def test_community_source_refuses_partial_phase4_directory(tmp_path: Path) -> None:
    phase4 = tmp_path / "phase4"
    phase4.mkdir()
    (phase4 / "analysis_index.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(BlindedReviewError, match="complete Phase 4 replay failed"):
        prepare_community_review_source(
            repository=ROOT,
            phase4_configuration_path=ROOT / "configs/study/phase4_analysis.json",
            phase4_output_root=phase4,
            selected_at=NOW,
        )
