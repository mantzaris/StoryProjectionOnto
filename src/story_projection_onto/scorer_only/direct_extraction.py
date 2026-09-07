"""Approved development-only query-blind extraction scope; never model-visible.

References are compiled before inspecting predictions from the source generator's
direct fact, causal and precedence witness maps. Query relevance is not extraction
eligibility. The existing strict matcher and its declared alternatives are reused.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from story_projection_onto.contracts import BenchmarkSplit, TemporalKind, canonical_sha256
from story_projection_onto.metrics.alignment import (
    AlignmentPlan,
    build_alignment_plan,
    score_alignment,
)
from story_projection_onto.metrics.common import maximum_cardinality_matching, precision_recall_f1

SCOPE_REVISION = "development-query-blind-direct-extraction-v1"
ELIGIBILITY_REVISION = "source-bound-direct-witness-v1"


@dataclass(frozen=True)
class DirectReference:
    witness_map: Mapping[str, dict[str, Any]]
    plans: tuple[AlignmentPlan, ...]
    target_to_witness: Mapping[str, str]
    manifest: Mapping[str, Any]


def compile_direct_reference(scorer: Any, evidence: Sequence[Any], root: Path) -> DirectReference:
    """No prediction argument: eligibility cannot depend on C0's emitted output."""
    from story_projection_onto.synthetic_benchmark import (
        build_seed_manifest,
        compile_alignment_alternatives,
        load_benchmark_configuration,
        realize_narrative,
    )

    if scorer.world_spec.split is not BenchmarkSplit.DEVELOPMENT:
        raise ValueError("direct competence reference forbids held-out worlds")
    config = load_benchmark_configuration(root / "configs/study/synthetic_benchmark.json")
    narrative = realize_narrative(scorer.world_spec, config, build_seed_manifest(config.root_seed))
    actual = {e.evidence_id: e.content_hash for e in evidence}
    regenerated = {e.evidence_id: e.content_hash for e in narrative.evidence}
    if actual != regenerated:
        raise ValueError("direct-witness generator differs from the frozen shared evidence")
    witnesses: dict[str, dict[str, Any]] = {}
    for family, mapping in (
        ("fact", narrative.fact_evidence_ids),
        ("causal", narrative.causal_evidence_ids),
        ("precedence", narrative.temporal_evidence_ids),
    ):
        for source_key, ids in mapping.items():
            if not ids:
                continue
            key = (
                family + ":" + (source_key if isinstance(source_key, str) else ":".join(source_key))
            )
            witnesses[key] = {
                "family": family,
                "source_key": source_key,
                "evidence_ids": list(ids),
                "evidence_hashes": {i: actual[i] for i in ids},
            }
    witnessed = {i for w in witnesses.values() for i in w["evidence_ids"]}
    if witnessed != set(actual):
        raise ValueError("direct-reference source coverage incomplete before prediction scoring")
    by_evidence = {tuple(sorted(w["evidence_ids"])): key for key, w in witnesses.items()}
    if len(by_evidence) != len(witnesses):
        raise ValueError("ambiguous source witness identity")
    plans, target_map, coverage = [], {}, []
    for gold, alternatives in zip(scorer.gold_projections, scorer.alternatives, strict=True):
        plan = build_alignment_plan(
            gold,
            alternatives,
            compiled_alternatives=compile_alignment_alternatives(gold, alternatives),
            include_context_excluded_assertions=True,
        )
        assertion_map = {a.assertion_id: a for a in gold.qualified_assertions}
        covered: set[str] = set()
        for target in plan.assertion_targets:
            key = by_evidence.get(tuple(sorted(assertion_map[target.target_id].evidence_ids)))
            if key is None:
                raise ValueError("reference assertion is not bound to a declared direct witness")
            if key in covered:
                raise ValueError("duplicate direct-reference target within a representation")
            covered.add(key)
            target_map[target.target_id] = key
        if covered != set(witnesses):
            raise ValueError(
                "missing direct-reference targets: " + str(sorted(set(witnesses) - covered))
            )
        coverage.append(
            {
                "gold_hash": gold.content_hash,
                "plan_hash": plan.content_hash,
                "witnesses_covered": len(covered),
            }
        )
        plans.append(plan)
    manifest = {
        "scope_revision": SCOPE_REVISION,
        "eligibility_revision": ELIGIBILITY_REVISION,
        "scorer_hash": scorer.content_hash,
        "world_id": scorer.world_spec.world_id,
        "generator_config_hash": config.content_hash,
        "evidence_hashes": actual,
        "witness_map_hash": canonical_sha256(witnesses),
        "target_to_witness": target_map,
        "representation_coverage": coverage,
        "complete_source_and_reference_coverage": True,
        "references_compiled_without_predictions": True,
        "strict_matcher_unchanged": True,
        "contextual_relevance_changes_eligibility_only_in_this_extraction_scope": True,
    }
    return DirectReference(witnesses, tuple(plans), target_map, manifest)


def score_direct_preconstruction(
    draft: Any, reference: DirectReference, evidence: Sequence[Any]
) -> dict:
    """One world, one denominator, one-to-one witness matching across frozen forms."""
    from story_projection_onto.scorer_only.development_assessment import (
        _cited_evidence_ids,
        _prediction_bundle,
    )

    ids = frozenset(e.evidence_id for e in evidence)
    adjacency: dict[str, set[str]] = defaultdict(set)
    matching_details = []
    comparable = []
    node_matches = False
    for plan in reference.plans:
        nodes, assertions, _ = _prediction_bundle(
            local_schema=draft.local_schema,
            instance_graph=draft.instance_graph,
            plan=plan,
            valid_evidence_ids=ids,
        )
        score = score_alignment(plan=plan, predicted_nodes=nodes, predicted_assertions=assertions)
        node_matches |= bool(score.all_anchor_node_matches)
        for match in score.strict_assertion_matches:
            witness = reference.target_to_witness[match.target_id]
            adjacency[match.prediction_id].add(witness)
            matching_details.append(
                {
                    "prediction_id": match.prediction_id,
                    "witness": witness,
                    "target_id": match.target_id,
                }
            )
        comparable.extend(
            m.model_dump(mode="json") for m in score.structurally_aligned_assertion_matches
        )
    pairs = maximum_cardinality_matching(
        {p: tuple(sorted(targets)) for p, targets in adjacency.items()}
    )
    predicted = draft.instance_graph.assertions
    metric = precision_recall_f1(
        true_positives=len(pairs),
        predicted_count=len(predicted),
        gold_count=len(reference.witness_map),
    )
    matched_ids = {p for p, _ in pairs}
    matched = [a for a in predicted if a.assertion_id in matched_ids]
    known = {TemporalKind.UNKNOWN, TemporalKind.NOT_APPLICABLE, TemporalKind.HORIZON_WITHHELD}
    families = {
        "explicit_entity": node_matches,
        "binary_relation": any(
            a.subject_id is not None and a.object_id is not None for a in matched
        ),
        "event_role": any(a.roles for a in matched),
        "story_time": any(a.temporal_scope.story_time.kind not in known for a in matched),
        "validity_time": any(a.temporal_scope.validity_time.kind not in known for a in matched),
    }
    citations = _cited_evidence_ids(draft)
    return {
        "scope_revision": SCOPE_REVISION,
        "eligibility_revision": ELIGIBILITY_REVISION,
        "draft_hash": draft.content_hash,
        "reference_manifest_hash": canonical_sha256(reference.manifest),
        "metric": metric.model_dump(mode="json"),
        "families": families,
        "valid_citations": sum(i in ids for i in citations),
        "citation_count": len(citations),
        "matched_pairs": pairs,
        "strict_representation_matches": matching_details,
        "structurally_aligned_matches": comparable,
        "unmatched_prediction_ids": [
            a.assertion_id for a in predicted if a.assertion_id not in matched_ids
        ],
        "unmatched_witness_ids": sorted(set(reference.witness_map) - {w for _, w in pairs}),
        "all_emitted_assertions_in_precision_denominator": True,
        "counts_are_world_once_not_sum_of_contexts": True,
    }


def aggregate_extraction(rows: Sequence[Mapping[str, Any]]) -> dict:
    if len(rows) != 4:
        raise ValueError("competence requires exactly four development preconstructions")
    totals = {
        k: sum(r["metric"][k] for r in rows)
        for k in ("true_positive_count", "predicted_count", "gold_count")
    }
    metric = precision_recall_f1(
        true_positives=totals["true_positive_count"],
        predicted_count=totals["predicted_count"],
        gold_count=totals["gold_count"],
    )
    families = {key: any(r["families"][key] for r in rows) for key in rows[0]["families"]}
    citations = sum(r["citation_count"] for r in rows)
    valid = sum(r["valid_citations"] for r in rows)
    evidence = valid / citations if citations else 0.0
    coverage = sum(families.values()) / len(families)
    return {
        "scope_revision": SCOPE_REVISION,
        "eligibility_revision": ELIGIBILITY_REVISION,
        "metric": metric.model_dump(mode="json"),
        "families": families,
        "explicit_family_coverage": coverage,
        "valid_evidence_reference_rate": evidence,
        "thresholds": {"precision": 0.85, "recall": 0.70, "family_coverage": 1, "evidence": 1},
        "passes": metric.precision >= 0.85
        and metric.recall >= 0.70
        and coverage == 1
        and evidence == 1,
        "implementation_change": False,
    }
