#!/usr/bin/env python3
"""Restricted development-only field comparisons; never changes gold or scores."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from story_projection_onto.conditions.base import ConditionAttemptRecord
from story_projection_onto.development_continuation import load_development_prequery_evidence
from story_projection_onto.development_runtime import load_development_call_manifest
from story_projection_onto.manifest import write_json_atomic
from story_projection_onto.metrics.alignment import score_alignment
from story_projection_onto.scorer_only.development_assessment import (
    DevelopmentScientificAssessmentProvider,
    _direct_assertion_ids,
    _filtered_alignment_plan,
    _prediction_bundle,
    resolve_development_world_id,
)
from story_projection_onto.synthetic_benchmark import ScorerWorldArtifact


def clean(value):
    if isinstance(value, dict):
        return {
            k: clean(v) for k, v in value.items() if k not in {"content_hash", "schema_version"}
        }
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


def run(root, calibration, output):
    if output.exists() or "restricted" not in output.parts:
        raise ValueError("new restricted diagnosis required")
    neutral, visible, _ = load_development_prequery_evidence(
        root, load_development_call_manifest(root)
    )
    rows, summaries = [], []
    for unit, artifact in neutral.items():
        world_id = resolve_development_world_id(
            root, visible[unit].content_hash, artifact.content_hash
        )
        scorer = ScorerWorldArtifact.model_validate_json(
            (root / f"data/synthetic/scorer_only/development/{world_id}.json").read_bytes()
        )
        assert scorer.world_spec.split.value == "development"
        evidence = {e.evidence_id: e for e in artifact.evidence}
        for ordinal in range(1, 4):
            attempt = ConditionAttemptRecord.model_validate_json(
                (calibration / f"{unit}.q{ordinal}.json").read_bytes()
            )
            projection = attempt.projection
            gold = scorer.gold_projections[ordinal - 1]
            plan = _filtered_alignment_plan(
                DevelopmentScientificAssessmentProvider._alignment_plan(scorer, ordinal),
                _direct_assertion_ids(scorer, gold),
            )
            nodes, assertions, _ = _prediction_bundle(
                local_schema=projection.local_schema,
                instance_graph=projection.instance_graph,
                plan=plan,
                valid_evidence_ids=frozenset(evidence),
            )
            result = score_alignment(
                plan=plan, predicted_nodes=nodes, predicted_assertions=assertions
            )
            summaries.append(
                {
                    "unit": unit,
                    "world": world_id,
                    "query": ordinal,
                    "node_matches": len(result.node_matches),
                    "predicted_nodes": len(nodes),
                    "strict_matches": len(result.strict_assertion_matches),
                    "structural_matches": len(result.structurally_aligned_assertion_matches),
                    "predicted_assertions": len(assertions),
                    "gold_assertions": len(plan.assertion_targets),
                }
            )
            originals = {a.assertion_id: a for a in projection.instance_graph.assertions}
            gold_originals = {a.assertion_id: a for a in gold.qualified_assertions}
            for assertion in assertions:
                candidates = [
                    (target, alt)
                    for target in plan.assertion_targets
                    for alt in target.alternatives
                    if set(alt.supporting_evidence_ids).intersection(assertion.evidence_ids)
                ]
                if not candidates:
                    rows.append(
                        {
                            "unit": unit,
                            "query": ordinal,
                            "prediction_id": assertion.prediction_id,
                            "category": "no_contextual_gold_shares_cited_evidence",
                            "evidence": [evidence[e].text for e in assertion.evidence_ids],
                        }
                    )
                    continue
                actual = clean(assertion.signature.model_dump(mode="json"))
                target, alternative = min(
                    candidates,
                    key=lambda pair: sum(
                        actual[k] != clean(pair[1].signature.model_dump(mode="json"))[k]
                        for k in actual
                    ),
                )
                expected = clean(alternative.signature.model_dump(mode="json"))
                differences = {
                    k: {"actual": actual[k], "expected": expected[k]}
                    for k in actual
                    if actual[k] != expected[k]
                }
                rows.append(
                    {
                        "unit": unit,
                        "query": ordinal,
                        "prediction_id": assertion.prediction_id,
                        "gold_id": target.target_id,
                        "evidence_ids": list(assertion.evidence_ids),
                        "evidence": [evidence[e].text for e in assertion.evidence_ids],
                        "actual_assertion": originals[assertion.prediction_id].model_dump(
                            mode="json"
                        ),
                        "gold_assertion": gold_originals[target.target_id].model_dump(mode="json"),
                        "actual_normalized": actual,
                        "expected_normalized": expected,
                        "mismatched_fields": differences,
                        "grounding_status": assertion.grounding_status.value,
                    }
                )
    counts = Counter(field for row in rows for field in row.get("mismatched_fields", {}))
    value = {
        "kind": "preserved_c0_development_field_diagnosis",
        "source_calibration": str(calibration),
        "summaries": summaries,
        "mismatched_field_counts": dict(counts),
        "rows": rows,
        "gold_changed": False,
        "held_out_gold_opened": False,
    }
    write_json_atomic(value, output)
    print(json.dumps({k: v for k, v in value.items() if k != "rows"}))


def validity_examples(source, output):
    """Reuse the preserved development comparison; no scoring or gold mutation."""
    if output.exists() or "restricted" not in output.parts:
        raise ValueError("new restricted diagnosis required")
    original = json.loads(source.read_bytes())
    rows = [r for r in original["rows"] if set(r.get("mismatched_fields", {})) == {"validity_time"}]
    # Declared rule: first comparison for each of these temporal-error families,
    # in existing deterministic artifact order; never condition-output favorability.
    selected = [
        next(r for r in rows if r["actual_normalized"]["predicate"] == predicate)
        for predicate in ("member_of", "coordinates_with", "acts_at")
    ]
    write_json_atomic(
        {
            "kind": "c0_validity_only_development_examples_not_rescoring",
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "validity_only_comparisons": len(rows),
            "selection_rule": "first preserved comparison for member_of, coordinates_with, acts_at",
            "examples": selected,
            "authoritative_rules": {
                "method_4_C0": "No new temporal/epistemic qualification after query reveal.",
                "method_7": "Story/event time and state/relation validity remain distinct; "
                "unknown time and temporal underdetermination are explicit.",
                "method_12": "Strict matching requires essential story/validity scope, "
                "aligned endpoints/roles, normalized predicate, direction, applicable "
                "epistemic status and valid supporting evidence.",
                "implementation_543": "C0 competence: directly stated development qualified "
                "assertions, precision .85, recall .70 and 100% valid references.",
            },
            "gold_changed": False,
            "threshold_changed": False,
            "held_out_gold_opened": False,
        },
        output,
    )
    print(json.dumps({"validity_only_comparisons": len(rows), "examples": len(selected)}))


def classify_validity(source, output):
    """Separate evidence support, latent intrinsic bounds and query clipping.

    Reads only the already-diagnosed DEVELOPMENT worlds. This classifies the
    reference construction; it does not change or rescore any assertion.
    """
    from story_projection_onto.synthetic_benchmark import StoryScopeBlock, _scope, _time_bounds

    if output.exists() or "restricted" not in output.parts:
        raise ValueError("new restricted classification required")
    root = Path.cwd()
    diagnosis = json.loads(source.read_bytes())
    neutral, visible, _ = load_development_prequery_evidence(
        root, load_development_call_manifest(root)
    )
    worlds = {}
    for unit, artifact in neutral.items():
        world_id = resolve_development_world_id(
            root, visible[unit].content_hash, artifact.content_hash
        )
        world = json.loads(
            (root / f"data/synthetic/scorer_only/development/{world_id}.json").read_bytes()
        )
        assert world["world_spec"]["split"] == "development"
        worlds[unit] = world
    rows = []
    for original in diagnosis["rows"]:
        if set(original.get("mismatched_fields", {})) != {"validity_time"}:
            continue
        world = worlds[original["unit"]]
        evidence_ids = set(original["evidence_ids"])
        candidates = [
            f
            for f in world["world_spec"]["facts"]
            if evidence_ids.intersection(world["fact_evidence_ids"].get(f["fact_id"], []))
            and f["relation"] == original["expected_normalized"]["predicate"]
        ]
        assert len(candidates) == 1
        fact = candidates[0]
        scope = _scope(
            StoryScopeBlock(world["query_assignment"]["story_scopes"][original["query"] - 1])
        )
        start, end = _time_bounds(scope)
        reference = original["expected_normalized"]["validity_time"]
        intrinsic = {"start": fact["story_position"], "end": fact["validity_end"]}
        clipped_end = min(end, fact["validity_end"] if fact["validity_end"] is not None else 10**9)
        assert reference["start"] == max(start, intrinsic["start"])
        assert reference["end"] == (None if clipped_end == 10**9 else clipped_end)
        evidence = [e for e in neutral[original["unit"]].evidence if e.evidence_id in evidence_ids]
        prose = " ".join(e.text for e in evidence)
        explicit_duration = "through step " in prose or " thereafter" in prose
        rows.append(
            {
                "unit": original["unit"],
                "query": original["query"],
                "prediction_id": original["prediction_id"],
                "reference_id": original["gold_id"],
                "evidence": [e.text for e in evidence],
                "temporal_clues": [
                    clue.normalized_expression for e in evidence for clue in e.temporal_clues
                ],
                "predicate": fact["relation"],
                "world_intrinsic_validity": intrinsic,
                "query_scope": clean(scope.model_dump(mode="json")),
                "reference_validity": reference,
                "c0_validity": original["actual_normalized"]["validity_time"],
                "explicit_duration_language": explicit_duration,
                "query_window_clipped": reference["start"] != intrinsic["start"]
                or reference["end"] != intrinsic["end"],
                "unstated_finite_intrinsic_end": fact["validity_end"] is not None
                and not explicit_duration,
                "authoritative_rule_supplies_exact_intrinsic_interval": False,
                "reference_interval_fully_explicit": explicit_duration,
                "interpretation": (
                    "Query visibility is a restriction, not a new onset/cessation claim. "
                    "A story point alone does not establish a full validity interval."
                ),
            }
        )
    counts = {
        key: sum(r[key] for r in rows)
        for key in (
            "query_window_clipped",
            "unstated_finite_intrinsic_end",
            "reference_interval_fully_explicit",
            "authoritative_rule_supplies_exact_intrinsic_interval",
        )
    }
    write_json_atomic(
        {
            "kind": "development_reference_validity_classification_not_rescoring",
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "comparison_count": len(rows),
            "overlapping_counts": counts,
            "rows": rows,
            "gold_changed": False,
            "scoring_changed": False,
            "held_out_opened": False,
        },
        output,
    )
    print(json.dumps({"comparisons": len(rows), "overlapping_counts": counts}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--calibration", type=Path)
    inputs.add_argument("--validity-from", type=Path)
    inputs.add_argument("--classify-validity-from", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.classify_validity_from:
        classify_validity(args.classify_validity_from, args.output)
    elif args.validity_from:
        validity_examples(args.validity_from, args.output)
    else:
        run(Path.cwd(), args.calibration, args.output)
