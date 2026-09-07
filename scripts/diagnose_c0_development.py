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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--calibration", type=Path)
    inputs.add_argument("--validity-from", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.validity_from:
        validity_examples(args.validity_from, args.output)
    else:
        run(Path.cwd(), args.calibration, args.output)
