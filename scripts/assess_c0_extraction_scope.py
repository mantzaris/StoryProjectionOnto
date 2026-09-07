#!/usr/bin/env python3
"""Score preserved CPU outputs under the approved extraction-scope amendment.

No inference, construction, changed predictions or threshold modifications.
Only four development scorers may be opened. Contextual precision is unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

from story_projection_onto.conditions.base import ConditionAttemptRecord, ConditionPreparation
from story_projection_onto.development_continuation import load_development_prequery_evidence
from story_projection_onto.development_runtime import load_development_call_manifest
from story_projection_onto.manifest import write_json_atomic
from story_projection_onto.metrics.alignment import score_alignment
from story_projection_onto.metrics.common import precision_recall_f1
from story_projection_onto.scorer_only.development_assessment import (
    DevelopmentScientificAssessmentProvider,
    _prediction_bundle,
    resolve_development_world_id,
)
from story_projection_onto.scorer_only.direct_extraction import (
    aggregate_extraction,
    compile_direct_reference,
    score_direct_preconstruction,
)
from story_projection_onto.synthetic_benchmark import ScorerWorldArtifact


def scientific_fields(value):
    if isinstance(value, dict):
        return {
            k: scientific_fields(v)
            for k, v in value.items()
            if k not in {"content_hash", "schema_version"}
        }
    if isinstance(value, list):
        return [scientific_fields(v) for v in value]
    return value


def diagnose_unmatched(draft, reference, evidence, scorer, unmatched):
    """Nearest source-sharing reference is a comparison, not a truth verdict."""
    candidates = {}
    for plan in reference.plans:
        _, assertions, _ = _prediction_bundle(
            local_schema=draft.local_schema,
            instance_graph=draft.instance_graph,
            plan=plan,
            valid_evidence_ids=frozenset(e.evidence_id for e in evidence),
        )
        for assertion in assertions:
            if assertion.prediction_id not in unmatched:
                continue
            actual = scientific_fields(assertion.signature.model_dump(mode="json"))
            for target in plan.assertion_targets:
                for alternative in target.alternatives:
                    if not set(assertion.evidence_ids).intersection(
                        alternative.supporting_evidence_ids
                    ):
                        continue
                    expected = scientific_fields(alternative.signature.model_dump(mode="json"))
                    differences = {
                        k: {"actual": actual[k], "expected": expected[k]}
                        for k in actual
                        if actual[k] != expected[k]
                    }
                    rank = (len(differences), target.target_id, alternative.alternative_id)
                    previous = candidates.get(assertion.prediction_id)
                    if previous is None or rank < previous[0]:
                        candidates[assertion.prediction_id] = (rank, target.target_id, differences)
    source = {e.evidence_id: e.text for e in evidence}
    gold = {a.assertion_id: a for g in scorer.gold_projections for a in g.qualified_assertions}
    rows = []
    for a in draft.instance_graph.assertions:
        if a.assertion_id not in unmatched:
            continue
        candidate = candidates.get(a.assertion_id)
        rows.append(
            {
                "prediction_id": a.assertion_id,
                "evidence": {e: source[e] for e in a.evidence_ids},
                "prediction": a.model_dump(mode="json"),
                "target": None if candidate is None else gold[candidate[1]].model_dump(mode="json"),
                "mismatched_fields": None if candidate is None else candidate[2],
                "category": "no_source_sharing_reference"
                if candidate is None
                else "duplicate_or_one_to_one_conflict"
                if not candidate[2]
                else "qualified_signature_mismatch",
                "selection_rule": (
                    "fewest differing strict signature fields; target/alternative ID tie-break"
                ),
                "absence_of_match_does_not_prove_unsupported": True,
            }
        )
    return rows


def run(root: Path, source: Path, output: Path):
    if output.exists() or "restricted" not in output.parts:
        raise ValueError("fresh restricted scoring directory required")
    tic = time.monotonic()
    output.mkdir(parents=True, mode=0o700)
    historical = json.loads((source / "calibration.json").read_bytes())
    for name, expected in historical["files"].items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise ValueError("preserved calibration artifact hash mismatch: " + name)
    neutral, visible, _ = load_development_prequery_evidence(
        root, load_development_call_manifest(root)
    )
    references, scorers = {}, {}
    # All source-bound references frozen BEFORE opening prediction payloads.
    for unit, artifact in neutral.items():
        world = resolve_development_world_id(
            root, visible[unit].content_hash, artifact.content_hash
        )
        scorer = ScorerWorldArtifact.model_validate_json(
            (root / f"data/synthetic/scorer_only/development/{world}.json").read_bytes()
        )
        reference = compile_direct_reference(scorer, artifact.evidence, root)
        references[unit], scorers[unit] = reference, scorer
        write_json_atomic(
            {"manifest": reference.manifest, "witness_map": reference.witness_map},
            output / f"{unit}.reference.json",
        )
    extraction, contextual = [], []
    for unit, artifact in neutral.items():
        preparation = ConditionPreparation.model_validate_json(
            (source / f"{unit}.preparation.json").read_bytes()
        )
        if (
            preparation.sealed_preontology is None
            or preparation.snapshot_hash != artifact.snapshot.content_hash
        ):
            raise ValueError("C0 sealed preconstruction source mismatch")
        draft = preparation.sealed_preontology.draft
        row = score_direct_preconstruction(draft, references[unit], artifact.evidence)
        row.update(
            unit=unit,
            preparation_hash=preparation.content_hash,
            seal_hash=preparation.sealed_preontology.construction_seal.content_hash,
        )
        extraction.append(row)
        write_json_atomic(row, output / f"{unit}.extraction.json")
        diagnosis = diagnose_unmatched(
            draft,
            references[unit],
            artifact.evidence,
            scorers[unit],
            set(row["unmatched_prediction_ids"]),
        )
        write_json_atomic(
            {
                "rows": diagnosis,
                "field_counts": dict(
                    Counter(
                        key for item in diagnosis for key in item.get("mismatched_fields", {}) or {}
                    )
                ),
            },
            output / f"{unit}.errors.json",
        )
        for ordinal in (1, 2, 3):
            attempt = ConditionAttemptRecord.model_validate_json(
                (source / f"{unit}.q{ordinal}.json").read_bytes()
            )
            if attempt.projection is None:
                raise ValueError("preserved contextual projection invalid")
            projection = attempt.projection
            plan = DevelopmentScientificAssessmentProvider._alignment_plan(scorers[unit], ordinal)
            nodes, assertions, _ = _prediction_bundle(
                local_schema=projection.local_schema,
                instance_graph=projection.instance_graph,
                plan=plan,
                valid_evidence_ids=frozenset(e.evidence_id for e in artifact.evidence),
            )
            score = score_alignment(
                plan=plan, predicted_nodes=nodes, predicted_assertions=assertions
            )
            full = references[unit].plans[ordinal - 1]
            full_nodes, full_assertions, _ = _prediction_bundle(
                local_schema=projection.local_schema,
                instance_graph=projection.instance_graph,
                plan=full,
                valid_evidence_ids=frozenset(e.evidence_id for e in artifact.evidence),
            )
            full_score = score_alignment(
                plan=full, predicted_nodes=full_nodes, predicted_assertions=full_assertions
            )
            relevant = {t.target_id for t in plan.assertion_targets}
            contextual.append(
                {
                    "unit": unit,
                    "query": ordinal,
                    "projection_hash": projection.content_hash,
                    "score": score.model_dump(mode="json"),
                    "strict_but_context_excluded_prediction_ids": [
                        m.prediction_id
                        for m in full_score.strict_assertion_matches
                        if m.target_id not in relevant
                    ],
                }
            )
    totals = {
        k: sum(r["score"]["strict_assertion_score"][k] for r in contextual)
        for k in ("true_positive_count", "predicted_count", "gold_count")
    }
    contextual_metric = precision_recall_f1(
        true_positives=totals["true_positive_count"],
        predicted_count=totals["predicted_count"],
        gold_count=totals["gold_count"],
    )
    result = {
        "historical_competence": historical,
        "revised_extraction_competence": aggregate_extraction(extraction),
        "contextual_projection_metric": contextual_metric.model_dump(mode="json"),
        "contextual_unit_scores": contextual,
        "extraction_world_scores": extraction,
        "predictions_reused_unchanged": True,
        "source_calibration": str(source.relative_to(root)),
        "extraction_implementation_changed": False,
        "source_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in (
                "scripts/assess_c0_extraction_scope.py",
                "src/story_projection_onto/scorer_only/direct_extraction.py",
                "src/story_projection_onto/scorer_only/development_assessment.py",
                "src/story_projection_onto/metrics/alignment.py",
                "src/story_projection_onto/synthetic_benchmark.py",
                "src/story_projection_onto/conditions/c0.py",
            )
        },
        "cpu_seconds": time.monotonic() - tic,
        "gpu_seconds": 0,
        "heldout_references_changed": False,
        "files": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob("*.json")
        },
    }
    write_json_atomic(result, output / "assessment.json")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "revised_extraction_competence",
                    "contextual_projection_metric",
                    "cpu_seconds",
                )
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(Path.cwd(), args.source.resolve(), args.output.resolve())
