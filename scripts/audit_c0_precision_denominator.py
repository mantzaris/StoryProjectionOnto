#!/usr/bin/env python3
"""Reproducible development-only sample; no automatic truth labels or rescoring."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from story_projection_onto.development_continuation import load_development_prequery_evidence
from story_projection_onto.development_runtime import load_development_call_manifest
from story_projection_onto.manifest import write_json_atomic
from story_projection_onto.scorer_only.development_assessment import resolve_development_world_id

SAMPLE_SEED = "c0-denominator-audit-20260907-v1"


def rank(row):
    return hashlib.sha256(
        f"{SAMPLE_SEED}|{row['unit']}|{row['query']}|{row['prediction_id']}".encode()
    ).hexdigest()


def run(root, source, calibration, output):
    if output.exists() or "restricted" not in output.parts:
        raise ValueError("new restricted output required")
    original = json.loads(source.read_bytes())
    population = [
        r
        for r in original["rows"]
        if r.get("category") == "no_contextual_gold_shares_cited_evidence"
    ]
    neutral, visible, _ = load_development_prequery_evidence(
        root, load_development_call_manifest(root)
    )
    selected = []
    eligibility_counts = Counter()
    for unit in sorted(neutral):
        world_id = resolve_development_world_id(
            root, visible[unit].content_hash, neutral[unit].content_hash
        )
        scorer = json.loads(
            (root / f"data/synthetic/scorer_only/development/{world_id}.json").read_bytes()
        )
        if scorer["world_spec"]["split"] != "development":
            raise ValueError("held-out scorer forbidden")
        for ordinal in range(1, 4):
            gold = scorer["gold_projections"][ordinal - 1]
            relevance = {item["target_id"]: item["is_relevant"] for item in gold["relevance"]}
            fact_evidence = {e for ids in scorer["fact_evidence_ids"].values() for e in ids}
            for row in population:
                if row["unit"] != unit or row["query"] != ordinal:
                    continue
                # This is an eligibility audit, NOT a truth judgment: shared
                # citation does not establish endpoint/predicate/time correctness.
                projection_value = json.loads(
                    (calibration / f"{unit}.q{ordinal}.json").read_bytes()
                )["projection"]
                prediction = next(
                    a
                    for a in projection_value["instance_graph"]["assertions"]
                    if a["assertion_id"] == row["prediction_id"]
                )
                matches = [
                    a
                    for a in gold["qualified_assertions"]
                    if set(prediction["evidence_ids"]).intersection(a["evidence_ids"])
                ]
                relevant = [a for a in matches if relevance.get(a["assertion_id"], True)]
                reason = (
                    "reference_exists_but_context_irrelevant"
                    if matches and not relevant
                    else "relevant_reference_excluded_by_direct_filter"
                    if relevant
                    and not any(fact_evidence.intersection(a["evidence_ids"]) for a in relevant)
                    else "needs_further_alignment_or_coverage_investigation"
                )
                eligibility_counts[reason] += 1
            rows = sorted(
                [r for r in population if r["unit"] == unit and r["query"] == ordinal], key=rank
            )[:2]
            projection = json.loads((calibration / f"{unit}.q{ordinal}.json").read_bytes())[
                "projection"
            ]
            graph = projection["instance_graph"]
            nodes = {
                n.get("entity_id", n.get("event_id")): n
                for n in graph["entities"] + graph["events"]
            }
            predicates = {p["predicate_id"]: p for p in projection["local_schema"]["predicates"]}
            for row in rows:
                assertion = next(
                    a for a in graph["assertions"] if a["assertion_id"] == row["prediction_id"]
                )
                evidence_ids = set(assertion["evidence_ids"])
                gold_by_query = [
                    [
                        a
                        for a in gold["qualified_assertions"]
                        if evidence_ids.intersection(a["evidence_ids"])
                    ]
                    for gold in scorer["gold_projections"]
                ]
                selected.append(
                    row
                    | {
                        "selection_hash": rank(row),
                        "world": world_id,
                        "assertion": assertion,
                        "subject": nodes.get(assertion["subject_id"]),
                        "object": nodes.get(assertion["object_id"]),
                        "predicate": predicates[assertion["predicate_id"]],
                        "gold_same_evidence_by_query": gold_by_query,
                        "query_assignment": scorer["query_assignment"],
                        "current_context_relevance": [
                            item
                            for item in gold["relevance"]
                            if item["target_id"]
                            in {a["assertion_id"] for a in gold_by_query[ordinal - 1]}
                        ],
                        "direct_world_facts": [
                            f
                            for f in scorer["world_spec"]["facts"]
                            if evidence_ids.intersection(
                                scorer["fact_evidence_ids"].get(f["fact_id"], [])
                            )
                        ],
                        "judgment": None,
                    }
                )
    write_json_atomic(
        {
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "population_count": len(population),
            "sample_seed": SAMPLE_SEED,
            "selection_rule": "lowest two SHA256 ranks within each of 12 development contexts",
            "sample_count": len(selected),
            "population_eligibility_counts_not_truth_labels": dict(eligibility_counts),
            "sample": selected,
            "scores_changed": False,
            "held_out_opened": False,
        },
        output,
    )
    print(json.dumps({"population": len(population), "sample": len(selected)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnosis", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(Path.cwd(), args.diagnosis, args.calibration, args.output)
