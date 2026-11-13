#!/usr/bin/env python3
"""Readable restricted comparison from immutable development-only CPU artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def run(before: Path, matcher: Path, after: Path, output: Path):
    if output.exists() or "restricted" not in output.parts:
        raise ValueError("fresh restricted report required")
    loaded = [json.loads((p / "assessment.json").read_bytes()) for p in (before, matcher, after)]
    if any(
        x["revised_extraction_competence"]["scope_revision"]
        != "development-query-blind-direct-extraction-v1"
        for x in loaded
    ):
        raise ValueError("approved extraction scope required for comparable rows")
    lines = [
        "# C0 CPU repair comparison (development only)\n",
        "No gold, threshold, eligibility or denominator-rule change in these comparisons. "
        "The earlier 29/269 precision and 29/84 recall record remains historical.\n",
        "| Stage | Extraction TP / emitted | TP / direct targets | P / R / F1 | "
        "Contextual TP / emitted / targets | Contextual P / R / F1 |",
        "|---|---|---|---|---|---|",
    ]
    for label, result in zip(
        ("Before these repairs", "Matcher only; original outputs", "Repaired C0; new outputs"),
        loaded,
        strict=True,
    ):
        e = result["revised_extraction_competence"]["metric"]
        c = result["contextual_projection_metric"]
        def rates(m):
            return " / ".join(f"{m[k]:.6f}" for k in ("precision", "recall", "f1"))
        lines.append(
            f"| {label} | {e['true_positive_count']}/{e['predicted_count']} | "
            f"{e['true_positive_count']}/{e['gold_count']} | {rates(e)} | "
            f"{c['true_positive_count']}/{c['predicted_count']}/{c['gold_count']} | "
            f"{rates(c)} |"
        )
    current = loaded[-1]
    excluded = sum(
        len(x["strict_but_context_excluded_prediction_ids"])
        for x in current["contextual_unit_scores"]
    )
    lines += [
        "\nCurrent competence record:\n\n```json\n"
        + json.dumps(current["revised_extraction_competence"], indent=2)
        + "\n```\n",
        f"All {excluded} strictly supported but context-excluded final assertions remain "
        "in contextual precision. No precision denominator removes an unmatched assertion.\n",
        "## Focused before/after examples\n",
        "Selection: lowest development unit; first source ID with an old mismatch for "
        "each of member_of, participates_in, enabled (fixed predicates of this repair). "
        "All assertions citing that source are shown; no best-looking prediction chosen.\n",
    ]
    unit = sorted(before.glob("*.errors.json"))[0].name.removesuffix(".errors.json")
    rows = json.loads((before / f"{unit}.errors.json").read_bytes())["rows"]
    root = Path.cwd()
    for predicate in ("member_of", "participates_in", "enabled"):
        choices = [
            r for r in rows if r["prediction"]["predicate_id"] == "c0-predicate-" + predicate
        ]
        if not choices:
            continue
        example = min(choices, key=lambda r: tuple(sorted(r["evidence"])))
        evidence_ids = set(example["evidence"])
        lines += [
            f"### {predicate}\n",
            *[f"{k}: {v}\n" for k, v in example["evidence"].items()],
            "Old strict mismatch fields:\n\n```json\n"
            + json.dumps(example["mismatched_fields"], indent=2)
            + "\n```\n",
        ]
        for label, assessment in (("Before", loaded[0]), ("After", loaded[2])):
            source = root / assessment["source_calibration"]
            draft = json.loads((source / f"{unit}.preparation.json").read_bytes())[
                "sealed_preontology"
            ]["draft"]
            g = draft["instance_graph"]
            nodes = {
                n.get("entity_id", n.get("event_id")): n for n in [*g["entities"], *g["events"]]
            }
            chosen = [a for a in g["assertions"] if set(a["evidence_ids"]) & evidence_ids]
            involved = {
                i
                for a in chosen
                for i in (a["subject_id"], a["object_id"], *(r["object_id"] for r in a["roles"]))
                if i
            }
            lines += [
                f"{label} emitted records (semantics unchanged in this rendering):\n\n```json\n"
                + json.dumps(
                    {"nodes": [nodes[i] for i in sorted(involved)], "assertions": chosen}, indent=2
                )
                + "\n```\n"
            ]
    lines += [
        "## Remaining failures\n",
        "All unmatched extraction predictions are retained. An empty mismatch object "
        "means a duplicate/one-to-one conflict, not proof of unsupported content. "
        "Reference organizations still impose their declared event/identity alternatives; "
        "these were not changed to match C0.\n",
    ]
    for path in sorted(after.glob("*.errors.json")):
        rows = json.loads(path.read_bytes())["rows"]
        for row in rows:
            lines.append(
                f"- {path.stem}: {row['prediction_id']}; "
                + " ".join(row["evidence"].values())
                + "\n\n```json\n"
                + json.dumps(row["mismatched_fields"], indent=2)
                + "\n```\n"
            )
    lines += [
        "## Source identities\n\n```json\n"
        + json.dumps(
            {
                str(p / "assessment.json"): hashlib.sha256(
                    (p / "assessment.json").read_bytes()
                ).hexdigest()
                for p in (before, matcher, after)
            },
            indent=2,
        )
        + "\n```\n"
    ]
    output.write_text("\n".join(lines), encoding="utf-8")
    print(str(output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("before", "matcher", "after", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    a = parser.parse_args()
    run(a.before, a.matcher, a.after, a.output)
