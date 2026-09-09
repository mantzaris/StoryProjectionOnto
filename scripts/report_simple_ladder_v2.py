"""Reproduce the v2 diagnostic report without modifying the original ladder."""

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path

from scripts.report_simple_ladder import graph
from story_projection_onto.manifest import write_json_atomic
from story_projection_onto.scorer_only.simple_ladder_v2 import office_comparison


def render(run, output):
    def read(name):
        return json.loads((run / name).read_text())

    cases, refs, terminal = (
        read("frozen-cases.json"),
        read("reference-answers-not-transmitted.json"),
        read("terminal.json"),
    )
    outcomes = terminal["outcomes"]
    by_id = {o["case_id"]: o for o in outcomes}
    rows, sections = [], []
    title = "Simple synthetic ladder v2 — format and fresh-counterpart diagnostics"
    md = [
        "# " + title,
        "",
        "Exploratory development only. No registered ontology acceptance, held-out efficacy, model reliability or production-timing claim.",
        "",
        "Eight requests and all references/rules were frozen before inference; one attempt per case, no adaptive repair. Unconstrained JSON, pinned Qwen3-8B-AWQ and nonthinking sampling are unchanged. V2 requires evidence_ids as supplied string lists, decomposed belief relationships with holder/attitude, and both requested interval bounds. Only predeclared endpoint aliases and original relation synonyms are permitted. Original v1 outputs and scores are unchanged.",
        "",
        "All predictions remain in the applicable denominators, including errors and duplicates. Underlying relationship scoring ignores citations/qualifications; complete-fact scoring requires them. Required-field compliance includes conditional requirements when the underlying relationship is resolved. Unresolved wording is not accepted or automatically called false.",
        "",
        "| Case | Cohort / task | Format | Citations | Underlying F1 | Complete P | Complete R | Complete F1 | Tokens in/out | Seconds |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for o in outcomes:
        k, e, c = o["case_id"], o["evaluation"], cases[o["case_id"]]
        response = o["response"] or {}
        f, u = e["full"], e["underlying"]
        row = dict(
            case_id=k,
            cohort=c["cohort"],
            task=c["level"],
            original_case=c["original_case"],
            parseable=e["parseable"],
            format_compliant=e["format_compliant"],
            shape_compliant=e["shape_compliant"],
            conditional_format_unresolved=e["format_unresolved"],
            citation_valid=e["citation_valid"],
            predictions=e["prediction_count"],
            underlying_correct=u["true_positive"],
            underlying_precision=u["precision"],
            underlying_recall=u["recall"],
            underlying_f1=u["f1"],
            complete_correct=f["true_positive"],
            reference_count=f["reference_count"],
            precision=f["precision"],
            recall=f["recall"],
            f1=f["f1"],
            input_tokens=response.get("prompt_tokens"),
            output_tokens=response.get("completion_tokens"),
            output_allowance=c["output_allowance"],
            finish_reason=response.get("finish_reason"),
            request_seconds=o["request_seconds"],
        )
        rows.append(row)
        md.append(
            f"| {k} | {c['cohort']} / {c['level']} | {e['format_compliant']}/{e['prediction_count']} | {e['citation_valid']}/{e['prediction_count']} | {u['f1']:.3f} | {f['precision']:.3f} | {f['recall']:.3f} | {f['f1']:.3f} | {row['input_tokens']}/{row['output_tokens']} | {o['request_seconds']:.3f} |"
        )
        evidence = "\n".join(f"{sid}: {text}" for sid, text in c["evidence"].items())
        details = html.escape(json.dumps(e, indent=2))
        records = o["parsed"]["facts"] if o["parsed"] else []
        record_rows = []
        for r in e["records"]:
            fact = html.escape(json.dumps(r["fact"], ensure_ascii=False))
            record_rows.append(
                f"<tr><td>{r['index'] + 1}</td><td>{fact}</td><td>{html.escape(r['assessment'])}</td><td>{html.escape('; '.join(r['format_issues'] + r['meaning_issues']))}</td></tr>"
            )
        sections.append(
            f'<section id="case-{k}"><h2>Case {k}: {c["cohort"]} / {c["level"]}</h2>'
            f"<pre>{html.escape(evidence)}</pre><p>{html.escape(c['question'])}</p>"
            "<details><summary>Frozen reference answers (not model output)</summary><pre>"
            + html.escape(json.dumps(refs[k], indent=2))
            + "</pre></details>"
            + f"<h3>Actual response</h3><pre>{html.escape(o['raw_text'])}</pre>{graph(records)}"
            + "<table><tr><th>#</th><th>Actual fact</th><th>Assessment</th><th>Errors</th></tr>"
            + "".join(record_rows)
            + "</table>"
            + f"<p>Complete P/R/F1: {f['precision']:.3f}/{f['recall']:.3f}/{f['f1']:.3f}; underlying F1 {u['f1']:.3f}. "
            + f"Tokens {row['input_tokens']}/{row['output_tokens']}; {o['request_seconds']:.3f} seconds; finish {row['finish_reason']}.</p>"
            + f"<details><summary>Qualification, citation, omission and denominator details</summary><pre>{details}</pre></details></section>"
        )
    comparisons = {}
    for label, pair in (("original", ("5", "6")), ("fresh", ("7", "8"))):
        comparisons[label] = (
            office_comparison(by_id[pair[0]]["parsed"], by_id[pair[1]]["parsed"], pair)
            if all(k in by_id for k in pair)
            else {"verdict": "not executed completely"}
        )
    supported = []
    for label, keys in (
        ("direct regression", ["1"]),
        ("explicit-interval regression", ["2"]),
        ("belief format on original and fresh examples", ["3", "4"]),
        ("small office factual/qualification coverage on both pairs", ["5", "6", "7", "8"]),
    ):
        if all(k in by_id and by_id[k]["evaluation"]["diagnostic_level_supported"] for k in keys):
            supported.append(label)
    conclusion = (
        "Highest tested coverage meeting exploratory P/R ≥.8: "
        + (supported[-1] if supported else "none")
        + ". This is not general reliability."
    )
    md += [
        "",
        conclusion,
        "",
        "## Office comparison",
        "",
        "Both person/office reference sets permit the same office-mediated organization. Therefore the pairs are insufficient to require an ontology-construction contrast; high factual scores alone are not construction success.",
        "",
    ]
    for label, comparison in comparisons.items():
        md += [f"- {label}: **{comparison['verdict']}**.", ""]
    original_path = output / "tables/simple_synthetic_results.json"
    original = json.loads(original_path.read_text()) if original_path.exists() else None
    original_comparison = []
    if original:
        old = {r["case_id"]: r for r in original["rows"]}
        md += [
            "## Original versus revised and fresh",
            "",
            "Original frozen scores are shown without re-evaluation. V2 changes formatting and prospective normalization, so score differences are not an isolated model-capability effect.",
            "",
            "| Task | Original v1 F1 | New revised response F1 | Fresh counterpart F1 |",
            "|---|---|---|---|",
        ]
        for label, prior, revised, fresh in (
            ("belief", "6", "3", "4"),
            ("person-office", "7", "5", "7"),
            ("office-continuity", "8", "6", "8"),
        ):

            def f1(k):
                return by_id[k]["evaluation"]["full"]["f1"] if k in by_id else None

            original_comparison.append(
                dict(
                    task=label,
                    original_f1=old[prior]["f1"],
                    revised_f1=f1(revised),
                    fresh_f1=f1(fresh),
                )
            )
            md.append(f"| {label} | {old[prior]['f1']:.3f} | {f1(revised)} | {f1(fresh)} |")
    md += ["", "## Evidence, generated facts and errors", ""]
    for o in outcomes:
        k, e, c = o["case_id"], o["evaluation"], cases[o["case_id"]]
        md += [
            f"### Case {k}: {c['cohort']} / {c['level']}",
            "",
            *[f"- {sid}: {text}" for sid, text in c["evidence"].items()],
            "",
            c["question"],
            "",
            "Actual facts:",
            "",
        ]
        if o["parsed"]:
            for r in e["records"]:
                md.append(
                    "- `" + json.dumps(r["fact"], ensure_ascii=False) + "` — " + r["assessment"]
                )
                if r["format_issues"] or r["meaning_issues"]:
                    md.append("  " + "; ".join(r["format_issues"] + r["meaning_issues"]))
        else:
            md += ["```text", o["raw_text"], "```", str(o["parse_error"])]
        md += [
            "",
            "Qualification measurements: `" + json.dumps(e["qualifications"]) + "`.",
            "",
            "Missing complete facts: `"
            + json.dumps(e["full"]["missing"], ensure_ascii=False)
            + "`.",
            "",
        ]
    unexecuted = [k for k in cases if k not in by_id]
    md += [
        "## Allocation and limits",
        "",
        f"New allocated GPU time **{terminal['new_allocated_seconds']:.6f} s**; cumulative **{terminal['actual_allocated_seconds']:.6f} s**. "
        f"Open allocation/service journals: {terminal['open_allocations']}/{terminal['open_service_journals']}. "
        f"Stop: `{terminal['stop_reason']}`. Unexecuted cases: {unexecuted or 'none'}.",
        "",
        "All service time is charged; request timings are not production p95. No benchmark/registered-acceptance change, C0 change or held-out execution. Full raw outputs, references, errors and qualification details appear in the HTML and JSON.",
        "",
        "[Readable graphs and complete records](figures/simple_synthetic_v2_comparison.html).",
        "",
    ]
    output.mkdir(exist_ok=True, parents=True)
    (output / "tables").mkdir(exist_ok=True)
    (output / "figures").mkdir(exist_ok=True)
    paths = [
        output / "SIMPLE_SYNTHETIC_V2_RESULTS.md",
        output / "tables/simple_synthetic_v2_results.json",
        output / "tables/simple_synthetic_v2_results.csv",
        output / "figures/simple_synthetic_v2_comparison.html",
    ]
    paths[0].write_text("\n".join(md))
    write_json_atomic(
        dict(
            scope="development diagnostic v2 only",
            rows=rows,
            outcomes=outcomes,
            cases=cases,
            references=refs,
            frozen_rules=read("frozen-scoring.json"),
            office_comparisons=comparisons,
            original_comparison=original_comparison,
            conclusion=conclusion,
            accounting={k: v for k, v in terminal.items() if k != "outcomes"},
        ),
        paths[1],
    )
    with paths[2].open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["case_id"])
        writer.writeheader()
        writer.writerows(rows)
    paths[3].write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>' + title + "</title>"
        "<style>body{font:16px system-ui;max-width:1000px;margin:30px auto;padding:0 20px;color:#172033}section{border-top:2px solid #cbd5e1;padding:25px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f5f9;padding:15px}svg{width:100%;font:14px system-ui}.edge{paint-order:stroke;stroke:white;stroke-width:4px}table{width:100%;border-collapse:collapse}td,th{border:1px solid #ccc;padding:8px;overflow-wrap:anywhere;text-align:left}</style>"
        "<h1>"
        + title
        + "</h1><p>Development diagnostics only; all executed outcomes are shown.</p><p>"
        + html.escape(conclusion)
        + "</p>"
        "<p>Identical office-mediated graphs satisfy both frozen question/reference sets; factual accuracy alone is not ontology-construction evidence.</p>"
        + "<pre>"
        + html.escape(json.dumps(comparisons, indent=2))
        + "</pre>"
        + "".join(sections)
        + "</html>"
    )
    write_json_atomic(
        {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        output / "tables/simple_synthetic_v2_manifest.json",
    )
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("reports"))
    args = p.parse_args()
    render(args.run, args.output)
