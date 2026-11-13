"""Reproduce the separate diagnostic report from immutable ladder records."""

import argparse
import csv
import hashlib
import html
import json
import math
import textwrap
from pathlib import Path

from story_projection_onto.manifest import write_json_atomic


def dump(value):
    return json.dumps(value, ensure_ascii=False, indent=2)


def graph(facts):
    # Exact strings only. No case normalization, aliases or inferred nodes.
    names = list(
        dict.fromkeys(
            f[k]
            for f in facts
            if isinstance(f, dict)
            for k in ("subject", "object")
            if isinstance(f.get(k), str)
        )
    )
    if not names:
        return "<p>No mechanically available endpoint graph.</p>"
    xy = {
        s: (
            430 + 300 * math.cos(2 * math.pi * i / len(names)),
            200 + 150 * math.sin(2 * math.pi * i / len(names)),
        )
        for i, s in enumerate(names)
    }
    out = [
        '<svg viewBox="0 0 860 420" role="img" aria-label="Actual generated relationships">',
        '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L0,6 L8,3 z" fill="#64748b"/></marker></defs>',
    ]
    for i, f in enumerate(facts):
        if (
            not isinstance(f, dict)
            or not all(isinstance(f.get(k), str) for k in ("subject", "object"))
            or f["subject"] not in xy
            or f["object"] not in xy
        ):
            continue
        x, y = xy[f["subject"]]
        a, b = xy[f["object"]]
        distance = max(1, math.hypot(a - x, b - y))
        boundary = min(
            95 / abs(a - x) if a != x else math.inf, 27 / abs(b - y) if b != y else math.inf
        )
        # End outside the rectangular node so the arrowhead is not painted over.
        ex = a - (a - x) * boundary - (a - x) * 5 / distance
        ey = b - (b - y) * boundary - (b - y) * 5 / distance
        out.append(
            f'<path d="M{x},{y} L{ex},{ey}" stroke="#64748b" fill="none" marker-end="url(#arrow)"/>'
        )
        label = html.escape(str(f.get("relation", "?")))
        out.append(
            f'<text x="{(x + a) / 2}" y="{(y + b) / 2 - 6 - (i % 2) * 13}" text-anchor="middle" class="edge">{i + 1}. {label}</text>'
        )
    for name, (x, y) in xy.items():
        out.append(
            f'<rect x="{x - 95}" y="{y - 27}" width="190" height="54" rx="8" fill="#e0f2fe" stroke="#0369a1"/>'
        )
        lines = textwrap.wrap(name, 24) or [name]
        for i, line in enumerate(lines):
            out.append(
                f'<text x="{x}" y="{y + 5 + (i - (len(lines) - 1) / 2) * 17}" text-anchor="middle">{html.escape(line)}</text>'
            )
    return "".join(out) + "</svg>"


def render(run, output):
    def read(name):
        return json.loads((run / name).read_text())

    cases, refs, terminal = (
        read("frozen-cases.json"),
        read("reference-answers-not-transmitted.json"),
        read("terminal.json"),
    )
    outcomes = terminal["outcomes"]
    tokenizer = (
        read("rendered-1.json").get("tokenizer_manifest", {})
        if (run / "rendered-1.json").exists()
        else {}
    )
    first_request = read("request-1.json")
    notes_path = run / "manual-diagnostic-notes.json"
    notes = read(notes_path.name) if notes_path.exists() else {"cases": {}}
    for k, note in notes["cases"].items():
        actual = next(o for o in outcomes if o["case_id"] == k)
        if note["response_sha256"] != actual["response"]["response_sha256"]:
            raise ValueError("manual diagnosis is bound to a different response")
    rows = []
    sections = []
    md = [
        "# Simple synthetic capability ladder",
        "",
        "Exploratory pedagogical development diagnostics—not registered C0/C1/C2 results, canonical ontologies, or held-out evidence.",
        "",
        f"Model: `{tokenizer.get('repository', first_request['model'])}@{tokenizer.get('revision', 'see run manifest')}`. "
        f"Nonthinking; seed {first_request['seed']}; temperature {first_request['temperature']}; "
        f"top-p {first_request['top_p']}; top-k {first_request['top_k']}; no guided schema.",
        "",
        "All eight requests, conservative matching rules, reference alternatives and the conditional plain-language control were frozen before inference. No guided JSON schema, expected answer, ontology contract or model repair was transmitted. Identical endpoint strings alone define graph nodes.",
        "",
        "Direct-fact progression requires combined precision and recall ≥80%. Later levels use the same exploratory P/R criterion on complete qualified facts. Every emitted fact remains in the denominator; duplicate, malformed and unresolved facts are not silently removed. Matching permits case/whitespace normalization and the frozen relation synonyms only. An unmatched wording is not automatically a false statement.",
        "",
        "| Case | Level | Finish | Correct / extracted / reference | Precision | Recall | F1 | Input / output tokens | Seconds |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for o in outcomes:
        k, case = o["case_id"], cases[o["case_id"]]
        ev = o.get("evaluation")
        full = ev["full"] if ev else {}
        response = o.get("response") or {}
        row = {
            "case_id": k,
            "level": case["level"],
            "complete": o["transport_complete"],
            "finish_reason": response.get("finish_reason"),
            "json_parseable": o["parsed"] is not None,
            **{
                f: full.get(f)
                for f in (
                    "true_positive",
                    "predicted",
                    "reference_count",
                    "precision",
                    "recall",
                    "f1",
                )
            },
            "input_tokens": response.get("prompt_tokens"),
            "output_tokens": response.get("completion_tokens"),
            "output_allowance": case["output_allowance"],
            "request_seconds": o["request_seconds"],
        }
        rows.append(row)

        def metric(f, values=full):
            return f"{values[f]:.3f}" if f in values else "not scored"

        md.append(
            f"| {k} | {case['level']} | {response.get('finish_reason', 'failed')} | {full.get('true_positive', '—')} / {full.get('predicted', '—')} / {full.get('reference_count', '—')} | {metric('precision')} | {metric('recall')} | {metric('f1')} | {row['input_tokens']} / {row['output_tokens']} | {row['request_seconds']:.3f} |"
        )
        predicted = o["parsed"]["facts"] if o["parsed"] else []
        evidence = "\n".join(f"{sid}: {s}" for sid, s in case["evidence"].items())
        reference_text = dump(
            refs.get(k, "Plain-language control; inspect against the same direct evidence.")
        )
        sections.append(
            f'<section id="case-{k}"><h2>Case {k}: {case["level"]}</h2><pre>{html.escape(evidence)}</pre>'
            f"<p><strong>Question:</strong> {html.escape(case['question'])}</p>"
            f"<details><summary>Reference answers (frozen alternatives, not model output)</summary><pre>{html.escape(reference_text)}</pre></details>"
            f"<h3>Actual model output</h3><pre>{html.escape(o['raw_text'])}</pre>{graph(predicted)}"
            f"<h3>Measured result</h3><p>Precision {metric('precision')}; recall {metric('recall')}; F1 {metric('f1')}.</p>"
            f"<details><summary>Correct, incorrect/unmatched, missing facts and qualification checks</summary><pre>{html.escape(dump(ev or o.get('failure')))}</pre></details>"
            f"<p>Tokens {row['input_tokens']} / {row['output_tokens']}; output cap {row['output_allowance']}; "
            f"{row['request_seconds']:.3f} seconds; finish {html.escape(str(row['finish_reason']))}; "
            f"normalization: {html.escape(o['normalization'])}.</p></section>"
        )
        if k in notes["cases"]:
            sections[-1] = sections[-1].replace(
                "</section>",
                "<h3>Manual diagnostic explanation (not rescoring)</h3><p>"
                + html.escape(notes["cases"][k]["finding"])
                + "</p></section>",
            )
    executed = {o["case_id"] for o in outcomes}
    levels = [
        ("direct extraction", ["1", "2"]),
        ("contextual selection", ["3", "4"]),
        ("explicit temporal intervals", ["5"]),
        ("belief versus reality", ["6"]),
        ("small office representation contrast", ["7", "8"]),
    ]
    by_id = {o["case_id"]: o for o in outcomes}
    supported = []
    for name, ids in levels:
        if all(
            i in by_id and (by_id[i].get("evaluation") or {}).get("diagnostic_level_supported")
            for i in ids
        ):
            supported.append(name)
    first_failure = next(
        (
            o["case_id"]
            for o in outcomes
            if o.get("evaluation") and o["evaluation"]["full"]["f1"] < 1
        ),
        None,
    )
    conclusion = (
        f"Highest tested level meeting the declared exploratory criterion: {supported[-1] if supported else 'none'}. "
        f"First observed scored error: {'case ' + first_failure if first_failure else 'none among executed cases'}. "
        "This does not establish model reliability, production ontology capacity, construction efficacy or p95 runtime."
    )
    md += [
        "",
        conclusion,
        "",
        f"Parseable JSON fact lists: {sum(o['parsed'] is not None for o in outcomes)}/{len(outcomes)}. "
        "The scores require the supplied evidence IDs; “underlying” drops temporal/epistemic qualifications but still checks citations. "
        "The manual notes below distinguish contract failures from factual errors without changing the frozen scores.",
        "",
        f"Not executed: {', '.join(k for k in cases if k not in executed) or 'none'}. Stop reason: `{terminal['stop_reason']}`.",
        "",
        f"New allocated GPU time: **{terminal['new_allocated_seconds']:.6f} s**. Cumulative: **{terminal['actual_allocated_seconds']:.6f} s**. "
        f"Open allocation/service journals: {terminal['open_allocations']}/{terminal['open_service_journals']}. "
        "Startup, idle time, monitoring, generation and shutdown are included; request times are not total allocation.",
        "",
        "Sampled peaks (bytes): "
        + ", ".join(
            f"{label} {max((s[field] for s in terminal.get('resource_samples', [])), default=0)}"
            for label, field in (
                ("VRAM", "gpu_vram_bytes"),
                ("process RAM", "process_ram_bytes"),
                ("project storage", "project_storage_bytes"),
            )
        )
        + ".",
        "",
        "## Evidence and actual answers",
        "",
    ]
    for o in outcomes:
        k, c = o["case_id"], cases[o["case_id"]]
        md += [
            f"### Case {k}: {c['level']}",
            "",
            *[f"{sid}: {s}  " for sid, s in c["evidence"].items()],
            "",
            c["question"],
            "",
            "Actual extracted relationships (verbatim JSON is in the HTML):",
            "",
        ]
        if o["parsed"]:
            for f in o["parsed"]["facts"]:
                if isinstance(f, dict):
                    qualifications = {
                        a: b for a, b in f.items() if a not in ("subject", "relation", "object")
                    }
                    md.append(
                        f"- {f.get('subject')} → **{f.get('relation')}** → {f.get('object')}; `{json.dumps(qualifications)}`"
                    )
                else:
                    md.append(f"- Malformed record: `{json.dumps(f)}`")
        else:
            md += ["```text", o["raw_text"], "```"]
        md.append("")
        if o.get("evaluation"):
            e = o["evaluation"]
            md += [
                f"Complete facts: {e['full']['true_positive']}/{e['full']['reference_count']}; "
                f"underlying relationships: {e['underlying']['true_positive']}/{e['underlying']['reference_count']}.",
                "",
                f"Unmatched predictions: {len(e['full']['incorrect_or_unmatched'])}; "
                f"missing complete reference facts: {len(e['full']['missing'])}; "
                f"automatically unresolved: {len(e['unresolved'])}. Full fact-level lists are in the HTML/JSON.",
                "",
            ]
        if k in notes["cases"]:
            md += ["Manual diagnosis (not rescoring): " + notes["cases"][k]["finding"], ""]
    baseline = read("toy-baseline.json")
    md += [
        "## Toy template baseline",
        "",
        "This three-pattern sentence parser is not registered C0; it reads the actual sentences and does not look up stored answers.",
        "",
    ]
    for k, b in baseline.items():
        f = b["evaluation"]["full"]
        md.append(
            f"Case {k}: {f['true_positive']}/{f['predicted']} precision, {f['true_positive']}/{f['reference_count']} recall; F1={f['f1']:.3f}."
        )
    md += [
        "",
        "Reference answers and every full scoring record are in the accompanying HTML and machine-readable JSON. The synthetic sources and expected answers are diagnostic material, separate from the original benchmark. C0 extraction competence, contextual results, historical failures, global ceilings and held-out review remain unchanged.",
        "",
        "Graph comparison: [simple_synthetic_comparison.html](figures/simple_synthetic_comparison.html).",
        "",
    ]
    output.mkdir(exist_ok=True, parents=True)
    (output / "tables").mkdir(exist_ok=True)
    (output / "figures").mkdir(exist_ok=True)
    paths = [
        output / "SIMPLE_SYNTHETIC_RESULTS.md",
        output / "tables/simple_synthetic_results.json",
        output / "tables/simple_synthetic_results.csv",
        output / "figures/simple_synthetic_comparison.html",
    ]
    paths[0].write_text("\n".join(md))
    write_json_atomic(
        {
            "scope": "exploratory diagnostics only",
            "model": {
                "repository": tokenizer.get("repository"),
                "revision": tokenizer.get("revision"),
                "seed": first_request["seed"],
                "template_sha256": tokenizer.get("chat_template_sha256"),
            },
            "rows": rows,
            "cases": cases,
            "references": refs,
            "outcomes": outcomes,
            "conclusion": conclusion,
            "toy_baseline": baseline,
            "manual_diagnostic_notes": notes,
            "accounting": {k: v for k, v in terminal.items() if k != "outcomes"},
        },
        paths[1],
    )
    with paths[2].open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["case_id"])
        writer.writeheader()
        writer.writerows(rows)
    paths[3].write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Simple synthetic ladder</title>'
        "<style>body{font:16px system-ui;max-width:1000px;margin:30px auto;padding:0 20px;color:#172033}"
        "section{border-top:2px solid #cbd5e1;padding:25px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f5f9;padding:15px}"
        "svg{width:100%;font:14px system-ui}.edge{paint-order:stroke;stroke:white;stroke-width:4px;stroke-linejoin:round}</style>"
        "<h1>Simple synthetic capability ladder</h1><p>Exploratory development only. All executed outputs shown; reference answers are not model outputs.</p>"
        + f"<p>{html.escape(conclusion)}</p>"
        + "".join(sections)
        + "</html>"
    )
    write_json_atomic(
        {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        output / "tables/simple_synthetic_manifest.json",
    )
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports"))
    args = parser.parse_args()
    render(args.run, args.output)
