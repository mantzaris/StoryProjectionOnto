"""Render all compact-story attempts and fixed selections; no inference or rescoring v1/v2."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import textwrap
from pathlib import Path

from story_projection_onto import compact_story as p
from story_projection_onto.contracts import canonical_sha256
from story_projection_onto.manifest import write_json_atomic
from story_projection_onto.scorer_only import compact_story as scorer


def fact_text(fact):
    if not isinstance(fact, dict):
        return json.dumps(fact)
    text = f"{fact.get('subject', '?')} — {fact.get('relation', '?')} → {fact.get('object', '?')}"
    if "valid_from" in fact or "valid_until" in fact:
        text += f" [{fact.get('valid_from', '?')}, {fact.get('valid_until', '?')})"
    if "holder" in fact or "attitude" in fact:
        text += f"; holder={fact.get('holder', '?')}, attitude={fact.get('attitude', '?')}"
    return text + "; evidence=" + json.dumps(fact.get("evidence_ids", "MISSING"))


def graph(facts, anchors, evaluation=None):
    """Exact endpoint strings only; common question-level anchors across A/B/reference."""
    e = html.escape
    status = {r["index"]: r["status"] for r in (evaluation or {}).get("rows", [])}
    pieces = [
        '<svg viewBox="0 0 850 580" role="img" aria-label="Generated fact graph"><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#66788c"/></marker></defs>'
    ]
    active = set()
    for i, row in enumerate(facts):
        if not isinstance(row, dict) or not all(
            isinstance(row.get(k), str) and row[k] in anchors for k in ("subject", "object")
        ):
            continue
        s, o = row["subject"], row["object"]
        active.update((s, o))
        x1, y1 = anchors[s]
        x2, y2 = anchors[o]
        dx, dy = x2 - x1, y2 - y1
        distance = max(1, math.hypot(dx, dy))
        a, b = (
            (x1 + dx / distance * 30, y1 + dy / distance * 30),
            (x2 - dx / distance * 35, y2 - dy / distance * 35),
        )
        color = (
            "#137a4c"
            if status.get(i) == "correct_complete_fact"
            else "#986000"
            if status.get(i) == "unresolved_wording"
            else "#b52b35"
            if evaluation
            else "#66788c"
        )
        # Numeric edge labels map directly to the readable assertion table, avoiding prose overlap.
        pieces.append(
            f'<line x1="{a[0]}" y1="{a[1]}" x2="{b[0]}" y2="{b[1]}" stroke="{color}" stroke-width="2" marker-end="url(#arrow)"/>'
        )
        mx, my = (x1 + x2) / 2 + (i % 3 - 1) * 12, (y1 + y2) / 2 + (i % 3 - 1) * 17
        pieces.append(
            f'<circle cx="{mx}" cy="{my}" r="12" fill="white"/><text x="{mx}" y="{my + 5}" text-anchor="middle" fill="{color}">{i + 1}</text>'
        )
    for name in sorted(active):
        x, y = anchors[name]
        lines = textwrap.wrap(name, 20)
        height = max(42, 18 * len(lines) + 12)
        pieces.append(
            f'<rect x="{x - 75}" y="{y - height / 2}" width="150" height="{height}" rx="8" fill="#e9f4ff" stroke="#4d7795"/><text x="{x}" y="{y - (len(lines) - 1) * 9 + 5}" text-anchor="middle">'
            + "".join(
                f'<tspan x="{x}" dy="{0 if j == 0 else 18}">{e(line)}</tspan>'
                for j, line in enumerate(lines)
            )
            + "</text>"
        )
    pieces.append("</svg>")
    return (
        '<button onclick="this.nextElementSibling.requestFullscreen()">Enlarge graph (Esc to return)</button>'
        + "".join(pieces)
        if active
        else "<p>No mechanically displayable endpoint records.</p>"
    )


def anchors_for(collections):
    names = sorted(
        {
            r[k]
            for facts in collections
            for r in facts
            if isinstance(r, dict)
            for k in ("subject", "object")
            if isinstance(r.get(k), str)
        }
    )
    return {
        name: (
            425 + 315 * math.cos(2 * math.pi * i / max(1, len(names))),
            290 + 235 * math.sin(2 * math.pi * i / max(1, len(names))),
        )
        for i, name in enumerate(names)
    }


def build(run):
    def read(name):
        return json.loads((run / name).read_text())

    terminal = read("terminal.json")
    cases, frozen_refs, rules = (
        read("frozen-cases.json"),
        read("reference-answers-not-transmitted.json"),
        read("frozen-scoring.json"),
    )
    if cases != p.cases() or frozen_refs != scorer.references() or rules != scorer.frozen_rules():
        raise ValueError("report code differs from frozen evaluation/request artifacts")
    calls = {o["case_id"]: o for o in terminal["outcomes"]}
    diagnosis = read("manual-diagnosis.json") if (run / "manual-diagnosis.json").exists() else {}
    for k, item in diagnosis.get("cases", {}).items():
        if item["response_sha256"] != (calls[k]["response"] or {}).get("response_sha256"):
            raise ValueError("manual diagnostic note does not match retained response")
    records = []
    for story_id in p.stories():
        base_id = next(
            k for k, c in cases.items() if c["story_id"] == story_id and c["task"] == "all"
        )
        base = calls.get(base_id)
        if base:
            seal = read(f"preextract-seal-{base_id}.json")
            if seal["outcome_hash"] != canonical_sha256(base) or seal[
                "parsed_hash"
            ] != canonical_sha256(base["parsed"]):
                raise ValueError("pre-extraction seal mismatch")
        for task in p.questions(p.stories()[story_id]):
            contextual_id = next(
                k for k, c in cases.items() if c["story_id"] == story_id and c["task"] == task
            )
            for approach, call_id in (
                ("A pre-extract/select", base_id),
                ("B contextual", contextual_id),
            ):
                outcome = calls.get(call_id)
                original = outcome.get("parsed") if outcome else None
                parsed, trace = (
                    p.select(original, story_id, task)
                    if approach.startswith("A") and original
                    else (original, [])
                )
                evaluation = scorer.evaluate_answer(story_id, task, parsed)
                response = (outcome or {}).get("response") or {}
                records.append(
                    {
                        "story_id": story_id,
                        "task": task,
                        "approach": approach,
                        "source_case_id": call_id,
                        "question": p.questions(p.stories()[story_id])[task],
                        "executed": outcome is not None,
                        "transport_complete": (outcome or {}).get("transport_complete", False),
                        "finish_reason": response.get("finish_reason"),
                        "parseable": original is not None,
                        "parsed": parsed,
                        "selection_trace": trace,
                        "evaluation": evaluation,
                        "raw_source_text": (outcome or {}).get("raw_text", ""),
                        "input_tokens": response.get("prompt_tokens", 0),
                        "output_tokens": response.get("completion_tokens", 0),
                        "request_seconds": (outcome or {}).get("request_seconds"),
                        "source_response_hash": response.get("response_sha256"),
                        "failure": (outcome or {}).get("failure")
                        or (outcome or {}).get("parse_error"),
                    }
                )
    return {
        "protocol": "compact-story-exploratory-v1",
        "registered_acceptance": False,
        "stories": p.stories(),
        "rules": rules,
        "references": frozen_refs,
        "packing": read("packing.json"),
        "calls": list(calls.values()),
        "manual_diagnosis": diagnosis,
        "results": records,
        "allocation": {
            k: terminal[k]
            for k in (
                "actual_allocated_seconds",
                "new_allocated_seconds",
                "open_allocations",
                "open_service_journals",
                "stop_reason",
            )
        },
        "resource_samples": terminal["resource_samples"],
    }


def render(run, output):
    data = build(run)
    output.mkdir(parents=True, exist_ok=True)
    (output / "tables").mkdir(exist_ok=True)
    (output / "figures").mkdir(exist_ok=True)
    table = []
    for r in data["results"]:
        ev = r["evaluation"]
        row = {
            k: r[k]
            for k in (
                "story_id",
                "task",
                "approach",
                "source_case_id",
                "executed",
                "transport_complete",
                "finish_reason",
                "parseable",
                "input_tokens",
                "output_tokens",
                "request_seconds",
            )
        }
        row.update(
            predictions=ev["full"]["predicted"],
            citation_valid=ev["citation_valid"],
            correct=ev["full"]["true_positive"],
            reference_count=ev["full"]["reference_count"],
            source_supported=ev["source_supported"],
            unresolved=len(ev["unresolved"]),
            confirmed_errors=len(ev["confirmed_errors"]),
            temporal_errors=sum(x["temporal_correct"] is False for x in ev["rows"]),
            attribution_errors=sum(x["epistemic_correct"] is False for x in ev["rows"]),
        )
        for prefix, key in (("bare", "underlying"), ("complete", "full")):
            row.update({prefix + "_" + m: ev[key][m] for m in ("precision", "recall", "f1")})
        table.append(row)
    write_json_atomic(data, output / "tables/compact_story_results.json")
    with (output / "tables/compact_story_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(table)
    md = [
        "# Compact story results — exploratory development",
        "",
        "A = query-blind pre-extraction followed by fixed CPU selection. B = direct contextual extraction. These are not registered C1/C2 results or evidence of ontology construction.",
        "",
        "| Story / question | Approach | Complete / parseable | Valid citations | Bare P/R/F1 | Qualified P/R/F1 |",
        "|---|---|---|---|---|---|",
    ]
    for r in table:
        measures = [
            " / ".join(f"{r[prefix + '_' + m]:.3f}" for m in ("precision", "recall", "f1"))
            for prefix in ("bare", "complete")
        ]
        md.append(
            f"| {r['story_id']} / {r['task']} | {r['approach']} | {r['transport_complete']} / {r['parseable']} | {r['citation_valid']}/{r['predictions']} | {measures[0]} | {measures[1]} |"
        )
    md += ["", "## What the comparison shows", ""]
    md += [
        data["manual_diagnosis"].get("summary", ""),
        "",
        "Unparseable answers retain the frozen zero-score answer-failure convention. Their per-fact denominator is not measurable; zero parsed records does not mean no candidate text was emitted. No trailing-comma removal or semantic repair was applied. Empty CPU selections are separate: their omission/zero-recall scores are measured outputs of the frozen filter.",
        "",
    ]
    for story_id in p.stories():
        means = {
            a: sum(
                r["complete_f1"]
                for r in table
                if r["story_id"] == story_id and r["approach"].startswith(a)
            )
            / 3
            for a in ("A", "B")
        }
        direction = (
            "higher" if means["B"] > means["A"] else "lower" if means["B"] < means["A"] else "equal"
        )
        md += [
            f"{story_id}: mean question-level qualified F1 is {means['A']:.3f} for A and {means['B']:.3f} for B ({direction}). This describes these three answers, not a general reliability or significance result.",
            "",
        ]
    md += [
        "Differences in target score can reflect relevance selection, omissions, or qualification errors; consult the assertion-level statuses below. Supported but irrelevant facts remain precision errors. Unrecognized wording remains unresolved, not automatically false or accepted. All predictions and duplicate records stay in their applicable denominators.",
        "",
        "## Readable stories and answers",
        "",
    ]
    examples = data["manual_diagnosis"].get("examples", [])
    if examples:
        # Put concrete examples immediately after the results, before the full inventory.
        position = md.index("## Readable stories and answers")
        lines = ["## Three readable examples", ""]
        for example in examples:
            lines += [
                f"- Evidence: {example['source']}\n  Generated: {example['actual']}\n  Assessment: {example['interpretation']}",
                "",
            ]
        md[position:position] = lines
    e = html.escape
    body = [
        '<!doctype html><meta charset="utf-8"><title>Compact story comparison</title><style>body{font:16px system-ui;margin:24px;color:#192b3c}table{border-collapse:collapse;width:100%;table-layout:fixed}td,th{border:1px solid #b8c4d0;padding:9px;vertical-align:top;overflow-wrap:anywhere}.compare{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.panel{border:1px solid #bbc8d3;padding:10px}svg{width:100%;font:14px system-ui}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}li{margin-bottom:12px}h2{border-top:2px solid #496c88;padding-top:20px}@media(max-width:1200px){.compare{grid-template-columns:1fr}}</style><h1>Compact story comparison</h1><p>Exploratory development; not registered ontology acceptance. Green edges: complete target match; amber: unresolved wording; red: error or irrelevant. Numbered edges map to every actual assertion below. Exact names share fixed anchors across each comparison.</p>'
    ]
    body += ["<p>" + e(data["manual_diagnosis"].get("summary", "")) + "</p>"]
    body += [
        "<style>svg:fullscreen{width:100vw;height:100vh;background:white}button{padding:6px 12px;cursor:pointer}svg text{paint-order:stroke;stroke:white;stroke-width:1px}</style>"
    ]
    for story_id, story in p.stories().items():
        md += (
            [f"### {story['title']}", ""]
            + [f"- {sid}: {sentence}" for sid, sentence in story["evidence"].items()]
            + [""]
        )
        body += (
            [f"<h2>{e(story['title'])}</h2><ol>"]
            + [
                f"<li><b>{sid}</b>: {e(sentence)}</li>"
                for sid, sentence in story["evidence"].items()
            ]
            + ["</ol>"]
        )
        for task, question in p.questions(story).items():
            pair = [r for r in data["results"] if r["story_id"] == story_id and r["task"] == task]
            refs = scorer.reference_for(story_id, task)
            collections = [refs] + [(r["parsed"] or {}).get("facts", []) for r in pair]
            anchors = anchors_for(collections)
            md += (
                [f"#### {task}: {question}", "", "Reference answers:", ""]
                + ["- " + fact_text(r) for r in refs]
                + [""]
            )
            body += (
                [
                    f'<h3>{e(question)}</h3><div class="compare"><section class="panel"><h4>Reference answers</h4>',
                    graph(refs, anchors),
                    "<ol>",
                ]
                + [f"<li>{e(fact_text(r))}</li>" for r in refs]
                + ["</ol></section>"]
            )
            for r in pair:
                ev = r["evaluation"]
                md += [r["approach"] + ":", ""]
                note = (
                    data["manual_diagnosis"]
                    .get("cases", {})
                    .get(r["source_case_id"], {})
                    .get("finding")
                )
                if note:
                    md += ["Manual source-output diagnosis (not rescoring): " + note, ""]
                body += [
                    f'<section class="panel" id="{story_id}-{task}-{r["approach"][0]}"><h4>{e(r["approach"])}</h4>',
                    graph((r["parsed"] or {}).get("facts", []), anchors, ev),
                    "<ol>",
                ]
                for row in ev["rows"]:
                    detail = f"{fact_text(row['fact'])} — {row['status']}" + (
                        "; " + "; ".join(row["issues"]) if row["issues"] else ""
                    )
                    md.append("- " + detail)
                    body.append(f"<li>{e(detail)}</li>")
                if not r["parseable"]:
                    md += [
                        "No complete parseable source response. Raw received output:",
                        "```text",
                        r["raw_source_text"],
                        "```",
                    ]
                md += ["", "Missing complete facts: " + json.dumps(ev["full"]["missing"]), ""]
                seconds = (
                    f"{r['request_seconds']:.3f}s"
                    if r["request_seconds"] is not None
                    else "not executed"
                )
                metrics = f"Qualified P/R/F1: {ev['full']['precision']:.3f}/{ev['full']['recall']:.3f}/{ev['full']['f1']:.3f}; tokens {r['input_tokens']}/{r['output_tokens']}; time {seconds}; finish {r['finish_reason']}."
                body += [
                    "</ol>",
                    "<p><em>Manual source-output diagnosis (not rescoring): "
                    + e(note)
                    + "</em></p>"
                    if note
                    else "",
                    f"<p>{e(metrics)}</p>",
                    "<details><summary>Missing facts / raw source response / full evaluation and selection trace</summary><pre>"
                    + e(
                        json.dumps(
                            {
                                "missing": ev["full"]["missing"],
                                "raw_source": r["raw_source_text"],
                                "evaluation": ev,
                                "selection_trace": r["selection_trace"],
                            },
                            indent=2,
                        )
                    )
                    + "</pre></details></section>",
                ]
            body += ["</div>"]
        base_id = next(
            k for k, c in p.cases().items() if c["story_id"] == story_id and c["task"] == "all"
        )
        base = next((o for o in data["calls"] if o["case_id"] == base_id), None)
        base_facts = ((base or {}).get("parsed") or {}).get("facts", [])
        body += [
            f'<h3>Complete query-blind source graph: {e(story["title"])}</h3><section class="panel" id="{story_id}-preextract">',
            graph(base_facts, anchors_for([base_facts]), (base or {}).get("evaluation")),
            "<ol>",
        ]
        body += [f"<li>{e(fact_text(row))}</li>" for row in base_facts]
        body += ["</ol></section>"]
        md += [
            "Query-blind source collection (actual, before CPU selection):",
            "```json",
            (base or {}).get("raw_text", "NOT EXECUTED"),
            "```",
            "",
        ]
        body += [
            "<details><summary>Actual full query-blind source collection and extraction score</summary><pre>"
            + e(json.dumps(base, indent=2))
            + "</pre></details>"
        ]
    input_tokens = sum((c.get("response") or {}).get("prompt_tokens", 0) for c in data["calls"])
    output_tokens = sum(
        (c.get("response") or {}).get("completion_tokens", 0) for c in data["calls"]
    )
    md += [
        "## Unique model calls",
        "",
        "| Call | Story/task | Completed / parseable | Finish | Tokens in/out | Request seconds |",
        "|---|---|---|---|---|---|",
    ]
    for call in data["calls"]:
        response = call.get("response") or {}
        case = p.cases()[call["case_id"]]
        md.append(
            f"| {call['case_id']} | {case['story_id']}/{case['task']} | {call['transport_complete']} / {call['parsed'] is not None} | {response.get('finish_reason')} | {response.get('prompt_tokens', 0)}/{response.get('completion_tokens', 0)} | {call['request_seconds']:.3f} |"
        )
    md += ["", "Query-blind extraction scores before selection:", ""]
    for call in data["calls"]:
        if p.cases()[call["case_id"]]["task"] == "all":
            ev = call["evaluation"]
            md.append(
                f"- Call {call['case_id']}: bare {ev['underlying']['true_positive']}/{ev['underlying']['predicted']}; complete qualified {ev['full']['true_positive']}/{ev['full']['predicted']}; valid citations {ev['citation_valid']}/{ev['full']['predicted']}."
            )
    passes = sum(r["complete_precision"] >= 0.8 and r["complete_recall"] >= 0.8 for r in table)
    md += [
        "",
        f"Complete qualified facts meet the exploratory 80% precision-and-recall criterion in {passes}/{len(table)} question/approach rows. Parseability, bare extraction and qualification are distinct outcomes; this small sample does not establish dependable temporal/epistemic formatting or contextual selection.",
        "",
    ]
    peaks = {
        k: max((sample[k] for sample in data["resource_samples"]), default=0)
        for k in ("gpu_vram_bytes", "process_ram_bytes", "project_storage_bytes")
    }
    md += ["Sampled resource peaks (bytes): `" + json.dumps(peaks, sort_keys=True) + "`.", ""]
    md += [
        "## Execution and boundaries",
        "",
        f"Executed calls: {len(data['calls'])}/8, one attempt each; no repairs. Unique-call input/output tokens: {input_tokens}/{output_tokens}. Baseline request costs shown on each of its three selections are shared, not three additional calls.",
        "",
        f"Additional allocated service time: {data['allocation']['new_allocated_seconds']:.6f} seconds; cumulative: {data['allocation']['actual_allocated_seconds']:.6f} seconds. Open allocation/service journals: {data['allocation']['open_allocations']}/{data['allocation']['open_service_journals']}. Stop reason: {data['allocation']['stop_reason']}. Exact retained values are in the JSON table.",
        "",
        "One 900-second envelope including loading and shutdown; all historical allocation retained. Same pinned Qwen3-8B-AWQ revision 4da05a8edb55c6046cce958586c33b61da07bb79, nonthinking, 12,288 context, seed 1988649846, temperature .7/top-p .8/top-k 20. Unconstrained JSON and a common 2,048-output allowance. Exact rendered packing and raw call timings are in the JSON table.",
        "",
        "Selection uses relation category, explicit interval overlap without clipping, and attributed-subject matching for belief/reality. It does not fix wrong facts, infer aliases, invent qualifications or deduplicate. Bare matching ignores qualifications/citations; qualified matching requires their source-supported values. Required missing content remains a recall failure. An absent time/holder field is not a literal null or an invented unknown value.",
        "",
        "Two fresh but deliberately clear, similarly structured stories cannot establish general reliability. No office continuity, event construction, identity merging, causal reasoning or registered ontology capability was tested. Original ladders, C0 scores, primary metrics, resource ceilings and held-out human-review gate are unchanged.",
        "",
        "Readable side-by-side graphs: [HTML](figures/compact_story_comparison.html). Exact tables: [CSV](tables/compact_story_results.csv), [JSON](tables/compact_story_results.json).",
        "",
        "CPU reproduction: `python -m scripts.report_compact_story --run <restricted-run-directory> --output reports`.",
    ]
    (output / "COMPACT_STORY_RESULTS.md").write_text("\n".join(md) + "\n")
    body += ["<h2>Accounting</h2><pre>" + e(json.dumps(data["allocation"], indent=2)) + "</pre>"]
    (output / "figures/compact_story_comparison.html").write_text("\n".join(body))
    files = [
        "COMPACT_STORY_RESULTS.md",
        "tables/compact_story_results.csv",
        "tables/compact_story_results.json",
        "figures/compact_story_comparison.html",
    ]
    write_json_atomic(
        {
            "files": {f: hashlib.sha256((output / f).read_bytes()).hexdigest() for f in files},
            "frozen_rules_hash": canonical_sha256(data["rules"]),
            "calls": len(data["calls"]),
        },
        output / "tables/compact_story_manifest.json",
    )
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports"))
    args = parser.parse_args()
    render(args.run, args.output)
