"""Generate safe exploratory tables/HTML from actual immutable development artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import math
import os
import re
import tempfile
from pathlib import Path

from story_projection_onto.contracts import EvidenceRecord, OntologyDraft
from story_projection_onto.development_adapter import DevelopmentConstructionConfiguration
from story_projection_onto.development_demo import read, sources, without_admin
from story_projection_onto.manifest import write_json_atomic
from story_projection_onto.scorer_only.development_demo_assessment import assess
from story_projection_onto.store import ArtifactRecord, BlobStore, Compression, ReleaseClass


def atomic_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
        temp = Path(f.name)
    temp.replace(path)


def streamed_content(run, request_hash):
    """Hash-check retained HTTP bytes and extract actual SSE content, no completion."""
    journals = sorted((run / "http").glob(request_hash[:16] + "-*/events.jsonl"))
    if not journals:
        return ""
    journal = journals[-1]
    blobs = BlobStore(journal.parent / "fragments")
    parts = []
    for line in journal.read_text().splitlines():
        row = json.loads(line)
        if row["event"] != "response_fragment":
            continue
        a = row["artifact"]
        a["compression"] = Compression(a["compression"])
        a["release_class"] = ReleaseClass(a["release_class"])
        parts.append(blobs.read_bytes(ArtifactRecord(**a), allow_restricted=True))
    content = []
    for line in b"".join(parts).decode("utf-8", errors="replace").splitlines():
        if not line.startswith("data:"):
            continue
        try:
            row = json.loads(line[5:])
        except json.JSONDecodeError:
            continue
        for choice in row.get("choices", []):
            value = choice.get("delta", {}).get("content")
            if isinstance(value, str):
                content.append(value)
    return "".join(content)


def complete_record_fragments(text):
    """Only complete array members from truncated JSON; explicitly not a draft."""
    groups = {}
    for field in (
        "entities",
        "events",
        "assertions",
        "contextual_types",
        "predicates",
        "decisions",
    ):
        match = re.search(r'"' + field + r'"\s*:\s*\[', text)
        out = []
        if match:
            rest = text[match.end() :].lstrip()
            while rest.startswith("{"):
                try:
                    value, end = json.JSONDecoder().raw_decode(rest)
                except json.JSONDecodeError:
                    break
                out.append(value)
                rest = rest[end:].lstrip()
                if not rest.startswith(","):
                    break
                rest = rest[1:].lstrip()
        groups[field] = out
    return {
        "instance_graph": {k: groups[k] for k in ("entities", "events", "assertions")},
        "local_schema": {k: groups[k] for k in ("contextual_types", "predicates")},
        "decisions": groups["decisions"],
        "partial_records_only": True,
    }


def metric_values(assessment):
    c = (assessment or {}).get("contextual_metrics")
    if not c:
        return {}
    a = c["alignment"]
    rare = c["rare"]
    strict = a["strict_assertion_score"]
    return {
        "strict_precision": strict["precision"],
        "strict_recall": strict["recall"],
        "strict_f1": strict["f1"],
        "node_f1": a["node_score"]["f1"],
        "essential_temporal_accuracy": a["essential_temporal_accuracy"]["value"],
        "grounding_precision": a["grounding_precision"]["value"],
        "citation_validity": a["evidence_citation_validity"]["value"],
        "rare_pivotal_recall": rare["qualified_assertion_recall"]["value"],
        "rare_support_path_survival": rare["complete_support_path_survival"]["value"],
    }


def safe_graph(draft):
    # Only model/CPU-authored graph fields. No scorer alignments, expected targets,
    # HTTP envelopes, absolute paths or source-side restricted diagnostics.
    return {
        k: without_admin(draft[k])
        for k in (
            "contextual_interpretation",
            "local_schema",
            "instance_graph",
            "decisions",
            "uncertainty_and_abstentions",
            "partial_records_only",
        )
        if k in draft
    }


def label_graph(graph):
    nodes = graph.get("instance_graph", {})
    schema = graph.get("local_schema", {})
    labels = {
        n.get("entity_id", n.get("event_id")): n.get("label", "?")
        for n in nodes.get("entities", []) + nodes.get("events", [])
    }
    predicates = {p["predicate_id"]: p["label"] for p in schema.get("predicates", [])}
    rows = []
    for a in nodes.get("assertions", []):
        b = a.get("content", a)
        time = a.get("temporal_scope")
        if time == "content":
            time = b.get("temporal_content")
        endpoints = (
            labels.get(b.get("subject_id"), b.get("subject_id")),
            labels.get(b.get("object_id"), b.get("object_id")),
        )
        binding = (
            " → ".join(str(x) for x in endpoints)
            if b.get("subject_id")
            else "; ".join(
                str(r["role"]) + "=" + str(labels.get(r["object_id"], r["object_id"]))
                for r in b.get("roles", [])
            )
        )
        rows.append(
            {
                "id": a["assertion_id"],
                "predicate": predicates.get(b.get("predicate_id"), b.get("predicate_id")),
                "bindings": binding,
                "time": time,
                "epistemic": a.get("epistemic_scope"),
                "evidence": a.get("evidence_ids"),
                "why_matters": a.get("why_matters"),
                "commitment": a.get("narrative_commitment"),
                "direction": a.get("direction"),
            }
        )
    return labels, predicates, rows


def graph_elements(graph, anchors):
    labels, predicates, _ = label_graph(graph)
    g = graph.get("instance_graph", {})
    elements = []
    for key, label in labels.items():
        elements.append({"data": {"id": key, "label": label}, "position": anchors[label]})
    for a in g.get("assertions", []):
        b = a.get("content", a)
        subject = b.get("subject_id")
        obj = b.get("object_id")
        pairs = []
        if subject in labels and obj in labels:
            pairs = [(subject, obj, predicates.get(b["predicate_id"], b["predicate_id"]))]
        elif b.get("roles"):
            aid = "display-" + a["assertion_id"]
            elements.append(
                {
                    "data": {
                        "id": aid,
                        "label": predicates.get(b["predicate_id"], b["predicate_id"]),
                    },
                    "position": {"x": 500, "y": 300 + 20 * len(elements)},
                    "classes": "assertion",
                }
            )
            pairs = [
                (aid, r["object_id"], r["role"])
                for r in b["roles"]
                if r["object_id"] in labels
            ]
        for i, (s, o, label) in enumerate(pairs):
            if a.get("direction") == "inverse":
                s, o = o, s
            elements.append(
                {
                    "data": {
                        "id": a["assertion_id"] + "-" + str(i),
                        "source": s,
                        "target": o,
                        "label": label,
                    }
                }
            )
    return elements


def build(root, run, output):
    _, unit, neutral = sources(root)
    terminal = (
        read(run / "terminal.json")
        if (run / "terminal.json").exists()
        else read(run / "guardian-terminal.json")
    )
    cfg = DevelopmentConstructionConfiguration.load(
        root / "configs/study/development_construction.json"
    )
    evidence = tuple(EvidenceRecord.model_validate(e) for e in neutral["evidence"])
    rows = []
    graphs = []
    restricted = []
    for ordinal in (1, 2):
        path = (
            root
            / f"artifacts/restricted/c0-calibration-identity-events-v9/dev-unit-01.q{ordinal}.json"
        )
        original = read(path)
        p = original["projection"]
        assert p["snapshot_hash"] == neutral["snapshot"]["content_hash"]
        d = OntologyDraft(
            contextual_interpretation="Previously executed, hash-verified C0 development projection",
            local_schema=p["local_schema"],
            instance_graph=p["instance_graph"],
            decisions=p["decisions"],
            omissions=p["omissions"],
            budget_accounting=p["budget_accounting"],
        )
        assessment = assess(
            d,
            evidence=evidence,
            upper=cfg.upper_ontology,
            horizon=neutral["snapshot"]["horizon"],
            budgets=cfg.projection_budgets_by_unit[unit["unit_id"]],
            root=root,
            ordinal=ordinal,
        )
        row = {
            "condition": "C0",
            "context": ordinal,
            "attempt_id": original["attempt_id"],
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "reused_actual_cpu_output": True,
            "schema_valid": True,
            "canonical_valid": True,
            "structure_valid": assessment["structure"]["validation_status"] == "accepted",
            "scientific_accepted": assessment["scientific_accepted"],
            "repair_parent": None,
            "finish_reason": None,
            "input_tokens": 0,
            "output_tokens": 0,
            "request_seconds": None,
            "failure": ""
            if assessment["scientific_accepted"]
            else "Exploratory source/description assessment includes unresolved matches; see HTML",
            **metric_values(assessment),
        }
        rows.append(row)
        graphs.append({"row": row, "graph": safe_graph(d.model_dump(mode="json"))})
        restricted.append({"attempt_id": row["attempt_id"], "assessment": assessment})
    for path in sorted(run.glob("*/outcome.json")):
        o = read(path)
        folder = path.parent
        assessment = o.get("assessment")
        response = o.get("response") or {}
        transport = o.get("transport_metadata", {})
        usage = transport.get("usage", {})
        ordinal = None if o["kind"] == "c1" else int(o["kind"][-1])
        schema_valid = (folder / "schema-errors.json").exists() and not read(
            folder / "schema-errors.json"
        )
        row = {
            "condition": o["condition"],
            "context": ordinal,
            "attempt_id": o["attempt_id"],
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "request_hash": o["request_hash"],
            "reused_actual_cpu_output": False,
            "schema_valid": schema_valid,
            "canonical_valid": o["canonical_valid"],
            "structure_valid": bool(
                assessment and assessment["structure"]["validation_status"] == "accepted"
            ),
            "scientific_accepted": o["scientific_accepted"],
            "repair_parent": o["repair_parent"],
            "finish_reason": response.get("finish_reason", transport.get("finish_reason")),
            "input_tokens": response.get("prompt_tokens", usage.get("prompt_tokens")),
            "output_tokens": response.get("completion_tokens", usage.get("completion_tokens")),
            "output_allowance": read(folder / "request.json")["max_tokens"],
            "request_seconds": o["generation_seconds"],
            "failure_stage": (o.get("failure") or {}).get("stage"),
            "failure": (o.get("failure") or {}).get("message", ""),
            "confirmed_source_defects": o.get("source_defects", []),
            **metric_values(assessment),
        }
        # Detailed exception chains stay restricted; concise public failure label.
        if len(row["failure"]) > 450:
            row["failure"] = (
                row["failure_stage"] + "; detailed diagnostic retained in restricted artifact"
            )
        rows.append(row)
        if (folder / "canonical.json").exists():
            graph = read(folder / "canonical.json")
        elif (folder / "decoded.json").exists():
            graph = read(folder / "decoded.json")
        else:
            graph = complete_record_fragments(streamed_content(run, o["request_hash"]))
        graphs.append({"row": row, "graph": safe_graph(graph)})
        restricted.append({"attempt_id": row["attempt_id"], "assessment": assessment})
    for ordinal in (1, 2):
        if not any(r["condition"] == "A-FixedSelect" and r["context"] == ordinal for r in rows):
            reason = "No accepted sealed C1; not attempted"
            packing = run / f"fixed-q{ordinal}-base-packing-failure.json"
            if packing.exists():
                reason = read(packing)["exception"]
            rows.append(
                {
                    "condition": "A-FixedSelect",
                    "context": ordinal,
                    "attempt_id": None,
                    "scientific_accepted": False,
                    "failure": reason,
                }
            )
    contexts = [
        read(root / unit["query_stages"][i - 1]["relative_path"] / "query.json")["query"]
        for i in (1, 2)
    ]
    samples = terminal.get("resource_samples", [])
    accounting = {
        "historical_seconds": 6716.108081,
        "phase_seconds": terminal["actual_allocated_seconds"] - 6716.108081,
        "cumulative_seconds": terminal["actual_allocated_seconds"],
        "phase_cap_seconds": 3600,
        "global_scheduled_seconds": 33660,
        "global_hard_seconds": 36000,
        "open_allocations": terminal["open_allocations"],
        "open_service_journals": terminal["open_service_journals"],
        "stop_reason": terminal.get("stop_reason", terminal.get("reason")),
        "peak_sampled_resources": {
            k: max((s.get(k, 0) or 0 for s in samples), default=0)
            for k in ("process_ram_bytes", "gpu_vram_bytes", "project_storage_bytes")
        },
    }
    result = {
        "label": "EXPLORATORY DEVELOPMENT ONLY — one easy world, not held-out efficacy",
        "unit": unit["unit_id"],
        "snapshot_hash": neutral["snapshot"]["content_hash"],
        "contexts": [without_admin(c) for c in contexts],
        "rows": rows,
        "accounting": accounting,
        "ordinary_admission": False,
        "human_review_still_required": True,
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json_atomic(result, output / "tables/preliminary_development_results.json")
    write_json_atomic(graphs, output / "tables/preliminary_development_graphs.json")
    write_json_atomic(restricted, run / "offline-report-assessments.json")
    columns = [
        "condition",
        "context",
        "attempt_id",
        "repair_parent",
        "schema_valid",
        "canonical_valid",
        "structure_valid",
        "scientific_accepted",
        "strict_precision",
        "strict_recall",
        "strict_f1",
        "node_f1",
        "essential_temporal_accuracy",
        "grounding_precision",
        "citation_validity",
        "rare_pivotal_recall",
        "rare_support_path_survival",
        "finish_reason",
        "input_tokens",
        "output_tokens",
        "output_allowance",
        "request_seconds",
        "failure",
    ]
    f = io.StringIO()
    w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
    atomic_text(output / "tables/preliminary_development_results.csv", f.getvalue())
    fmt = lambda x: "—" if x is None else f"{x:.4f}" if isinstance(x, float) else str(x)
    md = [
        "# Preliminary development results",
        "",
        "Exploratory development only. One preselected easy world and two contrasting contexts over identical evidence; no significance testing, held-out efficacy, production acceptance or p95 claim.",
        "",
        "## Results",
        "",
        "All base attempts and repairs are retained below. Scores describe canonical outputs even when other validation failed; they are not registered accepted-study results. Missing canonical outputs have no scored draft. Absence of a strict match is not proof of an invented claim.",
        "",
        "| Condition / context | Attempt | Schema / canonical / structure | Scientific acceptance | Strict P / R / F1 | Finish / tokens | Request s |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        md.append(
            "| "
            + " | ".join(
                [
                    r["condition"] + " / " + str(r.get("context") or "prequery"),
                    "repair"
                    if r.get("repair_parent")
                    else "base"
                    if r.get("attempt_id")
                    else "blocked",
                    " / ".join(
                        fmt(r.get(k))
                        for k in ("schema_valid", "canonical_valid", "structure_valid")
                    ),
                    str(r["scientific_accepted"]),
                    " / ".join(
                        fmt(r.get(k)) for k in ("strict_precision", "strict_recall", "strict_f1")
                    ),
                    str(r.get("finish_reason") or "CPU/blocked")
                    + " / "
                    + str(r.get("output_tokens", "—")),
                    fmt(r.get("request_seconds")),
                ]
            )
            + " |"
        )
    md += [
        "",
        "## Evidence, contexts and actual graphs",
        "",
        "The [interactive comparison](figures/preliminary_development_comparison.html) includes all supplied synthetic prose, both contexts, every available generated record, readable assertion bindings and complete qualifications. Truncated outputs show only complete recovered JSON members, explicitly not reconstructed/accepted graphs. Identical labels use fixed visual anchors; assertion junctions are display-only, not invented event semantics.",
        "",
        "[Machine-readable measures](tables/preliminary_development_results.csv) · [Graph records](tables/preliminary_development_graphs.json)",
        "",
        "## Allocation and gates",
        "",
        f"Historical allocation {accounting['historical_seconds']:.6f} s; this phase {accounting['phase_seconds']:.6f} s; cumulative {accounting['cumulative_seconds']:.6f} s. Open GPU/service journals: {accounting['open_allocations']}/{accounting['open_service_journals']}. The phase ceiling is 3,600 s. The global scheduled/hard limits remain 33,660/36,000 s; no complete-study admission is claimed.",
        "",
        "Remaining registered inventory is preserved in the restricted run manifest; it has not been removed or reset. A few development calls cannot establish production throughput. Held-out execution remains independently reviewed and gated.",
        "",
        "## Interpretation and limitations",
        "",
        "C0 rows reuse actual, unchanged, hash-verified CPU projections of this evidence; no authored graph is substituted for an LLM output. C0's complete extraction competence gate remains passed (104/122 precision, 104/114 recall, 5/5 fixture families, valid evidence references throughout); its separate aggregate contextual P=54/275, R=54/91, F1=0.295082 is unchanged. This report's two-context values are a subset, not a replacement gate.",
        "",
        "Scientific acceptance requires positive source support; unresolved matching/description coverage is not a pass. The conservative description recognizer only certifies direct source substrings and does not call unmatched paraphrases false. FixedSelect cannot run without an actual accepted C1 seal and intact packing. C2 runs independently of C1. Output-cap failures are capacity failures, not evidence of universal model incapability.",
        "",
    ]
    for r in rows:
        if r.get("failure"):
            md += [
                f"- {r['condition']} context {r.get('context')} {'repair' if r.get('repair_parent') else 'base'}: {r['failure']}"
            ]
    llm = [r for r in rows if r.get("request_hash")]
    if not any(r["scientific_accepted"] for r in llm):
        md += [
            "",
            "No GPU output was scientifically accepted. This development configuration failed within the used allowance; no further diagnostic phase is automatically initiated.",
        ]
    atomic_text(output / "PRELIMINARY_DEVELOPMENT_RESULTS.md", "\n".join(md) + "\n")
    labels = sorted({label for g in graphs for label in label_graph(g["graph"])[0].values()})
    anchors = {
        label: {
            "x": 500 + 410 * math.cos(2 * math.pi * i / max(1, len(labels))),
            "y": 410 + 330 * math.sin(2 * math.pi * i / max(1, len(labels))),
        }
        for i, label in enumerate(labels)
    }
    parts = [
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Exploratory development comparison</title><style>body{font:17px system-ui;margin:24px;color:#172335}h1,h2{line-height:1.2}.graph{height:850px;border:1px solid #bbb;background:#fcfcff}table{border-collapse:collapse;width:100%;font-size:15px}td,th{border:1px solid #ccd;padding:8px;vertical-align:top;overflow-wrap:anywhere}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:14px}section{margin:35px 0}.warning{background:#fff0ce;padding:14px}</style><h1>Exploratory development comparison</h1><p class="warning">One preselected easy development world. Graphs show actual outputs, including failures; they are not expected answers or confirmatory evidence. Drag/zoom nodes; read assertion qualifications below each graph.</p><h2>Shared evidence</h2><ol>'
    ]
    parts += [
        '<li id="e' + str(i) + '">' + html.escape(e.text) + "</li>"
        for i, e in enumerate(evidence, 1)
    ]
    parts.append("</ol><h2>Contrasting contexts</h2>")
    for i, c in enumerate(contexts, 1):
        parts.append(
            "<h3>Context "
            + str(i)
            + "</h3><pre>"
            + html.escape(json.dumps(without_admin(c), indent=2))
            + "</pre>"
        )
    js = []
    for i, g in enumerate(graphs):
        r = g["row"]
        graph = g["graph"]
        _, _, assertions = label_graph(graph)
        title = (
            r["condition"]
            + " / "
            + str(r.get("context") or "prequery")
            + (" / repair" if r.get("repair_parent") else " / base")
        )
        parts.append(
            "<section><h2>"
            + html.escape(title)
            + "</h2><pre>"
            + html.escape(
                json.dumps(
                    {
                        k: r.get(k)
                        for k in (
                            "attempt_id",
                            "schema_valid",
                            "canonical_valid",
                            "structure_valid",
                            "scientific_accepted",
                            "finish_reason",
                            "failure",
                        )
                    },
                    indent=2,
                )
            )
            + '</pre><div class="graph" id="g'
            + str(i)
            + '"></div><table><tr><th>Assertion / predicate</th><th>Bindings</th><th>Time / epistemic / commitment</th><th>Evidence / description</th></tr>'
        )
        for a in assertions:
            cells = [
                a["id"] + " / " + str(a["predicate"]),
                a["bindings"],
                json.dumps(
                    {k: a[k] for k in ("time", "epistemic", "commitment", "direction")}, indent=2
                ),
                json.dumps({k: a[k] for k in ("evidence", "why_matters")}, indent=2),
            ]
            parts.append(
                "<tr>"
                + "".join("<td><pre>" + html.escape(c) + "</pre></td>" for c in cells)
                + "</tr>"
            )
        parts.append(
            "</table><details><summary>All actual graph fields and construction decisions</summary><pre>"
            + html.escape(json.dumps(graph, indent=2))
            + "</pre></details></section>"
        )
        js.append({"id": "g" + str(i), "elements": graph_elements(graph, anchors)})
    parts.append(
        '<script src="../../ui/cytoscape.min.js"></script><script>const views='
        + json.dumps(js).replace("<", "\\u003c")
        + ';for(const v of views){cytoscape({container:document.getElementById(v.id),elements:v.elements,layout:{name:"preset",padding:50},style:[{selector:"node",style:{label:"data(label)","text-wrap":"wrap","text-max-width":150,"font-size":18,width:35,height:35,"background-color":"#387cb0","text-valign":"bottom","text-margin-y":8}},{selector:"edge",style:{label:"data(label)","font-size":14,"text-rotation":"autorotate","curve-style":"bezier","target-arrow-shape":"triangle",width:2,"line-color":"#8b738f","target-arrow-color":"#8b738f","text-background-color":"#fff","text-background-opacity":.9}},{selector:".assertion",style:{shape:"diamond","background-color":"#ae7739"}}]});}</script></html>'
    )
    atomic_text(output / "figures/preliminary_development_comparison.html", "\n".join(parts))
    manifest = {
        "run_id": run.name,
        "input_terminal_sha256": hashlib.sha256(
            (
                run
                / (
                    "terminal.json"
                    if (run / "terminal.json").exists()
                    else "guardian-terminal.json"
                )
            ).read_bytes()
        ).hexdigest(),
        "files": {
            str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                output / "PRELIMINARY_DEVELOPMENT_RESULTS.md",
                output / "tables/preliminary_development_results.json",
                output / "tables/preliminary_development_results.csv",
                output / "tables/preliminary_development_graphs.json",
                output / "figures/preliminary_development_comparison.html",
            ]
        },
    }
    write_json_atomic(manifest, output / "tables/preliminary_development_manifest.json")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports"))
    args = parser.parse_args()
    result = build(Path.cwd(), args.run, args.output)
    print(json.dumps({"rows": len(result["rows"]), "accounting": result["accounting"]}, indent=2))
