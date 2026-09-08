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
from contextlib import suppress
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
    result = {
        "instance_graph": {k: groups[k] for k in ("entities", "events", "assertions")},
        "local_schema": {k: groups[k] for k in ("contextual_types", "predicates")},
        "decisions": groups["decisions"],
        "partial_records_only": True,
    }
    m = re.search(r'"contextual_interpretation"\s*:\s*', text)
    if m:
        with suppress(json.JSONDecodeError):
            result["contextual_interpretation"] = json.JSONDecoder().raw_decode(text[m.end() :])[0]
    return result


def invalid_context_metrics(root, ordinal):
    """Existing invalid-output scoring, separate from canonical graph scoring."""
    from story_projection_onto.development_continuation import load_development_prequery_evidence
    from story_projection_onto.development_runtime import load_development_call_manifest
    from story_projection_onto.metrics.alignment import score_alignment
    from story_projection_onto.metrics.rare import annotations_from_gold, score_rare_pivotal
    from story_projection_onto.scorer_only.development_assessment import (
        DevelopmentScientificAssessmentProvider,
        resolve_development_world_id,
    )
    from story_projection_onto.synthetic_benchmark import ScorerWorldArtifact

    n, v, _ = load_development_prequery_evidence(root, load_development_call_manifest(root))
    world = resolve_development_world_id(
        root, v["dev-unit-01"].content_hash, n["dev-unit-01"].content_hash
    )
    scorer = ScorerWorldArtifact.model_validate_json(
        (root / f"data/synthetic/scorer_only/development/{world}.json").read_bytes()
    )
    assert scorer.world_spec.split.value == "development"
    plan = DevelopmentScientificAssessmentProvider._alignment_plan(scorer, ordinal)
    a = score_alignment(
        plan=plan, predicted_nodes=(), predicted_assertions=(), invalid_semantic_output=True
    )
    rare = score_rare_pivotal(
        annotations=annotations_from_gold(scorer.gold_projections[ordinal - 1]),
        strictly_matched_assertion_target_ids=frozenset(),
        invalid_semantic_output=True,
    )
    return metric_values(
        {
            "contextual_metrics": {
                "alignment": a.model_dump(mode="json"),
                "rare": rare.model_dump(mode="json"),
            }
        }
    )


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


def public_assessment(assessment):
    """Generated IDs/check findings only; no reference answers or scorer targets."""
    if not assessment:
        return {
            "assessment_available": False,
            "reason": "formal assessment unavailable; see canonical status and failure stage",
        }
    return {
        "assessment_available": True,
        "structural_diagnostics": [
            {k: d[k] for k in ("code", "path", "message") if k in d}
            for d in assessment["structure"].get("diagnostics", [])
        ],
        "grounding_supported_assertion_ids": assessment["grounding_supported_assertion_ids"],
        "grounding_unresolved_assertion_ids": assessment["grounding_unresolved_assertion_ids"],
        "description_assessments": assessment["description_assessments"],
        "unresolved_is_not_false_or_accepted": True,
    }


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
                (aid, r["object_id"], r["role"]) for r in b["roles"] if r["object_id"] in labels
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
            contextual_interpretation=(
                "Previously executed, hash-verified C0 development projection"
            ),
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
        graphs.append(
            {
                "row": row,
                "graph": safe_graph(d.model_dump(mode="json")),
                "assessment": public_assessment(assessment),
            }
        )
        restricted.append({"attempt_id": row["attempt_id"], "assessment": assessment})
    outcome_paths = sorted(
        set(run.glob("*/outcome.json")) | set(run.parent.glob("run-*/*/outcome.json"))
    )
    by_attempt = {p.parent.name: p.parent for p in outcome_paths}
    for path in outcome_paths:
        o = read(path)
        folder = path.parent
        assessment = o.get("assessment")
        response = o.get("response") or {}
        transport = o.get("transport_metadata", {})
        usage = transport.get("usage", {})
        ordinal = None if o["kind"] == "c1" else int(o["kind"][-1])
        schema_valid = (
            not read(folder / "schema-errors.json")
            if (folder / "schema-errors.json").exists()
            else None
        )
        row = {
            "protocol": o.get("protocol", "single-response-development"),
            "construction_stage": o.get("construction_stage"),
            "stage_valid": o.get("stage_valid"),
            "stage_complete": o.get("stage_complete"),
            "stage_dependencies": o.get("stage_dependencies", {}),
            "condition": o["condition"],
            "context": ordinal,
            "attempt_id": o["attempt_id"],
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "request_hash": o["request_hash"],
            "reused_actual_cpu_output": False,
            "schema_valid": schema_valid,
            "canonical_valid": o["canonical_valid"] if (folder / "decoded.json").exists() else None,
            "structure_valid": (
                assessment["structure"]["validation_status"] == "accepted" if assessment else None
            ),
            "scientific_accepted": o["scientific_accepted"],
            "repair_parent": o["repair_parent"],
            "finish_reason": response.get("finish_reason", transport.get("finish_reason")),
            "input_tokens": response.get("prompt_tokens", usage.get("prompt_tokens")),
            "prepared_input_tokens": read(folder / "packing.json")["input_token_count"],
            "output_tokens": response.get("completion_tokens", usage.get("completion_tokens")),
            "output_allowance": read(folder / "request.json")["max_tokens"],
            "request_seconds": o["generation_seconds"]
            if o["generation_seconds"] is not None
            else o["allocated_generation_seconds"],
            "request_time_source": "client wall clock"
            if o["generation_seconds"] is not None
            else "metered generation event; includes client/decoding",
            "failure_stage": (o.get("failure") or {}).get("stage"),
            "failure": (o.get("failure") or {}).get("message", ""),
            "confirmed_source_defects": o.get("source_defects", []),
            "transport_complete": transport.get("response_complete"),
            "http_status": transport.get("status_code", transport.get("http_status")),
            "request_transmitted": bool(
                list((folder.parent / "http").glob(o["request_hash"][:16] + "-*/events.jsonl"))
            ),
            "json_complete": (folder / "decoded.json").exists(),
            "assessment": public_assessment(assessment),
            "development_request_revision": o.get(
                "development_request_revision", "historical-base"
            ),
            "replaces_nontransmitted_reservation": o.get("replaces_nontransmitted_reservation"),
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
        elif (folder / "assembled-nested.json").exists():
            graph = read(folder / "assembled-nested.json")
        elif o.get("construction_stage") and (folder / "decoded.json").exists():
            s = o["construction_stage"]
            raw = read(folder / "decoded.json")
            dependencies = o.get("stage_dependencies", {})
            a = raw if s == "A" else read(by_attempt[dependencies["A"]] / "decoded.json")
            b = (
                raw
                if s == "B"
                else read(by_attempt[dependencies["B"]] / "decoded.json")
                if s == "C"
                else {}
            )
            graph = {
                "contextual_interpretation": a.get("contextual_interpretation"),
                "local_schema": a.get("local_schema", {}),
                "instance_graph": {
                    "entities": a.get("entities", []),
                    "events": a.get("events", []),
                    "assertions": b.get("assertions", []),
                },
                "decisions": raw.get("decisions", []) if s == "C" else [],
                "intermediate_stage": s,
            }
        elif (folder / "decoded.json").exists():
            graph = read(folder / "decoded.json")
        else:
            graph = complete_record_fragments(streamed_content(folder.parent, o["request_hash"]))
        if row["json_complete"]:
            instance = graph["instance_graph"]
            budget = (
                cfg.preconstruction_budgets
                if o["kind"] == "c1"
                else cfg.projection_budgets_by_unit[unit["unit_id"]]
            )
            row["nodes_received"] = len(instance["entities"]) + len(instance["events"])
            row["assertions_received"] = len(instance["assertions"])
            row["generation_object_budget_valid"] = (
                row["nodes_received"] <= budget.node_budget
                and row["assertions_received"] <= budget.assertion_budget
            )
        row["confirmed_semantic_categories"] = sorted(
            {
                d["category"]
                for d in o.get("source_defects", [])
                if d["category"] in {"unsupported_attribution", "unsupported_intrinsic_precision"}
            }
        )
        if assessment:
            row["grounding_unresolved_count"] = len(
                assessment["grounding_unresolved_assertion_ids"]
            )
            row["description_unresolved_count"] = sum(
                x["status"] == "unresolved" for x in assessment["description_assessments"]
            )
        if graph.get("partial_records_only") and row.get("output_tokens"):
            from story_projection_onto.development_demo import aliases
            from story_projection_onto.scorer_only.development_demo_assessment import (
                source_feedback,
            )

            row["partial_source_diagnostics"] = source_feedback(graph, evidence, aliases(evidence))
            row["confirmed_semantic_categories"] = sorted(
                {
                    d["category"]
                    for d in row["partial_source_diagnostics"]
                    if d["category"]
                    in {"unsupported_attribution", "unsupported_intrinsic_precision"}
                }
            )
            row["semantic_diagnosis_scope"] = (
                "Offline source-only checks of complete received members; not formal acceptance"
            )
            row["partial_record_counts"] = {k: len(v) for k, v in graph["instance_graph"].items()}
            row["received_construction_claims"] = [
                {
                    "operator": d["operator"],
                    "created_node_id_count": sum(
                        isinstance(x, str) and x.startswith(("nE", "nV"))
                        for x in d["created_object_ids"]
                    ),
                    "created_id_count": len(d["created_object_ids"]),
                }
                for d in graph["decisions"]
            ]
            received = graph["instance_graph"]["assertions"]
            signatures = {}
            for a in received:
                signature = json.dumps(
                    {k: v for k, v in a.items() if k != "assertion_id"}, sort_keys=True
                )
                signatures.setdefault(signature, []).append(a["assertion_id"])
            row["offline_prefix_observations"] = {
                "attributed_assertions": sum(
                    a.get("epistemic_scope") is not None for a in received
                ),
                "unique_claimed_nodes": len(
                    {
                        x
                        for d in graph["decisions"]
                        for x in d["created_object_ids"]
                        if isinstance(x, str) and x.startswith(("nE", "nV"))
                    }
                ),
                "node_budget": (
                    cfg.preconstruction_budgets.node_budget
                    if o["kind"] == "c1"
                    else cfg.projection_budgets_by_unit[unit["unit_id"]].node_budget
                ),
                "identical_assertion_groups_except_id": [
                    ids for ids in signatures.values() if len(ids) > 1
                ],
                "not_canonical_or_scientific_acceptance": True,
            }
        graphs.append(
            {"row": row, "graph": safe_graph(graph), "assessment": public_assessment(assessment)}
        )
        restricted.append({"attempt_id": row["attempt_id"], "assessment": assessment})
    # Reconcile a reserved request which failed before a model-call event. Do
    # not manufacture an outcome/response or overwrite the original terminal.
    for request_path in sorted(run.parent.glob("run-*/*/request.json")):
        folder = request_path.parent
        if (folder / "outcome.json").exists():
            continue
        reservation = next(
            (
                read(p)
                for p in run.parent.glob("attempt-*.json")
                if read(p)["attempt_id"] == folder.name
            ),
            None,
        )
        if reservation is None:
            continue
        packing = read(folder / "packing.json")
        # This is the previously diagnosed reservation, not a generic inference
        # that absent output proves non-transmission for an arbitrary failure.
        if folder.name != "nested-development-demonstration-20260908-attempt-04":
            raise ValueError("unreconciled reserved request requires explicit assessment")
        incident_run = folder.parent
        outer = read(incident_run / "controller-failure.json")
        reconciliation = {
            "attempt_id": folder.name,
            "kind": reservation["kind"],
            "parent": reservation["parent"],
            "request_hash": reservation["request_hash"],
            "observed_outer_failure": outer["message"],
            "original_pre_generation_exception_retained": False,
            "cpu_reproduced_defect": (
                "VLLMService.generate was called without repair=True for "
                "DecodingPass.REPAIR; its pre-event guard rejects this combination"
            ),
            "server_request_observed": False,
            "evidence": (
                "No HTTP journal and no GPU generation event; request and packing "
                "retained; original controller traceback identifies the "
                "missing-event bookkeeping failure"
            ),
            "status": "reserved repair; not transmitted; separately reconciled, not a model output",
        }
        write_json_atomic(
            reconciliation, incident_run / "offline-pre-generation-reconciliation.json"
        )
        rows.append(
            {
                "condition": "C1" if reservation["kind"] == "c1" else "C2",
                "context": None if reservation["kind"] == "c1" else int(reservation["kind"][-1]),
                "attempt_id": folder.name,
                "repair_parent": reservation["parent"],
                "request_hash": reservation["request_hash"],
                "prepared_input_tokens": packing["input_token_count"],
                "input_tokens": None,
                "output_tokens": None,
                "output_allowance": read(request_path)["max_tokens"],
                "request_seconds": None,
                "scientific_accepted": False,
                "schema_valid": None,
                "canonical_valid": None,
                "structure_valid": None,
                "execution_status": "not_transmitted",
                "failure_stage": "pre_generation_controller",
                "failure": (
                    "Missing repair flag rejected the prepared call before a GPU event; "
                    "subsequent bookkeeping masked the original exception. "
                    "CPU-reproduced diagnosis; no new model response."
                ),
            }
        )
    pending = (
        read(run / "pending-work.json")["queue"] if (run / "pending-work.json").exists() else []
    )
    for kind, parent, _ in pending:
        if not parent or any(r.get("repair_parent") == parent for r in rows):
            continue
        rows.append(
            {
                "condition": "C1" if kind == "c1" else "C2",
                "context": None if kind == "c1" else int(kind[-1]),
                "attempt_id": None,
                "repair_parent": parent,
                "scientific_accepted": False,
                "execution_status": "not_attempted",
                "failure": (
                    "Prepared repair not executed after the controller failure; no "
                    "replacement output."
                ),
            }
        )
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
    invalid_scores = {i: invalid_context_metrics(root, i) for i in (1, 2)}
    for r in rows:
        if (
            r.get("context")
            and r.get("attempt_id")
            and r["condition"] != "C0"
            and r.get("construction_stage") not in ("A", "B")
        ):
            values = {} if r["scientific_accepted"] else invalid_scores[r["context"]]
            r.update({"acceptance_gated_" + k: v for k, v in values.items()})
    transmitted_repairs = [
        r for r in rows if r.get("repair_parent") and r.get("request_transmitted")
    ]
    samples = terminal.get("resource_samples", [])
    for prior_terminal in run.parent.glob("run-*/terminal.json"):
        if prior_terminal != run / "terminal.json":
            samples += read(prior_terminal).get("resource_samples", [])
    accounting = {
        "historical_seconds": 6716.108081,
        "phase_seconds": round(terminal["actual_allocated_seconds"] - 6716.108081, 6),
        "cumulative_seconds": terminal["actual_allocated_seconds"],
        "phase_cap_seconds": 3600,
        "phase_maximum_starts": read(root / "configs/study/preliminary_development_demo.json")[
            "maximum_service_starts"
        ],
        "service_starts": len(list(run.parent.glob("start-*.json"))),
        "reserved_attempts": len(list(run.parent.glob("attempt-*.json"))),
        "transmitted_http_requests": sum(bool(r.get("request_transmitted")) for r in rows),
        "transmitted_generations": sum(
            (r.get("output_tokens") or 0) > 0 for r in rows if r["condition"] != "C0"
        ),
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
    staged_rows = [r for r in rows if r.get("construction_stage")]
    if staged_rows:
        accounting.update(
            phase_maximum_starts=6,
            phase_maximum_reservations=22,
            staged_baseline_seconds=8200.425164,
            staged_seconds=round(terminal["actual_allocated_seconds"] - 8200.425164, 6),
            staged_new_reservations=len(staged_rows),
        )
    previous_actual = accounting["historical_seconds"]
    allocation_sessions = []
    for p in sorted(run.parent.glob("run-*/terminal.json")):
        record = read(p)
        actual = record.get("actual_allocated_seconds")
        if actual is None or actual <= previous_actual:
            continue
        allocation_sessions.append(
            {
                "run_id": p.parent.name,
                "allocated_seconds": round(actual - previous_actual, 6),
                "cumulative_seconds": actual,
                "stop_reason": record.get("stop_reason"),
            }
        )
        previous_actual = actual
    accounting["allocation_sessions"] = allocation_sessions
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
    prior_gate_path = root / "artifacts/restricted/c0-repaired-extraction-v6/assessment.json"
    prior_gate = read(prior_gate_path)
    extraction_gate = prior_gate["revised_extraction_competence"]
    contextual_gate = prior_gate["contextual_projection_metric"]
    result["unchanged_c0_gate"] = {
        "source_sha256": hashlib.sha256(prior_gate_path.read_bytes()).hexdigest(),
        "extraction": extraction_gate,
        "contextual": contextual_gate,
    }
    result["model"] = {
        "repository": "Qwen/Qwen3-8B-AWQ",
        "revision": read(root / "configs/study/preliminary_development_demo.json")[
            "model_revision"
        ],
        "context_limit": 12288,
        "thinking": False,
        "backend": "vLLM 0.10.2 / XGrammar, no fallback, whitespace restriction enabled",
    }
    protocol = read(root / "configs/study/preliminary_development_demo.json")
    result["development_protocol"] = {
        k: protocol[k]
        for k in (
            "base_input_output_tokens",
            "repair_input_output_tokens",
            "instruction_path",
            "repair_policy_revision",
            "repair_previous_response_policy",
        )
    }
    if staged_rows:
        result["staged_development_protocol"] = read(root / "configs/study/staged_development.json")
    inventory_path = run / "remaining-registered-inventory.json"
    if inventory_path.exists():
        from story_projection_onto.output_capacity_gate import capacity_forecast

        inventory = read(inventory_path)["rows"]
        load_proxy = next(
            r["forecast_p95_seconds"] for r in inventory if r["call_class"] == "gpu_session_start"
        )
        proxy = capacity_forecast(
            inventory,
            pending_acceptance_resume_seconds=load_proxy,
            actual_allocated_seconds=terminal["actual_allocated_seconds"],
            additional_diagnostic_allowance=0,
        )
        result["registered_remaining_work"] = {
            "inventory": [
                {
                    "call_class": r["call_class"],
                    "remaining_count": r["remaining_count"],
                    "superseded_14b_row": r["superseded_without_execution"],
                }
                for r in inventory
            ],
            "fallback_gate_unpassed": {
                "c1": 1,
                "c2": 2,
                "fixed_select": 1,
                "conditional_short_repair_at_most": 1,
                "accounted_within_reserve_rows_not_added_again": True,
                "restart_resume_and_load_still_required": True,
            },
            "remaining_proxy_seconds": proxy["remaining_forecast_seconds"],
            "all_in_proxy_seconds": proxy["all_in_seconds"],
            "proxy_kind": proxy["kind"],
            "valid_production_forecast_established": False,
            "failed_throughput_credited": False,
        }
    output.mkdir(parents=True, exist_ok=True)
    write_json_atomic(result, output / "tables/preliminary_development_results.json")
    write_json_atomic(graphs, output / "tables/preliminary_development_graphs.json")
    write_json_atomic(restricted, run / "offline-report-assessments.json")
    columns = [
        "protocol",
        "construction_stage",
        "stage_valid",
        "stage_complete",
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
        "acceptance_gated_strict_f1",
        "finish_reason",
        "input_tokens",
        "prepared_input_tokens",
        "output_tokens",
        "output_allowance",
        "request_seconds",
        "failure",
        "transport_complete",
        "http_status",
        "request_transmitted",
        "json_complete",
        "failure_stage",
        "generation_object_budget_valid",
        "grounding_unresolved_count",
        "description_unresolved_count",
    ]
    f = io.StringIO()
    w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    atomic_text(output / "tables/preliminary_development_results.csv", f.getvalue())

    def fmt(x):
        return "—" if x is None else f"{x:.4f}" if isinstance(x, float) else str(x)

    md = [
        "# Preliminary development results",
        "",
        (
            "Exploratory development only. One preselected easy world and two "
            "contrasting contexts over identical evidence; no significance "
            "testing, held-out efficacy, production acceptance or p95 claim."
        ),
        "",
        "## Results",
        "",
        (
            "All base attempts and prepared repairs are retained below. Scores "
            "describe canonical outputs even when other validation failed; they "
            "are not registered accepted-study results. Missing canonical "
            "outputs have no scored draft. Separate acceptance_gated columns "
            "apply the existing invalid-output rule to failed LLM attempts "
            "(strict F1=0), without declaring every received statement false. "
            "Not-transmitted and not-attempted repairs are not model outputs. "
            "Absence of a strict match is not proof of an invented claim."
        ),
        "",
        (
            "| Condition / context | Attempt | Schema / canonical / structure | "
            "Scientific acceptance | Strict P / R / F1 | Finish / input : output "
            "tokens | Request s |"
        ),
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        md.append(
            "| "
            + " | ".join(
                [
                    r["condition"] + " / " + str(r.get("context") or "query-blind"),
                    (
                        "stage " + r["construction_stage"] + " "
                        if r.get("construction_stage")
                        else ""
                    )
                    + (
                        "repair"
                        if r.get("repair_parent")
                        else "base"
                        if r.get("attempt_id")
                        else "blocked"
                    ),
                    " / ".join(
                        fmt(r.get(k))
                        for k in ("schema_valid", "canonical_valid", "structure_valid")
                    ),
                    str(r["scientific_accepted"]),
                    " / ".join(
                        fmt(r.get(k)) for k in ("strict_precision", "strict_recall", "strict_f1")
                    ),
                    str(
                        r.get("finish_reason")
                        or (
                            "CPU"
                            if r["condition"] == "C0"
                            else r.get("execution_status", "blocked")
                        )
                    )
                    + " / "
                    + fmt(r.get("input_tokens"))
                    + " : "
                    + fmt(r.get("output_tokens")),
                    fmt(r.get("request_seconds")),
                ]
            )
            + " |"
        )
    md += [
        "",
        "## Available contextual measures",
        "",
        (
            "These use the existing scoring definitions. Unmatched reference "
            "grounding is a reference-coverage result, not proof of factual "
            "falsity. Conditional draft scores are available only when canonical "
            "reconstruction completes; failed LLM attempts also retain separate "
            "acceptance-gated scores."
        ),
        "",
        (
            "| Condition / context | Node F1 | Essential temporal accuracy | "
            "Reference grounding | Rare-pivotal recall | Complete rare support "
            "path |"
        ),
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        if "strict_f1" in r:
            md += [
                "| "
                + r["condition"]
                + " / "
                + str(r["context"])
                + " | "
                + " | ".join(
                    fmt(r.get(k))
                    for k in (
                        "node_f1",
                        "essential_temporal_accuracy",
                        "grounding_precision",
                        "rare_pivotal_recall",
                        "rare_support_path_survival",
                    )
                )
                + " |"
            ]
    md += [
        "",
        "## Evidence, contexts and actual graphs",
        "",
        (
            "The [interactive "
            "comparison](figures/preliminary_development_comparison.html) "
            "includes all supplied synthetic prose, both contexts, every "
            "available generated record, readable assertion bindings and "
            "complete qualifications. Truncated outputs show only complete "
            "recovered JSON members, explicitly not reconstructed/accepted "
            "graphs. Identical labels use fixed visual anchors; assertion "
            "junctions are display-only, not invented event semantics."
        ),
        "",
        (
            "[Machine-readable "
            "measures](tables/preliminary_development_results.csv) · [Graph "
            "records](tables/preliminary_development_graphs.json)"
        ),
        "",
        "## Allocation and gates",
        "",
        f"Historical allocation {accounting['historical_seconds']:.6f} s; "
        f"this phase {accounting['phase_seconds']:.6f} s; "
        f"cumulative {accounting['cumulative_seconds']:.6f} s. Open GPU/service journals: "
        f"{accounting['open_allocations']}/{accounting['open_service_journals']}. "
        "The phase ceiling is 3,600 s. The global scheduled/hard limits remain "
        "33,660/36,000 s; no complete-study admission is claimed.",
        "",
        f"Service starts: {accounting['service_starts']}; transmitted HTTP requests: "
        f"{accounting['transmitted_http_requests']}; responses with generation tokens: "
        f"{accounting['transmitted_generations']}; reserved attempts: "
        f"{accounting['reserved_attempts']}. The continuation allows at most "
        f"{accounting['phase_maximum_starts']} total starts within the same phase. "
        f"Transmitted parent repairs: {len(transmitted_repairs)}. "
        "vLLM is stopped; the pod remains intact. No further phase is initiated.",
        "",
        "Peak sampled GPU VRAM / process RAM / project occupancy (bytes): "
        + " / ".join(
            str(accounting["peak_sampled_resources"][k])
            for k in ("gpu_vram_bytes", "process_ram_bytes", "project_storage_bytes")
        )
        + (
            ". These are sampled peaks, not continuous maximum guarantees. "
            "Terminal full storage checks passed."
        ),
        "",
        (
            "Remaining registered inventory is preserved in the restricted run "
            "manifest; it has not been removed or reset. A few development calls "
            "cannot establish production throughput. Held-out execution remains "
            "independently reviewed and gated."
        ),
        "",
        "## Interpretation and limitations",
        "",
        "C0 rows reuse actual, unchanged, hash-verified CPU projections; no authored "
        "graph substitutes for an LLM output. Its existing extraction gate is "
        f"passed={extraction_gate['passes']}: precision "
        f"{extraction_gate['metric']['true_positive_count']}/"
        f"{extraction_gate['metric']['predicted_count']}, recall "
        f"{extraction_gate['metric']['true_positive_count']}/"
        f"{extraction_gate['metric']['gold_count']}, "
        f"{sum(extraction_gate['families'].values())}/{len(extraction_gate['families'])} "
        "fixture families, evidence-reference validity "
        f"{extraction_gate['valid_evidence_reference_rate']:.1%}. Its separate aggregate "
        f"contextual P={contextual_gate['true_positive_count']}/"
        f"{contextual_gate['predicted_count']}, R={contextual_gate['true_positive_count']}/"
        f"{contextual_gate['gold_count']}, F1={contextual_gate['f1']:.6f} is unchanged. "
        "These two-context values are a subset, not a replacement gate.",
        "",
        (
            "Scientific acceptance requires positive source support; unresolved "
            "matching/description coverage is not a pass. The conservative "
            "description recognizer only certifies direct source substrings and "
            "does not call unmatched paraphrases false. FixedSelect cannot run "
            "without an actual accepted C1 seal and intact packing. C2 runs "
            "independently of C1. Output-cap failures are capacity failures, not "
            "evidence of universal model incapability."
        ),
        "",
    ]
    if transmitted_repairs:
        cap_in, cap_out = protocol["repair_input_output_tokens"]
        md += [
            "",
            "### Continuation capacity and execution correction",
            "",
            f"The same C1/C2 development allocation is {cap_in:,} input / {cap_out:,} "
            f"output within {result['model']['context_limit']:,} total tokens. The named "
            "nested representation and lossless adapter are unchanged. Full evidence and "
            "essential source-grounded feedback are retained; prior responses remain "
            "complete in restricted lineage rather than being duplicated in model input.",
            "",
            "Existing graph and single-kind creation budgets were added to the grammar. "
            "The first transmitted C1 repair encountered a server-integration failure: "
            "vLLM returned HTTP 200 with a streaming rejection of `uniqueItems`, despite "
            "passing standalone XGrammar compilation. No generation occurred. This "
            "preflight coverage gap is an implementation error, not a model-semantic "
            "failure. C1 was not repeated. The final service start used the actual vLLM "
            "validator before allocation and removed only that unsupported decoder keyword; "
            "identical post-validation uniqueness constraints remained. Both C2 repairs "
            "were then transmitted on one service, independently of C1, with no blind "
            "base repetition or further repair. Authored capacity fits did not predict "
            "successful complete generation.",
        ]
    for r in rows:
        if r.get("failure"):
            md += [
                f"- {r['condition']} context {r.get('context')} "
                f"{'repair' if r.get('repair_parent') else 'base'}: {r['failure']}"
            ]
    if allocation_sessions:
        md += [
            "",
            "### Allocated service intervals",
            "",
            "| Start | Allocated seconds | Cumulative seconds | Stop reason |",
            "|---|---:|---:|---|",
        ]
        for i, interval in enumerate(allocation_sessions, 1):
            md.append(
                f"| {i} | {interval['allocated_seconds']:.6f} | "
                f"{interval['cumulative_seconds']:.6f} | {interval['stop_reason']} |"
            )
    llm = [r for r in rows if r.get("request_hash")]
    md += [
        "",
        f"Pinned model: {result['model']['repository']} at `{result['model']['revision']}`, "
        f"{result['model']['context_limit']:,} total tokens; {result['model']['backend']}. "
        "No model change. Historical rows used single-response construction. "
        "Separately labeled staged rows use the explicitly amended A/B/C protocol, "
        "not successful execution of the original protocol.",
        "",
        (
            "C1 repair requests remain model-query-blind. A late repair/seal cannot "
            "retrospectively satisfy the registered physical pre-query barrier. "
            "The old untransmitted reservation remains separately reported. No "
            "production adoption or complete acceptance is claimed."
        ),
    ]
    if "registered_remaining_work" in result:
        work = result["registered_remaining_work"]
        md += [
            "",
            "## Remaining registered work",
            "",
            "Unvalidated legacy remaining-work proxy: "
            f"{work['remaining_proxy_seconds']:.6f} s; all-in with preserved actual "
            f"allocation: {work['all_in_proxy_seconds']:.6f} s. This is not a calibrated "
            "production forecast. Failed completion speeds receive no credit. The "
            "original nine-hour target was not met; the amended scheduled ceiling "
            "remains unchanged.",
            "",
            (
                "The fallback gate still needs its C1, two C2, FixedSelect and "
                "conditional repair forms plus restart/resume validation. These "
                "calls remain earmarked within reserve rows, not added twice. "
                "Superseded historical 14B acceptance rows do not mean fallback "
                "acceptance passed."
            ),
            "",
            "| Remaining class | Count |",
            "|---|---|",
        ]
        md += [
            f"| {r['call_class']} | {r['remaining_count']} |"
            for r in work["inventory"]
            if not r["superseded_14b_row"]
        ]
    if not any(r["scientific_accepted"] for r in llm):
        md += [
            "",
            (
                "No GPU output was scientifically accepted. This development "
                "configuration failed within the used allowance. Structural "
                "completion, confirmed semantic errors and unresolved assessments "
                "are distinguished below. No further diagnostic phase is "
                "automatically initiated."
            ),
        ]
    md += [
        "",
        "## What changed and what was wrong",
        "",
        (
            "The two actual C0 projections retain the same named nodes. Context "
            "2 adds an office-holding assertion and a greeting to the context-1 "
            "selection, without forming a separate continuing-office node. Its "
            "lower contextual score is a projection/relevance result, not a "
            "reversal of the extraction-competence gate."
        ),
        "",
        (
            "The original base C2 prefixes differ: context 1 reports selection with "
            "an excessive created-entity inventory; context 2 reports "
            "contextual-type and relation operations with mixed created record "
            "kinds. Neither completed its schema/entity declarations, so these "
            "surface differences cannot establish correct contrastive ontology "
            "construction."
        ),
        "",
        (
            "For a readable source example, evidence e1 directly narrates Fara "
            "Cedar's membership in Cedar Circle at story step 1. Both original C2 "
            "base prefixes nevertheless attach holder-level knowledge and intrinsic "
            "point validity. Narration does not establish a participant's "
            "knowledge, and observation time does not establish intrinsic onset "
            "or duration. These are source-only findings on complete received "
            "assertion records, not full canonical validation. Missing "
            "endpoint/type declarations remain unresolved; the CPU does not fill "
            "them from descriptions."
        ),
        "",
        (
            "| Received LLM prefix | Complete entity / event / assertion records "
            "| Source-only confirmed defect categories | Construction claims |"
        ),
        "|---|---|---|---|",
    ]
    for r in rows:
        if "partial_record_counts" in r:
            counts = r["partial_record_counts"]
            categories = sorted({d["category"] for d in r["partial_source_diagnostics"]})
            claims = "; ".join(
                d["operator"] + f" (claims {d['created_node_id_count']} created node IDs)"
                for d in r["received_construction_claims"]
            )
            md.append(
                "| "
                + r["condition"]
                + " / "
                + str(r["context"] or "query-blind")
                + (" / repair" if r.get("repair_parent") else " / base")
                + " | "
                + " / ".join(str(counts[k]) for k in ("entities", "events", "assertions"))
                + " | "
                + ", ".join(categories)
                + " | "
                + claims
                + " |"
            )
    md += [
        "",
        (
            "Counts describe complete JSON members received before truncation, "
            "not complete graphs. A zero received-node count does not mean the "
            "model authored an empty graph. C2's excessive creation claims and "
            "repeated unsupported epistemic form show that output capacity is "
            "not the only remaining issue. These historical prefixes remain "
            "failed regardless of the separately reported repairs."
        ),
    ]
    if transmitted_repairs:
        md += [
            "",
            "## Parent repairs: completion versus scientific assessment",
            "",
            "No checker was weakened. Unmatched paraphrases are unresolved, not false. "
            "A generated repair is a replacement model output, not a CPU-completed prefix. "
            "Server rejection before generation is an integration failure, "
            "not a model verdict. "
            "The HTML shows every assertion, cited source passage and unresolved check.",
            "",
            "| Condition / context | HTTP / JSON complete | Object budget | "
            "Confirmed source-semantic categories | Grounding / description unresolved | "
            "Scientific accepted |",
            "|---|---|---|---|---|---|",
        ]
        for r in transmitted_repairs:
            md.append(
                "| "
                + " | ".join(
                    [
                        r["condition"] + " / " + str(r["context"] or "query-blind"),
                        str(r.get("transport_complete")) + " / " + str(r.get("json_complete")),
                        fmt(r.get("generation_object_budget_valid")),
                        ", ".join(r.get("confirmed_semantic_categories", []))
                        or "none established by source-only checks",
                        fmt(r.get("grounding_unresolved_count"))
                        + " / "
                        + fmt(r.get("description_unresolved_count")),
                        str(r["scientific_accepted"]),
                    ]
                )
                + " |"
            )
        md += [
            "",
            "### Before/after received structure",
            "",
            "Counts for truncated bases describe complete received members only. "
            "Different counts are not themselves evidence of semantic improvement.",
            "",
            "| Condition / context | Base received nodes / assertions | "
            "Repair received nodes / assertions | Repair failure stage |",
            "|---|---|---|---|",
        ]
        for r in transmitted_repairs:
            base = next(b for b in rows if b.get("attempt_id") == r["repair_parent"])
            counts = base.get("partial_record_counts", {})
            nodes = counts.get("entities", 0) + counts.get("events", 0)
            after = r.get("partial_record_counts", {})
            before_assertions = counts.get("assertions", base.get("assertions_received", "—"))
            after_nodes = r.get("nodes_received", after.get("entities", 0) + after.get("events", 0))
            md.append(
                "| "
                + " | ".join(
                    [
                        r["condition"] + " / " + str(r["context"] or "query-blind"),
                        f"{nodes} / {before_assertions}",
                        f"{after_nodes} / "
                        f"{r.get('assertions_received', after.get('assertions', '—'))}",
                        r.get("failure_stage") or "none",
                    ]
                )
                + " |"
            )
    received_repairs = [
        g
        for g in graphs
        if g["row"].get("repair_parent")
        and g["row"].get("offline_prefix_observations")
        and not g["row"].get("construction_stage")
    ]
    if received_repairs:
        md += [
            "",
            "### Observable semantic changes in the repairs",
            "",
            "Offline prefix inspection only. Counts cover received records, not complete "
            "ontologies or an estimate of repair effectiveness.",
            "",
            "| Context | Attributed records, base → repair | Claimed unique nodes / ceiling | "
            "Identical assertion groups except ID |",
            "|---|---|---|---|",
        ]
        for g in received_repairs:
            r = g["row"]
            obs = r["offline_prefix_observations"]
            base = next(b for b in rows if b.get("attempt_id") == r["repair_parent"])
            md.append(
                f"| {r['context']} | "
                f"{base['offline_prefix_observations']['attributed_assertions']} → "
                f"{obs['attributed_assertions']} | {obs['unique_claimed_nodes']} / "
                f"{obs['node_budget']} | "
                f"{json.dumps(obs['identical_assertion_groups_except_id'])} |"
            )
        md += [
            "",
            "The observed unsupported holder-attribution form was removed from the "
            "received repair assertions. Intrinsic point validity remains unsupported for "
            "observation-only evidence. Both repairs use `include_exclude` to claim more "
            "distinct nodes than the unchanged ceiling permits. The frozen grammar bounds "
            "actual graph objects and single-kind constructive operators, but mixed-kind "
            "creation lists remain cross-field post-checks. This unencoded path allowed "
            "excessive claims and repeated administrative content; the capacity repair did "
            "not eliminate that expansion. No final graph was available for formal checking.",
            "",
            "Readable example selection: first received assertion citing the first "
            "supplied evidence record, in each repaired context (not selected for favorable "
            "performance). The HTML contains every other received assertion.",
            "",
            "Supplied evidence: " + evidence[0].text,
            "",
        ]
        for g in received_repairs:
            a = next(
                (
                    a
                    for a in g["graph"]["instance_graph"]["assertions"]
                    if "e1" in a["evidence_ids"]
                ),
                None,
            )
            if a is not None:
                md += [
                    f"Context {g['row']['context']}, `{a['assertion_id']}`: "
                    f"`{a['content'].get('subject_id')} → {a['content']['predicate_id']} → "
                    f"{a['content'].get('object_id')}`. Generated explanation: " + a["why_matters"],
                    "",
                    "Generated temporal fields: `"
                    + json.dumps(a["content"]["temporal_content"], separators=(",", ":"))
                    + "`. Epistemic scope: `"
                    + json.dumps(a["epistemic_scope"])
                    + "`.",
                    "",
                ]
        md += [
            "No entity or predicate declarations were received in either repair. "
            "Consequently endpoint meanings, role/type compatibility, description mapping "
            "and substantive construction correctness remain unresolved. The context-2 "
            "first assertion uses the same ID at both endpoints despite a description "
            "naming a person and a collective; it cannot be repaired by assigning labels "
            "from prose. Unmatched descriptions are not automatically false, but neither "
            "are they positively grounded by absence of a known contradiction."
        ]
    if staged_rows:
        md += [
            "",
            "## Staged exploratory development",
            "",
            "A owns schema and graph objects; B owns qualified assertions; C owns descriptions "
            "and construction reporting. Only assembled canonical outputs receive contextual "
            "draft scores. Intermediate completion is not canonical or scientific success. "
            "C1 stages finish or fail before the new C2 query-bearing transmissions. "
            "C2 contexts do not inherit C1 or each other's records. No checker changed.",
            "",
            f"New reservations: {len(staged_rows)}. Additional staged allocation: "
            f"{accounting['staged_seconds']:.6f} s. Complete canonical stage-C outputs: "
            f"{sum(bool(r.get('canonical_valid')) for r in staged_rows)}; scientific accepts: "
            f"{sum(bool(r.get('scientific_accepted')) for r in staged_rows)}.",
            "",
            "| Condition/context/stage | Nodes/assertions available | Confirmed source defects | "
            "Grounding/description unresolved | Failure stage |",
            "|---|---|---|---|---|",
        ]
        for r in staged_rows:
            md.append(
                f"| {r['condition']}/{r.get('context')}/{r['construction_stage']} "
                f"{'repair' if r.get('repair_parent') else 'base'} | "
                f"{r.get('nodes_received', '—')}/{r.get('assertions_received', '—')} | "
                f"{', '.join(r.get('confirmed_semantic_categories', [])) or 'none established'} | "
                f"{r.get('grounding_unresolved_count', '—')}/"
                f"{r.get('description_unresolved_count', '—')} | "
                f"{r.get('failure_stage') or 'none at this stage'} |"
            )
        md += [
            "",
            "The HTML provides the actual stage inventories, full assembled graphs where "
            "available, and each assertion alongside its cited evidence. "
            "Earlier failed rows remain "
            "unchanged evidence; no authored fixture is reported as GPU output.",
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
        (
            '<!doctype html><html lang="en"><meta '
            'charset="utf-8"><title>Exploratory development '
            "comparison</title><style>body{font:17px "
            "system-ui;margin:24px;color:#172335}h1,h2{line-height:1.2}"
            ".graph{height:850px;border:1px solid #bbb;background:#fcfcff}"
            "table{border-collapse:collapse;width:100%;font-size:15px;table-layout:fixed}"
            "th:nth-child(1){width:15%}th:nth-child(2){width:17%}"
            "th:nth-child(3){width:28%}th:nth-child(4){width:40%}"
            "td,th{border:1px solid #ccd;padding:8px;vertical-align:top;overflow-wrap:anywhere}"
            "pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:14px}"
            "section{margin:35px 0}.warning{background:#fff0ce;padding:14px}</style>"
            '<h1>Exploratory development comparison</h1><p class="warning">One preselected '
            "easy development world. Graphs show actual outputs, including failures; "
            "they are not expected answers or confirmatory evidence. Drag/zoom nodes; "
            "read assertion qualifications below each graph.</p><h2>Shared evidence</h2><ol>"
        )
    ]
    parts += [
        '<li id="e' + str(i) + '">' + html.escape(e.text) + "</li>"
        for i, e in enumerate(evidence, 1)
    ]
    parts.append(
        "</ol><nav>Jump to output: "
        + " · ".join(
            '<a href="#view'
            + str(i)
            + '">'
            + html.escape(
                g["row"]["condition"]
                + " / "
                + str(g["row"].get("context") or "query-blind")
                + (" / repair" if g["row"].get("repair_parent") else " / base")
                + (
                    " / stage " + g["row"]["construction_stage"]
                    if g["row"].get("construction_stage")
                    else ""
                )
            )
            + "</a>"
            for i, g in enumerate(graphs)
        )
        + "</nav><h2>Contrasting contexts</h2>"
    )
    for i, c in enumerate(contexts, 1):
        parts.append(
            "<h3>Context "
            + str(i)
            + "</h3><pre>"
            + html.escape(json.dumps(without_admin(c), indent=2))
            + "</pre>"
        )
    js = []
    from story_projection_onto.development_demo import aliases

    source_aliases = aliases(evidence)
    source_text = {e.evidence_id: e.text for e in evidence}
    source_text.update({source_aliases[e.evidence_id]: e.text for e in evidence})
    for i, g in enumerate(graphs):
        r = g["row"]
        graph = g["graph"]
        labels_here, _, assertions = label_graph(graph)
        title = (
            r["condition"]
            + " / "
            + str(r.get("context") or "query-blind")
            + (" / repair" if r.get("repair_parent") else " / base")
            + (" / stage " + r["construction_stage"] if r.get("construction_stage") else "")
        )
        parts.append(
            '<section id="view'
            + str(i)
            + '"><h2>'
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
            + (
                '</pre><div class="graph" id="g' + str(i) + '"></div>'
                if labels_here
                else (
                    '</pre><p class="warning">'
                    + (
                        "Generation was truncated before entity/event declarations. "
                        "Only complete received assertion members appear below; null checks "
                        "were not reached. Labels are not inferred from prose."
                        if r.get("output_tokens")
                        else "No generated graph is available. The server rejected this request "
                        "before generation; this is not a model-authored empty graph."
                    )
                    + "</p>"
                )
            )
            + (
                "<table><tr><th>Assertion / predicate</th><th>Bindings</th><th>Time "
                "/ epistemic / commitment</th><th>Evidence / description / assessment</th></tr>"
            )
        )
        for a in assertions:
            assessment = g.get("assessment", {})
            checks = {
                "grounding": "supported"
                if a["id"] in assessment.get("grounding_supported_assertion_ids", [])
                else "unresolved"
                if a["id"] in assessment.get("grounding_unresolved_assertion_ids", [])
                else "not formally assessed",
                "description": next(
                    (
                        d
                        for d in assessment.get("description_assessments", [])
                        if d["id"] == a["id"]
                    ),
                    "not formally assessed",
                ),
            }
            cells = [
                a["id"] + " / " + str(a["predicate"]),
                a["bindings"],
                json.dumps(
                    {k: a[k] for k in ("time", "epistemic", "commitment", "direction")}, indent=2
                ),
                json.dumps(
                    {
                        **{k: a[k] for k in ("evidence", "why_matters")},
                        "cited_source_text": {
                            e: source_text.get(e, "unresolved citation")
                            for e in a["evidence"] or []
                        },
                        "assessment": checks,
                    },
                    indent=2,
                ),
            ]
            parts.append(
                "<tr>"
                + "".join("<td><pre>" + html.escape(c) + "</pre></td>" for c in cells)
                + "</tr>"
            )
        parts.append(
            "</table><details><summary>Structural and source-assessment diagnostics</summary><pre>"
            + html.escape(
                json.dumps(
                    {
                        "formal": g.get("assessment"),
                        "source_only": r.get("confirmed_source_defects", []),
                        "offline_partial_record_source_checks": r.get("partial_source_diagnostics"),
                    },
                    indent=2,
                )
            )
            + "</pre></details>"
        )
        parts.append(
            ("<details><summary>All actual graph fields and construction decisions</summary><pre>")
            + html.escape(json.dumps(graph, indent=2))
            + "</pre></details></section>"
        )
        if labels_here:
            js.append({"id": "g" + str(i), "elements": graph_elements(graph, anchors)})
    parts.append(
        '<script src="../../ui/cytoscape.min.js"></script><script>const views='
        + json.dumps(js).replace("<", "\\u003c")
        + (
            ";for(const v of "
            'views){cytoscape({container:document.getElementById(v.id),elements:v.elements,layout:{name:"preset",padding:50},style:[{selector:"node",style:{label:"data(label)","text-wrap":"wrap","text-max-width":150,"font-size":18,width:35,height:35,"background-color":"#387cb0","text-valign":"bottom","text-margin-y":8}},{selector:"edge",style:{label:"data(label)","font-size":14,"text-rotation":"autorotate","curve-style":"bezier","target-arrow-shape":"triangle",width:2,"line-color":"#8b738f","target-arrow-color":"#8b738f","text-background-color":"#fff","text-background-opacity":.9}},{selector:".assertion",style:{shape:"diamond","background-color":"#ae7739"}}]});}</script></html>'
        )
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
