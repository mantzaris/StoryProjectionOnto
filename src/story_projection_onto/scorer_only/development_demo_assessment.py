"""Exploratory development scoring; never imported by model request builders.

Feedback is source-only or structural. Gold matching is reported separately and
never exported as correct bindings/answers in model-facing feedback.
"""

from __future__ import annotations

import re
from pathlib import Path

from story_projection_onto.contracts import ConstructionCapabilities, SpoilerHorizon
from story_projection_onto.development_demo import reference_translation
from story_projection_onto.metrics.alignment import score_alignment
from story_projection_onto.metrics.rare import annotations_from_gold, score_rare_pivotal
from story_projection_onto.validate import validate_draft_structure


def normalized(text):
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def source_feedback(parsed, evidence, mapping):
    """Collect demonstrable issues even when reconstruction fails elsewhere."""
    value = reference_translation(parsed, {v: k for k, v in mapping.items()})
    source = {e.evidence_id: e for e in evidence}
    graph = value.get("instance_graph", {}) if isinstance(value, dict) else {}
    defects = []

    def error(path, category, constraint, actual):
        defects.append(
            {
                "path": path,
                "category": category,
                "constraint": constraint,
                "generated_value": actual,
            }
        )

    attitudes = {
        "known": r"\b(knew|knows|knowledge)\b",
        "believed": r"\b(believed|believes|belief)\b",
        "reported": r"\b(reported|reports|said|says)\b",
        "denied": r"\b(denied|denies|denial)\b",
        "uncertain": r"\b(uncertain|unsure|suspects|suspected)\b",
    }
    for i, a in enumerate(graph.get("assertions", [])):
        if not isinstance(a, dict):
            continue
        path = f"/instance_graph/assertions/{i}"
        texts = [source[e].text for e in a.get("evidence_ids", []) if e in source]
        scope = a.get("epistemic_scope")
        if isinstance(scope, dict):
            citations = [source[e].text for e in scope.get("evidence_ids", []) if e in source]
            attitude = scope.get("attitude")
            if (
                attitude in attitudes
                and citations
                and not any(re.search(attitudes[attitude], t, re.I) for t in citations)
            ):
                error(
                    path + "/epistemic_scope",
                    "unsupported_attribution",
                    "Cited direct narration does not state this holder attitude. Attribution needs positive evidence; it may be retracted, not defaulted to world truth.",
                    {"holder_id": scope.get("holder_id"), "attitude": attitude},
                )
        content = a.get("content", {})
        time = a.get("temporal_scope")
        if time == "content":
            time = content.get("temporal_content")
        if isinstance(time, dict):
            validity = time.get("validity_time", {})
            if validity.get("kind") in {"point", "interval"} and texts:
                # General evidence-grounded rule: observation is not intrinsic
                # onset/duration. Only diagnose when no duration statement exists.
                if not any(
                    re.search(r"\b(held|lasted|from .* through|valid|until)\b", t, re.I)
                    for t in texts
                ):
                    error(
                        path + "/temporal_scope/validity_time",
                        "unsupported_intrinsic_precision",
                        "Occurrence/observation time alone does not support intrinsic validity bounds. Use evidence-supported duration or explicit unknown.",
                        validity,
                    )
    return defects


def assess(draft, *, evidence, upper, horizon, budgets, root: Path, ordinal=None, fixed=False):
    from story_projection_onto.development_continuation import load_development_prequery_evidence
    from story_projection_onto.development_runtime import load_development_call_manifest
    from story_projection_onto.scorer_only.development_assessment import (
        DevelopmentScientificAssessmentProvider,
        _prediction_bundle,
        resolve_development_world_id,
    )
    from story_projection_onto.scorer_only.direct_extraction import (
        compile_direct_reference,
        score_direct_preconstruction,
    )
    from story_projection_onto.synthetic_benchmark import ScorerWorldArtifact

    structure = validate_draft_structure(
        draft=draft,
        upper_ontology=upper,
        evidence=evidence,
        horizon=SpoilerHorizon.model_validate(horizon),
        budgets=budgets,
        capabilities=ConstructionCapabilities.fixed_selection()
        if fixed
        else ConstructionCapabilities.active_construction(),
    )
    neutral, visible, _ = load_development_prequery_evidence(
        root, load_development_call_manifest(root)
    )
    world = resolve_development_world_id(
        root, visible["dev-unit-01"].content_hash, neutral["dev-unit-01"].content_hash
    )
    scorer = ScorerWorldArtifact.model_validate_json(
        (root / f"data/synthetic/scorer_only/development/{world}.json").read_bytes()
    )
    assert scorer.world_spec.split.value == "development"
    reference = compile_direct_reference(scorer, evidence, root)
    extraction = score_direct_preconstruction(draft, reference, evidence)
    positive = {r["prediction_id"] for r in extraction["strict_representation_matches"]}
    evidence_by_id = {e.evidence_id: e for e in evidence}
    descriptions = []
    for a in draft.instance_graph.assertions:
        texts = [
            normalized(evidence_by_id[e].text)
            for e in a.why_matters_evidence_ids
            if e in evidence_by_id
        ]
        descriptions.append(
            {
                "id": a.assertion_id,
                "status": "supported"
                if any(normalized(a.why_matters) in t for t in texts)
                else "unresolved",
                "check": "source-substring positive coverage only; paraphrase outside coverage is not false",
            }
        )
    # Structural validator independently checks description-to-assertion mapping,
    # type/role compatibility, event connectivity, target and budget correctness.
    result = {
        "structure": structure.model_dump(mode="json"),
        "source_direct_matching": extraction,
        "grounding_supported_assertion_ids": sorted(positive),
        "grounding_unresolved_assertion_ids": sorted(
            a.assertion_id
            for a in draft.instance_graph.assertions
            if a.assertion_id not in positive
        ),
        "description_assessments": descriptions,
        "contextual_metrics": None,
        "scientific_accepted": structure.accepted
        and bool(draft.instance_graph.assertions)
        and len(positive) == len(draft.instance_graph.assertions)
        and all(d["status"] == "supported" for d in descriptions),
        "scientific_rule": "unchanged structural validator plus positive existing source-bound strict matches and source-supported descriptions; unresolved is not accepted; contextual F1 is not an acceptance threshold",
        "reference_scope": "complete query-blind source direct-witness alternatives; no query-specific feedback",
    }
    if ordinal is not None:
        plan = DevelopmentScientificAssessmentProvider._alignment_plan(scorer, ordinal)
        nodes, assertions, _ = _prediction_bundle(
            local_schema=draft.local_schema,
            instance_graph=draft.instance_graph,
            plan=plan,
            valid_evidence_ids=frozenset(evidence_by_id),
        )
        score = score_alignment(plan=plan, predicted_nodes=nodes, predicted_assertions=assertions)
        matched = frozenset(m.target_id for m in score.strict_assertion_matches)
        rare = score_rare_pivotal(
            annotations=annotations_from_gold(scorer.gold_projections[ordinal - 1]),
            strictly_matched_assertion_target_ids=matched,
        )
        result["contextual_metrics"] = {
            "alignment": score.model_dump(mode="json"),
            "rare": rare.model_dump(mode="json"),
        }
    return result


def compact_feedback(source_defects, structure=None, canonical_error=None):
    items = list(source_defects)
    if structure:
        for d in structure.get("diagnostics", []):
            items.append({"category": d["code"], "path": d["path"], "constraint": d["message"]})
    if canonical_error:
        items.append(
            {
                "category": "reconstruction",
                "path": canonical_error.get("path", "/"),
                "constraint": canonical_error["message"],
            }
        )
    grouped = {}
    for item in items:
        import json

        key = (
            item["category"],
            item.get("constraint", ""),
            json.dumps(item.get("generated_value"), sort_keys=True),
        )
        row = grouped.setdefault(key, {"category": key[0], "constraint": key[1], "paths": []})
        if "generated_value" in item:
            row["generated_value"] = item["generated_value"]
        if item["path"] not in row["paths"]:
            row["paths"].append(item["path"])
    return list(grouped.values())


def exception_feedback(exc):
    """Compact typed errors only; full chain remains restricted, never truncated."""
    if hasattr(exc, "diagnostics"):
        return [
            {
                "category": d["category"],
                "path": p,
                "constraint": d["constraint"],
                "generated_value": d.get("referenced_id"),
            }
            for d in exc.diagnostics
            for p in d["paths"]
        ]
    if hasattr(exc, "errors"):
        return [
            {
                "category": "canonical_contract",
                "path": "/" + "/".join(map(str, d["loc"])),
                "constraint": d["msg"],
            }
            for d in exc.errors(include_input=False, include_url=False)
        ]
    return [{"category": "canonical_contract", "path": "/", "constraint": str(exc)}]
