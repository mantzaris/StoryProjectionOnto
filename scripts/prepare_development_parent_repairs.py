"""Prepare source-only, parent-bound feedback from complete retained JSON members.

Never turns a truncated prefix into a canonical draft or supplies missing semantics.
"""

import argparse
import json
from pathlib import Path

from scripts.report_preliminary_development import complete_record_fragments, streamed_content
from story_projection_onto.contracts import EvidenceRecord
from story_projection_onto.development_demo import aliases, prepare_request, read, sources
from story_projection_onto.gpu_runtime import TokenizerManifest
from story_projection_onto.manifest import write_json_atomic
from story_projection_onto.scorer_only.development_demo_assessment import source_feedback


def prepare(root, run, destination):
    from transformers import AutoTokenizer

    _, _, neutral = sources(root)
    evidence = tuple(EvidenceRecord.model_validate(e) for e in neutral["evidence"])
    mapping = aliases(evidence)
    t = AutoTokenizer.from_pretrained(
        root / "artifacts/restricted/pinned-tokenizer-cpu", local_files_only=True
    )
    m = TokenizerManifest(**read(run / "rendered-c1.json")["tokenizer_manifest"])
    result = {}
    for path in sorted(run.glob("*/outcome.json")):
        old = read(path)
        if old["repair_parent"] or old["scientific_accepted"]:
            continue
        if old["transport_metadata"].get("finish_reason") != "length":
            continue
        text = streamed_content(run, old["request_hash"])
        partial = complete_record_fragments(text)
        full = source_feedback(partial, evidence, mapping)
        feedback = [
            {
                "category": "incomplete_output",
                "path": "/",
                "constraint": "The previous answer ended at the output cap. Return a complete concise replacement, not a continuation. Budget ceilings are not fill targets. Prioritize a coherent evidence-supported graph; declare omissions. Do not emit speculative identifier ranges or repetitive citations.",
            }
        ]
        categories = {d["category"] for d in full}
        if "unsupported_attribution" in categories:
            feedback.append(
                {
                    "category": "unsupported_attribution",
                    "path": "/instance_graph/assertions/*/epistemic_scope",
                    "constraint": "The received assertions attribute known to holders, but their cited narration does not state holder knowledge. Reconsider attribution; use the non-attributed alternative where supported. A participant is not automatically a knower.",
                }
            )
        if "unsupported_intrinsic_precision" in categories:
            feedback.append(
                {
                    "category": "unsupported_intrinsic_precision",
                    "path": "/instance_graph/assertions/*/content/temporal_content/validity_time",
                    "constraint": "The received assertions give point validity from occurrence-only evidence. Observation is not intrinsic onset or duration; retain explicit unknown where bounds are unsupported.",
                }
            )
        nodes = set()
        for d in partial["decisions"]:
            nodes.update(
                x
                for x in d.get("created_object_ids", [])
                if isinstance(x, str) and x.startswith(("nE", "nV"))
            )
        if nodes:
            full.append(
                {
                    "category": "declared_node_inventory",
                    "declared_created_node_count": len(nodes),
                    "budget": 10 if old["kind"] != "c1" else 30,
                }
            )
        if partial["decisions"]:
            feedback.append(
                {
                    "category": "construction_reporting",
                    "path": "/decisions",
                    "constraint": "Operations must describe actual construction and only the corresponding created record kinds. Selection/supported_description alone is not substantive construction. Do not claim undeclared placeholder ranges; honor the stated total node budget.",
                }
            )

        def duplicates(value, path=""):
            if isinstance(value, dict):
                for k, v in value.items():
                    if (
                        k.endswith("_ids")
                        and isinstance(v, list)
                        and all(isinstance(x, str) for x in v)
                        and len(v) != len(set(v))
                    ):
                        full.append({"category": "duplicate_reference", "path": path + "/" + k})
                    duplicates(v, path + "/" + k)
            elif isinstance(value, list):
                for i, v in enumerate(value):
                    duplicates(v, path + "/" + str(i))

        duplicates(partial)
        if any(d["category"] == "duplicate_reference" for d in full):
            feedback.append(
                {
                    "category": "duplicate_reference",
                    "path": "/*/evidence_ids",
                    "constraint": "Identifier lists contain duplicates. Cite each supporting witness once, only where it supports that record; blanket repeated lists do not establish grounding.",
                }
            )
        # Complete declarations are not inferred from the truncated prefix.
        candidate = {
            "parent_request_hash": old["request_hash"],
            "parent_response_hash": old["transport_metadata"]["response_sha256"],
            "diagnostics": feedback,
            "full_source_diagnostics": full,
            "historical_status": "failed; immutable",
            "partial_inspection_only": True,
        }
        q, *_ = prepare_request(
            root, old["kind"], t, m, previous={"parent": old["attempt_id"]}, feedback=feedback
        )
        candidate["prepared_request_hash"] = q.request_hash
        candidate["input_tokens"] = q.rendered_input_token_count
        candidate["output_allowance"] = q.decoding.maximum_output_tokens
        result[old["attempt_id"]] = candidate
        write_json_atomic(
            q.wire_payload(), destination.parent / (old["kind"] + "-parent-repair-request.json")
        )
    write_json_atomic(result, destination)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--destination", type=Path, required=True)
    a = p.parse_args()
    result = prepare(Path.cwd(), a.run, a.destination)
    print(
        json.dumps(
            {
                k: {x: v[x] for x in ("input_tokens", "output_allowance", "prepared_request_hash")}
                for k, v in result.items()
            },
            indent=2,
        )
    )
