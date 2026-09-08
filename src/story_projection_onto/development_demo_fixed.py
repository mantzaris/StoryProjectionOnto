"""Development selection-only adapter; no graph completion or semantic mutation."""

import json
from dataclasses import asdict
from types import SimpleNamespace

from .contracts import (
    BudgetAccounting,
    ConditionName,
    FixedOntologyInput,
    InstanceGraph,
    LocalContextSchema,
    OntologyDraft,
    canonical_sha256,
)
from .development_demo import evidence_text, read, reference_translation, sources, without_admin
from .gpu_runtime import ChatMessage, GuidedJSONRequest
from .llm import (
    DecodingManifest,
    DecodingPass,
    PackingReport,
    PackingSection,
    enforce_fixed_select_draft,
    sealed_inventory_from_fixed_ontology,
)
from .semantic_generation import _array, _object

GROUPS = {
    "entities": "entity_id",
    "events": "event_id",
    "assertions": "assertion_id",
    "proposition_contents": "proposition_content_id",
    "contextual_types": "type_id",
    "predicates": "predicate_id",
}


def records(c1):
    return {
        k: getattr(
            c1.local_schema if k in {"contextual_types", "predicates"} else c1.instance_graph, k
        )
        for k in GROUPS
    }


def prepare_fixed(
    root, kind, tokenizer, manifest, *, c1, seal, evidence, mapping, config, feedback=None
):
    _, unit, neutral = sources(root)
    ordinal = int(kind[-1])
    budgets = config.projection_budgets_by_unit[unit["unit_id"]]
    schema = _object(
        {
            "selected_" + k: _array({"enum": [getattr(r, GROUPS[k]) for r in rows]})
            if rows
            else {"type": "array", "items": {"type": "string"}, "maxItems": 0}
            for k, rows in records(c1).items()
        }
        | {
            "contextual_interpretation": {"type": "string", "minLength": 1},
            "uncertainty_and_abstentions": _array({"type": "string", "minLength": 1}),
        }
    )
    body = {
        "evidence_index": evidence_text(evidence, mapping),
        "context": reference_translation(
            without_admin(
                read(root / unit["query_stages"][ordinal - 1]["relative_path"] / "query.json")[
                    "query"
                ]
            ),
            mapping,
        ),
        "sealed_horizon": without_admin(neutral["snapshot"]["horizon"]),
        "upper_ontology": without_admin(config.upper_ontology.model_dump(mode="json")),
        "budgets": without_admin(budgets.model_dump(mode="json")),
        "intact_c1_graph": reference_translation(
            without_admin(c1.model_dump(mode="json")), mapping
        ),
        "seal_hash": seal.content_hash,
    }
    messages = [
        ChatMessage(
            role="system",
            content="Select only existing records from the supplied intact sealed C1 ontology for this context. No creation, merging, splitting, predicate change, rewritten content or new qualifications. Return selected IDs in the corresponding arrays, an evidence-supported contextual interpretation and uncertainties. Select every required dependency explicitly, including types, predicates, endpoints, holders, propositions and description assertions; runtime will not add them. Never exceed object/display budgets. Return concise JSON matching this selection schema: "
            + json.dumps(schema, separators=(",", ":")),
        ),
        ChatMessage(role="user", content=json.dumps(body, separators=(",", ":"))),
    ]
    if feedback:
        messages.append(
            ChatMessage(
                role="user",
                content=json.dumps(
                    {
                        "repair": "Reconsider the previous selection; no semantic construction is allowed.",
                        "diagnostics": feedback,
                    },
                    separators=(",", ":"),
                ),
            )
        )
    count = len(
        tokenizer.apply_chat_template(
            [asdict(m) for m in messages],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )
    if count > 10240:
        raise ValueError(
            f"Intact C1 FixedSelect input {count} + 2048 output exceeds 12288; graph not truncated"
        )
    decoding = DecodingManifest(
        decoding_pass=DecodingPass.REPAIR if feedback else DecodingPass.FIRST_PASS,
        maximum_input_tokens=10240,
        maximum_output_tokens=2048,
        seed=1988649846,
        eos_token_id=manifest.eos_token_id,
        end_of_turn_token_ids=manifest.end_of_turn_token_ids,
        stop_token_ids=manifest.stop_token_ids,
        chat_template_hash=manifest.chat_template_sha256,
        structured_decoder="vllm-0.10.2-xgrammar-no-fallback",
        tokenizer_revision=manifest.revision,
        output_schema_hash=canonical_sha256(schema),
    )
    names = (
        "evidence_packet",
        "query_context",
        "upper_ontology",
        "output_schema",
        "system_prompt",
        "sealed_ontology",
    )
    sections = tuple(
        PackingSection(name=n, section_content_hash=canonical_sha256(body), token_count=0)
        for n in names
    ) + (
        PackingSection(
            name="complete_fixed_request",
            section_content_hash=canonical_sha256([asdict(m) for m in messages]),
            token_count=count,
        ),
    )
    packing = PackingReport.build(
        condition=ConditionName.A_FIXED_SELECT,
        tokenizer_revision=manifest.revision,
        maximum_model_tokens=12288,
        maximum_input_tokens=10240,
        reserved_output_tokens=2048,
        sections=sections,
        required_section_names=(*names, "complete_fixed_request"),
        complete_evidence_packet=True,
        complete_sealed_ontology=True,
    )
    return (
        GuidedJSONRequest(
            request_id="development-demo-" + kind + ("-repair" if feedback else ""),
            model_name="qwen3-8b-awq-fallback",
            condition=ConditionName.A_FIXED_SELECT,
            messages=tuple(messages),
            output_schema=schema,
            decoding=decoding,
            packing=packing,
            rendered_input_token_count=count,
            stream_response=True,
        ),
        evidence,
        mapping,
        budgets,
    )


def adapt_fixed(value, *, c1, seal, upper, execution):
    selected = {}
    for key, rows in records(c1).items():
        ids = value["selected_" + key]
        available = {getattr(r, GROUPS[key]): r for r in rows}
        if len(ids) != len(set(ids)) or set(ids) - set(available):
            raise ValueError(f"Duplicate/unknown selected {key}; no reference guessing")
        selected[key] = tuple(available[i] for i in ids)
    schema = LocalContextSchema(
        **{
            k: v
            for k, v in without_admin(c1.local_schema.model_dump(mode="python")).items()
            if k not in {"contextual_types", "predicates"}
        },
        contextual_types=selected.pop("contextual_types"),
        predicates=selected.pop("predicates"),
    )
    graph = InstanceGraph(**selected)
    node_count = len(graph.entities) + len(graph.events)
    draft = OntologyDraft(
        contextual_interpretation=value["contextual_interpretation"],
        local_schema=schema,
        instance_graph=graph,
        decisions=(),
        uncertainty_and_abstentions=tuple(value["uncertainty_and_abstentions"]),
        budget_accounting=BudgetAccounting(
            nodes_used=node_count,
            assertions_used=len(graph.assertions),
            display_nodes_used=node_count,
            display_assertions_used=len(graph.assertions),
            input_tokens=execution.input_tokens,
            output_tokens=execution.output_tokens,
        ),
    )
    fixed = FixedOntologyInput(
        construction_seal=seal,
        upper_ontology=upper,
        local_schema=c1.local_schema,
        instance_graph=c1.instance_graph,
    )
    inventory = sealed_inventory_from_fixed_ontology(fixed, seed_block=1, source_draft=c1)
    enforce_fixed_select_draft(draft, sealed=inventory, seed_block=1)
    return SimpleNamespace(
        draft=draft,
        provenance={
            "adapter": "development-selection-only-v1",
            "source_seal": seal.content_hash,
            "runtime_derived": "exact selected record copies and measured token counts; no dependency completion",
            "production_adoption_authorized": False,
        },
    )
