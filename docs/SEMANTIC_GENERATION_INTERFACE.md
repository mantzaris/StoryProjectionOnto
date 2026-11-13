# Named semantic interface — diagnostic only

This is the user-authorized CPU diagnostic interface and deterministic adapter,
not an activated study/held-out protocol. The existing C1/C2/FixedSelect builders,
transport, sampler, scheduling, accounting and scientific validators are unchanged.
The initial CPU-only patch performed no GPU start. The subsequent explicitly
authorized small live validation is bounded as recorded below. The model, tokenizer,
non-thinking template, sampling family and 12,288-token context remain pinned.

## Separation of model decisions and runtime facts

`semantic_generation.semantic_schema` derives its fields from `OntologyDraft`
and makes the following explicit representation changes. A field-inventory test
fails if a canonical scientific field disappears without a declared mapping.
All remaining scientific fields are required, including explicit empty arrays
and nullable fields. A missing confidence, citation, endpoint, qualification or
description fails; the adapter supplies no semantic substitute.

| Generated field(s) | Canonical destination and conversion |
|---|---|
| `contextual_interpretation` | Same field, copied |
| `local_schema.schema_id`, `abstraction` | Same fields, copied; schema ID remains model-authored because a construction decision may target this semantic container |
| `contextual_types`: `type_id`, `label`, `definition`, `parent_upper_type`, `abstraction`, `evidence_ids` | Same fields, copied |
| `predicates`: `predicate_id`, `label`, `definition`, `arity`, `domain_type_ids`, `range_type_ids`, `role_names`, `parent_upper_relation`, `evidence_ids` | Same fields, copied |
| Entities: `entity_id`, `label`, `supported_mention_candidate_ids`, `aliases`, `contextual_type_id`, `contextual_role`, `abstraction`, `temporal_state`, `uncertainty`, `confidence`, `evidence_ids`, `description`, `description_assertion_ids` | Same fields, copied |
| Events: `event_id`, `label`, `contextual_type_id`, `occurrence_time`, `reification_reason`, `uncertainty`, `confidence`, `evidence_ids`, `description`, `description_assertion_ids` | Same fields, copied |
| Assertion/proposition `form=binary`, `subject_id`, `object_id` | Remove only the redundant `form` tag; copy both endpoints; inactive canonical roles become empty |
| Assertion/proposition `form=nary`, `roles[{role,object_id,evidence_ids}]` | Remove only the redundant tag; copy every role/citation; inactive binary fields become null |
| Assertions: `assertion_id`, `predicate_id`, `proposition_content_id`, `direction`, `temporal_scope`, `epistemic_scope`, `narrative_commitment`, `confidence`, `evidence_ids`, `contextual_relevance`, `why_matters`, `why_matters_evidence_ids` | Same fields, copied; no default direction or confidence is inferred |
| Proposition contents: `proposition_content_id`, `predicate_id`, bindings, `temporal_content`, `evidence_ids` | Same fields, copied; not automatically world-committed |
| `TemporalScope.story_time`, `.validity_time`, entity/event time and holder-relative time | Copy the selected form and all supplied bounds/anchors/constraints/labels/reasons; inactive canonical fields become null/empty only |
| `discourse_position{passage_order,sentence_order,token_order}`, `revelation_position{revelation_order,label}` | Model-selected scientific coordinates copied; **not** derived by taking a minimum citation position or by using wall time |
| `epistemic_scope{holder_id,attitude,proposition_content_id,holder_relative_time,evidence_ids}` | Same fields, copied, or explicit null; no holder/commitment inferred |
| `provenance[{evidence_id,confidence}]` | Both fields copied; confidence is a separate model judgment, not replaced by source or assertion confidence |
| Decisions: `decision_id`, `operator`, `evidence_ids`, `rationale`, `input_object_ids`, `created_object_ids`, `removed_object_ids` | Same fields, copied |
| `omissions[{evidence_id,reason,confidence}]`, `uncertainty_and_abstentions` | Same fields, copied |

The adapter retains a separately hashed raw semantic payload identity and a
provenance receipt enumerating runtime additions. It deep-copies rather than
editing the generated object. Canonical string-edge whitespace normalization is
declared; semantic strings are neither summarized nor completed.

| Runtime-added field | Exact source |
|---|---|
| Every record's `schema_version` | Installed canonical contract constant `1.0.0` |
| Every record's `content_hash` | Canonical serialization's SHA-256, verified by the existing immutable-record validators |
| `decisions[i].decided_at` | Observed generation completion timestamp; receipt retains the observed start/end interval. This is **not** a fabricated internal model decision instant and cannot be backdated to the request |
| Each citation's `provenance_id` | Semantic-payload hash plus assertion/citation array indices; administrative identifier only |
| Citation `extraction_method` | `named-semantic-diagnostic-v1` adapter revision |
| Citation `locator`, `source_artifact_hash` | Exact fields from the frozen evidence record named by the **model-selected** citation; no source substitution |
| `budget_accounting.nodes_used`, `assertions_used` | Exact returned entity+event and qualified-assertion counts |
| `display_nodes_used`, `display_assertions_used` | Same counts: this diagnostic displays every returned object, with no clipping or selection |
| `input_tokens`, `output_tokens` | Observed request/response usage supplied through typed `ExecutionFacts`; not authored zero sentinels or estimates |
| Receipt request/response hashes, execution interval, source evidence hashes, adapter/schema hashes | Frozen effective request, retained response, clock observations, supplied source records and installed adapter/schema |

`ExecutionFacts` rejects malformed hashes, naive/reversed times and invalid token
counts. Values in CPU fixtures are explicitly authored test data, not timings or
usage of the study LLM. A live caller must bind these facts to the existing
request, response journal, metered call and generation outcome.

## Schema and prompt agreement

The model gets a readable field/type guide generated from the **same schema**
sent as `guided_json`, plus the complete one-passage development evidence, upper
vocabulary, horizon, capabilities and budgets. No expected graph is supplied.
There are no tuples or opaque handles. Local IDs start with `n` and remain
distinct from supplied evidence/candidate IDs. Citation/mention/upper-parent
vocabularies are bound by enums. Descriptions are ordinary nonempty text, not
restrictive identifier patterns.

Binary and n-ary assertions/propositions are closed tagged alternatives; neither
can lack bindings or mix both forms. Point time requires its integer coordinate.
Intervals require a start, an end or both; relative time requires anchor/relation;
partial-order time requires constraints. Unknown, not-applicable, horizon-withheld
and invalid are distinct closed alternatives. No branch requires irrelevant
coordinate fields. Optional labels/reasons are preserved; unknown/withheld/invalid
require a reason. Unsupported precision is never forced.

For the small diagnostic only, the grammar requires at least two entity/event
records, at least one assertion, one decision, types and predicates. Three graph
alternatives cover entity/entity, entity/event and event/event minima. The
combined maximum of four nodes and valid supported content still require
post-validation. The general schema permits empty graphs and abstention; their
registered ITT/NA treatment is unchanged, not declared successful by the adapter.

## Invariants retained after grammar decoding

The pinned backend is verified on CPU via both `Grammar.from_json_schema` and
`GrammarCompiler.compile_json_schema(..., any_whitespace=False)`, using the
pinned tokenizer. The no-arbitrary-whitespace setting remains. The grammar's
property-order restrictions are exposed by the generated guide; authored
capacity fixtures are serialized in that order. Retained model responses are
**never reordered or repaired** for replay.

The schema does not encode output-dependent membership/uniqueness, predicate
arity versus actual roles, interval bound ordering, temporal DAG acyclicity,
horizon/source support, description entailment, citation subsets/coverage,
source-confidence ceilings, evidence-supported temporal precision, epistemic
holder/proposition equality or commitment consistency, capability restrictions,
construction timing, or cross-array total object budgets. Existing canonical
and `validate_draft_structure` checks retain the applicable invariants, followed
by the existing scorer-only development grounding audit. Missing evidence or
referents are errors, never a reason for the adapter to generate them.

The existing controller now exposes `validate_named_semantic_diagnostic`: it
binds request/response hashes and observed usage, reconstructs the semantic
payload, then calls the **same** `_validate_small_canonical_draft` structural and
scorer-only grounding checks. Its legacy `validate_small_diagnostic` entry point
still requires raw zero-token sentinels; those are not incorrectly imposed on a
runtime-populated record. Neither scheduler nor service-start authority is
changed. This is a representation-stage distinction,
not removal of a scientific or budget validator. Grammar compliance,
reconstruction, canonical acceptance and scientific acceptance are separate
reported statuses. `reconstruct` does not claim scientific acceptance.

## Replay of actual A/B/C failures

Replay reassembles each original restricted SSE response from hash-verified
fragments and confirms its decoded identity. No response is corrected.

| Retained output | Revised contract rejects | Still requires canonical/scientific checking |
|---|---|---|
| A, unconstrained named | All three assertions lack endpoints/roles; point times lack coordinates; local parents are missing; legacy bookkeeping is forbidden | Duplicate arrival entity/event identity, supported descriptions and task fulfillment |
| B, constrained named | Its assertion lacks endpoints/roles; point plus partial-order fields are mutually exclusive; `carries` and `arrived_at` are not supplied upper parents; legacy bookkeeping is forbidden | Evidence support for numeric dawn coordinate, intrinsic validity, relation/type choices and decision substance |
| C, production tuples | Raw tuple representation is not named semantic JSON; unchanged decoded empty graph fails the small minima | Empty general abstention would remain representable but would not satisfy this task |

Old responses also lack newly required explicit fields and `form` tags. Therefore
their whole-output rejection alone is not evidence the new prompt will work.
Unchanged field-level subtrees isolate the specific observed contract defects;
they are diagnostic checks, not retrospective acceptance. The original A/B/C
outcomes and artifacts are preserved.

## Activation boundary and next live proposal

### Approved small validation session, 2026-09-07

The existing controller's `--semantic` mode implements the user's subsequent
authorization: one new start, at most three small attempts, at most 1,100 new
allocated seconds, preserving 5,363.502630 historical seconds. Its global ceiling
is 6,463.502630 seconds, not a reset of earlier blocks. Startup/live/generation/
validation/shutdown/guard caps are 360/15/180 per call/30/60/5 seconds; the whole
deadline takes precedence. Only complete-run forecast admission is excepted.
The 33,660 scheduled and strict 36,000 actual limits remain unchanged.

Both requests are prepared before allocation. The first is the frozen request
below. The second uses only development evidence `ev-03` (mechanic Ash repairs
the river pump while courier Lio remains at North Gate), selected before seeing
any first-call output. It runs only after first-task scientific acceptance. The
same semantic schema policy, tokenizer, settings and output allowance apply.
There are no full C1, FixedSelect, ordinary or held-out calls in this mode.

A remaining call may repair an observed schema/canonical structural defect by
versioning the prompt with that exact error. Scorer-only grounding feedback is
never supplied. No identical blind retry, semantic default, authored answer or
new startup is permitted. Every attempt has its own restricted request, response,
outcome and repair lineage; these are new diagnostics, not historical reserve
slots or confirmatory results. After useful calls stop, the service shuts down.
The existing guardian and repaired resource monitor enforce these bounds.

Final CPU verification is backed up under
`artifacts/restricted/semantic-interface.0TerjX/verified/`. The exact readable
request and schema are `MODEL_REQUEST_AND_SCHEMA.md`; authoritative payload
bytes are in `request.json`, with the rendered chat and binding alongside.
All **nine manifest-listed artifacts** and **four source-file hashes** reconcile.
The first CPU verifier import-path failure and subsequent CPU logs are preserved;
none involved GPU allocation.

**89 focused tests passed locally and on the pod** (remote 23.61 s), including
the original controller/tuple and intrinsic-validity checks. The final pinned
CPU verifier took **66.537044 s**, with **0.204064 s** for the small grammar's
conversion/compilation. This is CPU cost, not measured startup or inference speed.

| Capacity check, not model output | Completion tokens | Additional termination allowance |
|---|---:|---:|
| Authored two-node small fragment | 844 | 1 |
| Authored events plus four-role n-ary assertion | 1,881 | 1 |
| Authored holder and nonasserted proposition | 2,029 | 1 |

Each complete fixture is accepted by the pinned CPU grammar matcher and
reconstructs its scientific fields. The **actual small prompt is 4,712 tokens**
including the pinned chat template/special tokens; reserving **6,144 output
tokens** leaves **1,432 tokens** below 12,288. Whitespace is counted, not assumed
zero. These examples do not prove a demanding production C1 graph will fit,
that it can fit intact into FixedSelect, or that the model can generate them.

Pinned matcher rejection probes pass for missing endpoints, mixed temporal
forms, empty small graphs, invented evidence IDs, invalid upper parents, and
nonlocal/overlength identifiers. Positive controls pass. The A/B/C raw streams
are hash-verified and replayed unchanged, not converted into successful results.
Accepted model outputs under the new interface: **none attempted**.

Frozen diagnostic request SHA-256:
`bf083c095b1655da8bfb060f98e017c22e7554ef443e59c3bf22cbd5dc266ac4`.
Semantic schema SHA-256:
`e83dbf588e86700af90250f0fcd020620bcaa2c2ad48829d9e5a4dd30fea3bdd`.

Production adoption requires a documented pre-held-out method amendment naming
the new serialization/adapter, prompt/schema revisions, grammar-difference
manifest, decision-timestamp semantics and budget metadata provenance. Apply it
symmetrically to constructive C1/C2 and their allowed ablations; FixedSelect
still needs a sealed-ID selection-only schema and an intact genuine C1 graph.
No multi-call construction, model/context change, wider evidence, hidden graph,
validator relaxation or dropped comparison is authorized here. The general
diagnostic schema is **not** suitable for silently replacing FixedSelect or the
qualification-removal ablation. Full-study packing, acceptance including restart/
resume, development competence, timing admission and human review remain gates.

Proposed next authorization: **one diagnostic-only service session, at most
three small calls, at most 1,100 additional allocated seconds**, no full C1 or
ordinary study run. Startup 360 s, live checks 15 s, each call at most 180 s,
validation/drain 30 s, protected shutdown 60 s, guard 5 s; these sum to 1,010 s,
leaving 90 s for a bounded evidence-supported CPU repair while the same healthy
service is allocated. Enforce both stages and the whole deadline. Prepare the
base request and checks before allocation; call one tests this unchanged small
task. Further calls require a specific supported diagnostic repair or a declared
small discriminating check, not identical blind retries. Stop on unsafe
cancellation/service health or when no useful authorized work remains.

Historical allocation is **5,363.502630 s**; that proposal's maximum global
allocation would be **6,463.502630 s**. The 33,660/36,000 s limits remain. The
ordinary complete-run forecast still fails (last conservative total
45,525.515843 s); these CPU capacity checks supply no valid scientific throughput
estimate. A new live session therefore needs explicit authorization including
the diagnostic-only forecast exception. None is launched by this handoff.
