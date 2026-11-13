# Named-semantic ID diagnosis and CPU repair — 2026-09-08 UTC

No GPU start or inference. Original responses, ledgers, benchmark, thresholds and
review package are unchanged. This is a diagnostic adapter candidate, not an
activated production/held-out interface.

## Original response: not recoverable without a semantic choice

The first response is complete, generation-schema valid, four nodes and three
assertions. Every declaration and incoming reference is rendered in the restricted
`artifacts/restricted/semantic-id-diagnosis-v3/FAILED_GRAPH.md`; `audit.json`
records every reference's candidate targets and immutable input hashes.

| ID | Record kinds | Resolution |
|---|---|---|
| n000 | schema, decision | No incoming references; administrative cross-type reuse |
| n001 | entity, assertion | Typed fields resolve; decision created target is ambiguous |
| n002 | entity, assertion | Typed fields resolve; decision created target is ambiguous |
| n003 | entity, assertion | Typed fields resolve; decision created target is ambiguous |
| n004 | event, type, predicate | Typed fields resolve; event has no incoming node reference |
| n005 | type, predicate | Typed fields resolve |
| n006 | type, predicate | Typed fields resolve |
| n007 | type | Unique |

There are no same-type ID collisions and no exactly duplicated records. Two
predicate records have identical scientific payloads except their different IDs;
they are not merged. The three untyped `supported_description.created_object_ids`
could each refer to an entity OR an assertion. Rationale prose is not permission
for the runtime to select one. The new adapter therefore **rejects the original**,
returning no normalized/canonical graph. No response was relabeled or accepted.

Authority: implementation plan §5 specifies projection-local IDs and typed
references, not a universal namespace across all record kinds. Methodological
plan §2/§4.2 requires sealed IDs to survive C0/C1 projection; implementation plan
§6 prohibits CPU semantic repair. The actual v1 prompt explicitly required
globally distinct IDs; `InstanceGraph.graph_object_ids_are_unique` enforces that
within the graph. Schema type and predicate uniqueness is checked separately.
`validate.py` resolves typed endpoint/type/predicate/description references but
uses an untyped target set for this decision operator. This is not merely a
validator bug: the original decision really is ambiguous.

Further findings, independently diagnosed rather than formally accepted:

- One assertion makes the place carry the artifact; its own explanation names
  the person instead. Its declared domain also excludes the place's type.
- Two description-support assertions do not involve their purported node/event.
- The arrival event is isolated, with no event-role assertion or substantive
  construction decision. The sole operator is supported_description.
- Numeric story point 1 has no declared coordinate mapping; observation at dawn
  does not establish singleton intrinsic validity. Unknown remains available.
- The carrying predicate's located_at upper parent has a questionable meaning.
- Citations/mention references, disclosure order and confidence ceilings are
  individually valid. Null epistemic scope fits this unattitudinal passage.

These findings survive removing the first exception; they prevent any claim that
administrative ID repair would establish scientific acceptance.

## Proposed identifier contract

`semantic_identifiers.py` implements independent typed counters:
schema nS, type nT, predicate nR, entity nE, event nV, proposition nP, assertion nA,
decision nD. Each kind starts at 1; no shared numeric counter is required.
Endpoint and role fields still encode the model's chosen connections. Untyped
decision fields include the target's prefix. Supplied evidence/candidate IDs are
not renamed. Canonical IDs are runtime-generated from the original payload hash,
kind and local identifier; a bijective path-level translation receipt is retained.
No free text, endpoint choice, role, evidence, confidence or temporal meaning is
completed by the runtime. Same-type duplicate IDs and unresolved/ambiguous
references fail closed, including byte-identical duplicate records.

The opt-in `reconstruct_typed` adapter validates the candidate schema, translates
IDs, then invokes the existing canonical adapter. It distinguishes original
generated payload, translated payload, runtime metadata and canonical hashes.
No controller imports/activates it. Production approval would still have to bind
this representation consistently across conditions and preserve supplied sealed
C1 IDs for FixedSelect; this diagnostic translation is not a FixedSelect adapter.

Pinned xgrammar CPU compilation found a pre-existing limitation: pattern plus
minLength/maxLength ignored the length on two **role identifier** fields. The
candidate encodes exactly the same 1..192-character language in the regex.
Descriptions are untouched; no scientific validator is relaxed. The initial
warning and subsequent compile are retained separately in restricted storage.

## Retry evidence and smallest next live test

The second attempt changed **only** a system-message suffix quoting the first
Pydantic uniqueness error. Exact suffix is in the failed-graph rendering. Schema,
evidence, model, seed, sampling and output allowance were identical. It produced
repetitive auxiliary propositions and length/truncated JSON. The schema permits
unbounded proposition contents; the warning plausibly increased bookkeeping
burden, but one retry cannot establish causation or distinguish model behavior
from decoder interaction. The ID-only candidate does not claim to solve that
separate unbounded-generation or task-fulfillment risk.

Ready, **not launched**:
`artifacts/restricted/semantic-id-diagnosis-v3/CANDIDATE_REQUEST.md`,
`candidate-request.json`, `candidate-schema.json`.
Pinned template-inclusive input **5,605 tokens**, allowance **6,144**, total
**11,749 / 12,288**. Request hash
`fc57ab003859a7bfd7c8075fb2f1b1bec6b102f970238f778c15d392dac83693`.
This is a CPU packing check, not generation reliability or production capacity.

The smallest discriminating next test is **one** call on the same small evidence
and semantic task, with this typed-ID contract and unchanged model/sampling/
allowance. Require complete JSON, unambiguous chosen connections, canonical and
cross-field integrity, supported nonempty structure and substantive construction;
report each separately. Do not substitute full C1 or enlarge tokens. It requires
new explicit GPU-session authorization and the diagnostic-only forecast exception;
the prior session's unused seconds do not authorize a restart. Stop afterward.
No full-study budget, scientific validator or held-out protocol is changed here.
