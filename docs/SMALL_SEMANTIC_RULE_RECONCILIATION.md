# Small semantic diagnosis and CPU repair — 2026-09-08

Historical CPU-only decision record. The user subsequently approved the narrow
diagnostic reconciliation and one bounded session. Current frozen rules and
authorization: `SMALL_PILOT_RULES_V3.md`. The original proposal below is preserved;
its statement that authorization is pending no longer describes the new session.

No GPU start, inference, model/codec/ID redesign, gold edit or new acceptance.
Historical allocation remains **6,025.436171 seconds**. Original typed response,
canonical graph, ledger, validator records and prior manual review are immutable.

Readable per-assertion evidence/fields/checks:
`artifacts/restricted/semantic-diagnosis-v2/ASSERTION_RECORDS.md`.
Interpretation and error classification:
`artifacts/restricted/semantic-diagnosis-v2/SEMANTIC_DIAGNOSIS.md`.
Versioned CPU re-observation: `offline-diagnosis.json` in the same directory.
It is not a new model result or retrospective acceptance.

## What failed, and what did not

All three assertions share a local predicate defining a person's arrival **at a
place**, but bind an event as object. Two also use a place/object as subject. The
upper parent `participates_in` suggests a different relation; the runtime cannot
choose that interpretation and overwrite the model's local definition. This is
a genuine model-authored inconsistency, compounded by missing explicit binary
direction/role-order instructions in the old prompt. The old schema already
requires endpoints; adding or renaming IDs cannot fix their meaning.

The literal node descriptions and repeated `why_matters` sentence are supported
by the passage as text. Their assertion-backed support is not established by the
inconsistent edges. The old oracle's description rejection is also a cascade
from unmatched assertion witnesses, not a separate language-entailment judgment.

The response explicitly attributes point 1 to `time-clue-dawn`. Supplied input
contains normalized expression `day-1-dawn` and discourse/revelation position 1.
These are plausible cues, **not proof of the model's internal cause**. The old
prompt expressly forbids using a label alone as a numeric coordinate, distinguishes
occurrence from validity, and permits unknown time. The clue is defeasible and
targets the event, not every entity state or assertion's intrinsic duration.
There is no supplied unit/clock mapping or intrinsic-duration witness. Ambiguous
normalization may explain occurrence-time behavior; it cannot justify copying
point 1 into all three intrinsic-validity records. Unsupported precision remains
rejected. The revised general instruction clarifies digits-in-identifiers/clues
versus an explicit clock without giving this passage a replacement time.

The model did more than supply descriptions: it authored a local type, predicate,
three entities, an event and three assertions from evidence, not a supplied graph.
It reported those objects under `supported_description`, which does not certify
the substantive operations performed. The distinction is **construction objects
present, appropriate operation report absent**, not proof that no construction
occurred. The scientific quality of the created objects is a separate failure.

## Required operation: no compulsory reification

Methodological §§2–4 distinguish construction of local schema, identity, event and
qualification from selection/presentation; §7 permits binary assertions and gives
reasons for optional event reification. Implementation §§5–6 require local
schema/qualified assertions and construction-decision material. Neither plan
requires event reification, merge/split, or every operator in every tiny passage.
The explicit small diagnostic instruction adds **at least one supported
nonselection construction decision** as a plumbing/capability check, not a new
conference endpoint. The full development gate's operator coverage must not be
applied to each small example.

The existing `SUBSTANTIVE_CONSTRUCTION_OPERATORS` defines the permitted choices:
merge, split, contextual_type, schema_relation, event_reification, abstraction,
temporal_qualification, epistemic_qualification. Creating a local type or relation
can satisfy this requirement without an event. If a model chooses a justified
event, its decision should report that operation and appropriate targets. CPU
code cannot relabel the retained decision or assert that its reification was
scientifically correct. Selection, description, inclusion and rarity checks
alone are insufficient. No new scientific requirement has been added.

## Completed CPU changes

1. **General instruction v2**, opt-in only, in
   `prompts/diagnostics/semantic_instruction_v2.md`: explicit predicate argument
   order/direction, consistent local definition/upper-parent/role meanings,
   event-versus-participant distinctions, optional justified reification,
   source-grounded temporal uncertainty and accurate operation reporting.
   It contains no courier/pump names, expected endpoints, graph or replacement
   timestamp. Existing schemas, field guide, typed namespaces, adapter, evidence,
   template, sampling and allowances are unchanged. The original builder still
   reproduces the frozen live request.
2. **Observation-report errors fixed**, version `small-component-observations-v2`:
   unmatched oracle endpoints are `unknown`, not automatically false; structural
   validity/nonempty size are separate from substantive decision reporting; the
   canonical substantive-operator set excludes include/exclude and rarity-only
   records. All original scientific validators remain unchanged; no old output
   changes acceptance status.
3. **Existing controller repair branch corrected**: no unconditional scientific-
   failure shutdown. Only whitelisted fact-free contract violations (missing
   substantive report or incompatible event type) can support one specific repair;
   expected oracle facts, times and arbitrary feedback are excluded. Both frozen
   examples precede at most one repair of the earliest eligible failure. Transport
   uncertainty/unsafe cancellation still stops, and all existing time, call,
   ownership and shutdown controls remain. No new controller or approval schema.

The previous session's source `2584064` explicitly set `semantic_next=None` on
`failure_stage == scientific_validation`. Therefore stopping after one call was
an **automatic implementation restriction**, not a demonstrated absence of a
justified repair and not a user-required stop. The stopped service cannot be
resumed by spending unused seconds; the current exhausted session's authorization
and reservations have not been changed.

## Rule reconciliation requiring a decision before another live test

The general scientific standard stays fixed: evidence support, consistent roles,
essential temporal correctness, and explicit uncertainty. Changing the following
sealed pilot reference/gate assumptions needs a versioned fixture amendment; it
has **not** been implemented as an acceptance change in this repair.

| Current implementation | Proposed rule | Consequence |
|---|---|---|
| `_scope_supported` applies one `_KnownFact.story_point` to BOTH story and intrinsic validity; the arrival witness requires 1 for both. Unknown is rejected. | Separate source-bound story and validity expectations. With no explicit coordinate/duration, accept applicable unknown with reason/supported label; do not accept numeric bounds merely matching a hidden constant. Exact essential times still require exact evidence-supported values. Not-applicable is not interchangeable with unknown. | Removes unsupported mandatory precision and rejects invented precision that the legacy pilot sometimes accepts. The retained output still fails. This changes the sealed pilot reference/gate, not the scientific fidelity standard. |
| Arrival support has a direct actor→place witness but lacks role-preserving event-centric alternatives; unmatched is unknown and complete grounding fails. | Freeze condition-blind direct and event-role alternatives from evidence independently of model predictions; require consistent local definition, upper semantics, named participant roles, citations and qualifications. Never accept all `participates_in` edges merely because they point to an event. | A legitimate reification can be evaluated without forcing a direct triple. Inconsistent generated definitions/bindings remain rejected. Alternative coverage must be reviewed before use; no gold is tailored to this output. |

The implementation error in requiring hidden validity is established by
methodological §7 and implementation §§5/7.2, reinforced by the approved intrinsic-
validity amendment. The additional **acceptance-fixture revision and alternative
coverage** are explicitly proposed here rather than silently activated. As a
result, an otherwise reasonable unknown-time output can still fail the unchanged
legacy pilot audit; it must not be called a new model failure without inspection.

## Prepared validation session (proposal only)

First: the unchanged evidence task with the general instruction revision. Second:
previously frozen `ev-03` development passage, selected before the prior response,
not a newly cherry-picked passage and never drawn from held-out data. Complete
readable requests: `artifacts/restricted/semantic-diagnosis-v2/REQUESTS.md`.

Propose **one session, at most three calls, at most 1,100 allocated seconds**:
360 startup + 15 live checks + three (180 generation + 30 validation) + 30 brief
repair preparation + 60 protected shutdown + 5 guard = 1,100 seconds. Whole/stage
limits and per-call admission both apply. Run both preselected examples on the
same healthy service, then at most one specifically justified contract repair.
No blind retry or oracle-answer feedback. Brief repairs inside an approved session
need no separate approval; exhaustion, unhealthy cancellation, or a scientific
rule change ends permitted work. Baseline would remain 6,025.436171 s and the
session maximum would be 7,125.436171 s—not a reset of prior blocks. Retain the
diagnostic-only forecast exception and 33,660 / strictly-before-36,000 ceilings.

This proposal first needs the scoped pilot-rule/alternative decision above and
new session authority. It does not authorize ordinary execution, full C1 or
held-out calls. The CPU-prepared controller binds both exact v2 request hashes,
but the old exhausted allocation block is unchanged and rejects another start.
No new block or executable authority was created. Small success cannot prove production packing,
restart/resume acceptance, reliable throughput or full-study feasibility.

## CPU verification

The five focused interface/controller/identifier test modules passed **69 tests
in 10.86 seconds**; lint and diff checks pass. Nine original artifact hashes and
the preserved ledger hash verify unchanged. A separately versioned offline
observation still rejects the original output. No decoder recompilation or
remote execution was performed: both effective generation schemas are unchanged
from the previously compiled typed schemas.

Pinned-tokenizer template-inclusive input counts are **5,740** (first example)
and **5,992** (second), each with the unchanged **6,144** output allowance and
**12,288** context limit. Fact-free repair capacity checks with a neutral authored
record ID measured 5,789–5,797 and 6,041–6,049 input tokens, respectively. Actual
canonical paths can be longer and must pass the existing packing gate at repair
time. These are CPU capacity checks, not generation-reliability measurements or
production throughput. Exact identities, source hashes and measured counts are
in `artifacts/restricted/semantic-diagnosis-v2/CPU_VERIFICATION.json`.

## Preserved C0 results

The complete query-blind extraction gate remains **passed**: precision 104/122
(.852459), recall 104/114 (.912281), all five fixture families, and evidence-
reference validity 1.00. Matcher-only 59/154 and 59/114, four families, remains a
separate failure. Historical 29/269 and 29/84 records are retained.
Contextual precision **54/275**, recall **54/91**, F1 **.295082** are unchanged;
no extraction thresholds are applied to contextual projection. No C0 rerun,
denominator adjustment, gold edit or new calibration claim was made.
