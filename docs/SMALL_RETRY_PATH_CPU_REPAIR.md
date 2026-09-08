# Actual failed-response-to-retry repair — CPU only

Scope: diagnostic interface/checker only. No GPU start, inference, primary metric,
benchmark gold, C0 evaluation, allocation history or human-review gate changed.
The two historical outputs remain failed. CPU request candidate v3 is preserved;
**v4** adds the pinned decoder's verified single-space formatting requirement.

## Missing declarations: evidence and cause

For each row below, **both** fields under the indicated assertion reference the
same missing ID: `/proposition_content_id` and
`/epistemic_scope/proposition_content_id`. Array indices are zero-based.

| Retained call | Assertion index / ID | Missing content ID | Assertion's generated binding |
|---|---|---|---|
| First | 0 / nA1 | nP1 | Lio → nR1 → arrival event |
| First | 1 / nA2 | nP2 | arrival event → nR1 → North Gate |
| First | 2 / nA3 | nP3 | arrival event → nR1 → seal |
| Second | 0 / nA1 | nP1 | Ash → repaired → repair event |
| Second | 1 / nA2 | nP2 | Lio → remained at → stay event |
| Second | 2 / nA3 | nP1 | same binding as nA1 |

Thus six references per response are affected. Both generated
`instance_graph.proposition_contents=[]`. There is no unknown-ID normalization
or namespace collision: the typed references correctly identify a record kind,
but no record of that kind was declared.

The exact retained v2 prompt says belief/report/denial requires nonasserted
proposition content, requires assertion/scope IDs to agree, and uses null scope
and content ID without attribution. The complete guide describes the declaration
array and each content record's predicate, binary endpoints or roles,
`temporal_content`, and evidence IDs. The typed-ID instructions require exact
references to selected records. It did not explicitly name the declaration-array
path in the prose and singled out three attitudes rather than all five. That is
an instruction-emphasis gap, not an instruction to omit declarations.

Declaration is **conditional**, not mandatory for every assertion: any nonnull
content reference must resolve; any epistemic scope requires the matching content
reference and a declaration. No scope requires a null assertion content reference.
The array may be empty when no content is referenced. This includes known and
uncertain attitudes, not only belief/report/denial.

The assertions themselves contain predicates, endpoints, time and citations, and
their descriptions paraphrase parts of the source. But no nonasserted content
record or explicit `temporal_content` was supplied for any nP ID. These fields are
not permission to synthesize a missing record or choose its intended semantics.
The adapter correctly rejects them without filling, dropping or reinterpreting.

Two implementation gaps contributed to the failed workflow:

- The generation grammar allowed an independent nullable scope, nullable content
  ID, arbitrary commitment, and empty content array. It guaranteed ID syntax, not
  existence, equality or conditional commitment. The canonical validator was stricter.
- The controller passed a complete ID-audit dump to a formatter with a 4,000-character
  refusal limit. Actual errors were 13,530 and 13,915 characters. It also did not
  provide the previous response in the proposed repair conversation.

The model omitted the declarations and authored unsupported scopes (known by Lio
in the first, reported by Ash/Lio in the second). Both combined those scopes with
world commitment against explicit instructions. There is no evidence establishing
the model's internal reason for those choices. They are not codec decoding errors.

Authority: methodological plan §7 and implementation plan §§5, 6 and 7.2;
`QualifiedAssertion.validate_assertion_shape_and_commitment`, `EpistemicScope`,
and `PropositionContent` in `contracts.py`. These canonical meanings are unchanged.

## Completed implementation

- `IdentifierResolutionError` retains the full audit and exposes typed, deduplicated
  defects. Every offending JSON-pointer path survives compaction. Raw exception
  strings/scorer descriptions are never sufficient model-facing feedback.
- Repair feedback names category, referenced ID, paths and a general constraint.
  Repeated paths group under the same ID. A separate category identifies the
  observed nonnull-scope/world-commitment inconsistency.
- The previous response is included losslessly as untrusted assistant JSON; the
  complete evidence, upper vocabulary, original instructions and field guide remain.
  Runtime neither fills content nor recommends passage-specific bindings.
- Diagnostic schema v4 has explicit binary/n-ary × scope-present/scope-absent
  alternatives. No scope permits null scope/content and world/contested/unknown
  commitment. A scope requires nonnull scope/content and holder/contested/unknown
  commitment. Dynamic ID existence/equality and positive evidence support still
  require post-generation checks; grammar compliance is not scientific acceptance.
- The repair instruction explicitly covers all five attitudes and the declaration
  location. It preserves uncertainty and forbids inventing content or erasing
  supported attribution merely to pass.
- Shared controller response/reconstruction and next-step functions now retain
  full restricted evidence, prepare the actual retry, run both baselines despite
  ordinary validation failure, and record explicit stop reasons. Call/time limits,
  unsafe transport handling, the guardian and protected shutdown remain in place.

The pinned decoder's `any_whitespace=False` default requires **single-space colon
and comma separators**, not zero-space JSON. CPU controls initially failed on the
first value at character 29 because a required space was missing; the same strings
with standard separators succeeded. V4 clarifies output formatting without
relaxing that restriction. This was a CPU fixture/format-instruction mismatch,
not a newly observed model or GPU failure. Earlier CPU candidates/controls remain
preserved in `artifacts/restricted/small-retry-path-cpu-v3/`.

## Diagnostic checker v4, not primary scoring

The supplied `UpperOntology` is a primitive vocabulary, not a transitive subtype
DAG; local types name one direct upper parent. No undocumented universal hierarchy
was added. The source-bound manufactured seal and pump admit `artifact` as well as
their previously allowed generic `entity`. This does not allow persons/places/events
to become artifacts or ignore predicate direction, participant identity or roles.
The definition/paraphrase recognizer was not expanded to accept retained outputs.

Controls cover supported artifact patients/themes, incompatible types, a reversed
carrier/object relation, incorrect direction, unsupported intrinsic precision,
explicit unknown duration, and invented holder attribution even with a complete
content record. Assessments separately label unsupported precision/attribution,
binding/type contradictions and unresolved matching. Unresolved is never positive
grounding. A structurally complete attributed capacity fixture remains scientifically
rejected on the direct-narration evidence.

The two original responses still fail canonical reconstruction on missing IDs.
Separately replaying them against the stricter retry grammar also rejects their
scope/commitment combinations. Neither is a new generation or retrospective success.

## Exact reviewable requests and token accounting

Restricted current package: `artifacts/restricted/small-retry-path-cpu-v4/`.
Each `semantic-first/` and `semantic-second/` directory contains:

- `REQUEST_AND_FEEDBACK.md`: readable exact conversation and effective schema;
- `feedback.json`, `request.json`, `generation.schema.json`;
- `rendered-chat.json`, `packing.json`, `decoding.json`;
- full ID audit, separately versioned offline failure and preparation receipt.

`CPU_REPLAY.json` verifies unchanged original hashes and records the complete
template-inclusive counts; `CAPACITY.json` contains authored capacity checks.

| Tokens | First repair | Second repair |
|---|---:|---:|
| System instructions + complete field guide | 3,838 | 3,856 |
| Complete original evidence/upper/budget message | 2,096 | 2,330 |
| Complete previous response, lossless JSON serialization | 1,811 | 1,992 |
| Compact feedback | 277 | 256 |
| Chat-template/special-token delta | 27 | 27 |
| **Total input** | **8,049** | **8,461** |
| **Reserved completion** | **3,584** | **3,584** |
| **Input + reserved completion** | **11,633** | **12,045** |
| Context ceiling | 12,288 | 12,288 |
| Remaining context headroom | 655 | 243 |

The explicit **8,704-input / 3,584-output** diagnostic retry candidate replaces
the base small request's 6,144/6,144 allocation only for this named retry route.
The old base requests still hash-identically. No model/context expansion, evidence
truncation, hidden output removal or ordinary-protocol activation occurred. The
runtime rejects this new pair for an ordinary request ID. Full production adoption
would require its own packing, acceptance, fairness and timing gates.

Authored four-node/three-assertion outputs use 1,967 tokens with required separators;
a three-proposition attribution stress variant uses 2,596. Zero-space counts
(1,545 / 2,024) are recorded but **not** used as the decoder-compatible bound.
These are capacity fixtures, not model results or reliability/p95 evidence. Future
requests must be measured again; 243-token input headroom is not robust slack.

## Next decision

Verification: **111 focused tests passed** (19.99 s), plus **23 disjoint
transport/decoding tests** (0.65 s), **134 total**. The shared controller response
and scheduling functions replay both retained failures through simulated transport
and prepare a third parent-linked request; the simulated repair remains failed,
not an invented success. Time/call/unsafe-state/unavailable-repair stops and the
existing guardian cleanup check pass. This is CPU controller-path coverage, not a
new real service lifecycle or measured GPU shutdown. Both pinned schemas compile;
standard-separator positive controls pass, while invalid commitment and repeated
whitespace fail. Full CPU evidence and earlier failed controls are retained.

Recommend one new, separately authorized diagnostic-only service session containing
**the two prepared parent-linked repair calls**, with no unchanged baseline retry
and no full C1/study inference. Proposed envelope: startup 360 s, live checks 15 s,
two calls each 180 s + 30 s validation, protected shutdown 60 s, guard 5 s: **860 s**.
All preparation precedes allocation. This would cap cumulative actual consumption
at **7,181.388643 s** from the preserved **6,321.388643 s** baseline. It requires
explicit authorization and the diagnostic-only forecast exception; none is assumed.

Run both if safe and time remains. Stop on transport/cancellation uncertainty,
resource/deadline exhaustion or completion; no identical blind retries or semantic
CPU repair. Assess ID declaration, epistemic support, role meaning, time, descriptions
and actual construction separately. Success would establish these two small repairs,
not model reliability, full C1 capacity, FixedSelect packing or study feasibility.
If the model still fails this explicit contract, report that result without another
automatic representation change/startup.

C0 remains unchanged: extraction P 104/122, R 104/114, 5/5 families and 100% evidence
validity passed; contextual P 54/275, R 54/91, F1 .295082 stays separate. Historical
and matcher-only failures remain. Scheduled/hard global ceilings remain
33,660 / strictly before 36,000 seconds; human review still gates held-out execution.
