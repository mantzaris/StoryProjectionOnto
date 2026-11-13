# Small representation diagnostic (2026-09-07)

Explicit new user authorization: one service session, at most four small attempts,
1,200 additional allocated seconds. Historical actual allocation is
4,890.322639 s; the previous block and every failure remain intact. Scheduled
33,660 s and strict actual stop before 36,000 s are unchanged. Complete-forecast
admission is waived only for this diagnostic, not ordinary study work.

## Prior response inspection

The fifth small response completed HTTP 200 / SSE / JSON in 138 tokens, stop.
The first canonical-reference error was decision_id `I34`: not supplied (the map
ended at I33), invented in the supplied-handle namespace instead of a new `n` ID.
Its evidence_ids contained valid `ev-01`, plus I23 (event candidate), I27 (its
content hash), I28 (passage), and I14 (text hash). These supplied values were
misused as evidence IDs, not incorrectly decoded. `decided_at` was `","`.
Canonical Pydantic requires a datetime. Backend normalization had removed its
date-time format, leaving `string` in both effective schema and tuple legend.
The prose referred to a pre-query seal time that was not actually supplied.

Every tuple position, optional field, enum and `n` versus `I` namespace was
listed, but all reference kinds shared the generic ID type. The production
generic schema also allows empty arrays for abstention; it does not enforce the
small task's 2–4 nodes / 1–3 assertions. Neither omission is retroactive acceptance.
The response claimed reification but created no graph. It remains failed.

## Frozen comparison

All variants receive the exact same one-passage development evidence: a courier
arrives at a gate carrying a seal. No expected ontology is supplied. They must
construct 2–4 supported entity/event nodes, 1–3 qualified assertions, local
types/predicates and a supported decision. All canonical semantic fields and
unchanged structural/grounding validators remain common. This is not a smaller
semantic task for the named-field variants.

A uses ordinary named-field JSON without constrained decoding. B has exactly A's
messages and sampling plus a matching ordinary object JSON Schema (standard
named records/refs, no tuple transformation). C uses the production tuple/handle
codec and its grammar. Named fields are canonical directly, not parsed as tuples.
The matching object schema retains the full canonical optional semantic fields;
it is not an alternate scientific schema or a flat-triple approximation.

All prompts distinguish evidence, candidate, hash and new local IDs explicitly;
administrative decision time uses the supplied request timestamp in UTC ISO
format. Both constrained schemas enforce that timestamp spelling. No scientific
value is substituted into an output. Empty graphs remain post-validation failures.

Pinned model: Qwen3-8B-AWQ revision
`4da05a8edb55c6046cce958586c33b61da07bb79`. Same non-thinking template, seed and
sampling settings. Output allowance 6,144 each; total context remains 12,288.
Pinned-tokenizer input: A/B 4,506 each; C 3,250. Request hashes are frozen in
`representation_diagnostic.py` from the CPU-only preparation. The ordinary input
values are identical after reversing C's handles. No tuning between A/B/C.

CPU preparation took 64.421844 s outside allocation. Local focused unit tests:
47 passed; HTTP/SSE tests: 13 passed. The same 60 tests pass on the pod. Existing
real stuck-sampler shutdown tests also passed (included in a separate 20-test run).

The existing controller/guardian is reused. Startup cap 360 s; live checks 15 s;
generation stages 180 s each with 5 s exception drain inside that cap; validation
15 s each; shutdown 45 s protected plus a 5 s guard. The whole-start deadline
dominates all stages. The default frozen sequence is A/B/C only. A fourth call is
allowed only for an explicit outcome-supported repair, never an identical retry.
The service stops after the comparison or on an unhealthy/uncancelled request.

Report HTTP/completion, JSON syntax, reference integrity, nonempty structure,
canonical schema, scientific grounding and timing separately. A/B/C success would
not be full C1 acceptance, a p95 estimate, or full-study feasibility evidence.

## Actual outcome

Source checkpoint `5b298e9`; one start and three calls. All received HTTP 200,
complete SSE/usage/DONE, complete JSON and finish_reason=stop. No fourth call or
ordinary acceptance/development inference ran. The service is stopped.

| Variant | Output tokens | JSON syntax | Matching JSON schema | Canonical Pydantic | Required graph | Accepted |
|---|---:|---|---|---|---|---|
| A: named, unconstrained | 3,130 | pass | fail | fail | fail | no |
| B: named, constrained | 2,241 | pass | pass | fail | fail | no |
| C: production tuples | 62 | pass | pass | pass | empty | no |

Reference and semantic findings are separate from JSON syntax:

- **A:** generated Lio, North Gate, Seal and Arrival, but duplicated Arrival across
  entity/event arrays (five records, four distinct IDs). Its three assertions
  had predicates and descriptions, but **no subject/object or role bindings**.
  Evidence IDs were `ev-01`; required parents/schema ID/output-token sentinel were
  missing. Point times omitted numeric points, and copied provenance hashes did
  not match copied field values. The model wrote “The copper seal carried by Lio
  is a key object, providing context for the event.” That partially reflects the
  passage, but descriptive prose is not a valid linked qualified assertion.
- **B:** produced four distinct nodes and one assertion, again **without endpoints
  or roles**. Its JSON matched the grammar. Every point-time record also contained
  a partial-order self-equality, forbidden by canonical cross-field rules. It used
  point=0 for dawn despite the supplied day-1 clue, and gave intrinsic validity
  that same point without evidence. `carries`/`arrived_at` were invented as upper
  parents rather than local predicates under supplied upper terms. The node
  “Arrival at North Gate” is understandable, but this is not scientific success.
- **C:** lossless reconstruction passed with schema_id `nil`, no entities, events,
  assertions, types, predicates or decisions. Reference resolution was vacuous,
  not positive grounding evidence. The predefined meaningful-structure check
  rejected it. Its complete text was only 103 characters.

The full scientific-grounding audit was not reached: A/B failed canonical
validation; C failed required nonempty structure. The evidence-grounded semantic
task therefore passed for **zero** variants. The statements above are direct
field-level inspection of unchanged failed outputs, not retroactive repairs.

| Variant | First SSE event (s) | First content (s) | Allocated request event (s) | Client wall (s) |
|---|---:|---:|---:|---:|
| A | 1.302831 | 1.385437 | 74.960579 | 75.195217 |
| B | 1.534872 | 1.620860 | 63.803129 | 63.899934 |
| C | 0.452871 | 0.568566 | 3.224095 | 3.385467 |

No token ceiling was reached. A contained 3,341 whitespace characters of 10,345;
B 601/6,463; C 19/103. Authored capacity checks were 1,639 named / 1,152 tuple
tokens; these were never model answers or reliability evidence.

Allocation: startup **316.847788 s**, three request events **141.987803 s**,
remaining service allocation including checks/shutdown **14.344400 s**, total
**473.179991 s**. Global actual **5,363.502630 s**. New allowance unconsumed
**726.820009 s**, but its one permitted start is exhausted. Previous block usage
is unchanged. Sampled peaks: VRAM **22,793,945,088 bytes**, process RAM
**7,146,729,472 bytes**, project storage **16,654,564,864 bytes**; no sampled
violations. Stopped full storage census: **16,653,322,752 bytes**. No open GPU
allocation or service journal remains; GPU idle at 1 MiB; pod remains active.

Full remaining inventory proxy: **39,828.344344 s**, plus the already-declared
pending acceptance/restart/resume envelope **333.668869 s** = **40,162.013213 s**.
All-in **45,525.515843 s**, above scheduled 33,660 by **11,865.515843 s**. These
are conservative incomplete-study proxies, not successful-output p95. The initial
terminal's inventory subtotal and first derived summary omitted the extra envelope;
both are preserved, and `comparison-summary-complete-inventory-v2.json` explicitly
corrects that reporting omission. The controller calculation now includes it,
with a focused regression test. No admission threshold changed, and this omission
did not authorize ordinary work or expand the diagnostic's actual allocation cap.

### Conclusion and next decision

Constrained named fields improved syntactic compliance relative to unconstrained
named fields; the evidence does **not** establish a decoder integration outage.
Tuples reconstructed but yielded an empty answer. Named fields produced meaningful
prose and objects but still failed linked-assertion and temporal requirements.
Neither “named fields work” nor “the model cannot reason” is justified by one
example. This is evidence of representation/task-contract difficulty with unresolved
model behavior. A/B also show that the full canonical schema is still a demanding
model-facing contract, despite the small evidence task.

The next focused candidate should explain and encode assertion alternatives
(binary endpoints or n-ary roles), mutually exclusive temporal shapes and supplied
upper/reference vocabularies in a readable named-field diagnostic. Test CPU
conversion without inventing semantics before any new live authorization. Do not
redesign the full experiment or increase output allowances on this evidence. No
fourth call was spent on an identical request or an arbitrary token increase.

Restricted backup: `artifacts/restricted/representation-backup.maHyBy/`.
All **933** manifest-listed remote files match local hashes, including fragments,
requests, logs and ledger. Previous local ledgers remain preserved. Ledger SHA-256:
`be2e8207018ca6a9719f48716bc198c1b612dcba331768a9d1148046f8643918`.
No weights, novel, credentials or public model-output release was transferred.
