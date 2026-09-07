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
