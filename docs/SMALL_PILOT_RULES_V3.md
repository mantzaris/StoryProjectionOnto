# Authorized small-pilot rule reconciliation — frozen before execution

User-authorized scope: two preselected development evidence records only.
Revision: `small-source-bound-semantics-v3`. This does not change primary metrics,
synthetic gold, C0 scoring, held-out acceptance, or the production protocol.
Predecessor: `SMALL_SEMANTIC_RULE_RECONCILIATION.md`; old response/ledger records
and legacy scorer implementation remain immutable.

## Old and corrected rules

| Issue | Old pilot rule | Corrected diagnostic rule |
|---|---|---|
| Time | One hidden numeric story point also required as intrinsic validity. | Source-bound story and intrinsic validity are assessed independently. Neither passage supplies a numeric clock/duration. Applicable unknown with reason is supported; invented numeric bounds are rejected. NA is not missing information. Unassessed relative forms/labels remain unresolved. Direct narration reveals the proposition at its supplied passage position, never at a derived story time. |
| Representation | Direct witnesses omitted event-role equivalents; unmatched was often treated as false downstream. | Explicit binary arguments or named event roles must preserve the same source-bound participants and meaning. Inverse binary direction is interpreted explicitly. Wrong bindings, contradictory local definitions and incompatible upper types are rejected. Unrecognized definitions remain unresolved. |
| Descriptions | Unmatched assertion automatically contaminated description truth. | Textual evidence support and assertion mapping are separate. Whole source phrases and frozen ordered paraphrases can receive positive support. Linked supported assertions must involve the node and cover described referents. Unrecognized clauses/definitions are unresolved, not automatically false or true. |
| Construction | Description records could obscure whether a substantive operation occurred. | Require a substantive operator with actual matching created targets and supported content. Type/schema construction can qualify without reification. Reification must target an actual event. Description/selection alone cannot qualify. Unimplemented operator prerequisite assessments remain unresolved. |

Positive support always needs a witness. The rules never turn lack of a known
contradiction into grounding. The automated text/local-definition recognizer is
deliberately conservative and has **limited paraphrase coverage**: unrecognized
language is reported as unresolved, not a demonstrated model mistake. This
limitation must remain visible in any model-outcome report. The checker is not a
general entailment engine and cannot authorize production adoption on its own.

The heavily developed first passage is not a generalization test. The second
was selected before the prior live response and is also development material.
Both requests retain the exact frozen semantic-instruction-v2 system message,
typed-ID schema, complete evidence, pinned tokenizer/template/settings and
6,144-token output allowance. Their input counts remain 5,740 and 5,992.
No known answers, reference alternatives or checker feedback enter either base
request. No checker/rule changes are allowed after seeing new outputs.

## Controls and original-response replay

Authored controls live only in the test/scorer boundary:
`tests/unit/test_small_semantic_reconciliation.py`.

- Direct arrival, two binary participant/event edges, and an explicit n-ary
  arrival record expand to the same supported agent/location facts.
- A second-source direct repair control is positively supported.
- Reversed arguments, place-as-carrier, event-as-place and wrong named roles fail.
- Unsupported intrinsic point 1 and substituting NA for unknown fail.
- Unknown intrinsic duration passes without requiring post-query invention.
- Description-only and substantive operations targeting the wrong object kind fail.
- Unrecognized prose/predicate definitions remain unresolved, not positive.
- The real controller adapter uses these rules only when explicitly opted in;
  the old route retains its historical oracle behavior.
- The original canonical response is replayed byte-for-byte and remains rejected
  for endpoint meaning, intrinsic validity and construction reporting. This is
  separately versioned CPU re-evaluation, not a new model result.

Prelaunch focused run: **88 passed in 14.17 s** across the six changed/affected
interface, identifier and controller modules. Existing transport/monitor/guardian
tests are reused; neither transport nor resource-sampling code changed.
Rule, source, request and control identities are recorded in
`artifacts/restricted/small-rule-freeze-v3/CPU_FREEZE.json` before allocation.

## One newly authorized session, no historical reset

Block: `small-reconciled-semantic-validation-20260908`.
Historical actual **6,025.436171 s**; at most **1,100 s** new, hence maximum
**7,125.436171 s** actual under this authorization. One start, at most three small
generations, diagnostic-only complete-forecast exception. Scheduled 33,660 s and
strict actual stop before 36,000 s are unchanged. No full C1 or study inference.
Previous block directories/reservations and all ledger rows are retained.

Stage envelope: startup 360 + live checks 15 + three (generation 180 + validation
30) + brief repair preparation 30 + protected shutdown 60 + guard 5 = 1,100 s.
The whole deadline wins. Request timeouts leave a failure-drain margin; admission
also reserves shutdown globally. Both baselines run on the same healthy service
even after an ordinary semantic failure. Unsafe cancellation/transport stops.

Only after both baselines may one third call clarify one observed contract
violation. The fixed whitelist contains missing substantive operation report,
event type incompatibility, predicate/role mismatch and unsupported numeric time.
Feedback contains a code/path plus a general rule, never expected endpoints,
facts, replacement times, or an authored graph. Actual repair packing must pass;
unresolved semantic coverage alone does not justify answer-bearing feedback.
The same existing guardian/controller owns shutdown and accounting.

## Preserved boundaries

C0 full extraction competence remains passed: P 104/122, R 104/114, all five
fixture families, 100% valid evidence references. Contextual P 54/275, R 54/91,
F1 .295082 remain separate. No new C0 run, threshold, denominator or gold change.
The prior all-in complete-run proxy remains 46,187.449384 s before this session;
small diagnostic timings cannot establish a production p95 or feasibility.
Stop the service after the authorized diagnostic work. Human review still gates
held-out execution. Any later production adoption/acceptance needs its own gate.
