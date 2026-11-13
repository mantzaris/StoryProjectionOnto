# Fifth small streaming diagnostic: executed, not accepted

The authorized unchanged small request ran once at source checkpoint `55637c1`
on pinned Qwen3-8B-AWQ revision
`4da05a8edb55c6046cce958586c33b61da07bb79`. Request hash:
`cde6c6b6eedefab00aa46b7a01833998ca0b291ad8ff1daf55e574ad23681e7a`.
No full C1, C2, FixedSelect, retry or ordinary development inference ran.

The one development passage asks for a meaningful ontology fragment: 2–4
supported nodes, 1–3 qualified assertions, contextual types/predicates and a
supported construction decision. Complete SSE, canonical reconstruction, schema
validity and unchanged scientific validation were separate predefined gates.
Production codec, bounded identifiers, whitespace restriction and xgrammar were
used. Template-inclusive input was 3,453 tokens, allowance 6,144, total reserved
9,597 of the unchanged 12,288-token context.

## Actual observations

| Check | Observation |
|---|---|
| Startup | Ready in 238.839647 allocated seconds |
| Server/request | HTTP 200, SSE received, final usage and DONE received |
| First event / first content | 1.951377 / 2.022473 seconds from client request |
| Completion | 138 tokens, `finish_reason=stop`, complete JSON |
| Effective wire schema | Pass in independent CPU JSON Schema check; not the canonical contract |
| Canonical reconstruction | Rejected an unknown supplied opaque-reference handle |
| Canonical schema / scientific validators | Not reached; no accepted output |
| Predefined minimum structure | Failed: emitted an empty graph |
| Generation event / generation wall | 7.693043 / 7.730289 seconds |
| Generation plus failure handling | 7.807653 seconds |
| Entire allocated service start | 256.521126 seconds, below 560-second cap |

The wire schema permitted empty collections; wire validity is not scientific
validity. The preserved response includes a reification rationale but no actual
entities/events/assertions implementing it. This is not truncation or evidence
of a generation stall. Prompt/representation following and the permissive
structural grammar remain concerns; one small response cannot isolate model
behavior from representation difficulty or establish production reliability.

A post-hoc CPU field check also finds `decided_at` equal to a comma, not a valid
datetime. The effective guided schema admits a string at that position, whereas
canonical Pydantic validation rejects that value. The live path stopped earlier
at reference translation, so this additional finding is not a live canonical-
validation result or a repair. Original response bytes remain unchanged.

Server logs show weight loading completed in 32.20 seconds and engine
initialization in 1.52 seconds, within the longer observed startup. Earlier
initialization delay remains unassigned; no filesystem-causation claim is made.
Logs show the request running and shutdown. Detailed per-request grammar
preparation timing was not observed.

Sampled peaks: GPU 22,793,945,088 bytes; owned process-tree RAM 3,175,784,448
bytes; project storage 16,458,690,048 bytes. These are sampled maxima, not proof
that every instantaneous peak was captured. No recorded resource violation.
Verified shutdown leaves zero open allocation/service journals and GPU idle at
1 MiB. The pod remains active.

## Accounting and durable evidence

Historical actual allocation 4,633.801513 + this start 256.521126 =
**4,890.322639 seconds**. Same block: five starts, four generation attempts,
1,662.496315 / 1,965.975189 seconds; 303.478874 seconds unused. No further start
is authorized. The original nine-hour scheduled target was not met; amended
33,660 scheduled and strict actual stop before 36,000 remain unchanged.

Remaining mandatory proxy is 40,162.013213 seconds, including every generation,
reserve, service-load and pending acceptance/restart/resume envelope. All-in is
45,052.335852 seconds: 11,392.335852 above the scheduled ceiling. This remains an
unmeasured conservative capacity sensitivity, not a successful-output p95; the
138-token invalid output earns no throughput credit. Revised benchmark input
packing must also be rechecked. Ordinary admission and complete acceptance fail.

Restricted backup: `artifacts/restricted/fifth-small-backup.5NiPeo/`.
`fifth-start-receipt.json`, `small-response-analysis.json` and `block-summary.json`
are generated from the immutable response/ledger, not manually authored outcomes.
All 208 transferred diagnostic/CAS/ledger/log file hashes match the remote.
Ledger SHA-256:
`479357ade463d63332f0c27dbb2cc54325ed6776275fcd15fe4b8b384c071658`.
Read-only verification found 19 CAS artifacts, 7 attempt rows, 6 completed
model-call rows, 20 GPU events, 13 service sessions and zero integrity issues.
Older local ledgers, including stale V9, remain untouched.
