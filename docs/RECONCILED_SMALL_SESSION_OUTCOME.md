# Two-example small validation outcome — 2026-09-08

Frozen rules: `cbfb77a`, `docs/SMALL_PILOT_RULES_V3.md`.
Live controller checkpoint: **`4b36a4f`**. One service start, **two base small
calls**, no repair call, then verified shutdown. No ordinary study inference.
Both exact frozen request hashes and all previous ledger rows are preserved.

| Measure | First development example | Second development example |
|---|---:|---:|
| Template-inclusive input tokens | 5,740 | 5,992 |
| Output tokens / allowance | 2,324 / 6,144 | 2,539 / 6,144 |
| Request wall seconds | 54.612932 | 63.674449 |
| Metered inference seconds | 54.577201 | 63.631374 |
| First SSE event / content seconds | 2.029135 / 2.145275 | .873783 / .964057 |
| Complete JSON / generation schema | pass / pass | pass / pass |
| Finish reason | stop | stop |
| Unique declared records / collisions | 16 / 0 | 16 / 0 |
| Undeclared proposition IDs | nP1, nP2, nP3 | nP1, nP2 |
| Canonical reconstruction | fail | fail |
| Scientific checker reached | no | no |
| Raw nodes / assertions | 4 / 3 | 6 / 3 (node-budget excess) |
| Overall small-task acceptance | fail | fail |

**Two complete/schema-valid outputs, zero canonical/scientifically accepted
outputs.** These were not output-capacity or HTTP failures. Both referenced
proposition records that were never emitted; no missing semantics were filled in.
Manual review additionally finds unsupported numeric time and holder attribution,
inconsistent predicate/event bindings, and mixed construction reports. The second
output mis-types events and exceeds the total small-node budget. Descriptive prose
can be supported by the passage while its required assertion mapping is absent.
Manual findings are not presented as completed formal validation stages.

Readable actual evidence/graphs and exact semantic diagnoses:
`artifacts/restricted/reconciled-session-backup.gZFFyU/`:

- `artifacts/restricted/small-reconciled-semantic-validation-20260908/EVIDENCE_AND_GENERATED_GRAPHS.md`
- `MANUAL_DIAGNOSTIC_REVIEW.md`
- `TERMINAL_VERIFICATION.json`

## Why there was no third call

Canonical errors contained entire ID-audit dumps (13,530 / 13,915 characters).
The existing repair formatter refuses more than 4,000 characters, so it prepared
no targeted retry. The second baseline still ran, as required. The resulting
stop was a **formatter limitation**, not lack of useful possible repair or
exhausted authority. A compact missing-reference/conditional-scope diagnostic is
a justified next CPU task; no further startup or blind retry is proposed here.

The frozen v3 scientific checker was not changed during/after this session.
An additional implementation limitation must be corrected before future use:
its generic entity expectations for seal/pump omit compatible artifact types.
Its language recognizer also has limited coverage. Neither limitation caused
these two earlier canonical failures. Do not retroactively accept either output.

## Allocation and integrity

| Allocation class | Seconds |
|---|---:|
| Preserved historical allocation | 6,025.436171 |
| Service startup/load | 171.205433 |
| First inference | 54.577201 |
| Second inference | 63.631374 |
| Other allocated service/check/drain/shutdown time | 6.538464 |
| **New session** | **295.952472** |
| **Actual global total** | **6,321.388643** |
| Unused portion of 1,100-second allowance | 804.047528 |

The new session remained within 7,125.436171 global authorized seconds and all
stage/whole deadlines. Startup is measured, not an assumed 180-second event.
No historical time or failed attempt was reset. The stopped ledger passes
integrity checking; **4,504 historical rows retain their original values**, with
579 new rows. All **721 transferred files** match the closed remote hash manifest.
The previous local ledger remains untouched. Initial rsync ownership preservation
was unsupported by the remote filesystem; content hashes verified, and subsequent
copies omitted ownership changes. No transfer-control restriction was bypassed.

Sampled peaks: VRAM **22,793,945,088 bytes**, project process RAM
**7,172,423,680 bytes**, project storage **16,795,292,672 bytes**. No model offload,
new model, larger context, concurrent inference or resource-envelope expansion.
vLLM stopped; no GPU compute processes, no open allocation/service journals;
the pod remains active. Exact model revision remains
`Qwen/Qwen3-8B-AWQ@4da05a8edb55c6046cce958586c33b61da07bb79`.

Remaining mandatory-work proxy **40,162.013213 s**; updated all-in proxy
**46,483.401856 s**, above scheduled **33,660 s**. Strict actual stop before
36,000 s remains. No invalid-output timing was credited as successful production
throughput. These two timings do not establish a p95, reliability or feasibility.

## Preserved work and next boundary

The CPU rule/control suite passed **88 tests locally**; remote **85 passed / 3
retained-local-artifact skips**. The additional no-blind-retry boundary fix passed
41 local tests and 40 remote tests / 1 retained-local-artifact skip. These overlap;
do not add their counts. Rules/requests were frozen before either response.
Terminal checks reconfirm all 721 backup hashes, both frozen request identities,
153 unchanged deployed source hashes, ledger integrity and report numerical
consistency against the immutable summary. No unaffected suite was rerun merely
to produce this handoff.

C0's complete extraction gate remains passed: P **104/122**, R **104/114**,
**5/5** families, **100%** evidence-reference validity. Historical and matcher-only
failures remain separate. Contextual P **54/275**, R **54/91**, F1 **.295082** are
unchanged. No threshold, denominator, benchmark gold or primary metric changed.

Neither small example passed, so the conditional full-size packing/adoption
milestone was not entered. Full fallback acceptance, complete-run admission,
production-interface adoption and independent human review remain gates. No
held-out run, new results PDF or generalized recovery system was produced.
