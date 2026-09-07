# Bounded capacity diagnostics: measured outcome

Both authorized service starts were executed in the existing block. Two C1
attempts failed; no C2 or FixedSelect diagnostic was eligible. No output reached
schema/scientific validation. This is not fallback acceptance or development
execution, and invalid generations are not successful-throughput samples.

The model remains Qwen/Qwen3-8B-AWQ at revision
`4da05a8edb55c6046cce958586c33b61da07bb79`, vLLM 0.10.2, XGrammar 0.1.23,
context 12,288. Both requests had 4,479 template-inclusive input tokens and a
6,144-token output allowance. Request identity:
`b15526e7f5aeaf67845cf9167497ca4616815086a3feb96abe46f3abbc69436e`.
The changed service configuration is separately hash-bound in each attempt.

| Attempt | Actual completion | Generation allocation | Outcome |
|---|---:|---:|---|
| 1 | 6,144 tokens | 56.266611 s | HTTP 200 completed; length stop; 30,605/30,628 content characters whitespace, 6,119 newlines |
| 2 | 6,144 tokens | 55.974436 s | HTTP 200 completed; length stop; JSON truncated inside the first assertion-ID string containing repeated `0, ` |

The first failure justified the supported service-level
`--guided-decoding-disable-any-whitespace` repair. Its pinned CPU matcher rejects
the observed newline prefix, and the second service log confirms the flag was
active. Attempt 2 has 6,161 content characters and 2,047 whitespace characters;
these remaining spaces occur in the generated string, not a decoder-ignored
unbounded formatting gap. Merely increasing the token cap is not justified by
this repetitive, nonsemantic output.

`bounded_identifier_schema` implements a further **unactivated CPU candidate**:
generated ID strings use a 1–96-character ASCII identifier pattern. Closed
sealed-ID enums and descriptive text are unchanged. The pinned grammar accepts
a representative valid ID and rejects both the observed repeated-ID prefix and
a 97-character ID. This is a tested syntactic repair, not evidence that the LLM
will construct a scientifically valid ontology. No third service start occurred.

## Accounting and forecast

Historical pre-block allocation remains 3,227.826324 s. The block consumed
636.437522 s: 393.138483 s loading, 112.241047 s failed generation, and
131.057992 s other allocated service time (live checks, bookkeeping and shutdown).
Actual cumulative allocation is **3,864.263846 s**. Both services are physically
stopped and the authoritative ledger has no unresolved allocation or service
journals. The first controller's final resource-JSON serialization failed after
physical shutdown; its guardian preserved terminal accounting. The serializer
was repaired before the second start. No failed record was overwritten.

CPU-only invariant preparation was measured at 40.109124 s in the first preflight;
it preceded GPU allocation. Successful startup events measured 193.203338 and
199.935145 s. Neither measurement establishes valid generation p95 or permits
crediting another condition with C1's failed-generation speed.

| Remaining work | Conservative seconds |
|---|---:|
| 24 development calls | 3,660 |
| Held-out C1/C2/FixedSelect: 24/72/72 | 23,040 |
| 12 paraphrases | 1,800 |
| Six revisions and three traces | 1,350 |
| Reduced ablations: 12/8/8 | 4,200 |
| Narrative: four C1, eight C2, one full-index C2 | 2,310 |
| Remaining historical reserves: 1 long, 8 standard, 4 short | 1,800 |
| Five main-study service envelopes, including overhead | 1,668.344344 |
| Pending acceptance/resume service envelope after diagnostics stopped | 333.668869 |
| **Total remaining** | **40,162.013213** |
| **Actual consumed + all remaining** | **44,026.277059** |

These are the documented unmeasured output-allowance sensitivity proxies, capped
at unchanged watchdogs; they are not reliable successful-completion forecasts.
All 267 remaining generation/reserve slots are retained. C1 240 s, C2 300 s and
FixedSelect 90 s acceptance earmarks are already inside the remaining reserves;
they are not added again. The historical load proxy includes service overhead,
so no duplicate shutdown allowance is added to those envelopes. The extra
acceptance/resume envelope makes that unfinished lifecycle work explicit rather
than pretending the two failed diagnostic starts completed it.

Ordinary admission fails by **10,366.277059 s** against 33,660 s. Even the unchanged
inventory alone, without the extra acceptance envelope, would fail. The current
proxy also exceeds the strict actual 36,000 s limit; raising the scheduled
ceiling alone would not solve that problem. Actual allocation is safely below
both limits. No valid timing sample supports a reduced proxy yet.

Sampled block peaks: 22,793,945,088 VRAM bytes, 3,159,506,944 process-tree RAM bytes,
eight workers. Block-reported project occupancy is 16,429,932,544 bytes; apparent
ledger samples use a distinct accounting measure. Both remain within occupancy
and headroom limits. The pod is active but vLLM is stopped.

## Execution decision

No ordinary acceptance/development or held-out inference is admitted. The same
block has 563.562478 allocated seconds and one attempt unused, but **zero starts**
remaining. The current start authorization is insufficient for another real test.

Recommended single resource decision, **not authorized or applied**: extend this
same feasibility block to **three total starts, five total diagnostic attempts,
and 1,800 total allocated seconds**. Keep its diagnostic-only forecast exception,
the 33,660 scheduled ceiling, strict actual stop before 36,000, and all historical
usage. That permits one bounded service to test the identifier repair and, only
after a scientifically valid C1, measure C2 and actual-C1 FixedSelect. It does not
authorize ordinary study execution under a failing forecast or claim feasibility.
The original nine-hour scheduled target was not met.

The independently completed C0 production calibration still fails competence;
see `docs/C0_DEVELOPMENT_REPAIR.md` for genuine repairs and the separately proposed,
unapplied reference-validity correction. Human review remains required before
held-out inference. No PDF or public release was generated.

## Reproducible evidence

Focused final regression results: 119 passed (92 controller/codec/C0/metrics/
actual-client tests plus 27 selected runtime/acceptance/shutdown tests). Restricted
JUnit receipts are `capacity-block-focused-tests.xml` and
`capacity-block-runtime-tests.xml`. Unaffected passing checks were reused.

The local restricted terminal backup is
`artifacts/restricted/capacity-block-terminal.JD1MdQ/`. Its ledger SHA-256 is
`d187bc1b6b417e6e0606fe1fcc20b0fb69fc404a62f1d3e3efc5d3892cfa07d1`.
Read-only verification found 19 CAS artifacts, five failed model-call records
(three historical and two new), and zero issues. Both raw HTTP bodies are
reassembled from independently hash-verified compressed fragments.

`scripts/summarize_capacity_diagnostics.py` regenerates
`artifacts/restricted/capacity-diagnostic-outcome-v1.json` from that backup and
the preserved V10 mandatory inventory. No source of model weights, novel text,
credentials, or reviewer judgments enters this receipt.
