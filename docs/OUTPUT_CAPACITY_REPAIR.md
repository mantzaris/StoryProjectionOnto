# Output-capacity repair and feasibility gate

Status: CPU implementation and capacity checks, not GPU acceptance. No new GPU
service or generation was started. The active model remains Qwen/Qwen3-8B-AWQ at
`4da05a8edb55c6046cce958586c33b61da07bb79`; total context remains 12,288 tokens.
The original nine-hour scheduled target was not met. The approved scheduled
ceiling remains 33,660 seconds and the strict actual stop remains before 36,000.

## Recovered evidence and diagnosis

The expressly authorized restricted transfer succeeded. V10's authoritative
ledger, CAS, response fragments, logs and terminal verification are preserved
under `artifacts/restricted/v10-recovery.TI5xOY/`. Its ledger SHA-256 is
`10b82d5a51e95b1a21e2cf64483f7301a9287ec7b64117b7d6f67347c2e6ecd2`.
Local verification found 19 artifacts, three historical failed model calls,
zero integrity issues, and no open service/allocation journals. The original
local V9 ledger was not overwritten; its SHA-256 remains
`53e4341b14f53dce48482cd765eb4654777df76061ff4948e0ea9bcf8a93dc5c`.

V10 completed HTTP 200 with 7,229 response bytes. It reported 6,609 prompt and
exactly 2,048 completion tokens, `finish_reason=length`, and JSON truncated inside
a string. This is an output-capacity failure at decoding, before scientific
validation, not an established transport failure. V9's exact failure remains
unknown. The immutable failed-call row's zero token fields are not rewritten;
the actual V10 usage is retained in the separate HTTP diagnostic record.

The served path really used `guided_json` with pinned vLLM 0.10.2 and XGrammar
without fallback. Thinking was disabled in the actual chat template. The retained
message has null reasoning content and no think markers. Retokenized content is
2,013 tokens; the 35-token difference from server usage is not attributed to
reasoning. Its 1,535 whitespace characters include spaces inside prose, so they
are not all removable JSON formatting.

## Implemented serialization

`output_wire.py` provides opt-in, lossless schema-directed record tuples. Required
values remain required; optional fields retain the same canonical defaults.
Only deterministic `content_hash` and `schema_version` envelopes are derived.
No entities, assertions, temporal/epistemic qualifications, citations or missing
semantics are constructed by the decoder. Canonical and scientific validators
remain active after translation.

Input tables deduplicate syntax and repeated strings. Opaque ID/hash fields use
request-bound reversible handles; prose, descriptions and time values are not
rewritten. The inverse map, canonical schema and sealed-copy bindings are saved
in restricted HTTP diagnostics **before transport**, alongside the already
implemented response bytes, status/allowlisted headers and exception chain.
The new prompt requests compact JSON. Inspection of the pinned V1 XGrammar
backend found that the per-request `guided_whitespace_pattern` field is not used
there, so that ineffective field was removed. Compact whitespace is therefore
a formatting instruction, not a claimed decoder guarantee. Capacity fixtures
use compact JSON; actual generation reliability and extra whitespace remain
unmeasured until a permitted diagnostic succeeds.

FixedSelect additionally permits an exact `sealed_id` reference to a complete
record actually present in its supplied sealed C1 ontology. It cannot override
any field in that reference. The deterministic decoder copies that exact record;
the existing selection-only semantic audit still rejects constructive changes.
Permitted descriptive/relevance edits require a full record tuple. This is not
truncation of C1, and C1/C2 cannot use sealed-copy references.

The declared candidate allocation is 6,144 input / 6,144 output tokens for
constructive first-pass and repair calls (C1, C2 and LLM ablations). FixedSelect
uses 9,216 input / 3,072 output because exact sealed-record references encode
selection without repeating the selected semantic payload. Both retain the same
12,288-token total, complete evidence and final object budgets; C2 receives no
advantage over C1. This policy is recorded but **not activated or held-out-frozen**.
Controller integration, complete development packing and admission remain gates.

## Capacity measurements are not model results

The pinned tokenizer and actual non-thinking chat template were used on the
production pilot requests and authored/development fixtures only. No held-out
gold was used for tuning. A twenty-node authored C1 stress output takes 8,223
tokens in compact canonical JSON and 12,617 when pretty printed, versus 5,980
in record tuples. A twenty-node authored C2 stress output takes 7,149 compact
tokens versus 5,102 in tuples. Thus a blind 4,096-token increase is inadequate.

For the C1 stress fixture, isolated field-key strings account for 2,226 tokens,
identifier strings 2,303, descriptive strings 712 and other strings 932, with
421 identifier occurrences and 105 unique identifiers. These isolated counts
are **nonadditive** because tokenization depends on punctuation and boundaries.
Short-ID counterfactuals in the capacity artifact are explicitly not outputs or
throughput evidence. Oversized CPU C0 constructions are also not a proof of a
minimum necessary C1 output size.

The final exact prompt/output totals and decoder compilation results are in the
restricted `output-capacity-v9/capacity.json` receipt. The intact twenty-node C1
to FixedSelect stress check restores all canonical records and hashes without
cutting the supplied graph or packet. Stress cases establish representability,
not scientific validity of artificial disjoint copies, generation reliability,
or capacity for every future development/narrative case. Each real request and
full repair request still requires exact packing before admission.

| Capacity check | Template-inclusive input | Reserved output | Total | Result |
| --- | ---: | ---: | ---: | --- |
| C1 pilot | 4,479 | 6,144 | 10,623 | Fits; XGrammar compiles |
| C2 pilot 1 | 4,887 | 6,144 | 11,031 | Fits; XGrammar compiles |
| C2 pilot 2 | 4,888 | 6,144 | 11,032 | Fits; XGrammar compiles |
| FixedSelect pilot | 6,327 | 3,072 | 9,399 | Fits; XGrammar compiles |
| Intact twenty-node C1 to FixedSelect | 9,036 | 3,072 | 12,108 | Fits; canonical round-trip |

The final selection stress response uses 456 tokens to select the full supplied
twenty-node/ten-assertion graph by exact record references. It contains an authored
selection decision, not C1 construction decisions. Its closed-ID schema is
derived by the unchanged production builder from that larger authored inventory;
IDs are translated bijectively in both schema enums and values. Earlier fixture
errors were rejected and their logs retained. The final CPU measurement took
30.346590 seconds wall time after its logged start; this is not GPU throughput.

## Allocation and complete remaining inventory

Historical actual allocation remains **3,227.826324 seconds**; this work adds
**zero**. Accepted model outputs remain C1=0, C2=0, FixedSelect=0. The ledger
contains three failed historical C1 attempts (V3, V9, V10), no successful repair,
and no development inference. V10 alone used 291.587466 seconds including
161.043200 startup, 20.152956 failed generation and 110.391310 service overhead.

There are no valid revised-format generation timings from which to estimate
completion p95. The conservative sensitivity calculation scales the old
provisional generation proxy by the allowance ratio (constructive 3x, repairs
4x, FixedSelect 1.5x) and caps attempt cost at the **unchanged** watchdog. Calls
whose demand proxy exceeds their watchdog are marked; a timeout-capped cost is
not a valid-completion forecast. No speed gain is credited for compression or
the truncated response. This is a conservative gate calculation, not proof the
study necessarily takes this long.

| Remaining work | Count | Attempt-cost proxy, seconds |
| --- | ---: | ---: |
| Future base loads, including retained restart requirements | 5 | 1,668.344344 |
| Development, including the registered repair | 24 | 3,660 |
| Primary/secondary/mechanism condition calls | 168 | 23,040 |
| Paraphrase | 12 | 1,800 |
| Scripted feedback and researcher traces | 9 | 1,350 |
| Reduced held-out ablations | 28 | 4,200 |
| Narrative windows and full-index operation | 13 | 2,310 |
| Remaining historical long/standard/short reserves | 1 / 8 / 4 | 1,800 |
| **Remaining total** | | **39,828.344344** |

All 25 inventory rows are retained in the machine receipt. Superseded normal
acceptance rows are not counted as executed outputs. The two outstanding fallback
C2 calls and one FixedSelect call remain earmarked within the standard/short
reserves (150+150+90), not added twice. New capacity diagnostics are separate
development-recovery attempts; they do not consume or disguise themselves as the
last existing 240-second long reserve slot.

The all-in proxy is **43,056.170668 seconds before new recovery work**, exceeding
the scheduled ceiling by **9,396.170668** and the hard ceiling by **7,056.170668**.
Adding the full authorized 1,200-second recovery envelope gives **44,256.170668**.
Actual hard headroom is still 32,772.173676 seconds, exclusively; that fact does
not override the complete-run admission rule. The earlier 29,747.344344 remaining
forecast used the inadequate output allowances and is not fresh admission.

Read-only storage samples after CPU work are 10,347,766,286 apparent bytes
(`du -sb`) and 16,395,871,744 block-reported bytes (`du -sB1`). Use the larger
sample conservatively: it remains below 25 GB occupied and leaves 13,604,128,256
bytes of the planned 30 GB allocation, including the protected 5 GB headroom.
These are different filesystem accounting conventions, not evidence of a new
six-gigabyte download. No dependencies or weights were installed/downloaded
during this repair. Do not silently compare unlike historical storage samples.

The small CPU admission module tests preserved history, two starts, three
diagnostics, the whole 1,200-second allocation bound, shutdown reserve, strict
hard stop and all-in scheduled admission. It does **not** claim to be a deployed
live guardian. No launcher was activated; the historical V10 launcher must not
be replayed with a changed request. Required restart/resume acceptance remains.

## CPU baseline and focused validation

Production spaCy C0 now completed four preconstructions and twelve structurally
valid development projections. A bounded fix avoids reifying dependency-parser
occurrences with no shared evidence event-candidate anchor; their binary
relations remain. Structural failure records now preserve the actual report.
Scorer-only routing now resolves anonymized runtime units through the frozen,
hash-checked development routing instead of treating runtime IDs as gold filenames.
All projections precede scorer access; none of these results is an LLM output.

The existing competence calculation **fails**: family coverage 0.20, strict
direct qualified-assertion precision 0.00, recall 0.00, valid evidence references
1.00. The thresholds remain 1.00 / 0.85 / 0.70 / 1.00. This negative calibration
is durable under `c0-calibration-v4/calibration.json`, not a passed competence
gate or a substitute for integrated development. Inspection found lexical and
validity-interval alignment discrepancies; it does not establish that C0 lacks
all competence or that every gold qualification is supported. No gold, scoring
threshold or semantic validator was changed to turn this result into a pass.

Focused verification: 71 unit/calibration checks and 24 real loopback HTTP checks
pass. Tests cover canonical hash equality, reference restoration, rejection of
unknown/overridden sealed copies, validation after translation, and retained raw
evidence for transport/HTTP/decoding/schema/scientific failures. Older failed CPU
attempts and their logs are preserved. No PDF or public artifact bundle was
regenerated. The independent human-review gate remains unchanged.

Changed implementation files are `output_wire.py`, `gpu_runtime.py`, `llm.py`,
`output_capacity_gate.py`, `conditions/c0.py`, and
`scorer_only/development_assessment.py`. Reproduction entry points are
`scripts/assess_output_capacity.py`, `scripts/calibrate_c0_development.py`, and
`scripts/summarize_output_capacity_repair.py`; the candidate resource/serialization
policy is `configs/study/output_capacity_recovery.json`. Focused tests accompany
each change. The restricted aggregate receipt is
`artifacts/restricted/output-capacity-repair-receipt.json`.

## Concrete next resource decision

Ordinary acceptance/development cannot launch under the current complete-run
forecast. Increasing only the scheduled ceiling would not solve the conservative
hard-cap overrun. The concrete proposed decision is a **feasibility-only timing
exception to complete-run forecast admission for one revised C1 diagnostic**,
not an amendment to either allocation ceiling or the study comparisons.

Use at most one start and one generation, with the already tested 840-second
envelope (300 startup, 120 live checks, 240 generation, 120 validation/drain,
60 shutdown) inside the authorized 1,200-second block. Charge every second as
new recovery, with zero historical-reserve subtraction: actual allocation would
remain at most **4,067.826324 seconds**. This exception is **not yet approved**.
First bind the revised request/source to the persistent controller and guardian;
then obtain a scientifically valid timing, shut down if admission still fails,
and report the revised complete forecast. One timing is not a p95 estimate or
full acceptance. No development or held-out execution is included in this
proposed exception. Scientific validity, packing and review gates remain intact.

Reproduce the restricted CPU receipt locally with:

```bash
PYTHONPATH=src /tmp/spo-refresh-venv/bin/python scripts/summarize_output_capacity_repair.py \
  --recovery-root artifacts/restricted/v10-recovery.TI5xOY \
  --output artifacts/restricted/output-capacity-repair-receipt.json
```

The receipt is append-only: use a new output filename for verification reruns.
There is no presently admitted GPU resume command. vLLM is stopped; the pod
remains active. CPU engineering can continue without changing these gates.
