# Small semantic-interface validation — actual outcome

Authorized single session at checkpoint `f2354ae`, 2026-09-07; pinned
Qwen3-8B-AWQ revision `4da05a8edb55c6046cce958586c33b61da07bb79`, unchanged
template/sampling/context. One start and two small attempts, no full C1 or study
execution. The second preselected passage was not run because the first task
never passed. The third call was not spent on an identical failed request.

| Stage / measurement | First small request | Specific uniqueness repair |
|---|---|---|
| HTTP / stream | 200, complete SSE | 200, complete SSE |
| Finish reason | `stop` | `length` |
| Template-inclusive input / output tokens | 4,712 / 2,083 | 4,831 / 6,144 |
| Output allowance | 6,144 | 6,144 |
| Measured client request seconds | 62.219858 | 111.532163 |
| First event / first content seconds | 2.040002 / 2.126830 | .568787 / .646439 |
| JSON syntax / generation schema | Pass / pass | Truncated string / not reached |
| Canonical reconstruction | Rejected: graph ID collision | Not reached |
| Reference and cross-field integrity | Rejected: nonunique IDs | Not reached |
| Required structure | Four node records, three assertions; IDs invalid | Partial generated graph only |
| Scientific grounding / task fulfillment | Not reached; not accepted | Not reached; not accepted |

**Zero scientifically accepted outputs.** A complete HTTP stream is not complete
JSON or a successful ontology. No failed response was corrected or retrospectively
accepted. The first inference event's transport-success bit is not scientific
acceptance; the associated attempt/outcome remains failed.

## What the model actually generated

The evidence-only first passage describes courier Lio arriving at North Gate at
dawn carrying a copper seal. The model created Lio, North Gate and seal entities
plus an arrival event, but reused `n001`, `n002`, `n003` as both entity and assertion
IDs. The frozen prompt already explicitly required distinct IDs across record
kinds. The schema cannot enforce dynamic cross-array uniqueness; canonical
validation correctly rejected it. Endpoint fields and exclusive time shapes were
present, unlike the preceding representation comparison, but this is not semantic
acceptance.

The second request appended the exact observed canonical uniqueness error, with
the same schema, evidence, sampling and allowance. Its versioned request hash is
`451ddc597bb2ce726164296317cb13b8c4f2fbcd3f14137135fda1872ae0f609`;
repair lineage points to the first attempt. No scorer facts or expected graph
were supplied. It began many repetitive `proposition_contents`, cycling endpoint
choices and incrementing IDs through `n037`, despite world-committed assertions
with null proposition links. The 17,864-character content ends inside a string.
The decoder reports an unterminated string at character 17,853. The full stream,
including `length`, usage and `[DONE]`, is retained.

This exposes unresolved model compliance with cross-record identifiers and
unbounded auxiliary proposition generation. The prompt did not omit the uniqueness
rule, and HTTP delivery/stream capture worked. It does not establish a general
model-capability failure or uniquely identify a decoder fault. A task-specific
auxiliary-record bound and clearer distinct identifier namespaces would be
evidence-supported CPU candidates; another generic token increase is not justified.
Neither candidate is activated for production, and no new startup is proposed
or authorized by this outcome.

## Allocation and feasibility

| Allocated component | Seconds |
|---|---:|
| Startup | 209.945128 |
| First generation | 62.174294 |
| Failed repair generation | 111.478161 |
| Live checks, preparation, validation and shutdown residual | 8.122229 |
| New session total | 391.719812 |
| All prior history | 5,363.502630 |
| Global actual | 5,755.222442 |

The 1,100-second allowance has 708.280188 seconds unused; this does not authorize
another start. The 6,463.502630 session-global cap was respected. Zero unresolved
allocation/service journals; vLLM stopped, no GPU compute process; pod still active.
Sampled peaks: VRAM 22,793,945,088 bytes, process RAM 7,084,208,128 bytes, project
storage 16,720,313,856 bytes. Stopped full storage census: 16,718,024,704 bytes.
These are observations, not a claim of unsampled instantaneous maxima.

Remaining mandatory proxy: **40,162.013213 seconds**, including the unchanged
inventory and pending acceptance/restart/resume service envelope. Global actual
plus that proxy is **45,917.235655 seconds**, **12,257.235655** above the amended
33,660 scheduled ceiling. The strict actual stop before 36,000 remains unchanged.
Small failures are not valid production throughput or p95; watchdog-capped
capacity proxies are not a reliable complete-success forecast. No comparisons,
repair reserve or required restart/resume test were removed.

The controller's stored forecast template retained an old historical baseline;
actual admission and terminal metering used the current ledger. The immutable
template is preserved. The restricted summary recomputes all-in cost from terminal
actual usage, and a post-session focused repair makes future forecast display use
the supplied current ledger value rather than that old template default.

## Reproducible restricted evidence and remaining gates

`artifacts/restricted/semantic-session-backup.72UAKV/summary.json` is generated
from immutable outcomes and hash-verified SSE fragments. Readable supplied evidence,
validation outcomes and exact model content are in
`EVIDENCE_AND_GENERATED_GRAPHS.md`. `BACKUP_VERIFICATION.json` verifies all **729**
copied diagnostic/log/ledger files against the pod. Previous ledgers and failures
remain; no model weights, novel or secrets were transferred. The frozen requests,
schemas and rendered chat are in the nested run directory. The preselected second
request is preserved there too, explicitly not executed.

CPU preparation passed on the pinned grammar/tokenizer; 46 interface/controller
tests passed locally and remotely before allocation. C0's separately approved scope
amendment is documented in `C0_PRECISION_DENOMINATOR_AUDIT.md`; it still fails
competence. Small-interface validity, production-method approval/packing, full
fallback acceptance including restart/resume, ordinary timing admission, C0
competence and independent human review remain gates. No held-out activation,
ordinary 24-call development execution or interim PDF was performed.

Post-session verification at `e841572`: 136 focused local tests pass; 15 focused
deployed CPU tests pass, and both C0 evaluations reproduce exactly on the pod.
