# Fourth-start small streaming diagnostic

This is the explicitly authorized fourth start in `output-capacity-recovery-v1`,
not a new block. It permits exactly one small evidence-only call, then shutdown.
No full C1, C2, FixedSelect, retry, acceptance certificate, or ordinary study call
is authorized by this start. Historical actual allocation is 4,347.542032 seconds;
the same block has used 1,119.715708 seconds and three starts/attempts.

The complete new envelope is at most 550 seconds, leaving at least 130.284292
seconds in the 1,800-second block. The controller admits 549 seconds including
shutdown once and the guardian enforces the earlier stage/whole deadline.
Startup: 240 seconds; live checks: 60; generation: 120; exception bookkeeping: 5;
validation: 60; shutdown reserved: 60. Stage maxima sum to 545 seconds; all gaps
also count against the whole deadline. Remaining block and global ceilings still
apply. Only the diagnostic complete-forecast prerequisite is excepted.

## Frozen task and success criteria

The model receives one complete development evidence record describing a courier's
arrival with an object, its query-blind candidates, provenance, and temporal hint.
It is asked to construct 2–4 supported entity/event nodes, 1–3 qualified assertions,
the required local types/predicates, and an explicit supported construction decision.
It receives no expected graph, gold, or contextual answer. The task uses the
production record-tuple codec and pinned XGrammar constrained-decoding pathway.

Success requires separately: HTTP 200; complete SSE including usage and `[DONE]`,
`finish_reason=stop`; complete content JSON; lossless canonical reconstruction;
unchanged Pydantic and production structural validation; meaningful structure as
specified above; and the unchanged restricted development semantic-grounding audit.
Only the one supplied record is admissible evidence. CPU code restores syntax and
administrative envelopes, never missing semantics. The exact criteria are saved
before allocation in each run's `small-success-criteria.json`.

The complete model-facing rendered prompt was inspected. It specifies every tuple
position, named type, enum, optional field/default, ID alphabet, supplied opaque
handle convention, and new-ID convention using the schema-derived legend. Its
small scope has no expected answer. Pinned CPU preflight measured 3,453 input tokens
including the chat template, with 6,144 output tokens reserved: 9,597 total of 12,288.
An existing authored development fixture with four nodes, two assertions and five
decisions needs 1,152 compact wire tokens. This is a capacity check, not a small-task
answer, a worst-case bound, or proof of model comprehension/completion reliability.
No token allowance, model, context limit, sampling parameter or validator changed
for this start. CPU preparation took 45.241687 seconds, allocated GPU zero.

## Streaming and observations

The actual request explicitly sets `stream=true` and
`stream_options={include_usage:true,continuous_usage_stats:false}`. Installed pinned
vLLM 0.10.2 emits chat delta events, a final finish event, usage-only event and
`[DONE]`; it passes the same `guided_json` sampling configuration to the engine.
The global XGrammar whitespace restriction and bounded identifiers remain active.

The urllib client stores restricted, content-addressed fragments with fsynced
journal references before SSE parsing. It handles split frames/UTF-8, usage and
finish events. Timeout shuts down the exact response socket and preserves the
prefix/full exception chain. Socket shutdown is a cancellation request, not proof
of when engine cancellation completes. Verified service shutdown remains required.

HTTP status establishes request acceptance; time-to-first event/content is measured
at fragment receipt (not inferred from polling). Server logs retain engine aggregate
scheduling/generation statistics, warnings, exceptions and shutdown. Per-request
grammar preparation and scheduling duration are unknown unless explicitly logged.
Lack of client bytes alone is neither repetition nor a generation stall.

Focused CPU checks: 77 passing tests, including the actual production-codec socket
path, mid-stream durable capture before parsing, timeout disconnect, truncated SSE,
malformed event/UTF-8, HTTP 500, streaming server errors, canonical reconstruction
failure, scientific rejection, cumulative admission, one-small-call restriction,
guardian shutdown and unchanged non-streaming diagnostics. No GPU model output is
represented by these fixture tests.

## Actual execution: startup failed, small request not sent

The fourth start executed checkpoint `589f4ba` in the existing project. All 145
bound source/prompt/configuration hashes match. The prepared request hash is
`cde6c6b6eedefab00aa46b7a01833998ca0b291ad8ff1daf55e574ad23681e7a`.
The executed controller's invariant CPU preparation took 47.432633 seconds before
allocation (the earlier separate CPU check was 45.241687 seconds).

The service did not become healthy within its 240-second startup deadline. Logs
show architecture/configuration initialization and model loading starting at
16:37:46 UTC, with the first safetensors progress still at 0/2 shards when the
deadline arrived. The error chain is `RuntimeWatchdogTimeout` (not healthy), then
`startup resource sample exceeded the service-start deadline`, then
`cannot terminate vLLM while its owned resource sample remains in flight`.
The guardian retained the lease, completed cleanup and closed every allocation.
No diagnostic request was issued, reserved, or counted as a model call.

| Observation | Actual outcome |
|---|---|
| Server accepted small generation request | Not sent; unknown/not applicable HTTP status |
| First response event / first content | Not observed |
| Response content, usage, finish reason | None received; no inference attempted |
| JSON parsing / canonical reconstruction | Not reached / not reached |
| Canonical schema / structural validation | Not reached / not reached |
| Scientific validation | Not reached, not a scientific rejection |
| New complete / schema-valid / scientific outputs | 0 / 0 / 0 |
| Classified startup failure allocation | 239.985306 seconds |
| Shutdown and other allocated remainder | 46.274175 seconds |
| Whole new start | 286.259481 seconds (<550) |
| Preserved global actual allocation | 4,633.801513 seconds |
| Same block used / remaining | 1,405.975189 / 394.024811 seconds |
| Cumulative starts / diagnostic attempts | 4 / 3 |

Physical shutdown was verified at 16:38:55 UTC, followed by idle GPU at 1 MiB and
no service/controller processes. The pod remains active. The stopped ledger and
19 CAS artifacts verify with no integrity issue or unresolved allocation. All
historical attempts, logs and local ledgers are preserved. Restricted backup:
`artifacts/restricted/small-stream-terminal.EkOCND/`. Its run directory contains
the entire prepared request, rendered prompt, criteria, capacity receipt, source
binding, startup log, guardian terminal record and controller exception log.
There is no HTTP journal for this small request because transport never began.

Four completed startup samples were spaced roughly 48–56 seconds apart despite
the short requested sampling interval; the last reported 1,754,419,200 process-RAM
bytes, eight workers and zero process-attributed VRAM. These sparse observations
precede completion of weight loading and do not establish actual resource peaks.
`ResourceSampler.sample` takes its process snapshot before a full project storage
walk and only then samples GPU memory for those PIDs. An expensive storage walk
can therefore delay resource checks and leave the PID snapshot stale. Logs do
not isolate its component duration or prove it caused slow model initialization.
That is a concrete CPU-only profiling/repair target, not a reason to extend a
watchdog during execution. Project `du` occupancy is 16,442,096,640 bytes; the
smaller file-traversal sampler value is not used to assert a tighter storage peak.

Remaining mandatory-work proxy is unchanged at 40,162.013213 seconds. All-in with
actual use is 44,795.814726, exceeding 33,660 by 11,135.814726 seconds. No valid
production latency sample or p95 was obtained, no comparison/reserve was removed,
and no throughput credit is taken. Another 550-second start would not fit the
394.024811-second block remainder and is not authorized. Full acceptance,
restart/resume, development execution, and the independent review remain pending.
