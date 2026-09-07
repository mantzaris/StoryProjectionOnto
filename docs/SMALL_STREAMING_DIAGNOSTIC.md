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
