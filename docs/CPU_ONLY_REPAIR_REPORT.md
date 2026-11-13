# Bounded CPU-only repair

No GPU service, inference, model download, remote deployment, budget amendment,
or PDF generation occurred in this repair. The authoritative plans, scientific
thresholds, model revision, request token budgets, and mandatory comparisons are
unchanged. V9 remains terminal; its original ledger and result hashes still match.

## Evidence loss and repair

The executed fallback path is `FallbackAcceptanceRunner.run` →
`VLLMService.run_fallback_test` → `VLLMGuidedJSONClient.generate` →
`_urllib_transport` → `_decode_generation_response`.

Previously, `response.read()` kept the body only in memory. The decoder parsed
both the HTTP envelope and `message.content` before returning `GenerationResult`.
A decoding exception therefore prevented the runner's later CAS write. The
fallback failure handler retained the exception class; detailed restricted
diagnostics existed only for non-200 responses. V9's log demonstrates generation
activity, but establishes neither a particular HTTP response nor a finish reason.
Truncation remains a hypothesis, not a recovered diagnosis.

The repaired production fallback factory enables an ignored restricted journal
under `artifacts/restricted/http_diagnostics/<run-id>/`. It records HTTP status,
allowlisted response headers, atomic compressed body fragments, fragment offsets
and hashes, completion status, and full structured exception chains before JSON
interpretation. Exception causes, contexts, grouped exceptions, notes, and stack
locations are retained without local-variable values or source-line contents.
Authorization/cookie headers are excluded, sensitive exception text is redacted,
and no request headers or request body are journaled. Loopback requests bypass
proxies and cannot follow redirects.

Each response has an 8 MiB/1,024-fragment bound. Limit violations fail closed and
retain prior fragments. Interrupted event logs can have a torn final line; earlier
fsynced events and atomic fragment blobs remain recoverable. Journal files never
enter model-visible prompts or public result manifests.

Stages distinguish transport, HTTP, decoding, schema-validation, and scientific-
validation failures. New identifiable decoding failures are classified as invalid
outputs, not unexplained service failures. Historical V9 classification is not
rewritten. The original acceptance and grounding validators still reject invalid
schema, capability, temporal, epistemic, and semantic content. No CPU code repairs
or invents ontology content.

## Changed implementation

| File | Change |
| --- | --- |
| `src/story_projection_onto/http_diagnostics.py` | Restricted pre-parse journal and failure stages |
| `src/story_projection_onto/gpu_runtime.py` | Fragment-preserving real HTTP transport, decoder evidence, timeout lineage |
| `src/story_projection_onto/phase1_acceptance.py` | Diagnostic wrappers around unchanged schema/grounding gates |
| `src/story_projection_onto/controller_preparation.py` | Bounded readiness/release pipes for the separate CPU controller |
| `src/story_projection_onto/fallback_acceptance.py` | Production wiring, failure metadata, preallocation controller ordering |
| `scripts/measure_cpu_preparation.py` | Reproducible local measurements and unapproved budget calculation |

Tests changed/added: `tests/unit/test_fallback_acceptance.py`,
`tests/unit/test_controller_preparation.py`, and
`tests/integration/test_http_response_diagnostics.py`.

## Preparation measurement and limits

The run controller now prepares before the prepare controller may load the model,
then waits without opening its ledger writer, adopting a service, or revealing a
query. It keeps its separate PID. Parent loss, bad release, and timeout fail closed.
After release it rechecks guard authority, source bytes, model snapshot bytes, GPU
identity, and storage; subsequent ledger, service-identity, resource, and ordinary
admission checks remain in their original live execution paths. Snapshot hashing
is intentionally repeated, so that time is not claimed as a saving.

Five local samples, in seconds:

| Component | Local median | Credited RunPod/GPU saving |
| --- | ---: | ---: |
| Controller module import | 0.661676 | 0 |
| Policy/limit parsing | 0.000455 | 0 |
| Provenance loading | 0.010808 | 0 |
| Entire measured process-to-ready subset | 0.696744 | 0 |

The total includes process startup and is not additive with the component rows.
Pinned Torch imports, tokenizer setup, and RunPod filesystem/scheduling costs were
not measured: those dependencies/snapshot are not installed locally, and this
authorization did not permit new remote profiling jobs. V9's 149.819440 seconds of
service overhead include necessary live checks and shutdown; none is presumed
fully removable. No GPU-throughput improvement is claimed.

The source-bound canonical measurement is
`artifacts/public/manifests/cpu_preparation_repair_measurement_v1.json`.
Replay with a **new** output path:

```sh
PYTHONPATH=src /tmp/spo-refresh-venv/bin/python scripts/measure_cpu_preparation.py --output artifacts/public/manifests/cpu_preparation_repair_measurement_REPLAY.json
```

## Budget decision required — proposal, not an amendment

The next proposed call is the identical `fallback-c1-01` request, on the same pinned
Qwen3-8B-AWQ revision `4da05a8edb55c6046cce958586c33b61da07bb79`, with unchanged
prompt, seed, decoding, schema, and output cap. Its purpose is to obtain durable
response/finish-reason evidence through the repaired pathway, not to try a new
model or loosen validation.

Proposed recovery-gate envelope: 300 seconds startup, 120 seconds live controller
checks, 240 seconds for the C1 retry, 120 seconds validation/resource drain, and
60 seconds shutdown: **840 seconds**. Those new whole-gate/live-control caps are
not yet enforced by the existing runner and must be implemented/tested before
any subsequently approved launch. The measured preparation savings are credited
as zero.

| Forecast component | Seconds |
| --- | ---: |
| All prior actual allocation | 2,936.238858 |
| Remaining registered work and reserves | 29,987.344344 |
| Earmark one already-forecast reserve-long slot; do not count it twice | −240.000000 |
| Proposed bounded recovery gate | 840.000000 |
| All-in forecast bound | **33,523.583202** |
| Current scheduled ceiling | 32,400.000000 |
| Deficit under the current ceiling | **1,123.583202** |

Ordinary admission does **not** pass. The remaining forecast retains its five
planned service loads and every mandatory scientific comparison; no future work
is removed or optimistically accelerated. The retry earmarks one of the two
remaining long reserve slots; it creates no extra inference slot.

Concrete proposal: raise only the scheduled ceiling to **33,660 seconds
(9.35 hours)**, leaving 136.416798 seconds of scheduled reserve under this bound.
The strict actual hard ceiling stays **below 36,000 seconds**, with a projected
2,476.416798-second margin. All prior failed and unattended service allocation
remains charged. This would be a disclosed feasibility/resource amendment: the
original nine-hour scheduled target was not met. It would not change hypotheses,
conditions, independent units, endpoints, scientific validators, or the held-out
review gate. These forecasts remain provisional until valid model outputs yield
usable timing observations.

Exact authority needed: approve that scheduled-budget amendment and one new
service start for the bounded C1 recovery gate. If C1 fails or fresh admission
fails, stop and preserve evidence; no further startup is authorized. If it passes,
authorize continuation of the remaining registered acceptance and development
work **on that same service**, subject to their gates and the amended ceiling.
This avoids treating a standalone diagnostic as a complete acceptance pilot or
assuming another uncounted reload. Held-out execution stays blocked on independent
review. New source-bound preflight and cap enforcement are prerequisites, not
permission to launch now.

## Focused verification

256 unique focused tests pass across the final runs: 234 unit tests and 22 actual
loopback HTTP tests. The unit batch had 233 passes and one outdated exact-metadata
expectation; adding the newly required `failure_stage` to that expectation passed
its targeted rerun. Production code was unchanged for that correction. The HTTP
tests exercise valid fixture ontology validation, semantic rejection, malformed
inner/outer JSON, invalid UTF-8, incomplete fixed-length/chunked bodies, timeouts,
connection refusal, HTTP errors, redirect refusal, pre-decoder durability, secret
header exclusion, and response/fragment limits. No fixture is represented as LLM
output. Ruff and diff checks pass.

Logs/XML remain under `artifacts/restricted/cpu_repair_validation/`. The earlier
combined elevated-permission run was interrupted after 79 passes when a legacy
process-control fixture stalled outside its usual sandbox; its partial log is
preserved. Unit tests subsequently ran in the normal sandbox, separately from
the socket-permitted HTTP tests. Unrelated metric/report/review checks were reused.

The repair is local and not deployed to RunPod. No GPU resume command is currently
authorized. No results PDF was regenerated.
