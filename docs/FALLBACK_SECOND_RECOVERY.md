# Second fallback recovery overlay

This repository supports, but does not itself authorize, one fresh v7 recovery
after the terminal `fallback-qwen3-8b-awq-development-v3` decoder-transport
incident and the later zero-GPU v4, v5, and v6 control-plane incidents. V4,
v5, and v6 are immutable terminal attempts and cannot execute or resume. The
existing v3 amendment, its validator, the v3 failed result and incident
association, and all three later public incident records remain immutable
provenance.

The second overlay is intentionally fail-closed. It must bind all of the
following before a model process can start:

- the exact v3 result and incident file and manifest hashes;
- the exact self-hashed v4 zero-GPU control-plane incident;
- the exact self-hashed v5 and v6 zero-GPU control-plane incidents;
- the original v3 retry amendment and its predecessor through the original
  amendment validator;
- the cumulative 815.215409 GPU seconds, five ledger events, and four service
  sessions recorded after v3;
- the failed request and decoder-schema hashes;
- the finalized local/remote source association and current compatibility
  implementation hashes;
- the compatible decoder-schema hash and exact rebuilt C1 retry-request hash;
- the typed, self-hashed immutable-v3 evidence-provenance certificate, its
  resolver module, and its focused regression test, each by path and file hash;
- the exact model-visible `legacy_evidence_provenance` section and its packing
  hash, plus both the unchanged legacy-evidence value hash and the resolved
  evidence hash;
- the rebuilt retry's complete system/user messages, condition, model, seed,
  decoding manifest, and packing-section hashes reconstructed from the frozen
  C1 fixture/prompt plus the authenticated v3 tokenizer manifest, so a fresh
  overlay cannot rebaseline a modified request;
- the historical pre-data C0 two-fix layer, frozen from v3 hash
  `bb7ec5f00df86ec24eedeaea6b90a1bda473511d0ab9f334cecda9f1970f27c7`
  to implementation hash
  `cf907e89e6ab241ea3708fcf9b0ff083cc96a5052c61e9230906c0ad69863f8d`
  and regression hash
  `9fc632407a8eda7a7fcb8c6d3994ca204d290daefb71089b1d6f160fd36e0816`;
- the historical C1 semantic-status layer, frozen from v3 hash
  `828ab6e093d2baf9bf9147ad4202ba53c2bac9af3977be6cc3c216e5e294e55f`
  to implementation hash
  `a0bd4f61dff5df8fc668d4d185ec760c280af2c3a9848066223dfea7baf62151`;
- a separate projection-dependency correction that binds those exact C0 and C1
  predecessors to the current dependency-closed implementations, validation
  contracts, and regression files;
- a separate concurrent-integrity disclosure that hash-binds runtime semantic
  scope, registered-display feasibility, scorer grounding/ITT behavior, and
  append-only lifecycle/accounting sources and tests;
- zero accepted, base, or development predecessor outputs and zero model/GPU
  calls made by either new correction;
- one additional service allocation forecasted conservatively as the greater
  of the 300-second startup watchdog and the observed successful service
  allocation p95, plus exactly one `AttemptKind.RETRY` for `fallback-c1-01`
  charged to the next registered `reserve_long` slot;
- the corrected forecast and an explicit, dated user-authorization basis.

The retry does not add unreserved inference capacity. It moves cumulative
`reserve_long` consumption from one to two of four registered slots. The global
maximum remains 278 inference attempts. Only the second recovery service start
is new, moving the effective accounting-event ceiling from 287 to 288.

The corrected forecast retains the successful v3 service-start observation and
excludes the 0.852878-second request rejection from successful C1 timing. Before
any further allocation, the values are:

- actual allocation: 815.215409 seconds;
- remaining mandatory forecast: 30,516.0027264791 seconds;
- service-start watchdog: 300 seconds;
- observed successful service allocation p95: 391.40054529582005 seconds;
- additional service allocation forecast:
  `max(300, 391.40054529582005) = 391.40054529582005` seconds;
- projected scheduled allocation: 31,722.61868077492 seconds;
- scheduled reserve: 677.38131922508 seconds.

These values must be recomputed by the validator from immutable inputs; copying
them into an overlay is not sufficient.

## Immutable v4 zero-GPU control-plane incident

The authorized `fallback-qwen3-8b-awq-development-v4` attempt passed its
CPU-only preflight and wrote an orchestration invocation, first guard, and
guardian ticket. The guardian then rejected the invocation identity before
readiness, and the orchestrator exited with `fallback guardian exited before
readiness (1)`. No internal controller, checkpoint, handoff, model process,
service-start allocation, or result was created. The post-failure status kept
the cumulative allocation at 815.215409 seconds with zero unresolved GPU
allocations or service sessions; v4 therefore consumed zero additional GPU
seconds.

The contemporaneous status receipt's `resume_allowed: true` is superseded by
this postmortem. v4 must not be resumed, and none of its run ID, run root,
output paths, tmux session, invocation, guard, or guardian state may be reused.
Preserve those bytes and the immutable incident record at
`artifacts/public/manifests/fallback_gpu_acceptance_development_v4_control_plane_incident.json`.
Every v5 proposal, preflight, launch, status check, and resume check must bind
that record through `--prior-control-plane-incident`.

## Immutable-v3 evidence-provenance bridge

The three registered Phase-1 request fixtures remain byte-identical. They were
sealed before model-visible evidence records carried the source lineage now
required by the production validator. The only exception is therefore an
explicit, fail-closed bridge loaded from
`configs/study/phase1_legacy_evidence_provenance.json`. The second-recovery
overlay binds that certificate's self-hash and file hash together with
`src/story_projection_onto/phase1_legacy_provenance.py` and
`tests/unit/test_phase1_legacy_provenance.py`. A missing, altered, symlinked,
unregistered, or differently hashed fixture is rejected; missing provenance in
any other development, held-out, production, or case-study request does not
activate this compatibility path.

The bridge derives only administrative source identity from byte-verified
records: passage ID, exact text hash, record confidence, and provenance source
artifact hash. It performs no entity, event, relation, temporal, epistemic, or
other semantic construction. The model sees the resulting
`legacy_evidence_provenance` section, and the packing report binds its canonical
SHA-256 as a required section. Repair requests copy that exact section and must
retain the same section hash. Base construction, resume reconstruction, repair
construction, and validation all receive the same already verified bridge
instance; the repair builder independently resolves the registered base call
and rejects even a self-consistently rehashed replacement section.

Validation compares source-critical provenance, not whole
`ProvenanceReference` object identity. Every cited `evidence_id` must have one
provenance row; `locator` and `source_artifact_hash` must equal the indexed
source; and emitted confidence may not exceed the lower of the indexed record
and provenance confidence bounds. Assertion-specific `provenance_id` and
`extraction_method` may differ. This distinction permits the model to identify
its assertion-specific derivation without permitting it to substitute a source.

For `A-FixedSelect`, the raw sealed ontology and model output are compared
before any effective binding. The prompt preserves the historical raw null
source hashes, while naming the authenticated hash separately for validation.
Only after the raw seal check succeeds may the controller insert a missing
source hash into a separate validation-only view. The sealed ontology, model
output, and historical fixture bytes are never rewritten.

## Layered pre-data corrections

The first historical C0 layer corrects two deterministic behaviors: it parses
the development fixtures' `story step` point/validity form and admits the
evidence-supported `served_as` predicate. It is frozen to the three hashes shown
above. In particular, the historical record does not read the live C0 source or
test and cannot relabel later bytes as the two-fix result.

The first historical C1 layer replaces a falsely semantic-positive
query-projection runtime record with the shared structural-only record:
`accepted` plus `not_applicable` for evidence support, temporal correctness,
and commitment correctness, with the standardized non-semantic diagnostic. It
is likewise frozen: the historical validator checks the v3 and corrected C1
constants and the fixed hashes of
`tests/integration/test_condition_pathways.py` and
`tests/unit/test_development_assessment.py` as they existed at that layer. It
does not require the live C1 file or either live regression file to equal those
historical hashes.

The later correction has the fixed ID
`second-recovery-predata-projection-dependency-integrity-v1`.
It recursively retains assertion endpoints and roles, node-description
assertions, epistemic holders, proposition-content dependencies, all relevant
story/validity/holder-relative anchors, and all partial-order endpoints. It
then reruns deterministic structural validation on the final selected graph and
binds validation to the final projection's semantic payload without a circular
content hash. Repair parent hash and attempt are treated as one lineage pair.
The typed record binds the frozen C0 two-fix and C1 semantic-status predecessors
to the live C0, C1, `validate.py`, `contracts.py`, and exact focused regression
files.

This later correction changes selection feasibility and final
projection-validation targeting. Construction/selection code is therefore not
claimed byte-identical, and future projection outputs may change. It does not
change pre-query construction behavior or query-time ranking, and it was made
before any accepted/base/development output. The correction itself made zero
model calls and zero GPU calls. That projection correction does not itself
change the retry request; the separately declared provenance bridge does.

The concurrent typed disclosure separately binds four pre-data integrity
surfaces: required runtime semantic-assessment scope; one shared registered
final-projection display-feasibility rule with closure over binary/role
endpoints and epistemic holders, plus a confirmatory-geometry guard;
qualification-aware scorer grounding, including empty-output intention-to-treat behavior; and
append-only terminal lifecycle/GPU-accounting integrity. Its sorted source and
regression-test inventories are hashed file by file and re-derived by the
validator. These changes may alter later display or analysis outputs and are
not described as byte-identical construction/selection code.

The Phase-3 storage-admission receipt also binds the exact allocation,
occupancy, and minimum-headroom budget used for its sample. Replay recomputes
headroom and the ordered violation set from that sampled budget, requires the
live resume budget to match it, and rejects any runtime budget more permissive
than the registered 30/25/5 GB envelope. Ledger-first recovery therefore uses
the same authenticated live budget instead of silently substituting the global
allocation value for a differently sampled budget.

The semantic-scope inventory is exhaustive over explicit runtime-scope uses:
the shared contract, ledger schema/API/verifier, all six structural-record
writers, the additional ledger writers and readers, focused regressions, and
the three regenerated projection/validated-generation/schema-manifest JSON
contracts. The display inventory is exhaustive over production uses of
`REGISTERED_DISPLAY` or the shared selection compiler, including feedback,
Phase 5 capture/production/source materialization, Phase 4 geometry production,
renderer confirmation, UI compilation, validation, and the browser client.
Regression tests discover these call sites from the source tree and fail if a
new use is not added to the typed inventory.

The lifecycle inventory covers the ledger schema and independent verifier;
Phase 1, development, held-out, combined-block, case-study, Phase 4, and Phase 5
production writers/readers; canonical held-out binding; and the focused
regressions for those paths. An additional source-discovery regression requires
every production validation/projection/job/metric/visualization writer and
canonical ledger resolver to remain inside this hash-bound inventory.

The display gate applies only to final projections. C1's query-blind
`prequery_construction()` retains its historical comprehensive-prebuild
accounting and acceptance behavior, including the 30-node prebuild despite the
separate 20-node final display budget. Final C0, C1, C2, and `A-FixedSelect`
projections all receive the common display-feasibility gate.

The overlay does not establish these controls by rebuilding the current inputs
and asserting that they are unchanged. Instead it verifies the immutable v3
source manifest at file SHA-256
`47d249905194deca92eae9372e0392ee74d209323179be45dee825092f4fed0c`
and tree SHA-256
`81edb1ba173458970e93b903efa27981a96fb3008a8b4ec221f5810f4c515b20`,
then compares each relevant current file to its predecessor entry. The exact
comparison covers all five registered prompt files, the three fallback request
fixtures containing evidence/context/capabilities/budgets, the decoding and
fallback/development configurations, and the canonical `OntologyDraft`
schema. It separately verifies fixed aggregate hashes for prompt files,
fixtures, evidence values, budget values, and decoding parameter values, plus
the fallback seed tuple `(0, 0, 1, 1)`.

That record proves byte identity only for its declared fallback-micro-pilot
inputs and development-configuration scope. It does not claim that the whole
wire payload is byte-identical to the failed v3 request: the registered
decoder-compatibility transform changes guided-schema bytes and the hash-bound
provenance bridge adds one source-only model-visible section. It
also does not make a broad whole-study evidence or payload identity claim.
Every scientific wire-visible value of the one retry is nevertheless derived
again from immutable-v3 inputs; the decoder-facing schema is derived from the
unchanged canonical `OntologyDraft` schema through the registered compatibility
transform. Thus “not whole-wire byte-identical” does not permit changes to the
prompt, raw fixture evidence, semantic envelope, capability probe, model, seed,
or decoding policy values, nor any model-visible addition beyond the exact
certificate-derived provenance section.
As a fail-closed cross-check, validation first recreates the original decoder
schema at
`0b605bf4ed55dfe3bc7e14f988f9df002a7a295b8515be1274e774c4b03bf883`
and the complete failed v3 wire request at
`1cc73c5525e096a4df830892f37cdc8062899363a0b75835bb2f04b3a14a0d44`.
Only then does it derive and validate the retry. Its permitted wire delta is
the guided schema, the two schema-derived administrative hashes inside the
request envelope, and the exact hash-bound source-provenance section.
Authoritative plan hashes remain separately enforced. The single bounded-repair
limit and fact-free model-visible repair diagnostics remain unchanged; the
broader validation and repair policy is deliberately not claimed unchanged
because display feasibility and final validation targeting changed. The
model-facing `OntologyDraft` canonical schema is byte-identical at SHA-256
`270d0f9acab96a7ebf1edc0606da9494799633146d293992649836283c0d5210`.
The canonical `QueryContext` schema is likewise byte-identical at SHA-256
`fe32fcb5336f126ad212bf5382505e4b17a91e28774ae0f6d32ebe3868434d0c`.
The canonical projection and validated-generation schemas are not claimed
byte-identical because `ValidationRecord` now requires an explicit semantic
assessment scope. The C2 and `A-FixedSelect` condition implementation files are
also not claimed byte-identical: their finalization path now supplies the
registered horizon to validation, and their current model-visible requests
carry the declared provenance section. Their live bytes are still hash-bound
by the implementation inventory. The unchanged controls are instead the C2
query-dependent construction semantics, the mechanical `A-FixedSelect`
restriction, and its raw seal comparison before any effective provenance
binding.

## Authorization and dry validation

`SecondFallbackRecoveryOverlay` schema version 1.5.0 requires the typed evidence
provenance bridge binding, projection-dependency correction,
concurrent-integrity disclosure, and the exact v4, v5, and v6 control-plane
incident bindings. It
accepts either `proposed` or `authorized` so a complete proposal can be checked
without inventing approval. A proposal must
leave `authorized_by` and `recorded_at` null. An authorized overlay must name
`user`, contain the actual non-pending authorization basis, contain an aware
timestamp, and have a correct canonical manifest hash.

Use the fallback CLI's existing `--validate-only` mode with these additional
arguments:

```text
--retry-amendment configs/study/fallback_service_retry_amendment.json
--prior-fallback-failure <v1-terminal-result.json>
--second-recovery-overlay <proposed-or-authorized-overlay.json>
--second-recovery-v3-result artifacts/public/results/fallback_gpu_acceptance_development_v3.json
--second-recovery-v3-incident artifacts/public/manifests/fallback_gpu_acceptance_development_v3_incident.json
--prior-control-plane-incident artifacts/public/manifests/fallback_gpu_acceptance_development_v4_control_plane_incident.json
--prior-v5-control-plane-incident artifacts/public/manifests/fallback_gpu_acceptance_development_v5_control_plane_incident.json
--prior-v6-control-plane-incident artifacts/public/manifests/fallback_gpu_acceptance_development_v6_control_plane_incident.json
```

Validation is CPU-only. It verifies the bridge certificate and every registered
fixture, rehashes the source association and retry request, verifies the
provenance packing section, and directly compiles the decoder schema with
XGrammar 0.1.23 without importing vLLM. For a valid proposal it writes a report with `passed: false`,
`execution_authorized: false`, and the remaining authorization gap, then exits
with status 2. `--execute` uses the authorization-required validator and rejects
the same proposal before starting a service.

## Deterministic CPU-only builder

The builder derives all hashes, accounting, decoder and provenance bindings,
and forecast fields from the exact v3 artifacts, current source association,
local pinned tokenizer snapshot, verified provenance certificate, and
cumulative ledger. It does not import vLLM or construct
a model service. Its output root must be explicitly named `restricted`; writes
are append-only and an exact byte-identical replay is idempotent.

Build the inert proposal after the final source association exists:

`source_tree_fallback_second_recovery_v5.association.json` below is the required
fresh association; no v3/v4 association or earlier v5 candidate can bind the
implementation added here.

```bash
python -m story_projection_onto.fallback_acceptance \
  --build-second-recovery-overlay \
  --project-root <project-root> \
  --output <project-root>/artifacts/restricted/fallback-second-recovery-v5.proposed.json \
  --restricted-output-root <project-root>/artifacts/restricted \
  --run-id fallback-qwen3-8b-awq-development-v5 \
  --primary-result <project-root>/artifacts/public/results/phase1_gpu_acceptance_v2_failed.json \
  --activation-certificate <project-root>/artifacts/public/manifests/fallback_activation_v2.json \
  --snapshot <pinned-model-snapshot> \
  --source-association <project-root>/artifacts/public/manifests/source_tree_fallback_second_recovery_v5.association.json \
  --retry-amendment <project-root>/configs/study/fallback_service_retry_amendment.json \
  --prior-fallback-failure <project-root>/artifacts/public/results/fallback_gpu_acceptance_development_v1.json.controller-handoff.json \
  --second-recovery-v3-result <project-root>/artifacts/public/results/fallback_gpu_acceptance_development_v3.json \
  --second-recovery-v3-incident <project-root>/artifacts/public/manifests/fallback_gpu_acceptance_development_v3_incident.json \
  --prior-control-plane-incident <project-root>/artifacts/public/manifests/fallback_gpu_acceptance_development_v4_control_plane_incident.json \
  --ledger <project-root>/artifacts/restricted/phase1_acceptance.sqlite
```

The default status is `proposed`. Authorized construction additionally requires
`--second-recovery-authorization-status authorized`, the actual user instruction
via `--second-recovery-authorization-basis`, and an aware ISO-8601 timestamp via
`--second-recovery-authorized-at`. Those values must never be guessed.

## Complete validation and execution argument flow

First generate the two manifests independently from the byte-identical local
and remote trees after the tested recovery checkpoint.  Synchronize without
`--delete`; the remote manifest must be generated by the remote Python process,
not copied from the local machine.  Copy only that resulting manifest back into
the same local manifest directory, then create the association. Source-tree
discovery excludes Python bytecode by both directory (`__pycache__`) and direct
file suffix (`.pyc`/`.pyo`), so recovered interpreter artifacts cannot enter or
create a false mismatch in the scientific source association:

```bash
revision_label=fallback-second-recovery-v5
manifest_root=artifacts/public/manifests
checkpoint_commit=<full-tested-recovery-checkpoint-commit>
recorded_at=<actual-aware-UTC-recording-time>

PYTHONPATH=src python -m story_projection_onto.manifest \
  --root . --revision "$revision_label" \
  --output "$manifest_root/source_tree_fallback_second_recovery_v5.local.json"

# Run independently in the byte-identical remote project directory:
PYTHONPATH=src .venv/bin/python -m story_projection_onto.manifest \
  --root . --revision "$revision_label" \
  --output "$manifest_root/source_tree_fallback_second_recovery_v5.remote.json"

PYTHONPATH=src python scripts/associate_source_manifests.py \
  --local-manifest "$manifest_root/source_tree_fallback_second_recovery_v5.local.json" \
  --remote-manifest "$manifest_root/source_tree_fallback_second_recovery_v5.remote.json" \
  --git-commit "$checkpoint_commit" \
  --revision-label "$revision_label" \
  --recorded-at "$recorded_at" \
  --output "$manifest_root/source_tree_fallback_second_recovery_v5.association.json"
```

Use one fresh run root and outputs that do not already exist. The source
association below must be the newly generated post-test association, not a v3,
v4, or earlier v5 candidate. `--validate-only` and operator execution both use the
public `orchestrate` stage; never invoke `prepare`, `run`, or `cleanup` directly.

```bash
study_root=/workspace/StoryProjectionOnto
study_python="$study_root/.venv/bin/python"
run_id=fallback-qwen3-8b-awq-development-v5
run_root="$study_root/artifacts/restricted/fallback-development-v5"
shared_cache="$study_root/.cache/shared"
snapshot="$shared_cache/hub/models--Qwen--Qwen3-8B-AWQ/snapshots/4da05a8edb55c6046cce958586c33b61da07bb79"
source_association="$study_root/artifacts/public/manifests/source_tree_fallback_second_recovery_v5.association.json"
second_recovery_overlay="$study_root/artifacts/restricted/fallback-second-recovery-v5.authorized.json"
preflight="$study_root/artifacts/public/manifests/fallback_gpu_acceptance_development_v5.preflight.json"
result="$study_root/artifacts/public/results/fallback_gpu_acceptance_development_v5.json"
tmux_session=storyprojection-study-v5

common_arguments=(
  --controller-stage orchestrate
  --project-root "$study_root"
  --run-id "$run_id"
  --primary-result "$study_root/artifacts/public/results/phase1_gpu_acceptance_v2_failed.json"
  --activation-certificate "$study_root/artifacts/public/manifests/fallback_activation_v2.json"
  --cache-replacement-receipt "$study_root/artifacts/public/manifests/model_cache_replacement_runtime.json"
  --snapshot "$snapshot"
  --shared-cache "$shared_cache"
  --verified-model-manifest "$study_root/artifacts/public/manifests/model_snapshot_fallback.json"
  --source-association "$source_association"
  --retry-amendment "$study_root/configs/study/fallback_service_retry_amendment.json"
  --prior-fallback-failure "$study_root/artifacts/public/results/fallback_gpu_acceptance_development_v1.json.controller-handoff.json"
  --second-recovery-overlay "$second_recovery_overlay"
  --second-recovery-v3-result "$study_root/artifacts/public/results/fallback_gpu_acceptance_development_v3.json"
  --second-recovery-v3-incident "$study_root/artifacts/public/manifests/fallback_gpu_acceptance_development_v3_incident.json"
  --prior-control-plane-incident "$study_root/artifacts/public/manifests/fallback_gpu_acceptance_development_v4_control_plane_incident.json"
  --ledger "$study_root/artifacts/restricted/phase1_acceptance.sqlite"
  --artifact-root "$study_root/artifacts/blobs/phase1_acceptance"
  --checkpoint "$run_root/checkpoint.json"
  --quota-root "$study_root"
  --port 8000
)

PYTHONPATH="$study_root/src" "$study_python" \
  -m story_projection_onto.fallback_acceptance \
  --validate-only \
  --output "$preflight" \
  "${common_arguments[@]}"
```

Do not replace validation with a direct
`python -m story_projection_onto.fallback_acceptance --execute` invocation.
Execution belongs only inside the checked-in
`scripts/run_fallback_gpu_acceptance.py` launcher, run by the detached
`storyprojection-study-v5` tmux session. The launcher creates an immutable
invocation, an append-only orchestrator guard chain, and an independently
persistent lease guardian before the prepare controller can start the model.
Every controller result is bound to the exact invocation, guard, argument hash,
runner execution hash, and inner canonical result. Fresh execution requires the
checkpoint, all controller outputs, guardian state, logs, and invocation state
to be absent; it never overwrites or interprets a stale result as current.

The production vLLM launcher itself uses a durable pre-exec identity gate. Its
session-leading child cannot execute the model command until PID, process start
ticks, intended argv hash, supervisor argv hash, CPU affinity, and the service
lease have been persisted. If the controller is killed in the `Popen`-to-lease
window, pipe EOF makes that child exit without executing vLLM. The guardian can
then conservatively close the already-open service journal. The flock inode is
used only for exclusion; before each compatibility write to it, the same lease
is self-hashed and fsync'd through an atomic sidecar replacement. A torn or empty
flock payload therefore cannot erase the last complete exact PID and argv
identity after the gate releases. Exact live leases in `shutdown_unverified` or
`accounting_pending` state are cleanup-only: they can be adopted and killed, but
can never be treated as a ready service.

Launch the fresh orchestration remotely and detach it from the local SSH
connection (the common arguments above are unchanged):

```bash
run_log="$run_root/orchestrator.$(date -u +%Y%m%dT%H%M%SZ).log"
printf -v launch_command '%q ' \
  env PYTHONPATH="$study_root/src" "$study_python" \
  "$study_root/scripts/run_fallback_gpu_acceptance.py" \
  --execute --output "$result" "${common_arguments[@]}"
printf -v quoted_log '%q' "$run_log"
tmux new-session -d -s "$tmux_session" \
  "exec $launch_command >>$quoted_log 2>&1"
```

`printf %q` materializes the exact expanded array without depending on an array
inside the detached shell. Preserve that command in the timestamped run log.
Do not paste an internal `prepare`, `recover-prepare`, `run`, `cleanup`, or
`guardian` command; those stages require private hash-bound tickets generated
by the orchestrator.

The CPU-only status interface takes the same result/checkpoint identity and may
be run after a local disconnect:

```bash
PYTHONPATH="$study_root/src" "$study_python" \
  "$study_root/scripts/run_fallback_gpu_acceptance.py" \
  --status --output "$result" \
  "${common_arguments[@]}"
```

Once the immutable orchestration invocation exists, status requires the exact
original complete `common_arguments` array. This is necessary to rederive the
execution-argument hash and each internal controller command hash before any
receipt is trusted. A shortened status command is permitted only before an
invocation exists and cannot be used for launch monitoring or resume decisions.

If and only if status reports `resume_allowed: true`, restart the orchestrator
inside the same named tmux session with the exact original expanded execution
command plus `--resume-orchestrator`:

```bash
printf -v resume_command '%q ' \
  env PYTHONPATH="$study_root/src" "$study_python" \
  "$study_root/scripts/run_fallback_gpu_acceptance.py" \
  --execute --resume-orchestrator --output "$result" \
  "${common_arguments[@]}"
tmux new-session -d -s "$tmux_session" \
  "exec $resume_command >>$quoted_log 2>&1"
```

Resume replays the immutable argument hash
and contiguous guard chain. It adopts an already-started exact service, selects
`recover-prepare` when the first handoff was interrupted, never repeats a
started model call, and never creates a second service-start event. A missing,
extra, reordered, or modified invocation/guard/result fails closed. Once the
guardian receives a terminal request, reaches the hard-stop deadline, or
expires the bounded owner-loss grace, scientific resume is forbidden and only
terminal reconciliation remains.

The public launcher first makes the orchestrator a dedicated process-group
leader. Each orchestrator and internal controller persists its PID, procfs
start ticks, exact raw-argv hash, PGID, and SID in a contiguous self-hashed
receipt chain. At a hard deadline the guardian atomically writes a controller
takeover receipt before sending any signal. Controllers check that receipt both
before and after registration, closing the registration-to-spawn race; a
takeover can therefore never authorize a later model start. The guardian then
revalidates exact identities and signals only their dedicated groups, even when
a controller still holds the vLLM flock. An unrelated process that merely
reuses a PID, PGID, command fragment, or port is never signal authority.

If takeover kills a controller while its append-only GPU allocation journal is
open, the guardian uses the verified physical service-stop timestamp as one
shared recovery boundary. It recovers every crash-open classified allocation
at that timestamp before reconciling the encompassing service interval, so the
service row stores only the non-overlapping complement. Terminal success
requires both the allocation-journal and service-journal unresolved sets to be
empty; a later meter replay must add zero seconds.

New service leases and resume checkpoints additionally bind the model session
leader's exact PID/start-ticks/argv to PGID=SID and to a random private instance
token inherited by its workers. If the leader exits while GPU-worker
descendants remain, cleanup-only recovery verifies every non-zombie group
member's SID and token before signalling the persisted group and reconciling
the complete wall-clock service interval. Existing v3 leases remain compatible
while their exact leader is live; because they predate the inherited token,
they cannot authorize a blind group-only kill. Atomic lease state is consulted
before a checkpoint during guardian cleanup, and a lost runner checkpoint does
not suppress terminalization when an exact lease or service journal survives.

Guardian/controller durable files are intentionally retained beside the
checkpoint: `*.orchestrator-invocation.json`, `*.orchestrator-guard-NNNNNN.json`,
the matching hidden close receipts, `*.guardian-ticket.json`,
`*.guardian-ready-NNNNNN.json`, `*.guardian-terminal-request.json`,
`*.internal-controller-NNNNNN.json`, `*.guardian-controller-takeover.json`, and
`*.guardian-result.json`. They are operational/restricted provenance and must
not be deleted or reused for another attempt. The invocation-to-ticket
power-loss window is repairable only when no guard, controller, checkpoint,
output, guardian, or takeover state exists; all other partial combinations fail
closed.

A valid dry preflight must
report `execution_authorized: true`, `passed: true`, no GPU allocation or model
process start, the exact 815.215409-second predecessor accounting, the corrected
31,722.61868077492-second projection, 677.38131922508 seconds of scheduled
reserve, 4,217.38131922508 seconds of hard contingency after protected
shutdown, 288 effective accounting events, and 278 maximum inference attempts.

After the micro-pilot passes and before the automatic 24-call development
continuation prepares any input or invokes its adopter, the fallback owner loads
the untouched `phase_3` reservation from
`configs/study/storage_phase_allocations.json`, runs that exact preflight, and
appends its observation to the cumulative ledger. The checkpointed typed receipt
binds the plan-file hash, all four reservation components, ledger sample, and
measured headroom through the bootstrap, handoff, and completion receipt. A
rejected observation is retained before shutdown and blocks the continuation.
If power fails after the append-only ledger row but before its checkpoint write,
resume reconstructs that exact receipt as the next ordered checkpoint entry and
then takes a fresh observation; it never reuses the orphan as current admission.
Any other ledger/checkpoint ordering mismatch fails closed. Only a completed
continuation with a hash-valid receipt and the complete matching ledger history
may replay without another storage observation or adopter call.

## Historical v6 launch record (terminal; do not execute or resume)

V5 is a preserved zero-GPU incident and has no resume path. The bounded v6
repair kept every scientific input, call, seed, model, and forecast unchanged;
it extended only the fail-closed control plane. V6 bound both prior public
incident records and a byte-identical local/remote source association with
revision `fallback-second-recovery-v6`. V6 is now itself a terminal zero-GPU
incident. The commands in this subsection document the historical attempt and
must not be executed or used as a resume recipe.

Use the manifest commands above with `v6` substituted for `v5`, and bind the
association to the tested checkpoint commit. Then build the authorized overlay
with the v5 command above changed to run ID
`fallback-qwen3-8b-awq-development-v6`, v6 source association and output names,
plus this required argument:

```text
--prior-v5-control-plane-incident <project-root>/artifacts/public/manifests/fallback_gpu_acceptance_development_v5_control_plane_incident.json
```

The authorization basis and aware timestamp must come from the actual user
instruction; they must not be synthesized. For validation and execution, use a
fresh run root `artifacts/restricted/fallback-development-v6`, preflight
`artifacts/public/manifests/fallback_gpu_acceptance_development_v6.preflight.json`,
result `artifacts/public/results/fallback_gpu_acceptance_development_v6.json`,
and tmux session `storyprojection-study-v6`. Add the same
`--prior-v5-control-plane-incident` argument to `common_arguments`. All other
arguments and the checked-in launcher remain exactly as shown above.

The historical detached launch command was:

```bash
run_log="$run_root/orchestrator.$(date -u +%Y%m%dT%H%M%SZ).log"
printf -v launch_command '%q ' \
  env PYTHONPATH="$study_root/src" "$study_python" \
  "$study_root/scripts/run_fallback_gpu_acceptance.py" \
  --execute --output "$result" "${common_arguments[@]}"
printf -v quoted_log '%q' "$run_log"
tmux new-session -d -s storyprojection-study-v6 \
  "exec $launch_command >>$quoted_log 2>&1"
```

Guardian initialization is CPU-only and now uses the already authorized
300-second startup bound. Terminal-result verification retains an independent
90-second bound. A status read may observe the guardian-held SQLite WAL only
after deriving and verifying the exact guardian argv, PID, process group, and
session; attempted-start resume additionally requires exactly one matching open
service journal. Any failed proof keeps `resume_allowed` false.

## Fresh v7 execution after the terminal v6 control-plane incident

V6 failed during guardian construction, before readiness, because the frozen
development-continuation service-start identity still named v3+v5 while the
production factory derived v3+v6. V6 started no model service, allocated no GPU
time, consumed no inference attempt or recovery service-start slot, and created
no accepted output. Its only ledger delta is one retained storage observation.
The public incident is preserved at
`artifacts/public/manifests/fallback_gpu_acceptance_development_v6_control_plane_incident.json`.
Its post-failure status has `resume_allowed: false`; its run root, invocation,
ticket, guard, log, status, run ID, and output names must not be reused.

V7 is the only executable second-recovery run. It retains the unchanged 288
effective accounting events, 278 maximum inference attempts, one additional
service load, and the single reserve-long retry. Before overlay construction,
generate and independently associate byte-identical local and remote manifests
under revision `fallback-second-recovery-v7`; the association basename must be
`source_tree_fallback_second_recovery_v7.association.json`. Build schema-1.5.0
with run ID `fallback-qwen3-8b-awq-development-v7`, all existing v4/v5
arguments, and the additional mandatory argument:

```text
--prior-v6-control-plane-incident <project-root>/artifacts/public/manifests/fallback_gpu_acceptance_development_v6_control_plane_incident.json
```

Validation, launch, status, and any otherwise eligible orchestrator resume must
all use the same complete v4+v5+v6 incident chain. Use fresh v7 paths:
`artifacts/restricted/fallback-development-v7`,
`artifacts/restricted/fallback-second-recovery-v7.authorized.json`,
`artifacts/public/manifests/fallback_gpu_acceptance_development_v7.preflight.json`,
`artifacts/public/results/fallback_gpu_acceptance_development_v7.json`, and
tmux session `storyprojection-study-v7`. Only v7 may resume, and only if its own
status reports `resume_allowed: true`; terminal v4, v5, and v6 state can never
be converted into v7 state or used as a resumable checkpoint.

## Fields finalized only after approval

After the implementation is tested, committed, synchronized, and associated,
the builder fills the current source-association file/manifest/tree hashes,
compatibility and schema-builder file hashes, compatible decoder schema hash,
exact retry-request hash, provenance certificate/module/test and packing hashes,
current C0/C1/contracts/validation and regression
hashes, all concurrent-integrity source/test hashes, new run ID, actual
authorization basis, authorizer, timestamp, and finally the overlay manifest
hash. Any source or request change after that requires a new proposal;
historical records must not be edited.
