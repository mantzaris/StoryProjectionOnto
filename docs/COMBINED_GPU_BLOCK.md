# Combined 49-call GPU block

The final synthetic follow-up allocation is one indivisible, sequential GPU-service
block. It is admitted only after the development gate has passed, all 168 held-out
primary calls have terminal intention-to-treat records, the held-out runtime namespace
is closed, and the Phase 5 inputs are frozen. The controller never reads scorer gold.

## Frozen inventory

The tracked configuration is `configs/study/combined_gpu_block.json`. The compiler
loads and hash-checks the registered benchmark, seed, paraphrase, no-context,
eligibility, and feedback manifests, then emits exactly this order:

1. 12 one-seed C2 paraphrase calls;
2. 6 scripted C2 feedback regenerations;
3. 3 C2 researcher-interface traces;
4. 12 `A-NoContext` calls;
5. 8 `A-NoTemporalEpistemic` calls;
6. 8 `A-NoRareGuard` calls.

This is 49 base calls at the registered 95-second p95, or 4,655 seconds. One shared
model load adds 180 seconds, for a 4,835-second load-inclusive forecast under the
4,860-second activity ceiling. Calls have concurrency one. Each base call may request
at most one 90-second repair, but a durable global authority permits no more than the
unconsumed portion of the four-slot short reserve shared with earlier phases.
The load retains its registered 180-second p95 while using a 300-second operational
watchdog because the accepted fallback's recovered cold start exceeded 180 seconds.
Admission includes that full watchdog ceiling plus the still-required p95 schedule;
all actual service time remains charged continuously.

## Scientific boundaries

Every source binds the closed held-out execution, scorer-bridge authorization, final
review seal, pre-query barrier, and the 36-record ablation preparation registry. C2
paraphrases reuse only the seed-1 query-blind empty inventory. Their reworded
`QueryContext` receives a new, honestly post-freeze `QueryAccessEvent`; the original
evidence packet is reused byte-for-byte and that reuse is recorded in SQLite and CAS.
All structured context fields, horizon, gold signature, budgets, decoding, repair
policy, and seed remain fixed.

Each ablation uses its own condition-matching empty pre-query inventory:

- `A-NoContext` replaces only the model-visible structured context with the generic
  request; evidence, horizon, upper ontology, budgets, seed, decoder, and repair remain
  fixed.
- `A-NoTemporalEpistemic` alone disables temporal/epistemic fields and binds its
  distinct output schema, decoder, capability manifest, and later full-gold scorer.
- `A-NoRareGuard` removes only the rare-pivotal guard/check; it retains every evidence
  item and uses the ordinary C2 output grammar.

The trusted provider resolves the exact public synthetic evidence packet through a
dedicated public-packet pointer; preparations and model outputs remain separately
typed restricted objects. The model worker receives only
`CombinedPreparedCall.model_visible_payload()`. A blocking scan rejects scorer/gold
namespaces. Public output contains hashes, counts, statuses, and resource measurements
only; semantic requests, raw generations, and condition attempts remain restricted.
Synthetic query-access and packet-reuse receipts retain the public release class of
their source packet so SQLite/CAS release invariants remain exact.

## Lifecycle, metering, and recovery

`CombinedGpuController` is the sole lifecycle owner. It writes an activation intent
before starting the model. This intent is path-free and binds the exact activation
slot, configuration, session, and accounting event. The controller verifies exactly
one new SQLite `MODEL_LOAD` event and injects a lifecycle-free view of that same
service into the existing Phase 5 adapter. The view has inference and recovery methods
only; it cannot start, stop, or replace the service.

If a controller exits after writing the activation slot but before lifecycle
activation begins, replay permits the first load only after proving that no load
event, service journal, checkpoint, or lifecycle binding exists. If the process was
already launched, a new controller obtains exclusive ownership and may adopt only the
exact live process whose lease and open journal match the frozen configuration,
session, event, PID start ticks, argv hash, start time, accounting baseline, and healthy
endpoint. It then reconstructs the service checkpoint and combined binding without a
second launch. A terminal stale recovery, including an idempotent replay after terminal
accounting, is distinguished from live adoption and permanently consumes the load.
Every partial, mismatched, or ambiguous physical activation fails closed.

Before an ordinary request is sent, the controller persists its complete call slot. A
slot with no adapter intent, evidence-preparation artifact, GPU event, or model-call
row is safe to issue once. Exact terminal GPU-event, model-call, and raw-CAS lineage
reconstructs a missing semantic completion and ITT result without inference. Only a
genuinely in-flight or inconsistent state blocks;
an uncertain request is never resent. Base and optional repair GPU events, attempts,
model calls, condition-attempt CAS objects, and cumulative counters are replayed
independently. Invalid, failed, timed-out, and interrupted outcomes stay in the ITT
inventory. Repair claims are append-only and permanently consume their global slot.

After all 49 calls are verified in manifest order, the lifecycle owner physically
stops vLLM and closes allocation accounting. A crash between shutdown and final-index
publication resumes from the stopped receipt without adopting, loading, or resending
the service. The restricted execution index binds all 49 call specs, all 40 ordinary
ITT records, the nine-call Phase 5 index, repair claims, the sole model-load event, and
the physical shutdown receipt.

## Command-line entry point

The three production inputs must first be generated by the CPU-only trusted compiler
`scripts/materialize_combined_gpu_inputs.py`. It is the required route for producing
`runtime_binding.json`, `upstream_gate.json`, and `call_manifest.json` from the sealed
predecessor artifacts; do not hand-author or edit those files. The compiler performs no
model calls and also emits a hash-bound `compiler_manifest.json`. Run
`PYTHONPATH=src python scripts/materialize_combined_gpu_inputs.py --help` for its exact
required predecessor paths, snapshot/cache binding, output root, and timestamp.

Only after that compiler succeeds does the execution entry point accept the typed,
hash-bound outputs. A read-only audit is:

```bash
PYTHONPATH=src python scripts/run_combined_gpu_block.py \
  --repository . \
  --manifest artifacts/restricted/combined_gpu_inputs/call_manifest.json \
  --runtime-binding artifacts/restricted/combined_gpu_inputs/runtime_binding.json \
  --upstream-gate artifacts/restricted/combined_gpu_inputs/upstream_gate.json \
  --compiler-manifest artifacts/restricted/combined_gpu_inputs/compiler_manifest.json \
  --development-result artifacts/restricted/development/execution_result.json \
  --held-out-manifest artifacts/restricted/held_out/call_manifest.json \
  --held-out-execution artifacts/restricted/held_out/execution_manifest.json \
  --scorer-bridge artifacts/restricted/held_out/scorer_bridge.json \
  --final-schedule artifacts/restricted/held_out/final_schedule.json \
  --phase5-inputs artifacts/restricted/phase5/inputs.json \
  --phase5-primary-gate artifacts/restricted/held_out/primary_results_gate.json \
  --selected-model-freeze artifacts/restricted/fallback-development/selected-model-freeze.json \
  --source-association artifacts/public/manifests/combined-production-source.association.json \
  --validate-only
```

`--run` uses the sole factory frozen in the configuration:
`story_projection_onto.combined_gpu_factory:create_frozen_production_combined_bundle`.
An optional `--adapter-factory` value is accepted only when it is byte-for-byte that
same reference; arbitrary plugin factories are rejected before import. The run also
requires a run ID, global SQLite ledger, restricted CAS root, runtime root, quota root,
the pinned local model snapshot and shared cache, and the verified snapshot manifest.
The selected-model freeze and source association are already mandatory compiler-source
arguments in the audit command. For example, append:

```bash
  --run-id combined-v1 \
  --ledger artifacts/restricted/global-ledger.sqlite3 \
  --artifact-root artifacts/restricted/cas \
  --runtime-root artifacts/restricted/combined-runtime \
  --quota-root /workspace \
  --snapshot /workspace/hf-cache/models--Qwen--Qwen3-8B-AWQ/snapshots/4da05a8edb55c6046cce958586c33b61da07bb79 \
  --shared-cache /workspace/hf-cache \
  --verified-model-manifest artifacts/public/manifests/model_snapshot_fallback.json
```

Factory construction is CPU-only. It rehashes the model snapshot, tokenizer,
launcher, source association, prompts, schemas, decoders, upper ontology, CAS
evidence/preparations, resource limits, storage, and predecessor ledger before the
lifecycle owner may load vLLM. Do not place credentials, model weights, or copyrighted
corpus material in tracked manifests.

The controller gates each load, base request, and claimed repair against current live
service allocation plus that request's watchdog, every still-mandatory p95, and a
protected 60-second shutdown margin. The exact positive remaining-registered forecast
receipt is passed through every ordinary, repair, and Phase 5 generation call; zero or
nonfinite receipts are rejected. Ledger, CAS, runtime state, and semantic journals must
resolve beneath the explicit repository `artifacts/restricted` authority with no
ancestor symlink escape. The optional public summary must resolve at its distinct
frozen `artifacts/public` path. Filesystem journals use same-directory, content-addressed
temporary files, file and directory `fsync`, and immutable publish; truncated targets
or interrupted temporaries fail closed. Run IDs bind activation, service state,
semantic intents, recovery, shutdown, final index, and recomputed public summary.
