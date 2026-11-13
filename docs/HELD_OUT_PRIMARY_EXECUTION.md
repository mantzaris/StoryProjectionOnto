# Held-out primary execution control plane

This control plane is intentionally inert without three external inputs: a fully
materialized independent-review completion, a frozen passing development result,
and the tracked production live-service adapter named by the control
configuration. It cannot synthesize scientific outputs. The production factory
is frozen in the tracked source template; the development-result hash remains
`PENDING` there because that result can exist only after the same source tree has
passed the GPU/development gate. A restricted runtime binding supplies it without
mutating the accepted source tree.

## Frozen design

`configs/study/held_out_primary.json` binds the public benchmark, seed, GPU-call,
and development-exclusion manifests by file hash. Derivation produces exactly:

- 12 opaque held-out units and 36 query contexts;
- 12 deterministic C0 preconstructions;
- 24 query-blind C1 prebuild calls, two seeds per unit;
- 24 typed C2 empty `ConditionPreparation` records (two seeds per unit);
- 36 query-blind ablation empty preparations (12 units times three ablations,
  seed block 1), created without consulting the frozen selection manifests;
- 72 post-query C2 calls;
- 72 construction-disabled `A-FixedSelect` calls; and
- 36 C0 plus 72 C1 fixed CPU projections after preconstruction.

All 60 active-construction empty preparations, the 12 C0 constructions, and each
successful C1 preparation plus its deterministically derived same-seed
`A-FixedSelect` preparation are sealed in one persisted `PrequeryBarrier` before
the first `query.json` is opened. With successful C1 prebuilds this is exactly
120 bindings: 12 C0, 24 C1, 24 FixedSelect, 24 C2, and 36 ablation bindings.
Plan derivation reads only query-stage manifests and opaque query-artifact hashes;
it does not parse query semantics.

The call order separates C1, C2, and `A-FixedSelect` service sessions. Every
query-time comparison binds the same evidence-packet, horizon, and final-budget
hashes. Each FixedSelect request carries the complete same-unit, same-seed C1
graph and seal. A C1 failure becomes an intention-to-treat dependency failure;
it is never replaced by a fabricated graph or an unregistered GPU request.
The C0 builder receives an evidence-only `HeldOutPrequeryUnit`; no query-stage
path or hash is present in that interface. Runtime stage loading checks the exact
flat directory, typed evidence/query records, and embedded artifact hashes.

## Review gate

The production opener, `open_reviewed_held_out_plan`, first calls the full
independent-review completion verifier. Only after its 72 review decisions, nine
reviewed projections, scorer bindings, draft seal, final seal, and completion
manifest reproduce does it inspect public/model-visible held-out stage manifests.
It returns a `ReviewedHeldOutPlan` that cryptographically binds those review
records to the gold-free call manifest. The executor does not accept a bare call
manifest.

No scorer projection, alternative, reviewer note, or held-out gold identifier is
read into the runtime plan. The executor independently replays this gate rather
than accepting a caller-created `ReviewedHeldOutPlan`. Scoring is enabled only by
the hash-only bridge after the complete on-disk journal is replayed and all
success, invalid, timeout, and failure receipts are frozen.

## Injected adapters

`--run` accepts only the exact `module.path:factory` frozen in
`production_adapter_factory`; arbitrary injected factories are rejected. The
factory returns an object with `cpu`, `sessions`, and a shared audited `runtime`.
`cpu` implements deterministic C0 construction and sealed-graph CPU projection.
The runtime activates exactly three services sequentially: one allocation and
one model load for C1, then a full shutdown; one allocation/load for C2, then a
full shutdown; and one allocation/load for `A-FixedSelect`, followed by final
shutdown. Concurrent loads and one service shared across conditions are rejected.

The three sessions must have distinct process, service, allocation-event, and
model-load identities but one common global GPU-accounting scope. Each 180-second
registered service-start p95 is charged exactly once. The 300-second operational
startup watchdog accommodates the accepted fallback's observed cold-start time;
actual service allocation is charged continuously and the remaining-work gate is
recomputed rather than treating the watchdog as free time. Before and after
every started request, the runner checks
the immutable GPU inventory, monotonic allocated seconds, decreasing remaining
p95 schedule, ledger-chain binding, class-specific p95/watchdog, and one-repair
limit. A zeroed cumulative counter is rejected. A configured development result
must reproduce the initial allocated seconds and remaining mandatory forecast.
The post-call snapshot is checked against both the nine-hour schedule and the
strict pre-ten-hour stop. Shutdown receipts embed the exact transition snapshot,
so replay cannot substitute a later cumulative ledger state.

`held_out_production.py` supplies the lifecycle/CAS/query-opening control plane.
`held_out_execution.py` supplies the lifecycle-free semantic executor,
`held_out_c0.py` supplies the pinned local spaCy C0 adapter, and
`held_out_factory.py:create_frozen_production_held_out_bundle` binds them to
three resumable but distinct vLLM processes over one ledger and CAS. The
checked-in production factory is this exact implementation. The dynamic
development predecessor is materialized into a restricted binding, and
independent review is still required afterward; neither gate can route through
a fixture fallback.

After a successful fallback/development result has verified physical service
shutdown, materialize its restricted predecessor files without opening held-out
stages:

```bash
PYTHONPATH=src python scripts/finalize_held_out_runtime.py \
  --fallback-result artifacts/public/results/fallback_gpu_acceptance_development_v3.json \
  --source-association artifacts/public/manifests/source_tree_fallback_development_v5.association.json \
  --restricted-root artifacts/restricted \
  --ledger artifacts/restricted/ledger/study.sqlite3 \
  --artifact-root artifacts/restricted/cas \
  --created-at 2026-09-04T00:00:00Z
```

The explicit time above is illustrative; production uses the externally
recorded completion time. The binding stores both the source-template plan hash
and the separately reproduced post-development plan hash, plus the exact
post-shutdown GPU-event prefix and all 24 development model-call/CAS/validation
lineages. Execution replays that predecessor before loading a tokenizer or
model; a replacement ledger with the same scalar duration is rejected.
Re-execution accepts only byte-identical artifacts.

Validate the review and frozen plan without writing run artifacts:

```bash
PYTHONPATH=src python scripts/run_held_out_primary.py \
  --validate-only \
  --runtime-binding artifacts/restricted/held_out_primary/runtime_binding.json \
  --review-completion-root artifacts/restricted/scorer_only/independent_review
```

Production execution additionally requires `--run`, the frozen
`--adapter-factory`, and explicit
paths for the verified fallback snapshot/cache/manifest, selected-model freeze,
source association, cumulative ledger/CAS, private runtime root, quota root, and
the development-prequery CAS artifact hash. `--restricted-root` is mandatory;
the ledger, CAS, runtime, and output paths must all be real descendants of it,
with no symlink in the descendant chain. The entry point never searches for
these restricted inputs or silently selects a test adapter. The execution
completion time is captured only after the final service has been shut down and
its allocation has been reconciled.

For example, the private arguments include the common containment boundary:

```text
--restricted-root artifacts/restricted \
--ledger artifacts/restricted/ledger/study.sqlite3 \
--artifact-root artifacts/restricted/cas \
--runtime-root artifacts/restricted/held_out_primary/runtime \
--output-root artifacts/restricted/held_out_primary \
--results-gate-root artifacts/restricted/held_out
```

## Append-only recovery

Before each physical model load, the runtime fsyncs a path-free activation
intent binding the condition, plan, predecessor, launcher, logical service and
event IDs, watchdog, and remaining schedule. A restart adopts an exact live
lease without loading again, or treats a reconciled stale allocation as
terminal; a terminal recovery is never returned as a callable service. Before
physical shutdown it likewise fsyncs a path-free shutdown intent, allowing a
restart to reconstruct the terminal allocation and receipt.

Before calling a service, the runner fsyncs a call slot containing the exact
envelope, frozen service identity, and pre-request GPU snapshot.
If interrupted after this point, it invokes `recover_call`. A matching durable
result is reused; an unresolved semantic intent fails closed; and a slot proven
to predate any semantic intent can issue that exact call once. Recovery compares
the receipt to the original pre-request snapshot, so already-accounted GPU time
is not charged twice. ITT records retain both full schedule snapshots rather
than unverifiable hashes. Existing records must reproduce byte-for-byte.
Query openings additionally persist the exact typed query context and packet
materialization event so a controller restart never needs to reopen `query.json`.
Requests, base/repair packing reports, raw responses, validated generations,
condition outputs, C2/ablation preparations, model-call rows, and GPU-event rows are
resolved through typed CAS references that check physical bytes, logical hashes,
media type, and release class before scoring.
Symlinks, unknown paths, record drift, condition/session mismatch, and a partial
tree carrying a final execution manifest are rejected. Re-running a complete
journal performs no CPU or model calls and reproduces the same execution hash.

The output root is restricted by default:
`artifacts/restricted/held_out_primary`. It must not be included in public
bundles. After the terminal journal is replayed, the runner atomically publishes
a byte-identical execution-manifest copy, the hash-only scorer authorization,
and `primary_results_gate.json` in the separate restricted
`artifacts/restricted/held_out` namespace. Keeping this closure outside the
execution journal preserves the journal's strict no-extra-files audit. A restart
reuses the retained authorization and freeze timestamps and accepts only
byte-identical files. The current repository contains no held-out execution
outputs.
