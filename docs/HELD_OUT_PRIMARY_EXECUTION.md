# Held-out primary execution control plane

This control plane is intentionally inert without three external inputs: a fully
materialized independent-review completion, a frozen passing development result,
and the tracked production live-service adapter named by the control
configuration. It cannot start vLLM or synthesize scientific outputs. The tracked
configuration currently marks the latter two items `PENDING`, so scientific
execution is mechanically blocked rather than routed through a test adapter.

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

All 60 active-construction empty preparations, the 12 C0 constructions, and the
24 completed C1 preparations are sealed in one persisted `PrequeryBarrier`
before the first `query.json` is opened. Plan derivation reads only query-stage
manifests and opaque query-artifact hashes; it does not parse query semantics.

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
registered service-start reservation is charged exactly once. Before and after
every started request, the runner checks
the immutable GPU inventory, monotonic allocated seconds, decreasing remaining
p95 schedule, ledger-chain binding, class-specific p95/watchdog, and one-repair
limit. A zeroed cumulative counter is rejected. A configured development result
must reproduce the initial allocated seconds and remaining mandatory forecast.
The post-call snapshot is checked against both the nine-hour schedule and the
strict pre-ten-hour stop. Shutdown receipts embed the exact transition snapshot,
so replay cannot substitute a later cumulative ledger state.

`held_out_production.py` supplies the lifecycle/CAS/query-opening factory. It
requires a narrow `HeldOutSemanticExecutor` and three condition activators from
the selected vLLM integration. The checked-in configuration remains `PENDING`
until that exact executable binding and passing development predecessor are
frozen; this is an intentional launch gate, not a fixture fallback.

Validate the review and frozen plan without writing run artifacts:

```bash
PYTHONPATH=src python scripts/run_held_out_primary.py \
  --validate-only \
  --review-completion-root artifacts/restricted/scorer_only/independent_review
```

Production execution additionally requires `--run`, `--adapter-factory`, and an
aware, externally recorded `--completed-at`. Do not use a test adapter for a
scientific run.

## Append-only recovery

Before calling a service, the runner fsyncs a call slot containing the exact
envelope, frozen service identity, and pre-request GPU snapshot.
If interrupted after this point, it invokes `recover_call`; absence of a durable
service receipt fails closed. Recovery compares the receipt to the original
pre-request snapshot, so already-accounted GPU time is not charged twice and the
request is never resent. ITT records retain both full schedule snapshots rather
than unverifiable hashes. Existing records must reproduce byte-for-byte.
Requests, packing reports, raw responses, validated generations, condition
outputs, C2/ablation preparations, model-call rows, and GPU-event rows are
resolved through typed CAS references that check physical bytes, logical hashes,
media type, and release class before scoring.
Symlinks, unknown paths, record drift, condition/session mismatch, and a partial
tree carrying a final execution manifest are rejected. Re-running a complete
journal performs no CPU or model calls and reproduces the same execution hash.

The output root is restricted by default:
`artifacts/restricted/held_out_primary`. It must not be included in public
bundles. The current repository contains no held-out execution outputs.
