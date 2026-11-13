# Phase 5 execution

`phase5_execution.py` is the production bridge between the frozen feedback
contracts and actual held-out outputs. It does not create ontology semantics and
cannot run until independent review and the real primary held-out execution are
reproduced. The small primary-results wrapper is not an authority by itself: it
must byte- and logically bind a typed `HeldOutExecutionManifest` and its typed
`ScorerBridgeAuthorization`, which in turn must bind the final reviewed seal and
the complete intention-to-treat held-out inventory.

The input manifest contains exactly the six frozen scripted contexts and three
researcher trace slots from `configs/study/feedback.json`. Each script carries one
condition-independent `RevisionInstruction`, a pre-output freeze, the identical
packet, C0/C1/C2 parent projections, and three restricted scorer bindings. Trace
inputs are C2-only and contain no gold binding.

C0 and C1 have only two legal outcomes: deterministic reprojection of the same
construction seal, or `capability_limited`. Each produces a zero-GPU CPU-call
receipt. Every C2 episode—six scripts plus three traces—is one unique call slot
sent through the explicitly injected adapter. The adapter must return a ledger
bound receipt for success, invalid output, timeout, or failure. The orchestrator
does not catch an unrecorded adapter exception and invent a receipt.

## Semantic assessment and display scopes

Runtime validation is structural, not an evidence-support judgment. Every runtime
`ValidationRecord` therefore carries the explicit
`runtime_structural_only_not_assessed` scope and all three semantic status fields are
`not_applicable`; this combination means *not assessed*, not a favorable N/A verdict.
A post-hoc record must instead declare `posthoc_scorer_or_reviewer` and assess at least
one semantic dimension. The two scopes cannot be interchanged.

`build_visualization_bundle` defaults to `full_structural` plus the persistent
`pending_scorer_or_reviewer` disclosure so raw projection output remains inspectable.
Phase 5 execution paths request `registered_display` explicitly, applying the frozen
common display budgets and endpoint-closed visibility rule. A supported-only view is
available only when an immutable `VisualizationSemanticOverlay` is supplied explicitly
after construction. The overlay binds the exact projection, snapshot, evidence packet,
query context, and source scorer/reviewer artifact hashes and must assess every emitted
assertion with its exact content hash. Assertion support and description support are
separate. Unsupported assertions are omitted in supported-only mode; unverified node
descriptions and `why_matters` prose are replaced by fixed nonfactual disclosure text,
and only verified in-packet evidence from the assertion's own description lineage may
be shown as description support. Overlay records contain no gold answer and are never
serialized into model-visible requests.

## Production materialization

`phase5_production.py` closes the held-out-to-feedback boundary without assuming
that a bare held-out output hash is itself an `OntologyProjection`. The closed
gold-free runtime emits a `Phase5HeldOutProjectionExport` that binds its real
projection receipt and/or ITT record, model-call and GPU-event row hashes, and
separate restricted-CAS references for the validated projection, exact packet,
and sealed snapshot. C1 and C2 exports are fixed to seed block 1. The materializer
replays independent review, the primary gate, held-out execution, scorer bridge,
and held-out call manifest before resolving any episode.

The six scripted revisions must have an immutable pre-output
`ScriptedRevisionFreeze`. Actual CPU resolutions are supplied as typed,
restricted-CAS `CpuReprojectionInput` records and are checked against the exact C0
and C1 source projections. The three researcher traces contain their real UI
request receipt, replayable instruction, and C2 source. The capture-only real
`/api/revisions` endpoint binds the canonical request and instruction bytes,
before bundle, projection, action, episode, and honest timestamps. A separate post-freeze
`Phase5ScorerBindingAuthorization` exposes only 18 opaque known-answer hashes;
gold projections, target changes, and scorer artifacts are never opened by the
materializer or adapter.

Materialize and atomically retain the input plus its lineage receipt:

```bash
python scripts/materialize_phase5_inputs.py \
  --source-manifest /restricted/phase5/source_manifest.json \
  --primary-results-gate /restricted/held_out/primary_results_gate.json \
  --ledger /restricted/study.sqlite3 \
  --artifact-root /restricted/cas \
  --output-root /restricted/phase5/inputs
```

The source manifest contains references, not manually transcribed packets,
projections, timestamps, horizons, seeds, or model-call claims. Its records are
created by `persist_phase5_record` from the actual upstream objects. Materializing
does not call a model and records both `scorer_gold_read=false` and
`model_service_called=false`. Rerunning is byte-exact; missing, changed, extra, or
symlinked files fail closed.

## Production invocation

First validate all inputs and upstream gates without writing:

```bash
python scripts/run_phase5_feedback.py \
  --inputs /restricted/phase5/input_manifest.json \
  --materialization-receipt /restricted/phase5/materialization_receipt.json \
  --primary-results-gate /restricted/held_out/primary_results_gate.json \
  --validate-only
```

Execution additionally requires an aware, recorded completion timestamp and an
adapter factory. The factory is a zero-argument callable returning an object with
`regenerate(C2RegenerationRequest) -> C2RegenerationResult`:

```bash
python scripts/run_phase5_feedback.py \
  --inputs /restricted/phase5/input_manifest.json \
  --materialization-receipt /restricted/phase5/materialization_receipt.json \
  --primary-results-gate /restricted/held_out/primary_results_gate.json \
  --adapter-factory project_adapter:create_phase5_adapter \
  --ledger /restricted/study.sqlite3 \
  --completed-at 2026-09-04T12:00:00Z \
  --run
```

The default journal is `artifacts/restricted/phase5_feedback`. Input, freezes,
call slots, recovered ITT results, condition records, the
`FeedbackStudyExecutionManifest`, and completion index are
canonical and append-only. Resume skips byte- and hash-identical records; changed,
extra, partial-tampered, or symlinked records fail closed. The final inventory is
18 known-answer executions (C0/C1/C2 for each script), three C2-only traces, 12 CPU
receipts, nine unique C2 regeneration slots, and 18 scorer-only metric bindings.
Each slot has one base GPU ledger row and may have one explicitly linked bounded
repair row, so the completion index reports physical GPU requests and allocated
seconds separately from the nine scientific slots. Researcher trace gold fields
remain `NA`.

Each C2 request is durably appended and fsynced under `call_slots/` before the
adapter is entered. A pre-existing slot is never sent to `regenerate` again:
`recover` performs a side-effect-free ledger/CAS lookup and an absent result raises
`InterruptedFeedbackCallRecoveryRequired`. Every recovered or new result is stored
under `results/` before its condition record. The production CLI independently
reopens the supplied SQLite ledger read-only and verifies every model-call row,
GPU-event row, request hash, attempt, role, retry class, timestamps, outcome, and
allocated duration. The final index also binds the byte hash and logical hash of
all 47 journal artifacts.

`MeteredPhase5C2RegenerationAdapter` is the concrete boundary around an
`AlreadyOwnedPhase5C2Service`. The lease deliberately exposes only identity,
execution, and side-effect-free recovery—objects with start/load/shutdown methods
are rejected. The factory verifies the already-recorded model-load event and
nonempty cumulative GPU counter. For every regeneration it scans the model-visible
payload, checks hard-stop capacity, and reconstructs the receipt independently
from the append-only attempt, model-call, GPU-event, and CAS rows. A recovered slot
calls only `recover_feedback(request_hash)` and never sends another inference.
The external combined-block controller remains responsible for owning vLLM and
for providing the already-running service lease; Phase 5 never starts or stops it.

## Canonical report table

After the scorer-only 18-record bundle and receipt have been materialized, compile
the public-safe Phase 7 feedback source directly from the immutable execution and
scoring roots:

```bash
python scripts/materialize_phase5_feedback_table.py \
  --feedback-journal-root artifacts/restricted/phase5_feedback \
  --scoring-root artifacts/restricted/phase5_feedback_scoring \
  --output-root artifacts/public/phase5_feedback_report \
  --materialize
```

The output contains `feedback.csv` and a self-hashed
`feedback_table_receipt.json`. The compiler verifies all 18 scripted
condition-score records and derives the three C2 trace scores from their graph
diffs, latency, resolution, and replay records. It groups only by episode class,
condition, and action, retains explicit defined counts and rate denominators, and
keeps every trace gold field `NA`. It can be replayed without writing:

```bash
python scripts/materialize_phase5_feedback_table.py \
  --feedback-journal-root artifacts/restricted/phase5_feedback \
  --scoring-root artifacts/restricted/phase5_feedback_scoring \
  --output-root artifacts/public/phase5_feedback_report \
  --verify
```

Both files should be registered as Phase 5 predecessor artifacts for the Phase 7
`feedback` table. Set `source_table_artifact_id` to `feedback.csv` and
`source_receipt_artifact_id` to the public receipt artifact; give the receipt a
`content_hash` / `canonical_without_field` logical-hash contract and include both
artifact IDs in the table lineage and public allowlist. Phase 7 rejects a complete
feedback table unless the typed receipt, physical CSV hash, logical row hash,
columns, exact 18+3 inventory, and row count all agree. No episode text, rationale,
evidence, or projection content is copied into the public table.

The study is an implementation demonstration. It supports no participant or
usability claim.
