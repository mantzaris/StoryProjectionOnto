# Phase 7 reporting and release boundary

The Phase 7 renderer is deliberately downstream-only. It does not discover run
outputs, calculate scientific statistics, or turn missing values into estimates.
The production compiler in `src/story_projection_onto/phase7_compiler.py` closes the
manual handoff between analysis artifacts and the renderer. Its accepted chain is:

1. A run-specific self-hashed source-registry recipe (using
   `configs/study/phase7_source_registry.template.json` only to identify the required
   inventory) names all eight predecessor families, all seven phase states, all
   fourteen table slots, and their already-frozen paths. The registry materializer
   computes physical/logical hashes and pre/post-filter CSV row counts from those
   bytes; researchers do not copy them by hand.
2. `configs/study/phase7_compiler.json` freezes table columns and order, section/table
   attachments, figure columns, public static inputs, and the reporting-policy hash.
3. `scripts/compile_phase7_results.py` verifies the registry, emits content-addressed
   canonical CSVs, applies the six frozen qualitative rules, copies at most six public
   composite PNGs, and writes content-addressed ingestion and result manifests.
4. The generated ingestion manifest registers each predecessor file by exact path,
   hash, release class, logical contract, source family, and measurement domain.
5. `scripts/compile_report_ingestion.py` can independently replay that manifest and
   emit a sanitized receipt containing hashes and statuses but no restricted path or
   copyrighted content.
6. The result manifest may expose only tables marked available in that receipt. Its
   row counts, columns, table hashes, and upstream hashes must agree exactly.
7. `scripts/build_results_report.py` renders Markdown, PDF, and figures from that
   verified document. The result/figure manifest binds every displayed artifact,
   its report section, and the compact predeclared PDF columns. Full canonical CSVs
   remain unchanged and the editable Markdown retains every column.
8. The compiler emits a content-addressed public-bundle allowlist.
   `scripts/build_public_bundle.py` replays report verification before the release scan
   and deterministic bundle build.
9. `scripts/inspect_results_pdf.py prepare` rasterizes every PDF page and performs
   hash/page/text-inventory checks. These checks explicitly remain `pending` visual
   review. A researcher or agent must actually view all deterministic representative
   pages and use the `attest` command to record the six visual criteria. A complete
   public release requires an accepted receipt; automated checks never create one.

The registry always contains all eight source families (`fallback`, `development`,
`heldout`, `ablations`, `feedback`, `case_study`, `runtime`, and `storage`) and all
fourteen registered report table slots. A missing predecessor or table is represented
as `incomplete` or `blocked`, with a reason and no table path. It is never represented
by a blank or fabricated result row. A final table is rejected unless every required
predecessor family is complete and its immutable source lineage includes every such
family. The full-report gate additionally requires all seven phases, all predecessor
families, all final tables, and all registered figures and sections.

Paper statistics are permitted only in canonical, hash-bound tables. The complete
nineteen-section prose inventory is bound to the pre-result section-contract hash,
preventing post-output numeric prose from bypassing the table lineage. Interim tables
are visibly labeled and cannot satisfy the complete-study gate.

Compact comparison rows are not the complete metric inventory. Four frozen table
contracts additionally require the public Phase 4 row sources below before the table
may be marked complete:

- `primary_c2_vs_c1` binds all primary-unit node and strict qualified-assertion rows,
  evidence citation/grounding/unsupported diagnostics, the decision macro,
  unrepresentable-decision counts, all six decision-family precision/recall/F1 rows,
  and essential temporal qualification accuracy;
- `rare_pivotal` binds rare-pivotal recall, complete support-path survival, and all
  common/rare by pivotal/nonpivotal strata;
- `entropy_clutter` binds normalized and raw entropy, supports and denominators,
  topology counts, direct geometry, the two-part crossing inputs, irrelevant visible
  load, and rare-pivotal discoverability; and
- `community` binds all base/half/double Leiden cluster, modularity, AMI, purity,
  conductance, fragmentation, merging, and omission rows, plus cross-seed AMI,
  variation of information, and per-seed omission counts.

Each Phase 4 source binding names both a canonical public CSV and, through
`producer_manifest_artifact_id`, the public Phase 4 table manifest that produced it.
The manifest and source CSV must be in the binding lineage and the public allowlist;
supplemental rows must name that same manifest. The registry materializer derives the total, filtered, and
per-metric row counts; the compiler independently rechecks the physical hash, canonical
CSV serialization, exact key and condition partitions, one metric-version hash, and the
matching table-manifest entry. The CSV and table manifest must both be table lineage and
explicit public-bundle inputs. The PDF and editable report list the complete metric
inventory and immutable source paths; the source CSV retains every numerical row,
status, numerator, and denominator. Phase 7 never fills a missing comparison or
calculates a substitute statistic.

The source registry is a binding record, not a place to type paper values. Phase 4,
feedback, case, runtime, and storage producers first emit canonical CSV artifacts.
`scripts/materialize_phase7_source_registry.py` then derives their hashes and counts
from a self-hashed recipe. `source_row_count` is the complete upstream row count and
`output_row_count` is the count after the configuration's frozen exact-value
`registered_row_filters`; neither is accepted as manual recipe input. A recipe may not
choose or weaken those filters. The compiler also checks the configured row-identity
Cartesian sets for every comparison panel, the complete 12-row paraphrase producer,
and the complete 28-row ablation producer with its 12/8/8 condition and principal-metric
partition. A mismatch blocks replay. `study_status.csv` is
generated from the seven phase records. `qualitative_examples.csv` is generated only
from a self-hashed candidate set and the frozen policy rules. All other rows are copied
and sorted, never statistically recomputed.

For a complete one of the four supplemental panels, the run-specific recipe also uses
`supplemental_metric_sources` entries with `source_role`, `source_artifact_id`, and
`table_manifest_artifact_id`. Those artifacts must already appear in
`source_artifact_ids` and `public_artifact_ids`. Counts and metric identities are never
typed into the recipe; they are derived against `configs/study/phase7_compiler.json`.

Where a report table must expose the frozen Phase 4 mechanism or rare-pivotal gate,
its binding may name `report_gate_status.csv` as a singleton join. The compiler accepts
exactly one canonical row, requires that CSV in the table lineage, rejects column
collisions, and mechanically appends its gate fields to every selected comparison row.
It never recalculates or interprets the gate.

The completed `mechanism_c2_vs_fixed` table has an exact four-row contract from the
canonical Phase 4 comparison source: gated ontology-decision macro F1, the gated
rare-pivotal safeguard, and the two `mechanism_support` diagnostics for contrastive
decision-change F1 and ontological collapse. All four compare C2 with
`A-FixedSelect`; extra, missing, or substituted metric/family identities are rejected.
The separate `contrastive_mechanism_diagnostics` supplemental source binds every raw
primary-condition contrast-pair F1 and collapse observation (including C0), preserving
the 12-world analysis boundary while keeping these diagnostics outside the primary
two-endpoint Holm family.

The candidate set carries public-safe display summaries, opaque evidence IDs, source
hashes, one fixed-anchor composite PNG per example, and exactly two intention-to-treat
C2 seed scores for eligible hard-stratum counterexamples. A failed, invalid,
interrupted, or timed-out C2 seed must be zero before the minimum-mean rule is applied.
Narrative candidates additionally attest paraphrase-only, opaque-evidence-only content
and absence of verbatim copyrighted text. The release scan remains an independent
fail-closed check.

GPU-service time and total RunPod/pod wall time are separate measurement domains.
One source artifact cannot claim both domains. An observed resource-table value must
cite an artifact registered for its domain; absent pod wall time remains an explicitly
incomplete row with no value. This prevents scientific GPU allocation time from being
silently substituted for financial session duration.

## Commands

From the repository root, using the pinned environment:

```bash
python scripts/materialize_phase7_source_registry.py \
  --recipe artifacts/restricted/phase7/source_registry.recipe.json \
  --output artifacts/restricted/phase7/source_registry.json
python scripts/materialize_phase7_source_registry.py \
  --recipe artifacts/restricted/phase7/source_registry.recipe.json \
  --output artifacts/restricted/phase7/source_registry.json --verify
python scripts/compile_phase7_results.py \
  --registry artifacts/restricted/phase7/source_registry.json
python scripts/compile_phase7_results.py \
  --registry artifacts/restricted/phase7/source_registry.json --verify
python scripts/compile_report_ingestion.py --verify
python scripts/build_results_report.py --verify
python scripts/inspect_results_pdf.py prepare
# View every representative PNG named by the emitted raster manifest before attesting.
python scripts/inspect_results_pdf.py attest \
  --raster-manifest reports/visual_inspection/report_raster_manifest.<pdf-token>.json \
  --assessment artifacts/restricted/phase7/report_visual_assessment.json
python scripts/inspect_results_pdf.py verify \
  --raster-manifest reports/visual_inspection/report_raster_manifest.<pdf-token>.json \
  --receipt reports/visual_inspection/report_visual_inspection.json --require-accepted
python scripts/build_public_bundle.py \
  --phase7-current reports/phase7_current.json \
  --phase7-registry artifacts/restricted/phase7/source_registry.json \
  --visual-raster-manifest reports/visual_inspection/report_raster_manifest.<pdf-token>.json \
  --visual-inspection-receipt reports/visual_inspection/report_visual_inspection.json \
  --restricted-root /absolute/restricted/root \
  --protected-prose-canary-manifest release/protected_prose_canaries.json \
  --bundle-root /new/empty/output/path
```

The generic and production commands resolve predecessor artifacts from the live source
tree by default. To replay only the preserved recovery-interim receipt and report after
its benchmark lineage was superseded, opt in to the bound historical snapshot:

```bash
python scripts/compile_report_ingestion.py --verify \
  --source-snapshot artifacts/public/reporting_snapshots/conference-report-ingestion-v1/source_snapshot_manifest.json
python scripts/build_results_report.py --verify \
  --ingestion-source-snapshot artifacts/public/reporting_snapshots/conference-report-ingestion-v1/source_snapshot_manifest.json
```

Snapshot substitution is verification-only for receipt ingestion; it cannot create a
fresh receipt. Final and newly compiled reports must use their live, hash-bound inputs.

The release command first byte-reproduces the complete Phase 7 compiler tree from the
restricted source registry, then authenticates the current pointer, compilation
manifest, content-addressed report/PDF/allowlist, aliases, and exact entry inventory.
A self-hashed ad hoc or stale allowlist cannot satisfy this gate. The two
protected-prose arguments are mandatory whenever the allowlist contains a
case-study or novel artifact. The canary manifest remains restricted; only its hash is
recorded in the public bundle manifest.

The generic ingestion/report commands require the content-addressed paths named in
`reports/phase7_current.json` after production compilation. That pointer is self-hashed
and names the exact immutable result manifest, report PDF, public allowlist, and
compilation manifest. Standard report filenames are atomic convenience aliases; prior
content-addressed attempts remain append-only.

To advance the report after a predecessor freezes, create a new source-registry
recipe version, materialize a new source registry, and run the production compiler.
Never edit a content-addressed table, receipt,
selection, result, or compilation manifest. Never reuse an interim table as a final
table. Source and generated ingestion registries are operational metadata and need not
be public when they contain restricted paths; the sanitized receipt is public lineage.

The current checked-in report is intentionally incomplete. It reports observed
primary-pilot failures and resource samples only; it contains no synthetic efficacy,
ablation, feedback, case-study, or final runtime/storage result.

The checked-in source-registry template is even more conservative: compiling it emits
only a seven-row status table and explicit incomplete qualitative-selection records.
It is a software smoke input and must never be relabeled as a study result.

## Final failure and resource accounting producer

`scripts/build_interim_accounting_tables.py` remains recovery-interim provenance for
the two rejected primary-model pilots, the fallback-v1 watchdog attempt, and the
fallback-v3 decoder-schema rejection. It verifies their cumulative chain and emits
incremental failure allocations that sum exactly to the recovered ledger total; it is
not a final accounting producer. The production workflow
has a materialization step precisely so no one hand-authors the 278 call bindings,
outcomes, attempts, token totals, or resource peaks. First create a restricted,
self-hashed `final_phase7_accounting_source_recipe`. This source recipe is routing
metadata only. It names:

- the registered `gpu_call_inventory.json`;
- every frozen ledger/CAS snapshot, plus its lineage ID, sequence, and parent ledger;
- native self-hashed phase call/execution/result manifests that identify actual calls,
  each with an explicit `producer_role` from the closed accounting allowlist;
- each authorized service-start amendment; and
- wall-time samples already emitted by `scripts/capture_pod_wall_time.py`.

It contains no call slots, outcomes, repairs, ITT decisions, durations, token totals,
or peaks. Do not author its hashes or freeze timestamp by hand.
`build_final_accounting_source_recipe.py` accepts only explicit paths and producer
roles, validates their native contracts and ledger/CAS evidence, derives the freeze
from the latest supplied wall-time receipt, and emits a content-addressed source
recipe. It performs no broad path discovery. Capture the final wall sample, build,
materialize, compile, and replay with:

The closed `producer_role` values are `phase1_acceptance_result`,
`development_call_manifest`, `held_out_call_manifest`,
`held_out_execution_manifest`, `combined_call_manifest`,
`feedback_execution_manifest`, `case_execution_plan`, and
`case_execution_result`. Phase 1 result routes use `manifest_sha256`; the exact typed
immutable-record routes use `content_hash`. A completed execution supplies the
schedule role for each executed call family; execution/result roles may add terminal
lineage but cannot replace that schedule.

```bash
python scripts/capture_pod_wall_time.py \
  --output /absolute/frozen/study-root/artifacts/restricted/phase7/pod_wall_time_final.json
python scripts/build_final_accounting_source_recipe.py \
  --source-root /absolute/frozen/study-root \
  --accounting-id final-study-accounting \
  --base-call-inventory configs/study/gpu_call_inventory.json \
  --native phase1_acceptance_result artifacts/public/results/phase1_gpu_acceptance_v1_failed.json \
  --native phase1_acceptance_result artifacts/public/results/phase1_gpu_acceptance_v2_failed.json \
  --native phase1_acceptance_result artifacts/public/results/fallback_gpu_acceptance_development_v3.json \
  --amendment configs/study/fallback_service_retry_amendment.json \
  --ledger phase1-current phase1 0 - artifacts/restricted/phase1_acceptance.sqlite artifacts/blobs/phase1_acceptance \
  --wall-time-receipt artifacts/restricted/phase7/pod_wall_time_final.json \
  --output-root artifacts/restricted/phase7/final-accounting-sources
python scripts/materialize_final_accounting.py \
  --source-root /absolute/frozen/study-root \
  --source-recipe artifacts/restricted/phase7/final-accounting-sources/\
final_accounting_sources.<printed-hash>.json \
  --output-root artifacts/restricted/phase7/final-accounting-materialized
python scripts/materialize_final_accounting.py \
  --source-root /absolute/frozen/study-root \
  --source-recipe artifacts/restricted/phase7/final-accounting-sources/\
final_accounting_sources.<printed-hash>.json \
  --output-root artifacts/restricted/phase7/final-accounting-materialized \
  --verify
python scripts/compile_final_accounting.py \
  --source-root /absolute/frozen/study-root \
  --recipe artifacts/restricted/phase7/final-accounting-materialized/\
final_accounting_recipe.<hash>.json \
  --output-root artifacts/public/phase7/accounting
python scripts/compile_final_accounting.py \
  --source-root /absolute/frozen/study-root \
  --recipe artifacts/restricted/phase7/final-accounting-materialized/\
final_accounting_recipe.<hash>.json \
  --output-root artifacts/public/phase7/accounting \
  --verify
```

The explicit Phase 1 routes above form the bounded recovered-ledger smoke. On the
remote cumulative ledger they must reproduce 815.215409 allocated seconds, including
service overhead. Later invocations add only explicit `--native` and `--ledger`
routes for subsequently frozen phases. The builder's JSON output supplies the exact
source-recipe filename; the materializer similarly supplies the compiler recipe
filename. Use those printed names rather than a glob, and repeat the identical
arguments under `--verify` before proceeding.

`materialize_final_accounting.py` derives a content-addressed
`final_phase7_accounting_recipe` and all normalized adapter receipts. It uses call
classes already recorded in GPU-event details, falling back to a unique adjacent
`call_class`/call-identifier pair in a named native phase manifest. Generic JSON is
not an admissible native source: each route names one recognized producer role, uses
that producer's required self-hash field, and is validated against the exact typed
development, held-out, combined, feedback, or case-study record. A legacy primary
Phase 1 result has a closed kind, hash, run identity, and gate-status contract; the
fallback Phase 1 result additionally has the exact registered four-call contract.
Every included scientific attempt must share an exact identifier with the producer
family registered for its call class; the fixed call-manifest producers expose
call-specific identifiers for this association. Thus every executed Phase 1, 3, 5,
or 6 family requires its native source.
Ambiguous, unregistered, wrong-kind, or unassociated GPU/VLLM attempts fail
materialization. The compiler repeats the native/ledger replay and requires the
derived call, exclusion, and service slots to equal the materialized recipe. All
ledger attempts are then partitioned exactly into executed
slots or explicit CPU-only/hand-authored exclusions; an exclusion cannot carry a GPU
event or VLLM call.

The generated recipe enumerates all 278 registered inference slots, the eight base
service-start slots, any additional service starts authorized by self-hashed
methodological amendments, frozen cumulative ledger snapshots and CAS inventories,
and those derived exclusions. A call slot may be unstarted, but it may not disappear:
unused reserve slots become `not_used`, while unstarted non-reserve slots remain
visible as `incomplete`. ITT membership is a fixed code policy for held-out test,
paraphrase, and registered ablation classes; it is not inferred from success.
Executed reserve slots retain `reserve_long`/`reserve_standard`/`reserve_short` as
their call class but derive the underlying C1, C2, FixedSelect, or ablation condition
from the exact event/native call identity. A missing or ambiguous reserve condition
fails materialization.

Each ledger snapshot binds its physical file SHA-256 and a deterministic inventory
hash returned by `story_projection_onto.final_accounting.cas_inventory_sha256`.
Snapshots in one `lineage_id` use contiguous sequence numbers and name the exact
previous snapshot. Compilation verifies every ledger/CAS exhaustively and proves that
every earlier append-only row survives byte-for-byte in its successor. Only the last
snapshot in each lineage contributes to totals, preventing cumulative snapshots from
being double-counted. Distinct terminal lineages must partition registered calls,
attempts, GPU events, and services exactly.

The materializer generates the following self-hashed adapter receipts. Each embeds
the materialization source-recipe hash plus exact sorted physical hashes for the
native phase sources, all ledger snapshots, the base inventory, and amendments. The
compiler independently reconstructs the same facts and rejects any adapter drift.

- `final_execution_inventory_receipt` has `call_bindings`, each with exactly
  `slot_id`, `ledger_id`, `job_id`, `attempt_id`, nullable `model_call_id`, and ordered
  `gpu_event_ids`. These are generated from ledger jobs, attempts, model calls, and
  GPU events after native call-manifest association.
- `final_result_inventory_receipt` partitions its `registered_slot_ids` into exact
  `itt_slot_ids` and `non_itt_slot_ids`, and gives one `terminal_outcomes` binding for
  every slot. Outcomes are generated from immutable model-call success, failure rows,
  and repair ancestry; unstarted slots come from the registered inventory.
- `final_repair_inventory_receipt` has `repair_bindings` containing `slot_id` and
  `parent_slot_id`; these come only from ledger `parent_attempt_id` links, including
  explicit links to an excluded hand-authored base fixture.
- `final_service_inventory_receipt` lists all `registered_service_slot_ids`, exact
  executed `service_bindings`, and `all_services_stopped`. Base versus amended
  sessions are partitioned by exact scalar equality with the hash-bound
  `authorized_recovery_run_id` (including parsed journal detail objects), not by
  chronology or substring; one session may match zero or one amendment and never two.
- One `final_token_accounting_receipt` per terminal ledger lists the sorted VLLM
  `model_call_ids` and sums prompt/completion tokens directly from those rows.
- One `final_storage_accounting_receipt` per terminal ledger lists sorted
  `storage_sample_ids`, actual/projected peaks, minimum effective headroom, and the
  all-samples-allowed result, all derived from storage/resource sample rows. Every
  terminal ledger requires storage evidence. Every resource sample requires the
  paired `resource_sample:<sample_id>` storage row emitted by the runtime sampler,
  and every allocated service interval requires an in-interval resource sample. The
  sole bounded exception is an exact, terminal, failed service-start event whose
  complete lifetime is below the registered one-second sampler cadence. Its service
  ID is enumerated as incomplete evidence, and RAM/VRAM/worker peaks become observed
  lower bounds; no zero or resource-limit compliance is inferred. Longer, successful,
  or otherwise unbound uncovered services fail materialization.
- `runpod_container_wall_time_sample` receipts are those emitted by
  `scripts/capture_pod_wall_time.py`. The compiler takes the latest cumulative sample
  per distinct PID-1 start instant and reports their sum as an operational lower bound,
  never as GPU time or provider billing time. It requires the exact producer scope and
  method, verifies elapsed time against the two timestamps, rejects samples after the
  source freeze or before terminal ledger evidence, and cannot double-count one start
  by changing a free-form scope. They are already native measurement receipts, so the
  materializer binds rather than rewrites them.
- `final_attempt_exclusion_receipt` is always generated, even when empty, and records
  every excluded `ledger_id`, `attempt_id`, and permitted reason.

Every input also has a physical SHA-256 in the recipe. Paths must remain below a real,
symlink-free source root; ledger, receipt, and CAS files must be singly linked regular
files, and ledgers must be closed and checkpointed. The compiler derives outcomes,
failure classes, repair success, token totals, peaks, and allocation from the ledger;
the recipe cannot assert those results. It partitions every GPU event and service
session exactly once and requires the failure-table allocation sum to equal cumulative
`gpu_summary` totals, including `service_overhead` (service lifetime minus classified
events). If an allocated service outside the exact pre-sampler exception lacks
resource sampling, a resource sample lacks its paired storage row, or any terminal
ledger lacks storage evidence, compilation fails rather than reporting unsupported
RAM/VRAM/worker peaks or storage compliance. The source recipe must not predate the
terminal scientific ledger evidence.

Outputs are immutable `failure_accounting.<hash>.csv`,
`resource_accounting.<hash>.csv`, and a self-hashed
`final_accounting_receipt.<hash>.json`. The receipt binds both table hashes, exact
columns and row counts, snapshot row hashes, amendments, total allocation, overhead,
ITT cardinality, incomplete mandatory calls, wall-time lower bound, and final stopped
service state. Register those content-addressed CSVs and the receipt as the runtime and
storage predecessors; do not copy them to the old interim filenames. `--verify`
re-derives or recompiles everything in memory, requires byte-identical existing
outputs, and writes nothing. The builder, materializer, and compiler `--verify`
commands are all required before Phase 7 source-registry materialization.
