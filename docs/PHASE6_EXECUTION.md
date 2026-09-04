# Phase 6 production execution boundary

The Phase 6 controller is implemented in
`story_projection_onto.case_study_execution`. It keeps the lawful source, index,
queries, packets, raw responses, and projections restricted. Novel-bearing semantic
and rendered model requests are held only in memory; durable call receipts contain
their exact hashes plus prose-free packing and alias metadata. Synthetic
tests construct an artificial four-chapter work under `tmp_path`; they never read
or reproduce the target novel.

The controller enforces the registered 4/8/1 GPU inventory: four query-blind C1
preconstructions, eight bounded post-query C2 constructions, and one separately
labeled post-access full-index FTS/BM25 C2 demonstration. It also schedules four
C0 preparations, four bounded empty inventories, one operational empty inventory,
seals all thirteen preparations, and then records exactly 25 ITT outputs. Bounded
packets exist in memory, with prose-free durable receipts. Query-access and packet
materialization events are persisted to restricted CAS and SQLite before a query is
returned to a condition. The operational provider is invoked only after its access
event is committed.

`ProductionCaseClassicalAdapter` loads the pinned spaCy C0 implementation and fails
closed if the installed package/model or frozen rules differ. `preflight_case_c1_requests`
uses the accepted lossless wire codec and selected tokenizer to pack all four C1
requests before service start; no truncation is allowed. C2 requests remain dynamic
and must be packed by the GPU adapter after audited query access.

The controller owns exactly one case-study model load. The concrete
`ProductionCaseStudyGpuAdapter` implements `CaseStudyLifecycleGpuAdapter` with
complete lossless C1/C2 request packing, guided JSON inference, SQLite/CAS attempt
lineage, deterministic boundary validation, at most one fact-free repair, physical
PID/start-tick identity, append-only restart state, and an explicit shutdown receipt.
`build_production_case_study_gpu_adapter` is the semantic adapter factory.
`story_projection_onto.case_study_factory:create_frozen_production_case_study_bundle`
is the registered outer factory. It accepts only explicitly named restricted inputs,
the existing cumulative ledger/CAS, the one verified fallback snapshot/cache, and a
current local/remote source association. It verifies the launcher and tokenizer
against the execution plan, runs storage and global GPU admission, constructs the
credible C0 adapter, and creates (but does not start) the sole model service. The
controller commits a path-free activation intent before starting that service, and
starts only after all four C1 requests pack losslessly. It uses a 300-second
operational startup watchdog while retaining the registered load/call inventory and
remaining-work forecast. Recovery adopts the exact checkpointed PID/start-ticks
identity, or the exact live service lease if power failed before the checkpoint. A
path-free shutdown intent is durable before physical termination; the terminal
service journal can reconstruct the shutdown receipt after another loss. A terminal
sole load never authorizes a replacement load.

The repository CLI has a read-only validation/status mode and an explicit production
mode. No path is discovered from the host. `--execute` requires every lawful input,
gate bundle, cumulative-ledger path/hash, model-cache input, and runtime directory;
missing inputs fail before model construction. Its stdout contains only counts,
hashes, outcomes, and accounting totals. Paths, queries, packets, raw responses, and
review material remain restricted. The read-only form validates a compiled plan,
confirms the frozen outer factory is importable, and optionally audits a resume:

```bash
python scripts/run_case_study_controller.py \
  --validate-only \
  --restricted-root /absolute/restricted/root \
  --plan /absolute/restricted/root/case-execution-plan.json \
  --resume /absolute/restricted/root/resume/<hash>.json
```

After all eight pre-case gate records and the admission attestation exist, stage
canonical restricted CAS copies and a typed bundle/reference without opening a
model service:

```bash
python scripts/prepare_case_study_run.py stage-admission-evidence \
  --restricted-root /absolute/restricted/root \
  --admission-attestation /absolute/restricted/root/admission.json \
  --ledger /absolute/restricted/root/study.sqlite \
  --artifact-root /absolute/restricted/root/blobs/study \
  --synthetic-run-closure /absolute/gates/synthetic-run-closure.json \
  --timing-lineage-audit /absolute/gates/timing-lineage.json \
  --gold-firewall-audit /absolute/gates/gold-firewall.json \
  --registered-metric-regeneration /absolute/gates/metric-regeneration.json \
  --blinded-error-review /absolute/gates/blinded-error-review.json \
  --storage-preflight /absolute/gates/storage-preflight.json \
  --gpu-schedule-admission /absolute/gates/gpu-schedule.json \
  --public-release-scan /absolute/gates/release-scan.json \
  --bundle-output /absolute/restricted/root/admission-evidence.json \
  --reference-output /absolute/restricted/root/admission-evidence-reference.json \
  --staged-at 2026-09-04T00:00:00Z
```

The production invocation adds `--execute` plus the explicit index, index manifest,
preregistration, two input/admission attestations, selected-model freeze, admission
bundle/reference, cumulative ledger and expected SHA-256, CAS/runtime/quota roots,
verified snapshot manifest, snapshot/shared cache, source association/revision, and
optional loopback port. Run it inside the named persistent `storyprojection-study`
tmux session with timestamped stdout/stderr. Repeating the same command after a loss
between completed calls adopts the exact still-live PID from the private checkpoint;
it never starts a replacement load. A completed replay returns the same terminal
result hash and performs no generation. If interruption occurs after an individual
call is marked active, the adapter checks its exact terminal GPU event, model-call
row, restricted raw-response artifact, and validation lineage. It reconstructs the
receipt when possible, or records the terminal attempt as an ITT failure when the
response was not durably linked. It never repeats a call with an existing terminal
GPU event.

Production admission requires the existing cumulative study ledger, compatible CAS,
and runtime directory to be stable real paths beneath the explicit restricted root.
Every extant ancestor is checked for symbolic links. A fresh SQLite file is rejected.
Admission reserves the 300-second service-start watchdog, 2,310 base
call seconds, and the next 240-second repair against the global nine-hour schedule
and strict ten-hour hard stop. A path-free bootstrap intent makes admission creation
replayable if power fails between its CAS writes and final reference pointer.

No case execution is authorized until the lawful restricted corpus path, exact input
attestation, all eight pre-case gate artifacts, selected-model freeze, current source
manifest, storage preflight, and cumulative-ledger predecessor hash are supplied.
The terminal handoff contains a blank eight-unit/four-detailed-unit review template
and zero reviewer judgments.

Controller, adapter, resume, and GPU-state records are first published immutably
with no-replace semantics, file and parent-directory fsync, and CAS hashes. Mutable
`current.json` files are convenience pointers backed by immutable pointer histories;
a missing or torn pointer is reconstructed from the unique latest valid history
entry. CLI exception text is generic because validation errors can include restricted
input values; detailed diagnostics stay in restricted artifacts.

The synthetic adapter integration test uses only artificial text under `tmp_path`.
It drives all 13 registered calls through a metered fake service, including one real
invalid-parent repair, produces all 25 ITT receipts, proves completed-run replay
without another generation, reconstructs an interrupted terminal call without a
second inference, exercises activation and shutdown crash boundaries, and verifies
one start and one shutdown. It is software validation only and creates no scientific
case-study output.
