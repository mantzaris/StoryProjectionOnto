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
`build_production_case_study_gpu_adapter` is the registered fail-closed factory. It
accepts only an explicitly loaded attested case, the existing cumulative ledger and
CAS, the selected tokenizer/model manifests, and an already configured service whose
launcher hash equals the execution plan. The owner shuts that exact service down on
success or failure and rejects an incomplete state whose sole load was already
stopped.

The repository CLI is deliberately a read-only contract/status command because the
lawful corpus and admission artifacts are not repository configuration. Its
`--execute` switch fails before model start and tells the lifecycle owner to invoke
the registered factory with those restricted objects. This is a missing run input,
not a missing semantic adapter. The read-only form validates a compiled plan,
confirms the registered factory is importable, and optionally audits a resume:

```bash
python scripts/run_case_study_controller.py \
  --restricted-root /absolute/restricted/root \
  --plan /absolute/restricted/root/case-execution-plan.json \
  --resume /absolute/restricted/root/resume/<hash>.json
```

Production admission must use the existing cumulative Phase-1 ledger (currently
`artifacts/restricted/phase1_acceptance.sqlite`) and its compatible blob root, or an
explicit cryptographically attested predecessor. A fresh SQLite file is rejected;
the case schedule reserves 2,310 base seconds and the next 240-second repair against
the global 9-hour admission and 10-hour hard stop.

No case execution is authorized until the lawful restricted corpus path, exact input
attestation, all eight pre-case gate artifacts, selected-model freeze, current source
manifest, storage preflight, and cumulative-ledger predecessor hash are supplied.
The terminal handoff contains a blank eight-unit/four-detailed-unit review template
and zero reviewer judgments.

The synthetic adapter integration test uses only artificial text under `tmp_path`.
It drives all 13 registered calls through a metered fake service, including one real
invalid-parent repair, produces all 25 ITT receipts, proves completed-run replay
without another generation, rejects an interrupted active-call reissue, and verifies
one start and one shutdown. It is software validation only and creates no scientific
case-study output.
