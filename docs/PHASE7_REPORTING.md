# Phase 7 reporting and release boundary

The Phase 7 renderer is deliberately downstream-only. It does not discover run
outputs, calculate scientific statistics, or turn missing values into estimates.
The production compiler in `src/story_projection_onto/phase7_compiler.py` closes the
manual handoff between analysis artifacts and the renderer. Its accepted chain is:

1. A run-specific self-hashed source registry (using
   `configs/study/phase7_source_registry.template.json` only as a structural template)
   registers all eight predecessor families, all seven phase states, and all fourteen
   table slots. Every available source has a bounded relative path, physical SHA-256,
   release class, and logical hash where one exists.
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
   verified document. The result/figure manifest binds every displayed artifact.
8. The compiler emits a content-addressed public-bundle allowlist.
   `scripts/build_public_bundle.py` replays report verification before the release scan
   and deterministic bundle build.

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

The source registry is a binding record, not a place to type paper values. Phase 4,
feedback, case, runtime, and storage producers first emit canonical CSV artifacts;
their hashes and row counts are then registered. `source_row_count` is the complete
upstream row count and `output_row_count` is the count after any declarative exact-value
`row_filters`. A mismatch blocks compilation. `study_status.csv` is generated from the
seven phase records. `qualitative_examples.csv` is generated only from a self-hashed
candidate set and the frozen policy rules. All other rows are copied and sorted, never
statistically recomputed.

Where a report table must expose the frozen Phase 4 mechanism or rare-pivotal gate,
its binding may name `report_gate_status.csv` as a singleton join. The compiler accepts
exactly one canonical row, requires that CSV in the table lineage, rejects column
collisions, and mechanically appends its gate fields to every selected comparison row.
It never recalculates or interprets the gate.

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
python scripts/compile_phase7_results.py \
  --registry artifacts/restricted/phase7/source_registry.json
python scripts/compile_phase7_results.py \
  --registry artifacts/restricted/phase7/source_registry.json --verify
python scripts/compile_report_ingestion.py --verify
python scripts/build_results_report.py --verify
python scripts/build_public_bundle.py --bundle-root /new/empty/output/path
```

The generic ingestion/report commands require the content-addressed paths named in
`reports/phase7_current.json` after production compilation. That pointer is self-hashed
and names the exact immutable result manifest, report PDF, public allowlist, and
compilation manifest. Standard report filenames are atomic convenience aliases; prior
content-addressed attempts remain append-only.

To advance the report after a predecessor freezes, create a new source-registry
version and run the production compiler. Never edit a content-addressed table, receipt,
selection, result, or compilation manifest. Never reuse an interim table as a final
table. Source and generated ingestion registries are operational metadata and need not
be public when they contain restricted paths; the sanitized receipt is public lineage.

The current checked-in report is intentionally incomplete. It reports observed
primary-pilot failures and resource samples only; it contains no synthetic efficacy,
ablation, feedback, case-study, or final runtime/storage result.

The checked-in source-registry template is even more conservative: compiling it emits
only a seven-row status table and explicit incomplete qualitative-selection records.
It is a software smoke input and must never be relabeled as a study result.
