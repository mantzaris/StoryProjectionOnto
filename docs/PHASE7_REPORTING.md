# Phase 7 reporting and release boundary

The Phase 7 renderer is deliberately downstream-only. It does not discover run
outputs, calculate scientific statistics, or turn missing values into estimates.
Its accepted input chain is:

1. `configs/study/report_ingestion.json` registers each predecessor file by exact
   relative path, physical SHA-256, release class, logical hash where available,
   required JSON fields, source family, and measurement domain.
2. `scripts/compile_report_ingestion.py` replays that registry against the source
   tree and canonical CSV directory. It emits a sanitized receipt containing hashes
   and statuses, but no restricted path or copyrighted content.
3. `reports/results_manifest.json` may expose only tables marked available in that
   receipt. Its row counts, columns, table hashes, and upstream hashes must agree
   exactly with the receipt.
4. `scripts/build_results_report.py` renders Markdown, PDF, and figures from that
   verified in-memory document. `reports/result_figure_manifest.json` binds every
   displayed table, report, and figure to its immutable source hashes.
5. `scripts/build_public_bundle.py` replays ingestion and report verification before
   running the allowlist-only public-release scan and deterministic bundle build.

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

GPU-service time and total RunPod/pod wall time are separate measurement domains.
One source artifact cannot claim both domains. An observed resource-table value must
cite an artifact registered for its domain; absent pod wall time remains an explicitly
incomplete row with no value. This prevents scientific GPU allocation time from being
silently substituted for financial session duration.

## Commands

From the repository root, using the pinned environment:

```bash
python scripts/compile_report_ingestion.py --verify
python scripts/build_results_report.py --verify
python scripts/build_public_bundle.py --bundle-root /new/empty/output/path
```

To advance the report after a predecessor freezes, add its exact immutable artifact
and table hashes to a new ingestion-manifest version, compile a new append-only receipt,
then update the result manifest. Never overwrite a scientific receipt or reuse an
interim table as a final table. The source ingestion registry is operational metadata
and need not be public when it contains a restricted path; the sanitized receipt is
the public lineage object.

The current checked-in report is intentionally incomplete. It reports observed
primary-pilot failures and resource samples only; it contains no synthetic efficacy,
ablation, feedback, case-study, or final runtime/storage result.
