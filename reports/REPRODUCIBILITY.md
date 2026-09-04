# Reproducibility report

Study status: `incomplete`.

Result manifest: `ab5408139af322219e8503d1a0175b6fe48591b1113626183e98331539c7f473`.

Code revision: `e09f83eca015c194552c4e53a4138f6fa36befa2`; dirty: `true`.

The report builder verifies UTF-8/LF CSV bytes, SHA-256 values, row counts, required
columns, predecessor ingestion, phase gates, PDF presence, figure hashes, and exact
Markdown regeneration.
Scientific statistics are computed upstream; this layer never recomputes or imputes them.

## Inputs

Ingestion receipt: `5d7d86c714e599184206fbe459cb3408b216fdc39980c9ae75d3035fb541cc0e`.

- `study_status`: `d1cae85a93158406ed710d39a7b6eb69cb4eee6d69bb91b3a32b7bc16051143d`, 7 rows, status `complete`.
- `resource_accounting`: `6d655ceb4c91d2875463be129cc766ac4230036f8286215d5cfe3e09d0c03927`, 8 rows, status `complete`.
- `failure_accounting`: `1a9b8a298001a0fb4bb0548bd749a349845b7eaac3331d70ad99196e364cec92`, 2 rows, status `complete`.

## Regeneration

Run `python scripts/build_results_report.py --manifest reports/results_manifest.json`
from the repository root in the pinned study environment. Add `--verify` for a
read-only consistency check.

The public bundle is a separate allowlist-only build. Restricted novel text, detailed
offsets, FTS indexes, model weights, caches, raw prompts containing protected prose,
and private paths are never public-bundle inputs.
