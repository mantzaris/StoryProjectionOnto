# Reproducibility report

Study status: `incomplete`.

Result manifest: `157a5261a5847af75a5831e96fc97d62eb87a35ef5b457075dff409d168411e7`.

Code revision: `fb3399690fe121e9d314f53e40337cdec415782c`; dirty: `true`.

The report builder verifies UTF-8/LF CSV bytes, SHA-256 values, row counts, required
columns, predecessor ingestion, phase gates, PDF presence, figure hashes, and exact
Markdown regeneration.
Scientific statistics are computed upstream; this layer never recomputes or imputes them.

## Inputs

Ingestion receipt: `dd64d597f3497c2b9ad7fc60205e90ad5540716e586f885b7aa0aada90b245fd`.

- `study_status`: `d1cae85a93158406ed710d39a7b6eb69cb4eee6d69bb91b3a32b7bc16051143d`, 7 rows, status `complete`.
- `resource_accounting`: `228538ad967640a7d4919d95b3c86e4a43ca6ae167b4fcfb5eb1dc72eb004e7b`, 8 rows, status `complete`.
- `failure_accounting`: `b268d277b6e992a4f50b68d718f7db05e5b22a084bb0ca25a16e8342dc2d45ef`, 5 rows, status `complete`.

## Regeneration

Run `python scripts/build_results_report.py --manifest reports/results_manifest.json`
from the repository root in the pinned study environment. Add `--verify` for a
read-only consistency check.

The public bundle is a separate allowlist-only build. Restricted novel text, detailed
offsets, FTS indexes, model weights, caches, raw prompts containing protected prose,
and private paths are never public-bundle inputs.
