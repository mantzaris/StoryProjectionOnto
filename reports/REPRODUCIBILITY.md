# Reproducibility report

Study status: `incomplete`.

Result manifest: `016232981064343020e0335a399d43744ba64f4f5c723978dc931f4326f2f255`.

Code revision: `cadfb25c39e4f0b1998ab15ad89dcc5e97d1fbf1`; dirty: `true`.

The report builder verifies UTF-8/LF CSV bytes, SHA-256 values, row counts, required
columns, predecessor ingestion, phase gates, PDF presence, figure hashes, and exact
Markdown regeneration.
Scientific statistics are computed upstream; this layer never recomputes or imputes them.

## Inputs

Ingestion receipt: `004b2e7ace7c75d1fcabd3f59a7f5686e1f088c4038db1f07ef6a5fc6a25bfef`.

- `study_status`: `d1cae85a93158406ed710d39a7b6eb69cb4eee6d69bb91b3a32b7bc16051143d`, 7 rows, status `complete`.
- `resource_accounting`: `228538ad967640a7d4919d95b3c86e4a43ca6ae167b4fcfb5eb1dc72eb004e7b`, 8 rows, status `complete`.
- `failure_accounting`: `339b205aa3ea22d6bca593b76fb0bd9ecf4a2641961b902d9b1ccee4452ef75a`, 4 rows, status `complete`.

## Regeneration

Run `python scripts/build_results_report.py --manifest reports/results_manifest.json`
from the repository root in the pinned study environment. Add `--verify` for a
read-only consistency check.

The public bundle is a separate allowlist-only build. Restricted novel text, detailed
offsets, FTS indexes, model weights, caches, raw prompts containing protected prose,
and private paths are never public-bundle inputs.
