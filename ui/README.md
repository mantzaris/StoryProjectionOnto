# Local graph interface

This directory contains the bounded, single-user Phase 5 research interface. It is not a
hosted service and makes no usability claim.

The repository vendors the exact Cytoscape.js 3.30.4 minified asset. Its npm tarball
bytes were checked against an independently fetched unpkg copy; `cytoscape.lock.json`
records the resulting SHA-256 and byte count. To reproduce the installation, independently
obtain the SHA-256 of the exact `dist/cytoscape.min.js` asset, then run:

```bash
./ui/vendor_cytoscape.sh EXPECTED_64_CHARACTER_SHA256
```

The bounded script downloads only the pinned npm tarball over HTTPS, extracts the one
minified asset, enforces a 2 MB asset ceiling, verifies the supplied checksum before
installation, and writes `cytoscape.lock.json`. The API verifies the installed bytes
against that lock before reporting the renderer as verified.

Application code should instantiate `LocalUiRepository`, add immutable
`VisualizationBundle` records, inject the real metered revision runner into `create_app`,
and serve the returned FastAPI app locally. If no runner is injected, revision submission
returns HTTP 503 and never claims a regeneration.
