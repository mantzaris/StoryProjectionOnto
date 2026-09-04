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

The one study geometry, style, font, viewport, and root-derived layout seed are frozen in
`configs/study/visualization.json`. The exact six-script/three-trace inventory and its
claim boundary are frozen in `configs/study/feedback.json`. Execution records are built by
`story_projection_onto.feedback_runtime`; every C2 feedback episode requires a unique
reference to its separately metered model-call ledger artifact. A configured episode may
therefore be planned without being mislabeled as an executed regeneration.

Before any condition output is inspected, each known-answer episode also requires a
`ScriptedRevisionFreeze` binding the exact shared instruction and evidence/mention-anchor
hashes. The tracked protocol selects the six script slots and three trace slots; it does
not fabricate a completed freeze, trace, model call, or result.

For a scientific trace, pass the protocol's `llm_seed` to `create_app` as
`revision_seed`. The API refuses to start an injected runner when that frozen seed is
absent or differs from the submission. The browser transports the 63-bit value as decimal
text so JavaScript cannot round it.

Runner responses include an auditable condition-call record. C2 responses must identify a
metered post-revision GPU reconstruction. C0/C1 may return a same-seal CPU reprojected view
or an explicit `capability_limited` result with no invented after view.

When a comparison is selected, current objects keep their anchor-derived coordinates and
removed or pre-merge/pre-split objects are drawn as translucent `Before:` ghosts. This is
renderer state only; filtering, pan, zoom, focus, and temporary hiding preserve the source
projection's semantic hash.
