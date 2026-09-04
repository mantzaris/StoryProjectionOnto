# Run status

Updated: 2026-09-04 (fallback v3 terminal failure reconciled)

## Verified durable state

- Branch: `implementation/query-dependent-temporal-ontology`
- Last pre-recovery commit: `568c61af0b29fe99f1c8dca513d88b4d0b724263`
- Recovery checkpoint commit:
  `8a77a3aafa0acc834a6563b0015b5fdff29151f8`.
- The independently generated local and RunPod 288-file manifests are
  byte-identical (file SHA-256
  `47d249905194deca92eae9372e0392ee74d209323179be45dee825092f4fed0c`)
  and bind source-tree SHA-256
  `81edb1ba173458970e93b903efa27981a96fb3008a8b4ec221f5810f4c515b20`.
  Source-association manifest SHA-256:
  `1ce70d1c0796e5cc085d55ec535594b5b1e4d913eff3314b200562c5c095d4b4`.
- Remote source snapshot at commit `6acf3077f2a33ebc1dc28cbc431034b9aafdfced`
  was independently hashed and is byte-identical to the corresponding local Git
  tree. Unique later remote manifests, ledgers, CAS blobs, checkpoints, failure
  logs, and test logs were recovered without overwriting the local worktree.
- The remote GPU is an NVIDIA RTX 4090 with 24,564 MiB reported memory. No
  model server, study runner, download, `tmux`, or `screen` job survived the
  interruption; vLLM is stopped.
- The cumulative GPU ledger is internally consistent and fully closed at
  422.961986 allocated seconds across three failed startup sessions. The
  recoverable unattended-allocation uncertainty is zero seconds.
- A refreshed 2026-09-04 22:01 UTC remote audit found the RTX 4090 idle at
  1 MiB/0%, no vLLM or study runner, no `tmux`/`screen` session, and no listener
  on port 8000. The live project directory occupied 9,959,306,622 bytes; vLLM
  remains stopped.
- One stale Python 2.7 bytecode file found outside the reconciled source
  inventory was moved, without deletion, into the restricted recovery
  quarantine before the final RunPod manifest was generated.
- The validation-only v3 fallback controller preflight passed without loading
  the model or allocating the GPU. Its manifest SHA-256 is
  `121d042d0dc6647011454b6e4bb8a5f7d1526290c7cefff03dbb2cca9d996e6b`
  (file SHA-256
  `cf55e1ef833381cdc2d39d8f970ebe6ec42eae5f362b9a967109c909437c5eb4`).
- The sole authorized v3 retry ran from 23:01:36Z through 23:09:49Z. Model
  startup, controller restart, and live-service adoption succeeded, but the
  first `fallback-c1-01` request was rejected before generation with a
  `RuntimeTransportError`. No output was accepted and no development call ran.
  The service lease is `stopped_verified`; the GPU is idle and port 8000 is
  closed. Incident manifest:
  `2e33bcca745dfd5e85b02e5f0f1039444cb082cd7bb88ed252677414f83e249a`.
- Current project-controlled remote occupancy at recovery was 15,135,812,608
  bytes. The maximum persisted ledger sample was 13,623,907,840 bytes; the
  difference is retained recovery/source material rather than a larger study.

## Phase state

| Phase | State | Last verified evidence |
|---|---|---|
| 1 — contracts and GPU acceptance | Blocked before another allocation | Contracts, storage/GPU controls, ledger/CAS, schemas, primary rejection, pinned fallback, and the consumed v3 retry exist. v3 loaded successfully but its first decoder schema was rejected before generation; no further service start is currently authorized. |
| 2 — synthetic benchmark | Blocked only at external review | Four development worlds, 12 held-out worlds, 36 held-out contexts, mutation checks, sealed stages, and the unchanged condition-blind three-world/nine-projection review package exist and hash-reproduce. The source-bound lineage was safely resealed after runtime-boundary hardening; no scientific payload changed. Independent reviewer decisions and adjudication are absent. |
| 3 — conditions and primary run | In progress | C0/C1/C2/FixedSelect implementations and production control planes exist. The 24 development calls and all 168 held-out calls remain unexecuted. |
| 4 — metrics and ablations | Software implemented; execution pending | Registered metric/statistical primitives exist. The 28 ablation calls and 12 paraphrase calls remain unexecuted. |
| 5 — interface and feedback | Software implemented; execution pending | The minimal Cytoscape interface and revision contracts exist. Six scripted revisions and three researcher traces remain unexecuted. |
| 6 — one-novel case study | Input pending | Restricted indexing/runtime software exists. No lawful novel path has been supplied and no narrative inference has run. |
| 7 — results and release | Interim only | The report, table, figure, and public-release validation substrate exists; current outputs explicitly report an incomplete study and contain no fabricated efficacy result. |

## Validation state

- Definitive reconciled-tree gate: Ruff clean; 734 tests passed and one expected
  system-browser wrapper skipped in 313.01 seconds on Python 3.12. The skipped
  wrapper was exercised separately against real Chrome and passed rendering,
  spoiler filtering, evidence inspection, `REFINE_CONTEXT`, and
  `REQUEST_MERGE_SPLIT`.
- Twenty-two JSON Schemas reproduce byte-for-byte; schema manifest
  `6e2e91d574ec3414409083dac0e8923f9e6b991ba34ce9cac7bb249c5a7f3176`.
- Synthetic benchmark verification, the full benchmark-to-report lineage,
  report replay, and a fresh 29-entry public-bundle build pass. The benchmark
  manifest is `22bd5e47245938724fc25d289e86e286f4f19e6a4ffd45468649e4e8f68acb78`;
  the independent-review package and scorer bindings remain byte-identical.
- The recovered ledger/CAS verifies with 17 artifacts, three closed GPU events,
  three closed service sessions, zero model calls, zero attempts, zero unresolved
  allocations, and exactly 422.961986 allocated GPU seconds.
- After v3, the ledger/CAS again verifies with 17 artifacts, five closed GPU
  events, four closed service sessions, one failed attempt/model call, zero
  unresolved allocations, and exactly 815.215409 cumulative GPU seconds.
  Peak v3 VRAM was 22,525,509,632 bytes; peak process RAM was 2,955,644,928
  bytes; no resource limit was violated.
- A CUDA-disabled reconstruction reproduced the exact failed request hash
  `1cc73c5525e096a4df830892f37cdc8062899363a0b75835bb2f04b3a14a0d44`.
  The pinned XGrammar converter rejects 22 empty-alternative `content_hash`
  patterns while translating the schema; after those are removed, pinned vLLM
  would separately reject the single string `date-time` format. The canonical
  validation schema remains valid; only a decoder-compatibility projection is
  required.
- A final bounded adversarial audit found no remaining high- or medium-severity
  duplicate-call, accounting, gold/query-boundary, TOCTOU, or public-path leak.

## Remaining registered GPU inventory

- No GPU allocation is currently authorized. The single amended fallback v3
  service start and one long-reserve failed request were consumed.
- Remaining base scientific calls before repairs: 258 (four fallback
  micro-pilot calls, 24 development calls, 168 held-out primary calls, 49
  combined synthetic calls, and 13 case-study calls).
- Actual allocated GPU time is now 815.215409 seconds. The v3 result's remaining
  forecast is not valid admission evidence because it incorrectly uses the
  0.852878-second failed transport duration as a successful C1 latency proxy.
  Recompute the forecast from provisional C1 timing after fixing that defect and
  before proposing any new amendment.
- The preflight counted 287 effective accounting events and no more than 278
  inference attempts. It projected 11,058,371,456 occupied bytes and
  18,941,628,544 bytes of effective storage headroom.

## Gates and blockers

- The v5 tree and preflight remain immutable evidence for the consumed v3 run.
  Do not reuse them to authorize another call. A decoder-compatibility repair,
  corrected forecast, complete CPU validation, fresh source association, a new
  explicit service-start/inference amendment, and a fresh validation-only gate
  are required before another GPU allocation.
- Do not open held-out query payloads or start held-out inference until the
  independent review completion reproduces.
- Do not start the case-study phase until all synthetic pre-case gates pass and
  an exact lawful local novel path is supplied.
- No destructive cleanup, cache replacement, model download, model switch, or
  additional service start is authorized.

## Exact resume command

Resume with CPU-only implementation and tests for decoder-schema compatibility,
failed-transport diagnostics, and successful-only timing forecasts. Preserve the
v3 result, ledger, CAS, checkpoints, lease, and logs. Do not run
`run_fallback_gpu_acceptance.py --execute` or start vLLM: v3 is terminal and no
additional service start or inference attempt is authorized.
