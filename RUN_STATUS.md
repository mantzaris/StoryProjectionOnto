# Run status

Updated: 2026-09-06 23:57 UTC

## CPU-only repair completed locally — GPU remains unauthorized

The transport now preserves restricted response fragments and full diagnostic
chains before decoding. Schema and scientific validation failures are distinguished
without weakening their gates. The separate run controller now prepares before
model allocation and retains fresh live identity, resource, and budget checks.
See `docs/CPU_ONLY_REPAIR_REPORT.md` for changed files, measurements, and limits.

Focused verification: 256 unique tests pass across the final runs (234 unit and
22 real loopback HTTP tests). An exact-metadata test was updated for the required
new failure-stage field and passed its targeted rerun; unaffected passes were
reused. The earlier interrupted host-permission test batch and all subsequent
logs/XML are preserved in ignored restricted storage. Ruff/diff checks pass.

The local measured preparation subset has a 0.696744-second median over five
samples; credited pod/GPU savings are **zero**. No remote execution or deployment,
new model output, GPU allocation, plan/config amendment, or PDF generation occurred.
The canonical V9 ledger and result hashes remain unchanged; actual allocated
time remains **2,936.238858 seconds**.

The proposed (not authorized) next C1 recovery-gate envelope is 840 seconds,
including startup, live checks, one retry, drain, and shutdown. Earmarking its
already-forecast 240-second long-reserve slot gives an all-in bound of
**33,523.583202 seconds**, so the current 32,400-second ordinary gate still fails.
The concrete proposal is **33,660 scheduled seconds (9.35 hours)** with the strict
actual hard stop still below 36,000 seconds. No mandatory work or prior second is
removed. This resource amendment requires the user's decision; whole-gate cap
enforcement and a fresh source-bound preflight are still prerequisites to any
future launch. Acceptance/development continuation must retain the same live
service if separately authorized and all gates pass. Held-out review remains
required. The older V9 outcome below is preserved as historical evidence.

## V9 terminal outcome — no further GPU start authorized

The single authorized V9 start completed in **183.849429 seconds**, within its
300-second timeout. The pre-start hard-budget check passed with the protected
180-second drain/shutdown allowance. The ordinary rule remains **all actual
allocation, including live service overhead, plus all remaining registered work
<= 32,400 seconds**. The exception was used only for service startup.

Post-start ordinary admission passed: the recorded controller-handoff calculation
was 2,795.0750912889002 + 29,459 = 32,254.0750912889 seconds. The fresh stage-two
gate also passed before the first scientific request. V9 then attempted the
authorized `fallback-c1-01` reserve-long retry; it failed after **21.302286
seconds** with `RuntimeTransportError`. There are **zero accepted outputs**, zero
repairs, and zero development calls. The server log records generation activity,
but the decoder retained neither the raw response nor the detailed error for
this failure path. The exact decoding/transport cause cannot be recovered; token
truncation is not an established diagnosis. No scientific validity or throughput
gate is claimed to have passed.

The guardian verified physical shutdown at 2026-09-06 23:13:51 UTC. V9 used
354.971155 service seconds: 183.849429 startup, 21.302286 failed call, and
149.819440 service overhead (including controller adoption and shutdown).
Total study allocation is **2,936.238858 seconds**. The updated terminal
full-inventory forecast is **29,987.34434394846 remaining seconds**, so the all-in
forecast is **32,923.58320194846 seconds**, exceeding ordinary admission by
**523.58320194846 seconds**. This is not authorization for another start or retry.
The strict actual hard budget has 33,063.761142 seconds remaining, exclusively;
the projected hard margin is 3,076.41679805154 seconds before any additional
recovery costs. Historical failed allocation remains counted.

The remaining forecast still includes all mandatory development, held-out,
feedback, ablation, and narrative work and unconsumed registered reserves. Its
five remaining planned service loads use 333.6688687896926 seconds each (observed
V9 startup plus service overhead); no valid generation-speed observation exists.
No mandatory comparison was removed. Any subsequent recovery must explicitly
budget its new startup and retry, and needs new authority. V9 cannot resume.

Peak sampled V9 VRAM was 22,793,945,088 bytes; stage-two process RAM was
3,172,536,320 bytes and project occupancy 10,347,346,432 bytes. The cumulative
resource gate passes. The terminal ledger/CAS audit passes: 18 hashed artifacts,
2 failed model calls total, 9 GPU events, 7 closed service sessions, and no open
allocation/service journals. The terminal ledger SHA-256 is
`53e4341b14f53dce48482cd765eb4654777df76061ff4948e0ea9bcf8a93dc5c`.
The public V9 result SHA-256 is
`dbd5c1f5defccba571879548b152de59ffd59e90e5e1a6a6ea6f22eede2a03dc`.
The source checkpoint is `c094c1e67137c84d0746e66263439b80f53e2161`;
authorized preflight checkpoint is `0ca9158`.

Exact terminal records, logs, ledger, blobs, file inventory, and verification
script are preserved locally under
`artifacts/restricted/recovery_validation/v9_terminal_20260906T2313Z/`.
Original records remain on the remote project. Passing source tests were reused;
no production code changed during this attempt. Newly run checks verified
ledger/CAS integrity, canonical result/handoff/guardian hashes, accounting sums,
and physical shutdown. No new PDF was made. The RunPod pod remains active;
vLLM and the V9 tmux job are stopped.

Smallest next engineering work: preserve exact HTTP response bytes and detailed
decoder failure in restricted storage before parsing, with CPU regression tests;
do not relax structured-output validation or claim a retroactive V9 repair.
Separately, move invariant CPU controller preparation ahead of allocation and
demonstrate a complete forecast saving at least 523.583202 seconds **plus any
additional recovery costs**. A budget amendment is not silently applied. No V10
has been implemented or launched. Held-out work remains gated on human review.

There is **no authorized GPU resume command**. Read-only remote status command,
from the remote project root:
`.venv/bin/python artifacts/restricted/recovery_validation/v9_authorization_20260906T2256Z/v9_control.py status`.
The following pre-launch section and older ledger totals are historical.

## Authorized V9 execution

The user explicitly authorized exactly one V9 essential-recovery service-only
contingency start. It permits zero inference or development calls under
contingency, no V10 or further startup, no accounting reset, and no change to the
32,400-second ordinary gate or strict stop before 36,000 actual seconds.

The CPU-only builder and preflight completed at 2026-09-06 23:02:51 UTC. The
preflight passes only through the separately recorded service-start contingency:
ordinary admission remains false. It confirms 2,581.267703 consumed seconds,
29,459 remaining mandatory seconds, a 391.40054529582005-second startup forecast,
a 300-second startup timeout, 180 protected drain/shutdown seconds, and
3,388.33175170418 seconds of hard margin. Current controlled storage is
10,345,472,000 bytes; projected occupancy is 11,431,016,320 bytes with
18,568,983,680 bytes of allocation headroom.

Independent local and remote source manifests are byte-identical at file SHA-256
`c7d2e80a33e851ef1b917adeb209d3c29c32f00b52cff5b1da7733e4dd790d17`,
binding tested source commit `c094c1e67137c84d0746e66263439b80f53e2161`.
The association file SHA-256 is
`16300d106e25b5d56add6eb8a761eff622c906a7a04fdb7803fdcc333676859f`;
the authorized restricted overlay file SHA-256 is
`57b9fdfbcea988ce2ebd1d072398604c0e27a155232c90f0e92f475eefb87d5d`;
the public preflight file SHA-256 is
`14b8f763e3ae219b1ff475a5a7ecdeeff896db1130220cd0974f16c7655f4fc3`.
The canonical ledger remains byte-identical to terminal V8 before launch.

The next persistent job is `storyprojection-study-v9`. Its exact operator
wrapper and logs are preserved in ignored restricted recovery storage.
Acceptance and the 24 development calls continue automatically only if the
fresh measured ordinary gate passes. Held-out execution remains blocked on
human review. The earlier authorization-blocker statements below are historical
and superseded by this section; the source and failure records remain preserved.

## Verified durable state

- Branch: `implementation/query-dependent-temporal-ontology`.
- Reconciled recovery checkpoint:
  `d21f2c0de9d6c8d82a5ca386ef880d767e7346d4`.
- Previously verified status checkpoint:
  `fb3399690fe121e9d314f53e40337cdec415782c`.
- Tested v4-incident/v5-recovery checkpoint:
  `48852dc61cfb6224779ed28ee11a20f2fe951882`.
- Tested v5 source checkpoint:
  `bfe43bca2f5a242a694b4a6998b7e384fe7dfc1b`.
- Tested v6 launch-hardening checkpoint:
  `96ee3a39f38f8d5f49a58d2babd53fd4c1b5d5a4`.
- Tested v7 authorization/source checkpoint:
  `0758e487740c608b7eb1060a7a940352032069ef`.
- Tested v7 incident/v8 runtime-hardening checkpoint:
  `703cede70a45b97452b4d68f7612bf5a467fb1d6`.
- Tested fixed-mode RunPod lease-storage checkpoint:
  `4b5b0cc768844f7661fc7ca316bd7e63cc15a2a6`.
- Tested v8 authorization/preflight checkpoint:
  `ef1049927cdc247b5ffb6d838658f64467f0c5ae`.
- Tested v8 incident/lease and blind-review alignment checkpoint:
  `232ed1624f9175dd84e0ff36be6082e479a4bbd0`.
- Tested post-review execution-lineage refresh checkpoint:
  `4934a5c187542c64d1bf6770f8141cbb25190c07`.
- Tested historical-report replay and source-inventory checkpoint:
  `a5f1224e5862bef3fc9aec1d0241861b965c1906`.
- Tested CPU-only v9 recovery-admission checkpoint:
  `2574cdc73c64d9bce947fb84d315a4ac3ade7ee3`.
- Local project: `/home/resort/Documents/repos/StoryProjectionOnto`.
- Remote project: `/workspace/StoryProjectionOnto`.
- The reconciled canonical Phase 1 ledger is schema v8, file SHA-256
  `33b18e16478b8c73951ff0269479ae697874ca1772669d87539ff2d7ac8884e2`.
  Its read-only ledger/CAS audit passes with 17 artifacts, one failed attempt/model call,
  seven closed GPU events, six closed service sessions, and no unresolved allocation or
  service journal. V8 added no inference attempt, model call, accepted output, or repair.
- The stale 422.961986-second local ledger remains preserved in ignored recovery storage;
  it was not discarded or used for admission.
- The authorized v4 orchestration failed before guardian readiness because an internal
  stage output was incorrectly reinterpreted as the outer result path. Its exact six-file
  restricted record remains preserved. The typed public incident has file SHA-256
  `ff74489a367c6e24c693ba8c00ccc3de505ef9af1f84f3a90518ef573ad06052`
  and logical SHA-256
  `e8b30397068a97f4f169395ec0a70ec1f709515c1df07b6a17400c7568c2993c`.
  It proves zero new GPU events, service sessions, attempts, model calls, accepted outputs,
  retry consumption, or allocated GPU microseconds. v4 is terminal and cannot be resumed.
- The v5 CPU-only preflight passed (file SHA-256
  `7fd9dd36d15a12129a68b22dbb60e0ba5c12bd68268b9f09ebd8e887fd108ad5`,
  logical SHA-256
  `f3b90b16a323f81066b4392fccecdb5af69718e76b67f0325dec9d8adaf70b24`).
  Its real guarded launch then failed safely before model or GPU start: the guardian's
  heavyweight initialization exceeded the 30-second readiness watchdog. The guardian
  later published readiness, but the required status check could not open the ledger while
  that guardian held its WAL writer. The guardian performed owner-loss takeover and terminal
  reconciliation at 16:02:32 UTC. Exact terminal status certifies physical shutdown,
  `resume_allowed: false`, no checkpoint/handoff/result, and zero unresolved journals.
  V5 is terminal, preserved, and must not be resumed. Its independently rebuilt public
  incident has file SHA-256
  `47ce61fe2d65eab23e967eb06ff8eb3920d3e21f6191564f8b2a73bea4f4709f`
  and logical SHA-256
  `06b7bf28427266efa9ebae3640a8a4fe883b0233956d313d98fb9103775dbfdf`.
- The bounded v6 control-plane repair was implemented and independently reviewed. Guardian
  readiness now has the already authorized 300-second CPU-only bound, terminal verification
  retains its separate 90-second bound, status permits a coordinated live-WAL snapshot only
  after proving the exact guardian argv/PID/process-group/session identity, and attempted-start
  resume requires exactly one matching open service journal. Adversarial probes fail closed.
- V6 passed its CPU-only preflight, but its guarded launch then failed during guardian
  construction before readiness, model loading, or GPU allocation: a stale development
  invariant expected v3+v5 service-event IDs while the production factory correctly derived
  v3+v6. V6 is terminal and cannot resume. Its six-file restricted record and before/after
  ledgers are preserved. The typed public incident has file SHA-256
  `e8fb3aac180838a4212435ca51905178b7a22ca76990321027f01e182311df56`
  and logical SHA-256
  `d052713bd262745afc2060e0a75a3b565cf91db93365832b1a8b27dac09a81e9`.
  Independent rebuild, SQLite diff, release scan, and focused tests confirm exactly one
  storage row and zero GPU, model-service, internal-controller, inference, retry, or output
  consumption.
- The bounded v7 correction now derives only the exact v3+v7 service-event lineage, rejects
  v4/v5/v6 execution, and requires a schema-1.5.0 overlay binding all three terminal
  control-plane incidents plus a fresh v7 source association. Scientific inputs, the one
  reserve-long retry, the 288 effective accounting events, 278 maximum inference attempts,
  and the complete-run forecast are unchanged. An adversarial prelaunch review also exposed
  and repaired a v4 outer-record substitution gap: v4 is now pinned by both its exact file
  and canonical manifest hashes, with a timestamp-plus-rehash regression.
- The real v7 launch reached the pinned fallback endpoint but failed before inference when a
  slow startup resource sample exceeded the controller's derived join window. Its guardian
  preserved the failure records, and a manually identity-bound stop verified process-group,
  endpoint, and GPU absence. The typed incident records 692.635556 service seconds, including
  224.234089 classified failure seconds and 468.401467 service-overhead seconds; peak observed
  VRAM remained 22,525,509,632 bytes. The incident has file SHA-256
  `222c4011d458be9776d75214db59a9da567a5406480735425e62cffa61fa2edf`
  and logical SHA-256
  `4a550f74c1627e196d07db475acf9200fda2127f59516a59ad14b40e4867aa15`.
  V7 is terminal and cannot be resumed. A bounded incident-derived utility for restoring only
  its null-identity terminal lease passes five focused tests and is not permitted to mutate
  the ledger.
- The RunPod network volume exposes fixed permissive mode bits and ignores successful POSIX
  chmod requests. The first lease-repair precondition step therefore stopped before opening the
  ledger or changing lease state. Exact pre-repair copies remain in restricted recovery
  storage. A narrow compatibility proof now accepts only the root-owned RunPod mfs FUSE mount
  class after positive kernel mountinfo, fixed-mode chmod-no-op, namespace, symlink, and hash
  checks; ordinary filesystems still require mode 0600. The exact observed mount rendering and
  adversarial cases pass 14/14, and the full incident/lease/fallback subset passes 100/100.
- The first remote invocation of the repaired lease CLI also failed closed before opening the
  ledger or changing the lease because the final typed v7 incident had not yet been synchronized
  to the remote public-manifest directory. The one missing file was restored only after its
  SHA-256 matched the local public record. The bounded repair then restored the exact v7 lease to
  `stopped_verified`, added zero accounting rows and zero inference calls, and preserved the
  ledger byte-for-byte. Its restricted receipt has file SHA-256
  `bdfd002a21ce75cc164889d33bfdc5fbb11d858dca509413decd44a09674c388`;
  an immediate idempotent replay preserved the receipt, lease, and ledger hashes.
- The independently generated local and remote v8 source manifests are byte-identical with file
  SHA-256 `806b237d29b78809f6a796c9c9671ec0c0b18e91a2571d289313fe02faae5150`.
  Their association has file SHA-256
  `5dd314003f802717feed6c72bf801d89adf4aca848351f8686409554bc46bc75`.
  The authorized schema-1.6.0 v8 overlay is preserved in restricted storage with file SHA-256
  `3b215d02c73066546257da53e067ba4d47ab3f0f986640cede5d08c14b67c775`.
- The fresh v8 CPU-only preflight passed and authorized execution without starting a model or
  allocating GPU time. Its file SHA-256 is
  `2f59e51263f352dafd212d575d492bec21528b823b5bb83e75c5758074bba1c2`,
  and logical manifest SHA-256 is
  `070da9bf50161c43cfec94d7880b51f9e550bb614140746786d4857ee58a3305`.
- The persistent v8 launch then reached a healthy pinned Qwen3-8B-AWQ endpoint. Before any
  inference request, controller-restart adoption rejected the EngineCore member because
  vLLM 0.10.2 had replaced that child's inherited token-bearing argv/environment region with
  its `VLLM::EngineCore` process title. The strict owner check and automated shutdown failed
  closed. The public result, controller handoff, and cleanup records are preserved with file
  SHA-256s `c023b29626bae21421f5bc26e307bf47e6973cd7164cda38ab9d9a7813ef112e`,
  `c5ca07fe734898766c053d08a8c1bac14511a2f1b17370016cad50ed6350e003`,
  and `d0738f06692a94833ab1cf12177308b64a6c9cfde3d6ff5be2e3cc668d3a290a`.
  The result records zero completed calls, zero accepted outputs, and zero repairs.
- After all controllers and the guardian were terminal, an exact-identity, one-shot restricted
  utility verified the three-member process group, the sole GPU owner, and port owner, wrote
  its stop intent durably, and sent only `SIGTERM` to that exact group. Its outcome verifies no
  remaining group member, GPU process, or port-8000 listener at
  2026-09-05 20:40:09.324832 UTC. vLLM is stopped and the GPU is idle.
- V8 terminal accounting has been reconstructed conservatively from the service start and
  verified stop: 1,073.416738 service seconds, comprising 227.686586 seconds already
  classified as successful service start and 845.730152 seconds of additional service
  overhead. The terminal recovery receipt has file SHA-256
  `0cd173be577f5c66717dc6a9365075b3471a860e8dd3432601f0f098ef87e420`
  and logical SHA-256
  `8588ad6cda3531a8d7a53052248c910d977b6127a3a4daa1a2e9dbd1c05d3041`.
  Replay is idempotent and the final ledger has no unresolved allocation or service row.
- The exact typed public v8 runtime incident has file SHA-256
  `c44f34c0599a00264235d13c1f8c9ad80e27a81751102ea09ae759ce9631a3e0`
  and logical SHA-256
  `fa5c75d67c690db828b1eaefebfb6d291908fb89ad6b534bc8d251633519136e`.
  It binds all public and restricted controller evidence, exact manual-stop and accounting
  receipts, all three immutable ledger states, zero-inference deltas, resource maxima, and
  terminal absence without publishing commands, tokens, absolute remote paths, or log text.
  The earlier underbound draft remains preserved in restricted recovery storage with file
  SHA-256 `2caee1ca1f0c3331abc16da7936ce20912e8f1109ad44b121bb371804f388d1d`.
- A bounded source repair accepts only a tokenless, exact NUL-padded `VLLM::EngineCore`
  immediate child of the still-live token-bound PGID/SID leader, after repeated stable
  identity/title/token reads. Wrong tokens, arbitrary tokenless members, ancestry/session/title
  near-misses, zombies, missing leaders, and observed races fail closed. A separate V8-only
  CPU utility can restore the exact incident-bound terminal lease without starting/adopting a
  service or opening the canonical ledger as SQLite. It rejects symlinked inventory entries,
  unreceipted already-restored leases, identity drift, live processes/endpoints/GPU compute,
  and ledger drift; scratch and canonical ledger invariants run on exceptional exit.
- That utility restored the exact remote V8 lease from `shutdown_unverified` to
  `stopped_verified` at 2026-09-05 22:05:57.996391 UTC. The canonical ledger remained
  byte-identical at
  `33b18e16478b8c73951ff0269479ae697874ca1772669d87539ff2d7ac8884e2`;
  zero GPU events, accounting rows, inference attempts, or model calls were added. The
  restricted receipt has file SHA-256
  `544ffd07e57c9bd0c4c3915b8329ce46389e874564364d42edeeec7e2eaa7a58`
  and logical SHA-256
  `99550aa3bbc7a9d921d069cc6c1c1a6542330b400ec5e00f3c1a74c8fb8076e9`.
  Immediate replay preserved the receipt, restored lease, and ledger byte-for-byte.
- Four interrupted remote quarantine directories were packed losslessly before their exact
  unpacked copies were removed. The retained local and remote archive has 94,138 members,
  file SHA-256
  `617a13b5dabf7376f342550d217168d705ab328f0517cf44d663525c6562d721`,
  and passed full decompression/member-count validation.
- The latest terminal remote observation at 2026-09-05 22:42 UTC found one NVIDIA RTX 4090
  (24,564 MiB), 1 MiB used, 0% utilization, no study/vLLM process, no `tmux` or `screen`
  session, and no listener on port 8000. vLLM is stopped; the pod remains active.
- The latest bounded remote project-tree allocated-byte measurement is 16,313,550,336 bytes;
  the immutable v4 preflight's controlled-path sample was 10,337,457,664 apparent bytes. Both are
  safely below the 25 GB occupied limit and preserve the required 5 GB headroom inside the
  registered 30 GB allocation.
- Pinned permitted fallback model: `Qwen/Qwen3-8B-AWQ`, exact revision
  `4da05a8edb55c6046cce958586c33b61da07bb79`.

## Validation state

- Current collected test inventory: 1,244 tests. The last complete stable pre-v8 tree had
  1,101 passes and three expected local-environment skips; the bounded v8 recovery and review
  changes were verified with the focused current-tree selections below rather than another
  broad execution cycle.
- On that last full stable-tree run, the unit suite had 1,016 passes in 347.06 seconds, the
  property suite had 16 passes in 46.89 seconds, and the integration suite had 69 passes plus
  three expected environment skips across 72 tests.
  The skipped local checks are two XGrammar checks (package absent locally) and one pinned
  tokenizer-snapshot check (remote-only snapshot). The renderer, real-browser, and six API
  interface tests were rerun with loopback access and all eight passed.
- Prior full-tree aggregate: 1,101 passed, three expected local-environment skips, zero
  implementation failures at that checkpoint.
- The post-v6 affected suite has 109 passing focused tests for fallback orchestration,
  development lineage, live read-only ledger status, typed incidents, and schema replay.
  A separate complete run of the five directly changed test modules has 108 passing tests.
  The final post-repair run of those modules has 109 passing tests. Independent v6 incident
  reconstruction and v7 prelaunch review found no remaining launch blocker in scope.
- The final v7 incident and lease-repair suite has 25 passing tests; the independent-review
  handoff/runtime/benchmark suites now pass 27 targeted tests. After the final periodic-watchdog and coordinator
  publication hardening, the stable focused lineage/runtime suite passed 203 tests, the full
  runtime/fallback pair passed 182 tests, and independent targeted audit runs passed 45/45
  lineage checks plus 16/16 repeated publication-race checks.
- The final nine-module prelaunch suite passes 248/248. An immediately preceding run had one
  heartbeat-test polling-window miss under concurrent local test load; the isolated test then
  passed 11/11 and the clean full rerun passed. No production exception or remote activity was
  involved.
- The post-v8 bounded EngineCore repair previously passed all 118 GPU-runtime tests and Ruff.
  The final changed-runtime selection passes 18 tests, and the combined V7/V8 incident,
  lease, schema, review, benchmark, and release selection passes 99 tests. One broader
  213-test run had 212 passes plus the known one-second heartbeat polling-window miss under
  parallel local load; that exact test passed immediately in isolation. Independent integrity
  reviews found and closed the run-root symlink and unreceipted-restored-lease gaps before
  remote use.
- Ruff, Python compilation, JSON parsing, `git diff --check`, package/CLI import smoke,
  report ingestion replay, report replay, and public-release hash checks pass.
- Twenty-eight generated JSON Schemas reproduce byte-for-byte; together with their manifest,
  the checked directory has 29 JSON files. Schema logical manifest SHA-256:
  `57cea3d884ef286b7bec3f02dbf92f2e7a5d4e9a5f75228d51cab288728d7a9b`.
- Synthetic benchmark verify-only and the lineage-only semantic guard pass. Removing the
  implementation-only adjudicator identity restriction changed only compiler lineage: every
  substantive artifact proved byte-identical, and only the draft seal and benchmark manifest
  changed. Benchmark logical manifest SHA-256:
  `98589db781a254cfa2524c85e59a7bbc048bedf99accc0bdccd812b5aeef2514`;
  draft-seal logical SHA-256:
  `2d928894e6154b0a0fe4b8922fb8c98cb1d58c94d3fab77d97a197564864651d`.
- The benchmark lineage-only refresh initially left five downstream execution-control
  configurations pinned to the prior manifest or draft-seal hashes. Checkpoint `4934a5c`
  refreshes only those bindings. The exact 24-call development dry-run, 168-call held-out
  derivation, combined block, feedback protocol, and Phase-5 runner now reproduce; a
  72-test affected dependency suite and a separate 52-test entrypoint/property suite pass.
  No evidence, gold, review question, call count, condition, seed, budget, or output changed.
- The prior report-ingestion receipt reproduces at
  `dd64d597f3497c2b9ad7fc60205e90ad5540716e586f885b7aa0aada90b245fd`
  through an explicit, self-hashed historical-source snapshot that archives only the exact
  predecessor benchmark bytes. Snapshot substitution is verification-only, requires exact
  manifest/receipt and override-inventory bindings, and is never the default for new or final
  reports. The 52-test reporting selection, adversarial path/hash cases, Ruff, and both explicit
  replay commands pass. The interim PDF, Markdown, receipt, result metadata, figure manifest,
  reproducibility file, and figure remain byte-identical; no interim report was regenerated.
  The current public input allowlist retains the current benchmark at its canonical path and
  carries the predecessor only under its versioned snapshot path; its logical SHA-256 is
  `e755537eb45856637063b97b108e6c616bde2b50845a69e2b4c2e35a11a42641`.
- The fail-closed Phase-7 source inventory now reconciles all post-recovery additions: 29 JSON
  schemas, 66 scripts, 104 science-code paths, and 128 verification paths. Its compiler and
  registry selections pass 30/30; no scientific output or report number changed.
- The condition-blind human review handoff is materialized in ignored restricted scorer-only
  storage. It contains three worlds, nine projections, all shared evidence and proposed gold
  structures—including 18 readable executable alternative representations—72 exact questions,
  and a blank response worksheet; it contains no method outputs, reviewer judgments, or launch
  authorization. The packet and worksheet SHA-256s are
  `38c3b76ed5deb8afdf2aa34fdaa4e1801109e85344911edc802c8d09e26e9e94`
  and `780b62b49076f68dfd592db864de25fee24abb6916aa239d1e63e27649bdeae2`;
  handoff logical SHA-256 is
  `4744303d9f081e913a05d46ceac7b3fb8d010ac2db95dd6af91103912e70eb56`.
  The prior fingerprint-only rendering is preserved in restricted recovery storage.
- The CPU-only schema-1.7 v9 scaffold preserves full-validator replay of every immutable v4-v8
  overlay while making all five historical run IDs terminal on every execution surface. V9
  binds the exact v8 incident and lease-repair receipt, distinguishes the 300-second process
  watchdog from the 391.40054529582005-second admission proxy, and permits the registered
  hard contingency to reach only one service-start event. The runner independently repeats
  exact-path, canonical-payload, and dependency validation at construction and before prepare,
  recovered-prepare, and run boundaries. Every post-adoption path rechecks the ordinary
  32,400-second gate before a checkpoint handoff, reserve consumption, job creation, attempt,
  or model call. No v9 source association, authorization overlay, preflight, launch artifact,
  model process, GPU event, inference attempt, or model output has been created.
- The final cross-module fallback/development/runtime/schema selection passed 291 tests before
  the last bounded recovered-prepare gate was added. The complete fallback suite then passed
  95/95, including that regression and five exact historical-overlay replays; Ruff and
  `git diff --check` pass. The independent final audit also passed 15 critical lineage and
  authorization tests, 10 accounting/contingency tests, and 51 incident tests. The review
  handoff dry-run reproduces and its handoff/runtime tests pass 14/14. The 24-call development
  manifest validates without permitting model load, service start, shutdown, or artifact
  writes.

## Phase state

| Phase | State | Verified position |
|---|---|---|
| 1 — contracts and GPU acceptance | Awaiting explicit essential-recovery authorization | Contracts, ledgers/CAS, storage and GPU controls, evidence/ontology boundary, provenance bridge, decoder projection, process-group hardening, and the CPU-only v9 fail-closed path pass. V4–v6 are preserved zero-GPU control-plane incidents; v7 and v8 are preserved terminal runtime incidents with no inference. Ordinary admission is false by 31.668248 seconds. V9 may start only if the plan's service-only essential-recovery contingency is explicitly unlocked; it cannot make a call until live allocation plus all remaining work separately fits the strict 9-hour gate. |
| 2 — synthetic benchmark | Software/data complete; independent review pending | Four development worlds, 12 held-out worlds, 36 primary contexts, contrastive pairs, rare-pivotal/temporal/epistemic gold, mutation tests, and the condition-blind 3-world/9-projection review package reproduce. |
| 3 — conditions and primary run | Software complete; execution/calibration pending | C0, C1, C2, and A-FixedSelect pathways and timing/capability/equal-evidence implementations pass. C0 implementation is complete; its four-world competence calibration is intentionally still unfrozen. The 24 development and 168 held-out calls remain. |
| 4 — metrics and ablations | Software complete; execution pending | Registered metrics, world-level inference, 4,096 sign flips, Holm correction, bootstrap sensitivity, community analysis, and three reduced ablations are implemented. |
| 5 — interface and feedback | Software complete; execution pending | Cytoscape interface, two registered actions, six scripted revisions, three trace captures, fixed anchors, and provenance-aware before/after views pass local tests. |
| 6 — one-novel case study | Software complete; lawful input pending | Query-blind indexing, window/C1/C2 control planes, restricted storage, and descriptive analysis are implemented. No novel inference has run. |
| 7 — results and release | Interim only | Immutable-table reporting, visual inspection, accounting, and public-release controls pass. Final numerical outputs await registered execution. |

## GPU and call accounting

- Actual allocated GPU time consumed: 2,581.267703 seconds.
- In the historical v5 preflight, `31,722.618681` seconds was the complete
  all-in projection, not a remaining-work value. Its exact decomposition was
  `815.215409` seconds already consumed + `30,516.0027264791` seconds of
  remaining mandatory work + `391.40054529582005` seconds for the next service
  allocation. Adding the consumed allocation to `31,722.618681` a second time
  double-counts it. The corresponding scheduled reserve was
  `32,400 - 31,722.618681 = 677.381319` seconds.
- Recovered unattended-allocation uncertainty: zero seconds.
- The v4 control-plane launch consumed zero GPU seconds and did not consume an inference,
  retry, or service-start slot; it remains visible as one operational failure row.
- The v5 control-plane launch likewise consumed zero GPU seconds and no inference, retry,
  or service-start slot; it added only one valid storage sample and remains visible as a
  distinct operational failure.
- The v6 control-plane launch likewise consumed zero GPU seconds and no inference, retry,
  service-start, or internal-controller slot; it added only one valid storage sample and
  remains visible as a distinct operational failure.
- The v7 service allocation consumed 692.635556 seconds and one recovery service-start slot,
  but zero inference or retry attempts and zero accepted model outputs.
- The v8 service allocation consumed 1,073.416738 seconds and one recovery service-start
  slot, but zero inference or retry attempts and zero accepted model outputs. All three
  currently authorized post-v3 recovery service starts are therefore consumed.
- Peak recorded GPU VRAM: 22,525,509,632 bytes (20.98 GiB), below 23 GB.
- Peak recorded process RAM: 2,972,696,576 bytes (2.77 GiB), below 25 GB.
- Peak recorded project-controlled storage during v8: 10,344,769,024 bytes (9.63 GiB),
  below 25 GB with the required headroom.
- Remaining registered scientific generations: 258: four fallback acceptance calls,
  24 development calls, 168 held-out primary calls, 49 combined synthetic calls
  (12 paraphrase, nine feedback/interface, 28 ablation), and 13 case-study calls.
- The authoritative admission calculation is all-in: actual allocation already consumed,
  plus the forecast for every remaining mandatory call, plus the next service-load allowance.
  After terminal v8 accounting, the unchanged calculation for any fresh start is
  `2,581.267703 + 29,459.000000 + 391.40054529582005 =
  32,431.66824829582005` seconds (9.008796736 hours). The registered scheduled reserve is
  therefore `-31.66824829582005` seconds: the 9-hour admission gate fails. The conservative
  hard contingency after the bounded 120-second sampler drain and 60-second shutdown reserve
  is still `3,388.33175170417995` seconds, but hard-cap headroom does not override the failed
  scheduled gate. Every mandatory call remains in the forecast; no budget or inventory has
  been silently changed.
- No new GPU call is admitted merely because platform credit changed. The registered
  9-hour scheduled ceiling and hard stop before 10 actual allocated hours remain binding.

## Gates and blockers

- Ordinary admission for a fresh GPU start is forbidden because the all-in forecast exceeds
  the registered 9-hour scheduled ceiling by 31.668248 seconds. The plan permits one
  essential-failure-recovery use of locked hard contingency after scheduled reserve is
  exhausted, but it must be explicitly authorized for the single v9 service-start event.
  That authorization would cover zero inference attempts and zero development calls. After
  startup, the measured increment must be at most 359.732297 seconds so actual allocation plus
  the unchanged 29,459-second remaining forecast fits 32,400 seconds; otherwise the service
  is stopped without a scientific call. Mandatory calls cannot be silently removed.
- The v5 and v8 preflights are historical: each passed for its then-current immutable source,
  overlay, and ledger state, but neither authorizes or describes v9. The 8B fallback has no
  accepted output yet. The tested v9 implementation can materialize the next CPU-only source
  association, schema-1.7 authorization overlay, and preflight only after explicit dated
  authorization, then launch once in a fresh detached persistent job. Inference calls remain
  subject to the ordinary 9-hour gate.
- Held-out inference is forbidden until the mandatory independent review and adjudication
  reproduce. No second review has been fabricated.
- Phase 6 requires the exact lawful local novel path only when synthetic work is complete.
- No destructive cache replacement, model switch, hidden-ontology shortcut, held-out gold
  access, or untracked foreground inference is authorized.

## Exact resume command

V7 and v8 are terminal and have no permitted resume command. No GPU resume command is
authorized while the scheduled-budget gate above fails. The existing 24-call development
block can still be checked without model load, service start, or artifact writes with:

```bash
PYTHONPATH=src /tmp/spo-refresh-venv/bin/python scripts/run_development_block.py \
  --project-root /home/resort/Documents/repos/StoryProjectionOnto --validate-only
```

Any later GPU recovery must use a fresh run ID, run root, result path, source association,
incident binding, authorization overlay, and detached persistent session. It must first pass
the all-in 9-hour admission calculation and may not reuse v4/v5/v6/v7/v8 mutable state.
