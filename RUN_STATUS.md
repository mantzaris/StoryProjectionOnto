# Run status

Updated: 2026-09-05 17:30 UTC

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
- Local project: `/home/resort/Documents/repos/StoryProjectionOnto`.
- Remote project: `/workspace/StoryProjectionOnto`.
- The reconciled canonical Phase 1 ledger is schema v8, file SHA-256
  `8698837637cc3d5c86e3abb3593f5647d25a17bfdaaf01fb35a36ff1e0c34a9d`.
  Its read-only ledger/CAS audit passes with 17 artifacts, one failed attempt/model call,
  five closed GPU events, four closed service sessions, and no unresolved allocation.
  V5 and v6 each added exactly one allowed storage-preflight row and no GPU event,
  service session, attempt, call, artifact, or unresolved journal; the ledger now contains
  115 storage samples.
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
- Four interrupted remote quarantine directories were packed losslessly before their exact
  unpacked copies were removed. The retained local and remote archive has 94,138 members,
  file SHA-256
  `617a13b5dabf7376f342550d217168d705ab328f0517cf44d663525c6562d721`,
  and passed full decompression/member-count validation.
- The latest read-only remote audit at 2026-09-05 17:12 UTC found one NVIDIA RTX 4090
  (24,564 MiB), 1 MiB used, 0% utilization, no study/vLLM process, no `tmux` or `screen`
  session, and no listener on port 8000. The v6 public result and guardian-result receipt are
  absent. vLLM is stopped.
- The latest bounded remote project-tree measurement is 10,327,091,099 apparent bytes;
  the immutable v4 preflight's controlled-path sample was 10,337,457,664 bytes. Both are
  safely below the 25 GB occupied limit and preserve the required 5 GB headroom inside the
  registered 30 GB allocation.
- Pinned permitted fallback model: `Qwen/Qwen3-8B-AWQ`, exact revision
  `4da05a8edb55c6046cce958586c33b61da07bb79`.

## Validation state

- Complete collected test inventory: 1,104 tests.
- Unit suite: 1,016 passed in 347.06 seconds on the final isolated stable-tree run.
- Property suite: 16 passed in 46.89 seconds.
- Integration suite: 69 passed and three expected environment skips across 72 tests.
  The skipped local checks are two XGrammar checks (package absent locally) and one pinned
  tokenizer-snapshot check (remote-only snapshot). The renderer, real-browser, and six API
  interface tests were rerun with loopback access and all eight passed.
- Aggregate verified result: 1,101 passed, three expected local-environment skips, zero
  implementation failures.
- The post-v6 affected suite has 109 passing focused tests for fallback orchestration,
  development lineage, live read-only ledger status, typed incidents, and schema replay.
  A separate complete run of the five directly changed test modules has 108 passing tests.
  The final post-repair run of those modules has 109 passing tests. Independent v6 incident
  reconstruction and v7 prelaunch review found no remaining launch blocker in scope.
- Ruff, Python compilation, JSON parsing, `git diff --check`, package/CLI import smoke,
  report ingestion replay, report replay, and public-release hash checks pass.
- Twenty-five generated JSON Schemas reproduce byte-for-byte; together with their manifest,
  the checked directory has 26 files. Schema logical manifest SHA-256:
  `e40a3c5ea4ca00d935e254cf261396c15883b6487def4e07a73acc1d5b0562e6`.
- Synthetic benchmark verify-only and the semantic refresh guard pass. Benchmark logical
  manifest SHA-256:
  `c5bce978a9006b67f13c40701fca2d7b558d62239615c49a2c035221cd1b723f`.
  Refresh receipt SHA-256:
  `2396c3a7c308a964b348022a6811cde1dc0b0f1d4af05f3838c68a73bc6f9696`.
- The report-ingestion receipt verifies at
  `dd64d597f3497c2b9ad7fc60205e90ad5540716e586f885b7aa0aada90b245fd`.
  The interim report remains explicitly incomplete and contains no fabricated efficacy
  results. Its eight-page PDF was rendered and inspected. The current public bundle input
  manifest is self-consistent at logical SHA-256
  `bb2644ebd0460efe8eaa7b566a86603ff6c423c5710f026ed194fc5a997c3ead`.
- The condition-blind human review handoff is materialized in ignored restricted scorer-only
  storage. It contains three worlds, nine projections, all shared evidence and proposed gold
  structures, 72 exact questions, and a blank response worksheet; it contains no method
  outputs, reviewer judgments, or launch authorization. The packet and worksheet SHA-256s are
  `5003724e1dae0909b729980eb334b19b32073a2df03dc33ea17aa7f96a021ac1`
  and `780b62b49076f68dfd592db864de25fee24abb6916aa239d1e63e27649bdeae2`.

## Phase state

| Phase | State | Verified position |
|---|---|---|
| 1 — contracts and GPU acceptance | In progress | Contracts, ledgers/CAS, storage and GPU controls, evidence/ontology boundary, provenance bridge, decoder projection, and process-group hardening pass. V4, v5, and v6 are preserved zero-GPU control-plane incidents. The bounded v7 lineage repair passes focused CPU tests; its fresh source association, overlay, preflight, and real fallback execution are next. |
| 2 — synthetic benchmark | Software/data complete; independent review pending | Four development worlds, 12 held-out worlds, 36 primary contexts, contrastive pairs, rare-pivotal/temporal/epistemic gold, mutation tests, and the condition-blind 3-world/9-projection review package reproduce. |
| 3 — conditions and primary run | Software complete; execution pending | C0, C1, C2, and A-FixedSelect pathways and timing/capability/equal-evidence gates pass. The 24 development and 168 held-out calls remain. |
| 4 — metrics and ablations | Software complete; execution pending | Registered metrics, world-level inference, 4,096 sign flips, Holm correction, bootstrap sensitivity, community analysis, and three reduced ablations are implemented. |
| 5 — interface and feedback | Software complete; execution pending | Cytoscape interface, two registered actions, six scripted revisions, three trace captures, fixed anchors, and provenance-aware before/after views pass local tests. |
| 6 — one-novel case study | Software complete; lawful input pending | Query-blind indexing, window/C1/C2 control planes, restricted storage, and descriptive analysis are implemented. No novel inference has run. |
| 7 — results and release | Interim only | Immutable-table reporting, visual inspection, accounting, and public-release controls pass. Final numerical outputs await registered execution. |

## GPU and call accounting

- Actual allocated GPU time consumed: 815.215409 seconds.
- Recovered unattended-allocation uncertainty: zero seconds.
- The v4 control-plane launch consumed zero GPU seconds and did not consume an inference,
  retry, or service-start slot; it remains visible as one operational failure row.
- The v5 control-plane launch likewise consumed zero GPU seconds and no inference, retry,
  or service-start slot; it added only one valid storage sample and remains visible as a
  distinct operational failure.
- The v6 control-plane launch likewise consumed zero GPU seconds and no inference, retry,
  service-start, or internal-controller slot; it added only one valid storage sample and
  remains visible as a distinct operational failure.
- Peak recorded v3 GPU VRAM: 22,525,509,632 bytes.
- Peak recorded v3 process RAM: 2,955,644,928 bytes.
- Remaining registered scientific generations: 258: four fallback acceptance calls,
  24 development calls, 168 held-out primary calls, 49 combined synthetic calls
  (12 paraphrase, nine feedback/interface, 28 ablation), and 13 case-study calls.
- The v6 preflight's `31,722.618681` seconds is the projected **all-in final allocation**,
  not remaining time. It equals 815.215409 seconds already consumed plus
  30,516.002726 seconds of remaining mandatory work plus a 391.400545-second recovery
  service-start allowance (differences at the sixth decimal are display rounding).
  Therefore the registered scheduled reserve is
  `32,400 - 31,722.618681 = 677.381319` seconds, and the hard contingency is
  `36,000 - 31,722.618681 - 60 = 4,217.381319` seconds after protecting the registered
  60-second shutdown reserve. Admission remains valid without changing the budget or calls.
- No new GPU call is admitted merely because platform credit changed. The registered
  9-hour scheduled ceiling and hard stop before 10 actual allocated hours remain binding.

## Gates and blockers

- Before any further GPU service start: commit the tested bounded v7 repair, independently
  associate byte-identical local/remote v7 source trees, build an authorized schema-1.5.0
  v7 overlay bound to the v4/v5/v6 incidents, reproduce the all-in forecast from the current
  ledger, pass a fresh CPU-only v7 preflight, and repeat the exact remote idle/resource audit.
- Held-out inference is forbidden until the mandatory independent review and adjudication
  reproduce. No second review has been fabricated.
- Phase 6 requires the exact lawful local novel path only when synthetic work is complete.
- No destructive cache replacement, model switch, hidden-ontology shortcut, held-out gold
  access, or untracked foreground inference is authorized.

## Exact resume command

V6 is terminal and has no permitted resume command. After committing this tested checkpoint,
generate the fresh local v7 source manifest:

```bash
PYTHONPATH=src /tmp/spo-refresh-venv/bin/python -m story_projection_onto.manifest \
  --root . --revision fallback-second-recovery-v7 \
  --output artifacts/public/manifests/source_tree_fallback_second_recovery_v7.local.json
```

Synchronize that exact source inventory without deletion, independently generate the remote
v7 manifest, associate both with the tested commit, build the authorized v7 overlay bound to
all three terminal incidents, and run the CPU-only v7 preflight. If it passes, the next real-output
job is the checked-in `scripts/run_fallback_gpu_acceptance.py --execute` launcher in detached
tmux session `storyprojection-study-v7`, using run ID
`fallback-qwen3-8b-awq-development-v7`, fresh restricted run root
`artifacts/restricted/fallback-development-v7`, and public result
`artifacts/public/results/fallback_gpu_acceptance_development_v7.json`. Do not reuse any
v4/v5/v6 run root, result path, tmux session, overlay, or invocation state.
