# Run status

Updated: 2026-09-05 15:38 UTC

## Verified durable state

- Branch: `implementation/query-dependent-temporal-ontology`.
- Reconciled recovery checkpoint:
  `d21f2c0de9d6c8d82a5ca386ef880d767e7346d4`.
- Previously verified status checkpoint:
  `fb3399690fe121e9d314f53e40337cdec415782c`.
- Tested v4-incident/v5-recovery checkpoint:
  `48852dc61cfb6224779ed28ee11a20f2fe951882`.
- Local project: `/home/resort/Documents/repos/StoryProjectionOnto`.
- Remote project: `/workspace/StoryProjectionOnto`.
- The reconciled canonical Phase 1 ledger is schema v8, file SHA-256
  `38775d1fe3c27cb93afbf78f3c692300a05ed52cb083cb4656d0378b0b94429f`.
  Its read-only ledger/CAS audit passes with 17 artifacts, one failed attempt/model call,
  five closed GPU events, four closed service sessions, and no unresolved allocation.
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
- Four interrupted remote quarantine directories were packed losslessly before their exact
  unpacked copies were removed. The retained local and remote archive has 94,138 members,
  file SHA-256
  `617a13b5dabf7376f342550d217168d705ab328f0517cf44d663525c6562d721`,
  and passed full decompression/member-count validation.
- The latest read-only remote audit at 2026-09-05 15:04 UTC found one NVIDIA RTX 4090
  (24,564 MiB), 1 MiB used, 0% utilization, no study/vLLM process, no `tmux` or `screen`
  session, and no listener on port 8000. vLLM is stopped.
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
- Ruff, Python compilation, JSON parsing, `git diff --check`, package/CLI import smoke,
  report ingestion replay, report replay, and public-release hash checks pass.
- Twenty-three generated JSON Schemas reproduce byte-for-byte; together with their manifest,
  the checked directory has 24 files. Schema logical manifest SHA-256:
  `e596dab7a6163f5a76f222c783aa996eee4501b509ad5a13cfc7c60534620198`.
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

## Phase state

| Phase | State | Verified position |
|---|---|---|
| 1 — contracts and GPU acceptance | In progress | Contracts, ledgers/CAS, storage and GPU controls, evidence/ontology boundary, provenance bridge, decoder projection, restart guardian, and recovery validation are implemented. The zero-GPU v4 incident is frozen; the outer-result identity repair and process-group hardening pass. Fresh v5 source association and remote validation-only admission are next. |
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
- Peak recorded v3 GPU VRAM: 22,525,509,632 bytes.
- Peak recorded v3 process RAM: 2,955,644,928 bytes.
- Remaining registered scientific generations: 258: four fallback acceptance calls,
  24 development calls, 168 held-out primary calls, 49 combined synthetic calls
  (12 paraphrase, nine feedback/interface, 28 ablation), and 13 case-study calls.
- Documentary v5 forecast pending fresh validation-only admission: 31,722.618681 seconds
  remaining; 677.381319 seconds scheduled reserve and 4,217.381319 seconds hard
  contingency. It must be reproduced from the checkpointed remote tree before launch.
- No new GPU call is admitted merely because platform credit changed. The registered
  9-hour scheduled ceiling and hard stop before 10 actual allocated hours remain binding.

## Gates and blockers

- Before any GPU service start: independently hash byte-identical local and remote source
  trees at checkpoint `48852dc61cfb6224779ed28ee11a20f2fe951882`, create the v5
  association and authorized overlay bound to the v4 incident, verify the remote
  ledger/model/runtime/storage state, and pass the CPU-only v5 preflight.
- Held-out inference is forbidden until the mandatory independent review and adjudication
  reproduce. No second review has been fabricated.
- Phase 6 requires the exact lawful local novel path only when synthetic work is complete.
- No destructive cache replacement, model switch, hidden-ontology shortcut, held-out gold
  access, or untracked foreground inference is authorized.

## Exact resume command

The tested v5 repair checkpoint is complete. Resume by generating its fresh local source
manifest:

```bash
PYTHONPATH=src /tmp/spo-refresh-venv/bin/python -m story_projection_onto.manifest \
  --root . --revision fallback-second-recovery-v5 \
  --output artifacts/public/manifests/source_tree_fallback_second_recovery_v5.local.json
```

Then back up every overwritten remote source file, synchronize the checkpointed source
inventory without `--delete`, independently generate the remote manifest, associate both
manifests with the full checkpoint SHA, build the append-only authorized v5 overlay, and
follow `docs/FALLBACK_SECOND_RECOVERY.md` from “Complete validation and execution argument
flow.” Do not execute the GPU launcher until the fresh validation-only preflight reports
admission and a final idle-GPU/ledger/storage audit passes.
