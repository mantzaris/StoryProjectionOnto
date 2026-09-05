# Run status

Updated: 2026-09-05 14:15 UTC

## Verified durable state

- Branch: `implementation/query-dependent-temporal-ontology`.
- Last immutable predecessor commit: `cadfb25c39e4f0b1998ab15ad89dcc5e97d1fbf1`.
- Recovery checkpoint: pending the immediate commit of this fully validated tree; its full
  SHA will be recorded in a status-only follow-up commit before remote source association.
- Local project: `/home/resort/Documents/repos/StoryProjectionOnto`.
- Remote project: `/workspace/StoryProjectionOnto`.
- The reconciled canonical Phase 1 ledger is schema v8, file SHA-256
  `38775d1fe3c27cb93afbf78f3c692300a05ed52cb083cb4656d0378b0b94429f`.
  Its read-only ledger/CAS audit passes with 17 artifacts, one failed attempt/model call,
  five closed GPU events, four closed service sessions, and no unresolved allocation.
- The stale 422.961986-second local ledger remains preserved in ignored recovery storage;
  it was not discarded or used for admission.
- Four interrupted remote quarantine directories were packed losslessly before their exact
  unpacked copies were removed. The retained local and remote archive has 94,138 members,
  file SHA-256
  `617a13b5dabf7376f342550d217168d705ab328f0517cf44d663525c6562d721`,
  and passed full decompression/member-count validation.
- The latest read-only remote audit at 2026-09-05 14:08 UTC found one NVIDIA RTX 4090
  (24,564 MiB), 1 MiB used, 0% utilization, no study/vLLM process, no `tmux` or `screen`
  session, and no listener on port 8000. vLLM is stopped.
- Current remote project-controlled occupancy is 16,111,649,792 bytes. This is
  8,888,350,208 bytes below the 25 GB occupied limit and preserves more than the required
  5 GB headroom inside the registered 30 GB allocation.
- Pinned permitted fallback model: `Qwen/Qwen3-8B-AWQ`, exact revision
  `4da05a8edb55c6046cce958586c33b61da07bb79`.

## Validation state

- Complete collected test inventory: 1,089 tests.
- Unit suite: 1,001 passed in 337.17 seconds.
- Property suite: 16 passed in 47.74 seconds.
- Integration suite: 69 passed and three expected environment skips across 72 tests.
  The skipped local checks are two XGrammar checks (package absent locally) and one pinned
  tokenizer-snapshot check (remote-only snapshot). The renderer, real-browser, and six API
  interface tests were rerun with loopback access and all eight passed.
- Aggregate verified result: 1,086 passed, three expected local-environment skips, zero
  implementation failures.
- Ruff, Python compilation, JSON parsing, `git diff --check`, package/CLI import smoke,
  report ingestion replay, report replay, and public-release hash checks pass.
- Twenty-two JSON Schemas reproduce byte-for-byte. Schema logical manifest SHA-256:
  `ec99077520896fa0ded7bbeebd2a24bd0dc255f54e5e1765751e74130757a9f1`.
- Synthetic benchmark verify-only and the semantic refresh guard pass. Benchmark logical
  manifest SHA-256:
  `c5bce978a9006b67f13c40701fca2d7b558d62239615c49a2c035221cd1b723f`.
  Refresh receipt SHA-256:
  `2396c3a7c308a964b348022a6811cde1dc0b0f1d4af05f3838c68a73bc6f9696`.
- The report-ingestion receipt verifies at
  `004b2e7ace7c75d1fcabd3f59a7f5686e1f088c4038db1f07ef6a5fc6a25bfef`.
  The interim report remains explicitly incomplete and contains no fabricated efficacy
  results. The current public bundle input manifest has 37/37 matching leaves and logical
  SHA-256 `fbc729b15a3bf5118ecfc2caade545dc615e63464269d757786ca1dc91bb13d0`.

## Phase state

| Phase | State | Verified position |
|---|---|---|
| 1 — contracts and GPU acceptance | In progress | Contracts, ledgers/CAS, storage and GPU controls, evidence/ontology boundary, provenance bridge, decoder projection, restart guardian, and recovery validation are implemented. Fresh v4 source association and remote validation-only admission are next. |
| 2 — synthetic benchmark | Software/data complete; independent review pending | Four development worlds, 12 held-out worlds, 36 primary contexts, contrastive pairs, rare-pivotal/temporal/epistemic gold, mutation tests, and the condition-blind 3-world/9-projection review package reproduce. |
| 3 — conditions and primary run | Software complete; execution pending | C0, C1, C2, and A-FixedSelect pathways and timing/capability/equal-evidence gates pass. The 24 development and 168 held-out calls remain. |
| 4 — metrics and ablations | Software complete; execution pending | Registered metrics, world-level inference, 4,096 sign flips, Holm correction, bootstrap sensitivity, community analysis, and three reduced ablations are implemented. |
| 5 — interface and feedback | Software complete; execution pending | Cytoscape interface, two registered actions, six scripted revisions, three trace captures, fixed anchors, and provenance-aware before/after views pass local tests. |
| 6 — one-novel case study | Software complete; lawful input pending | Query-blind indexing, window/C1/C2 control planes, restricted storage, and descriptive analysis are implemented. No novel inference has run. |
| 7 — results and release | Interim only | Immutable-table reporting, visual inspection, accounting, and public-release controls pass. Final numerical outputs await registered execution. |

## GPU and call accounting

- Actual allocated GPU time consumed: 815.215409 seconds.
- Recovered unattended-allocation uncertainty: zero seconds.
- Peak recorded v3 GPU VRAM: 22,525,509,632 bytes.
- Peak recorded v3 process RAM: 2,955,644,928 bytes.
- Remaining registered scientific generations: 258: four fallback acceptance calls,
  24 development calls, 168 held-out primary calls, 49 combined synthetic calls
  (12 paraphrase, nine feedback/interface, 28 ablation), and 13 case-study calls.
- Documentary v4 forecast pending fresh validation-only admission: 31,722.618681 seconds
  remaining; 677.381319 seconds scheduled reserve and 4,217.381319 seconds hard
  contingency. It must be reproduced from the checkpointed remote tree before launch.
- No new GPU call is admitted merely because platform credit changed. The registered
  9-hour scheduled ceiling and hard stop before 10 actual allocated hours remain binding.

## Gates and blockers

- Before any GPU service start: commit this tree, independently hash byte-identical local
  and remote source trees, create the v4 association and authorized overlay, verify the
  remote ledger/model/runtime/storage state, and pass the CPU-only v4 preflight.
- Held-out inference is forbidden until the mandatory independent review and adjudication
  reproduce. No second review has been fabricated.
- Phase 6 requires the exact lawful local novel path only when synthetic work is complete.
- No destructive cache replacement, model switch, hidden-ontology shortcut, held-out gold
  access, or untracked foreground inference is authorized.

## Exact resume command

If interrupted before the checkpoint is recorded, rerun the complete local CPU gate:

```bash
PYTHONHASHSEED=0 /tmp/spo-refresh-venv/bin/python -m pytest -q tests/unit tests/property
```

Then rerun the 72 integration tests, granting loopback access only to the renderer,
system-browser, and UI smoke files. On success, create the recovery checkpoint and follow
`docs/FALLBACK_SECOND_RECOVERY.md` from “Complete validation and execution argument flow”
using revision label `fallback-second-recovery-v4`. Do not execute the GPU launcher until
the fresh validation-only preflight reports admission.
