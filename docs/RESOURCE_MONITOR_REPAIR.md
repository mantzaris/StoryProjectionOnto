# CPU-only resource monitor and shutdown repair

2026-09-07. No model service start, inference, budget change, or scientific
validator change was performed for this repair. Fourth-start evidence and all
allocation records remain authoritative. The next start is **not authorized**.

## Measured stopped-pod costs

The three restricted `resource-component-profile*.json` records include UTC and
monotonic timestamps. Each small component was measured three times per profile;
these are observations, not p95 estimates or loaded-service throughput.

| Component | Observed duration |
|---|---:|
| Process enumeration + RAM (two enumerations per profile row) | 0.14–0.35 ms |
| GPU query including nearby process enumeration | 16.5–31.5 ms |
| Read-only canonical ledger query | 1.9–90.3 ms |
| Status-file read | 2.2–22.5 ms |
| 1,000 uncontended in-process lock cycles | 0.096–0.171 ms |
| Existing controller file-lock acquire/release | 0.84–3.49 ms |
| Old recursive storage walk | 49.884 s; 50.213 s |
| Corrected standalone full traversal including directories | 41.410 s |
| Repaired full census, including storage-worker startup | 9.620 s; 10.223 s |
| Repaired fast observation, including first fast-worker startup | 201.6 ms; 224.4 ms |
| Subsequent fast observations | 39.2–122.4 ms (four observations) |
| Cancel and reap both idle observer workers | 16.5–16.6 ms |

The old walk counted files but omitted directory blocks. The repaired census
includes directories and symlinks, deduplicates hard links, and measured
16,442,173,952 occupied bytes initially. The final source-bound standalone census
measured 16,443,250,688 bytes; subsequent writes were retained by the event index.
The old approximately 10.36 GB samples
are retained as historical undercounts, not rewritten. Both the native monitor
and `StoragePreflight` now include directory occupancy. The 25 GB occupied,
30 GB planned allocation and protected 5 GB headroom gates are unchanged.

These stopped-pod measurements identify an unnecessarily expensive sampling
path. They **do not establish that storage traversal caused slow model loading**.
No GPU throughput improvement or complete-study forecast reduction is credited.
Locks were uncontended; contention cost is not established by these timings.

## Implemented behavior

`ResourceSampler.prepare()` performs a bounded full census before allocation.
The separate storage worker uses an allocated-block index and inotify events
for project writes, including atomic replacements, truncations and hard links.
Each observation drains that event stream and refreshes filesystem headroom.
It records baseline time, current observation time, write-event count, occupied
bytes and change from baseline. It is not a size prediction or a TTL cache.
Overflow, missing watches, ambiguous directory moves, device changes or excessive
backlog invalidate the index and fail closed. External writers on another host
are not an allowed live-storage assumption: the index covers this pod's kernel
events; deployments and other bulk changes occur while stopped, followed by a
new census. A new storage checkpoint is required after event-stream loss.

The fast worker takes process/RAM observations before and immediately after its
GPU query, checks the root's birth ticks, and timestamps each component. GPU
queries have a 2 s subprocess timeout; fast RPC has a 3 s deadline; storage-update
RPC has a 1 s deadline. A combined observation older than 5 s fails. There is no
filesystem traversal between process identity and GPU observation. Full scans
occur at initial admission and bounded stopped-service checkpoints; the diagnostic
guardian performs a final census only after physical shutdown and reconciliation.
Same-controller model restart preserves the storage event stream; a fresh
controller must establish a new bounded storage checkpoint before inference.
Required restart/resume acceptance is retained and remains uncompleted on GPU.

Workers have separate owned sessions, immutable-name restricted diagnostic
journals, parent-death protection and no SQLite connection. Complete observations
and exception details are fsynced privately before IPC delivery. The watchdog
queues observations; the controller alone commits resource samples, with bounded
checkpoint synchronization. No sampling worker competes for the ledger.

Shutdown sets stop flags and signals the verified owned service **before**
draining observers or joining heartbeat writers. Exact PID birth, process-group,
session and token ownership checks remain. SIGTERM, bounded grace, SIGKILL and
absence verification share one absolute deadline. Cancellation does not acquire
the sample/RPC lock. Probe processes are killed/reaped; then threads are joined
and accounting is reconciled. The verified physical-stop timestamp is retained
even if later CPU accounting fails. Such failure keeps the lease unresolved,
rather than claiming a clean shutdown or erasing allocation.

CPU fault injection SIGSTOPs a real production storage-probe process during a
fake service's live monitor request. Assertions require the service signal in
less than 0.3 s, complete cleanup in less than 1 s, no live probe/thread, retained
diagnostics, and no unresolved fixture allocation or service journal. Separate
tests cover RPC timeout/reaping, event loss, identity changes, storage/headroom
failure and restart/checkpoint continuity. These are CPU tests, not a claim of
measured live vLLM shutdown latency.

Final verification: **179 focused tests passed locally and 179 on the pod**;
**31 unchanged C0/alignment tests passed locally**. The classification reproduced
byte-for-byte. Lint and whitespace checks pass. The first remote collection
attempt lacked the unchanged streaming fixture (zero tests run); both that
record and the successful rerun are retained. Deployed source/test checksums
match the local source checkpoint `00db0fa`; nothing was pushed.

Restricted backup: `artifacts/restricted/resource-monitor-repair.V995HH/`.
Canonical ledger SHA-256 remains
`cae5d0dacd6382d68603d19acdfef155ee2c3c3ccf96599dbe0ada875514faff`.

## Single proposed next envelope — approval required

Keep the exact small evidence-only streaming task and success criteria in
`docs/SMALL_STREAMING_DIAGNOSTIC.md`; no full C1 or second call. Input remains
3,453 template-inclusive tokens plus 6,144 output tokens, within 12,288 total.
Request SHA-256:
`cde6c6b6eedefab00aa46b7a01833998ca0b291ad8ff1daf55e574ad23681e7a`.

| Stage | Maximum allocated seconds |
|---|---:|
| Startup/readiness | 360 |
| Live controller checks | 15 |
| One small SSE diagnostic, including timeout cancellation | 120 |
| Validation/exception drain | 15 |
| Protected verified shutdown | 45 |
| Whole-start scheduling guard | 5 |
| **Whole start, all gaps included** | **560** |

360 s startup is an estimate: earlier successful starts were about 193–202 s,
but the latest was not ready by 240 s while loading weights. Slow loading remains
unresolved. The next start would retain loading logs and fresh RAM/GPU samples,
request receipt, observable scheduling/generation logs, first SSE event/content,
received fragments, cancellation and verified shutdown. Unobservable per-request
grammar preparation remains unknown. 120 s permits a bounded observation, not a
promise that the model completes. Pre-allocation census and final stopped-service
census are CPU-only; any preparation performed while live counts inside 560 s.

Historical block use is **1,405.975189 s**, leaving **394.024811 s** of 1,800 s.
This proposal requires one fifth cumulative start and **165.975189 s** additional
block authority: `1,405.975189 + 560 = 1,965.975189 s`. It is not another 550 s
start squeezed into the old remainder, nor a fresh block. At most one actual call
would increase cumulative attempts from three to four, still within five.

Global actual allocation remains **4,633.801513 s**; the proposed maximum would
be **5,193.801513 s**, below unchanged 33,660 scheduled and strict 36,000 actual
ceilings. The diagnostic-only complete-forecast exception would still be needed.
Ordinary admission still fails: remaining mandatory proxy **40,162.013213 s**,
all-in **44,795.814726 s**, scheduled deficit **11,135.814726 s**. There are no
new valid scientific generation timings. All mandatory calls, repairs, loads and
restart/resume requirements remain. No full-study budget change is proposed here.

The existing executable still rejects the exhausted four-start limit. The
proposed envelope has deliberately not been activated or launched.
