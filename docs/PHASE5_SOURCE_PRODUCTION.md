# Phase 5 source production

`scripts/materialize_phase5_sources.py` is the append-only bridge between a closed
held-out primary journal and `scripts/materialize_phase5_inputs.py`. It performs no
model call and does not author an ontology, feedback trace, or known answer.

The producer requires all of the following restricted inputs:

- the pre-output `Phase5ScriptCommitmentManifest` and its six CAS records;
- the closed `PrimaryHeldOutResultsGate`, copied execution manifest, scorer bridge,
  complete held-out journal, CAS, and SQLite ledger;
- the durable query-blind C0 preparation-state directory;
- one explicitly authored six-entry `Phase5KnownAnswerSourceManifest` whose review
  hashes match the execution and whose timestamp precedes the script commitment;
- exactly three `RevisionInstruction` files and their three append-only
  `/api/revisions` submission receipts, one pair for each registered researcher
  trace slot. Each receipt embeds and hashes the canonical UI request; the CLI
  replays that request into the exact instruction and never creates defaults.

`--restricted-root` must resolve exactly to the repository's
`artifacts/restricted` directory; naming an arbitrary private-looking directory is
not sufficient to establish the registered release boundary.

The output contains 21 seed-1 held-out projection exports (C0/C1/C2 for six scripts
and C2 for three traces), 12 C0/C1 CPU reprojection inputs, one 18-entry hash-only
scorer authorization, and one nine-episode source manifest. Synthetic packets and
snapshots reuse their existing public CAS bytes. Instructions, projections, exports,
CPU inputs, scorer bindings, and manifests remain restricted.

For `REFINE_CONTEXT`, C0 and C1 run their frozen projector over the identical sealed
preontology and may emit only `selection` decisions. `REQUEST_MERGE_SPLIT` returns an
explicit `capability_limited` CPU input with no after projection. The current bounded
same-packet runner rejects spoiler-horizon changes; lens and story-time refinements
remain supported. This prevents later-horizon evidence from entering a narrower
revision packet implicitly.

An invalid, failed, or timed-out selected held-out parent is retained in the primary
ITT journal but cannot serve as a feedback before-projection. The producer raises
`Phase5SelectedParentUnavailable` with its episode, condition, outcome, and exact
receipt/ITT hash before writing any Phase 5 source artifact. It never substitutes an
empty or synthetic ontology.

Validate first, then rerun the same command with `--materialize`:

```bash
python scripts/materialize_phase5_sources.py \
  --repository . \
  --restricted-root artifacts/restricted \
  --script-commitment <restricted-commitment-manifest> \
  --primary-results-gate <restricted-primary-results-gate> \
  --held-out-journal-root <restricted-held-out-journal> \
  --known-answer-source <restricted-known-answer-source> \
  --trace-instruction researcher-trace-easy=<restricted-instruction-file> \
  --trace-instruction researcher-trace-medium=<restricted-instruction-file> \
  --trace-instruction researcher-trace-hard=<restricted-instruction-file> \
  --trace-submission-receipt researcher-trace-easy=<restricted-receipt-file> \
  --trace-submission-receipt researcher-trace-medium=<restricted-receipt-file> \
  --trace-submission-receipt researcher-trace-hard=<restricted-receipt-file> \
  --c0-state-root <restricted-c0-state-directory> \
  --ledger <restricted-ledger> \
  --artifact-root <restricted-cas-root> \
  --validate-only
```

The first successful materialization writes `production_session.json` before all
other output records. That session freezes the observed access/start/completion
times and every source-file hash, so recovery replays byte-for-byte without
backdating. Every subsequent record is canonical, mode `0600`, atomically linked,
and append-only. Interrupted temporary files are preserved and ignored as evidence,
while unexpected or out-of-order published files fail closed.

## Capturing the three researcher requests

After the primary held-out gate is closed, validate the exact seed-1 C2 parent
selection without starting a model service:

```bash
python scripts/serve_phase5_trace_capture.py \
  --repository . \
  --restricted-root artifacts/restricted \
  --primary-results-gate <restricted-primary-results-gate> \
  --held-out-journal-root <restricted-held-out-journal> \
  --capture-root artifacts/restricted/phase5_trace_capture \
  --ledger <restricted-ledger> \
  --artifact-root <restricted-cas-root> \
  --validate-only
```

Rerun with `--serve` to bind the interface to localhost. The factory loads only
the three registered C2 before bundles and maps them to their registered episode
IDs. A real `POST /api/revisions` writes, in order, canonical
`<episode>.submission.json`, `<episode>.instruction.json`, then
`<episode>.receipt.json`, all mode `0600` below canonical restricted storage. The
receipt records the endpoint, action, actual request/record timestamps, request,
instruction, before-projection, and before-bundle hashes. This route is
capture-only: it has no runner and cannot start regeneration. Exact retries are
idempotent; rebinding an episode or using a symlinked/out-of-bound destination
fails closed.

## Scorer-only compilation

After the complete 21-record feedback journal is closed, compile the 18 scripted
known-answer rows in the scorer-only namespace:

```bash
python scripts/score_phase5_feedback.py \
  --repository . \
  --restricted-root artifacts/restricted \
  --source-manifest <restricted-phase5-source-manifest> \
  --known-answer-source <restricted-known-answer-source> \
  --feedback-journal-root <restricted-closed-feedback-journal> \
  --ledger <restricted-ledger> \
  --artifact-root <restricted-cas-root> \
  --output-root artifacts/restricted/phase5_feedback_scoring \
  --validate-only
```

The known-answer manifest must resolve each before/after reviewed gold projection
and each target-change rule through typed restricted CAS references. The compiler
replays every journal byte/logical hash and calls `score_scripted_feedback` once
for every script-condition pair. CPU merge/split limitations retain projection
gold endpoints as `NA` while recording the registered target as unrealized. A
failed, timed-out, or invalid C2 regeneration has no fabricated
after projection or diff; intention-to-treat scoring records the real before
endpoint and a zero after endpoint against nonempty after-context gold, with zero
target realization. Researcher traces remain excluded from gold scoring. Rerun
with `--materialize` to write the append-only session, canonical metrics, and
source-lineage receipt.
