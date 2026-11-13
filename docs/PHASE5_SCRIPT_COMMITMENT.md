# Phase 5 scripted-revision commitment

The six known-answer revisions must be committed after the independent-review
completion and post-development runtime binding reproduce, but before the held-out
execution journal contains any record. The command first validates only the
restricted hash-only review completion manifest and final reviewed seal. Only then
does it open the frozen feedback protocol, researcher-authored restricted draft,
runtime-bound public held-out plan, and six public model-visible query stages. It
has no condition-output or scorer-gold input and does not open reviewed gold
artifacts.

Each draft declares a future `scheduled_activation_at`, which is copied to the
typed `UserRevision.created_at`. The command records the actual wall-clock
`committed_at`; it never accepts a caller-supplied commitment timestamp. This
separates honest pre-output authorship from the later time at which the committed
revision is applied to completed projections.

Validate without writing:

```bash
python scripts/freeze_phase5_scripted_revisions.py \
  --repository . \
  --draft-manifest artifacts/restricted/phase5_script_draft.json \
  --review-completion-root \
    artifacts/restricted/scorer_only/independent_review \
  --held-out-control configs/study/held_out_primary.json \
  --held-out-runtime-binding \
    artifacts/restricted/held_out_primary/runtime_binding.json \
  --held-out-journal-root artifacts/restricted/held_out_primary \
  --ledger artifacts/restricted/study.sqlite3 \
  --artifact-root artifacts/blobs/phase5 \
  --restricted-root artifacts/restricted \
  --validate-only
```

Replace `--validate-only` with `--materialize` to publish the six instruction and
freeze records to the restricted CAS and atomically create
`artifacts/restricted/phase5_script_commitment/commitment_manifest.json`. Repeating
the command accepts only byte-identical materialization; it never overwrites a
prior commitment.

Run this command immediately before the held-out launch, not during development
or review. The scheduled activation must be strictly later than the registered
held-out watchdog floor reproduced from the exact 168-call order: 23,040 base-call
seconds, 2,520 repair-reserve seconds, and 900 seconds for three service starts,
for a 26,460-second floor. The registered study does not give the final C0/C1 CPU
projection tail a watchdog, so the commitment deliberately does not claim that
26,460 seconds is a wall-time upper bound. It records this limitation and the
post-held-out materializer rejects the episode unless all selected C0, C1, and C2
parents actually completed before the precommitted activation. Do not backdate a
commitment or activation to repair a missed window; preserve the attempt and make
a new prospective commitment if the execution has not started.

Post-held-out Phase 5 materialization requires both the commitment file and its
exact hash in `Phase5MaterializationSourceManifest`:

```bash
python scripts/materialize_phase5_inputs.py \
  --source-manifest artifacts/restricted/phase5_source_manifest.json \
  --script-commitment \
    artifacts/restricted/phase5_script_commitment/commitment_manifest.json \
  --primary-results-gate artifacts/restricted/primary_results_gate.json \
  --ledger artifacts/restricted/study.sqlite3 \
  --artifact-root artifacts/blobs/phase5
```

The materializer replays the manifest and rejects any episode whose protocol slot,
context, action, public query/evidence hashes, instruction CAS reference, freeze CAS
reference, activation time, or pre-output chronology differs from the commitment.
The resulting execution manifest and receipt retain the commitment hash.

Drafts, commitments, constructed/instruction CAS payloads, and execution inputs are
restricted artifacts and must not be copied into the public bundle. Canonical
synthetic evidence packets and snapshots may retain their existing public CAS
classification; the Phase 5 reader verifies that classification instead of
re-registering identical bytes as restricted. The command refuses noncanonical
restricted roots, symlinked paths, a broad blob root, or any held-out journal that
has already started.
