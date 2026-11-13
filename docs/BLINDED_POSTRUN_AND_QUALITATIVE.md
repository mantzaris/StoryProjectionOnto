# Blinded post-run review and qualitative production

`scripts/materialize_blinded_postrun_reviews.py` owns the two scorer-only human
review handoffs that occur after held-out outputs close. The input manifests, review
panels, rejoin maps, reviewer notes, and adjudication records must all live below one
explicit restricted root. Every intermediate path component is checked for symlinks.
Each materialization is content addressed, atomic, and append-only; its directory name
is the manifest content hash. Replay accepts an existing directory only when its
complete byte inventory is identical. Symlinks and non-regular filesystem objects are
rejected recursively, including objects that would otherwise be omitted by an
`is_file()` inventory.
Only the `reviewer/` subtree is distributed for annotation. The condition/world/seed
rejoin map is physically separated in the sibling `scorer_only/` subtree and must not
be exposed to the reviewer.

`error-source` deterministically replays the complete immutable Phase 4 index and
all primary held-out receipts, then enumerates every registered failure signal into
condition-neutral panels. The source manifest binds the exact 252-cell score
denominator and 288 primary output receipts; a partial but internally consistent
Phase 4 directory is rejected.

The `error-prepare` command applies
`configs/study/held_out_error_review_taxonomy.json`, assigns deterministic opaque item IDs, and
copies condition-neutral panels without world, context, seed, or condition metadata.
`error-complete` requires one externally authored judgment and one final adjudication
for every blinded failure. Its canonical CSV has one row per failed output and marks
`world` as the independent unit; contexts and seeds are never treated as samples.

`community-source` replays the same Phase 4 output root and applies the frozen,
quality-blind context rank. The first hash-ranked community-eligible context is fixed
before output validity or partition availability is inspected. It must supply the
same-context quartet spanning C0, C1, C2, and `A-FixedSelect`; an unavailable cell
fails the review source instead of causing selection of a later, easier context. LLM
cells use seed block 1 and deterministic C0 has no seed. The producer recomputes the
metric adapter and all registered half/base/double Leiden-CPM partitions from the exact
restricted projection, pinned metric configuration, nodes, and edges before accepting
the persisted Phase 4 objects.

The resulting 12 neutral JSON panels expose enough gold-free material to score the
registered semantic-coherence, interpretability, and evidence-support rubric: safe
context wording and scope, evidence text and provenance method, pseudonymized but
readable nodes and relations, temporal/epistemic qualifications, cited evidence,
confidence, and supported `why_matters` text. They expose no gold, expected effects,
condition/world/context/seed identity, raw source IDs, projection/partition hashes, or
resolution labels. Blinding scans every decoded key and string leaf, including embedded
identity substrings after Unicode normalization. Source/rejoin records containing the
hidden identities remain physically restricted and are never reviewer inputs.

The `community-prepare` command applies the frozen five-point rubric in
`configs/study/community_review_template.json`. `community-complete` requires one
rubric record per opaque partition before rejoining condition metadata in restricted
storage. The resulting canonical table records the three rubric scores and their mean,
while retaining `world` as the independent unit. Phase 6 admission binds source
selection, package, response, adjudication, and finalization bytes and requires review
completion/finalization before the gold-firewall gate, which must itself precede the
semantic bundle freeze.

All six modes support `--validate-only`, which performs every read, hash, lineage,
coverage, and panel check without writing. Omitting that flag creates (or exactly
replays) the content-addressed bundle below `--output-root`.

`scripts/materialize_qualitative_candidates.py` validates a restricted typed source
manifest, applies the six rules already frozen in `configs/study/reporting.json`, and
renders equal-size fixed-grid PNGs. A context's four condition panels must share the
same fixed-anchor hash; candidate lineage must include every panel and composite.
The output binds the candidate set, selected IDs/rule hashes, every panel hash, grid
order, and composite hash. Narrative candidates remain subject to the existing
paraphrase-only and opaque-evidence-only attestations.

For a public bundle containing a path component with `case` or `novel`,
`scripts/build_public_bundle.py` now requires both `--restricted-root` and
`--protected-prose-canary-manifest`. The restricted self-hashed manifest contains
canonical base64 exact-byte canaries and their hashes. Canaries are decoded only for
the scan and are never copied or printed; the public bundle records only the manifest
hash. Case publication fails closed when the manifest is absent, empty, malformed,
outside the root, symlinked, or does not bind its bytes.
