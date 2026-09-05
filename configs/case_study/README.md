# Restricted first-novel case configuration

`indexing.json` freezes only query-blind segmentation and FTS settings. It does
not contain a corpus path, novel text, windows, annotations, or query outputs.

The source path and enclosing restricted root must be supplied explicitly to
`scripts/prepare_novel_case.py`; both must remain under an ignored restricted
directory such as `.local_data/restricted/`. The script will not search for the
novel and requires an affirmative lawful-copy attestation. It writes the index
and its manifest only inside that restricted root and refuses to overwrite
existing files.

The four windows, eight bounded contexts, fixed per-window horizons, and one
separate operational FTS query are intentionally absent until they can be
preregistered from the lawfully supplied source. Their typed contract is
`CaseStudyPreregistration` in `story_projection_onto.novel_case`.

`runtime.json` freezes the production orchestration policy but contains no
corpus path or case result. It pins the concrete outer adapter factory and the
300-second operational service-start watchdog. `scripts/prepare_case_study_run.py compile-plan`
accepts only explicitly named files under one restricted root: the verified
index, its manifest, the four-window preregistration, and a later input
attestation that binds their exact hashes. A separate pre-case admission
attestation must bind the completed synthetic/review/fairness/metric/storage/GPU
and release gates, plus the raw file hash and canonical manifest hash of the
materialized selected-model freeze. These attestations are evidence of work
performed elsewhere; the compiler never manufactures or silently marks a gate
passed.

The eight gate files must implement the typed semantic records in
`case_study_runtime.py`. Before attestation, run
`scripts/prepare_case_study_run.py semantic-admission-bundle`; it rejects drift
among the held-out closure, benchmark, output inventory, selected model, metric
tables, cumulative ledger, GPU inventory/time total, storage snapshot, and
release scan. The attestation binds this bundle's canonical hash, so bare hashes
plus asserted pass booleans cannot admit the narrative phase.
The command also requires `--evidence-root`, the live `--ledger`, and a
restricted `--native-artifact-map` naming the exact role-to-path inventory.
Compilation replays the native benchmark, held-out journal, analysis/review,
firewall, feedback, storage, GPU, and release artifacts; placeholder hashes fail.
The exact native map also includes the completed descriptive community rubric under
the roles `blinded_community_rubric_template`,
`blinded_community_source_manifest`, `blinded_community_package`,
`blinded_community_rejoin`, `blinded_community_completion`,
`blinded_community_finalization`, and `blinded_community_table`. The source,
review package, and final bundle must retain their content-addressed canonical
layouts. Admission replays all producer bytes, the 12 condition-blind panels, the
four-condition paired design, all three registered Leiden resolutions, and the
C0-unseeded/LLM-seed-1 rule before accepting the narrative phase.

`configs/study/model.json` intentionally remains the original primary-model
runtime template because the common launcher validates that provenance even
when its allowlisted `model_candidate=fallback` switch is active. It is not the
case-study model-identity authority. The case compiler derives repository and
revision only from an explicitly supplied, internally hash-valid
`selected_llm_model_freeze`, verifies that identity against
`configs/study/fallback_model.json`, and verifies the freeze file/hash against
the pre-case admission. Thus the rejected 14B identity cannot silently enter a
case plan, and no fallback is treated as accepted before its real freeze exists.

The resulting restricted execution plan has exactly four query-blind C1 GPU
preconstructions, eight post-query C2 GPU constructions, eight credible C0
projections, and one separately labeled full-index FTS/BM25 C2 operational
demonstration. All 13 LLM slots use one frozen model/runtime and the low 31 bits
of frozen `llm_block_1`. Bounded packets are all-admissible, reused unchanged
across C0/C1/C2, and retained only in memory with prose-free hash receipts.

The runtime module provides strict adapter protocols rather than a hidden CPU
C2 implementation. Its append-only resume receipts require a pre-query barrier,
C1 construction seals, C2 empty inventories and construction certificates,
one-repair lineage, intention-to-treat failures, and exact evidence equality.
Review-template generation produces eight blank broad-review units and four
blank detailed assertion/event matching sections; no reviewer judgment is
inferred or prefilled. Plans, resume states, review material, raw attempts,
projections, the FTS index, packets, and paths remain restricted. Public output
is limited to hashes/counts and separately reviewed high-level paraphrases with
opaque evidence identifiers.

After all 25 ITT outputs terminate, a named human must populate the typed
`CaseStudyCompletedReview`. Validation requires all five dimensions for C0, C1,
and C2 in each of eight bounded contexts, detailed count-based matching in the
four preregistered contexts, and exact output hashes. Optional second-reader
disagreement is retained only for declared paper examples. Then
`scripts/compile_case_study_analysis.py` emits the canonical `novel_case` CSV:
descriptive scores/counts only, with the full-index query explicitly noncausal.
It cannot run without the lawful index, terminal model outputs, and real human
judgments, and it exports neither prose nor reconstructive offsets.
The CSV uses only a restricted `CaseStudyPublicAliasManifest` mechanically
derived from the identifiers sealed in the pre-query execution plan. The
mapping is sorted and versioned, and therefore offers no post-output alias
selection discretion.
It is written to the restricted target first and remains
`restricted_pending_canaries` unless an exact protected-prose canary manifest
is supplied, its corpus hash and every canary are verified against the exact
restricted source named by the sealed index manifest, and the public payload
passes the release scanner. Alias assignment is the deterministic ordering of
the plan's private identifiers, so it cannot be relabeled after outputs exist.

The executable controller boundary, concrete fail-closed GPU adapter/factory,
cumulative-ledger rule, one-load lifecycle, and still-required lawful restricted
run inputs are documented in `docs/PHASE6_EXECUTION.md`.

Once the eight admitted pre-case gate files exist,
`scripts/prepare_case_study_run.py stage-admission-evidence` copies their validated
JSON values into the same cumulative restricted CAS and writes a path-free typed
bundle/reference under the restricted root. This staging step performs no GPU work.
It requires the compiled plan and an explicit restricted transition directory. An
H0 snapshot captures the complete checkpointed predecessor ledger and shared-CAS
inventory before any write; an H1 receipt then proves an exact nine-payload delta
(the eight gates plus their bundle), with no mutation of predecessor rows or blobs.
Execution replays semantic admission against H0 and requires the live ledger to
equal H1. The shared CAS must be a real directory below the repository's ignored
`artifacts/blobs/` namespace; the ledger, transition records, and runtime directory
must be real, non-symlinked descendants of the explicit restricted root. A
process-scoped runtime lock rejects duplicate controllers before either can open the
ledger or construct a model service. Production activation and shutdown intents are
path-free immutable records; pointer histories and the service journal make the
one-load lifecycle recoverable across interruption boundaries.
