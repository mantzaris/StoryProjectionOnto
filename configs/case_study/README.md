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

The executable controller boundary, concrete fail-closed GPU adapter/factory,
cumulative-ledger rule, one-load lifecycle, and still-required lawful restricted
run inputs are documented in `docs/PHASE6_EXECUTION.md`.

Once the eight admitted pre-case gate files exist,
`scripts/prepare_case_study_run.py stage-admission-evidence` copies their validated
JSON values into the same cumulative restricted CAS and writes a path-free typed
bundle/reference under the restricted root. This staging step performs no GPU work.
The ledger SHA supplied to the production controller is measured only after staging.
The cumulative ledger, CAS, and runtime directory must all be real, non-symlinked
descendants of that explicit restricted root. Production activation and shutdown
intents are path-free immutable records; pointer histories and the service journal
make the one-load lifecycle recoverable across interruption boundaries.
