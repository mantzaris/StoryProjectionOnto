# First-novel runtime attestations

These records must be created in the ignored restricted case directory after
the underlying checks have actually occurred. They are not example pass
results, and this repository intentionally ships no populated attestation.

`CaseStudyInputAttestation` binds the exact SQLite index SHA-256, raw manifest
file SHA-256 plus its canonical content hash, and raw preregistration file
SHA-256 plus its canonical content hash. It also records the person and time who
attested a lawful copy and output-blind window selection. The attestation must be
later than the preregistration.

`CaseStudyAdmissionAttestation` contains hash references to completed synthetic
closure, timing-lineage, gold-firewall, registered-metric regeneration, blinded
error review, storage headroom, GPU schedule, public-release scan, selected
model freeze, validator/upper-ontology freeze, and C1/C2 capability evidence.
Every Boolean is a required literal `true`; a missing or failed gate must remain
missing/failed and blocks plan compilation.

The semantic bundle's exact native inventory additionally binds the frozen
community rubric template and the source manifest, blinded package, scorer-only
rejoin map, named-reviewer completion, finalization, and canonical 12-row table.
These seven `blinded_community_*` roles are descriptive evidence, not a ninth
confirmatory gate, but any missing role, altered byte, identity-bearing reviewer
panel, or incomplete condition/resolution quartet blocks semantic admission.

The eight referenced files are not arbitrary JSON. They must validate as the
eight `Case*Gate` contracts and compile into one
`CaseStudySemanticAdmissionBundle`. The bundle enforces shared held-out closure,
benchmark, output-inventory, selected-model, metric-table, ledger, GPU-inventory,
allocated-time, and release-scan lineage. Record its `content_hash` as
`semantic_gate_bundle_hash`, along with the exact held-out closure, cumulative
ledger, and GPU-event inventory hashes in the admission attestation.

The admission must include both `selected_model_freeze_file_sha256` (the exact
bytes of the explicitly supplied freeze/result JSON) and
`selected_model_freeze_hash` (the freeze's internally verified
`manifest_sha256`). The compiler accepts either a standalone freeze object or a
result object with a `selected_model_freeze` member, but it requires the actual
file and verifies the accepted fallback repository/revision and gate-lineage
fields against the tracked single-fallback policy. A bare hash is insufficient.

Use `scripts/prepare_case_study_run.py contract-hashes` to record the current
schema identities. Never place the corpus path, source prose, packet contents,
offsets, model responses, or reviewer judgments in a tracked configuration.

The repository intentionally provides no populated `CaseStudyCompletedReview`.
That restricted record must be supplied by named human reviewers after all 25
ITT outputs terminate; software must never prefill or infer its verdicts.
