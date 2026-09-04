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
