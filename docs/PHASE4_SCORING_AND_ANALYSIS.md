# Phase 4 scorer and registered analysis

`scripts/run_phase4_analysis.py` is the only production bridge from the frozen
synthetic execution journals to held-out gold. It runs on the CPU, lives under the
`scorer_only` namespace, never constructs a model request, never opens the model
runtime, and never changes the intended experimental denominator after observing an
output.

## Inputs and firewall

The command requires the immutable held-out call and execution manifests, the closed
scorer bridge, the 40 ordinary calls from the combined execution index, the completed
independent-review tree, the scorer-only benchmark tree, the shared CAS and SQLite
ledger, and a current local/remote source-tree association. The association is
revalidated against the current source tree. Review completion is reproduced from its
response, adjudication, reviewed artifacts, and final seal; a status flag is not
sufficient.

The primary intended manifest contains exactly 252 cells: 36 C0, 72 C1, 72 C2, and
72 `A-FixedSelect`. The combined intended manifest contains exactly 40 ordinary
paraphrase and reduced-ablation cells. Terminal failures remain in both manifests.
Invalid, timed-out, interrupted, or unrepaired outputs receive the registered ITT
semantic zeros when gold is nonempty; structural quantities remain `NA`.

Runtime modules do not import this module. Gold projections, permissible alternatives,
rare/pivotal labels, and community assignments enter only after the runtime-closed
scorer bridge has been verified. Scorer plans and grounding audits are stored as
restricted artifacts. A scorer-side grounding verdict requires an exact qualified
signature—including story time, validity, epistemic holder/attitude, and narrative
commitment—and only packet-admissible citations registered as support for that exact
signature; an unrelated in-packet citation makes the assertion unsupported. Each
valid complete score bundle also embeds a restricted,
gold-free replay record containing the exact scored projection, query context, and
evidence packet. Read-only Phase 4 admission reruns the grounding audit and complete
metric pipeline from those inputs and exact-compares the bundle and intended-cell
score; it also regenerates every invalid ITT score from its frozen failure kind,
failure-artifact hash, allocated GPU seconds, scorer plan, and metric configuration.
Thus a coordinated edit to rows and their linking hashes cannot pass merely because
the edited files remain internally self-consistent.

## Geometry gate

Visual clutter is measured from renderer output, never estimated from string length or
invented by the scorer. For every structurally valid successful projection, provide one
`GeometryMetricInput` at:

```text
<geometry-root>/<projection-content-hash>.json
```

The record must bind the projection hash and the frozen layout, style, font, viewport,
visibility, label rectangles, and rare-pivotal discoverability trace. Preflight lists
only the missing count in its public stdout summary. It refuses the analysis until all
required geometry exists and `materialization_receipt.json` reproduces the successful
source-result inventory. Invalid or empty projections do not receive favorable geometry
values.

The confirmatory renderer source is the immutable `registered_display` view, not the
raw full-output inspection view and not a post-hoc support-filtered view. The shared
selection rule ranks assertions by contextual relevance and confidence with canonical
hash/ID tie breaking, preserves endpoint closure, then fills unused node slots by the
same condition-neutral rule. It must materialize exactly
`budget_accounting.display_nodes_used` and
`budget_accounting.display_assertions_used`; those counts must equal the emitted graph
counts capped by the common display budgets. An impossible endpoint-closed declaration
fails the geometry gate. `full_structural` remains available for raw intention-to-treat
inspection, but cannot enter registered clutter. Likewise a scorer/reviewer overlay can
never enter confirmatory geometry: unsupported omissions therefore cannot earn lower
clutter or better discoverability.

Geometry is produced through the real vendored Cytoscape browser renderer, not by the
scorer. First prepare the restricted, post-execution capture inputs; this joins relevance
and rare-pivotal labels only after the model runtime is closed. Then validate, capture,
and replay the output (timestamps are explicit for byte-exact recovery):

```bash
python scripts/materialize_phase4_geometry.py --prepare \
  --prepared-at 2026-09-04T18:00:00Z \
  --source-root "$GEOMETRY_SOURCE_ROOT" \
  --held-out-root "$HELD_OUT_ROOT" --scorer-bridge "$SCORER_BRIDGE" \
  --combined-root "$COMBINED_ROOT" --review-root "$REVIEW_ROOT" \
  --ledger "$STUDY_LEDGER" --artifact-root "$STUDY_CAS" \
  --source-association "$SOURCE_ASSOCIATION"

python scripts/materialize_phase4_geometry.py --validate-only \
  --source-root "$GEOMETRY_SOURCE_ROOT" --geometry-root "$GEOMETRY_ROOT"

python scripts/materialize_phase4_geometry.py --run \
  --source-root "$GEOMETRY_SOURCE_ROOT" --geometry-root "$GEOMETRY_ROOT" \
  --completed-at 2026-09-04T18:15:00Z

python scripts/materialize_phase4_geometry.py --replay \
  --source-root "$GEOMETRY_SOURCE_ROOT" --geometry-root "$GEOMETRY_ROOT"
```

The capture fixes a 1200x800 unit-scale viewport and 14px
`system-ui,sans-serif` node, edge, and n-ary-hub typography. It waits for browser font
readiness and records the effective styles and font probe. N-ary hubs receive distinct,
stable preset positions derived from their role nodes and assertion identity; hub and
role labels retain separate rectangles rather than counting the empty area between
them. A content-addressed copy accompanies each required
`<projection-hash>.json` file. Before either Phase 4 validation or scoring, the scorer
replays the restricted source manifest, every renderer DTO, every raw capture, every
source-entry/raw hash, the public geometry, its content-addressed copy, and the receipt.
The receipt explicitly records `world` as the independent confirmatory unit;
per-projection/context geometry rows are observations, never independent samples. No
command in this geometry sequence starts or calls the model.

## Commands

Run the non-inference preflight first, substituting the durable run directories:

```bash
python scripts/run_phase4_analysis.py --validate-only \
  --held-out-root "$HELD_OUT_ROOT" \
  --scorer-bridge "$SCORER_BRIDGE" \
  --combined-root "$COMBINED_ROOT" \
  --review-root "$REVIEW_ROOT" \
  --benchmark-root data/synthetic \
  --ledger "$STUDY_LEDGER" \
  --artifact-root "$STUDY_CAS" \
  --source-association "$SOURCE_ASSOCIATION" \
  --geometry-source-root "$GEOMETRY_SOURCE_ROOT" \
  --geometry-root "$GEOMETRY_ROOT" \
  --output-root "$PHASE4_OUTPUT"
```

After `ready` is true, use one recorded timezone-aware timestamp. Reuse the exact same
timestamp if an interrupted append-only run is resumed:

```bash
python scripts/run_phase4_analysis.py --run \
  --completed-at 2026-09-04T18:30:00Z \
  --held-out-root "$HELD_OUT_ROOT" \
  --scorer-bridge "$SCORER_BRIDGE" \
  --combined-root "$COMBINED_ROOT" \
  --review-root "$REVIEW_ROOT" \
  --benchmark-root data/synthetic \
  --ledger "$STUDY_LEDGER" \
  --artifact-root "$STUDY_CAS" \
  --source-association "$SOURCE_ASSOCIATION" \
  --geometry-source-root "$GEOMETRY_SOURCE_ROOT" \
  --geometry-root "$GEOMETRY_ROOT" \
  --output-root "$PHASE4_OUTPUT"
```

Run this command in the persistent study `tmux` session and tee stdout/stderr to the
timestamped study log. It performs no GPU call and must not be counted as GPU
allocation.

## Deterministic outputs

Every file is write-once: an exact replay is accepted, while any byte drift is refused.
Files are also stored in the shared compressed CAS and metric rows are linked to the
restricted Phase 4 study in the SQLite ledger. `analysis_index.json` binds all input
manifests, the reproduced review seal, the current source tree, the metric version,
every output hash, and the completion timestamp.

`table_manifest.json` is the self-hashed reporting contract. Each entry has a stable
table ID, path, ordered columns, row count, physical SHA-256, logical hash, release
class, independent unit, and a `selection_ready` flag. In particular,
`unit_metric_observations`, `primary_world_metrics`, `contrast_pair_scores`,
`paraphrase_pairs`, and `ablation_pairs` can be consumed without hand-transcribing
numbers or choosing examples from rendered figures.

The report-facing CSV contracts are:

| File | Ordered columns |
|---|---|
| `tables/comparisons.csv` | `comparison`, `metric`, `estimate`, `paired_median`, `standardized_effect`, `ci_lower`, `ci_upper`, `p_value`, `adjusted_p_value`, `sign_flip_p_value`, `bootstrap_lower`, `bootstrap_upper`, `paired_world_count`, `status`, `analysis_family`, `p_value_kind`, `multiplicity_method`, `independent_unit` |
| `tables/unit_metrics.csv` | `source_block`, `unit_id`, `condition`, `world_id`, `context_id`, `seed_block`, `output_valid`, `projection_id`, `unit_hash`, `metric_name`, `metric_status`, `value`, `numerator`, `denominator`, `metric_version_hash` |
| `tables/world_metrics.csv` | `condition`, `world_id`, `metric_name`, `value`, `context_count`, `row_count`, `undefined_context_count` |
| `tables/contrast_metrics.csv` | `world_id`, `before_context_id`, `after_context_id`, `condition`, `seed_block`, `unit_hash`, `metric_name`, `metric_status`, `value`, `numerator`, `denominator`, `metric_version_hash` |
| `tables/contrast_world_metrics.csv` | same columns as `world_metrics.csv` |
| `tables/cross_seed_metrics.csv` | `world_id`, `context_id`, `condition`, `seed_blocks`, `unit_hash`, `metric_name`, `metric_status`, `value`, `numerator`, `denominator`, `metric_version_hash` |
| `tables/cross_seed_world_metrics.csv` | same columns as `world_metrics.csv` |
| `tables/rare_pivotal_observations.csv` | `condition`, `world_id`, `context_id`, `seed_block`, `true_positive_count`, `gold_count`, `output_valid`, `scorer_plan_hash` |
| `tables/paraphrase.csv` | `call_id`, `world_id`, `context_id`, `status`, `signature_divergence`, `strict_f1_change`, `base_strict_f1`, `paraphrase_strict_f1`, `base_score_hash`, `paraphrase_score_hash`, `base_projection_hash`, `paraphrase_projection_hash` |
| `tables/ablations.csv` | `call_id`, `condition`, `world_id`, `context_id`, `principal_metric`, `parent_value`, `ablation_value`, `difference`, `parent_score_hash`, `ablation_score_hash` |
| `tables/report_gate_status.csv` | `gate_status_hash`, `registered_analysis_hash`, `alpha`, `rare_safeguard_margin`, `organization_adjusted_p_value`, `mechanism_one_sided_p_value`, `c2_vs_c1_rare_lower_bound`, `c2_vs_fixed_rare_lower_bound`, `c2_vs_fixed_rare_p_value`, `organization_gate_passed`, `construction_freedom_gate_passed`, `c2_vs_c1_rare_safeguard_gate_passed`, `c2_vs_fixed_rare_safeguard_status`, `c2_vs_fixed_rare_safeguard_gate_passed` |

`comparisons.csv` uses only these frozen `analysis_family` values:
`registered_primary`, `required_secondary`, `rare_pivotal_safeguard`,
`gated_mechanism`, `gated_mechanism_safeguard`,
`mechanism_support`,
`secondary_complete_world_panel`, `gold_aligned_community`,
`cross_seed_community`, and `crossing_two_part`.

`analysis/report_gate_status.json` is the self-hashed source of the one-row gate CSV.
Its logical hash and physical file SHA-256 are also direct fields of the restricted
`analysis_index.json`, so a report compiler cannot silently substitute a gate record.
The three report-critical fields—`organization_gate_passed`,
`construction_freedom_gate_passed`, and
`c2_vs_fixed_rare_safeguard_gate_passed`—are derived from the registered analysis,
not inferred later from formatted p-values. The fixed-select rare safeguard carries
the frozen `-0.05` margin plus an explicit `passed`, `failed`, or `not_tested` status.

The registered analysis first averages the two LLM seeds within context and then the
three contexts within each of the 12 worlds. C0 retains its single deterministic value.
It runs the two primary C2-C1 tests with Holm adjustment, required C2-C0 estimates, all
4,096 world-level sign assignments, paired intervals and effects, 10,000-resample world
bootstrap sensitivity intervals, and the -0.05 rare-pivotal noninferiority gate. The
C2-`A-FixedSelect` mechanism test remains gated by the organization result. The entropy
two-sided family receives Benjamini-Hochberg adjustment across its ten planned
metric/comparison tests; unestimable panels stay explicit. Crossing opportunities and
conditional rates use the registered two-part analysis.

Two additional 12-world, two-sided C2-`A-FixedSelect` rows report contrastive
decision-change F1 and ontological collapse under the `mechanism_support` family.
They are required mechanism diagnostics, but are not added to the two-endpoint primary
Holm family and do not replace the gated ontology-decision endpoint or rare-pivotal
safeguard.

The complete output also contains contrast/collapse, paraphrase, reduced-ablation,
rare-support-path, entropy, direct clutter, Leiden-CPM resolution sensitivity,
gold-aligned community, and cross-seed stability observations. These secondary rows do
not turn contexts, seeds, nodes, assertions, or edges into independent samples.
