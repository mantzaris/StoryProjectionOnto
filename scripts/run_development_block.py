#!/usr/bin/env python3
"""Validate and print the frozen Phase-3 development call manifest.

This inspection command never owns a model lifecycle.  The production GPU
entrypoint is ``scripts/run_fallback_gpu_acceptance.py --execute
--controller-stage orchestrate ...``; it loads the registered factory
``story_projection_onto.development_continuation:create_production_development_adopter``
and retains final shutdown authority around the synchronous development block.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from story_projection_onto.development_runtime import (  # noqa: E402
    canonical_public_summary,
    load_development_call_manifest,
)
from story_projection_onto.gpu_runtime import (  # noqa: E402
    FALLBACK_MODEL_REPOSITORY,
    FALLBACK_MODEL_REVISION,
    capture_tokenizer_manifest,
)


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument(
        "--plan",
        type=Path,
        help="Optional plan path; production should use the tracked default",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--validate-only",
        action="store_true",
        help="Explicit acknowledgement that this command performs no inference",
    )
    mode.add_argument(
        "--packing-preflight",
        action="store_true",
        help="Persist the exact four complete C1 chat-packing checks without GPU use",
    )
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--launcher-configuration-hash")
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    options = parse_arguments(arguments)
    root = options.project_root.resolve(strict=True)
    plan = None if options.plan is None else options.plan.resolve(strict=True)
    manifest = load_development_call_manifest(root, plan)
    if options.validate_only:
        print(canonical_public_summary(manifest))
        return 0
    missing = tuple(
        name
        for name in ("snapshot", "ledger", "artifact_root", "launcher_configuration_hash")
        if getattr(options, name) is None
    )
    if missing:
        raise SystemExit(
            "--packing-preflight requires "
            + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        )
    launcher_hash = options.launcher_configuration_hash
    if (
        not isinstance(launcher_hash, str)
        or len(launcher_hash) != 64
        or any(character not in "0123456789abcdef" for character in launcher_hash)
    ):
        raise SystemExit("--launcher-configuration-hash must be lowercase SHA-256")

    # Imports that may initialize tokenizer libraries are delayed until the
    # explicit packing mode.  No CUDA API or model service is touched.
    from transformers import AutoTokenizer

    from story_projection_onto.development_adapter import (
        DevelopmentConstructionConfiguration,
    )
    from story_projection_onto.development_continuation import (
        build_c1_packing_preflight,
        build_development_run_configurations,
        development_seed_manifest_hash,
        development_validator_hash,
        load_development_prequery_evidence,
    )
    from story_projection_onto.store import ArtifactStore, BlobStore, Ledger

    snapshot = options.snapshot.resolve(strict=True)
    tokenizer_manifest = capture_tokenizer_manifest(
        snapshot,
        repository=FALLBACK_MODEL_REPOSITORY,
        revision=FALLBACK_MODEL_REVISION,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot),
        local_files_only=True,
        trust_remote_code=False,
        revision=FALLBACK_MODEL_REVISION,
    )
    construction = DevelopmentConstructionConfiguration.load(
        root / "configs/study/development_construction.json"
    )
    neutral, _visible, _certificates = load_development_prequery_evidence(
        root, manifest
    )
    configs, _fixed_plans = build_development_run_configurations(
        root=root,
        manifest=manifest,
        construction=construction,
        tokenizer_manifest=tokenizer_manifest,
        model_stack_hash=launcher_hash,
        seed_manifest_hash=development_seed_manifest_hash(root, manifest),
        validator_hash=development_validator_hash(root),
    )
    with Ledger(options.ledger) as ledger:
        preflight, artifact_hash = build_c1_packing_preflight(
            root=root,
            manifest=manifest,
            construction=construction,
            neutral_by_unit=neutral,
            run_configs=configs,
            tokenizer=tokenizer,
            tokenizer_manifest=tokenizer_manifest,
            artifacts=ArtifactStore(BlobStore(options.artifact_root), ledger),
            clock=lambda: datetime.now(UTC),
        )
    print(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "kind": "development_c1_cpu_packing_preflight",
                "preflight_artifact_hash": artifact_hash,
                "preflight_hash": preflight.content_hash,
                "tokenizer_manifest_hash": tokenizer_manifest.manifest_sha256,
                "call_token_counts": {
                    item.call_id: item.rendered_input_token_count
                    for item in preflight.receipts
                },
                "worst_rendered_input_tokens": preflight.worst_rendered_input_tokens,
                "all_complete_and_within_cap": preflight.all_complete_and_within_cap,
                "gpu_executed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
