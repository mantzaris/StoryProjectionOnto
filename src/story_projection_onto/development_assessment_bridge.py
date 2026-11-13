"""Late-bound bridge from gold-free development execution to scorer-only review.

This module deliberately has no top-level import of ``scorer_only``.  The
production continuation calls it only after all 24 model calls and all CPU
projection receipts have been durably indexed in the public assessment bundle.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Callable
from pathlib import Path

from story_projection_onto.contracts import canonical_json
from story_projection_onto.development_runtime import (
    DevelopmentCallManifest,
    DevelopmentITTRecord,
    DevelopmentPrequeryInputs,
    DevelopmentScientificAssessment,
)
from story_projection_onto.store import BlobStore, Ledger


def _write_canonical_manifest(path: Path, payload: object) -> str:
    """Atomically persist exact canonical bytes and return their SHA-256."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError("assessment input manifest cannot be a symlink")
    encoded = (canonical_json(payload) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(encoded).hexdigest()


def build_post_run_development_assessment_provider(
    *,
    root: Path,
    ledger: Ledger,
    blobs: BlobStore,
    prequery_inputs: DevelopmentPrequeryInputs,
    assessment_bundle_artifact_hash: str,
    assessment_manifest_path: Path,
) -> Callable[
    [DevelopmentCallManifest, tuple[DevelopmentITTRecord, ...]],
    DevelopmentScientificAssessment,
]:
    """Verify the neutral bundle, persist scorer routing, then return a provider."""

    # Delayed by design: importing the scorer namespace before generation would
    # weaken the executable gold firewall even if no scorer file were opened.
    from story_projection_onto.scorer_only.development_assessment import (
        build_development_assessment_input_manifest,
        build_development_scientific_assessment_provider,
    )

    manifest = build_development_assessment_input_manifest(
        root=root,
        ledger=ledger,
        blobs=blobs,
        assessment_bundle_artifact_hash=assessment_bundle_artifact_hash,
    )
    manifest_sha256 = _write_canonical_manifest(
        assessment_manifest_path,
        manifest.model_dump(mode="json"),
    )
    return build_development_scientific_assessment_provider(
        root=root,
        ledger=ledger,
        blobs=blobs,
        prequery_inputs=prequery_inputs,
        assessment_manifest_path=assessment_manifest_path,
        assessment_manifest_file_sha256=manifest_sha256,
    )


__all__ = ["build_post_run_development_assessment_provider"]
