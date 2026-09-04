from __future__ import annotations

import hashlib
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from story_projection_onto.contracts import AllenRelation, PartialOrderConstraint
from story_projection_onto.store import BlobStore, Compression, ReleaseClass
from story_projection_onto.temporal import (
    TemporalDiagnosticCode,
    TemporalValidationStatus,
    validate_partial_order,
)


@settings(max_examples=16, deadline=None)
@given(st.integers(min_value=2, max_value=8), st.booleans())
def test_precedence_chain_cycle_detection_is_order_independent(
    node_count: int,
    reverse_input_order: bool,
) -> None:
    node_ids = tuple(f"event-{index}" for index in range(node_count))
    chain = tuple(
        PartialOrderConstraint(
            left_id=node_ids[index],
            relation=AllenRelation.BEFORE,
            right_id=node_ids[index + 1],
        )
        for index in range(node_count - 1)
    )
    supplied_chain = tuple(reversed(chain)) if reverse_input_order else chain

    acyclic = validate_partial_order(supplied_chain, known_node_ids=node_ids)
    assert acyclic.status is TemporalValidationStatus.VALID

    closing_edge = PartialOrderConstraint(
        left_id=node_ids[-1],
        relation=AllenRelation.BEFORE,
        right_id=node_ids[0],
    )
    cyclic = validate_partial_order((*supplied_chain, closing_edge), known_node_ids=node_ids)
    assert cyclic.status is TemporalValidationStatus.CONTRADICTION
    cycle_diagnostic = next(
        item
        for item in cyclic.diagnostics
        if item.code is TemporalDiagnosticCode.PARTIAL_ORDER_CYCLE
    )
    assert cycle_diagnostic.involved_ids == node_ids


@settings(max_examples=16, deadline=None)
@given(st.binary(min_size=0, max_size=2_048))
def test_content_addressed_gzip_writes_deduplicate_for_arbitrary_bytes(
    tmp_path: Path,
    payload: bytes,
) -> None:
    example_root = tmp_path / hashlib.sha256(payload).hexdigest()
    store = BlobStore(example_root / "blobs", compression=Compression.GZIP)

    first = store.put_bytes(
        payload,
        media_type="application/octet-stream",
        release_class=ReleaseClass.PUBLIC,
    )
    second = store.put_bytes(
        payload,
        media_type="application/octet-stream",
        release_class=ReleaseClass.PUBLIC,
    )

    assert second.content_hash == first.content_hash
    assert second.relative_path == first.relative_path
    assert store.read_bytes(second) == payload
    assert len(tuple((example_root / "blobs").rglob(f"*{store.suffix}"))) == 1
