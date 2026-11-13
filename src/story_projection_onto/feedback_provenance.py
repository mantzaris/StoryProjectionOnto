"""Durable request provenance for the three researcher-driven Phase 5 traces.

The receipt is created by the real ``POST /api/revisions`` boundary.  It records
only request lineage; it does not claim that the downstream condition resolved
the request successfully and it contains no scorer-only fields.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Literal, Self

from pydantic import AwareDatetime, Field, model_validator

from story_projection_onto.contracts import (
    ConditionName,
    FeedbackAction,
    Identifier,
    ImmutableRecord,
    ReleaseClass,
    Sha256Digest,
)


class ResearcherTraceSubmissionReceipt(ImmutableRecord):
    """One accepted researcher-trace request at the actual UI endpoint."""

    receipt_id: Identifier
    episode_id: Identifier
    endpoint: Literal["/api/revisions"] = "/api/revisions"
    http_method: Literal["POST"] = "POST"
    action: FeedbackAction
    requested_at: AwareDatetime
    before_condition: Literal[ConditionName.C2_LLM_QUERY] = ConditionName.C2_LLM_QUERY
    before_projection_id: Identifier
    before_projection_hash: Sha256Digest
    before_bundle_hash: Sha256Digest
    submission_hash: Sha256Digest
    submission_canonical_json: str = Field(min_length=1)
    submission_file_sha256: Sha256Digest
    instruction_hash: Sha256Digest
    instruction_file_sha256: Sha256Digest
    recorded_at: AwareDatetime
    generated_by_ui_endpoint: Literal[True] = True
    scorer_fields_included: Literal[False] = False
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED

    @model_validator(mode="after")
    def chronology_is_honest(self) -> Self:
        if self.recorded_at < self.requested_at:
            raise ValueError("trace submission receipt predates its UI request")
        try:
            submitted = json.loads(self.submission_canonical_json)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("trace receipt embeds invalid submission JSON") from error
        submission_bytes = (self.submission_canonical_json + "\n").encode("utf-8")
        if (
            not isinstance(submitted, dict)
            or submitted.get("content_hash") != self.submission_hash
            or hashlib.sha256(submission_bytes).hexdigest()
            != self.submission_file_sha256
        ):
            raise ValueError("trace receipt does not bind its canonical submission bytes")
        return self


class AppendOnlyResearcherTraceCaptureStore:
    """Restricted, fsync-backed instruction/receipt sink used by ``create_app``.

    One episode maps to one immutable receipt.  An exact retry is idempotent;
    any attempted rebinding fails before the revision runner is invoked.
    """

    def __init__(self, root: Path) -> None:
        self.root = root.absolute()

    @staticmethod
    def _assert_no_symlink_chain(path: Path) -> None:
        current = path.absolute()
        while True:
            if current.is_symlink():
                raise ValueError(f"symlinked trace-receipt path is forbidden: {path}")
            if current.parent == current:
                return
            current = current.parent

    @staticmethod
    def _safe_episode_id(value: str) -> str:
        if not value or Path(value).name != value or value in {".", ".."} or "\\" in value:
            raise ValueError("unsafe researcher-trace episode ID")
        return value

    def _append(self, path: Path, payload: bytes) -> None:
        self._assert_no_symlink_chain(path)
        if path.exists():
            if not path.is_file() or path.read_bytes() != payload:
                raise ValueError("append-only researcher-trace receipt changed")
            return
        parent_existed = self.root.exists()
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self.root, 0o700)
        self._assert_no_symlink_chain(self.root)
        if not parent_existed:
            parent_fd = os.open(self.root.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=self.root
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                os.fchmod(stream.fileno(), 0o600)
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                if not path.is_file() or path.read_bytes() != payload:
                    raise ValueError("concurrent researcher-trace receipt changed") from None
            root_fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(root_fd)
            finally:
                os.close(root_fd)
        finally:
            temporary.unlink(missing_ok=True)

    def __call__(
        self,
        instruction: ImmutableRecord,
        receipt: ResearcherTraceSubmissionReceipt,
    ) -> None:
        episode_id = self._safe_episode_id(receipt.episode_id)
        instruction_payload = (instruction.to_canonical_json() + "\n").encode("utf-8")
        if (
            instruction.content_hash != receipt.instruction_hash
            or hashlib.sha256(instruction_payload).hexdigest()
            != receipt.instruction_file_sha256
        ):
            raise ValueError("trace receipt does not bind its canonical instruction bytes")
        # The receipt is published last: its existence therefore certifies that the
        # bound submission and instruction bytes were already made durable.
        self._append(
            self.root / f"{episode_id}.submission.json",
            (receipt.submission_canonical_json + "\n").encode("utf-8"),
        )
        self._append(self.root / f"{episode_id}.instruction.json", instruction_payload)
        self._append(
            self.root / f"{episode_id}.receipt.json",
            (receipt.to_canonical_json() + "\n").encode("utf-8"),
        )


__all__ = [
    "AppendOnlyResearcherTraceCaptureStore",
    "ResearcherTraceSubmissionReceipt",
]
