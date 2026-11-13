"""Restricted, append-only response evidence, written before JSON interpretation.

This is a diagnostic journal, not a validator or a source of model-visible data.
It deliberately never records request headers, request bodies, or local variables.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
import threading
import traceback
from collections.abc import Callable, Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, TypeVar

from story_projection_onto.contracts import ReleaseClass, canonical_json
from story_projection_onto.store import BlobStore

FailureStage = Literal[
    "transport", "http", "decoding", "schema_validation", "scientific_validation"
]
T = TypeVar("T")
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
MAX_RESPONSE_FRAGMENTS = 1024
SAFE_RESPONSE_HEADERS = frozenset(
    {
        "content-type",
        "content-length",
        "transfer-encoding",
        "date",
        "x-request-id",
    }
)
_SECRET = re.compile(
    r"(?i)((?:authorization|proxy-authorization|api[-_]key|access[-_]token|"
    r"password|set-cookie|cookie)\s*[:=]\s*)([^\r\n]+)|\b(Bearer\s+)\S+"
)


class ResponseEvidenceLimitError(RuntimeError):
    """The response exceeds the bounded diagnostic/storage envelope."""


class RestrictedResponseJournal:
    """One response attempt with atomic fragment blobs and an fsynced event log."""

    def __init__(
        self,
        root: Path,
        *,
        request_hash: str,
        maximum_bytes: int = MAX_RESPONSE_BYTES,
        maximum_fragments: int | None = None,
    ):
        if maximum_fragments is None:
            maximum_fragments = MAX_RESPONSE_FRAGMENTS
        root = Path(root).absolute()
        if "restricted" not in root.parts or any(p.is_symlink() for p in (root, *root.parents)):
            raise ValueError("HTTP evidence must use a non-symlink restricted location")
        if not re.fullmatch(r"[0-9a-f]{64}", request_hash):
            raise ValueError("request_hash must be SHA-256")
        if not 0 < maximum_bytes <= MAX_RESPONSE_BYTES:
            raise ValueError("invalid HTTP evidence bound")
        if not 0 < maximum_fragments <= 8192:
            raise ValueError("invalid HTTP fragment bound")
        self.maximum_fragments = maximum_fragments
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = Path(tempfile.mkdtemp(prefix=f"{request_hash[:16]}-", dir=root))
        self.blobs = BlobStore(self.path / "fragments", max_raw_bytes=MAX_RESPONSE_BYTES)
        self.maximum_bytes = maximum_bytes
        self.received_bytes = 0
        self.fragment_count = 0
        self.digest = hashlib.sha256()
        self._lock = threading.RLock()
        self._sequence = 0
        self._secret_values: set[str] = set()
        self._recorded_errors: list[BaseException] = []
        self._cancelled = False
        self.response_complete = False
        self.event("request", request_hash=request_hash, release_class="restricted")

    def event(self, name: str, **values: object) -> None:
        with self._lock:
            payload = {
                "event": name,
                "sequence": self._sequence,
                "recorded_at": datetime.now(UTC).isoformat(),
                **values,
            }
            # Each complete line is durable. A torn last line remains evidence
            # of interruption; earlier lines and atomic blobs are recoverable.
            descriptor = os.open(
                self.path / "events.jsonl", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600
            )
            with os.fdopen(descriptor, "ab") as stream:
                stream.write((canonical_json(payload) + "\n").encode())
                stream.flush()
                os.fsync(stream.fileno())
            directory = os.open(self.path, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            self._sequence += 1

    def headers(self, status: int, headers: Mapping[str, str]) -> None:
        with self._lock:
            for key, value in headers.items():
                if key.lower() in {
                    "authorization",
                    "proxy-authorization",
                    "set-cookie",
                    "cookie",
                    "x-api-key",
                }:
                    self._secret_values.add(value)
            safe = {
                key.lower(): self.redact(value)
                for key, value in headers.items()
                if key.lower() in SAFE_RESPONSE_HEADERS
            }
            self.event("response_headers", http_status=status, headers=safe)

    def fragment(self, payload: bytes) -> None:
        if not payload:
            return
        with self._lock:
            if self.fragment_count >= self.maximum_fragments:
                self.event("fragment_limit", complete=False, retained_bytes=self.received_bytes)
                raise ResponseEvidenceLimitError("HTTP response exceeded diagnostic fragment limit")
            remaining = self.maximum_bytes - self.received_bytes
            kept = payload[:remaining]
            if kept:
                record = self.blobs.put_bytes(
                    kept,
                    media_type="application/octet-stream",
                    release_class=ReleaseClass.RESTRICTED,
                )
                offset = self.received_bytes
                self.received_bytes += len(kept)
                self.fragment_count += 1
                self.digest.update(kept)
                self.event("response_fragment", offset=offset, artifact=asdict(record))
            if len(payload) > remaining:
                self.event("response_limit", complete=False, retained_bytes=self.received_bytes)
                raise ResponseEvidenceLimitError("HTTP response exceeded diagnostic byte limit")
            if self._cancelled:
                raise TimeoutError("response arrived after the total-wall watchdog")

    def complete(self) -> None:
        self.response_complete = True
        self.event(
            "response_complete",
            complete=True,
            received_bytes=self.received_bytes,
            response_sha256=self.digest.hexdigest(),
        )

    def redact(self, message: str) -> str:
        for value in sorted(self._secret_values, key=len, reverse=True):
            if value:
                message = message.replace(value, "[REDACTED]")
        return _SECRET.sub(lambda m: (m.group(1) or m.group(3)) + "[REDACTED]", message)

    def _chain(self, error: BaseException, seen: set[int]) -> dict[str, object]:
        if id(error) in seen:
            return {"cycle": True, "type": type(error).__qualname__}
        seen.add(id(error))
        return {
            "type": f"{type(error).__module__}.{type(error).__qualname__}",
            "message": self.redact(str(error)),
            "frames": [
                {"file": frame.filename, "line": frame.lineno, "function": frame.name}
                for frame in traceback.extract_tb(error.__traceback__)
            ],
            "cause": None if error.__cause__ is None else self._chain(error.__cause__, seen),
            "context": None if error.__context__ is None else self._chain(error.__context__, seen),
            "suppress_context": error.__suppress_context__,
            "notes": [self.redact(note) for note in getattr(error, "__notes__", ())],
            "group": [self._chain(child, seen) for child in error.exceptions]
            if isinstance(error, BaseExceptionGroup)
            else [],
        }

    def failure(self, stage: FailureStage, error: BaseException, *, cancel: bool = False) -> None:
        with self._lock:
            if any(previous is error for previous in self._recorded_errors):
                return
            self._recorded_errors.append(error)
            self._cancelled |= cancel
            self.event(
                "failure",
                stage=stage,
                exception_chain=self._chain(error, set()),
                received_bytes=self.received_bytes,
                response_complete=self.response_complete,
            )

    def validate(self, stage: FailureStage, operation: Callable[[], T]) -> T:
        self.event("validation_started", stage=stage)
        try:
            result = operation()
        except BaseException as error:
            self.failure(stage, error)
            raise
        self.event("validation_passed", stage=stage)
        return result
