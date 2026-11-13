"""A bounded pipe barrier for preparing the next controller before model loading.

The child retains its own PID. No service, ledger writer, query, or inference is
created by this helper. EOF and timeout fail closed; the normal live checks run
after release. This is not a resume or recovery authorization mechanism.
"""

from __future__ import annotations

import os
import select
import subprocess
import time
from collections.abc import Sequence

READY_FD = "SPO_CPU_PREPARED_FD"
RELEASE_FD = "SPO_CPU_RELEASE_FD"
CPU_PREPARATION_TIMEOUT_SECONDS = 300.0
CPU_RELEASE_TIMEOUT_SECONDS = 900.0


def wait_for_allocation_release(*, timeout: float = CPU_RELEASE_TIMEOUT_SECONDS) -> bool:
    """Signal completed invariant preparation, then await the exact parent pipe."""
    raw_ready = os.environ.pop(READY_FD, None)
    raw_release = os.environ.pop(RELEASE_FD, None)
    if raw_ready is None and raw_release is None:
        return False
    if raw_ready is None or raw_release is None:
        raise RuntimeError("incomplete CPU preparation pipe pair")
    ready, release = int(raw_ready), int(raw_release)
    try:
        os.write(ready, b"ready\n")
        os.close(ready)
        ready = -1
        if not select.select([release], [], [], timeout)[0]:
            raise TimeoutError("CPU-prepared controller was not released")
        if os.read(release, 32) != b"continue\n":
            raise RuntimeError("CPU preparation owner disappeared or sent an invalid release")
    finally:
        if ready >= 0:
            os.close(ready)
        os.close(release)
    return True


class PreparedController:
    """Own a specific prestarted subprocess; never signal an inferred process."""

    def __init__(self, command: Sequence[str], *, timeout: float = CPU_PREPARATION_TIMEOUT_SECONDS):
        started = time.monotonic()
        ready_read, ready_write = os.pipe()
        release_read, release_write = os.pipe()
        self.release_fd = release_write
        self.process: subprocess.Popen[bytes] | None = None
        try:
            self.process = subprocess.Popen(
                command,
                pass_fds=(ready_write, release_read),
                env={**os.environ, READY_FD: str(ready_write), RELEASE_FD: str(release_read)},
            )
            os.close(ready_write)
            ready_write = -1
            os.close(release_read)
            release_read = -1
            if not select.select([ready_read], [], [], timeout)[0]:
                raise TimeoutError("CPU controller preparation exceeded its bound")
            if os.read(ready_read, 32) != b"ready\n":
                raise RuntimeError("CPU controller exited before readiness")
            self.preparation_seconds = time.monotonic() - started
        except BaseException:
            self.close()
            raise
        finally:
            os.close(ready_read)
            if ready_write >= 0:
                os.close(ready_write)
            if release_read >= 0:
                os.close(release_read)

    def release_and_wait(self) -> int:
        if self.release_fd < 0 or self.process is None:
            raise RuntimeError("prepared controller cannot be released twice")
        os.write(self.release_fd, b"continue\n")
        os.close(self.release_fd)
        self.release_fd = -1
        return self.process.wait()

    def close(self) -> None:
        if self.release_fd >= 0:
            os.close(self.release_fd)
            self.release_fd = -1
        if self.process is not None and self.process.poll() is None:
            # First let EOF stop a waiting controller; otherwise stop only the
            # Popen-owned process. Guardian ownership/accounting is unchanged.
            try:
                self.process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
