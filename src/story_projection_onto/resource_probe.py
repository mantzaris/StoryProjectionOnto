"""Stdlib-only read-only monitor workers. No ledger, model, or recovery controller.

Executed directly so sampling does not import the inference stack. Each worker
owns one session; its parent can cancel/reap it without joining a filesystem walk.
"""

from __future__ import annotations

import ctypes
import json
import os
import signal
import stat
import struct
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime


def stamp():
    return {"utc": datetime.now(UTC).isoformat(), "monotonic": time.monotonic()}


def process_tree(pid):
    pending, identities, rss = [pid], {}, 0
    while pending:
        current = pending.pop()
        if current in identities:
            continue
        try:
            with open(f"/proc/{current}/stat") as file:
                fields = file.read().rsplit(")", 1)[1].split()
            start_ticks = int(fields[19])
            with open(f"/proc/{current}/status") as file:
                status = file.read()
            identities[current] = start_ticks
            rss += next(
                (
                    int(line.split()[1]) * 1024
                    for line in status.splitlines()
                    if line.startswith("VmRSS:")
                ),
                0,
            )
            for task in os.listdir(f"/proc/{current}/task"):
                with open(f"/proc/{current}/task/{task}/children") as file:
                    pending.extend(int(child) for child in file.read().split())
        except (FileNotFoundError, ProcessLookupError):
            continue
    if pid not in identities:
        raise RuntimeError("owned root process absent during resource observation")
    return identities, rss


def fast(pid, expected_start=None):
    began = stamp()
    identities, rss = process_tree(pid)
    process_at = stamp()
    if expected_start is not None and identities[pid] != expected_start:
        raise RuntimeError("resource root PID identity changed")
    gpu_started = stamp()
    query = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
        timeout=2,
    )
    gpu_at = stamp()
    refreshed, fresh_rss = process_tree(pid)
    refreshed_at = stamp()
    if refreshed.get(pid) != identities[pid]:
        raise RuntimeError("resource root identity changed across GPU query")
    gpu = 0
    for line in query.stdout.splitlines():
        if not line.strip():
            continue
        process, memory = (int(value.strip()) for value in line.split(","))
        if process in refreshed or process in identities:
            gpu += memory * 1024**2
    with open("/proc/meminfo") as file:
        available = next(
            int(line.split()[1]) * 1024 for line in file if line.startswith("MemAvailable:")
        )
    return {
        "process_ids": sorted(set(identities) | set(refreshed)),
        "identities": refreshed,
        "root_start_ticks": identities[pid],
        "process_ram_bytes": max(rss, fresh_rss),
        "gpu_vram_bytes": gpu,
        "system_available_ram_bytes": available,
        "timing": {
            "began": began,
            "process": process_at,
            "gpu_started": gpu_started,
            "gpu": gpu_at,
            "refreshed_process": refreshed_at,
        },
    }


class StorageIndex:
    """Exact allocated-block census plus loss-detecting Linux inotify updates.

    Includes directories/symlinks and deduplicates hard links. Queue loss, mount
    changes and ambiguous directory moves invalidate the index, never an estimate.
    """

    MASK = 0x2 | 0x4 | 0x8 | 0x40 | 0x80 | 0x100 | 0x200 | 0x400 | 0x800

    def __init__(self, root):
        self.root = os.path.realpath(root)
        self.device = os.stat(self.root).st_dev
        self.libc = ctypes.CDLL(None, use_errno=True)
        self.fd = self.libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
        if self.fd < 0:
            raise OSError(ctypes.get_errno(), "inotify_init1")
        self.paths, self.inodes, self.watches = {}, {}, {}
        self.total = 0
        self.events = 0
        self.began = stamp()
        try:
            self.scan(self.root)
            self.drain()
        except BaseException:
            os.close(self.fd)
            raise
        self.baseline_at = stamp()
        self.baseline_bytes = self.total

    @staticmethod
    def allocated(metadata):
        return metadata.st_blocks * 512 if metadata.st_blocks else metadata.st_size

    def update(self, path):
        old = self.paths.pop(path, None)
        if old is not None:
            size, refs = self.inodes[old]
            if refs == 1:
                self.total -= size
                del self.inodes[old]
            else:
                self.inodes[old] = (size, refs - 1)
        try:
            metadata = os.lstat(path)
        except FileNotFoundError:
            return None
        if metadata.st_dev != self.device:
            raise RuntimeError("storage index crossed its quota device")
        identity = (metadata.st_dev, metadata.st_ino)
        size = self.allocated(metadata)
        previous, refs = self.inodes.get(identity, (0, 0))
        self.total += size - previous
        self.inodes[identity] = (size, refs + 1)
        self.paths[path] = identity
        return metadata

    def scan(self, path):
        metadata = self.update(path)
        if metadata is None or not stat.S_ISDIR(metadata.st_mode):
            return
        watch = self.libc.inotify_add_watch(self.fd, os.fsencode(path), self.MASK)
        if watch < 0:
            raise OSError(ctypes.get_errno(), "cannot completely watch project storage")
        self.watches[watch] = path
        with os.scandir(path) as entries:
            for entry in entries:
                self.scan(entry.path)

    def drain(self):
        started = time.monotonic()
        while True:
            try:
                data = os.read(self.fd, 1024 * 1024)
            except BlockingIOError:
                return
            if time.monotonic() - started > 2:
                raise RuntimeError("storage event backlog exceeds freshness deadline")
            offset = 0
            while offset < len(data):
                watch, mask, _, length = struct.unpack_from("iIII", data, offset)
                name = os.fsdecode(data[offset + 16 : offset + 16 + length].split(b"\0")[0])
                offset += 16 + length
                self.events += 1
                if mask & 0x4000:
                    raise RuntimeError("inotify overflow: full storage checkpoint required")
                if mask & 0x8000:  # explicit watch removal after known deletion
                    self.watches.pop(watch, None)
                    continue
                parent = self.watches.get(watch)
                if parent is None:
                    raise RuntimeError("unmapped storage event")
                if mask & (0x800 | 0x2000):
                    raise RuntimeError("directory moved/unmounted: full checkpoint required")
                path = os.path.join(parent, name) if name else parent
                if mask & 0x40000000 and mask & (0x40 | 0x80):
                    raise RuntimeError("directory rename requires full storage checkpoint")
                if mask & 0x40000000 and mask & 0x100:
                    self.scan(path)
                else:
                    self.update(path)
                self.update(parent)

    def observation(self):
        self.drain()
        fs = os.statvfs(self.root)
        return {
            "occupied_bytes": self.total,
            "filesystem_free_bytes": fs.f_bavail * fs.f_frsize,
            "baseline_bytes": self.baseline_bytes,
            "baseline_at": self.baseline_at,
            "baseline_began": self.began,
            "observed_at": stamp(),
            "write_events": self.events,
            "change_bytes": self.total - self.baseline_bytes,
            "method": "full_allocated_block_census_plus_loss_detecting_inotify",
            "watched_directories": len(self.watches),
            "valid": True,
        }


def main():
    expected_parent = int(sys.argv[2])
    # Kill this dedicated probe group on controller death, including nvidia-smi.
    signal.signal(signal.SIGTERM, lambda *_: os.killpg(os.getpgrp(), signal.SIGKILL))
    libc = ctypes.CDLL(None)
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0 or os.getppid() != expected_parent:
        raise RuntimeError("probe parent identity unavailable")
    mode = sys.argv[1]
    journal = sys.argv[3]
    index = None
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if mode == "storage":
                if index is None:
                    index = StorageIndex(request["root"])
                answer = index.observation()
            elif mode == "fast":
                answer = fast(request["pid"], request.get("expected_start"))
            else:
                raise ValueError("unknown monitor operation")
            record = {"ok": True, "value": answer}
        except BaseException as error:
            record = {
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error),
                "exception_chain": traceback.format_exc(),
                "observed_at": stamp(),
            }
        # One private diagnostic writer per worker; never a ledger connection.
        encoded = json.dumps(record) + "\n"
        with open(journal, "a") as file:
            file.write(encoded)
            file.flush()
            os.fsync(file.fileno())
        sys.stdout.write(encoded)
        sys.stdout.flush()
        if not record["ok"]:
            return


if __name__ == "__main__":
    main()
