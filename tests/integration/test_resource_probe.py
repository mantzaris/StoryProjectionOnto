"""CPU-only real Linux monitoring/cancellation, with no model service."""

import json
import os
import signal
import struct
import threading
import time
from types import SimpleNamespace

import pytest

from story_projection_onto.experiment import AllocatedGPUMeter
from story_projection_onto.gpu_runtime import (
    ResourceSampler,
    ResourceWatchdog,
    RuntimeWatchdogTimeout,
    VLLMService,
    _ResourceProbe,
)
from story_projection_onto.resource_probe import StorageIndex, fast
from story_projection_onto.store import Ledger, StorageBudgetExceeded, StoragePreflight
from tests.unit.test_gpu_runtime import FakeClock, FakeProcess, FakeServiceClient, limits
from tests.unit.test_gpu_runtime import launch_configuration as configuration_fixture


@pytest.fixture
def launch_configuration(tmp_path):
    return configuration_fixture.__wrapped__(tmp_path)


def census(root):
    seen, total = set(), 0
    for p in [root, *root.rglob("*")]:
        s = p.lstat()
        identity = s.st_dev, s.st_ino
        if identity not in seen:
            seen.add(identity)
            total += s.st_blocks * 512 if s.st_blocks else s.st_size
    return total


def test_live_storage_tracks_atomic_writes_hardlinks_truncation_and_directories(tmp_path):
    index = StorageIndex(tmp_path)
    try:
        assert index.observation()["occupied_bytes"] == census(tmp_path)
        folder = tmp_path / "new"
        folder.mkdir()
        file = folder / "tmp"
        file.write_bytes(b"x" * 100000)
        file.rename(folder / "final")
        os.link(folder / "final", folder / "linked")
        os.symlink("final", folder / "symbolic")
        assert index.observation()["occupied_bytes"] == census(tmp_path)
        (folder / "final").write_bytes(b"tiny")
        assert index.observation()["occupied_bytes"] == census(tmp_path)
        (folder / "linked").unlink()
        assert index.observation()["occupied_bytes"] == census(tmp_path)
        (folder / "final").unlink()
        (folder / "symbolic").unlink()
        folder.rmdir()
        observed = index.observation()
        assert observed["occupied_bytes"] == census(tmp_path)
        assert observed["write_events"] > 0
    finally:
        os.close(index.fd)


def test_ambiguous_directory_move_fails_closed(tmp_path):
    (tmp_path / "a").mkdir()
    index = StorageIndex(tmp_path)
    try:
        (tmp_path / "a").rename(tmp_path / "b")
        with pytest.raises(RuntimeError, match="directory"):
            index.observation()
    finally:
        os.close(index.fd)


def test_storage_overflow_fails_closed_instead_of_using_stale_count(tmp_path, monkeypatch):
    index = StorageIndex(tmp_path)
    try:
        monkeypatch.setattr(
            "story_projection_onto.resource_probe.os.read",
            lambda *args: struct.pack("iIII", -1, 0x4000, 0, 0),
        )
        with pytest.raises(RuntimeError, match=r"overflow.*full storage checkpoint"):
            index.observation()
    finally:
        os.close(index.fd)


def test_fast_query_refreshes_process_identity_and_records_individual_times(monkeypatch):
    observations = iter([({41: 100, 42: 101}, 200), ({41: 100, 43: 102}, 300)])
    monkeypatch.setattr(
        "story_projection_onto.resource_probe.process_tree", lambda pid: next(observations)
    )

    def gpu(*args, **kwargs):
        assert kwargs["timeout"] == 2
        return SimpleNamespace(stdout="42, 10\n43, 20\n99, 200\n")

    monkeypatch.setattr("story_projection_onto.resource_probe.subprocess.run", gpu)
    measured = fast(41, 100)
    assert measured["gpu_vram_bytes"] == 30 * 1024**2  # no unrelated PID 99
    assert measured["process_ram_bytes"] == 300
    times = measured["timing"]
    assert times["process"]["monotonic"] <= times["gpu"]["monotonic"]
    assert times["gpu"]["monotonic"] <= times["refreshed_process"]["monotonic"]
    assert measured["identities"] == {41: 100, 43: 102}


def test_fast_query_rejects_root_pid_reuse_across_gpu_observation(monkeypatch):
    observations = iter([({41: 100}, 200), ({41: 999}, 200)])
    monkeypatch.setattr(
        "story_projection_onto.resource_probe.process_tree", lambda pid: next(observations)
    )
    monkeypatch.setattr(
        "story_projection_onto.resource_probe.subprocess.run",
        lambda *a, **k: SimpleNamespace(stdout="41, 20\n"),
    )
    with pytest.raises(RuntimeError, match="identity changed across GPU"):
        fast(41, 100)


def test_preallocation_storage_failure_reaps_probe_and_writes_no_ledger(tmp_path, monkeypatch):
    sampler = ResourceSampler(limits=limits(), storage=StoragePreflight(tmp_path))
    original = sampler.storage.require

    def reject(**values):
        return original(**{**values, "filesystem_free_bytes": 1})

    monkeypatch.setattr(sampler.storage, "require", reject)
    with pytest.raises(StorageBudgetExceeded):
        sampler.prepare(timeout_seconds=3)
    assert sampler._storage_probe is None
    assert sampler._fast_probe is None
    assert not list(tmp_path.rglob("*.sqlite"))


def test_checkpoint_recounts_and_same_controller_restart_can_keep_write_stream(tmp_path):
    sampler = ResourceSampler(limits=limits(), storage=StoragePreflight(tmp_path))
    try:
        sampler.prepare(timeout_seconds=3)
        original = sampler._storage_probe
        sampler.cancel_pending()  # restart cancellation never stops the write stream
        (tmp_path / "while-model-reloads").write_bytes(b"x" * 10000)
        assert sampler._storage_probe is original
        observed = original.request({}, 1)
        assert observed["write_events"] > 0
        sampler.prepare(force=True, timeout_seconds=3)
        assert original._process.poll() is not None
        assert sampler._storage_probe is not original
        assert sampler._storage_observation["valid"]
    finally:
        sampler.close_probes()


def test_stopped_worker_is_killed_and_reaped_on_observation_deadline(tmp_path):
    probe = _ResourceProbe("storage", tmp_path)
    try:
        first = probe.request({"root": str(tmp_path)}, 3)
        assert first["valid"]
        assert json.loads(probe.journal.read_text().splitlines()[0])["ok"]
        os.kill(probe._process.pid, signal.SIGSTOP)
        start = time.monotonic()
        with pytest.raises(RuntimeWatchdogTimeout, match="observation deadline"):
            probe.request({}, 0.05)
        assert time.monotonic() - start < 0.5
        assert probe._process.poll() is not None
    finally:
        probe.close()


def test_owned_service_signaled_before_stuck_storage_collection_and_accounting_reconciles(
    tmp_path, launch_configuration
):
    # The service handle is a CPU fixture. The wedged storage observer is a REAL
    # subprocess using production IPC, signal ownership, and wait/reaping logic.
    process = FakeProcess(10991)
    signals = []

    def signal_service(pid, number):
        assert pid == process.pid
        signals.append((number, time.monotonic()))
        process.running = False

    with Ledger(tmp_path / "ledger.sqlite") as ledger:
        sampler = ResourceSampler(
            limits=limits(), storage=StoragePreflight(tmp_path), ledger=ledger
        )
        sampler.prepare(timeout_seconds=3)
        storage_child = sampler._storage_probe._process
        os.kill(storage_child.pid, signal.SIGSTOP)
        entered = threading.Event()

        class FastFixture:
            _process = SimpleNamespace(poll=lambda: None)

            def request(self, *args):
                entered.set()
                return {
                    "root_start_ticks": 1,
                    "process_ids": [process.pid],
                    "identities": {},
                    "process_ram_bytes": 1,
                    "gpu_vram_bytes": 0,
                    "system_available_ram_bytes": 100000000000,
                    "timing": {"began": {"monotonic": time.monotonic()}},
                }

            def cancel(self):
                pass

            def close(self):
                pass

        sampler._fast_probe = FastFixture()
        service = VLLMService(
            configuration=launch_configuration,
            client=FakeServiceClient(FakeClock()),
            meter=AllocatedGPUMeter(ledger),
            popen_factory=lambda *a, **k: process,
            process_group_signaler=signal_service,
            process_group_liveness_check=lambda pid: process.running,
            available_cpu_sampler=lambda: set(range(16)),
            affinity_setter=lambda *a: None,
        )
        service.start(session_id="cpu-only", event_id="cpu-fixture-load", watchdog_seconds=1)
        watchdog = ResourceWatchdog(
            sampler=sampler, root_pid=process.pid, sample_prefix="stuck", interval_seconds=10
        )
        service.start_periodic_resource_watchdog(watchdog)
        assert entered.wait(0.5)
        start = time.monotonic()
        uptime = service.shutdown(shutdown_seconds=1)
        assert signals[0][1] - start < 0.3
        assert time.monotonic() - start < 1
        assert storage_child.poll() is not None
        assert not watchdog.running and not watchdog.sample_in_flight
        assert uptime is not None
        assert not ledger.unresolved_gpu_service_journals()
        assert not ledger.unresolved_gpu_allocations()
        assert sampler._storage_probe is None
        # The initial durable census survives cancellation; no worker opens SQLite.
        journals = list((tmp_path / "artifacts/restricted/resource-monitor").glob("*.jsonl"))
        assert any(p.read_text().strip() for p in journals)
