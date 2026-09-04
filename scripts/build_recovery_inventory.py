#!/usr/bin/env python3
"""Build a safe, self-hashed inventory of interrupted-session recovery files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

UTC = timezone.utc  # noqa: UP017 -- recovery host may provide only Python 3.8


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _safe_relative(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError(f"unsafe recovery inventory path: {value}")
    return path.as_posix()


def _without_prefix(value: str, prefix: str) -> str:
    if not value.startswith(prefix):
        raise ValueError(f"{value!r} does not begin with {prefix!r}")
    return value[len(prefix) :]


def _git_blob(repository: Path, relative: str) -> bytes | None:
    result = subprocess.run(
        ["git", "show", f"HEAD:{relative}"],
        cwd=repository,
        check=False,
        capture_output=True,
    )
    return result.stdout if result.returncode == 0 else None


def _apparent_phase(path: str) -> str:
    lowered = path.casefold()
    if any(token in lowered for token in ("case_study", "novel_case", "phase6")):
        return "phase_6_narrative"
    if any(token in lowered for token in ("report", "figure", "table", "phase7")):
        return "phase_7_reporting"
    if any(token in lowered for token in ("feedback", "phase5", "ui/", "cytoscape")):
        return "phase_5_interaction"
    if any(token in lowered for token in ("metric", "analysis", "combined_gpu", "phase4")):
        return "phase_4_analysis"
    if any(token in lowered for token in ("held_out", "development", "conditions/")):
        return "phase_3_conditions"
    if any(
        token in lowered
        for token in ("synthetic", "benchmark", "independent_review", "review/")
    ):
        return "phase_2_benchmark"
    if any(
        token in lowered
        for token in (
            "phase1",
            "fallback",
            "model_",
            "gpu_",
            "contracts.py",
            "store.py",
            "schemas/",
        )
    ):
        return "phase_1_contracts_acceptance"
    return "cross_phase"


def _roles(apparent_path: str, recovery_path: str) -> tuple[str, ...]:
    path = apparent_path.casefold()
    recovery = recovery_path.casefold()
    source_prefixes = (
        "configs/",
        "docs/",
        "plan_notes/",
        "prompts/",
        "schemas/",
        "scripts/",
        "src/",
        "tests/",
        "ui/",
    )
    roles: set[str] = set()
    if path.startswith(source_prefixes) or (
        "/live_root/" in f"/{recovery}" and "/" not in path
    ):
        roles.add("source")
    if any(
        token in path
        for token in (
            "data/synthetic/",
            "reports/",
            "artifacts/",
            ".cache/",
            ".sqlite",
            ".log",
            "schemas/jsonschema/",
        )
    ):
        roles.add("generated")
    if any(
        token in recovery
        for token in (
            "runpod_snapshots/",
            "test_logs/",
            "artifacts/restricted/",
            "artifacts/blobs/",
            ".cache/",
        )
    ):
        roles.add("restricted")
    if any(
        token in path
        for token in (
            "reports/",
            "schemas/jsonschema/",
            "data/synthetic/condition_inputs/",
            "data/synthetic/model_visible/",
            ".log",
            ".sqlite-shm",
            ".sqlite-wal",
        )
    ):
        roles.add("regenerable")
    return tuple(sorted(roles or {"generated"}))


def _origin_mapping(repository: Path, path: Path) -> tuple[str, str]:
    relative = path.relative_to(repository).as_posix()
    marker = ".local_data/runpod_recovery/20260904_interrupted_remote/"
    if relative.startswith(marker + "live_root/"):
        return "remote_live_project", _without_prefix(relative, marker + "live_root/")
    if relative.startswith(marker + "public_manifests/"):
        name = _without_prefix(relative, marker + "public_manifests/")
        return "remote_public_manifest_copy", f"artifacts/public/manifests/{name}"
    if relative.startswith(marker + "test_logs/"):
        name = _without_prefix(relative, marker + "test_logs/")
        return "remote_test_log_copy", f"artifacts/restricted/recovery_logs/{name}"
    if relative.startswith(".local_data/runpod_snapshots/"):
        return "remote_failure_snapshot_copy", relative
    return "reconciled_runtime_artifact", relative


def _entry(repository: Path, path: Path) -> dict[str, Any]:
    recovery_path = path.relative_to(repository).as_posix()
    origin_scope, apparent_path = _origin_mapping(repository, path)
    recovery_path = _safe_relative(recovery_path)
    apparent_path = _safe_relative(apparent_path)
    blob = _git_blob(repository, apparent_path)
    digest = _sha256(path)
    if blob is None:
        git_status = "untracked_or_not_applicable"
    elif hashlib.sha256(blob).hexdigest() == digest:
        git_status = "tracked_exact_at_head"
    else:
        git_status = "tracked_content_differs_from_head"
    stat = path.stat()
    return {
        "recovery_path": recovery_path,
        "apparent_original_path": apparent_path,
        "origin_scope": origin_scope,
        "size_bytes": stat.st_size,
        "sha256": digest,
        "modification_time_utc": datetime.fromtimestamp(stat.st_mtime, UTC)
        .isoformat()
        .replace("+00:00", "Z"),
        "git_status": git_status,
        "apparent_phase": _apparent_phase(apparent_path),
        "artifact_roles": list(_roles(apparent_path, recovery_path)),
    }


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path("."))
    parser.add_argument(
        "--recovery-root",
        type=Path,
        default=Path(".local_data/runpod_recovery/20260904_interrupted_remote"),
    )
    parser.add_argument(
        "--runtime-path",
        action="append",
        type=Path,
        default=[],
        help="Additional recovered runtime file or directory, repository-relative.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/public/manifests/interrupted_recovery_inventory.json"),
    )
    parser.add_argument("--recorded-at", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    options = parse_arguments(argv)
    repository = options.repository.resolve(strict=True)
    recorded_at = datetime.fromisoformat(options.recorded_at.replace("Z", "+00:00"))
    if recorded_at.tzinfo is None or recorded_at.utcoffset() is None:
        raise ValueError("--recorded-at must be timezone-aware")
    roots = [options.recovery_root, *options.runtime_path]
    files: set[Path] = set()
    for raw in roots:
        candidate = raw if raw.is_absolute() else repository / raw
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(repository)
        if resolved.is_symlink():
            raise ValueError("recovery inventory refuses symlink roots")
        if resolved.is_file():
            files.add(resolved)
        else:
            files.update(
                item.resolve(strict=True)
                for item in resolved.rglob("*")
                if item.is_file() and not item.is_symlink()
            )
    entries = tuple(
        _entry(repository, path)
        for path in sorted(files, key=lambda item: item.relative_to(repository).as_posix())
    )
    payload: dict[str, object] = {
        "schema_version": "1.0.0",
        "kind": "interrupted_session_recovery_inventory",
        "recorded_at": recorded_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "repository_branch": subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "repository_head": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "file_count": len(entries),
        "total_bytes": sum(item["size_bytes"] for item in entries),
        "files": entries,
        "deliberately_excluded": [
            "model weights",
            "virtual environments",
            "general package caches",
            "SSH material",
            "copyrighted narrative corpus",
            "credentials and access tokens",
        ],
        "contains_endpoint_or_credential": False,
    }
    manifest = {**payload, "manifest_sha256": _canonical_sha256(payload)}
    output = options.output if options.output.is_absolute() else repository / options.output
    output.resolve(strict=False).relative_to(repository)
    _atomic_json(output, manifest)
    print(
        json.dumps(
            {
                "file_count": len(entries),
                "manifest_sha256": manifest["manifest_sha256"],
                "output": output.relative_to(repository).as_posix(),
                "total_bytes": payload["total_bytes"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
