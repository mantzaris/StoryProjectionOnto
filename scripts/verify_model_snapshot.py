#!/usr/bin/env python3
"""Verify and hash the one permitted pinned model snapshot without copying it."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

MAXIMUM_MODEL_BYTES = 11_000_000_000
REQUIRED_FILES = frozenset(
    {
        "LICENSE",
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def atomic_write_json(path: Path, value: object) -> None:
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
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def verify_snapshot(
    *,
    shared_cache: Path,
    snapshot: Path,
    model_configuration_path: Path,
) -> dict[str, object]:
    shared_cache = shared_cache.resolve(strict=True)
    snapshot = snapshot.resolve(strict=True)
    snapshot.relative_to(shared_cache)
    model_configuration_path = model_configuration_path.resolve(strict=True)
    configured = json.loads(model_configuration_path.read_text(encoding="utf-8"))
    revision = configured["revision"]
    repository = configured["repository"]
    if snapshot.name != revision:
        raise ValueError("snapshot directory does not match the pinned revision")

    model_cache_directories = tuple(
        path
        for path in (shared_cache / "hub").glob("models--*")
        if path.is_dir() and not path.is_symlink()
    )
    if len(model_cache_directories) != 1:
        raise ValueError("shared cache must contain exactly one model repository")
    snapshots = tuple(
        path for path in (model_cache_directories[0] / "snapshots").iterdir() if path.is_dir()
    )
    if len(snapshots) != 1 or snapshots[0].name != revision:
        raise ValueError("shared cache must contain exactly one pinned snapshot")
    incomplete = tuple(shared_cache.rglob("*.incomplete"))
    if incomplete:
        raise ValueError("shared cache contains incomplete model downloads")

    paths = tuple(
        sorted(
            (path for path in snapshot.rglob("*") if path.is_file()),
            key=lambda path: path.relative_to(snapshot).as_posix(),
        )
    )
    relative_names = {path.relative_to(snapshot).as_posix() for path in paths}
    missing = sorted(REQUIRED_FILES - relative_names)
    if missing:
        raise ValueError(f"snapshot is missing required files: {', '.join(missing)}")
    if not any(name.endswith(".safetensors") for name in relative_names):
        raise ValueError("snapshot contains no safetensors weights")

    for path in paths:
        resolved_file = path.resolve(strict=True)
        resolved_file.relative_to(shared_cache)
    file_entries = tuple(
        {
            "path": path.relative_to(snapshot).as_posix(),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in paths
    )
    total_bytes = sum(int(entry["size_bytes"]) for entry in file_entries)
    if total_bytes > MAXIMUM_MODEL_BYTES:
        raise ValueError("model snapshot exceeds the registered 11 GB category")

    license_text = (snapshot / "LICENSE").read_text(encoding="utf-8")
    if "Apache License" not in license_text or "Version 2.0" not in license_text:
        raise ValueError("model snapshot does not contain the expected Apache-2.0 license")
    model_config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    quantization = model_config.get("quantization_config", {})
    if str(quantization.get("quant_method", "")).casefold() != "awq":
        raise ValueError("model configuration is not the registered AWQ quantization")

    immutable = {
        "schema_version": "1.0.0",
        "repository": repository,
        "revision": revision,
        "license": "Apache-2.0",
        "quantization": "awq",
        "file_count": len(file_entries),
        "total_bytes": total_bytes,
        "files": file_entries,
        "model_configuration_sha256": sha256_file(model_configuration_path),
        "single_repository_in_shared_cache": True,
        "single_snapshot_in_shared_cache": True,
        "incomplete_file_count": 0,
    }
    return {
        **immutable,
        "manifest_sha256": hashlib.sha256(canonical_bytes(immutable)).hexdigest(),
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-cache", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument(
        "--model-configuration",
        type=Path,
        default=Path("configs/study/model.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    manifest = verify_snapshot(
        shared_cache=arguments.shared_cache,
        snapshot=arguments.snapshot,
        model_configuration_path=arguments.model_configuration,
    )
    atomic_write_json(arguments.output, manifest)
    print(
        f"verified {manifest['file_count']} files, {manifest['total_bytes']} bytes; "
        f"manifest={manifest['manifest_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
