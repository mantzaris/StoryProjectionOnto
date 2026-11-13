#!/usr/bin/env python3
"""Build or verify the restricted query-blind first-novel index.

This command never searches for a corpus and never prints source paths or prose.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from story_projection_onto.novel_case import (
    LawfulNovelSourceAuthorization,
    RestrictedNovelIndexManifest,
    build_restricted_novel_index,
    load_segmentation_config,
    public_contract_hashes,
    verify_restricted_novel_index,
    write_restricted_manifest,
)


def _aware_datetime(value: str) -> datetime:
    timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise argparse.ArgumentTypeError("timestamp must include a UTC offset")
    return timestamp


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    schemas = subcommands.add_parser("contract-hashes")
    schemas.set_defaults(handler=_contract_hashes)

    build = subcommands.add_parser("build-index")
    build.add_argument("--source", type=Path, required=True)
    build.add_argument("--restricted-root", type=Path, required=True)
    build.add_argument("--index", type=Path, required=True)
    build.add_argument("--manifest", type=Path, required=True)
    build.add_argument("--config", type=Path, required=True)
    build.add_argument("--built-at", type=_aware_datetime, required=True)
    build.add_argument("--expected-source-sha256")
    build.add_argument("--attest-lawful-copy", action="store_true")
    build.set_defaults(handler=_build_index)

    verify = subcommands.add_parser("verify-index")
    verify.add_argument("--index", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    verify.set_defaults(handler=_verify_index)
    return parser.parse_args(arguments)


def _contract_hashes(_: argparse.Namespace) -> int:
    print(json.dumps(public_contract_hashes(), indent=2, sort_keys=True))
    return 0


def _build_index(options: argparse.Namespace) -> int:
    config = load_segmentation_config(options.config)
    authorization = LawfulNovelSourceAuthorization(
        source_path=options.source,
        restricted_root=options.restricted_root,
        lawful_copy_attested=options.attest_lawful_copy,
        expected_source_sha256=options.expected_source_sha256,
    )
    manifest = build_restricted_novel_index(
        authorization,
        config,
        destination=options.index,
        built_at=options.built_at,
    )
    write_restricted_manifest(
        manifest,
        options.manifest,
        restricted_root=options.restricted_root,
    )
    print(
        json.dumps(
            {
                "chapter_count": manifest.chapter_count,
                "index_sha256": manifest.index_sha256,
                "manifest_hash": manifest.content_hash,
                "passage_count": manifest.passage_count,
                "query_blind": manifest.query_blind,
                "release_class": manifest.release_class.value,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _verify_index(options: argparse.Namespace) -> int:
    manifest = RestrictedNovelIndexManifest.model_validate_json(
        options.manifest.read_text(encoding="utf-8")
    )
    verify_restricted_novel_index(options.index, manifest)
    print(
        json.dumps(
            {
                "index_sha256": manifest.index_sha256,
                "manifest_hash": manifest.content_hash,
                "verified": True,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def main(arguments: Sequence[str] | None = None) -> int:
    options = parse_args(arguments)
    return options.handler(options)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
