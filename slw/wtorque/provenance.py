"""Deterministic source hashing and equation/source provenance manifests."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import __version__
from .config import RunConfig

SOURCE_PROVENANCE_IDS = (
    "MC2023-EQ13-LOCAL-PARTITION",
    "MC2023-EQ73-77-TR-SPLIT",
    "MC2023-EQ100-XC-ROTATION",
    "MLE2023-EQ20-29-MIXED-RESPONSE",
    "PROJECT-EXTENSION-FINITE-Q-QPAIR",
    "PROJECT-EXTENSION-EXTERNAL-BOSON-PROJECTION",
)


def sha256_file(path: str | Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def build_run_manifest(config: RunConfig) -> dict[str, Any]:
    """Build a deterministic manifest; host and wall-clock fields are excluded."""

    sources: dict[str, Path | None] = {
        "electrons": config.electrons.file,
        "dfpt": config.dfpt.file,
        "phonons": config.phonons_file,
        "magnons": config.magnons_file,
        "onsite_soc_win": (
            None
            if config.electrons.onsite_soc is None
            else config.electrons.onsite_soc.win_file
        ),
    }
    hashes: dict[str, str] = {}
    for name, path in sources.items():
        if path is not None:
            if not path.is_file():
                raise FileNotFoundError(f"configured {name} source does not exist: {path}")
            hashes[name] = sha256_file(path)
    return {
        "schema_version": 1,
        "wtorque_version": __version__,
        "config": config.resolved_dict(),
        "source_hashes": hashes,
        "source_provenance_ids": list(SOURCE_PROVENANCE_IDS),
    }


def canonical_manifest_json(manifest: Mapping[str, Any]) -> str:
    return json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def manifest_digest(manifest: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_manifest_json(manifest).encode("utf-8")).hexdigest()


__all__ = [
    "SOURCE_PROVENANCE_IDS",
    "build_run_manifest",
    "canonical_manifest_json",
    "manifest_digest",
    "sha256_file",
]
