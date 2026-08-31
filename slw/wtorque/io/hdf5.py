"""Restartable q-group HDF5 output with manifest and payload checksums."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Self

import h5py
import numpy as np

from slw.wtorque.errors import IncompleteRunError
from slw.wtorque.provenance import canonical_manifest_json, manifest_digest


def _dataset_checksum(group: h5py.Group) -> str:
    digest = hashlib.sha256()
    leaves: list[tuple[str, h5py.Dataset]] = []

    def visit(name: str, value: h5py.Dataset | h5py.Group) -> None:
        if isinstance(value, h5py.Dataset):
            leaves.append((name, value))

    group.visititems(visit)
    for name, dataset in sorted(leaves):
        array = np.ascontiguousarray(dataset[...])
        digest.update(name.encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _write_nested(group: h5py.Group, path: str, value: object) -> None:
    clean = path.strip("/")
    if not clean or clean.startswith("q_data/"):
        raise ValueError(f"invalid q payload path {path!r}")
    parent_path, _, leaf = clean.rpartition("/")
    parent = group.require_group(parent_path) if parent_path else group
    if leaf in parent:
        del parent[leaf]
    parent.create_dataset(leaf, data=np.asarray(value))


class RestartableHDF5:
    """Manage per-q states: pending -> running -> complete/failed."""

    def __init__(
        self,
        path: str | Path,
        qpoints: object,
        manifest: Mapping[str, Any],
        *,
        resume: bool,
    ) -> None:
        self.path = Path(path)
        self.qpoints = np.asarray(qpoints, dtype=np.float64)
        if self.qpoints.ndim != 2 or self.qpoints.shape[1] != 3:
            raise ValueError("qpoints must have shape (nq,3)")
        self.manifest = dict(manifest)
        self._manifest_digest = manifest_digest(manifest)
        if self.path.exists() and not resume:
            raise FileExistsError(f"refusing to overwrite existing output: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = h5py.File(self.path, "r+" if self.path.exists() else "w")
        if "meta" in self.handle:
            stored = str(self.handle["meta"].attrs.get("manifest_sha256", ""))
            if stored != self._manifest_digest:
                self.handle.close()
                raise IncompleteRunError("restart manifest does not match the existing output")
            if self.handle["qpoints"].shape != self.qpoints.shape or not np.allclose(
                self.handle["qpoints"][...], self.qpoints, atol=1e-12
            ):
                self.handle.close()
                raise IncompleteRunError("restart q-point mesh does not match the existing output")
        else:
            meta = self.handle.create_group("meta")
            text_dtype = h5py.string_dtype("utf-8")
            meta.create_dataset(
                "run_manifest_json",
                data=canonical_manifest_json(manifest),
                dtype=text_dtype,
            )
            meta.attrs["manifest_sha256"] = self._manifest_digest
            hashes = manifest.get("source_hashes", {})
            if isinstance(hashes, Mapping):
                meta.create_dataset(
                    "source_hashes",
                    data=np.asarray(
                        [f"{name}={digest}" for name, digest in sorted(hashes.items())],
                        dtype=object,
                    ),
                    dtype=text_dtype,
                )
            provenance = manifest.get("source_provenance_ids", ())
            if isinstance(provenance, (list, tuple)):
                meta.create_dataset(
                    "source_provenance_ids",
                    data=np.asarray([str(value) for value in provenance], dtype=object),
                    dtype=text_dtype,
                )
            resolved = manifest.get("config", {})
            if isinstance(resolved, Mapping):
                electrons = resolved.get("electrons", {})
                magnetic = resolved.get("magnetic_subspace", {})
                labels = {
                    "spin_coordinate": magnetic.get("spin_coordinate")
                    if isinstance(magnetic, Mapping)
                    else None,
                    "site_projection": magnetic.get("site_projection")
                    if isinstance(magnetic, Mapping)
                    else None,
                    "exchange_extraction": electrons.get("exchange_extraction")
                    if isinstance(electrons, Mapping)
                    else None,
                }
                for name, value in labels.items():
                    if value is not None:
                        meta.create_dataset(name, data=str(value), dtype=text_dtype)
            self.handle.create_dataset("qpoints", data=self.qpoints)
            q_data = self.handle.create_group("q_data")
            for index in range(self.qpoints.shape[0]):
                group = q_data.create_group(f"q_{index:06d}")
                group.attrs["status"] = "pending"
            self.handle.flush()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def close(self) -> None:
        if self.handle:
            self.handle.close()

    def status(self, index: int) -> str:
        return str(self.handle[f"q_data/q_{int(index):06d}"].attrs["status"])

    def write_q(self, index: int, payload: Mapping[str, object]) -> None:
        group = self.handle[f"q_data/q_{int(index):06d}"]
        group.attrs["status"] = "running"
        self.handle.flush()
        try:
            for name, value in payload.items():
                _write_nested(group, name, value)
            checksum = _dataset_checksum(group)
            group.attrs["payload_sha256"] = checksum
            self.handle.flush()
            group.attrs["status"] = "complete"
            self.handle.flush()
        except Exception:
            group.attrs["status"] = "failed"
            self.handle.flush()
            raise

    def verify_complete_q(self, index: int) -> None:
        group = self.handle[f"q_data/q_{int(index):06d}"]
        if str(group.attrs.get("status", "pending")) != "complete":
            raise IncompleteRunError(f"q index {index} is not complete")
        expected = str(group.attrs.get("payload_sha256", ""))
        actual = _dataset_checksum(group)
        if not expected or expected != actual:
            raise IncompleteRunError(
                f"q index {index} checksum mismatch: expected {expected}, got {actual}"
            )

    def pending_indices(self) -> tuple[int, ...]:
        pending: list[int] = []
        for index in range(self.qpoints.shape[0]):
            try:
                self.verify_complete_q(index)
            except IncompleteRunError:
                pending.append(index)
        return tuple(pending)

    def materialize(self) -> None:
        """Stack common q payloads into the canonical root output datasets."""

        groups = [self.handle[f"q_data/q_{index:06d}"] for index in range(self.qpoints.shape[0])]
        for index in range(len(groups)):
            self.verify_complete_q(index)
        leaf_names: set[str] | None = None
        for group in groups:
            current: set[str] = set()

            def collect(
                name: str,
                value: h5py.Dataset | h5py.Group,
                target: set[str] = current,
            ) -> None:
                if isinstance(value, h5py.Dataset):
                    target.add(name)

            group.visititems(collect)
            leaf_names = current if leaf_names is None else leaf_names.intersection(current)
        for name in sorted(leaf_names or ()):
            values = np.stack([group[name][...] for group in groups], axis=0)
            parent_path, _, leaf = name.rpartition("/")
            parent = self.handle.require_group(parent_path) if parent_path else self.handle
            if leaf in parent:
                del parent[leaf]
            parent.create_dataset(leaf, data=values)
        self.handle.flush()


__all__ = ["RestartableHDF5"]
