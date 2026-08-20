"""Atomic native output for magnon lifetime grids."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .pipeline import LifetimeGridResult

LIFETIME_OUTPUT_SCHEMA_VERSION = 1


def write_lifetime_npz(
    path: str | Path,
    result: LifetimeGridResult,
    *,
    metadata: Mapping[str, Any],
    overwrite: bool = False,
) -> Path:
    """Write one complete lifetime result atomically from rank zero."""

    output = Path(path).expanduser().resolve()
    if output.suffix.lower() != ".npz":
        raise ValueError("native lifetime output must use the .npz suffix")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        raise FileExistsError(f"lifetime output already exists: {output}")

    payload_metadata = dict(metadata)
    payload_metadata.update(
        {
            "broadening_mev": result.broadening_mev,
            "negative_tolerance_mev": result.negative_tolerance_mev,
            "schema_version": LIFETIME_OUTPUT_SCHEMA_VERSION,
            "temperature_k": result.temperature_k,
        }
    )
    metadata_json = json.dumps(
        payload_metadata,
        sort_keys=True,
        separators=(",", ":"),
    )

    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp.npz",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(
            temporary,
            schema_version=np.asarray(LIFETIME_OUTPUT_SCHEMA_VERSION, dtype=np.int64),
            metadata_json=np.asarray(metadata_json),
            k_points_frac=result.k_points_frac,
            energy_mev=result.energy_mev,
            self_energy_onshell_mev=result.self_energy_onshell_mev,
            gamma_hwhm_mev=result.gamma_hwhm_mev,
            fwhm_mev=result.fwhm_mev,
            scattering_rate_ps_inv=result.scattering_rate_ps_inv,
            lifetime_ps=result.lifetime_ps,
            valid_damping=result.valid_damping,
        )
        if overwrite:
            os.replace(temporary, output)
        else:
            try:
                os.link(temporary, output)
            except FileExistsError as exc:
                raise FileExistsError(
                    f"lifetime output already exists: {output}"
                ) from exc
            temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)
    return output


__all__ = ["LIFETIME_OUTPUT_SCHEMA_VERSION", "write_lifetime_npz"]
