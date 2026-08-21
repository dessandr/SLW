"""Atomic native output for magnon dispersion and lifetime grids."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .dispersion import MagnonDispersionResult
from .pipeline import LifetimeGridResult

LIFETIME_OUTPUT_SCHEMA_VERSION = 1
DISPERSION_OUTPUT_SCHEMA_VERSION = 1


def _atomic_target(path: Path, temporary: Path, *, overwrite: bool) -> None:
    if overwrite:
        os.replace(temporary, path)
        return
    try:
        os.link(temporary, path)
    except FileExistsError as exc:
        raise FileExistsError(f"output already exists: {path}") from exc
    temporary.unlink()


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


def write_dispersion_npz(
    path: str | Path,
    result: MagnonDispersionResult,
    *,
    metadata: Mapping[str, Any],
    overwrite: bool = False,
) -> Path:
    """Write a complete native magnon dispersion atomically from rank zero."""

    output = Path(path).expanduser().resolve()
    if output.suffix.lower() != ".npz":
        raise ValueError("native dispersion output must use the .npz suffix")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        raise FileExistsError(f"dispersion output already exists: {output}")
    payload_metadata = dict(metadata)
    payload_metadata["schema_version"] = DISPERSION_OUTPUT_SCHEMA_VERSION
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
            schema_version=np.asarray(DISPERSION_OUTPUT_SCHEMA_VERSION, dtype=np.int64),
            metadata_json=np.asarray(metadata_json),
            k_points_frac=result.path.k_points_frac,
            x_coordinate_inv_ang=result.path.x_coordinate_inv_ang,
            energy_mev=result.energy_mev,
            goldstone_mask=result.goldstone_mask,
            segment_offsets=result.path.segment_offsets,
            tick_positions_inv_ang=result.path.tick_positions_inv_ang,
            tick_labels=np.asarray(result.path.tick_labels, dtype=np.str_),
        )
        _atomic_target(output, temporary, overwrite=overwrite)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def write_dispersion_plot(
    path: str | Path,
    result: MagnonDispersionResult,
    *,
    title: str | None = None,
    dpi: int = 180,
    overwrite: bool = False,
) -> Path:
    """Write a noninteractive native magnon-band figure atomically."""

    output = Path(path).expanduser().resolve()
    if output.suffix.lower() not in {".png", ".pdf", ".svg"}:
        raise ValueError("dispersion plot must use .png, .pdf, or .svg")
    resolution = int(dpi)
    if resolution < 1:
        raise ValueError("plot dpi must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        raise FileExistsError(f"dispersion plot already exists: {output}")
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axis = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    colors = plt.get_cmap("tab10")
    for segment in range(result.path.n_segments):
        start = int(result.path.segment_offsets[segment])
        stop = int(result.path.segment_offsets[segment + 1])
        x_value = result.path.x_coordinate_inv_ang[start:stop]
        for mode in range(result.energy_mev.shape[1]):
            axis.plot(
                x_value,
                result.energy_mev[start:stop, mode],
                color=colors(mode % 10),
                linewidth=1.5,
                label=f"mode {mode + 1}" if segment == 0 else None,
            )
    for position in result.path.tick_positions_inv_ang:
        axis.axvline(float(position), color="0.82", linewidth=0.8, zorder=0)
    axis.axhline(0.0, color="0.35", linewidth=0.8, zorder=0)
    axis.set_xlim(
        float(result.path.tick_positions_inv_ang[0]),
        float(result.path.tick_positions_inv_ang[-1]),
    )
    axis.set_ylim(bottom=0.0)
    axis.set_xticks(
        result.path.tick_positions_inv_ang,
        labels=result.path.tick_labels,
    )
    axis.set_ylabel("Magnon energy (meV)")
    if title:
        axis.set_title(str(title))
    if result.energy_mev.shape[1] > 1:
        axis.legend(frameon=False)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.stem}.",
        suffix=output.suffix,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        figure.savefig(temporary, dpi=resolution)
        plt.close(figure)
        _atomic_target(output, temporary, overwrite=overwrite)
    finally:
        plt.close(figure)
        temporary.unlink(missing_ok=True)
    return output


__all__ = [
    "DISPERSION_OUTPUT_SCHEMA_VERSION",
    "LIFETIME_OUTPUT_SCHEMA_VERSION",
    "write_dispersion_npz",
    "write_dispersion_plot",
    "write_lifetime_npz",
]
