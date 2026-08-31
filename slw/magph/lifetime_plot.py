"""Native Brillouin-zone plots for magnon lifetime products."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.spatial import Voronoi, cKDTree

from .lifetime import linewidth_observables
from .lswt import magnon_mode_chirality, solve_isotropic_lswt
from .model import SingleIonAnisotropy
from .screening import load_exchange_h5, screen_magnetic_configuration


@dataclass(frozen=True)
class LifetimePlotData:
    """Validated plotting view of a native or retained lifetime product."""

    source: Path
    metadata: dict[str, Any]
    k_points_frac: np.ndarray
    energy_mev: np.ndarray
    gamma_hwhm_mev: np.ndarray
    fwhm_mev: np.ndarray
    scattering_rate_ps_inv: np.ndarray
    lifetime_ps: np.ndarray
    valid_damping: np.ndarray
    magnon_chirality: np.ndarray | None
    mode_labels: tuple[str, ...]
    lattice_ang: np.ndarray | None
    tau_frac: np.ndarray | None
    atom_labels: tuple[str, ...]
    magnetic_atom_indices: np.ndarray | None
    spin_pattern: np.ndarray | None

    @property
    def mode_count(self) -> int:
        return int(self.energy_mev.shape[1])


@dataclass(frozen=True)
class MagneticSymmetryOperation:
    index: int
    rotation: np.ndarray
    translation: np.ndarray
    channel_map: np.ndarray
    rotation_order: int
    determinant: int


@dataclass(frozen=True)
class PlaneGeometry:
    polygons: tuple[np.ndarray, ...]
    source_indices: np.ndarray
    outlines: tuple[tuple[np.ndarray, bool], ...]
    clip_boundary: np.ndarray | None
    minimum: np.ndarray
    maximum: np.ndarray
    add_axis_margin: bool


def _metadata_from_payload(payload: dict[str, np.ndarray]) -> dict[str, Any]:
    if "metadata_json" not in payload:
        return {}
    raw = np.asarray(payload["metadata_json"])
    if raw.size != 1:
        raise ValueError(f"metadata_json must be scalar, got {raw.shape}")
    try:
        metadata = json.loads(str(raw.reshape(()).item()))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("metadata_json is not valid JSON") from exc
    if not isinstance(metadata, dict):
        raise TypeError("metadata_json must contain one JSON object")
    return metadata


def _array_from_aliases(
    payload: dict[str, np.ndarray],
    aliases: tuple[str, ...],
    *,
    required: bool = True,
) -> np.ndarray | None:
    for key in aliases:
        if key in payload:
            return np.asarray(payload[key])
    if required:
        raise KeyError(
            "lifetime product is missing "
            + "/".join(aliases)
            + f"; available keys={sorted(payload)}"
        )
    return None


def _scalar(payload: dict[str, np.ndarray], key: str, default: Any = None) -> Any:
    if key not in payload:
        return default
    value = np.asarray(payload[key])
    if value.size != 1:
        raise ValueError(f"{key} must be scalar, got {value.shape}")
    return value.reshape(-1)[0].item()


def _decode_labels(values: np.ndarray) -> tuple[str, ...]:
    result = []
    for value in np.asarray(values).reshape(-1):
        if isinstance(value, bytes):
            result.append(value.decode("utf-8"))
        else:
            result.append(str(value))
    return tuple(result)


def _geometry_from_exchange(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    with h5py.File(path, "r") as handle:
        required = (
            "basic_data/lattice_ang",
            "basic_data/tau_frac",
            "basic_data/atom_labels",
        )
        missing = [key for key in required if key not in handle]
        if missing:
            raise ValueError(
                f"exchange geometry source {path} is missing {', '.join(missing)}"
            )
        lattice = np.asarray(handle[required[0]], dtype=np.float64)
        tau = np.asarray(handle[required[1]], dtype=np.float64)
        labels = _decode_labels(np.asarray(handle[required[2]]))
    return lattice, tau, labels


def _resolve_exchange_source(
    source: Path,
    metadata: dict[str, Any],
    explicit: str | Path | None,
) -> Path | None:
    candidate = explicit if explicit is not None else metadata.get("exchange_h5")
    if candidate is None:
        return None
    path = Path(candidate).expanduser()
    if not path.is_absolute():
        relative = source.parent / path
        if relative.is_file():
            path = relative
    if explicit is not None and not path.is_file():
        raise FileNotFoundError(f"exchange HDF5 file not found: {path}")
    return path.resolve() if path.is_file() else None


def _anisotropy_from_metadata(
    metadata: dict[str, Any],
) -> SingleIonAnisotropy | None:
    raw = metadata.get("single_ion_anisotropy")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise TypeError("single_ion_anisotropy metadata must be an object or null")
    required = ("energy_mev", "axis", "spin_normalization")
    missing = [name for name in required if name not in raw]
    if missing:
        raise ValueError(
            "single_ion_anisotropy metadata is missing " + ", ".join(missing)
        )
    return SingleIonAnisotropy(
        energy_mev=raw["energy_mev"],
        axis=raw["axis"],
        spin_normalization=raw["spin_normalization"],
        model=raw.get("model", "uniaxial"),
        hamiltonian_sign=raw.get("hamiltonian_sign", "minus"),
    )


def _reconstruct_magnon_chirality(
    exchange_source: Path,
    metadata: dict[str, Any],
    k_points_frac: np.ndarray,
    stored_energy_mev: np.ndarray,
) -> np.ndarray:
    """Rebuild cheap LSWT eigenvectors for a schema-v1 AFM product."""

    required = (
        "magnetic_order",
        "spin_magnitudes",
        "spin_pattern",
        "quantization_axis",
    )
    missing = [name for name in required if metadata.get(name) is None]
    if missing:
        raise ValueError(
            "AFM chirality reconstruction needs lifetime metadata fields: "
            + ", ".join(missing)
        )
    exchange, _ = load_exchange_h5(
        exchange_source,
        reciprocity_atol_mev=float(
            metadata.get("exchange_reciprocity_atol_mev", 1.0e-8)
        ),
    )
    anisotropy = _anisotropy_from_metadata(metadata)
    configuration = screen_magnetic_configuration(
        exchange,
        order=metadata["magnetic_order"],
        spin_pattern=metadata["spin_pattern"],
        spin_magnitudes=metadata["spin_magnitudes"],
        quantization_axis=metadata["quantization_axis"],
        anisotropy=anisotropy,
    )
    spectrum = solve_isotropic_lswt(
        exchange,
        configuration,
        k_points_frac,
        anisotropy=anisotropy,
    )
    mode_count = stored_energy_mev.shape[1]
    raw_energy = spectrum.physical_energies_mev[:, :mode_count]
    raw_chirality = magnon_mode_chirality(
        spectrum.transformation[:, :, :mode_count],
        configuration.spin_pattern,
        physical_mode_count=mode_count,
    )
    if metadata.get("magnon_mode_order") == "chirality_descending":
        permutation = np.argsort(-raw_chirality, axis=1, kind="stable")
        expected_energy = np.take_along_axis(raw_energy, permutation, axis=1)
        current_chirality = np.take_along_axis(
            raw_chirality, permutation, axis=1
        )
    else:
        expected_energy = raw_energy
        current_chirality = raw_chirality
    energy_error = float(
        np.max(np.abs(expected_energy - stored_energy_mev), initial=0.0)
    )
    energy_scale = max(
        float(np.max(np.abs(expected_energy), initial=0.0)),
        1.0,
    )
    if energy_error > 1024.0 * np.finfo(np.float64).eps * energy_scale:
        raise ValueError(
            "reconstructed LSWT energies do not match the lifetime product; "
            f"maximum error={energy_error:.6g} meV"
        )
    return np.asarray(current_chirality, dtype=np.float64)


def _chirality_labels(chirality: np.ndarray) -> tuple[str, ...]:
    representative = np.median(chirality, axis=0)
    labels = []
    for value in representative:
        if np.isclose(abs(value), 1.0, atol=1.0e-6, rtol=0.0):
            labels.append(rf"$\chi={'+' if value >= 0.0 else '-'}1$")
        else:
            labels.append(rf"$\chi\approx{value:+.3g}$")
    return tuple(labels)


def _metadata_geometry(
    metadata: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]] | None:
    keys = ("lattice_ang", "tau_frac", "atom_labels")
    if not all(metadata.get(key) is not None for key in keys):
        return None
    return (
        np.asarray(metadata["lattice_ang"], dtype=np.float64),
        np.asarray(metadata["tau_frac"], dtype=np.float64),
        tuple(str(item) for item in metadata["atom_labels"]),
    )


def _validate_geometry(
    geometry: tuple[np.ndarray, np.ndarray, tuple[str, ...]] | None,
) -> tuple[np.ndarray | None, np.ndarray | None, tuple[str, ...]]:
    if geometry is None:
        return None, None, ()
    lattice, tau, labels = geometry
    lattice = np.asarray(lattice, dtype=np.float64)
    tau = np.mod(np.asarray(tau, dtype=np.float64), 1.0)
    if lattice.shape != (3, 3) or not np.all(np.isfinite(lattice)):
        raise ValueError(
            f"lattice_ang must have finite shape (3,3), got {lattice.shape}"
        )
    if abs(float(np.linalg.det(lattice))) <= np.finfo(np.float64).eps:
        raise ValueError("lattice_ang must be nonsingular")
    if tau.ndim != 2 or tau.shape[1:] != (3,) or not np.all(np.isfinite(tau)):
        raise ValueError(f"tau_frac must have finite shape (natom,3), got {tau.shape}")
    if len(labels) != tau.shape[0]:
        raise ValueError(
            f"atom_labels count {len(labels)} != tau_frac atom count {tau.shape[0]}"
        )
    return lattice, tau, tuple(labels)


def load_lifetime_plot_data(
    path: str | Path,
    *,
    exchange_h5: str | Path | None = None,
    physical_mode_count: int | None = None,
) -> LifetimePlotData:
    """Load native lifetime data and canonicalize AFM chirality branches."""

    source = Path(path).expanduser().resolve()
    selected_keys = {
        "metadata_json",
        "k_points_frac",
        "k_mesh_frac",
        "energy_mev",
        "energy",
        "gamma_hwhm_mev",
        "linewidth",
        "fwhm_mev",
        "scattering_rate_ps_inv",
        "lifetime_ps",
        "valid_damping",
        "magnon_chirality",
        "physical_channel_count",
    }
    with np.load(source, allow_pickle=False) as loaded:
        payload = {key: loaded[key] for key in loaded.files if key in selected_keys}
    metadata = _metadata_from_payload(payload)
    exchange_source = _resolve_exchange_source(source, metadata, exchange_h5)
    k_points = np.asarray(
        _array_from_aliases(payload, ("k_points_frac", "k_mesh_frac")),
        dtype=np.float64,
    )
    energy = np.asarray(
        _array_from_aliases(payload, ("energy_mev", "energy")),
        dtype=np.float64,
    )
    gamma = np.asarray(
        _array_from_aliases(payload, ("gamma_hwhm_mev", "linewidth")),
        dtype=np.float64,
    )
    if k_points.ndim != 2 or k_points.shape[1:] != (3,) or k_points.shape[0] == 0:
        raise ValueError(
            f"k_points_frac must have nonempty shape (nk,3), got {k_points.shape}"
        )
    if energy.ndim != 2 or gamma.ndim != 2 or energy.shape != gamma.shape:
        raise ValueError(
            f"energy/linewidth must share shape (nk,nmode), got {energy.shape}/{gamma.shape}"
        )
    if energy.shape[0] != k_points.shape[0]:
        raise ValueError(
            f"k-point count {k_points.shape[0]} != observable count {energy.shape[0]}"
        )
    if not np.all(np.isfinite(k_points)) or not np.all(np.isfinite(energy)):
        raise ValueError("k_points_frac and energy_mev must be finite")

    stored_count = _scalar(payload, "physical_channel_count")
    requested_count = (
        physical_mode_count if physical_mode_count is not None else stored_count
    )
    mode_count = energy.shape[1] if requested_count is None else int(requested_count)
    if mode_count < 1 or mode_count > energy.shape[1]:
        raise ValueError(
            f"physical_mode_count={mode_count} is invalid for {energy.shape[1]} stored modes"
        )
    energy = energy[:, :mode_count]
    gamma = gamma[:, :mode_count]
    derived = linewidth_observables(
        gamma,
        negative_tolerance_mev=float(metadata.get("negative_tolerance_mev", 0.0)),
    )

    aliases = {
        "fwhm": ("fwhm_mev",),
        "rate": ("scattering_rate_ps_inv",),
        "lifetime": ("lifetime_ps",),
        "valid": ("valid_damping",),
    }
    optional = {
        name: _array_from_aliases(payload, keys, required=False)
        for name, keys in aliases.items()
    }
    fwhm = (
        derived.fwhm_mev
        if optional["fwhm"] is None
        else np.asarray(optional["fwhm"], dtype=np.float64)[:, :mode_count]
    )
    rate = (
        derived.scattering_rate_ps_inv
        if optional["rate"] is None
        else np.asarray(optional["rate"], dtype=np.float64)[:, :mode_count]
    )
    lifetime = (
        derived.lifetime_ps
        if optional["lifetime"] is None
        else np.asarray(optional["lifetime"], dtype=np.float64)[:, :mode_count]
    )
    valid = (
        derived.valid_damping
        if optional["valid"] is None
        else np.asarray(optional["valid"], dtype=np.bool_)[:, :mode_count]
    )
    for name, value in (
        ("fwhm_mev", fwhm),
        ("scattering_rate_ps_inv", rate),
        ("lifetime_ps", lifetime),
        ("valid_damping", valid),
    ):
        if value.shape != energy.shape:
            raise ValueError(
                f"{name} shape {value.shape} != energy shape {energy.shape}"
            )

    chirality_raw = _array_from_aliases(
        payload, ("magnon_chirality",), required=False
    )
    chirality = (
        None
        if chirality_raw is None
        else np.asarray(chirality_raw, dtype=np.float64)[:, :mode_count]
    )
    is_afm = metadata.get("magnetic_order") == "collinear_afm"
    if chirality is None and is_afm and exchange_source is not None:
        chirality = _reconstruct_magnon_chirality(
            exchange_source,
            metadata,
            k_points,
            energy,
        )
    if chirality is not None:
        if chirality.shape != energy.shape:
            raise ValueError(
                f"magnon_chirality shape {chirality.shape} != energy shape {energy.shape}"
            )
        if not np.all(np.isfinite(chirality)):
            raise ValueError("magnon_chirality must contain only finite values")
        tolerance = 256.0 * np.finfo(np.float64).eps
        if np.any(np.abs(chirality) > 1.0 + tolerance):
            raise ValueError("magnon_chirality must be bounded by [-1,1]")
        permutation = np.argsort(-chirality, axis=1, kind="stable")

        def reorder(value: np.ndarray) -> np.ndarray:
            return np.take_along_axis(value, permutation, axis=1)

        energy = reorder(energy)
        gamma = reorder(gamma)
        fwhm = reorder(fwhm)
        rate = reorder(rate)
        lifetime = reorder(lifetime)
        valid = reorder(valid)
        chirality = reorder(chirality)
        mode_labels = _chirality_labels(chirality)
    else:
        mode_labels = tuple(f"mode {index + 1}" for index in range(mode_count))

    geometry = _metadata_geometry(metadata)
    if geometry is None and exchange_source is not None:
        geometry = _geometry_from_exchange(exchange_source)
    lattice, tau, labels = _validate_geometry(geometry)

    magnetic_raw = metadata.get("magnetic_atom_indices")
    spin_raw = metadata.get("spin_pattern")
    magnetic = (
        None if magnetic_raw is None else np.asarray(magnetic_raw, dtype=np.int64)
    )
    spins = None if spin_raw is None else np.asarray(spin_raw, dtype=np.float64)
    if magnetic is not None and magnetic.shape != (mode_count,):
        raise ValueError(
            f"magnetic_atom_indices shape {magnetic.shape} != mode count {(mode_count,)}"
        )
    if spins is not None and spins.shape != (mode_count,):
        raise ValueError(
            f"spin_pattern shape {spins.shape} != mode count {(mode_count,)}"
        )

    return LifetimePlotData(
        source=source,
        metadata=metadata,
        k_points_frac=np.mod(k_points, 1.0),
        energy_mev=energy,
        gamma_hwhm_mev=gamma,
        fwhm_mev=fwhm,
        scattering_rate_ps_inv=rate,
        lifetime_ps=lifetime,
        valid_damping=valid,
        magnon_chirality=chirality,
        mode_labels=mode_labels,
        lattice_ang=lattice,
        tau_frac=tau,
        atom_labels=labels,
        magnetic_atom_indices=magnetic,
        spin_pattern=spins,
    )


def _validate_plane_axes(axes: Any) -> tuple[int, int]:
    result = tuple(int(axis) for axis in axes)
    if (
        len(result) != 2
        or len(set(result)) != 2
        or any(axis not in (0, 1, 2) for axis in result)
    ):
        raise ValueError(
            f"plane_axes must contain two different axes in [0,2], got {result}"
        )
    return result


def _slice_axis(plane_axes: tuple[int, int]) -> int:
    return next(iter({0, 1, 2} - set(plane_axes)))


def _select_plane_slice(
    points: np.ndarray,
    *,
    axis: int,
    value: float,
    decimals: int,
) -> tuple[np.ndarray, float]:
    coordinate = np.round(np.mod(points[:, axis], 1.0), int(decimals))
    available = np.unique(coordinate)
    target = float(value) % 1.0
    distance = np.abs(((available - target + 0.5) % 1.0) - 0.5)
    chosen = float(available[int(np.argmin(distance))])
    return np.isclose(coordinate, chosen, atol=10.0 ** (-int(decimals))), chosen


def _requested_slices(
    points: np.ndarray,
    *,
    plane_axes: tuple[int, int],
    requested: tuple[float, ...],
    all_slices: bool,
    decimals: int,
) -> tuple[tuple[float, np.ndarray], ...]:
    normal = _slice_axis(plane_axes)
    values = (
        np.unique(np.round(np.mod(points[:, normal], 1.0), int(decimals)))
        if all_slices
        else np.asarray(requested, dtype=np.float64)
    )
    selected: list[tuple[float, np.ndarray]] = []
    for value in values:
        mask, chosen = _select_plane_slice(
            points,
            axis=normal,
            value=float(value),
            decimals=decimals,
        )
        if not any(
            np.isclose(chosen, item[0], atol=10.0 ** (-decimals)) for item in selected
        ):
            selected.append((chosen, mask))
    if not selected:
        raise ValueError("at least one reciprocal-space slice is required")
    return tuple(selected)


def _reciprocal_plane_basis_2d(
    lattice_ang: np.ndarray,
    axes: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    reciprocal = 2.0 * np.pi * np.linalg.inv(lattice_ang).T
    first = np.asarray(reciprocal[axes[0]], dtype=np.float64)
    second = np.asarray(reciprocal[axes[1]], dtype=np.float64)
    e1 = first / np.linalg.norm(first)
    perpendicular = second - float(np.dot(second, e1)) * e1
    norm = float(np.linalg.norm(perpendicular))
    if norm < 1.0e-12:
        raise ValueError("selected reciprocal plane vectors are nearly collinear")
    e2 = perpendicular / norm
    return (
        np.asarray((np.dot(first, e1), np.dot(first, e2))),
        np.asarray((np.dot(second, e1), np.dot(second, e2))),
    )


def _plane_points(
    points_frac: np.ndarray,
    axes: tuple[int, int],
    first: np.ndarray,
    second: np.ndarray,
) -> np.ndarray:
    f1 = ((points_frac[:, axes[0]] + 0.5) % 1.0) - 0.5
    f2 = ((points_frac[:, axes[1]] + 0.5) % 1.0) - 0.5
    return f1[:, None] * first[None, :] + f2[:, None] * second[None, :]


def _first_bz_polygon(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    lattice_points = np.asarray(
        [ia * first + ib * second for ia in range(-2, 3) for ib in range(-2, 3)],
        dtype=np.float64,
    )
    origin = 12
    voronoi = Voronoi(lattice_points)
    region = voronoi.regions[voronoi.point_region[origin]]
    if not region or -1 in region:
        raise RuntimeError("could not construct the first Brillouin-zone polygon")
    polygon = voronoi.vertices[region]
    center = np.mean(polygon, axis=0)
    angle = np.arctan2(polygon[:, 1] - center[1], polygon[:, 0] - center[0])
    return polygon[np.argsort(angle)]


def _periodic_voronoi_cells(
    points: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    tile: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    tile_range = range(-int(tile), int(tile) + 1)
    tiled = [
        points + ia * first + ib * second for ia in tile_range for ib in tile_range
    ]
    all_points = np.vstack(tiled)
    voronoi = Voronoi(all_points)
    count = points.shape[0]
    polygons: list[np.ndarray] = []
    indices: list[int] = []
    for tile_index in range(len(tiled)):
        offset = tile_index * count
        for point_index in range(count):
            region = voronoi.regions[voronoi.point_region[offset + point_index]]
            if not region or -1 in region:
                continue
            polygons.append(voronoi.vertices[region])
            indices.append(point_index)
    return polygons, np.asarray(indices, dtype=np.int64)


def _intersects_box(
    polygon: np.ndarray, minimum: np.ndarray, maximum: np.ndarray
) -> bool:
    return bool(
        np.all(np.max(polygon, axis=0) >= minimum)
        and np.all(np.min(polygon, axis=0) <= maximum)
    )


def _build_plane_geometry(
    points_frac: np.ndarray,
    lattice_ang: np.ndarray,
    axes: tuple[int, int],
    *,
    tile: int,
    bz_mode: str,
    periodic_repeats: int,
    periodic_view: str,
    periodic_padding: float,
    decimals: int,
) -> PlaneGeometry:
    unique = np.unique(np.round(points_frac[:, axes], int(decimals)), axis=0)
    if unique.shape[0] != points_frac.shape[0]:
        raise ValueError("selected plane contains duplicate in-plane k points")
    if points_frac.shape[0] < 4:
        raise ValueError("selected plane needs at least four points for a BZ map")
    first, second = _reciprocal_plane_basis_2d(lattice_ang, axes)
    points = _plane_points(points_frac, axes, first, second)
    boundary = _first_bz_polygon(first, second)
    mode = str(bz_mode).strip().lower()
    view = str(periodic_view).strip().lower()
    repeats = int(periodic_repeats)
    padding = float(periodic_padding)
    if mode not in {"clip", "periodic"}:
        raise ValueError("bz_mode must be 'clip' or 'periodic'")
    if view not in {"central", "neighbors"}:
        raise ValueError("periodic_view must be 'central' or 'neighbors'")
    if repeats < 0 or padding < 0.0:
        raise ValueError("periodic repeats and padding must be non-negative")
    construction_tile = max(
        int(tile), repeats + 1 if mode == "periodic" and view == "neighbors" else 1
    )
    if mode == "periodic" and view == "central":
        construction_tile = max(construction_tile, 2)
    polygons, source_indices = _periodic_voronoi_cells(
        points, first, second, construction_tile
    )
    outlines: list[tuple[np.ndarray, bool]]
    clip_boundary: np.ndarray | None
    add_margin = True
    if mode == "periodic" and view == "neighbors":
        outlines = [
            (boundary + ia * first + ib * second, ia == 0 and ib == 0)
            for ia in range(-repeats, repeats + 1)
            for ib in range(-repeats, repeats + 1)
        ]
        visible_vertices = np.vstack([polygon for polygon, _ in outlines])
        minimum = np.min(visible_vertices, axis=0)
        maximum = np.max(visible_vertices, axis=0)
        clip_boundary = None
    elif mode == "periodic":
        outlines = [(boundary, True)]
        minimum = np.min(boundary, axis=0)
        maximum = np.max(boundary, axis=0)
        span = np.maximum(maximum - minimum, np.finfo(np.float64).eps)
        minimum = minimum - padding * span
        maximum = maximum + padding * span
        clip_boundary = None
        add_margin = False
    else:
        outlines = [(boundary, True)]
        minimum = np.min(boundary, axis=0)
        maximum = np.max(boundary, axis=0)
        clip_boundary = boundary
    keep = np.asarray(
        [_intersects_box(polygon, minimum, maximum) for polygon in polygons],
        dtype=np.bool_,
    )
    return PlaneGeometry(
        polygons=tuple(
            polygon for polygon, visible in zip(polygons, keep, strict=True) if visible
        ),
        source_indices=source_indices[keep],
        outlines=tuple(outlines),
        clip_boundary=clip_boundary,
        minimum=minimum,
        maximum=maximum,
        add_axis_margin=add_margin,
    )


def _rotation_order(rotation: np.ndarray, maximum: int = 24) -> int:
    product = np.eye(3, dtype=np.int64)
    for order in range(1, maximum + 1):
        product = product @ rotation
        if np.array_equal(product, np.eye(3, dtype=np.int64)):
            return order
    return 0


def _species(labels: tuple[str, ...]) -> tuple[tuple[str, ...], np.ndarray]:
    names = tuple(re.sub(r"\d+$", "", label) for label in labels)
    mapping = {name: index + 1 for index, name in enumerate(dict.fromkeys(names))}
    return names, np.asarray([mapping[name] for name in names], dtype=np.int32)


def _atom_map(
    positions: np.ndarray,
    species: tuple[str, ...],
    lattice: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    tolerance_ang: float,
) -> np.ndarray:
    transformed = np.mod(positions @ rotation.T + translation, 1.0)
    result = np.full(positions.shape[0], -1, dtype=np.int64)
    error = np.full(positions.shape[0], np.inf, dtype=np.float64)
    for source, target in enumerate(transformed):
        candidates = np.asarray(
            [index for index, name in enumerate(species) if name == species[source]],
            dtype=np.int64,
        )
        delta = target[None, :] - positions[candidates]
        delta -= np.rint(delta)
        distance = np.linalg.norm(delta @ lattice, axis=1)
        nearest = int(np.argmin(distance))
        result[source] = int(candidates[nearest])
        error[source] = float(distance[nearest])
    if np.any(error > tolerance_ang) or np.unique(result).size != positions.shape[0]:
        raise ValueError("symmetry operation does not produce a bijective atom map")
    return result


def _find_transposing_operations(
    data: LifetimePlotData,
    *,
    symprec: float,
    atom_tolerance_ang: float,
) -> tuple[MagneticSymmetryOperation, ...]:
    if data.lattice_ang is None or data.tau_frac is None or not data.atom_labels:
        raise ValueError(
            "automatic rotation-error plotting requires lattice/atom geometry"
        )
    if data.magnetic_atom_indices is None or data.spin_pattern is None:
        raise ValueError("automatic rotation-error plotting requires magnetic metadata")
    import spglib

    species_names, numbers = _species(data.atom_labels)
    symmetry = spglib.get_symmetry(
        (data.lattice_ang, data.tau_frac, numbers), symprec=float(symprec)
    )
    if symmetry is None:
        raise RuntimeError("spglib did not find crystal symmetry operations")
    local_from_global = {
        int(atom): index for index, atom in enumerate(data.magnetic_atom_indices)
    }
    operations: list[MagneticSymmetryOperation] = []
    for index, (rotation, translation) in enumerate(
        zip(symmetry["rotations"], symmetry["translations"], strict=True)
    ):
        try:
            atom_map = _atom_map(
                data.tau_frac,
                species_names,
                data.lattice_ang,
                np.asarray(rotation, dtype=np.int64),
                np.asarray(translation, dtype=np.float64),
                float(atom_tolerance_ang),
            )
        except ValueError:
            continue
        mapped = atom_map[data.magnetic_atom_indices]
        if any(int(atom) not in local_from_global for atom in mapped):
            continue
        channel_map = np.asarray(
            [local_from_global[int(atom)] for atom in mapped], dtype=np.int64
        )
        if not np.all(data.spin_pattern[channel_map] == -data.spin_pattern):
            continue
        rotation_array = np.asarray(rotation, dtype=np.int64)
        operations.append(
            MagneticSymmetryOperation(
                index=index,
                rotation=rotation_array,
                translation=np.asarray(translation, dtype=np.float64),
                channel_map=channel_map,
                rotation_order=_rotation_order(rotation_array),
                determinant=round(float(np.linalg.det(rotation_array))),
            )
        )
    return tuple(operations)


def _select_operation(
    operations: tuple[MagneticSymmetryOperation, ...],
    requested: int | None,
    k_points: np.ndarray,
    *,
    tolerance: float,
    workers: int,
) -> MagneticSymmetryOperation:
    if not operations:
        raise ValueError(
            "no crystal symmetry operation exchanges all magnetic sublattices"
        )
    if requested is not None:
        for operation in operations:
            if operation.index == requested:
                _momentum_map(
                    k_points,
                    operation.rotation,
                    tolerance=tolerance,
                    workers=workers,
                )
                return operation
        raise ValueError(
            f"symmetry operation {requested} is unavailable; "
            f"available={[item.index for item in operations]}"
        )
    compatible = []
    for operation in operations:
        try:
            _momentum_map(
                k_points,
                operation.rotation,
                tolerance=tolerance,
                workers=workers,
            )
        except ValueError:
            continue
        compatible.append(operation)
    if not compatible:
        raise ValueError(
            "no magnetic sublattice-transposing operation maps the stored shifted "
            "k mesh bijectively"
        )
    return max(
        compatible,
        key=lambda item: (
            item.determinant == 1,
            item.rotation_order,
            -item.index,
        ),
    )


def _momentum_map(
    k_points: np.ndarray,
    rotation: np.ndarray,
    *,
    tolerance: float,
    workers: int,
) -> tuple[np.ndarray, np.ndarray]:
    transformed = np.mod(k_points @ np.linalg.inv(rotation), 1.0)
    tree = cKDTree(k_points, boxsize=1.0)
    distance, momentum_map = tree.query(transformed, k=1, workers=int(workers))
    if (
        np.any(distance > tolerance)
        or np.unique(momentum_map).size != k_points.shape[0]
    ):
        raise ValueError(
            "selected rotation does not map the stored k mesh bijectively; "
            f"maximum distance={float(np.max(distance)):.6g}"
        )
    return np.asarray(momentum_map, dtype=np.int64), np.asarray(
        distance, dtype=np.float64
    )


def _rotation_comparison(
    values: np.ndarray,
    k_points: np.ndarray,
    rotation: np.ndarray,
    channel_map: np.ndarray,
    *,
    tolerance: float,
    workers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    momentum_map, distance = _momentum_map(
        k_points,
        rotation,
        tolerance=tolerance,
        workers=workers,
    )
    rotated = values[momentum_map][:, channel_map]
    absolute = np.abs(values - rotated)
    floor = np.finfo(np.float64).eps * max(
        float(np.max(np.abs(values[np.isfinite(values)]), initial=1.0)), 1.0
    )
    relative = 2.0 * absolute / (np.abs(values) + np.abs(rotated) + floor)
    return absolute, relative, distance


def _finite_percentile(values: np.ndarray, percentile: float, default: float) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.percentile(finite, percentile)) if finite.size else float(default)


def _safe_limits(low: float, high: float) -> tuple[float, float]:
    if not np.isfinite(low) or not np.isfinite(high):
        return 0.0, 1.0
    if high < low:
        low, high = high, low
    if np.isclose(low, high):
        padding = max(abs(low), abs(high), 1.0) * 1.0e-6
        low -= padding
        high += padding
    return float(low), float(high)


def _slice_tag(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".").replace("-", "m").replace(".", "p")


def _decorate_axis(
    axis: Any,
    collection: Any,
    geometry: PlaneGeometry,
    axes: tuple[int, int],
    args: argparse.Namespace,
) -> None:
    from matplotlib import pyplot as plt

    if geometry.clip_boundary is not None:
        clip = plt.Polygon(
            geometry.clip_boundary, closed=True, facecolor="none", edgecolor="none"
        )
        axis.add_patch(clip)
        collection.set_clip_path(clip)
    for outline, central in geometry.outlines:
        closed = np.vstack((outline, outline[0]))
        axis.plot(
            closed[:, 0],
            closed[:, 1],
            color=args.bz_color,
            linewidth=args.bz_lw if central else args.neighbor_bz_lw,
            alpha=1.0 if central else args.neighbor_bz_alpha,
            zorder=4,
        )
    span = np.maximum(geometry.maximum - geometry.minimum, np.finfo(np.float64).eps)
    margin = args.margin if geometry.add_axis_margin else 0.0
    axis.set_xlim(
        geometry.minimum[0] - margin * span[0],
        geometry.maximum[0] + margin * span[0],
    )
    axis.set_ylim(
        geometry.minimum[1] - margin * span[1],
        geometry.maximum[1] + margin * span[1],
    )
    axis.set_aspect("equal")
    axis.set_xlabel(rf"$k_{{{axes[0] + 1}}}$ ($\AA^{{-1}}$)")
    axis.set_ylabel(rf"$k_{{{axes[1] + 1}}}$ ($\AA^{{-1}}$)")


def _save_figure(
    figure: Any,
    *,
    output_dir: Path,
    stem: str,
    formats: tuple[str, ...],
    dpi: int,
    overwrite: bool,
) -> list[Path]:
    from matplotlib import pyplot as plt

    paths = [output_dir / f"{stem}.{extension}" for extension in formats]
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "plot output already exists: " + ", ".join(str(path) for path in existing)
        )
    for path in paths:
        figure.savefig(path, dpi=int(dpi), bbox_inches="tight")
    plt.close(figure)
    return paths


def _plot_modes(
    values: np.ndarray,
    *,
    mode_labels: tuple[str, ...],
    geometry: PlaneGeometry,
    axes: tuple[int, int],
    title: str,
    label: str,
    cmap: str,
    limits: tuple[float, float],
    args: argparse.Namespace,
    output_dir: Path,
    stem: str,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.collections import PolyCollection

    mode_count = values.shape[1]
    figure, panels = plt.subplots(
        1,
        mode_count,
        figsize=(args.panel_width * mode_count, args.panel_height),
        squeeze=False,
        constrained_layout=True,
    )
    collection = None
    for mode in range(mode_count):
        collection = PolyCollection(
            geometry.polygons,
            array=values[geometry.source_indices, mode],
            cmap=cmap,
            edgecolor=args.edgecolor,
            linewidth=args.linewidth,
        )
        collection.set_clim(*limits)
        panel = panels[0, mode]
        panel.add_collection(collection)
        _decorate_axis(panel, collection, geometry, axes, args)
        panel.set_title(mode_labels[mode])
    figure.suptitle(title)
    figure.colorbar(collection, ax=list(panels.flat), label=label, shrink=0.86)
    return _save_figure(
        figure,
        output_dir=output_dir,
        stem=stem,
        formats=tuple(args.formats),
        dpi=args.dpi,
        overwrite=args.overwrite,
    )


def _plot_splitting(
    energy: np.ndarray,
    gamma: np.ndarray,
    *,
    geometry: PlaneGeometry,
    axes: tuple[int, int],
    channels: tuple[int, int],
    chirality_ordered: bool,
    limits: tuple[tuple[float, float], tuple[float, float]],
    slice_title: str,
    args: argparse.Namespace,
    output_dir: Path,
    stem: str,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.collections import PolyCollection

    first, second = channels
    values = np.column_stack(
        (
            energy[:, first] - energy[:, second],
            (gamma[:, first] - gamma[:, second]) * args.linewidth_scale,
        )
    )
    if chirality_ordered and channels == (0, 1):
        labels = (
            r"$E_{\chi+}-E_{\chi-}$ (meV)",
            rf"$\Gamma_{{\chi+}}-\Gamma_{{\chi-}}$ ({args.linewidth_unit})",
        )
    else:
        labels = (
            rf"$E_{{{first + 1}}}-E_{{{second + 1}}}$ (meV)",
            rf"$\Gamma_{{{first + 1}}}-\Gamma_{{{second + 1}}}$ ({args.linewidth_unit})",
        )
    titles = ("magnon energy splitting", "linewidth splitting")
    figure, panels = plt.subplots(
        1,
        2,
        figsize=(2.0 * args.panel_width, args.panel_height),
        squeeze=False,
        constrained_layout=True,
    )
    for panel_index in range(2):
        collection = PolyCollection(
            geometry.polygons,
            array=values[geometry.source_indices, panel_index],
            cmap=args.diverging_cmap,
            edgecolor=args.edgecolor,
            linewidth=args.linewidth,
        )
        collection.set_clim(*limits[panel_index])
        panel = panels[0, panel_index]
        panel.add_collection(collection)
        _decorate_axis(panel, collection, geometry, axes, args)
        panel.set_title(titles[panel_index])
        figure.colorbar(collection, ax=panel, label=labels[panel_index], shrink=0.86)
    figure.suptitle(f"Mode splitting, {slice_title}")
    return _save_figure(
        figure,
        output_dir=output_dir,
        stem=stem,
        formats=tuple(args.formats),
        dpi=args.dpi,
        overwrite=args.overwrite,
    )


def _plot_rotation_error(
    absolute: np.ndarray,
    relative: np.ndarray,
    *,
    mode_labels: tuple[str, ...],
    geometry: PlaneGeometry,
    axes: tuple[int, int],
    limits: tuple[tuple[float, float], tuple[float, float]],
    slice_title: str,
    args: argparse.Namespace,
    output_dir: Path,
    stem: str,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.collections import PolyCollection

    mode_count = absolute.shape[1]
    combined = np.concatenate(
        (absolute * args.linewidth_scale, 100.0 * relative), axis=1
    )
    figure, panels = plt.subplots(
        2,
        mode_count,
        figsize=(args.panel_width * mode_count, 1.85 * args.panel_height),
        squeeze=False,
        constrained_layout=True,
    )
    row_collections = []
    for row in range(2):
        row_collection = None
        for mode in range(mode_count):
            value_index = mode if row == 0 else mode_count + mode
            row_collection = PolyCollection(
                geometry.polygons,
                array=combined[geometry.source_indices, value_index],
                cmap=args.error_cmap,
                edgecolor=args.edgecolor,
                linewidth=args.linewidth,
            )
            row_collection.set_clim(*limits[row])
            panel = panels[row, mode]
            panel.add_collection(row_collection)
            _decorate_axis(panel, row_collection, geometry, axes, args)
            panel.set_title(
                f"{mode_labels[mode]}: "
                + (r"$|\Delta\Gamma|$" if row == 0 else "relative error")
            )
        row_collections.append(row_collection)
    figure.suptitle(f"Rotation + mode-swap residual, {slice_title}")
    figure.colorbar(
        row_collections[0],
        ax=list(panels[0]),
        label=rf"$|\Delta\Gamma|$ ({args.linewidth_unit})",
        shrink=0.84,
    )
    figure.colorbar(
        row_collections[1],
        ax=list(panels[1]),
        label="symmetric relative error (%)",
        shrink=0.84,
    )
    return _save_figure(
        figure,
        output_dir=output_dir,
        stem=stem,
        formats=tuple(args.formats),
        dpi=args.dpi,
        overwrite=args.overwrite,
    )


def plot_lifetime(args: argparse.Namespace) -> tuple[list[Path], Path, dict[str, Any]]:
    """Create mode-resolved BZ maps from one lifetime result."""

    data = load_lifetime_plot_data(
        args.input,
        exchange_h5=args.exchange_h5,
        physical_mode_count=args.physical_mode_count,
    )
    if data.lattice_ang is None:
        raise ValueError(
            "BZ plotting requires lattice geometry; use exchange_h5 or regenerate "
            "the lifetime product with embedded native geometry"
        )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "plot_summary.json"
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(f"plot summary already exists: {summary_path}")
    axes = _validate_plane_axes(args.plane_axes)
    normal = _slice_axis(axes)
    slices = _requested_slices(
        data.k_points_frac,
        plane_axes=axes,
        requested=tuple(args.slices),
        all_slices=bool(args.all_slices),
        decimals=int(args.k_round),
    )
    plots = set(args.plots)
    if "all" in plots:
        plots = {"energy", "linewidth", "scattering-rate", "lifetime"}
        if data.mode_count >= 2:
            plots.add("splitting")
    selected_union = np.logical_or.reduce([mask for _, mask in slices])

    energy = data.energy_mev
    gamma = data.gamma_hwhm_mev * args.linewidth_scale
    fwhm = data.fwhm_mev * args.linewidth_scale
    rate = data.scattering_rate_ps_inv * args.scattering_rate_scale
    log_lifetime = np.full(data.lifetime_ps.shape, np.nan, dtype=np.float64)
    positive = np.isfinite(data.lifetime_ps) & (data.lifetime_ps > 0.0)
    log_lifetime[positive] = np.log10(data.lifetime_ps[positive])
    quantities = {
        "energy": (
            energy,
            "Magnon energy",
            "Energy (meV)",
            args.energy_cmap,
        ),
        "linewidth": (
            gamma,
            "Magnon HWHM linewidth",
            rf"$\Gamma=-\mathrm{{Im}}\,\Sigma^R$ ({args.linewidth_unit})",
            args.linewidth_cmap,
        ),
        "fwhm": (
            fwhm,
            "Magnon FWHM linewidth",
            f"FWHM ({args.linewidth_unit})",
            args.linewidth_cmap,
        ),
        "scattering-rate": (
            rate,
            "Magnon scattering rate",
            rf"$\tau^{{-1}}$ ({args.scattering_rate_unit})",
            args.scattering_rate_cmap,
        ),
        "lifetime": (
            log_lifetime,
            "Magnon lifetime",
            r"$\log_{10}[\tau\,/\,\mathrm{ps}]$",
            args.lifetime_cmap,
        ),
    }
    limits = {
        name: _safe_limits(
            _finite_percentile(values[selected_union], args.vmin_percentile, 0.0),
            _finite_percentile(values[selected_union], args.vmax_percentile, 1.0),
        )
        for name, (values, _, _, _) in quantities.items()
    }
    split_channels = tuple(int(index) for index in args.split_modes)
    if "splitting" in plots:
        if (
            data.metadata.get("magnetic_order") == "collinear_afm"
            and data.magnon_chirality is None
        ):
            raise ValueError(
                "AFM splitting requires chirality-resolved branches; provide "
                "exchange_h5 to reconstruct schema-v1 output labels"
            )
        if any(index < 0 or index >= data.mode_count for index in split_channels):
            raise ValueError(
                f"split_modes={split_channels} is invalid for {data.mode_count} modes"
            )
        energy_split = energy[:, split_channels[0]] - energy[:, split_channels[1]]
        gamma_split = gamma[:, split_channels[0]] - gamma[:, split_channels[1]]
        split_limits = tuple(
            _safe_limits(-limit, limit)
            for limit in (
                _finite_percentile(
                    np.abs(energy_split[selected_union]), args.error_percentile, 1.0
                ),
                _finite_percentile(
                    np.abs(gamma_split[selected_union]), args.error_percentile, 1.0
                ),
            )
        )
    else:
        split_limits = ((_safe_limits(-1.0, 1.0)), (_safe_limits(-1.0, 1.0)))

    operation: MagneticSymmetryOperation | None = None
    mapping_distance: np.ndarray | None = None
    rotation_absolute: np.ndarray | None = None
    rotation_relative: np.ndarray | None = None
    if "rotation-error" in plots:
        if args.rotation is not None:
            if args.channel_map is None:
                raise ValueError("explicit rotation requires channel_map")
            rotation = np.asarray(args.rotation, dtype=np.int64).reshape(3, 3)
            channel_map = np.asarray(args.channel_map, dtype=np.int64)
            translation = np.zeros(3, dtype=np.float64)
        else:
            operation = _select_operation(
                _find_transposing_operations(
                    data,
                    symprec=args.symprec,
                    atom_tolerance_ang=args.atom_tolerance_ang,
                ),
                args.symmetry_operation_index,
                data.k_points_frac,
                tolerance=args.mesh_tolerance,
                workers=args.workers,
            )
            rotation = operation.rotation
            translation = operation.translation
            channel_map = operation.channel_map
        if (
            channel_map.shape != (data.mode_count,)
            or np.unique(channel_map).size != data.mode_count
        ):
            raise ValueError(
                "channel_map must be one permutation of all physical modes"
            )
        rotation_absolute, rotation_relative, mapping_distance = _rotation_comparison(
            data.gamma_hwhm_mev,
            data.k_points_frac,
            rotation,
            channel_map,
            tolerance=args.mesh_tolerance,
            workers=args.workers,
        )
        rotation_limits = (
            _safe_limits(
                0.0,
                _finite_percentile(
                    rotation_absolute[selected_union] * args.linewidth_scale,
                    args.error_percentile,
                    1.0,
                ),
            ),
            _safe_limits(
                0.0,
                _finite_percentile(
                    100.0 * rotation_relative[selected_union],
                    args.error_percentile,
                    100.0,
                ),
            ),
        )
    else:
        rotation = None
        translation = None
        channel_map = None
        rotation_limits = ((_safe_limits(0.0, 1.0)), (_safe_limits(0.0, 100.0)))

    outputs: list[Path] = []
    selected_values = []
    for selected, mask in slices:
        geometry = _build_plane_geometry(
            data.k_points_frac[mask],
            data.lattice_ang,
            axes,
            tile=args.tile,
            bz_mode=args.bz_mode,
            periodic_repeats=args.periodic_repeats,
            periodic_view=args.periodic_view,
            periodic_padding=args.periodic_padding,
            decimals=args.k_round,
        )
        selected_values.append(float(selected))
        tag = _slice_tag(selected)
        slice_title = rf"$k_{{{normal + 1}}}={selected:.6g}$"
        for name in ("energy", "linewidth", "fwhm", "scattering-rate", "lifetime"):
            if name not in plots:
                continue
            values, title, label, cmap = quantities[name]
            outputs.extend(
                _plot_modes(
                    values[mask],
                    mode_labels=data.mode_labels,
                    geometry=geometry,
                    axes=axes,
                    title=f"{title}, {slice_title}",
                    label=label,
                    cmap=cmap,
                    limits=limits[name],
                    args=args,
                    output_dir=output_dir,
                    stem=f"{name.replace('-', '_')}.k{normal + 1}_{tag}",
                )
            )
        if "splitting" in plots:
            outputs.extend(
                _plot_splitting(
                    data.energy_mev[mask],
                    data.gamma_hwhm_mev[mask],
                    geometry=geometry,
                    axes=axes,
                    channels=split_channels,
                    chirality_ordered=data.magnon_chirality is not None,
                    limits=split_limits,
                    slice_title=slice_title,
                    args=args,
                    output_dir=output_dir,
                    stem=f"mode_splitting.k{normal + 1}_{tag}",
                )
            )
        if "rotation-error" in plots:
            assert rotation_absolute is not None and rotation_relative is not None
            outputs.extend(
                _plot_rotation_error(
                    rotation_absolute[mask],
                    rotation_relative[mask],
                    mode_labels=data.mode_labels,
                    geometry=geometry,
                    axes=axes,
                    limits=rotation_limits,
                    slice_title=slice_title,
                    args=args,
                    output_dir=output_dir,
                    stem=f"rotation_error.k{normal + 1}_{tag}",
                )
            )

    summary = {
        "input": str(data.source),
        "native_schema_version": data.metadata.get("schema_version"),
        "output_dir": str(output_dir),
        "kmesh": data.metadata.get("kmesh"),
        "kshift": data.metadata.get("kshift"),
        "mode_count": data.mode_count,
        "mode_labels": list(data.mode_labels),
        "magnon_mode_order": (
            "chirality_descending"
            if data.magnon_chirality is not None
            else data.metadata.get("magnon_mode_order", "stored")
        ),
        "magnon_chirality_range": (
            None
            if data.magnon_chirality is None
            else [
                float(np.min(data.magnon_chirality)),
                float(np.max(data.magnon_chirality)),
            ]
        ),
        "splitting_convention": (
            "E_chi_plus_minus_E_chi_minus"
            if data.magnon_chirality is not None and split_channels == (0, 1)
            else f"E_mode_{split_channels[0] + 1}_minus_E_mode_{split_channels[1] + 1}"
        ),
        "plane_axes": list(axes),
        "normal_axis": normal,
        "requested_slices": [float(value) for value in args.slices],
        "selected_slices": selected_values,
        "plots": sorted(plots),
        "linewidth_scale": float(args.linewidth_scale),
        "linewidth_unit": str(args.linewidth_unit),
        "color_limits": {name: list(value) for name, value in limits.items()},
        "split_limits": [list(value) for value in split_limits],
        "rotation_direct_frac": None if rotation is None else rotation.tolist(),
        "translation_frac": None if translation is None else translation.tolist(),
        "channel_map": None if channel_map is None else channel_map.tolist(),
        "symmetry_operation_index": None if operation is None else operation.index,
        "mesh_max_mapping_distance": None
        if mapping_distance is None
        else float(np.max(mapping_distance, initial=0.0)),
        "files": [str(path) for path in outputs],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        "[magph:lifetime-plot] "
        f"input={data.source}, modes={data.mode_count}, "
        f"slices={selected_values}, plots={sorted(plots)}"
    )
    for output in outputs:
        print(f"[magph:lifetime-plot] written={output}")
    print(f"[magph:lifetime-plot] summary={summary_path}")
    return outputs, summary_path, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot native magnon energy, linewidth, rate, and lifetime BZ maps"
    )
    parser.add_argument(
        "input", help="Native MnTe.lifetime.npz or retained lifetime NPZ"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--exchange-h5",
        help="Geometry source for old outputs that do not embed lattice metadata",
    )
    parser.add_argument("--slices", nargs="+", type=float, default=(0.0, 0.5))
    parser.add_argument("--all-slices", action="store_true")
    parser.add_argument(
        "--plots",
        nargs="+",
        choices=(
            "all",
            "energy",
            "linewidth",
            "fwhm",
            "scattering-rate",
            "lifetime",
            "splitting",
            "rotation-error",
        ),
        default=("all",),
    )
    parser.add_argument("--plane-axes", nargs=2, type=int, default=(0, 1))
    parser.add_argument("--physical-mode-count", "--physical-channel-count", type=int)
    parser.add_argument(
        "--split-modes", "--split-channels", nargs=2, type=int, default=(0, 1)
    )
    parser.add_argument("--rotation", nargs=9, type=int)
    parser.add_argument("--channel-map", nargs="+", type=int)
    parser.add_argument("--symmetry-operation-index", type=int)
    parser.add_argument("--symprec", type=float, default=1.0e-5)
    parser.add_argument("--atom-tolerance-ang", type=float, default=1.0e-4)
    parser.add_argument("--mesh-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--workers", type=int, default=-1)
    parser.add_argument("--k-round", type=int, default=8)
    parser.add_argument(
        "--formats", nargs="+", choices=("png", "pdf", "svg"), default=("png", "pdf")
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--panel-width", type=float, default=4.4)
    parser.add_argument("--panel-height", type=float, default=4.0)
    parser.add_argument("--tile", type=int, default=1)
    parser.add_argument("--bz-mode", choices=("clip", "periodic"), default="clip")
    parser.add_argument("--periodic-repeats", type=int, default=1)
    parser.add_argument(
        "--periodic-view", choices=("central", "neighbors"), default="central"
    )
    parser.add_argument("--periodic-padding", type=float, default=0.06)
    parser.add_argument("--margin", type=float, default=0.035)
    parser.add_argument("--linewidth-scale", type=float, default=1.0)
    parser.add_argument("--linewidth-unit", default="meV")
    parser.add_argument("--scattering-rate-scale", type=float, default=1.0)
    parser.add_argument("--scattering-rate-unit", default=r"ps$^{-1}$")
    parser.add_argument("--vmin-percentile", type=float, default=1.0)
    parser.add_argument("--vmax-percentile", type=float, default=99.0)
    parser.add_argument("--error-percentile", type=float, default=99.0)
    parser.add_argument("--energy-cmap", default="viridis")
    parser.add_argument("--linewidth-cmap", default="magma")
    parser.add_argument("--lifetime-cmap", default="viridis")
    parser.add_argument("--scattering-rate-cmap", default="magma")
    parser.add_argument("--error-cmap", default="cividis")
    parser.add_argument("--diverging-cmap", default="RdBu_r")
    parser.add_argument("--edgecolor", default="none")
    parser.add_argument("--linewidth", type=float, default=0.0)
    parser.add_argument("--bz-color", default="0.25")
    parser.add_argument("--bz-lw", type=float, default=1.1)
    parser.add_argument("--neighbor-bz-lw", type=float, default=0.65)
    parser.add_argument("--neighbor-bz-alpha", type=float, default=0.55)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    plot_lifetime(build_parser().parse_args(argv))
    return 0


__all__ = ["LifetimePlotData", "load_lifetime_plot_data", "plot_lifetime"]


if __name__ == "__main__":
    raise SystemExit(main())
