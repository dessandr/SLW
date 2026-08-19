"""Fukui Berry curvature for the effective hybrid magnon-phonon Hamiltonian."""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import Voronoi

from . import lswt
from .adapter import build_phonon_cache_from_epr, load_phonon_cache
from .hybrid import (
    MEV_PER_THZ,
    _load_atom_frac,
    _load_structure,
    build_hybrid_from_static_tensor,
)
from .tensor_adapter import load_exchange_tensor_realspace_h5
from .vertex import build_site_basis_vertex, project_site_vertex_to_phonon_modes


def plane_mesh(plane: str, mesh, fixed: float = 0.0):
    n1, n2 = [int(x) for x in mesh]
    u = np.arange(n1, dtype=np.float64) / float(n1)
    v = np.arange(n2, dtype=np.float64) / float(n2)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    q = np.zeros((n1, n2, 3), dtype=np.float64)
    p = str(plane).strip().lower()
    if p in {"kz", "xy"}:
        q[..., 0] = uu
        q[..., 1] = vv
        q[..., 2] = float(fixed)
        axes = (0, 1)
    elif p in {"ky", "xz"}:
        q[..., 0] = uu
        q[..., 1] = float(fixed)
        q[..., 2] = vv
        axes = (0, 2)
    elif p in {"kx", "yz"}:
        q[..., 0] = float(fixed)
        q[..., 1] = uu
        q[..., 2] = vv
        axes = (1, 2)
    else:
        raise ValueError(f"Unknown plane={plane!r}; use kz/xy, ky/xz, or kx/yz")
    return q.reshape(-1, 3), q, axes


def _unit_link(a, b):
    ov = np.einsum("...i,...i->...", np.conjugate(a), b, optimize=True)
    abs_ov = np.abs(ov)
    return np.divide(ov, abs_ov, out=np.ones_like(ov, dtype=np.complex128), where=abs_ov > 1.0e-14)


def fukui_band_flux(eigenvectors_grid: np.ndarray, band: int) -> np.ndarray:
    """Return plaquette Berry flux in radians for one band on a periodic mesh."""
    vec = np.asarray(eigenvectors_grid, dtype=np.complex128)[..., int(band)]
    ux = _unit_link(vec, np.roll(vec, -1, axis=0))
    uy = _unit_link(vec, np.roll(vec, -1, axis=1))
    loop = ux * np.roll(uy, -1, axis=0) * np.conjugate(np.roll(ux, -1, axis=1)) * np.conjugate(uy)
    return np.angle(loop)


def fukui_all_band_flux(eigenvectors_grid: np.ndarray) -> np.ndarray:
    """Return plaquette Berry flux for all bands on a periodic mesh."""
    vec = np.asarray(eigenvectors_grid, dtype=np.complex128)
    ovx = np.einsum("...in,...in->...n", np.conjugate(vec), np.roll(vec, -1, axis=0), optimize=True)
    ovy = np.einsum("...in,...in->...n", np.conjugate(vec), np.roll(vec, -1, axis=1), optimize=True)
    absx = np.abs(ovx)
    absy = np.abs(ovy)
    ux = np.divide(ovx, absx, out=np.ones_like(ovx, dtype=np.complex128), where=absx > 1.0e-14)
    uy = np.divide(ovy, absy, out=np.ones_like(ovy, dtype=np.complex128), where=absy > 1.0e-14)
    loop = ux * np.roll(uy, -1, axis=0) * np.conjugate(np.roll(ux, -1, axis=1)) * np.conjugate(uy)
    return np.angle(loop)


def kubo_berry_curvature_grid(
    hamiltonian_grid: np.ndarray,
    eigenvalues_grid: np.ndarray,
    eigenvectors_grid: np.ndarray,
    lattice,
    axes,
    mesh,
    *,
    gap_floor: float = 1.0e-8,
) -> np.ndarray:
    """Return Kubo Berry curvature normal to the selected reciprocal plane.

    The finite differences follow the two reciprocal-lattice mesh directions.
    The final division by sin(theta) converts the two directional derivatives
    to the normal Berry-curvature component in Cartesian reciprocal units.
    """
    H = np.asarray(hamiltonian_grid, dtype=np.complex128)
    E = np.asarray(eigenvalues_grid, dtype=np.float64)
    U = np.asarray(eigenvectors_grid, dtype=np.complex128)
    nx, ny = [int(x) for x in mesh]
    recip = 2.0 * np.pi * np.linalg.inv(np.asarray(lattice, dtype=np.float64)).T
    b1 = np.asarray(recip[int(axes[0])], dtype=np.float64)
    b2 = np.asarray(recip[int(axes[1])], dtype=np.float64)
    step1 = float(np.linalg.norm(b1)) / float(nx)
    step2 = float(np.linalg.norm(b2)) / float(ny)
    sin_theta = float(np.linalg.norm(np.cross(b1, b2)) / max(np.linalg.norm(b1) * np.linalg.norm(b2), 1.0e-30))
    if step1 < 1.0e-30 or step2 < 1.0e-30 or sin_theta < 1.0e-12:
        raise ValueError("Invalid reciprocal mesh geometry for Kubo Berry curvature")

    dH1 = (np.roll(H, -1, axis=0) - np.roll(H, 1, axis=0)) / (2.0 * step1)
    dH2 = (np.roll(H, -1, axis=1) - np.roll(H, 1, axis=1)) / (2.0 * step2)
    v1 = np.einsum("...an,...ab,...bm->...nm", U.conj(), dH1, U, optimize=True)
    v2 = np.einsum("...an,...ab,...bm->...nm", U.conj(), dH2, U, optimize=True)
    dE = E[..., :, None] - E[..., None, :]
    denom = dE * dE
    valid = np.abs(dE) > float(gap_floor)
    term = np.divide(
        v1 * np.swapaxes(v2, -1, -2),
        denom,
        out=np.zeros_like(v1, dtype=np.complex128),
        where=valid,
    )
    return -2.0 * np.imag(np.sum(term, axis=-1)) / sin_theta


def _parse_band_spec(spec) -> list[int]:
    if isinstance(spec, (int, np.integer)):
        return [int(spec)]
    bands: list[int] = []
    for part in str(spec).replace(",", " ").split():
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo = int(lo_s)
            hi = int(hi_s)
            step = 1 if hi >= lo else -1
            bands.extend(range(lo, hi + step, step))
        else:
            bands.append(int(part))
    out = []
    seen = set()
    for b in bands:
        if b not in seen:
            out.append(b)
            seen.add(b)
    if not out:
        raise ValueError(f"Empty band specification: {spec!r}")
    return out


def _band_suffix_path(path: str, band: int, multiple: bool) -> str:
    if not multiple:
        return path
    root, ext = os.path.splitext(path)
    return f"{root}_band{int(band)}{ext or '.npz'}"


def _band_sum_label(bands) -> str:
    b = [int(x) for x in bands]
    if not b:
        raise ValueError("Empty band sum")
    if b == list(range(b[0], b[-1] + 1)):
        return f"{b[0]}-{b[-1]}" if len(b) > 1 else str(b[0])
    return ",".join(str(x) for x in b)


def _match_overlap(prev_ordered_vecs: np.ndarray, curr_raw_vecs: np.ndarray):
    overlap = np.abs(prev_ordered_vecs.conj().T @ curr_raw_vecs)
    row, col = linear_sum_assignment(-overlap)
    order = np.empty(overlap.shape[0], dtype=np.int32)
    order[row] = col
    matched = overlap[row, col]
    return order, matched


def track_band_order_by_overlap(eigenvectors_grid: np.ndarray):
    """Track band order on a 2D mesh by nearest-neighbor eigenvector overlap."""
    U = np.asarray(eigenvectors_grid, dtype=np.complex128)
    if U.ndim != 4 or U.shape[-2] != U.shape[-1]:
        raise ValueError(f"Expected eigenvectors_grid shape (nx,ny,nband,nband), got {U.shape}")
    nx, ny, ndim, nband = U.shape
    band_map = np.zeros((nx, ny, nband), dtype=np.int32)
    band_map[0, 0] = np.arange(nband, dtype=np.int32)
    overlaps = []

    def assign_from(prev_i, prev_j, i, j):
        prev_order = band_map[prev_i, prev_j]
        prev_vecs = U[prev_i, prev_j, :, prev_order]
        curr_vecs = U[i, j]
        order, matched = _match_overlap(prev_vecs, curr_vecs)
        band_map[i, j] = order
        overlaps.append(matched)

    prev_i, prev_j = 0, 0
    for i in range(nx):
        if i == 0:
            js = range(1, ny)
        else:
            start_j = ny - 1 if i % 2 == 1 else 0
            assign_from(prev_i, prev_j, i, start_j)
            prev_i, prev_j = i, start_j
            js = range(start_j - 1, -1, -1) if i % 2 == 1 else range(start_j + 1, ny)
        for j in js:
            assign_from(prev_i, prev_j, i, j)
            prev_i, prev_j = i, j

    if overlaps:
        ov = np.concatenate([np.asarray(x, dtype=np.float64).reshape(-1) for x in overlaps])
        stats = {
            "track_overlap_min": np.asarray(float(np.min(ov)), dtype=np.float64),
            "track_overlap_mean": np.asarray(float(np.mean(ov)), dtype=np.float64),
            "track_overlap_p05": np.asarray(float(np.percentile(ov, 5.0)), dtype=np.float64),
        }
    else:
        stats = {
            "track_overlap_min": np.asarray(1.0, dtype=np.float64),
            "track_overlap_mean": np.asarray(1.0, dtype=np.float64),
            "track_overlap_p05": np.asarray(1.0, dtype=np.float64),
        }
    return band_map, stats


def _take_tracked_bands(arr: np.ndarray, band_map: np.ndarray):
    a = np.asarray(arr)
    idx = np.asarray(band_map, dtype=np.int32)
    if a.ndim == 3:
        return np.take_along_axis(a, idx, axis=2)
    if a.ndim == 4:
        return np.take_along_axis(a, idx[:, :, None, :], axis=3)
    raise ValueError(f"Unsupported tracked-band array shape: {a.shape}")


def reciprocal_plaquette_area(lattice, axes, mesh) -> float:
    recip = 2.0 * np.pi * np.linalg.inv(np.asarray(lattice, dtype=np.float64)).T
    b1 = recip[int(axes[0])] / int(mesh[0])
    b2 = recip[int(axes[1])] / int(mesh[1])
    return float(np.linalg.norm(np.cross(b1, b2)))


def reciprocal_plane_basis_2d(lattice, axes):
    recip = 2.0 * np.pi * np.linalg.inv(np.asarray(lattice, dtype=np.float64)).T
    b1 = np.asarray(recip[int(axes[0])], dtype=np.float64)
    b2 = np.asarray(recip[int(axes[1])], dtype=np.float64)
    e1 = b1 / max(float(np.linalg.norm(b1)), 1.0e-30)
    b2_perp = b2 - float(np.dot(b2, e1)) * e1
    if np.linalg.norm(b2_perp) < 1.0e-12:
        raise ValueError("Selected reciprocal plane vectors are nearly collinear")
    e2 = b2_perp / float(np.linalg.norm(b2_perp))
    b1_2d = np.array([np.dot(b1, e1), np.dot(b1, e2)], dtype=np.float64)
    b2_2d = np.array([np.dot(b2, e1), np.dot(b2, e2)], dtype=np.float64)
    return b1_2d, b2_2d


def _plane_plot_points(qgrid_frac, axes, b1_2d, b2_2d, *, center_plaquettes: bool):
    q = np.asarray(qgrid_frac, dtype=np.float64).copy()
    nx, ny = q.shape[:2]
    if center_plaquettes:
        q[..., int(axes[0])] += 0.5 / float(nx)
        q[..., int(axes[1])] += 0.5 / float(ny)
    f1 = ((q[..., int(axes[0])] + 0.5) % 1.0) - 0.5
    f2 = ((q[..., int(axes[1])] + 0.5) % 1.0) - 0.5
    pts = f1[..., None] * b1_2d[None, None, :] + f2[..., None] * b2_2d[None, None, :]
    return pts.reshape(-1, 2)


def _periodic_voronoi_polygons(pts, vals, b1, b2, tile):
    pts = np.asarray(pts, dtype=np.float64)
    vals = np.asarray(vals, dtype=np.float64)
    if vals.ndim == 0 or vals.shape[0] != pts.shape[0]:
        raise ValueError(f"Voronoi values must start with npoints={pts.shape[0]}, got {vals.shape}")
    tile_range = range(-int(tile), int(tile) + 1)
    tiled = [pts + ia * b1 + ib * b2 for ia in tile_range for ib in tile_range]
    all_pts = np.vstack(tiled)
    vor = Voronoi(all_pts)

    n = pts.shape[0]
    polys = []
    colors = []
    for itile in range(len(tiled)):
        offset = itile * n
        for ip in range(n):
            region = vor.regions[vor.point_region[offset + ip]]
            if not region or -1 in region:
                continue
            polys.append(vor.vertices[region])
            colors.append(vals[ip])
    return polys, np.asarray(colors, dtype=np.float64)


def _first_bz_polygon_2d(b1, b2):
    lattice_pts = []
    origin_index = None
    for ia in range(-2, 3):
        for ib in range(-2, 3):
            if ia == 0 and ib == 0:
                origin_index = len(lattice_pts)
            lattice_pts.append(ia * b1 + ib * b2)
    lattice_pts = np.asarray(lattice_pts, dtype=np.float64)
    vor = Voronoi(lattice_pts)
    region = vor.regions[vor.point_region[origin_index]]
    if not region or -1 in region:
        return None
    poly = vor.vertices[region]
    center = np.mean(poly, axis=0)
    angle = np.arctan2(poly[:, 1] - center[1], poly[:, 0] - center[0])
    return poly[np.argsort(angle)]


def _finite_percentile(values, pct):
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan
    return float(np.percentile(v, float(pct)))


def _select_plot_quantity(result: dict, quantity: str):
    q = str(quantity).strip().lower()
    if q in {"flux", "berry_flux"}:
        return result["berry_flux"], "Berry flux (rad)", True, True
    if q in {"curvature", "berry_curvature"}:
        return result["berry_curvature"], "Berry curvature", True, True
    if q in {"kubo", "kubo_curvature", "berry_curvature_kubo", "omega_kubo"}:
        if "berry_curvature_kubo" not in result:
            raise ValueError("Kubo curvature was not computed. Use --berry_method kubo or --berry_method both.")
        return result["berry_curvature_kubo"], "Kubo Berry curvature", False, True
    if q in {"kubo_flux", "berry_flux_kubo"}:
        if "berry_flux_kubo" not in result:
            raise ValueError("Kubo curvature was not computed. Use --berry_method kubo or --berry_method both.")
        return result["berry_flux_kubo"], "Kubo Berry flux estimate (rad)", False, True
    if q in {"energy", "energy_thz"}:
        return result["energy_thz"], "Energy (THz)", False, False
    if q in {"weight", "magnon_weight"}:
        return result["magnon_weight"], "Magnon weight", False, False
    if q in {"chirality", "hybrid_chirality"}:
        return result["hybrid_chirality"], "Hybrid chirality", False, True
    raise ValueError(f"Unknown plot quantity {quantity!r}")


def compute_hybrid_berry(
    *,
    tensor_h5: str,
    j_tensor_h5: str,
    epr_phonon: str | None = None,
    phonon_cache: str | None = None,
    component: str = "dmi",
    static_component: str = "iso",
    S: float = 2.5,
    spin_direction=(0.0, 1.0, 0.0),
    spin_pattern="auto",
    phase_convention: str = "basis",
    plane: str = "kz",
    fixed: float = 0.0,
    mesh=(31, 31),
    band: int = 0,
    threshold: float = 0.0,
    structure: str | None = None,
    coupling_scale: float = 1.0,
    berry_method: str = "fukui",
    kubo_gap_floor: float = 1.0e-8,
    track_bands: str = "energy",
):
    results = compute_hybrid_berry_multi(
        tensor_h5=tensor_h5,
        j_tensor_h5=j_tensor_h5,
        epr_phonon=epr_phonon,
        phonon_cache=phonon_cache,
        component=component,
        static_component=static_component,
        S=S,
        spin_direction=spin_direction,
        spin_pattern=spin_pattern,
        phase_convention=phase_convention,
        plane=plane,
        fixed=fixed,
        mesh=mesh,
        bands=[int(band)],
        threshold=threshold,
        structure=structure,
        coupling_scale=coupling_scale,
        berry_method=berry_method,
        kubo_gap_floor=kubo_gap_floor,
        track_bands=track_bands,
    )
    return results[int(band)]


def compute_hybrid_berry_multi(
    *,
    tensor_h5: str,
    j_tensor_h5: str,
    epr_phonon: str | None = None,
    phonon_cache: str | None = None,
    component: str = "dmi",
    static_component: str = "iso",
    S: float = 2.5,
    spin_direction=(0.0, 1.0, 0.0),
    spin_pattern="auto",
    phase_convention: str = "basis",
    plane: str = "kz",
    fixed: float = 0.0,
    mesh=(31, 31),
    bands=(0,),
    threshold: float = 0.0,
    structure: str | None = None,
    coupling_scale: float = 1.0,
    berry_method: str = "fukui",
    kubo_gap_floor: float = 1.0e-8,
    track_bands: str = "energy",
):
    lattice, _, _ = _load_structure(structure, fallback_h5=j_tensor_h5)
    qpts, qgrid, axes = plane_mesh(plane, mesh, fixed=fixed)
    components = "dmi,iso,aniso" if component == "full" else component
    payload = load_exchange_tensor_realspace_h5(tensor_h5, components=components)
    nmag = int(max(int(payload.bond_i.max(initial=0)), int(payload.bond_j.max(initial=0))) + 1)
    spin_info = lswt.build_spin_frame_info(
        nmag=nmag,
        spin_direction=spin_direction,
        spin_pattern=spin_pattern,
    )
    atom_frac = None
    if phase_convention == "basis":
        atom_frac = _load_atom_frac(structure, fallback_h5=j_tensor_h5)
    site = build_site_basis_vertex(
        payload,
        spin_info,
        qpts_frac=qpts,
        atom_frac=atom_frac,
        component=component,
        S=S,
        phase_convention=phase_convention,
        threshold=threshold,
    )
    if phonon_cache:
        ph_cache = load_phonon_cache(phonon_cache)
    elif epr_phonon:
        ph_cache = build_phonon_cache_from_epr(epr_phonon, qpts_frac=qpts)
    else:
        raise ValueError("Use epr_phonon or phonon_cache")
    mode = project_site_vertex_to_phonon_modes(site, ph_cache)
    hybrid = build_hybrid_from_static_tensor(
        j_tensor_h5=j_tensor_h5,
        structure=structure,
        mode_vertex=mode,
        S=S,
        spin_info=spin_info,
        kpts_frac=np.zeros((1, 3), dtype=np.float64),
        static_component=static_component,
        coupling_scale=coupling_scale,
    )
    nx, ny = [int(x) for x in mesh]
    nband = hybrid.eigenvalues.shape[-1]
    bands = [int(b) for b in bands]
    bad = [b for b in bands if b < 0 or b >= nband]
    if bad:
        raise ValueError(f"band(s)={bad} out of range for nband={nband}")
    evec_grid_raw = hybrid.eigenvectors[0].reshape(nx, ny, nband, nband)
    energy_raw = hybrid.eigenvalues[0].reshape(nx, ny, nband)
    magnon_weight_raw = hybrid.magnon_weight[0].reshape(nx, ny, nband)
    chirality_raw = hybrid.hybrid_chirality[0].reshape(nx, ny, nband)
    tracking = str(track_bands).strip().lower()
    track_stats = {}
    if tracking in {"energy", "none", "off"}:
        band_map = np.broadcast_to(np.arange(nband, dtype=np.int32), (nx, ny, nband)).copy()
        evec_grid = evec_grid_raw
        energy_all = energy_raw
        magnon_weight_all = magnon_weight_raw
        chirality_all = chirality_raw
        tracking = "energy"
    elif tracking in {"overlap", "ovlp"}:
        band_map, track_stats = track_band_order_by_overlap(evec_grid_raw)
        evec_grid = _take_tracked_bands(evec_grid_raw, band_map)
        energy_all = _take_tracked_bands(energy_raw, band_map)
        magnon_weight_all = _take_tracked_bands(magnon_weight_raw, band_map)
        chirality_all = _take_tracked_bands(chirality_raw, band_map)
        tracking = "overlap"
    else:
        raise ValueError("track_bands must be energy or overlap")
    flux_all = fukui_all_band_flux(evec_grid)
    area = reciprocal_plaquette_area(lattice, axes, mesh)
    curvature_all = flux_all / max(area, 1.0e-30)
    chern_all = np.sum(flux_all, axis=(0, 1)) / (2.0 * np.pi)
    gaps = np.abs(energy_raw.reshape(nx * ny, nband, 1) - energy_raw.reshape(nx * ny, 1, nband))
    diag = np.arange(nband)
    gaps[:, diag, diag] = np.inf
    band_min_gap_all = np.nanmin(gaps, axis=(0, 2))
    method = str(berry_method).strip().lower()
    kubo_all = None
    chern_kubo_all = None
    if method in {"kubo", "both"}:
        H_grid = hybrid.hamiltonian[0].reshape(nx, ny, nband, nband)
        E_grid = hybrid.eigenvalues[0].reshape(nx, ny, nband)
        kubo_all = kubo_berry_curvature_grid(
            H_grid,
            _take_tracked_bands(E_grid, band_map),
            evec_grid,
            lattice,
            axes,
            mesh,
            gap_floor=kubo_gap_floor,
        )
        chern_kubo_all = np.sum(kubo_all, axis=(0, 1)) * area / (2.0 * np.pi)
    elif method != "fukui":
        raise ValueError("berry_method must be fukui, kubo, or both")
    common = {
        "hybrid": hybrid,
        "qpts_frac": qpts,
        "qgrid_frac": qgrid,
        "lattice_ang": np.asarray(lattice, dtype=np.float64),
        "plane_axes": np.asarray(axes, dtype=np.int32),
        "mesh": np.asarray(mesh, dtype=np.int32),
        "plane": np.asarray(str(plane), dtype=object),
        "fixed": np.asarray(float(fixed), dtype=np.float64),
        "plaquette_area": np.asarray(area, dtype=np.float64),
        "track_bands": np.asarray(tracking, dtype=object),
        "band_map": band_map,
        **track_stats,
    }
    results = {}
    for b in bands:
        result = {
            **common,
            "band": np.asarray(int(b), dtype=np.int32),
            "berry_flux": flux_all[..., b],
            "berry_curvature": curvature_all[..., b],
            "chern": np.asarray(float(chern_all[b]), dtype=np.float64),
            "energy_mev": energy_all[..., b],
            "energy_thz": energy_all[..., b] / MEV_PER_THZ,
            "band_min_gap_mev": np.asarray(float(band_min_gap_all[b]), dtype=np.float64),
            "raw_band_index": band_map[..., b],
            "magnon_weight": magnon_weight_all[..., b],
            "hybrid_chirality": chirality_all[..., b],
        }
        if kubo_all is not None:
            result["berry_curvature_kubo"] = kubo_all[..., b]
            result["berry_flux_kubo"] = kubo_all[..., b] * area
            result["chern_kubo"] = np.asarray(float(chern_kubo_all[b]), dtype=np.float64)
        results[int(b)] = result
    return results


def sum_band_results(results: dict[int, dict], bands):
    bands = [int(b) for b in bands]
    if not bands:
        raise ValueError("No bands provided for sum")
    missing = [b for b in bands if b not in results]
    if missing:
        raise ValueError(f"Band-sum requested missing bands: {missing}")
    first = results[bands[0]]
    label = _band_sum_label(bands)
    stacked_energy = np.stack([results[b]["energy_mev"] for b in bands], axis=-1)
    stacked_weight = np.stack([results[b]["magnon_weight"] for b in bands], axis=-1)
    stacked_chi = np.stack([results[b]["hybrid_chirality"] for b in bands], axis=-1)
    weight_sum = np.sum(stacked_weight, axis=-1)
    chi_weighted = np.sum(stacked_weight * stacked_chi, axis=-1)
    raw_stack = np.stack([results[b]["raw_band_index"] for b in bands], axis=-1)
    result = {
        **first,
        "band": np.asarray(-1, dtype=np.int32),
        "band_sum": np.asarray(bands, dtype=np.int32),
        "band_label": np.asarray(label, dtype=object),
        "raw_band_index": raw_stack,
        "berry_flux": np.sum([results[b]["berry_flux"] for b in bands], axis=0),
        "berry_curvature": np.sum([results[b]["berry_curvature"] for b in bands], axis=0),
        "chern": np.asarray(float(np.sum([float(results[b]["chern"]) for b in bands])), dtype=np.float64),
        "energy_mev": np.mean(stacked_energy, axis=-1),
        "energy_thz": np.mean(stacked_energy, axis=-1) / MEV_PER_THZ,
        "energy_min_mev": np.min(stacked_energy, axis=-1),
        "energy_max_mev": np.max(stacked_energy, axis=-1),
        "magnon_weight": weight_sum,
        "hybrid_chirality": np.divide(
            chi_weighted,
            weight_sum,
            out=np.zeros_like(chi_weighted, dtype=np.float64),
            where=np.abs(weight_sum) > 1.0e-14,
        ),
    }
    evals = np.asarray(first["hybrid"].eigenvalues[0], dtype=np.float64)
    nb = evals.shape[-1]
    in_set = np.asarray(sorted(set(bands)), dtype=np.int32)
    out_set = np.asarray([i for i in range(nb) if i not in set(in_set.tolist())], dtype=np.int32)
    if out_set.size:
        boundary_gap = np.min(np.abs(evals[:, in_set, None] - evals[:, None, out_set]))
    else:
        boundary_gap = np.inf
    result["band_min_gap_mev"] = np.asarray(float(boundary_gap), dtype=np.float64)
    if "berry_curvature_kubo" in first:
        result["berry_curvature_kubo"] = np.sum([results[b]["berry_curvature_kubo"] for b in bands], axis=0)
        result["berry_flux_kubo"] = np.sum([results[b]["berry_flux_kubo"] for b in bands], axis=0)
        result["chern_kubo"] = np.asarray(float(np.sum([float(results[b]["chern_kubo"]) for b in bands])), dtype=np.float64)
    return result


def save_berry_npz(path: str, result: dict):
    hybrid = result["hybrid"]
    arrays = dict(
        qpts_frac=result["qpts_frac"],
        qgrid_frac=result["qgrid_frac"],
        lattice_ang=result["lattice_ang"],
        plane_axes=result["plane_axes"],
        mesh=result["mesh"],
        plane=result["plane"],
        fixed=result["fixed"],
        band=result["band"],
        track_bands=result["track_bands"],
        raw_band_index=result["raw_band_index"],
        band_map=result["band_map"],
        berry_flux=result["berry_flux"],
        berry_curvature=result["berry_curvature"],
        plaquette_area=result["plaquette_area"],
        chern=result["chern"],
        energy_mev=result["energy_mev"],
        energy_thz=result["energy_thz"],
        band_min_gap_mev=result["band_min_gap_mev"],
        magnon_weight=result["magnon_weight"],
        hybrid_chirality=result["hybrid_chirality"],
        hybrid_eigenvalues=hybrid.eigenvalues,
        hybrid_eigenvectors=hybrid.eigenvectors,
    )
    for key in ("band_sum", "band_label", "energy_min_mev", "energy_max_mev"):
        if key in result:
            arrays[key] = result[key]
    for key in ("berry_curvature_kubo", "berry_flux_kubo", "chern_kubo"):
        if key in result:
            arrays[key] = result[key]
    for key in ("track_overlap_min", "track_overlap_mean", "track_overlap_p05"):
        if key in result:
            arrays[key] = result[key]
    np.savez_compressed(path, **arrays)


def plot_berry(path: str, result: dict, *, quantity: str = "flux"):
    q = str(quantity).strip().lower()
    data, label, _, diverging = _select_plot_quantity(result, quantity)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    vmax = float(np.nanpercentile(np.abs(data), 99.0)) if data.size else 1.0
    if diverging:
        im = ax.imshow(data.T, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
    else:
        im = ax.imshow(data.T, origin="lower", cmap="viridis", interpolation="nearest")
    ax.set_xlabel("mesh axis 1")
    ax.set_ylabel("mesh axis 2")
    band_title = f"bands {str(result['band_label'])}" if "band_label" in result else f"band {int(result['band'])}"
    ax.set_title(f"{band_title}, C={float(result['chern']):.6g}")
    fig.colorbar(im, ax=ax, label=label)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    print(f"[hybrid-berry] wrote plot {path}")


def plot_berry_bz(
    path: str,
    result: dict,
    *,
    quantity: str = "flux",
    tile: int = 2,
    cmap: str | None = None,
    edgecolor: str = "none",
    linewidth: float = 0.0,
    vmin: float | None = None,
    vmax: float | None = None,
    vmin_percentile: float = 1.0,
    vmax_percentile: float = 99.0,
    fig_width: float = 5.4,
    fig_height: float = 5.0,
):
    data, label, center_plaquettes, diverging = _select_plot_quantity(result, quantity)
    lattice = np.asarray(result["lattice_ang"], dtype=np.float64)
    axes = tuple(int(x) for x in np.asarray(result["plane_axes"], dtype=np.int32))
    b1, b2 = reciprocal_plane_basis_2d(lattice, axes)
    pts = _plane_plot_points(result["qgrid_frac"], axes, b1, b2, center_plaquettes=center_plaquettes)
    polys, colors = _periodic_voronoi_polygons(pts, np.asarray(data).reshape(-1), b1, b2, tile)
    if len(polys) == 0:
        raise RuntimeError("Voronoi construction produced no finite cells")

    if vmin is None:
        vmin = -_finite_percentile(np.abs(colors), vmax_percentile) if diverging else _finite_percentile(colors, vmin_percentile)
    if vmax is None:
        vmax = _finite_percentile(np.abs(colors), vmax_percentile) if diverging else _finite_percentile(colors, vmax_percentile)
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin, vmax = (-1.0, 1.0) if diverging else (0.0, 1.0)
    if abs(float(vmax) - float(vmin)) < 1.0e-14:
        pad = max(abs(float(vmax)), 1.0) * 1.0e-6
        vmin = float(vmin) - pad
        vmax = float(vmax) + pad

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    pc = PolyCollection(
        polys,
        array=colors,
        cmap=cmap or ("RdBu_r" if diverging else "viridis"),
        edgecolor=edgecolor,
        linewidth=linewidth,
    )
    pc.set_clim(float(vmin), float(vmax))
    ax.add_collection(pc)

    bz = _first_bz_polygon_2d(b1, b2)
    if bz is not None:
        clip_patch = plt.Polygon(bz, closed=True, facecolor="none", edgecolor="none")
        ax.add_patch(clip_patch)
        pc.set_clip_path(clip_patch)
        closed = np.vstack([bz, bz[0]])
        ax.plot(closed[:, 0], closed[:, 1], color="0.45", lw=1.2, alpha=0.95)
        xmin, ymin = np.min(bz, axis=0)
        xmax, ymax = np.max(bz, axis=0)
    else:
        xmin, ymin = np.min(pts, axis=0)
        xmax, ymax = np.max(pts, axis=0)

    dx = max(float(xmax - xmin), 1.0e-12)
    dy = max(float(ymax - ymin), 1.0e-12)
    ax.set_xlim(float(xmin) - 0.04 * dx, float(xmax) + 0.04 * dx)
    ax.set_ylim(float(ymin) - 0.04 * dy, float(ymax) + 0.04 * dy)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$q_1$ ($\AA^{-1}$)")
    ax.set_ylabel(r"$q_2$ ($\AA^{-1}$)")
    band_title = f"bands {str(result['band_label'])}" if "band_label" in result else f"band {int(result['band'])}"
    ax.set_title(f"{band_title}, C={float(result['chern']):.6g}")
    fig.colorbar(pc, ax=ax, label=label)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    print(f"[hybrid-berry] wrote BZ Voronoi plot {path}")


def _print_summary(result: dict, out_path: str):
    print("[hybrid-berry] summary")
    print(f"  mesh: {tuple(int(x) for x in result['mesh'])}")
    if "band_label" in result:
        print(f"  band_sum: {str(result['band_label'])}")
    else:
        print(f"  band: {int(result['band'])}")
    print(f"  track_bands: {str(result['track_bands'])}")
    if "track_overlap_min" in result:
        print(
            "  track_overlap[min,p05,mean]: "
            f"({float(result['track_overlap_min'])}, {float(result['track_overlap_p05'])}, {float(result['track_overlap_mean'])})"
        )
    print(f"  chern: {float(result['chern'])}")
    print(f"  band_min_gap_meV: {float(result['band_min_gap_mev'])}")
    print(f"  flux_minmax: ({float(np.nanmin(result['berry_flux']))}, {float(np.nanmax(result['berry_flux']))})")
    print(f"  curvature_minmax: ({float(np.nanmin(result['berry_curvature']))}, {float(np.nanmax(result['berry_curvature']))})")
    if "berry_curvature_kubo" in result:
        print(f"  chern_kubo: {float(result['chern_kubo'])}")
        print(f"  kubo_curvature_minmax: ({float(np.nanmin(result['berry_curvature_kubo']))}, {float(np.nanmax(result['berry_curvature_kubo']))})")
    print(f"  energy_THz_minmax: ({float(np.nanmin(result['energy_thz']))}, {float(np.nanmax(result['energy_thz']))})")
    print(f"[hybrid-berry] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Fukui Berry curvature for effective hybrid magnon-phonon Hamiltonian")
    ap.add_argument("--dJ_tensor_h5", default=None, help="dynamic dJ/du tensor HDF5 from compute_dJ_epr_tensor")
    ap.add_argument("--J_tensor_h5", default=None, help="static exchange tensor HDF5 from compute_J_epr_tensor")
    ap.add_argument("--tensor_h5", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--j_tensor_h5", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--epr_phonon", default=None)
    ap.add_argument("--phonon_cache", default=None)
    ap.add_argument("--component", default="dmi", choices=["dmi", "iso", "aniso", "full"])
    ap.add_argument("--static_component", default="iso",
                    help="Static exchange tensor for bare magnons: iso, aniso, dmi, full, or combinations like iso+dmi")
    ap.add_argument("--S", type=float, default=2.5)
    ap.add_argument("--spin_direction", type=float, nargs=3, default=[0.0, 1.0, 0.0])
    ap.add_argument("--spin_pattern", default="auto")
    ap.add_argument("--phase_convention", choices=["cell", "basis"], default="basis")
    ap.add_argument("--plane", default="kz", choices=["kz", "xy", "ky", "xz", "kx", "yz"])
    ap.add_argument("--fixed", type=float, default=0.0)
    ap.add_argument("--mesh", type=int, nargs=2, default=[31, 31])
    ap.add_argument("--band", default="0", help="Band index, range, or list. Examples: 0, 0-5, '0 2 4', '0,2,4'")
    ap.add_argument("--sum_below", action=argparse.BooleanOptionalAction, default=False,
                    help="Sum Berry curvature from band 0 through each --band value, WannierTools occupied-subspace style")
    ap.add_argument("--band_sum", default=None,
                    help="Explicit band subspace to sum, e.g. 0-12 or '0 1 2'. Produces one summed output")
    ap.add_argument("--threshold", type=float, default=0.0)
    ap.add_argument("--structure", default=None)
    ap.add_argument("--coupling_scale", type=float, default=1.0)
    ap.add_argument("--berry_method", choices=["fukui", "kubo", "both"], default="fukui",
                    help="Fukui is plaquette-link curvature; Kubo uses finite-difference dH/dq matrix elements")
    ap.add_argument("--kubo_gap_floor", type=float, default=1.0e-8,
                    help="Energy denominator floor in meV for Kubo Berry curvature")
    ap.add_argument("--track_bands", choices=["energy", "overlap"], default="energy",
                    help="energy keeps per-q eigenvalue order; overlap tracks branches by nearest-neighbor eigenvector overlap")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--plot", default=None)
    ap.add_argument("--plot_quantity", default="flux")
    ap.add_argument("--plot_bz", action=argparse.BooleanOptionalAction, default=False,
                    help="Draw plot as a periodic Voronoi map clipped to the first BZ")
    ap.add_argument("--plot_tile", type=int, default=2,
                    help="Periodic image range used to fill Voronoi cells before first-BZ clipping")
    ap.add_argument("--plot_cmap", default=None)
    ap.add_argument("--plot_edgecolor", default="none")
    ap.add_argument("--plot_linewidth", type=float, default=0.0)
    ap.add_argument("--plot_vmin", type=float, default=None)
    ap.add_argument("--plot_vmax", type=float, default=None)
    ap.add_argument("--plot_vmin_percentile", type=float, default=1.0)
    ap.add_argument("--plot_vmax_percentile", type=float, default=99.0)
    args = ap.parse_args()
    args.dJ_tensor_h5 = args.dJ_tensor_h5 or args.tensor_h5
    args.J_tensor_h5 = args.J_tensor_h5 or args.j_tensor_h5
    if not args.dJ_tensor_h5:
        ap.error("one of --dJ_tensor_h5 or legacy --tensor_h5 is required")
    if not args.J_tensor_h5:
        ap.error("one of --J_tensor_h5 or legacy --j_tensor_h5 is required")

    berry_method = args.berry_method
    if str(args.plot_quantity).strip().lower() in {"kubo", "kubo_curvature", "berry_curvature_kubo", "omega_kubo", "kubo_flux", "berry_flux_kubo"}:
        berry_method = "both" if berry_method == "fukui" else berry_method
    if args.band_sum is not None:
        bands = _parse_band_spec(args.band_sum)
        sum_bands = bands
        summed_output = True
    else:
        requested_bands = _parse_band_spec(args.band)
        if args.sum_below:
            bands = sorted(set(b for hi in requested_bands for b in range(0, int(hi) + 1)))
            sum_bands = bands
            summed_output = True
        else:
            bands = requested_bands
            sum_bands = []
            summed_output = False
    multiple = len(bands) > 1 and not summed_output
    results = compute_hybrid_berry_multi(
        tensor_h5=args.dJ_tensor_h5,
        j_tensor_h5=args.J_tensor_h5,
        epr_phonon=args.epr_phonon,
        phonon_cache=args.phonon_cache,
        component=args.component,
        static_component=args.static_component,
        S=args.S,
        spin_direction=args.spin_direction,
        spin_pattern=args.spin_pattern,
        phase_convention=args.phase_convention,
        plane=args.plane,
        fixed=args.fixed,
        mesh=args.mesh,
        bands=bands,
        threshold=args.threshold,
        structure=args.structure,
        coupling_scale=args.coupling_scale,
        berry_method=berry_method,
        kubo_gap_floor=args.kubo_gap_floor,
        track_bands=args.track_bands,
    )
    output_items = []
    if summed_output:
        output_items.append((None, sum_band_results(results, sum_bands), args.output, args.plot))
    else:
        for band in bands:
            output_items.append((
                int(band),
                results[int(band)],
                _band_suffix_path(args.output, int(band), multiple),
                _band_suffix_path(args.plot, int(band), multiple) if args.plot else None,
            ))
    for _, result, out_path, plot_path in output_items:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        save_berry_npz(out_path, result)
        _print_summary(result, out_path)
        if plot_path:
            os.makedirs(os.path.dirname(os.path.abspath(plot_path)) or ".", exist_ok=True)
            if args.plot_bz:
                plot_berry_bz(
                    plot_path,
                    result,
                    quantity=args.plot_quantity,
                    tile=args.plot_tile,
                    cmap=args.plot_cmap,
                    edgecolor=args.plot_edgecolor,
                    linewidth=args.plot_linewidth,
                    vmin=args.plot_vmin,
                    vmax=args.plot_vmax,
                    vmin_percentile=args.plot_vmin_percentile,
                    vmax_percentile=args.plot_vmax_percentile,
                )
            else:
                plot_berry(plot_path, result, quantity=args.plot_quantity)


if __name__ == "__main__":
    main()
