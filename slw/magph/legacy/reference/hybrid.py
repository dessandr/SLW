"""Hybridized magnon-phonon Hamiltonian helpers."""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import h5py
import numpy as np

import matplotlib

def _argv_value(flag: str):
    if flag not in sys.argv:
        return None
    idx = sys.argv.index(flag)
    if idx + 1 >= len(sys.argv):
        return None
    return sys.argv[idx + 1]


if "--show" in sys.argv:
    requested_backend = _argv_value("--mpl_backend")
    candidates = [requested_backend] if requested_backend else ["QtAgg", "TkAgg", "MacOSX"]
    for backend in candidates:
        if not backend:
            continue
        try:
            matplotlib.use(backend, force=True)
            break
        except Exception:
            continue
else:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from slw.core.cli_paths import resolve_in_dir, resolve_out_dir, resolve_path, resolve_workdir

from .. import lswt
from ..input_parser import parse_input_file
from ..utils import parsing_POSCAR
from ..vertex import ModeBasisVertex, build_site_basis_vertex, project_site_vertex_to_phonon_modes
from ..tensor_adapter import load_exchange_tensor_realspace_h5
from ..adapter import build_phonon_cache_from_epr, load_phonon_cache
from ..hybrid_html import save_hybrid_interactive_html

MEV_PER_THZ = 4.135667696


@dataclass
class HybridHamiltonian:
    kpts_frac: np.ndarray
    qpts_frac: np.ndarray
    magnon_energies: np.ndarray
    phonon_energies: np.ndarray
    magnon_vertex: np.ndarray
    hamiltonian: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    magnon_weight: np.ndarray
    magnon_chirality: np.ndarray
    hybrid_chirality: np.ndarray
    component: str
    phonon_angular_momentum: np.ndarray | None = None
    hybrid_phonon_angular_momentum: np.ndarray | None = None
    units: str = "meV"
    zeeman_field_mev: np.ndarray | None = None

    @property
    def shape_summary(self) -> dict[str, Any]:
        return {
            "nk": int(self.kpts_frac.shape[0]),
            "nq": int(self.qpts_frac.shape[0]),
            "n_magnon": int(self.magnon_energies.shape[-1]),
            "n_phonon": int(self.phonon_energies.shape[-1]),
            "vertex_shape": tuple(int(x) for x in self.magnon_vertex.shape),
            "hamiltonian_shape": tuple(int(x) for x in self.hamiltonian.shape),
            "eigenvectors_shape": tuple(int(x) for x in self.eigenvectors.shape),
            "magnon_weight_shape": tuple(int(x) for x in self.magnon_weight.shape),
            "magnon_chirality_shape": tuple(int(x) for x in self.magnon_chirality.shape),
            "hybrid_chirality_shape": tuple(int(x) for x in self.hybrid_chirality.shape),
            "phonon_angular_momentum_shape": None
            if self.phonon_angular_momentum is None
            else tuple(int(x) for x in self.phonon_angular_momentum.shape),
            "hybrid_phonon_angular_momentum_shape": None
            if self.hybrid_phonon_angular_momentum is None
            else tuple(int(x) for x in self.hybrid_phonon_angular_momentum.shape),
            "component": self.component,
            "units": self.units,
        }


def _load_j_tensor_h5(path: str, component: str = "full"):
    with h5py.File(path, "r") as h5:
        comp = str(component).strip().lower().replace(",", "+")
        aliases = {
            "gamma": "aniso",
            "dmi_tensor": "dmi",
            "iso_gamma": "iso+aniso",
            "symmetric": "iso+aniso",
        }
        comp = aliases.get(comp, comp)
        if comp == "full":
            parts = ("full",)
        else:
            raw_parts = tuple(aliases.get(x.strip(), x.strip()) for x in comp.split("+") if x.strip())
            allowed = {"iso", "aniso", "dmi"}
            if not raw_parts or any(x not in allowed for x in raw_parts):
                raise ValueError(
                    f"Unknown static tensor component {component!r}; "
                    "use iso, aniso, dmi, full, or combinations like iso+dmi"
                )
            parts = tuple(dict.fromkeys(raw_parts))
        part_keys = {
            "full": "J_tensor_r",
            "iso": "J_iso_tensor_r",
            "aniso": "J_gamma_r",
            "dmi": "J_dmi_tensor_r",
        }
        required = ["bonds/mag_i_atom", "bonds/mag_j_atom", "bonds/R"]
        required.extend(part_keys[x] for x in parts)
        missing = [x for x in required if x not in h5]
        if missing:
            raise KeyError(f"Missing static J tensor datasets in {path}: {missing}")
        tensor = None
        for part in parts:
            arr = np.asarray(h5[part_keys[part]], dtype=np.float64)
            tensor = arr if tensor is None else tensor + arr
        ii = np.asarray(h5["bonds/mag_i_atom"], dtype=np.int32).reshape(-1)
        jj = np.asarray(h5["bonds/mag_j_atom"], dtype=np.int32).reshape(-1)
        rr = np.asarray(h5["bonds/R"], dtype=np.int32).reshape(-1, 3)
    if tensor.ndim != 3 or tensor.shape[-2:] != (3, 3):
        raise ValueError(f"Unsupported static J tensor shape {tensor.shape}")
    return tensor, ii, jj, rr


def _filter_static_bonds(tensor, ii, jj, rr, *, include_onsite_self: bool = False):
    keep = np.all(np.isfinite(tensor), axis=(1, 2))
    if not include_onsite_self:
        keep &= ~((ii == jj) & np.all(rr == 0, axis=1))
    if not np.any(keep):
        raise ValueError("No static J tensor bonds remain after filtering")
    return tensor[keep], ii[keep], jj[keep], rr[keep]


def _load_structure(structure: str | None, fallback_h5: str | None = None):
    if structure:
        lattice, atom_labels, atom_pos = parsing_POSCAR(structure)
        return np.asarray(lattice, dtype=np.float64), atom_labels, np.asarray(atom_pos, dtype=np.float64)
    if fallback_h5:
        with h5py.File(fallback_h5, "r") as h5:
            if "basic_data/lattice_ang" in h5 and "basic_data/tau_cart_ang" in h5:
                lattice = np.asarray(h5["basic_data/lattice_ang"], dtype=np.float64).reshape(3, 3)
                atom_pos = np.asarray(h5["basic_data/tau_cart_ang"], dtype=np.float64).reshape(-1, 3)
                labels = []
                if "basic_data/atom_labels" in h5:
                    for x in np.asarray(h5["basic_data/atom_labels"]).tolist():
                        labels.append(x.decode() if isinstance(x, bytes) else str(x))
                return lattice, labels, atom_pos
    raise ValueError("Structure is required: use --structure or provide J tensor HDF5 with basic_data/lattice_ang")


def _load_atom_frac(structure: str | None, fallback_h5: str | None = None):
    if structure:
        lattice, _, atom_pos = parsing_POSCAR(structure)
        return (np.asarray(atom_pos, dtype=np.float64) @ np.linalg.inv(np.asarray(lattice, dtype=np.float64))) % 1.0
    if fallback_h5:
        with h5py.File(fallback_h5, "r") as h5:
            if "basic_data/tau_frac" in h5:
                return np.asarray(h5["basic_data/tau_frac"], dtype=np.float64).reshape(-1, 3) % 1.0
    lattice, _, atom_pos = _load_structure(None, fallback_h5)
    return (atom_pos @ np.linalg.inv(lattice)) % 1.0


def _fractional_mesh(mesh) -> np.ndarray:
    n1, n2, n3 = [int(x) for x in mesh]
    grid = np.meshgrid(
        np.arange(n1, dtype=np.float64) / n1,
        np.arange(n2, dtype=np.float64) / n2,
        np.arange(n3, dtype=np.float64) / n3,
        indexing="ij",
    )
    return np.stack([g.reshape(-1) for g in grid], axis=1)


def _default_kpath_points():
    return {
        "G": np.array([0.0, 0.0, 0.0], dtype=np.float64),
        "GAMMA": np.array([0.0, 0.0, 0.0], dtype=np.float64),
        "Γ": np.array([0.0, 0.0, 0.0], dtype=np.float64),
        "M": np.array([0.5, 0.0, 0.0], dtype=np.float64),
        "K": np.array([1.0 / 3.0, 1.0 / 3.0, 0.0], dtype=np.float64),
        "A": np.array([0.0, 0.0, 0.5], dtype=np.float64),
        "L": np.array([0.5, 0.0, 0.5], dtype=np.float64),
        "H": np.array([1.0 / 3.0, 1.0 / 3.0, 0.5], dtype=np.float64),
        "LP": np.array([0.0, 0.5, 0.5], dtype=np.float64),
        "L'": np.array([0.0, 0.5, 0.5], dtype=np.float64),
    }


def _parse_kpoint_token(token: str) -> np.ndarray:
    points = _default_kpath_points()
    key = str(token).strip()
    lookup = key.upper().replace("Γ", "G")
    if lookup in points:
        return points[lookup].copy()
    vals = [float(x) for x in key.replace(",", " ").split()]
    if len(vals) != 3:
        raise ValueError(f"Unknown k/q path token {token!r}; use symbols or 'kx,ky,kz'")
    return np.asarray(vals, dtype=np.float64)


def _format_fraction_label(value: float, max_denominator: int = 12) -> str:
    frac = Fraction(float(value)).limit_denominator(int(max_denominator))
    if frac.denominator == 1:
        return str(frac.numerator)
    return f"{frac.numerator}/{frac.denominator}"


def _format_miller_label(vec, *, max_denominator: int = 12) -> str:
    vals = np.asarray(vec, dtype=np.float64).reshape(3)
    parts = [_format_fraction_label(x, max_denominator=max_denominator) for x in vals]
    return r"$(" + r"\ ".join(parts) + r")$"


def generate_fractional_path(path: str, points_per_segment: int, recip=None, label_style: str = "symbol"):
    tokens = [x for x in str(path).replace(";", " ").split() if x]
    if len(tokens) < 2:
        raise ValueError("--qpath needs at least two points, e.g. 'G M K G'")
    nodes = np.asarray([_parse_kpoint_token(tok) for tok in tokens], dtype=np.float64)
    qpts = []
    tick_indices = []
    for iseg in range(len(nodes) - 1):
        if iseg == 0:
            tick_indices.append(0)
        segment = np.linspace(nodes[iseg], nodes[iseg + 1], int(points_per_segment), endpoint=False)
        qpts.extend(segment.tolist())
        tick_indices.append(len(qpts))
    qpts.append(nodes[-1].tolist())
    qpts = np.asarray(qpts, dtype=np.float64)
    metric = np.eye(3) if recip is None else np.asarray(recip, dtype=np.float64)
    cart = qpts @ metric
    kdist = np.zeros(qpts.shape[0], dtype=np.float64)
    if qpts.shape[0] > 1:
        kdist[1:] = np.cumsum(np.linalg.norm(cart[1:] - cart[:-1], axis=1))
    symbol_labels = [r"$\Gamma$" if t.upper() in {"G", "GAMMA"} or t == "Γ" else t for t in tokens]
    style = str(label_style).strip().lower()
    if style == "symbol":
        labels = symbol_labels
    elif style == "miller":
        labels = [_format_miller_label(v) for v in nodes]
    elif style == "both":
        labels = [f"{sym}\n{_format_miller_label(v)}" for sym, v in zip(symbol_labels, nodes)]
    else:
        raise ValueError(f"Unknown qpath label style {label_style!r}; use symbol, miller, or both")
    return qpts % 1.0, kdist, np.asarray(tick_indices, dtype=np.int32), labels


def transform_vertex_to_magnon_modes(mode_vertex: ModeBasisVertex, magnon_modes_kq: np.ndarray) -> np.ndarray:
    """Transform site/Nambu vertex to positive magnon mode basis.

    `mode_vertex.vertex` shape is `(nq, n_ph, 2*nmag)`.
    `magnon_modes_kq` shape is `(nk, nq, 2*nmag, nmag)`.
    Returned shape is `(nk, nq, n_ph, nmag)`.
    """
    site_v = np.asarray(mode_vertex.vertex, dtype=np.complex128)
    modes = np.asarray(magnon_modes_kq, dtype=np.complex128)
    if modes.ndim != 4 or modes.shape[1] != site_v.shape[0] or modes.shape[2] != site_v.shape[2]:
        raise ValueError(f"shape mismatch: site_vertex={site_v.shape}, magnon_modes_kq={modes.shape}")
    return np.einsum("qpn,kqnm->kqpm", site_v, modes, optimize=True)


def regularize_phonon_energies(
    mode_vertex: ModeBasisVertex,
    *,
    negative_tol_mev: float = 0.0,
    gamma_zero_tol: float = 1.0e-8,
    gamma_acoustic_modes: int = 0,
) -> ModeBasisVertex:
    """Clip small negative phonon energies before building the hybrid Hamiltonian."""
    ph = np.array(mode_vertex.phonon_energies, dtype=np.float64, copy=True)
    negative_tol = float(max(0.0, negative_tol_mev))
    if negative_tol > 0.0:
        small_negative = (ph < 0.0) & (ph >= -negative_tol)
        n_small = int(np.count_nonzero(small_negative))
        if n_small:
            print(f"[magph-hybrid] clipped {n_small} small negative phonon energies to 0 meV (tol={negative_tol:g})")
            ph[small_negative] = 0.0
        large_negative = ph < -negative_tol
        n_large = int(np.count_nonzero(large_negative))
        if n_large:
            print(
                "[magph-hybrid][WARN] "
                f"{n_large} phonon energies are below -{negative_tol:g} meV; leaving them unchanged."
            )
    nac = int(max(0, gamma_acoustic_modes))
    if nac > 0:
        q = np.asarray(mode_vertex.qpts_frac, dtype=np.float64)
        qwrap = q - np.rint(q)
        gamma_mask = np.linalg.norm(qwrap, axis=1) <= float(gamma_zero_tol)
        n_gamma = int(np.count_nonzero(gamma_mask))
        if n_gamma:
            nclip = min(nac, ph.shape[1])
            ph[gamma_mask, :nclip] = 0.0
            print(
                f"[magph-hybrid] set lowest {nclip} acoustic phonon modes to 0 meV "
                f"at {n_gamma} Gamma q-point(s)"
            )
    return ModeBasisVertex(
        qpts_frac=mode_vertex.qpts_frac,
        phonon_energies=ph,
        phonon_vectors=mode_vertex.phonon_vectors,
        vertex=mode_vertex.vertex,
        site_vertex=mode_vertex.site_vertex,
        phonon_factor=mode_vertex.phonon_factor,
        units=mode_vertex.units,
        phonon_masses=mode_vertex.phonon_masses,
    )


def magnon_mode_chirality(magnon_modes_kq: np.ndarray, spin_pattern) -> np.ndarray:
    """Return a signed chirality proxy for positive BdG magnon modes.

    The sign is the ordered-spin-axis angular-momentum proxy
    `-sum_i eta_i (|u_i|^2-|v_i|^2)`, normalized by the absolute local metric
    density.  For a two-sublattice collinear AFM this separates the two circular
    magnon branches while phonon-like hybrid states remain near zero after
    projection.
    """
    modes = np.asarray(magnon_modes_kq, dtype=np.complex128)
    nk, nq, n2, nmode = modes.shape
    nmag = n2 // 2
    if n2 != 2 * nmag:
        raise ValueError(f"Expected Nambu dimension 2*nmag, got {n2}")
    signs = np.asarray(spin_pattern, dtype=np.float64).reshape(nmag)
    u = modes[:, :, :nmag, :]
    v = modes[:, :, nmag:, :]
    metric_density = np.abs(u) ** 2 - np.abs(v) ** 2
    num = -np.einsum("i,kqim->kqm", signs, metric_density, optimize=True)
    den = np.sum(np.abs(metric_density), axis=2)
    return np.divide(num, den, out=np.zeros_like(num, dtype=np.float64), where=den > 1.0e-14)


def phonon_mode_angular_momentum(
    phonon_vectors: np.ndarray,
    *,
    masses: np.ndarray | None,
    normalize: bool = True,
) -> np.ndarray:
    """Return physical phonon angular momentum in units of hbar.

    Magph caches store physical displacements ``u=e/sqrt(M)``, so the mass
    metric is mandatory.  Refusing a missing mass array prevents mixed-mass
    systems from silently receiving the old unweighted polarization proxy.
    """
    if masses is None:
        raise ValueError(
            "phonon angular momentum requires atom masses because cached "
            "phonon vectors are physical displacements u=e/sqrt(M); "
            "regenerate the phonon cache with the current adapter"
        )
    from ..rotational import phonon_angular_momentum_over_hbar

    return phonon_angular_momentum_over_hbar(
        phonon_vectors,
        np.asarray(masses, dtype=np.float64),
        normalize=normalize,
    )


def build_hybrid_hamiltonian_blocks(
    *,
    kpts_frac,
    qpts_frac,
    magnon_energies_kq,
    magnon_modes_kq,
    mode_vertex: ModeBasisVertex,
    magnon_chirality_kq=None,
    coupling_scale: float = 1.0,
    component: str = "dmi",
) -> HybridHamiltonian:
    mag_e = np.asarray(magnon_energies_kq, dtype=np.float64)
    ph_e = np.asarray(mode_vertex.phonon_energies, dtype=np.float64)
    g = transform_vertex_to_magnon_modes(mode_vertex, magnon_modes_kq) * float(coupling_scale)
    nk, nq, nph, nmag = g.shape
    if mag_e.shape != (nk, nq, nmag):
        raise ValueError(f"magnon energy shape mismatch: {mag_e.shape}, expected={(nk, nq, nmag)}")
    if ph_e.shape != (nq, nph):
        raise ValueError(f"phonon energy shape mismatch: {ph_e.shape}, expected={(nq, nph)}")
    if magnon_chirality_kq is None:
        mag_chi = np.zeros((nk, nq, nmag), dtype=np.float64)
    else:
        mag_chi = np.asarray(magnon_chirality_kq, dtype=np.float64)
        if mag_chi.shape != (nk, nq, nmag):
            raise ValueError(f"magnon chirality shape mismatch: {mag_chi.shape}, expected={(nk, nq, nmag)}")
    ndim = nmag + nph
    H = np.zeros((nk, nq, ndim, ndim), dtype=np.complex128)
    idx_m = np.arange(nmag)
    idx_p = np.arange(nph)
    H[:, :, idx_m, idx_m] = mag_e
    H[:, :, nmag + idx_p, nmag + idx_p] = ph_e[None, :, :]
    H[:, :, :nmag, nmag:] = np.swapaxes(g, -1, -2)
    H[:, :, nmag:, :nmag] = np.conj(g)
    H = 0.5 * (H + np.swapaxes(H.conj(), -1, -2))
    vals, vecs = np.linalg.eigh(H.reshape(-1, ndim, ndim))
    evals = vals.real.reshape(nk, nq, ndim)
    evecs = vecs.reshape(nk, nq, ndim, ndim)
    mag_amp2 = np.abs(evecs[:, :, :nmag, :]) ** 2
    ph_amp2 = np.abs(evecs[:, :, nmag:, :]) ** 2
    weights = np.sum(mag_amp2, axis=2).real
    hybrid_chi = np.einsum("kqmn,kqm->kqn", mag_amp2, mag_chi, optimize=True).real
    ph_l = phonon_mode_angular_momentum(
        mode_vertex.phonon_vectors,
        masses=mode_vertex.phonon_masses,
    )
    hybrid_ph_l = np.einsum("kqpn,qpv->kqnv", ph_amp2, ph_l, optimize=True).real
    return HybridHamiltonian(
        kpts_frac=np.asarray(kpts_frac, dtype=np.float64),
        qpts_frac=np.asarray(qpts_frac, dtype=np.float64),
        magnon_energies=mag_e,
        phonon_energies=ph_e,
        magnon_vertex=g,
        hamiltonian=H,
        eigenvalues=evals,
        eigenvectors=evecs,
        magnon_weight=weights,
        magnon_chirality=mag_chi,
        hybrid_chirality=hybrid_chi,
        phonon_angular_momentum=ph_l,
        hybrid_phonon_angular_momentum=hybrid_ph_l,
        component=str(component),
        zeeman_field_mev=None,
    )


def build_hybrid_from_static_tensor(
    *,
    j_tensor_h5: str,
    structure: str | None,
    mode_vertex: ModeBasisVertex,
    S: float,
    spin_info: lswt.SpinFrameInfo,
    kpts_frac=None,
    kmesh=(1, 1, 1),
    static_component: str = "iso",
    include_onsite_self: bool = False,
    bond_factor: float = 1.0,
    anisotropy_mev: float = 0.0,
    zeeman_field_mev=None,
    coupling_scale: float = 1.0,
    nproc: int = 1,
) -> HybridHamiltonian:
    lattice, _, atom_pos = _load_structure(structure, fallback_h5=j_tensor_h5)
    recip = 2.0 * np.pi * np.linalg.inv(lattice).T
    tensor, ii, jj, rr = _load_j_tensor_h5(j_tensor_h5, component=static_component)
    tensor, ii, jj, rr = _filter_static_bonds(tensor, ii, jj, rr, include_onsite_self=include_onsite_self)

    mag_ids = np.asarray(sorted(set(int(x) for x in ii.tolist() + jj.tolist())), dtype=np.int32)
    remap = {int(g): iloc for iloc, g in enumerate(mag_ids)}
    ii_local = np.asarray([remap[int(x)] for x in ii], dtype=np.int32)
    jj_local = np.asarray([remap[int(x)] for x in jj], dtype=np.int32)
    atom_pos_mag = atom_pos[mag_ids]
    if spin_info.nmag != len(mag_ids):
        raise ValueError(f"spin_info.nmag={spin_info.nmag}, static J nmag={len(mag_ids)}")

    if kpts_frac is None:
        kpts_frac = _fractional_mesh(kmesh)
    kpts_frac = np.asarray(kpts_frac, dtype=np.float64).reshape(-1, 3)
    qpts_frac = np.asarray(mode_vertex.qpts_frac, dtype=np.float64)
    nk = kpts_frac.shape[0]
    nq = qpts_frac.shape[0]
    nmag = spin_info.nmag
    lswt_nproc = max(1, int(nproc))
    kq_frac = (kpts_frac[:, None, :] + qpts_frac[None, :, :]) % 1.0
    kq_cart = kq_frac.reshape(-1, 3) @ recip
    print(
        f"[magph-hybrid] solving bare magnons: nk={nk} nq={nq} "
        f"n_kq={int(kq_cart.shape[0])} nproc={lswt_nproc}",
        flush=True,
    )
    omega, modes = lswt.solve_tensor_lswt(
        kq_cart,
        S,
        tensor,
        ii_local,
        jj_local,
        rr,
        lattice,
        atom_pos_mag,
        spin_info,
        nproc=lswt_nproc,
        bond_factor=bond_factor,
        anisotropy_mev=anisotropy_mev,
        zeeman_field_mev=zeeman_field_mev,
    )
    mag_e = omega.reshape(nk, nq, nmag)
    mag_u = modes.reshape(nk, nq, 2 * nmag, nmag)
    mag_chi = magnon_mode_chirality(mag_u, spin_info.spin_pattern)
    out = build_hybrid_hamiltonian_blocks(
        kpts_frac=kpts_frac,
        qpts_frac=qpts_frac,
        magnon_energies_kq=mag_e,
        magnon_modes_kq=mag_u,
        magnon_chirality_kq=mag_chi,
        mode_vertex=mode_vertex,
        coupling_scale=coupling_scale,
        component=mode_vertex.site_vertex.component,
    )
    out.zeeman_field_mev = None if zeeman_field_mev is None else np.asarray(zeeman_field_mev, dtype=np.float64)
    return out


def save_hybrid_npz(path: str, data: HybridHamiltonian):
    np.savez_compressed(
        path,
        kpts_frac=data.kpts_frac,
        qpts_frac=data.qpts_frac,
        magnon_energies=data.magnon_energies,
        phonon_energies=data.phonon_energies,
        magnon_vertex=data.magnon_vertex,
        hybrid_hamiltonian=data.hamiltonian,
        hybrid_eigenvalues=data.eigenvalues,
        hybrid_eigenvectors=data.eigenvectors,
        magnon_weight=data.magnon_weight,
        magnon_chirality=data.magnon_chirality,
        hybrid_chirality=data.hybrid_chirality,
        phonon_angular_momentum=np.asarray([] if data.phonon_angular_momentum is None else data.phonon_angular_momentum),
        hybrid_phonon_angular_momentum=np.asarray(
            [] if data.hybrid_phonon_angular_momentum is None else data.hybrid_phonon_angular_momentum
        ),
        component=np.asarray(data.component, dtype=object),
        units=np.asarray(data.units, dtype=object),
        zeeman_field_mev=np.asarray([] if data.zeeman_field_mev is None else data.zeeman_field_mev, dtype=np.float64),
    )


def save_hybrid_band_npz(path: str, data: HybridHamiltonian, *, kdist=None, tick_indices=None, kpath_symbols=None):
    payload = {
        "kpts_frac": data.kpts_frac,
        "qpts_frac": data.qpts_frac,
        "magnon_energies": data.magnon_energies,
        "phonon_energies": data.phonon_energies,
        "magnon_vertex": data.magnon_vertex,
        "hybrid_hamiltonian": data.hamiltonian,
        "hybrid_eigenvalues": data.eigenvalues,
        "hybrid_eigenvectors": data.eigenvectors,
        "magnon_weight": data.magnon_weight,
        "magnon_chirality": data.magnon_chirality,
        "hybrid_chirality": data.hybrid_chirality,
        "phonon_angular_momentum": np.asarray([] if data.phonon_angular_momentum is None else data.phonon_angular_momentum),
        "hybrid_phonon_angular_momentum": np.asarray(
            [] if data.hybrid_phonon_angular_momentum is None else data.hybrid_phonon_angular_momentum
        ),
        "component": np.asarray(data.component, dtype=object),
        "units": np.asarray(data.units, dtype=object),
    }
    if kdist is not None:
        payload["kdist"] = np.asarray(kdist, dtype=np.float64)
    if tick_indices is not None:
        payload["tick_indices"] = np.asarray(tick_indices, dtype=np.int32)
    if kpath_symbols is not None:
        payload["kpath_symbols"] = np.asarray(kpath_symbols, dtype=object)
    np.savez_compressed(path, **payload)


def _convert_energy(e_mev, unit: str):
    if str(unit).lower() == "thz":
        return np.asarray(e_mev, dtype=np.float64) / MEV_PER_THZ, "Frequency (THz)"
    if str(unit).lower() == "mev":
        return np.asarray(e_mev, dtype=np.float64), "Energy (meV)"
    raise ValueError(f"Unsupported unit {unit!r}")


def plot_bare_band(
    data: HybridHamiltonian,
    out_png: str,
    *,
    kdist=None,
    tick_indices=None,
    kpath_symbols=None,
    unit="THz",
    xtick_fontsize=None,
    show=False,
):
    if data.magnon_energies.shape[0] != 1:
        raise ValueError("plot_bare_band expects nk=1 path data")
    x = np.arange(data.qpts_frac.shape[0], dtype=np.float64) if kdist is None else np.asarray(kdist, dtype=np.float64)
    mag_y, ylabel = _convert_energy(data.magnon_energies[0], unit)
    ph_y, _ = _convert_energy(data.phonon_energies, unit)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for ib in range(ph_y.shape[1]):
        ax.plot(x, ph_y[:, ib], color="0.68", lw=0.8, alpha=0.85)
    for ib in range(mag_y.shape[1]):
        ax.plot(x, mag_y[:, ib], color="#b2182b", lw=1.4, alpha=0.95)
    if tick_indices is not None and kpath_symbols is not None:
        ticks = np.asarray(tick_indices, dtype=np.int32)
        ticks = ticks[(ticks >= 0) & (ticks < len(x))]
        ax.set_xticks(x[ticks])
        ax.set_xticklabels(list(kpath_symbols)[: len(ticks)], fontsize=xtick_fontsize)
        for t in x[ticks]:
            ax.axvline(t, color="0.82", lw=0.5, zorder=0)
    ax.set_xlim(float(x[0]), float(x[-1]))
    ax.set_ylabel(ylabel)
    ax.set_xlabel("q path")
    ax.plot([], [], color="#b2182b", lw=1.4, label="bare magnon")
    ax.plot([], [], color="0.68", lw=0.8, label="bare phonon")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    print(f"[magph-hybrid] wrote bare plot {out_png}")
    if show:
        plt.show()


def plot_hybrid_band(
    data: HybridHamiltonian,
    out_png: str,
    *,
    kdist=None,
    tick_indices=None,
    kpath_symbols=None,
    unit="THz",
    overlay_bare=False,
    color_by="magnon_weight",
    xtick_fontsize=None,
    show=False,
):
    if data.eigenvalues.shape[0] != 1:
        raise ValueError("plot_hybrid_band expects nk=1 path data")
    x = np.arange(data.qpts_frac.shape[0], dtype=np.float64) if kdist is None else np.asarray(kdist, dtype=np.float64)
    y_mev = data.eigenvalues[0]
    color_key = str(color_by).strip().lower()
    if color_key in {"magnon_weight", "weight", "mw"}:
        color_values = np.clip(data.magnon_weight[0], 0.0, 1.0)
        norm = plt.Normalize(0.0, 1.0)
        cmap = "viridis"
        color_label = "Magnon weight"
    elif color_key in {"chirality", "chi", "magnon_chirality"}:
        color_values = np.clip(data.hybrid_chirality[0], -1.0, 1.0)
        norm = plt.Normalize(-1.0, 1.0)
        cmap = "seismic"
        color_label = "Magnon chirality"
    elif color_key in {"phonon_angular_momentum", "phonon_l", "phonon_lz", "angular_momentum", "lz"}:
        if data.hybrid_phonon_angular_momentum is None:
            raise ValueError("hybrid_phonon_angular_momentum is not available")
        color_values = np.clip(data.hybrid_phonon_angular_momentum[0, :, :, 2], -1.0, 1.0)
        norm = plt.Normalize(-1.0, 1.0)
        cmap = "seismic"
        color_label = r"Phonon $L_z$"
    else:
        raise ValueError(f"Unknown color_by={color_by!r}; use magnon_weight, chirality, or phonon_lz")
    y, ylabel = _convert_energy(y_mev, unit)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    if overlay_bare:
        mag_y, _ = _convert_energy(data.magnon_energies[0], unit)
        ph_y, _ = _convert_energy(data.phonon_energies, unit)
        for ib in range(ph_y.shape[1]):
            ax.plot(x, ph_y[:, ib], color="0.78", lw=0.6, alpha=0.5, zorder=0)
        for ib in range(mag_y.shape[1]):
            ax.plot(x, mag_y[:, ib], color="black", lw=0.8, alpha=0.65, ls="--", zorder=1)
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    for ib in range(y.shape[1]):
        pts = np.column_stack([x, y[:, ib]])
        segments = np.stack([pts[:-1], pts[1:]], axis=1)
        segment_weight = 0.5 * (color_values[:-1, ib] + color_values[1:, ib])
        lc = LineCollection(
            segments,
            array=segment_weight,
            cmap=cmap,
            norm=norm,
            linewidth=1.2,
            alpha=0.98,
            zorder=2,
        )
        ax.add_collection(lc)
    if tick_indices is not None and kpath_symbols is not None:
        ticks = np.asarray(tick_indices, dtype=np.int32)
        ticks = ticks[(ticks >= 0) & (ticks < len(x))]
        ax.set_xticks(x[ticks])
        ax.set_xticklabels(list(kpath_symbols)[: len(ticks)], fontsize=xtick_fontsize)
        for t in x[ticks]:
            ax.axvline(t, color="0.82", lw=0.5, zorder=0)
    ax.set_xlim(float(x[0]), float(x[-1]))
    finite_y = y[np.isfinite(y)]
    if finite_y.size:
        ymin = float(np.min(finite_y))
        ymax = float(np.max(finite_y))
        pad = 0.04 * max(ymax - ymin, 1.0e-12)
        ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("q path")
    cbar = fig.colorbar(mappable, ax=ax)
    cbar.set_label(color_label)
    if overlay_bare:
        ax.plot([], [], color="black", lw=0.8, ls="--", label="bare magnon")
        ax.plot([], [], color="0.78", lw=0.6, label="bare phonon")
        ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    print(f"[magph-hybrid] wrote plot {out_png}")
    if show:
        plt.show()


def save_bare_npz(path: str, data: HybridHamiltonian, *, kdist=None, tick_indices=None, kpath_symbols=None):
    payload = {
        "kpts_frac": data.kpts_frac,
        "qpts_frac": data.qpts_frac,
        "bare_magnon_mev": data.magnon_energies,
        "bare_phonon_mev": data.phonon_energies,
        "bare_magnon_thz": data.magnon_energies / MEV_PER_THZ,
        "bare_phonon_thz": data.phonon_energies / MEV_PER_THZ,
        "bare_magnon_chirality": data.magnon_chirality,
        "bare_phonon_angular_momentum": np.asarray([] if data.phonon_angular_momentum is None else data.phonon_angular_momentum),
        "mev_per_thz": np.asarray(MEV_PER_THZ, dtype=np.float64),
    }
    if kdist is not None:
        payload["kdist"] = np.asarray(kdist, dtype=np.float64)
    if tick_indices is not None:
        payload["tick_indices"] = np.asarray(tick_indices, dtype=np.int32)
    if kpath_symbols is not None:
        payload["kpath_symbols"] = np.asarray(kpath_symbols, dtype=object)
    np.savez_compressed(path, **payload)


def _arg_or_cfg(args, cfg, name, default=None, *cfg_names):
    value = getattr(args, name, None)
    if value is not None:
        return value
    for key in cfg_names or (name,):
        if key in cfg and cfg[key] is not None:
            return cfg[key]
    return default


def _resolve_optional_path(base_dir, path):
    if path is None:
        return None
    return resolve_path(base_dir, path)


def _save_phonon_cache(path, cache, *, compressed=True):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    writer = np.savez_compressed if compressed else np.savez
    writer(path, **cache)


def _normalize_calculation_mode(mode):
    text = str(mode).strip().lower().replace("-", "_")
    aliases = {
        "cache": "cache",
        "cached": "cache",
        "reuse": "cache",
        "use_cache": "cache",
        "from_scratch": "from_scratch",
        "scratch": "from_scratch",
        "recompute": "from_scratch",
        "force": "from_scratch",
        "force_recompute": "from_scratch",
    }
    if text not in aliases:
        raise ValueError("calculation_mode must be cache/reuse or from_scratch/recompute")
    return aliases[text]


def _stage_timer(label):
    t0 = time.perf_counter()

    def done():
        print(f"[magph-hybrid] {label} finished in {time.perf_counter() - t0:.2f}s", flush=True)

    return done


def main():
    ap = argparse.ArgumentParser(description="Build hybridized magnon-phonon Hamiltonian from tensor dJ and static J")
    ap.add_argument("--workdir", default=None, help="Workflow root directory (default: current directory)")
    ap.add_argument("--input_file", default=None, help="Optional input.in/magph.in with hybrid variables")
    ap.add_argument(
        "--calculation_mode",
        default=None,
        help="Phonon cache mode: cache/reuse or from_scratch/recompute",
    )
    ap.add_argument("--in_dir", default=None, help="Input directory for tensor/phonon/structure files")
    ap.add_argument("--out_dir", default=None, help="Output directory for hybrid outputs")
    ap.add_argument("--dJ_tensor_h5", default=None, help="dynamic dJ/du tensor HDF5 from compute_dJ_epr_tensor")
    ap.add_argument("--tensor_h5", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--epr_phonon", default=None)
    ap.add_argument("--phonon_cache", default=None)
    ap.add_argument("--phonon_cache_out", default=None, help="Write qpath/qmesh phonon cache built from --epr_phonon")
    ap.add_argument(
        "--phonon_cache_compressed",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Compress --phonon_cache_out; use --no-phonon_cache_compressed for faster writes",
    )
    ap.add_argument("--phonon_qmesh", type=int, nargs=3, default=None)
    ap.add_argument("--phonon_nproc", type=int, default=None, help="Worker processes when building phonon cache from --epr_phonon")
    ap.add_argument("--hybrid_nproc", type=int, default=None, help="Worker processes for bare magnon LSWT along the hybrid path")
    ap.add_argument("--J_tensor_h5", default=None, help="static exchange tensor HDF5 from compute_J_epr_tensor")
    ap.add_argument("--j_tensor_h5", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--structure", default=None, help="POSCAR/QE input with structure; optional if J tensor HDF5 stores lattice/tau")
    ap.add_argument("--component", default=None, choices=["dmi", "iso", "aniso", "full"])
    ap.add_argument(
        "--static_component",
        default=None,
        help="Static exchange tensor for bare magnons: iso, aniso, dmi, full, or combinations like iso+dmi.",
    )
    ap.add_argument("--S", type=float, default=None)
    ap.add_argument("--spin_direction", type=float, nargs=3, default=None)
    ap.add_argument("--spin_pattern", default=None)
    ap.add_argument("--phase_convention", choices=["cell", "basis"], default=None)
    ap.add_argument("--kmesh", type=int, nargs=3, default=None)
    ap.add_argument("--qpath", default=None, help="Band path for q, e.g. 'G M K G A'. Coordinates like '0,0,0 0.5,0,0' also work.")
    ap.add_argument("--q_points", default=None, help="Explicit list of q-points, overriding qpath and qmesh")
    ap.add_argument(
        "--qpath_label_style",
        choices=["symbol", "miller", "both"],
        default=None,
        help="X tick label style for qpath nodes.",
    )
    ap.add_argument("--path_points", type=int, default=None, help="Points per q-path segment")
    ap.add_argument("--plot", default=None, help="Optional hybrid band plot colored by magnon weight")
    ap.add_argument("--html_plot", default=None, help="Optional interactive standalone HTML hybrid band plot")
    ap.add_argument("--html_data", default=None, help="Optional JSON sidecar path for --html_plot")
    ap.add_argument("--plot_unit", choices=["meV", "THz"], default=None)
    ap.add_argument("--xtick_fontsize", type=float, default=None, help="Font size for q-path x tick labels")
    ap.add_argument("--color_by", choices=["magnon_weight", "chirality", "phonon_lz"], default=None)
    ap.add_argument(
        "--overlay_bare",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Overlay bare magnon/phonon branches on the hybrid plot",
    )
    ap.add_argument("--bare_output", default=None, help="Optional npz containing only bare magnon/phonon branches")
    ap.add_argument("--bare_plot", default=None, help="Optional bare magnon/phonon branch plot")
    ap.add_argument("--show", action=argparse.BooleanOptionalAction, default=None, help="Show matplotlib windows after saving plots")
    ap.add_argument("--mpl_backend", default=None, help="Matplotlib interactive backend for --show, e.g. TkAgg or QtAgg")
    ap.add_argument("--coupling_scale", type=float, default=None)
    ap.add_argument("--bond_factor", type=float, default=None)
    ap.add_argument("--anisotropy_mev", type=float, default=None)
    ap.add_argument(
        "--zeeman_field_mev",
        type=float,
        nargs=3,
        default=None,
        help="Zeeman energy vector h=g*mu_B*B in meV, Cartesian coordinates.",
    )
    ap.add_argument(
        "--field_tesla",
        type=float,
        nargs=3,
        default=None,
        help="Magnetic field vector in Tesla; converted to meV using --g_factor.",
    )
    ap.add_argument("--g_factor", type=float, default=None, help="g factor for --field_tesla conversion")
    ap.add_argument(
        "--phonon_negative_tol_mev",
        type=float,
        default=None,
        help="Clip phonon energies in [-tol, 0) to 0 before hybridization.",
    )
    ap.add_argument(
        "--gamma_acoustic_zero",
        type=int,
        default=None,
        help="Set the lowest N phonon modes at Gamma to 0 before hybridization, usually N=3.",
    )
    ap.add_argument("--gamma_zero_tol", type=float, default=None, help="Fractional-q tolerance for --gamma_acoustic_zero")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("-o", "--output", default=None)
    args = ap.parse_args()

    workdir = resolve_workdir(args.workdir)
    cfg = {"workdir": workdir}
    if args.input_file:
        input_path = resolve_path(workdir, args.input_file)
        cfg.update(parse_input_file(input_path))
        cfg["_input_file_path"] = input_path
    in_dir = resolve_in_dir(workdir, _arg_or_cfg(args, cfg, "in_dir", None), default_subdir=".")
    out_dir = resolve_out_dir(workdir, _arg_or_cfg(args, cfg, "out_dir", None), default_subdir=".")

    args.dJ_tensor_h5 = args.dJ_tensor_h5 or args.tensor_h5 or cfg.get("dJ_tensor_h5")
    args.J_tensor_h5 = args.J_tensor_h5 or args.j_tensor_h5 or cfg.get("J_tensor_h5")
    if not args.dJ_tensor_h5:
        ap.error("one of --dJ_tensor_h5 or legacy --tensor_h5 is required")
    if not args.J_tensor_h5:
        ap.error("one of --J_tensor_h5 or legacy --j_tensor_h5 is required")
    args.dJ_tensor_h5 = resolve_path(in_dir, args.dJ_tensor_h5)
    args.J_tensor_h5 = resolve_path(in_dir, args.J_tensor_h5)

    args.epr_phonon = _resolve_optional_path(in_dir, _arg_or_cfg(args, cfg, "epr_phonon", None))
    args.phonon_cache = _resolve_optional_path(in_dir, _arg_or_cfg(args, cfg, "phonon_cache", None))
    args.phonon_cache_out = _resolve_optional_path(out_dir, _arg_or_cfg(args, cfg, "phonon_cache_out", None, "hybrid_phonon_cache_out"))
    args.structure = _resolve_optional_path(in_dir, _arg_or_cfg(args, cfg, "structure", None, "structure_file", "POSCAR"))
    args.output = _resolve_optional_path(out_dir, _arg_or_cfg(args, cfg, "output", None, "hybrid_output"))
    args.plot = _resolve_optional_path(out_dir, _arg_or_cfg(args, cfg, "plot", None, "hybrid_plot"))
    args.html_plot = _resolve_optional_path(out_dir, _arg_or_cfg(args, cfg, "html_plot", None, "hybrid_html_plot"))
    args.html_data = _resolve_optional_path(out_dir, _arg_or_cfg(args, cfg, "html_data", None, "hybrid_html_data"))
    args.bare_output = _resolve_optional_path(out_dir, _arg_or_cfg(args, cfg, "bare_output", None, "hybrid_bare_output"))
    args.bare_plot = _resolve_optional_path(out_dir, _arg_or_cfg(args, cfg, "bare_plot", None, "hybrid_bare_plot"))
    if not args.output:
        ap.error("-o/--output or hybrid_output in --input_file is required")

    args.component = _arg_or_cfg(args, cfg, "component", "dmi", "hybrid_component")
    args.static_component = _arg_or_cfg(args, cfg, "static_component", "iso", "hybrid_static_component")
    args.S = float(_arg_or_cfg(args, cfg, "S", 2.5))
    args.spin_direction = _arg_or_cfg(args, cfg, "spin_direction", [0.0, 1.0, 0.0])
    args.spin_pattern = _arg_or_cfg(args, cfg, "spin_pattern", "auto")
    args.phase_convention = _arg_or_cfg(args, cfg, "phase_convention", "cell")
    args.kmesh = _arg_or_cfg(args, cfg, "kmesh", [1, 1, 1])
    args.phonon_qmesh = _arg_or_cfg(args, cfg, "phonon_qmesh", None)
    args.phonon_nproc = max(1, int(_arg_or_cfg(args, cfg, "phonon_nproc", 1)))
    args.hybrid_nproc = max(1, int(_arg_or_cfg(args, cfg, "hybrid_nproc", args.phonon_nproc)))
    args.phonon_cache_compressed = bool(_arg_or_cfg(args, cfg, "phonon_cache_compressed", True))
    args.calculation_mode = _normalize_calculation_mode(_arg_or_cfg(args, cfg, "calculation_mode", "cache"))
    args.qpath = _arg_or_cfg(args, cfg, "qpath", None)
    args.q_points = _arg_or_cfg(args, cfg, "q_points", None)
    args.qpath_label_style = _arg_or_cfg(args, cfg, "qpath_label_style", "symbol")
    args.path_points = int(_arg_or_cfg(args, cfg, "path_points", 80))
    args.plot_unit = _arg_or_cfg(args, cfg, "plot_unit", "THz")
    args.xtick_fontsize = _arg_or_cfg(args, cfg, "xtick_fontsize", None)
    args.color_by = _arg_or_cfg(args, cfg, "color_by", "magnon_weight")
    args.overlay_bare = bool(_arg_or_cfg(args, cfg, "overlay_bare", False))
    args.show = bool(_arg_or_cfg(args, cfg, "show", False))
    args.coupling_scale = float(_arg_or_cfg(args, cfg, "coupling_scale", 1.0))
    args.bond_factor = float(_arg_or_cfg(args, cfg, "bond_factor", 1.0))
    args.anisotropy_mev = float(_arg_or_cfg(args, cfg, "anisotropy_mev", 0.0))
    args.zeeman_field_mev = _arg_or_cfg(args, cfg, "zeeman_field_mev", None)
    args.field_tesla = _arg_or_cfg(args, cfg, "field_tesla", None)
    args.g_factor = float(_arg_or_cfg(args, cfg, "g_factor", 2.0))
    args.phonon_negative_tol_mev = float(_arg_or_cfg(args, cfg, "phonon_negative_tol_mev", 0.0))
    args.gamma_acoustic_zero = int(_arg_or_cfg(args, cfg, "gamma_acoustic_zero", 0))
    args.gamma_zero_tol = float(_arg_or_cfg(args, cfg, "gamma_zero_tol", 1.0e-8))
    args.threshold = float(_arg_or_cfg(args, cfg, "threshold", 0.0))

    lattice0, _, _ = _load_structure(args.structure, fallback_h5=args.J_tensor_h5)
    recip0 = 2.0 * np.pi * np.linalg.inv(lattice0).T
    qpts_frac = None
    kdist = None
    tick_indices = None
    kpath_symbols = None
    if args.q_points:
        vals = [float(x) for x in str(args.q_points).replace(",", " ").split()]
        qpts_frac = np.asarray(vals, dtype=np.float64).reshape(-1, 3)
        print(f"[magph-hybrid] explicit q_points: nq={int(qpts_frac.shape[0])}", flush=True)
    elif args.qpath:
        qpts_frac, kdist, tick_indices, kpath_symbols = generate_fractional_path(
            args.qpath,
            args.path_points,
            recip=recip0,
            label_style=args.qpath_label_style,
        )
        print(f"[magph-hybrid] qpath points: nq={int(qpts_frac.shape[0])}", flush=True)

    components = "dmi,iso,aniso" if args.component == "full" else args.component
    done = _stage_timer(f"load dJ tensor ({components})")
    payload = load_exchange_tensor_realspace_h5(args.dJ_tensor_h5, components=components)
    done()
    print(
        "[magph-hybrid] dJ tensor payload: "
        f"targets={int(payload.target_atoms.size)} bonds={int(payload.bond_i.size)} "
        f"rp={int(payload.rp_grid.shape[0])} disp={len(payload.disp_axes)}",
        flush=True,
    )
    nmag = int(max(int(payload.bond_i.max(initial=0)), int(payload.bond_j.max(initial=0))) + 1)
    spin_info = lswt.build_spin_frame_info(
        nmag=nmag,
        spin_direction=args.spin_direction,
        spin_pattern=args.spin_pattern,
    )
    atom_frac = None
    if args.phase_convention == "basis":
        atom_frac = _load_atom_frac(args.structure, fallback_h5=args.J_tensor_h5)
    done = _stage_timer("site-basis vertex")
    site = build_site_basis_vertex(
        payload,
        spin_info,
        qpts_frac=qpts_frac,
        qmesh=args.phonon_qmesh,
        atom_frac=atom_frac,
        component=args.component,
        S=args.S,
        phase_convention=args.phase_convention,
        threshold=args.threshold,
        verbose=True,
    )
    done()
    if args.phonon_cache and args.calculation_mode == "cache":
        done = _stage_timer("load phonon cache")
        ph_cache = load_phonon_cache(args.phonon_cache)
        done()
    elif args.epr_phonon:
        if args.phonon_cache and args.calculation_mode == "from_scratch":
            print(f"[magph-hybrid] calculation_mode=from_scratch; ignoring phonon_cache={args.phonon_cache}", flush=True)
        ph_cache = build_phonon_cache_from_epr(
            args.epr_phonon,
            qmesh=args.phonon_qmesh,
            qpts_frac=qpts_frac,
            nproc=args.phonon_nproc,
            verbose=True,
        )
        if args.phonon_cache_out:
            _save_phonon_cache(args.phonon_cache_out, ph_cache, compressed=args.phonon_cache_compressed)
            print(f"[magph-hybrid] wrote phonon cache {args.phonon_cache_out}")
    else:
        if args.calculation_mode == "from_scratch":
            raise ValueError("calculation_mode=from_scratch requires epr_phonon to rebuild the phonon cache")
        raise ValueError("Use --phonon_cache or --epr_phonon")
    done = _stage_timer("phonon-mode projection")
    mode = project_site_vertex_to_phonon_modes(site, ph_cache)
    done()
    if args.phonon_negative_tol_mev > 0.0 or args.gamma_acoustic_zero > 0:
        mode = regularize_phonon_energies(
            mode,
            negative_tol_mev=args.phonon_negative_tol_mev,
            gamma_zero_tol=args.gamma_zero_tol,
            gamma_acoustic_modes=args.gamma_acoustic_zero,
        )
    zeeman_field_mev = None
    if args.zeeman_field_mev is not None:
        zeeman_field_mev = np.asarray(args.zeeman_field_mev, dtype=np.float64)
    if args.field_tesla is not None:
        muB_mev_per_t = 5.7883818060e-2
        field_mev_from_t = float(args.g_factor) * muB_mev_per_t * np.asarray(args.field_tesla, dtype=np.float64)
        zeeman_field_mev = field_mev_from_t if zeeman_field_mev is None else zeeman_field_mev + field_mev_from_t
    if zeeman_field_mev is not None:
        print(f"[magph-hybrid] Zeeman field h(meV) = {zeeman_field_mev.tolist()}")
        local_h = np.einsum("ia,a->i", spin_info.frames[:, :, 2], zeeman_field_mev, optimize=True)
        print(f"[magph-hybrid] Zeeman local projection h.e3_i(meV) = {local_h.tolist()}")
    done = _stage_timer("bare magnons and hybrid diagonalization")
    hybrid = build_hybrid_from_static_tensor(
        j_tensor_h5=args.J_tensor_h5,
        structure=args.structure,
        mode_vertex=mode,
        S=args.S,
        spin_info=spin_info,
        kmesh=args.kmesh,
        static_component=args.static_component,
        coupling_scale=args.coupling_scale,
        bond_factor=args.bond_factor,
        anisotropy_mev=args.anisotropy_mev,
        zeeman_field_mev=zeeman_field_mev,
        nproc=args.hybrid_nproc,
    )
    done()
    done = _stage_timer("write hybrid npz")
    if args.qpath:
        save_hybrid_band_npz(args.output, hybrid, kdist=kdist, tick_indices=tick_indices, kpath_symbols=kpath_symbols)
    else:
        save_hybrid_npz(args.output, hybrid)
    done()
    print("[magph-hybrid] summary")
    for key, value in hybrid.shape_summary.items():
        print(f"  {key}: {value}")
    print(f"  eigen_minmax: ({float(np.nanmin(hybrid.eigenvalues))}, {float(np.nanmax(hybrid.eigenvalues))})")
    print(
        "  eigen_minmax_THz: "
        f"({float(np.nanmin(hybrid.eigenvalues) / MEV_PER_THZ)}, "
        f"{float(np.nanmax(hybrid.eigenvalues) / MEV_PER_THZ)})"
    )
    print(
        "  bare_phonon_max_THz: "
        f"{float(np.nanmax(hybrid.phonon_energies) / MEV_PER_THZ)}"
    )
    print(
        "  bare_magnon_max_THz: "
        f"{float(np.nanmax(hybrid.magnon_energies) / MEV_PER_THZ)}"
    )
    print(f"[magph-hybrid] wrote {args.output}")
    if args.html_plot:
        done = _stage_timer("write interactive HTML")
        html_path, html_data_path = save_hybrid_interactive_html(
            args.html_plot,
            hybrid,
            data_path=args.html_data,
            kdist=kdist,
            tick_indices=tick_indices,
            kpath_symbols=kpath_symbols,
            source=args.output,
        )
        print(f"[magph-hybrid] wrote interactive HTML {html_path}")
        print(f"[magph-hybrid] wrote interactive HTML data {html_data_path}")
        done()
    if args.bare_output:
        done = _stage_timer("write bare npz")
        save_bare_npz(args.bare_output, hybrid, kdist=kdist, tick_indices=tick_indices, kpath_symbols=kpath_symbols)
        print(f"[magph-hybrid] wrote bare branches {args.bare_output}")
        done()
    if args.bare_plot:
        done = _stage_timer("write bare plot")
        plot_bare_band(
            hybrid,
            args.bare_plot,
            kdist=kdist,
            tick_indices=tick_indices,
            kpath_symbols=kpath_symbols,
            unit=args.plot_unit,
            xtick_fontsize=args.xtick_fontsize,
            show=bool(args.show),
        )
        done()
    if args.plot:
        done = _stage_timer("write hybrid plot")
        plot_hybrid_band(
            hybrid,
            args.plot,
            kdist=kdist,
            tick_indices=tick_indices,
            kpath_symbols=kpath_symbols,
            unit=args.plot_unit,
            overlay_bare=bool(args.overlay_bare),
            color_by=args.color_by,
            xtick_fontsize=args.xtick_fontsize,
            show=bool(args.show),
        )
        done()


if __name__ == "__main__":
    main()
