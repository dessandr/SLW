"""Prototype site-basis magnon-phonon vertices from real-space dJ tensors."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import numpy as np

from .adapter import build_phonon_cache_from_epr, load_phonon_cache
from . import lswt
from .tensor_adapter import ExchangeTensorRealSpacePayload, load_exchange_tensor_realspace_h5

HBAR_MEV_PS = 0.6582119569
LEGACY_PHONON_UNIT_CONV = 3.1062


@dataclass
class SiteBasisVertex:
    qpts_frac: np.ndarray
    vertex: np.ndarray
    target_atoms: np.ndarray
    disp_axes: list[str]
    spin_info: lswt.SpinFrameInfo
    component: str
    phase_convention: str
    units: str = "meV/A"

    @property
    def shape_summary(self) -> dict[str, Any]:
        return {
            "nq": int(self.qpts_frac.shape[0]),
            "n_phonon_dof": int(self.vertex.shape[1]),
            "n_nambu": int(self.vertex.shape[2]),
            "nmag": int(self.spin_info.nmag),
            "component": self.component,
            "phase_convention": self.phase_convention,
            "units": self.units,
        }


@dataclass
class ModeBasisVertex:
    qpts_frac: np.ndarray
    phonon_energies: np.ndarray
    phonon_vectors: np.ndarray
    vertex: np.ndarray
    site_vertex: SiteBasisVertex
    phonon_factor: np.ndarray
    units: str = "meV"
    # The cache stores physical displacements u=e/sqrt(M).  Masses are
    # therefore required for norms and phonon angular momentum.  None is kept
    # only so older manually constructed objects fail at the observable that
    # needs the metric, rather than at unrelated vertex operations.
    phonon_masses: np.ndarray | None = None

    @property
    def shape_summary(self) -> dict[str, Any]:
        return {
            "nq": int(self.vertex.shape[0]),
            "n_phonon_modes": int(self.vertex.shape[1]),
            "n_nambu": int(self.vertex.shape[2]),
            "site_vertex_shape": tuple(int(x) for x in self.site_vertex.vertex.shape),
            "phonon_energy_shape": tuple(int(x) for x in self.phonon_energies.shape),
            "units": self.units,
        }


def generate_fractional_mesh(mesh) -> np.ndarray:
    nx, ny, nz = [int(x) for x in mesh]
    axes = [np.arange(n, dtype=np.float64) / float(n) for n in (nx, ny, nz)]
    grid = np.meshgrid(*axes, indexing="ij")
    return np.stack([g.reshape(-1) for g in grid], axis=1)


def component_tensor(payload: ExchangeTensorRealSpacePayload, component: str = "dmi") -> np.ndarray:
    key = str(component).strip().lower()
    if key in {"dmi", "ddmi", "ddmi_r"}:
        return lswt.compose_exchange_tensor(dmi=payload.component("dmi"))
    if key in {"iso", "dj_iso", "dj_iso_r"}:
        return lswt.compose_exchange_tensor(iso=payload.component("iso"))
    if key in {"aniso", "gamma", "dj_gamma", "dj_gamma_r"}:
        return lswt.compose_exchange_tensor(aniso=payload.component("aniso"))
    if key in {"full", "all"}:
        return lswt.compose_exchange_tensor(
            iso=None if payload.dJ_iso is None else payload.dJ_iso,
            aniso=None if payload.dJ_aniso is None else payload.dJ_aniso,
            dmi=None if payload.dDMI is None else payload.dDMI,
        )
    raise ValueError(f"Unknown vertex component {component!r}")


def flatten_tensor_entries(
    payload: ExchangeTensorRealSpacePayload,
    *,
    component: str = "dmi",
    threshold: float = 0.0,
) -> dict[str, np.ndarray]:
    tensor = component_tensor(payload, component)
    n_t, n_b, n_rp, n_disp = tensor.shape[:4]
    target_idx, bond_idx, rp_idx, disp_idx = np.indices((n_t, n_b, n_rp, n_disp), sparse=False)
    values = tensor.reshape(-1, 3, 3)
    norms = np.linalg.norm(values.reshape(values.shape[0], -1), axis=1)
    mask = norms > float(threshold)
    return {
        "target_slot": target_idx.reshape(-1)[mask].astype(np.int32, copy=False),
        "target_atom": payload.target_atoms[target_idx.reshape(-1)[mask]].astype(np.int32, copy=False),
        "bond_index": bond_idx.reshape(-1)[mask].astype(np.int32, copy=False),
        "rp_index": rp_idx.reshape(-1)[mask].astype(np.int32, copy=False),
        "disp_axis_index": disp_idx.reshape(-1)[mask].astype(np.int32, copy=False),
        "i_atom": payload.bond_i[bond_idx.reshape(-1)[mask]].astype(np.int32, copy=False),
        "j_atom": payload.bond_j[bond_idx.reshape(-1)[mask]].astype(np.int32, copy=False),
        "R": payload.bond_R[bond_idx.reshape(-1)[mask]].astype(np.int32, copy=False),
        "Rp": payload.rp_grid[rp_idx.reshape(-1)[mask]].astype(np.int32, copy=False),
        "tensor": values[mask].astype(np.float64, copy=False),
    }


def _phase_arguments(entries: dict[str, np.ndarray], atom_frac, *, phase_convention: str) -> tuple[np.ndarray, np.ndarray]:
    convention = str(phase_convention).strip().lower()
    rp = np.asarray(entries["Rp"], dtype=np.float64)
    rr = np.asarray(entries["R"], dtype=np.float64)
    if convention in {"cell", "cell-only", "cell_only"}:
        return rp, rp - rr
    if convention in {"basis", "basis-position", "basis_position"}:
        if atom_frac is None:
            raise ValueError("atom_frac/structure is required for basis-position phase convention")
        tau = np.asarray(atom_frac, dtype=np.float64)
        target = np.asarray(entries["target_atom"], dtype=np.int32)
        ii = np.asarray(entries["i_atom"], dtype=np.int32)
        jj = np.asarray(entries["j_atom"], dtype=np.int32)
        return rp + tau[target] - tau[ii], rp + tau[target] - rr - tau[jj]
    raise ValueError(f"Unknown phase convention {phase_convention!r}")


def _compressed_vertex_terms(entries, coeff, dof, arg_i, arg_j, *, n_ph, n_nambu):
    nmag = n_nambu // 2
    channel_i = dof * n_nambu + entries["i_atom"]
    channel_ci = dof * n_nambu + nmag + entries["i_atom"]
    channel_j = dof * n_nambu + entries["j_atom"]
    channel_cj = dof * n_nambu + nmag + entries["j_atom"]

    args = np.concatenate([arg_i, arg_i, arg_j, arg_j], axis=0)
    channels = np.concatenate([channel_i, channel_ci, channel_j, channel_cj]).astype(np.int64, copy=False)
    values = np.concatenate(
        [
            coeff["annihilate_i"],
            coeff["create_i"],
            coeff["annihilate_j"],
            coeff["create_j"],
        ]
    ).astype(np.complex128, copy=False)

    uniq_args, inv_arg = np.unique(np.asarray(args, dtype=np.float64), axis=0, return_inverse=True)
    n_channel = int(n_ph) * int(n_nambu)
    flat = inv_arg.astype(np.int64, copy=False) * n_channel + channels
    uniq_flat, inv_flat = np.unique(flat, return_inverse=True)
    summed = np.zeros(uniq_flat.shape[0], dtype=np.complex128)
    np.add.at(summed, inv_flat, values)

    arg_index = uniq_flat // n_channel
    channel_index = uniq_flat % n_channel
    coeff_matrix = np.zeros((uniq_args.shape[0], n_channel), dtype=np.complex128)
    coeff_matrix[arg_index, channel_index] = summed
    return uniq_args, coeff_matrix.reshape(uniq_args.shape[0], int(n_ph), int(n_nambu))


def build_site_basis_vertex(
    payload: ExchangeTensorRealSpacePayload,
    spin_info: lswt.SpinFrameInfo,
    *,
    qpts_frac=None,
    qmesh=None,
    atom_frac=None,
    component: str = "dmi",
    S: float = 2.5,
    phase_convention: str = "cell",
    threshold: float = 0.0,
    chunk_q: int = 128,
    verbose: bool = False,
) -> SiteBasisVertex:
    """Build `V(q, phonon_dof, magnon_nambu)` in the site/Nambu basis."""
    if qpts_frac is None:
        qpts_frac = generate_fractional_mesh(payload.qmesh if qmesh is None else qmesh)
    qpts_frac = np.asarray(qpts_frac, dtype=np.float64).reshape(-1, 3)
    nmag = int(spin_info.nmag)
    n_disp = int(len(payload.disp_axes))
    n_ph = int(payload.target_atoms.size) * n_disp
    entries = flatten_tensor_entries(payload, component=component, threshold=threshold)
    if verbose:
        print(
            "[magph-vertex] site-basis entries: "
            f"nq={int(qpts_frac.shape[0])} n_entries={int(entries['tensor'].shape[0])} "
            f"n_ph={n_ph} n_nambu={2 * nmag} threshold={float(threshold):g}",
            flush=True,
        )
    coeff = lswt.linear_hp_coefficients_from_tensor(
        entries["tensor"],
        entries["i_atom"],
        entries["j_atom"],
        spin_info.frames,
        S,
    )
    dof = entries["target_slot"] * n_disp + entries["disp_axis_index"]
    arg_i, arg_j = _phase_arguments(entries, atom_frac, phase_convention=phase_convention)
    term_args, term_coeff = _compressed_vertex_terms(
        entries,
        coeff,
        dof,
        arg_i,
        arg_j,
        n_ph=n_ph,
        n_nambu=2 * nmag,
    )
    if verbose:
        print(
            "[magph-vertex] compressed terms: "
            f"n_phase={int(term_args.shape[0])} n_channels={n_ph * 2 * nmag}",
            flush=True,
        )

    vertex = np.zeros((qpts_frac.shape[0], n_ph, 2 * nmag), dtype=np.complex128)
    chunk = max(1, int(chunk_q))
    for start in range(0, qpts_frac.shape[0], chunk):
        stop = min(start + chunk, qpts_frac.shape[0])
        q = qpts_frac[start:stop]
        phase = np.exp(2.0j * np.pi * (q @ term_args.T))
        vertex[start:stop] = np.einsum("qu,upn->qpn", phase, term_coeff, optimize=True)

    return SiteBasisVertex(
        qpts_frac=qpts_frac,
        vertex=vertex,
        target_atoms=payload.target_atoms,
        disp_axes=list(payload.disp_axes),
        spin_info=spin_info,
        component=str(component),
        phase_convention=str(phase_convention),
        units=payload.units,
    )


def _qpts_match(a: np.ndarray, b: np.ndarray, tol: float = 1.0e-8) -> bool:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if aa.shape != bb.shape:
        return False
    diff = ((aa - bb + 0.5) % 1.0) - 0.5
    return bool(np.max(np.abs(diff)) <= float(tol))


def project_site_vertex_to_phonon_modes(
    site: SiteBasisVertex,
    phonon_cache: dict[str, np.ndarray],
    *,
    phonon_eta_mev: float = 1.0e-3,
    include_quantum_factor: bool = True,
    unit_conv: float = LEGACY_PHONON_UNIT_CONV,
    qpt_tol: float = 1.0e-8,
) -> ModeBasisVertex:
    """Contract site-basis vertex with phonon eigenvectors.

    `site.vertex` has shape `(nq, target_dof, n_nambu)`.
    `ph_vec_flat` has shape `(nq, n_mode, nat, 3)`.
    The result has shape `(nq, n_mode, n_nambu)`.
    """
    q_ph = np.asarray(phonon_cache["q_mesh_flat_frac"], dtype=np.float64)
    ph_en = np.asarray(phonon_cache["ph_en_flat"], dtype=np.float64)
    ph_vec = np.asarray(phonon_cache["ph_vec_flat"], dtype=np.complex128)
    phonon_masses = (
        None
        if "atom_mass_electron" not in phonon_cache
        else np.asarray(phonon_cache["atom_mass_electron"], dtype=np.float64)
    )
    if not _qpts_match(site.qpts_frac, q_ph, tol=qpt_tol):
        raise ValueError(
            "site vertex q-points do not match phonon cache q-points. "
            f"site={site.qpts_frac.shape}, phonon={q_ph.shape}"
        )
    if ph_vec.ndim != 4 or ph_vec.shape[0] != site.vertex.shape[0]:
        raise ValueError(f"Unsupported ph_vec_flat shape {ph_vec.shape} for site vertex {site.vertex.shape}")
    if phonon_masses is not None and phonon_masses.shape != (ph_vec.shape[2],):
        raise ValueError(
            "atom_mass_electron shape is incompatible with phonon vectors: "
            f"{phonon_masses.shape} != {(ph_vec.shape[2],)}"
        )
    target = np.asarray(site.target_atoms, dtype=np.int32)
    if int(target.max(initial=0)) >= ph_vec.shape[2]:
        raise ValueError(f"target atom index exceeds phonon nat: max={int(target.max())} nat={ph_vec.shape[2]}")
    pol = ph_vec[:, :, target, :].reshape(ph_vec.shape[0], ph_vec.shape[1], -1)
    if pol.shape[2] != site.vertex.shape[1]:
        raise ValueError(f"phonon dof mismatch: pol={pol.shape}, site={site.vertex.shape}")
    if include_quantum_factor:
        factor = (HBAR_MEV_PS / np.sqrt(2.0 * np.maximum(ph_en, 0.0) + float(phonon_eta_mev))) * float(unit_conv)
    else:
        factor = np.ones_like(ph_en, dtype=np.float64)
    eps = pol * factor[:, :, None]
    mode_vertex = np.einsum("qdn,qmd->qmn", site.vertex, eps, optimize=True)
    return ModeBasisVertex(
        qpts_frac=site.qpts_frac,
        phonon_energies=ph_en,
        phonon_vectors=ph_vec,
        vertex=mode_vertex,
        site_vertex=site,
        phonon_factor=factor,
        phonon_masses=phonon_masses,
    )


def _load_atom_frac_from_structure(path: str) -> np.ndarray:
    from .utils import parsing_POSCAR

    lattice, _, atom_pos = parsing_POSCAR(path)
    lattice = np.asarray(lattice, dtype=np.float64)
    atom_pos = np.asarray(atom_pos, dtype=np.float64)
    return (atom_pos @ np.linalg.inv(lattice)) % 1.0


def save_site_basis_vertex_npz(path: str, data: SiteBasisVertex):
    np.savez_compressed(
        path,
        qpts_frac=data.qpts_frac,
        vertex=data.vertex,
        target_atoms=data.target_atoms,
        disp_axes=np.asarray(data.disp_axes, dtype=object),
        spin_directions=data.spin_info.spin_directions,
        spin_pattern=data.spin_info.spin_pattern,
        frames=data.spin_info.frames,
        eta=data.spin_info.eta,
        component=np.asarray(data.component, dtype=object),
        phase_convention=np.asarray(data.phase_convention, dtype=object),
        units=np.asarray(data.units, dtype=object),
    )


def save_mode_basis_vertex_npz(path: str, data: ModeBasisVertex):
    payload = dict(
        qpts_frac=data.qpts_frac,
        phonon_energies=data.phonon_energies,
        vertex=data.vertex,
        site_vertex=data.site_vertex.vertex,
        target_atoms=data.site_vertex.target_atoms,
        disp_axes=np.asarray(data.site_vertex.disp_axes, dtype=object),
        spin_directions=data.site_vertex.spin_info.spin_directions,
        spin_pattern=data.site_vertex.spin_info.spin_pattern,
        frames=data.site_vertex.spin_info.frames,
        eta=data.site_vertex.spin_info.eta,
        phonon_factor=data.phonon_factor,
        component=np.asarray(data.site_vertex.component, dtype=object),
        phase_convention=np.asarray(data.site_vertex.phase_convention, dtype=object),
        units=np.asarray(data.units, dtype=object),
    )
    if data.phonon_masses is not None:
        payload["atom_mass_electron"] = np.asarray(
            data.phonon_masses, dtype=np.float64
        )
    np.savez_compressed(path, **payload)


def main():
    ap = argparse.ArgumentParser(description="Build prototype site-basis magnon-phonon vertex from dJ tensor HDF5")
    ap.add_argument("--dJ_tensor_h5", default=None, help="dynamic dJ/du tensor HDF5 from compute_dJ_epr_tensor")
    ap.add_argument("--tensor_h5", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--component", default="dmi", choices=["dmi", "iso", "aniso", "full"])
    ap.add_argument("--S", type=float, default=2.5)
    ap.add_argument("--spin_direction", type=float, nargs=3, default=[0.0, 1.0, 0.0])
    ap.add_argument("--spin_pattern", default="auto")
    ap.add_argument("--qmesh", type=int, nargs=3, default=None)
    ap.add_argument("--phase_convention", choices=["cell", "basis"], default="cell")
    ap.add_argument("--structure", default=None, help="POSCAR/QE-style input file for basis-position phases")
    ap.add_argument("--phonon_cache", default=None, help="Phonon cache npz with ph_en_flat/ph_vec_flat")
    ap.add_argument("--epr_phonon", default=None, help="Build phonon cache directly from EPR HDF5")
    ap.add_argument("--phonon_qmesh", type=int, nargs=3, default=None, help="q mesh for --epr_phonon")
    ap.add_argument("--no_phonon_factor", action="store_true", help="Do not multiply HBAR/sqrt(2w) phonon factor")
    ap.add_argument("--phonon_eta_mev", type=float, default=1.0e-3)
    ap.add_argument("--threshold", type=float, default=0.0)
    ap.add_argument("--chunk_q", type=int, default=128)
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--site_output", default=None, help="Optional output for the pre-phonon site-basis vertex")
    ap.add_argument("--show", type=int, default=3)
    args = ap.parse_args()
    args.dJ_tensor_h5 = args.dJ_tensor_h5 or args.tensor_h5
    if not args.dJ_tensor_h5:
        ap.error("one of --dJ_tensor_h5 or legacy --tensor_h5 is required")

    components = "dmi,iso,aniso" if args.component == "full" else args.component
    payload = load_exchange_tensor_realspace_h5(args.dJ_tensor_h5, components=components)
    nmag = int(max(int(payload.bond_i.max(initial=0)), int(payload.bond_j.max(initial=0))) + 1)
    spin_info = lswt.build_spin_frame_info(
        nmag=nmag,
        spin_direction=args.spin_direction,
        spin_pattern=args.spin_pattern,
    )
    atom_frac = None
    if args.phase_convention == "basis":
        if not args.structure:
            raise ValueError("--structure is required for --phase_convention basis")
        atom_frac = _load_atom_frac_from_structure(args.structure)

    data = build_site_basis_vertex(
        payload,
        spin_info,
        qmesh=args.qmesh,
        atom_frac=atom_frac,
        component=args.component,
        S=args.S,
        phase_convention=args.phase_convention,
        threshold=args.threshold,
        chunk_q=args.chunk_q,
    )
    print("[magph-vertex] summary")
    for key, value in data.shape_summary.items():
        print(f"  {key}: {value}")
    norms = np.linalg.norm(data.vertex.reshape(data.vertex.shape[0], -1), axis=1)
    print(f"  max_q_norm: {float(norms.max()) if norms.size else 0.0}")
    print(f"  mean_q_norm: {float(norms.mean()) if norms.size else 0.0}")
    if int(args.show) > 0:
        flat = np.abs(data.vertex).reshape(-1)
        order = np.argsort(flat)[::-1][: int(args.show)]
        nq, nph, nn = data.vertex.shape
        for idx in order:
            iq = idx // (nph * nn)
            rem = idx % (nph * nn)
            iph = rem // nn
            inu = rem % nn
            print(
                f"  entry q={iq} qfrac={data.qpts_frac[iq].tolist()} "
                f"phonon_dof={iph} nambu={inu} value={data.vertex[iq, iph, inu]}"
            )
    if args.site_output:
        save_site_basis_vertex_npz(args.site_output, data)
        print(f"[magph-vertex] wrote site vertex {args.site_output}")

    mode_data = None
    if args.phonon_cache or args.epr_phonon:
        if args.phonon_cache:
            ph_cache = load_phonon_cache(args.phonon_cache)
        else:
            ph_cache = build_phonon_cache_from_epr(args.epr_phonon, qmesh=args.phonon_qmesh or args.qmesh)
        mode_data = project_site_vertex_to_phonon_modes(
            data,
            ph_cache,
            phonon_eta_mev=args.phonon_eta_mev,
            include_quantum_factor=not bool(args.no_phonon_factor),
        )
        print("[magph-vertex] phonon-mode summary")
        for key, value in mode_data.shape_summary.items():
            print(f"  {key}: {value}")
        mode_norms = np.linalg.norm(mode_data.vertex.reshape(mode_data.vertex.shape[0], -1), axis=1)
        print(f"  max_q_mode_norm: {float(mode_norms.max()) if mode_norms.size else 0.0}")
        print(f"  mean_q_mode_norm: {float(mode_norms.mean()) if mode_norms.size else 0.0}")

    if args.output:
        if mode_data is None:
            save_site_basis_vertex_npz(args.output, data)
        else:
            save_mode_basis_vertex_npz(args.output, mode_data)
        print(f"[magph-vertex] wrote {args.output}")


if __name__ == "__main__":
    main()
