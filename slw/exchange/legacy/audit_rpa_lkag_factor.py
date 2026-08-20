# Legacy exchange implementation; use the native engine for new workflows.
"""Compare transverse-RPA poles with LKAG LSWT under explicit J conventions.

This diagnostic intentionally reads only already-generated, small HDF5 files.
It does not rebuild EPR Hamiltonians.  The central comparison is between the
legacy SLW directed-bond scale (bond_factor=1) and the TB2J Hamiltonian
convention, whose i!=j sum contains both directions without a prefactor 1/2
and therefore requires bond_factor=2 in the generic LSWT builder.
"""

from __future__ import annotations

import argparse
import os

import h5py
import numpy as np

from slw.magph.legacy.kernels import Magnon_Hamiltonian_collinear_bdg


def _decode_scalar(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def _read_scalar_j(h5, source):
    key = str(source)
    if key not in h5:
        raise KeyError(f"J source '{key}' is absent from {h5.filename}")
    values = np.asarray(h5[key], dtype=np.float64)
    if values.ndim == 1:
        return values
    if values.ndim == 3 and values.shape[1:] == (3, 3):
        return np.trace(values, axis1=1, axis2=2) / 3.0
    raise ValueError(f"J source '{key}' must be scalar or (...,3,3), got {values.shape}")


def load_audit_inputs(rpa_path, jr_path, j_source):
    with h5py.File(rpa_path, "r") as h5:
        qpoints = np.asarray(h5["basic_data/qpts_frac"], dtype=np.float64)
        moments = np.asarray(h5["basic_data/signed_moments"], dtype=np.float64)
        full_pm = np.asarray(h5["modes/roots_full_pm_ev"], dtype=np.float64) * 1000.0
        full_mp = np.asarray(h5["modes/roots_full_mp_ev"], dtype=np.float64) * 1000.0

    with h5py.File(jr_path, "r") as h5:
        unit = _decode_scalar(h5["basic_data/unit"][()]).strip().lower()
        if unit != "mev":
            raise ValueError(f"J input must use meV, got {unit!r}")
        lattice = np.asarray(h5["basic_data/lattice_ang"], dtype=np.float64)
        atom_pos_all = np.asarray(h5["basic_data/tau_cart_ang"], dtype=np.float64)
        if "bonds/mag_i_local" in h5 and "bonds/mag_j_local" in h5:
            ii = np.asarray(h5["bonds/mag_i_local"], dtype=np.int64)
            jj = np.asarray(h5["bonds/mag_j_local"], dtype=np.int64)
            global_i = np.asarray(h5["bonds/mag_i_atom"], dtype=np.int64)
            global_j = np.asarray(h5["bonds/mag_j_atom"], dtype=np.int64)
            mag_atom_ids = np.unique(np.concatenate((global_i, global_j)))
        else:
            global_i = np.asarray(h5["bonds/mag_i_atom"], dtype=np.int64)
            global_j = np.asarray(h5["bonds/mag_j_atom"], dtype=np.int64)
            mag_atom_ids = np.unique(np.concatenate((global_i, global_j)))
            local = {int(atom): index for index, atom in enumerate(mag_atom_ids)}
            ii = np.asarray([local[int(atom)] for atom in global_i], dtype=np.int64)
            jj = np.asarray([local[int(atom)] for atom in global_j], dtype=np.int64)
        rr = np.asarray(h5["bonds/R"], dtype=np.int64)
        j_mev = _read_scalar_j(h5, j_source)
    atom_pos = atom_pos_all[mag_atom_ids]
    if len(moments) != len(mag_atom_ids):
        raise ValueError(
            f"RPA moments ({len(moments)}) and J magnetic sites "
            f"({len(mag_atom_ids)}) differ"
        )
    return {
        "qpoints": qpoints,
        "moments": moments,
        "full_pm_mev": full_pm,
        "full_mp_mev": full_mp,
        "lattice": lattice,
        "atom_pos": atom_pos,
        "ii": ii,
        "jj": jj,
        "rr": rr,
        "j_mev": j_mev,
    }


def _first_nonnegative_mode(row, zero_tol_mev):
    values = np.asarray(row, dtype=np.float64)
    values = values[np.isfinite(values) & (values >= -float(zero_tol_mev))]
    return max(0.0, float(np.min(values))) if values.size else np.nan


def rpa_chiral_modes(full_pm_mev, full_mp_mev, *, zero_tol_mev=1.0e-5):
    pm = np.asarray(
        [_first_nonnegative_mode(row, zero_tol_mev) for row in full_pm_mev]
    )
    mp = np.asarray(
        [_first_nonnegative_mode(row, zero_tol_mev) for row in full_mp_mev]
    )
    return np.sort(np.column_stack((pm, mp)), axis=1)


def compute_lswt_conventions(data, spin, spin_pattern, bond_factors):
    reciprocal = 2.0 * np.pi * np.linalg.inv(data["lattice"]).T
    kcart = np.asarray(data["qpoints"], dtype=np.float64) @ reciprocal
    j_payload = (data["j_mev"], data["ii"], data["jj"], data["rr"])
    output = {}
    for factor in bond_factors:
        bands = []
        for kpoint in kcart:
            omega, _ = Magnon_Hamiltonian_collinear_bdg(
                kpoint,
                float(spin),
                j_payload,
                data["lattice"],
                data["atom_pos"],
                spin_pattern,
                len(spin_pattern),
                bond_factor=float(factor),
            )
            bands.append(np.sort(np.asarray(omega, dtype=np.float64)))
        output[float(factor)] = np.asarray(bands)
    return output


def matched_ratio(reference, candidate, *, zero_tol_mev):
    ref = np.asarray(reference, dtype=np.float64)
    val = np.asarray(candidate, dtype=np.float64)
    mask = (
        np.isfinite(ref)
        & np.isfinite(val)
        & (np.abs(ref) > float(zero_tol_mev))
        & (np.abs(val) > float(zero_tol_mev))
    )
    return ref[mask] / val[mask]


def write_table(path, qpoints, rpa_modes, lswt_by_factor):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    factors = tuple(lswt_by_factor)
    with open(path, "w", encoding="utf-8") as handle:
        header = ["iq", "q1", "q2", "q3", "rpa_chiral_0_meV", "rpa_chiral_1_meV"]
        for factor in factors:
            header.extend(
                [
                    f"lswt_bond_factor_{factor:g}_mode_0_meV",
                    f"lswt_bond_factor_{factor:g}_mode_1_meV",
                ]
            )
        handle.write("\t".join(header) + "\n")
        for iq, qpoint in enumerate(qpoints):
            fields = [str(iq), *(f"{value:.12e}" for value in qpoint)]
            fields.extend(f"{value:.12e}" for value in rpa_modes[iq])
            for factor in factors:
                fields.extend(f"{value:.12e}" for value in lswt_by_factor[factor][iq])
            handle.write("\t".join(fields) + "\n")


def run(args):
    data = load_audit_inputs(args.rpa, args.jr, args.j_source)
    if str(args.S).strip().lower() == "projected":
        spin = 0.5 * float(np.mean(np.abs(data["moments"])))
    else:
        spin = float(args.S)
    if spin <= 0.0:
        raise ValueError("S must be positive")
    pattern = np.sign(np.asarray(args.spin_pattern, dtype=np.float64))
    if pattern.size != len(data["moments"]) or np.any(pattern == 0.0):
        raise ValueError("--spin_pattern must contain one nonzero sign per magnetic site")
    factors = tuple(float(value) for value in args.bond_factors)
    if not factors or any(value <= 0.0 for value in factors):
        raise ValueError("--bond_factors must contain positive values")

    rpa_modes = rpa_chiral_modes(
        data["full_pm_mev"], data["full_mp_mev"], zero_tol_mev=args.zero_tol_mev
    )
    lswt = compute_lswt_conventions(data, spin, pattern, factors)
    print(
        f"[rpa-lkag-audit] projected_M={np.abs(data['moments']).tolist()} S={spin:.10g}"
    )
    for factor in factors:
        ratio = matched_ratio(
            rpa_modes, lswt[factor], zero_tol_mev=args.zero_tol_mev
        )
        if ratio.size:
            print(
                f"[rpa-lkag-audit] bond_factor={factor:g} "
                f"RPA/LSWT min={np.min(ratio):.8f} mean={np.mean(ratio):.8f} "
                f"max={np.max(ratio):.8f} n={ratio.size}"
            )
    write_table(args.output, data["qpoints"], rpa_modes, lswt)
    print(f"[rpa-lkag-audit] wrote {args.output}")


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Audit factor conventions between RPA poles and LKAG LSWT"
    )
    parser.add_argument("--rpa", required=True, help="RPA diagnostic HDF5")
    parser.add_argument("--jr", required=True, help="LKAG/TB2J-style J HDF5")
    parser.add_argument("--j_source", "--j-source", dest="j_source", default="J_iso_r")
    parser.add_argument(
        "--S",
        default="projected",
        help="Spin magnitude or 'projected' to use mean(abs(M_i))/2 from RPA",
    )
    parser.add_argument(
        "--spin_pattern",
        "--spin-pattern",
        dest="spin_pattern",
        type=float,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--bond_factors",
        "--bond-factors",
        dest="bond_factors",
        type=float,
        nargs="+",
        default=[1.0, 2.0],
        help="1=current generic BdG; 2=TB2J double-counted i!=j Hamiltonian",
    )
    parser.add_argument(
        "--zero_tol_mev",
        "--zero-tol-mev",
        dest="zero_tol_mev",
        type=float,
        default=1.0e-5,
    )
    parser.add_argument("--output", default="rpa_lkag_factor_audit.tsv")
    return parser


def main():
    run(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
