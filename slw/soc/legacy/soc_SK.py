"""Fit PRB-style interatomic Slater-Koster SOC templates.

This module reuses the I/O and least-squares fitting machinery from
``slw.soc.legacy.nonlocal_fit`` but replaces the nonlocal kernel with the
two-center interatomic SOC form

    H_SOC(R) = i lambda_SK(R) . sigma

from Kato and Ogata, Phys. Rev. B 111, 035137 (2025).  The first implemented
kernel is the explicit p-p table from the main text.  The code is organized so
additional s-p, p-d, d-d tables can be added as new template builders without
changing the fitting path.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from slw.core.cli_paths import resolve_path, resolve_workdir
from slw.soc.legacy.band_splitting import (
    compute_alignment_shift,
    parse_band_selection,
    read_kpoints,
)

from slw.soc.legacy.nonlocal_fit import (
    block_atom_fractional,
    block_orbital_index,
    build_lambda_bounds,
    build_ls_operator,
    build_static_onsite_term,
    choose_unique_candidate,
    clip_initial_to_bounds,
    find_hr_seed_sidecar,
    find_seeded_sidecar,
    hamiltonian_k,
    hr_seed_from_path,
    infer_blocks_from_win,
    infer_seed_from_paths,
    k_phases,
    load_hr,
    local_spin_moments,
    make_shifted_kmesh,
    model_hk,
    moment_blocks_from_selector,
    operator_name,
    parse_blocks,
    parse_moment_components,
    prepare_terms_hk,
    read_atoms_from_win,
    read_unit_cell_cart_from_win,
    real_harmonic_rotation,
    bond_frame,
    require_spglib_symmetry,
    ridge_initial_guess,
    selected_residual,
    selected_residual_kparallel,
    species_name,
    write_fit_summary_csv,
    write_fitted_hr,
    write_lambda_csv,
)


PAULI_X = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
PAULI_Y = np.asarray([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
PAULI_Z = np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
PAULI = np.stack([PAULI_X, PAULI_Y, PAULI_Z], axis=0)

AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
EPS = np.zeros((3, 3, 3), dtype=np.float64)
EPS[0, 1, 2] = EPS[1, 2, 0] = EPS[2, 0, 1] = 1.0
EPS[1, 0, 2] = EPS[2, 1, 0] = EPS[0, 2, 1] = -1.0
AXIS_TESSERAL_MU = [1, -1, 0]  # Cartesian x,y,z in the paper's p tesseral labels.
W90_TESSERAL_MU = {
    1: [0, 1, -1],       # pz, px, py
    2: [0, 1, -1, 2, -2],  # dz2, dxz, dyz, dx2-y2, dxy
}

SK_SYMBOL_SPECS = {
    "pd": {
        "sym": {
            "Kpd_sigma": (1, 0, 0, -1),
            "Kpd_pi": (1, 1, 0, -2),
            "Kpd_pi0": (1, -1, 0, 0),
            "Kpd_pi_prime": (1, 1, -1, -1),
        },
        "asym": {
            "Ktilde_pd_sigma": (1, 0, 0, -1),
            "Ktilde_pd_pi": (1, 1, 0, -2),
            "Ktilde_pd_pi0": (1, -1, 0, 0),
            "Ktilde_pd_pi_prime": (1, 1, -1, -1),
        },
    },
    "dd": {
        "sym": {
            "Kdd_sigma": (1, 0, 0, -1),
            "Kdd_pi": (1, 1, 0, -2),
            "Kdd_pi_prime": (1, 1, -1, -1),
            "Kdd_delta_prime": (1, 2, -1, -2),
        },
        "asym": {
            "Ktilde_dd_sigma": (1, 0, 0, -1),
            "Ktilde_dd_pi": (1, 1, 0, -2),
        },
    },
}


@dataclass(frozen=True)
class SKBond:
    ir: int
    ir_neg: int
    row_block: dict
    col_block: dict
    row_block_id: int
    col_block_id: int
    unit: np.ndarray
    distance_ang: float
    pair_key: tuple[str, str]
    pair_type: str


def _parse_p_order(text: str) -> list[int]:
    order = [x.strip().lower().replace("p", "") for x in str(text).replace(";", ",").split(",") if x.strip()]
    if sorted(order) != ["x", "y", "z"]:
        raise ValueError(f"p orbital order must be a permutation of px,py,pz; got {text!r}")
    return [AXIS_INDEX[x] for x in order]


def _canonical_pair_key(a: str, b: str) -> tuple[str, str]:
    aa = species_name(a)
    bb = species_name(b)
    return tuple(sorted((aa, bb)))


def _canonical_bond_key(ir: int, row_id: int, col_id: int) -> tuple[int, int, int]:
    return (int(ir), int(row_id), int(col_id))


def _spin_block_from_lambda(lambda_vec: np.ndarray) -> np.ndarray:
    return 1.0j * np.einsum("a,aij->ij", np.asarray(lambda_vec, dtype=np.complex128), PAULI, optimize=True)


def _tesseral_c(l_val: int, mu: int, m: int) -> complex:
    if abs(mu) > int(l_val) or abs(m) > int(l_val):
        return 0.0
    if mu > 0:
        if m == mu:
            return ((-1) ** mu) / np.sqrt(2.0)
        if m == -mu:
            return 1.0 / np.sqrt(2.0)
    elif mu == 0:
        if m == 0:
            return 1.0
    else:
        if m == mu:
            return 1.0j / np.sqrt(2.0)
        if m == -mu:
            return -((-1) ** mu) * 1.0j / np.sqrt(2.0)
    return 0.0


def _symbol_closure(seed: dict[tuple[int, int, int, int], float], *, same_l: bool, inversion_parity: float) -> dict[tuple[int, int, int, int], float]:
    out: dict[tuple[int, int, int, int], float] = {}
    stack = list(seed.items())
    while stack:
        key, value = stack.pop()
        old = out.get(key)
        if old is not None:
            continue
        out[key] = float(value)
        j, n, jp, np_ = key
        transforms = [
            ((jp, n, j, np_), -value),          # derivative exchange in Eq. (27)
            ((-j, -n, -jp, -np_), value),       # complex conjugation
        ]
        if same_l:
            transforms.append(((j, np_, jp, n), inversion_parity * value))
        for new_key, new_value in transforms:
            if new_key not in out:
                stack.append((new_key, float(new_value)))
    return out


def _local_symbol_tensor(lrow: int, lcol: int, symbol_key: tuple[int, int, int, int], *, symmetric: bool) -> np.ndarray:
    row_mus = W90_TESSERAL_MU[int(lrow)]
    col_mus = W90_TESSERAL_MU[int(lcol)]
    seed_sign = -1.0
    same_l_exchange = int(lrow) == int(lcol) and int(symbol_key[2]) == 0
    symbols = _symbol_closure(
        {tuple(symbol_key): seed_sign},
        same_l=same_l_exchange,
        inversion_parity=1.0 if symmetric else -1.0,
    )
    out = np.zeros((len(row_mus), len(col_mus), 3), dtype=np.complex128)
    for irow, mu in enumerate(row_mus):
        for icol, mup in enumerate(col_mus):
            for tau in range(3):
                total = 0.0j
                for nu in range(3):
                    for nup in range(3):
                        eps = EPS[nu, nup, tau]
                        if eps == 0.0:
                            continue
                        for m in range(-int(lrow), int(lrow) + 1):
                            c_mu = _tesseral_c(lrow, mu, m)
                            if c_mu == 0.0:
                                continue
                            for mp in range(-int(lcol), int(lcol) + 1):
                                c_mup = _tesseral_c(lcol, mup, mp)
                                if c_mup == 0.0:
                                    continue
                                for deriv in (-1, 0, 1):
                                    c_nu = _tesseral_c(1, AXIS_TESSERAL_MU[nu], deriv)
                                    if c_nu == 0.0:
                                        continue
                                    for deriv_p in (-1, 0, 1):
                                        c_nup = _tesseral_c(1, AXIS_TESSERAL_MU[nup], deriv_p)
                                        if c_nup == 0.0:
                                            continue
                                        total += (
                                            0.5
                                            * eps
                                            * c_mu
                                            * c_mup
                                            * c_nu
                                            * c_nup
                                            * symbols.get((deriv, m, deriv_p, mp), 0.0)
                                        )
                out[irow, icol, tau] = total
    return np.real_if_close(out, tol=1000).astype(np.complex128)


def _rotate_local_lambda_tensor(local_tensor: np.ndarray, lrow: int, lcol: int, unit: np.ndarray) -> np.ndarray:
    frame = bond_frame(unit)
    row_rot = real_harmonic_rotation(lrow, frame)
    col_rot = real_harmonic_rotation(lcol, frame)
    return np.einsum("ia,jb,abv,tv->ijt", row_rot, col_rot, local_tensor, frame, optimize=True)


def _pp_lambda_templates(mu: int, mup: int, unit: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return p-p lambda vectors for Kpp_sigma, Kpp_pi_prime, Ktilde_pp_sigma."""
    rhat = np.asarray(unit, dtype=np.float64).reshape(3)
    eps_vec = EPS[mu, mup].copy()
    angular = np.einsum("n,nt->t", eps_vec * rhat, np.outer(np.ones(3), rhat), optimize=True)
    # Equivalent to angular[t] = sum_n eps[mu,mup,n] R_n R_t.
    sigma = eps_vec - angular
    pi_prime = -angular
    asym = np.zeros(3, dtype=np.float64)
    for tau in range(3):
        asym[tau] = np.sum(
            rhat
            * (
                EPS[mu, tau] * rhat[mup]
                + EPS[mup, tau] * rhat[mu]
            )
        )
    return sigma, pi_prime, asym


def _add_spinful_pair(matrix: np.ndarray, row_block: dict, col_block: dict, mu_i: int, mu_j: int, spin_block: np.ndarray, dim: int, block_layout: str) -> None:
    for srow in range(2):
        row = block_orbital_index(row_block, mu_i, srow, dim, block_layout)
        for scol in range(2):
            val = spin_block[srow, scol]
            if abs(val) > 1.0e-14:
                col = block_orbital_index(col_block, mu_j, scol, dim, block_layout)
                matrix[row, col] += val


def _add_pp_soc_bond_terms(
    term_mats: dict[str, np.ndarray],
    bond: SKBond,
    dim: int,
    block_layout: str,
    p_axes: list[int],
    include_asym: bool,
) -> None:
    for iorb, mu in enumerate(p_axes):
        for jorb, mup in enumerate(p_axes):
            lam_sigma, lam_pi, lam_asym = _pp_lambda_templates(mu, mup, bond.unit)
            for name, lam in (("Kpp_sigma", lam_sigma), ("Kpp_pi_prime", lam_pi)):
                block = _spin_block_from_lambda(lam)
                _add_spinful_pair(term_mats[name][bond.ir], bond.row_block, bond.col_block, iorb, jorb, block, dim, block_layout)
                _add_spinful_pair(
                    term_mats[name][bond.ir_neg],
                    bond.col_block,
                    bond.row_block,
                    jorb,
                    iorb,
                    block.conj().T,
                    dim,
                    block_layout,
                )
            if include_asym:
                block = _spin_block_from_lambda(lam_asym)
                _add_spinful_pair(term_mats["Ktilde_pp_sigma"][bond.ir], bond.row_block, bond.col_block, iorb, jorb, block, dim, block_layout)
                _add_spinful_pair(
                    term_mats["Ktilde_pp_sigma"][bond.ir_neg],
                    bond.col_block,
                    bond.row_block,
                    jorb,
                    iorb,
                    block.conj().T,
                    dim,
                    block_layout,
                )


def _group_bonds_by_shell(bonds: list[SKBond], tol: float) -> list[list[SKBond]]:
    groups: list[list[SKBond]] = []
    for bond in sorted(bonds, key=lambda b: (b.pair_key, b.distance_ang, b.ir, b.row_block_id, b.col_block_id)):
        for group in groups:
            ref = group[0]
            if ref.pair_key == bond.pair_key and abs(ref.distance_ang - bond.distance_ang) <= tol:
                group.append(bond)
                break
        else:
            groups.append([bond])
    groups.sort(key=lambda g: (float(np.mean([b.distance_ang for b in g])), g[0].pair_key))
    return groups


def _group_bonds_by_orbit(bonds: list[SKBond], rotations: np.ndarray, cell: np.ndarray, tol: float) -> list[list[SKBond]]:
    raw = [
        {
            "bond_frac": np.asarray(b.unit * b.distance_ang) @ np.linalg.inv(np.asarray(cell, dtype=float)),
            "distance_ang": b.distance_ang,
            "pair_key": b.pair_key,
            "index": i,
        }
        for i, b in enumerate(bonds)
    ]
    unused = set(range(len(raw)))
    groups: list[list[SKBond]] = []
    while unused:
        seed_idx = min(unused)
        seed = raw[seed_idx]
        targets = [np.asarray(seed["bond_frac"] @ rot.T, dtype=float) for rot in rotations]
        orbit = []
        for idx in list(unused):
            cand = raw[idx]
            if cand["pair_key"] != seed["pair_key"]:
                continue
            diffs = np.asarray([(cand["bond_frac"] - target - np.rint(cand["bond_frac"] - target)) @ cell for target in targets])
            if float(np.min(np.linalg.norm(diffs, axis=1))) <= tol:
                orbit.append(idx)
        for idx in orbit:
            unused.remove(idx)
        groups.append([bonds[raw[idx]["index"]] for idx in orbit])
    groups.sort(key=lambda g: (float(np.mean([b.distance_ang for b in g])), g[0].pair_key))
    return groups


def build_prb_pp_sk_terms(
    coll_hr,
    blocks: list[dict],
    block_layout: str,
    win_path: str,
    *,
    p_order: str = "pz,px,py",
    num_shells: int = 1,
    shell_tol: float = 1.0e-2,
    grouping: str = "shell",
    include_same_species_asym: bool = False,
    strict_pairs: bool = False,
) -> list[dict]:
    """Build p-p interatomic SOC terms from PRB 111, 035137 Eq. (51),(58)."""
    p_blocks = [dict(b, _block_id=i) for i, b in enumerate(blocks) if int(b["l"]) == 1]
    unsupported = [b for b in blocks if int(b["l"]) not in {1, 2}]
    if strict_pairs and unsupported:
        labels = [f"{b.get('atom_label', '')}:l{b['l']}" for b in unsupported]
        raise NotImplementedError(f"soc_SK currently implements PRB p-p, p-d, d-d only; unsupported blocks={labels}")
    if len(p_blocks) < 1:
        raise ValueError("PRB p-p SK SOC needs at least one p block")

    p_axes = _parse_p_order(p_order)
    atom_frac_by_label = {
        label: np.asarray(coord, dtype=float)
        for label, coord in zip(*read_atoms_from_win(win_path)[:2])
    }

    cell = read_unit_cell_cart_from_win(win_path)
    r_to_ir = {tuple(r): ir for ir, r in enumerate(coll_hr.r_keys)}
    bonds: list[SKBond] = []
    for ir, r in enumerate(coll_hr.r_keys):
        r_tuple = tuple(int(x) for x in r)
        r_neg = tuple(-int(x) for x in r_tuple)
        if r_neg not in r_to_ir:
            continue
        ir_neg = r_to_ir[r_neg]
        for row_block in p_blocks:
            row_id = int(row_block["_block_id"])
            row_frac = block_atom_fractional(row_block, atom_frac_by_label, win_path)
            row_label = str(row_block.get("atom_label", f"p{row_id + 1}"))
            for col_block in p_blocks:
                col_id = int(col_block["_block_id"])
                partner_key = _canonical_bond_key(ir_neg, col_id, row_id)
                this_key = _canonical_bond_key(ir, row_id, col_id)
                if this_key > partner_key:
                    continue
                col_frac = block_atom_fractional(col_block, atom_frac_by_label, win_path)
                col_label = str(col_block.get("atom_label", f"p{col_id + 1}"))
                bond_frac = np.asarray(r, dtype=float) + col_frac - row_frac
                bond_cart = bond_frac @ cell
                distance = float(np.linalg.norm(bond_cart))
                if distance <= float(shell_tol):
                    continue
                bonds.append(
                    SKBond(
                        ir=ir,
                        ir_neg=ir_neg,
                        row_block=row_block,
                        col_block=col_block,
                        row_block_id=row_id,
                        col_block_id=col_id,
                        unit=bond_cart / distance,
                        distance_ang=distance,
                        pair_key=_canonical_pair_key(row_label, col_label),
                        pair_type="pp",
                    )
                )
    if not bonds:
        raise ValueError("No p-p interatomic bonds were generated for PRB SK SOC")

    grouping = str(grouping).strip().lower()
    if grouping == "shell":
        groups = _group_bonds_by_shell(bonds, shell_tol)
    elif grouping == "orbit":
        sym_data = require_spglib_symmetry(win_path, shell_tol)
        groups = _group_bonds_by_orbit(bonds, sym_data["rotations"], cell, shell_tol)
    else:
        raise ValueError(f"Unknown --sk-grouping {grouping!r}; use shell or orbit")
    if int(num_shells) > 0:
        selected = []
        counts: dict[tuple[str, str], int] = {}
        for group in groups:
            key = group[0].pair_key
            used = int(counts.get(key, 0))
            if used >= int(num_shells):
                continue
            selected.append(group)
            counts[key] = used + 1
        groups = selected

    terms = []
    for igroup, group in enumerate(groups):
        pair_species = group[0].pair_key
        same_species = pair_species[0] == pair_species[1]
        include_asym = include_same_species_asym or not same_species
        mats = {
            "Kpp_sigma": np.zeros_like(coll_hr.h_norm),
            "Kpp_pi_prime": np.zeros_like(coll_hr.h_norm),
        }
        if include_asym:
            mats["Ktilde_pp_sigma"] = np.zeros_like(coll_hr.h_norm)
        for bond in group:
            _add_pp_soc_bond_terms(mats, bond, coll_hr.dim, block_layout, p_axes, include_asym)
        distances = np.asarray([b.distance_ang for b in group], dtype=float)
        meta = {
            "orbit": igroup,
            "grouping": grouping,
            "num_bonds": len(group),
            "distance_mean_ang": float(np.mean(distances)),
            "distance_std_ang": float(np.std(distances)),
            "distance_min_ang": float(np.min(distances)),
            "distance_max_ang": float(np.max(distances)),
            "pair_keys": "-".join(pair_species),
            "sk_soc_form": "PRB111_035137_pp_Eq51_Eq58",
            "p_order": p_order,
        }
        for pname, r_mats in mats.items():
            norm = float(np.linalg.norm(r_mats.reshape(r_mats.shape[0], -1)))
            if norm <= 1.0e-14:
                continue
            terms.append(
                {
                    "name": f"prb_pp_{pname}_{grouping}_{igroup}",
                    "operator": f"prb_pp_{pname}",
                    "l": "pp",
                    "kind": "rdep",
                    "r_matrices": r_mats,
                    "metadata": dict(meta, template_norm=norm),
                    "blocks": p_blocks,
                }
            )
    if not terms:
        raise ValueError("All PRB p-p SK SOC templates vanished")
    return terms


def _collect_interatomic_bonds(
    coll_hr,
    blocks: list[dict],
    win_path: str,
    *,
    pair_type: str,
    shell_tol: float,
) -> list[SKBond]:
    labels, frac, _numbers = read_atoms_from_win(win_path)
    atom_frac_by_label = {label: np.asarray(coord, dtype=float) for label, coord in zip(labels, frac)}
    cell = read_unit_cell_cart_from_win(win_path)
    r_to_ir = {tuple(r): ir for ir, r in enumerate(coll_hr.r_keys)}
    blocks_l = [dict(b, _block_id=i) for i, b in enumerate(blocks)]
    if pair_type == "pd":
        row_blocks = [b for b in blocks_l if int(b["l"]) == 1]
        col_blocks = [b for b in blocks_l if int(b["l"]) == 2]
    elif pair_type == "dd":
        row_blocks = [b for b in blocks_l if int(b["l"]) == 2]
        col_blocks = row_blocks
    else:
        raise ValueError(f"Unsupported generic SK pair_type={pair_type!r}")
    if not row_blocks or not col_blocks:
        return []

    bonds: list[SKBond] = []
    for ir, r in enumerate(coll_hr.r_keys):
        r_tuple = tuple(int(x) for x in r)
        r_neg = tuple(-int(x) for x in r_tuple)
        if r_neg not in r_to_ir:
            continue
        ir_neg = r_to_ir[r_neg]
        for row_block in row_blocks:
            row_id = int(row_block["_block_id"])
            row_frac = block_atom_fractional(row_block, atom_frac_by_label, win_path)
            row_label = str(row_block.get("atom_label", f"b{row_id + 1}"))
            for col_block in col_blocks:
                col_id = int(col_block["_block_id"])
                if pair_type == "dd":
                    partner_key = _canonical_bond_key(ir_neg, col_id, row_id)
                    this_key = _canonical_bond_key(ir, row_id, col_id)
                    if this_key > partner_key:
                        continue
                col_frac = block_atom_fractional(col_block, atom_frac_by_label, win_path)
                col_label = str(col_block.get("atom_label", f"b{col_id + 1}"))
                bond_frac = np.asarray(r, dtype=float) + col_frac - row_frac
                bond_cart = bond_frac @ cell
                distance = float(np.linalg.norm(bond_cart))
                if distance <= float(shell_tol):
                    continue
                bonds.append(
                    SKBond(
                        ir=ir,
                        ir_neg=ir_neg,
                        row_block=row_block,
                        col_block=col_block,
                        row_block_id=row_id,
                        col_block_id=col_id,
                        unit=bond_cart / distance,
                        distance_ang=distance,
                        pair_key=_canonical_pair_key(row_label, col_label),
                        pair_type=pair_type,
                    )
                )
    return bonds


def _generic_groups_for_bonds(bonds: list[SKBond], win_path: str, grouping: str, shell_tol: float) -> list[list[SKBond]]:
    grouping = str(grouping).strip().lower()
    if grouping == "shell":
        return _group_bonds_by_shell(bonds, shell_tol)
    if grouping == "orbit":
        cell = read_unit_cell_cart_from_win(win_path)
        sym_data = require_spglib_symmetry(win_path, shell_tol)
        return _group_bonds_by_orbit(bonds, sym_data["rotations"], cell, shell_tol)
    raise ValueError(f"Unknown --sk-grouping {grouping!r}; use shell or orbit")


def _trim_groups_per_pair(groups: list[list[SKBond]], num_shells: int) -> list[list[SKBond]]:
    if int(num_shells) <= 0:
        return groups
    selected = []
    counts: dict[tuple[str, str], int] = {}
    for group in groups:
        key = group[0].pair_key
        used = int(counts.get(key, 0))
        if used >= int(num_shells):
            continue
        selected.append(group)
        counts[key] = used + 1
    return selected


def _add_generic_soc_bond_terms(
    term_mats: dict[str, np.ndarray],
    local_tensors: dict[str, np.ndarray],
    bond: SKBond,
    dim: int,
    block_layout: str,
    lrow: int,
    lcol: int,
) -> None:
    for name, local_tensor in local_tensors.items():
        lambda_tensor = _rotate_local_lambda_tensor(local_tensor, lrow, lcol, bond.unit)
        for iorb in range(lambda_tensor.shape[0]):
            for jorb in range(lambda_tensor.shape[1]):
                block = _spin_block_from_lambda(lambda_tensor[iorb, jorb])
                _add_spinful_pair(term_mats[name][bond.ir], bond.row_block, bond.col_block, iorb, jorb, block, dim, block_layout)
                _add_spinful_pair(
                    term_mats[name][bond.ir_neg],
                    bond.col_block,
                    bond.row_block,
                    jorb,
                    iorb,
                    block.conj().T,
                    dim,
                    block_layout,
                )


def build_prb_generic_sk_terms(
    coll_hr,
    blocks: list[dict],
    block_layout: str,
    win_path: str,
    *,
    pair_type: str,
    num_shells: int = 1,
    shell_tol: float = 1.0e-2,
    grouping: str = "shell",
    include_same_species_asym: bool = False,
) -> list[dict]:
    pair_type = str(pair_type).strip().lower()
    if pair_type not in SK_SYMBOL_SPECS:
        raise ValueError(f"Unsupported PRB SK pair_type={pair_type!r}")
    lrow, lcol = (1, 2) if pair_type == "pd" else (2, 2)
    bonds = _collect_interatomic_bonds(coll_hr, blocks, win_path, pair_type=pair_type, shell_tol=shell_tol)
    if not bonds:
        return []
    groups = _trim_groups_per_pair(_generic_groups_for_bonds(bonds, win_path, grouping, shell_tol), num_shells)
    if not groups:
        return []

    terms = []
    for igroup, group in enumerate(groups):
        pair_species = group[0].pair_key
        same_species = pair_species[0] == pair_species[1]
        include_asym = include_same_species_asym or not same_species
        tensor_specs = dict(SK_SYMBOL_SPECS[pair_type]["sym"])
        if include_asym:
            tensor_specs.update(SK_SYMBOL_SPECS[pair_type]["asym"])
        local_tensors = {
            name: _local_symbol_tensor(lrow, lcol, symbol, symmetric=not name.startswith("Ktilde"))
            for name, symbol in tensor_specs.items()
        }
        mats = {name: np.zeros_like(coll_hr.h_norm) for name in local_tensors}
        for bond in group:
            _add_generic_soc_bond_terms(mats, local_tensors, bond, coll_hr.dim, block_layout, lrow, lcol)
        distances = np.asarray([b.distance_ang for b in group], dtype=float)
        meta = {
            "orbit": igroup,
            "grouping": grouping,
            "num_bonds": len(group),
            "distance_mean_ang": float(np.mean(distances)),
            "distance_std_ang": float(np.std(distances)),
            "distance_min_ang": float(np.min(distances)),
            "distance_max_ang": float(np.max(distances)),
            "pair_keys": "-".join(pair_species),
            "sk_soc_form": f"PRB111_035137_{pair_type}_Eq26_symbols",
            "orbital_order": "wannier90",
        }
        for pname, r_mats in mats.items():
            norm = float(np.linalg.norm(r_mats.reshape(r_mats.shape[0], -1)))
            if norm <= 1.0e-14:
                continue
            terms.append(
                {
                    "name": f"prb_{pair_type}_{pname}_{grouping}_{igroup}",
                    "operator": f"prb_{pair_type}_{pname}",
                    "l": pair_type,
                    "kind": "rdep",
                    "r_matrices": r_mats,
                    "metadata": dict(meta, template_norm=norm),
                    "blocks": [b for b in blocks if int(b["l"]) in ({1, 2} if pair_type == "pd" else {2})],
                }
            )
    return terms


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit PRB interatomic SK SOC corrections.")
    parser.add_argument("--workdir", type=str, default=None)
    parser.add_argument("--target-dir", type=str, default=None)
    parser.add_argument("--coll-dir", type=str, default=None)
    parser.add_argument("--wsoc-dir", type=str, default=None, help="Legacy alias for --target-dir")
    parser.add_argument("--wosoc-dir", type=str, default=None, help="Legacy alias for --coll-dir")
    parser.add_argument("--seed", type=str, default=None)
    parser.add_argument("--coll-hr", type=str, default=None)
    parser.add_argument("--target-hr", type=str, default=None)
    parser.add_argument("--kpt", type=str, default=None)
    parser.add_argument("--kmesh", nargs=3, type=int, default=None)
    parser.add_argument("--mesh-shift", nargs=3, type=float, default=(0.173205, 0.318310, 0.414214))
    parser.add_argument("--win", type=str, default=None)
    parser.add_argument("--centres", type=str, default=None, help="Optional centres.xyz for inferred block sanity check")
    parser.add_argument("--blocks", nargs="+", default=None)
    parser.add_argument("--block-layout", choices=["spinor-block", "separated-spin"], default="spinor-block")
    parser.add_argument("--bands", type=str, required=True)
    parser.add_argument("--align", choices=["none", "bottom", "vbm"], default="none")
    parser.add_argument("--align-bands", type=str, default="1")
    parser.add_argument("--align-reducer", choices=["mean", "min", "max"], default="mean")
    parser.add_argument("--num-occupied", type=int, default=None)
    parser.add_argument("--target-l", nargs="+", type=int, default=None, help="Optional onsite L.S channels to include")
    parser.add_argument("--onsite-positive", action="store_true")
    parser.add_argument("--term-bound", action="append", default=None, metavar="NAME:LOWER:UPPER")
    parser.add_argument("--include-pp-sk", action="store_true", default=True, help="Include PRB p-p SK SOC templates")
    parser.add_argument("--no-pp-sk", action="store_false", dest="include_pp_sk")
    parser.add_argument("--include-pd-sk", action="store_true", default=True, help="Include PRB p-d SK SOC templates")
    parser.add_argument("--no-pd-sk", action="store_false", dest="include_pd_sk")
    parser.add_argument("--include-dd-sk", action="store_true", default=True, help="Include PRB d-d SK SOC templates")
    parser.add_argument("--no-dd-sk", action="store_false", dest="include_dd_sk")
    parser.add_argument("--p-order", default="pz,px,py", help="Deprecated; Wannier90 p order is used internally")
    parser.add_argument("--num-shells", type=int, default=1)
    parser.add_argument("--shell-tol", type=float, default=1.0e-2)
    parser.add_argument("--sk-grouping", choices=["shell", "orbit"], default="shell")
    parser.add_argument("--include-same-species-asym", action="store_true")
    parser.add_argument("--strict-pairs", action="store_true")
    parser.add_argument("--ridge", type=float, default=1.0e-8)
    parser.add_argument("--max-nfev", type=int, default=200)
    parser.add_argument("--k-workers", type=int, default=1)
    parser.add_argument("--k-batch-size", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--remove-center", action="store_true")
    parser.add_argument("--moment-constraint", choices=["none", "collinear", "target"], default="none")
    parser.add_argument("--moment-subspace", default="")
    parser.add_argument("--moment-components", default="z")
    parser.add_argument("--moment-weight", type=float, default=1.0)
    parser.add_argument("--moment-num-occupied", type=int, default=None)
    parser.add_argument("--out-prefix", type=str, default="soc_SK_fit")
    return parser


def main(argv=None) -> None:
    args = build_argparser().parse_args(argv)
    workdir = resolve_workdir(args.workdir)
    target_dir_arg = args.target_dir if args.target_dir else args.wsoc_dir
    coll_dir_arg = args.coll_dir if args.coll_dir else args.wosoc_dir
    target_dir = resolve_path(workdir, target_dir_arg) if target_dir_arg else None
    coll_dir = resolve_path(workdir, coll_dir_arg) if coll_dir_arg else None
    if target_dir is not None and not os.path.isdir(target_dir):
        raise ValueError(f"Target directory does not exist: {target_dir}")
    if coll_dir is not None and not os.path.isdir(coll_dir):
        raise ValueError(f"Collinear directory does not exist: {coll_dir}")

    if args.seed:
        seed = args.seed
    elif args.target_hr:
        seed = infer_seed_from_paths([resolve_path(workdir, args.target_hr)])
    elif target_dir:
        seed = infer_seed_from_paths(glob.glob(os.path.join(target_dir, "*_hr.dat")))
    elif args.coll_hr:
        seed = infer_seed_from_paths([resolve_path(workdir, args.coll_hr)])
    elif coll_dir:
        seed = infer_seed_from_paths(glob.glob(os.path.join(coll_dir, "*_hr.dat")))
    else:
        seed = infer_seed_from_paths(glob.glob(os.path.join(workdir, "**", "*_hr.dat"), recursive=True))

    if args.coll_hr:
        coll_path = resolve_path(workdir, args.coll_hr)
    elif coll_dir:
        coll_path = os.path.join(coll_dir, f"{seed}_hr.dat")
    else:
        coll_path = choose_unique_candidate(
            glob.glob(os.path.join(workdir, "**", f"{seed}_hr.dat"), recursive=True),
            "coll",
            exclude=[resolve_path(workdir, args.target_hr)] if args.target_hr else [],
        )
    if args.target_hr:
        target_path = resolve_path(workdir, args.target_hr)
    elif target_dir:
        target_path = os.path.join(target_dir, f"{seed}_hr.dat")
    else:
        target_path = choose_unique_candidate(
            glob.glob(os.path.join(workdir, "**", f"{seed}_hr.dat"), recursive=True),
            "target",
            exclude=[coll_path],
        )
    if not os.path.exists(coll_path):
        raise ValueError(f"Collinear/noSOC hr.dat not found: {coll_path}")
    if not os.path.exists(target_path):
        raise ValueError(f"Target SOC hr.dat not found: {target_path}")

    if args.kmesh is not None:
        kpts = make_shifted_kmesh(args.kmesh, args.mesh_shift)
        k_source = f"shifted mesh {tuple(args.kmesh)} with shift {tuple(float(x) for x in args.mesh_shift)}"
    else:
        kpt_path = resolve_path(workdir, args.kpt) if args.kpt else find_seeded_sidecar(target_path, seed, "_band.kpt")
        if kpt_path is None:
            raise ValueError("Could not find kpt file near target hr.dat; pass --kpt or --kmesh")
        kpts, _weights = read_kpoints(kpt_path)
        k_source = kpt_path

    win_path = resolve_path(workdir, args.win) if args.win else (
        find_seeded_sidecar(target_path, seed, ".win") or find_seeded_sidecar(coll_path, seed, ".win")
    )
    if win_path is None:
        raise ValueError("soc_SK requires --win or a seed .win next to input hr.dat")
    centres_path = (
        resolve_path(workdir, args.centres)
        if args.centres
        else (
            find_hr_seed_sidecar(coll_path, "_centres.xyz")
            or find_seeded_sidecar(coll_path, seed, "_centres.xyz")
            or find_hr_seed_sidecar(target_path, "_centres.xyz")
            or find_seeded_sidecar(target_path, seed, "_centres.xyz")
        )
    )

    coll_hr = load_hr(coll_path)
    target_hr = load_hr(target_path)
    if coll_hr.dim != target_hr.dim:
        raise ValueError(f"Dimension mismatch: coll={coll_hr.dim}, target={target_hr.dim}")

    blocks = parse_blocks(args.blocks)
    if not blocks:
        blocks = infer_blocks_from_win(win_path, coll_hr.dim, args.block_layout, centres_path=centres_path)

    terms = []
    used_blocks_by_operator = []
    for l_val in args.target_l or []:
        op_matrix, used_blocks = build_ls_operator(coll_hr.dim, blocks, target_l=int(l_val), block_layout=args.block_layout)
        ls_term = build_static_onsite_term(coll_hr, operator_name(l_val), int(l_val), op_matrix, used_blocks)
        terms.append(ls_term)
        used_blocks_by_operator.append({"operator": operator_name(l_val), "l": int(l_val), "blocks": used_blocks})

    if args.include_pp_sk:
        pp_terms = build_prb_pp_sk_terms(
            coll_hr,
            blocks,
            args.block_layout,
            win_path,
            p_order=args.p_order,
            num_shells=args.num_shells,
            shell_tol=args.shell_tol,
            grouping=args.sk_grouping,
            include_same_species_asym=args.include_same_species_asym,
            strict_pairs=args.strict_pairs,
        )
        terms.extend(pp_terms)
        used_blocks_by_operator.extend({"operator": t["name"], "l": t["l"], "blocks": t["blocks"]} for t in pp_terms)
    if args.include_pd_sk:
        pd_terms = build_prb_generic_sk_terms(
            coll_hr,
            blocks,
            args.block_layout,
            win_path,
            pair_type="pd",
            num_shells=args.num_shells,
            shell_tol=args.shell_tol,
            grouping=args.sk_grouping,
            include_same_species_asym=args.include_same_species_asym,
        )
        terms.extend(pd_terms)
        used_blocks_by_operator.extend({"operator": t["name"], "l": t["l"], "blocks": t["blocks"]} for t in pd_terms)
    if args.include_dd_sk:
        dd_terms = build_prb_generic_sk_terms(
            coll_hr,
            blocks,
            args.block_layout,
            win_path,
            pair_type="dd",
            num_shells=args.num_shells,
            shell_tol=args.shell_tol,
            grouping=args.sk_grouping,
            include_same_species_asym=args.include_same_species_asym,
        )
        terms.extend(dd_terms)
        used_blocks_by_operator.extend({"operator": t["name"], "l": t["l"], "blocks": t["blocks"]} for t in dd_terms)
    if not terms:
        raise ValueError("No fit terms were selected")

    bands = parse_band_selection(args.bands, coll_hr.dim, name="fit bands")
    align_bands = parse_band_selection(args.align_bands, coll_hr.dim, name="align bands")
    hk_coll = hamiltonian_k(coll_hr, kpts)
    hk_target = hamiltonian_k(target_hr, kpts)
    evals_coll = np.linalg.eigvalsh(hk_coll)
    evals_target = np.linalg.eigvalsh(hk_target)
    shift, alignment = compute_alignment_shift(
        evals_target,
        evals_coll,
        args.align,
        bottom_bands=align_bands,
        bottom_reducer=args.align_reducer,
        num_occupied=args.num_occupied,
    )
    evals_target_aligned = evals_target - shift

    moment_active = args.moment_constraint != "none"
    moment_blocks = []
    moment_components = []
    moment_reference = None
    moment_nocc = None
    if moment_active:
        moment_nocc = args.moment_num_occupied if args.moment_num_occupied is not None else args.num_occupied
        if moment_nocc is None:
            raise ValueError("--moment-constraint requires --moment-num-occupied or --num-occupied")
        moment_blocks = moment_blocks_from_selector(blocks, args.moment_subspace)
        moment_components = parse_moment_components(args.moment_components)
        ref_hk = hk_coll if args.moment_constraint == "collinear" else hk_target
        moment_reference = local_spin_moments(ref_hk, moment_blocks, moment_components, moment_nocc, coll_hr.dim, args.block_layout)

    phases = k_phases(kpts, coll_hr.r_vecs)
    terms = prepare_terms_hk(terms, phases)
    x0 = ridge_initial_guess(hk_coll, terms, evals_target_aligned, bands, ridge=args.ridge)
    lower_bounds, upper_bounds = build_lambda_bounds(terms, onsite_positive=args.onsite_positive, term_bounds=args.term_bound)
    x0 = clip_initial_to_bounds(x0, lower_bounds, upper_bounds)

    progress_every = max(0, int(args.progress_every))
    progress_state = {"calls": 0, "t0": time.time()}
    progress_block = len(x0) + 1
    k_workers = max(1, int(args.k_workers))
    if int(args.k_batch_size) > 0:
        k_batch_size = int(args.k_batch_size)
    elif k_workers > 1:
        k_batch_size = max(1, int(np.ceil(len(kpts) / float(4 * k_workers))))
    else:
        k_batch_size = len(kpts)

    def residual_fn(values):
        moment_resid = np.empty(0, dtype=float)
        if k_workers > 1 or moment_active:
            data_resid, moment_current = selected_residual_kparallel(
                values,
                hk_coll,
                terms,
                evals_target_aligned,
                bands,
                remove_center=args.remove_center,
                workers=k_workers,
                batch_size=k_batch_size,
                moment_blocks=moment_blocks if moment_active else None,
                moment_components=moment_components if moment_active else None,
                moment_nocc=moment_nocc,
                dim=coll_hr.dim,
                block_layout=args.block_layout,
            )
            if moment_active:
                moment_resid = float(args.moment_weight) * np.ravel(moment_current - moment_reference)
        else:
            data_resid = selected_residual(values, hk_coll, terms, evals_target_aligned, bands, remove_center=args.remove_center)
        reg = np.sqrt(float(args.ridge)) * np.asarray(values, dtype=float)
        progress_state["calls"] += 1
        ncall = int(progress_state["calls"])
        nfev = 1 + (ncall - 1) // progress_block
        is_base_eval = (ncall - 1) % progress_block == 0
        if progress_every > 0 and is_base_eval and (nfev == 1 or nfev % progress_every == 0):
            total_resid = np.concatenate([data_resid, moment_resid, reg])
            cost = 0.5 * float(np.dot(total_resid, total_resid))
            rms = float(np.sqrt(np.mean(data_resid * data_resid)))
            print(f"     iter {nfev:5d} cost={cost:.6e} rms={rms:.6e} elapsed={time.time() - progress_state['t0']:.1f}s", flush=True)
        return np.concatenate([data_resid, moment_resid, reg])

    result = least_squares(residual_fn, x0, bounds=(lower_bounds, upper_bounds), max_nfev=args.max_nfev)
    lambdas = result.x
    final_residual = selected_residual(lambdas, hk_coll, terms, evals_target_aligned, bands, remove_center=args.remove_center)
    hk_fit = model_hk(hk_coll, terms, lambdas)
    evals_fit = np.linalg.eigvalsh(hk_fit)
    moment_fit = None
    if moment_active:
        moment_fit = local_spin_moments(hk_fit, moment_blocks, moment_components, moment_nocc, coll_hr.dim, args.block_layout)

    out_prefix = resolve_path(workdir, args.out_prefix)
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    lambda_csv = f"{out_prefix}_lambdas.csv"
    summary_csv = f"{out_prefix}_fit_summary.csv"
    npz_path = f"{out_prefix}.npz"
    hr_path = f"{out_prefix}_hr.dat"
    for input_path, role in ((coll_path, "collinear"), (target_path, "target")):
        if os.path.abspath(hr_path) == os.path.abspath(input_path):
            raise ValueError(f"Refusing to overwrite {role} input hr.dat: {hr_path}")
    write_lambda_csv(lambda_csv, terms, lambdas, lower_bounds=lower_bounds, upper_bounds=upper_bounds)
    write_fit_summary_csv(summary_csv, kpts, final_residual, evals_fit[:, bands], evals_target_aligned[:, bands], bands)
    write_fitted_hr(hr_path, coll_hr, terms, lambdas)
    np.savez(
        npz_path,
        kpts=kpts,
        bands_1based=bands + 1,
        lambdas=lambdas,
        lambda_lower_bounds=lower_bounds,
        lambda_upper_bounds=upper_bounds,
        term_names=np.asarray([term["name"] for term in terms], dtype=object),
        operator_names=np.asarray([term.get("operator", term["name"]) for term in terms], dtype=object),
        operator_l=np.asarray([term["l"] for term in terms], dtype=object),
        term_metadata=np.asarray([term.get("metadata", {}) for term in terms], dtype=object),
        kmesh=np.asarray(args.kmesh if args.kmesh is not None else [], dtype=int),
        mesh_shift=np.asarray(args.mesh_shift, dtype=float),
        evals_coll=evals_coll,
        evals_target=evals_target,
        evals_target_aligned=evals_target_aligned,
        evals_fit=evals_fit,
        residual=final_residual.reshape(len(kpts), len(bands)),
        energy_shift_target_minus_coll_eV=float(shift),
        alignment=np.asarray(alignment, dtype=object),
        used_blocks=np.asarray(used_blocks_by_operator, dtype=object),
        block_layout=args.block_layout,
        success=bool(result.success),
        cost=float(result.cost),
        message=str(result.message),
        moment_constraint=args.moment_constraint,
        moment_subspace=args.moment_subspace,
        moment_components=np.asarray(moment_components, dtype=object),
        moment_blocks=np.asarray(moment_blocks, dtype=object),
        moment_num_occupied=-1 if moment_nocc is None else int(moment_nocc),
        moment_reference=np.asarray([] if moment_reference is None else moment_reference),
        moment_fit=np.asarray([] if moment_fit is None else moment_fit),
        moment_weight=float(args.moment_weight),
    )

    rms = float(np.sqrt(np.mean(final_residual * final_residual)))
    max_abs = float(np.max(np.abs(final_residual)))
    print(f"Using seed: {seed}")
    print(f"Loaded k-points: {len(kpts)} from {k_source}")
    print(f"Loaded collinear: dim={coll_hr.dim}, R={len(coll_hr.r_keys)} from {coll_path}")
    print(f"Loaded target SOC: dim={target_hr.dim}, R={len(target_hr.r_keys)} from {target_path}")
    print(f"Loaded lattice: {win_path}")
    print(f"Block layout: {args.block_layout}")
    print(
        "PRB SK: "
        f"pp={bool(args.include_pp_sk)}, pd={bool(args.include_pd_sk)}, dd={bool(args.include_dd_sk)}, "
        f"grouping={args.sk_grouping}, shells_per_pair={args.num_shells}, orbital_order=wannier90"
    )
    preview_terms = [term["name"] for term in terms[:12]]
    suffix = " ..." if len(terms) > len(preview_terms) else ""
    print(f"Selected terms ({len(terms)}): {preview_terms}{suffix}")
    print(f"Selected bands: {bands + 1}")
    print(f"K residual workers: {k_workers}, batch_size={k_batch_size}")
    print(f"Alignment mode: {args.align}, shift={shift:.12g} eV")
    print(f"Fit success: {result.success}, cost={result.cost:.12g}")
    print(f"Fit residual RMS={rms:.12g} eV, max_abs={max_abs:.12g} eV")
    print(f"Saved lambdas: {lambda_csv}")
    print(f"Saved fit summary: {summary_csv}")
    print(f"Saved fitted hr: {hr_path}")
    print(f"Saved npz: {npz_path}")


if __name__ == "__main__":
    main()
