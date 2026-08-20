"""Compact numerical primitives for native LKAG integrations."""

from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True)
def _accumulate_tb2j_A_numba(
    GR,
    GRm,
    ni_arr,
    nj_arr,
    iu_up,
    iu_dn,
    jv_up,
    jv_dn,
    P_i,
    P_j,
    acc_A,
    weight_over_pi,
):
    """Accumulate TB2J Pauli ``A`` tensors over independent pair/R tasks."""

    n_pairs = ni_arr.shape[0]
    n_r = GR.shape[0]
    for task in prange(n_pairs * n_r):
        pair = task // n_r
        r_index = task - pair * n_r
        ni = ni_arr[pair]
        nj = nj_arr[pair]
        gij = np.zeros((4, ni, nj), dtype=np.complex128)
        gji = np.zeros((4, nj, ni), dtype=np.complex128)
        forward = GR[r_index]
        reverse = GRm[r_index]
        for i in range(ni):
            i_up = iu_up[pair, i]
            i_down = iu_dn[pair, i]
            for j in range(nj):
                j_up = jv_up[pair, j]
                j_down = jv_dn[pair, j]
                uu = forward[i_up, j_up]
                ud = forward[i_up, j_down]
                du = forward[i_down, j_up]
                dd = forward[i_down, j_down]
                gij[0, i, j] = 0.5 * (uu + dd)
                gij[1, i, j] = 0.5 * (ud + du)
                gij[2, i, j] = 0.5j * (ud - du)
                gij[3, i, j] = 0.5 * (uu - dd)

                uu = reverse[j_up, i_up]
                ud = reverse[j_up, i_down]
                du = reverse[j_down, i_up]
                dd = reverse[j_down, i_down]
                gji[0, j, i] = 0.5 * (uu + dd)
                gji[1, j, i] = 0.5 * (ud + du)
                gji[2, j, i] = 0.5j * (ud - du)
                gji[3, j, i] = 0.5 * (uu - dd)

        left = np.zeros((4, ni, nj), dtype=np.complex128)
        right = np.zeros((4, nj, ni), dtype=np.complex128)
        for component in range(4):
            for i in range(ni):
                for j in range(nj):
                    value = 0.0j
                    for orbital in range(ni):
                        value += P_i[pair, i, orbital] * gij[
                            component, orbital, j
                        ]
                    left[component, i, j] = value
            for j in range(nj):
                for i in range(ni):
                    value = 0.0j
                    for orbital in range(nj):
                        value += P_j[pair, j, orbital] * gji[
                            component, orbital, i
                        ]
                    right[component, j, i] = value

        for first in range(4):
            for second in range(4):
                trace = 0.0j
                for i in range(ni):
                    for j in range(nj):
                        trace += left[first, i, j] * right[second, j, i]
                acc_A[pair, r_index, first, second] += trace * weight_over_pi


def get_semicircle_contour(
    emin: float = -15.0,
    emax: float = 0.0,
    npoints: int = 40,
) -> list[tuple[np.complex128, np.complex128]]:
    """Return an upper-half-plane semicircle and oriented integration weights."""

    lower = float(emin)
    upper = float(emax)
    count = int(npoints)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError(f"require finite emin < emax, got {emin}, {emax}")
    if count < 2:
        raise ValueError(f"npoints must be at least 2, got {npoints}")
    radius = 0.5 * (upper - lower)
    centre = 0.5 * (upper + lower)
    theta = np.linspace(np.pi, 0.0, count)
    path = centre + radius * np.exp(1j * theta)
    delta_theta = -np.pi / (count - 1)
    weights = 1j * radius * np.exp(1j * theta) * delta_theta
    return list(zip(path.astype(np.complex128), weights.astype(np.complex128)))


def get_cfr_pole_mesh(
    npoles: int = 40,
    beta_eV_inv: float = 400.0,
) -> list[tuple[np.complex128, np.complex128]]:
    """Return the retained Matsubara-pole integration scaffold."""

    count = int(npoles)
    beta = float(beta_eV_inv)
    if count < 1 or not np.isfinite(beta) or beta <= 0.0:
        raise ValueError(f"require npoles >= 1 and beta > 0, got {npoles}, {beta}")
    indices = np.arange(1, count + 1, dtype=np.float64)
    poles = 1j * (2.0 * indices - 1.0) * np.pi / beta
    weights = np.full(count, (-2.0j * np.pi) / beta, dtype=np.complex128)
    return list(zip(poles.astype(np.complex128), weights))


def get_cfr_ozaki_mesh(
    npoles: int = 40,
    beta_eV_inv: float = 400.0,
) -> list[tuple[np.complex128, np.complex128]]:
    """Return an Ozaki continued-fraction pole mesh in deterministic order."""

    count = int(npoles)
    beta = float(beta_eV_inv)
    if count < 1 or not np.isfinite(beta) or beta <= 0.0:
        raise ValueError(f"require npoles >= 1 and beta > 0, got {npoles}, {beta}")
    if count == 1:
        pole = np.complex128(1j * np.pi / beta)
        weight = np.complex128((-2.0j * np.pi) / beta)
        return [(pole, weight)]
    indices = np.arange(1, count, dtype=np.float64)
    off_diagonal = 1.0 / (
        2.0 * np.sqrt((2.0 * indices - 1.0) * (2.0 * indices + 1.0))
    )
    continued_fraction = np.diag(off_diagonal, 1) + np.diag(off_diagonal, -1)
    poles, eigenvectors = np.linalg.eigh(continued_fraction)
    mask = poles > 1.0e-14
    positive = poles[mask]
    if positive.size == 0:
        return get_cfr_pole_mesh(npoles=1, beta_eV_inv=beta)
    residues = 0.25 * np.abs(eigenvectors[0, mask]) ** 2 / positive**2
    path = np.concatenate((1j / (beta * positive), -1j / (beta * positive)))
    weights = np.concatenate(
        ((2.0j / beta) * residues, (2.0j / beta) * residues)
    )
    order = np.argsort(np.imag(path))
    return list(
        zip(
            path[order].astype(np.complex128),
            weights[order].astype(np.complex128),
        )
    )


__all__ = [
    "_accumulate_tb2j_A_numba",
    "get_cfr_ozaki_mesh",
    "get_cfr_pole_mesh",
    "get_semicircle_contour",
]
