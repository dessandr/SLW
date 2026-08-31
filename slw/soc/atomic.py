r"""Atomic onsite :math:`\lambda\,\mathbf L\cdot\mathbf S` operators.

The orbital manifolds are resolved from a Wannier90 ``.win`` file.  No
material, atom, or orbital indices are inferred from hard-coded tables beyond
Wannier90's declared real-harmonic ordering conventions.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .manifold import clean_species, read_projection_groups
from .model import AtomicSOCSpec, SpinorGroupBy
from .spinor import reorder_spinor_matrix

WANNIER90_P_ORDER = "pz,px,py"
WANNIER90_D_ORDER = "dz2,dxz,dyz,dx2-y2,dxy"


@dataclass(frozen=True)
class ResolvedAtomicSOCManifold:
    """One input SOC selector resolved to explicit non-spin Wannier indices."""

    selector: str
    orbital: str
    lambda_ev: float
    groups: tuple[tuple[int, ...], ...]
    matched_labels: tuple[str, ...]


def p_orbital_l_matrices(
    order: str = WANNIER90_P_ORDER,
) -> tuple[NDArray[np.complex128], ...]:
    """Return ``Lx, Ly, Lz`` in a declared real-p-orbital order."""

    base = ["px", "py", "pz"]
    order_list = [
        value.strip().lower()
        for value in str(order).replace(";", ",").split(",")
        if value.strip()
    ]
    if sorted(order_list) != sorted(base):
        raise ValueError(
            f"unsupported p orbital order {order_list}; expected a permutation of {base}"
        )
    lx = np.array(
        [[0, 0, 0], [0, 0, -1j], [0, 1j, 0]], dtype=np.complex128
    )
    ly = np.array(
        [[0, 0, 1j], [0, 0, 0], [-1j, 0, 0]], dtype=np.complex128
    )
    lz = np.array(
        [[0, -1j, 0], [1j, 0, 0], [0, 0, 0]], dtype=np.complex128
    )
    permutation = [base.index(value) for value in order_list]
    return tuple(matrix[np.ix_(permutation, permutation)] for matrix in (lx, ly, lz))


def d_orbital_l_matrices(
    order: str = WANNIER90_D_ORDER,
) -> tuple[NDArray[np.complex128], ...]:
    """Return ``Lx, Ly, Lz`` in a declared real-d-orbital order."""

    base = ["dz2", "dxz", "dyz", "dx2-y2", "dxy"]
    aliases = {
        "dz^2": "dz2",
        "d_z2": "dz2",
        "d_z^2": "dz2",
        "dx2y2": "dx2-y2",
        "dx^2-y^2": "dx2-y2",
        "d_x2-y2": "dx2-y2",
        "d_x^2-y^2": "dx2-y2",
    }
    order_list = [
        aliases.get(value.strip().lower(), value.strip().lower())
        for value in str(order).replace(";", ",").split(",")
        if value.strip()
    ]
    if sorted(order_list) != sorted(base):
        raise ValueError(
            f"unsupported d orbital order {order_list}; expected a permutation of {base}"
        )
    root_three = np.sqrt(3.0)
    lx = np.array(
        [
            [0, 0, 1j * root_three, 0, 0],
            [0, 0, 0, 0, 1j],
            [-1j * root_three, 0, 0, -1j, 0],
            [0, 0, 1j, 0, 0],
            [0, -1j, 0, 0, 0],
        ],
        dtype=np.complex128,
    )
    ly = np.array(
        [
            [0, -1j * root_three, 0, 0, 0],
            [1j * root_three, 0, 0, 1j, 0],
            [0, 0, 0, 0, 1j],
            [0, -1j, 0, 0, 0],
            [0, 0, -1j, 0, 0],
        ],
        dtype=np.complex128,
    )
    lz = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, -1j, 0, 0],
            [0, 1j, 0, 0, 0],
            [0, 0, 0, 0, -2j],
            [0, 0, 0, 2j, 0],
        ],
        dtype=np.complex128,
    )
    permutation = [base.index(value) for value in order_list]
    return tuple(matrix[np.ix_(permutation, permutation)] for matrix in (lx, ly, lz))


def _atomic_soc_block(
    lambda_ev: float,
    angular_momentum: tuple[NDArray[np.complex128], ...],
) -> NDArray[np.complex128]:
    lx, ly, lz = angular_momentum
    value = float(lambda_ev)
    if not math.isfinite(value):
        raise ValueError(f"SOC lambda must be finite, got {lambda_ev!r}")
    norb = lx.shape[0]
    block = np.zeros((2 * norb, 2 * norb), dtype=np.complex128)
    block[:norb, :norb] = 0.5 * value * lz
    block[:norb, norb:] = 0.5 * value * (lx - 1j * ly)
    block[norb:, :norb] = 0.5 * value * (lx + 1j * ly)
    block[norb:, norb:] = -0.5 * value * lz
    return block


def atomic_p_soc_block(
    lambda_ev: float,
    order: str = WANNIER90_P_ORDER,
) -> NDArray[np.complex128]:
    """Return a spin-major p-shell ``lambda L.S`` block in eV."""

    return _atomic_soc_block(lambda_ev, p_orbital_l_matrices(order))


def atomic_d_soc_block(
    lambda_ev: float,
    order: str = WANNIER90_D_ORDER,
) -> NDArray[np.complex128]:
    """Return a spin-major d-shell ``lambda L.S`` block in eV."""

    return _atomic_soc_block(lambda_ev, d_orbital_l_matrices(order))


def atomic_soc_block_diagonal(
    l_values: Iterable[int],
    lambda_ev: float,
) -> NDArray[np.complex128]:
    """Build spin-major atomic ``lambda L.S`` blocks for s, p, and d shells."""

    blocks: list[NDArray[np.complex128]] = []
    for raw_value in l_values:
        value = int(raw_value)
        if value == 0:
            blocks.append(np.zeros((2, 2), dtype=np.complex128))
        elif value == 1:
            blocks.append(atomic_p_soc_block(lambda_ev))
        elif value == 2:
            blocks.append(atomic_d_soc_block(lambda_ev))
        else:
            raise ValueError(f"atomic SOC supports s, p, and d shells; received l={value}")
    dimension = sum(block.shape[0] for block in blocks)
    result = np.zeros((dimension, dimension), dtype=np.complex128)
    offset = 0
    for block in blocks:
        stop = offset + block.shape[0]
        result[offset:stop, offset:stop] = block
        offset = stop
    return result


def _add_atomic_soc_block(
    h_spin: object,
    orbital_groups: Iterable[Iterable[int]],
    soc_block: NDArray[np.complex128],
    *,
    inplace: bool,
) -> NDArray[np.complex128]:
    matrix = (
        np.asarray(h_spin)
        if inplace
        else np.array(h_spin, dtype=np.complex128, copy=True)
    )
    if matrix.dtype != np.dtype(np.complex128):
        raise TypeError(f"h_spin must use complex128; got {matrix.dtype}")
    if matrix.ndim < 2 or matrix.shape[-1] != matrix.shape[-2] or matrix.shape[-1] % 2:
        raise ValueError(f"h_spin must have even square trailing shape, got {matrix.shape}")
    nwan = matrix.shape[-1] // 2
    norb = soc_block.shape[0] // 2
    for group in orbital_groups:
        orbital_indices = np.asarray(group, dtype=np.int64).reshape(norb)
        if np.any(orbital_indices < 0) or np.any(orbital_indices >= nwan):
            raise ValueError(
                f"orbital indices out of range for nwan={nwan}: {orbital_indices.tolist()}"
            )
        spinor_indices = np.concatenate((orbital_indices, orbital_indices + nwan))
        matrix[..., spinor_indices[:, None], spinor_indices[None, :]] += soc_block
    return np.asarray(matrix, dtype=np.complex128)


def add_atomic_p_soc(
    h_spin: object,
    p_orbital_groups: Iterable[Iterable[int]],
    lambda_ev: float,
    order: str = WANNIER90_P_ORDER,
    inplace: bool = False,
) -> NDArray[np.complex128]:
    """Add onsite atomic p SOC to a spin-major spinor matrix."""

    return _add_atomic_soc_block(
        h_spin,
        p_orbital_groups,
        atomic_p_soc_block(lambda_ev, order),
        inplace=inplace,
    )


def add_atomic_d_soc(
    h_spin: object,
    d_orbital_groups: Iterable[Iterable[int]],
    lambda_ev: float,
    order: str = WANNIER90_D_ORDER,
    inplace: bool = False,
) -> NDArray[np.complex128]:
    """Add onsite atomic d SOC to a spin-major spinor matrix."""

    return _add_atomic_soc_block(
        h_spin,
        d_orbital_groups,
        atomic_d_soc_block(lambda_ev, order),
        inplace=inplace,
    )


def resolve_atomic_soc_manifolds(
    spec: AtomicSOCSpec,
    win_path: str | Path,
    *,
    nwan: int,
) -> tuple[ResolvedAtomicSOCManifold, ...]:
    """Resolve site/species selectors against ordered Wannier90 projections."""

    _, projection_groups, projection_count = read_projection_groups(win_path)
    if projection_count != int(nwan):
        raise ValueError(
            f"{Path(win_path)} projection count={projection_count} "
            f"but Hamiltonian nwan={nwan}"
        )
    used: set[int] = set()
    resolved: list[ResolvedAtomicSOCManifold] = []
    for manifold in spec.manifolds:
        selector_label = manifold.label
        orbital = manifold.orbital
        exact = [
            group
            for group in projection_groups
            if str(group["atom_label"]).casefold() == selector_label.casefold()
            and str(group["orbital"]).casefold() == orbital.casefold()
        ]
        matching = exact or [
            group
            for group in projection_groups
            if str(group["element"]).casefold() == clean_species(selector_label)
            and str(group["orbital"]).casefold() == orbital.casefold()
        ]
        if not matching:
            known = sorted(
                f"{group['atom_label']}-{group['orbital']}"
                for group in projection_groups
            )
            raise ValueError(
                f"SOC selector {manifold.selector!r} matched no Wannier manifold; "
                f"known site manifolds={known}"
            )
        resolved_groups: list[tuple[int, ...]] = []
        for group in matching:
            raw_indices = group["indices"]
            if not isinstance(raw_indices, list):
                raise TypeError("Wannier projection indices must be stored as a list")
            resolved_groups.append(tuple(int(index) for index in raw_indices))
        indices = {index for group in resolved_groups for index in group}
        overlap = sorted(used & indices)
        if overlap:
            raise ValueError(
                f"SOC selector {manifold.selector!r} overlaps a previous selector "
                f"at Wannier indices {overlap}"
            )
        used.update(indices)
        resolved.append(
            ResolvedAtomicSOCManifold(
                selector=manifold.selector,
                orbital=orbital,
                lambda_ev=manifold.lambda_ev,
                groups=tuple(resolved_groups),
                matched_labels=tuple(str(group["atom_label"]) for group in matching),
            )
        )
    return tuple(resolved)


def build_atomic_soc_matrix(
    spec: AtomicSOCSpec,
    win_path: str | Path,
    *,
    nwan: int,
    groupby: SpinorGroupBy | str,
    scale: float = 1.0,
    p_order: str = WANNIER90_P_ORDER,
    d_order: str = WANNIER90_D_ORDER,
) -> tuple[NDArray[np.complex128], tuple[ResolvedAtomicSOCManifold, ...]]:
    """Build a full onsite SOC matrix in the requested spinor basis order."""

    scale_value = float(scale)
    if not math.isfinite(scale_value):
        raise ValueError(f"SOC scale must be finite, got {scale!r}")
    resolved = resolve_atomic_soc_manifolds(spec, win_path, nwan=nwan)
    blocked = np.zeros((2 * int(nwan), 2 * int(nwan)), dtype=np.complex128)
    for manifold in resolved:
        value = scale_value * manifold.lambda_ev
        if manifold.orbital == "p":
            blocked = add_atomic_p_soc(
                blocked,
                manifold.groups,
                value,
                order=p_order,
            )
        elif manifold.orbital == "d":
            blocked = add_atomic_d_soc(
                blocked,
                manifold.groups,
                value,
                order=d_order,
            )
        else:  # AtomicSOCManifold currently admits only p and d.
            raise ValueError(f"unsupported SOC orbital {manifold.orbital!r}")
    output = reorder_spinor_matrix(
        blocked,
        source=SpinorGroupBy.SPIN,
        target=groupby,
    )
    residual = float(np.max(np.abs(output - output.conj().T), initial=0.0))
    if residual > 1.0e-12:
        raise ValueError(f"atomic SOC Hermiticity residual is {residual:.3e}")
    return np.asarray(output, dtype=np.complex128), resolved


__all__ = [
    "WANNIER90_D_ORDER",
    "WANNIER90_P_ORDER",
    "ResolvedAtomicSOCManifold",
    "add_atomic_d_soc",
    "add_atomic_p_soc",
    "atomic_d_soc_block",
    "atomic_p_soc_block",
    "atomic_soc_block_diagonal",
    "build_atomic_soc_matrix",
    "d_orbital_l_matrices",
    "p_orbital_l_matrices",
    "resolve_atomic_soc_manifolds",
]
