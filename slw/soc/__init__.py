"""Typed SOC terms and TB2J spinor-basis utilities."""

from .atomic import (
    WANNIER90_D_ORDER,
    WANNIER90_P_ORDER,
    ResolvedAtomicSOCManifold,
    add_atomic_d_soc,
    add_atomic_p_soc,
    atomic_d_soc_block,
    atomic_p_soc_block,
    atomic_soc_block_diagonal,
    build_atomic_soc_matrix,
    d_orbital_l_matrices,
    p_orbital_l_matrices,
    resolve_atomic_soc_manifolds,
)
from .manifold import clean_species, read_projection_groups
from .model import AtomicSOCManifold, AtomicSOCSpec, SpinorGroupBy
from .spinor import (
    groupby_to_spin_major_indices,
    normalize_groupby,
    reorder_spinor_matrix,
    reorder_spinor_rows,
    spin_major_to_groupby_indices,
)

__all__ = [
    "WANNIER90_D_ORDER",
    "WANNIER90_P_ORDER",
    "AtomicSOCManifold",
    "AtomicSOCSpec",
    "ResolvedAtomicSOCManifold",
    "SpinorGroupBy",
    "add_atomic_d_soc",
    "add_atomic_p_soc",
    "atomic_d_soc_block",
    "atomic_p_soc_block",
    "atomic_soc_block_diagonal",
    "build_atomic_soc_matrix",
    "clean_species",
    "d_orbital_l_matrices",
    "groupby_to_spin_major_indices",
    "normalize_groupby",
    "p_orbital_l_matrices",
    "read_projection_groups",
    "reorder_spinor_matrix",
    "reorder_spinor_rows",
    "resolve_atomic_soc_manifolds",
    "spin_major_to_groupby_indices",
]
