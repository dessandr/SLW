"""Typed SOC terms and TB2J spinor-basis utilities."""

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
    "AtomicSOCManifold",
    "AtomicSOCSpec",
    "SpinorGroupBy",
    "clean_species",
    "groupby_to_spin_major_indices",
    "normalize_groupby",
    "read_projection_groups",
    "reorder_spinor_matrix",
    "reorder_spinor_rows",
    "spin_major_to_groupby_indices",
]
