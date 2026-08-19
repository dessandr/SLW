"""
Magnon-phonon (magph) integration helpers for slw.

This package currently provides:
- I/O adapter utilities to convert SLW dJ/du payloads into legacy magph tuple format
- A preparation CLI to validate inputs and emit intermediate artifacts
"""

from .adapter import (
    load_djdu_npz,
    load_djr_h5,
    select_rp_slice,
    build_legacy_djdu_tuple,
    load_phonon_cache,
    load_j_payload,
)
from .runtime import (
    load_manifest,
    build_shapes,
    load_legacy_j_tuple,
    load_legacy_djdu_tuple,
    load_djdu_tuple,
    load_djdu_payload,
    estimate_lifetime_memory_gb,
    estimate_solver_memory_gb,
    recommend_solver_partition,
)
from .input_parser import parse_input_file
from .backend import load_legacy_backend

__all__ = [
    "load_djdu_npz",
    "load_djr_h5",
    "select_rp_slice",
    "build_legacy_djdu_tuple",
    "load_phonon_cache",
    "load_j_payload",
    "ExchangeTensorRealSpacePayload",
    "load_exchange_tensor_realspace_h5",
    "load_dmi_realspace_h5",
    "load_j_iso_realspace_h5",
    "load_j_aniso_realspace_h5",
    "summarize_tensor_payload",
    "summarize_dmi_payload",
    "SpinFrameInfo",
    "build_spin_frame_info",
    "compose_exchange_tensor",
    "dmi_vector_to_matrix",
    "linear_hp_coefficients_from_tensor",
    "bdg_matrix_from_tensor",
    "solve_tensor_lswt",
    "SiteBasisVertex",
    "ModeBasisVertex",
    "build_site_basis_vertex",
    "project_site_vertex_to_phonon_modes",
    "save_site_basis_vertex_npz",
    "save_mode_basis_vertex_npz",
    "HybridHamiltonian",
    "transform_vertex_to_magnon_modes",
    "build_hybrid_hamiltonian_blocks",
    "build_hybrid_from_static_tensor",
    "save_hybrid_npz",
    "compute_hybrid_berry",
    "fukui_band_flux",
    "load_manifest",
    "build_shapes",
    "load_legacy_j_tuple",
    "load_legacy_djdu_tuple",
    "load_djdu_tuple",
    "load_djdu_payload",
    "estimate_lifetime_memory_gb",
    "estimate_solver_memory_gb",
    "recommend_solver_partition",
    "HelicityProjectors",
    "AtomicRotationDecomposition",
    "ChiralizedPhononModes",
    "helicity_projectors",
    "decompose_atomistic_rotation",
    "phonon_angular_momentum_over_hbar",
    "chiralize_degenerate_subspaces",
    "apply_mode_rotations",
    "parse_input_file",
    "load_legacy_backend",
]


def __getattr__(name):
    tensor_adapter_names = {
        "ExchangeTensorRealSpacePayload",
        "load_exchange_tensor_realspace_h5",
        "load_dmi_realspace_h5",
        "load_j_iso_realspace_h5",
        "load_j_aniso_realspace_h5",
        "summarize_tensor_payload",
        "summarize_dmi_payload",
    }
    if name in tensor_adapter_names:
        from . import tensor_adapter

        return getattr(tensor_adapter, name)
    lswt_names = {
        "SpinFrameInfo",
        "build_spin_frame_info",
        "compose_exchange_tensor",
        "dmi_vector_to_matrix",
        "linear_hp_coefficients_from_tensor",
        "bdg_matrix_from_tensor",
        "solve_tensor_lswt",
    }
    if name in lswt_names:
        from . import lswt

        return getattr(lswt, name)
    vertex_names = {
        "SiteBasisVertex",
        "ModeBasisVertex",
        "build_site_basis_vertex",
        "project_site_vertex_to_phonon_modes",
        "save_site_basis_vertex_npz",
        "save_mode_basis_vertex_npz",
    }
    if name in vertex_names:
        from . import vertex

        return getattr(vertex, name)
    hybrid_names = {
        "HybridHamiltonian",
        "transform_vertex_to_magnon_modes",
        "build_hybrid_hamiltonian_blocks",
        "build_hybrid_from_static_tensor",
        "save_hybrid_npz",
    }
    if name in hybrid_names:
        from . import hybrid

        return getattr(hybrid, name)
    hybrid_berry_names = {
        "compute_hybrid_berry",
        "fukui_band_flux",
    }
    if name in hybrid_berry_names:
        from . import hybrid_berry

        return getattr(hybrid_berry, name)
    rotational_names = {
        "HelicityProjectors",
        "AtomicRotationDecomposition",
        "ChiralizedPhononModes",
        "helicity_projectors",
        "decompose_atomistic_rotation",
        "phonon_angular_momentum_over_hbar",
        "chiralize_degenerate_subspaces",
        "apply_mode_rotations",
    }
    if name in rotational_names:
        from . import rotational

        return getattr(rotational, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
