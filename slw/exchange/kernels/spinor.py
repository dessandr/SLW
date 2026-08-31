"""Spinor-model helpers for native exchange Hamiltonians.

These routines intentionally operate on already-built H(k) or g(k,q) matrices
instead of hr.dat files.  The first production use is planned for model SOC
and spin-flip EPR vertices in compute_dJ_epr_tensor.
"""

from __future__ import annotations

import numpy as np

from slw.soc.atomic import (
    WANNIER90_D_ORDER,
    WANNIER90_P_ORDER,
    add_atomic_d_soc,
    add_atomic_p_soc,
    atomic_d_soc_block,
    atomic_p_soc_block,
    atomic_soc_block_diagonal,
    d_orbital_l_matrices,
    p_orbital_l_matrices,
)


def normalize_spin_direction(n):
    arr = np.asarray(n, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(arr))
    if norm <= 0.0:
        raise ValueError(f"Spin direction must be nonzero, got {arr.tolist()}")
    return arr / norm


def spinor_from_collinear(up, down, n=(0.0, 0.0, 1.0)):
    """Embed collinear up/down matrices into spin-major spinor basis.

    Basis convention is [up orbitals..., down orbitals...].
    For n=(0,0,1), this returns block_diag(up, down).
    """
    hup = np.asarray(up, dtype=np.complex128)
    hdn = np.asarray(down, dtype=np.complex128)
    if hup.shape != hdn.shape or hup.ndim < 2 or hup.shape[-1] != hup.shape[-2]:
        raise ValueError(f"up/down matrix shape mismatch: {hup.shape} vs {hdn.shape}")
    nx, ny, nz = normalize_spin_direction(n)
    h0 = 0.5 * (hup + hdn)
    hz = 0.5 * (hup - hdn)
    out_shape = hup.shape[:-2] + (2 * hup.shape[-2], 2 * hup.shape[-1])
    out = np.zeros(out_shape, dtype=np.complex128)
    nwan = hup.shape[-1]
    out[..., :nwan, :nwan] = h0 + nz * hz
    out[..., :nwan, nwan:] = (nx - 1j * ny) * hz
    out[..., nwan:, :nwan] = (nx + 1j * ny) * hz
    out[..., nwan:, nwan:] = h0 - nz * hz
    return out



def spinor_from_blocks(uu, dd, ud=None, du=None):
    """Assemble a spin-major spinor matrix from explicit spin blocks.

    This is the placeholder entry point for future spin-flip EPR g datasets.
    If ud/du are omitted they are filled with zeros.
    """
    uu = np.asarray(uu, dtype=np.complex128)
    dd = np.asarray(dd, dtype=np.complex128)
    if uu.shape != dd.shape or uu.ndim < 2 or uu.shape[-1] != uu.shape[-2]:
        raise ValueError(f"uu/dd matrix shape mismatch: {uu.shape} vs {dd.shape}")
    if ud is None:
        ud = np.zeros_like(uu)
    if du is None:
        du = np.zeros_like(uu)
    ud = np.asarray(ud, dtype=np.complex128)
    du = np.asarray(du, dtype=np.complex128)
    if ud.shape != uu.shape or du.shape != uu.shape:
        raise ValueError(f"spin-flip block shape mismatch: uu={uu.shape} ud={ud.shape} du={du.shape}")
    n = uu.shape[-1]
    out = np.zeros(uu.shape[:-2] + (2 * n, 2 * n), dtype=np.complex128)
    out[..., :n, :n] = uu
    out[..., :n, n:] = ud
    out[..., n:, :n] = du
    out[..., n:, n:] = dd
    return out


__all__ = [
    "WANNIER90_D_ORDER",
    "WANNIER90_P_ORDER",
    "add_atomic_d_soc",
    "add_atomic_p_soc",
    "atomic_d_soc_block",
    "atomic_p_soc_block",
    "atomic_soc_block_diagonal",
    "d_orbital_l_matrices",
    "normalize_spin_direction",
    "p_orbital_l_matrices",
    "spinor_from_blocks",
    "spinor_from_collinear",
]
