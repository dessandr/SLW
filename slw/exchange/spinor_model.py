"""Small spinor-model helpers for EPR collinear Hamiltonians.

These routines intentionally operate on already-built H(k) or g(k,q) matrices
instead of hr.dat files.  The first production use is planned for model SOC
and spin-flip EPR vertices in compute_dJ_epr_tensor.
"""

from __future__ import annotations

import numpy as np


WANNIER90_P_ORDER = "pz,px,py"
WANNIER90_D_ORDER = "dz2,dxz,dyz,dx2-y2,dxy"


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


def p_orbital_l_matrices(order=WANNIER90_P_ORDER):
    """Return Lx,Ly,Lz in the real p-orbital basis.

    Default order follows Wannier90 real harmonics: (pz, px, py).
    Matrices use hbar=1.
    """
    base = ["px", "py", "pz"]
    order_list = [x.strip().lower() for x in str(order).replace(";", ",").split(",") if x.strip()]
    if not order_list:
        order_list = base
    if sorted(order_list) != sorted(base):
        raise ValueError(f"Unsupported p orbital order {order_list}; expected a permutation of {base}")
    Lx0 = np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]], dtype=np.complex128)
    Ly0 = np.array([[0, 0, 1j], [0, 0, 0], [-1j, 0, 0]], dtype=np.complex128)
    Lz0 = np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]], dtype=np.complex128)
    perm = [base.index(x) for x in order_list]
    return tuple(mat[np.ix_(perm, perm)] for mat in (Lx0, Ly0, Lz0))


def atomic_p_soc_block(lambda_ev, order=WANNIER90_P_ORDER):
    """Return 6x6 lambda L.S block in spin-major p basis."""
    Lx, Ly, Lz = p_orbital_l_matrices(order=order)
    lam = float(lambda_ev)
    block = np.zeros((6, 6), dtype=np.complex128)
    block[:3, :3] = 0.5 * lam * Lz
    block[:3, 3:] = 0.5 * lam * (Lx - 1j * Ly)
    block[3:, :3] = 0.5 * lam * (Lx + 1j * Ly)
    block[3:, 3:] = -0.5 * lam * Lz
    return block


def d_orbital_l_matrices(order=WANNIER90_D_ORDER):
    """Return Lx,Ly,Lz in the Wannier90 real d-orbital basis.

    Default order follows Wannier90 real harmonics:
    (dz2, dxz, dyz, dx2-y2, dxy).  Matrices use hbar=1.
    """
    base = ["dz2", "dxz", "dyz", "dx2-y2", "dxy"]
    aliases = {
        "dz^2": "dz2",
        "d_z2": "dz2",
        "d_z^2": "dz2",
        "dx2y2": "dx2-y2",
        "dx2-y2": "dx2-y2",
        "dx^2-y^2": "dx2-y2",
        "d_x2-y2": "dx2-y2",
        "d_x^2-y^2": "dx2-y2",
    }
    order_list = [x.strip().lower() for x in str(order).replace(";", ",").split(",") if x.strip()]
    order_list = [aliases.get(x, x) for x in order_list]
    if not order_list:
        order_list = base
    if sorted(order_list) != sorted(base):
        raise ValueError(f"Unsupported d orbital order {order_list}; expected a permutation of {base}")
    rt3 = np.sqrt(3.0)
    Lx0 = np.array(
        [
            [0, 0, 1j * rt3, 0, 0],
            [0, 0, 0, 0, 1j],
            [-1j * rt3, 0, 0, -1j, 0],
            [0, 0, 1j, 0, 0],
            [0, -1j, 0, 0, 0],
        ],
        dtype=np.complex128,
    )
    Ly0 = np.array(
        [
            [0, -1j * rt3, 0, 0, 0],
            [1j * rt3, 0, 0, 1j, 0],
            [0, 0, 0, 0, 1j],
            [0, -1j, 0, 0, 0],
            [0, 0, -1j, 0, 0],
        ],
        dtype=np.complex128,
    )
    Lz0 = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, -1j, 0, 0],
            [0, 1j, 0, 0, 0],
            [0, 0, 0, 0, -2j],
            [0, 0, 0, 2j, 0],
        ],
        dtype=np.complex128,
    )
    perm = [base.index(x) for x in order_list]
    return tuple(mat[np.ix_(perm, perm)] for mat in (Lx0, Ly0, Lz0))


def atomic_d_soc_block(lambda_ev, order=WANNIER90_D_ORDER):
    """Return 10x10 lambda L.S block in spin-major d basis."""
    Lx, Ly, Lz = d_orbital_l_matrices(order=order)
    lam = float(lambda_ev)
    block = np.zeros((10, 10), dtype=np.complex128)
    block[:5, :5] = 0.5 * lam * Lz
    block[:5, 5:] = 0.5 * lam * (Lx - 1j * Ly)
    block[5:, :5] = 0.5 * lam * (Lx + 1j * Ly)
    block[5:, 5:] = -0.5 * lam * Lz
    return block


def atomic_soc_block_diagonal(l_values, lambda_ev):
    """Build spin-major atomic ``lambda L.S`` blocks for s, p, and d shells."""
    blocks = []
    for value in l_values:
        angular_momentum = int(value)
        if angular_momentum == 0:
            blocks.append(np.zeros((2, 2), dtype=np.complex128))
        elif angular_momentum == 1:
            blocks.append(atomic_p_soc_block(lambda_ev))
        elif angular_momentum == 2:
            blocks.append(atomic_d_soc_block(lambda_ev))
        else:
            raise ValueError(
                "Atomic SOC supports s, p, and d shells; "
                f"received l={angular_momentum}"
            )
    dimension = sum(block.shape[0] for block in blocks)
    result = np.zeros((dimension, dimension), dtype=np.complex128)
    offset = 0
    for block in blocks:
        stop = offset + block.shape[0]
        result[offset:stop, offset:stop] = block
        offset = stop
    return result


def _add_atomic_soc_block(h_spin, orbital_groups, soc_block, inplace=False):
    mat = h_spin if inplace else np.array(h_spin, dtype=np.complex128, copy=True)
    if mat.ndim < 2 or mat.shape[-1] != mat.shape[-2] or mat.shape[-1] % 2 != 0:
        raise ValueError(f"h_spin must have even square trailing shape, got {mat.shape}")
    nwan = mat.shape[-1] // 2
    norb = soc_block.shape[0] // 2
    for group in orbital_groups:
        oidx = np.asarray(group, dtype=np.int64).reshape(norb)
        if np.any(oidx < 0) or np.any(oidx >= nwan):
            raise ValueError(f"orbital indices out of range for nwan={nwan}: {oidx.tolist()}")
        idx = np.concatenate([oidx, oidx + nwan])
        mat[..., idx[:, None], idx[None, :]] += soc_block
    return mat


def add_atomic_p_soc(h_spin, p_orbital_groups, lambda_ev, order=WANNIER90_P_ORDER, inplace=False):
    """Add onsite atomic p-orbital SOC to a spin-major spinor matrix.

    p_orbital_groups is an iterable of 3 orbital indices in the non-spin
    orbital basis, e.g. [[10, 11, 12], [25, 26, 27]].
    """
    soc = atomic_p_soc_block(lambda_ev, order=order)
    return _add_atomic_soc_block(h_spin, p_orbital_groups, soc, inplace=inplace)


def add_atomic_d_soc(h_spin, d_orbital_groups, lambda_ev, order=WANNIER90_D_ORDER, inplace=False):
    """Add onsite atomic d-orbital SOC to a spin-major spinor matrix."""
    soc = atomic_d_soc_block(lambda_ev, order=order)
    return _add_atomic_soc_block(h_spin, d_orbital_groups, soc, inplace=inplace)


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
