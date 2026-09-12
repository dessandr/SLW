"""Fixed-frame derivative of the spinor TB2J exchange functional.

H and dH come from one full spinor EPR, including spin-flip matrix elements.
An explicitly selected frame defines the local orbital projectors. Both electronic
endpoints are transformed to its *cell-periodic* trial basis, then reordered
to the spin-major convention used by the exchange kernel. This is a frozen
projector/local-direction force-theorem model, not a self-consistent dJ.

For an existing TB2J spinor-Wannier model, identity endpoint frames retain its
interleaved Wannier-pair Pauli definition. AMN frames define a different local
spin model and must not be substituted without comparing the static exchange.
"""
from __future__ import annotations

import numpy as np

from slw.exchange.kernels.dj_tensor_epr import (
    _build_local_tb2j_projector_deriv_from_block,
    _build_local_tb2j_projector_from_block,
    _compute_band_dg,
    _compute_tensor_chunk_analytic,
    _local_tb2j_projector_direction,
    _pauli_block_all,
)


def uniform_points(mesh):
    raw = np.asarray(mesh)
    if raw.shape != (3,) or not np.issubdtype(raw.dtype, np.integer) or np.any(raw < 1):
        raise ValueError("mesh must contain three positive integers")
    return np.indices(tuple(raw)).reshape(3, -1).T / raw


def commensurate_map(kpoints, qpoint, mesh):
    """Exact k+q index for C-ordered unshifted k grid; retain actual q."""
    k, q, mesh = np.asarray(kpoints), np.asarray(qpoint), np.asarray(mesh)
    if k.shape != (int(np.prod(mesh)), 3) or not np.allclose(k, uniform_points(mesh)):
        raise ValueError("kpoints must be the complete C-ordered uniform mesh")
    scaled = (k + q) * mesh
    if not np.allclose(scaled, np.rint(scaled), atol=1.e-7, rtol=0):
        raise ValueError("q is not commensurate with the electronic k mesh")
    return np.ravel_multi_index(tuple((np.rint(scaled).astype(int) % mesh).T), tuple(mesh))


def periodic_spinor_transform(operator, source_frame, final_frame):
    """Q(k+q)^dagger O Q(k), interleaved -> spin-major; no ionic phase."""
    op = np.asarray(operator, complex)
    source, final = np.asarray(source_frame, complex), np.asarray(final_frame, complex)
    if source.shape != final.shape or source.ndim != 3 or source.shape[-1] % 2:
        raise ValueError("endpoint frames must have matching even square dimensions")
    if source.shape[-1] != source.shape[-2] or op.shape[0] != len(source) or op.shape[-2:] != source.shape[-2:]:
        raise ValueError("operator and endpoint frame dimensions disagree")
    expansion = (slice(None),) + (None,) * (op.ndim - 3)
    value = final[expansion].conj().swapaxes(-1, -2) @ op @ source[expansion]
    n = source.shape[-1]
    order = np.r_[np.arange(0, n, 2), np.arange(1, n, 2)]
    return value[..., order[:, None], order]


def prepare_kdata(h_spin, orbital_masks, fermi_energy_eV):
    """Prepare all local projectors, allowing noncontiguous orbital masks."""
    h = np.asarray(h_spin, complex)
    nw = h.shape[-1] // 2
    if h.ndim != 3 or h.shape[-2:] != (2 * nw, 2 * nw):
        raise ValueError("H must have shape (nk,2*norb,2*norb)")
    if np.max(abs(h - h.conj().swapaxes(-1, -2))) > 1.e-8:
        raise ValueError("H is not Hermitian")
    masks = np.asarray(orbital_masks)
    if masks.ndim != 2 or masks.shape[1] != nw or not np.all((masks == 0) | (masks == 1)):
        raise ValueError("orbital masks must be binary with shape (nsite,norb)")
    if np.any(masks.sum(axis=0) > 1) or np.any(masks.sum(axis=1) == 0):
        raise ValueError("magnetic orbital masks must be nonempty and disjoint")
    energies, vectors = np.linalg.eigh(h)
    result = dict(evals=energies-float(fermi_energy_eV), evecs=vectors,
                  site_coeffs={}, site_indices={}, p_ops={}, p_dirs={})
    onsite = h.mean(axis=0)
    for site, mask in enumerate(masks):
        orbitals = np.flatnonzero(mask)
        indices = np.r_[orbitals, orbitals + nw]
        block = onsite[np.ix_(indices, indices)]
        result["site_indices"][site] = indices
        result["site_coeffs"][site] = vectors[:, indices]
        result["p_ops"][site] = _build_local_tb2j_projector_from_block(block)
        result["p_dirs"][site] = _local_tb2j_projector_direction(block)
    return result


def wannier_pair_masks(wannier_centers, magnetic_positions, lattice_columns,
                       *, maximum_pair_distance_ang, site_radius_ang):
    """Validate an explicitly declared interleaved Wannier-pair spin model.

    Geometry checks pairing and site membership, not physical spin purity.
    This reproduces the local Pauli assumption of the spinor TB2J model;
    static exchange agreement is a separate required workflow check.
    """
    centers, sites, lattice = map(np.asarray, (wannier_centers, magnetic_positions, lattice_columns))
    if centers.ndim != 2 or centers.shape[1] != 3 or len(centers)%2 or sites.ndim != 2 or sites.shape[1] != 3 or lattice.shape != (3,3):
        raise ValueError("invalid Wannier pair/site geometry")
    if not np.isfinite([maximum_pair_distance_ang,site_radius_ang]).all() or min(maximum_pair_distance_ang,site_radius_ang)<=0:
        raise ValueError("pair distance and site radius must be finite and positive")
    # Cell representatives must be aligned to those in the static TB2J model.
    # No independent minimum-image shifts of paired Wannier functions.
    if np.max(np.linalg.norm((centers[::2]-centers[1::2])@lattice.T,axis=1))>maximum_pair_distance_ang:
        raise ValueError("declared neighboring Wannier spin partners are not colocated")
    midpoint=.5*(centers[::2]+centers[1::2])
    distance=np.linalg.norm((sites[:,None,:]-midpoint[None,:,:])@lattice.T,axis=-1)
    masks=distance<site_radius_ang
    if np.any(masks.sum(axis=1)==0) or np.any(masks.sum(axis=0)>1):
        raise ValueError("Wannier site projectors are empty or overlap")
    return masks


def semicircle_gauss(energy_min_eV, energy_max_eV, count):
    """Upper-half-plane contour from lower to upper, Gauss-Legendre weights."""
    if not np.isfinite([energy_min_eV, energy_max_eV]).all() or energy_min_eV >= energy_max_eV or count < 2:
        raise ValueError("invalid contour bounds or quadrature count")
    x, w = np.polynomial.legendre.leggauss(count)
    theta = (1. - x) * np.pi / 2
    radius = (energy_max_eV - energy_min_eV) / 2
    center = (energy_max_eV + energy_min_eV) / 2
    z = center + radius * np.exp(1j * theta)
    dz = -1j * radius * np.exp(1j * theta) * np.pi / 2 * w
    return list(zip(z, dz))


def static_exchange_A(kdata, kpoints, pair_meta, contour):
    """Static A_uv in eV, same projectors/contour as its analytic derivative."""
    phase = np.exp(-2j*np.pi*np.asarray([p["R"] for p in pair_meta]) @ np.asarray(kpoints).T)
    out = np.zeros((len(pair_meta), 4, 4), complex)
    coeff = kdata["site_coeffs"]
    for z, dz in contour:
        inv = 1 / (z - kdata["evals"])
        blocks = {(i, j): (ci * inv[:, None, :]) @ cj.conj().swapaxes(-1, -2)
                  for i, ci in coeff.items() for j, cj in coeff.items()}
        for b, pair in enumerate(pair_meta):
            i, j = pair["li"], pair["lj"]
            gij = np.einsum("k,kij->ij", phase[b]/len(kpoints), blocks[i, j])
            gji = np.einsum("k,kij->ij", phase[b].conj()/len(kpoints), blocks[j, i])
            x = kdata["p_ops"][i] @ np.asarray(_pauli_block_all(gij))
            y = kdata["p_ops"][j] @ np.asarray(_pauli_block_all(gji))
            out[b] += np.einsum("uij,vji->uv", x, y) * dz/np.pi
    return out


def derivative_A(kdata, g_spin, kpoints, qpoint, kmesh, pair_meta, contour):
    """Return dA_uv(q) in eV/Angstrom for one Cartesian perturbation.

    All magnetic sites receive the k-average local exchange-field response,
    including when the displaced atom is nonmagnetic. Projector directions
    and the AMN frame are frozen. Imaginary parts must be taken *after* the
    inverse q transform, since dA(q) itself need not be real.
    """
    mapping = commensurate_map(kpoints, qpoint, kmesh)[None]
    g = np.asarray(g_spin, complex)[None]
    band = _compute_band_dg(g, kdata["evecs"], mapping)
    onsite = g[0].mean(axis=0)
    dp = {}
    for site, indices in kdata["site_indices"].items():
        dp[site] = _build_local_tb2j_projector_deriv_from_block(
            onsite[np.ix_(indices, indices)], kdata["p_dirs"][site])[None]
    bonds = np.asarray([p["R"] for p in pair_meta])
    phase = np.exp(-2j*np.pi*bonds @ np.asarray(kpoints).T)
    endpoint = np.exp(2j*np.pi*np.asarray(qpoint) @ bonds.T)[None]
    return _compute_tensor_chunk_analytic(
        contour, kdata, band, None, -1, mapping, pair_meta, phase,
        endpoint, 1/len(kpoints), ("x", "y", "z"),
        onsite_projector_derivatives=dp,
    )[0]


def isotropic_from_A(real_space_A):
    a = np.asarray(real_space_A)
    return 1000 * np.imag(a[..., 0, 0] - a[..., 1, 1] - a[..., 2, 2] - a[..., 3, 3])
