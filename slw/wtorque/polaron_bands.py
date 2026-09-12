"""Finite-q magnon-polaron bands with explicit q/-q Nambu partners.

Energies and mode couplings are in eV.  The full projected coupling is the
coefficient of ``Psi_m(q)^dagger V(q) Psi_p(q)`` with ``Psi=(c(q),c†(-q))``.
It obeys ``V(q)=X_m V(-q)* X_p``.  In the combined Nambu Hamiltonian, using
the actual -q block is essential: pairing is symmetric across q and -q,
and need not be symmetric at one q.  No gap or stability shift is applied.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from slw.core.structure import read_wannier90_kpoint_path
from slw.wtorque.projection.polaron import assemble_rwa


def _energies(name: str, value: object) -> NDArray[np.float64]:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or result.size == 0 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite nonempty one-dimensional array")
    return result


def _swap(half: int) -> NDArray[np.int64]:
    return np.r_[np.arange(half, 2 * half), np.arange(half)]


def coupling_qpair_residual(full_nambu: object, full_nambu_minus: object) -> float:
    """Relative residual of the physical mode-coupling q-pair identity."""
    value = np.asarray(full_nambu, dtype=np.complex128)
    minus = np.asarray(full_nambu_minus, dtype=np.complex128)
    if (value.ndim != 2 or value.shape != minus.shape
            or any(size == 0 or size % 2 for size in value.shape)):
        raise ValueError("full_nambu q partners must share shape (2*nmag,2*nphonon)")
    if not np.all(np.isfinite(value)) or not np.all(np.isfinite(minus)):
        raise ValueError("full_nambu q partners must be finite")
    expected = minus[_swap(value.shape[0] // 2)][:, _swap(value.shape[1] // 2)].conj()
    return float(np.linalg.norm(value - expected) / max(
        np.linalg.norm(value), np.linalg.norm(minus), np.finfo(float).tiny,
    ))


def assemble_qpair_bdg(
    magnon_energies: object,
    phonon_energies: object,
    full_nambu: object,
    magnon_energies_minus: object,
    phonon_energies_minus: object,
    full_nambu_minus: object,
    *,
    qpair_tolerance: float = 1.0e-8,
) -> NDArray[np.complex128]:
    """Return the Hessian in ``(a_q,b_q,a†_-q,b†_-q)`` order.

    The quadratic Hamiltonian is ``1/2 sum_q Psi† H Psi``.  Populating both
    cross blocks with V and V† accounts for that half without dividing V.
    The supplied -q coupling is checked, never silently averaged with q.
    """
    em = _energies("magnon energies", magnon_energies)
    ep = _energies("phonon energies", phonon_energies)
    emm = _energies("minus-q magnon energies", magnon_energies_minus)
    epm = _energies("minus-q phonon energies", phonon_energies_minus)
    if em.shape != emm.shape or ep.shape != epm.shape:
        raise ValueError("q and -q energy dimensions differ")
    if not np.isfinite(qpair_tolerance) or qpair_tolerance < 0:
        raise ValueError("qpair_tolerance must be finite and nonnegative")
    value = np.asarray(full_nambu, dtype=np.complex128)
    if value.shape != (2 * em.size, 2 * ep.size):
        raise ValueError("full_nambu dimensions do not match energies")
    residual = coupling_qpair_residual(value, full_nambu_minus)
    if residual > qpair_tolerance:
        raise ValueError(f"mode-coupling q-pair residual {residual:.6g} exceeds {qpair_tolerance:.6g}")
    count = em.size + ep.size
    result = np.diag(np.r_[em, ep, emm, epm]).astype(np.complex128)
    im = np.r_[np.arange(em.size), count + np.arange(em.size)]
    ip = np.r_[em.size + np.arange(ep.size), count + em.size + np.arange(ep.size)]
    result[np.ix_(im, ip)] = value
    result[np.ix_(ip, im)] = value.conj().T
    return result


@dataclass(frozen=True)
class PolaronBands:
    energies_eV: NDArray[np.float64]
    magnon_weights: NDArray[np.float64]
    eigenvectors: NDArray[np.complex128]
    rwa_energies_eV: NDArray[np.float64]
    dynamic_eigenvalues_eV: NDArray[np.complex128]
    minimum_hessian_eigenvalue_eV: NDArray[np.float64]
    maximum_imaginary_eigenvalue_eV: NDArray[np.float64]
    qpair_residual: NDArray[np.float64]
    paraunitarity_residual: NDArray[np.float64]
    eigen_residual_eV: NDArray[np.float64]
    valid: NDArray[np.bool_]
    status: tuple[str, ...]
    mode_valid: NDArray[np.bool_] | None = None
    translation_zero_mask: NDArray[np.bool_] | None = None


def solve_polaron_bands(
    magnon_energies: object,
    phonon_energies: object,
    full_nambu: object,
    magnon_energies_minus: object,
    phonon_energies_minus: object,
    full_nambu_minus: object,
    *,
    valid_q_mask: object | None = None,
    qpair_tolerance: float = 1.0e-8,
    stability_tolerance_eV: float = 1.0e-10,
    imaginary_tolerance_eV: float = 1.0e-10,
    phonon_translation_mask: object | None = None,
) -> PolaronBands:
    """Solve positive-norm bands; preserve unstable/zero modes as diagnostics.

    All arguments have a leading q axis; the minus arrays must refer to the
    actual -q in compatible magnon and phonon gauges.  Caller-masked endpoints
    can contain NaNs.  Unstable and zero-mode points return NaN bands, with
    the reason and unmodified dynamic spectrum retained.  For positive
    Hessians the Hermitian Cholesky reduction produces a complete metric-
    normalized eigensystem, including degenerate subspaces.

    ``magnon_weights`` is the bounded Euclidean Nambu component fraction,
    ``sum_m(|u|²+|v|²)/sum_all(|u|²+|v|²)``.  It is a mode character, not a
    neutron spectral intensity or a signed bosonic-metric fraction.
    """
    em, ep, emm, epm = (np.asarray(v, dtype=np.float64) for v in (
        magnon_energies, phonon_energies, magnon_energies_minus, phonon_energies_minus,
    ))
    vq, vm = (np.asarray(v, dtype=np.complex128) for v in (full_nambu, full_nambu_minus))
    if em.ndim != 2 or ep.ndim != 2 or em.shape != emm.shape or ep.shape != epm.shape:
        raise ValueError("energies must have compatible (nq,nmode) q-pair dimensions")
    nq, nm = em.shape
    np_ = ep.shape[1]
    if not nq or not nm or not np_ or ep.shape[0] != nq:
        raise ValueError("energy dimensions must be nonempty with matching nq")
    if vq.shape != (nq, 2 * nm, 2 * np_) or vm.shape != vq.shape:
        raise ValueError("full_nambu must have shape (nq,2*nmag,2*nphonon)")
    for name, value in (("stability_tolerance_eV", stability_tolerance_eV),
                        ("imaginary_tolerance_eV", imaginary_tolerance_eV)):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    valid = np.ones(nq, dtype=bool) if valid_q_mask is None else np.asarray(valid_q_mask, dtype=bool).copy()
    if valid.shape != (nq,):
        raise ValueError("valid_q_mask must have shape (nq,)")
    if phonon_translation_mask is not None:
        translations = np.asarray(phonon_translation_mask, bool)
        if translations.shape != ep.shape:
            raise ValueError("phonon_translation_mask must have shape (nq,nphonon)")
        if np.any(translations & valid[:, None]):
            return _solve_with_translations(
                em, ep, vq, emm, epm, vm, valid, translations,
                qpair_tolerance=qpair_tolerance, stability_tolerance_eV=stability_tolerance_eV,
                imaginary_tolerance_eV=imaginary_tolerance_eV,
            )
    count = nm + np_
    bands = np.full((nq, count), np.nan)
    weights = bands.copy()
    rwa = bands.copy()
    eigenvectors = np.full((nq, 2 * count, count), np.nan + 0j)
    dynamic = np.full((nq, 2 * count), np.nan + 0j)
    hminimum, imax, qres, pres, eres = (np.full(nq, np.nan) for _ in range(5))
    metric = np.r_[np.ones(count), -np.ones(count)]
    magnon_indices = np.r_[np.arange(nm), count + np.arange(nm)]
    status = []
    for iq in range(nq):
        if not valid[iq]:
            status.append("masked_by_caller")
            continue
        hessian = assemble_qpair_bdg(
            em[iq], ep[iq], vq[iq], emm[iq], epm[iq], vm[iq],
            qpair_tolerance=qpair_tolerance,
        )
        qres[iq] = coupling_qpair_residual(vq[iq], vm[iq])
        hminimum[iq] = np.linalg.eigvalsh(hessian)[0]
        eigenvalues = np.linalg.eigvals(metric[:, None] * hessian)
        dynamic[iq] = eigenvalues[np.lexsort((eigenvalues.imag, eigenvalues.real))]
        imax[iq] = np.max(np.abs(eigenvalues.imag))
        rwa[iq] = np.linalg.eigvalsh(assemble_rwa(em[iq], ep[iq], vq[iq, :nm, :np_]))
        if hminimum[iq] < -stability_tolerance_eV or imax[iq] > imaginary_tolerance_eV:
            valid[iq] = False
            status.append("unstable")
            continue
        if hminimum[iq] <= stability_tolerance_eV:
            valid[iq] = False
            status.append("zero_or_unresolved_mode")
            continue
        lower = np.linalg.cholesky(hessian)
        hermitian_dynamic = lower.conj().T @ (metric[:, None] * lower)
        energies, unitary = np.linalg.eigh(hermitian_dynamic)
        full_vectors = np.linalg.solve(lower.conj().T, unitary) * np.sqrt(np.abs(energies))[None, :]
        positive = np.flatnonzero(energies > 0)
        if positive.size != count:
            raise ValueError("positive Hessian produced an invalid bosonic mode count")
        selected = full_vectors[:, positive]
        eigenvectors[iq] = selected
        bands[iq] = energies[positive]
        full_norm = np.sum(np.abs(selected) ** 2, axis=0)
        weights[iq] = np.sum(np.abs(selected[magnon_indices]) ** 2, axis=0) / full_norm
        pres[iq] = np.max(np.abs(full_vectors.conj().T @ (metric[:, None] * full_vectors)
                                  - np.diag(np.sign(energies))))
        eres[iq] = np.max(np.abs(metric[:, None] * hessian @ selected - selected * bands[iq]))
        status.append("stable")
    return PolaronBands(bands, weights, eigenvectors, rwa, dynamic, hminimum, imax,
                        qres, pres, eres, valid, tuple(status),
                        np.broadcast_to(valid[:, None], bands.shape).copy(), np.zeros_like(bands, bool))


def _solve_with_translations(em, ep, vq, emm, epm, vm, valid, translations, **tolerances):
    """Solve finite oscillators and retain three unnormalized zero endpoints.

    No eigenvector or phonon AM is assigned to a free rigid translation.
    Finite-mode vectors are embedded back into the original Nambu rows.
    """
    from dataclasses import fields
    nq, nm = em.shape
    np_ = ep.shape[1]; n = nm+np_
    base = solve_polaron_bands(em, ep, vq, emm, epm, vm,
        valid_q_mask=np.zeros(nq, bool), **tolerances)
    arrays = {f.name: np.array(getattr(base, f.name), copy=True)
              for f in fields(base) if f.name != "status"}
    status = list(base.status)
    for iq in np.flatnonzero(valid):
        removed = np.flatnonzero(translations[iq]); active = np.flatnonzero(~translations[iq])
        if len(removed) not in (0, 3):
            raise ValueError("Gamma elimination requires exactly three rigid translations")
        if np.any(ep[iq, removed] != 0) or np.any(epm[iq, removed] != 0):
            raise ValueError("removed translation energies must be exactly zero at both q signs")
        inactive_columns = np.r_[removed, np_+removed]
        if len(removed) and (np.max(abs(vq[iq][:, inactive_columns])) > 1.e-14
                            or np.max(abs(vm[iq][:, inactive_columns])) > 1.e-14):
            raise ValueError("rigid translations must decouple before the finite-mode BdG solve")
        columns = np.r_[active, np_+active]
        result = solve_polaron_bands(em[iq:iq+1], ep[iq:iq+1, active], vq[iq:iq+1, :, columns],
            emm[iq:iq+1], epm[iq:iq+1, active], vm[iq:iq+1, :, columns], **tolerances)
        status[iq] = result.status[0]
        # Low three columns are energy-only acoustic endpoints, not bosons.
        offset = len(removed)
        target = np.arange(offset, n)
        rows = np.r_[np.arange(nm), nm+active, n+np.arange(nm), n+nm+active]
        for name in ('energies_eV', 'magnon_weights', 'rwa_energies_eV', 'mode_valid'):
            arrays[name][iq, target] = getattr(result, name)[0]
        arrays['eigenvectors'][iq][:, target] = 0.
        arrays['eigenvectors'][iq][np.ix_(rows, target)] = result.eigenvectors[0]
        for name in ('minimum_hessian_eigenvalue_eV', 'maximum_imaginary_eigenvalue_eV',
                     'qpair_residual', 'paraunitarity_residual', 'eigen_residual_eV', 'valid'):
            arrays[name][iq] = getattr(result, name)[0]
        dynamic = np.r_[result.dynamic_eigenvalues_eV[0], np.zeros(2*offset)]
        arrays['dynamic_eigenvalues_eV'][iq] = dynamic[np.lexsort((dynamic.imag, dynamic.real))]
        if result.valid[0] and offset:
            arrays['energies_eV'][iq, :offset] = 0.
            arrays['rwa_energies_eV'][iq, :offset] = 0.
            arrays['translation_zero_mask'][iq, :offset] = True
            arrays['minimum_hessian_eigenvalue_eV'][iq] = min(0., result.minimum_hessian_eigenvalue_eV[0])
            status[iq] = "stable_optical_with_rigid_translations"
    return PolaronBands(**arrays, status=tuple(status))


@dataclass(frozen=True)
class SampledBandPath:
    qpoints: NDArray[np.float64]
    distance_inv_ang: NDArray[np.float64]
    segment_index: NDArray[np.int64]
    tick_positions: NDArray[np.float64]
    tick_labels: tuple[str, ...]
    gamma_mask: NDArray[np.bool_]


@dataclass(frozen=True)
class NativePathModes:
    """Paired native modes plus full-path arrays containing explicit NaN holes.

    The native objects contain ``[q[valid_indices], -q[valid_indices]]``;
    they retain masses, frames, atom positions, and source diagnostics.
    """
    valid_indices: NDArray[np.int64]
    valid: NDArray[np.bool_]
    status: tuple[str, ...]
    magnons: object
    phonons: object
    magnon_energies_eV: NDArray[np.float64]
    phonon_energies_eV: NDArray[np.float64]
    magnon_energies_minus_eV: NDArray[np.float64]
    phonon_energies_minus_eV: NDArray[np.float64]
    magnon_transform: NDArray[np.complex128]
    phonon_eigenvectors: NDArray[np.complex128]
    magnon_transform_minus: NDArray[np.complex128]
    phonon_eigenvectors_minus: NDArray[np.complex128]


def native_path_modes(
    epr_path: str | Path,
    exchange_path: str | Path,
    qpoints: object,
    *,
    spin_lengths: object,
    source_directed_bond_weight: float,
    magnetic_atom_labels: tuple[str, ...] | list[str] | None = None,
    gauge: str = "atomic",
    loto_mode: str = "auto",
    valid_q_mask: object | None = None,
    anisotropy_mev: object | None = None,
    anisotropy_spin_normalization: str | None = None,
    gamma_policy: str = "omit",
    gamma_asr_tolerance: float = 1.e-6,
) -> NativePathModes:
    """Batch q/-q modes, with an opt-in finite-mode treatment of Gamma.

    A zero-point normalized acoustic mode or a metric-normalized AFM
    Goldstone eigenvector is undefined at Γ. By default these points are
    masked. ``gamma_policy='optical'`` separates rigid translations and
    requires finite magnon modes, optionally from input anisotropy.
    No zero-mode energy is shifted. A batch with additional unstable/zero modes falls back
    to individual checks to record their locations.  Source, gauge, and
    other input errors still raise.  Independent q subsets may be passed
    on separate MPI ranks by the workflow owner.
    """
    from slw.wtorque.native_modes import native_magnon_modes, native_phonon_modes

    points = np.asarray(qpoints, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not points.shape[0] or not np.all(np.isfinite(points)):
        raise ValueError("qpoints must have finite nonempty shape (nq,3)")
    valid = np.ones(points.shape[0], dtype=bool) if valid_q_mask is None else np.asarray(valid_q_mask, dtype=bool).copy()
    if valid.shape != (points.shape[0],):
        raise ValueError("valid_q_mask must have shape (nq,)")
    status = ["stable" if value else "masked_by_caller" for value in valid]
    gamma = np.max(np.abs(points - np.rint(points)), axis=1) < 1.e-12
    if gamma_policy not in {"omit", "optical"}:
        raise ValueError("gamma_policy must be omit or optical")
    for index in np.flatnonzero(gamma & valid):
        status[index] = "gamma_zero_mode_omitted" if gamma_policy == "omit" else "gamma_optical_with_rigid_translations"
    if gamma_policy == "omit":
        valid &= ~gamma

    def evaluate(indices):
        paired = np.concatenate((points[indices], -points[indices]), axis=0)
        magnons = native_magnon_modes(
            exchange_path, paired, spin_lengths=spin_lengths,
            source_directed_bond_weight=source_directed_bond_weight,
            magnetic_atom_labels=magnetic_atom_labels, gauge=gauge,
            anisotropy_mev=anisotropy_mev,
            anisotropy_spin_normalization=anisotropy_spin_normalization,
        )
        phonons = native_phonon_modes(epr_path, paired, gauge=gauge, loto_mode=loto_mode,
            gamma_policy="optical" if gamma_policy == "optical" else "reject",
            gamma_asr_tolerance=gamma_asr_tolerance)
        return magnons, phonons

    def physical_mode_error(exc):
        message = str(exc).lower()
        return any(term in message for term in (
            "nonpositive native phonon mode", "zero or unstable magnon modes",
            "goldstone", "unstable mode", "hessian is unstable", "unstable metric energy",
        ))

    indices = np.flatnonzero(valid)
    if not indices.size:
        raise ValueError("path has no unmasked finite-q points for vertex projection")
    try:
        magnons, phonons = evaluate(indices)
    except ValueError as exc:
        if not physical_mode_error(exc):
            raise
        for index in indices:
            try:
                evaluate(np.asarray([index]))
            except ValueError as point_error:
                if not physical_mode_error(point_error):
                    raise
                valid[index] = False
                status[index] = "unprojectable_mode: " + str(point_error)
        indices = np.flatnonzero(valid)
        if not indices.size:
            raise ValueError("all finite-q path modes are unstable or singular") from exc
        magnons, phonons = evaluate(indices)

    def expand(values):
        plus, minus = np.split(values, 2, axis=0)
        target_shape = (points.shape[0], *values.shape[1:])
        full = np.full(target_shape, np.nan, dtype=values.dtype)
        full_minus = full.copy()
        full[indices], full_minus[indices] = plus, minus
        return full, full_minus

    em, emm = expand(magnons.energies_eV)
    ep, epm = expand(phonons.energies_eV)
    tm, tmm = expand(magnons.transform)
    tp, tpm = expand(phonons.eigenvectors)
    return NativePathModes(indices, valid, tuple(status), magnons, phonons,
                           em, ep, emm, epm, tm, tp, tmm, tpm)


def sample_wannier_path(
    win_path: str | Path,
    lattice_ang: object,
    *,
    points_per_segment: int = 40,
) -> SampledBandPath:
    """Sample the input cell's Wannier90 path, retaining disconnected pieces.

    ``points_per_segment`` counts intervals (both endpoints are retained).
    Shared endpoints are repeated deliberately so every segment is an
    independent plotting unit; reciprocal lattice vectors include 2 pi.
    """
    lattice = np.asarray(lattice_ang, dtype=np.float64)
    if lattice.shape != (3, 3) or not np.all(np.isfinite(lattice)):
        raise ValueError("lattice_ang must contain three finite row lattice vectors")
    if isinstance(points_per_segment, bool) or int(points_per_segment) != points_per_segment or points_per_segment < 1:
        raise ValueError("points_per_segment must be a positive integer")
    reciprocal = 2 * np.pi * np.linalg.inv(lattice).T
    qpoints, distances, ids, ticks, labels = [], [], [], [], []
    offset = 0.0
    previous_end = None
    for index, (left_label, left, right_label, right) in enumerate(read_wannier90_kpoint_path(win_path)):
        length = float(np.linalg.norm((right - left) @ reciprocal))
        if not np.isfinite(length) or length <= 0:
            raise ValueError(f"path segment {index} must have finite nonzero length")
        qpoints.append(np.linspace(left, right, int(points_per_segment) + 1))
        distances.append(offset + np.linspace(0, length, int(points_per_segment) + 1))
        ids.extend([index] * (int(points_per_segment) + 1))
        if index == 0:
            ticks.append(offset)
            labels.append(left_label)
        elif not np.allclose(left, previous_end, atol=1.0e-12, rtol=0) or labels[-1] != left_label:
            labels[-1] = labels[-1] + " | " + left_label
        offset += length
        ticks.append(offset)
        labels.append(right_label)
        previous_end = right
    points = np.concatenate(qpoints)
    gamma_mask = np.max(np.abs(points - np.rint(points)), axis=1) < 1.0e-12
    return SampledBandPath(points, np.concatenate(distances), np.asarray(ids, dtype=np.int64),
                           np.asarray(ticks), tuple(labels), gamma_mask)


def plot_polaron_bands(
    path: SampledBandPath,
    bands: PolaronBands,
    output_path: str | Path,
    *,
    magnon_energies: object | None = None,
    phonon_energies: object | None = None,
    energy_limits_meV: tuple[float, float] | None = None,
    title: str = "Magnon–polaron bands",
) -> None:
    """Save a standalone figure with mode-character color and no gap bridging."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    if path.qpoints.shape[0] != bands.energies_eV.shape[0]:
        raise ValueError("path and band q counts differ")
    fig, ax = plt.subplots(figsize=(9.0, 5.3), constrained_layout=True)
    valid_edges = (path.segment_index[1:] == path.segment_index[:-1]) & bands.valid[1:] & bands.valid[:-1]
    for bare, color in ((phonon_energies, "0.75"), (magnon_energies, "0.45")):
        if bare is None:
            continue
        values = np.asarray(bare, dtype=float)
        if values.ndim != 2 or values.shape[0] != path.qpoints.shape[0]:
            raise ValueError("bare energies must have shape (nq,nmode)")
        for segment in np.unique(path.segment_index):
            selected = path.segment_index == segment
            ax.plot(path.distance_inv_ang[selected], values[selected] * 1.e3,
                    color=color, linewidth=0.7, linestyle="--", zorder=1)
    norm = Normalize(0, 1)
    collection = None
    for branch in range(bands.energies_eV.shape[1]):
        vertices = np.column_stack((path.distance_inv_ang, bands.energies_eV[:, branch] * 1.e3))
        lines = np.stack((vertices[:-1], vertices[1:]), axis=1)[valid_edges]
        weights = bands.magnon_weights[:, branch].copy()
        # Energy-only acoustic endpoints have no normalized boson character.
        # Their plotting color may inherit the adjoining finite-q value.
        if bands.translation_zero_mask is not None:
            for iq in np.flatnonzero(bands.translation_zero_mask[:, branch]):
                neighbors = [j for j in (iq-1, iq+1) if 0 <= j < len(weights)
                             and path.segment_index[j] == path.segment_index[iq]
                             and np.isfinite(weights[j])]
                weights[iq] = np.mean(weights[neighbors]) if neighbors else 0.
        color = ((weights[:-1] + weights[1:]) / 2)[valid_edges]
        collection = LineCollection(lines, cmap="coolwarm", norm=norm, linewidth=1.35, zorder=2)
        collection.set_array(color)
        ax.add_collection(collection)
    ax.set_xticks(path.tick_positions, [label.replace("G", r"$\Gamma$") if label == "G" else label for label in path.tick_labels])
    for tick in path.tick_positions:
        ax.axvline(tick, color="0.85", linewidth=0.7, zorder=0)
    xmin, xmax = float(path.distance_inv_ang.min()), float(path.distance_inv_ang.max())
    ax.set_xlim((xmin, xmax) if xmax > xmin else (xmin - .5, xmax + .5))
    if path.qpoints.shape[0] == 1 and bands.valid[0]:
        ax.scatter(np.full(bands.energies_eV.shape[1], xmin), bands.energies_eV[0] * 1.e3,
                   c=bands.magnon_weights[0], cmap="coolwarm", norm=norm, s=12, zorder=3)
    finite = bands.energies_eV[np.isfinite(bands.energies_eV)]
    if energy_limits_meV is None and finite.size:
        ax.set_ylim(min(0.0, float(finite.min() * 1.e3)), float(finite.max() * 1.e3) * 1.04)
    elif energy_limits_meV is not None:
        ax.set_ylim(*energy_limits_meV)
    ax.set_ylabel("Energy (meV)")
    ax.set_title(title)
    if collection is not None:
        fig.colorbar(collection, ax=ax, label="Magnon Nambu component fraction", pad=0.02)
    target = Path(output_path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=180)
    plt.close(fig)


__all__ = ["NativePathModes", "PolaronBands", "SampledBandPath", "assemble_qpair_bdg",
           "coupling_qpair_residual", "native_path_modes", "plot_polaron_bands", "sample_wannier_path",
           "solve_polaron_bands"]
