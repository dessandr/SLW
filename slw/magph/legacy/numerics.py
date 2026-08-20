import numpy as np
from numba import njit, prange

from .kernels import HBAR_MEV_PS, bose_function


def bosonic_metric_diag(nchannel, physical_count=None):
    """Return the diagonal BdG metric for particle-then-hole ordering."""
    nchannel = int(nchannel)
    if nchannel < 1:
        raise ValueError(f"nchannel must be positive, got {nchannel}")
    if physical_count is None:
        if nchannel % 2:
            raise ValueError(
                "physical_count is required for an odd number of BdG channels"
            )
        physical_count = nchannel // 2
    physical_count = int(physical_count)
    if physical_count < 0 or physical_count > nchannel:
        raise ValueError(
            f"physical_count={physical_count} is invalid for nchannel={nchannel}"
        )
    metric = -np.ones(nchannel, dtype=np.float64)
    metric[:physical_count] = 1.0
    return metric


def linewidth_observables(gamma_hwhm_mev):
    """Convert ``gamma=-Im Sigma^R`` (HWHM, meV) to physical observables.

    Negative damping is retained as NaN rather than silently clipped.  A
    strictly zero linewidth has zero scattering rate and infinite lifetime.
    """
    gamma = np.asarray(gamma_hwhm_mev, dtype=np.float64)
    valid = np.isfinite(gamma) & (gamma >= 0.0)
    positive = valid & (gamma > 0.0)
    zero = valid & (gamma == 0.0)

    fwhm = np.full(gamma.shape, np.nan, dtype=np.float64)
    rate = np.full(gamma.shape, np.nan, dtype=np.float64)
    lifetime = np.full(gamma.shape, np.nan, dtype=np.float64)
    fwhm[valid] = 2.0 * gamma[valid]
    rate[valid] = fwhm[valid] / HBAR_MEV_PS
    lifetime[positive] = HBAR_MEV_PS / fwhm[positive]
    lifetime[zero] = np.inf
    return {
        "gamma_hwhm_mev": gamma.copy(),
        "fwhm_mev": fwhm,
        "lifetime_ps": lifetime,
        "scattering_rate_ps_inv": rate,
        "valid_damping": valid,
    }


def retarded_damping_matrix(sigma_retarded):
    r"""Return ``-(Sigma^R-Sigma^{R\dagger})/(2i)``.

    Unlike elementwise ``-imag(Sigma)``, this is the Hermitian dissipative
    matrix required when off-diagonal self-energy elements are complex.
    """
    sigma = np.asarray(sigma_retarded, dtype=np.complex128)
    if sigma.ndim < 2 or sigma.shape[-1] != sigma.shape[-2]:
        raise ValueError(f"sigma_retarded must end in square axes, got {sigma.shape}")
    sigma_dagger = np.swapaxes(np.conjugate(sigma), -1, -2)
    damping = 0.5j * (sigma - sigma_dagger)
    return 0.5 * (damping + np.swapaxes(np.conjugate(damping), -1, -2))


def paraunitary_residual(transform, metric_diag):
    r"""Return ``U^\dagger eta U - eta`` for a square BdG transform."""
    matrix = np.asarray(transform, dtype=np.complex128)
    metric = np.asarray(metric_diag, dtype=np.float64).reshape(-1)
    if matrix.shape != (metric.size, metric.size):
        raise ValueError(
            f"transform shape {matrix.shape} is incompatible with metric size {metric.size}"
        )
    eta = np.diag(metric.astype(np.complex128))
    return matrix.conj().T @ eta @ matrix - eta


@njit(parallel=True, fastmath=True)
def _compute_linewidth_vectorized(
    g_pq_cache,
    e_kq_cache,
    ph_en_flat,
    e_mag_k,
    T,
    eta,
    total_q,
    metric_diag,
):
    num_modes = ph_en_flat.shape[1]
    nchannel = e_mag_k.shape[0]
    linewidth = np.zeros(nchannel, dtype=np.float64)

    for iq in prange(total_q):
        local_lw = np.zeros(nchannel, dtype=np.float64)
        for nu in range(num_modes):
            w_ph = ph_en_flat[iq, nu]
            nb = bose_function(w_ph, T)
            for m in range(nchannel):
                omega = e_mag_k[m]
                for l in range(nchannel):
                    e_m_kq = e_kq_cache[iq, l]
                    nm = bose_function(e_m_kq, T)
                    g_elem = g_pq_cache[iq, nu, l, m]
                    m_sq = (g_elem * np.conj(g_elem)).real
                    x1 = omega - e_m_kq - w_ph
                    x2 = omega - e_m_kq + w_ph
                    im_d1 = -eta / (x1 * x1 + eta * eta)
                    im_d2 = -eta / (x2 * x2 + eta * eta)
                    term = (nb + 1.0 + nm) * im_d1 + (nb - nm) * im_d2
                    local_lw[m] += metric_diag[l] * m_sq * term
        linewidth += local_lw
    return linewidth / total_q


def compute_linewidth_vectorized(
    g_pq_cache,
    e_kq_cache,
    ph_en_flat,
    e_mag_k,
    T,
    eta,
    total_q,
    metric_diag=None,
):
    """
    Return diagonal ``Im Sigma^R_mm(E_m)`` for every external channel.

    ``metric_diag`` is explicit so normal-boson and BdG kernels share the same
    implementation without embedding a fixed channel count.
    """
    energies = np.asarray(e_mag_k, dtype=np.float64)
    if metric_diag is None:
        metric_diag = bosonic_metric_diag(energies.shape[0])
    metric = np.asarray(metric_diag, dtype=np.float64)
    if metric.shape != energies.shape:
        raise ValueError(f"metric shape {metric.shape} != energy shape {energies.shape}")
    return _compute_linewidth_vectorized(
        g_pq_cache,
        e_kq_cache,
        ph_en_flat,
        energies,
        float(T),
        float(eta),
        int(total_q),
        metric,
    )


@njit(parallel=True, fastmath=True)
def _compute_linewidth_components_vectorized(
    g_components,
    e_kq_cache,
    ph_en_flat,
    e_mag_k,
    T,
    eta,
    total_q,
    metric_diag,
):
    """Numba kernel for process- and polarization-resolved Im Sigma."""
    ncomp = g_components.shape[0]
    num_modes = ph_en_flat.shape[1]
    nchannel = e_mag_k.shape[0]
    contribution = np.zeros(
        (2, ncomp, ncomp, nchannel), dtype=np.float64
    )

    for iq in prange(total_q):
        local = np.zeros((2, ncomp, ncomp, nchannel), dtype=np.float64)
        for nu in range(num_modes):
            w_ph = ph_en_flat[iq, nu]
            nb = bose_function(w_ph, T)
            for m in range(nchannel):
                omega = e_mag_k[m]
                for l in range(nchannel):
                    e_m_kq = e_kq_cache[iq, l]
                    nm = bose_function(e_m_kq, T)
                    x_emission = omega - e_m_kq - w_ph
                    x_absorption = omega - e_m_kq + w_ph
                    im_emission = -eta / (
                        x_emission * x_emission + eta * eta
                    )
                    im_absorption = -eta / (
                        x_absorption * x_absorption + eta * eta
                    )
                    emission_weight = (
                        metric_diag[l] * (nb + 1.0 + nm) * im_emission
                    )
                    absorption_weight = (
                        metric_diag[l] * (nb - nm) * im_absorption
                    )
                    for first in range(ncomp):
                        g_first = g_components[first, iq, nu, l, m]
                        for second in range(first, ncomp):
                            g_second = g_components[second, iq, nu, l, m]
                            cross = (g_first * np.conj(g_second)).real
                            value_emission = cross * emission_weight
                            value_absorption = cross * absorption_weight
                            local[0, first, second, m] += value_emission
                            local[1, first, second, m] += value_absorption
                            if second != first:
                                # Store both ordered cross terms.  Summing the
                                # component axes then gives the full factor
                                # 2 Re(g_first g_second*) exactly once.
                                local[0, second, first, m] += value_emission
                                local[1, second, first, m] += value_absorption
        contribution += local
    return contribution / total_q


def compute_linewidth_components_vectorized(
    g_components,
    e_kq_cache,
    ph_en_flat,
    e_mag_k,
    T,
    eta,
    total_q,
    metric_diag,
):
    r"""Resolve raw diagonal ``Im Sigma^R`` by process and vertex component.

    ``g_components`` has shape
    ``(ncomp,nq,nmode,nInternalChannel,nExternalChannel)``.  The result has
    shape ``(2,ncomp,ncomp,nExternalChannel)``, with process index 0 for
    emission (the ``omega-E-w`` pole) and 1 for absorption (the
    ``omega-E+w`` pole).  Diagonal component entries are self terms, while
    the two symmetric off-diagonal entries together are
    ``2 Re(g_c g_d*)`` interference.

    The BdG metric is deliberately mandatory.  If ``sum_c g_c`` is the full
    vertex, summing all three leading result axes reconstructs
    :func:`compute_linewidth_vectorized` for that full vertex (up to floating
    point summation order).
    """
    components = np.asarray(g_components, dtype=np.complex128)
    if components.ndim != 5:
        raise ValueError(
            "g_components must have shape "
            "(ncomp,nq,nmode,nInternalChannel,nExternalChannel), "
            f"got {components.shape}"
        )
    if components.shape[0] < 1:
        raise ValueError("g_components must contain at least one component")
    nq = components.shape[1]
    nmode = components.shape[2]
    ninternal = components.shape[3]
    nexternal = components.shape[4]
    if ninternal != nexternal:
        raise ValueError(
            "g_components must end in equal internal/external BdG channel axes, "
            f"got {(ninternal, nexternal)}"
        )
    internal_energy = np.asarray(e_kq_cache, dtype=np.float64)
    phonon_energy = np.asarray(ph_en_flat, dtype=np.float64)
    external_energy = np.asarray(e_mag_k, dtype=np.float64).reshape(-1)
    if internal_energy.shape != (nq, ninternal):
        raise ValueError(
            f"e_kq_cache shape {internal_energy.shape} != {(nq, ninternal)}"
        )
    if phonon_energy.shape != (nq, nmode):
        raise ValueError(
            f"ph_en_flat shape {phonon_energy.shape} != {(nq, nmode)}"
        )
    if external_energy.shape != (nexternal,):
        raise ValueError(
            f"e_mag_k shape {external_energy.shape} != {(nexternal,)}"
        )
    metric_raw = np.asarray(metric_diag)
    if np.iscomplexobj(metric_raw) and np.any(np.imag(metric_raw) != 0.0):
        raise ValueError("metric_diag must be real")
    metric = np.asarray(np.real(metric_raw), dtype=np.float64).reshape(-1)
    if metric.shape != (ninternal,):
        raise ValueError(
            f"metric shape {metric.shape} != internal channel shape {(ninternal,)}"
        )
    if not np.all(np.isfinite(metric)):
        raise ValueError("metric_diag contains non-finite values")
    normalized_q = int(total_q)
    if normalized_q != nq or normalized_q < 1:
        raise ValueError(
            f"total_q={normalized_q} must equal the positive q dimension {nq}"
        )
    broadening = float(eta)
    if not np.isfinite(broadening) or broadening <= 0.0:
        raise ValueError(f"eta must be positive and finite, got {eta}")
    return _compute_linewidth_components_vectorized(
        components,
        internal_energy,
        phonon_energy,
        external_energy,
        float(T),
        broadening,
        normalized_q,
        metric,
    )


@njit(parallel=True, fastmath=True)
def _compute_self_energy_at_frequencies(
    omega_eval,
    g_pq_cache,
    e_kq_cache,
    ph_en_flat,
    T,
    eta,
    total_q,
    metric_diag,
):
    num_modes = ph_en_flat.shape[1]
    nchannel = e_kq_cache.shape[1]
    sigma = np.zeros((omega_eval.shape[0], nchannel, nchannel), dtype=np.complex128)

    for iq in prange(total_q):
        local_sigma = np.zeros(
            (omega_eval.shape[0], nchannel, nchannel), dtype=np.complex128
        )
        for nu in range(num_modes):
            w_ph = ph_en_flat[iq, nu]
            nb = bose_function(w_ph, T)
            for l in range(nchannel):
                e_m_kq = e_kq_cache[iq, l]
                nm = bose_function(e_m_kq, T)
                for iw in range(omega_eval.shape[0]):
                    d1 = (omega_eval[iw] + 1j * eta) - (e_m_kq + w_ph)
                    d2 = (omega_eval[iw] + 1j * eta) - (e_m_kq - w_ph)
                    weight = metric_diag[l] * (
                        (nb + 1.0 + nm) / d1 + (nb - nm) / d2
                    )
                    for m in range(nchannel):
                        g_left = np.conj(g_pq_cache[iq, nu, l, m])
                        for n in range(nchannel):
                            local_sigma[iw, m, n] += (
                                g_left * g_pq_cache[iq, nu, l, n] * weight
                            )
        sigma += local_sigma
    return sigma / total_q


def compute_self_energy_at_frequencies(
    omega_eval,
    g_pq_cache,
    e_kq_cache,
    ph_en_flat,
    T,
    eta,
    total_q,
    metric_diag,
):
    """Evaluate the complete retarded self-energy at common frequencies."""
    omega = np.asarray(omega_eval, dtype=np.float64).reshape(-1)
    metric = np.asarray(metric_diag, dtype=np.float64).reshape(-1)
    nchannel = int(np.asarray(e_kq_cache).shape[1])
    if metric.shape != (nchannel,):
        raise ValueError(
            f"metric shape {metric.shape} != internal channel count {(nchannel,)}"
        )
    return _compute_self_energy_at_frequencies(
        omega,
        g_pq_cache,
        e_kq_cache,
        ph_en_flat,
        float(T),
        float(eta),
        int(total_q),
        metric,
    )


def compute_self_energy_onshell_matrix(
    g_pq_cache,
    e_kq_cache,
    ph_en_flat,
    e_mag_k,
    T,
    eta,
    total_q,
    metric_diag=None,
):
    """Return Sigma_mn(omega_m) for all on-shell external channels.

    The diagonal imaginary part matches compute_linewidth_vectorized before
    the final sign conversion: linewidth_raw[m] = Im Sigma_mm(omega_m).
    """
    energies = np.asarray(e_mag_k, dtype=np.float64).reshape(-1)
    if metric_diag is None:
        metric_diag = bosonic_metric_diag(energies.shape[0])
    sigma_by_energy = compute_self_energy_at_frequencies(
        energies,
        g_pq_cache,
        e_kq_cache,
        ph_en_flat,
        T,
        eta,
        total_q,
        metric_diag,
    )
    # Backward-compatible row-on-shell representation: row m is evaluated at
    # E_m.  This object must not be treated as one matrix at a common omega.
    index = np.arange(energies.shape[0])
    return sigma_by_energy[index, index, :]
