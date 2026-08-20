import numpy as np
from numba import njit, prange


HBAR_MEV_PS = 0.6582119569
KB_MEV = 0.08617333262


@njit(parallel=True, fastmath=True)
def precompute_phase_factors(q_mesh_flat, delta_bonds):
    total_q = q_mesh_flat.shape[0]
    n_bonds = delta_bonds.shape[0]
    out = np.zeros((total_q, n_bonds), dtype=np.complex128)
    for iq in prange(total_q):
        for ib in range(n_bonds):
            dot_val = (
                q_mesh_flat[iq, 0] * delta_bonds[ib, 0]
                + q_mesh_flat[iq, 1] * delta_bonds[ib, 1]
                + q_mesh_flat[iq, 2] * delta_bonds[ib, 2]
            )
            out[iq, ib] = np.exp(1j * dot_val)
    return out


@njit(fastmath=True)
def Magnon_BdG_kernel_v1(k, S, J0, R, atom_pos, spin_pattern, bond_factor=1.0, anisotropy=0.0):
    """Two-sublattice BdG Hamiltonian in the current normalized-spin J convention.

    ``J0`` is a mate-complete directed bond list.  Each directed entry carries
    one half of the full two-endpoint Nambu bond kernel, preserving the scale
    of the historical ``Magnon_Hamiltonian_v1`` implementation.  The basis is
    ``(a_k, b_k, a^dagger_-k, b^dagger_-k)``.
    """
    nmag = spin_pattern.shape[0]
    if nmag != 2:
        raise ValueError("Magnon_BdG_kernel_v1 requires exactly two magnetic sublattices")
    H = np.zeros((4, 4), dtype=np.complex128)
    onsite = anisotropy / S
    for i0 in range(4):
        H[i0, i0] = onsite + 0.0j

    for idx in range(len(J0[0])):
        J_eff = J0[0][idx]
        i = int(J0[1][idx])
        j = int(J0[2][idx])
        rv = J0[3][idx]
        dx = atom_pos[j, 0] - atom_pos[i, 0] + rv[0] * R[0, 0] + rv[1] * R[1, 0] + rv[2] * R[2, 0]
        dy = atom_pos[j, 1] - atom_pos[i, 1] + rv[0] * R[0, 1] + rv[1] * R[1, 1] + rv[2] * R[2, 1]
        dz = atom_pos[j, 2] - atom_pos[i, 2] + rv[0] * R[0, 2] + rv[1] * R[1, 2] + rv[2] * R[2, 2]
        dot_val = k[0] * dx + k[1] * dy + k[2] * dz
        phase = np.exp(1.0j * dot_val)
        phase_conj = np.conjugate(phase)
        align = spin_pattern[i] * spin_pattern[j]
        coefficient = -bond_factor * J_eff / (2.0 * S)

        # N_b(k,0) = -eta_b(P_i+P_j) + p_b X_b(k,0)
        H[i, i] += coefficient * (-align)
        H[j, j] += coefficient * (-align)
        H[nmag + i, nmag + i] += coefficient * (-align)
        H[nmag + j, nmag + j] += coefficient * (-align)
        if align > 0.0:
            H[i, j] += coefficient * phase
            H[j, i] += coefficient * phase_conj
            H[nmag + i, nmag + j] += coefficient * phase
            H[nmag + j, nmag + i] += coefficient * phase_conj
        else:
            # C_b(k,0) = X_b(k,0) for antiparallel bonds.
            H[i, nmag + j] += coefficient * phase
            H[j, nmag + i] += coefficient * phase_conj
            H[nmag + i, j] += coefficient * phase
            H[nmag + j, i] += coefficient * phase_conj
    return 0.5 * (H + np.conjugate(H.T))


@njit(fastmath=True)
def Magnon_Hamiltonian_v1(k, S, J0, R, atom_pos, spin_pattern, bond_factor=1.0, anisotropy=0.0):
    H = Magnon_BdG_kernel_v1(
        k, S, J0, R, atom_pos, spin_pattern, bond_factor, anisotropy
    )
    omega = np.zeros(4, dtype=np.float64)
    transform = np.zeros((4, 4), dtype=np.complex128)
    # The A-type kernel splits into the two chirality sectors (a,b^dagger) and
    # (b,a^dagger).  Solving them independently fixes the physical channel
    # labels even when the two positive branches are exactly degenerate.
    sector_rows = np.array([[0, 3], [1, 2]], dtype=np.int64)
    sector_metric = np.array([1.0, -1.0], dtype=np.float64)
    for sector in range(2):
        rows = sector_rows[sector]
        block = np.zeros((2, 2), dtype=np.complex128)
        for row in range(2):
            for col in range(2):
                block[row, col] = sector_metric[row] * H[rows[row], rows[col]]
        values, vectors = np.linalg.eig(block)
        norms = np.zeros(2, dtype=np.float64)
        for index in range(2):
            for row in range(2):
                norms[index] += sector_metric[row] * (
                    np.conjugate(vectors[row, index]) * vectors[row, index]
                ).real
        positive_index = 0 if norms[0] > norms[1] else 1
        negative_index = 1 - positive_index
        if norms[positive_index] <= 1.0e-10 or norms[negative_index] >= -1.0e-10:
            order = np.argsort(values.real)
            negative_index = order[0]
            positive_index = order[1]

        positive_column = sector
        negative_column = 3 if sector == 0 else 2
        omega[positive_column] = values[positive_index].real
        omega[negative_column] = values[negative_index].real
        positive_norm = np.abs(norms[positive_index])
        negative_norm = np.abs(norms[negative_index])
        if positive_norm < 1.0e-12:
            positive_norm = np.sum(np.abs(vectors[:, positive_index]) ** 2)
        if negative_norm < 1.0e-12:
            negative_norm = np.sum(np.abs(vectors[:, negative_index]) ** 2)
        for row in range(2):
            transform[rows[row], positive_column] = (
                vectors[row, positive_index] / np.sqrt(positive_norm)
            )
            transform[rows[row], negative_column] = (
                vectors[row, negative_index] / np.sqrt(negative_norm)
            )
    return omega, transform


@njit(fastmath=True)
def Magnon_Hamiltonian_fm_normal(k, S, J0, R, atom_pos, nmag, anisotropy=0.0, bond_factor=1.0):
    """Normal LSWT Hamiltonian for collinear FM order.

    The input J convention follows the rest of magph: J_eff is assumed to be
    J*S^2 in meV, so the magnon scale is J_eff/S. Positive J is FM for the
    Hamiltonian convention H=-sum J_ij S_i.S_j.
    """
    h = np.zeros((nmag, nmag), dtype=np.complex128)
    k0 = float(anisotropy) / float(S)
    for i0 in range(nmag):
        h[i0, i0] = k0 + 0j

    for idx in range(len(J0[0])):
        J_eff = bond_factor * J0[0][idx] / S
        i = int(J0[1][idx])
        j = int(J0[2][idx])
        rv = J0[3][idx]
        delta = atom_pos[j] - atom_pos[i] + R.T @ np.array([rv[0], rv[1], rv[2]], dtype=np.float64)
        dot_val = k[0] * delta[0] + k[1] * delta[1] + k[2] * delta[2]
        phase = np.exp(1j * dot_val)
        if i == j:
            h[i, i] += 2.0 * J_eff * (1.0 - np.cos(dot_val))
        else:
            h[i, i] += J_eff
            h[j, j] += J_eff
            h[i, j] -= J_eff * phase
            h[j, i] -= J_eff * np.conj(phase)

    h = 0.5 * (h + np.conjugate(h.T))
    omega, vec = np.linalg.eigh(h)
    return omega.real, vec


@njit(fastmath=True)
def Magnon_Hamiltonian_collinear_bdg(k, S, J0, R, atom_pos, spin_pattern, nmag, anisotropy=0.0, bond_factor=1.0):
    """General collinear LSWT BdG kernel using site spin alignment.

    spin_pattern entries are +/-1 in the magnetic basis.  The input exchange is
    assumed to follow the TB2J normalized-spin convention and J_eff=J*S^2, so
    the magnon scale is J_eff/S.  Directed i!=j bond lists should use
    bond_factor=1.
    """
    A = np.zeros((nmag, nmag), dtype=np.complex128)
    B = np.zeros((nmag, nmag), dtype=np.complex128)
    k0 = float(anisotropy) / float(S)
    for i0 in range(nmag):
        A[i0, i0] = k0 + 0j

    for idx in range(len(J0[0])):
        J_eff = bond_factor * J0[0][idx] / S
        i = int(J0[1][idx])
        j = int(J0[2][idx])
        rv = J0[3][idx]
        delta = atom_pos[j] - atom_pos[i] + R.T @ np.array([rv[0], rv[1], rv[2]], dtype=np.float64)
        dot_val = k[0] * delta[0] + k[1] * delta[1] + k[2] * delta[2]
        phase = np.exp(1j * dot_val)
        align = spin_pattern[i] * spin_pattern[j]
        # Positive stiffness for a stable bond in the chosen collinear state.
        Kij = J_eff * align
        if align > 0.0:
            A[i, i] += Kij
            A[i, j] -= Kij * phase
        else:
            A[i, i] += Kij
            B[i, j] += Kij * phase

    A = 0.5 * (A + np.conjugate(A.T))
    B = 0.5 * (B + np.conjugate(B.T))
    H = np.zeros((2 * nmag, 2 * nmag), dtype=np.complex128)
    H[:nmag, :nmag] = A
    H[:nmag, nmag:] = B
    H[nmag:, :nmag] = np.conjugate(B.T)
    H[nmag:, nmag:] = np.conjugate(A)

    eta = np.zeros((2 * nmag, 2 * nmag), dtype=np.complex128)
    for i0 in range(nmag):
        eta[i0, i0] = 1.0 + 0j
        eta[nmag + i0, nmag + i0] = -1.0 + 0j

    vals, vecs = np.linalg.eig(eta @ H)
    order = np.argsort(vals.real)
    omega = np.zeros(nmag, dtype=np.float64)
    Uk = np.zeros((2 * nmag, nmag), dtype=np.complex128)
    count = 0
    for io in range(order.shape[0]):
        idx = order[io]
        val = vals[idx].real
        vec = vecs[:, idx]
        norm = 0.0
        for j0 in range(2 * nmag):
            sgn = 1.0 if j0 < nmag else -1.0
            norm += sgn * (np.conjugate(vec[j0]) * vec[j0]).real
        if val > -1.0e-10 and norm > 1.0e-10 and count < nmag:
            omega[count] = max(0.0, val)
            Uk[:, count] = vec / np.sqrt(norm)
            count += 1
    if count < nmag:
        # Fallback for unstable/noisy cases: take the largest positive-real
        # eigenvalues so plotting still exposes the instability.
        pos = np.argsort(vals.real)[-nmag:]
        pos = pos[np.argsort(vals.real[pos])]
        for ic in range(nmag):
            omega[ic] = vals[pos[ic]].real
            Uk[:, ic] = vecs[:, pos[ic]]
    return omega, Uk


@njit(fastmath=True)
def Magnon_Hamiltonian_4_v3(k, S, J0, R, atom_pos):
    K = 0.005
    A = np.array([2.0 * K * S + 0j, 2.0 * K * S + 0j], dtype=np.complex128)
    B = np.array([0j, 0j], dtype=np.complex128)

    for idx in range(len(J0[0])):
        J_eff = J0[0][idx]
        i = int(J0[1][idx])
        j = int(J0[2][idx])
        rv = J0[3][idx]
        delta = atom_pos[j] - atom_pos[i] + R.T @ np.array([rv[0], rv[1], rv[2]], dtype=np.float64)
        dot_val = k[0] * delta[0] + k[1] * delta[1] + k[2] * delta[2]
        J_mag = np.abs(J_eff)
        if J_eff < 0.0:
            phase = np.exp(1j * dot_val)
            A[i] += J_mag / S
            B[i] += J_mag / S * phase
        else:
            phase = np.cos(dot_val)
            A[i] += J_mag / S * (1.0 - phase)

    H_up = np.zeros((2, 2), dtype=np.complex128)
    H_up[0, 0] = A[0]
    H_up[0, 1] = B[1]
    H_up[1, 0] = B[0]
    H_up[1, 1] = A[1]
    sz_up = np.array([[1 + 0j, 0 + 0j], [0 + 0j, -1 + 0j]], dtype=np.complex128)
    w_up, v_up = np.linalg.eig(sz_up @ H_up)
    idx_up = np.argsort(w_up.real)[::-1]
    w_up = w_up[idx_up]
    v_up = v_up[:, idx_up]

    H_dn = np.zeros((2, 2), dtype=np.complex128)
    H_dn[0, 0] = np.conj(A[0])
    H_dn[0, 1] = np.conj(B[0])
    H_dn[1, 0] = np.conj(B[1])
    H_dn[1, 1] = np.conj(A[1])
    sz_dn = np.array([[-1 + 0j, 0 + 0j], [0 + 0j, 1 + 0j]], dtype=np.complex128)
    w_dn, v_dn = np.linalg.eig(sz_dn @ H_dn)
    idx_dn = np.argsort(w_dn.real)
    w_dn = w_dn[idx_dn]
    v_dn = v_dn[:, idx_dn]

    omega = np.zeros(4, dtype=np.float64)
    omega[0] = w_up[0].real
    omega[1] = w_up[1].real
    omega[2] = w_dn[0].real
    omega[3] = w_dn[1].real

    Uk = np.zeros((4, 4), dtype=np.complex128)
    Uk[0:2, 0:2] = v_up
    Uk[2:4, 2:4] = v_dn
    n0 = (np.conj(v_up[:, 0]).T @ sz_up @ v_up[:, 0]).real
    n1 = (np.conj(v_up[:, 1]).T @ sz_up @ v_up[:, 1]).real
    n2 = (np.conj(v_dn[:, 0]).T @ sz_dn @ v_dn[:, 0]).real
    n3 = (np.conj(v_dn[:, 1]).T @ sz_dn @ v_dn[:, 1]).real
    Uk[0:2, 0] /= np.sqrt(np.abs(n0) + 1.0e-6)
    Uk[0:2, 1] /= np.sqrt(np.abs(n1) + 1.0e-6)
    Uk[2:4, 2] /= np.sqrt(np.abs(n2) + 1.0e-6)
    Uk[2:4, 3] /= np.sqrt(np.abs(n3) + 1.0e-6)
    return omega, Uk


@njit(fastmath=True, inline="always")
def bose_function(energy, T):
    # A bosonic BdG spectrum contains negative-energy hole partners.  Their
    # occupation is fixed by n_B(-E) = -(1 + n_B(E)), including the T -> 0
    # limit n_B(-E) = -1.  Returning zero for every energy at T=0 breaks the
    # cancellation between the negative-norm metric and the hole occupation.
    if T < 1.0e-9:
        if energy < -1.0e-9:
            return -1.0
        return 0.0
    if energy < -1.0e-9:
        x = -energy / (KB_MEV * T)
        if x > 700.0:
            n_abs = 0.0
        elif x < 1.0e-10:
            n_abs = 1.0 / x
        else:
            n_abs = 1.0 / (np.exp(x) - 1.0)
        return -(1.0 + n_abs)
    if energy > 1.0e-9:
        x = energy / (KB_MEV * T)
        if x > 700.0:
            return 0.0
        if x < 1.0e-10:
            return 1.0 / x
        return 1.0 / (np.exp(x) - 1.0)
    return 0.0


@njit(fastmath=True)
def calc_vertex_4_optimized_mpi(ik, iq, exp_iqd, exp_ikd, ph_freq, ph_pol, Uk, Ukq, gradJ, S):
    eta = 0.001
    unit_conv = 3.1062
    num_modes = ph_pol.shape[0]
    num_atoms = ph_pol.shape[1]
    eps_q = np.zeros((num_modes, num_atoms, 3), dtype=np.complex128)
    for mode_idx in range(num_modes):
        freq = ph_freq[mode_idx]
        if freq < 0.0:
            freq = 0.0
        factor = (HBAR_MEV_PS / np.sqrt(2.0 * freq + eta)) * unit_conv
        for ia in range(num_atoms):
            for ax in range(3):
                eps_q[mode_idx, ia, ax] = ph_pol[mode_idx, ia, ax] * factor

    Vq = np.zeros((num_modes, 4, 4), dtype=np.complex128)
    for idx in range(len(gradJ[0])):
        k_atom_idx = int(gradJ[0][idx])
        i_idx = int(gradJ[1][idx])
        j_idx = int(gradJ[2][idx])
        J_grad_vec = gradJ[4][idx]
        exp_iq = exp_iqd[iq, idx]
        exp_ik = exp_ikd[ik, idx]
        exp_ikq = exp_iq * exp_ik
        Gamma_kq = (1.0 - exp_iq) * (1.0 + np.conj(exp_iq) - exp_ik - exp_ikq)

        for mode in range(num_modes):
            K_dot = (1.0 / (2.0 * S)) * (
                eps_q[mode, k_atom_idx, 0] * J_grad_vec[0]
                + eps_q[mode, k_atom_idx, 1] * J_grad_vec[1]
                + eps_q[mode, k_atom_idx, 2] * J_grad_vec[2]
            )
            if i_idx == j_idx:
                g = K_dot * Gamma_kq
                if i_idx == 0:
                    Vq[mode, 0, 0] += g
                    Vq[mode, 2, 2] += np.conj(g)
                else:
                    Vq[mode, 1, 1] += g
                    Vq[mode, 3, 3] += np.conj(g)
            else:
                phase_q = 1.0 - exp_iq
                phase_mq = 1.0 - np.conj(exp_iq)
                Vq[mode, 0, 1] += phase_q * np.conj(-K_dot) * exp_ik
                Vq[mode, 1, 0] += -phase_mq * np.conj(-K_dot) * np.conj(exp_ik)
                Vq[mode, 2, 3] += phase_mq * -K_dot * np.conj(exp_ikq)
                Vq[mode, 3, 2] += -phase_q * -K_dot * exp_ikq
                Vq[mode, 0, 0] += -K_dot * phase_q
                Vq[mode, 2, 2] += -np.conj(-K_dot) * phase_mq
                Vq[mode, 1, 1] += -K_dot * -phase_mq
                Vq[mode, 3, 3] += np.conj(-K_dot) * -phase_q

    g = np.zeros((num_modes, 4, 4), dtype=np.complex128)
    gm = np.zeros((num_modes, 4, 4), dtype=np.complex128)
    U_kq_H = np.conj(Ukq).T
    for mode in range(num_modes):
        g[mode] = U_kq_H @ Vq[mode] @ Uk
        gm[mode] = np.conj(g[mode]).T
    return g, gm


@njit(fastmath=True)
def calc_vertex_fm_normal_mpi(ik, iq, exp_iqd, exp_ikd, ph_freq, ph_pol, Uk, Ukq, gradJ, S, nmag):
    eta = 0.001
    unit_conv = 3.1062
    num_modes = ph_pol.shape[0]
    num_atoms = ph_pol.shape[1]
    eps_q = np.zeros((num_modes, num_atoms, 3), dtype=np.complex128)
    for mode_idx in range(num_modes):
        freq = ph_freq[mode_idx]
        if freq < 0.0:
            freq = 0.0
        factor = (HBAR_MEV_PS / np.sqrt(2.0 * freq + eta)) * unit_conv
        for ia in range(num_atoms):
            for ax in range(3):
                eps_q[mode_idx, ia, ax] = ph_pol[mode_idx, ia, ax] * factor

    Vq = np.zeros((num_modes, nmag, nmag), dtype=np.complex128)
    for idx in range(len(gradJ[0])):
        k_atom_idx = int(gradJ[0][idx])
        i_idx = int(gradJ[1][idx])
        j_idx = int(gradJ[2][idx])
        J_grad_vec = gradJ[4][idx]
        exp_iq = exp_iqd[iq, idx]
        exp_ik = exp_ikd[ik, idx]
        exp_ikq = exp_iq * exp_ik
        Gamma_kq = (1.0 - exp_iq) * (1.0 + np.conj(exp_iq) - exp_ik - exp_ikq)

        for mode in range(num_modes):
            K_dot = (1.0 / (2.0 * S)) * (
                eps_q[mode, k_atom_idx, 0] * J_grad_vec[0]
                + eps_q[mode, k_atom_idx, 1] * J_grad_vec[1]
                + eps_q[mode, k_atom_idx, 2] * J_grad_vec[2]
            )
            if i_idx == j_idx:
                Vq[mode, i_idx, i_idx] += K_dot * Gamma_kq
            else:
                phase_q = 1.0 - exp_iq
                phase_mq = 1.0 - np.conj(exp_iq)
                Vq[mode, i_idx, j_idx] += phase_q * np.conj(-K_dot) * exp_ik
                Vq[mode, j_idx, i_idx] += -phase_mq * np.conj(-K_dot) * np.conj(exp_ik)
                Vq[mode, i_idx, i_idx] += -K_dot * phase_q
                Vq[mode, j_idx, j_idx] += K_dot * phase_mq

    g = np.zeros((num_modes, nmag, nmag), dtype=np.complex128)
    U_kq_H = np.conj(Ukq).T
    for mode in range(num_modes):
        g[mode] = U_kq_H @ Vq[mode] @ Uk
    return g


@njit(parallel=True, fastmath=True)
def g_kq_loop(ik, exp_iqd, exp_ikd, total_q, q_mesh_flat_cart, k_cart, dJdu, Uk, ph_en_flat, ph_vec_flat, S, J0, R_vec, G_vec, atom_pos, spin_pattern):
    num_modes = ph_vec_flat.shape[1]
    g_pq_cache = np.zeros((total_q, num_modes, 4, 4), dtype=np.complex128)
    e_kq_cache = np.zeros((total_q, 4), dtype=np.float64)
    for iq in prange(total_q):
        kq_cart = k_cart + q_mesh_flat_cart[iq]
        e_mag_kq, Ukq = Magnon_Hamiltonian_v1(kq_cart, S, J0, R_vec, atom_pos, spin_pattern)
        g_kq, _ = calc_vertex_4_optimized_mpi(
            ik,
            iq,
            exp_iqd,
            exp_ikd,
            ph_en_flat[iq],
            ph_vec_flat[iq],
            Uk,
            Ukq,
            dJdu,
            S,
        )
        g_pq_cache[iq] = g_kq
        e_kq_cache[iq] = e_mag_kq
    return g_pq_cache, e_kq_cache


@njit(parallel=True, fastmath=True)
def g_kq_loop_fm_normal(
    ik,
    exp_iqd,
    exp_ikd,
    total_q,
    q_mesh_flat_cart,
    k_cart,
    dJdu,
    Uk,
    ph_en_flat,
    ph_vec_flat,
    S,
    J0,
    R_vec,
    G_vec,
    atom_pos,
    nmag,
    anisotropy,
    bond_factor,
):
    num_modes = ph_vec_flat.shape[1]
    g_pq_cache = np.zeros((total_q, num_modes, nmag, nmag), dtype=np.complex128)
    e_kq_cache = np.zeros((total_q, nmag), dtype=np.float64)
    for iq in prange(total_q):
        kq_cart = k_cart + q_mesh_flat_cart[iq]
        e_mag_kq, Ukq = Magnon_Hamiltonian_fm_normal(
            kq_cart,
            S,
            J0,
            R_vec,
            atom_pos,
            nmag,
            anisotropy,
            bond_factor,
        )
        g_kq = calc_vertex_fm_normal_mpi(
            ik,
            iq,
            exp_iqd,
            exp_ikd,
            ph_en_flat[iq],
            ph_vec_flat[iq],
            Uk,
            Ukq,
            dJdu,
            S,
            nmag,
        )
        g_pq_cache[iq] = g_kq
        e_kq_cache[iq] = e_mag_kq
    return g_pq_cache, e_kq_cache


@njit(parallel=True, fastmath=True)
def fixed_q_gbar_afm(
    k_mesh_cart,
    q_cart,
    exp_iqd,
    exp_ikd,
    ph_freq,
    ph_pol,
    dJdu,
    S,
    J0,
    R_vec,
    atom_pos,
    spin_pattern,
    physical_only=True,
):
    """Compute sqrt(sum_(m,nu)|G_mnnu(Q,k)|^2/Nm) for every AFM k and n."""
    total_k = k_mesh_cart.shape[0]
    nchannel = 4
    nphysical = nchannel // 2 if physical_only else nchannel
    result = np.zeros((total_k, nphysical), dtype=np.float64)

    for ik in prange(total_k):
        k_cart = k_mesh_cart[ik]
        _, Uk = Magnon_Hamiltonian_v1(k_cart, S, J0, R_vec, atom_pos, spin_pattern)
        _, Ukq = Magnon_Hamiltonian_v1(k_cart + q_cart, S, J0, R_vec, atom_pos, spin_pattern)
        vertex, _ = calc_vertex_4_optimized_mpi(
            ik,
            0,
            exp_iqd,
            exp_ikd,
            ph_freq,
            ph_pol,
            Uk,
            Ukq,
            dJdu,
            S,
        )
        for n in range(nphysical):
            norm_squared = 0.0
            for nu in range(vertex.shape[0]):
                for m in range(nphysical):
                    value = vertex[nu, m, n]
                    norm_squared += value.real * value.real + value.imag * value.imag
            result[ik, n] = np.sqrt(norm_squared / float(nphysical))
    return result


@njit(parallel=True, fastmath=True)
def fixed_q_gbar_fm_normal(
    k_mesh_cart,
    q_cart,
    exp_iqd,
    exp_ikd,
    ph_freq,
    ph_pol,
    dJdu,
    S,
    J0,
    R_vec,
    atom_pos,
    nmag,
    anisotropy=0.0,
    bond_factor=1.0,
):
    """Compute the fixed-Q branch-resolved coupling norm for normal FM LSWT."""
    total_k = k_mesh_cart.shape[0]
    result = np.zeros((total_k, nmag), dtype=np.float64)

    for ik in prange(total_k):
        k_cart = k_mesh_cart[ik]
        _, Uk = Magnon_Hamiltonian_fm_normal(
            k_cart, S, J0, R_vec, atom_pos, nmag, anisotropy, bond_factor
        )
        _, Ukq = Magnon_Hamiltonian_fm_normal(
            k_cart + q_cart, S, J0, R_vec, atom_pos, nmag, anisotropy, bond_factor
        )
        vertex = calc_vertex_fm_normal_mpi(
            ik,
            0,
            exp_iqd,
            exp_ikd,
            ph_freq,
            ph_pol,
            Uk,
            Ukq,
            dJdu,
            S,
            nmag,
        )
        for n in range(nmag):
            norm_squared = 0.0
            for nu in range(vertex.shape[0]):
                for m in range(nmag):
                    value = vertex[nu, m, n]
                    norm_squared += value.real * value.real + value.imag * value.imag
            result[ik, n] = np.sqrt(norm_squared / float(nmag))
    return result


@njit(parallel=True, fastmath=True)
def compute_all_omega_sigma_vectorized(omega_full, g_pq_cache, e_kq_cache, ph_en_flat, T, eta, total_q):
    num_w = len(omega_full)
    sigma_k_all_w = np.zeros((num_w, 4, 4), dtype=np.complex128)
    sz = np.array([1 + 0j, 1 + 0j, -1 + 0j, -1 + 0j], dtype=np.complex128)
    for iq in prange(total_q):
        local_sigma = np.zeros((num_w, 4, 4), dtype=np.complex128)
        for nu in range(ph_en_flat.shape[1]):
            w_ph = ph_en_flat[iq, nu]
            nb = bose_function(w_ph, T)
            for l in range(4):
                e_m_kq = e_kq_cache[iq, l]
                nm = bose_function(e_m_kq, T)
                for iw in range(num_w):
                    d1 = (omega_full[iw] + 1j * eta) - (e_m_kq + w_ph)
                    d2 = (omega_full[iw] + 1j * eta) - (e_m_kq - w_ph)
                    weight = (nb + 1.0 + nm) / d1 + (nb - nm) / d2
                    for m in range(4):
                        g_mq_ml = np.conj(g_pq_cache[iq, nu, l, m])
                        for n in range(4):
                            local_sigma[iw, m, n] += g_mq_ml * g_pq_cache[iq, nu, l, n] * sz[l] * weight
        sigma_k_all_w += local_sigma
    return sigma_k_all_w / total_q


@njit(parallel=True, fastmath=True)
def compute_all_omega_sigma_fm_normal_vectorized(omega_full, g_pq_cache, e_kq_cache, ph_en_flat, T, eta, total_q):
    num_w = len(omega_full)
    nchan = e_kq_cache.shape[1]
    sigma_k_all_w = np.zeros((num_w, nchan, nchan), dtype=np.complex128)
    for iq in prange(total_q):
        local_sigma = np.zeros((num_w, nchan, nchan), dtype=np.complex128)
        for nu in range(ph_en_flat.shape[1]):
            w_ph = ph_en_flat[iq, nu]
            nb = bose_function(w_ph, T)
            for l in range(nchan):
                e_m_kq = e_kq_cache[iq, l]
                nm = bose_function(e_m_kq, T)
                for iw in range(num_w):
                    d1 = (omega_full[iw] + 1j * eta) - (e_m_kq + w_ph)
                    d2 = (omega_full[iw] + 1j * eta) - (e_m_kq - w_ph)
                    weight = (nb + 1.0 + nm) / d1 + (nb - nm) / d2
                    for m in range(nchan):
                        g_mq_ml = np.conj(g_pq_cache[iq, nu, l, m])
                        for n in range(nchan):
                            local_sigma[iw, m, n] += g_mq_ml * g_pq_cache[iq, nu, l, n] * weight
        sigma_k_all_w += local_sigma
    return sigma_k_all_w / total_q
