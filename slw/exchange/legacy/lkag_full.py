"""Archived full LKAG solver retained for historical diagnostics."""
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from numba import njit, prange
from scipy.optimize import linear_sum_assignment

from slw.core.wannier_io import read_wannier_hr
from slw.exchange.kernels.gkq_adapter import _load_single_gkq

# Constants
RY_TO_EV = 13.605693122994
HA_TO_EV = 2.0 * RY_TO_EV

@njit(parallel=True)
def _compute_Gk_mesh_njit(evals_mesh, evecs_mesh, z, Nk1, Nk2, Nk3):
    """
    Compute G(k, z) for the entire k-mesh at energy z.
    Returns: Gk_mesh (Nk1, Nk2, Nk3, dim, dim)
    """
    dim = evals_mesh.shape[3]
    Gk_mesh = np.zeros((Nk1, Nk2, Nk3, dim, dim), dtype=np.complex128)
    for k1 in prange(Nk1):
        for k2 in range(Nk2):
            for k3 in range(Nk3):
                evals = evals_mesh[k1, k2, k3]
                evecs = evecs_mesh[k1, k2, k3]
                inv_den = 1.0 / (z - evals)
                for i in range(dim):
                    for j in range(dim):
                        # Sum over bands b
                        val = 0.0 + 0.0j
                        for b in range(dim):
                            val += (evecs[i, b] * evecs[j, b].conjugate()) * inv_den[b]
                        Gk_mesh[k1, k2, k3, i, j] = val
    return Gk_mesh

@njit(parallel=True)
def _sum_GR_bulk_njit(Gk_mesh, R_list_tensor, Nk1, Nk2, Nk3):
    """
    Fourier transform G(k) -> G(R) for multiple R vectors.
    R_list_tensor: (n_R, 3)
    Returns: GR_mesh (n_R, dim, dim)
    """
    n_R = R_list_tensor.shape[0]
    dim = Gk_mesh.shape[3]
    GR_mesh = np.zeros((n_R, dim, dim), dtype=np.complex128)
    Nk_inv = 1.0 / (Nk1 * Nk2 * Nk3)

    for ir in prange(n_R):
        R = R_list_tensor[ir]
        for k1 in range(Nk1):
            for k2 in range(Nk2):
                for k3 in range(Nk3):
                    phase = np.exp(1j * 2 * np.pi * (k1/Nk1*R[0] + k2/Nk2*R[1] + k3/Nk3*R[2]))
                    for i in range(dim):
                        for j in range(dim):
                            GR_mesh[ir, i, j] += Gk_mesh[k1, k2, k3, i, j] * phase
        for i in range(dim):
            for j in range(dim):
                GR_mesh[ir, i, j] *= Nk_inv
    return GR_mesh

@njit(parallel=True)
def _accumulate_tb2j_A_numba(
    GR,
    GRm,
    ni_arr,
    nj_arr,
    iu_up,
    iu_dn,
    jv_up,
    jv_dn,
    P_i,
    P_j,
    acc_A,
    weight_over_pi,
):
    """
    Accumulate TB2J-like A^{uv} for one contour energy point.
    Parallel over flattened (pair, R) tasks.
    """
    n_pairs = ni_arr.shape[0]
    n_R = GR.shape[0]
    n_tasks = n_pairs * n_R

    for task in prange(n_tasks):
        ip = task // n_R
        ir = task - ip * n_R

        ni = ni_arr[ip]
        nj = nj_arr[ip]

        gij_u = np.zeros((4, ni, nj), dtype=np.complex128)
        gji_u = np.zeros((4, nj, ni), dtype=np.complex128)

        G = GR[ir]
        Gm = GRm[ir]

        for i in range(ni):
            iu = iu_up[ip, i]
            idn = iu_dn[ip, i]
            for j in range(nj):
                ju = jv_up[ip, j]
                jdn = jv_dn[ip, j]

                uu = G[iu, ju]
                ud = G[iu, jdn]
                du = G[idn, ju]
                dd = G[idn, jdn]
                gij_u[0, i, j] = 0.5 * (uu + dd)
                gij_u[1, i, j] = 0.5 * (ud + du)
                gij_u[2, i, j] = 0.5j * (ud - du)
                gij_u[3, i, j] = 0.5 * (uu - dd)

                uu_m = Gm[ju, iu]
                ud_m = Gm[ju, idn]
                du_m = Gm[jdn, iu]
                dd_m = Gm[jdn, idn]
                gji_u[0, j, i] = 0.5 * (uu_m + dd_m)
                gji_u[1, j, i] = 0.5 * (ud_m + du_m)
                gji_u[2, j, i] = 0.5j * (ud_m - du_m)
                gji_u[3, j, i] = 0.5 * (uu_m - dd_m)

        X = np.zeros((4, ni, nj), dtype=np.complex128)
        Y = np.zeros((4, nj, ni), dtype=np.complex128)

        for u in range(4):
            for i in range(ni):
                for j in range(nj):
                    s = 0.0 + 0.0j
                    for k in range(ni):
                        s += P_i[ip, i, k] * gij_u[u, k, j]
                    X[u, i, j] = s

        for v in range(4):
            for j in range(nj):
                for i in range(ni):
                    s = 0.0 + 0.0j
                    for k in range(nj):
                        s += P_j[ip, j, k] * gji_u[v, k, i]
                    Y[v, j, i] = s

        for u in range(4):
            for v in range(4):
                trv = 0.0 + 0.0j
                for i in range(ni):
                    for j in range(nj):
                        trv += X[u, i, j] * Y[v, j, i]
                acc_A[ip, ir, u, v] += trv * weight_over_pi

@njit(parallel=True)
def _compute_dGk_mesh_njit(Gk_mesh, gK_mesh, Nk1, Nk2, Nk3):
    """
    Compute dGk = Gk @ gK @ Gk for entire mesh.
    """
    dim = Gk_mesh.shape[3]
    dGk_mesh = np.zeros((Nk1, Nk2, Nk3, dim, dim), dtype=np.complex128)
    for k1 in prange(Nk1):
        for k2 in range(Nk2):
            for k3 in range(Nk3):
                G = Gk_mesh[k1, k2, k3]
                g = gK_mesh[k1, k2, k3]
                dGk_mesh[k1, k2, k3] = G @ g @ G
    return dGk_mesh


def _onsite_block_dDelta(block_up, block_dn):
    """
    Onsite local-block derivative convention.

    Keep the full local orbital block:

      dDelta_i^{mn}(Rp) = g_up^{mn}(Re=0, Rp) - g_dn^{mn}(Re=0, Rp)
    """
    return block_up - block_dn

@njit(parallel=True)
def _accumulate_dJ_from_GR_njit(GR_u, dGR_u, GR_d_m, dGR_d_m, D_i, dD_i, D_j, dD_j):
    """
    Accumulate dJ/du contribution from already Fourier-transformed real-space blocks.

    Inputs are per-R matrices:
      - GR_u[ir]    = G_ij^up(R_ir)
      - GR_d_m[ir]  = G_ji^dn(-R_ir)
      - dGR_u[ir]   = dG_ij^up(R_ir)
      - dGR_d_m[ir] = dG_ji^dn(-R_ir)
    """
    n_R = GR_u.shape[0]
    out = np.zeros(n_R, dtype=np.complex128)
    for ir in prange(n_R):
        out[ir] = lkag_deriv_trace_njit(
            D_i, dD_i,
            GR_u[ir], dGR_u[ir],
            D_j, dD_j,
            GR_d_m[ir], dGR_d_m[ir]
        )
    return out


@njit(parallel=True)
def _accumulate_dJ_tensor_from_GR_njit(
    GR, dGR,
    GRm, dGRm,
    Da_i, dDa_i,
    Db_j, dDb_j
):
    """
    Accumulate dJ^{ab}/du for one energy point in spinor basis.
    Parallel over R-vectors.
    """
    n_R = GR.shape[0]
    out = np.zeros(n_R, dtype=np.complex128)
    for ir in prange(n_R):
        out[ir] = lkag_tensor_deriv_trace_njit(
            Da_i, dDa_i,
            GR[ir], dGR[ir],
            Db_j, dDb_j,
            GRm[ir], dGRm[ir]
        )
    return out

@njit
def compute_Gk_njit(evals, evecs, z):
    """
    Compute G(k, z) = C(k) (z - epsilon(k))^-1 C(k)^dagger
    """
    dim = evals.shape[0]
    G = np.zeros((dim, dim), dtype=np.complex128)
    for b in range(dim):
        denominator = z - evals[b]
        inv_den = 1.0 / denominator
        for i in range(dim):
            c_i = evecs[i, b]
            for j in range(dim):
                G[i, j] += (c_i * evecs[j, b].conjugate()) * inv_den
    return G

@njit
def lkag_trace_njit(Delta_i, Gij_up, Delta_j, Gji_down):
    """
    Scalar LKAG trace for the bond convention:

      Gij_up(R) = < i, R | G^up | j, 0 >

    so the physical pair is (j, 0) <-> (i, R).
    By cyclicity of the trace,

      Tr[ Delta_i Gij_up Delta_j Gji_down ]

    is equivalent to the more intuitive loop reading

      Tr[ Delta_j Gji_down Delta_i Gij_up ],

    i.e. start at j@0, propagate to i@R, scatter on i, propagate back,
    and scatter on j.
    """
    tmp = Delta_i @ Gij_up @ Delta_j @ Gji_down
    res = 0.0 + 0.0j
    for i in range(tmp.shape[0]):
        res += tmp[i, i]
    return res

@njit
def lkag_deriv_trace_njit(D_i, dD_i, G_ij_u, dG_ij_u, D_j, dD_j, G_ji_d, dG_ji_d):
    """
    Trace for dJ/du in the same bond convention:

      G_ij_u(R) = < i, R | G^up | j, 0 >
      G_ji_d(-R) = < j, 0 | G^dn | i, R >

    Physical reading is a closed loop starting at j@0.  The code keeps the
    algebraically equivalent ordering

      Tr [ dD_i G_ij_u D_j G_ji_d
          + D_i dG_ij_u D_j G_ji_d
          + D_i G_ij_u dD_j G_ji_d
          + D_i G_ij_u D_j dG_ji_d ]

    but the caller must provide:
      - dD_j evaluated at R_p        (origin-site derivative)
      - dD_i evaluated at R_p - R    (target-site derivative)
    """
    term1 = dD_i @ G_ij_u @ D_j @ G_ji_d
    term2 = D_i @ dG_ij_u @ D_j @ G_ji_d
    term3 = D_i @ G_ij_u @ dD_j @ G_ji_d
    term4 = D_i @ G_ij_u @ D_j @ dG_ji_d

    res = 0.0 + 0.0j
    for i in range(D_i.shape[0]):
        res += term1[i,i] + term2[i,i] + term3[i,i] + term4[i,i]
    return res


@njit
def lkag_tensor_deriv_trace_njit(
    Da_i, dDa_i,
    G_ij, dG_ij,
    Db_j, dDb_j,
    G_ji, dG_ji
):
    """
    General spinor/tensor trace for dJ^{ab}/du.

    Formula:
      Tr [ dD_i^a G_ij D_j^b G_ji
         + D_i^a dG_ij D_j^b G_ji
         + D_i^a G_ij dD_j^b G_ji
         + D_i^a G_ij D_j^b dG_ji ]
    """
    term1 = dDa_i @ G_ij @ Db_j @ G_ji
    term2 = Da_i @ dG_ij @ Db_j @ G_ji
    term3 = Da_i @ G_ij @ dDb_j @ G_ji
    term4 = Da_i @ G_ij @ Db_j @ dG_ji

    res = 0.0 + 0.0j
    for i in range(Da_i.shape[0]):
        res += term1[i,i] + term2[i,i] + term3[i,i] + term4[i,i]
    return res


@njit
def lkag_deriv_terms_njit(D_i, dD_i, G_ij_u, dG_ij_u, D_j, dD_j, G_ji_d, dG_ji_d):
    """
    Return the four separate dJ trace terms before the final Im/(4 pi) post-factor.
    Ordering:
      term1 = dD_i G_ij_u D_j G_ji_d
      term2 = D_i dG_ij_u D_j G_ji_d
      term3 = D_i G_ij_u dD_j G_ji_d
      term4 = D_i G_ij_u D_j dG_ji_d
    """
    term1_mat = dD_i @ G_ij_u @ D_j @ G_ji_d
    term2_mat = D_i @ dG_ij_u @ D_j @ G_ji_d
    term3_mat = D_i @ G_ij_u @ dD_j @ G_ji_d
    term4_mat = D_i @ G_ij_u @ D_j @ dG_ji_d

    t1 = 0.0 + 0.0j
    t2 = 0.0 + 0.0j
    t3 = 0.0 + 0.0j
    t4 = 0.0 + 0.0j
    for i in range(D_i.shape[0]):
        t1 += term1_mat[i, i]
        t2 += term2_mat[i, i]
        t3 += term3_mat[i, i]
        t4 += term4_mat[i, i]
    return t1, t2, t3, t4

@njit(parallel=True)
def _compute_gK_mesh_njit(g_Rp, Nk1, Nk2, Nk3):
    """
    Vectorized FT for derivatives. (nx, ny, nz, dim, dim) -> (Nk1, Nk2, Nk3, dim, dim)
    """
    nx, ny, nz = g_Rp.shape[:3]
    dim = g_Rp.shape[3]
    res = np.zeros((Nk1, Nk2, Nk3, dim, dim), dtype=np.complex128)

    # Precompute R coordinates
    rx = np.zeros(nx)
    for i in range(nx): rx[i] = i if i < nx//2 else i - nx
    ry = np.zeros(ny)
    for i in range(ny): ry[i] = i if i < ny//2 else i - ny
    rz = np.zeros(nz)
    for i in range(nz): rz[i] = i if i < nz//2 else i - nz

    for k1 in prange(Nk1):
        for k2 in range(Nk2):
            for k3 in range(Nk3):
                # Phase for this k
                kx, ky, kz = k1/Nk1, k2/Nk2, k3/Nk3
                for ix in range(nx):
                    px = np.exp(1j * 2 * np.pi * kx * rx[ix])
                    for iy in range(ny):
                        pxy = px * np.exp(1j * 2 * np.pi * ky * ry[iy])
                        for iz in range(nz):
                            phase = pxy * np.exp(1j * 2 * np.pi * kz * rz[iz])
                            res[k1, k2, k3] += g_Rp[ix, iy, iz] * phase
    return res

@njit
def _get_gK_core(g_Rp, kvec, nx, ny, nz):
    """
    Fourier transform Hamiltonian derivatives from electronic real-space (Re) to k-space.

    [Phase 0 Tensor Spec]
    - g_Rp: (nx, ny, nz, dim, dim) - Electronic Re-grid components for a FIXED phonon site Rp.
    - kvec: (3,) fractional k-point coordinate.
    """
    m, n = g_Rp.shape[3], g_Rp.shape[4]
    res = np.zeros((m, n), dtype=np.complex128)
    for ix in range(nx):
        rx = ix if ix < nx//2 else ix - nx
        for iy in range(ny):
            ry = iy if iy < ny//2 else iy - ny
            for iz in range(nz):
                rz = iz if iz < nz//2 else iz - nz
                phase = np.exp(1j * 2 * np.pi * (kvec[0]*rx + kvec[1]*ry + kvec[2]*rz))
                for im in range(m):
                    for in_ in range(n):
                        res[im, in_] += g_Rp[ix, iy, iz, im, in_] * phase
    return res

class LKAGSolver:
    r"""
    Core solver for Exchange parameters (J) and their derivatives (dJ/du).
    """
    def __init__(self, result_dir, efermi=0.0, formalism="collinear"):
        if formalism not in {"collinear", "noncollinear"}:
            raise ValueError(f"Unsupported formalism: {formalism}")
        self.result_dir = result_dir
        self.efermi = efermi
        self.formalism = formalism
        self.H_R_dict = {'up': {}, 'down': {}}
        self.R_list = None
        self.dim = None
        self.k_cache = {'up': None, 'down': None, 'grid': None}
        self.last_debug_trace = None
        self.spinor_hr_path = None
        self.spinor_dim = None
        self.spinor_basis_order = "spin_major"
        self.spinor_site_indices = None
        self.H_spinor_R_dict = None
        self.H_spinor_R_tensor = None
        self.R_list_spinor = None
        self.R_to_idx_spinor = None
        self.idx_000_spinor = None
        self.k_cache_spinor = {'evals': None, 'evecs': None, 'grid': None}

        self.mag_atoms = []
        self.atom_names = {}
        self.orbital_slices = {}
        self.orbital_mapping_summary = None
        self.g_provider = None

    def configure_magnetic_subspace(self, slices, names=None):
        """Configure material-independent local orbital slices explicitly."""
        if self.dim is None:
            raise RuntimeError(
                "Solver dimension is not initialized. Load Hamiltonian data first (load_hr/load_hr_spinor)."
            )
        normalized = {}
        for key, value in dict(slices).items():
            site = int(key)
            if not isinstance(value, slice) or value.step not in (None, 1):
                raise TypeError(f"Site {site} must map to a contiguous slice")
            start = 0 if value.start is None else int(value.start)
            stop = self.dim if value.stop is None else int(value.stop)
            if start < 0 or stop <= start or stop > self.dim:
                raise ValueError(f"Invalid slice for site {site}: {value}, dimension={self.dim}")
            normalized[site] = slice(start, stop)
        if not normalized:
            raise ValueError("At least one magnetic-site slice is required")
        self.mag_atoms = sorted(normalized)
        self.atom_names = {
            site: str(names[site]) if names and site in names else f"Site{site + 1}"
            for site in self.mag_atoms
        }
        self.orbital_slices = normalized
        self.orbital_mapping_summary = {
            "source": "explicit_slices",
            "sites": [
                {
                    "local_idx": site,
                    "name": self.atom_names[site],
                    "slice": [normalized[site].start, normalized[site].stop],
                    "n_orb": normalized[site].stop - normalized[site].start,
                }
                for site in self.mag_atoms
            ],
        }

    def load_wannier_centers(self, xyz_file):
        """Parse Wannier90 centres.xyz file to get orbital centers in Cartesian (Angstrom)."""
        centers = []
        with open(xyz_file, 'r') as f:
            lines = f.readlines()
            # First line is count, second is comment
            for line in lines[2:]:
                parts = line.split()
                if len(parts) >= 4 and parts[0] == 'X':
                    centers.append([float(parts[1]), float(parts[2]), float(parts[3])])
        self.wannier_centers = np.array(centers) # shape (N_wan, 3)
        print(f"Loaded {len(self.wannier_centers)} Wannier centers from {xyz_file}")

    def load_hr(self, prefix_up, prefix_down, *, unit="ev"):
        """Parse Wannier90 hr.dat files."""
        if self.formalism != "collinear":
            raise RuntimeError(
                "load_hr(up/down) is only valid in collinear mode. "
                "Use load_hr_spinor() for noncollinear scaffold path."
            )

        u = str(unit).strip().lower()
        if u == "ev":
            scale = 1.0
        elif u in {"ry", "ryd", "rydberg"}:
            scale = RY_TO_EV
        elif u in {"ha", "hartree"}:
            scale = HA_TO_EV
        else:
            raise ValueError(f"Unsupported hr unit='{unit}'. Use ev|ry|ha.")

        def parse_file(fname):
            with open(fname, 'r') as f:
                f.readline() # header
                nw = int(f.readline())
                nr = int(f.readline())
                degen = []
                while len(degen) < nr:
                    degen.extend(map(int, f.readline().split()))
                degen = np.array(degen)

                hr_dict = {}
                for ir in range(nr):
                    w = degen[ir]
                    for _ in range(nw * nw):
                        line = f.readline().split()
                        if not line: break
                        r1, r2, r3, m, n = map(int, line[:5])
                        val = ((float(line[5]) + 1j * float(line[6])) / w) * scale
                        R = (r1, r2, r3)
                        if R not in hr_dict:
                            hr_dict[R] = np.zeros((nw, nw), dtype=complex)
                        hr_dict[R][m-1, n-1] = val
                return hr_dict, nw

        path_up = os.path.join(self.result_dir, prefix_up + "_hr.dat")
        path_dn = os.path.join(self.result_dir, prefix_down + "_hr.dat")

        self.H_R_dict['up'], self.dim = parse_file(path_up)
        self.H_R_dict['down'], _ = parse_file(path_dn)
        print(f"[load_hr] loaded in unit='{u}' and converted to eV (scale={scale:.12g})")

        # Phase 0: Migrate to Tensor-based structure for vectorization
        self.R_list = np.array(list(self.H_R_dict['up'].keys()))

        # O(1) lookup for R-vector indices (e.g., origin)
        self.R_to_idx = {tuple(R): i for i, R in enumerate(self.R_list)}
        self.idx_000 = self.R_to_idx.get((0,0,0), None)

        self.H_R_tensor = np.stack([
            np.array([self.H_R_dict['up'][tuple(R)] for R in self.R_list]),
            np.array([self.H_R_dict['down'][tuple(R)] for R in self.R_list])
        ])

    def diagnose_orbital_permutation(self, *, apply=False, topk=12):
        """
        Diagnose possible up/down orbital index mismatch and optionally align down-spin basis.

        Returns
        -------
        dict
            Diagnostic summary including suggested permutation and signature costs.
        """
        if self.H_R_tensor is None or self.R_list is None:
            raise RuntimeError("Hamiltonian tensors are not loaded. Call load_hr() first.")
        hup = self.H_R_tensor[0]  # (nR, dim, dim)
        hdn = self.H_R_tensor[1]
        dim = int(hup.shape[1])
        if hdn.shape[1] != dim:
            raise ValueError("up/down dimensions do not match.")

        if self.idx_000 is None:
            raise KeyError("R=(0,0,0) onsite block is missing; cannot diagnose permutation robustly.")

        h0u = hup[self.idx_000]
        h0d = hdn[self.idx_000]
        diag_u = np.real(np.diag(h0u))
        diag_d = np.real(np.diag(h0d))
        row_u = np.sqrt(np.sum(np.abs(hup) ** 2, axis=(0, 2)))
        row_d = np.sqrt(np.sum(np.abs(hdn) ** 2, axis=(0, 2)))
        # Lightweight orbital signatures; robust to tiny phase noise.
        sig_u = np.stack([diag_u, row_u], axis=1)
        sig_d = np.stack([diag_d, row_d], axis=1)

        # Scale features to avoid norm-dominated matching.
        scale = np.maximum(np.std(np.vstack([sig_u, sig_d]), axis=0), 1e-12)
        su = sig_u / scale[None, :]
        sd = sig_d / scale[None, :]
        cost = np.linalg.norm(su[:, None, :] - sd[None, :, :], axis=2)
        rows, cols = linear_sum_assignment(cost)
        if not np.array_equal(rows, np.arange(dim)):
            raise RuntimeError("Internal assignment row order mismatch.")
        perm = cols.astype(np.int64)  # up-index i corresponds to down old-index perm[i]

        # Consistency metrics before/after permutation on onsite block.
        before = float(np.linalg.norm(h0u - h0d) / max(np.linalg.norm(h0u), 1e-14))
        h0d_p = h0d[np.ix_(perm, perm)]
        after = float(np.linalg.norm(h0u - h0d_p) / max(np.linalg.norm(h0u), 1e-14))
        moved = int(np.count_nonzero(perm != np.arange(dim)))
        top_pairs = np.argsort(cost[np.arange(dim), perm])[: min(int(topk), dim)]

        report = {
            "dim": dim,
            "moved_orbitals": moved,
            "onsite_rel_before": before,
            "onsite_rel_after": after,
            "perm_up_to_dn_old": perm,
            "best_cost_mean": float(np.mean(cost[np.arange(dim), perm])),
            "top_matches": [(int(i), int(perm[i]), float(cost[i, perm[i]])) for i in top_pairs],
        }

        if apply and moved > 0:
            # Reorder down-spin basis globally: H_dn'[i,j] = H_dn_old[perm[i], perm[j]]
            self.H_R_tensor[1] = self.H_R_tensor[1][:, perm, :][:, :, perm]
            # Keep dict view consistent with tensor view.
            for ir, R in enumerate(self.R_list):
                self.H_R_dict["down"][tuple(R)] = self.H_R_tensor[1, ir]
            print(
                f"[perm-check] Applied down-spin orbital permutation: moved={moved}, "
                f"onsite_rel {before:.3e} -> {after:.3e}"
            )
        else:
            print(
                f"[perm-check] Suggested moved={moved}, onsite_rel {before:.3e} -> {after:.3e} "
                f"(apply={bool(apply)})"
            )
        return report

    def load_hr_spinor(self, hr_path, basis_order="spin_major"):
        """
        Load spinor hr.dat for noncollinear scaffold mode.
        Currently used for IO/metadata validation before numerical kernel rollout.
        """
        if self.formalism != "noncollinear":
            raise RuntimeError("load_hr_spinor() is only valid in noncollinear mode.")
        if not os.path.isabs(hr_path):
            hr_path = os.path.join(self.result_dir, hr_path)
        hr_path = os.path.abspath(hr_path)
        if not os.path.exists(hr_path):
            raise FileNotFoundError(f"Spinor hr.dat not found: {hr_path}")

        parsed = read_wannier_hr(hr_path)
        if len(parsed) == 3:
            dim, _degens, h_dict = parsed
        elif len(parsed) == 4:
            dim, _nrpts, _degens, h_dict = parsed
        else:
            raise ValueError(f"Unexpected read_wannier_hr return length={len(parsed)}: {hr_path}")

        order = str(basis_order).strip().lower().replace("-", "_")
        aliases = {
            "spin_major": "spin_major",
            "spin": "spin_major",
            "up_down": "spin_major",
            "site_interleaved": "site_interleaved",
            "site": "site_interleaved",
            "atom_interleaved": "site_interleaved",
            "atom": "site_interleaved",
        }
        if order not in aliases:
            raise ValueError(
                f"Unsupported spinor basis_order={basis_order!r}. "
                "Use spin_major or site_interleaved."
            )
        order = aliases[order]
        if int(dim) % 2 != 0:
            raise ValueError(f"Spinor hr.dat dimension must be even, got {dim}")

        self.spinor_hr_path = hr_path
        self.spinor_dim = int(dim)
        self.dim = int(dim) // 2
        self.spinor_basis_order = order
        self.H_spinor_R_dict = h_dict
        self.R_list_spinor = np.array(list(h_dict.keys()))
        self.R_to_idx_spinor = {tuple(R): i for i, R in enumerate(self.R_list_spinor)}
        self.idx_000_spinor = self.R_to_idx_spinor.get((0, 0, 0), None)
        if self.idx_000_spinor is None:
            raise KeyError("Spinor hr.dat does not contain R=(0,0,0) onsite block.")
        self.H_spinor_R_tensor = np.array([h_dict[tuple(R)] for R in self.R_list_spinor])
        print(
            f"[noncollinear] Loaded spinor hr.dat: {hr_path} "
            f"(dim={self.spinor_dim}, R={len(h_dict)}, basis_order={self.spinor_basis_order})"
        )

    def get_HK_spinor(self, kvec):
        """
        Fourier transform spinor H(R) -> H(k), using loaded spinor hr tensor.
        """
        if self.H_spinor_R_tensor is None or self.R_list_spinor is None:
            raise RuntimeError("Spinor Hamiltonian is not loaded. Call load_hr_spinor() first.")
        phases = np.exp(1j * 2 * np.pi * (self.R_list_spinor @ kvec))
        return np.tensordot(phases, self.H_spinor_R_tensor, axes=(0, 0))

    def precompute_k_mesh_spinor(self, k_grid):
        """
        Precompute spinor eigenvalues/eigenvectors for noncollinear tensor calculation.
        """
        if self.formalism != "noncollinear":
            raise RuntimeError("precompute_k_mesh_spinor() is only valid in noncollinear mode.")
        if self.H_spinor_R_tensor is None:
            raise RuntimeError("Spinor Hamiltonian not loaded. Call load_hr_spinor() first.")

        Nk1, Nk2, Nk3 = k_grid
        print(f"Precomputing spinor eigensystems for {Nk1}x{Nk2}x{Nk3} grid...")
        evals_mesh = np.zeros((Nk1, Nk2, Nk3, self.spinor_dim))
        evecs_mesh = np.zeros((Nk1, Nk2, Nk3, self.spinor_dim, self.spinor_dim), dtype=complex)

        for k1 in range(Nk1):
            for k2 in range(Nk2):
                for k3 in range(Nk3):
                    kvec = np.array([k1 / Nk1, k2 / Nk2, k3 / Nk3])
                    Hk = self.get_HK_spinor(kvec)
                    evals, evecs = np.linalg.eigh(Hk)
                    evals_mesh[k1, k2, k3] = evals
                    evecs_mesh[k1, k2, k3] = evecs

        self.k_cache_spinor['evals'] = evals_mesh
        self.k_cache_spinor['evecs'] = evecs_mesh
        self.k_cache_spinor['grid'] = k_grid
        print("Spinor eigensystem caching complete.")

    def _spinor_atom_indices(self, atom_idx):
        """
        Build spinor index array for a magnetic atom:
          [up-orbitals(atom), down-orbitals(atom)].
        """
        up_idx, dn_idx = self._spinor_atom_up_dn_indices(atom_idx)
        return np.concatenate([up_idx, dn_idx])

    def _spinor_atom_up_dn_indices(self, atom_idx):
        """
        Build separate up/down spinor index arrays for a magnetic atom.

        The returned arrays preserve orbital order, so orbital k in up_idx is
        paired with orbital k in dn_idx. This is the canonical representation
        used internally before forming TB2J-style paired blocks.
        """
        if self.spinor_dim is None:
            raise RuntimeError("Spinor dimension is undefined. Load spinor hr first.")
        if self.spinor_site_indices is not None:
            atom_i = int(atom_idx)
            if atom_i not in self.spinor_site_indices:
                raise KeyError(
                    f"No explicit spinor slice for local magnetic site {atom_i}. "
                    f"Known sites: {sorted(self.spinor_site_indices)}"
                )
            idx = self.spinor_site_indices[atom_i]
            n2 = int(idx.shape[0])
            return idx[: n2 // 2], idx[n2 // 2 :]
        half = self.spinor_dim // 2
        slc = self.orbital_slices[atom_idx]
        if self.spinor_basis_order == "spin_major":
            up_idx = np.arange(slc.start, slc.stop)
            dn_idx = up_idx + half
        elif self.spinor_basis_order == "site_interleaved":
            n = int(slc.stop - slc.start)
            base = 2 * int(slc.start)
            up_idx = np.arange(base, base + n)
            dn_idx = np.arange(base + n, base + 2 * n)
        else:
            raise RuntimeError(f"Unsupported spinor_basis_order={self.spinor_basis_order}")
        return up_idx, dn_idx

    def configure_spinor_site_indices(self, site_indices):
        """
        Override spinor basis indices per local magnetic site.

        Parameters
        ----------
        site_indices : dict[int, array-like]
            Each value must be ordered as [up indices..., down indices...] for
            that local magnetic site.
        """
        if self.spinor_dim is None:
            raise RuntimeError("Load spinor hr before configuring explicit spinor site indices.")
        out = {}
        for site, idx in site_indices.items():
            arr = np.asarray(idx, dtype=np.int64).reshape(-1)
            if arr.size == 0 or arr.size % 2 != 0:
                raise ValueError(f"Spinor indices for site {site} must be nonempty and even-length, got {arr.size}.")
            if np.any(arr < 0) or np.any(arr >= int(self.spinor_dim)):
                raise ValueError(f"Spinor indices for site {site} exceed spinor dimension {self.spinor_dim}: {arr}")
            if len(set(int(x) for x in arr.tolist())) != arr.size:
                raise ValueError(f"Duplicate spinor indices for site {site}: {arr}")
            out[int(site)] = arr
        self.spinor_site_indices = out
        print(f"[noncollinear] Explicit spinor site indices configured: {out}")

    @staticmethod
    def _extract_exchange_fields_from_spinor_block(h_loc):
        """
        Decompose local spinor onsite block into Pauli components:
          h_loc = h0 + sigma_x*Bx + sigma_y*By + sigma_z*Bz
        where each B* is an orbital-space matrix.
        h_loc shape: (2n, 2n)
        Returns Bx, By, Bz as (n, n).
        """
        n2 = h_loc.shape[0]
        if n2 % 2 != 0:
            raise ValueError(f"Local spinor block dimension must be even, got {n2}.")
        n = n2 // 2
        huu = h_loc[:n, :n]
        hud = h_loc[:n, n:]
        hdu = h_loc[n:, :n]
        hdd = h_loc[n:, n:]

        bx = 0.5 * (hud + hdu)
        by = (hdu - hud) / (2.0j)
        bz = 0.5 * (huu - hdd)
        return bx, by, bz

    @staticmethod
    def _compose_spinor_operator(axis, bmat):
        """
        Compose D^axis from orbital-space bmat using Pauli structure.
        Returns spinor-space matrix with shape (2n, 2n).
        """
        n = bmat.shape[0]
        z = np.zeros((n, n), dtype=complex)
        if axis == 'x':
            return np.block([[z, bmat], [bmat, z]])
        if axis == 'y':
            return np.block([[z, -1j * bmat], [1j * bmat, z]])
        if axis == 'z':
            return np.block([[bmat, z], [z, -bmat]])
        raise ValueError(f"Unsupported axis: {axis}")

    def _build_local_exchange_operators(self, atom_idx):
        """
        Build local D_i^x, D_i^y, D_i^z operators for one magnetic atom
        from the spinor onsite block at R=(0,0,0).
        """
        h0 = self.H_spinor_R_tensor[self.idx_000_spinor]
        idx = self._spinor_atom_indices(atom_idx)
        h_loc = h0[np.ix_(idx, idx)]
        bx, by, bz = self._extract_exchange_fields_from_spinor_block(h_loc)
        return {
            'x': self._compose_spinor_operator('x', bx),
            'y': self._compose_spinor_operator('y', by),
            'z': self._compose_spinor_operator('z', bz),
        }

    @staticmethod
    def _pauli_block_all_contiguous(block):
        """
        Pauli decomposition for a spinor block ordered as [[uu, ud], [du, dd]]
        where each sub-block is orbital-space.
        Returns (I, x, y, z) orbital-space blocks.
        """
        nrow2, ncol2 = block.shape
        if nrow2 % 2 != 0 or ncol2 % 2 != 0:
            raise ValueError(f"Spinor block must have even shape, got {block.shape}.")
        nrow = nrow2 // 2
        ncol = ncol2 // 2
        guu = block[:nrow, :ncol]
        gud = block[:nrow, ncol:]
        gdu = block[nrow:, :ncol]
        gdd = block[nrow:, ncol:]
        g0 = 0.5 * (guu + gdd)
        gx = 0.5 * (gud + gdu)
        gy = 0.5j * (gud - gdu)
        gz = 0.5 * (guu - gdd)
        return g0, gx, gy, gz

    @staticmethod
    def _pauli_block_all_tb2j(block):
        """
        TB2J Pauli decomposition for orbital-paired spinor order:
          [orb0_up, orb0_dn, orb1_up, orb1_dn, ...].
        """
        if block.shape[0] % 2 != 0 or block.shape[1] % 2 != 0:
            raise ValueError(f"Spinor block must have even shape, got {block.shape}.")
        a00 = block[::2, ::2]
        a01 = block[::2, 1::2]
        a10 = block[1::2, ::2]
        a11 = block[1::2, 1::2]
        g0 = 0.5 * (a00 + a11)
        gx = 0.5 * (a01 + a10)
        gy = 0.5j * (a01 - a10)
        gz = 0.5 * (a00 - a11)
        return g0, gx, gy, gz

    def _build_local_tb2j_projector(self, atom_idx):
        """
        TB2J-like local projector:
          P_i = e_x Mx + e_y My + e_z Mz,
        where e is the normalized spin-vector estimated from traces of Pauli blocks.
        """
        h0 = self.H_spinor_R_tensor[self.idx_000_spinor]
        up_idx, dn_idx = self._spinor_atom_up_dn_indices(atom_idx)
        if up_idx.shape[0] != dn_idx.shape[0]:
            raise ValueError(
                f"Up/down orbital counts differ for atom {atom_idx}: "
                f"{up_idx.shape[0]} vs {dn_idx.shape[0]}"
            )
        idx = np.empty(up_idx.shape[0] * 2, dtype=np.int64)
        idx[0::2] = up_idx
        idx[1::2] = dn_idx
        h_loc = h0[np.ix_(idx, idx)]
        _, mx, my, mz = self._pauli_block_all_tb2j(h_loc)
        ex = np.trace(mx)
        ey = np.trace(my)
        ez = np.trace(mz)
        evec = np.array([ex, ey, ez], dtype=np.complex128)
        norm = np.linalg.norm(evec)
        if norm < 1e-14:
            return np.zeros_like(mx)
        evec = evec / norm
        return mx * evec[0] + my * evec[1] + mz * evec[2]

    def compute_J_tensor_bulk(
        self,
        idx_pairs,
        R_list,
        k_grid,
        energy_mesh,
        axes=('x', 'y', 'z'),
        kernel="tb2j",
        tb2j_debug_pairing=False,
    ):
        r"""
        Noncollinear exchange tensor calculation.

        kernel="tb2j":
          TB2J-like route:
            1) build A^{uv}(R) from Pauli blocks and local projectors P_i
            2) map to anisotropic tensor with TB2J convention:
               Jani_{ab}(R) = Im[A^{ab}(R) + A^{ab}(-R, j, i)], a,b in (x,y,z)

        kernel="direct":
          Direct spinor-operator route:
            J_{ij}^{ab}(R) = (1000 / 4pi) Im \int dE Tr[D_i^a G_{ij}(E,R) D_j^b G_{ji}(E,-R)].

        Returns
        -------
        dict: {(idx_i, idx_j, R_tuple, a, b): value_meV}
        """
        if self.formalism != "noncollinear":
            raise RuntimeError("compute_J_tensor_bulk() requires formalism='noncollinear'.")
        if self.H_spinor_R_tensor is None:
            raise RuntimeError("Spinor Hamiltonian not loaded. Call load_hr_spinor().")

        # Ensure spinor cache
        if (
            self.k_cache_spinor['evals'] is None
            or self.k_cache_spinor['grid'] != k_grid
        ):
            self.precompute_k_mesh_spinor(k_grid)

        evals = self.k_cache_spinor['evals']
        evecs = self.k_cache_spinor['evecs']
        Nk1, Nk2, Nk3 = k_grid

        req_idx_pairs = [tuple(p) for p in idx_pairs]
        acc_idx_pairs = list(req_idx_pairs)
        req_R_list = [tuple(int(x) for x in r) for r in R_list]
        acc_R_list = list(req_R_list)
        raw_R_req = np.array(req_R_list, dtype=np.int64)
        n_pairs_req = len(req_idx_pairs)
        n_R_req = len(req_R_list)
        n_ax = len(axes)
        kernel_norm = str(kernel).strip().lower()
        if kernel_norm not in {"tb2j", "direct"}:
            raise ValueError(f"Unsupported kernel={kernel}. Use 'tb2j' or 'direct'.")
        if kernel_norm == "tb2j":
            # TB2J A->Jani mapping needs (j, i, -R) mates.
            pair_set = set(acc_idx_pairs)
            for i, j in req_idx_pairs:
                if (j, i) not in pair_set:
                    acc_idx_pairs.append((j, i))
                    pair_set.add((j, i))
            r_set = set(acc_R_list)
            for r in req_R_list:
                rm = (-r[0], -r[1], -r[2])
                if rm not in r_set:
                    acc_R_list.append(rm)
                    r_set.add(rm)
        n_pairs_acc = len(acc_idx_pairs)
        n_R_acc = len(acc_R_list)
        raw_R_acc = np.array(acc_R_list, dtype=np.int64)

        if kernel_norm == "direct":
            # Prebuild local exchange operators per atom
            d_ops = {}
            for pair in req_idx_pairs:
                for atom_idx in pair:
                    if atom_idx not in d_ops:
                        d_ops[atom_idx] = self._build_local_exchange_operators(atom_idx)

            acc = np.zeros((n_pairs_req, n_R_req, n_ax, n_ax), dtype=complex)
            pair_ctx = []
            for idx_i, idx_j in req_idx_pairs:
                idx_i_spin = self._spinor_atom_indices(idx_i)
                idx_j_spin = self._spinor_atom_indices(idx_j)
                d_i_stack = np.stack([d_ops[idx_i][ax] for ax in axes], axis=0)
                d_j_stack = np.stack([d_ops[idx_j][ax] for ax in axes], axis=0)
                pair_ctx.append((idx_i_spin, idx_j_spin, d_i_stack, d_j_stack))
        else:
            # TB2J-like accumulator in Pauli space (u,v in 0,x,y,z)
            acc_A = np.zeros((n_pairs_acc, n_R_acc, 4, 4), dtype=np.complex128)
            p_ops = {}
            pair_ctx = []
            for idx_i, idx_j in acc_idx_pairs:
                if idx_i not in p_ops:
                    p_ops[idx_i] = self._build_local_tb2j_projector(idx_i)
                if idx_j not in p_ops:
                    p_ops[idx_j] = self._build_local_tb2j_projector(idx_j)
                idx_i_spin = self._spinor_atom_indices(idx_i)
                idx_j_spin = self._spinor_atom_indices(idx_j)
                pair_ctx.append((idx_i, idx_j, idx_i_spin, idx_j_spin, p_ops[idx_i], p_ops[idx_j]))

            # Dense metadata for njit kernel
            ni_arr = np.zeros((n_pairs_acc,), dtype=np.int64)
            nj_arr = np.zeros((n_pairs_acc,), dtype=np.int64)
            max_ni = 0
            max_nj = 0
            for ip, (_, _, idx_i_spin, idx_j_spin, _, _) in enumerate(pair_ctx):
                ni2 = idx_i_spin.shape[0]
                nj2 = idx_j_spin.shape[0]
                if ni2 % 2 != 0 or nj2 % 2 != 0:
                    raise RuntimeError("Spinor atom index length must be even.")
                ni = ni2 // 2
                nj = nj2 // 2
                ni_arr[ip] = ni
                nj_arr[ip] = nj
                max_ni = max(max_ni, ni)
                max_nj = max(max_nj, nj)

            P_i = np.zeros((n_pairs_acc, max_ni, max_ni), dtype=np.complex128)
            P_j = np.zeros((n_pairs_acc, max_nj, max_nj), dtype=np.complex128)
            iu_up = np.zeros((n_pairs_acc, max_ni), dtype=np.int64)
            iu_dn = np.zeros((n_pairs_acc, max_ni), dtype=np.int64)
            jv_up = np.zeros((n_pairs_acc, max_nj), dtype=np.int64)
            jv_dn = np.zeros((n_pairs_acc, max_nj), dtype=np.int64)
            for ip, (_, _, idx_i_spin, idx_j_spin, pi, pj) in enumerate(pair_ctx):
                ni = idx_i_spin.shape[0] // 2
                nj = idx_j_spin.shape[0] // 2
                P_i[ip, :ni, :ni] = pi
                P_j[ip, :nj, :nj] = pj
                iu_up[ip, :ni] = idx_i_spin[:ni]
                iu_dn[ip, :ni] = idx_i_spin[ni:]
                jv_up[ip, :nj] = idx_j_spin[:nj]
                jv_dn[ip, :nj] = idx_j_spin[nj:]

            # Index trace snapshot for runtime sanity checks.
            self.last_tb2j_index_trace = {
                "n_pairs_req": int(n_pairs_req),
                "n_pairs_acc": int(n_pairs_acc),
                "n_R_req": int(n_R_req),
                "n_R_acc": int(n_R_acc),
                "max_ni": int(max_ni),
                "max_nj": int(max_nj),
                "sample_pairs": [],
            }
            n_show = min(3, n_pairs_acc)
            for ip in range(n_show):
                ni = int(ni_arr[ip])
                nj = int(nj_arr[ip])
                self.last_tb2j_index_trace["sample_pairs"].append(
                    {
                        "pair": tuple(int(x) for x in acc_idx_pairs[ip]),
                        "ni": ni,
                        "nj": nj,
                        "i_up_head": [int(x) for x in iu_up[ip, : min(4, ni)]],
                        "i_dn_head": [int(x) for x in iu_dn[ip, : min(4, ni)]],
                        "j_up_head": [int(x) for x in jv_up[ip, : min(4, nj)]],
                        "j_dn_head": [int(x) for x in jv_dn[ip, : min(4, nj)]],
                    }
                )
            print(
                f"[tb2j-index] pairs_req={n_pairs_req}, pairs_acc={n_pairs_acc}, "
                f"R_req={n_R_req}, R_acc={n_R_acc}, max_ni={max_ni}, max_nj={max_nj}, "
                f"sample={self.last_tb2j_index_trace['sample_pairs'][0] if n_show > 0 else 'none'}"
            )

        # Single FT call for both real-space Green functions.
        # _sum_GR_bulk_njit uses exp(+i k.R), while the EPR tensor reference uses
        # G_ij(R) = sum_k exp(-i k.R) G_ij(k).  Therefore request -R first for
        # G_ij(R), and +R second for G_ji(-R).
        R_all = np.concatenate([-raw_R_acc, raw_R_acc], axis=0)

        for z, weight in energy_mesh:
            z_eff = z + self.efermi
            Gk = _compute_Gk_mesh_njit(evals, evecs, z_eff, Nk1, Nk2, Nk3)
            GR_all = _sum_GR_bulk_njit(Gk, R_all, Nk1, Nk2, Nk3)
            GR = GR_all[:n_R_acc]
            GRm = GR_all[n_R_acc:]

            if kernel_norm == "direct":
                for ip, (idx_i_spin, idx_j_spin, d_i_stack, d_j_stack) in enumerate(pair_ctx):
                    for ir in range(n_R_req):
                        gij = GR[ir][np.ix_(idx_i_spin, idx_j_spin)]
                        gji = GRm[ir][np.ix_(idx_j_spin, idx_i_spin)]

                        # Vectorized over tensor axes:
                        # A[a] = D_i^a @ G_ij, B[b] = D_j^b @ G_ji
                        # T[a,b] = Tr[A[a] @ B[b]]
                        A = np.einsum("aij,jk->aik", d_i_stack, gij, optimize=True)
                        B = np.einsum("bij,jk->bik", d_j_stack, gji, optimize=True)
                        acc[ip, ir] += np.einsum("aij,bji->ab", A, B, optimize=True) * weight
            else:
                _accumulate_tb2j_A_numba(
                    GR=GR,
                    GRm=GRm,
                    ni_arr=ni_arr,
                    nj_arr=nj_arr,
                    iu_up=iu_up,
                    iu_dn=iu_dn,
                    jv_up=jv_up,
                    jv_dn=jv_dn,
                    P_i=P_i,
                    P_j=P_j,
                    acc_A=acc_A,
                    weight_over_pi=weight / np.pi,
                )

        out = {}
        pref_direct = 1000.0 / (4.0 * np.pi)
        if kernel_norm == "direct":
            for ip, (idx_i, idx_j) in enumerate(req_idx_pairs):
                for ir, r in enumerate(req_R_list):
                    rt = tuple(r)
                    for ia, axa in enumerate(axes):
                        for ib, axb in enumerate(axes):
                            out[(idx_i, idx_j, rt, axa, axb)] = float(pref_direct * np.imag(acc[ip, ir, ia, ib]))
            return out

        # TB2J-like mapping:
        #   M_ab(R) = Im[Aab(R,i,j) + Aab(-R,j,i)], a,b in x,y,z
        # Build symmetric tensor first, then decompose:
        #   Ms = (M + M^T)/2 = Jiso*I + Gamma (Gamma traceless symmetric)
        #   Jiso = Tr(Ms)/3, Gamma = Ms - Jiso*I
        #
        # Exported tensor is Jfull = Jiso*I + Gamma = Ms.
        # (Antisymmetric part is DMI-like and intentionally excluded here.)
        #
        # Note: TB2J A-tensor path does not use LKAG 1/(4pi) prefactor explicitly
        # in A->J decomposition; export here in meV with eV->meV conversion only.
        pref_tb2j = 1000.0
        pair_to_ip = {}
        for ip, pair in enumerate(acc_idx_pairs):
            pair_to_ip.setdefault(tuple(pair), ip)
        r_to_ir = {}
        for ir, r in enumerate(acc_R_list):
            r_to_ir.setdefault(tuple(r), ir)
        ax_to_pauli = {'x': 1, 'y': 2, 'z': 3}

        if tb2j_debug_pairing:
            pairing_records = []
            missing_mates = 0
        else:
            pairing_records = None
            missing_mates = 0

        for ip_req, (idx_i, idx_j) in enumerate(req_idx_pairs):
            for ir_req, r in enumerate(req_R_list):
                rt = tuple(r)
                r_neg = tuple(-x for x in rt)
                ip = pair_to_ip[(idx_i, idx_j)]
                ip_m = pair_to_ip.get((idx_j, idx_i))
                ir_m = r_to_ir.get(r_neg)
                if ip_m is not None and ir_m is not None:
                    val_m = acc_A[ip_m, ir_m]
                    mate_found = True
                else:
                    val_m = np.zeros((4, 4), dtype=np.complex128)
                    mate_found = False
                    missing_mates += 1
                val = acc_A[ip, r_to_ir[rt]]

                # Build raw TB2J-like 3x3 tensor from A^{ab} (a,b in x,y,z).
                m = np.zeros((3, 3), dtype=np.float64)
                for ia in range(3):
                    ua = ia + 1
                    for ib in range(3):
                        vb = ib + 1
                        m[ia, ib] = float(np.imag(val[ua, vb] + val_m[ua, vb]))

                # Symmetric-traceless decomposition for anisotropy
                ms = 0.5 * (m + m.T)
                jiso_ms = float(np.trace(ms) / 3.0)
                gamma = ms - np.eye(3, dtype=np.float64) * jiso_ms
                ma = 0.5 * (m - m.T)

                # TB2J-like isotropic definition (exchange_Jdict convention):
                # Jiso = Im[A00 - Axx - Ayy - Azz] from val(R,i,j) only (no val_m).
                jiso_tb2j = float(np.imag(val[0, 0] - val[1, 1] - val[2, 2] - val[3, 3]))
                # Full tensor = isotropic + symmetric-traceless + antisymmetric(DMI-like)
                jfull = gamma + np.eye(3, dtype=np.float64) * jiso_tb2j + ma

                # TB2J DMI convention:
                # D_i = Re(A^{0i} - A^{i0}), i in {x,y,z}, from val(R,i,j).
                dmi_tb2j = np.array(
                    [
                        np.real(val[0, 1] - val[1, 0]),
                        np.real(val[0, 2] - val[2, 0]),
                        np.real(val[0, 3] - val[3, 0]),
                    ],
                    dtype=np.float64,
                )

                for axa in axes:
                    ia = ax_to_pauli[axa] - 1
                    for axb in axes:
                        ib = ax_to_pauli[axb] - 1
                        out[(idx_i, idx_j, rt, axa, axb)] = float(pref_tb2j * jfull[ia, ib])
                # Store TB2J raw anisotropy block for reporting/comparison.
                out[(idx_i, idx_j, rt, "jani_tb2j", "xx")] = float(pref_tb2j * m[0, 0])
                out[(idx_i, idx_j, rt, "jani_tb2j", "xy")] = float(pref_tb2j * m[0, 1])
                out[(idx_i, idx_j, rt, "jani_tb2j", "xz")] = float(pref_tb2j * m[0, 2])
                out[(idx_i, idx_j, rt, "jani_tb2j", "yx")] = float(pref_tb2j * m[1, 0])
                out[(idx_i, idx_j, rt, "jani_tb2j", "yy")] = float(pref_tb2j * m[1, 1])
                out[(idx_i, idx_j, rt, "jani_tb2j", "yz")] = float(pref_tb2j * m[1, 2])
                out[(idx_i, idx_j, rt, "jani_tb2j", "zx")] = float(pref_tb2j * m[2, 0])
                out[(idx_i, idx_j, rt, "jani_tb2j", "zy")] = float(pref_tb2j * m[2, 1])
                out[(idx_i, idx_j, rt, "jani_tb2j", "zz")] = float(pref_tb2j * m[2, 2])
                out[(idx_i, idx_j, rt, "jiso_tb2j", "s")] = float(pref_tb2j * jiso_tb2j)
                out[(idx_i, idx_j, rt, "jiso_trace", "s")] = float(pref_tb2j * float(np.trace(jfull) / 3.0))
                out[(idx_i, idx_j, rt, "dmi_tb2j", "x")] = float(pref_tb2j * dmi_tb2j[0])
                out[(idx_i, idx_j, rt, "dmi_tb2j", "y")] = float(pref_tb2j * dmi_tb2j[1])
                out[(idx_i, idx_j, rt, "dmi_tb2j", "z")] = float(pref_tb2j * dmi_tb2j[2])
                if tb2j_debug_pairing and len(pairing_records) < 64:
                    pairing_records.append(
                        {
                            "pair": (int(idx_i), int(idx_j)),
                            "R": (int(rt[0]), int(rt[1]), int(rt[2])),
                            "mate_pair": (int(idx_j), int(idx_i)),
                            "mate_R": (int(r_neg[0]), int(r_neg[1]), int(r_neg[2])),
                            "mate_found": bool(mate_found),
                            "ip_m": None if ip_m is None else int(ip_m),
                            "ir_req": int(ir_req),
                            "ir_m": None if ir_m is None else int(ir_m),
                            "A11_im": float(np.imag(val[1, 1])),
                            "A11m_im": float(np.imag(val_m[1, 1])),
                            "Jiso_trace": float(jiso_ms),
                            "Jiso_tb2j": float(jiso_tb2j),
                            "D_tb2j": [float(dmi_tb2j[0]), float(dmi_tb2j[1]), float(dmi_tb2j[2])],
                        }
                    )
        if tb2j_debug_pairing:
            self.last_tb2j_pairing_trace = {
                "n_pairs_req": len(req_idx_pairs),
                "n_pairs_acc": len(acc_idx_pairs),
                "n_R_req": len(req_R_list),
                "n_R_acc": len(acc_R_list),
                "missing_mates": int(missing_mates),
                "sample_records": pairing_records,
            }
            print(
                f"[tb2j-pairing] missing_mates={missing_mates} / "
                f"{len(req_idx_pairs) * len(req_R_list)}"
            )
            if pairing_records:
                print(f"[tb2j-pairing] sample={pairing_records[0]}")
        return out

    def precompute_k_mesh(self, k_grid, nproc=1):
        """
        Precompute all eigenvalues and eigenvectors for the given k-grid.
        This is the core of Phase 1 optimization.
        """
        Nk1, Nk2, Nk3 = k_grid
        nproc_eff = max(1, int(nproc))
        print(f"Precomputing eigensystems for {Nk1}x{Nk2}x{Nk3} grid... (kcache_nproc={nproc_eff})")

        self.k_cache['grid'] = k_grid
        for spin_idx, s_label in [(0, 'up'), (1, 'down')]:
            evals_mesh = np.zeros((Nk1, Nk2, Nk3, self.dim))
            evecs_mesh = np.zeros((Nk1, Nk2, Nk3, self.dim, self.dim), dtype=complex)

            # Keep BLAS thread count conservative when we fan out over k1 workers.
            if nproc_eff > 1:
                os.environ["OMP_NUM_THREADS"] = "1"
                os.environ["MKL_NUM_THREADS"] = "1"
                os.environ["OPENBLAS_NUM_THREADS"] = "1"

            def _solve_k1_slice(k1_idx):
                evals_k1 = np.zeros((Nk2, Nk3, self.dim), dtype=np.float64)
                evecs_k1 = np.zeros((Nk2, Nk3, self.dim, self.dim), dtype=np.complex128)
                for k2 in range(Nk2):
                    for k3 in range(Nk3):
                        kvec = np.array([k1_idx / Nk1, k2 / Nk2, k3 / Nk3], dtype=np.float64)
                        Hk = self.get_HK(kvec, s_label)
                        evals, evecs = np.linalg.eigh(Hk)
                        evals_k1[k2, k3] = evals
                        evecs_k1[k2, k3] = evecs
                return int(k1_idx), evals_k1, evecs_k1

            if nproc_eff == 1 or Nk1 == 1:
                for k1 in range(Nk1):
                    k1_idx, ev1, vc1 = _solve_k1_slice(k1)
                    evals_mesh[k1_idx] = ev1
                    evecs_mesh[k1_idx] = vc1
            else:
                with ThreadPoolExecutor(max_workers=min(nproc_eff, Nk1)) as ex:
                    futs = [ex.submit(_solve_k1_slice, k1) for k1 in range(Nk1)]
                    for fut in as_completed(futs):
                        k1_idx, ev1, vc1 = fut.result()
                        evals_mesh[k1_idx] = ev1
                        evecs_mesh[k1_idx] = vc1

            self.k_cache[s_label] = (evals_mesh, evecs_mesh)
        print("Eigensystem caching complete.")

    def load_g_real(self, filepath):
        """
        Load real-space dH/du derivatives.

        Supported inputs:
        - legacy pickle payload (*.real.pkl)
        - single g(k,q) HDF5 (*.h5)
        - directory containing multiple g(k,q) HDF5 files
        - JSON manifest path (directory scan fallback for sibling *.h5)
        """
        path = os.path.abspath(filepath)
        data = None

        if os.path.isdir(path):
            data = self._load_g_real_from_gkq_h5s(path)
        elif path.lower().endswith(".pkl"):
            with open(path, "rb") as f:
                data = pickle.load(f)
        elif path.lower().endswith(".h5"):
            data = self._load_g_real_from_gkq_h5s([path])
        elif path.lower().endswith(".json"):
            data = self._load_g_real_from_gkq_h5s(os.path.dirname(path))
        else:
            raise ValueError(
                f"Unsupported g_real input: {filepath}. "
                "Use *.real.pkl, g_kq*.h5, or a directory/manifest containing g_kq HDF5 files."
            )

        self.g_real_dict = data["g_real_dict"]
        self.nk_grid = data["nk_grid"]
        self.nq_grid = data["nq_grid"]

        # Normalize keys: some use 'down', some use 'dn'
        new_g = {}
        for (s, a, ax), val in self.g_real_dict.items():
            s_new = "down" if s == "dn" else s
            new_g[(s_new, a, ax)] = val
        self.g_real_dict = new_g

        # Phase 0: Smart Unit Detection
        # New files may include units='eV'. Otherwise assume Hartree and convert.
        if data.get("units") == "eV":
            print("Detected eV units in g_real payload. No conversion needed.")
        else:
            print(f"Warning: No 'units' metadata found. Assuming Hartree units and applying HA_TO_EV ({HA_TO_EV}).")
            for key in self.g_real_dict:
                self.g_real_dict[key] *= HA_TO_EV

        print(f"Loaded {len(self.g_real_dict)} entries from g_real payload.")
        self.g_provider = None

    def load_g_real_epr_native(
        self,
        epr_up,
        epr_dn,
        *,
        atom_labels=None,
        divide_by_ndeg=False,
        value_unit="ry",
    ):
        from slw.exchange.kernels.eph_provider import EPRNativeProvider

        self.g_provider = EPRNativeProvider(
            epr_up=epr_up,
            epr_dn=epr_dn,
            atom_labels=atom_labels,
            divide_by_ndeg=bool(divide_by_ndeg),
            value_unit=value_unit,
        )
        self.g_real_dict = {}
        self.nk_grid = list(self.g_provider.nk_grid)
        self.nq_grid = list(self.g_provider.nq_grid)
        print(f"Loaded EPR-native g provider: nk_grid={tuple(self.nk_grid)} nq_grid={tuple(self.nq_grid)}")

    def _get_q_dims(self, sample_key):
        if self.g_provider is not None:
            return tuple(int(x) for x in self.g_provider.nq_grid)
        return tuple(int(x) for x in self.g_real_dict[sample_key].shape[3:6])

    def _get_gRp(self, spin, atom, ax, rp_idx):
        if self.g_provider is not None:
            return self.g_provider.get_gRp(spin, atom, ax, rp_idx)
        g_real = self.g_real_dict[(spin, atom, ax)]
        rp = self._normalize_rp_idx(rp_idx, g_real.shape[3:6])
        return g_real[:, :, :, rp[0], rp[1], rp[2]]

    @staticmethod
    def _gkq_h5_candidates(path_or_list):
        if isinstance(path_or_list, (list, tuple)):
            return [os.path.abspath(p) for p in path_or_list if str(p).lower().endswith(".h5")]
        root = os.path.abspath(path_or_list)
        if os.path.isdir(root):
            files = []
            for f in os.listdir(root):
                if not f.lower().endswith(".h5"):
                    continue
                fp = os.path.join(root, f)
                if os.path.isfile(fp):
                    files.append(fp)
            return sorted(files)
        if root.lower().endswith(".h5"):
            return [root]
        return []

    def _load_g_real_from_gkq_h5s(self, path_or_list):
        paths = self._gkq_h5_candidates(path_or_list)
        if not paths:
            raise FileNotFoundError(f"No .h5 files found under: {path_or_list}")

        g_real_dict = {}
        key_meta = {}
        nk_grid_ref = None
        nq_grid_ref = None
        loaded = 0
        skipped = 0
        duplicated = 0

        def _basis_rank(basis_name):
            b = str(basis_name).strip().lower()
            if b == "wannier":
                return 2
            if b == "band":
                return 1
            return 0

        for p in paths:
            try:
                info = _load_single_gkq(p)
            except Exception:
                skipped += 1
                continue

            key = info["key"]
            if nk_grid_ref is None:
                nk_grid_ref = tuple(int(x) for x in info["nk_grid"])
                nq_grid_ref = tuple(int(x) for x in info["nq_grid"])
            else:
                if tuple(info["nk_grid"]) != nk_grid_ref or tuple(info["nq_grid"]) != nq_grid_ref:
                    raise ValueError(
                        f"Grid mismatch in {p}: nk/nq={info['nk_grid']}/{info['nq_grid']} "
                        f"vs ref {nk_grid_ref}/{nq_grid_ref}"
                    )

            if key in g_real_dict:
                duplicated += 1
                prev = key_meta[key]
                prev_rank = _basis_rank(prev["basis"])
                new_rank = _basis_rank(info.get("basis", ""))
                prev_mtime = float(prev["mtime"])
                new_mtime = float(os.path.getmtime(p))

                # Priority:
                # 1) higher basis rank (wannier > band > unknown)
                # 2) newer file mtime
                replace = (new_rank > prev_rank) or (new_rank == prev_rank and new_mtime > prev_mtime)
                if replace:
                    g_real_dict[key] = info["g_real"]
                    key_meta[key] = {"path": p, "basis": info.get("basis", ""), "mtime": new_mtime}
                    print(
                        f"[load_g_real] duplicate key {key}: replacing "
                        f"{prev['path']} ({prev['basis']}) -> {p} ({info.get('basis', '')})"
                    )
                else:
                    print(
                        f"[load_g_real] duplicate key {key}: keeping "
                        f"{prev['path']} ({prev['basis']}), skipping {p} ({info.get('basis', '')})"
                    )
                continue

            g_real_dict[key] = info["g_real"]
            key_meta[key] = {"path": p, "basis": info.get("basis", ""), "mtime": float(os.path.getmtime(p))}
            loaded += 1

        if loaded == 0:
            raise RuntimeError(
                f"Found {len(paths)} .h5 files but none had valid g(k,q) payload "
                "(dataset 'g_band' + attrs channel/label/axis)."
            )

        print(
            f"[load_g_real] loaded g(k,q) HDF5 unique_entries={loaded}, duplicates={duplicated}, skipped_non_gkq={skipped}, "
            f"nk_grid={nk_grid_ref}, nq_grid={nq_grid_ref}"
        )
        return {
            "g_real_dict": g_real_dict,
            "nk_grid": list(nk_grid_ref),
            "nq_grid": list(nq_grid_ref),
            # compute_gkq_full outputs are Hartree-level unless explicitly post-processed
            "units": "ha",
        }

    def get_HK(self, kvec, spin='up'):
        """
        Fourier transform H(R) -> H(k) using 4D tensor logic (Phase 0).
        Hk = sum_R H(R) * exp(i 2pi k.R)
        """
        s_idx = 0 if spin == 'up' else 1
        phases = np.exp(1j * 2 * np.pi * (self.R_list @ kvec))
        Hk = np.tensordot(phases, self.H_R_tensor[s_idx], axes=(0, 0))
        return Hk

    def get_gK(self, atom_name, axis, spin, R_p_idx, kvec):
        """
        Compute g(k, Rp) for a fixed perturbation-cell index Rp.

        Here R_p_idx is interpreted as the canonical index of the perturbation
        cell on the stored q-grid. The returned matrix is the electronic
        Fourier transform over Re only.
        """
        sample_key = (spin, atom_name, axis)
        q_dims = self._get_q_dims(sample_key)
        rp = self._normalize_rp_idx(R_p_idx, q_dims)
        g_Rp = self._get_gRp(spin, atom_name, axis, rp)
        nx, ny, nz = g_Rp.shape[:3]
        return _get_gK_core(g_Rp, kvec, nx, ny, nz)

    @staticmethod
    def _normalize_rp_idx(R_p_idx, q_dims):
        """
        Validate/normalize Rp index to be inside q-grid bounds.
        Allows negative Python-style indices and converts them to canonical [0, N-1].
        """
        if len(R_p_idx) != 3:
            raise ValueError(f"R_p_idx must have 3 integers, got: {R_p_idx}")
        norm = []
        for iax, (idx, n) in enumerate(zip(R_p_idx, q_dims)):
            if idx < -n or idx >= n:
                raise IndexError(
                    f"R_p_idx[{iax}]={idx} out of bounds for q-grid axis size {n}. "
                    f"Valid range is [0, {n-1}] or negative [-{n}, -1]. "
                    f"Note: do not pass supercell size itself as index."
                )
            norm.append(idx % n)
        return tuple(norm)

    def get_G_R(self, k_grid, z, spin='up', R=(0,0,0)):
        """Compute G(R, z) using cache if available."""
        if self.k_cache[spin] is not None and self.k_cache['grid'] == k_grid:
            # Phase 1: Use precomputed cache
            evals_mesh, evecs_mesh = self.k_cache[spin]
            Nk1, Nk2, Nk3 = k_grid
            # Compute only for this z
            Gk_mesh = _compute_Gk_mesh_njit(evals_mesh, evecs_mesh, z, Nk1, Nk2, Nk3)
            # Use single R Fourier transform
            R_tensor = np.array([R])
            GR_mesh = _sum_GR_bulk_njit(Gk_mesh, R_tensor, Nk1, Nk2, Nk3)
            return GR_mesh[0]
        else:
            # Fallback to slow implementation
            Nk1, Nk2, Nk3 = k_grid
            Nk = Nk1 * Nk2 * Nk3
            GR = np.zeros((self.dim, self.dim), dtype=complex)
            for k1 in range(Nk1):
                for k2 in range(Nk2):
                    for k3 in range(Nk3):
                        kvec = np.array([k1/Nk1, k2/Nk2, k3/Nk3])
                        Hk = self.get_HK(kvec, spin)
                        evals, evecs = np.linalg.eigh(Hk)
                        Gk = compute_Gk_njit(evals, evecs, z)
                        phase = np.exp(-1j * 2 * np.pi * np.dot(kvec, R))
                        GR += Gk * phase
            return GR / Nk

    def compute_J_bulk(self, idx_pairs, R_list, k_grid, energy_mesh):
        """
        Phase 1: High-performance bulk J calculation.
        idx_pairs: list of (idx_i, idx_j)
        R_list: list of (r1, r2, r3)
        Returns: results_dict {(idx_i, idx_j, R): value}
        """
        if self.formalism != "collinear":
            raise NotImplementedError(
                "Noncollinear J kernel is not implemented yet. "
                "Scaffold is ready (formalism flag + spinor hr input)."
            )
        # Ensure k_mesh is precomputed
        if self.k_cache['up'] is None or self.k_cache['grid'] != k_grid:
            self.precompute_k_mesh(k_grid)

        up_evals, up_evecs = self.k_cache['up']
        dn_evals, dn_evecs = self.k_cache['down']
        Nk1, Nk2, Nk3 = k_grid

        # Prepare Deltas and signs
        H0_up = self.H_R_tensor[0, self.idx_000]
        H0_dn = self.H_R_tensor[1, self.idx_000]
        Deltas = {}
        signs = {}
        for pair in idx_pairs:
            for atom_idx in pair:
                if atom_idx not in Deltas:
                    slc = self.orbital_slices[atom_idx]
                    D = H0_up[slc, slc] - H0_dn[slc, slc]
                    Deltas[atom_idx] = D
                    signs[atom_idx] = -1.0 if np.real(np.trace(D)) < 0 else 1.0

        if len(idx_pairs) != len(R_list):
            raise ValueError(f"idx_pairs and R_list must be 1:1 matched, got {len(idx_pairs)} and {len(R_list)}")

        total_J_acc = np.zeros((len(idx_pairs),), dtype=complex)

        for z, weight in energy_mesh:
            z_eff = z + self.efermi
            # 1. Compute Gk_mesh for both spins
            Gk_mesh_up = _compute_Gk_mesh_njit(up_evals, up_evecs, z_eff, Nk1, Nk2, Nk3)
            Gk_mesh_dn = _compute_Gk_mesh_njit(dn_evals, dn_evecs, z_eff, Nk1, Nk2, Nk3)

            # 2. Sum contributions bond-by-bond for matched (pair, R)
            for ip, ((idx_i, idx_j), R) in enumerate(zip(idx_pairs, R_list)):
                Delta_i = Deltas[idx_i]
                Delta_j = Deltas[idx_j]
                slc_i = self.orbital_slices[idx_i]
                slc_j = self.orbital_slices[idx_j]
                R_tensor = np.asarray([R], dtype=np.int64)
                Rm_tensor = -R_tensor
                GR_up = _sum_GR_bulk_njit(Gk_mesh_up, R_tensor, Nk1, Nk2, Nk3)[0]
                GR_dn_m = _sum_GR_bulk_njit(Gk_mesh_dn, Rm_tensor, Nk1, Nk2, Nk3)[0]
                Gij_up = GR_up[slc_i, slc_j]
                Gji_dn = GR_dn_m[slc_j, slc_i]
                total_J_acc[ip] += lkag_trace_njit(Delta_i, Gij_up, Delta_j, Gji_dn) * weight

        # Post-processing
        results = {}
        for ip, ((idx_i, idx_j), R) in enumerate(zip(idx_pairs, R_list)):
            rel_sign = signs[idx_i] * signs[idx_j]
            val = 1000.0 * np.imag(total_J_acc[ip]) / (4.0 * np.pi * rel_sign)
            results[(idx_i, idx_j, tuple(R))] = val
        return results

    def compute_J_R(self, idx_i, idx_j, R, k_grid, energy_mesh):
        """Baseline J calculation."""
        if self.formalism != "collinear":
            raise NotImplementedError("Noncollinear compute_J_R is not implemented yet.")
        # Potential at R=0
        H0_up = self.H_R_tensor[0, self.idx_000]
        H0_dn = self.H_R_tensor[1, self.idx_000]
        slc_i = self.orbital_slices[idx_i]
        slc_j = self.orbital_slices[idx_j]
        # Spin-splitting potential Delta = H_up - H_down
        Delta_i = H0_up[slc_i, slc_i] - H0_dn[slc_i, slc_i]
        Delta_j = H0_up[slc_j, slc_j] - H0_dn[slc_j, slc_j]

        def get_spin_sign(D):
            return -1.0 if np.real(np.trace(D)) < 0 else 1.0
        rel_sign = get_spin_sign(Delta_i) * get_spin_sign(Delta_j)

        total_J = 0.0 + 0.0j
        for z, weight in energy_mesh:
            GR_up = self.get_G_R(k_grid, z + self.efermi, spin='up', R=R)
            Rm = tuple(-x for x in R)
            GR_dn_m = self.get_G_R(k_grid, z + self.efermi, spin='down', R=Rm)
            Gij_up = GR_up[slc_i, slc_j]
            Gji_dn = GR_dn_m[slc_j, slc_i]
            total_J += lkag_trace_njit(Delta_i, Gij_up, Delta_j, Gji_dn) * weight
        return 1000.0 * np.imag(total_J) / (4.0 * np.pi * rel_sign)

    def compute_dJ_bulk(self, idx_pairs, R_list, atom_target, axis, R_p_idx, k_grid, energy_mesh):
        """Vectorized dJ_ij(R_ij) / du_{atom_target, axis, R_p_idx}."""
        if self.formalism != "collinear":
            raise NotImplementedError(
                "Noncollinear dJ/du kernel is not implemented yet. "
                "Scaffold is ready (formalism flag + spinor hr input)."
            )
        # Ensure cache for G calculation
        if self.k_cache['up'] is None or self.k_cache['grid'] != k_grid:
            self.precompute_k_mesh(k_grid)

        up_evals, up_evecs = self.k_cache['up']
        dn_evals, dn_evecs = self.k_cache['down']
        Nk1, Nk2, Nk3 = k_grid

        # Prepare Deltas (R=0)
        H0_up = self.H_R_tensor[0, self.idx_000]
        H0_dn = self.H_R_tensor[1, self.idx_000]
        Deltas = {}
        signs = {}
        for pair in idx_pairs:
            for atom_idx in pair:
                if atom_idx not in Deltas:
                    slc = self.orbital_slices[atom_idx]
                    D = H0_up[slc, slc] - H0_dn[slc, slc]
                    Deltas[atom_idx] = D
                    signs[atom_idx] = -1.0 if np.real(np.trace(D)) < 0 else 1.0

        # Vectorize gK mesh
        # We need a mesh of k-vectors for the EPC grid FT
        kx = np.linspace(0, 1, Nk1, endpoint=False)
        ky = np.linspace(0, 1, Nk2, endpoint=False)
        kz = np.linspace(0, 1, Nk3, endpoint=False)
        kvecs = np.stack(np.meshgrid(kx, ky, kz, indexing='ij'), axis=-1)

        gK_up = np.zeros((Nk1, Nk2, Nk3, self.dim, self.dim), dtype=complex)
        gK_dn = np.zeros((Nk1, Nk2, Nk3, self.dim, self.dim), dtype=complex)

        q_dims = self._get_q_dims(('up', atom_target, axis))
        rp = self._normalize_rp_idx(R_p_idx, q_dims)
        g_up_Rp = self._get_gRp('up', atom_target, axis, rp)
        g_dn_Rp = self._get_gRp('down', atom_target, axis, rp)

        # FT for derivatives (optimized)
        gK_up = _compute_gK_mesh_njit(g_up_Rp, Nk1, Nk2, Nk3)
        gK_dn = _compute_gK_mesh_njit(g_dn_Rp, Nk1, Nk2, Nk3)

        if len(idx_pairs) != len(R_list):
            raise ValueError(f"idx_pairs and R_list must be 1:1 matched, got {len(idx_pairs)} and {len(R_list)}")
        dJ_R_acc = np.zeros((len(idx_pairs),), dtype=complex)

        for z, weight in energy_mesh:
            G_u = _compute_Gk_mesh_njit(up_evals, up_evecs, z + self.efermi, Nk1, Nk2, Nk3)
            G_d = _compute_Gk_mesh_njit(dn_evals, dn_evecs, z + self.efermi, Nk1, Nk2, Nk3)
            dG_u = _compute_dGk_mesh_njit(G_u, gK_up, Nk1, Nk2, Nk3)
            dG_d = _compute_dGk_mesh_njit(G_d, gK_dn, Nk1, Nk2, Nk3)

            # Sum over matched bonds
            for ip, ((idx_i, idx_j), R) in enumerate(zip(idx_pairs, R_list)):
                slc_i = self.orbital_slices[idx_i]
                slc_j = self.orbital_slices[idx_j]
                ni = slc_i.stop - slc_i.start
                nj = slc_j.stop - slc_j.start

                Di = Deltas[idx_i]
                Dj = Deltas[idx_j]

                # Onsite local-block dDelta at Re=(0,0,0) for the selected Rp.
                dDi_const = _onsite_block_dDelta(
                    g_up_Rp[0, 0, 0, slc_i, slc_i],
                    g_dn_Rp[0, 0, 0, slc_i, slc_i],
                )
                dDj_const = _onsite_block_dDelta(
                    g_up_Rp[0, 0, 0, slc_j, slc_j],
                    g_dn_Rp[0, 0, 0, slc_j, slc_j],
                )

                R_tensor = np.asarray([R], dtype=np.int64)
                Rm_tensor = -R_tensor
                GR_u = _sum_GR_bulk_njit(G_u[:, :, :, slc_i, slc_j], R_tensor, Nk1, Nk2, Nk3)[0]
                dGR_u = _sum_GR_bulk_njit(dG_u[:, :, :, slc_i, slc_j], R_tensor, Nk1, Nk2, Nk3)[0]
                GR_d_m = _sum_GR_bulk_njit(G_d[:, :, :, slc_j, slc_i], Rm_tensor, Nk1, Nk2, Nk3)[0]
                dGR_d_m = _sum_GR_bulk_njit(dG_d[:, :, :, slc_j, slc_i], Rm_tensor, Nk1, Nk2, Nk3)[0]

                dJ_R_acc[ip] += lkag_deriv_trace_njit(
                    Di, dDi_const, GR_u, dGR_u, Dj, dDj_const, GR_d_m, dGR_d_m
                ) * weight

        # Return meV units
        results = {}
        for ip, ((idx_i, idx_j), R) in enumerate(zip(idx_pairs, R_list)):
            rel_sign = signs[idx_i] * signs[idx_j]
            results[(idx_i, idx_j, tuple(R))] = 1000.0 * np.imag(dJ_R_acc[ip]) / (4.0 * np.pi * rel_sign)
        return results

    def compute_dJ_batch(
        self,
        idx_pairs,
        R_list,
        atom_targets,
        axes=['x', 'y', 'z'],
        R_p_idx=(0,0,0),
        k_grid=(1,1,1),
        energy_mesh=None,
        ddelta_mode="re0",
        debug=False,
        debug_every=10,
        debug_terms=False,
    ):
        """
        Batch version of dJ/du calculation.
        Reuses Gu, Gd across all axes and atoms to maximize performance.

        ddelta_mode:
            re0 : dDelta from the real-space onsite local block at Re=(0,0,0) for selected Rp
            k0  : dDelta from the k-space Gamma onsite local block g(k=0, Rp)
            off : disable onsite dDelta terms

        Returns:
            dJ_dict: {(atom, axis): {(i, j, R): val}, ...}
        """
        if self.formalism != "collinear":
            raise NotImplementedError(
                "Noncollinear batch dJ/du kernel is not implemented yet. "
                "Scaffold is ready (formalism flag + spinor hr input)."
            )
        if energy_mesh is None:
            energy_mesh = get_semicircle_contour(emin=-25.0, emax=0.0, npoints=100)

        Nk1, Nk2, Nk3 = k_grid
        t_all0 = time.perf_counter()
        self.last_debug_trace = None
        if debug:
            debug_trace = {
                "k_grid": [int(Nk1), int(Nk2), int(Nk3)],
                "n_energy": len(energy_mesh),
                "n_pairs": len(idx_pairs),
                "n_R": len(R_list),
                "n_targets": len(atom_targets),
                "axes": list(axes),
                "target_ctx": [],
                "energy_samples": [],
            }
        else:
            debug_trace = None

        # 1. Prepare eigensystems
        if self.k_cache['up'] is None or self.k_cache['grid'] != k_grid:
            self.precompute_k_mesh(k_grid)
        up_evals, up_evecs = self.k_cache['up']
        dn_evals, dn_evecs = self.k_cache['down']

        # 2. Prepare Deltas and signs (R=0)
        Deltas = {}
        signs = {}
        H0_up = self.H_R_tensor[0, self.idx_000]
        H0_dn = self.H_R_tensor[1, self.idx_000]
        for idx in set([p[0] for p in idx_pairs] + [p[1] for p in idx_pairs]):
            slc = self.orbital_slices[idx]
            D = H0_up[slc, slc] - H0_dn[slc, slc]
            Deltas[idx] = D
            signs[idx] = -1.0 if np.real(np.trace(D)) < 0 else 1.0

        # 3. Initialize results
        dJ_batch_acc = {}
        dJ_terms_acc = {}
        for atom in atom_targets:
            for ax in axes:
                dJ_batch_acc[(atom, ax)] = np.zeros((len(idx_pairs),), dtype=complex)
                if debug_terms:
                    dJ_terms_acc[(atom, ax)] = np.zeros((4, len(idx_pairs)), dtype=np.complex128)

        # Precompute pair metadata once
        if len(idx_pairs) != len(R_list):
            raise ValueError(f"idx_pairs and R_list must be 1:1 matched, got {len(idx_pairs)} and {len(R_list)}")

        pair_meta = []
        for (idx_i, idx_j), R in zip(idx_pairs, R_list):
            slc_i = self.orbital_slices[idx_i]
            slc_j = self.orbital_slices[idx_j]
            pair_meta.append((idx_i, idx_j, tuple(int(x) for x in R), slc_i, slc_j, Deltas[idx_i], Deltas[idx_j]))

        # Precompute target-axis data once (major speedup)
        # gK is independent of contour-energy z.
        target_ctx = {}
        sample_key = ('up', atom_targets[0], axes[0])
        if self.g_provider is None and sample_key not in self.g_real_dict:
            raise KeyError(f"Missing g_real entry for {sample_key}")
        # Rp convention: extract_v2 stores Rp = R_orbit - R_displaced.
        # If we move an atom at R_p_idx, then for an orbital at R_cell, the relative index is (R_cell - R_p_idx).
        # For the bond origin (cell 0), the index is -R_p_idx.
        # For the bond target (cell R), the index is R - R_p_idx.
        rp_origin_eff = -np.asarray(R_p_idx)
        rp = self._normalize_rp_idx(rp_origin_eff, self._get_q_dims(sample_key))

        ddelta_mode = str(ddelta_mode).strip().lower()
        if ddelta_mode not in {"re0", "k0", "off"}:
            raise ValueError(f"Unsupported ddelta_mode: {ddelta_mode}. Use one of: re0, k0, off.")
        for atom in atom_targets:
            for ax in axes:
                g_up_Rp = self._get_gRp('up', atom, ax, rp)
                g_dn_Rp = self._get_gRp('down', atom, ax, rp)

                gK_up = _compute_gK_mesh_njit(g_up_Rp, Nk1, Nk2, Nk3)
                gK_dn = _compute_gK_mesh_njit(g_dn_Rp, Nk1, Nk2, Nk3)

                # dDelta onsite terms for each pair (also z-independent).
                dD_terms = []
                for _, _, R, slc_i, slc_j, _, _ in pair_meta:
                    ni = slc_i.stop - slc_i.start
                    nj = slc_j.stop - slc_j.start
                    if ddelta_mode == "off":
                        dDi_const = np.zeros((ni, ni), dtype=complex)
                        dDj_const = np.zeros((nj, nj), dtype=complex)
                    else:
                        nq0, nq1, nq2 = self._get_q_dims(('up', atom, ax))
                        # Target site is at cell R. Relevant Rp = R - R_p_idx = R + rp_origin_eff
                        rp_i = ((rp_origin_eff[0] + R[0]) % nq0, (rp_origin_eff[1] + R[1]) % nq1, (rp_origin_eff[2] + R[2]) % nq2)
                        if ddelta_mode == "k0":
                            g_up_Rp_i_sum = np.sum(self._get_gRp('up', atom, ax, rp_i), axis=(0, 1, 2))
                            g_dn_Rp_i_sum = np.sum(self._get_gRp('down', atom, ax, rp_i), axis=(0, 1, 2))
                            dDi_const = _onsite_block_dDelta(
                                g_up_Rp_i_sum[slc_i, slc_i],
                                g_dn_Rp_i_sum[slc_i, slc_i],
                            )
                            dDj_const = _onsite_block_dDelta(
                                gK_up[0, 0, 0, slc_j, slc_j],
                                gK_dn[0, 0, 0, slc_j, slc_j],
                            )
                        else:  # re0
                            g_up_Rp_i = self._get_gRp('up', atom, ax, rp_i)
                            g_dn_Rp_i = self._get_gRp('down', atom, ax, rp_i)
                            dDi_const = _onsite_block_dDelta(
                                g_up_Rp_i[0, 0, 0, slc_i, slc_i],
                                g_dn_Rp_i[0, 0, 0, slc_i, slc_i],
                            )
                            dDj_const = _onsite_block_dDelta(
                                g_up_Rp[0, 0, 0, slc_j, slc_j],
                                g_dn_Rp[0, 0, 0, slc_j, slc_j],
                            )
                        if dDi_const.shape != (ni, ni) or dDj_const.shape != (nj, nj):
                            # Safety fallback (should not happen with consistent slicing)
                            dDi_const = np.zeros((ni, ni), dtype=complex)
                            dDj_const = np.zeros((nj, nj), dtype=complex)
                    dD_terms.append((dDi_const, dDj_const))

                target_ctx[(atom, ax)] = (gK_up, gK_dn, dD_terms)
                if debug:
                    debug_trace["target_ctx"].append({
                        "atom": atom,
                        "axis": ax,
                        "g_up_shape": list(g_up_Rp.shape),
                        "g_dn_shape": list(g_dn_Rp.shape),
                        "gK_up_maxabs": float(np.max(np.abs(gK_up))),
                        "gK_dn_maxabs": float(np.max(np.abs(gK_dn))),
                        "gK_up_has_nan": bool(np.isnan(gK_up).any()),
                        "gK_dn_has_nan": bool(np.isnan(gK_dn).any()),
                        "ddelta_mode": ddelta_mode,
                    })

        # 4. Energy loop (Outer)
        print(f"Starting batch dJ/du calculation for {len(atom_targets)} atoms and {len(axes)} axes...")
        for iz, (z, weight) in enumerate(energy_mesh):
            if iz % 10 == 0:
                print(f"  Energy point {iz+1}/{len(energy_mesh)}...")
            t_e0 = time.perf_counter()

            # Compute Gu, Gd ONCE per energy point
            G_u = _compute_Gk_mesh_njit(up_evals, up_evecs, z + self.efermi, Nk1, Nk2, Nk3)
            G_d = _compute_Gk_mesh_njit(dn_evals, dn_evecs, z + self.efermi, Nk1, Nk2, Nk3)
            if debug and (iz % max(1, debug_every) == 0 or iz == len(energy_mesh) - 1):
                e_sample = {
                    "iz": int(iz),
                    "z_real": float(np.real(z)),
                    "z_imag": float(np.imag(z)),
                    "weight_abs": float(np.abs(weight)),
                    "Gu_maxabs": float(np.max(np.abs(G_u))),
                    "Gd_maxabs": float(np.max(np.abs(G_d))),
                    "Gu_has_nan": bool(np.isnan(G_u).any()),
                    "Gd_has_nan": bool(np.isnan(G_d).any()),
                }
            else:
                e_sample = None

            # Loop over targets
            for atom in atom_targets:
                for ax in axes:
                    gK_up, gK_dn, dD_terms = target_ctx[(atom, ax)]

                    # dG
                    dG_u = _compute_dGk_mesh_njit(G_u, gK_up, Nk1, Nk2, Nk3)
                    dG_d = _compute_dGk_mesh_njit(G_d, gK_dn, Nk1, Nk2, Nk3)
                    if e_sample is not None and atom == atom_targets[0] and ax == axes[0]:
                        e_sample["dGu_sample_maxabs"] = float(np.max(np.abs(dG_u)))
                        e_sample["dGd_sample_maxabs"] = float(np.max(np.abs(dG_d)))
                        e_sample["dGu_has_nan"] = bool(np.isnan(dG_u).any())
                        e_sample["dGd_has_nan"] = bool(np.isnan(dG_d).any())

                    # Bond summation
                    acc = dJ_batch_acc[(atom, ax)]
                    for ip, (_, _, R, slc_i, slc_j, Di, Dj) in enumerate(pair_meta):
                        dDi_const, dDj_const = dD_terms[ip]
                        R_tensor = np.asarray([R], dtype=np.int64)
                        Rm_tensor = -R_tensor
                        GR_u = _sum_GR_bulk_njit(G_u[:, :, :, slc_i, slc_j], R_tensor, Nk1, Nk2, Nk3)[0]
                        dGR_u = _sum_GR_bulk_njit(dG_u[:, :, :, slc_i, slc_j], R_tensor, Nk1, Nk2, Nk3)[0]
                        GR_d_m = _sum_GR_bulk_njit(G_d[:, :, :, slc_j, slc_i], Rm_tensor, Nk1, Nk2, Nk3)[0]
                        dGR_d_m = _sum_GR_bulk_njit(dG_d[:, :, :, slc_j, slc_i], Rm_tensor, Nk1, Nk2, Nk3)[0]
                        if debug_terms:
                            t1, t2, t3, t4 = lkag_deriv_terms_njit(
                                Di, dDi_const, GR_u, dGR_u, Dj, dDj_const, GR_d_m, dGR_d_m
                            )
                            dJ_terms_acc[(atom, ax)][0, ip] += t1 * weight
                            dJ_terms_acc[(atom, ax)][1, ip] += t2 * weight
                            dJ_terms_acc[(atom, ax)][2, ip] += t3 * weight
                            dJ_terms_acc[(atom, ax)][3, ip] += t4 * weight
                            acc[ip] += (t1 + t2 + t3 + t4) * weight
                        else:
                            acc[ip] += lkag_deriv_trace_njit(
                                Di, dDi_const, GR_u, dGR_u, Dj, dDj_const, GR_d_m, dGR_d_m
                            ) * weight
            if e_sample is not None:
                e_sample["elapsed_s"] = float(time.perf_counter() - t_e0)
                debug_trace["energy_samples"].append(e_sample)

        # 5. Format output
        final_results = {}
        for (atom, ax), matrix in dJ_batch_acc.items():
            atom_ax_res = {}
            for ip, ((idx_i, idx_j), R) in enumerate(zip(idx_pairs, R_list)):
                rel_sign = signs[idx_i] * signs[idx_j]
                atom_ax_res[(idx_i, idx_j, tuple(R))] = 1000.0 * np.imag(matrix[ip]) / (4.0 * np.pi * rel_sign)
            final_results[(atom, ax)] = atom_ax_res
        if debug:
            debug_trace["total_elapsed_s"] = float(time.perf_counter() - t_all0)
            debug_trace["rp_idx_normalized"] = [int(rp[0]), int(rp[1]), int(rp[2])]
            debug_trace["ddelta_mode"] = ddelta_mode
            if debug_terms:
                term_payload = {}
                for (atom, ax), arr in dJ_terms_acc.items():
                    key = f"{atom}:{ax}"
                    rel = np.zeros_like(arr.real)
                    for ip, (idx_pair, R) in enumerate(zip(idx_pairs, R_list)):
                        rel_sign = signs[idx_pair[0]] * signs[idx_pair[1]]
                        rel[:, ip] = 1000.0 * np.imag(arr[:, ip]) / (4.0 * np.pi * rel_sign)
                    term_payload[key] = {
                        "terms_meV_per_A": rel.tolist(),
                    }
                debug_trace["term_breakdown"] = {
                    "term_order": [
                        "term1=dD_i G_ij^up D_j G_ji^dn",
                        "term2=D_i dG_ij^up D_j G_ji^dn",
                        "term3=D_i G_ij^up dD_j G_ji^dn",
                        "term4=D_i G_ij^up D_j dG_ji^dn",
                    ],
                    "pair_keys": [
                        [int(i), int(j), [int(R[0]), int(R[1]), int(R[2])]]
                        for (i, j), R in zip(idx_pairs, R_list)
                    ],
                    "by_target_axis": term_payload,
                }
            self.last_debug_trace = debug_trace
        return final_results

def get_semicircle_contour(emin=-15.0, emax=0.0, npoints=40):
    radius = (emax - emin) / 2.0
    center = (emax + emin) / 2.0
    theta = np.linspace(np.pi, 0, npoints)
    z_list = center + radius * np.exp(1j * theta)
    dtheta = -np.pi / (npoints - 1)
    dz_list = 1j * radius * np.exp(1j * theta) * dtheta
    return list(zip(z_list, dz_list))


def get_cfr_pole_mesh(npoles=40, beta_eV_inv=400.0):
    """
    Experimental CFR-like pole mesh using Matsubara poles.

    Notes
    -----
    - This is a lightweight pole-sum scaffold, not a full Ozaki/CFR implementation.
    - Kept optional via CLI integrator switch; default contour path remains unchanged.
    - z is relative to E_F (same convention as get_semicircle_contour usage).
    """
    n = int(max(1, npoles))
    beta = float(beta_eV_inv)
    # Fermionic Matsubara poles: i*(2m-1)pi/beta
    m = np.arange(1, n + 1, dtype=np.float64)
    omega = (2.0 * m - 1.0) * np.pi / beta
    z_list = 1j * omega
    # Residue factor for Fermi poles in contour sum (experimental scaffold).
    w = (-2.0j * np.pi) / beta
    dz_list = np.full(n, w, dtype=np.complex128)
    return list(zip(z_list.astype(np.complex128), dz_list))


def get_cfr_ozaki_mesh(npoles=40, beta_eV_inv=400.0):
    """
    Ozaki/TB2J-style continued-fraction pole mesh (experimental).

    This follows the same pole/residue construction used in TB2J `mycfr.py`:
      1) build tridiagonal continued-fraction matrix with
         b_j = 1 / [2 * sqrt((2j-1)(2j+1))], j=1..nz-1
      2) diagonalize to obtain poles p_n
      3) residues r_n = 0.25 * |v_{0n}|^2 / p_n^2
      4) map to energy poles z_n = i / (beta * p_n), with weights
         w_n = 2 i r_n / beta, including ±z_n pairs.

    Notes
    -----
    - Returned shape is compatible with existing energy_mesh API:
      list of (z, weight) complex pairs.
    - Kept experimental to allow side-by-side validation against contour.
    """
    nz = int(max(1, npoles))
    beta = float(beta_eV_inv)
    if beta <= 0.0:
        raise ValueError(f"beta_eV_inv must be > 0, got {beta_eV_inv}")
    if nz == 1:
        # Fallback to minimal Matsubara-like single pole.
        z = 1j * np.pi / beta
        w = (-2.0j * np.pi) / beta
        return [(np.complex128(z), np.complex128(w))]

    j = np.arange(1, nz, dtype=np.float64)
    b = 1.0 / (2.0 * np.sqrt((2.0 * j - 1.0) * (2.0 * j + 1.0)))
    b_mat = np.diag(b, 1) + np.diag(b, -1)

    poles, eigvecs = np.linalg.eigh(b_mat)
    eps = 1e-14
    mask = poles > eps
    poles = poles[mask]
    if poles.size == 0:
        z = 1j * np.pi / beta
        w = (-2.0j * np.pi) / beta
        return [(np.complex128(z), np.complex128(w))]

    residues = 0.25 * (np.abs(eigvecs[0, mask]) ** 2) / (poles ** 2)

    path = []
    weights = []
    for p, r in zip(poles, residues):
        z = 1j / (beta * p)
        w = (2.0j / beta) * r
        path.append(np.complex128(z))
        weights.append(np.complex128(w))
        path.append(np.complex128(-z))
        weights.append(np.complex128(w))

    # Deterministic ordering helps reproducibility/debugging.
    order = np.argsort(np.imag(np.array(path)))
    path = np.array(path, dtype=np.complex128)[order]
    weights = np.array(weights, dtype=np.complex128)[order]
    return list(zip(path, weights))
