# Legacy exchange implementation; use the native engine for new workflows.
"""Build a full spinor Wannier90 hr.dat with model SOC from collinear up/down hr.dat.

Two SOC modes are supported:

  wannier (default)
    Adds the onsite atomic lambda*L.S block directly in the Wannier orbital
    index space.  Basis convention is spin-major (all-up, all-down).
    Off-diagonal (pd cross-species) blocks receive NO contribution.

  rotated
    Rotates the atomic SOC projector into the Wannier basis via the Wannier90
    U(k) matrices:

        H_SOC_W(k) = U†(k) @ H_SOC_atomic @ U(k)       [udag_soc_u]
                 or  U(k)  @ H_SOC_atomic @ U†(k)       [u_soc_udag]

    Then the k-space correction is Fourier-transformed back to R-space and
    added to the collinear base.  This mode CAN populate the off-diagonal
    (pd) blocks if the Wannier U(k) mixes Cr-d and I-p character.

Inputs
------
  --up_hr / --dn_hr       Collinear spin-up / spin-down Wannier90 hr.dat
  --win                   Wannier90 .win file (for orbital-group parsing and --symmetrize structure)
  --soc                   SOC specs: element:orbital:lambda_eV  e.g. "I:p:0.6;Cr:d:0.05"
  --soc_gauge             wannier | rotated
  --u_mat                 Common U(k) file (*_u.mat / .npy / .npz)  [rotated mode]
  --u_up_mat / --u_dn_mat Spin-resolved U(k) files                  [rotated mode]
  --soc_rotation          udag_soc_u (default) | u_soc_udag
  --kmesh N1 N2 N3        Regular k-mesh for rotated FFT (must match U.mat mesh)
  --spin_direction x y z  Quantisation axis (default: 0 0 1)
  --drop_tol              Threshold below which HR blocks are discarded (default 1e-12)
  --hermitianize          Symmetrise H(R) = 0.5*(H(R)+H†(R)) after assembly
  --dump_txt              Path for human-readable block-norm diagnostic report
  -o / --output           Output spinor hr.dat path

Basis ordering (--basis_order)
------------------------------
  spin_major (default)
    Output basis: [w1↑, w2↑, ..., wN↑, w1↓, w2↓, ..., wN↓]
    This is the standard convention for two concatenated collinear calculations.

  win_interleaved
    Output basis: [w1↑, w1↓, w2↑, w2↓, ..., wN↑, wN↓]  grouped by site/projection.
    Matches the spinor hr.dat produced by Wannier90 non-collinear runs where the
    up/down components of each projection are adjacent.
    Requires --win to be provided so the projection groups can be read.

  group_interleaved  (alias: site_interleaved, atom_interleaved)
    Same as win_interleaved.  Groups are defined by the projections block in .win,
    so all orbitals of the same atom/orbital-type appear consecutively with their
    spin partner before moving to the next group.
"""

from __future__ import annotations

import argparse
import os
import re

import numpy as np
import spglib

from slw.core.constants import BOHR_TO_ANG
from slw.core.wannier_io import read_wannier_hr, write_wannier_hr
from slw.core.structure import ELEMENT_TO_Z



# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WANNIER90_P_ORDER = "pz,px,py"
WANNIER90_D_ORDER = "dz2,dxz,dyz,dx2-y2,dxy"


# ---------------------------------------------------------------------------
# Wannier90 .win parser  (standalone, no exchange module dependency)
# ---------------------------------------------------------------------------

def _clean_species(label: str) -> str:
    return re.sub(r"\d+$", "", str(label).strip()).lower()


def _projection_orbitals(text: str) -> list[str]:
    clean = re.sub(r"\b(l|ang|angular)_?mom(entum)?\s*=\s*0\b", "s", text, flags=re.I)
    clean = re.sub(r"\b(l|ang|angular)_?mom(entum)?\s*=\s*1\b", "p", clean, flags=re.I)
    clean = re.sub(r"\b(l|ang|angular)_?mom(entum)?\s*=\s*2\b", "d", clean, flags=re.I)
    clean = re.sub(r"\b(l|ang|angular)_?mom(entum)?\s*=\s*3\b", "f", clean, flags=re.I)
    fields = [x.strip().lower() for x in re.split(r"[,; \t]+", clean) if x.strip()]
    return [f.strip("{}()") for f in fields if f.strip("{}()") in {"s", "p", "d", "f"}]


def _orbital_count(kind: str) -> int:
    return {"s": 1, "p": 3, "d": 5, "f": 7}[kind]


def _win_projection_groups(win_path: str):
    """Parse atoms + projections from a Wannier90 .win file.

    Returns
    -------
    atoms  : list of str  atom labels in order
    groups : list of dict  with keys: atom_index, atom_label, element, orbital, indices
    nproj  : int  total number of projections (= num_wann for no disentanglement)
    """
    atoms: list[str] = []
    projections: list[tuple[str, list[str]]] = []
    block = None
    with open(win_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.split("!", 1)[0].split("#", 1)[0].strip()
            low = line.lower()
            if not line:
                continue
            if low.startswith("begin atoms_frac") or low.startswith("begin atoms_cart"):
                block = "atoms"
                continue
            if low.startswith("begin projections"):
                block = "projections"
                continue
            if low.startswith("end "):
                block = None
                continue
            if block == "atoms":
                toks = line.split()
                if len(toks) >= 4:
                    atoms.append(toks[0])
            elif block == "projections":
                lhs, rhs = (line.split(":", 1) + [line])[:2] if ":" in line else (line, line)
                orbs = _projection_orbitals(rhs)
                if orbs:
                    projections.append((lhs.strip(), orbs))

    if not atoms:
        raise ValueError(f"No atoms_frac/atoms_cart block found in {win_path}")
    if not projections:
        raise ValueError(f"No projections block found in {win_path}")

    groups = []
    offset = 0
    used = [False] * len(atoms)
    for label, orbitals in projections:
        label_clean = _clean_species(label)
        exact = [i for i, a in enumerate(atoms) if a.lower() == label.lower()]
        species = [i for i, a in enumerate(atoms) if _clean_species(a) == label_clean]
        matches = exact if exact else species
        if not matches:
            raise ValueError(f"Projection label '{label}' did not match any atom in {win_path}")
        for ia in matches:
            if used[ia]:
                raise ValueError(
                    f"Atom '{atoms[ia]}' matched by multiple projection lines in {win_path}"
                )
            used[ia] = True
            for orb in orbitals:
                n = _orbital_count(orb)
                groups.append({
                    "atom_index": ia,
                    "atom_label": atoms[ia],
                    "element": _clean_species(atoms[ia]),
                    "orbital": orb,
                    "indices": list(range(offset, offset + n)),
                })
                offset += n
    return atoms, groups, offset


def _parse_soc_specs(text: str) -> list[tuple[str, str, float]]:
    """Parse 'I:p:0.6;Cr:d:0.05' -> [(elem, orb, lam), ...]."""
    specs: list[tuple[str, str, float]] = []
    if not text:
        return specs
    for item in re.split(r"[;]+", str(text)):
        item = item.strip()
        if not item:
            continue
        parts = [x.strip() for x in re.split(r"[:,=\s\t]+", item) if x.strip()]
        if len(parts) != 3:
            raise ValueError(f"SOC spec must be element:orbital:lambda_eV, got {item!r}")
        elem, orb, lam = parts
        orb = orb.lower()
        if orb not in {"p", "d"}:
            raise ValueError(f"Unsupported SOC orbital {orb!r}; use p or d")
        specs.append((_clean_species(elem), orb, float(lam)))
    return specs


def _selected_groups(win_path: str, specs: list[tuple[str, str, float]], nwan: int):
    """Return (entries, groups_per_spec) where entries carry all needed info."""
    _, all_groups, nproj = _win_projection_groups(win_path)
    if int(nproj) != int(nwan):
        raise ValueError(
            f"{win_path} projection count={nproj} but Hamiltonian nwan={nwan}"
        )
    entries = []
    for elem, orb, lam in specs:
        matched = [g for g in all_groups if g["element"] == elem and g["orbital"] == orb]
        if not matched:
            known = sorted({(g["element"], g["orbital"]) for g in all_groups})
            raise ValueError(f"No {elem}:{orb} projection in {win_path}; known={known}")
        entries.append({
            "element": elem,
            "orbital": orb,
            "lambda_ev": lam,
            "groups": matched,
        })
    return entries


# ---------------------------------------------------------------------------
# Atomic L·S blocks
# ---------------------------------------------------------------------------

def _p_orbital_l_matrices(order: str = WANNIER90_P_ORDER):
    base = ["px", "py", "pz"]
    order_list = [x.strip().lower() for x in str(order).replace(";", ",").split(",") if x.strip()]
    if not order_list:
        order_list = base
    if sorted(order_list) != sorted(base):
        raise ValueError(f"Unsupported p order {order_list}; expected permutation of {base}")
    Lx0 = np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]], dtype=np.complex128)
    Ly0 = np.array([[0, 0, 1j], [0, 0, 0], [-1j, 0, 0]], dtype=np.complex128)
    Lz0 = np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]], dtype=np.complex128)
    perm = [base.index(x) for x in order_list]
    return tuple(mat[np.ix_(perm, perm)] for mat in (Lx0, Ly0, Lz0))


def _d_orbital_l_matrices(order: str = WANNIER90_D_ORDER):
    base = ["dz2", "dxz", "dyz", "dx2-y2", "dxy"]
    aliases = {
        "dz^2": "dz2", "d_z2": "dz2", "d_z^2": "dz2",
        "dx2y2": "dx2-y2", "dx^2-y^2": "dx2-y2", "d_x2-y2": "dx2-y2",
    }
    order_list = [aliases.get(x.strip().lower(), x.strip().lower())
                  for x in str(order).replace(";", ",").split(",") if x.strip()]
    if not order_list:
        order_list = base
    if sorted(order_list) != sorted(base):
        raise ValueError(f"Unsupported d order {order_list}; expected permutation of {base}")
    rt3 = np.sqrt(3.0)
    Lx0 = np.array([
        [0,        0,        1j*rt3,  0,   0  ],
        [0,        0,        0,       0,   1j ],
        [-1j*rt3,  0,        0,      -1j,  0  ],
        [0,        0,        1j,      0,   0  ],
        [0,       -1j,       0,       0,   0  ],
    ], dtype=np.complex128)
    Ly0 = np.array([
        [0,        -1j*rt3,  0,   0,   0  ],
        [1j*rt3,   0,        0,   1j,  0  ],
        [0,        0,        0,   0,   1j ],
        [0,       -1j,       0,   0,   0  ],
        [0,        0,       -1j,  0,   0  ],
    ], dtype=np.complex128)
    Lz0 = np.array([
        [0,   0,   0,    0,    0   ],
        [0,   0,  -1j,   0,    0   ],
        [0,   1j,  0,    0,    0   ],
        [0,   0,   0,    0,   -2j  ],
        [0,   0,   0,    2j,   0   ],
    ], dtype=np.complex128)
    perm = [base.index(x) for x in order_list]
    return tuple(mat[np.ix_(perm, perm)] for mat in (Lx0, Ly0, Lz0))


def _atomic_p_soc_block(lam: float, order: str = WANNIER90_P_ORDER) -> np.ndarray:
    """6x6 lambda*L·S in spin-major p basis."""
    Lx, Ly, Lz = _p_orbital_l_matrices(order)
    block = np.zeros((6, 6), dtype=np.complex128)
    block[:3, :3] = 0.5 * lam * Lz
    block[:3, 3:] = 0.5 * lam * (Lx - 1j * Ly)
    block[3:, :3] = 0.5 * lam * (Lx + 1j * Ly)
    block[3:, 3:] = -0.5 * lam * Lz
    return block


def _atomic_d_soc_block(lam: float, order: str = WANNIER90_D_ORDER) -> np.ndarray:
    """10x10 lambda*L·S in spin-major d basis."""
    Lx, Ly, Lz = _d_orbital_l_matrices(order)
    block = np.zeros((10, 10), dtype=np.complex128)
    block[:5, :5] = 0.5 * lam * Lz
    block[:5, 5:] = 0.5 * lam * (Lx - 1j * Ly)
    block[5:, :5] = 0.5 * lam * (Lx + 1j * Ly)
    block[5:, 5:] = -0.5 * lam * Lz
    return block


def _build_atomic_soc_projector(entries, nwan: int, p_order: str, d_order: str) -> np.ndarray:
    """Build the full (2*nwan, 2*nwan) atomic SOC matrix in spin-major Wannier space."""
    mat = np.zeros((2 * nwan, 2 * nwan), dtype=np.complex128)
    for entry in entries:
        orb = entry["orbital"]
        lam = float(entry["lambda_ev"])
        if orb == "p":
            block = _atomic_p_soc_block(lam, order=p_order)
        else:
            block = _atomic_d_soc_block(lam, order=d_order)
        norb = block.shape[0] // 2
        for group in entry["groups"]:
            idx0 = np.asarray(group["indices"], dtype=np.int64).reshape(norb)
            if np.any(idx0 >= nwan):
                raise ValueError(
                    f"Orbital index {idx0.tolist()} out of range for nwan={nwan}"
                )
            idx = np.concatenate([idx0, idx0 + nwan])
            mat[idx[:, None], idx[None, :]] += block
    return mat


# ---------------------------------------------------------------------------
# HR helpers
# ---------------------------------------------------------------------------

def _read_wannier_hr_compat(path: str):
    parsed = read_wannier_hr(path)
    # Support both 3-tuple and legacy 4-tuple returns
    if len(parsed) == 4:
        dim, _nrpts, degens, hmap = parsed
    elif len(parsed) == 3:
        dim, degens, hmap = parsed
    else:
        raise ValueError(f"Unexpected read_wannier_hr return length={len(parsed)} for {path}")
    dim = int(dim)
    if dim <= 0 or not hmap:
        raise ValueError(f"Failed to read non-empty Wannier90 hr.dat: {path}")
    return dim, list(degens), hmap


def _hmap_to_hk(hmap: dict, kpts: np.ndarray, dim: int) -> np.ndarray:
    """Compute H(k) from an HR dict via Fourier sum.

    H(k) = sum_R H(R) exp(+2pi i k.R),  then Hermitianize.
    """
    rvec = np.asarray(sorted(hmap), dtype=np.float64)          # (nR, 3)
    h_r = np.asarray(
        [hmap[tuple(int(x) for x in r)] for r in rvec], dtype=np.complex128
    )                                                            # (nR, dim, dim)
    for ir, r in enumerate(rvec):
        if h_r[ir].shape != (dim, dim):
            raise ValueError(
                f"R={tuple(r.astype(int))} block shape={h_r[ir].shape}, expected={(dim, dim)}"
            )
    phase = np.exp(2j * np.pi * (np.asarray(kpts, dtype=np.float64) @ rvec.T))  # (nk, nR)
    hk = np.einsum("kr,rij->kij", phase, h_r, optimize=True)
    return 0.5 * (hk + np.swapaxes(hk.conj(), 1, 2))


def _hk_to_hmap(hk: np.ndarray, kmesh: tuple[int, int, int], drop_tol: float) -> dict:
    """Inverse FFT H(k) -> H(R).

    kpts must be on a regular Gamma-centred mesh with shape kmesh.
    Uses np.fft.ifftn and np.fft.fftshift to match standard Wannier90 convention.
    """
    n1, n2, n3 = kmesh
    dim = hk.shape[-1]
    grid = hk.reshape(n1, n2, n3, dim, dim)

    # 3D IFFT and shift grid center to middle
    hr_grid = np.fft.ifftn(grid, axes=(0, 1, 2))
    hr_grid = np.fft.fftshift(hr_grid, axes=(0, 1, 2))

    r1 = np.arange(n1) - n1 // 2
    r2 = np.arange(n2) - n2 // 2
    r3 = np.arange(n3) - n3 // 2

    hmap: dict = {}
    for i, a in enumerate(r1):
        for j, b in enumerate(r2):
            for k, c in enumerate(r3):
                r = (int(a), int(b), int(c))
                block = np.asarray(hr_grid[i, j, k], dtype=np.complex128)
                if np.linalg.norm(block) > drop_tol:
                    hmap[r] = block
    return hmap


def _spinor_from_collinear(h_up: dict, h_dn: dict, nwan: int,
                            spin_dir=(0.0, 0.0, 1.0)) -> dict:
    """Embed collinear HR maps into a spin-major spinor HR.

    Basis: [up_0..up_{N-1}, dn_0..dn_{N-1}].
    For spin_dir = (0,0,1) this is block_diag(H_up, H_dn).
    """
    nvec = np.asarray(spin_dir, dtype=np.float64)
    norm = float(np.linalg.norm(nvec))
    if norm <= 0:
        raise ValueError("spin_direction must be nonzero")
    nx, ny, nz = nvec / norm

    all_r = sorted(set(h_up) | set(h_dn))
    zero = np.zeros((nwan, nwan), dtype=np.complex128)
    hmap: dict = {}
    for r in all_r:
        up = np.asarray(h_up.get(r, zero), dtype=np.complex128)
        dn = np.asarray(h_dn.get(r, zero), dtype=np.complex128)
        h0 = 0.5 * (up + dn)
        hz = 0.5 * (up - dn)
        block = np.zeros((2 * nwan, 2 * nwan), dtype=np.complex128)
        block[:nwan, :nwan] = h0 + nz * hz
        block[:nwan, nwan:] = (nx - 1j * ny) * hz
        block[nwan:, :nwan] = (nx + 1j * ny) * hz
        block[nwan:, nwan:] = h0 - nz * hz
        hmap[r] = block
    return hmap


def _add_hmap(out: dict, correction: dict, dim: int) -> None:
    """Add correction HR map into out in-place."""
    zero = np.zeros((dim, dim), dtype=np.complex128)
    for r, block in correction.items():
        r = tuple(int(x) for x in r)
        if r not in out:
            out[r] = zero.copy()
        out[r] = out[r] + np.asarray(block, dtype=np.complex128)


# ---------------------------------------------------------------------------
# Space Group Symmetrization Helpers (Direction B)
# ---------------------------------------------------------------------------

def _parse_win_structure(win_path: str):
    """Parse lattice vectors, atom positions, and atomic numbers from a Wannier90 .win file.

    Returns
    -------
    cell      : ndarray (3, 3) float - lattice vectors in row format (always in Angstrom)
    positions : ndarray (N_atoms, 3) float - fractional coordinates
    atomic_numbers : ndarray (N_atoms,) int
    """
    cell_lines = []
    atom_lines = []
    block = None
    cell_unit = "ang"  # Wannier90 default for unit_cell_cart is Angstrom.
    atom_unit = "ang"
    atom_coord_type = "frac"

    with open(win_path, "r", encoding="utf-8") as f:
        for raw in f:
            # strip comments
            line = raw.split("!", 1)[0].split("#", 1)[0].strip()
            if not line:
                continue
            low = line.lower()
            if low.startswith("begin unit_cell_cart"):
                block = "unit_cell"
                continue
            if low.startswith("begin atoms_frac"):
                block = "atoms"
                atom_coord_type = "frac"
                continue
            if low.startswith("begin atoms_cart"):
                block = "atoms"
                atom_coord_type = "cart"
                continue
            if low.startswith("end "):
                block = None
                continue

            if block == "unit_cell":
                toks = line.split()
                unit = toks[0].lower()
                if len(toks) == 1 and unit in {"bohr", "ang", "angstrom"}:
                    cell_unit = unit
                elif len(toks) >= 3:
                    cell_lines.append([float(x) for x in toks[:3]])
            elif block == "atoms":
                toks = line.split()
                unit = toks[0].lower()
                if atom_coord_type == "cart" and len(toks) == 1 and unit in {"bohr", "ang", "angstrom"}:
                    atom_unit = unit
                elif len(toks) >= 4:
                    atom_lines.append((toks[0], [float(x) for x in toks[1:4]]))

    if len(cell_lines) != 3:
        raise ValueError(f"Failed to find a valid 3x3 unit_cell block in {win_path}")
    if not atom_lines:
        raise ValueError(f"Failed to find atoms block in {win_path}")

    cell = np.array(cell_lines, dtype=np.float64)
    if cell_unit == "bohr":
        cell *= BOHR_TO_ANG

    positions = []
    atomic_numbers = []

    for symbol, pos in atom_lines:
        elem = re.sub(r"\d+$", "", symbol).upper()
        z = ELEMENT_TO_Z.get(elem, 1)
        atomic_numbers.append(z)
        positions.append(pos)

    positions = np.array(positions, dtype=np.float64)
    atomic_numbers = np.array(atomic_numbers, dtype=np.int64)

    if atom_coord_type == "cart":
        if atom_unit == "bohr":
            positions *= BOHR_TO_ANG
        inv_cell = np.linalg.inv(cell)
        positions = positions @ inv_cell
    positions = positions - np.floor(positions)

    return cell, positions, atomic_numbers


def _get_symmetry_operations(win_path: str, symprec: float = 1e-5):
    """Read Wannier90 win file, get symmetry operations from spglib and build atom mappings.

    Returns
    -------
    rotations   : (nsym, 3, 3) int   - Rotation matrices (fractional coordinates)
    translations: (nsym, 3) float    - Translation vectors (fractional coordinates)
    atom_maps   : (nsym, natoms) int - Mapping of source atom to target atom
    atom_shifts : (nsym, natoms, 3) int - Lattice shift vector in fractional coordinates
    positions   : (natoms, 3) float  - Fractional atomic positions
    """
    cell, positions, atomic_numbers = _parse_win_structure(win_path)

    # spglib expects cell tuple: (lattice, positions, numbers)
    # where lattice has cell vectors as row vectors
    spg_cell = (cell, positions, atomic_numbers)
    dataset = spglib.get_symmetry_dataset(spg_cell, symprec=symprec)
    if dataset is None:
        raise ValueError("spglib failed to find symmetry dataset")

    rotations = dataset.rotations          # (nsym, 3, 3)
    translations = dataset.translations    # (nsym, 3)
    nsym = len(rotations)
    natoms = len(positions)

    atom_maps = np.zeros((nsym, natoms), dtype=np.int64)
    atom_shifts = np.zeros((nsym, natoms, 3), dtype=np.int64)

    for isym in range(nsym):
        rot = rotations[isym]
        trans = translations[isym]
        for ia in range(natoms):
            p = positions[ia]
            target = rot @ p + trans
            same_species = np.flatnonzero(atomic_numbers == atomic_numbers[ia])
            diff = target[None, :] - positions[same_species]
            diff_wrap = diff - np.rint(diff)
            err = np.linalg.norm(diff_wrap, axis=1)
            ibest = int(np.argmin(err))
            best_ja = int(same_species[ibest])
            best_err = float(err[ibest])
            best_shift = np.rint(diff[ibest]).astype(np.int64)

            if best_err > 1e-4:
                raise ValueError(
                    f"Failed to map atom {ia} under symmetry operation {isym} (best_err={best_err})"
                )
            atom_maps[isym, ia] = best_ja
            atom_shifts[isym, ia] = best_shift

    return rotations, translations, atom_maps, atom_shifts, positions



def _p_orbital_rotation_matrix(R_cart: np.ndarray) -> np.ndarray:
    """Compute 3x3 rotation matrix for p-orbitals in Wannier90 'pz, px, py' basis."""
    # R_cart maps [x, y, z]^T -> [x', y', z']^T.
    # Wannier90 p basis order: [pz, px, py] (z=2, x=0, y=1).
    perm = [2, 0, 1]
    return R_cart[np.ix_(perm, perm)]


def _eval_d_orbitals(r_pts: np.ndarray) -> np.ndarray:
    """Evaluate Wannier90 orthonormal real spherical harmonics d-orbitals at r_pts."""
    x, y, z = r_pts[0], r_pts[1], r_pts[2]
    r2 = x**2 + y**2 + z**2

    dz2 = 0.5 * (3.0 * z**2 - r2)
    dxz = np.sqrt(3.0) * x * z
    dyz = np.sqrt(3.0) * y * z
    dx2y2 = 0.5 * np.sqrt(3.0) * (x**2 - y**2)
    dxy = np.sqrt(3.0) * x * y

    return np.array([dz2, dxz, dyz, dx2y2, dxy])


def _d_orbital_rotation_matrix(R_cart: np.ndarray) -> np.ndarray:
    """Compute 5x5 rotation matrix for d-orbitals in Wannier90 basis.

    Wannier90 d basis order: dz2, dxz, dyz, dx2-y2, dxy.
    """
    # Numerically fit the 5x5 rotation matrix using random points on a unit sphere.
    np.random.seed(42)
    pts = np.random.randn(3, 20)
    pts /= np.linalg.norm(pts, axis=0)

    M_orig = _eval_d_orbitals(pts)           # (5, 20)
    pts_back = R_cart.T @ pts                 # Rotate points backward: R^T * r
    M_back = _eval_d_orbitals(pts_back)       # (5, 20)

    # D @ M_orig = M_back  =>  D = M_back @ M_orig^T @ (M_orig @ M_orig^T)^-1
    D = M_back @ M_orig.T @ np.linalg.inv(M_orig @ M_orig.T)
    return D


def _su2_matrix_from_rot3d(R_cart: np.ndarray) -> np.ndarray:
    """Compute SU(2) spin-1/2 rotation matrix from 3x3 Cartesian rotation matrix.

    Handles both proper and improper rotations (spinor is invariant under inversion).
    """
    det = np.linalg.det(R_cart)
    R_proper = R_cart * det                   # Proper rotation part (det = +1)

    tr = np.trace(R_proper)
    cos_theta = 0.5 * (tr - 1.0)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    sigma_y = np.array([[0.0, -1j], [1j, 0.0]], dtype=np.complex128)
    sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    I2 = np.eye(2, dtype=np.complex128)

    if abs(theta) < 1e-9:
        return I2

    if abs(theta - np.pi) < 1e-9:
        # Pi rotation: axis n is the eigenvector of R_proper for eigenvalue +1
        # which is parallel to columns of (R_proper + I)
        mat = R_proper + np.eye(3)
        norms = np.linalg.norm(mat, axis=0)
        idx = np.argmax(norms)
        n = mat[:, idx] / norms[idx]
    else:
        # Standard axis extraction
        s = 2.0 * np.sin(theta)
        nx = (R_proper[2, 1] - R_proper[1, 2]) / s
        ny = (R_proper[0, 2] - R_proper[2, 0]) / s
        nz = (R_proper[1, 0] - R_proper[0, 1]) / s
        n = np.array([nx, ny, nz])
        n /= np.linalg.norm(n)

    n_dot_sigma = n[0] * sigma_x + n[1] * sigma_y + n[2] * sigma_z
    U_spin = np.cos(theta / 2.0) * I2 - 1j * np.sin(theta / 2.0) * n_dot_sigma
    return U_spin


def _build_wannier_rotation_matrix(rot_cart: np.ndarray, atom_map: np.ndarray,
                                    entries: list, nwan: int) -> np.ndarray:
    """Build the (2*nwan, 2*nwan) basis rotation matrix D^W in spin-major order."""
    dim = 2 * nwan
    Dw = np.zeros((dim, dim), dtype=np.complex128)

    U_spin = _su2_matrix_from_rot3d(rot_cart)
    D_p = _p_orbital_rotation_matrix(rot_cart)
    D_d = _d_orbital_rotation_matrix(rot_cart)

    atom_groups = {}
    for entry in entries:
        orb = entry["orbital"]
        for group in entry["groups"]:
            atom_idx = group["atom_index"]
            atom_groups.setdefault(atom_idx, []).append({
                "orbital": orb,
                "indices": group["indices"]
            })

    for ia in range(len(atom_map)):
        ja = atom_map[ia]
        groups_i = atom_groups.get(ia, [])
        groups_j = atom_groups.get(ja, [])

        for gi in groups_i:
            orb_i = gi["orbital"]
            idx_i = gi["indices"]
            gj = next((g for g in groups_j if g["orbital"] == orb_i), None)
            if gj is None:
                raise ValueError(
                    f"Atom {ia} has orbital {orb_i} but mapped atom {ja} does not"
                )
            idx_j = gj["indices"]

            if orb_i == "p":
                D_orb = D_p
            elif orb_i == "d":
                D_orb = D_d
            else:
                D_orb = np.eye(1, dtype=np.complex128)

            for i_spin in range(2):
                for j_spin in range(2):
                    row_idx = j_spin * nwan + np.array(idx_j)
                    col_idx = i_spin * nwan + np.array(idx_i)
                    coeff = U_spin[j_spin, i_spin]
                    Dw[np.ix_(row_idx, col_idx)] = coeff * D_orb

    return Dw


def _symmetrize_hamiltonian(hmap: dict, rotations: np.ndarray, translations: np.ndarray,
                            atom_maps: np.ndarray, atom_shifts: np.ndarray,
                            entries: list, nwan: int, drop_tol: float = 1e-12) -> dict:
    """Symmetrize the Hamiltonian map under space group operations."""
    nsym = len(rotations)
    dim = 2 * nwan

    Dw_list = []
    for isym in range(nsym):
        rot = rotations[isym]
        amap = atom_maps[isym]
        Dw = _build_wannier_rotation_matrix(rot, amap, entries, nwan)
        Dw_list.append(Dw)

    atom_spinor_indices = {}
    for entry in entries:
        for group in entry["groups"]:
            atom_idx = group["atom_index"]
            indices = group["indices"]
            spinor_idx = list(indices) + [idx + nwan for idx in indices]
            atom_spinor_indices.setdefault(atom_idx, []).extend(spinor_idx)

    natoms = len(atom_maps[0])
    sym_hmap = {}

    for R, block in hmap.items():
        block = np.asarray(block, dtype=np.complex128)
        R_arr = np.array(R, dtype=np.float64)

        for isym in range(nsym):
            rot = rotations[isym]
            amap = atom_maps[isym]
            ashift = atom_shifts[isym]
            Dw = Dw_list[isym]

            for ia in range(natoms):
                idx_i = atom_spinor_indices.get(ia, [])
                if not idx_i:
                    continue
                ia_prime = amap[ia]
                shift_i = ashift[ia]
                idx_ia_prime = atom_spinor_indices[ia_prime]

                for ja in range(natoms):
                    idx_j = atom_spinor_indices.get(ja, [])
                    if not idx_j:
                        continue
                    ja_prime = amap[ja]
                    shift_j = ashift[ja]
                    idx_ja_prime = atom_spinor_indices[ja_prime]

                    h_sub = block[np.ix_(idx_j, idx_i)]
                    Dw_j = Dw[np.ix_(idx_ja_prime, idx_j)]
                    Dw_i = Dw[np.ix_(idx_ia_prime, idx_i)]
                    h_sub_trans = Dw_j @ h_sub @ Dw_i.conj().T

                    # R' = rot @ R + shift_j - shift_i
                    R_prime_arr = rot @ R_arr + shift_j - shift_i
                    R_prime = tuple(int(x) for x in np.round(R_prime_arr))

                    if R_prime not in sym_hmap:
                        sym_hmap[R_prime] = np.zeros((dim, dim), dtype=np.complex128)
                    sym_hmap[R_prime][np.ix_(idx_ja_prime, idx_ia_prime)] += h_sub_trans

    final_hmap = {}
    for R, block in sym_hmap.items():
        avg_block = block / nsym
        if np.linalg.norm(avg_block) > drop_tol:
            final_hmap[R] = avg_block

    return final_hmap


def _hermitianize_hmap(hmap: dict, dim: int) -> dict:
    """Enforce Hermiticity H(R) = 0.5 * (H(R) + H†(-R)) for all R."""
    all_r = set(hmap.keys())
    for r in list(all_r):
        r_neg = tuple(-x for x in r)
        all_r.add(r_neg)

    zero = np.zeros((dim, dim), dtype=np.complex128)
    new_hmap = {}
    for r in all_r:
        r_neg = tuple(-x for x in r)
        h = hmap.get(r, zero)
        h_neg = hmap.get(r_neg, zero)
        block = 0.5 * (h + h_neg.conj().T)
        if np.linalg.norm(block) > 1e-20:
            new_hmap[r] = block
    return new_hmap


# ---------------------------------------------------------------------------
# Basis ordering / permutation
# ---------------------------------------------------------------------------

_GROUP_INTERLEAVED_ALIASES = {
    "win_interleaved", "site_interleaved", "atom_interleaved", "group_interleaved",
}
_ORBITAL_INTERLEAVED_ALIASES = {
    "orbital_interleaved", "groupby_orbital", "orb_interleaved", "orbital",
}


def _basis_permutation_from_win(win_path: str, nwan: int, basis_order: str):
    """Compute the permutation that reorders the spin-major spinor basis.

    In spin-major order the spinor Wannier index is::

        [w0↑, w1↑, ..., w_{N-1}↑,  w0↓, w1↓, ..., w_{N-1}↓]
         0    1    ...  N-1           N   N+1  ...  2N-1

    Two interleaved modes:
    1. group_interleaved:
       [grp0_up..., grp0_dn..., grp1_up..., grp1_dn...]
    2. orbital_interleaved:
       [grp0_o0_up, grp0_o0_dn, grp0_o1_up, grp0_o1_dn, ...]

    Returns
    -------
    perm        : ndarray (2*nwan,) int64  — permutation array
    basis_label : str                      — canonical name of chosen basis order
    group_labels: list of str
    """
    order = str(basis_order).strip().lower().replace("-", "_")
    if order in {"spin_major", "spin"}:
        return np.arange(2 * nwan, dtype=np.int64), "spin_major", []

    if order not in _GROUP_INTERLEAVED_ALIASES and order not in _ORBITAL_INTERLEAVED_ALIASES:
        raise ValueError(
            f"Unsupported --basis_order {basis_order!r}; "
            "use spin_major, group_interleaved (site_interleaved), or orbital_interleaved"
        )
    if not win_path:
        raise ValueError(f"--basis_order {basis_order} requires --win")

    _, groups, nproj = _win_projection_groups(win_path)
    if int(nproj) != int(nwan):
        raise ValueError(
            f"{win_path} projection count={nproj} but HR nwan={nwan}"
        )

    # Verify all indices cover 0..nwan-1 exactly once
    seen: list[int] = []
    for g in groups:
        seen.extend(int(x) for x in g["indices"])
    if sorted(seen) != list(range(int(nwan))):
        raise ValueError(
            f".win projection groups do not cover 0..{nwan - 1} exactly: "
            f"got {sorted(seen)[:16]}..."
        )

    perm: list[int] = []
    labels: list[str] = []

    if order in _GROUP_INTERLEAVED_ALIASES:
        # 2번 옵션: 원자 그룹 단위로 up 다 모으고 down 다 모으고
        for g in groups:
            idx = [int(x) for x in g["indices"]]
            perm.extend(idx)
            perm.extend([int(nwan) + i for i in idx])
            a = g["atom_label"]
            o = g["orbital"]
            i0, i1 = idx[0], idx[-1] + 1
            labels.append(f"{a}:{o}[{i0}:{i1}]")
        basis_label = "group_interleaved"
    else:
        # 1번 옵션: 개별 오비탈 단위로 up/down 번갈아서 배치
        for g in groups:
            idx = [int(x) for x in g["indices"]]
            for i in idx:
                perm.append(i)
                perm.append(int(nwan) + i)
            a = g["atom_label"]
            o = g["orbital"]
            i0, i1 = idx[0], idx[-1] + 1
            labels.append(f"{a}:{o}_orb_interleaved[{i0}:{i1}]")
        basis_label = "orbital_interleaved"

    return np.asarray(perm, dtype=np.int64), basis_label, labels


def _apply_basis_permutation(hmap: dict, perm: np.ndarray) -> dict:
    """Apply a basis permutation P to every HR block: H_new = H[P, :][:, P].

    If perm is the identity (i.e. arange), returns the original dict unchanged.
    """
    if np.array_equal(perm, np.arange(len(perm), dtype=np.int64)):
        return hmap
    return {
        r: np.asarray(block, dtype=np.complex128)[np.ix_(perm, perm)]
        for r, block in hmap.items()
    }


# ---------------------------------------------------------------------------
# Wannier90 U-matrix reader
# ---------------------------------------------------------------------------

def _read_wannier_u_mat(path: str):
    """Read a Wannier90 *_u.mat file.

    Returns (kpts, mats) where kpts is (nk, 3) fractional and
    mats is (nk, nwann, nwann) complex.
    """
    with open(path, "r", encoding="utf-8") as f:
        f.readline()                         # header comment
        nk, n1, n2 = (int(x) for x in f.readline().split()[:3])
        kpts = np.empty((nk, 3), dtype=np.float64)
        mats = np.empty((nk, n1, n2), dtype=np.complex128)
        for ik in range(nk):
            line = f.readline()
            while line and not line.strip():
                line = f.readline()
            parts = line.split()
            if len(parts) < 3:
                raise ValueError(f"Missing k-point line before U matrix {ik + 1} in {path}")
            kpts[ik] = [float(parts[0]), float(parts[1]), float(parts[2])]
            vals = np.empty(n1 * n2, dtype=np.complex128)
            for i in range(n1 * n2):
                re_im = f.readline().split()
                if len(re_im) < 2:
                    raise ValueError(f"Missing complex entry {i + 1} for U matrix {ik + 1} in {path}")
                vals[i] = float(re_im[0]) + 1j * float(re_im[1])
            mats[ik] = vals.reshape((n1, n2))
    return kpts, mats


def _read_u_matrix(path: str):
    """Dispatch U-matrix reader by file extension."""
    ext = os.path.splitext(str(path))[1].lower()
    if ext == ".npy":
        arr = np.load(path)
        return None, np.asarray(arr, dtype=np.complex128)
    if ext == ".npz":
        data = np.load(path, allow_pickle=True)
        key = next((k for k in ("u", "U", "u_matrix") if k in data), None)
        if key is None:
            raise ValueError(f"{path} must contain array 'u', 'U', or 'u_matrix'")
        kpts = None
        for kk in ("kpts", "kpoints", "kpts_frac"):
            if kk in data:
                kpts = np.asarray(data[kk], dtype=np.float64)
                break
        return kpts, np.asarray(data[key], dtype=np.complex128)
    return _read_wannier_u_mat(path)


def _match_u_to_kmesh(u_kpts, u_mats, kpts_target, tol: float = 1e-6):
    """Match U-matrix k-grid to the target k-mesh.

    If u_kpts is None and shapes match, returns u_mats directly.
    """
    u = np.asarray(u_mats, dtype=np.complex128)
    if u.ndim == 2:
        return np.broadcast_to(u, (len(kpts_target),) + u.shape).copy()
    if u.ndim != 3:
        raise ValueError(f"U matrix must have shape (nk,n,m) or (n,m), got {u.shape}")
    if u.shape[0] == len(kpts_target) and u_kpts is None:
        return u
    if u_kpts is None:
        raise ValueError(
            f"U matrix nk={u.shape[0]} != target nk={len(kpts_target)}, no kpts provided"
        )
    src = np.asarray(u_kpts, dtype=np.float64).reshape(-1, 3) % 1.0
    dst = np.asarray(kpts_target, dtype=np.float64).reshape(-1, 3) % 1.0
    out = np.empty((dst.shape[0],) + u.shape[1:], dtype=np.complex128)
    max_dist = 0.0
    for ik, kp in enumerate(dst):
        diff = src - kp[None, :]
        diff -= np.rint(diff)
        dist = np.linalg.norm(diff, axis=1)
        idx = int(np.argmin(dist))
        dmin = float(dist[idx])
        max_dist = max(max_dist, dmin)
        if dmin > tol:
            raise ValueError(
                f"No matching U(k) for k[{ik}]={kp.tolist()} within tol={tol:g}; "
                f"nearest dist={dmin:g}. Check --kmesh matches U.mat grid."
            )
        out[ik] = u[idx]
    if max_dist > 1e-9:
        print(
            f"[build-full-soc-hr][WARN] U(k) nearest-match max fractional dist={max_dist:g}",
            flush=True,
        )
    return out


def _kmesh_kpoints(kmesh: tuple[int, int, int]) -> np.ndarray:
    n1, n2, n3 = kmesh
    return np.array(
        [(i / n1, j / n2, k / n3) for i in range(n1) for j in range(n2) for k in range(n3)],
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# Rotated SOC: H_SOC_W(k) = U†(k) H_SOC_atomic U(k)  →  H_SOC(R)
# ---------------------------------------------------------------------------

def _build_rotated_soc_hmap(
    hsoc_atomic: np.ndarray,
    u_up: np.ndarray,
    u_dn: np.ndarray,
    kmesh: tuple[int, int, int],
    soc_rotation: str,
    drop_tol: float,
    hermitianize: bool,
) -> dict:
    """Rotate H_SOC_atomic into Wannier basis and FFT to R-space.

    Parameters
    ----------
    hsoc_atomic : (2*nwan, 2*nwan) complex  — full spin-major SOC projector
    u_up, u_dn  : (nk, nwan, nwan) complex  — Wannier U matrices per spin
    kmesh       : (n1, n2, n3)              — must match u_up/u_dn k-order
    soc_rotation: 'udag_soc_u' or 'u_soc_udag'
    """
    nk, nwan, _ = u_up.shape
    dim = 2 * nwan

    # Build spinor U: block-diagonal in spin-major order
    uspin = np.zeros((nk, dim, dim), dtype=np.complex128)
    uspin[:, :nwan, :nwan] = u_up
    uspin[:, nwan:, nwan:] = u_dn

    hsoc = np.asarray(hsoc_atomic, dtype=np.complex128)    # (dim, dim)

    if soc_rotation == "udag_soc_u":
        # H_SOC_W(k) = U†(k) @ H_SOC_atomic @ U(k)
        u_dag = np.swapaxes(uspin.conj(), 1, 2)             # (nk, dim, dim)
        hsoc_k = u_dag @ hsoc[None] @ uspin                 # (nk, dim, dim)
    elif soc_rotation == "u_soc_udag":
        u_dag = np.swapaxes(uspin.conj(), 1, 2)
        hsoc_k = uspin @ hsoc[None] @ u_dag
    else:
        raise ValueError(f"Unknown --soc_rotation {soc_rotation!r}")

    if hermitianize:
        hsoc_k = 0.5 * (hsoc_k + np.swapaxes(hsoc_k.conj(), 1, 2))

    return _hk_to_hmap(hsoc_k, kmesh, drop_tol)


# ---------------------------------------------------------------------------
# Block-norm diagnostic
# ---------------------------------------------------------------------------

def _block_norms(hmap: dict, nwan: int, d_groups=None, p_groups=None) -> dict:
    """Compute norms of uu/ud/du/dd and cross-species sub-blocks at R=(0,0,0)."""
    r0 = (0, 0, 0)
    if r0 not in hmap:
        return {}
    h = np.asarray(hmap[r0], dtype=np.complex128)
    dim = 2 * nwan
    assert h.shape == (dim, dim)
    up = np.arange(nwan)
    dn = np.arange(nwan, 2 * nwan)

    norms = {
        "uu_full": float(np.linalg.norm(h[np.ix_(up, up)])),
        "dd_full": float(np.linalg.norm(h[np.ix_(dn, dn)])),
        "ud_full": float(np.linalg.norm(h[np.ix_(up, dn)])),
        "du_full": float(np.linalg.norm(h[np.ix_(dn, up)])),
    }

    # Per-species sub-blocks if group info provided
    if d_groups and p_groups:
        d_idx = np.concatenate([g["indices"] for g in d_groups]).astype(np.int64)
        p_idx = np.concatenate([g["indices"] for g in p_groups]).astype(np.int64)
        for label, ri, ci in [
            ("dd_uu", d_idx, d_idx),
            ("pp_uu", p_idx, p_idx),
            ("dp_uu", d_idx, p_idx),
            ("pd_uu", p_idx, d_idx),
            ("dd_ud", d_idx, d_idx + nwan),
            ("pp_ud", p_idx, p_idx + nwan),
            ("dp_ud", d_idx, p_idx + nwan),
            ("pd_ud", p_idx, d_idx + nwan),
        ]:
            norms[label] = float(np.linalg.norm(h[np.ix_(ri, ci)]))

    return norms


# ---------------------------------------------------------------------------
# Hermiticity diagnostic
# ---------------------------------------------------------------------------

def _hermiticity_check(hmap: dict) -> float:
    """Return max |H(R) - H†(-R)| / max |H(R)|."""
    max_err = 0.0
    max_norm = 0.0
    for r, block in hmap.items():
        h = np.asarray(block, dtype=np.complex128)
        r_neg = tuple(-x for x in r)
        if r_neg in hmap:
            h_neg = np.asarray(hmap[r_neg], dtype=np.complex128)
            err = float(np.linalg.norm(h - h_neg.conj().T))
        else:
            err = float(np.linalg.norm(h))
        norm = float(np.linalg.norm(h))
        max_err = max(max_err, err)
        max_norm = max(max_norm, norm)
    return max_err / max(max_norm, 1e-30)


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_full_soc_hr(args):
    # ---- read collinear HR maps ----
    dim_up, degens_up, h_up = _read_wannier_hr_compat(args.up_hr)
    dim_dn, degens_dn, h_dn = _read_wannier_hr_compat(args.dn_hr)
    if dim_up != dim_dn:
        raise ValueError(f"HR dimension mismatch: up={dim_up} dn={dim_dn}")
    nwan = int(dim_up)
    print(f"[build-full-soc-hr] nwan={nwan}  nrpts_up={len(h_up)}  nrpts_dn={len(h_dn)}", flush=True)

    # ---- basis ordering ----
    # Compute the permutation BEFORE building the spinor HR so that SOC
    # index groups (already in spin-major space) are applied correctly first,
    # then the permutation is applied to the assembled matrix at the end.
    perm, basis_label, basis_groups = _basis_permutation_from_win(
        args.win, nwan, args.basis_order
    )
    print(f"[build-full-soc-hr] basis_order={basis_label}", flush=True)
    if basis_groups:
        print(f"[build-full-soc-hr] basis_groups={basis_groups}", flush=True)

    # ---- parse SOC specs ----
    specs = _parse_soc_specs(args.soc)
    if not specs:
        raise ValueError("--soc is required, e.g. 'I:p:0.6;Cr:d:0.05'")

    entries = _selected_groups(args.win, specs, nwan)

    # ---- spin-major spinor base H(R) ----
    spin_dir = tuple(float(x) for x in args.spin_direction)
    out_hmap = _spinor_from_collinear(h_up, h_dn, nwan, spin_dir=spin_dir)
    dim = 2 * nwan

    soc_gauge = str(args.soc_gauge).strip().lower()

    if soc_gauge == "wannier":
        # --- Mode 1: add atomic SOC directly in Wannier index space ---
        hsoc = _build_atomic_soc_projector(entries, nwan, args.p_order, args.d_order)
        r0 = (0, 0, 0)
        soc_hmap = {r0: hsoc}
        print("[build-full-soc-hr] mode=wannier: built atomic SOC at R=(0,0,0)", flush=True)
        for entry in entries:
            print(
                f"  {entry['element']}:{entry['orbital']} lambda={entry['lambda_ev']:g} eV "
                f"groups={[g['indices'] for g in entry['groups']]}",
                flush=True,
            )

    elif soc_gauge == "rotated":
        # --- Mode 2: U†(k) H_SOC_atomic U(k)  →  FFT  →  H_SOC(R) ---
        if args.kmesh is None:
            raise ValueError("--soc_gauge rotated requires an explicit --kmesh")
        kmesh = tuple(int(x) for x in args.kmesh)
        kpts = _kmesh_kpoints(kmesh)
        print(
            f"[build-full-soc-hr] mode=rotated  kmesh={kmesh}  nk={len(kpts)}  "
            f"rotation={args.soc_rotation}",
            flush=True,
        )

        # Load U matrices
        if args.u_up_mat or args.u_dn_mat:
            if not (args.u_up_mat and args.u_dn_mat):
                raise ValueError("Provide both --u_up_mat and --u_dn_mat, or a common --u_mat")
            ku_up, u_up_raw = _read_u_matrix(args.u_up_mat)
            ku_dn, u_dn_raw = _read_u_matrix(args.u_dn_mat)
            u_up = _match_u_to_kmesh(ku_up, u_up_raw, kpts, tol=args.u_match_tol)
            u_dn = _match_u_to_kmesh(ku_dn, u_dn_raw, kpts, tol=args.u_match_tol)
            print(
                f"[build-full-soc-hr] U_up shape={u_up.shape}  U_dn shape={u_dn.shape}",
                flush=True,
            )
        elif args.u_mat:
            ku, u_raw = _read_u_matrix(args.u_mat)
            u_common = _match_u_to_kmesh(ku, u_raw, kpts, tol=args.u_match_tol)
            if u_common.shape[1:] == (nwan, nwan):
                u_up = u_dn = u_common
            else:
                raise ValueError(
                    f"--u_mat shape {u_common.shape} incompatible with nwan={nwan}"
                )
            print(f"[build-full-soc-hr] U (common) shape={u_up.shape}", flush=True)
        else:
            raise ValueError("--soc_gauge rotated requires --u_mat or --u_up_mat/--u_dn_mat")

        if u_up.shape[1:] != (nwan, nwan) or u_dn.shape[1:] != (nwan, nwan):
            raise ValueError(
                f"U shapes up={u_up.shape} dn={u_dn.shape}; expected (nk,{nwan},{nwan})"
            )

        # Build full-spin atomic SOC projector (dim x dim)
        hsoc_atomic = _build_atomic_soc_projector(entries, nwan, args.p_order, args.d_order)

        # Rotate and FFT
        soc_hmap = _build_rotated_soc_hmap(
            hsoc_atomic,
            u_up,
            u_dn,
            kmesh,
            soc_rotation=args.soc_rotation,
            drop_tol=args.drop_tol,
            hermitianize=bool(args.hermitianize),
        )
        print(
            f"[build-full-soc-hr] rotated SOC nrpts={len(soc_hmap)} "
            f"(drop_tol={args.drop_tol:g})",
            flush=True,
        )
        for entry in entries:
            print(
                f"  {entry['element']}:{entry['orbital']} lambda={entry['lambda_ev']:g} eV",
                flush=True,
            )

    else:
        raise ValueError(f"Unknown --soc_gauge {args.soc_gauge!r}; use wannier or rotated")

    # ---- Space Group Symmetrization ----
    # Structural space-group operations are only valid for the nonmagnetic SOC
    # correction here. Applying them to the collinear base Hamiltonian averages
    # away the spin splitting / magnetic moment.
    if args.symmetrize:
        print(f"[build-full-soc-hr] Symmetrising SOC correction under space group...", flush=True)
        rotations, translations, atom_maps, atom_shifts, positions = _get_symmetry_operations(
            args.win, symprec=args.symprec
        )
        print(f"  found {len(rotations)} symmetry operations in space group", flush=True)
        soc_hmap = _symmetrize_hamiltonian(
            soc_hmap, rotations, translations, atom_maps, atom_shifts, entries, nwan,
            drop_tol=args.drop_tol
        )
        print(f"  symmetrized SOC nrpts={len(soc_hmap)}", flush=True)

    soc_hmap_for_diag = soc_hmap
    _add_hmap(out_hmap, soc_hmap, dim)

    # ---- Hermiticity Symmetrization ----
    if args.hermitianize:
        print("[build-full-soc-hr] Symmetrising H(R) = 0.5 * (H(R) + H†(-R))...", flush=True)
        out_hmap = _hermitianize_hmap(out_hmap, dim)
        print(f"  hermitianized spinor HR nrpts={len(out_hmap)}", flush=True)

    # ---- apply basis permutation (spin_major → win_interleaved etc.) ----
    # NOTE: block-norm diagnostics use spin-major indices (pre-permutation).
    # The permutation is applied right before writing.
    out_hmap_spinmaj = out_hmap          # keep reference for diagnostics
    out_hmap = _apply_basis_permutation(out_hmap, perm)

    # ---- hermiticity diagnostic (in final basis) ----
    herm_rel = _hermiticity_check(out_hmap)

    # ---- block-norm diagnostic ----
    # Try to identify d and p groups from entries for cross-species norms
    d_groups = [g for e in entries for g in e["groups"] if e["orbital"] == "d"]
    p_groups = [g for e in entries for g in e["groups"] if e["orbital"] == "p"]
    norms_base  = _block_norms(
        {r: (out_hmap[r] - soc_hmap_for_diag.get(r, np.zeros((dim, dim), dtype=np.complex128)))
         for r in out_hmap if (0, 0, 0) == r or True},
        nwan, d_groups, p_groups,
    )
    norms_soc  = _block_norms(soc_hmap_for_diag, nwan, d_groups, p_groups)
    norms_full = _block_norms(out_hmap, nwan, d_groups, p_groups)

    # ---- write output ----
    out_path = args.output
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

    # Build a lookup for the original up_hr degeneracy mapping.
    # Note: h_up keys are R-tuples.
    original_rvecs = sorted(h_up.keys())
    degen_map = {r: degens_up[ir] for ir, r in enumerate(original_rvecs)}

    # Map degeneracy for each R-vector in the sorted order of out_hmap keys
    out_rvecs = sorted(out_hmap.keys())
    degens_out = [degen_map.get(r, 1) for r in out_rvecs]

    write_wannier_hr(
        out_path,
        dim,
        degens_out,
        out_hmap,
        header=(
            "SLW full SOC spinor HR"
            f"; gauge={soc_gauge}"
            f"; basis_order={basis_label}"
            f"; soc={args.soc}"
        ),
    )

    # ---- summary ----
    print("[build-full-soc-hr] summary", flush=True)
    print(f"  up_hr:          {args.up_hr}", flush=True)
    print(f"  dn_hr:          {args.dn_hr}", flush=True)
    print(f"  win:            {args.win}", flush=True)
    print(f"  output:         {out_path}", flush=True)
    print(f"  spinor_dim:     {dim}  (nwan={nwan})", flush=True)
    print(f"  nrpts_out:      {len(out_hmap)}", flush=True)
    print(f"  soc_gauge:      {soc_gauge}", flush=True)
    print(f"  basis_order:    {basis_label}", flush=True)
    if basis_groups:
        print(f"  basis_groups:   {basis_groups}", flush=True)
    print(f"  hermiticity_rel:{herm_rel:.3e}", flush=True)

    _print_block_norms("H_base (R=0)",   norms_base)
    _print_block_norms("H_SOC (R=0)",    norms_soc)
    _print_block_norms("H_final (R=0)",  norms_full)

    if args.dump_txt:
        _write_dump_txt(args, out_hmap_spinmaj, norms_base, norms_soc, norms_full,
                        herm_rel, soc_gauge, d_groups, p_groups, dim)
        print(f"  dump_txt:       {args.dump_txt}", flush=True)

    return out_path


def _print_block_norms(label: str, norms: dict):
    if not norms:
        return
    print(f"  [{label}]", flush=True)
    keys_order = [
        ("uu_full", "full uu"),  ("dd_full", "full dd"),
        ("ud_full", "full ud"),  ("du_full", "full du"),
        ("dd_uu",   "d-d  ↑↑"), ("pp_uu",   "p-p  ↑↑"),
        ("dp_uu",   "d-p  ↑↑"), ("pd_uu",   "p-d  ↑↑"),
        ("dd_ud",   "d-d  ↑↓"), ("pp_ud",   "p-p  ↑↓"),
        ("dp_ud",   "d-p  ↑↓"), ("pd_ud",   "p-d  ↑↓"),
    ]
    for key, desc in keys_order:
        if key in norms:
            print(f"    {desc}: {norms[key]:.6g} eV", flush=True)


def _write_dump_txt(args, out_hmap, norms_base, norms_soc, norms_full,
                    herm_rel, soc_gauge, d_groups, p_groups, dim):
    nwan = dim // 2
    rows = []
    for r, block in out_hmap.items():
        nd = block.shape[0] // 2
        uu = block[:nd, :nd]
        ud = block[:nd, nd:]
        du = block[nd:, :nd]
        dd = block[nd:, nd:]
        rows.append((
            float(np.linalg.norm(block)),
            tuple(int(x) for x in r),
            float(np.linalg.norm(uu)),
            float(np.linalg.norm(ud)),
            float(np.linalg.norm(du)),
            float(np.linalg.norm(dd)),
        ))
    rows.sort(reverse=True, key=lambda x: x[0])

    def _fmt_norms(d):
        return "  ".join(f"{k}={v:.6g}" for k, v in d.items() if k in {
            "uu_full","dd_full","ud_full","du_full",
            "dp_uu","pd_uu","dp_ud","pd_ud",
        }) if d else ""

    os.makedirs(os.path.dirname(os.path.abspath(args.dump_txt)) or ".", exist_ok=True)
    with open(args.dump_txt, "w", encoding="utf-8") as f:
        f.write("# SLW build_full_soc_hr diagnostic report\n")
        f.write(f"# up_hr = {args.up_hr}\n")
        f.write(f"# dn_hr = {args.dn_hr}\n")
        f.write(f"# win = {args.win}\n")
        f.write(f"# soc = {args.soc}\n")
        f.write(f"# soc_gauge = {soc_gauge}\n")
        if soc_gauge == "rotated":
            f.write(f"# soc_rotation = {args.soc_rotation}\n")
            f.write(f"# kmesh = {tuple(int(x) for x in args.kmesh)}\n")
        f.write(f"# nwan = {nwan}  spinor_dim = {dim}\n")
        f.write(f"# hermiticity_rel_max = {herm_rel:.6e}\n")
        f.write(f"# block_norms_base:  {_fmt_norms(norms_base)}\n")
        f.write(f"# block_norms_soc:   {_fmt_norms(norms_soc)}\n")
        f.write(f"# block_norms_final: {_fmt_norms(norms_full)}\n\n")
        f.write("# R_x R_y R_z  norm_full  norm_uu  norm_ud  norm_du  norm_dd\n")
        for norm, r, nuu, nud, ndu, ndd in rows:
            f.write(
                f"{r[0]:5d} {r[1]:5d} {r[2]:5d} "
                f"{norm: .10e} {nuu: .10e} {nud: .10e} {ndu: .10e} {ndd: .10e}\n"
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    src = ap.add_argument_group("collinear HR input")
    src.add_argument("--up_hr", required=True, help="Collinear spin-up Wannier90 hr.dat")
    src.add_argument("--dn_hr", required=True, help="Collinear spin-down Wannier90 hr.dat")

    soc = ap.add_argument_group("SOC specification")
    soc.add_argument(
        "--soc", required=True,
        help="SOC specs: element:orbital:lambda_eV e.g. 'I:p:0.6;Cr:d:0.05'",
    )
    soc.add_argument("--win", required=True, help="Wannier90 .win file for projection groups")
    soc.add_argument(
        "--soc_gauge", choices=["wannier", "rotated"], default="wannier",
        help="wannier: add lambda*L.S directly in Wannier index space.  "
             "rotated: rotate via U(k) then FFT to R-space (can populate pd blocks).",
    )
    soc.add_argument("--p_order", default=WANNIER90_P_ORDER, help="p orbital order in each group")
    soc.add_argument("--d_order", default=WANNIER90_D_ORDER, help="d orbital order in each group")

    rot = ap.add_argument_group("rotated mode (--soc_gauge rotated)")
    rot.add_argument("--u_mat", default=None, help="Common Wannier90 *_u.mat / .npy / .npz")
    rot.add_argument("--u_up_mat", default=None, help="Spin-up U(k) file")
    rot.add_argument("--u_dn_mat", default=None, help="Spin-down U(k) file")
    rot.add_argument(
        "--soc_rotation", choices=["udag_soc_u", "u_soc_udag"], default="udag_soc_u",
        help="Rotation convention.  "
             "udag_soc_u: U†·H_SOC·U  (standard Wannier90 gauge).  "
             "u_soc_udag: U·H_SOC·U†.",
    )
    rot.add_argument("--kmesh", type=int, nargs=3, default=None,
                     help="Regular k-mesh for FFT (must match U.mat mesh)")
    rot.add_argument("--u_match_tol", type=float, default=1e-6,
                     help="Fractional-k tolerance for matching U(k) to kmesh")

    misc = ap.add_argument_group("output / misc")
    misc.add_argument("--spin_direction", type=float, nargs=3, default=[0.0, 0.0, 1.0])
    misc.add_argument("--hermitianize", action=argparse.BooleanOptionalAction, default=False,
                      help="Symmetrise H(R) = 0.5*(H + H†) after assembly")
    misc.add_argument("--drop_tol", type=float, default=1e-12,
                      help="Drop HR blocks with norm below this value (rotated mode)")
    misc.add_argument("--symprec", type=float, default=1e-5, help="spglib symmetry tolerance")
    misc.add_argument("--symmetrize", action=argparse.BooleanOptionalAction, default=False,
                      help="Symmetrise only the SOC correction under structural space-group operations")
    misc.add_argument(
        "--basis_order",
        choices=["spin_major", "win_interleaved", "site_interleaved",
                 "atom_interleaved", "group_interleaved", "orbital_interleaved",
                 "groupby_orbital", "orbital"],
        default="spin_major",
        help=(
            "Output spinor basis ordering.  "
            "spin_major (default): [all_up | all_down] — standard for collinear.  "
            "group_interleaved (win_interleaved): [grp0_up_all, grp0_dn_all, ...] — groups up then dn per atom.  "
            "orbital_interleaved: [grp0_o0_up, grp0_o0_dn, grp0_o1_up, grp0_o1_dn, ...] — groups up/dn per orbital."
        ),
    )
    misc.add_argument("--dump_txt", default=None, help="Write human-readable diagnostic report")
    misc.add_argument("-o", "--output", required=True, help="Output spinor hr.dat path")

    args = ap.parse_args()
    build_full_soc_hr(args)


if __name__ == "__main__":
    main()
