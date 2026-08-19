"""Acoustic Sum Rule (ASR) and point-group symmetry diagnostic for dJ/du tensors.

ASR Theory
----------
Physical requirement: a rigid translation of the whole crystal (all atoms
displaced uniformly by the same vector u) must leave the exchange interaction
unchanged.  Expanding J_{ij}(R) to first order in displacements gives the ASR:

    Sum_{kappa in UC}  Sum_{Rp}  dJ^{ab}_{ij}(R) / du^{alpha}_{kappa}(Rp)  =  0

for every bond (i,j,R), cartesian axis alpha, and spin tensor components (a,b).

In phonon-vertex language, this is equivalent to saying that the acoustic
phonon mode at q=0 (eigenvector e^alpha_{kappa} = 1/sqrt(N_kappa) for all kappa)
does NOT couple to the exchange interaction:

    V^{ab, alpha}_{ij}(R, q=0) = Sum_{kappa} (dJ^{ab}_{ij}(R) / du^alpha_{kappa})|_{q=0}
                                = Sum_{kappa} Sum_{Rp} dJ^{ab}_{ij,kappa}(R, Rp)  =  0

In the HDF5 produced by compute_dJ_epr_tensor(_mpi), the Rp dependence is
already Fourier-transformed to real space:

    dJ_tensor_r/m{kappa+1}_b{bond+1}[iRp, idisp, 3, 3]
        iRp    -> phonon unit cell lattice vector index (Rp grid from qmesh)
        idisp  -> cartesian displacement axis of target atom kappa (x=0,y=1,z=2)
        3x3    -> spin-space exchange tensor (rows/cols = x,y,z spin components)

The stored real-space derivative follows the inverse discrete Fourier transform

    dJ(Rp) = (1/Nq) * Sum_q exp(-i q.Rp) dJ(q),

so the forward transform, and hence the physical q=0 contribution, is

    X_{kappa,bond,alpha}(q=0) = Sum_{Rp} dJ[kappa,bond,Rp,alpha,:,:].

There is no additional 1/Nq factor in the physical q=0 amplitude.  Dividing
by the number of Rp cells is available only as an explicitly labelled
cell-average diagnostic.

The ASR residual (which should be zero) is:
    residual[bond, alpha, :, :] = Sum_{kappa} X_{kappa, bond, alpha}(q=0)

Point-Group Symmetry Theory
---------------------------
For a space group operation g = {Rot | t} that maps bond (i,j,R) -> (i',j',R')
and target atom kappa -> kappa', the dJ/du coupling must transform as:

    ||dJ_{i'j'}(R') / du_{kappa'}^{(R*alpha)}||_F  =  ||dJ_{ij}(R) / du_{kappa}^{alpha}||_F

For collinear magnets where the spin Hamiltonian is approximately rotationally
symmetric in spin space (or where only the Frobenius norm is checked), bonds
within the same symmetry orbit must yield the same magnitude of dJ/du(q=0)
for corresponding target atom types.

Note on partial ASR
-------------------
If NOT all unit-cell atoms are included as displaced targets (e.g. only
magnetic atoms but not ligands), the sum is only partial and a non-zero
residual is expected and not a violation.  When ALL nat atoms are displaced
(the typical phonon-vertex workflow where mag atoms + ligands are all targeted),
the full ASR can be evaluated and the residual should be ~0 up to numerical
accuracy.

For non-magnetic (ligand) target atoms, the onsite dP/du term is automatically
zero in compute_dJ_epr_tensor, since they have no local exchange projector.
Their contribution comes entirely from the dg_band (EPR electron-phonon) term.

Usage
-----
    # ASR only:
    python check_dJ_asr.py dJ_tensor_analytic.h5

    # ASR + point-group symmetry check (requires EPR h5 for lattice info):
    python check_dJ_asr.py dJ_tensor_analytic.h5 --epr_h5 epr_up.h5
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings

import h5py
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────

def _decode(arr):
    out = []
    for x in np.asarray(arr):
        out.append(x.decode("utf-8") if isinstance(x, bytes) else str(x))
    return out


def _fmt_r(v):
    return f"({int(v[0])},{int(v[1])},{int(v[2])})"


def _load_h5(path):
    """Read everything needed for ASR analysis from a dJ_tensor HDF5."""
    with h5py.File(path, "r") as h5:
        labels = _decode(h5["basic_data/atom_labels"][()])
        qmesh  = tuple(int(x) for x in h5["basic_data/qmesh"][()])
        gi     = np.asarray(h5["bonds/mag_i_atom"][()], dtype=int)
        gj     = np.asarray(h5["bonds/mag_j_atom"][()], dtype=int)
        rvec   = np.asarray(h5["bonds/R"][()],            dtype=int)
        dist   = np.asarray(h5["bonds/distance_ang"][()], dtype=float)
        rp_grid = np.asarray(h5["displacements/Rp"][()],  dtype=int)
        targets  = np.asarray(h5["displacements/target_atom"][()], dtype=int)
        disp_axes = _decode(h5["displacements/axes"][()])

        n_bond = len(gi)
        n_rp   = len(rp_grid)
        n_disp = len(disp_axes)
        nq     = int(np.prod(qmesh))  # = n_rp

        # Collect per-(target,bond) tensors: shape (n_rp, n_disp, 3, 3)
        tensors = {}   # (target, bond_idx) -> array
        grp = h5.get("dJ_tensor_r")
        if grp is None:
            raise KeyError("dJ_tensor_r group not found in HDF5; "
                           "is this a compute_dJ_epr_tensor output?")
        for key in grp:
            m = re.fullmatch(r"m(\d+)_b(\d+)", str(key))
            if m is None:
                continue
            t = int(m.group(1)) - 1   # 0-based target atom
            b = int(m.group(2)) - 1   # 0-based bond index
            tensors[(t, b)] = np.asarray(grp[key][()], dtype=float)
            # expected shape: (n_rp, n_disp, 3, 3)

    return {
        "labels":    labels,
        "qmesh":     qmesh,
        "nq":        nq,
        "gi":        gi,
        "gj":        gj,
        "rvec":      rvec,
        "dist":      dist,
        "rp_grid":   rp_grid,
        "targets":   targets,
        "disp_axes": disp_axes,
        "tensors":   tensors,
        "n_bond":    n_bond,
        "n_rp":      n_rp,
        "n_disp":    n_disp,
    }


# ──────────────────────────────────────────────────────────────────────────────
# ASR analysis
# ──────────────────────────────────────────────────────────────────────────────

_PHYSICAL_Q0 = "physical_sum"
_CELL_AVERAGE = "cell_average"


def _resolve_q0_normalization(normalization=None, *, q0_mode=None):
    """Resolve the q=0 reporting convention, including the legacy API alias.

    ``q0_mode`` predates the Fourier-convention audit.  Its historical
    semantics are preserved for callers that pass it explicitly:

      * ``q0_mode=True``  -> cell average, ``sum_Rp / n_Rp``
      * ``q0_mode=False`` -> physical q=0 sum, ``sum_Rp``

    New callers should use ``normalization``; omitting both selects the
    physical q=0 Fourier amplitude.
    """
    if normalization is not None and q0_mode is not None:
        raise TypeError("pass either normalization or legacy q0_mode, not both")

    if q0_mode is not None:
        warnings.warn(
            "q0_mode is deprecated: True means the non-physical cell average "
            "and False means the physical sum_Rp; use normalization=... instead",
            DeprecationWarning,
            stacklevel=3,
        )
        return _CELL_AVERAGE if bool(q0_mode) else _PHYSICAL_Q0

    if normalization is None:
        return _PHYSICAL_Q0

    aliases = {
        _PHYSICAL_Q0: _PHYSICAL_Q0,
        "sum": _PHYSICAL_Q0,
        "sum_Rp": _PHYSICAL_Q0,
        _CELL_AVERAGE: _CELL_AVERAGE,
        "average": _CELL_AVERAGE,
        "mean": _CELL_AVERAGE,
    }
    try:
        return aliases[str(normalization)]
    except KeyError as exc:
        allowed = f"{_PHYSICAL_Q0!r} or {_CELL_AVERAGE!r}"
        raise ValueError(f"normalization must be {allowed}, got {normalization!r}") from exc


def _q0_component(arr, *, normalization=_PHYSICAL_Q0):
    """Return the physical q=0 sum, or an explicit cell-average diagnostic."""
    normalization = _resolve_q0_normalization(normalization)
    summed = np.asarray(arr).sum(axis=0)
    if normalization == _CELL_AVERAGE:
        n_rp = int(np.asarray(arr).shape[0])
        if n_rp == 0:
            raise ValueError("cannot form a cell average from an empty Rp axis")
        return summed / float(n_rp)
    return summed


def compute_asr(data, *, normalization=None, q0_mode=None):
    """
    Compute ASR residuals.

    The ASR residual for a bond is:
        residual[alpha, a, b] = Sum_{kappa} X_{kappa}(q=0)[alpha, a, b]

    where X_{kappa}(q=0) = Sum_{Rp} dJ[kappa, bond, Rp, alpha, a, b]

    In the phonon-vertex context, this residual is proportional to the
    magnon-phonon coupling vertex for the acoustic q=0 mode, which must
    vanish by translational invariance.

    Parameters
    ----------
    data          : dict from _load_h5
    normalization : ``"physical_sum"`` (default) for the physical Fourier
                    amplitude X(q=0)=sum_Rp dJ(Rp), or ``"cell_average"``
                    for the optional non-physical per-cell diagnostic.
    q0_mode       : deprecated compatibility alias.  If explicitly supplied,
                    True selects ``"cell_average"`` and False selects
                    ``"physical_sum"``, matching its historical behavior.

    Returns
    -------
    results : dict with keys:
        "per_bond"    : list of per-bond dicts with residual info
        "global_max"  : overall max |residual| across all bonds and components
        "normalization": explicit normalization name
        "q0_mode"     : deprecated compatibility flag (True for cell average)
    """
    normalization = _resolve_q0_normalization(normalization, q0_mode=q0_mode)
    labels    = data["labels"]
    targets   = data["targets"]
    n_bond    = data["n_bond"]
    n_disp    = data["n_disp"]
    disp_axes = data["disp_axes"]
    tensors   = data["tensors"]

    per_bond = []
    for ib in range(n_bond):
        gi_b   = int(data["gi"][ib])
        gj_b   = int(data["gj"][ib])
        rvec_b = tuple(int(x) for x in data["rvec"][ib])
        dist_b = float(data["dist"][ib])
        ilab = labels[gi_b] if gi_b < len(labels) else f"Atom{gi_b+1}"
        jlab = labels[gj_b] if gj_b < len(labels) else f"Atom{gj_b+1}"

        # X_{kappa}(q=0) shape: (n_disp, 3, 3)
        # residual = Sum_{kappa} X_{kappa}(q=0)
        partial_sum = np.zeros((n_disp, 3, 3), dtype=float)
        per_target  = {}
        missing     = []

        for t in targets:
            t = int(t)
            arr = tensors.get((t, ib))
            if arr is None:
                missing.append(t)
                continue
            # arr shape: (n_rp, n_disp, 3, 3)
            # Physical q=0 component is the forward DFT at q=0: sum_Rp arr.
            s = _q0_component(arr, normalization=normalization)
            per_target[t] = s   # (n_disp, 3, 3)
            partial_sum  += s

        # ASR residual norms
        residual_norm = float(np.linalg.norm(partial_sum))   # scalar
        per_axis_norm = np.array([np.linalg.norm(partial_sum[i]) for i in range(n_disp)])

        # Per-target contribution magnitudes for reference
        contrib_norms = {t: float(np.linalg.norm(v)) for t, v in per_target.items()}

        # "Scale" = RMS of all per-target magnitudes → denominator for rel_err
        all_norms = list(contrib_norms.values())
        scale = float(np.sqrt(np.mean(np.array(all_norms)**2))) if all_norms else 0.0

        per_bond.append({
            "bond_idx":      ib,
            "gi":            gi_b,
            "gj":            gj_b,
            "rvec":          rvec_b,
            "dist":          dist_b,
            "ilab":          ilab,
            "jlab":          jlab,
            "partial_sum":   partial_sum,      # (n_disp, 3, 3)  — ASR residual
            "residual_norm": residual_norm,    # scalar
            "per_axis_norm": per_axis_norm,    # (n_disp,)
            "per_target":    per_target,       # {t: (n_disp, 3, 3)}
            "contrib_norms": contrib_norms,    # {t: float}
            "scale":         scale,             # RMS of per-target norms
            "missing":       missing,
        })

    global_max = max((b["residual_norm"] for b in per_bond), default=0.0)
    global_rms = float(np.sqrt(np.mean([b["residual_norm"]**2 for b in per_bond]))) if per_bond else 0.0
    return {
        "per_bond": per_bond,
        "global_max": global_max,
        "global_rms": global_rms,
        "normalization": normalization,
        # Retained for readers of old result dictionaries only.
        "q0_mode": normalization == _CELL_AVERAGE,
    }


def _relative_error(residual_norm, scale):
    """ASR |residual| / RMS(per-target contributions).  Dimensionless.

    A value << 1 means good ASR satisfaction.  Note: this is meaningful
    only when all unit-cell atoms are included as targets.
    """
    if scale < 1e-30:
        return float("nan")
    return residual_norm / scale


# ──────────────────────────────────────────────────────────────────────────────
# pretty-printing
# ──────────────────────────────────────────────────────────────────────────────

def _print_report(data, results, *, top_n=10, verbose=False, threshold=None):
    labels    = data["labels"]
    targets   = data["targets"]
    disp_axes = data["disp_axes"]
    per_bond  = results["per_bond"]
    normalization = results.get(
        "normalization",
        _CELL_AVERAGE if results.get("q0_mode", False) else _PHYSICAL_Q0,
    )
    x_name = "X(q=0)" if normalization == _PHYSICAL_Q0 else "X_cell_avg"
    x_target = "X_kappa(q=0)" if normalization == _PHYSICAL_Q0 else "X_cell_avg,kappa"

    n_targets_available = len(targets)
    nat_total = len(labels)
    full_asr  = (n_targets_available == nat_total)

    print("=" * 72)
    print("  dJ/du Acoustic Sum Rule (ASR) diagnostic")
    print("=" * 72)
    print(f"  H5 targets (0-based): {list(int(t) for t in targets)}")
    print(f"  labels ({nat_total} total): {labels}")
    print(f"  qmesh : {data['qmesh']}  Nq={data['nq']}  n_Rp={data['n_rp']}  n_bond={data['n_bond']}")
    print(f"  disp_axes: {disp_axes}")
    if normalization == _PHYSICAL_Q0:
        print("  normalization: physical q=0 Fourier amplitude X(q=0)=sum_Rp dJ(Rp)")
    else:
        print("  normalization: diagnostic cell average=(1/n_Rp)*sum_Rp dJ(Rp)")
        print("                 [not the physical dJ(q=0) Fourier amplitude]")
    print()
    if full_asr:
        print(f"  [FULL ASR] All {nat_total} unit-cell atoms displaced — full ASR can be checked.")
        print(f"  Residual = Sum_{{all kappa}} {x_target}  should be 0 by translational invariance.")
        print("  In phonon vertex language: V(acoustic, q=0) = 0")
    else:
        print(f"  [PARTIAL ASR] Only {n_targets_available}/{nat_total} atoms displaced.")
        print("  Zero residual is NOT guaranteed — this is a partial sum only.")
        print("  Include all unit-cell atoms as --targets for the full ASR check.")
    print()
    print("  ASR condition:")
    print(f"    Sum_{{kappa in targets}} {x_name}_{{kappa,bond,alpha}}  =  0")
    if normalization == _PHYSICAL_Q0:
        print("    X(q=0) = Sum_{Rp} dJ_tensor_r[kappa, bond, Rp, alpha, :, :]")
    else:
        print("    X_cell_avg = (1/n_Rp) * Sum_{Rp} dJ_tensor_r[kappa, bond, Rp, alpha, :, :]")
    print()

    # Sort bonds by residual (worst first)
    sorted_bonds = sorted(per_bond, key=lambda b: b["residual_norm"], reverse=True)

    print(f"  Global max  |ASR residual| = {results['global_max']:.4e}  meV/A")
    print(f"  Global RMS  |ASR residual| = {results['global_rms']:.4e}  meV/A")
    print()

    # Table header
    hdr = f"  {'bond':>4}  {'i':>6}  {'j':>6}  {'R':>12}  {'dist/A':>8}"
    for ax in disp_axes:
        hdr += f"  |res_{ax}|  "
    hdr += f"  {'|res_tot|':>11}  {'rel_err':>8}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for b in sorted_bonds:
        if threshold is not None and b["residual_norm"] < threshold:
            continue
        rel = _relative_error(b["residual_norm"], b["scale"])
        row = (f"  {b['bond_idx']+1:>4}  "
               f"{b['ilab']:>6}  {b['jlab']:>6}  "
               f"{_fmt_r(b['rvec']):>12}  {b['dist']:8.4f}")
        for i in range(len(disp_axes)):
            row += f"  {b['per_axis_norm'][i]:>9.3e}"
        row += f"  {b['residual_norm']:>11.4e}  {rel:>8.4f}"
        print(row)

    if verbose:
        print()
        print("  Per-bond detailed breakdown (worst bonds):")
        print()
        for b in sorted_bonds[:top_n]:
            print(f"  Bond {b['bond_idx']+1}: {b['ilab']} - {b['jlab']}  "
                  f"R={_fmt_r(b['rvec'])}  dist={b['dist']:.4f} A")
            print(f"    |residual| = {b['residual_norm']:.4e}  meV/A  "
                  f"rel_err={_relative_error(b['residual_norm'], b['scale']):.4f}")
            print(f"    Per-target {x_target} magnitudes (meV/A):")
            for t, norm in sorted(b["contrib_norms"].items()):
                tlab = labels[t] if t < len(labels) else f"Atom{t+1}"
                print(f"      kappa={t} ({tlab:>8}):  {norm:.4e}")
            if b["missing"]:
                print(f"    [!] Missing targets in H5: {b['missing']}")
            print(f"    ASR residual tensor  Sum_kappa {x_target} [idisp, 3x3]  (meV/A):")
            for i, ax in enumerate(disp_axes):
                mat = b["partial_sum"][i]
                print(f"      disp={ax}:")
                for row_ in mat:
                    print("        " + "  ".join(f"{v:+.4e}" for v in row_))
            print()

    print()
    print("  Legend:")
    if normalization == _PHYSICAL_Q0:
        print("    X(q=0)       : physical q=0 Fourier component, Sum_Rp dJ_r")
    else:
        print("    X_cell_avg   : diagnostic per-cell average, X(q=0)/n_Rp")
    print(f"    |res_alpha|  : ||Sum_kappa {x_target}||_F for displacement axis alpha")
    print(f"    |res_tot|    : ||Sum_kappa {x_target}||_F over all axes and tensor components")
    print(f"    rel_err      : |res_tot| / RMS_kappa(||{x_target}||_F)  [dimensionless]")
    print("                   << 1 means good ASR satisfaction")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# per-axis violin / stats summary
# ──────────────────────────────────────────────────────────────────────────────

def _stats_summary(data, results):
    per_bond  = results["per_bond"]
    disp_axes = data["disp_axes"]
    n_disp    = len(disp_axes)

    print("  Per-displacement-axis statistics of |ASR residual| (meV/A):")
    print(f"  {'axis':>5}  {'min':>10}  {'mean':>10}  {'max':>10}  {'rms':>10}")
    for i, ax in enumerate(disp_axes):
        vals = np.array([b["per_axis_norm"][i] for b in per_bond])
        print(f"  {ax:>5}  {vals.min():10.4e}  {vals.mean():10.4e}  "
              f"{vals.max():10.4e}  {np.sqrt(np.mean(vals**2)):10.4e}")
    print()

    # Global contribution norms (how large each target's total coupling is)
    target_total_norm = {}
    for b in per_bond:
        for t, v in b["per_target"].items():
            target_total_norm.setdefault(t, []).append(np.linalg.norm(v))
    labels = data["labels"]
    normalization = results.get("normalization", _PHYSICAL_Q0)
    quantity = "sum_Rp dJ_kappa" if normalization == _PHYSICAL_Q0 else "cell-average dJ_kappa"
    print(f"  Per-target overall coupling magnitude (||{quantity}||, meV/A):")
    print(f"  {'kappa':>6}  {'label':>8}  {'mean':>10}  {'max':>10}")
    for t in sorted(target_total_norm):
        arr = np.array(target_total_norm[t])
        tlab = labels[t] if t < len(labels) else f"Atom{t+1}"
        print(f"  {t:>6}  {tlab:>8}  {arr.mean():10.4e}  {arr.max():10.4e}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Point-group symmetry check (requires EPR H5 for lattice)
# ──────────────────────────────────────────────────────────────────────────────

def _get_spglib_ops(epr_h5_path, symprec=1e-4, labels=None):
    """Return spglib symmetry operations for the crystal in the EPR H5.

    Uses slw.exchange.compute_J_epr_kspace._read_epr_spglib_cell which
    handles all atomic-number fallbacks (atomic_numbers, zatom, ityp, mass,
    atom_labels).

    Returns dict with:
        rotations   : (Nsym, 3, 3) int  - rotation matrices in fractional coords
        translations: (Nsym, 3)   float - translation vectors in fractional coords
        spacegroup  : str
        lattice_ang : (3,3) Angstrom row-vector lattice for spglib
        tau_frac    : (nat, 3) fractional positions
    or None if spglib is unavailable.
    """
    try:
        import spglib
        from slw.exchange.compute_J_epr_kspace import _read_epr_spglib_cell
    except ImportError as exc:
        print(f"[check_dJ_asr][sym] import failed: {exc}; skipping symmetry check", flush=True)
        return None

    try:
        lattice_ang, tau_frac, numbers, species_source = _read_epr_spglib_cell(
            epr_h5_path, labels=labels
        )
    except Exception as exc:
        print(f"[check_dJ_asr][sym] _read_epr_spglib_cell failed: {exc}", flush=True)
        return None

    cell = (lattice_ang, tau_frac, numbers)
    sym  = spglib.get_symmetry(cell, symprec=float(symprec))
    if sym is None or len(sym.get("rotations", [])) == 0:
        print(f"[check_dJ_asr][sym] spglib returned no operations (symprec={symprec})", flush=True)
        return None

    ds  = spglib.get_symmetry_dataset(cell, symprec=float(symprec))
    try:
        sg = f"{ds.international} #{int(ds.number)}"
    except Exception:
        try:
            sg = f"{ds['international']} #{int(ds['number'])}"
        except Exception:
            sg = "unknown"
    print(
        f"[check_dJ_asr][sym] space group: {sg}  "
        f"n_ops={len(sym['rotations'])}  species_source={species_source}",
        flush=True
    )
    return {
        "rotations":    sym["rotations"],
        "translations": sym["translations"],
        "spacegroup":   sg,
        "latt": {
            "tau_frac":  tau_frac,
            "lattice_ang": lattice_ang,
            "numbers":   numbers,
        },
    }


def _apply_sym_to_bond(rot, trans, pos_frac, i, j, R, *, tol=1e-3):
    """Apply space group operation {rot|trans} to bond (i,j,R).

    Returns (i', j', R') or None if atoms cannot be matched.
    Here rot is a (3,3) int matrix acting on fractional column vectors:
        new_frac = rot @ old_frac + trans
    """
    nat = len(pos_frac)

    def map_atom(idx):
        frac_new = rot @ pos_frac[idx] + trans
        frac_mod = frac_new % 1.0
        shift    = np.round(frac_new - frac_mod).astype(int)
        for k in range(nat):
            diff = frac_mod - pos_frac[k]
            diff = diff - np.round(diff)
            if np.linalg.norm(diff) < tol:
                return k, shift
        return None, None

    ip, shift_i = map_atom(i)
    jp, shift_j = map_atom(j)
    if ip is None or jp is None:
        return None

    # Bond vector in fractional: d = pos[j] + R - pos[i]
    # After op: d' = rot @ d
    R_arr = np.asarray(R, dtype=float)
    d  = pos_frac[j] + R_arr - pos_frac[i]
    dp = rot @ d
    # Rp s.t. dp = pos[jp] + Rp - pos[ip]
    Rp = dp - (pos_frac[jp] - pos_frac[ip])
    Rp = np.round(Rp).astype(int)
    return (ip, jp, tuple(int(x) for x in Rp))


def _build_bond_orbits(data, sym_ops):
    """Group bonds into symmetry orbits using spglib operations.

    Returns list of orbits: each orbit is a list of bond indices (into data['gi'] etc.)
    that are related by space group operations.
    """
    pos_frac  = sym_ops["latt"]["tau_frac"]
    rotations = sym_ops["rotations"]
    trans     = sym_ops["translations"]
    n_bond    = data["n_bond"]

    # Build lookup: (gi, gj, R) -> bond_index
    bond_by_key = {}
    for ib in range(n_bond):
        key = (int(data["gi"][ib]), int(data["gj"][ib]),
               tuple(int(x) for x in data["rvec"][ib]))
        bond_by_key[key] = ib

    assigned = [False] * n_bond
    orbits   = []

    for ib in range(n_bond):
        if assigned[ib]:
            continue
        i = int(data["gi"][ib])
        j = int(data["gj"][ib])
        R = tuple(int(x) for x in data["rvec"][ib])
        orbit = [ib]
        assigned[ib] = True

        for rot, t in zip(rotations, trans):
            result = _apply_sym_to_bond(rot, t, pos_frac, i, j, R)
            if result is None:
                continue
            ip, jp, Rp = result
            # Check forward bond (ip,jp,Rp)
            key_fwd = (ip, jp, Rp)
            if key_fwd in bond_by_key:
                ib2 = bond_by_key[key_fwd]
                if not assigned[ib2]:
                    orbit.append(ib2)
                    assigned[ib2] = True
            # Check reverse bond (jp,ip,-Rp) — bonds may be stored canonically
            key_rev = (jp, ip, tuple(-x for x in Rp))
            if key_rev in bond_by_key:
                ib2 = bond_by_key[key_rev]
                if not assigned[ib2]:
                    orbit.append(ib2)
                    assigned[ib2] = True

        orbits.append(orbit)

    return orbits


def compute_symmetry_check(
    data, sym_ops, asr_results, *, normalization=None, q0_mode=None
):
    """Check point-group symmetry of dJ/du across bonds in the same orbit.

    For each orbit of symmetry-equivalent bonds, the total coupling
    magnitude Sum_kappa ||X_kappa(q=0)||_F should be the same for all
    bonds in the orbit (to within numerical accuracy).

    Returns
    -------
    orbits_info : list of dicts with
        'bonds'          : list of bond indices in this orbit
        'norms'          : per-bond total coupling norms  (sum over kappa)
        'sym_residual'   : max - min of norms (should be ~0)
        'rel_sym_err'    : sym_residual / mean_norm
    """
    if normalization is None and q0_mode is None and asr_results is not None:
        normalization = asr_results.get("normalization")
        if normalization is None and "q0_mode" in asr_results:
            q0_mode = asr_results["q0_mode"]
    normalization = _resolve_q0_normalization(normalization, q0_mode=q0_mode)

    targets  = data["targets"]
    tensors  = data["tensors"]
    n_bond   = data["n_bond"]
    labels   = data["labels"]

    orbits = _build_bond_orbits(data, sym_ops)

    # Compute per-bond total coupling norm = Sum_kappa ||X_kappa(q=0)||_F
    bond_total_norm = np.zeros(n_bond, dtype=float)
    for ib in range(n_bond):
        for t in targets:
            t = int(t)
            arr = tensors.get((t, ib))
            if arr is None:
                continue
            xq0 = _q0_component(arr, normalization=normalization)
            bond_total_norm[ib] += float(np.linalg.norm(xq0))

    orbits_info = []
    for orbit in orbits:
        norms = np.array([bond_total_norm[ib] for ib in orbit])
        mean_norm    = float(np.mean(norms))
        sym_residual = float(np.max(norms) - np.min(norms)) if len(norms) > 1 else 0.0
        rel_sym_err  = sym_residual / mean_norm if mean_norm > 1e-30 else float("nan")
        bond_labels  = [
            (int(data["gi"][ib]), int(data["gj"][ib]),
             tuple(int(x) for x in data["rvec"][ib]),
             float(data["dist"][ib]))
            for ib in orbit
        ]
        orbits_info.append({
            "bonds":        orbit,
            "bond_labels":  bond_labels,
            "norms":        norms,
            "mean_norm":    mean_norm,
            "sym_residual": sym_residual,
            "rel_sym_err":  rel_sym_err,
            "normalization": normalization,
        })

    return orbits_info


def _print_symmetry_report(data, orbits_info, *, threshold=None, verbose=False):
    labels = data["labels"]
    normalization = (
        orbits_info[0].get("normalization", _PHYSICAL_Q0)
        if orbits_info else _PHYSICAL_Q0
    )
    x_name = "X_kappa(q=0)" if normalization == _PHYSICAL_Q0 else "X_cell_avg,kappa"

    print("=" * 72)
    print("  Point-group symmetry check for dJ/du")
    print("=" * 72)
    print("  For each symmetry orbit: bonds related by space group ops should")
    print(f"  have equal Sum_kappa ||{x_name}||_F  (total coupling magnitude).")
    print(f"  Checking {len(orbits_info)} orbits from {data['n_bond']} bonds.")
    print()

    # Sort by rel_sym_err (worst first)
    sorted_orbits = sorted(
        orbits_info, key=lambda o: o["rel_sym_err"] if not np.isnan(o["rel_sym_err"]) else 0.0,
        reverse=True
    )

    # Summary table
    print(f"  {'orbit':>5}  {'mult':>4}  {'dist/A':>8}  {'mean|X|':>10}  {'max-min':>10}  {'rel_err':>8}")
    print("  " + "-" * 58)
    for io, orb in enumerate(sorted_orbits):
        if threshold is not None and orb["sym_residual"] < threshold:
            continue
        rel = orb["rel_sym_err"]
        dist_ref = orb["bond_labels"][0][3]
        print(f"  {io+1:>5}  {len(orb['bonds']):>4}  {dist_ref:8.4f}  "
              f"{orb['mean_norm']:10.4e}  {orb['sym_residual']:10.4e}  "
              f"{rel:>8.4f}")

    if verbose:
        print()
        print("  Per-orbit bond details:")
        for io, orb in enumerate(sorted_orbits[:10]):
            rel = orb["rel_sym_err"]
            print(f"  Orbit {io+1}  mult={len(orb['bonds'])}  "
                  f"mean|X|={orb['mean_norm']:.4e}  rel_err={rel:.4f}")
            for ib, (gi, gj, Rv, dist) in zip(orb["bonds"], orb["bond_labels"]):
                ilab = labels[gi] if gi < len(labels) else f"A{gi+1}"
                jlab = labels[gj] if gj < len(labels) else f"A{gj+1}"
                norm = float(orb["norms"][list(orb["bonds"]).index(ib)])
                print(f"    b{ib+1}: {ilab}-{jlab} R={_fmt_r(Rv)} d={dist:.4f}A  |X_sum|={norm:.4e}")
            print()

    # Global symmetry error
    all_rel = [o["rel_sym_err"] for o in orbits_info if not np.isnan(o["rel_sym_err"]) and len(o["bonds"]) > 1]
    if all_rel:
        print(f"  Global max rel_sym_err = {max(all_rel):.4e}")
        print(f"  Global RMS rel_sym_err = {float(np.sqrt(np.mean(np.array(all_rel)**2))):.4e}")
        print("  (rel_err << 1 means bonds in the same symmetry orbit have consistent")
        print("   dJ/du coupling magnitudes, as required by crystal symmetry)")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Uniform ASR Correction and H5 generation
# ──────────────────────────────────────────────────────────────────────────────

def perform_asr_correction(data, *, normalization=None, q0_mode=None):
    """
    Perform uniform ASR correction on the tensors.

    For a fixed bond b and displacement axis d:
        total_sum[d] = Sum_kappa Sum_Rp  tensor[kappa, b][Rp, d]
        delta[d] = total_sum[d] / (n_targets * n_rp)
        corrected[kappa, b][Rp, d] = tensor[kappa, b][Rp, d] - delta[d]

    Since the correction delta is uniform across all available target/Rp
    entries, it is the minimum-Frobenius-norm correction subject to the ASR
    constraint.  It is independent of whether the residual is displayed as a
    physical sum or as a cell average; both vanish after projection.

    ``normalization`` and legacy ``q0_mode`` are accepted only for API
    compatibility and validation; they do not alter the correction.

    Returns
    -------
    tensors_corrected : dict
        (target, bond_idx) -> corrected array of shape (n_rp, n_disp, 3, 3)
    """
    _resolve_q0_normalization(normalization, q0_mode=q0_mode)

    targets  = data["targets"]
    n_bond   = data["n_bond"]
    n_rp     = data["n_rp"]
    n_disp   = data["n_disp"]
    tensors  = data["tensors"]

    if len(targets) == 0:
        return {}

    tensors_corrected = {}
    for ib in range(n_bond):
        # Sum the physical q=0 amplitudes over available targets.
        total_sum = np.zeros((n_disp, 3, 3), dtype=float)
        present = []
        for t in targets:
            t = int(t)
            arr = tensors.get((t, ib))
            if arr is not None:
                # arr shape: (n_rp, n_disp, 3, 3)
                total_sum += arr.sum(axis=0)
                present.append((t, arr))

        if not present:
            continue

        # Orthogonal projection: the same delta on every constrained entry.
        delta = total_sum / float(len(present) * n_rp)  # (n_disp, 3, 3)

        for t, arr in present:
            # Subtract delta from each Rp index.
            tensors_corrected[(t, ib)] = arr - delta[None, :, :, :]

    return tensors_corrected


def _copy_attributes(src, dst):
    for k, v in src.attrs.items():
        dst.attrs[k] = v


def copy_and_correct_h5(src_path, dst_path, tensors_corrected):
    """Copy all structures, groups, and attributes to new H5, replacing dJ_tensor_r."""
    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        _copy_attributes(src, dst)
        for name in src:
            if name == "dJ_tensor_r":
                grp = dst.create_group(name)
                _copy_attributes(src[name], grp)
                for key in src[name]:
                    m = re.fullmatch(r"m(\d+)_b(\d+)", str(key))
                    if m is not None:
                        t = int(m.group(1)) - 1
                        b = int(m.group(2)) - 1
                        if (t, b) in tensors_corrected:
                            arr = tensors_corrected[(t, b)]
                            grp.create_dataset(key, data=arr)
                            _copy_attributes(src[name][key], grp[key])
                        else:
                            # Fallback if somehow missing
                            src.copy(src[name][key], grp)
                    else:
                        src.copy(src[name][key], grp)
            else:
                src.copy(src[name], dst)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Check ASR and point-group symmetry for dJ/du tensors.\n\n"
                    "ASR: Sum_{kappa} X_{kappa,bond,alpha}(q=0) = 0 (acoustic phonon decouples).\n"
                    "Sym: bonds in the same crystal symmetry orbit must have equal coupling magnitudes."
    )
    ap.add_argument("h5", help="Path to dJ_tensor HDF5 (compute_dJ_epr_tensor output)")
    ap.add_argument(
        "--output", "-o", default=None,
        help="Path to save the ASR-corrected HDF5 file. If specified, performs uniform ASR correction "
             "and writes the corrected file, then runs checks on it."
    )
    ap.add_argument(
        "--epr_h5", default=None,
        help="Path to EPR HDF5 (epr_up.h5) to read lattice for symmetry orbit analysis. "
             "If not given, symmetry check is skipped."
    )
    ap.add_argument(
        "--symprec", type=float, default=1e-4,
        help="spglib symmetry tolerance for space group detection (default: 1e-4)"
    )
    ap.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print full per-bond 3x3 ASR residual matrices and per-orbit bond details"
    )
    ap.add_argument(
        "--top_n", type=int, default=10,
        help="Number of worst bonds to show in verbose ASR mode (default: 10)"
    )
    ap.add_argument(
        "--threshold", type=float, default=None,
        help="Hide bonds/orbits with |residual| or sym_residual below this value (meV/A)"
    )
    norm_group = ap.add_mutually_exclusive_group()
    norm_group.add_argument(
        "--cell_average", action="store_true",
        help="Report the optional (1/n_Rp)*sum_Rp cell-average diagnostic instead "
             "of the default physical q=0 Fourier sum"
    )
    norm_group.add_argument(
        "--no_q0_norm", action="store_true",
        help="Deprecated compatibility alias; the physical sum_Rp is now the default (no effect)"
    )
    args = ap.parse_args()

    normalization = _CELL_AVERAGE if args.cell_average else _PHYSICAL_Q0
    if args.no_q0_norm:
        print(
            "[check_dJ_asr] --no_q0_norm is deprecated; physical sum_Rp is already the default.",
            file=sys.stderr,
            flush=True,
        )

    print(f"[check_dJ_asr] Loading: {args.h5}", flush=True)
    data    = _load_h5(args.h5)
    results = compute_asr(data, normalization=normalization)
    _print_report(data, results, top_n=args.top_n, verbose=args.verbose, threshold=args.threshold)
    _stats_summary(data, results)

    if args.epr_h5 is not None:
        print(f"[check_dJ_asr] Loading EPR lattice from: {args.epr_h5}", flush=True)
        sym_ops = _get_spglib_ops(args.epr_h5, symprec=args.symprec)
        if sym_ops is not None:
            orbits_info = compute_symmetry_check(
                data, sym_ops, results, normalization=normalization
            )
            _print_symmetry_report(data, orbits_info, threshold=args.threshold, verbose=args.verbose)
    else:
        print("[check_dJ_asr] No --epr_h5 given; skipping point-group symmetry check.", flush=True)

    if args.output is not None:
        print("\n" + "=" * 72)
        print(f"  Performing ASR Correction -> {args.output}")
        print("=" * 72)
        tensors_corrected = perform_asr_correction(data, normalization=normalization)
        copy_and_correct_h5(args.h5, args.output, tensors_corrected)
        print(f"[check_dJ_asr] Corrected file saved: {args.output}", flush=True)

        # Verify the corrected file
        print("\n[check_dJ_asr] Running diagnostic validation on corrected file...", flush=True)
        data_c = _load_h5(args.output)
        results_c = compute_asr(data_c, normalization=normalization)
        print(f"  [Verify] New Global max |ASR residual| = {results_c['global_max']:.4e}  meV/A", flush=True)
        print(f"  [Verify] New Global RMS |ASR residual| = {results_c['global_rms']:.4e}  meV/A", flush=True)
        if args.epr_h5 is not None and sym_ops is not None:
            orbits_info_c = compute_symmetry_check(
                data_c, sym_ops, results_c, normalization=normalization
            )
            all_rel = [o["rel_sym_err"] for o in orbits_info_c if not np.isnan(o["rel_sym_err"]) and len(o["bonds"]) > 1]
            if all_rel:
                print(f"  [Verify] New Global max rel_sym_err = {max(all_rel):.4e}", flush=True)


if __name__ == "__main__":
    main()
