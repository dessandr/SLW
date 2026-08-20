"""Atomic-gauge scalar-exchange magnon--phonon scattering vertices.

The lifetime pipeline stores scalar exchange parameters in the normalized-spin
convention used by the legacy magph code.  For a mate-complete directed bond
list, its quadratic BdG kernel can be written as

    H_k = -bond_factor/(2 S) sum_b J_b F_b(k, 0),

where the factor one half converts the directed list to the full two-endpoint
bond kernel.  The scattering vertex below is the exact derivative of this
kernel, using all displacement cells in dJ(R, Rp).

Phonon eigenvectors produced by :mod:`slw.magph.legacy.adapter` are mass-normalized
cell-gauge vectors.  In that gauge the atomic-position phase

    exp[i q . (Rp + tau_kappa - tau_i)] e_atomic(kappa)

reduces to ``exp[i q . (Rp - tau_i)] e_cell(kappa)``.  This cancellation is
used explicitly so that magnon and phonon gauge conventions are not mixed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit, prange

from .kernels import HBAR_MEV_PS, Magnon_Hamiltonian_v1

LEGACY_PHONON_UNIT_CONV = 3.1062


def canonical_exchange_derivative_units(payload, *, require_explicit=False) -> str:
    """Validate that scalar exchange derivatives are in meV/angstrom."""
    raw = payload.get("units")
    if raw is None:
        if require_explicit:
            raise ValueError(
                "dJ payload must explicitly declare units compatible with meV/angstrom"
            )
        return "meV/angstrom (implicit legacy)"
    array = np.asarray(raw)
    if array.size != 1:
        raise ValueError(f"dJ units must be a scalar string, got shape {array.shape}")
    value = array.reshape(()).item()
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="strict")
    normalized = str(value).strip().lower().replace(" ", "")
    normalized = normalized.replace("ångström", "angstrom").replace("å", "angstrom")
    accepted = {
        "mev/a",
        "mev/ang",
        "mev/angstrom",
        "mev/angstroms",
    }
    if normalized not in accepted:
        raise ValueError(
            "dJ units are incompatible with the atomic-gauge lifetime prefactor: "
            f"got {value!r}, require meV/angstrom"
        )
    return "meV/angstrom"


@dataclass(frozen=True)
class ExchangeDerivativeASRReport:
    mode: str
    max_abs_before: float
    max_abs_after: float
    rms_before: float
    rms_after: float
    tolerance: float
    violated_before: bool
    violated_after: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "max_abs_before": self.max_abs_before,
            "max_abs_after": self.max_abs_after,
            "rms_before": self.rms_before,
            "rms_after": self.rms_after,
            "tolerance": self.tolerance,
            "violated_before": self.violated_before,
            "violated_after": self.violated_after,
        }


def _asr_residual(values: np.ndarray) -> np.ndarray:
    """Return sum_(Rp,kappa) dJ_b/du_(Rp,kappa,axis)."""
    arr = np.asarray(values)
    if arr.ndim != 4:
        raise ValueError(
            "dJ values must have shape (nRp,nTarget,3,nBond), "
            f"got {arr.shape}"
        )
    return np.sum(arr, axis=(0, 1))


def apply_exchange_derivative_asr(
    values,
    *,
    mode: str = "check",
    tolerance: float = 1.0e-8,
) -> tuple[np.ndarray, ExchangeDerivativeASRReport]:
    """Check or minimally project the per-bond exchange-derivative ASR.

    ``project`` applies the Euclidean minimum-norm correction under the
    constraint ``sum_(Rp,kappa) dJ = 0``.  ``check`` leaves the data unchanged,
    and ``none`` records diagnostics without treating a residual as a failure.
    """
    selected = str(mode).strip().lower()
    aliases = {"off": "none", "false": "none", "on": "project", "enforce": "project"}
    selected = aliases.get(selected, selected)
    if selected not in {"none", "check", "project"}:
        raise ValueError("dJ_asr must be one of: none, check, project")
    tol = max(float(tolerance), 0.0)
    arr = np.asarray(values)
    dtype = np.complex128 if np.iscomplexobj(arr) else np.float64
    corrected = np.array(arr, dtype=dtype, copy=True)
    before = _asr_residual(corrected)
    if selected == "project" and corrected.shape[0] * corrected.shape[1] > 0:
        corrected -= before[None, None, :, :] / float(
            corrected.shape[0] * corrected.shape[1]
        )
    after = _asr_residual(corrected)

    def stats(residual):
        absolute = np.abs(residual)
        return (
            float(np.max(absolute, initial=0.0)),
            float(np.sqrt(np.mean(absolute**2))) if absolute.size else 0.0,
        )

    max_before, rms_before = stats(before)
    max_after, rms_after = stats(after)
    report = ExchangeDerivativeASRReport(
        mode=selected,
        max_abs_before=max_before,
        max_abs_after=max_after,
        rms_before=rms_before,
        rms_after=rms_after,
        tolerance=tol,
        violated_before=bool(selected != "none" and max_before > tol),
        violated_after=bool(selected != "none" and max_after > tol),
    )
    return corrected, report


def target_atom_indices(payload) -> np.ndarray:
    if "target_indices" in payload:
        target = np.asarray(payload["target_indices"], dtype=np.int32).reshape(-1)
    else:
        import re

        indices = []
        for label in payload["targets"]:
            match = re.search(r"(\d+)$", str(label))
            if match is None:
                raise ValueError(
                    "dJ payload has no target_indices and target label "
                    f"{label!r} has no trailing atom number"
                )
            indices.append(int(match.group(1)) - 1)
        target = np.asarray(indices, dtype=np.int32)
    return target


def filter_dj_payload_bonds(payload, allowed_keys) -> dict[str, object]:
    """Return a shallow payload copy restricted to directed bond keys."""
    allowed = set(allowed_keys)
    pair = np.asarray(payload["pair_R"], dtype=np.int32).reshape(-1, 5)
    keep = np.asarray(
        [
            (int(row[0]), int(row[1]), tuple(int(x) for x in row[2:5])) in allowed
            for row in pair
        ],
        dtype=bool,
    )
    if not np.any(keep):
        raise ValueError("Bond filtering removed all real-space dJ entries")
    out = dict(payload)
    out["pair_R"] = pair[keep]
    out["values_rp"] = np.asarray(payload["values_rp"])[..., keep]
    return out


def _prepare_atomic_gauge_lambda_inputs(
    payload,
    qpts_frac,
    phonon_energies,
    phonon_vectors_cell_mass_normalized,
    atom_frac,
    *,
    phonon_floor_mev: float = 1.0e-3,
    unit_conv: float = LEGACY_PHONON_UNIT_CONV,
    asr_mode: str = "check",
    asr_tolerance: float = 1.0e-8,
):
    """Validate and normalize the inputs shared by the Lambda builders."""
    derivative_units = canonical_exchange_derivative_units(payload)
    qpts = np.asarray(qpts_frac, dtype=np.float64).reshape(-1, 3)
    energies = np.asarray(phonon_energies, dtype=np.float64)
    vectors = np.asarray(phonon_vectors_cell_mass_normalized, dtype=np.complex128)
    tau = np.asarray(atom_frac, dtype=np.float64).reshape(-1, 3)
    pair = np.asarray(payload["pair_R"], dtype=np.int32).reshape(-1, 5)
    rp = np.asarray(payload["rp_list"], dtype=np.float64).reshape(-1, 3)
    target = target_atom_indices(payload)
    values, asr = apply_exchange_derivative_asr(
        payload["values_rp"], mode=asr_mode, tolerance=asr_tolerance
    )

    nq = qpts.shape[0]
    if energies.ndim != 2 or energies.shape[0] != nq:
        raise ValueError(f"phonon energy shape {energies.shape} is incompatible with nq={nq}")
    if vectors.ndim != 4 or vectors.shape[:2] != energies.shape:
        raise ValueError(
            f"phonon vector shape {vectors.shape} is incompatible with energies {energies.shape}"
        )
    if vectors.shape[-1] != 3:
        raise ValueError(
            "phonon vectors must end in three Cartesian components, "
            f"got {vectors.shape}"
        )
    if vectors.shape[2] != tau.shape[0]:
        raise ValueError(
            "phonon vector/structure atom mismatch: "
            f"vectors have {vectors.shape[2]} atoms, atom_frac has {tau.shape[0]}"
        )
    if values.shape != (rp.shape[0], target.size, 3, pair.shape[0]):
        raise ValueError(
            "dJ array/payload mismatch: "
            f"values={values.shape}, Rp={rp.shape}, target={target.shape}, bonds={pair.shape}"
        )
    if target.size and (np.min(target) < 0 or np.max(target) >= vectors.shape[2]):
        raise ValueError(
            f"dJ target atoms {target.tolist()} exceed phonon atom count {vectors.shape[2]}"
        )
    target_coverage_complete = bool(
        target.size == vectors.shape[2]
        and np.array_equal(np.sort(np.unique(target)), np.arange(vectors.shape[2]))
    )
    if str(asr_mode).strip().lower() in {"project", "on", "enforce"} and not target_coverage_complete:
        raise ValueError(
            "dJ_asr=project requires derivatives for every phonon atom; "
            f"targets={np.sort(np.unique(target)).tolist()}, nat={vectors.shape[2]}"
        )
    if pair.size and (np.min(pair[:, :2]) < 0 or np.max(pair[:, :2]) >= tau.shape[0]):
        raise ValueError("dJ magnetic bond atom index exceeds the supplied structure")
    floor = float(phonon_floor_mev)
    if floor <= 0.0:
        raise ValueError("phonon_floor_mev must be positive")

    factor = (
        HBAR_MEV_PS
        * float(unit_conv)
        / np.sqrt(2.0 * np.maximum(energies, floor))
    )
    report = {
        "formalism": "atomic_gauge_full_Rp",
        "phonon_vector_gauge": "cell",
        "n_rp": int(rp.shape[0]),
        "n_targets": int(target.size),
        "phonon_natoms": int(vectors.shape[2]),
        "target_coverage_complete": target_coverage_complete,
        "n_bonds": int(pair.shape[0]),
        "single_rp_legacy_input": bool(payload.get("single_rp_legacy", False)),
        "phonon_floor_mev": floor,
        "phonon_unit_conversion": float(unit_conv),
        "dJ_units": derivative_units,
        "asr": asr.as_dict(),
    }
    return qpts, energies, vectors, tau, pair, rp, target, values, factor, report


def build_atomic_gauge_lambda(
    payload,
    qpts_frac,
    phonon_energies,
    phonon_vectors_cell_mass_normalized,
    atom_frac,
    *,
    phonon_floor_mev: float = 1.0e-3,
    unit_conv: float = LEGACY_PHONON_UNIT_CONV,
    asr_mode: str = "check",
    asr_tolerance: float = 1.0e-8,
    q_chunk: int = 32,
    bond_chunk: int = 64,
) -> tuple[np.ndarray, dict[str, object]]:
    """Build ``Lambda[q,nu,b]`` from every dJ(R_b,Rp) displacement cell.

    The result includes the phonon zero-point displacement and therefore has
    energy units when dJ is supplied in meV/Angstrom.
    """
    (
        qpts,
        energies,
        vectors,
        tau,
        pair,
        rp,
        target,
        values,
        factor,
        report,
    ) = _prepare_atomic_gauge_lambda_inputs(
        payload,
        qpts_frac,
        phonon_energies,
        phonon_vectors_cell_mass_normalized,
        atom_frac,
        phonon_floor_mev=phonon_floor_mev,
        unit_conv=unit_conv,
        asr_mode=asr_mode,
        asr_tolerance=asr_tolerance,
    )

    nq = qpts.shape[0]
    output = np.zeros((nq, energies.shape[1], pair.shape[0]), dtype=np.complex128)
    qstep = max(1, int(q_chunk))
    bstep = max(1, int(bond_chunk))
    origin_tau = tau[pair[:, 0]]
    for qstart in range(0, nq, qstep):
        qstop = min(nq, qstart + qstep)
        q = qpts[qstart:qstop]
        phase_rp = np.exp(2.0j * np.pi * (q @ rp.T))
        pol = vectors[qstart:qstop, :, target, :]
        for bstart in range(0, pair.shape[0], bstep):
            bstop = min(pair.shape[0], bstart + bstep)
            contracted = np.einsum(
                "qmta,rtab,qr->qmb",
                pol,
                values[:, :, :, bstart:bstop],
                phase_rp,
                optimize=True,
            )
            origin_phase = np.exp(
                -2.0j * np.pi * (q @ origin_tau[bstart:bstop].T)
            )
            output[qstart:qstop, :, bstart:bstop] = (
                contracted
                * origin_phase[:, None, :]
                * factor[qstart:qstop, :, None]
            )

    return output, report


def _validate_polarization_projectors(
    projectors,
    *,
    phonon_natoms: int,
    tolerance: float,
) -> tuple[np.ndarray, str, dict[str, object]]:
    """Return checked global or atom-resolved Cartesian projectors.

    Components need not be mutually orthogonal, idempotent, or complete.  This
    permits overlapping local rotational subspaces and weighted resolutions
    of identity.  Hermiticity is required so each reported component is a
    physical polarization observable rather than an arbitrary change of
    basis.
    """
    raw = np.asarray(projectors)
    if raw.ndim == 3 and raw.shape[1:] == (3, 3):
        scope = "global"
    elif raw.ndim == 4 and raw.shape[1:] == (phonon_natoms, 3, 3):
        scope = "atom_resolved"
    else:
        raise ValueError(
            "projectors must have shape (ncomp,3,3) or "
            f"(ncomp,{phonon_natoms},3,3), got {raw.shape}"
        )
    if raw.shape[0] < 1:
        raise ValueError("at least one polarization component is required")
    arr = np.asarray(raw, dtype=np.complex128)
    if not np.all(np.isfinite(arr)):
        raise ValueError("polarization projectors contain non-finite values")
    tol = max(float(tolerance), 0.0)
    dagger = np.swapaxes(np.conjugate(arr), -1, -2)
    hermiticity_max_abs = float(np.max(np.abs(arr - dagger), initial=0.0))
    projector_scale = max(float(np.max(np.abs(arr), initial=0.0)), 1.0)
    if hermiticity_max_abs > tol * projector_scale:
        raise ValueError(
            "polarization projectors must be Hermitian: "
            f"max|P-P^dagger|={hermiticity_max_abs:.6g}, "
            f"scaled tolerance={tol * projector_scale:.6g}"
        )

    identity = np.eye(3, dtype=np.complex128)
    summed = np.sum(arr, axis=0)
    if scope == "global":
        closure = summed - identity
        closure_frobenius_by_atom = np.asarray(
            [np.linalg.norm(closure)], dtype=np.float64
        )
    else:
        closure = summed - identity[None, :, :]
        closure_frobenius_by_atom = np.linalg.norm(closure, axis=(-2, -1))
    closure_max_abs = float(np.max(np.abs(closure), initial=0.0))
    closure_relative_frobenius = float(
        np.linalg.norm(closure) / max(np.linalg.norm(np.broadcast_to(identity, summed.shape)), np.finfo(float).tiny)
    )
    report = {
        "count": int(arr.shape[0]),
        "scope": scope,
        "hermiticity_max_abs": hermiticity_max_abs,
        "projector_sum_identity_max_abs": closure_max_abs,
        "projector_sum_identity_relative_frobenius": closure_relative_frobenius,
        "projector_sum_identity_max_frobenius_per_atom": float(
            np.max(closure_frobenius_by_atom, initial=0.0)
        ),
        "projector_tolerance": tol,
        "projector_sum_is_identity": bool(closure_max_abs <= tol),
    }
    return arr, scope, report


def build_atomic_gauge_lambda_components(
    payload,
    qpts_frac,
    phonon_energies,
    phonon_vectors_cell_mass_normalized,
    atom_frac,
    projectors,
    *,
    phonon_floor_mev: float = 1.0e-3,
    unit_conv: float = LEGACY_PHONON_UNIT_CONV,
    asr_mode: str = "check",
    asr_tolerance: float = 1.0e-8,
    projector_tolerance: float = 1.0e-10,
    reconstruction_tolerance: float = 1.0e-10,
    q_chunk: int = 32,
    bond_chunk: int = 64,
) -> tuple[np.ndarray, dict[str, object]]:
    r"""Resolve atomic-gauge ``Lambda`` into polarization components.

    ``projectors`` may be global Cartesian operators with shape
    ``(ncomp,3,3)`` or atom-resolved operators with shape
    ``(ncomp,nPhononAtom,3,3)``.  The latter are indexed with the global atom
    indices in ``payload['target_indices']`` before the dJ contraction.  Thus
    atom-local circular motion remains resolved even if its cell-summed
    angular momentum cancels by inversion symmetry.

    All components, plus an identity reference used only for diagnostics, are
    contracted together.  Consequently q/Rp phases and bond chunks are shared
    in a single vectorized pass.  The returned array has shape
    ``(ncomp,nq,nmode,nbond)``.  Components are allowed to overlap or to be
    incomplete; the report quantifies whether their sum reconstructs the full
    vertex.
    """
    (
        qpts,
        energies,
        vectors,
        tau,
        pair,
        rp,
        target,
        values,
        factor,
        report,
    ) = _prepare_atomic_gauge_lambda_inputs(
        payload,
        qpts_frac,
        phonon_energies,
        phonon_vectors_cell_mass_normalized,
        atom_frac,
        phonon_floor_mev=phonon_floor_mev,
        unit_conv=unit_conv,
        asr_mode=asr_mode,
        asr_tolerance=asr_tolerance,
    )
    projector_array, scope, component_report = _validate_polarization_projectors(
        projectors,
        phonon_natoms=vectors.shape[2],
        tolerance=projector_tolerance,
    )
    identity = np.eye(3, dtype=np.complex128)
    if scope == "global":
        augmented = np.concatenate((projector_array, identity[None, :, :]), axis=0)
        selected_projectors = augmented
    else:
        atom_identities = np.broadcast_to(
            identity, (1, vectors.shape[2], 3, 3)
        )
        augmented = np.concatenate((projector_array, atom_identities), axis=0)
        selected_projectors = augmented[:, target, :, :]

    ncomp = projector_array.shape[0]
    nq = qpts.shape[0]
    output = np.zeros(
        (ncomp, nq, energies.shape[1], pair.shape[0]), dtype=np.complex128
    )
    qstep = max(1, int(q_chunk))
    bstep = max(1, int(bond_chunk))
    origin_tau = tau[pair[:, 0]]
    reconstruction_max_abs = 0.0
    reconstruction_l2_squared = 0.0
    reference_l2_squared = 0.0

    for qstart in range(0, nq, qstep):
        qstop = min(nq, qstart + qstep)
        q = qpts[qstart:qstop]
        phase_rp = np.exp(2.0j * np.pi * (q @ rp.T))
        pol = vectors[qstart:qstop, :, target, :]
        if scope == "global":
            projected_pol = np.einsum(
                "cab,qmtb->cqmta", selected_projectors, pol, optimize=True
            )
        else:
            projected_pol = np.einsum(
                "ctab,qmtb->cqmta", selected_projectors, pol, optimize=True
            )
        for bstart in range(0, pair.shape[0], bstep):
            bstop = min(pair.shape[0], bstart + bstep)
            contracted = np.einsum(
                "cqmta,rtab,qr->cqmb",
                projected_pol,
                values[:, :, :, bstart:bstop],
                phase_rp,
                optimize=True,
            )
            origin_phase = np.exp(
                -2.0j * np.pi * (q @ origin_tau[bstart:bstop].T)
            )
            resolved = (
                contracted
                * origin_phase[None, :, None, :]
                * factor[None, qstart:qstop, :, None]
            )
            output[:, qstart:qstop, :, bstart:bstop] = resolved[:ncomp]
            difference = np.sum(resolved[:ncomp], axis=0) - resolved[ncomp]
            reconstruction_max_abs = max(
                reconstruction_max_abs,
                float(np.max(np.abs(difference), initial=0.0)),
            )
            reconstruction_l2_squared += float(np.vdot(difference, difference).real)
            reference_l2_squared += float(
                np.vdot(resolved[ncomp], resolved[ncomp]).real
            )

    reconstruction_l2 = float(np.sqrt(reconstruction_l2_squared))
    reference_l2 = float(np.sqrt(reference_l2_squared))
    reconstruction_relative_l2 = reconstruction_l2 / max(
        reference_l2, np.finfo(float).tiny
    )
    recon_tol = max(float(reconstruction_tolerance), 0.0)
    component_report.update(
        {
            "lambda_sum_minus_full_max_abs": reconstruction_max_abs,
            "lambda_sum_minus_full_l2": reconstruction_l2,
            "lambda_full_l2": reference_l2,
            "lambda_sum_minus_full_relative_l2": reconstruction_relative_l2,
            "reconstruction_tolerance": recon_tol,
            "lambda_reconstructs_full": bool(
                reconstruction_max_abs <= recon_tol
                or reconstruction_relative_l2 <= recon_tol
            ),
        }
    )
    report = dict(report)
    report["component_decomposition"] = component_report
    return output, report


@njit(fastmath=True)
def bare_vertex_atomic_gauge(
    k_cart,
    q_cart,
    lambda_modes_bonds,
    pair_R,
    lattice,
    atom_pos,
    spin_pattern,
    S,
    bond_factor,
):
    """Return the site/Nambu Hamiltonian kernel V_H[nu] for one (k,q)."""
    nmode = lambda_modes_bonds.shape[0]
    nmag = spin_pattern.shape[0]
    nchannel = 2 * nmag
    vertex = np.zeros((nmode, nchannel, nchannel), dtype=np.complex128)
    prefactor = -bond_factor / (2.0 * S)
    for ib in range(pair_R.shape[0]):
        i = int(pair_R[ib, 0])
        j = int(pair_R[ib, 1])
        r0 = pair_R[ib, 2]
        r1 = pair_R[ib, 3]
        r2 = pair_R[ib, 4]
        dx = atom_pos[j, 0] - atom_pos[i, 0] + r0 * lattice[0, 0] + r1 * lattice[1, 0] + r2 * lattice[2, 0]
        dy = atom_pos[j, 1] - atom_pos[i, 1] + r0 * lattice[0, 1] + r1 * lattice[1, 1] + r2 * lattice[2, 1]
        dz = atom_pos[j, 2] - atom_pos[i, 2] + r0 * lattice[0, 2] + r1 * lattice[1, 2] + r2 * lattice[2, 2]
        kd = k_cart[0] * dx + k_cart[1] * dy + k_cart[2] * dz
        qd = q_cart[0] * dx + q_cart[1] * dy + q_cart[2] * dz
        phase_k = np.exp(1.0j * kd)
        phase_mkq = np.exp(-1.0j * (kd + qd))
        phase_mq = np.exp(-1.0j * qd)
        align = spin_pattern[i] * spin_pattern[j]
        p = 1.0 if align > 0.0 else 0.0
        m = 1.0 - p
        for mode in range(nmode):
            coefficient = prefactor * lambda_modes_bonds[mode, ib]
            # N_b = -eta_b D_b + p_b X_b
            vertex[mode, i, i] += coefficient * (-align)
            vertex[mode, j, j] += coefficient * (-align) * phase_mq
            vertex[mode, nmag + i, nmag + i] += coefficient * (-align)
            vertex[mode, nmag + j, nmag + j] += coefficient * (-align) * phase_mq
            if p > 0.0:
                vertex[mode, i, j] += coefficient * phase_k
                vertex[mode, j, i] += coefficient * phase_mkq
                vertex[mode, nmag + i, nmag + j] += coefficient * phase_k
                vertex[mode, nmag + j, nmag + i] += coefficient * phase_mkq
            if m > 0.0:
                # C_b = m_b X_b: antiparallel bonds are anomalous.
                vertex[mode, i, nmag + j] += coefficient * phase_k
                vertex[mode, j, nmag + i] += coefficient * phase_mkq
                vertex[mode, nmag + i, j] += coefficient * phase_k
                vertex[mode, nmag + j, i] += coefficient * phase_mkq
    return vertex


@njit(parallel=True, fastmath=True)
def g_kq_loop_atomic_gauge(
    q_mesh_flat_cart,
    k_cart,
    lambda_qnu_b,
    pair_R,
    Uk,
    S,
    J0,
    lattice,
    atom_pos,
    spin_pattern,
    anisotropy,
    bond_factor,
):
    """Build band-basis vertices and signed intermediate magnon energies."""
    total_q = q_mesh_flat_cart.shape[0]
    nmode = lambda_qnu_b.shape[1]
    nchannel = 2 * spin_pattern.shape[0]
    g_cache = np.zeros((total_q, nmode, nchannel, nchannel), dtype=np.complex128)
    e_cache = np.zeros((total_q, nchannel), dtype=np.float64)
    for iq in prange(total_q):
        q_cart = q_mesh_flat_cart[iq]
        e_kq, Ukq = Magnon_Hamiltonian_v1(
            k_cart + q_cart,
            S,
            J0,
            lattice,
            atom_pos,
            spin_pattern,
            bond_factor,
            anisotropy,
        )
        bare = bare_vertex_atomic_gauge(
            k_cart,
            q_cart,
            lambda_qnu_b[iq],
            pair_R,
            lattice,
            atom_pos,
            spin_pattern,
            S,
            bond_factor,
        )
        Ukq_h = np.conjugate(Ukq).T
        for mode in range(nmode):
            g_cache[iq, mode] = Ukq_h @ bare[mode] @ Uk
        e_cache[iq] = e_kq
    return g_cache, e_cache


@njit(parallel=True, fastmath=True)
def g_kq_loop_atomic_gauge_components(
    q_mesh_flat_cart,
    k_cart,
    lambda_components_qnu_b,
    pair_R,
    Uk,
    S,
    J0,
    lattice,
    atom_pos,
    spin_pattern,
    anisotropy,
    bond_factor,
):
    """Build component-resolved band vertices without repeated eigensolves.

    The Lambda input and vertex output have shapes
    ``(ncomp,nq,nmode,nbond)`` and
    ``(ncomp,nq,nmode,nchannel,nchannel)``, respectively.  For each q point,
    the internal magnon eigensystem is computed once.  Component and mode axes
    are flattened for one bare-vertex bond pass, then restored after the band
    transformation.  The returned energy cache is therefore shared by every
    component.
    """
    ncomp = lambda_components_qnu_b.shape[0]
    total_q = q_mesh_flat_cart.shape[0]
    nmode = lambda_components_qnu_b.shape[2]
    nbond = lambda_components_qnu_b.shape[3]
    nchannel = 2 * spin_pattern.shape[0]
    g_cache = np.zeros(
        (ncomp, total_q, nmode, nchannel, nchannel), dtype=np.complex128
    )
    e_cache = np.zeros((total_q, nchannel), dtype=np.float64)
    for iq in prange(total_q):
        q_cart = q_mesh_flat_cart[iq]
        e_kq, Ukq = Magnon_Hamiltonian_v1(
            k_cart + q_cart,
            S,
            J0,
            lattice,
            atom_pos,
            spin_pattern,
            bond_factor,
            anisotropy,
        )
        flattened_lambda = np.empty(
            (ncomp * nmode, nbond), dtype=np.complex128
        )
        for component in range(ncomp):
            for mode in range(nmode):
                flattened_mode = component * nmode + mode
                for bond in range(nbond):
                    flattened_lambda[flattened_mode, bond] = (
                        lambda_components_qnu_b[component, iq, mode, bond]
                    )
        flattened_bare = bare_vertex_atomic_gauge(
            k_cart,
            q_cart,
            flattened_lambda,
            pair_R,
            lattice,
            atom_pos,
            spin_pattern,
            S,
            bond_factor,
        )
        Ukq_h = np.conjugate(Ukq).T
        for component in range(ncomp):
            for mode in range(nmode):
                flattened_mode = component * nmode + mode
                g_cache[component, iq, mode] = (
                    Ukq_h @ flattened_bare[flattened_mode] @ Uk
                )
        e_cache[iq] = e_kq
    return g_cache, e_cache


@njit(parallel=True, fastmath=True)
def fixed_q_gbar_atomic_gauge(
    k_mesh_cart,
    q_cart,
    lambda_modes_bonds,
    pair_R,
    S,
    J0,
    lattice,
    atom_pos,
    spin_pattern,
    anisotropy,
    bond_factor,
    physical_only=True,
):
    """Branch-resolved fixed-phonon-q norm using the atomic-gauge vertex."""
    total_k = k_mesh_cart.shape[0]
    nchannel = 2 * spin_pattern.shape[0]
    nphysical = spin_pattern.shape[0] if physical_only else nchannel
    result = np.zeros((total_k, nphysical), dtype=np.float64)
    for ik in prange(total_k):
        k_cart = k_mesh_cart[ik]
        _, Uk = Magnon_Hamiltonian_v1(
            k_cart,
            S,
            J0,
            lattice,
            atom_pos,
            spin_pattern,
            bond_factor,
            anisotropy,
        )
        _, Ukq = Magnon_Hamiltonian_v1(
            k_cart + q_cart,
            S,
            J0,
            lattice,
            atom_pos,
            spin_pattern,
            bond_factor,
            anisotropy,
        )
        bare = bare_vertex_atomic_gauge(
            k_cart,
            q_cart,
            lambda_modes_bonds,
            pair_R,
            lattice,
            atom_pos,
            spin_pattern,
            S,
            bond_factor,
        )
        Ukq_h = np.conjugate(Ukq).T
        for n in range(nphysical):
            norm_squared = 0.0
            for mode in range(bare.shape[0]):
                band_vertex = Ukq_h @ bare[mode] @ Uk
                for m in range(nphysical):
                    value = band_vertex[m, n]
                    norm_squared += value.real * value.real + value.imag * value.imag
            result[ik, n] = np.sqrt(norm_squared / float(nphysical))
    return result
