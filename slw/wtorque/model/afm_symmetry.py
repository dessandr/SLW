"""Optional joint reciprocity/PT projection of full spinor e-ph vertices.

Only A={-I|tau}T with paired real trial orbitals is supported. This operation
is compatible with SOC; neither bare T nor an independent spin rotation is
imposed. All formulas use the atomic-position gauge at unwrapped endpoints.
This is an explicit model correction, not a repair of the underlying DFPT data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np

from slw.wtorque.model.native_spinor_frame import read_spinor_projection_metadata
from slw.wtorque.model.time_reversal import atomic_spin_time_reversal


def _dagger(x: np.ndarray) -> np.ndarray:
    return x.conj().swapaxes(-1, -2)


def _residual(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    delta = (x-y).reshape(len(x), -1)
    norm = np.linalg.norm(x.reshape(len(x), -1), axis=1)
    return {
        "relative": float(np.linalg.norm(delta) / max(np.linalg.norm(x), 1.e-30)),
        "max_k_relative": float(np.max(np.linalg.norm(delta, axis=1) / np.maximum(norm, 1.e-30))),
        "max_absolute_element": float(np.max(abs(x-y))),
    }


@dataclass(frozen=True)
class AFMInversionSymmetry:
    """A's electronic sewing without its common Bloch phase, and atom map.

    ``atom_permutation[i]`` is the target of atom i. For inversion it is an
    involution. The electronic sewing includes real-orbital parity and iσ_y.
    For displacement covectors the electronic endpoint and displacement
    Bloch phases cancel: (A g)_(pκ,α) = -S g_(κ,α)* S†.
    """

    electronic_sewing: np.ndarray
    atom_permutation: np.ndarray
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        s = np.array(self.electronic_sewing, dtype=np.complex128, copy=True)
        p = np.asarray(self.atom_permutation)
        if (s.ndim != 2 or s.shape[0] != s.shape[1] or s.shape[0] < 2
                or s.shape[0] % 2 or not np.all(np.isfinite(s))):
            raise ValueError("AFM sewing must be a finite even-dimensional square matrix")
        if (p.ndim != 1 or not len(p) or not np.issubdtype(p.dtype, np.integer)
                or not np.array_equal(np.sort(p), np.arange(len(p)))
                or not np.array_equal(p[p], np.arange(len(p)))):
            raise ValueError("AFM atom permutation must be an involution")
        eye = np.eye(len(s)); j = atomic_spin_time_reversal(len(s), "interleaved")
        if (np.max(abs(s @ _dagger(s)-eye)) > 1.e-10
                or np.max(abs(s @ s.conj()+eye)) > 1.e-10
                or np.max(abs(s @ j-j @ s.conj())) > 1.e-10):
            raise ValueError("AFM sewing must be unitary, square to -1, and commute with spin time reversal")
        p = np.array(p, dtype=int, copy=True)
        s.setflags(write=False); p.setflags(write=False)
        object.__setattr__(self, "electronic_sewing", s)
        object.__setattr__(self, "atom_permutation", p)

    def transform_vertex(self, vertex: object) -> np.ndarray:
        g = np.asarray(vertex, dtype=np.complex128)
        nw, nat = len(self.electronic_sewing), len(self.atom_permutation)
        if (g.ndim != 4 or not len(g) or g.shape[1:] != (3*nat, nw, nw)
                or not np.all(np.isfinite(g))):
            raise ValueError("AFM vertex must have finite nonempty shape (nk,3*nat,nwan,nwan)")
        s = self.electronic_sewing
        transformed = (s @ g.conj() @ _dagger(s)).reshape(len(g), nat, 3, nw, nw)
        return -transformed[:, self.atom_permutation].reshape(g.shape)

    def project_pair(self, forward: object, reverse: object) -> tuple[np.ndarray, np.ndarray, dict]:
        """Real-Frobenius orthogonal projection onto both commuting constraints.

        Reverse is g(k+q,-q), with the same ordered source k samples; it is
        not g(k,-q). The single forward matrix need not be Hermitian.
        """
        a, b = np.asarray(forward, complex), np.asarray(reverse, complex)
        if a.shape != b.shape:
            raise ValueError("forward and reverse vertices must have identical shapes")
        aa, ab = self.transform_vertex(a), self.transform_vertex(b)
        pair = .5 * (a + _dagger(b))
        apair = self.transform_vertex(pair)
        projected = .5 * (pair + apair)
        reversed_projected = _dagger(projected).copy()
        report = {
            "raw_reciprocity": _residual(a, _dagger(b)),
            "raw_afm_forward": _residual(a, aa), "raw_afm_reverse": _residual(b, ab),
            "pair_only_afm": _residual(pair, apair),
            "projected_afm": _residual(projected, self.transform_vertex(projected)),
            "projected_reciprocity": _residual(projected, _dagger(reversed_projected)),
            "correction_from_raw_forward": _residual(a, projected),
            "correction_from_raw_reverse": _residual(b, reversed_projected),
            "correction_from_pair_only": _residual(pair, projected),
            "afm_pair_commutator": _residual(apair, .5 * (aa + _dagger(ab))),
        }
        return projected, reversed_projected, report

    def project_wannier_pair(self, forward: object, reverse: object, *,
                             source_frame: object, final_frame: object,
                             q_red: object, perturbation_positions: object) -> tuple[np.ndarray, np.ndarray, dict]:
        """Transform both actual endpoints, project, then restore Wannier gauge."""
        a, b = np.asarray(forward, complex), np.asarray(reverse, complex)
        fs, ff = np.asarray(source_frame, complex), np.asarray(final_frame, complex)
        q, pos = np.asarray(q_red, float), np.asarray(perturbation_positions, float)
        nw = len(self.electronic_sewing)
        if (a.ndim != 4 or b.shape != a.shape or fs.shape != (len(a), nw, nw)
                or ff.shape != fs.shape or q.shape != (3,) or pos.shape != (3*len(self.atom_permutation), 3)
                or not all(np.all(np.isfinite(x)) for x in (fs, ff, q, pos))):
            raise ValueError("invalid endpoint frames, q, or perturbation positions")
        if max(np.max(abs(_dagger(f) @ f-np.eye(nw))) for f in (fs, ff)) > 1.e-8:
            raise ValueError("endpoint frames must be unitary")
        phase = np.exp(2j*np.pi*(pos @ q))[None, :, None, None]
        ga = (_dagger(ff)[:, None] @ a @ fs[:, None]) * phase
        gb = (_dagger(fs)[:, None] @ b @ ff[:, None]) * phase.conj()
        pa, pb, report = self.project_pair(ga, gb)
        wa = (ff[:, None] @ pa @ _dagger(fs)[:, None]) * phase.conj()
        wb = (fs[:, None] @ pb @ _dagger(ff)[:, None]) * phase
        report["projected_Wannier_reciprocity"] = _residual(wa, _dagger(wb))
        return wa, wb, report


def native_afm_inversion_symmetry(frame: Any, metadata: Any, config: dict[str, Any]) -> AFMInversionSymmetry:
    """Derive the map from QE species/geometry and complete real NNKP trials.

    The translation is explicitly supplied. QE starting moments and the
    parsed exchange model's ordered local axes/spin lengths must both obey
    the map; this does not certify converged DFT magnetic symmetry.
    """
    tau = np.asarray(config.get("afm_inversion_translation"), float)
    if tau.shape != (3,) or not np.all(np.isfinite(tau)):
        raise ValueError("afm_inversion_translation must be a finite reduced three-vector")
    root = ET.parse(config["qe_xml"]).getroot()
    structure = root.find("output/atomic_structure")
    if structure is None:
        raise ValueError("AFM symmetry requires QE output atomic_structure")
    atoms = list(structure.find("atomic_positions"))
    lattice = np.array([np.fromstring(a.text, sep=" ") for a in structure.find("cell")])
    positions = np.array([np.fromstring(a.text, sep=" ") for a in atoms]) @ np.linalg.inv(lattice)
    if positions.shape != np.asarray(metadata.tau).shape or not np.allclose(positions, metadata.tau, atol=1.e-7, rtol=0):
        raise ValueError("QE and EPR ordered atomic positions disagree")
    species = {a.attrib["name"]: a for a in root.find("input/atomic_species")}
    identities, moments = [], []
    for atom in atoms:
        sp = species[atom.attrib["name"]]
        pseudo, mass = sp.findtext("pseudo_file"), sp.findtext("mass")
        if not pseudo or mass is None:
            raise ValueError("AFM mapping requires species pseudo_file and mass")
        identities.append((pseudo.strip(), float(mass)))
        m, theta, phi = [float(sp.findtext(tag, "0")) for tag in
                         ("starting_magnetization", "spin_teta", "spin_phi")]
        # QE schema spin_teta/spin_phi are in radians (input-file angles are degrees).
        moments.append(m*np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]))
    moments = np.asarray(moments)

    def match(position, candidates, compatible):
        delta = tau-position-candidates
        hits = np.flatnonzero((np.max(abs(delta-np.rint(delta)), axis=1) < 1.e-7) & compatible)
        if len(hits) != 1:
            raise ValueError("AFM inversion has missing or ambiguous geometry/orbital partners")
        j = int(hits[0])
        return j, np.rint(delta[j]).astype(int)

    pa, atom_shifts = [], []
    for i, pos in enumerate(positions):
        j, shift = match(pos, positions, np.array([v == identities[i] for v in identities]))
        pa.append(j); atom_shifts.append(shift)
    pa = np.array(pa)
    if not np.allclose(moments[pa], -moments, atol=1.e-7, rtol=0):
        raise ValueError("QE starting moments do not obey the requested AFM inversion")
    projection = read_spinor_projection_metadata(config["nnkp"], atomic_spin_order=config["atomic_spin_order"])
    lines = [line.split("!")[0].strip() for line in Path(config["nnkp"]).read_text().splitlines()]
    lower = [line.lower() for line in lines]; begin = lower.index("begin spinor_projections")
    block = [line for line in lines[begin+1:lower.index("end spinor_projections", begin)] if line]
    spatial = np.array([np.r_[np.fromstring(block[1+3*i], sep=" "), np.fromstring(block[2+3*i], sep=" ")]
                        for i in range(int(block[0]))])[projection.projection_pairs[:, 0]]
    centers = projection.orbital_centers
    if (centers.shape != np.asarray(frame.orbital_centers).shape
            or not np.allclose(centers, frame.orbital_centers, atol=1.e-7, rtol=0)
            or len(centers)*2 != metadata.nwan):
        raise ValueError("NNKP trials and native canonical orbital frame disagree")
    angular = spatial[:, 3]
    if np.any(angular < 0) or np.any(angular != np.rint(angular)):
        raise ValueError("AFM inversion requires definite-parity real spherical/cubic trials, not hybrid orbitals")
    po, orbital_shifts = [], []
    for i, pos in enumerate(centers):
        j, shift = match(pos, centers, np.max(abs(spatial[:, 3:]-spatial[i, 3:]), axis=1) < 1.e-10)
        po.append(j); orbital_shifts.append(shift)
    po = np.asarray(po)
    pmat = np.eye(len(po))[:, po]  # inversion is its own inverse; checked below
    sewing = np.kron(pmat * ((-1.)**angular)[None, :], np.array([[0., 1.], [-1., 0.]]))
    # Magnetic projectors, local axes, and spin magnitudes must close as well.
    masks = np.asarray(frame.orbital_masks, bool); local = np.asarray(frame.local_frames)
    spins = np.asarray(config["spin_lengths"], float)
    if spins.shape != (len(masks),) or np.any(spins <= 0) or not np.all(np.isfinite(spins)):
        raise ValueError("AFM mapping requires one positive spin length per magnetic site")
    magnetic_map = []
    for i, pos in enumerate(frame.magnetic_site_positions):
        j, _ = match(pos, np.asarray(frame.magnetic_site_positions), np.ones(len(masks), bool))
        magnetic_map.append(j)
        if (not np.array_equal(masks[i], masks[j, po])
                or not np.allclose(local[j, 2]*spins[j], -local[i, 2]*spins[i], atol=1.e-7, rtol=0)):
            raise ValueError("magnetic projectors or ordered local moments violate AFM inversion")
    return AFMInversionSymmetry(sewing, pa, {
        "operation": "{-I|tau}T", "tau_fractional": tau.tolist(),
        "atom_permutation": pa.tolist(), "atom_lattice_shifts": np.asarray(atom_shifts).tolist(),
        "orbital_permutation": po.tolist(), "orbital_lattice_shifts": np.asarray(orbital_shifts).tolist(),
        "orbital_parities": ((-1.)**angular).tolist(), "magnetic_site_permutation": magnetic_map,
        "QE_spinorbit": root.findtext("output/magnetization/spinorbit"),
        "QE_noncolin": root.findtext("output/magnetization/noncolin"),
        "moment_check": "QE starting moments and exchange local axes; not converged DFT moments",
        "model_correction": True, "spin_rotation_without_lattice_rotation": False,
        "formula_atomic": "(A g)_(p_kappa,alpha) = -S_e g_(kappa,alpha)* S_e^dagger",
    })
