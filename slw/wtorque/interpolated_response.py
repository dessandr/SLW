"""Rank-local arbitrary-q torque response on a uniform integration k mesh.

H and the full Cartesian g (including the EPR polar add-back) are interpolated
in their original periodic Wannier gauge. Both endpoints are then transformed
to the interpolated rigid atomic-spin frame. No interpolation of previously
projected magnon/phonon mode coefficients is performed here.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from slw.wtorque.model.interpolated_spinor_frame import InterpolatedSpinorFrame
from slw.wtorque.model.afm_symmetry import native_afm_inversion_symmetry
from slw.wtorque.model.time_reversal import atomic_spin_time_reversal
from slw.wtorque.torque.direct_vertex import (
    finite_q_direct_vertices, retarded_direct_loop_eigh_zero_temperature,
)
from slw.wtorque.torque.kernel import retarded_bubble_loop_eigh_zero_temperature_finite_q
from slw.wtorque.torque.qpair import finalize_q_pair
from slw.wtorque.torque.vertices import finite_q_vertices


def _dagger(value: np.ndarray) -> np.ndarray:
    return value.conj().swapaxes(-1, -2)


def _relative(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.linalg.norm(first - second) / max(np.linalg.norm(first), np.finfo(float).tiny))


class ArbitraryQResponse:
    """Compute a completed ±q Cartesian bubble and fixed-frame direct kernel.

    MPI ownership belongs to the caller: construct one object per active rank
    and assign independent q pairs. Only one pair of responses is returned;
    transient electronic vertices are released after each call. The dense k
    mesh is independent of q; all shifted endpoints are evaluated explicitly.
    """

    def __init__(self, coarse_frame: Any, backend: Any, kmesh: object, config: dict[str, Any]) -> None:
        raw_mesh = np.asarray(kmesh)
        if (raw_mesh.shape != (3,) or not np.issubdtype(raw_mesh.dtype, np.integer)
                or np.any(raw_mesh < 1)):
            raise ValueError("dense kmesh must contain three positive integers")
        self.kmesh = raw_mesh.astype(np.int64)
        self.kpoints = np.asarray([
            (i / self.kmesh[0], j / self.kmesh[1], k / self.kmesh[2])
            for i in range(self.kmesh[0]) for j in range(self.kmesh[1]) for k in range(self.kmesh[2])
        ], dtype=float)
        indices = np.indices(self.kmesh).reshape(3, -1).T
        opposite = (-indices) % self.kmesh
        self.minus = ((opposite[:, 0] * self.kmesh[1] + opposite[:, 1]) * self.kmesh[2] + opposite[:, 2]).astype(int)
        self.k_weights = np.full(len(self.kpoints), 1. / len(self.kpoints))
        self.backend, self.coarse_frame = backend, coarse_frame
        self.meta = backend.metadata
        self.config = dict(config)
        self.include_direct = bool(config.get("include_direct_vertex", False))
        if self.include_direct and config.get("g_xc_source") != "fixed_frame_tr_odd":
            raise ValueError("dense native direct term requires g_xc_source=fixed_frame_tr_odd")
        self.pair_policy = config.get("g_pair_policy", "raw")
        if self.pair_policy not in {"raw", "hermitian_pair_average"}:
            raise ValueError("g_pair_policy must be raw or hermitian_pair_average")
        self.symmetry_policy = config.get("vertex_symmetry_policy", "none")
        if self.symmetry_policy not in {"none", "afm_inversion_pair"}:
            raise ValueError("vertex_symmetry_policy must be none or afm_inversion_pair")
        self.vertex_symmetry = None
        if self.symmetry_policy == "afm_inversion_pair":
            if self.pair_policy != "hermitian_pair_average":
                raise ValueError("afm_inversion_pair requires g_pair_policy=hermitian_pair_average")
            self.vertex_symmetry = native_afm_inversion_symmetry(coarse_frame, self.meta, config)
        self.chunk = int(config.get("perturbation_chunk", 3))
        if self.chunk < 1:
            raise ValueError("perturbation_chunk must be positive")
        self.integral = {
            "energy_min_eV": float(config["energy_min_eV"]),
            "occupied_energy_max_eV": float(config["fermi_energy_eV"]),
            "eta_eV": float(config["eta_eV"]),
            "perturbation_chunk": self.chunk,
        }
        if (not all(np.isfinite(value) for value in self.integral.values())
                or self.integral["eta_eV"] <= 0
                or self.integral["occupied_energy_max_eV"] <= self.integral["energy_min_eV"]):
            raise ValueError("response energy interval and positive broadening must be finite")
        if config.get("temperature_K", 0.) != 0.:
            raise ValueError("analytic dense native response requires temperature_K=0")
        self.frame_interpolator = InterpolatedSpinorFrame(
            coarse_frame, kmesh=self.meta.nk_grid, lattice_column_vectors=self.meta.at,
            wannier_centers=self.meta.wc,
            rank_tolerance=float(config.get("frame_interpolation_rank_tolerance", 1.e-8)),
        )
        self.base = self.frame_interpolator.evaluate(self.kpoints)
        self.j = atomic_spin_time_reversal(self.meta.nwan, "interleaved")
        self.b0 = self.base.periodic_frame @ self.j @ self.base.periodic_frame[self.minus].swapaxes(-1, -2)
        h = backend.evaluate_h(self.kpoints)
        self.h0 = _dagger(self.base.atomic_frame) @ h @ self.base.atomic_frame
        self.xc0 = self._exchange(h, h[self.minus], self.b0, self.base.atomic_frame)
        self.positions = np.repeat(self.meta.tau, 3, axis=0)
        eigenvalues = np.linalg.eigvalsh(self.h0)
        mu = self.integral["occupied_energy_max_eV"]
        self.diagnostics = {
            "kmesh": self.kmesh.tolist(), "nk": len(self.kpoints),
            "coarse_kmesh": list(self.meta.nk_grid), "coarse_qmesh": list(self.meta.nq_grid),
            "frame_interpolation": self.frame_interpolator.diagnostics,
            "base_frame": self.base.diagnostics,
            "occupied_band_counts": sorted(set(np.sum(eigenvalues < mu, axis=1).tolist())),
            "valence_max_eV": float(eigenvalues[eigenvalues < mu].max()) if np.any(eigenvalues < mu) else None,
            "conduction_min_eV": float(eigenvalues[eigenvalues > mu].min()) if np.any(eigenvalues > mu) else None,
            "g_pair_policy": self.pair_policy, "direct_term_enabled": self.include_direct,
            "vertex_symmetry_policy": self.symmetry_policy,
            "vertex_symmetry": self.vertex_symmetry.diagnostics if self.vertex_symmetry else None,
            "raw_g_reciprocity_policy": "diagnostic before optional pair averaging",
            "configured_coarse_g_reciprocity_tolerance": config.get("g_reciprocity_tolerance"),
            "configured_coarse_g_reciprocity_gate_applied_to_interpolation": False,
            "projector_motion_enabled": False, "temperature_K": 0.,
            "kernel_units": "eV/angstrom", "arbitrary_q_endpoints": True,
        }

    @staticmethod
    def _exchange(h: np.ndarray, hm: np.ndarray, sewing: np.ndarray, frame: np.ndarray) -> np.ndarray:
        odd = .5 * (h - sewing @ hm.conj() @ _dagger(sewing))
        return _dagger(frame) @ odd @ frame

    def _bubble(self, q: np.ndarray, final: Any, hfinal: np.ndarray, xcfinal: np.ndarray,
                vertex: np.ndarray) -> np.ndarray:
        forward = finite_q_vertices(
            self.xc0, xcfinal, orbital_masks=self.coarse_frame.orbital_masks,
            local_frames=self.coarse_frame.local_frames, q_red=q,
            orbital_centers=self.coarse_frame.orbital_centers,
            magnetic_site_positions=self.coarse_frame.magnetic_site_positions,
        )
        reverse = _dagger(forward).reshape(len(self.kpoints), -1, self.meta.nwan, self.meta.nwan)
        transformed = self.frame_interpolator.transform_vertex(
            vertex, self.base, final, perturbation_positions=self.positions,
        )
        return retarded_bubble_loop_eigh_zero_temperature_finite_q(
            self.h0, hfinal, reverse, transformed, self.k_weights, **self.integral,
        )

    def _direct(self, q: np.ndarray, forward: np.ndarray, shifted: np.ndarray) -> np.ndarray:
        nt, npert = 2 * len(self.coarse_frame.local_frames), 3 * self.meta.nat
        result = np.empty((nt, npert), complex)
        for start in range(0, npert, self.chunk):
            stop = min(start + self.chunk, npert)
            vertex = finite_q_direct_vertices(
                forward[:, start:stop], g_xc_at_k_minus_q=shifted[:, start:stop],
                kpoints=self.kpoints, q_red=q, orbital_masks=self.coarse_frame.orbital_masks,
                local_frames=self.coarse_frame.local_frames, orbital_centers=self.coarse_frame.orbital_centers,
                magnetic_site_positions=self.coarse_frame.magnetic_site_positions,
            )
            result[:, start:stop] = retarded_direct_loop_eigh_zero_temperature(
                self.h0, vertex.reshape(len(self.kpoints), nt, stop-start, self.meta.nwan, self.meta.nwan),
                self.k_weights, **self.integral,
            )
        return result

    def evaluate_pair(self, qpoint: object) -> dict[str, Any]:
        q = np.asarray(qpoint, dtype=float)
        if q.shape != (3,) or not np.all(np.isfinite(q)):
            raise ValueError("qpoint must be a finite reduced three-vector")
        plus = self.frame_interpolator.evaluate(self.kpoints + q)
        minus = self.frame_interpolator.evaluate(self.kpoints - q)
        bp = plus.periodic_frame @ self.j @ minus.periodic_frame[self.minus].swapaxes(-1, -2)
        bm = minus.periodic_frame @ self.j @ plus.periodic_frame[self.minus].swapaxes(-1, -2)
        hp, hm = self.backend.evaluate_h(self.kpoints + q), self.backend.evaluate_h(self.kpoints - q)
        xp = self._exchange(hp, hm[self.minus], bp, plus.atomic_frame)
        xm = self._exchange(hm, hp[self.minus], bm, minus.atomic_frame)
        hp = _dagger(plus.atomic_frame) @ hp @ plus.atomic_frame
        hm = _dagger(minus.atomic_frame) @ hm @ minus.atomic_frame
        # A/B and C/D are Hermitian pairs at exactly opposite endpoints.
        a = self.backend.evaluate_g(self.kpoints, q)
        b = self.backend.evaluate_g(self.kpoints + q, -q)
        c = self.backend.evaluate_g(self.kpoints, -q)
        d = self.backend.evaluate_g(self.kpoints - q, q)
        reciprocity = [_relative(a, _dagger(b)), _relative(c, _dagger(d))]
        symmetry_diagnostics = None
        if self.vertex_symmetry is not None:
            project = self.vertex_symmetry.project_wannier_pair
            a, b, forward_diagnostics = project(
                a, b, source_frame=self.base.atomic_frame, final_frame=plus.atomic_frame,
                q_red=q, perturbation_positions=self.positions,
            )
            c, d, reverse_diagnostics = project(
                c, d, source_frame=self.base.atomic_frame, final_frame=minus.atomic_frame,
                q_red=-q, perturbation_positions=self.positions,
            )
            symmetry_diagnostics = {"plus_q": forward_diagnostics, "minus_q": reverse_diagnostics}
        elif self.pair_policy == "hermitian_pair_average":
            a = .5 * (a + _dagger(b)); b = _dagger(a)
            c = .5 * (c + _dagger(d)); d = _dagger(c)
        retarded_bubble = np.asarray([
            self._bubble(q, plus, hp, xp, a), self._bubble(-q, minus, hm, xm, c),
        ])
        retarded_direct = np.zeros_like(retarded_bubble)
        direct_diagnostics: dict[str, Any] | None = None
        if self.include_direct:
            ax = .5 * (a - bp[:, None] @ c[self.minus].conj() @ _dagger(self.b0)[:, None])
            dx = .5 * (d - self.b0[:, None] @ b[self.minus].conj() @ _dagger(bm)[:, None])
            cx = .5 * (c - bm[:, None] @ a[self.minus].conj() @ _dagger(self.b0)[:, None])
            bx = .5 * (b - self.b0[:, None] @ d[self.minus].conj() @ _dagger(bp)[:, None])
            direct_diagnostics = {
                "g_xc_relative_norm": [float(np.linalg.norm(ax) / max(np.linalg.norm(a), np.finfo(float).tiny)),
                                       float(np.linalg.norm(cx) / max(np.linalg.norm(c), np.finfo(float).tiny))],
                "g_xc_hermitian_pair_relative": [_relative(ax, _dagger(bx)), _relative(cx, _dagger(dx))],
            }
            transform = self.frame_interpolator.transform_vertex
            retarded_direct[0] = self._direct(
                q, transform(ax, self.base, plus, perturbation_positions=self.positions),
                transform(dx, minus, self.base, perturbation_positions=self.positions),
            )
            retarded_direct[1] = self._direct(
                -q, transform(cx, self.base, minus, perturbation_positions=self.positions),
                transform(bx, plus, self.base, perturbation_positions=self.positions),
            )
        shape = (2, len(self.coarse_frame.local_frames), 2, self.meta.nat, 3)
        bubble = np.asarray([finalize_q_pair(retarded_bubble[0], retarded_bubble[1]),
                             finalize_q_pair(retarded_bubble[1], retarded_bubble[0])]).reshape(shape)
        direct = np.asarray([finalize_q_pair(retarded_direct[0], retarded_direct[1]),
                             finalize_q_pair(retarded_direct[1], retarded_direct[0])]).reshape(shape)
        total = bubble + direct
        if not np.all(np.isfinite(total)):
            raise ValueError("nonfinite interpolated response")
        return {
            "qpoints": np.asarray([q, -q]), "bubble": bubble, "direct": direct, "total": total,
            "retarded_bubble": retarded_bubble, "retarded_direct": retarded_direct,
            "diagnostics": {
                "g_reciprocity_relative": reciprocity,
                "nominal_g_reciprocity_tolerance": 1.e-5,
                "nominal_g_reciprocity_gate_passed": max(reciprocity) <= 1.e-5,
                "q_pair_kernel_residual_eV_per_angstrom": float(np.max(abs(total[1] - total[0].conj()))),
                "plus_frame": plus.diagnostics, "minus_frame": minus.diagnostics,
                "direct": direct_diagnostics,
                "vertex_symmetry": symmetry_diagnostics,
            },
        }


__all__ = ["ArbitraryQResponse"]
