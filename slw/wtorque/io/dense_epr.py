"""Cached qe2pert H and Cartesian g at arbitrary reduced k and q.

The default reproduces Perturbo's pair-specific Wigner--Seitz interpolation
and source 3D dipole add-back. Stored coefficients contain their WS weights.
Independent options replace phonon images using both electronic endpoints
and replace the identity LR by a point-Wannier-center LR with consistent
coarse re-splitting. Transform both electronic endpoints only after summing
SR and LR in the original Wannier basis.

The evaluator itself is rank-local.  Callers distribute q points with MPI;
``cache_dir`` shares read-only, memory-mapped coefficient files between ranks
on one filesystem instead of keeping one 1--2 GB heap copy per rank.
"""

from __future__ import annotations

from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.constants import physical_constants

from slw.core.constants import RY_TO_EV
from slw.core.qe2pert_ws import (
    init_rvec_images, set_wigner_seitz_cell, triangular_pair_index,
)
from slw.epc.epr_io import energy_scale_to_ev, read_epr_metadata
from slw.epc.epr_phonon import read_epr_phonon_metadata
from slw.wtorque.io.epr_longrange import (
    PolarLongRange3D, electronic_longrange_3d as _source_longrange_3d,
)
from slw.wtorque.io.epr_short_range import TwoCenterShortRangePlan

_CACHE_VERSION = 1
_BOHR_ANG = physical_constants["Bohr radius"][0] / 1.e-10


def _points(value: object, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must have finite shape (n,3)")
    return array


def _qpoint(value: object) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError("qpoint must be a finite reduced three-vector")
    return array


def electronic_longrange_3d(meta: dict[str, Any], qpoint: object) -> np.ndarray:
    """Source Perturbo 3D dipole scalar, Ry/bohr, with unchanged q policy."""
    return _source_longrange_3d(meta, qpoint)


@dataclass
class _HGroup:
    vectors: np.ndarray
    pairs: np.ndarray
    values: np.ndarray


@dataclass
class _GGroup:
    electron_vectors: np.ndarray
    phonon_vectors: np.ndarray
    # Each row is atom, bra, ket (all zero based).
    pairs: np.ndarray
    # [nRe, npair*3, nRp], so both contractions are contiguous BLAS calls.
    values: np.ndarray | None = None


class DenseEPREvaluator:
    """Load WS coefficients once and interpolate in the original cell gauge.

    ``evaluate_h(k)`` returns ``[nk,nw,nw]`` eV; ``evaluate_g(k,q)`` returns
    ``[nk,3*nat,nw,nw]`` eV/Angstrom.  k need not lie on a uniform grid or in
    the first BZ.  General-q g is never made Hermitian or averaged here.

    The short-range sum is periodic in both k and q, but the source's fixed
    finite G box in the polar correction is not exactly q+G periodic.  We
    preserve the supplied continuous q representatives, including their -q
    partners, and the exact source Gamma convention.  Folding q independently
    would change the source model and can introduce boundary discontinuities.
    This finite-sum limitation is recorded explicitly in ``diagnostics``.

    A cache is keyed by absolute source path, size, mtime, inode, and format
    version.  Files are atomically installed under a process lock; a partial
    build cannot be mistaken for a completed cache.  Immutable EPR inputs are
    expected, as in the rest of the native workflow.
    """

    def __init__(
        self, epr_path: str | Path, *, energy_unit: str, displacement_unit: str,
        expected_spinor: bool = True, cache_dir: str | Path | None = None,
        longrange_model: str = "source", short_range_model: str = "source",
        longrange_coarse_qpoints: object | None = None,
    ) -> None:
        for name, value, allowed in (
            ("longrange_model", longrange_model, {"source", "point_center"}),
            ("short_range_model", short_range_model, {"source", "two_center"}),
        ):
            if not isinstance(value, str) or value not in allowed:
                raise ValueError(f"{name} must be one of {sorted(allowed)}")
        self.longrange_model = longrange_model
        self.short_range_model = short_range_model
        self._short_range_plan: TwoCenterShortRangePlan | None = None
        if longrange_model == "point_center" and longrange_coarse_qpoints is None:
            raise ValueError("point_center requires longrange_coarse_qpoints with the actual source q representatives")
        if longrange_model == "source" and longrange_coarse_qpoints is not None:
            raise ValueError("longrange_coarse_qpoints requires longrange_model=point_center")
        self._polar_model: PolarLongRange3D | None = None
        self.path = Path(epr_path).expanduser().resolve()
        self.energy_scale = float(energy_scale_to_ev(energy_unit))
        if displacement_unit == "bohr":
            self.derivative_scale = self.energy_scale / _BOHR_ANG
        elif displacement_unit == "angstrom":
            self.derivative_scale = self.energy_scale
        else:
            raise ValueError("displacement_unit must be bohr or angstrom")
        self.polar_metadata: dict[str, Any] | None = None
        self._hgroups: list[_HGroup] = []
        self._ggroups: list[_GGroup] = []
        with h5py.File(self.path, "r") as handle:
            self.metadata = read_epr_metadata(handle)
            if "basic_data/spinor" not in handle:
                raise KeyError("EPR requires basic_data/spinor")
            self.spinor = bool(handle["basic_data/spinor"][()])
            if self.spinor != bool(expected_spinor):
                raise ValueError("EPR spinor declaration disagrees with expected_spinor")
            basic = handle["basic_data"]
            lquad = "qtensor" in basic or (bool(basic["lquad"][()]) if "lquad" in basic else False)
            lpolar = bool(basic["lpolar"][()]) if "lpolar" in basic else False
            if longrange_model == "point_center" and not lpolar:
                raise ValueError("longrange_model=point_center requires a polar EPR (lpolar=true)")
            if lquad:
                raise NotImplementedError("native dense EPR quadrupole add-back is not implemented")
            if lpolar:
                self.polar_metadata = read_epr_phonon_metadata(self.path)
                self.polar_metadata["polar_alpha"] = float(basic["polar_alpha"][()]) if "polar_alpha" in basic else 1.
                self.polar_metadata["zstar"] = self.polar_metadata["zstar"].swapaxes(-1, -2)
                if self.polar_metadata["system_2d"]:
                    raise NotImplementedError("native dense EPR 2D polar add-back is not implemented")
            self._build_groups(handle)
            if cache_dir is None:
                self._load_g(handle)
                self.cache_path = None
            else:
                self._load_cached_g(handle, Path(cache_dir).expanduser().resolve())
        if short_range_model == "two_center":
            self._short_range_plan = TwoCenterShortRangePlan(self)
        if self.polar_metadata is not None and longrange_model == "point_center":
            self._polar_model = PolarLongRange3D(
                self.polar_metadata, self.metadata.wc, coarse_qpoints=longrange_coarse_qpoints,
                at=self.metadata.at, tau=self.metadata.tau,
                ws_cells=(self._short_range_plan.diagonal_ws_cells()
                          if self._short_range_plan is not None else None),
            )
        self.pert_atom = np.repeat(np.arange(self.metadata.nat, dtype=np.int32), 3)
        self.pert_cart = np.tile(np.arange(3, dtype=np.int32), self.metadata.nat)
        self.diagnostics = {
            "source": str(self.path), "coefficient_bytes": int(sum(g.values.nbytes for g in self._ggroups)),
            "electron_ws_groups": len(self._hgroups), "eph_ws_groups": len(self._ggroups),
            "eph_pairs": sum(len(g.pairs) for g in self._ggroups),
            "cache_path": None if self.cache_path is None else str(self.cache_path),
            "longrange": "perturbo_3d_dipole" if self.polar_metadata is not None else "none",
            "longrange_model": longrange_model, "short_range_model": short_range_model,
            "short_range_geometry": self._short_range_plan.diagnostics if self._short_range_plan is not None else None,
            "longrange_resplitting": self._polar_model is not None,
            "longrange_resplitting_geometry": short_range_model if self._polar_model is not None else None,
            "longrange_coarse_qpoints": (
                np.asarray(longrange_coarse_qpoints, float).tolist()
                if self._polar_model is not None else None
            ),
            "electronic_form_factor": (
                "diagonal point-Wannier-center approximation; finite-size/off-diagonal overlaps omitted"
                if self._polar_model is not None else "source Wannier identity"
            ),
            "longrange_q_policy": "source_representative_without_folding",
            "longrange_exact_reciprocal_periodicity": self.polar_metadata is None,
            "longrange_limitation": (
                "source fixed finite G box; retain continuous q representatives and exact minus partners; "
                "no independent q folding or converged-G periodic completion"
                if self.polar_metadata is not None else None
            ),
            "ws_degeneracy": "already_in_coefficients", "g_units": "eV/angstrom", "h_units": "eV",
        }

    def _build_groups(self, handle: h5py.File) -> None:
        m = self.metadata
        eimages = init_rvec_images(m.nk_grid, m.at)
        pimages = init_rvec_images(m.nq_grid, m.at)
        ews = {
            (i, j): set_wigner_seitz_cell(eimages, m.at, m.wc[i], m.wc[j])
            for i in range(m.nwan) for j in range(m.nwan)
        }
        pws = {
            (a, i): set_wigner_seitz_cell(pimages, m.at, m.wc[i], m.tau[a])
            for a in range(m.nat) for i in range(m.nwan)
        }
        hsource = handle["electron_wannier"]
        hgroups: dict[bytes, tuple[np.ndarray, list, list]] = {}
        for j in range(m.nwan):
            for i in range(j + 1):
                index = triangular_pair_index(i + 1, j + 1)
                real, imag = f"hopping_r{index}", f"hopping_i{index}"
                if (real in hsource) != (imag in hsource):
                    raise KeyError(f"incomplete complex EPR hopping pair {index}")
                if real not in hsource:
                    continue
                ws = ews[i, j]
                data = hsource[real][()] + 1j * hsource[imag][()]
                if data.shape != (ws.nr,) or not np.all(np.isfinite(data)):
                    raise ValueError(f"invalid EPR hopping pair {index}: WS length or finiteness")
                key = ws.vectors.tobytes()
                entry = hgroups.setdefault(key, (ws.vectors, [], []))
                entry[1].append((i, j)); entry[2].append(data)
        self._hgroups = [_HGroup(v, np.asarray(p), np.asarray(d).T.copy()) for v, p, d in hgroups.values()]
        gsource = handle["eph_matrix_wannier"]
        ggroups: dict[tuple[bytes, bytes], tuple[np.ndarray, np.ndarray, list]] = {}
        for a in range(m.nat):
            for j in range(m.nwan):
                for i in range(m.nwan):
                    real, imag = f"ep_hop_r_{a+1}_{j+1}_{i+1}", f"ep_hop_i_{a+1}_{j+1}_{i+1}"
                    if (real in gsource) != (imag in gsource):
                        raise KeyError(f"incomplete complex EPR g pair {a+1},{j+1},{i+1}")
                    if real not in gsource:
                        continue
                    re, rp = ews[i, j], pws[a, i]
                    shape = (rp.nr, re.nr, 3)
                    if gsource[real].shape != shape or gsource[imag].shape != shape:
                        raise ValueError(f"EPR g WS shape mismatch {real}; expected {shape}")
                    key = re.vectors.tobytes(), rp.vectors.tobytes()
                    entry = ggroups.setdefault(key, (re.vectors, rp.vectors, []))
                    entry[2].append((a, i, j))
        self._ggroups = [_GGroup(re, rp, np.asarray(p)) for re, rp, p in ggroups.values()]
        if not self._hgroups or not self._ggroups:
            raise ValueError("EPR requires nonempty electron and electron-phonon hopping data")

    @staticmethod
    def _fill_group(source: h5py.Group, group: _GGroup, target: np.ndarray) -> None:
        for ipair, (a, i, j) in enumerate(group.pairs):
            stem = f"_{a+1}_{j+1}_{i+1}"
            block = source[f"ep_hop_r{stem}"][()] + 1j * source[f"ep_hop_i{stem}"][()]
            if not np.all(np.isfinite(block)):
                raise ValueError(f"nonfinite EPR g coefficients {stem}")
            target[:, 3*ipair:3*ipair+3, :] = block.transpose(1, 2, 0)

    def _load_g(self, handle: h5py.File) -> None:
        for group in self._ggroups:
            shape = (len(group.electron_vectors), 3*len(group.pairs), len(group.phonon_vectors))
            group.values = np.empty(shape, complex)
            self._fill_group(handle["eph_matrix_wannier"], group, group.values)

    def _load_cached_g(self, handle: h5py.File, directory: Path) -> None:
        stat = self.path.stat()
        identity = (str(self.path), stat.st_size, stat.st_mtime_ns, stat.st_ino, _CACHE_VERSION)
        key = hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:24]
        self.cache_path = directory / ("dense-epr-" + key)
        self.cache_path.mkdir(parents=True, exist_ok=True)
        manifest = self.cache_path / "complete.json"
        with (self.cache_path / "build.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not manifest.is_file():
                for index, group in enumerate(self._ggroups):
                    path = self.cache_path / f"g_{index:05d}.npy"
                    temporary = path.with_suffix(".tmp.npy")
                    shape = (len(group.electron_vectors), 3*len(group.pairs), len(group.phonon_vectors))
                    data = np.lib.format.open_memmap(temporary, mode="w+", dtype=complex, shape=shape)
                    self._fill_group(handle["eph_matrix_wannier"], group, data)
                    data.flush(); del data
                    os.replace(temporary, path)
                temporary_manifest = manifest.with_suffix(".tmp")
                temporary_manifest.write_text(json.dumps({"source_identity": identity, "groups": len(self._ggroups)}))
                os.replace(temporary_manifest, manifest)
            for index, group in enumerate(self._ggroups):
                group.values = np.load(self.cache_path / f"g_{index:05d}.npy", mmap_mode="r")
                shape = (len(group.electron_vectors), 3*len(group.pairs), len(group.phonon_vectors))
                if group.values.shape != shape or group.values.dtype != np.complex128:
                    raise ValueError(f"invalid dense EPR cache group {index}")

    def evaluate_h(self, kpoints: object) -> np.ndarray:
        """Return H(k) in eV in the original periodic Wannier cell basis."""
        k = _points(kpoints, "kpoints")
        values = np.zeros((len(k), self.metadata.nwan, self.metadata.nwan), complex)
        for group in self._hgroups:
            phase = np.exp(2j * np.pi * (k @ group.vectors.T))
            block = (phase @ group.values) * self.energy_scale
            i, j = group.pairs.T
            values[:, i, j] = block
            off = i != j
            values[:, j[off], i[off]] = block[:, off].conj()
        return values

    def longrange(self, qpoint: object) -> np.ndarray:
        """Return the source dipole scalar (legacy API), irrespective of model.

        Use ``longrange_diagonal`` for the selected model's full diagonal.
        Both methods return eV/Angstrom.
        """
        q = _qpoint(qpoint)
        if self.polar_metadata is None:
            return np.zeros(3*self.metadata.nat, complex)
        # The analytical Perturbo kernel is always in Ry/bohr, independently
        # of an explicit alternative unit supplied for stored coefficients.
        return electronic_longrange_3d(self.polar_metadata, q) * (RY_TO_EV / _BOHR_ANG)

    def longrange_diagonal(self, qpoint: object) -> np.ndarray:
        """Selected LR add-back, eV/Angstrom, shape (3*nat,nwan)."""
        q = _qpoint(qpoint)
        if self._polar_model is not None:
            return self._polar_model.center(q) * (RY_TO_EV / _BOHR_ANG)
        return np.broadcast_to(self.longrange(q)[:, None],
                               (3*self.metadata.nat, self.metadata.nwan)).copy()

    def _evaluate_source_short_range(self, k: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Stored SR in the source WS geometry, eV/Angstrom."""
        m = self.metadata
        values = np.zeros((len(k), 3*m.nat, m.nwan, m.nwan), complex)
        phases_k: dict[bytes, np.ndarray] = {}
        phases_q: dict[bytes, np.ndarray] = {}
        for group in self._ggroups:
            ekey, pkey = group.electron_vectors.tobytes(), group.phonon_vectors.tobytes()
            if ekey not in phases_k:
                phases_k[ekey] = np.exp(2j * np.pi * (k @ group.electron_vectors.T))
            if pkey not in phases_q:
                phases_q[pkey] = np.exp(2j * np.pi * (group.phonon_vectors @ q))
            nre, ncoefficient, nrp = group.values.shape
            electron = (group.values.reshape(nre*ncoefficient, nrp) @ phases_q[pkey]).reshape(nre, ncoefficient)
            block = (phases_k[ekey] @ electron).reshape(len(k), len(group.pairs), 3) * self.derivative_scale
            atoms, i, j = group.pairs.T
            perturbations = 3*atoms[:, None] + np.arange(3)[None, :]
            values[:, perturbations, i[:, None], j[:, None]] = block
        return values

    def evaluate_g(self, kpoints: object, qpoint: object, *, include_longrange: bool = True) -> np.ndarray:
        """Selected g(k,q), bra at k+q and ket at k, without q-pair averaging.

        With include_longrange=False the result is the SR remainder for the
        selected LR model, including its consistent coarse subtraction.
        """
        k, q = _points(kpoints, "kpoints"), _qpoint(qpoint)
        m = self.metadata
        if self._short_range_plan is None:
            values = self._evaluate_source_short_range(k, q)
        else:
            values = self._short_range_plan.evaluate(k, q) * self.derivative_scale
        diagonal = np.arange(m.nwan)
        # This correction belongs to SR and remains active when add-back is
        # disabled. Changing fine LR alone would not preserve coarse data.
        if self._polar_model is not None:
            values[:, :, diagonal, diagonal] += (
                self._polar_model.sr_correction(q) * (RY_TO_EV / _BOHR_ANG)
            )[None]
        if include_longrange:
            values[:, :, diagonal, diagonal] += self.longrange_diagonal(q)[None]
        return values


__all__ = ["DenseEPREvaluator", "electronic_longrange_3d"]
