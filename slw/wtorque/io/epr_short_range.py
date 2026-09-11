"""Short-range EPR interpolation plans.

The EPR writer stores one phonon Wigner--Seitz cell for ``(atom, bra)``.
That choice is sufficient for a single Fourier transform, but it is not tied
to the second electronic centre.  For a real-space matrix element

``C_ij(Re, Rp) = <0,i|dV(Rp)|Re,j>``

the Hermitian real-space relation is

``C_ij(Re, Rp) = conj(C_ji(-Re, Rp-Re))``.

``TwoCenterShortRangePlan`` changes only the finite-q phonon interpolation.
For every electron image it selects phonon images by minimizing the summed
distance to both electronic centres.  The source p-axis is first collapsed by
``Rp mod qmesh`` and that residue total is distributed uniformly over the new
images.  Consequently every q on the stored coarse mesh is reproduced exactly
while off-grid q values use the two-centre geometry.

The plan consumes groups already loaded by :class:`DenseEPREvaluator`; it does
not reopen the HDF5 payload or duplicate its cached coefficients.  Values are
returned in the native stored coefficient units.  The caller applies the same
energy/displacement conversion used by the source evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from slw.core.qe2pert_ws import WSCell, VectorImages, init_rvec_images


_TWO_PI = 2.0 * np.pi
_DEFAULT_EPS = 1.0e-6


def _points(value: object, name: str) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError(f"{name} must have finite shape (n,3)")
    return points


def _qpoint(value: object) -> np.ndarray:
    point = np.asarray(value, dtype=np.float64)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError("qpoint must be a finite reduced three-vector")
    return point


def _mesh(value: object, name: str) -> tuple[int, int, int]:
    array = np.asarray(value)
    if array.shape != (3,) or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must contain three positive integers")
    result = tuple(int(item) for item in array)
    if any(item <= 0 for item in result):
        raise ValueError(f"{name} must contain three positive integers")
    return result


def _residue_indices(vectors: object, mesh: tuple[int, int, int]) -> np.ndarray:
    values = np.asarray(vectors)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("real-space vectors must have shape (n,3)")
    if not np.issubdtype(values.dtype, np.integer):
        rounded = np.rint(values)
        if not np.allclose(values, rounded, atol=0.0, rtol=0.0):
            raise ValueError("real-space vectors must be integral")
        values = rounded.astype(np.int64)
    else:
        values = values.astype(np.int64, copy=False)
    dims = np.asarray(mesh, dtype=np.int64)
    reduced = np.mod(values, dims[None, :])
    # ``init_rvec_images`` follows qe2pert's Fortran order: the first mesh
    # coordinate is fastest, then the second, then the third.
    return (reduced[:, 0] + dims[0] * reduced[:, 1] + dims[0] * dims[1] * reduced[:, 2]).astype(
        np.int64,
        copy=False,
    )


def _candidate_images(
    mesh: tuple[int, int, int],
    lattice: np.ndarray,
    candidate_images: VectorImages | None,
) -> VectorImages:
    """Return a finite candidate set for two-centre p-cell selection.

    The large qe2pert image sphere is used by default.  Its extra margin is
    needed because the partner image ``Rp-Re`` can be outside the source
    small-cutoff sphere even though both tied minima are local.  Closure is
    checked after construction; callers can pass an explicitly larger image
    set when a custom mesh needs it.
    """

    if candidate_images is None:
        return init_rvec_images(mesh, lattice, large_cutoff=True)
    if not isinstance(candidate_images, VectorImages):
        raise TypeError("candidate_images must be a qe2pert VectorImages object")
    if candidate_images.mesh != mesh:
        raise ValueError(
            f"candidate_images mesh {candidate_images.mesh} disagrees with qmesh {mesh}"
        )
    if candidate_images.vec.ndim != 2 or candidate_images.vec.shape[1] != 3:
        raise ValueError("candidate_images.vec must have shape (n,3)")
    if not np.issubdtype(candidate_images.vec.dtype, np.integer):
        raise ValueError("candidate_images.vec must contain integer vectors")
    if candidate_images.vec.shape[0] < 1:
        raise ValueError("candidate_images.vec must be nonempty")
    if not np.issubdtype(candidate_images.idx.dtype, np.integer) or not np.issubdtype(
        candidate_images.nim.dtype, np.integer
    ):
        raise ValueError("candidate_images.idx/nim must contain integers")
    if candidate_images.idx.shape != (int(np.prod(mesh)),):
        raise ValueError("candidate_images.idx has an invalid shape")
    if candidate_images.nim.shape != candidate_images.idx.shape:
        raise ValueError("candidate_images.nim has an invalid shape")
    if np.any(candidate_images.nim < 1):
        raise ValueError("candidate_images.nim must be positive")
    if int(candidate_images.vec.shape[0]) != int(np.sum(candidate_images.nim)):
        raise ValueError("candidate_images.vec is inconsistent with idx/nim")
    starts = candidate_images.idx.astype(np.int64, copy=False)
    counts = candidate_images.nim.astype(np.int64, copy=False)
    if (
        int(starts[0]) != 0
        or np.any(starts[1:] != starts[:-1] + counts[:-1])
        or int(starts[-1] + counts[-1]) != int(candidate_images.vec.shape[0])
    ):
        raise ValueError("candidate_images.idx/nim must describe contiguous residue blocks")
    residue_ids = np.arange(int(np.prod(mesh)), dtype=np.int64)
    residues = np.column_stack(
        (
            residue_ids % mesh[0],
            (residue_ids // mesh[0]) % mesh[1],
            residue_ids // (mesh[0] * mesh[1]),
        )
    )
    vectors = candidate_images.vec.astype(np.int64, copy=False)
    for residue, base in enumerate(residues):
        start = int(starts[residue])
        stop = start + int(counts[residue])
        if np.any(np.mod(vectors[start:stop] - base[None, :], np.asarray(mesh)) != 0):
            raise ValueError("candidate_images vectors disagree with their residue addresses")
    return candidate_images


def _select_tied_images(
    electron_vectors: np.ndarray,
    *,
    lattice: np.ndarray,
    atom_position: np.ndarray,
    bra_center: np.ndarray,
    ket_center: np.ndarray,
    candidates: VectorImages,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Select tied p images for every electron vector.

    The result is a flattened candidate-index array and an ``(nRe,nResidue)``
    count array.  Candidate indices are ordered by electron vector and then by
    q-mesh residue, matching the source image ordering.
    """

    re = np.asarray(electron_vectors, dtype=np.int64)
    if re.ndim != 2 or re.shape[1] != 3:
        raise ValueError("electron_vectors must have shape (n,3)")
    mesh = candidates.mesh
    nresidue = int(np.prod(mesh))
    target = (re.astype(np.float64) + ket_center[None, :]) @ lattice.T
    candidate_cart = (candidates.vec.astype(np.float64) + atom_position[None, :]) @ lattice.T
    first_distance = np.linalg.norm(
        candidate_cart - (bra_center @ lattice.T)[None, :], axis=1
    )

    selected_by_residue: list[np.ndarray] = []
    rows_by_residue: list[np.ndarray] = []
    counts = np.empty((len(re), nresidue), dtype=np.int64)
    for residue in range(nresidue):
        start = int(candidates.idx[residue])
        stop = start + int(candidates.nim[residue])
        cart = candidate_cart[start:stop]
        score = np.linalg.norm(target[:, None, :] - cart[None, :, :], axis=2)
        score += first_distance[start:stop][None, :]
        minima = np.min(score, axis=1)
        # ``flatnonzero`` scans this C-order mask by Re and then candidate,
        # retaining qe2pert's candidate order within every tied row.  Keeping
        # the rows explicitly lets us perform the residue-major to Re-major
        # transpose below without a Python loop over every electron image.
        ties = score - minima[:, None] < float(eps)
        row_counts = np.count_nonzero(ties, axis=1).astype(np.int64, copy=False)
        if np.any(row_counts < 1):
            raise RuntimeError("two-centre WS selection produced no image")
        counts[:, residue] = row_counts
        selected = np.flatnonzero(ties)
        local_count = int(stop - start)
        selected_by_residue.append((selected % local_count + start).astype(np.int64, copy=False))
        rows_by_residue.append((selected // local_count).astype(np.int64, copy=False))

    # The arrays above are residue-major.  A stable sort by (Re, residue)
    # produces the exact Re-major layout expected by ``_PairPlan`` while
    # preserving candidate order for multiple ties in one row.
    flat_residue = np.concatenate(selected_by_residue)
    flat_rows = np.concatenate(rows_by_residue)
    flat_residues = np.concatenate(
        [np.full(len(values), residue, dtype=np.int64) for residue, values in enumerate(selected_by_residue)]
    )
    order = np.argsort(flat_rows * nresidue + flat_residues, kind="stable")
    flat = flat_residue[order].astype(np.uint32, copy=False)
    if candidates.vec.shape[0] < np.iinfo(np.uint16).max:
        flat = flat.astype(np.uint16, copy=False)
    return flat, counts


@dataclass(frozen=True)
class _PairPlan:
    atom: int
    bra: int
    ket: int
    electron_vectors: np.ndarray
    source_residues: np.ndarray
    tied_indices: np.ndarray
    tied_counts: np.ndarray

    @property
    def nre(self) -> int:
        return int(self.electron_vectors.shape[0])

    @property
    def nresidue(self) -> int:
        return int(self.tied_counts.shape[1])

    def phase_weights(
        self,
        qpoint: np.ndarray,
        candidate_vectors: np.ndarray,
        candidate_phase: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return ``W[Re,Rp]`` that replaces the source p phase."""

        if candidate_phase is None:
            candidate_phase = np.exp(_TWO_PI * 1j * (candidate_vectors @ qpoint))
        qphase = candidate_phase[self.tied_indices]
        counts = self.tied_counts.reshape(-1).astype(np.float64)
        starts = np.cumsum(counts, dtype=np.int64) - counts.astype(np.int64)
        averages = np.add.reduceat(qphase, starts) / counts
        averages = averages.reshape(self.nre, self.nresidue)
        return averages[:, self.source_residues]


@dataclass(frozen=True)
class _GroupPlan:
    group: Any
    pairs: tuple[_PairPlan, ...]


class TwoCenterShortRangePlan:
    """Evaluate EPR short-range coefficients with tied two-centre p cells.

    Parameters
    ----------
    backend:
        A loaded ``DenseEPREvaluator``-compatible object.  It must expose
        ``metadata`` and the already loaded ``_ggroups`` list.
    candidate_images:
        Optional finite q-mesh image set.  The default uses the large qe2pert
        image sphere and verifies reciprocal closure when both swapped blocks
        are present.
    eps:
        WS tie tolerance in reduced-cell distance units, matching qe2pert's
        ``eps6`` default.
    require_reciprocal_closure:
        If true, reject a finite candidate set that cannot map every selected
        ``(Re,Rp)`` image to ``(-Re,Rp-Re)`` in the swapped block.
    """

    def __init__(
        self,
        backend: Any,
        *,
        candidate_images: VectorImages | None = None,
        eps: float = _DEFAULT_EPS,
        require_reciprocal_closure: bool = True,
    ) -> None:
        if not np.isfinite(eps) or float(eps) <= 0.0:
            raise ValueError("eps must be finite and positive")
        self._eps = float(eps)
        if not hasattr(backend, "metadata") or not hasattr(backend, "_ggroups"):
            raise TypeError("backend must expose metadata and loaded _ggroups")
        metadata = backend.metadata
        self.metadata = metadata
        self.nat = int(metadata.nat)
        self.nwan = int(metadata.nwan)
        self.nq_grid = _mesh(metadata.nq_grid, "metadata.nq_grid")
        self.lattice = np.asarray(metadata.at, dtype=np.float64)
        self.centres = np.asarray(metadata.wc, dtype=np.float64)
        self.tau = np.asarray(metadata.tau, dtype=np.float64)
        if self.lattice.shape != (3, 3) or not np.all(np.isfinite(self.lattice)):
            raise ValueError("metadata.at must be finite with shape (3,3)")
        if abs(float(np.linalg.det(self.lattice))) <= np.finfo(float).eps:
            raise ValueError("metadata.at must be nonsingular")
        if self.centres.shape != (self.nwan, 3) or not np.all(np.isfinite(self.centres)):
            raise ValueError("metadata.wc must have finite shape (nwan,3)")
        if self.tau.shape != (self.nat, 3) or not np.all(np.isfinite(self.tau)):
            raise ValueError("metadata.tau must have finite shape (nat,3)")
        if not np.isfinite(self.nat) or self.nat < 1 or self.nwan < 1:
            raise ValueError("metadata nat and nwan must be positive")
        self.candidates = _candidate_images(self.nq_grid, self.lattice, candidate_images)
        self._candidate_vectors = np.asarray(self.candidates.vec, dtype=np.int64)
        self._geometry_cache: dict[tuple[object, ...], tuple[np.ndarray, np.ndarray]] = {}
        self._geometry_cache_hits = 0
        self._geometry_cache_misses = 0
        self._groups = self._build_groups(backend._ggroups, self._eps)
        if not self._groups:
            raise ValueError("backend has no loaded EPR electron-phonon groups")
        self._pair_lookup_cache = self._pair_lookup()
        if require_reciprocal_closure:
            closure = self._check_reciprocal_closure()
            if closure["misses"] or closure["missing_reverse_images"]:
                missing = closure["missing_reverse_images"]
                misses = closure["misses"]
                raise ValueError(
                    "two-centre candidate images do not close under "
                    f"(Re,Rp)->(-Re,Rp-Re): {misses} image misses and "
                    f"{missing} missing swapped electron images; "
                    "load swapped pairs or pass a larger candidate_images set"
                )
        else:
            closure = self._closure_diagnostics()
        self.diagnostics = {
            "mode": "two_center",
            "candidate_image_count": int(len(self._candidate_vectors)),
            "candidate_image_policy": "large_qe2pert_sphere"
            if candidate_images is None
            else "caller_supplied",
            "ws_eps": self._eps,
            "eph_groups": len(self._groups),
            "eph_pairs": int(sum(len(group.pairs) for group in self._groups)),
            "tied_image_count": int(
                sum(int(pair.tied_counts.sum()) for group in self._groups for pair in group.pairs)
            ),
            "coarse_q_policy": "exact_residue_preservation",
            "reciprocal_candidate_closure": closure,
            "units": "native_stored_coefficients",
            "geometry_cache": {
                "requests": int(self._geometry_cache_hits + self._geometry_cache_misses),
                "hits": int(self._geometry_cache_hits),
                "misses": int(self._geometry_cache_misses),
                "unique_geometries": int(len(self._geometry_cache)),
            },
        }

    def _build_groups(self, groups: Iterable[Any], eps: float) -> list[_GroupPlan]:
        result: list[_GroupPlan] = []
        for group in groups:
            if group.values is None:
                raise ValueError("two-centre plan requires loaded EPR group values")
            electron = np.asarray(group.electron_vectors)
            phonon = np.asarray(group.phonon_vectors)
            pairs = np.asarray(group.pairs)
            values = group.values
            if electron.ndim != 2 or electron.shape[1] != 3:
                raise ValueError("EPR group electron_vectors must have shape (n,3)")
            if phonon.ndim != 2 or phonon.shape[1] != 3:
                raise ValueError("EPR group phonon_vectors must have shape (n,3)")
            if pairs.ndim != 2 or pairs.shape[1] != 3:
                raise ValueError("EPR group pairs must have shape (n,3)")
            expected = (len(electron), 3 * len(pairs), len(phonon))
            if tuple(values.shape) != expected:
                raise ValueError(f"EPR group values shape {values.shape} disagrees with {expected}")
            pair_plans: list[_PairPlan] = []
            for index, raw_pair in enumerate(pairs):
                atom, bra, ket = (int(value) for value in raw_pair)
                if not (0 <= atom < self.nat and 0 <= bra < self.nwan and 0 <= ket < self.nwan):
                    raise ValueError(f"EPR pair {(atom, bra, ket)} is outside metadata bounds")
                # Tied image geometry depends on the electron-image array and
                # the three real-space centres, but not on the coefficient
                # values.  Duplicate Wannier centres and repeated group
                # layouts are common in dense EPR payloads, so reuse the
                # immutable selection arrays across those pairs.
                geometry_key = (
                    electron.shape,
                    electron.tobytes(),
                    self.tau[atom].tobytes(),
                    self.centres[bra].tobytes(),
                    self.centres[ket].tobytes(),
                )
                cached = self._geometry_cache.get(geometry_key)
                if cached is None:
                    tied, counts = _select_tied_images(
                        electron,
                        lattice=self.lattice,
                        atom_position=self.tau[atom],
                        bra_center=self.centres[bra],
                        ket_center=self.centres[ket],
                        candidates=self.candidates,
                        eps=eps,
                    )
                    self._geometry_cache[geometry_key] = (tied, counts)
                    self._geometry_cache_misses += 1
                else:
                    tied, counts = cached
                    self._geometry_cache_hits += 1
                pair_plans.append(
                    _PairPlan(
                        atom=atom,
                        bra=bra,
                        ket=ket,
                        electron_vectors=electron,
                        source_residues=_residue_indices(phonon, self.nq_grid),
                        tied_indices=tied,
                        tied_counts=counts,
                    )
                )
            result.append(_GroupPlan(group=group, pairs=tuple(pair_plans)))
        return result

    def _pair_lookup(self) -> dict[tuple[int, int, int], _PairPlan]:
        result: dict[tuple[int, int, int], _PairPlan] = {}
        for group in self._groups:
            for pair in group.pairs:
                key = (pair.atom, pair.bra, pair.ket)
                if key in result:
                    raise ValueError(f"duplicate EPR pair {key} in loaded groups")
                result[key] = pair
        return result

    def _closure_diagnostics(self) -> dict[str, Any]:
        lookup = self._pair_lookup_cache
        checked = 0
        missing_reverse = 0
        misses = 0
        for pair in lookup.values():
            reverse = lookup.get((pair.atom, pair.ket, pair.bra))
            if reverse is None:
                missing_reverse += pair.nre
                continue
            reverse_rows = {tuple(vector): index for index, vector in enumerate(reverse.electron_vectors.tolist())}
            for row, vector in enumerate(pair.electron_vectors.tolist()):
                reverse_row = reverse_rows.get(tuple(-np.asarray(vector, dtype=np.int64)))
                if reverse_row is None:
                    misses += 1
                    continue
                checked += 1
                forward_start = int(np.sum(pair.tied_counts[:row]))
                forward_stop = forward_start + int(pair.tied_counts[row].sum())
                reverse_start = int(np.sum(reverse.tied_counts[:reverse_row]))
                reverse_stop = reverse_start + int(reverse.tied_counts[reverse_row].sum())
                forward = self._candidate_vectors[pair.tied_indices[forward_start:forward_stop]]
                expected = np.asarray(
                    sorted(tuple(vector) for vector in (forward - np.asarray(vector, dtype=np.int64))),
                    dtype=np.int64,
                )
                actual = self._candidate_vectors[reverse.tied_indices[reverse_start:reverse_stop]]
                actual = np.asarray(sorted(tuple(vector) for vector in actual), dtype=np.int64)
                if expected.shape != actual.shape or not np.array_equal(expected, actual):
                    misses += 1
        return {
            "checked_electron_images": int(checked),
            "missing_reverse_images": int(missing_reverse),
            "misses": int(misses),
            "checked": bool(checked > 0),
        }

    def _check_reciprocal_closure(self) -> dict[str, Any]:
        return self._closure_diagnostics()

    def phase_weights(self, atom: int, bra: int, ket: int, electron_index: int, qpoint: object) -> np.ndarray:
        """Expose one source-axis replacement weight for diagnostics/tests."""

        q = _qpoint(qpoint)
        pair = self._pair_lookup_cache.get((int(atom), int(bra), int(ket)))
        if pair is None:
            raise KeyError(f"unknown EPR pair {(atom, bra, ket)}")
        index = int(electron_index)
        if index < 0 or index >= pair.nre:
            raise IndexError(f"electron_index {index} is outside 0..{pair.nre - 1}")
        candidate_phase = np.exp(_TWO_PI * 1j * (self._candidate_vectors @ q))
        return pair.phase_weights(q, self._candidate_vectors, candidate_phase)[index].copy()

    def diagonal_ws_cells(self) -> dict[tuple[int, int], WSCell]:
        """Return the tied two-centre q cells for every ``(atom, Re=0)``.

        The polar long-range resplitting needs the same Re=0 phonon image
        geometry as the short-range plan.  In particular, the two-centre
        score is evaluated with the configured ``eps``; it is not replaced by
        a single-centre WS call, since near ties can then select a different
        degeneracy.  A diagonal EPR block supplies the actual Re=0 row.  If a
        caller loaded only a subset of blocks, the Re=0 geometry is built from
        metadata and no coefficient data are read.
        """

        result: dict[tuple[int, int], WSCell] = {}
        for atom in range(self.nat):
            for wannier in range(self.nwan):
                pair = self._pair_lookup_cache.get((atom, wannier, wannier))
                if pair is None:
                    electron = np.zeros((1, 3), dtype=np.int64)
                    tied, counts = _select_tied_images(
                        electron,
                        lattice=self.lattice,
                        atom_position=self.tau[atom],
                        bra_center=self.centres[wannier],
                        ket_center=self.centres[wannier],
                        candidates=self.candidates,
                        eps=self._eps,
                    )
                    row = 0
                else:
                    zero = np.flatnonzero(np.all(pair.electron_vectors == 0, axis=1))
                    if zero.size != 1:
                        raise ValueError(
                            f"diagonal EPR pair {(atom, wannier, wannier)} must contain one Re=0 row"
                        )
                    row = int(zero[0])
                    tied = pair.tied_indices
                    counts = pair.tied_counts
                row_start = int(np.sum(counts[:row]))
                row_counts = counts[row].astype(np.int64, copy=True)
                row_stop = row_start + int(row_counts.sum())
                indices = np.asarray(tied[row_start:row_stop], dtype=np.int64)
                if indices.size != int(row_counts.sum()):
                    raise RuntimeError("diagonal two-centre WS row bookkeeping failed")
                result[(atom, wannier)] = WSCell(
                    vectors=self._candidate_vectors[indices].copy(),
                    ndeg=np.repeat(row_counts, row_counts),
                    raw_indices=indices.copy(),
                )
        return result

    def evaluate(self, kpoints: object, qpoint: object) -> np.ndarray:
        """Return two-centre SR ``g`` in native stored coefficient units.

        The output shape is ``(nk, 3*nat, nwan, nwan)``.  No long-range term or
        energy/displacement conversion is included.
        """

        k = _points(kpoints, "kpoints")
        q = _qpoint(qpoint)
        # Every pair draws phases from the same finite candidate image set.
        # Compute one phase per candidate once per q instead of once for every
        # tied (Re,Rp) entry.
        candidate_phase = np.exp(_TWO_PI * 1j * (self._candidate_vectors @ q))
        output = np.zeros((len(k), 3 * self.nat, self.nwan, self.nwan), dtype=np.complex128)
        phase_cache: dict[bytes, np.ndarray] = {}
        for plan in self._groups:
            group = plan.group
            key = np.asarray(group.electron_vectors).tobytes()
            phases = phase_cache.get(key)
            if phases is None:
                phases = np.exp(_TWO_PI * 1j * (k @ np.asarray(group.electron_vectors).T))
                phase_cache[key] = phases
            for index, pair in enumerate(plan.pairs):
                weights = pair.phase_weights(q, self._candidate_vectors, candidate_phase)
                source = np.asarray(group.values[:, 3 * index : 3 * index + 3, :])
                phonon_sum = np.einsum("rp,rcp->rc", weights, source, optimize=True)
                block = phases @ phonon_sum
                output[:, 3 * pair.atom : 3 * pair.atom + 3, pair.bra, pair.ket] = block
        return output

__all__ = ["TwoCenterShortRangePlan"]
