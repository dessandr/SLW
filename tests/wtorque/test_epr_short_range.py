from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from slw.core.qe2pert_ws import VectorImages, init_rvec_images, set_wigner_seitz_cell
from slw.wtorque.io.epr_short_range import TwoCenterShortRangePlan, _residue_indices


def _metadata(*, nwan: int = 2, qmesh: tuple[int, int, int] = (2, 1, 1)):
    centres = np.zeros((nwan, 3), dtype=float)
    if nwan > 1:
        centres[1, 0] = 0.27
    return SimpleNamespace(
        nat=1,
        nwan=nwan,
        nk_grid=(2, 1, 1),
        nq_grid=qmesh,
        at=np.eye(3),
        tau=np.array([[0.13, 0.0, 0.0]]),
        wc=centres,
    )


def _groups(metadata, pair_values: dict[tuple[int, int, int], np.ndarray] | None = None):
    electron_images = init_rvec_images(metadata.nk_grid, metadata.at)
    phonon_images = init_rvec_images(metadata.nq_grid, metadata.at)
    result = []
    rng = np.random.default_rng(2718)
    for atom, bra, ket in ((0, 0, 1), (0, 1, 0)):
        electron = set_wigner_seitz_cell(
            electron_images, metadata.at, metadata.wc[bra], metadata.wc[ket]
        )
        phonon = set_wigner_seitz_cell(
            phonon_images, metadata.at, metadata.wc[bra], metadata.tau[atom]
        )
        values = (
            pair_values[(atom, bra, ket)]
            if pair_values is not None and (atom, bra, ket) in pair_values
            else rng.normal(size=(electron.nr, 3, phonon.nr))
            + 1j * rng.normal(size=(electron.nr, 3, phonon.nr))
        )
        result.append(
            SimpleNamespace(
                electron_vectors=electron.vectors,
                phonon_vectors=phonon.vectors,
                pairs=np.asarray([[atom, bra, ket]], dtype=np.int64),
                values=np.asarray(values, dtype=np.complex128),
            )
        )
    return result


def _residue_vector(index: int, mesh: tuple[int, int, int]) -> np.ndarray:
    n1, n2, _ = mesh
    return np.asarray((index % n1, (index // n1) % n2, index // (n1 * n2)), dtype=int)


def _hermitian_groups(metadata):
    """Make weighted source residues obey C_ij(Re,Rp)=C_ji(-Re,Rp-Re)^*.

    Values in each old p cell are divided by the number of images sharing a
    residue, just as qe2pert's writer divides by ``ndeg_q``.  The plan must
    collapse those weighted values before applying the new tied-cell phases.
    """

    electron_images = init_rvec_images(metadata.nk_grid, metadata.at)
    phonon_images = init_rvec_images(metadata.nq_grid, metadata.at)
    cells = {}
    for atom, bra, ket in ((0, 0, 1), (0, 1, 0)):
        electron = set_wigner_seitz_cell(
            electron_images, metadata.at, metadata.wc[bra], metadata.wc[ket]
        )
        phonon = set_wigner_seitz_cell(
            phonon_images, metadata.at, metadata.wc[bra], metadata.tau[atom]
        )
        cells[(atom, bra, ket)] = (electron, phonon)

    mesh = metadata.nq_grid
    rng = np.random.default_rng(581)
    coefficients: dict[tuple[int, int, tuple[int, int, int], int], np.ndarray] = {}

    def residue(vector):
        reduced = np.mod(np.asarray(vector, dtype=int), np.asarray(mesh))
        return int(reduced[0] + mesh[0] * reduced[1] + mesh[0] * mesh[1] * reduced[2])

    def coefficient(atom, bra, ket, re, source_residue):
        key = (atom, bra, ket, tuple(int(x) for x in re), int(source_residue))
        residue_vector = _residue_vector(int(source_residue), mesh)
        partner = (
            atom,
            ket,
            bra,
            tuple(int(x) for x in -np.asarray(re, dtype=int)),
            residue(residue_vector - np.asarray(re, dtype=int)),
        )
        if key in coefficients:
            return coefficients[key]
        if partner in coefficients:
            value = coefficients[partner].conj()
        elif key == partner:
            value = rng.normal(size=3)
        else:
            value = rng.normal(size=3) + 1j * rng.normal(size=3)
            coefficients[partner] = value.conj()
        coefficients[key] = value
        return value

    groups = []
    for atom, bra, ket in ((0, 0, 1), (0, 1, 0)):
        electron, phonon = cells[(atom, bra, ket)]
        source_residues = _residue_indices(phonon.vectors, mesh)
        multiplicity = np.bincount(source_residues, minlength=int(np.prod(mesh)))
        values = np.empty((electron.nr, 3, phonon.nr), dtype=np.complex128)
        for ire, re in enumerate(electron.vectors):
            for irp, residue_index in enumerate(source_residues):
                values[ire, :, irp] = coefficient(atom, bra, ket, re, residue_index) / multiplicity[residue_index]
        groups.append(
            SimpleNamespace(
                electron_vectors=electron.vectors,
                phonon_vectors=phonon.vectors,
                pairs=np.asarray([[atom, bra, ket]], dtype=np.int64),
                values=values,
            )
        )
    return groups


def _backend(metadata, groups):
    return SimpleNamespace(metadata=metadata, _ggroups=groups)


def _source_evaluate(metadata, groups, kpoints, qpoint):
    k = np.asarray(kpoints, dtype=float)
    q = np.asarray(qpoint, dtype=float)
    result = np.zeros((len(k), 3 * metadata.nat, metadata.nwan, metadata.nwan), complex)
    for group in groups:
        phase_k = np.exp(2j * np.pi * (k @ group.electron_vectors.T))
        for pair_index, (atom, bra, ket) in enumerate(group.pairs):
            source = group.values[:, 3 * pair_index : 3 * pair_index + 3, :]
            phase_q = np.exp(2j * np.pi * (group.phonon_vectors @ q))
            block = phase_k @ np.einsum("rcp,p->rc", source, phase_q)
            result[:, 3 * atom : 3 * atom + 3, bra, ket] = block
    return result


def test_two_center_preserves_physical_hermiticity_off_grid():
    metadata = _metadata()
    plan = TwoCenterShortRangePlan(_backend(metadata, _hermitian_groups(metadata)))
    k = np.asarray([[0.11, 0.0, 0.0], [0.31, 0.0, 0.0]])
    q = np.asarray([0.217, 0.0, 0.0])
    forward = plan.evaluate(k, q)
    reverse = plan.evaluate(k + q, -q)
    np.testing.assert_allclose(
        forward[:, 0:3, 0, 1], reverse[:, 0:3, 1, 0].conj(), atol=2.0e-13
    )
    np.testing.assert_allclose(
        forward[:, 0:3, 1, 0], reverse[:, 0:3, 0, 1].conj(), atol=2.0e-13
    )


def test_two_center_reproduces_every_stored_q_without_projection():
    metadata = _metadata()
    groups = _groups(metadata)
    plan = TwoCenterShortRangePlan(_backend(metadata, groups))
    k = np.asarray([[0.07, 0.0, 0.0], [0.39, 0.0, 0.0]])
    for q in ([0.0, 0.0, 0.0], [0.5, 0.0, 0.0]):
        np.testing.assert_allclose(plan.evaluate(k, q), _source_evaluate(metadata, groups, k, q), atol=3.0e-14)
    # The random source pair is not projected to its swapped partner; only
    # the residue-preserving interpolation changes the off-grid value.
    off_grid = plan.evaluate(k, [0.217, 0.0, 0.0])
    source = _source_evaluate(metadata, groups, k, [0.217, 0.0, 0.0])
    assert np.linalg.norm(off_grid - source) > 1.0e-8


def test_two_center_uses_average_phase_for_alias_degeneracy():
    metadata = _metadata(nwan=1, qmesh=(1, 1, 1))
    metadata.tau = np.array([[0.5, 0.0, 0.0]])
    electron_images = init_rvec_images(metadata.nk_grid, metadata.at)
    phonon_images = init_rvec_images(metadata.nq_grid, metadata.at)
    electron = set_wigner_seitz_cell(electron_images, metadata.at, metadata.wc[0], metadata.wc[0])
    phonon = set_wigner_seitz_cell(phonon_images, metadata.at, metadata.wc[0], [0.5, 0.0, 0.0])
    assert phonon.nr == 2
    values = np.full((electron.nr, 3, phonon.nr), 0.5, complex)
    group = SimpleNamespace(
        electron_vectors=electron.vectors,
        phonon_vectors=phonon.vectors,
        pairs=np.asarray([[0, 0, 0]], dtype=np.int64),
        values=values,
    )
    plan = TwoCenterShortRangePlan(_backend(metadata, [group]))
    q = np.asarray([0.25, 0.0, 0.0])
    expected_weight = np.mean(np.exp(2j * np.pi * phonon.vectors[:, 0] * q[0]))
    weights = plan.phase_weights(0, 0, 0, 0, q)
    np.testing.assert_allclose(weights, expected_weight, atol=2.0e-15)
    got = plan.evaluate([[0.0, 0.0, 0.0]], q)[0, 0, 0, 0]
    np.testing.assert_allclose(got, 3.0 * expected_weight, atol=2.0e-15)


def test_two_center_cached_candidate_phase_matches_direct_tied_phase():
    metadata = _metadata()
    plan = TwoCenterShortRangePlan(_backend(metadata, _groups(metadata)))
    pair = plan._pair_lookup_cache[(0, 0, 1)]
    q = np.asarray([0.217, 0.0, 0.0])
    direct_phase = np.exp(
        2j * np.pi * (plan._candidate_vectors[pair.tied_indices] @ q)
    )
    counts = pair.tied_counts.reshape(-1).astype(float)
    starts = np.cumsum(counts, dtype=np.int64) - counts.astype(np.int64)
    direct = np.add.reduceat(direct_phase, starts) / counts
    direct = direct.reshape(pair.nre, pair.nresidue)[:, pair.source_residues]
    cached_phase = np.exp(2j * np.pi * (plan._candidate_vectors @ q))
    cached = pair.phase_weights(q, plan._candidate_vectors, cached_phase)
    np.testing.assert_allclose(cached, direct, atol=2.0e-14)


def test_diagonal_ws_cells_reuse_two_center_near_tie_selection():
    metadata = _metadata(nwan=1, qmesh=(1, 1, 1))
    metadata.nk_grid = (1, 1, 1)
    metadata.tau = np.array([[0.500000375, 0.0, 0.0]])
    electron_images = init_rvec_images(metadata.nk_grid, metadata.at)
    phonon_images = init_rvec_images(metadata.nq_grid, metadata.at)
    electron = set_wigner_seitz_cell(
        electron_images, metadata.at, metadata.wc[0], metadata.wc[0]
    )
    phonon = set_wigner_seitz_cell(
        phonon_images, metadata.at, metadata.wc[0], metadata.tau[0]
    )
    group = SimpleNamespace(
        electron_vectors=electron.vectors,
        phonon_vectors=phonon.vectors,
        pairs=np.asarray([[0, 0, 0]], dtype=np.int64),
        values=np.zeros((electron.nr, 3, phonon.nr), dtype=np.complex128),
    )
    source_cell = phonon
    plan = TwoCenterShortRangePlan(_backend(metadata, [group]))
    tied_cell = plan.diagonal_ws_cells()[(0, 0)]
    assert source_cell.nr == 2
    assert tied_cell.nr == 1
    np.testing.assert_array_equal(tied_cell.ndeg, np.ones(1, dtype=np.int64))


def test_diagonal_ws_cells_build_missing_diagonal_from_geometry():
    metadata = _metadata()
    plan = TwoCenterShortRangePlan(_backend(metadata, _groups(metadata)))
    cells = plan.diagonal_ws_cells()
    assert set(cells) == {(0, 0), (0, 1)}
    for cell in cells.values():
        assert cell.vectors.shape == (cell.ndeg.size, 3)
        assert cell.raw_indices.shape == cell.ndeg.shape
        assert np.all(cell.ndeg >= 1)


def test_two_center_rejects_insufficient_reciprocal_candidate_set():
    metadata = _metadata()
    groups = _groups(metadata)
    small = VectorImages(
        vec=np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.int64),
        idx=np.asarray([0, 1], dtype=np.int64),
        nim=np.asarray([1, 1], dtype=np.int64),
        mesh=(2, 1, 1),
    )
    with pytest.raises(ValueError, match="do not close"):
        TwoCenterShortRangePlan(_backend(metadata, groups), candidate_images=small)


def test_two_center_rejects_noncontiguous_candidate_metadata():
    metadata = _metadata()
    groups = _groups(metadata)
    malformed = VectorImages(
        vec=np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.int64),
        idx=np.asarray([0, 0], dtype=np.int64),
        nim=np.asarray([1, 1], dtype=np.int64),
        mesh=(2, 1, 1),
    )
    with pytest.raises(ValueError, match="contiguous"):
        TwoCenterShortRangePlan(_backend(metadata, groups), candidate_images=malformed)


def test_two_center_rejects_candidate_with_wrong_residue_address():
    metadata = _metadata()
    groups = _groups(metadata)
    malformed = VectorImages(
        vec=np.asarray([[1, 0, 0], [0, 0, 0]], dtype=np.int64),
        idx=np.asarray([0, 1], dtype=np.int64),
        nim=np.asarray([1, 1], dtype=np.int64),
        mesh=(2, 1, 1),
    )
    with pytest.raises(ValueError, match="residue addresses"):
        TwoCenterShortRangePlan(_backend(metadata, groups), candidate_images=malformed)


def test_two_center_requires_loaded_swapped_pairs_for_closure():
    metadata = _metadata()
    forward_only = _groups(metadata)[:1]
    with pytest.raises(ValueError, match="missing swapped"):
        TwoCenterShortRangePlan(_backend(metadata, forward_only))


def test_two_center_validates_loaded_groups_and_points():
    metadata = _metadata()
    group = _groups(metadata)[0]
    group.values = None
    with pytest.raises(ValueError, match="loaded EPR"):
        TwoCenterShortRangePlan(_backend(metadata, [group]))

    plan = TwoCenterShortRangePlan(_backend(metadata, _groups(metadata)))
    with pytest.raises(ValueError, match="kpoints"):
        plan.evaluate([[np.nan, 0.0, 0.0]], [0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="qpoint"):
        plan.evaluate([[0.0, 0.0, 0.0]], [np.inf, 0.0, 0.0])
