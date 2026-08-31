from __future__ import annotations

import h5py
import numpy as np

from slw.core.qe2pert_ws import init_rvec_images, set_wigner_seitz_cell
from slw.wtorque.green.provider import green_matrix
from slw.wtorque.io.native_epr import parse_projection_indices
from slw.wtorque.io.native_epr_g import reconstruct_epr_g_at_q
from slw.wtorque.model.magnetic_subspace import (
    build_projected_collinear_exchange,
    rotate_projected_exchange,
)
from slw.wtorque.torque.kernel import contract_bubble_batch


def _unitary(rng: np.random.Generator, dimension: int) -> np.ndarray:
    raw = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(
        size=(dimension, dimension)
    )
    q, r = np.linalg.qr(raw)
    diagonal = np.diag(r)
    return np.asarray(q * (diagonal / np.abs(diagonal)).conj(), dtype=np.complex128)


def _projected_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nk = 2
    nw = 6
    magnetic_dimension = 4
    frame = np.zeros((nk, nw, magnetic_dimension), dtype=np.complex128)
    frame[:, :magnetic_dimension] = np.eye(magnetic_dimension, dtype=np.complex128)
    positive = np.diag([1.6, 1.3, 1.1, 0.9]).astype(np.complex128)
    overlap = frame @ positive

    orbital_even = np.array([[0.1, 0.03], [0.03, -0.2]], dtype=np.complex128)
    orbital_exchange = np.array([[1.2, 0.17], [0.17, 0.8]], dtype=np.complex128)
    identity_spin = np.eye(2, dtype=np.complex128)
    sigma_z = np.diag([1.0, -1.0]).astype(np.complex128)
    h_subspace = np.kron(orbital_even, identity_spin) + np.kron(
        orbital_exchange, sigma_z
    )
    hamiltonian = np.zeros((nk, nw, nw), dtype=np.complex128)
    hamiltonian[:, :magnetic_dimension, :magnetic_dimension] = h_subspace
    hamiltonian[:, 4, 4] = 2.1
    hamiltonian[:, 5, 5] = -1.7
    minus = np.array([0, 1], dtype=np.int64)
    return hamiltonian, overlap, minus


def test_projection_index_parser_is_explicit_and_one_based() -> None:
    assert parse_projection_indices("9-12,15,18", 18) == (8, 9, 10, 11, 14, 17)


def test_projected_collinear_exchange_is_gauge_covariant_and_longitudinal_zero() -> (
    None
):
    hamiltonian, overlap, minus = _projected_fixture()
    projected = build_projected_collinear_exchange(
        hamiltonian,
        overlap,
        minus,
        magnetization_direction=(0.0, 0.0, 1.0),
        atomic_spin_order="interleaved",
    )
    assert np.max(np.abs(projected.longitudinal_vertex)) < 1.0e-13
    assert projected.raw_rotation_vertices.shape == (2, 3, 6, 6)
    assert np.max(np.abs(projected.raw_rotation_vertices[:, 2])) < 1.0e-13
    assert np.max(projected.noncollinear_fraction_per_k) < 1.0e-13

    delta = 1.0e-6
    numerical = (
        rotate_projected_exchange(projected, projected.local_frame[0], delta)
        - rotate_projected_exchange(projected, projected.local_frame[0], -delta)
    ) / (2.0 * delta)
    np.testing.assert_allclose(
        numerical,
        -projected.transverse_vertices[:, 1],
        rtol=2.0e-10,
        atol=2.0e-10,
    )

    rng = np.random.default_rng(20260826)
    gauge = np.stack([_unitary(rng, hamiltonian.shape[-1]) for _ in range(2)])
    gauge_dagger = gauge.conj().transpose(0, 2, 1)
    transformed = build_projected_collinear_exchange(
        gauge_dagger @ hamiltonian @ gauge,
        gauge_dagger @ overlap,
        minus,
        magnetization_direction=(0.0, 0.0, 1.0),
        atomic_spin_order="interleaved",
    )
    expected_vertices = (
        gauge_dagger[:, None] @ projected.transverse_vertices @ gauge[:, None]
    )
    np.testing.assert_allclose(
        transformed.transverse_vertices,
        expected_vertices,
        atol=2.0e-12,
    )

    raw_g = rng.normal(size=(2, 1, 6, 6)) + 1j * rng.normal(size=(2, 1, 6, 6))
    perturbation = np.asarray(
        0.5 * (raw_g + raw_g.conj().transpose(0, 1, 3, 2)),
        dtype=np.complex128,
    )
    z = 0.4 + 0.07j
    green = green_matrix(hamiltonian, z)
    reference = contract_bubble_batch(
        projected.transverse_vertices,
        perturbation,
        green,
        green,
        np.array([0.5, 0.5]),
    )
    transformed_g = gauge_dagger[:, None] @ perturbation @ gauge[:, None]
    transformed_green = green_matrix(gauge_dagger @ hamiltonian @ gauge, z)
    actual = contract_bubble_batch(
        transformed.transverse_vertices,
        transformed_g,
        transformed_green,
        transformed_green,
        np.array([0.5, 0.5]),
    )
    np.testing.assert_allclose(actual, reference, rtol=2.0e-12, atol=2.0e-12)


def test_single_q_epr_fft_matches_direct_positive_phase_sum(tmp_path) -> None:
    path = tmp_path / "toy_epr.h5"
    kmesh = (2, 1, 1)
    qmesh = (1, 1, 1)
    lattice = np.eye(3, dtype=np.float64)
    centres = np.zeros((2, 3), dtype=np.float64)
    atom_positions = np.zeros((1, 3), dtype=np.float64)
    electron_images = init_rvec_images(kmesh, lattice)
    phonon_images = init_rvec_images(qmesh, lattice)
    ws_electron = set_wigner_seitz_cell(
        electron_images, lattice, centres[0], centres[0]
    )
    ws_phonon = set_wigner_seitz_cell(
        phonon_images, lattice, centres[0], atom_positions[0]
    )
    assert ws_phonon.nr == 1

    blocks: dict[int, np.ndarray] = {}
    with h5py.File(path, "w") as handle:
        basic = handle.create_group("basic_data")
        basic.create_dataset("nat", data=np.int32(1))
        basic.create_dataset("num_wann", data=np.int32(2))
        basic.create_dataset("kc_dim", data=np.asarray(kmesh, dtype=np.int32))
        basic.create_dataset("qc_dim", data=np.asarray(qmesh, dtype=np.int32))
        basic.create_dataset("at", data=lattice.T)
        basic.create_dataset("tau", data=atom_positions)
        basic.create_dataset("wannier_center_cryst", data=centres)
        basic.create_dataset("spinor", data=np.int32(1))
        eph = handle.create_group("eph_matrix_wannier")
        for orbital in (1, 2):
            block = np.empty((3, ws_electron.nr, ws_phonon.nr), dtype=np.complex128)
            for axis in range(3):
                block[axis, :, 0] = (
                    orbital + 0.2 * axis + np.arange(ws_electron.nr)
                ) + 1j * (0.1 * axis)
            blocks[orbital] = block
            stored = block.transpose(2, 1, 0)
            eph.create_dataset(f"ep_hop_r_1_{orbital}_{orbital}", data=stored.real)
            eph.create_dataset(f"ep_hop_i_1_{orbital}_{orbital}", data=stored.imag)

    result = reconstruct_epr_g_at_q(
        path,
        0,
        energy_unit="ev",
        displacement_unit="angstrom",
    )
    for orbital, block in blocks.items():
        direct = np.einsum(
            "kr,ar->ka",
            np.exp(
                2j * np.pi * (result.kpoints @ ws_electron.vectors.astype(np.float64).T)
            ),
            block[:, :, 0],
            optimize=True,
        )
        np.testing.assert_allclose(
            result.values[:, :, orbital - 1, orbital - 1],
            direct,
            atol=1.0e-13,
        )
