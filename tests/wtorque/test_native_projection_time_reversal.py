from __future__ import annotations

import numpy as np
import pytest

from slw.core.wannier_io import (
    expand_compact_wannier_u_dis,
    read_wannier_eig,
)
from slw.core.wannier_spin_io import iter_wannier_amn_chunks
from slw.wtorque.model.time_reversal import (
    atomic_spin_time_reversal,
    projection_anchored_sewing,
    time_reversed,
)


def _unitary(rng: np.random.Generator, dimension: int) -> np.ndarray:
    raw = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(
        size=(dimension, dimension)
    )
    q, r = np.linalg.qr(raw)
    phase = np.diag(r)
    return np.asarray(q * (phase / np.abs(phase)).conj(), dtype=np.complex128)


def test_chunked_amn_reader_preserves_band_fast_layout(tmp_path) -> None:
    path = tmp_path / "toy.amn"
    lines = ["synthetic AMN", "2 3 2"]
    expected = np.empty((3, 2, 2), dtype=np.complex128)
    for ik in range(3):
        for projection in range(2):
            for band in range(2):
                value = complex(100 * ik + 10 * projection + band, band - projection)
                expected[ik, band, projection] = value
                lines.append(
                    f"{band + 1} {projection + 1} {ik + 1} "
                    f"{value.real:.8f} {value.imag:.8f}"
                )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    chunks = list(iter_wannier_amn_chunks(path, kpoint_chunk=2))

    assert [item[0] for item in chunks] == [slice(0, 2), slice(2, 3)]
    assert np.array_equal(np.concatenate([item[1] for item in chunks]), expected)


def test_eig_reader_and_compact_u_dis_restore_global_band_indices(tmp_path) -> None:
    eig_path = tmp_path / "toy.eig"
    energies = np.array(
        [[-4.0, -0.5, 0.2, 1.1, 8.0], [-3.0, -0.2, 0.4, 1.3, 9.0]],
        dtype=np.float64,
    )
    lines = []
    for ik in range(2):
        for band in range(5):
            lines.append(f"{band + 1} {ik + 1} {energies[ik, band]:.8f}")
    eig_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    parsed = read_wannier_eig(
        eig_path, expected_bands=5, expected_kpoints=2
    ).eigenvalues
    assert np.array_equal(parsed, energies)

    compact = np.zeros((2, 5, 2), dtype=np.complex128)
    compact[:, :3] = np.arange(12, dtype=np.float64).reshape(2, 3, 2)
    expanded = expand_compact_wannier_u_dis(
        compact,
        parsed,
        outer_window_min=-1.0,
        outer_window_max=2.0,
    )
    assert np.array_equal(expanded[:, 1:4], compact[:, :3])
    assert np.count_nonzero(expanded[:, [0, 4]]) == 0

    compact[0, 4, 0] = 1.0e-3
    with pytest.raises(ValueError, match="nonzero padded rows"):
        expand_compact_wannier_u_dis(
            compact,
            parsed,
            outer_window_min=-1.0,
            outer_window_max=2.0,
        )


def test_projection_anchored_sewing_is_covariant_under_wannier_gauge() -> None:
    rng = np.random.default_rng(4217)
    nk = 4
    dimension = 4
    minus = np.array([0, 3, 2, 1], dtype=np.int64)
    overlap = np.stack([_unitary(rng, dimension) for _ in range(nk)])
    atomic = atomic_spin_time_reversal(dimension, "interleaved")
    projected = projection_anchored_sewing(overlap, minus, atomic)
    identity = np.eye(dimension, dtype=np.complex128)
    assert (
        np.max(
            np.abs(
                projected.sewing @ projected.sewing.conj().transpose(0, 2, 1) - identity
            )
        )
        < 1.0e-12
    )
    assert (
        np.max(np.abs(projected.sewing @ projected.sewing[minus].conj() + identity))
        < 1.0e-12
    )

    gauge = np.stack([_unitary(rng, dimension) for _ in range(nk)])
    transformed_overlap = gauge.conj().transpose(0, 2, 1) @ overlap
    transformed = projection_anchored_sewing(transformed_overlap, minus, atomic)
    expected_sewing = (
        gauge.conj().transpose(0, 2, 1) @ projected.sewing @ gauge[minus].conj()
    )
    assert np.allclose(transformed.sewing, expected_sewing, atol=1.0e-12)

    raw_h = rng.normal(size=(nk, dimension, dimension)) + 1j * rng.normal(
        size=(nk, dimension, dimension)
    )
    hamiltonian = np.asarray(
        0.5 * (raw_h + raw_h.conj().transpose(0, 2, 1)),
        dtype=np.complex128,
    )
    transformed_hamiltonian = gauge.conj().transpose(0, 2, 1) @ hamiltonian @ gauge
    theta_h = time_reversed(hamiltonian[minus], projected.sewing)
    transformed_theta_h = time_reversed(
        transformed_hamiltonian[minus], transformed.sewing
    )
    expected_theta_h = gauge.conj().transpose(0, 2, 1) @ theta_h @ gauge
    assert np.allclose(transformed_theta_h, expected_theta_h, atol=1.0e-11)
