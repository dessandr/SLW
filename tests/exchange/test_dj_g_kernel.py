from __future__ import annotations

import unittest

import numpy as np

from slw.exchange.kernels.dj_epr import (
    _cartesian_axis_batches,
    _partition_derivative_work,
    _rotate_g_to_eigenbasis,
    _spectral_projected_blocks,
)


def _random_unitaries(rng: np.random.Generator, nk: int, nw: int) -> np.ndarray:
    result = np.empty((nk, nw, nw), dtype=np.complex128)
    for ik in range(nk):
        raw = rng.normal(size=(nw, nw)) + 1j * rng.normal(size=(nw, nw))
        q, r = np.linalg.qr(raw)
        phases = np.diag(r)
        phases = np.where(np.abs(phases) > 0.0, phases / np.abs(phases), 1.0)
        result[ik] = q * np.conjugate(phases)[None, :]
    return result


class ScalarDjGKernelTests(unittest.TestCase):
    def test_cartesian_axis_batches_keep_all_targets_in_axis_order(self) -> None:
        self.assertEqual(
            _cartesian_axis_batches((3, 7), ("z", "x")),
            (
                ("z", ((3, "z"), (7, "z"))),
                ("x", ((3, "x"), (7, "x"))),
            ),
        )

    def test_mpi_partition_distributes_cache_before_excess_rank_energy(self) -> None:
        work_items = tuple((target, axis) for target in range(4) for axis in "xyz")
        energies = tuple(range(23))

        assigned: dict[tuple[int, str], list[int]] = {
            item: [] for item in work_items
        }
        for rank in range(20):
            local_items, local_energy, replicas = _partition_derivative_work(
                work_items,
                energies,
                rank,
                20,
            )
            self.assertEqual(len(local_items), 1)
            self.assertIn(replicas, (1, 2))
            assigned[local_items[0]].extend(local_energy)

        for values in assigned.values():
            self.assertEqual(sorted(values), list(energies))

        ownership = []
        for rank in range(4):
            local_items, local_energy, replicas = _partition_derivative_work(
                work_items,
                energies,
                rank,
                4,
            )
            self.assertEqual(len(local_items), 3)
            self.assertEqual(local_energy, energies)
            self.assertEqual(replicas, 1)
            ownership.extend(local_items)
        self.assertEqual(sorted(ownership), sorted(work_items))

    def test_complete_spectral_endpoint_blocks_match_direct_ggg(self) -> None:
        rng = np.random.default_rng(810_2026)
        nk, nq, nw = 4, 3, 7
        coeffs = _random_unitaries(rng, nk, nw)
        kq_map = np.asarray(((0, 1, 2, 3), (1, 2, 3, 0), (2, 3, 0, 1)), dtype=np.int64)
        g_wannier = rng.normal(size=(nq, nk, nw, nw)) + 1j * rng.normal(
            size=(nq, nk, nw, nw)
        )
        poles = rng.normal(size=(nk, nw))
        z = 0.37 + 0.19j
        inverse = 1.0 / (z - poles)
        slices = {0: slice(0, 2), 1: slice(2, 5), 2: slice(5, 7)}
        endpoint_pairs = ((0, 1), (0, 2), (1, 0), (2, 1), (2, 2))

        g_band = _rotate_g_to_eigenbasis(g_wannier, coeffs, kq_map)
        green = np.einsum(
            "kni,ki,kmi->knm",
            coeffs,
            inverse,
            np.conjugate(coeffs),
            optimize=True,
        )
        for iq in range(nq):
            blocks = _spectral_projected_blocks(
                coeffs,
                inverse,
                g_band[iq],
                kq_map[iq],
                slices,
                endpoint_pairs,
            )
            direct = green[kq_map[iq]] @ g_wannier[iq] @ green
            for li, lj in endpoint_pairs:
                with self.subTest(iq=iq, li=li, lj=lj):
                    np.testing.assert_allclose(
                        blocks[(li, lj)],
                        direct[:, slices[li], slices[lj]],
                        rtol=2.0e-13,
                        atol=2.0e-13,
                    )

    def test_spectral_rotation_retains_the_full_wannier_dimension(self) -> None:
        rng = np.random.default_rng(811_2026)
        coeffs = _random_unitaries(rng, 2, 5)
        g = rng.normal(size=(1, 2, 5, 5)) + 1j * rng.normal(size=(1, 2, 5, 5))
        rotated = _rotate_g_to_eigenbasis(
            g, coeffs, np.asarray(((1, 0),), dtype=np.int64)
        )
        self.assertEqual(rotated.shape, g.shape)
        expected_norm = np.linalg.norm(g, axis=(-2, -1))
        actual_norm = np.linalg.norm(rotated, axis=(-2, -1))
        np.testing.assert_allclose(
            actual_norm, expected_norm, rtol=1.0e-13, atol=1.0e-13
        )


if __name__ == "__main__":
    unittest.main()
