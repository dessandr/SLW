import unittest

import numpy as np

from slw.exchange.kernels.j_epr import _compute_j_direct


def _toy_exchange_problem():
    """Return a deterministic two-site, one-orbital-per-site problem."""
    hk_up = np.array(
        [
            [[-0.90, 0.18 + 0.04j], [0.18 - 0.04j, -0.45]],
            [[-0.70, -0.11 + 0.03j], [-0.11 - 0.03j, -0.35]],
        ],
        dtype=np.complex128,
    )
    hk_dn = np.array(
        [
            [[0.35, 0.13 - 0.02j], [0.13 + 0.02j, 0.75]],
            [[0.50, -0.08 - 0.01j], [-0.08 + 0.01j, 0.90]],
        ],
        dtype=np.complex128,
    )
    slices = {0: slice(0, 1), 1: slice(1, 2)}
    pair_meta = [
        {"gi": 0, "gj": 1, "li": 0, "lj": 1, "R": (0, 0, 0)},
        {"gi": 1, "gj": 0, "li": 1, "lj": 0, "R": (1, 0, 0)},
    ]
    kpts = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    energy_mesh = [
        (-1.8 + 0.45j, 0.17 + 0.03j),
        (-1.1 + 0.55j, 0.22 - 0.02j),
        (-0.4 + 0.65j, 0.19 + 0.04j),
        (0.2 + 0.70j, 0.13 - 0.01j),
    ]
    return hk_up, hk_dn, slices, pair_meta, kpts, energy_mesh


def _dense_reference(hk_up, hk_dn, slices, pair_meta, kpts, energy_mesh, efermi):
    """Evaluate the same LKAG expression using explicit dense inverses."""
    dim = hk_up.shape[-1]
    eye = np.eye(dim, dtype=np.complex128)
    h_up = 0.5 * (hk_up + np.swapaxes(hk_up.conj(), -1, -2)) - efermi * eye
    h_dn = 0.5 * (hk_dn + np.swapaxes(hk_dn.conj(), -1, -2)) - efermi * eye

    delta_full = np.mean(hk_up - hk_dn, axis=0)
    delta = {}
    signs = {}
    for site, site_slice in slices.items():
        local = delta_full[site_slice, site_slice]
        delta[site] = 0.5 * (local + local.conj().T)
        signs[site] = -1.0 if np.real(np.trace(delta[site])) < 0.0 else 1.0

    j_acc = np.zeros(len(pair_meta), dtype=np.complex128)
    trace_acc = np.zeros(len(pair_meta), dtype=np.complex128)
    nk = len(kpts)
    for z, dz in energy_mesh:
        g_up = [np.linalg.inv(z * eye - matrix) for matrix in h_up]
        g_dn = [np.linalg.inv(z * eye - matrix) for matrix in h_dn]
        for bond, meta in enumerate(pair_meta):
            site_i = int(meta["li"])
            site_j = int(meta["lj"])
            slice_i = slices[site_i]
            slice_j = slices[site_j]
            shape_ij = (slice_i.stop - slice_i.start, slice_j.stop - slice_j.start)
            shape_ji = shape_ij[::-1]
            green_up = np.zeros(shape_ij, dtype=np.complex128)
            green_dn = np.zeros(shape_ji, dtype=np.complex128)
            for ik, kpoint in enumerate(kpts):
                phase = np.exp(-2j * np.pi * np.dot(meta["R"], kpoint)) / nk
                green_up += phase * g_up[ik][slice_i, slice_j]
                green_dn += phase.conjugate() * g_dn[ik][slice_j, slice_i]
            relative_sign = signs[site_i] * signs[site_j]
            trace = np.trace(
                delta[site_i] @ green_up @ delta[site_j] @ green_dn
            ) / relative_sign
            trace_acc[bond] += trace
            j_acc[bond] += trace * dz

    j_mev = 1000.0 * np.imag(j_acc) / (4.0 * np.pi)
    return j_mev, trace_acc


class DirectExchangeKernelTests(unittest.TestCase):
    def test_small_problem_matches_dense_inverse_reference(self):
        problem = _toy_exchange_problem()
        hk_up, hk_dn, slices, pair_meta, kpts, energy_mesh = problem

        result, trace, kdata, stats = _compute_j_direct(
            hk_up,
            hk_dn,
            slices,
            pair_meta,
            kpts,
            energy_mesh,
            0.0,
            nproc=1,
        )
        expected, expected_trace = _dense_reference(
            hk_up, hk_dn, slices, pair_meta, kpts, energy_mesh, 0.0
        )

        self.assertEqual(result.shape, (len(pair_meta),))
        self.assertEqual(trace.shape, (len(pair_meta),))
        self.assertTrue(np.all(np.isfinite(result)))
        self.assertTrue(np.all(np.isfinite(trace)))
        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(trace, expected_trace, rtol=1e-12, atol=1e-12)
        self.assertEqual(kdata["evals_up"].shape, (len(kpts), hk_up.shape[-1]))
        self.assertEqual(kdata["evals_dn"].shape, (len(kpts), hk_dn.shape[-1]))
        self.assertEqual(stats, {"n_chunks": 1, "nproc": 1})

    def test_two_process_result_matches_serial(self):
        hk_up, hk_dn, slices, pair_meta, kpts, energy_mesh = _toy_exchange_problem()

        serial = _compute_j_direct(
            hk_up, hk_dn, slices, pair_meta, kpts, energy_mesh, 0.0, nproc=1
        )
        parallel = _compute_j_direct(
            hk_up, hk_dn, slices, pair_meta, kpts, energy_mesh, 0.0, nproc=2
        )

        np.testing.assert_allclose(parallel[0], serial[0], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(parallel[1], serial[1], rtol=1e-12, atol=1e-12)
        self.assertEqual(parallel[3], {"n_chunks": 2, "nproc": 2})

    def test_identical_spin_channels_give_zero_exchange(self):
        hk_up, _, slices, pair_meta, kpts, energy_mesh = _toy_exchange_problem()

        result, trace, kdata, _ = _compute_j_direct(
            hk_up,
            hk_up.copy(),
            slices,
            pair_meta,
            kpts,
            energy_mesh,
            0.0,
            nproc=1,
        )

        np.testing.assert_array_equal(result, np.zeros(len(pair_meta)))
        np.testing.assert_array_equal(trace, np.zeros(len(pair_meta), dtype=np.complex128))
        for local_delta in kdata["delta"].values():
            np.testing.assert_array_equal(local_delta, np.zeros_like(local_delta))


if __name__ == "__main__":
    unittest.main()
