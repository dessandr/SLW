import unittest

import numpy as np

from slw.exchange.kernels.dj_tensor_epr import _decompose_tb2j_dA
from slw.exchange.kernels.j_tensor_epr import _decompose_tb2j_pair


def _dmi_matrix(vector):
    """Build the antisymmetric matrix used by the TB2J convention."""
    vector = np.asarray(vector)
    matrix = np.zeros(vector.shape[:-1] + (3, 3), dtype=vector.dtype)
    matrix[..., 0, 1] = vector[..., 2]
    matrix[..., 1, 0] = -vector[..., 2]
    matrix[..., 0, 2] = -vector[..., 1]
    matrix[..., 2, 0] = vector[..., 1]
    matrix[..., 1, 2] = vector[..., 0]
    matrix[..., 2, 1] = -vector[..., 0]
    return matrix


class StaticTensorDecompositionTests(unittest.TestCase):
    def test_pair_decomposition_invariants(self):
        rng = np.random.default_rng(20260820)
        val = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        mate = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))

        result = _decompose_tb2j_pair(val, mate)

        expected_raw = np.imag(val[1:4, 1:4] + mate[1:4, 1:4])
        expected_dmi = np.real(val[0, 1:4] - val[1:4, 0])
        expected_jiso = np.imag(val[0, 0] - val[1, 1] - val[2, 2] - val[3, 3])
        np.testing.assert_allclose(result["m_raw"], expected_raw)
        np.testing.assert_allclose(result["dmi"], expected_dmi)
        self.assertAlmostEqual(result["jiso"], expected_jiso)

        np.testing.assert_allclose(result["gamma"], result["gamma"].T, atol=1e-14)
        self.assertAlmostEqual(float(np.trace(result["gamma"])), 0.0, places=14)
        np.testing.assert_allclose(result["ma"], -result["ma"].T, atol=1e-14)
        np.testing.assert_allclose(result["dmi_mat"], -result["dmi_mat"].T, atol=1e-14)
        np.testing.assert_allclose(result["dmi_mat"], _dmi_matrix(result["dmi"]))

        isotropic = np.eye(3) * result["jiso"]
        np.testing.assert_allclose(
            result["jfull_dmi"], isotropic + result["gamma"] + result["dmi_mat"]
        )
        np.testing.assert_allclose(
            result["jfull_aab"], isotropic + result["gamma"] + result["ma"]
        )


class DerivativeTensorDecompositionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(731)
        self.d_a = rng.normal(size=(3, 2, 2, 4, 4)) + 1j * rng.normal(
            size=(3, 2, 2, 4, 4)
        )
        self.pair_meta = [
            {"gi": 0, "gj": 1, "R": (1, 0, 0)},
            {"gi": 1, "gj": 0, "R": (-1, 0, 0)},
        ]
        self.rp_grid = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        self.result = _decompose_tb2j_dA(
            self.d_a,
            self.pair_meta,
            rp_grid=self.rp_grid,
            qmesh=(3, 1, 1),
        )

    def test_derivative_decomposition_invariants(self):
        result = self.result
        np.testing.assert_allclose(
            result["dJ_tensor_r"],
            result["dJ_iso_tensor_r"]
            + result["dJ_gamma_r"]
            + result["dJ_dmi_tensor_r"],
            rtol=1e-13,
            atol=1e-11,
        )
        np.testing.assert_allclose(
            result["dJ_iso_tensor_r"],
            result["dJ_iso_r"][..., None, None] * np.eye(3),
        )
        np.testing.assert_allclose(
            result["dJ_gamma_r"],
            np.swapaxes(result["dJ_gamma_r"], -1, -2),
            atol=1e-11,
        )
        np.testing.assert_allclose(
            np.trace(result["dJ_gamma_r"], axis1=-2, axis2=-1),
            0.0,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            result["dJ_dmi_tensor_r"],
            -np.swapaxes(result["dJ_dmi_tensor_r"], -1, -2),
            atol=1e-11,
        )
        np.testing.assert_allclose(
            result["dJ_dmi_tensor_r"], _dmi_matrix(result["dDMI_r"])
        )

        expected_jiso = 1000.0 * np.imag(
            self.d_a[..., 0, 0]
            - self.d_a[..., 1, 1]
            - self.d_a[..., 2, 2]
            - self.d_a[..., 3, 3]
        )
        expected_dmi = 1000.0 * np.real(
            self.d_a[..., 0, 1:4] - self.d_a[..., 1:4, 0]
        )
        np.testing.assert_allclose(result["dJ_iso_r"], expected_jiso)
        np.testing.assert_allclose(result["dDMI_r"], expected_dmi)
        for value in result.values():
            self.assertTrue(np.all(np.isfinite(value)))

    def test_directed_mate_uses_periodic_rp_minus_r(self):
        rp_index = self.rp_grid[:, 0]
        for bond, meta in enumerate(self.pair_meta):
            mate = 1 - bond
            shifted_index = np.mod(rp_index - int(meta["R"][0]), 3)
            expected = 1000.0 * np.imag(
                self.d_a[:, bond, :, 1:4, 1:4]
                + self.d_a[shifted_index, mate, :, 1:4, 1:4]
            )
            np.testing.assert_allclose(self.result["dA_jani_r"][:, bond], expected)

        minus_one = np.mod(rp_index - 1, 3)
        np.testing.assert_allclose(
            self.result["dA_jani_r"][:, 0],
            self.result["dA_jani_r"][minus_one, 1],
        )


if __name__ == "__main__":
    unittest.main()
