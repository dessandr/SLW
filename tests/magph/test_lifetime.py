import unittest

import numpy as np

from slw.core.constants import HBAR_MEV_PS
from slw.magph.lifetime import (
    compute_lifetime,
    lifetime_from_onshell_self_energy,
    linewidth_observables,
)
from slw.magph.self_energy import compute_onshell_self_energy_diagonal


class LifetimeObservableTests(unittest.TestCase):
    def test_hwhm_converts_to_fwhm_rate_and_lifetime(self):
        gamma = np.array([0.25, 1.5])
        result = linewidth_observables(gamma)
        expected_fwhm = 2.0 * gamma
        np.testing.assert_allclose(result.gamma_hwhm_mev, gamma)
        np.testing.assert_allclose(result.fwhm_mev, expected_fwhm)
        np.testing.assert_allclose(
            result.scattering_rate_ps_inv, expected_fwhm / HBAR_MEV_PS
        )
        np.testing.assert_allclose(result.lifetime_ps, HBAR_MEV_PS / expected_fwhm)
        np.testing.assert_array_equal(result.valid_damping, [True, True])
        self.assertFalse(result.gamma_hwhm_mev.flags.writeable)

    def test_zero_vertex_linewidth_has_zero_rate_and_infinite_lifetime(self):
        self_energy = compute_onshell_self_energy_diagonal(
            np.array([1.0, 2.0, 3.0]),
            np.zeros((2, 1, 2, 3), dtype=np.complex128),
            np.array([[0.7, 1.4], [0.9, 1.8]]),
            np.array([[0.3], [0.6]]),
            temperature_k=20.0,
            broadening_mev=0.1,
            metric=np.array([1.0, 1.0]),
        )
        result = lifetime_from_onshell_self_energy(self_energy.sigma_diagonal_mev)
        np.testing.assert_array_equal(result.fwhm_mev, np.zeros(3))
        np.testing.assert_array_equal(result.scattering_rate_ps_inv, np.zeros(3))
        np.testing.assert_array_equal(result.lifetime_ps, np.full(3, np.inf))
        np.testing.assert_array_equal(result.valid_damping, np.ones(3, dtype=bool))

        shorthand = compute_lifetime(np.zeros(3))
        np.testing.assert_array_equal(shorthand.lifetime_ps, result.lifetime_ps)

    def test_roundoff_negative_is_clipped_only_with_explicit_tolerance(self):
        gamma = np.array([-2.0e-13, 0.0, 0.4])
        strict = linewidth_observables(gamma)
        self.assertTrue(strict.invalid_negative_damping[0])
        self.assertTrue(np.isnan(strict.gamma_hwhm_mev[0]))

        tolerant = linewidth_observables(gamma, negative_tolerance_mev=1.0e-12)
        np.testing.assert_array_equal(tolerant.roundoff_clipped, [True, False, False])
        np.testing.assert_array_equal(
            tolerant.invalid_negative_damping, [False, False, False]
        )
        self.assertEqual(tolerant.raw_gamma_hwhm_mev[0], gamma[0])
        self.assertEqual(tolerant.gamma_hwhm_mev[0], 0.0)
        self.assertEqual(tolerant.scattering_rate_ps_inv[0], 0.0)
        self.assertTrue(np.isinf(tolerant.lifetime_ps[0]))

    def test_materially_negative_damping_is_marked_invalid_not_clipped(self):
        gamma = np.array([-0.03, -1.0e-13, 0.2])
        result = linewidth_observables(gamma, negative_tolerance_mev=1.0e-12)
        np.testing.assert_array_equal(result.valid_damping, [False, True, True])
        np.testing.assert_array_equal(
            result.invalid_negative_damping, [True, False, False]
        )
        np.testing.assert_array_equal(result.roundoff_clipped, [False, True, False])
        self.assertEqual(result.raw_gamma_hwhm_mev[0], -0.03)
        self.assertTrue(np.isnan(result.gamma_hwhm_mev[0]))
        self.assertTrue(np.isnan(result.fwhm_mev[0]))
        self.assertTrue(np.isnan(result.scattering_rate_ps_inv[0]))
        self.assertTrue(np.isnan(result.lifetime_ps[0]))
        self.assertTrue(result.has_invalid_damping)

    def test_nonfinite_damping_is_invalid(self):
        result = linewidth_observables(np.array([np.nan, np.inf, 0.1]))
        np.testing.assert_array_equal(result.valid_damping, [False, False, True])
        np.testing.assert_array_equal(
            result.invalid_negative_damping, [False, False, False]
        )
        self.assertTrue(np.isnan(result.lifetime_ps[0]))
        self.assertTrue(np.isnan(result.lifetime_ps[1]))

    def test_retarded_onshell_vector_uses_minus_imaginary_part(self):
        sigma = np.array([1.2 - 0.1j, -3.0 - 0.4j, 0.5 + 0.02j])
        result = lifetime_from_onshell_self_energy(sigma, negative_tolerance_mev=1.0e-3)
        np.testing.assert_allclose(
            result.raw_gamma_hwhm_mev, np.array([0.1, 0.4, -0.02])
        )
        np.testing.assert_array_equal(result.valid_damping, [True, True, False])

    def test_retarded_onshell_matrix_extracts_diagonal(self):
        sigma = np.array(
            [
                [2.0 - 0.2j, 3.0 + 4.0j],
                [5.0 - 1.0j, 7.0 - 0.6j],
            ]
        )
        result = lifetime_from_onshell_self_energy(sigma)
        np.testing.assert_allclose(result.gamma_hwhm_mev, [0.2, 0.6])

    def test_invalid_tolerance_complex_gamma_and_bad_sigma_shape_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "negative_tolerance_mev"):
            linewidth_observables([0.1], negative_tolerance_mev=-1.0)
        with self.assertRaisesRegex(ValueError, "complex"):
            linewidth_observables(np.array([0.1 + 0.2j]))
        with self.assertRaisesRegex(ValueError, "square axes"):
            lifetime_from_onshell_self_energy(np.zeros((2, 3), dtype=complex))


if __name__ == "__main__":
    unittest.main()
