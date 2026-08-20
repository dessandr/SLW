import unittest

import numpy as np

from slw.core.constants import KB_MEV_PER_K
from slw.magph.legacy.numerics import compute_self_energy_at_frequencies
from slw.magph.self_energy import (
    bose_occupation,
    compute_onshell_self_energy_diagonal,
    compute_retarded_self_energy,
    normalize_q_weights,
)


def _dense_self_energy(
    omega,
    vertex,
    internal_energy,
    phonon_energy,
    temperature,
    broadening,
    metric,
    q_weights,
):
    result = np.zeros(
        (len(omega), vertex.shape[-1], vertex.shape[-1]),
        dtype=np.complex128,
    )
    for iw, frequency in enumerate(omega):
        for iq in range(vertex.shape[0]):
            for mode in range(vertex.shape[1]):
                nphonon = float(bose_occupation(phonon_energy[iq, mode], temperature))
                for internal in range(vertex.shape[2]):
                    energy = internal_energy[iq, internal]
                    nmagnon = float(bose_occupation(energy, temperature))
                    weight = (
                        q_weights[iq]
                        * metric[internal]
                        * (
                            (nphonon + 1.0 + nmagnon)
                            / (
                                frequency
                                + 1j * broadening
                                - energy
                                - phonon_energy[iq, mode]
                            )
                            + (nphonon - nmagnon)
                            / (
                                frequency
                                + 1j * broadening
                                - energy
                                + phonon_energy[iq, mode]
                            )
                        )
                    )
                    coupling = vertex[iq, mode, internal]
                    result[iw] += np.outer(coupling.conj(), coupling) * weight
    return result


def _random_problem(seed=731):
    rng = np.random.default_rng(seed)
    nq, nmode, ninternal, nexternal = 5, 3, 4, 3
    vertex = rng.normal(size=(nq, nmode, ninternal, nexternal))
    vertex = vertex + 1j * rng.normal(size=vertex.shape)
    metric = np.array([1.0, 1.0, -1.0, -1.0])
    internal_energy = rng.uniform(0.2, 12.0, size=(nq, ninternal)) * metric[None, :]
    phonon_energy = rng.uniform(0.1, 8.0, size=(nq, nmode))
    omega = np.array([-1.2, 0.4, 3.8, 9.1])
    q_weights = np.array([1.0, 3.0, 2.0, 0.5, 4.0])
    return omega, vertex, internal_energy, phonon_energy, metric, q_weights


class BoseOccupationTests(unittest.TestCase):
    def test_negative_bdg_identity_and_zero_temperature_limit(self):
        positive = np.array([0.3, 2.0, 20.0])
        at_temperature = bose_occupation(positive, 80.0)
        negative = bose_occupation(-positive, 80.0)
        np.testing.assert_allclose(negative, -(1.0 + at_temperature))
        np.testing.assert_array_equal(
            bose_occupation(np.array([-2.0, 0.0, 2.0]), 0.0),
            np.array([-1.0, 0.0, 0.0]),
        )

    def test_finite_temperature_zero_energy_is_rejected_without_a_cutoff(self):
        with self.assertRaisesRegex(ValueError, "Bose occupation diverges"):
            bose_occupation(np.array([0.0]), 1.0e-15)

        small_nonzero = 1.0e-12
        temperature = 300.0
        expected = 1.0 / np.expm1(small_nonzero / (KB_MEV_PER_K * temperature))
        np.testing.assert_allclose(
            bose_occupation(np.array([small_nonzero]), temperature),
            np.array([expected]),
            rtol=2.0e-15,
        )

    def test_invalid_temperature_and_energy_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "temperature_k"):
            bose_occupation([1.0], -1.0)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            bose_occupation([np.nan], 10.0)
        with self.assertRaisesRegex(ValueError, "complex"):
            bose_occupation([1.0 + 1.0j], 10.0)


class RetardedSelfEnergyTests(unittest.TestCase):
    def test_single_mode_zero_temperature_has_analytic_value(self):
        broadening = 0.5
        result = compute_retarded_self_energy(
            np.array([4.0]),
            np.array([[[[2.0 + 0.0j]]]]),
            np.array([[3.0]]),
            np.array([[2.0]]),
            temperature_k=0.0,
            broadening_mev=broadening,
            metric=np.array([1.0]),
        )
        expected = 4.0 / (-1.0 + 1j * broadening)
        np.testing.assert_allclose(result.sigma_retarded_mev[0, 0, 0], expected)
        self.assertFalse(result.sigma_retarded_mev.flags.writeable)
        np.testing.assert_array_equal(result.q_weights, np.array([1.0]))

    def test_vectorized_result_matches_dense_loops_and_normalized_weights(self):
        omega, vertex, internal, phonon, metric, raw_weights = _random_problem()
        normalized = raw_weights / np.sum(raw_weights)
        result = compute_retarded_self_energy(
            omega,
            vertex,
            internal,
            phonon,
            temperature_k=65.0,
            broadening_mev=0.17,
            metric=metric,
            q_weights=raw_weights,
            q_chunk_size=2,
            omega_chunk_size=3,
        )
        expected = _dense_self_energy(
            omega,
            vertex,
            internal,
            phonon,
            65.0,
            0.17,
            metric,
            normalized,
        )
        np.testing.assert_allclose(
            result.sigma_retarded_mev, expected, rtol=2.0e-13, atol=2.0e-13
        )
        np.testing.assert_allclose(result.q_weights, normalized)

    def test_uniform_weights_match_legacy_numerics_for_signed_bdg_channels(self):
        rng = np.random.default_rng(90)
        nq, nmode, nchannel = 4, 2, 4
        vertex = rng.normal(size=(nq, nmode, nchannel, nchannel))
        vertex = vertex + 1j * rng.normal(size=vertex.shape)
        positive = rng.uniform(0.4, 7.0, size=(nq, nchannel // 2))
        internal = np.concatenate((positive, -positive), axis=1)
        phonon = rng.uniform(0.2, 5.0, size=(nq, nmode))
        omega = np.array([-3.0, 0.5, 4.5])
        metric = np.array([1.0, 1.0, -1.0, -1.0])

        native = compute_retarded_self_energy(
            omega,
            vertex,
            internal,
            phonon,
            temperature_k=125.0,
            broadening_mev=0.23,
            metric=metric,
            q_chunk_size=2,
            omega_chunk_size=2,
        ).sigma_retarded_mev
        legacy = compute_self_energy_at_frequencies(
            omega,
            vertex,
            internal,
            phonon,
            125.0,
            0.23,
            nq,
            metric,
        )
        np.testing.assert_allclose(native, legacy, rtol=2.0e-12, atol=2.0e-12)

    def test_fm_and_afm_use_positive_and_signed_energy_branches_respectively(self):
        vertex = np.ones((1, 1, 2, 1), dtype=np.complex128)
        phonon = np.array([[1.0]])
        kwargs = {
            "omega_mev": np.array([2.5]),
            "vertex_mev": vertex,
            "phonon_energy_mev": phonon,
            "temperature_k": 0.0,
            "broadening_mev": 0.2,
        }
        fm = compute_retarded_self_energy(
            **kwargs,
            internal_energy_mev=np.array([[2.0, 2.0]]),
            metric=np.array([1.0, 1.0]),
        ).sigma_retarded_mev
        afm_result = compute_retarded_self_energy(
            **kwargs,
            internal_energy_mev=np.array([[2.0, -2.0]]),
            metric=np.array([1.0, -1.0]),
        )
        afm = afm_result.sigma_retarded_mev
        expected_afm = _dense_self_energy(
            kwargs["omega_mev"],
            vertex,
            np.array([[2.0, -2.0]]),
            phonon,
            0.0,
            0.2,
            np.array([1.0, -1.0]),
            np.array([1.0]),
        )
        self.assertGreater(abs(fm[0, 0, 0]), 0.0)
        self.assertGreater(abs(afm[0, 0, 0]), 0.0)
        np.testing.assert_allclose(afm, expected_afm, rtol=2.0e-14, atol=2.0e-14)
        self.assertEqual(afm_result.minimum_metric_energy_mev, 2.0)

    def test_metric_energy_sign_mismatch_is_strict_by_default_and_auditable(self):
        base = {
            "omega_mev": np.array([1.0]),
            "vertex_mev": np.ones((1, 1, 2, 1), dtype=np.complex128),
            "internal_energy_mev": np.array([[2.0, 1.0e-10]]),
            "phonon_energy_mev": np.array([[0.5]]),
            "temperature_k": 0.0,
            "broadening_mev": 0.1,
            "metric": np.array([1.0, -1.0]),
        }
        with self.assertRaisesRegex(ValueError, "signed BdG metric"):
            compute_retarded_self_energy(**base)

        result = compute_retarded_self_energy(
            **base, metric_energy_tolerance_mev=2.0e-10
        )
        self.assertEqual(result.metric_energy_tolerance_mev, 2.0e-10)
        self.assertEqual(result.minimum_metric_energy_mev, -1.0e-10)

    def test_zero_bose_modes_are_allowed_only_at_zero_temperature(self):
        base = {
            "omega_mev": np.array([1.0]),
            "vertex_mev": np.ones((1, 1, 1, 1), dtype=np.complex128),
            "internal_energy_mev": np.array([[1.0]]),
            "phonon_energy_mev": np.array([[0.0]]),
            "broadening_mev": 0.1,
            "metric": np.array([1.0]),
        }
        zero_temperature = compute_retarded_self_energy(**base, temperature_k=0.0)
        self.assertTrue(np.all(np.isfinite(zero_temperature.sigma_retarded_mev)))
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            compute_retarded_self_energy(**base, temperature_k=1.0)

        with self.assertRaisesRegex(ValueError, "internal_energy_mev.*exact zero"):
            compute_retarded_self_energy(
                **{
                    **base,
                    "internal_energy_mev": np.array([[0.0]]),
                    "phonon_energy_mev": np.array([[0.5]]),
                    "temperature_k": 1.0,
                }
            )

    def test_chunk_sizes_do_not_change_the_result(self):
        omega, vertex, internal, phonon, metric, weights = _random_problem(99)
        dense_blocks = compute_retarded_self_energy(
            omega,
            vertex,
            internal,
            phonon,
            temperature_k=30.0,
            broadening_mev=0.31,
            metric=metric,
            q_weights=weights,
        ).sigma_retarded_mev
        scalar_blocks = compute_retarded_self_energy(
            omega,
            vertex,
            internal,
            phonon,
            temperature_k=30.0,
            broadening_mev=0.31,
            metric=metric,
            q_weights=weights,
            q_chunk_size=1,
            omega_chunk_size=1,
        ).sigma_retarded_mev
        np.testing.assert_allclose(
            scalar_blocks, dense_blocks, rtol=2.0e-13, atol=2.0e-13
        )

    def test_direct_onshell_diagonal_matches_full_frequency_evaluation(self):
        _, vertex, internal, phonon, metric, weights = _random_problem(47)
        external = np.array([0.8, 2.3, 6.1])
        full = compute_retarded_self_energy(
            external,
            vertex,
            internal,
            phonon,
            temperature_k=42.0,
            broadening_mev=0.11,
            metric=metric,
            q_weights=weights,
            q_chunk_size=3,
            omega_chunk_size=2,
        ).sigma_retarded_mev
        direct = compute_onshell_self_energy_diagonal(
            external,
            vertex,
            internal,
            phonon,
            temperature_k=42.0,
            broadening_mev=0.11,
            metric=metric,
            q_weights=weights,
            q_chunk_size=2,
            channel_chunk_size=2,
        )
        expected = np.diagonal(full, axis1=-2, axis2=-1)[
            np.arange(external.size), np.arange(external.size)
        ]
        np.testing.assert_allclose(
            direct.sigma_diagonal_mev, expected, rtol=2.0e-13, atol=2.0e-13
        )
        np.testing.assert_allclose(
            direct.gamma_hwhm_mev,
            -np.imag(expected),
            rtol=2.0e-13,
            atol=2.0e-13,
        )

    def test_input_contract_rejects_invalid_shapes_and_controls(self):
        vertex = np.ones((2, 1, 2, 1), dtype=np.complex128)
        base = {
            "omega_mev": np.array([1.0]),
            "vertex_mev": vertex,
            "internal_energy_mev": np.array([[1.0, -1.0], [1.0, -1.0]]),
            "phonon_energy_mev": np.ones((2, 1)),
            "temperature_k": 10.0,
            "broadening_mev": 0.1,
            "metric": np.array([1.0, -1.0]),
        }
        with self.assertRaisesRegex(ValueError, "broadening_mev"):
            compute_retarded_self_energy(**{**base, "broadening_mev": 0.0})
        with self.assertRaisesRegex(ValueError, "temperature_k"):
            compute_retarded_self_energy(**{**base, "temperature_k": -0.1})
        with self.assertRaisesRegex(ValueError, "metric entries"):
            compute_retarded_self_energy(**{**base, "metric": np.array([1.0, 0.0])})
        with self.assertRaisesRegex(ValueError, "internal_energy_mev shape"):
            compute_retarded_self_energy(
                **{**base, "internal_energy_mev": np.ones((2, 1))}
            )
        with self.assertRaisesRegex(ValueError, "negative"):
            compute_retarded_self_energy(
                **{**base, "phonon_energy_mev": -np.ones((2, 1))}
            )
        with self.assertRaisesRegex(ValueError, "q_weights"):
            compute_retarded_self_energy(**base, q_weights=[1.0, -1.0])
        with self.assertRaisesRegex(ValueError, "q_chunk_size"):
            compute_retarded_self_energy(**base, q_chunk_size=0)
        with self.assertRaisesRegex(ValueError, "metric_energy_tolerance_mev"):
            compute_retarded_self_energy(**base, metric_energy_tolerance_mev=-1.0e-12)

    def test_q_weight_normalization_is_strict(self):
        np.testing.assert_allclose(normalize_q_weights([2.0, 3.0], 2), [0.4, 0.6])
        with self.assertRaisesRegex(ValueError, "shape"):
            normalize_q_weights([1.0], 2)
        with self.assertRaisesRegex(ValueError, "positive sum"):
            normalize_q_weights([0.0, 0.0], 2)


if __name__ == "__main__":
    unittest.main()
