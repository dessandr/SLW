from __future__ import annotations

import unittest

import numpy as np

from slw.magph.coupling import ModeResolvedExchangeDerivative
from slw.magph.lswt import uniform_fractional_mesh
from slw.magph.mesh import build_magnon_mesh_cache
from slw.magph.phonon_renormalization import compute_phonon_renormalization_grid
from slw.magph.phonon_self_energy import compute_phonon_self_energy_onshell
from slw.magph.screening import screen_magnetic_configuration
from slw.magph.self_energy import bose_occupation
from tests.magph.test_vertex import _model


class PhononSelfEnergyTests(unittest.TestCase):
    def test_normal_fm_bubble_vanishes_at_zero_temperature(self) -> None:
        result = compute_phonon_self_energy_onshell(
            [5.0],
            np.asarray([[[[2.0 + 0.0j]]]]),
            np.asarray([[1.0]]),
            np.asarray([[3.0]]),
            temperature_k=0.0,
            broadening_mev=0.2,
            metric=[1.0],
        )
        np.testing.assert_array_equal(result.sigma_onshell_mev, [0.0j])
        self.assertEqual(result.nambu_prefactor, 1.0)

    def test_normal_fm_bubble_matches_single_transition(self) -> None:
        omega = 5.0
        initial = 1.0
        final = 3.0
        coupling = 2.0
        temperature = 100.0
        broadening = 0.2
        result = compute_phonon_self_energy_onshell(
            [omega],
            np.asarray([[[[coupling + 0.0j]]]]),
            np.asarray([[initial]]),
            np.asarray([[final]]),
            temperature_k=temperature,
            broadening_mev=broadening,
            metric=[1.0],
        )
        expected = (
            coupling**2
            * (
                float(bose_occupation(initial, temperature))
                - float(bose_occupation(final, temperature))
            )
            / (omega + 1.0j * broadening + initial - final)
        )
        np.testing.assert_allclose(result.sigma_onshell_mev, [expected])

    def test_bdg_pair_creation_has_half_nambu_factor(self) -> None:
        omega = 6.0
        magnon = 2.0
        coupling = 1.7
        broadening = 0.3
        vertex = np.zeros((1, 1, 2, 2), dtype=np.complex128)
        vertex[0, 0, 0, 1] = coupling
        vertex[0, 0, 1, 0] = coupling
        result = compute_phonon_self_energy_onshell(
            [omega],
            vertex,
            np.asarray([[magnon, -magnon]]),
            np.asarray([[magnon, -magnon]]),
            temperature_k=0.0,
            broadening_mev=broadening,
            metric=[1.0, -1.0],
        )
        expected = 0.5 * coupling**2 * (
            1.0 / (omega + 1.0j * broadening - 2.0 * magnon)
            - 1.0 / (omega + 1.0j * broadening + 2.0 * magnon)
        )
        np.testing.assert_allclose(result.sigma_onshell_mev, [expected])
        self.assertEqual(result.nambu_prefactor, 0.5)
        self.assertGreater(result.gamma_hwhm_mev[0], 0.0)

    def test_k_weights_and_chunks_do_not_change_result(self) -> None:
        rng = np.random.default_rng(773)
        nk, nmode, nchannel = 5, 4, 2
        vertex = rng.normal(size=(nk, nmode, nchannel, nchannel))
        vertex = vertex + 1.0j * rng.normal(size=vertex.shape)
        metric = np.asarray((1.0, -1.0))
        positive_initial = rng.uniform(0.5, 8.0, size=(nk, 1))
        positive_final = rng.uniform(0.5, 8.0, size=(nk, 1))
        initial = np.concatenate((positive_initial, -positive_initial), axis=1)
        final = np.concatenate((positive_final, -positive_final), axis=1)
        phonon = np.linspace(2.0, 9.0, nmode)
        weights = np.asarray((1.0, 2.0, 4.0, 3.0, 0.5))
        dense = compute_phonon_self_energy_onshell(
            phonon,
            vertex,
            initial,
            final,
            temperature_k=50.0,
            broadening_mev=0.15,
            metric=metric,
            k_weights=weights,
        )
        chunked = compute_phonon_self_energy_onshell(
            phonon,
            vertex,
            initial,
            final,
            temperature_k=50.0,
            broadening_mev=0.15,
            metric=metric,
            k_weights=weights,
            mode_chunk_size=1,
            channel_chunk_size=1,
        )
        np.testing.assert_allclose(chunked.sigma_onshell_mev, dense.sigma_onshell_mev)
        np.testing.assert_allclose(chunked.k_weights, weights / np.sum(weights))

    def test_finite_temperature_goldstone_and_metric_mismatch_fail_closed(self) -> None:
        base = {
            "phonon_energy_mev": [4.0],
            "vertex_mev": np.ones((1, 1, 2, 2), dtype=np.complex128),
            "initial_magnon_energy_mev": np.asarray([[0.0, -1.0]]),
            "final_magnon_energy_mev": np.asarray([[1.0, -1.0]]),
            "temperature_k": 20.0,
            "broadening_mev": 0.1,
            "metric": [1.0, -1.0],
        }
        with self.assertRaisesRegex(ValueError, "exact zero"):
            compute_phonon_self_energy_onshell(**base)

        with self.assertRaisesRegex(ValueError, "signed BdG metric"):
            compute_phonon_self_energy_onshell(
                **{
                    **base,
                    "initial_magnon_energy_mev": np.asarray([[1.0, 1.0e-3]]),
                    "temperature_k": 0.0,
                }
            )

    def test_zero_exchange_striction_preserves_bare_phonons_end_to_end(self) -> None:
        exchange = _model(afm=True)
        configuration = screen_magnetic_configuration(
            exchange,
            order="collinear_afm",
            spin_magnitudes=1.0,
        )
        q_points = np.asarray(((0.0, 0.0, 0.0), (0.5, 0.0, 0.0)))
        coupling = ModeResolvedExchangeDerivative(
            q_points_frac=q_points,
            phonon_energy_mev=np.full((2, 1), 5.0),
            lambda_mev=np.zeros((2, 1, exchange.n_bonds), dtype=np.complex128),
            target_atom_indices=np.asarray((0,)),
            target_coverage_complete=True,
            frequency_regularized=np.zeros((2, 1), dtype=bool),
            q_mesh_shape=(2, 1, 1),
        )
        k_points = uniform_fractional_mesh((2, 1, 1), shift=(0.5, 0.0, 0.0))
        cache = build_magnon_mesh_cache(
            exchange,
            configuration,
            coupling,
            k_points,
            k_mesh_shape=(2, 1, 1),
            kshift_grid=(0.5, 0.0, 0.0),
        )
        progress: list[tuple[int, int]] = []
        distributed = compute_phonon_renormalization_grid(
            exchange,
            configuration,
            coupling,
            cache,
            np.full((2, 1), 5.0),
            temperature_k=0.0,
            broadening_mev=0.1,
            q_chunk_size=1,
            mode_chunk_size=1,
            progress=lambda completed, total: progress.append((completed, total)),
        )
        self.assertEqual(progress, [(1, 2), (2, 2)])
        self.assertIsNotNone(distributed.global_result)
        result = distributed.global_result
        assert result is not None
        np.testing.assert_array_equal(result.self_energy_onshell_mev, 0.0)
        np.testing.assert_array_equal(result.renormalized_frequency_mev, 5.0)
        np.testing.assert_array_equal(result.frequency_shift_mev, 0.0)
        np.testing.assert_array_equal(result.dynamically_stable, True)
        np.testing.assert_array_equal(result.valid_renormalization, True)

    def test_afm_pair_channel_softens_a_finite_q_phonon_at_zero_temperature(
        self,
    ) -> None:
        exchange = _model(afm=True)
        configuration = screen_magnetic_configuration(
            exchange,
            order="collinear_afm",
            spin_magnitudes=1.0,
        )
        q_points = np.asarray(((0.0, 0.0, 0.0), (0.5, 0.0, 0.0)))
        coupling = ModeResolvedExchangeDerivative(
            q_points_frac=q_points,
            phonon_energy_mev=np.full((2, 1), 5.0),
            lambda_mev=np.full(
                (2, 1, exchange.n_bonds), 0.2, dtype=np.complex128
            ),
            target_atom_indices=np.asarray((0,)),
            target_coverage_complete=True,
            frequency_regularized=np.zeros((2, 1), dtype=bool),
            q_mesh_shape=(2, 1, 1),
        )
        k_points = uniform_fractional_mesh((2, 1, 1), shift=(0.5, 0.0, 0.0))
        cache = build_magnon_mesh_cache(
            exchange,
            configuration,
            coupling,
            k_points,
            k_mesh_shape=(2, 1, 1),
            kshift_grid=(0.5, 0.0, 0.0),
        )
        distributed = compute_phonon_renormalization_grid(
            exchange,
            configuration,
            coupling,
            cache,
            np.full((2, 1), 5.0),
            temperature_k=0.0,
            broadening_mev=0.1,
            q_chunk_size=1,
        )
        result = distributed.global_result
        assert result is not None
        self.assertLess(result.frequency_shift_mev[1, 0], 0.0)
        self.assertGreater(result.gamma_hwhm_mev[1, 0], 0.0)
        self.assertTrue(result.valid_renormalization[1, 0])


if __name__ == "__main__":
    unittest.main()
