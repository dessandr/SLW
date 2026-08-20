from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from slw.cli.mpi import MPIContext
from slw.magph.coupling import ModeResolvedExchangeDerivative
from slw.magph.model import (
    ExchangeConvention,
    ExchangeModel,
    ExchangeRepresentation,
    ExchangeSpinNormalization,
)
from slw.magph.pipeline import (
    compute_lifetime_grid,
    evaluate_scattering_problem,
)
from slw.magph.screening import screen_magnetic_configuration
from slw.magph.vertex import build_scattering_problem


def _fm_chain() -> ExchangeModel:
    isotropic = np.asarray((2.0, 2.0))
    return ExchangeModel(
        source=Path("exchange.h5"),
        representation=ExchangeRepresentation.ISOTROPIC,
        source_dataset="J_r/value",
        convention=ExchangeConvention(
            spin_normalization=ExchangeSpinNormalization.UNIT_VECTOR,
            source_spin_magnitude=1.0,
            kernel_family="scalar_lkag",
        ),
        isotropic_mev=isotropic,
        tensor_mev=isotropic[:, None, None] * np.eye(3),
        bond_i=np.asarray((0, 0)),
        bond_j=np.asarray((0, 0)),
        bond_i_atom=np.asarray((0, 0)),
        bond_j_atom=np.asarray((0, 0)),
        cell_shift=np.asarray(((1, 0, 0), (-1, 0, 0))),
        magnetic_atom_indices=np.asarray((0,)),
        mirror_index=np.asarray((1, 0)),
    )


def _mode_coupling(value: float) -> ModeResolvedExchangeDerivative:
    q_points = np.asarray(((0.25, 0.0, 0.0), (0.5, 0.0, 0.0)))
    lambda_mev = np.full((2, 1, 2), value, dtype=np.complex128)
    return ModeResolvedExchangeDerivative(
        q_points_frac=q_points,
        phonon_energy_mev=np.full((2, 1), 5.0),
        lambda_mev=lambda_mev,
        target_atom_indices=np.asarray((0,)),
        target_coverage_complete=True,
        frequency_regularized=np.zeros((2, 1), dtype=bool),
    )


class NativeLifetimePipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.exchange = _fm_chain()
        self.configuration = screen_magnetic_configuration(
            self.exchange,
            order="fm",
            spin_magnitudes=1.0,
        )

    def test_one_k_problem_runs_through_self_energy_and_lifetime(self) -> None:
        problem = build_scattering_problem(
            self.exchange,
            self.configuration,
            _mode_coupling(0.1),
            external_k_frac=(0.125, 0.0, 0.0),
            q_chunk_size=1,
        )
        result = evaluate_scattering_problem(
            problem,
            temperature_k=0.0,
            broadening_mev=0.2,
            q_chunk_size=1,
            channel_chunk_size=1,
        )
        self.assertEqual(result.physical_sigma_mev.shape, (1,))
        self.assertTrue(np.all(np.isfinite(result.physical_sigma_mev)))
        self.assertTrue(np.all(result.lifetime.valid_damping))
        self.assertTrue(np.all(result.lifetime.gamma_hwhm_mev >= 0.0))

    def test_serial_grid_is_root_assembled_and_zero_vertex_is_infinite_lifetime(
        self,
    ) -> None:
        k_points = np.asarray(((0.125, 0.0, 0.0), (0.25, 0.0, 0.0), (0.375, 0.0, 0.0)))
        distributed = compute_lifetime_grid(
            self.exchange,
            self.configuration,
            _mode_coupling(0.0),
            k_points,
            temperature_k=0.0,
            broadening_mev=0.2,
            vertex_q_chunk_size=1,
            self_energy_q_chunk_size=1,
            channel_chunk_size=1,
            context=MPIContext(),
        )
        self.assertEqual(distributed.mpi_size, 1)
        np.testing.assert_array_equal(distributed.local_indices, (0, 1, 2))
        self.assertIsNotNone(distributed.global_result)
        assert distributed.global_result is not None
        result = distributed.global_result
        self.assertEqual(result.energy_mev.shape, (3, 1))
        np.testing.assert_allclose(result.self_energy_onshell_mev, 0.0)
        np.testing.assert_allclose(result.gamma_hwhm_mev, 0.0)
        np.testing.assert_allclose(result.scattering_rate_ps_inv, 0.0)
        self.assertTrue(np.all(np.isinf(result.lifetime_ps)))
        self.assertFalse(result.energy_mev.flags.writeable)


if __name__ == "__main__":
    unittest.main()
