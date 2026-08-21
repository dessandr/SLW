from __future__ import annotations

import concurrent.futures
import unittest
from pathlib import Path
from unittest import mock

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
from tests.magph.test_parallel import _CollectiveState, _ThreadComm
from tests.magph.test_vertex import _model


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

    def test_streaming_fm_matches_materialized_vertex_route(self) -> None:
        coupling = ModeResolvedExchangeDerivative(
            q_points_frac=np.asarray(((0.0, 0.0, 0.0), (0.5, 0.0, 0.0))),
            phonon_energy_mev=np.full((2, 1), 5.0),
            lambda_mev=np.full((2, 1, 2), 0.1, dtype=np.complex128),
            target_atom_indices=np.asarray((0,)),
            target_coverage_complete=True,
            frequency_regularized=np.zeros((2, 1), dtype=bool),
            q_mesh_shape=(2, 1, 1),
        )
        k_points = np.column_stack(
            (((np.arange(4) + 0.5) / 4.0), np.zeros(4), np.zeros(4))
        )
        common = {
            "temperature_k": 0.0,
            "broadening_mev": 0.2,
            "vertex_q_chunk_size": 1,
            "self_energy_q_chunk_size": 1,
            "channel_chunk_size": 1,
            "context": MPIContext(),
        }
        materialized = compute_lifetime_grid(
            self.exchange,
            self.configuration,
            coupling,
            k_points,
            **common,
        )
        with mock.patch(
            "slw.magph.pipeline.build_scattering_problem",
            side_effect=AssertionError("streaming route materialized a full vertex"),
        ):
            streaming = compute_lifetime_grid(
                self.exchange,
                self.configuration,
                coupling,
                k_points,
                k_mesh_shape=(4, 1, 1),
                kshift_grid=(0.5, 0.0, 0.0),
                **common,
            )
        assert materialized.global_result is not None
        assert streaming.global_result is not None
        np.testing.assert_allclose(
            streaming.global_result.energy_mev,
            materialized.global_result.energy_mev,
        )
        np.testing.assert_allclose(
            streaming.global_result.self_energy_onshell_mev,
            materialized.global_result.self_energy_onshell_mev,
            rtol=1.0e-12,
            atol=1.0e-12,
        )

    def test_streaming_afm_matches_materialized_vertex_route(self) -> None:
        exchange = _model(afm=True)
        configuration = screen_magnetic_configuration(
            exchange,
            order="collinear_afm",
            spin_magnitudes=1.0,
        )
        coupling = ModeResolvedExchangeDerivative(
            q_points_frac=np.asarray(((0.0, 0.0, 0.0), (0.5, 0.0, 0.0))),
            phonon_energy_mev=np.full((2, 1), 5.0),
            lambda_mev=np.full((2, 1, 4), 0.1, dtype=np.complex128),
            target_atom_indices=np.asarray((0, 1)),
            target_coverage_complete=True,
            frequency_regularized=np.zeros((2, 1), dtype=bool),
            q_mesh_shape=(2, 1, 1),
        )
        k_points = np.column_stack(
            (((np.arange(4) + 0.5) / 4.0), np.zeros(4), np.zeros(4))
        )
        common = {
            "temperature_k": 0.0,
            "broadening_mev": 0.2,
            "vertex_q_chunk_size": 1,
            "self_energy_q_chunk_size": 1,
            "channel_chunk_size": 1,
            "context": MPIContext(),
        }
        materialized = compute_lifetime_grid(
            exchange,
            configuration,
            coupling,
            k_points,
            **common,
        )
        streaming = compute_lifetime_grid(
            exchange,
            configuration,
            coupling,
            k_points,
            k_mesh_shape=(4, 1, 1),
            kshift_grid=(0.5, 0.0, 0.0),
            **common,
        )
        assert materialized.global_result is not None
        assert streaming.global_result is not None
        np.testing.assert_allclose(
            streaming.global_result.energy_mev,
            materialized.global_result.energy_mev,
        )
        np.testing.assert_allclose(
            streaming.global_result.self_energy_onshell_mev,
            materialized.global_result.self_energy_onshell_mev,
            rtol=1.0e-11,
            atol=1.0e-11,
        )

    def test_two_rank_streaming_pipeline_matches_serial(self) -> None:
        coupling = ModeResolvedExchangeDerivative(
            q_points_frac=np.asarray(((0.0, 0.0, 0.0), (0.5, 0.0, 0.0))),
            phonon_energy_mev=np.full((2, 1), 5.0),
            lambda_mev=np.full((2, 1, 2), 0.1, dtype=np.complex128),
            target_atom_indices=np.asarray((0,)),
            target_coverage_complete=True,
            frequency_regularized=np.zeros((2, 1), dtype=bool),
            q_mesh_shape=(2, 1, 1),
        )
        k_points = np.column_stack(
            (((np.arange(4) + 0.5) / 4.0), np.zeros(4), np.zeros(4))
        )
        common = {
            "temperature_k": 0.0,
            "broadening_mev": 0.2,
            "vertex_q_chunk_size": 1,
            "self_energy_q_chunk_size": 1,
            "channel_chunk_size": 1,
            "k_mesh_shape": (4, 1, 1),
            "kshift_grid": (0.5, 0.0, 0.0),
        }
        serial = compute_lifetime_grid(
            self.exchange,
            self.configuration,
            coupling,
            k_points,
            context=MPIContext(),
            **common,
        )
        state = _CollectiveState(2)

        def run_rank(rank: int):
            return compute_lifetime_grid(
                self.exchange,
                self.configuration,
                coupling,
                k_points,
                context=MPIContext(
                    comm=_ThreadComm(state, rank),
                    rank=rank,
                    size=2,
                ),
                **common,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(run_rank, rank) for rank in range(2)]
            distributed = [future.result(timeout=5.0) for future in futures]

        assert serial.global_result is not None
        assert distributed[0].global_result is not None
        self.assertIsNone(distributed[1].global_result)
        np.testing.assert_allclose(
            distributed[0].global_result.energy_mev,
            serial.global_result.energy_mev,
        )
        np.testing.assert_allclose(
            distributed[0].global_result.self_energy_onshell_mev,
            serial.global_result.self_energy_onshell_mev,
        )
        np.testing.assert_array_equal(distributed[0].local_indices, (0, 1))
        np.testing.assert_array_equal(distributed[1].local_indices, (2, 3))


if __name__ == "__main__":
    unittest.main()
