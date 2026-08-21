from __future__ import annotations

import concurrent.futures
import unittest
from unittest import mock

import numpy as np

from slw.cli.mpi import MPIContext
from slw.magph.coupling import ModeResolvedExchangeDerivative
from slw.magph.lswt import solve_isotropic_lswt, uniform_fractional_mesh
from slw.magph.mesh import build_magnon_mesh_cache
from slw.magph.screening import screen_magnetic_configuration
from tests.magph.test_parallel import _CollectiveState, _ThreadComm
from tests.magph.test_vertex import _model


def _coupling(*, afm: bool) -> ModeResolvedExchangeDerivative:
    bonds = 4 if afm else 2
    return ModeResolvedExchangeDerivative(
        q_points_frac=np.asarray(((0.0, 0.0, 0.0), (0.5, 0.0, 0.0))),
        phonon_energy_mev=np.full((2, 1), 5.0),
        lambda_mev=np.full((2, 1, bonds), 0.1, dtype=np.complex128),
        target_atom_indices=np.asarray((0, 1) if afm else (0,)),
        target_coverage_complete=True,
        frequency_regularized=np.zeros((2, 1), dtype=bool),
        q_mesh_shape=(2, 1, 1),
    )


class MagnonMeshCacheTests(unittest.TestCase):
    def _build(self, *, afm: bool):
        exchange = _model(afm=afm)
        configuration = screen_magnetic_configuration(
            exchange,
            order="collinear_afm" if afm else "fm",
            spin_magnitudes=1.0,
        )
        k_points = uniform_fractional_mesh((4, 1, 1), shift=(0.5, 0.0, 0.0))
        return exchange, configuration, k_points, _coupling(afm=afm)

    def test_union_mapping_matches_explicit_k_plus_q(self) -> None:
        exchange, configuration, k_points, coupling = self._build(afm=False)
        cache = build_magnon_mesh_cache(
            exchange,
            configuration,
            coupling,
            k_points,
            k_mesh_shape=(4, 1, 1),
            kshift_grid=(0.5, 0.0, 0.0),
            context=MPIContext(),
        )

        self.assertEqual(cache.union_mesh_shape, (4, 1, 1))
        for ik, k_point in enumerate(k_points):
            indices = cache.internal_union_indices(ik)
            expected = np.mod(k_point[None, :] + coupling.q_points_frac, 1.0)
            np.testing.assert_allclose(cache.union_points_frac[indices], expected)

    def test_union_lswt_is_solved_once_and_matches_direct_afm(self) -> None:
        exchange, configuration, k_points, coupling = self._build(afm=True)
        with mock.patch(
            "slw.magph.mesh.solve_isotropic_lswt",
            wraps=solve_isotropic_lswt,
        ) as solve:
            cache = build_magnon_mesh_cache(
                exchange,
                configuration,
                coupling,
                k_points,
                k_mesh_shape=(4, 1, 1),
                kshift_grid=(0.5, 0.0, 0.0),
                context=MPIContext(),
            )

        self.assertEqual(solve.call_count, 1)
        direct = solve_isotropic_lswt(exchange, configuration, cache.union_points_frac)
        np.testing.assert_allclose(
            cache.signed_energies_mev, direct.signed_energies_mev
        )
        np.testing.assert_allclose(cache.transformation, direct.transformation)

    def test_two_ranks_distribute_then_broadcast_identical_union_cache(self) -> None:
        exchange, configuration, k_points, coupling = self._build(afm=False)
        state = _CollectiveState(2)

        def run_rank(rank: int):
            return build_magnon_mesh_cache(
                exchange,
                configuration,
                coupling,
                k_points,
                k_mesh_shape=(4, 1, 1),
                kshift_grid=(0.5, 0.0, 0.0),
                context=MPIContext(
                    comm=_ThreadComm(state, rank),
                    rank=rank,
                    size=2,
                ),
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(run_rank, rank) for rank in range(2)]
            caches = [future.result(timeout=5.0) for future in futures]
        np.testing.assert_array_equal(
            caches[0].signed_energies_mev,
            caches[1].signed_energies_mev,
        )
        np.testing.assert_array_equal(
            caches[0].transformation,
            caches[1].transformation,
        )


if __name__ == "__main__":
    unittest.main()
