from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from slw.magph.coupling import ModeResolvedExchangeDerivative
from slw.magph.lswt import _quadratic_blocks
from slw.magph.model import (
    ExchangeConvention,
    ExchangeModel,
    ExchangeRepresentation,
    ExchangeSpinNormalization,
)
from slw.magph.screening import screen_magnetic_configuration
from slw.magph.vertex import (
    build_bare_isotropic_vertex,
    build_scattering_problem,
)


def _model(*, afm=False, values=None) -> ExchangeModel:
    if afm:
        default = np.full(4, -2.0)
        bond_i = np.asarray((0, 1, 0, 1))
        bond_j = np.asarray((1, 0, 1, 0))
        shifts = np.asarray(((0, 0, 0), (0, 0, 0), (-1, 0, 0), (1, 0, 0)))
        mirror = np.asarray((1, 0, 3, 2))
        atoms = np.asarray((0, 1))
    else:
        default = np.full(2, 2.0)
        bond_i = np.asarray((0, 0))
        bond_j = np.asarray((0, 0))
        shifts = np.asarray(((1, 0, 0), (-1, 0, 0)))
        mirror = np.asarray((1, 0))
        atoms = np.asarray((0,))
    isotropic = default if values is None else np.asarray(values, dtype=np.float64)
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
        bond_i=bond_i,
        bond_j=bond_j,
        bond_i_atom=bond_i,
        bond_j_atom=bond_j,
        cell_shift=shifts,
        magnetic_atom_indices=atoms,
        mirror_index=mirror,
    )


def _coupling(lambda_bonds, q_points=((0.0, 0.0, 0.0),)):
    q_points = np.asarray(q_points, dtype=np.float64)
    values = np.broadcast_to(
        np.asarray(lambda_bonds, dtype=np.complex128)[None, None, :],
        (q_points.shape[0], 1, len(lambda_bonds)),
    ).copy()
    return ModeResolvedExchangeDerivative(
        q_points_frac=q_points,
        phonon_energy_mev=np.full((q_points.shape[0], 1), 5.0),
        lambda_mev=values,
        target_atom_indices=np.asarray((0,)),
        target_coverage_complete=True,
        frequency_regularized=np.zeros((q_points.shape[0], 1), dtype=bool),
    )


def _finite_difference_model(base: ExchangeModel, delta, epsilon, sign):
    values = base.isotropic_mev + sign * epsilon * np.asarray(delta)
    return _model(afm=base.n_magnetic_sites == 2, values=values)


def _bdg_matrix(exchange, configuration, k):
    points = np.asarray((k,), dtype=np.float64)
    normal, pairing = _quadratic_blocks(exchange, configuration, points)
    normal_minus, pairing_minus = _quadratic_blocks(exchange, configuration, -points)
    return np.block(
        [
            [normal[0], pairing[0]],
            [pairing_minus[0].conj(), normal_minus[0].conj()],
        ]
    )


class NativeVertexTests(unittest.TestCase):
    def test_fm_q0_vertex_is_static_lswt_finite_difference(self) -> None:
        exchange = _model()
        configuration = screen_magnetic_configuration(
            exchange, order="fm", spin_magnitudes=2.0
        )
        delta = np.asarray((0.3, 0.3))
        k = np.asarray((0.23, 0.0, 0.0))
        bare = build_bare_isotropic_vertex(
            exchange, configuration, _coupling(delta), k
        )[0, 0]
        epsilon = 1.0e-6
        plus = _finite_difference_model(exchange, delta, epsilon, 1.0)
        minus = _finite_difference_model(exchange, delta, epsilon, -1.0)
        normal_plus, _ = _quadratic_blocks(plus, configuration, k[None, :])
        normal_minus, _ = _quadratic_blocks(minus, configuration, k[None, :])
        finite_difference = (normal_plus[0] - normal_minus[0]) / (2.0 * epsilon)
        np.testing.assert_allclose(bare, finite_difference, rtol=1.0e-8, atol=1.0e-10)

    def test_afm_q0_vertex_is_static_bdg_finite_difference(self) -> None:
        exchange = _model(afm=True)
        configuration = screen_magnetic_configuration(
            exchange, order="collinear_afm", spin_magnitudes=1.0
        )
        delta = np.full(4, 0.2)
        k = np.asarray((0.23, 0.0, 0.0))
        bare = build_bare_isotropic_vertex(
            exchange, configuration, _coupling(delta), k
        )[0, 0]
        epsilon = 1.0e-6
        plus = _finite_difference_model(exchange, delta, epsilon, 1.0)
        minus = _finite_difference_model(exchange, delta, epsilon, -1.0)
        finite_difference = (
            _bdg_matrix(plus, configuration, k)
            - _bdg_matrix(minus, configuration, k)
        ) / (2.0 * epsilon)
        np.testing.assert_allclose(bare, finite_difference, rtol=1.0e-8, atol=1.0e-10)

    def test_band_problem_has_self_energy_array_contract(self) -> None:
        exchange = _model()
        configuration = screen_magnetic_configuration(
            exchange, order="fm", spin_magnitudes=1.0
        )
        coupling = _coupling(
            (0.1, 0.1),
            q_points=((0.0, 0.0, 0.0), (0.25, 0.0, 0.0)),
        )
        problem = build_scattering_problem(
            exchange,
            configuration,
            coupling,
            external_k_frac=(0.25, 0.0, 0.0),
            q_chunk_size=1,
        )
        self.assertEqual(problem.vertex_mev.shape, (2, 1, 1, 1))
        self.assertEqual(problem.internal_energy_mev.shape, (2, 1))
        np.testing.assert_array_equal(problem.metric, (1.0,))
        self.assertFalse(problem.vertex_mev.flags.writeable)


if __name__ == "__main__":
    unittest.main()
