from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from slw.magph.coupling import build_mode_resolved_isotropic_derivative
from slw.magph.derivative import ExchangeDerivativeModel
from slw.magph.model import (
    ExchangeConvention,
    ExchangeModel,
    ExchangeRepresentation,
    ExchangeSpinNormalization,
)
from slw.magph.phonon import (
    CELL_GAUGE_FOURIER_PHASE_CONVENTION,
    CELL_GAUGE_VECTOR_CONVENTION,
    PhononCache,
    PhononMassUnit,
    zero_point_displacements,
)


def _exchange() -> ExchangeModel:
    isotropic = np.asarray((2.0, 2.0))
    return ExchangeModel(
        source=Path("static.h5"),
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


def _phonons() -> PhononCache:
    # Two atoms and six orthonormal Cartesian modes on each of two q points.
    vectors = np.zeros((2, 6, 2, 3), dtype=np.complex128)
    for mode in range(6):
        vectors[:, mode, mode // 3, mode % 3] = 1.0
    return PhononCache(
        source="phonon.npz",
        schema_version=3,
        q_points_frac=np.asarray(((0.0, 0.0, 0.0), (0.5, 0.0, 0.0))),
        frequencies_mev=np.full((2, 6), 4.0),
        eigenvectors_mass_normalized=vectors,
        masses=np.ones(2),
        mass_unit=PhononMassUnit.AMU,
        q_mesh_shape=(2, 1, 1),
        vector_convention=CELL_GAUGE_VECTOR_CONVENTION,
        fourier_phase_convention=CELL_GAUGE_FOURIER_PHASE_CONVENTION,
    )


def _derivative(exchange: ExchangeModel, *, partial=False) -> ExchangeDerivativeModel:
    targets = np.asarray((0,)) if partial else np.asarray((0, 1))
    values = np.zeros((targets.size, 2, 2, 3), dtype=np.float64)
    values[0, :, 0, 0] = 1.0
    values[0, :, 1, 0] = 2.0
    if not partial:
        values[1] = -values[0]
    return ExchangeDerivativeModel(
        source=Path("dynamic.h5"),
        static_exchange_source=exchange.source,
        representation=ExchangeRepresentation.ISOTROPIC,
        source_dataset="dJ_r",
        target_atom_indices=targets,
        rp_cell_shifts=np.asarray(((0, 0, 0), (-1, 0, 0))),
        q_mesh_shape=(2, 1, 1),
        isotropic_mev_per_ang=values,
        tensor_mev_per_ang=values[..., None, None] * np.eye(3),
    )


class ModeResolvedCouplingTests(unittest.TestCase):
    def test_vectorized_fourier_and_phonon_contraction(self) -> None:
        exchange = _exchange()
        phonons = _phonons()
        zero_point = zero_point_displacements(phonons)
        result = build_mode_resolved_isotropic_derivative(
            exchange,
            _derivative(exchange),
            phonons,
            zero_point,
            q_chunk_size=1,
            bond_chunk_size=1,
        )

        amplitude = float(np.abs(zero_point.values_ang[0, 0, 0, 0]))
        # q=0: 1 + 2; q=1/2: 1 - 2. Mode zero lives on target 0.
        np.testing.assert_allclose(result.lambda_mev[0, 0], 3.0 * amplitude)
        np.testing.assert_allclose(result.lambda_mev[1, 0], -1.0 * amplitude)
        # Mode three lives on target 1 and has the opposite derivative.
        np.testing.assert_allclose(result.lambda_mev[0, 3], -3.0 * amplitude)
        np.testing.assert_allclose(result.lambda_mev[1, 3], 1.0 * amplitude)
        self.assertTrue(result.target_coverage_complete)
        self.assertFalse(result.lambda_mev.flags.writeable)

    def test_chunking_does_not_change_result(self) -> None:
        exchange = _exchange()
        phonons = _phonons()
        zero_point = zero_point_displacements(phonons)
        derivative = _derivative(exchange)
        full = build_mode_resolved_isotropic_derivative(
            exchange, derivative, phonons, zero_point
        )
        chunked = build_mode_resolved_isotropic_derivative(
            exchange,
            derivative,
            phonons,
            zero_point,
            q_chunk_size=1,
            bond_chunk_size=1,
        )
        np.testing.assert_array_equal(full.lambda_mev, chunked.lambda_mev)

    def test_partial_target_coverage_is_fail_closed_by_default(self) -> None:
        exchange = _exchange()
        phonons = _phonons()
        zero_point = zero_point_displacements(phonons)
        derivative = _derivative(exchange, partial=True)
        with self.assertRaisesRegex(ValueError, "every phonon atom"):
            build_mode_resolved_isotropic_derivative(
                exchange, derivative, phonons, zero_point
            )
        result = build_mode_resolved_isotropic_derivative(
            exchange,
            derivative,
            phonons,
            zero_point,
            require_complete_targets=False,
        )
        self.assertFalse(result.target_coverage_complete)


if __name__ == "__main__":
    unittest.main()
