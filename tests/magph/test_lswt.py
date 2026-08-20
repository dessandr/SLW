from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from slw.magph.lswt import solve_isotropic_lswt, uniform_fractional_mesh
from slw.magph.model import (
    ExchangeConvention,
    ExchangeModel,
    ExchangeRepresentation,
    ExchangeSpinNormalization,
)
from slw.magph.screening import screen_magnetic_configuration


def _fm_chain(j_mev=2.0) -> ExchangeModel:
    isotropic = np.full(2, float(j_mev))
    return ExchangeModel(
        source=Path("fm.h5"),
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


def _afm_chain(j_mev=-2.0) -> ExchangeModel:
    isotropic = np.full(4, float(j_mev))
    return ExchangeModel(
        source=Path("afm.h5"),
        representation=ExchangeRepresentation.ISOTROPIC,
        source_dataset="J_r/value",
        convention=ExchangeConvention(
            spin_normalization=ExchangeSpinNormalization.UNIT_VECTOR,
            source_spin_magnitude=1.0,
            kernel_family="scalar_lkag",
        ),
        isotropic_mev=isotropic,
        tensor_mev=isotropic[:, None, None] * np.eye(3),
        bond_i=np.asarray((0, 1, 0, 1)),
        bond_j=np.asarray((1, 0, 1, 0)),
        bond_i_atom=np.asarray((0, 1, 0, 1)),
        bond_j_atom=np.asarray((1, 0, 1, 0)),
        cell_shift=np.asarray(((0, 0, 0), (0, 0, 0), (-1, 0, 0), (1, 0, 0))),
        magnetic_atom_indices=np.asarray((0, 1)),
        mirror_index=np.asarray((1, 0, 3, 2)),
    )


class NativeLSWTTests(unittest.TestCase):
    def test_uniform_mesh_is_vectorized_and_explicitly_shifted(self) -> None:
        mesh = uniform_fractional_mesh((2, 2, 1), shift=(0.5, 0.0, 0.0))
        self.assertEqual(mesh.shape, (4, 3))
        np.testing.assert_allclose(np.unique(mesh[:, 0]), (0.25, 0.75))
        np.testing.assert_allclose(np.unique(mesh[:, 1]), (0.0, 0.5))
        self.assertFalse(mesh.flags.writeable)

    def test_unit_vector_fm_chain_matches_analytic_dispersion(self) -> None:
        exchange = _fm_chain(j_mev=2.0)
        configuration = screen_magnetic_configuration(
            exchange,
            order="fm",
            spin_magnitudes=2.0,
        )
        points = np.asarray(((0.0, 0.0, 0.0), (0.25, 0.0, 0.0), (0.5, 0.0, 0.0)))
        spectrum = solve_isotropic_lswt(exchange, configuration, points)
        # omega(k)=2 J/S [1-cos(2*pi*k)] for the mate-complete chain.
        expected = (2.0 * 2.0 / 2.0) * (1.0 - np.cos(2.0 * np.pi * points[:, 0]))
        np.testing.assert_allclose(spectrum.physical_energies_mev[:, 0], expected)
        np.testing.assert_allclose(spectrum.metric, (1.0,))
        self.assertLess(spectrum.max_eigen_residual_mev, 1.0e-12)

    def test_bipartite_afm_has_signed_paraunitary_channels(self) -> None:
        exchange = _afm_chain(j_mev=-2.0)
        configuration = screen_magnetic_configuration(
            exchange,
            order="collinear_afm",
            spin_magnitudes=(1.0, 1.0),
            quantization_axis=(0.0, 0.0, 1.0),
        )
        points = np.asarray(((0.25, 0.0, 0.0), (0.2, 0.0, 0.0)))
        spectrum = solve_isotropic_lswt(exchange, configuration, points)
        expected = 2.0 * abs(-2.0) * np.abs(np.sin(np.pi * points[:, 0]))
        np.testing.assert_allclose(
            spectrum.signed_energies_mev[:, :2], np.repeat(expected[:, None], 2, axis=1)
        )
        np.testing.assert_allclose(
            spectrum.signed_energies_mev[:, 2:],
            -np.repeat(expected[:, None], 2, axis=1),
        )
        np.testing.assert_array_equal(spectrum.metric, (1.0, 1.0, -1.0, -1.0))
        for transform in spectrum.transformation:
            residual = transform.conj().T @ (spectrum.metric[:, None] * transform)
            np.testing.assert_allclose(residual, np.diag(spectrum.metric), atol=1.0e-12)
        self.assertLess(spectrum.max_paraunitary_residual, 1.0e-12)

    def test_unstable_declared_fm_is_rejected_before_or_during_lswt(self) -> None:
        exchange = _fm_chain(j_mev=-2.0)
        with self.assertRaisesRegex(ValueError, "locally stationary/stable"):
            screen_magnetic_configuration(
                exchange,
                order="fm",
                spin_magnitudes=1.0,
            )

    def test_exact_afm_goldstone_point_fails_with_actionable_message(self) -> None:
        exchange = _afm_chain()
        configuration = screen_magnetic_configuration(
            exchange,
            order="collinear_afm",
            spin_magnitudes=1.0,
        )
        with self.assertRaisesRegex(ValueError, "Goldstone.*shifted mesh"):
            solve_isotropic_lswt(
                exchange,
                configuration,
                np.asarray(((0.0, 0.0, 0.0),)),
            )


if __name__ == "__main__":
    unittest.main()
