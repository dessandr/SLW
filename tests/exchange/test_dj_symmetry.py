from __future__ import annotations

import unittest

import numpy as np

from slw.exchange.kernels.dj_symmetry import apply_covariant_derivative_symmetry


def _c4_rotations() -> np.ndarray:
    return np.asarray(
        (
            ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
            ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
            ((-1, 0, 0), (0, -1, 0), (0, 0, 1)),
            ((0, 1, 0), (-1, 0, 0), (0, 0, 1)),
        ),
        dtype=np.int64,
    )


class CovariantDerivativeSymmetryTests(unittest.TestCase):
    def _project(self, values: np.ndarray, *, policy: str = "project"):
        rotations = _c4_rotations()
        return apply_covariant_derivative_symmetry(
            values,
            bond_i_atom=np.zeros(4, dtype=np.int64),
            bond_j_atom=np.zeros(4, dtype=np.int64),
            bond_cell_shifts=np.asarray(
                ((1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0)),
                dtype=np.int64,
            ),
            target_atom_indices=np.asarray((0,), dtype=np.int64),
            rp_cell_shifts=np.zeros((1, 3), dtype=np.int64),
            q_mesh_shape=(1, 1, 1),
            lattice_ang=np.eye(3),
            positions_frac=np.zeros((1, 3)),
            species_numbers=np.asarray((1,), dtype=np.int32),
            rotations=rotations,
            translations=np.zeros((rotations.shape[0], 3)),
            policy=policy,
            tolerance_mev_per_ang=1.0e-10,
            spacegroup="P4 test",
            species_source="test",
        )

    def test_projection_rotates_vector_and_bond_together(self) -> None:
        values = np.zeros((1, 4, 1, 3), dtype=np.float64)
        values[0, :, 0] = np.asarray(
            ((2.0, 0.0, 0.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, -1.0, 0.0))
        )

        result = self._project(values)

        np.testing.assert_allclose(
            result.values_mev_per_ang[0, :, 0],
            1.25
            * np.asarray(
                ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, -1.0, 0.0))
            ),
        )
        self.assertTrue(result.applied)
        self.assertEqual(result.n_operations, 4)
        self.assertEqual(result.spacegroup, "P4 test")
        self.assertAlmostEqual(result.max_abs_residual_mev_per_ang, 0.75)

        second = self._project(result.values_mev_per_ang)
        np.testing.assert_allclose(second.values_mev_per_ang, result.values_mev_per_ang)
        self.assertLess(second.max_abs_residual_mev_per_ang, 1.0e-14)

    def test_fail_policy_rejects_covariance_residual(self) -> None:
        values = np.zeros((1, 4, 1, 3), dtype=np.float64)
        values[0, 0, 0, 0] = 1.0
        with self.assertRaisesRegex(ValueError, "violates space-group covariance"):
            self._project(values, policy="fail")

    def test_rp_cell_and_polar_vector_transform_under_inversion(self) -> None:
        rotations = np.asarray((np.eye(3), -np.eye(3)), dtype=np.int64)
        values = np.zeros((1, 2, 3, 3), dtype=np.float64)
        values[0, 0, :, 0] = (1.0, 2.0, 3.0)
        values[0, 1, :, 0] = (-1.0, -3.0, -2.0)

        result = apply_covariant_derivative_symmetry(
            values,
            bond_i_atom=np.zeros(2, dtype=np.int64),
            bond_j_atom=np.zeros(2, dtype=np.int64),
            bond_cell_shifts=np.asarray(((1, 0, 0), (-1, 0, 0))),
            target_atom_indices=np.asarray((0,)),
            rp_cell_shifts=np.asarray(((0, 0, 0), (1, 0, 0), (-1, 0, 0))),
            q_mesh_shape=(3, 1, 1),
            lattice_ang=np.eye(3),
            positions_frac=np.zeros((1, 3)),
            species_numbers=np.asarray((1,)),
            rotations=rotations,
            translations=np.zeros((2, 3)),
            policy="project",
        )

        np.testing.assert_allclose(result.values_mev_per_ang, values)
        self.assertLess(result.max_abs_residual_mev_per_ang, 1.0e-14)

    def test_nonclosed_bond_range_is_rejected(self) -> None:
        rotations = _c4_rotations()
        with self.assertRaisesRegex(ValueError, "bond set is not closed"):
            apply_covariant_derivative_symmetry(
                np.zeros((1, 2, 1, 3)),
                bond_i_atom=np.zeros(2, dtype=np.int64),
                bond_j_atom=np.zeros(2, dtype=np.int64),
                bond_cell_shifts=np.asarray(((1, 0, 0), (-1, 0, 0))),
                target_atom_indices=np.asarray((0,)),
                rp_cell_shifts=np.zeros((1, 3), dtype=np.int64),
                q_mesh_shape=(1, 1, 1),
                lattice_ang=np.eye(3),
                positions_frac=np.zeros((1, 3)),
                species_numbers=np.asarray((1,)),
                rotations=rotations,
                translations=np.zeros((rotations.shape[0], 3)),
                policy="project",
            )


if __name__ == "__main__":
    unittest.main()
