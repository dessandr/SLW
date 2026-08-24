from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from slw.magph.derivative import (
    DerivativeASRPolicy,
    load_exchange_derivative_h5,
)
from slw.magph.model import (
    ExchangeConvention,
    ExchangeModel,
    ExchangeRepresentation,
    ExchangeSpinNormalization,
)


def _exchange_model() -> ExchangeModel:
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


def _long_range_exchange_model() -> ExchangeModel:
    isotropic = np.asarray((2.0, 2.0, 0.5, 0.5))
    return ExchangeModel(
        source=Path("static_long_range.h5"),
        representation=ExchangeRepresentation.ISOTROPIC,
        source_dataset="J_r/value",
        convention=ExchangeConvention(
            spin_normalization=ExchangeSpinNormalization.UNIT_VECTOR,
            source_spin_magnitude=1.0,
            kernel_family="scalar_lkag",
        ),
        isotropic_mev=isotropic,
        tensor_mev=isotropic[:, None, None] * np.eye(3),
        bond_i=np.asarray((0, 0, 0, 0)),
        bond_j=np.asarray((0, 0, 0, 0)),
        bond_i_atom=np.asarray((0, 0, 0, 0)),
        bond_j_atom=np.asarray((0, 0, 0, 0)),
        cell_shift=np.asarray(
            ((1, 0, 0), (-1, 0, 0), (2, 0, 0), (-2, 0, 0))
        ),
        magnetic_atom_indices=np.asarray((0,)),
        mirror_index=np.asarray((1, 0, 3, 2)),
    )


def _write_derivative(
    path: Path,
    values: np.ndarray,
    *,
    rp: np.ndarray | None = None,
    qmesh=(1, 1, 1),
    unit="meV/A",
    include_provenance=True,
    shifts: np.ndarray | None = None,
    include_covariance: bool = False,
) -> None:
    # values: target, source-bond, Rp, Cartesian axis.  Source bonds are stored
    # in the reverse order to exercise key-based matching.
    target_count, source_bond_count, nrp, _ = values.shape
    target_atoms = np.arange(target_count, dtype=np.int64)
    rp_values = np.zeros((nrp, 3), dtype=np.int64) if rp is None else np.asarray(rp)
    text = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as handle:
        handle.attrs["complete"] = 1
        if include_provenance:
            handle.attrs["fourier_phase_convention"] = "exp(+i2pi_q_dot_Rp)"
            handle.attrs["directed_bond_mate"] = "(j,i,-R;Rp-R)_periodic"
        basic = handle.create_group("basic_data")
        basic.create_dataset("qmesh", data=np.asarray(qmesh, dtype=np.int64))
        basic.create_dataset("unit", data=np.asarray(unit, dtype=object), dtype=text)
        if include_covariance:
            symmetry = handle.create_group("symmetry")
            symmetry.create_dataset(
                "policy", data=np.asarray("project", dtype=object), dtype=text
            )
            symmetry.create_dataset("applied", data=np.asarray(True))
            symmetry.create_dataset(
                "max_abs_residual_raw_mev_per_ang", data=np.asarray(0.25)
            )
            symmetry.create_dataset("n_operations", data=np.asarray(8))
            symmetry.create_dataset(
                "spacegroup", data=np.asarray("P test", dtype=object), dtype=text
            )
        bonds = handle.create_group("bonds")
        bonds.create_dataset("mag_i_atom", data=np.zeros(source_bond_count, dtype=int))
        bonds.create_dataset("mag_j_atom", data=np.zeros(source_bond_count, dtype=int))
        source_shifts = (
            np.asarray(((-1, 0, 0), (1, 0, 0)), dtype=np.int64)
            if shifts is None
            else np.asarray(shifts, dtype=np.int64)
        )
        if source_shifts.shape != (source_bond_count, 3):
            raise ValueError("test derivative shifts must match the source bond count")
        bonds.create_dataset("R", data=source_shifts)
        displacement = handle.create_group("displacements")
        displacement.create_dataset("target_atom", data=target_atoms)
        displacement.create_dataset(
            "axes",
            data=np.asarray(("z", "x", "y"), dtype=object),
            dtype=text,
        )
        displacement.create_dataset("Rp", data=rp_values)
        group = handle.create_group("dJ_r")
        group.attrs["unit"] = unit
        for target_slot, atom in enumerate(target_atoms):
            for source_bond in range(source_bond_count):
                dataset_values = values[target_slot, source_bond][..., (2, 0, 1)]
                dataset = group.create_dataset(
                    f"dJ_r_m{int(atom) + 1}_b{source_bond + 1}",
                    data=dataset_values,
                )
                dataset.attrs["unit"] = unit


class ExchangeDerivativeTests(unittest.TestCase):
    def test_scalar_loader_matches_bonds_axes_and_promotes_exactly(self) -> None:
        exchange = _exchange_model()
        # Source order is R=-1, R=+1 and both directed mates agree.
        values = np.zeros((2, 2, 1, 3), dtype=np.float64)
        values[0, :, 0] = (1.0, 2.0, 3.0)
        values[1, :, 0] = (-1.0, -2.0, -3.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dj.h5"
            _write_derivative(path, values)
            model, report = load_exchange_derivative_h5(path, exchange)

        self.assertEqual(model.representation, ExchangeRepresentation.ISOTROPIC)
        np.testing.assert_array_equal(model.target_atom_indices, (0, 1))
        np.testing.assert_allclose(
            model.isotropic_mev_per_ang[0, :, 0],
            np.tile((1, 2, 3), (2, 1)),
        )
        np.testing.assert_allclose(
            model.isotropic_mev_per_ang[1, :, 0],
            np.tile((-1, -2, -3), (2, 1)),
        )
        expected_tensor = model.isotropic_mev_per_ang[..., None, None] * np.eye(3)
        np.testing.assert_array_equal(model.tensor_mev_per_ang, expected_tensor)
        self.assertEqual(report.max_asr_residual_after_mev_per_ang, 0.0)
        self.assertFalse(model.tensor_mev_per_ang.flags.writeable)

    def test_nonbinary_values_retain_exact_isotropic_tensor_invariant(self) -> None:
        exchange = _exchange_model()
        values = np.zeros((2, 2, 1, 3), dtype=np.float64)
        values[0, :, 0] = (0.1, 0.2, 0.3)
        values[1] = -values[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dj.h5"
            _write_derivative(path, values)
            model, _report = load_exchange_derivative_h5(path, exchange)

        np.testing.assert_array_equal(
            model.tensor_mev_per_ang,
            model.isotropic_mev_per_ang[..., None, None] * np.eye(3),
        )

    def test_rp_aware_mate_relation_is_enforced(self) -> None:
        exchange = _exchange_model()
        rp = np.asarray(((0, 0, 0), (-1, 0, 0)), dtype=np.int64)
        values = np.zeros((2, 2, 2, 3), dtype=np.float64)
        # Source bonds: 0=R- and 1=R+.  R+ at Rp0 maps to R- at Rp-1.
        values[0, 1, 0, 0] = 2.0
        values[0, 0, 1, 0] = 2.0
        values[0, 1, 1, 0] = -2.0
        values[0, 0, 0, 0] = -2.0
        values[1] = -values[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dj.h5"
            _write_derivative(path, values, rp=rp, qmesh=(2, 1, 1))
            model, _ = load_exchange_derivative_h5(path, exchange)
        self.assertEqual(model.n_rp, 2)

        values[0, 0, 1, 0] += 0.25
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.h5"
            _write_derivative(path, values, rp=rp, qmesh=(2, 1, 1))
            with self.assertRaisesRegex(ValueError, "Rp-aware"):
                load_exchange_derivative_h5(path, exchange)

    def test_asr_fail_report_and_project_are_explicit(self) -> None:
        exchange = _exchange_model()
        values = np.ones((2, 2, 1, 3), dtype=np.float64)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dj.h5"
            _write_derivative(path, values)
            with self.assertRaisesRegex(ValueError, "acoustic sum rule"):
                load_exchange_derivative_h5(path, exchange)
            reported, report = load_exchange_derivative_h5(
                path,
                exchange,
                asr_policy=DerivativeASRPolicy.REPORT,
            )
            projected, projected_report = load_exchange_derivative_h5(
                path,
                exchange,
                asr_policy=DerivativeASRPolicy.PROJECT,
            )

        self.assertGreater(report.max_asr_residual_after_mev_per_ang, 0.0)
        self.assertFalse(reported.tensor_mev_per_ang.flags.writeable)
        self.assertTrue(projected_report.projected_asr)
        self.assertLessEqual(
            projected_report.max_asr_residual_after_mev_per_ang,
            projected_report.asr_tolerance_mev_per_ang,
        )
        np.testing.assert_allclose(
            np.sum(projected.isotropic_mev_per_ang, axis=(0, 2)), 0.0
        )

    def test_missing_phase_provenance_is_rejected(self) -> None:
        values = np.zeros((2, 2, 1, 3), dtype=np.float64)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dj.h5"
            _write_derivative(path, values, include_provenance=False)
            with self.assertRaisesRegex(ValueError, "fourier_phase_convention"):
                load_exchange_derivative_h5(path, _exchange_model())

    def test_covariant_symmetry_provenance_is_screened_and_reported(self) -> None:
        values = np.zeros((2, 2, 1, 3), dtype=np.float64)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dj.h5"
            _write_derivative(path, values, include_covariance=True)
            _model, report = load_exchange_derivative_h5(
                path,
                _exchange_model(),
            )

        self.assertEqual(report.covariant_symmetry_policy, "project")
        self.assertTrue(report.covariant_symmetry_applied)
        self.assertEqual(report.max_covariance_residual_mev_per_ang, 0.25)
        self.assertEqual(report.covariant_symmetry_operation_count, 8)
        self.assertEqual(report.covariant_symmetry_spacegroup, "P test")

    def test_short_range_derivative_is_zero_embedded_in_long_range_static_bonds(
        self,
    ) -> None:
        exchange = _long_range_exchange_model()
        values = np.zeros((2, 2, 1, 3), dtype=np.float64)
        values[0, :, 0] = (1.0, 2.0, 3.0)
        values[1] = -values[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "short_dj.h5"
            _write_derivative(path, values)
            model, report = load_exchange_derivative_h5(path, exchange)

        self.assertEqual(model.n_bonds, 4)
        self.assertEqual(model.n_source_bonds, 2)
        self.assertFalse(model.bond_coverage_complete)
        np.testing.assert_array_equal(model.source_static_bond_indices, (0, 1))
        np.testing.assert_array_equal(
            model.isotropic_mev_per_ang[:, 2:],
            np.zeros((2, 2, 1, 3)),
        )
        self.assertEqual(report.static_bond_count, 4)
        self.assertEqual(report.source_bond_count, 2)
        self.assertEqual(report.zero_filled_static_bond_count, 2)
        self.assertFalse(report.bond_coverage_complete)

    def test_subset_asr_projection_preserves_zero_filled_static_bonds(self) -> None:
        exchange = _long_range_exchange_model()
        values = np.ones((2, 2, 1, 3), dtype=np.float64)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "short_dj.h5"
            _write_derivative(path, values)
            model, report = load_exchange_derivative_h5(
                path,
                exchange,
                asr_policy=DerivativeASRPolicy.PROJECT,
            )

        self.assertTrue(report.projected_asr)
        np.testing.assert_allclose(
            np.sum(model.isotropic_mev_per_ang[:, :2], axis=(0, 2)),
            0.0,
        )
        np.testing.assert_array_equal(
            model.isotropic_mev_per_ang[:, 2:],
            np.zeros((2, 2, 1, 3)),
        )

    def test_derivative_bonds_outside_static_range_are_rejected(self) -> None:
        values = np.zeros((2, 2, 1, 3), dtype=np.float64)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "extra_dj.h5"
            _write_derivative(
                path,
                values,
                shifts=np.asarray(((-2, 0, 0), (2, 0, 0))),
            )
            with self.assertRaisesRegex(ValueError, "must be a subset"):
                load_exchange_derivative_h5(path, _exchange_model())


if __name__ == "__main__":
    unittest.main()
