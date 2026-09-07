from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np

from slw.cli.mpi import MPIContext
from slw.magph.config import (
    build_dispersion_request,
    build_lifetime_request,
    build_phonon_renormalization_request,
)
from slw.magph.engine import (
    run_dispersion,
    run_lifetime,
    run_phonon_renormalization,
)
from slw.magph.parallel import CollectiveExecutionError
from tests.magph.test_derivative import _write_derivative
from tests.magph.test_epr_phonon import _write_one_atom_epr
from tests.magph.test_exchange_screening import _write_scalar
from tests.magph.test_phonon import _write_cache


class NativeMagphEngineTests(unittest.TestCase):
    def test_phonon_renormalization_runs_without_an_fm_phonon_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            exchange = root / "J.h5"
            derivative = root / "dJ.h5"
            phonon = root / "phonon.npz"
            output = root / "phonon_renormalization.npz"
            _write_scalar(
                exchange,
                [2.0, 2.0],
                atom_i=[0, 0],
                atom_j=[0, 0],
                shifts=[(1, 0, 0), (-1, 0, 0)],
            )
            _write_derivative(
                derivative,
                np.zeros((1, 2, 1, 3), dtype=np.float64),
            )
            _write_cache(phonon, frequency=5.0, masses=(4.0,))
            parameters = {
                "exchange_h5": exchange,
                "derivative_h5": derivative,
                "phonon_cache": phonon,
                "output": output,
                "magnetic_order": "fm",
                "spin_magnitudes": 1.0,
                "quantization_axis": (0.0, 0.0, 1.0),
                "anisotropy_model": "uniaxial",
                "anisotropy_mev": 0.1,
                "anisotropy_axis": (0.0, 0.0, 1.0),
                "anisotropy_normalization": "unit_vector",
                "kmesh": (2, 1, 1),
                "kshift": (0.5, 0.0, 0.0),
                "temperature_k": 0.0,
                "broadening_mev": 0.2,
            }
            request = build_phonon_renormalization_request(
                parameters,
                prefix="sample",
                savedir=root,
            )
            result = run_phonon_renormalization(
                request,
                context=MPIContext(),
                verbosity="quiet",
            )

            self.assertEqual(result.output, output.resolve())
            with np.load(output, allow_pickle=False) as payload:
                np.testing.assert_allclose(payload["bare_frequency_mev"], 5.0)
                np.testing.assert_allclose(
                    payload["renormalized_frequency_mev"], 5.0
                )
                np.testing.assert_allclose(payload["frequency_shift_mev"], 0.0)
                metadata = json.loads(str(payload["metadata_json"]))
                self.assertEqual(metadata["calculation"], "phonon_renormalization")
                self.assertEqual(metadata["external_quasiparticle"], "phonon")
                self.assertFalse(
                    metadata["static_exchange_second_derivative_included"]
                )
                self.assertEqual(
                    metadata["algorithm"]["mpi_distribution"],
                    "phonon_q",
                )

            restarted = run_phonon_renormalization(
                build_phonon_renormalization_request(
                    {**parameters, "restart_mode": "restart"},
                    prefix="sample",
                    savedir=root,
                ),
                context=MPIContext(),
                verbosity="quiet",
            )
            self.assertEqual(restarted.output, output.resolve())

    def test_missing_phonon_cache_is_built_from_epr_before_lifetime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            exchange = root / "J.h5"
            derivative = root / "dJ.h5"
            epr = root / "sample_epr.h5"
            output = root / "lifetime.npz"
            _write_scalar(
                exchange,
                [2.0, 2.0],
                atom_i=[0, 0],
                atom_j=[0, 0],
                shifts=[(1, 0, 0), (-1, 0, 0)],
            )
            _write_derivative(
                derivative,
                np.zeros((1, 2, 1, 3), dtype=np.float64),
            )
            _write_one_atom_epr(epr)
            request = build_lifetime_request(
                {
                    "exchange_h5": exchange,
                    "derivative_h5": derivative,
                    "phonon_epr": epr,
                    "phonon_qmesh": (2, 1, 1),
                    "output": output,
                    "magnetic_order": "fm",
                    "spin_magnitudes": 1.0,
                    "quantization_axis": (0.0, 0.0, 1.0),
                    "anisotropy_model": "uniaxial",
                    "anisotropy_mev": 0.1,
                    "anisotropy_axis": (0.0, 0.0, 1.0),
                    "anisotropy_normalization": "unit_vector",
                    "kmesh": (3, 1, 1),
                    "kshift": (0.5, 0.0, 0.0),
                    "temperature_k": 0.0,
                    "broadening_mev": 0.2,
                },
                prefix="sample",
                savedir=root,
            )

            result = run_lifetime(request, context=MPIContext(), verbosity="quiet")

            self.assertEqual(result.output, output.resolve())
            self.assertTrue(request.phonon_cache.is_file())
            self.assertEqual(result.k_point_count, 3)
            with np.load(request.phonon_cache, allow_pickle=False) as cache:
                np.testing.assert_array_equal(cache["q_mesh_shape"], (2, 1, 1))
            with np.load(output, allow_pickle=False) as payload:
                metadata = json.loads(str(payload["metadata_json"]))
                self.assertEqual(metadata["phonon_epr"], str(epr.resolve()))
                self.assertEqual(metadata["phonon_loto"], "auto")
                self.assertEqual(metadata["derivative_source_qmesh"], [1, 1, 1])
                self.assertEqual(metadata["phonon_evaluation_qmesh"], [2, 1, 1])
                self.assertTrue(metadata["derivative_q_interpolation"])
                self.assertEqual(metadata["union_kq_mesh"], [6, 1, 1])

    def test_explicit_phonon_qmesh_rejects_stale_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            exchange = root / "J.h5"
            derivative = root / "dJ.h5"
            phonon = root / "phonon.npz"
            _write_scalar(
                exchange,
                [2.0, 2.0],
                atom_i=[0, 0],
                atom_j=[0, 0],
                shifts=[(1, 0, 0), (-1, 0, 0)],
            )
            _write_derivative(
                derivative,
                np.zeros((1, 2, 1, 3), dtype=np.float64),
            )
            _write_cache(phonon, frequency=5.0, masses=(4.0,))
            request = build_lifetime_request(
                {
                    "exchange_h5": exchange,
                    "derivative_h5": derivative,
                    "phonon_cache": phonon,
                    "phonon_qmesh": (2, 1, 1),
                    "magnetic_order": "fm",
                    "spin_magnitudes": 1.0,
                    "quantization_axis": (0.0, 0.0, 1.0),
                    "kmesh": (1, 1, 1),
                    "kshift": (0.5, 0.0, 0.0),
                    "temperature_k": 0.0,
                    "broadening_mev": 0.2,
                },
                prefix="sample",
                savedir=root,
            )

            with self.assertRaisesRegex(
                CollectiveExecutionError, "phonon cache q mesh.*phonon_qmesh"
            ):
                run_lifetime(request, context=MPIContext(), verbosity="quiet")

    def test_generated_files_run_through_native_engine(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            exchange = root / "J.h5"
            derivative = root / "dJ.h5"
            phonon = root / "phonon.npz"
            output = root / "lifetime.npz"
            _write_scalar(
                exchange,
                [2.0, 2.0],
                atom_i=[0, 0],
                atom_j=[0, 0],
                shifts=[(1, 0, 0), (-1, 0, 0)],
            )
            _write_derivative(
                derivative,
                np.zeros((1, 2, 1, 3), dtype=np.float64),
            )
            _write_cache(phonon, frequency=5.0, masses=(4.0,))
            request = build_lifetime_request(
                {
                    "exchange_h5": exchange,
                    "derivative_h5": derivative,
                    "phonon_cache": phonon,
                    "output": output,
                    "magnetic_order": "fm",
                    "spin_magnitudes": 1.0,
                    "quantization_axis": (0.0, 0.0, 1.0),
                    "anisotropy_model": "uniaxial",
                    "anisotropy_mev": 0.1,
                    "anisotropy_axis": (0.0, 0.0, 1.0),
                    "anisotropy_normalization": "unit_vector",
                    "kmesh": (2, 1, 1),
                    "kshift": (0.5, 0.0, 0.0),
                    "temperature_k": 0.0,
                    "broadening_mev": 0.2,
                },
                prefix="sample",
                savedir=root,
            )
            result = run_lifetime(request, context=MPIContext(), verbosity="quiet")

            self.assertEqual(result.output, output.resolve())
            self.assertEqual(result.k_point_count, 2)
            self.assertEqual(result.magnetic_site_count, 1)
            with np.load(output, allow_pickle=False) as payload:
                self.assertEqual(payload["energy_mev"].shape, (2, 1))
                np.testing.assert_allclose(payload["energy_mev"], 4.2)
                np.testing.assert_allclose(payload["gamma_hwhm_mev"], 0.0)
                self.assertTrue(np.all(np.isinf(payload["lifetime_ps"])))
                metadata = json.loads(str(payload["metadata_json"]))
                self.assertEqual(
                    metadata["algorithm"]["coupling"],
                    "mpi_q_distributed_cache",
                )
                self.assertEqual(
                    metadata["algorithm"]["lswt"],
                    "uniform_k_plus_q_union_cache",
                )
                self.assertEqual(
                    metadata["algorithm"]["vertex_self_energy"],
                    "q_block_streaming",
                )
                self.assertFalse(metadata["algorithm"]["full_vertex_materialized"])
                self.assertEqual(metadata["derivative_static_bond_count"], 2)
                self.assertEqual(metadata["derivative_source_bond_count"], 2)
                self.assertEqual(
                    metadata["derivative_zero_filled_static_bond_count"],
                    0,
                )
                self.assertTrue(metadata["derivative_bond_coverage_complete"])
                self.assertEqual(
                    metadata["single_ion_anisotropy"]["spin_normalization"],
                    "unit_vector",
                )
                self.assertEqual(metadata["union_kq_mesh"], [2, 1, 1])
                self.assertGreater(
                    metadata["cache_bytes_per_rank"]["coupling"],
                    0,
                )
                self.assertGreater(metadata["cache_bytes_per_rank"]["lswt"], 0)

    def test_native_dispersion_writes_exact_path_and_sia_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            exchange = root / "J.h5"
            kpath = root / "bands.win"
            output = root / "dispersion.npz"
            plot_output = root / "dispersion.png"
            _write_scalar(
                exchange,
                [2.0, 2.0],
                atom_i=[0, 0],
                atom_j=[0, 0],
                shifts=[(1, 0, 0), (-1, 0, 0)],
            )
            with h5py.File(exchange, "r+") as handle:
                handle["basic_data"].create_dataset("lattice_ang", data=np.eye(3))
            kpath.write_text(
                """begin kpoint_path
G 0 0 0 X 0.5 0 0
end kpoint_path
""",
                encoding="utf-8",
            )
            parameters = {
                "exchange_h5": exchange,
                "kpath_file": kpath,
                "output": output,
                "plot": True,
                "plot_output": plot_output,
                "points_per_segment": 2,
                "magnetic_order": "fm",
                "spin_magnitudes": 2.0,
                "quantization_axis": (0.0, 0.0, 1.0),
                "anisotropy_model": "uniaxial",
                "anisotropy_mev": 0.1,
                "anisotropy_axis": (0.0, 0.0, 1.0),
                "anisotropy_normalization": "unit_vector",
            }
            request = build_dispersion_request(
                parameters,
                prefix="sample",
                savedir=root,
            )
            result = run_dispersion(
                request,
                context=MPIContext(),
                verbosity="quiet",
            )
            assert result.output == output.resolve()
            assert result.plot_output == plot_output.resolve()
            assert plot_output.is_file()
            with np.load(output, allow_pickle=False) as payload:
                np.testing.assert_allclose(payload["energy_mev"][:, 0], (0.1, 2.1, 4.1))
                metadata = json.loads(str(payload["metadata_json"]))
                assert metadata["single_ion_anisotropy"]["energy_mev"] == [0.1]
                assert metadata["parallel"]["distribution"] == "kpath_points"
                assert len(metadata["restart_signature"]) == 64

            restart_request = build_dispersion_request(
                {**parameters, "restart_mode": "restart"},
                prefix="sample",
                savedir=root,
            )
            plot_output.unlink()
            with patch(
                "slw.magph.engine.compute_magnon_dispersion",
                side_effect=AssertionError("completed restart must not recompute"),
            ):
                restarted = run_dispersion(
                    restart_request,
                    context=MPIContext(),
                    verbosity="quiet",
                )
            self.assertEqual(restarted.k_point_count, 3)
            self.assertTrue(plot_output.is_file())

            mismatch_request = build_dispersion_request(
                {**parameters, "points_per_segment": 3, "restart_mode": "restart"},
                prefix="sample",
                savedir=root,
            )
            with self.assertRaisesRegex(
                CollectiveExecutionError, "restart signature.*does not match"
            ):
                run_dispersion(
                    mismatch_request,
                    context=MPIContext(),
                    verbosity="quiet",
                )

            output.write_bytes(b"stale output")
            scratch_request = build_dispersion_request(
                {**parameters, "restart_mode": "from_scratch"},
                prefix="sample",
                savedir=root,
            )
            run_dispersion(
                scratch_request,
                context=MPIContext(),
                verbosity="quiet",
            )
            with np.load(output, allow_pickle=False) as payload:
                self.assertEqual(payload["energy_mev"].shape, (3, 1))

    def test_lifetime_accepts_shorter_derivative_range_than_static_exchange(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            exchange = root / "J_long.h5"
            derivative = root / "dJ_short.h5"
            phonon = root / "phonon.npz"
            output = root / "lifetime.npz"
            _write_scalar(
                exchange,
                [2.0, 2.0, 0.5, 0.5],
                atom_i=[0, 0, 0, 0],
                atom_j=[0, 0, 0, 0],
                shifts=[(1, 0, 0), (-1, 0, 0), (2, 0, 0), (-2, 0, 0)],
            )
            _write_derivative(
                derivative,
                np.zeros((1, 2, 1, 3), dtype=np.float64),
            )
            _write_cache(phonon, frequency=5.0, masses=(4.0,))
            request = build_lifetime_request(
                {
                    "exchange_h5": exchange,
                    "derivative_h5": derivative,
                    "phonon_cache": phonon,
                    "output": output,
                    "magnetic_order": "fm",
                    "spin_magnitudes": 1.0,
                    "quantization_axis": (0.0, 0.0, 1.0),
                    "anisotropy_model": "uniaxial",
                    "anisotropy_mev": 0.1,
                    "anisotropy_axis": (0.0, 0.0, 1.0),
                    "anisotropy_normalization": "unit_vector",
                    "kmesh": (2, 1, 1),
                    "kshift": (0.5, 0.0, 0.0),
                    "temperature_k": 0.0,
                    "broadening_mev": 0.2,
                },
                prefix="sample",
                savedir=root,
            )

            run_lifetime(request, context=MPIContext(), verbosity="quiet")

            with np.load(output, allow_pickle=False) as payload:
                metadata = json.loads(str(payload["metadata_json"]))
                self.assertEqual(metadata["derivative_static_bond_count"], 4)
                self.assertEqual(metadata["derivative_source_bond_count"], 2)
                self.assertEqual(
                    metadata["derivative_zero_filled_static_bond_count"],
                    2,
                )
                self.assertFalse(metadata["derivative_bond_coverage_complete"])
                np.testing.assert_array_equal(payload["gamma_hwhm_mev"], 0.0)


if __name__ == "__main__":
    unittest.main()
