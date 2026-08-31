from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from slw.magph.dispersion import MagnonDispersionResult, MagnonKPath
from slw.magph.output import (
    DISPERSION_OUTPUT_SCHEMA_VERSION,
    LIFETIME_OUTPUT_SCHEMA_VERSION,
    write_dispersion_npz,
    write_dispersion_plot,
    write_lifetime_npz,
)
from slw.magph.pipeline import LifetimeGridResult


def _result() -> LifetimeGridResult:
    return LifetimeGridResult(
        k_points_frac=np.asarray(((0.25, 0.0, 0.0),)),
        energy_mev=np.asarray(((2.0,),)),
        self_energy_onshell_mev=np.asarray(((-0.1j,),)),
        gamma_hwhm_mev=np.asarray(((0.1,),)),
        fwhm_mev=np.asarray(((0.2,),)),
        scattering_rate_ps_inv=np.asarray(((0.3,),)),
        lifetime_ps=np.asarray(((3.0,),)),
        valid_damping=np.asarray(((True,),)),
        temperature_k=10.0,
        broadening_mev=0.05,
        negative_tolerance_mev=0.0,
    )


class NativeLifetimeOutputTests(unittest.TestCase):
    def test_atomic_output_contains_units_and_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.npz"
            written = write_lifetime_npz(
                output,
                _result(),
                metadata={"calculation": "lifetime", "unit": "meV"},
            )
            self.assertEqual(written, output.resolve())
            with np.load(output, allow_pickle=False) as payload:
                self.assertEqual(
                    int(payload["schema_version"]), LIFETIME_OUTPUT_SCHEMA_VERSION
                )
                metadata = json.loads(str(payload["metadata_json"]))
                self.assertEqual(metadata["calculation"], "lifetime")
                self.assertEqual(
                    metadata["schema_version"], LIFETIME_OUTPUT_SCHEMA_VERSION
                )
                np.testing.assert_allclose(payload["gamma_hwhm_mev"], ((0.1,),))

    def test_lifetime_output_stores_validated_magnon_chirality(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.npz"
            chirality = np.asarray(((1.0,),))
            write_lifetime_npz(
                output,
                _result(),
                metadata={"magnon_mode_order": "chirality_descending"},
                magnon_chirality=chirality,
            )
            with np.load(output, allow_pickle=False) as payload:
                np.testing.assert_allclose(payload["magnon_chirality"], chirality)

            with self.assertRaisesRegex(ValueError, "chirality shape"):
                write_lifetime_npz(
                    Path(directory) / "bad.npz",
                    _result(),
                    metadata={},
                    magnon_chirality=np.ones((2, 1)),
                )

    def test_existing_output_is_not_clobbered_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.npz"
            output.write_bytes(b"original")
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                write_lifetime_npz(output, _result(), metadata={})
            self.assertEqual(output.read_bytes(), b"original")

    def test_dispersion_npz_and_plot_share_the_same_path_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = MagnonKPath(
                source=root / "bands.win",
                k_points_frac=np.asarray(((0.0, 0.0, 0.0), (0.5, 0.0, 0.0))),
                x_coordinate_inv_ang=np.asarray((0.0, 1.0)),
                segment_offsets=np.asarray((0, 2)),
                tick_positions_inv_ang=np.asarray((0.0, 1.0)),
                tick_labels=("Γ", "X"),
                points_per_segment=1,
            )
            result = MagnonDispersionResult(
                path=path,
                energy_mev=np.asarray(((0.0, 1.0), (2.0, 3.0))),
                goldstone_mask=np.asarray(((True, False), (False, False))),
                mpi_size=1,
            )
            npz = write_dispersion_npz(
                root / "bands.npz",
                result,
                metadata={"calculation": "dispersion"},
            )
            image = write_dispersion_plot(root / "bands.png", result, title="test")
            assert image.stat().st_size > 0
            with np.load(npz, allow_pickle=False) as payload:
                assert (
                    int(payload["schema_version"]) == DISPERSION_OUTPUT_SCHEMA_VERSION
                )
                np.testing.assert_allclose(payload["energy_mev"], result.energy_mev)
                np.testing.assert_array_equal(payload["tick_labels"], ("Γ", "X"))


if __name__ == "__main__":
    unittest.main()
