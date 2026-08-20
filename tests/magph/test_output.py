from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from slw.magph.output import LIFETIME_OUTPUT_SCHEMA_VERSION, write_lifetime_npz
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
                self.assertEqual(metadata["schema_version"], 1)
                np.testing.assert_allclose(payload["gamma_hwhm_mev"], ((0.1,),))

    def test_existing_output_is_not_clobbered_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.npz"
            output.write_bytes(b"original")
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                write_lifetime_npz(output, _result(), metadata={})
            self.assertEqual(output.read_bytes(), b"original")


if __name__ == "__main__":
    unittest.main()
