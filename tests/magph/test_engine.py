from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from slw.cli.mpi import MPIContext
from slw.magph.config import build_lifetime_request
from slw.magph.engine import run_lifetime
from tests.magph.test_derivative import _write_derivative
from tests.magph.test_exchange_screening import _write_scalar
from tests.magph.test_phonon import _write_cache


class NativeMagphEngineTests(unittest.TestCase):
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
                np.testing.assert_allclose(payload["gamma_hwhm_mev"], 0.0)
                self.assertTrue(np.all(np.isinf(payload["lifetime_ps"])))


if __name__ == "__main__":
    unittest.main()
