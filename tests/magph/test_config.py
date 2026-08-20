from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from slw.magph.config import MagphInputError, build_lifetime_request


def _parameters() -> dict[str, object]:
    return {
        "input_format": "default",
        "exchange_h5": "J.h5",
        "derivative_h5": "dJ.h5",
        "phonon_cache": "phonon.npz",
        "magnetic_order": "fm",
        "spin_magnitudes": 2.5,
        "quantization_axis": (0.0, 0.0, 1.0),
        "kmesh": (4, 3, 2),
        "kshift": (0.5, 0.5, 0.5),
        "temperature_k": 300.0,
        "broadening_mev": 0.2,
    }


class NativeMagphConfigTests(unittest.TestCase):
    def test_required_input_is_normalized_without_opening_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            request = build_lifetime_request(
                _parameters(), prefix="mn", savedir=directory
            )
            self.assertEqual(request.kmesh, (4, 3, 2))
            self.assertEqual(request.kshift, (0.5, 0.5, 0.5))
            self.assertEqual(request.spin_magnitudes, (2.5,))
            self.assertEqual(
                request.output, Path(directory).resolve() / "mn.lifetime.npz"
            )

    def test_explicit_kshift_is_required(self) -> None:
        parameters = _parameters()
        del parameters["kshift"]
        with self.assertRaisesRegex(MagphInputError, "requires kshift"):
            build_lifetime_request(parameters, prefix="slw", savedir=".")

    def test_unknown_parameter_is_rejected(self) -> None:
        parameters = _parameters()
        parameters["material"] = "MnTe"
        with self.assertRaisesRegex(MagphInputError, "unknown.*material"):
            build_lifetime_request(parameters, prefix="slw", savedir=".")


if __name__ == "__main__":
    unittest.main()
