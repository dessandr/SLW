from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from slw.magph.config import (
    MagphInputError,
    RestartMode,
    build_dispersion_request,
    build_lifetime_request,
)


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

    def test_lifetime_parses_complete_single_ion_anisotropy(self) -> None:
        parameters = _parameters()
        parameters.update(
            {
                "anisotropy_model": "uniaxial",
                "anisotropy_mev": (0.1, 0.2),
                "anisotropy_axis": (0.0, 0.0, 1.0),
                "anisotropy_normalization": "unit_vector",
            }
        )
        request = build_lifetime_request(parameters, prefix="slw", savedir=".")
        assert request.anisotropy is not None
        built = request.anisotropy.build(2)
        self.assertEqual(built.energy_mev.tolist(), [0.1, 0.2])
        self.assertEqual(built.axis.tolist(), [[0.0, 0.0, 1.0]] * 2)

    def test_partial_single_ion_anisotropy_is_rejected(self) -> None:
        parameters = _parameters()
        parameters["anisotropy_mev"] = 0.1
        with self.assertRaisesRegex(MagphInputError, "requires.*anisotropy_axis"):
            build_lifetime_request(parameters, prefix="slw", savedir=".")

    def test_dispersion_request_has_atomic_output_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            request = build_dispersion_request(
                {
                    "exchange_h5": "J.h5",
                    "kpath_file": "bands.win",
                    "magnetic_order": "collinear_afm",
                    "spin_magnitudes": (2.5, 2.5),
                    "quantization_axis": (0.0, 0.0, 1.0),
                },
                prefix="mn",
                savedir=directory,
            )
            self.assertEqual(
                request.output,
                Path(directory).resolve() / "mn.dispersion.npz",
            )
            self.assertEqual(
                request.plot_output,
                Path(directory).resolve() / "mn.dispersion.png",
            )
            self.assertEqual(request.points_per_segment, 50)
            self.assertIs(request.restart_mode, RestartMode.ERROR)

    def test_restart_mode_and_deprecated_overwrite_alias_are_explicit(self) -> None:
        parameters = _parameters()
        parameters["restart_mode"] = "restart"
        request = build_lifetime_request(parameters, prefix="slw", savedir=".")
        self.assertIs(request.restart_mode, RestartMode.RESTART)
        self.assertFalse(request.overwrite)

        parameters = _parameters()
        parameters["overwrite"] = True
        request = build_lifetime_request(parameters, prefix="slw", savedir=".")
        self.assertIs(request.restart_mode, RestartMode.FROM_SCRATCH)
        self.assertTrue(request.overwrite)

        parameters["restart_mode"] = "from_scratch"
        with self.assertRaisesRegex(MagphInputError, "mutually exclusive"):
            build_lifetime_request(parameters, prefix="slw", savedir=".")

    def test_invalid_restart_mode_is_rejected(self) -> None:
        parameters = _parameters()
        parameters["restart_mode"] = "append"
        with self.assertRaisesRegex(MagphInputError, "error, restart, from_scratch"):
            build_lifetime_request(parameters, prefix="slw", savedir=".")


if __name__ == "__main__":
    unittest.main()
