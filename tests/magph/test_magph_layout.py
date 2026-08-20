import importlib.util
import unittest
from pathlib import Path

import slw.magph

HISTORICAL_MODULES = (
    "adapter",
    "analyze_lifetime",
    "analyze_rotational_coupling",
    "analyze_rotational_selectivity",
    "backend",
    "build_phonon_cache",
    "hybrid",
    "hybrid_berry",
    "hybrid_html",
    "input_parser",
    "kernels",
    "lifetime_mpi_dynamic",
    "lswt",
    "numerics",
    "plot",
    "plot_coupling_bz",
    "plot_coupling_kbz",
    "plot_coupling_kpath",
    "plot_lifetime_symmetry",
    "plot_scattering_kbz",
    "plot_scattering_qbz",
    "plotting",
    "prepare_lifetime",
    "rotational",
    "run_chirality_plane_mpi",
    "runtime",
    "scattering",
    "solver_mpi",
    "tensor_adapter",
    "utils",
    "vertex",
)

REFERENCE_DRIVERS = (
    "build_phonon_cache",
    "hybrid",
    "hybrid_berry",
    "solver_mpi",
    "lifetime_mpi_dynamic",
    "plot_scattering_kbz",
    "plot_scattering_qbz",
    "analyze_rotational_coupling",
    "run_chirality_plane_mpi",
    "prepare_lifetime",
)


class MagphLayoutTests(unittest.TestCase):
    def test_package_root_contains_no_historical_modules(self):
        root = Path(slw.magph.__file__).resolve().parent
        modules = {path.name for path in root.glob("*.py")}
        self.assertEqual(modules, {"__init__.py"})

    def test_historical_module_paths_are_not_preserved(self):
        for name in HISTORICAL_MODULES:
            with self.subTest(name=name):
                self.assertIsNone(importlib.util.find_spec(f"slw.magph.{name}"))

    def test_active_drivers_are_isolated_below_legacy_reference(self):
        for name in REFERENCE_DRIVERS:
            with self.subTest(name=name):
                self.assertIsNotNone(
                    importlib.util.find_spec(
                        f"slw.magph.legacy.reference.{name}"
                    )
                )


if __name__ == "__main__":
    unittest.main()
