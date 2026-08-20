import importlib.util
import unittest
from pathlib import Path

import slw.exchange


class ExchangeLayoutTests(unittest.TestCase):
    def test_package_root_contains_only_native_exchange_modules(self):
        root = Path(slw.exchange.__file__).resolve().parent
        modules = {path.name for path in root.glob("*.py")}
        self.assertEqual(
            modules,
            {"__init__.py", "config.py", "engine.py", "model.py"},
        )

    def test_historical_compute_shims_are_not_reexported(self):
        for name in (
            "compute_J_epr_kspace",
            "compute_J_epr_tensor",
            "compute_J_wannier_tensor",
            "compute_dJ_epr_kspace",
            "compute_dJ_epr_tensor",
            "compute_dJ_epr_tensor_mpi",
        ):
            with self.subTest(name=name):
                self.assertIsNone(importlib.util.find_spec(f"slw.exchange.{name}"))

    def test_parity_drivers_are_isolated_below_legacy_reference(self):
        for name in (
            "compute_J_epr_kspace",
            "compute_J_epr_tensor",
            "compute_J_wannier_tensor",
            "compute_dJ_epr_kspace",
            "compute_dJ_epr_tensor",
            "compute_dJ_epr_tensor_mpi",
        ):
            with self.subTest(name=name):
                self.assertIsNotNone(
                    importlib.util.find_spec(
                        f"slw.exchange.legacy.reference.{name}"
                    )
                )


if __name__ == "__main__":
    unittest.main()
