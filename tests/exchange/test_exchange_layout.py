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

    def test_numerical_drivers_are_native_private_modules(self):
        for name in (
            "j_epr",
            "j_tensor_epr",
            "j_wannier",
            "dj_epr",
            "dj_tensor_epr",
            "dj_tensor_mpi",
        ):
            with self.subTest(name=name):
                self.assertIsNotNone(
                    importlib.util.find_spec(
                        f"slw.exchange.kernels.{name}"
                    )
                )

    def test_active_exchange_tree_has_no_legacy_imports(self):
        root = Path(slw.exchange.__file__).resolve().parent
        for path in sorted(root.rglob("*.py")):
            if "legacy" in path.relative_to(root).parts:
                continue
            with self.subTest(path=path.relative_to(root)):
                self.assertNotIn("slw.exchange.legacy", path.read_text())


if __name__ == "__main__":
    unittest.main()
