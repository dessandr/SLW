import unittest

from slw.cli.registry import RegistryError, resolve_action


class RegistryTests(unittest.TestCase):
    def test_exchange_source_is_explicit(self):
        with self.assertRaisesRegex(RegistryError, "requires input_format"):
            resolve_action("exchange", "j", {})

    def test_wannier_j_uses_native_exchange_handler(self):
        resolved = resolve_action(
            "exchange",
            "j",
            {"input_format": "wannier", "spinor_hr": "spinor_hr.dat"},
        )
        self.assertEqual(
            resolved.backend.handler,
            "slw.exchange.engine:prepare_run",
        )
        self.assertEqual(resolved.source, "wannier")
        self.assertNotIn("kernel", resolved.parameters)

    def test_tensor_dj_alias_injects_ltensor(self):
        resolved = resolve_action(
            "exchange",
            "dj_tensor",
            {"input_format": "epr"},
        )
        self.assertTrue(resolved.parameters["ltensor"])
        self.assertEqual(resolved.action.name, "dj")
        self.assertTrue(resolved.backend.is_native)

    def test_lifetime_uses_native_magph_handler(self):
        resolved = resolve_action("magph", "lifetime", {})
        self.assertEqual(resolved.backend.handler, "slw.magph.engine:prepare_run")
        self.assertTrue(resolved.backend.is_native)

    def test_magnon_dispersion_uses_native_magph_handler(self):
        resolved = resolve_action("magph", "dispersion", {})
        self.assertEqual(resolved.backend.handler, "slw.magph.engine:prepare_run")
        self.assertTrue(resolved.backend.is_native)

    def test_dispersion_rejects_implicit_material_path(self):
        with self.assertRaisesRegex(RegistryError, "explicit win or kpath"):
            resolve_action("epr", "dispersion", {"epr": "input.h5"})

    def test_help_resolution_can_skip_runtime_requirements(self):
        resolved = resolve_action(
            "epr",
            "dispersion",
            {},
            validate_requirements=False,
        )
        self.assertEqual(
            resolved.backend.module,
            "slw.epc.check_qe2pert_epr_dispersion",
        )


if __name__ == "__main__":
    unittest.main()
