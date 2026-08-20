import unittest

from slw.cli.adapter import (
    ArgumentAdapterError,
    arguments_from_parameters,
    parser_for,
)


class ArgumentAdapterTests(unittest.TestCase):
    def test_required_arguments_and_vectors(self):
        parser = parser_for("slw.epc.compute_gkq_from_epr")
        argv = arguments_from_parameters(
            parser,
            {"epr": "input.h5", "out": "gkq.h5", "rotate": False, "nproc": 4},
        )
        self.assertEqual(
            argv,
            ["--epr", "input.h5", "--out", "gkq.h5", "--nproc", "4"],
        )

    def test_boolean_optional_false_uses_negative_option(self):
        parser = parser_for("slw.magph.legacy.reference.hybrid")
        argv = arguments_from_parameters(
            parser,
            {"input_file": "legacy.in", "phonon_cache_compressed": False},
        )
        self.assertIn("--no-phonon_cache_compressed", argv)

    def test_store_false_destination_is_selected(self):
        parser = parser_for(
            "slw.exchange.legacy.reference.compute_J_epr_tensor"
        )
        parameters = {
            "epr_up": "up.h5",
            "epr_dn": "dn.h5",
            "efermi": 0.0,
            "kmesh": [2, 2, 2],
            "mag_atoms": [0],
            "slices": "0:0:5",
            "all_bonds": False,
        }
        argv = arguments_from_parameters(parser, parameters)
        self.assertIn("--canonical_bonds", argv)

    def test_unknown_parameter_is_rejected_before_calculation(self):
        parser = parser_for("slw.epc.compute_gkq_from_epr")
        with self.assertRaisesRegex(ArgumentAdapterError, "unknown input parameter"):
            arguments_from_parameters(parser, {"epr": "input.h5", "threads": 4})

    def test_missing_required_parameter_is_rejected(self):
        parser = parser_for("slw.epc.compute_gkq_from_epr")
        with self.assertRaises(ArgumentAdapterError):
            arguments_from_parameters(parser, {"nproc": 2})


if __name__ == "__main__":
    unittest.main()
