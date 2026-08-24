from __future__ import annotations

import pickle
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from slw.exchange.config import (
    ExchangeInputError,
    ExchangeRequest,
    build_exchange_request,
)
from slw.exchange.kernels.dispatch import build_namespace
from slw.exchange.model import (
    ExchangeCalculation,
    ExchangeSource,
    TensorKernel,
)
from slw.soc.model import SpinorGroupBy


def _common(**updates):
    values = {
        "epr_up": "up_epr.h5",
        "epr_dn": "dn_epr.h5",
        "efermi": 1.25,
        "kmesh": [4, 3, 2],
        "mag_atoms": [0, 2],
        "slices": "0:0:5,1:5:10",
    }
    values.update(updates)
    return values


class ExchangeConfigTests(unittest.TestCase):
    def test_builds_frozen_epr_j_request_with_deterministic_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            savedir = Path(directory) / "sample.save"
            request = build_exchange_request(
                "j",
                _common(input_format="epr"),
                prefix="sample",
                savedir=savedir,
            )

            self.assertIsInstance(request, ExchangeRequest)
            self.assertEqual(request.calculation, ExchangeCalculation.J)
            self.assertFalse(request.ltensor)
            self.assertEqual(request.source, ExchangeSource.EPR)
            self.assertEqual(request.tensor_kernel, TensorKernel.TB2J)
            self.assertEqual(request.kmesh, (4, 3, 2))
            self.assertEqual(request.mag_atoms, (0, 2))
            self.assertEqual(request.slice_map, {0: slice(0, 5), 1: slice(5, 10)})
            self.assertEqual(request.output.h5_path, str(savedir / "sample.j.h5"))
            self.assertEqual(request.output.text_path, str(savedir / "sample.j.txt"))
            self.assertEqual(
                request.output.table_path,
                str(savedir / "sample.j.all_bonds.tsv"),
            )
            with self.assertRaises(FrozenInstanceError):
                request.ltensor = True  # type: ignore[misc]

    def test_tensor_alias_normalizes_and_conflict_is_rejected(self):
        request = build_exchange_request(
            "j_tensor",
            _common(input_format="epr", tensor_kernel="direct"),
            prefix="x",
            savedir="save",
        )
        self.assertEqual(request.calculation, ExchangeCalculation.J)
        self.assertTrue(request.ltensor)
        self.assertEqual(request.tensor_kernel, TensorKernel.DIRECT)
        self.assertEqual(request.mode_name, "j_tensor")

        with self.assertRaisesRegex(ExchangeInputError, "requires ltensor=true"):
            build_exchange_request(
                "dj_tensor",
                _common(ltensor=False),
                prefix="x",
                savedir="save",
            )

        for option in ("tensor_kernel", "kernel"):
            with (
                self.subTest(option=option),
                self.assertRaisesRegex(ExchangeInputError, "ltensor=true"),
            ):
                build_exchange_request(
                    "j",
                    _common(input_format="epr", **{option: "tb2j"}),
                    prefix="x",
                    savedir="save",
                )

    def test_j_requires_explicit_source(self):
        with self.assertRaisesRegex(ExchangeInputError, "requires explicit input_format"):
            build_exchange_request("j", _common(), prefix="x", savedir="save")

        request = build_exchange_request(
            ExchangeCalculation.J,
            _common(input_format=ExchangeSource.EPR),
            prefix="x",
            savedir="save",
        )
        self.assertEqual(request.calculation, ExchangeCalculation.J)

    def test_scalar_and_tensor_wannier_file_combinations(self):
        base = {
            "input_format": "wannier",
            "efermi": 0.0,
            "kmesh": (2, 2, 2),
            "mag_atoms": 0,
            "slices": "0:0:5",
        }
        scalar = build_exchange_request(
            "j",
            {**base, "up_hr": "up_hr.dat", "dn_hr": "dn_hr.dat"},
            prefix="w",
            savedir="save",
        )
        self.assertEqual(scalar.files.up_hr, "up_hr.dat")

        tensor = build_exchange_request(
            "j_tensor",
            {
                **base,
                "spinor_hr": "spinor_hr.dat",
                "win": "model.win",
                "groupby": "orbital",
            },
            prefix="w",
            savedir="save",
        )
        self.assertEqual(tensor.files.spinor_hr, "spinor_hr.dat")

        with self.assertRaisesRegex(ExchangeInputError, "not both"):
            build_exchange_request(
                "j_tensor",
                {
                    **base,
                    "spinor_hr": "spinor_hr.dat",
                    "up_hr": "up_hr.dat",
                    "dn_hr": "dn_hr.dat",
                },
                prefix="w",
                savedir="save",
            )
        with self.assertRaisesRegex(ExchangeInputError, "requires collinear"):
            build_exchange_request(
                "j",
                {**base, "spinor_hr": "spinor_hr.dat"},
                prefix="w",
                savedir="save",
            )

    def test_dj_is_epr_only_and_spinor_base_requires_tensor(self):
        request = build_exchange_request(
            "dj_tensor",
            _common(
                input_format="epr",
                spinor_hr="spinor_hr.dat",
                groupby="orbital",
            ),
            prefix="d",
            savedir="save",
        )
        self.assertEqual(request.source, ExchangeSource.EPR)
        self.assertEqual(request.files.spinor_hr, "spinor_hr.dat")
        self.assertEqual(request.groupby, SpinorGroupBy.ORBITAL)

        with self.assertRaisesRegex(ExchangeInputError, "supports only"):
            build_exchange_request(
                "dj",
                _common(input_format="wannier"),
                prefix="d",
                savedir="save",
            )
        with self.assertRaisesRegex(ExchangeInputError, "scalar dJ"):
            build_exchange_request(
                "dj",
                _common(input_format="epr", spinor_hr="spinor_hr.dat"),
                prefix="d",
                savedir="save",
            )

        with self.assertRaisesRegex(ExchangeInputError, "requires explicit input_format='epr'"):
            build_exchange_request(
                "dj",
                _common(),
                prefix="d",
                savedir="save",
            )

        with self.assertRaisesRegex(ExchangeInputError, "supports tensor_kernel='tb2j'"):
            build_exchange_request(
                "dj_tensor",
                _common(input_format="epr", tensor_kernel="direct"),
                prefix="d",
                savedir="save",
            )

        with self.assertRaisesRegex(ExchangeInputError, "requires explicit groupby"):
            build_exchange_request(
                "dj_tensor",
                _common(input_format="epr", spinor_hr="spinor_hr.dat"),
                prefix="d",
                savedir="save",
            )

    def test_site_keys_are_validated_for_backend_conventions(self):
        with self.assertRaisesRegex(ExchangeInputError, "at least two"):
            build_exchange_request(
                "j",
                _common(
                    input_format="epr",
                    mag_atoms=[0],
                    slices="0:0:5",
                ),
                prefix="x",
                savedir="save",
            )

        with self.assertRaisesRegex(ExchangeInputError, "local magnetic-site keys"):
            build_exchange_request(
                "j",
                _common(input_format="epr", slices="0:0:5,2:5:10"),
                prefix="x",
                savedir="save",
            )

        tensor_dj = build_exchange_request(
            "dj_tensor",
            _common(input_format="epr", slices="0:0:5,2:5:10"),
            prefix="d",
            savedir="save",
        )
        self.assertEqual(set(tensor_dj.slice_map), {0, 2})

        with self.assertRaisesRegex(ExchangeInputError, "normalized global atom keys"):
            build_exchange_request(
                "dj_tensor",
                _common(input_format="epr", slices="0:0:5,3:5:10"),
                prefix="d",
                savedir="save",
            )

    def test_common_numerical_validation(self):
        cases = (
            ({"kmesh": [2, 0, 2]}, "positive"),
            ({"kmesh": [2, 2]}, "exactly three"),
            ({"efermi": float("nan")}, "finite"),
            ({"mag_atoms": []}, "at least one"),
            ({"slices": "0:5:5"}, "0<=start<stop"),
            ({"slices": "0:0:5,1:4:8"}, "overlap"),
        )
        for updates, message in cases:
            with (
                self.subTest(updates=updates),
                self.assertRaisesRegex(ExchangeInputError, message),
            ):
                build_exchange_request(
                    "j",
                    _common(input_format="epr", **updates),
                    prefix="x",
                    savedir="save",
                )

    def test_one_based_atoms_are_normalized_to_zero_based(self):
        request = build_exchange_request(
            "j",
            _common(input_format="epr", mag_atoms=[1, 3], mag_atoms_base=1),
            prefix="x",
            savedir="save",
        )
        self.assertEqual(request.mag_atoms, (0, 2))
        self.assertEqual(request.atom_index_base, 1)
        self.assertNotIn("mag_atoms_base", request.options)

    def test_advanced_options_are_allowlisted_immutable_and_pickleable(self):
        request = build_exchange_request(
            "dj",
            _common(
                input_format="epr",
                qmesh=[2, 1, 1],
                nproc=4,
                ddelta_mode="onsite",
                g_kernel="spectral",
            ),
            prefix="d",
            savedir="save",
        )
        self.assertEqual(request.options["qmesh"], (2, 1, 1))
        self.assertEqual(request.options["nproc"], 4)
        self.assertEqual(request.options["g_kernel"], "spectral")
        self.assertEqual(pickle.loads(pickle.dumps(request)), request)

        with self.assertRaisesRegex(ExchangeInputError, "unknown or unsupported"):
            build_exchange_request(
                "dj",
                _common(input_format="epr", material_name="MnTe"),
                prefix="d",
                savedir="save",
            )

        tensor = build_exchange_request(
            "dj_tensor",
            _common(
                input_format="epr",
                onsite_deriv_exchange_field=".false.",
            ),
            prefix="d",
            savedir="save",
        )
        self.assertFalse(tensor.options["onsite_deriv_projector"])
        self.assertNotIn("onsite_deriv_exchange_field", tensor.options)

        with self.assertRaisesRegex(ExchangeInputError, "different boolean values"):
            build_exchange_request(
                "dj_tensor",
                _common(
                    input_format="epr",
                    onsite_deriv_exchange_field=False,
                    onsite_deriv_projector=True,
                ),
                prefix="d",
                savedir="save",
            )

    def test_scalar_j_orbit_symmetry_defaults_and_grouping_contract(self):
        request = build_exchange_request(
            "j",
            _common(input_format="epr"),
            prefix="x",
            savedir="save",
        )
        _module, _function, namespace = build_namespace(request)
        self.assertEqual(namespace.orbit_symmetry, "project")
        self.assertEqual(namespace.orbit_symmetry_tolerance_mev, 1.0e-8)

        diagnostic = build_exchange_request(
            "j",
            _common(input_format="epr", no_symmetry_orbits=True),
            prefix="x",
            savedir="save",
        )
        _module, _function, namespace = build_namespace(diagnostic)
        self.assertEqual(namespace.orbit_symmetry, "report")

        with self.assertRaisesRegex(ExchangeInputError, "requires orbit_grouping='spglib'"):
            build_exchange_request(
                "j",
                _common(
                    input_format="epr",
                    orbit_grouping="shell",
                    orbit_symmetry="project",
                ),
                prefix="x",
                savedir="save",
            )

    def test_scalar_dj_covariant_symmetry_defaults_and_partial_contract(self):
        request = build_exchange_request(
            "dj",
            _common(input_format="epr"),
            prefix="x",
            savedir="save",
        )
        _module, _function, namespace = build_namespace(request)
        self.assertEqual(namespace.covariant_symmetry, "project")
        self.assertEqual(
            namespace.covariant_symmetry_tolerance_mev_per_ang,
            1.0e-8,
        )

        partial = build_exchange_request(
            "dj",
            _common(input_format="epr", axes="x"),
            prefix="x",
            savedir="save",
        )
        _module, _function, namespace = build_namespace(partial)
        self.assertEqual(namespace.covariant_symmetry, "none")

        with self.assertRaisesRegex(ExchangeInputError, "requires axes='xyz'"):
            build_exchange_request(
                "dj",
                _common(
                    input_format="epr",
                    axes="x",
                    covariant_symmetry="project",
                ),
                prefix="x",
                savedir="save",
            )

    def test_source_file_dependencies_and_tensor_target_base(self):
        wannier = {
            "input_format": "wannier",
            "up_hr": "up_hr.dat",
            "dn_hr": "dn_hr.dat",
            "efermi": 0.0,
            "kmesh": (2, 2, 2),
            "mag_atoms": [0],
            "slices": "0:0:5",
        }
        with self.assertRaisesRegex(ExchangeInputError, "provided together"):
            build_exchange_request(
                "j",
                {**wannier, "ref_epr_up": "ref_up.h5"},
                prefix="w",
                savedir="save",
            )

        with self.assertRaisesRegex(ExchangeInputError, "unknown or unsupported"):
            build_exchange_request(
                "j_tensor",
                {
                    **wannier,
                    "up_hr": None,
                    "dn_hr": None,
                    "spinor_hr": "spinor_hr.dat",
                    "groupby": "spin",
                    "dynamic_soc": True,
                },
                prefix="w",
                savedir="save",
            )

        target_request = build_exchange_request(
            "dj_tensor",
            _common(
                input_format="epr",
                mag_atoms=[1, 3],
                mag_atoms_base=1,
                targets=[1, 3],
            ),
            prefix="d",
            savedir="save",
        )
        self.assertEqual(target_request.options["targets"], (0, 2))

    def test_declarative_advanced_schema_rejects_invalid_values(self):
        cases = (
            ("j", {"input_format": "epr", "integrator": "garbage"}, "integrator"),
            ("j", {"input_format": "epr", "nproc": 0}, "nproc"),
            ("dj", {"input_format": "epr", "qmesh": [2, 0, 1]}, "qmesh"),
            (
                "dj",
                {"input_format": "epr", "g_kernel": "truncated"},
                "g_kernel",
            ),
            (
                "j_tensor",
                {"input_format": "epr", "spin_direction": [0.0, 0.0, 0.0]},
                "zero vector",
            ),
            (
                "j_tensor",
                {"input_format": "epr", "axes": "xx"},
                "duplicate axes",
            ),
            ("j", {"input_format": "epr", "hr_unit": "kelvin"}, "hr_unit"),
        )
        for calculation, updates, message in cases:
            with (
                self.subTest(calculation=calculation, updates=updates),
                self.assertRaisesRegex(ExchangeInputError, message),
            ):
                build_exchange_request(
                    calculation,
                    _common(**updates),
                    prefix="x",
                    savedir="save",
                )

        normalized = build_exchange_request(
            "j",
            _common(input_format="epr", no_h5=".false.", cfr_beta="400"),
            prefix="x",
            savedir="save",
        )
        self.assertFalse(normalized.options["no_h5"])
        self.assertEqual(normalized.options["cfr_beta"], 400.0)

    def test_soc_card_and_groupby_contract(self):
        base = {
            "input_format": "wannier",
            "up_hr": "up_hr.dat",
            "dn_hr": "dn_hr.dat",
            "efermi": 0.0,
            "kmesh": (2, 2, 2),
            "mag_atoms": [0],
            "slices": "0:0:5",
        }
        with self.assertRaisesRegex(ExchangeInputError, "unknown or unsupported"):
            build_exchange_request(
                "j_tensor",
                {**base, "soc_p_groups": "0,1,2", "lambda_te": 0.0},
                prefix="w",
                savedir="save",
            )
        with self.assertRaisesRegex(ExchangeInputError, "explicit win"):
            build_exchange_request(
                "j_tensor",
                {
                    **base,
                    "soc_card": {
                        "mode": "atomic",
                        "entries": [{"selector": "Te-p", "lambda_ev": 0.5}],
                    },
                },
                prefix="w",
                savedir="save",
            )

        request = build_exchange_request(
            "j_tensor",
            {
                **base,
                "win": "model.win",
                "soc_card": {
                    "mode": "atomic",
                    "entries": [
                        {"selector": "Te-p", "lambda_ev": 0.5},
                        {"selector": "Mn1-d", "lambda_ev": 0.05},
                    ],
                },
            },
            prefix="w",
            savedir="save",
        )
        self.assertEqual(
            tuple(item.selector for item in request.soc.manifolds),
            ("Te-p", "Mn1-d"),
        )

        with self.assertRaisesRegex(ExchangeInputError, "duplicate manifold"):
            build_exchange_request(
                "j_tensor",
                {
                    **base,
                    "win": "model.win",
                    "soc_card": {
                        "mode": "atomic",
                        "entries": [
                            {"selector": "Te-p", "lambda_ev": 0.5},
                            {"selector": "te-P", "lambda_ev": 0.4},
                        ],
                    },
                },
                prefix="w",
                savedir="save",
            )

        spinor = {**base, "up_hr": None, "dn_hr": None, "spinor_hr": "s.hr"}
        with self.assertRaisesRegex(ExchangeInputError, "requires explicit groupby"):
            build_exchange_request("j_tensor", spinor, prefix="w", savedir="save")
        with self.assertRaisesRegex(ExchangeInputError, "spin or orbital"):
            build_exchange_request(
                "j_tensor",
                {**spinor, "groupby": "wannier_spin"},
                prefix="w",
                savedir="save",
            )

    def test_output_overrides_are_resolved_against_out_dir(self):
        request = build_exchange_request(
            "j",
            _common(
                input_format="epr",
                out_dir="custom",
                out_h5="result.h5",
                out_name="report.txt",
            ),
            prefix="x",
            savedir="ignored.save",
        )
        self.assertEqual(request.output.h5_path, "custom/result.h5")
        self.assertEqual(request.output.text_path, "custom/report.txt")


if __name__ == "__main__":
    unittest.main()
