import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from slw.cli.mpi import MPIContext
from slw.cli.runner import run_stage
from slw.cli.schema import parse_run_config
from slw.exchange.config import build_exchange_request
from slw.exchange.engine import ExchangeRunResult, _execute, prepare_run, run_exchange
from slw.exchange.kernels.dispatch import KernelArtifacts, build_namespace
from slw.exchange.kernels.dj_epr import _axis_list, _validate_local_parallelism


def _parameters(*, ltensor=False, source="epr"):
    values = {
        "input_format": source,
        "ltensor": ltensor,
        "efermi": 0.0,
        "kmesh": [2, 2, 1],
        "mag_atoms": [0, 1],
        "slices": "0:0:2,1:2:4",
    }
    if source == "epr":
        values.update(epr_up="up.h5", epr_dn="dn.h5")
    else:
        values.update(up_hr="up_hr.dat", dn_hr="dn_hr.dat")
    return values


def _run_config(calculation, parameters, *, execution="auto"):
    groups = {
        "control": {"calculation": calculation, "prefix": "toy", "outdir": "out"},
        "parallel": {"execution": execution},
        "exchange": dict(parameters),
    }
    return parse_run_config(groups, stage="exchange", cwd="/")


class ExchangeEngineTests(unittest.TestCase):
    def test_j_tensor_prepares_one_native_serial_plan(self):
        parameters = _parameters(ltensor=True)
        config = _run_config("j", parameters)
        plan = prepare_run(
            calculation="j",
            requested_name="j",
            parameters=parameters,
            config=config,
            context=MPIContext(),
            program="slw_exchange.x",
        )

        self.assertFalse(plan.all_ranks)
        self.assertIn("tensor J", plan.backend_label)
        self.assertIn(("form", "tensor"), plan.summary)

    def test_all_exchange_modes_select_mpi_in_auto_mode(self):
        parameters = _parameters(ltensor=True)
        config = _run_config("dj", parameters)
        plan = prepare_run(
            calculation="dj",
            requested_name="dj",
            parameters=parameters,
            config=config,
            context=MPIContext(comm=object(), rank=0, size=4),
            program="slw_exchange.x",
        )
        self.assertTrue(plan.all_ranks)

        scalar_parameters = _parameters(ltensor=False)
        scalar_config = _run_config("dj", scalar_parameters)
        scalar = prepare_run(
            calculation="dj",
            requested_name="dj",
            parameters=scalar_parameters,
            config=scalar_config,
            context=MPIContext(comm=object(), rank=0, size=4),
            program="slw_exchange.x",
        )
        self.assertTrue(scalar.all_ranks)
        self.assertIsNone(scalar.warning)

    def test_scalar_dj_accepts_rank_local_workers_under_mpi(self):
        parameters = {**_parameters(), "nproc": 4, "omp_threads": 2}
        config = _run_config("dj", parameters)
        plan = prepare_run(
            calculation="dj",
            requested_name="dj",
            parameters=parameters,
            config=config,
            context=MPIContext(comm=object(), rank=0, size=2),
            program="slw_exchange.x",
        )

        self.assertTrue(plan.all_ranks)
        self.assertIn(("local workers/rank", 4), plan.summary)
        self.assertIn(("threads/worker", 2), plan.summary)

        tensor_parameters = {
            key: value for key, value in parameters.items() if key != "omp_threads"
        }
        tensor_parameters["ltensor"] = True
        tensor_config = _run_config("dj", tensor_parameters)
        with self.assertRaisesRegex(ValueError, "requires nproc=1 per rank"):
            prepare_run(
                calculation="dj",
                requested_name="dj",
                parameters=tensor_parameters,
                config=tensor_config,
                context=MPIContext(comm=object(), rank=0, size=2),
                program="slw_exchange.x",
            )

    def test_scalar_dj_local_workers_respect_rank_cpu_affinity(self):
        with (
            mock.patch(
                "slw.exchange.kernels.dj_epr.os.sched_getaffinity",
                return_value={0, 1, 2, 3},
            ),
            self.assertRaisesRegex(ValueError, r"nproc\*omp_threads=8"),
        ):
            _validate_local_parallelism(
                nproc=4,
                omp_threads=2,
                precache_workers=1,
            )

        with mock.patch(
            "slw.exchange.kernels.dj_epr.os.sched_getaffinity",
            return_value={0, 1, 2, 3},
        ):
            self.assertEqual(
                _validate_local_parallelism(
                    nproc=2,
                    omp_threads=2,
                    precache_workers=1,
                ),
                4,
            )

    def test_explicit_mpi_accepts_static_exchange(self):
        parameters = _parameters(ltensor=True)
        config = _run_config("j", parameters, execution="mpi")
        plan = prepare_run(
            calculation="j",
            requested_name="j",
            parameters=parameters,
            config=config,
            context=MPIContext(comm=object(), rank=0, size=2),
            program="slw_exchange.x",
        )
        self.assertTrue(plan.all_ranks)

    def test_reference_namespace_is_built_without_argv_translation(self):
        request = build_exchange_request(
            "j",
            _parameters(source="wannier"),
            prefix="toy",
            savedir="/tmp/toy.save",
        )
        module, function, namespace = build_namespace(request)

        self.assertEqual(module, "slw.exchange.kernels.j_wannier")
        self.assertEqual(function, "run")
        self.assertEqual(namespace.kernel, "scalar")
        self.assertEqual(namespace.mag_atoms_base, 0)
        self.assertEqual(namespace.slices, "0:0:2,1:2:4")
        self.assertTrue(os.path.isabs(namespace.out_dir))
        self.assertTrue(os.path.isabs(namespace.out_name))
        self.assertFalse(
            namespace.out_name.startswith(namespace.out_dir + namespace.out_dir)
        )

    def test_tensor_dj_normalizes_one_based_targets_with_atoms(self):
        parameters = _parameters(ltensor=True)
        parameters["mag_atoms"] = [1, 2]
        parameters["mag_atoms_base"] = 1
        parameters["targets"] = [1, 2]
        request = build_exchange_request(
            "dj",
            parameters,
            prefix="toy",
            savedir="/tmp/toy.save",
        )
        _, _, namespace = build_namespace(request)

        self.assertEqual(namespace.mag_atoms, (0, 1))
        self.assertEqual(namespace.mag_atoms_base, 0)
        self.assertEqual(namespace.targets, (0, 1))

    def test_scalar_dj_axis_input_survives_config_and_kernel_boundary(self):
        for axes in ("xyz", "x,y,z", "xy", "x,y"):
            with self.subTest(axes=axes):
                request = build_exchange_request(
                    "dj",
                    {**_parameters(), "axes": axes},
                    prefix="toy",
                    savedir="/tmp/toy.save",
                )
                _, _, namespace = build_namespace(request)

                self.assertEqual(
                    _axis_list(namespace.axes), list(axes.replace(",", ""))
                )

    def test_scalar_dj_g_kernel_defaults_to_direct_at_kernel_boundary(self):
        request = build_exchange_request(
            "dj",
            _parameters(),
            prefix="toy",
            savedir="/tmp/toy.save",
        )
        _, _, namespace = build_namespace(request)
        self.assertEqual(namespace.g_kernel, "direct")

        spectral = build_exchange_request(
            "dj",
            {**_parameters(), "g_kernel": "spectral"},
            prefix="toy",
            savedir="/tmp/toy.save",
        )
        _, _, spectral_namespace = build_namespace(spectral)
        self.assertEqual(spectral_namespace.g_kernel, "spectral")

    def test_spinor_groupby_and_soc_card_reach_native_namespace(self):
        parameters = {
            "input_format": "wannier",
            "ltensor": True,
            "spinor_hr": "spinor_hr.dat",
            "groupby": "orbital",
            "win": "model.win",
            "efermi": 0.0,
            "kmesh": [1, 1, 1],
            "mag_atoms": [0],
            "slices": "0:0:3",
            "soc_card": {
                "mode": "atomic",
                "entries": [{"selector": "Te-p", "lambda_ev": 0.5}],
            },
        }
        request = build_exchange_request(
            "j", parameters, prefix="toy", savedir="/tmp/toy.save"
        )
        _, _, namespace = build_namespace(request)

        self.assertEqual(namespace.groupby, "orbital")
        self.assertEqual(
            namespace.soc_manifolds,
            ({"selector": "Te-p", "lambda_ev": 0.5},),
        )
        self.assertNotIn("groupby", request.options)

    def test_exchange_dry_run_uses_ltensor_and_touches_no_inputs(self):
        text = """
        &control
          calculation = 'j',
          prefix = 'toy',
          outdir = './scratch'
        /
        &exchange
          input_format = 'epr',
          ltensor = .true.,
          epr_up = 'missing-up.h5',
          epr_dn = 'missing-dn.h5',
          efermi = 0.0,
          kmesh = 2, 2, 1,
          mag_atoms = 0, 1,
          slices = '0:0:2,1:2:4'
        /
        """
        stdout = io.StringIO()
        stderr = io.StringIO()
        previous_stdin = sys.stdin
        previous_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                sys.stdin = io.StringIO(text)
                with (
                    contextlib.redirect_stdout(stdout),
                    contextlib.redirect_stderr(stderr),
                ):
                    code = run_stage("exchange", ["--dry-run"])
                self.assertFalse(os.path.exists(os.path.join(directory, "scratch")))
            finally:
                sys.stdin = previous_stdin
                os.chdir(previous_cwd)

        self.assertEqual(code, 0, stderr.getvalue())
        self.assertIn("form            = tensor", stdout.getvalue())
        self.assertIn("slw.exchange.engine", stdout.getvalue())
        self.assertIn("toy.j_tensor.h5", stdout.getvalue())

    def test_execution_emits_qe_style_phases_progress_and_timing(self):
        with tempfile.TemporaryDirectory() as directory:
            request = build_exchange_request(
                "j",
                {
                    **_parameters(ltensor=True),
                    "out_dir": directory,
                },
                prefix="toy",
                savedir=directory,
            )

            def fake_kernel(received, *, mpi, comm):
                self.assertIs(received, request)
                self.assertFalse(mpi)
                self.assertIsNone(comm)
                print("[J-epr-tensor] completed bond block 1/1")
                paths = (received.output.h5_path, received.output.text_path)
                for path in paths:
                    Path(path).touch()
                return KernelArtifacts(paths=paths)

            stdout = io.StringIO()
            with (
                mock.patch(
                    "slw.exchange.engine.execute_kernel", side_effect=fake_kernel
                ),
                contextlib.redirect_stdout(stdout),
            ):
                result = _execute(
                    request,
                    context=MPIContext(),
                    program="slw_exchange.x",
                    verbosity="normal",
                    all_ranks=False,
                )

        output = stdout.getvalue()
        self.assertEqual(
            result.artifacts, (request.output.h5_path, request.output.text_path)
        )
        self.assertGreaterEqual(result.wall_seconds, 0.0)
        self.assertIn("Running tensor J kernel ...", output)
        self.assertIn("completed bond block 1/1", output)
        self.assertNotIn("Validating exchange output", output)
        self.assertIn("SLW_EXCHANGE : CPU", output)
        self.assertIn("WALL", output)

    def test_public_api_discovers_mpi_and_runs_serial_mode_on_root_only(self):
        request = build_exchange_request(
            "j",
            _parameters(),
            prefix="toy",
            savedir="/tmp/toy.save",
        )

        class RootComm:
            def bcast(self, value, root=0):
                self.values = getattr(self, "values", []) + [value]
                self.root = root
                return value

        comm = RootComm()
        context = MPIContext(comm=comm, rank=0, size=4)
        expected = ExchangeRunResult(request, (), 1.0, 2.0, 1)
        with (
            mock.patch.object(MPIContext, "discover", return_value=context) as discover,
            mock.patch(
                "slw.exchange.engine._execute", return_value=expected
            ) as execute,
        ):
            result = run_exchange(request, execution="serial")

        discover.assert_called_once_with()
        execute.assert_called_once()
        self.assertIs(result, expected)
        self.assertEqual(comm.root, 0)
        self.assertEqual(len(comm.values), 2)

    def test_public_api_nonroot_receives_serial_result_without_calculating(self):
        request = build_exchange_request(
            "j",
            _parameters(),
            prefix="toy",
            savedir="/tmp/toy.save",
        )
        expected = ExchangeRunResult(request, (), 1.0, 2.0, 1)

        class NonRootComm:
            def __init__(self):
                self.calls = 0

            def bcast(self, value, root=0):
                self.calls += 1
                if self.calls == 1:
                    return request, "serial", "slw_exchange.x", "normal"
                self.received = value
                return expected, None

        comm = NonRootComm()
        context = MPIContext(comm=comm, rank=2, size=4)
        with mock.patch("slw.exchange.engine._execute") as execute:
            result = run_exchange(request, context=context, execution="serial")

        execute.assert_not_called()
        self.assertIs(result, expected)
        self.assertEqual(comm.received, (None, None))

    def test_public_api_broadcasts_serial_root_failure(self):
        request = build_exchange_request(
            "j",
            _parameters(),
            prefix="toy",
            savedir="/tmp/toy.save",
        )

        class RootComm:
            def bcast(self, value, root=0):
                return value

        context = MPIContext(comm=RootComm(), rank=0, size=2)
        with (
            mock.patch(
                "slw.exchange.engine._execute",
                side_effect=OSError("write failed"),
            ),
            self.assertRaisesRegex(
                RuntimeError,
                "exchange calculation failed on rank 0: OSError: write failed",
            ),
        ):
            run_exchange(request, context=context, execution="serial")


if __name__ == "__main__":
    unittest.main()
