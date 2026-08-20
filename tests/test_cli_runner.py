import contextlib
import io
import os
import sys
import tempfile
import unittest

from slw.cli.runner import run_stage


@contextlib.contextmanager
def working_directory(path):
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class RunnerTests(unittest.TestCase):
    def test_native_help_can_describe_all_exchange_sources(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = run_stage("exchange", ["--help-calculation", "j"])
        self.assertEqual(code, 0, stderr.getvalue())
        self.assertIn("input_format='epr' or 'wannier'", stdout.getvalue())

    def test_stdin_dry_run_validates_without_creating_savedir(self):
        input_text = """
        &control
          calculation = 'gkq',
          prefix = 'sample',
          outdir = './scratch'
        /
        &epr
          epr = 'input.h5',
          out = '${savedir}/gkq.h5',
          nproc = 2
        /
        """
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            tempfile.TemporaryDirectory() as directory,
            working_directory(directory),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            previous = sys.stdin
            sys.stdin = io.StringIO(input_text)
            try:
                code = run_stage("epr", ["--dry-run"])
            finally:
                sys.stdin = previous
            self.assertFalse(os.path.exists(os.path.join(directory, "scratch")))
        self.assertEqual(code, 0, stderr.getvalue())
        self.assertIn("Input and calculation parameters validated", stdout.getvalue())
        self.assertIn("sample.save/gkq.h5", stdout.getvalue())

    def test_invalid_backend_argument_returns_usage_error(self):
        input_text = """
        &control calculation = 'gkq' /
        &epr epr = 'input.h5', worker_count = 2 /
        """
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            previous = sys.stdin
            sys.stdin = io.StringIO(input_text)
            try:
                code = run_stage("epr", ["--dry-run"])
            finally:
                sys.stdin = previous
        self.assertEqual(code, 2)
        self.assertIn("unknown input parameter", stderr.getvalue())

    def test_native_lifetime_dry_run_uses_qe_style_input(self):
        input_text = """
        &control
          calculation = 'lifetime',
          prefix = 'sample',
          outdir = './scratch'
        /
        &magph
          exchange_h5 = 'J.h5',
          derivative_h5 = 'dJ.h5',
          phonon_cache = 'phonon.npz',
          magnetic_order = 'fm',
          spin_magnitudes = 2.5,
          quantization_axis = 0.0, 0.0, 1.0,
          kmesh = 4, 4, 2,
          kshift = 0.5, 0.5, 0.5,
          temperature_k = 300.0,
          broadening_mev = 0.2
        /
        """
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            tempfile.TemporaryDirectory() as directory,
            working_directory(directory),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            previous = sys.stdin
            sys.stdin = io.StringIO(input_text)
            try:
                code = run_stage("magph", ["--dry-run"])
            finally:
                sys.stdin = previous
            self.assertFalse(os.path.exists(os.path.join(directory, "scratch")))
        self.assertEqual(code, 0, stderr.getvalue())
        self.assertIn("slw.magph.engine (native lifetime)", stdout.getvalue())
        self.assertIn("execution       = rank-0 serial", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
