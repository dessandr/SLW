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
        &parallel
          workers_per_rank = 1,
          threads_per_worker = 2,
          q_chunk_size = 16,
          bond_chunk_size = 8,
          vertex_q_chunk_size = 8,
          self_energy_q_chunk_size = 8
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
        self.assertIn("workers/rank    = 1", stdout.getvalue())
        self.assertIn("threads/worker  = 2", stdout.getvalue())

    def test_legacy_hybrid_maps_common_worker_controls(self):
        input_text = """
        &control
          calculation = 'hybrid',
          verbosity = 'high'
        /
        &parallel
          workers_per_rank = 4,
          threads_per_worker = 2
        /
        &magph
          j_tensor_h5 = 'J.h5',
          dj_tensor_h5 = 'dJ.h5',
          phonon_cache = 'phonon.npz',
          s = 2.5,
          spin_direction = 0.0, 0.0, 1.0,
          kmesh = 4, 4, 2,
          output = 'hybrid.npz'
        /
        """
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            previous = sys.stdin
            sys.stdin = io.StringIO(input_text)
            try:
                code = run_stage("magph", ["--dry-run"])
            finally:
                sys.stdin = previous
        self.assertEqual(code, 0, stderr.getvalue())
        self.assertIn("workers/rank    = 4", stdout.getvalue())
        self.assertIn("threads/worker  = 2", stdout.getvalue())
        self.assertIn("--phonon_nproc 4", stdout.getvalue())
        self.assertIn("--hybrid_nproc 4", stdout.getvalue())

    def test_legacy_magph_rejects_unused_worker_pool(self):
        input_text = """
        &control calculation = 'berry' /
        &parallel workers_per_rank = 4 /
        &magph output = 'berry.npz' /
        """
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            previous = sys.stdin
            sys.stdin = io.StringIO(input_text)
            try:
                code = run_stage("magph", ["--dry-run"])
            finally:
                sys.stdin = previous
        self.assertEqual(code, 2)
        self.assertIn("workers_per_rank", stderr.getvalue())

    def test_native_lifetime_rejects_unused_numba_override(self):
        input_text = """
        &control calculation = 'lifetime' /
        &parallel numba_threads = 4 /
        &magph
          exchange_h5 = 'J.h5',
          derivative_h5 = 'dJ.h5',
          phonon_cache = 'phonon.npz',
          magnetic_order = 'fm',
          spin_magnitudes = 2.5,
          quantization_axis = 0.0, 0.0, 1.0,
          kmesh = 2, 2, 2,
          kshift = 0.5, 0.5, 0.5,
          temperature_k = 300.0,
          broadening_mev = 0.2
        /
        """
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            previous = sys.stdin
            sys.stdin = io.StringIO(input_text)
            try:
                code = run_stage("magph", ["--dry-run"])
            finally:
                sys.stdin = previous
        self.assertEqual(code, 2)
        self.assertIn("does not use &parallel numba_threads", stderr.getvalue())

    def test_native_dispersion_dry_run_accepts_complete_sia(self):
        input_text = """
        &control calculation = 'dispersion', prefix = 'mn' /
        &parallel workers_per_rank = 1, threads_per_worker = 4 /
        &magph
          exchange_h5 = 'J.h5',
          kpath_file = 'bands.win',
          magnetic_order = 'collinear_afm',
          spin_magnitudes = 2.5, 2.5,
          spin_pattern = 1.0, -1.0,
          quantization_axis = 0.0, 0.0, 1.0,
          anisotropy_model = 'uniaxial',
          anisotropy_mev = 0.05,
          anisotropy_axis = 0.0, 0.0, 1.0,
          anisotropy_normalization = 'unit_vector',
          plot = .false.
        /
        """
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            previous = sys.stdin
            sys.stdin = io.StringIO(input_text)
            try:
                code = run_stage("magph", ["--dry-run"])
            finally:
                sys.stdin = previous
        self.assertEqual(code, 0)
        self.assertIn("native dispersion", stdout.getvalue())
        self.assertIn("points/segment", stdout.getvalue())
        self.assertNotIn("Numba threads", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
