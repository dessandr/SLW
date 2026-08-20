import os
import sys
import unittest
from unittest import mock

from slw.cli.mpi import MPIContext, MPIUnavailableError
from slw.cli.registry import Backend
from slw.cli.runner import _execution_plan


class MPIFrontendTests(unittest.TestCase):
    def test_multi_rank_environment_requires_mpi4py(self):
        with (
            mock.patch.dict(os.environ, {"OMPI_COMM_WORLD_SIZE": "4"}, clear=True),
            mock.patch.dict(sys.modules, {"mpi4py": None}),
            self.assertRaisesRegex(MPIUnavailableError, "4-rank launch"),
        ):
            MPIContext.discover()

    def test_auto_uses_registered_mpi_backend(self):
        backend = Backend(module="serial.backend", mpi_module="mpi.backend")
        context = MPIContext(comm=object(), rank=0, size=8)
        plan = _execution_plan(backend, requested="auto", context=context)
        self.assertEqual(plan.module, "mpi.backend")
        self.assertTrue(plan.all_ranks)

    def test_auto_guards_serial_backend_under_mpi(self):
        backend = Backend(module="serial.backend")
        context = MPIContext(comm=object(), rank=0, size=8)
        plan = _execution_plan(backend, requested="auto", context=context)
        self.assertEqual(plan.module, "serial.backend")
        self.assertFalse(plan.all_ranks)
        self.assertIn("rank 0", plan.warning)


if __name__ == "__main__":
    unittest.main()
