import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from slw.exchange.config import build_exchange_request
from slw.exchange.legacy.reference import adapter
from slw.exchange.legacy.reference import compute_dJ_epr_tensor_mpi as mpi_driver


class _SingleRankComm:
    """Minimal communicator that intentionally has no Barrier method."""

    def __init__(self):
        self.gather_calls = 0
        self.broadcasts = []

    def Get_rank(self):
        return 0

    def Get_size(self):
        return 1

    def allgather(self, value):
        return [value]

    def gather(self, value, root=0):
        self.gather_calls += 1
        return [value]

    def bcast(self, value, root=0):
        self.broadcasts.append(value)
        return value


def _args():
    return SimpleNamespace(
        checkpoint=0,
        nproc=1,
        out_dir=".",
        out_h5="tensor.h5",
        disp_axes="xyz",
    )


def _setup(*, tasks=()):
    return {
        "axes": "xyz",
        "qmesh": (1, 1, 1),
        "qpts": np.zeros((1, 3)),
        "labels": ["M"],
        "pair_meta": [],
        "tasks": list(tasks),
        "common_payload": {},
    }


class ExchangeReferenceMPITests(unittest.TestCase):
    def test_explicit_communicator_reaches_reference_adapter_unchanged(self):
        request = build_exchange_request(
            "dj",
            {
                "input_format": "epr",
                "ltensor": True,
                "epr_up": "up.h5",
                "epr_dn": "dn.h5",
                "efermi": 0.0,
                "kmesh": (1, 1, 1),
                "mag_atoms": (0,),
                "slices": "0:0:2",
            },
            prefix="toy",
            savedir="/tmp/slw-test",
        )
        comm = object()
        received = []

        def fake_run(namespace, *, comm=None):
            received.append((namespace, comm))

        module = SimpleNamespace(run_mpi=fake_run)
        with mock.patch.object(adapter.importlib, "import_module", return_value=module):
            adapter.execute(request, mpi=True, comm=comm)

        self.assertEqual(len(received), 1)
        self.assertIs(received[0][1], comm)

    def test_setup_failure_is_synchronized_before_gather(self):
        comm = _SingleRankComm()
        with (
            mock.patch.object(
                mpi_driver,
                "_build_common_payload",
                side_effect=ValueError("bad setup"),
            ),
            self.assertRaisesRegex(
                RuntimeError,
                r"MPI setup failed \(rank 0: ValueError: bad setup\)",
            ),
        ):
            mpi_driver.run_mpi(_args(), comm=comm)

        self.assertEqual(comm.gather_calls, 0)
        self.assertEqual(comm.broadcasts, [])

    def test_local_compute_failure_is_synchronized_before_gather(self):
        comm = _SingleRankComm()
        task = (0, "x", object(), "up.h5", "dn.h5", "ry")
        with (
            mock.patch.object(
                mpi_driver,
                "_build_common_payload",
                return_value=_setup(tasks=(task,)),
            ),
            mock.patch.object(
                mpi_driver,
                "_run_target_axis_energy_parallel",
                side_effect=ArithmeticError("kernel failed"),
            ),
            self.assertRaisesRegex(
                RuntimeError,
                r"MPI local compute failed \(rank 0: ArithmeticError: kernel failed\)",
            ),
        ):
            mpi_driver.run_mpi(_args(), comm=comm)

        self.assertEqual(comm.gather_calls, 0)
        self.assertEqual(comm.broadcasts, [])

    def test_root_write_status_is_broadcast_and_no_final_barrier_is_needed(self):
        comm = _SingleRankComm()
        with (
            mock.patch.object(
                mpi_driver,
                "_build_common_payload",
                return_value=_setup(),
            ),
            mock.patch.object(
                mpi_driver,
                "_save_results",
                side_effect=OSError("disk full"),
            ),
            self.assertRaisesRegex(
                RuntimeError,
                r"MPI root write failed \(rank 0: OSError: disk full\)",
            ),
        ):
            mpi_driver.run_mpi(_args(), comm=comm)

        self.assertEqual(comm.gather_calls, 1)
        self.assertEqual(comm.broadcasts, [(0, "OSError", "disk full")])

    def test_peer_phase_error_raises_the_same_message_on_healthy_rank(self):
        comm = mock.Mock()
        comm.allgather.return_value = [None, (1, "ValueError", "peer failed")]

        with self.assertRaisesRegex(
            RuntimeError,
            r"MPI setup failed \(rank 1: ValueError: peer failed\)",
        ):
            mpi_driver._raise_synchronized_phase_error(
                comm,
                rank=0,
                phase="setup",
                error=None,
            )


if __name__ == "__main__":
    unittest.main()
