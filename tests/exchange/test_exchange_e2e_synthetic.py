from __future__ import annotations

import concurrent.futures
import tempfile
import threading
import unittest
from pathlib import Path

import h5py
import numpy as np

from slw.cli.mpi import MPIContext
from slw.exchange.config import build_exchange_request
from slw.exchange.engine import run_exchange
from slw.exchange.kernels.dispatch import execute

from .fixture_factory import common_parameters, write_toy_epr, write_toy_wannier


class _CollectiveState:
    def __init__(self, size: int) -> None:
        self.size = size
        self.condition = threading.Condition()
        self.allgather: dict[int, object] = {}
        self.allgather_result: list[object] | None = None
        self.gather: dict[int, object] = {}
        self.gather_result: list[object] | None = None


class _ThreadComm:
    def __init__(self, state: _CollectiveState, rank: int) -> None:
        self.state = state
        self.rank = rank

    def Get_rank(self):
        return self.rank

    def Get_size(self):
        return self.state.size

    def allgather(self, value):
        with self.state.condition:
            self.state.allgather[self.rank] = value
            if len(self.state.allgather) == self.state.size:
                self.state.allgather_result = [
                    self.state.allgather[index] for index in range(self.state.size)
                ]
                self.state.condition.notify_all()
            else:
                self.state.condition.wait_for(
                    lambda: self.state.allgather_result is not None
                )
            return list(self.state.allgather_result)

    def gather(self, value, root=0):
        with self.state.condition:
            self.state.gather[self.rank] = value
            if len(self.state.gather) == self.state.size:
                self.state.gather_result = [
                    self.state.gather[index] for index in range(self.state.size)
                ]
                self.state.condition.notify_all()
            else:
                self.state.condition.wait_for(lambda: self.state.gather_result is not None)
            return list(self.state.gather_result) if self.rank == root else None


class SyntheticExchangeEndToEndTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="slw-exchange-test-")
        self.root = Path(self.temp.name)
        self.up = self.root / "up.h5"
        self.down = self.root / "down.h5"
        write_toy_epr(self.up, spin="up")
        write_toy_epr(self.down, spin="down")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _request(self, calculation: str, prefix: str, **updates):
        parameters = common_parameters(self.up, self.down)
        parameters.update(updates)
        return build_exchange_request(
            calculation,
            parameters,
            prefix=prefix,
            savedir=str(self.root / "save"),
        )

    def _execute_two_ranks(self, request):
        state = _CollectiveState(2)

        def run_rank(rank: int):
            return execute(request, mpi=True, comm=_ThreadComm(state, rank))

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(run_rank, rank) for rank in range(2)]
            return [future.result(timeout=15.0) for future in futures]

    @staticmethod
    def _numeric_group_payload(handle: h5py.File, path: str) -> list[np.ndarray]:
        arrays: list[np.ndarray] = []
        node = handle[path]
        if isinstance(node, h5py.Dataset):
            return [np.asarray(node)]
        node.visititems(
            lambda _name, item: arrays.append(np.asarray(item))
            if isinstance(item, h5py.Dataset)
            and np.issubdtype(item.dtype, np.number)
            else None
        )
        return arrays

    def test_all_four_epr_products_are_numerical_and_complete(self) -> None:
        requests = (
            self._request("j", "scalar_j", no_symmetry_orbits=True),
            self._request(
                "j",
                "tensor_j",
                ltensor=True,
                tensor_kernel="direct",
                axes="xyz",
                no_symmetry=True,
            ),
            self._request(
                "dj",
                "scalar_dj",
                eph_unit="ev",
                targets="0",
                axes="x",
                qmesh=(1, 1, 1),
                no_symmetry_orbits=True,
            ),
            self._request(
                "dj",
                "tensor_dj",
                ltensor=True,
                tensor_kernel="tb2j",
                eph_unit="ev",
                targets=(0,),
                disp_axes="x",
                tensor_axes="xyz",
                qmesh=(1, 1, 1),
                numba_threads=1,
                blas_threads=1,
                checkpoint=0,
            ),
        )
        datasets = ("J_r/value", "J_tensor_r", "dJ_r", "dJ_tensor_r")
        for request, expected in zip(requests, datasets, strict=True):
            with self.subTest(mode=request.mode_name):
                result = run_exchange(
                    request,
                    context=MPIContext(),
                    execution="serial",
                    verbosity="quiet",
                )
                self.assertTrue(Path(result.artifacts[0]).is_file())
                with h5py.File(request.output.h5_path) as handle:
                    self.assertIn(expected, handle)
                    finite_payloads = self._numeric_group_payload(handle, expected)
                    self.assertTrue(finite_payloads)
                    self.assertTrue(
                        all(np.all(np.isfinite(values)) for values in finite_payloads)
                    )
                    if request.mode_name == "dj_tensor":
                        self.assertEqual(int(handle.attrs["complete"]), 1)

    def test_two_rank_static_j_matches_serial(self) -> None:
        serial = self._request("j", "serial_j", no_symmetry_orbits=True)
        run_exchange(
            serial,
            context=MPIContext(),
            execution="serial",
            verbosity="quiet",
        )
        with h5py.File(serial.output.h5_path) as handle:
            expected = np.asarray(handle["J_r/value"])

        parallel = self._request("j", "parallel_j", no_symmetry_orbits=True)
        results = self._execute_two_ranks(parallel)
        self.assertEqual(results[0].paths, results[1].paths)
        with h5py.File(parallel.output.h5_path) as handle:
            actual = np.asarray(handle["J_r/value"])
            self.assertEqual(int(handle["basic_data/mpi_size"][()]), 2)
        np.testing.assert_allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12)

    def test_two_rank_tensor_j_and_scalar_dj_match_serial(self) -> None:
        cases = (
            (
                "j",
                {
                    "ltensor": True,
                    "tensor_kernel": "direct",
                    "axes": "xyz",
                    "no_symmetry": True,
                },
                "J_tensor_r",
            ),
            (
                "dj",
                {
                    "eph_unit": "ev",
                    "targets": "0",
                    "axes": "x",
                    "qmesh": (1, 1, 1),
                    "no_symmetry_orbits": True,
                },
                "dJ_r",
            ),
        )
        for calculation, options, dataset in cases:
            with self.subTest(calculation=calculation, dataset=dataset):
                serial = self._request(
                    calculation, f"serial_{dataset.lower()}", **options
                )
                run_exchange(
                    serial,
                    context=MPIContext(),
                    execution="serial",
                    verbosity="quiet",
                )
                with h5py.File(serial.output.h5_path) as handle:
                    expected = self._numeric_group_payload(handle, dataset)

                parallel = self._request(
                    calculation, f"parallel_{dataset.lower()}", **options
                )
                self._execute_two_ranks(parallel)
                with h5py.File(parallel.output.h5_path) as handle:
                    actual = self._numeric_group_payload(handle, dataset)
                    self.assertEqual(int(handle["basic_data/mpi_size"][()]), 2)
                self.assertEqual(len(actual), len(expected))
                for got, want in zip(actual, expected, strict=True):
                    np.testing.assert_allclose(
                        got, want, rtol=1.0e-11, atol=1.0e-11
                    )

    def test_wannier_and_epr_scalar_j_agree_for_same_hamiltonian(self) -> None:
        epr = self._request("j", "epr_j", no_symmetry_orbits=True)
        run_exchange(epr, context=MPIContext(), execution="serial", verbosity="quiet")
        with h5py.File(epr.output.h5_path) as handle:
            epr_j = np.sort(np.asarray(handle["J_r/value"]))

        up_hr, down_hr, win = write_toy_wannier(self.root)
        wannier = build_exchange_request(
            "j",
            {
                "input_format": "wannier",
                "up_hr": str(up_hr),
                "dn_hr": str(down_hr),
                "win": str(win),
                "efermi": 0.0,
                "kmesh": (1, 1, 1),
                "mag_atoms": (0, 1),
                "slices": "0:0:1,1:1:2",
                "hr_unit": "ev",
                "n_shells": 1,
                "d_max": 3.0,
                "emin": -0.5,
                "empoints": 6,
                "nproc": 1,
                "orbit_grouping": "none",
            },
            prefix="wannier_j",
            savedir=str(self.root / "save"),
        )
        run_exchange(
            wannier,
            context=MPIContext(),
            execution="serial",
            verbosity="quiet",
        )
        with h5py.File(wannier.output.h5_path) as handle:
            wannier_j = np.sort(np.asarray(handle["J_tensor_r"])[:, 0, 0])
        np.testing.assert_allclose(wannier_j, epr_j, rtol=1.0e-11, atol=1.0e-11)


if __name__ == "__main__":
    unittest.main()
