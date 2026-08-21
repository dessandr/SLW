from __future__ import annotations

import concurrent.futures
import threading
import unittest

import numpy as np

from slw.cli.mpi import MPIContext
from slw.magph.parallel import (
    CollectiveExecutionError,
    balanced_partition,
    distributed_array_map,
)


class _CollectiveState:
    def __init__(self, size: int):
        self.size = size
        self.condition = threading.Condition()
        self.allgather_values: dict[int, dict[int, object]] = {}
        self.allgather_results: dict[int, list[object]] = {}
        self.gather_values: dict[int, dict[int, object]] = {}
        self.gather_results: dict[int, list[object]] = {}
        self.broadcast_values: dict[int, object] = {}


class _ThreadComm:
    def __init__(self, state: _CollectiveState, rank: int):
        self.state = state
        self.rank = rank
        self.allgather_generation = 0
        self.gather_generation = 0
        self.broadcast_generation = 0

    def allgather(self, value):
        state = self.state
        generation = self.allgather_generation
        with state.condition:
            values = state.allgather_values.setdefault(generation, {})
            values[self.rank] = value
            if len(values) == state.size:
                state.allgather_results[generation] = [
                    values[index] for index in range(state.size)
                ]
                state.condition.notify_all()
            else:
                state.condition.wait_for(lambda: generation in state.allgather_results)
            result = list(state.allgather_results[generation])
        self.allgather_generation += 1
        return result

    def gather(self, value, root=0):
        state = self.state
        generation = self.gather_generation
        with state.condition:
            values = state.gather_values.setdefault(generation, {})
            values[self.rank] = value
            if len(values) == state.size:
                state.gather_results[generation] = [
                    values[index] for index in range(state.size)
                ]
                state.condition.notify_all()
            else:
                state.condition.wait_for(lambda: generation in state.gather_results)
            result = (
                list(state.gather_results[generation]) if self.rank == root else None
            )
        self.gather_generation += 1
        return result

    def bcast(self, value, root=0):
        state = self.state
        generation = self.broadcast_generation
        with state.condition:
            if self.rank == root:
                state.broadcast_values[generation] = value
                state.condition.notify_all()
            else:
                state.condition.wait_for(lambda: generation in state.broadcast_values)
            result = state.broadcast_values[generation]
        self.broadcast_generation += 1
        return result


class NativeMagphParallelTests(unittest.TestCase):
    def test_balanced_partition_is_complete_and_contiguous(self) -> None:
        partitions = [balanced_partition(10, rank, 3) for rank in range(3)]
        self.assertEqual([item.count for item in partitions], [4, 3, 3])
        combined = np.concatenate([item.indices for item in partitions])
        np.testing.assert_array_equal(combined, np.arange(10))

    def test_serial_uses_the_same_vectorized_worker_contract(self) -> None:
        calls = []

        def worker(indices):
            calls.append(indices.copy())
            return np.stack((indices, indices**2), axis=1)

        result = distributed_array_map(
            5,
            worker,
            item_shape=(2,),
            dtype=np.int64,
            context=MPIContext(),
        )
        self.assertEqual(len(calls), 1)
        np.testing.assert_array_equal(calls[0], np.arange(5))
        np.testing.assert_array_equal(
            result.global_values,
            np.stack((np.arange(5), np.arange(5) ** 2), axis=1),
        )
        self.assertFalse(result.global_values.flags.writeable)

    def test_three_ranks_assemble_in_global_index_order(self) -> None:
        state = _CollectiveState(3)

        def run_rank(rank):
            context = MPIContext(
                comm=_ThreadComm(state, rank),
                rank=rank,
                size=3,
            )
            return distributed_array_map(
                8,
                lambda indices: (indices + 1)[:, None] * np.asarray((1.0, 10.0)),
                item_shape=(2,),
                context=context,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(run_rank, rank) for rank in range(3)]
            results = [future.result(timeout=5.0) for future in futures]
        expected = (np.arange(8) + 1)[:, None] * np.asarray((1.0, 10.0))
        np.testing.assert_allclose(results[0].global_values, expected)
        self.assertIsNone(results[1].global_values)
        self.assertIsNone(results[2].global_values)
        np.testing.assert_array_equal(results[0].local_indices, [0, 1, 2])
        np.testing.assert_array_equal(results[1].local_indices, [3, 4, 5])
        np.testing.assert_array_equal(results[2].local_indices, [6, 7])

    def test_rank_failure_is_raised_on_every_rank_before_gather(self) -> None:
        state = _CollectiveState(3)

        def run_rank(rank):
            context = MPIContext(
                comm=_ThreadComm(state, rank),
                rank=rank,
                size=3,
            )

            def worker(indices):
                if rank == 1:
                    raise OSError("local k block failed")
                return indices.astype(np.float64)

            return distributed_array_map(6, worker, context=context)

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(run_rank, rank) for rank in range(3)]
            for future in futures:
                with self.assertRaisesRegex(
                    CollectiveExecutionError,
                    "rank 1 OSError: local k block failed",
                ):
                    future.result(timeout=5.0)
        self.assertEqual(state.gather_values, {})


if __name__ == "__main__":
    unittest.main()
