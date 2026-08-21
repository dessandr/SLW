from __future__ import annotations

import concurrent.futures
from dataclasses import replace
from pathlib import Path

import numpy as np

from slw.cli.mpi import MPIContext
from slw.magph.dispersion import (
    build_wannier90_kpath,
    compute_magnon_dispersion,
)
from slw.magph.screening import screen_magnetic_configuration
from tests.magph.test_lswt import _fm_chain
from tests.magph.test_parallel import _CollectiveState, _ThreadComm


def _write_path(path: Path) -> None:
    path.write_text(
        """begin kpoint_path
G 0.0 0.0 0.0 X 0.5 0.0 0.0
X 0.5 0.0 0.0 G 0.0 0.0 0.0
end kpoint_path
""",
        encoding="utf-8",
    )


def test_wannier_path_uses_reciprocal_cartesian_distance(tmp_path: Path) -> None:
    source = tmp_path / "bands.win"
    _write_path(source)
    path = build_wannier90_kpath(
        source,
        lattice_ang=np.eye(3),
        points_per_segment=2,
    )
    assert path.n_segments == 2
    assert path.n_points == 6
    np.testing.assert_allclose(path.tick_positions_inv_ang, (0.0, np.pi, 2.0 * np.pi))
    assert path.tick_labels == ("Γ", "X", "Γ")
    np.testing.assert_array_equal(path.segment_offsets, (0, 3, 6))


def test_native_dispersion_is_root_assembled_and_keeps_fm_goldstone(
    tmp_path: Path,
) -> None:
    source = tmp_path / "bands.win"
    _write_path(source)
    exchange = replace(_fm_chain(), lattice_ang=np.eye(3))
    configuration = screen_magnetic_configuration(
        exchange,
        order="fm",
        spin_magnitudes=2.0,
    )
    path = build_wannier90_kpath(
        source,
        lattice_ang=exchange.lattice_ang,
        points_per_segment=2,
    )
    distributed = compute_magnon_dispersion(
        exchange,
        configuration,
        path,
        context=MPIContext(),
    )
    assert distributed.global_result is not None
    result = distributed.global_result
    np.testing.assert_allclose(
        result.energy_mev[:, 0],
        (0.0, 2.0, 4.0, 4.0, 2.0, 0.0),
    )
    np.testing.assert_array_equal(
        result.goldstone_mask[:, 0],
        (True, False, False, False, False, True),
    )


def test_two_mpi_ranks_partition_path_and_only_root_assembles(tmp_path: Path) -> None:
    source = tmp_path / "bands.win"
    _write_path(source)
    exchange = replace(_fm_chain(), lattice_ang=np.eye(3))
    configuration = screen_magnetic_configuration(
        exchange,
        order="fm",
        spin_magnitudes=2.0,
    )
    path = build_wannier90_kpath(
        source,
        lattice_ang=exchange.lattice_ang,
        points_per_segment=2,
    )
    state = _CollectiveState(2)

    def run_rank(rank: int):
        return compute_magnon_dispersion(
            exchange,
            configuration,
            path,
            context=MPIContext(
                comm=_ThreadComm(state, rank),
                rank=rank,
                size=2,
            ),
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        results = [
            future.result(timeout=5.0)
            for future in [executor.submit(run_rank, rank) for rank in range(2)]
        ]
    assert results[0].global_result is not None
    assert results[1].global_result is None
    np.testing.assert_array_equal(results[0].local_indices, (0, 1, 2))
    np.testing.assert_array_equal(results[1].local_indices, (3, 4, 5))
