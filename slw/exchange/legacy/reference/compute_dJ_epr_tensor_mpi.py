# Numerical parity implementation retained behind the native exchange engine.
"""MPI driver for analytic EPR dJ/du tensor calculations.

The work unit is one displaced target atom and Cartesian displacement axis.
Each MPI rank receives a subset of these target-axis tasks.  Within each rank,
the existing compute_dJ_epr_tensor energy-parallel worker can still be used via
--nproc.
"""

from __future__ import annotations

import os
import time
from typing import Any

import h5py
import numba
import numpy as np

from slw.exchange.legacy.diagnose_J_epr_kspace import (
    _build_hk_from_epr,
    _find_nearest_neighbours_from_epr,
    _full_k_mesh,
)
from slw.exchange.legacy.lkag_solver import get_cfr_pole_mesh, get_semicircle_contour
from slw.exchange.legacy.reference.compute_dJ_epr_kspace import (
    _atom_labels,
    _bond_target_q_phase,
)
from slw.exchange.legacy.reference.compute_dJ_epr_tensor import (
    _apply_model_soc,
    _kq_map,
    _load_slices_for_global_atoms,
    _load_spinor_hr_hk,
    _load_spinor_slices_for_global_atoms,
    _normalize_mag_atoms,
    _normalize_targets,
    _parse_axes,
    _precompute_spinor_kdata,
    _run_target_axis_energy_parallel,
    _save_results,
    build_arg_parser,
)
from slw.exchange.legacy.spinor_model import spinor_from_collinear


def _rank_print(rank, msg):
    print(f"[dJ-epr-tensor-mpi][rank {rank}] {msg}", flush=True)


def _exception_record(rank: int, exc: Exception) -> tuple[int, str, str]:
    """Return a small, pickle-safe description of a rank-local exception."""

    detail = str(exc).strip() or repr(exc)
    return int(rank), type(exc).__name__, detail


def _raise_synchronized_phase_error(
    comm: Any,
    *,
    rank: int,
    phase: str,
    error: Exception | None,
) -> None:
    """Raise the same phase failure on every rank before the next collective.

    Every caller must enter this helper.  In particular, a failed rank must not
    raise directly while its peers continue into a gather or broadcast.
    """

    local = None if error is None else _exception_record(rank, error)
    records = comm.allgather(local)
    failures = [record for record in records if record is not None]
    if not failures:
        return
    details = "; ".join(
        f"rank {failed_rank}: {kind}: {message}"
        for failed_rank, kind, message in failures
    )
    synchronized = RuntimeError(f"MPI {phase} failed ({details})")
    if error is not None:
        raise synchronized from error
    raise synchronized


def _raise_broadcast_root_error(
    comm: Any,
    *,
    rank: int,
    phase: str,
    error: Exception | None,
) -> None:
    """Broadcast a root-only phase status and raise identically on all ranks."""

    local = _exception_record(rank, error) if rank == 0 and error is not None else None
    failure = comm.bcast(local, root=0)
    if failure is None:
        return
    failed_rank, kind, message = failure
    synchronized = RuntimeError(
        f"MPI {phase} failed (rank {failed_rank}: {kind}: {message})"
    )
    if error is not None:
        raise synchronized from error
    raise synchronized


def _build_common_payload(args, rank=0):
    blas_threads = max(1, int(args.blas_threads))
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = str(blas_threads)
    try:
        import mkl

        mkl.set_num_threads(blas_threads)
    except ImportError:
        pass
    except Exception as exc:  # noqa: BLE001 - optional MKL runtime
        _rank_print(rank, f"[warn] failed to set blas_threads={args.blas_threads}: {exc}")
    _rank_print(rank, f"blas_threads={blas_threads}")
    try:
        numba.set_num_threads(max(1, int(args.numba_threads)))
        _rank_print(rank, f"numba_threads={numba.get_num_threads()}")
    except Exception as exc:  # noqa: BLE001 - optional Numba runtime
        _rank_print(rank, f"[warn] failed to set numba_threads={args.numba_threads}: {exc}")

    axes = _parse_axes(args.tensor_axes)
    kpts = _full_k_mesh(args.kmesh)
    with h5py.File(args.epr_up, "r") as h5:
        epr_qmesh = tuple(int(x) for x in h5["basic_data/qc_dim"][()])
    qmesh = tuple(int(x) for x in (args.qmesh if args.qmesh is not None else epr_qmesh))
    qpts = _full_k_mesh(qmesh)

    _rank_print(
        rank,
        f"meshes: kmesh={tuple(args.kmesh)} nk={len(kpts)} qmesh={qmesh} "
        f"nq={len(qpts)} tensor_axes={axes}",
    )

    spinor_input = bool(args.spinor_hr)
    mag_atoms = _normalize_mag_atoms(args)
    target_ids = _normalize_targets(args, [mag_atoms[0]])
    all_involved = sorted(set(mag_atoms + target_ids))
    _rank_print(
        rank,
        f"atoms: mag_atoms_0based={mag_atoms} targets_0based={target_ids} "
        f"all_involved={all_involved}",
    )

    if spinor_input:
        if str(args.soc).strip() or abs(float(args.lambda_te)) > 0.0 or str(args.soc_p_groups).strip():
            raise ValueError("--spinor_hr is already a SOC/noncollinear Hamiltonian; remove --soc/--lambda_te/--soc_p_groups")
        _rank_print(
            rank,
            f"Loading base H(k) from spinor_hr={args.spinor_hr} "
            f"basis_order={args.spinor_basis_order} unit={args.spinor_hr_unit}",
        )
        h_spin, spinor_meta = _load_spinor_hr_hk(
            args.spinor_hr,
            kpts,
            apply_degeneracy=bool(args.apply_degeneracy),
            hr_unit=args.spinor_hr_unit,
            basis_order=args.spinor_basis_order,
            win=args.win,
            centres=args.centres,
        )
        dim_col = int(spinor_meta["nwan"])
        slices, slice_labels = _load_spinor_slices_for_global_atoms(args, dim_col, mag_atoms)
        if slice_labels:
            _rank_print(rank, f"mag_subspace={slice_labels}")
    else:
        _rank_print(rank, "Loading H(k) up/dn")
        hk_up = _build_hk_from_epr(args.epr_up, kpts, unit=args.hr_unit)
        hk_dn = _build_hk_from_epr(args.epr_dn, kpts, unit=args.hr_unit)
        dim_col = int(hk_up.shape[1])
        slices = _load_slices_for_global_atoms(args, dim_col, mag_atoms)

        _rank_print(rank, "Building spinor H(k)")
        h_spin = spinor_from_collinear(hk_up, hk_dn, n=args.spin_direction)
        h_spin, soc_entries, soc_win = _apply_model_soc(h_spin, args, dim_col)
        if soc_entries:
            _rank_print(rank, f"applied model SOC entries={len(soc_entries)} win={soc_win or '<manual>'}")

    kdata = _precompute_spinor_kdata(h_spin, slices, args.efermi)
    kdata["kpts"] = kpts
    kdata["qpts"] = qpts

    kq_map_idx = _kq_map(args.kmesh, qmesh)
    labels = _atom_labels(args.epr_up, None)
    neighbours = _find_nearest_neighbours_from_epr(args.epr_up, mag_atoms, n_shells=args.n_shells, d_max=args.d_max)
    pair_meta = []
    for n in neighbours:
        gi, gj = int(n["i"]), int(n["j"])
        if gi in mag_atoms and gj in mag_atoms:
            pair_meta.append(
                {
                    "gi": gi,
                    "gj": gj,
                    "li": gi,
                    "lj": gj,
                    "R": n["R"],
                    "dist": n["distance"],
                    "shell": n.get("shell_idx", 0),
                }
            )
    _rank_print(
        rank,
        f"bonds: n_neighbours={len(neighbours)} n_pair_meta={len(pair_meta)} "
        f"n_shells={args.n_shells} d_max={args.d_max}",
    )

    r_arr = np.asarray([m["R"] for m in pair_meta], dtype=np.float64)
    phase = np.exp(-1j * 2.0 * np.pi * (r_arr @ kpts.T))
    bond_target_q_phase = _bond_target_q_phase(qpts, pair_meta)
    wk = 1.0 / float(len(kpts))
    energy_mesh = (
        get_semicircle_contour(emin=args.emin, emax=0.0, npoints=args.empoints)
        if args.integrator == "contour"
        else get_cfr_pole_mesh(beta_eV_inv=400.0, npoles=args.empoints)
    )
    disp_axes = list(args.disp_axes)
    tasks = [(ia, dax, energy_mesh, args.epr_up, args.epr_dn, args.eph_unit) for ia in target_ids for dax in disp_axes]
    common_payload = {
        "kdata": kdata,
        "kq_map": kq_map_idx,
        "pair_meta": pair_meta,
        "phase": phase,
        "bond_target_q_phase": bond_target_q_phase,
        "wk": wk,
        "axes": axes,
        "slices": slices,
        "spin_direction": tuple(float(x) for x in args.spin_direction),
        "numba_threads": int(args.numba_threads),
        "blas_threads": int(args.blas_threads),
        "progress_every": int(args.progress_every),
        "verbose_worker_init": int(args.verbose_worker_init),
        "onsite_deriv_projector": bool(args.onsite_deriv_projector),
    }
    return {
        "axes": axes,
        "qmesh": qmesh,
        "qpts": qpts,
        "labels": labels,
        "pair_meta": pair_meta,
        "tasks": tasks,
        "common_payload": common_payload,
    }


def run_mpi(args, comm=None):
    """Run tensor dJ collectively on ``comm``.

    Passing the communicator explicitly keeps the numerical driver on the same
    communicator selected by the native frontend.  Standalone legacy use falls
    back to ``MPI.COMM_WORLD``.
    """

    if comm is None:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    t0 = time.time()
    if rank == 0 and int(args.checkpoint) != 0:
        _rank_print(rank, "checkpoint is ignored in MPI mode; rank 0 writes one final HDF5 after gather")
    setup = None
    setup_error = None
    try:
        setup = _build_common_payload(args, rank=rank)
    except Exception as exc:  # noqa: BLE001 - synchronize before collectives
        setup_error = exc
    _raise_synchronized_phase_error(
        comm,
        rank=rank,
        phase="setup",
        error=setup_error,
    )
    if setup is None:  # pragma: no cover - enforced by synchronized failure above
        raise RuntimeError("MPI setup completed without a payload")

    tasks = setup["tasks"]
    local_tasks = [task for it, task in enumerate(tasks) if it % size == rank]
    _rank_print(
        rank,
        f"world_size={size} total_tasks={len(tasks)} local_tasks={len(local_tasks)} "
        f"nproc_per_rank={int(args.nproc)}",
    )
    _rank_print(
        rank,
        "local task list: " + (", ".join(f"(target={ia},axis={dax})" for ia, dax, *_ in local_tasks) or "<none>"),
    )

    local_results = {}
    local_error = None
    nq = len(setup["qpts"])
    qmesh = setup["qmesh"]
    pair_meta = setup["pair_meta"]
    try:
        for task in local_tasks:
            ia, dax, acc = _run_target_axis_energy_parallel(
                task,
                setup["common_payload"],
                int(args.nproc),
            )
            grid = acc.reshape(
                qmesh[0],
                qmesh[1],
                qmesh[2],
                len(pair_meta),
                4,
                4,
            )
            real_grid = np.fft.fftn(grid, axes=(0, 1, 2)) / float(nq)
            local_results[(ia, dax)] = real_grid.reshape(
                -1,
                len(pair_meta),
                4,
                4,
            )
            _rank_print(rank, f"completed target={ia} axis={dax}")
    except Exception as exc:  # noqa: BLE001 - synchronize before result gather
        local_error = exc
    _raise_synchronized_phase_error(
        comm,
        rank=rank,
        phase="local compute",
        error=local_error,
    )

    gathered = None
    gather_error = None
    try:
        gathered = comm.gather(local_results, root=0)
    except Exception as exc:  # noqa: BLE001 - synchronize recoverable MPI errors
        gather_error = exc
    _raise_synchronized_phase_error(
        comm,
        rank=rank,
        phase="result gather",
        error=gather_error,
    )

    write_error = None
    if rank == 0:
        try:
            if gathered is None:
                raise RuntimeError("root did not receive gathered MPI results")
            results = {}
            for part in gathered:
                for key, val in part.items():
                    if key in results:
                        raise RuntimeError(f"Duplicate MPI task result for {key}")
                    results[key] = val
            expected = {(ia, dax) for ia, dax, *_ in tasks}
            missing = sorted(expected - set(results.keys()))
            if missing:
                raise RuntimeError(f"Missing MPI task results: {missing}")
            _save_results(
                args,
                results,
                pair_meta,
                setup["labels"],
                setup["axes"],
                list(args.disp_axes),
                qmesh,
                complete=True,
            )
            _rank_print(rank, f"wrote {os.path.join(args.out_dir, args.out_h5)}")
            _rank_print(rank, f"Done. Total time: {time.time() - t0:.2f}s")
        except Exception as exc:  # noqa: BLE001 - peers await status broadcast
            write_error = exc
    _raise_broadcast_root_error(
        comm,
        rank=rank,
        phase="root write",
        error=write_error,
    )


def main():
    ap = build_arg_parser()
    ap.description = "MPI analytic dJ/du Tensor Calculator for EPR"
    args = ap.parse_args()
    run_mpi(args)


if __name__ == "__main__":
    main()
