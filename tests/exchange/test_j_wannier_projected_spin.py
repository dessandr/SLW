from __future__ import annotations

import concurrent.futures
import threading
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

import h5py
import numpy as np
import pytest

from slw.core.wannier_io import write_wannier_hr
from slw.exchange.config import build_exchange_request
from slw.exchange.kernels import j_wannier, wannier_projected
from slw.exchange.kernels.dispatch import build_namespace
from slw.exchange.kernels.lkag import get_semicircle_contour
from slw.exchange.kernels.wannier_projected import (
    _minus_k_indices_from_points,
    compute_projected_tb2j,
    load_projected_wannier_context,
    spin_product_separability,
)

_PAULI = np.asarray(
    (
        ((0.0, 1.0), (1.0, 0.0)),
        ((0.0, -1.0j), (1.0j, 0.0)),
        ((1.0, 0.0), (0.0, -1.0)),
    ),
    dtype=np.complex128,
)
_KPOINTS = np.asarray(((0.0, 0.0, 0.0), (0.5, 0.0, 0.0)))
_ACTIVE_BANDS = np.asarray((1, 2, 4, 5), dtype=np.int64)


@dataclass(frozen=True)
class _WannierBundle:
    hr: Path
    win: Path
    amn: Path
    eig: Path
    spn: Path
    u_mat: Path
    u_dis_global: Path
    u_dis_compact: Path
    hamiltonian_atomic: np.ndarray


class _CollectiveState:
    def __init__(self, size: int) -> None:
        self.size = size
        self.condition = threading.Condition()
        self.allgather: dict[int, dict[int, object]] = {}
        self.allgather_result: dict[int, list[object]] = {}
        self.gather: dict[int, dict[int, object]] = {}
        self.gather_result: dict[int, list[object]] = {}
        self.broadcast: dict[int, object] = {}


class _ThreadComm:
    """Small blocking communicator for deterministic thread-rank tests."""

    def __init__(self, state: _CollectiveState, rank: int) -> None:
        self.state = state
        self.rank = rank
        self.allgather_sequence = 0
        self.gather_sequence = 0
        self.broadcast_sequence = 0

    def Get_rank(self) -> int:
        return self.rank

    def Get_size(self) -> int:
        return self.state.size

    def allgather(self, value):
        sequence = self.allgather_sequence
        self.allgather_sequence += 1
        with self.state.condition:
            pending = self.state.allgather.setdefault(sequence, {})
            pending[self.rank] = value
            if len(pending) == self.state.size:
                self.state.allgather_result[sequence] = [
                    pending[index] for index in range(self.state.size)
                ]
                self.state.condition.notify_all()
            else:
                self.state.condition.wait_for(
                    lambda: sequence in self.state.allgather_result
                )
            return list(self.state.allgather_result[sequence])

    def gather(self, value, root: int = 0):
        sequence = self.gather_sequence
        self.gather_sequence += 1
        with self.state.condition:
            pending = self.state.gather.setdefault(sequence, {})
            pending[self.rank] = value
            if len(pending) == self.state.size:
                self.state.gather_result[sequence] = [
                    pending[index] for index in range(self.state.size)
                ]
                self.state.condition.notify_all()
            else:
                self.state.condition.wait_for(
                    lambda: sequence in self.state.gather_result
                )
            if self.rank == root:
                return list(self.state.gather_result[sequence])
            return None

    def bcast(self, value, root: int = 0):
        sequence = self.broadcast_sequence
        self.broadcast_sequence += 1
        with self.state.condition:
            if self.rank == root:
                self.state.broadcast[sequence] = value
                self.state.condition.notify_all()
            else:
                self.state.condition.wait_for(
                    lambda: sequence in self.state.broadcast
                )
            return self.state.broadcast[sequence]


def _atomic_hamiltonians() -> np.ndarray:
    """Two AFM sites with spin-independent hopping and no SOC."""

    result = np.zeros((2, 4, 4), dtype=np.complex128)
    exchange = 0.72
    scalar_staggering = 0.11
    for ik, hopping in enumerate((0.31, 0.13)):
        result[ik] = np.diag(
            (
                scalar_staggering + exchange,
                scalar_staggering - exchange,
                -scalar_staggering - exchange,
                -scalar_staggering + exchange,
            )
        )
        result[ik, 0, 2] = result[ik, 2, 0] = hopping
        result[ik, 1, 3] = result[ik, 3, 1] = hopping
    return result


def _random_unitary(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(4, 4)) + 1.0j * rng.normal(size=(4, 4))
    unitary, triangular = np.linalg.qr(matrix)
    phases = np.diag(triangular).copy()
    phases /= np.abs(phases)
    return np.asarray(unitary * phases.conj()[None, :], dtype=np.complex128)


def _write_u_matrix(path: Path, kpoints: np.ndarray, matrices: np.ndarray) -> None:
    nk, nband, nwann = matrices.shape
    rows = ["synthetic formatted Wannier90 rotation", f"{nk} {nwann} {nband}"]
    for ik, kpoint in enumerate(kpoints):
        rows.append("")
        rows.append(" ".join(f"{value:.17e}" for value in kpoint))
        for iw in range(nwann):
            for ib in range(nband):
                value = matrices[ik, ib, iw]
                rows.append(f"{value.real:.17e} {value.imag:.17e}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_amn(path: Path, matrices: np.ndarray) -> None:
    nk, nband, nproj = matrices.shape
    rows = ["synthetic exact atomic projections", f"{nband} {nk} {nproj}"]
    for ik in range(nk):
        for projection in range(nproj):
            for band in range(nband):
                value = matrices[ik, band, projection]
                rows.append(
                    f"{band + 1} {projection + 1} {ik + 1} "
                    f"{value.real:.17e} {value.imag:.17e}"
                )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_eig(path: Path, eigenvalues: np.ndarray) -> None:
    rows = []
    for ik in range(eigenvalues.shape[0]):
        for band in range(eigenvalues.shape[1]):
            rows.append(f"{band + 1} {ik + 1} {eigenvalues[ik, band]:.17e}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_formatted_spn(path: Path, spin_matrices: np.ndarray) -> None:
    nk, nspin, nband, nband_second = spin_matrices.shape
    assert nspin == 3 and nband == nband_second
    rows = ["synthetic formatted physical Pauli matrices", f"{nband} {nk}"]
    for ik in range(nk):
        for column in range(nband):
            for row in range(column + 1):
                values = spin_matrices[ik, :, row, column]
                rows.append(
                    " ".join(
                        f"{part:.17e}"
                        for value in values
                        for part in (value.real, value.imag)
                    )
                )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _make_bundle(
    root: Path,
    *,
    gauge_rotated: bool,
    spin_axis: str = "z",
) -> _WannierBundle:
    root.mkdir(parents=True, exist_ok=True)
    h_atomic = _atomic_hamiltonians()
    if spin_axis == "y":
        spin_rotation = (
            np.eye(2, dtype=np.complex128) - 1.0j * _PAULI[0]
        ) / np.sqrt(2.0)
        full_rotation = np.kron(
            np.eye(2, dtype=np.complex128), spin_rotation
        )
        h_atomic = np.asarray(
            full_rotation[None, ...]
            @ h_atomic
            @ full_rotation.conj().T[None, ...],
            dtype=np.complex128,
        )
    elif spin_axis != "z":
        raise ValueError(f"unsupported synthetic spin axis {spin_axis!r}")
    gauges = (
        np.stack((_random_unitary(731), _random_unitary(991)))
        if gauge_rotated
        else np.broadcast_to(np.eye(4, dtype=np.complex128), (2, 4, 4)).copy()
    )

    # The Bloch space has one extra, nonmagnetic spin pair outside the outer
    # window.  Its interspersed band indices make the compact-U_dis test a real
    # row-remapping test instead of an identity-shaped special case.
    global_eigenvalues = np.empty((2, 6), dtype=np.float64)
    amn = np.zeros((2, 6, 4), dtype=np.complex128)
    spn = np.empty((2, 3, 6, 6), dtype=np.complex128)
    u = np.empty((2, 4, 4), dtype=np.complex128)
    h_wannier = np.empty_like(h_atomic)
    spin_physical = np.asarray(
        [np.kron(np.eye(3, dtype=np.complex128), axis) for axis in _PAULI]
    )
    for ik in range(2):
        active_eigenvalues, active_eigenvectors = np.linalg.eigh(h_atomic[ik])
        global_eigenvalues[ik] = (
            -8.0,
            active_eigenvalues[0],
            active_eigenvalues[1],
            8.0,
            active_eigenvalues[2],
            active_eigenvalues[3],
        )
        bloch_vectors = np.zeros((6, 6), dtype=np.complex128)
        bloch_vectors[4, 0] = 1.0
        bloch_vectors[:4, _ACTIVE_BANDS] = active_eigenvectors
        bloch_vectors[5, 3] = 1.0
        amn[ik] = bloch_vectors.conj().T[:, :4]
        spn[ik] = np.einsum(
            "ib,aij,jc->abc",
            bloch_vectors.conj(),
            spin_physical,
            bloch_vectors,
            optimize=True,
        )
        u[ik] = active_eigenvectors.conj().T @ gauges[ik]
        h_wannier[ik] = gauges[ik].conj().T @ h_atomic[ik] @ gauges[ik]

    hr = root / "synthetic_hr.dat"
    write_wannier_hr(
        hr,
        4,
        (1, 1),
        {
            (0, 0, 0): 0.5 * (h_wannier[0] + h_wannier[1]),
            (1, 0, 0): 0.5 * (h_wannier[0] - h_wannier[1]),
        },
    )
    win = root / "synthetic.win"
    win.write_text(
        """num_bands = 6
num_wann = 4
dis_win_min = -2.0
dis_win_max = 2.0
begin unit_cell_cart
ang
4 0 0
0 4 0
0 0 4
end unit_cell_cart
begin atoms_frac
Mn1 0.0 0.0 0.0
Mn2 0.5 0.0 0.0
end atoms_frac
begin projections
Mn1:s
Mn2:s
end projections
""",
        encoding="utf-8",
    )
    eig = root / "synthetic.eig"
    amn_path = root / "synthetic.amn"
    spn_path = root / "synthetic.spn"
    u_path = root / "synthetic_u.mat"
    u_dis_global = root / "synthetic_u_dis_global.mat"
    u_dis_compact = root / "synthetic_u_dis_compact.mat"
    _write_eig(eig, global_eigenvalues)
    _write_amn(amn_path, amn)
    _write_formatted_spn(spn_path, spn)
    _write_u_matrix(u_path, _KPOINTS, u)

    expanded_dis = np.zeros((2, 6, 4), dtype=np.complex128)
    expanded_dis[:, _ACTIVE_BANDS, :] = np.eye(4, dtype=np.complex128)
    compact_dis = np.zeros_like(expanded_dis)
    compact_dis[:, :4, :] = np.eye(4, dtype=np.complex128)
    _write_u_matrix(u_dis_global, _KPOINTS, expanded_dis)
    _write_u_matrix(u_dis_compact, _KPOINTS, compact_dis)
    return _WannierBundle(
        hr=hr,
        win=win,
        amn=amn_path,
        eig=eig,
        spn=spn_path,
        u_mat=u_path,
        u_dis_global=u_dis_global,
        u_dis_compact=u_dis_compact,
        hamiltonian_atomic=h_atomic,
    )


def _load(bundle: _WannierBundle, *, compact: bool = False):
    return load_projected_wannier_context(
        spinor_hr=bundle.hr,
        win=bundle.win,
        amn=bundle.amn,
        eig=bundle.eig,
        spn=bundle.spn,
        u_mat=bundle.u_mat,
        u_dis_mat=(bundle.u_dis_compact if compact else bundle.u_dis_global),
        u_dis_layout=("compact_outer_window" if compact else "global_bands"),
        groupby="orbital",
        mag_atoms=(0, 1),
        kmesh=(2, 1, 1),
        kpoints=_KPOINTS,
        efermi=0.0,
    )


def _exchange_result(context):
    pair_meta = [{"li": 0, "lj": 1, "R": (0, 0, 0)}]
    return compute_projected_tb2j(
        context,
        pair_meta,
        get_semicircle_contour(emin=-2.0, emax=0.0, npoints=48),
        ("x", "y", "z"),
        collinear_override=True,
    )


def _projected_run_request(
    bundle: _WannierBundle,
    savedir: Path,
    *,
    prefix: str,
):
    return build_exchange_request(
        "j_tensor",
        {
            "input_format": "wannier",
            "spinor_hr": str(bundle.hr),
            "win": str(bundle.win),
            "amn": str(bundle.amn),
            "eig": str(bundle.eig),
            "spn": str(bundle.spn),
            "u_mat": str(bundle.u_mat),
            "u_dis_mat": str(bundle.u_dis_global),
            "u_dis_layout": "global_bands",
            "spin_operator": "spn",
            "groupby": "orbital",
            "efermi": 0.0,
            "kmesh": (2, 1, 1),
            "mag_atoms": (0, 1),
            "tensor_kernel": "tb2j",
            "axes": "xyz",
            "n_shells": 1,
            "d_max": 2.1,
            "all_bonds": True,
            "orbit_grouping": "none",
            "integrator": "contour",
            "emin": -2.0,
            # Five points split 3+2 over two ranks and therefore exercise an
            # uneven collective reduction rather than two identical chunks.
            "empoints": 5,
            "nproc": 1,
            "collinear_override": True,
        },
        prefix=prefix,
        savedir=savedir,
    )


def _run_two_thread_ranks(request, state: _CollectiveState):
    def run_rank(rank: int):
        _module, _function, namespace = build_namespace(request)
        return j_wannier.run(namespace, comm=_ThreadComm(state, rank))

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(run_rank, rank) for rank in range(2)]
        return [future.result(timeout=30.0) for future in futures]


def _read_exchange_h5(path: str | Path):
    with h5py.File(path, "r") as handle:
        return {
            "tensor": handle["J_tensor_r"][:],
            "trace": handle["trace_acc"][:],
            "extra": {
                name: dataset[:]
                for name, dataset in handle["extra"].items()
            },
            "mpi_size": int(handle["basic_data/mpi_size"][()]),
            "nE": int(handle["basic_data/nE"][()]),
            "spin_direction": handle["basic_data/spin_direction"][:],
            "native_kpoints": handle[
                "basic_data/native_kpoints_crystal"
            ][:],
            "spin_direction_source": handle[
                "basic_data/spin_direction_source"
            ][()].decode(),
            "groupby_semantics": handle[
                "basic_data/groupby_semantics"
            ][()].decode(),
            "additional_soc_mode": handle[
                "basic_data/additional_soc_mode"
            ][()].decode(),
            "base_soc_provenance": handle[
                "basic_data/base_hamiltonian_soc_provenance"
            ][()].decode(),
            "collinear_override": bool(
                handle["basic_data/collinear_override"][()]
            ),
            "selector_index_basis": str(
                handle["magnetic_subspace"].attrs["selector_index_basis"]
            ),
        }


def test_loader_reconstructs_exact_projected_atomic_frame(tmp_path: Path) -> None:
    bundle = _make_bundle(tmp_path / "identity", gauge_rotated=False)
    context = _load(bundle)

    assert context.eigenvalues.shape == (2, 4)
    assert context.coefficients.shape == (2, 4, 4)
    np.testing.assert_array_equal(context.site_offsets, (0, 2, 4))
    np.testing.assert_array_equal(context.site_orbital_counts, (1, 1))
    np.testing.assert_array_equal(context.selected_amn_columns, (0, 1, 2, 3))
    np.testing.assert_array_equal(context.selected_spatial_indices, (0, 1))
    np.testing.assert_allclose(
        context.site_spin_directions,
        ((0.0, 0.0, 1.0), (0.0, 0.0, -1.0)),
        atol=2.0e-14,
    )

    reconstructed = np.einsum(
        "kia,ka,kja->kij",
        context.coefficients,
        context.eigenvalues,
        context.coefficients.conj(),
        optimize=True,
    )
    np.testing.assert_allclose(reconstructed, bundle.hamiltonian_atomic, atol=2.0e-13)
    diagnostics = context.diagnostics
    assert diagnostics["operator_mode"] == (
        "amn_anchored_atomic_pauli_spn_validated"
    )
    assert diagnostics["exchange_vertex_operator"] == "atomic_trial_pauli"
    assert diagnostics["spn_usage"] == "projection_frame_validation_only"
    assert diagnostics["projection_labels"] == ("Mn1:s", "Mn2:s")
    assert diagnostics["u_isometry_residual"] < 2.0e-15
    assert diagnostics["atomic_frame_isometry_residual"] < 2.0e-15
    assert diagnostics["hamiltonian_max_residual_ev"] < 2.0e-13
    assert diagnostics["spn_hermiticity_residual"] < 2.0e-15
    assert diagnostics["spin_projection_relative_max"] < 2.0e-15
    assert diagnostics["projection_min_singular_value"] > 1.0 - 2.0e-15
    assert diagnostics["threshold_spin_projection_tolerance"] == 0.4
    assert diagnostics["threshold_intersite_xc_tolerance"] == 0.1


def test_compact_u_dis_expands_interspersed_outer_window_rows(tmp_path: Path) -> None:
    bundle = _make_bundle(tmp_path / "compact", gauge_rotated=True)
    global_context = _load(bundle)
    compact_context = _load(bundle, compact=True)

    assert compact_context.diagnostics["u_dis_layout"] == "compact_outer_window"
    np.testing.assert_allclose(compact_context.eigenvalues, global_context.eigenvalues)
    np.testing.assert_allclose(
        compact_context.coefficients,
        global_context.coefficients,
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        compact_context.site_spin_directions,
        global_context.site_spin_directions,
        atol=2.0e-13,
    )


def test_minus_k_lookup_uses_native_shifted_order() -> None:
    points = np.asarray(
        (
            (0.75, 0.5, 0.0),
            (0.25, 0.0, 0.0),
            (0.75, 0.0, 0.0),
            (0.25, 0.5, 0.0),
        )
    )
    partners = _minus_k_indices_from_points(points)
    np.testing.assert_array_equal(partners, (3, 2, 1, 0))
    summed = points + points[partners]
    np.testing.assert_allclose(summed - np.rint(summed), 0.0, atol=1.0e-15)


def test_nonfinite_spn_is_rejected_before_projection(tmp_path: Path) -> None:
    bundle = _make_bundle(tmp_path / "nonfinite_spn", gauge_rotated=False)
    invalid = np.zeros((3, 6, 6), dtype=np.complex128)
    invalid[0, 0, 0] = np.nan
    with (
        mock.patch.object(
            wannier_projected,
            "iter_wannier_spn",
            return_value=iter((invalid, invalid)),
        ),
        pytest.raises(ValueError, match="SPN contains non-finite values"),
    ):
        _load(bundle)


def test_no_soc_collinear_afm_has_only_nonzero_isotropic_exchange(
    tmp_path: Path,
) -> None:
    context = _load(_make_bundle(tmp_path / "no_soc", gauge_rotated=False))
    tensor, _accumulator, extra, _stats = _exchange_result(context)

    assert abs(extra["jiso_tb2j"][0]) > 1.0e-3
    np.testing.assert_allclose(extra["dmi_tb2j"], 0.0, atol=2.0e-11)
    np.testing.assert_allclose(extra["J_gamma_r"], 0.0, atol=2.0e-11)
    np.testing.assert_allclose(
        tensor[0],
        np.eye(3) * extra["jiso_tb2j"][0],
        rtol=2.0e-11,
        atol=2.0e-11,
    )


def test_no_soc_y_axis_override_is_coordinate_covariant(tmp_path: Path) -> None:
    context = _load(
        _make_bundle(
            tmp_path / "no_soc_y",
            gauge_rotated=True,
            spin_axis="y",
        )
    )
    np.testing.assert_allclose(
        np.abs(context.site_spin_directions),
        ((0.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
        atol=2.0e-13,
    )
    tensor, _accumulator, extra, _stats = _exchange_result(context)
    assert abs(extra["jiso_tb2j"][0]) > 1.0e-3
    np.testing.assert_allclose(extra["dmi_tb2j"], 0.0, atol=2.0e-11)
    np.testing.assert_allclose(extra["J_gamma_r"], 0.0, atol=2.0e-11)
    np.testing.assert_allclose(
        tensor[0],
        np.eye(3) * extra["jiso_tb2j"][0],
        rtol=2.0e-11,
        atol=2.0e-11,
    )


def test_projected_exchange_is_invariant_under_k_dependent_wannier_gauge(
    tmp_path: Path,
) -> None:
    identity = _load(_make_bundle(tmp_path / "identity", gauge_rotated=False))
    rotated = _load(_make_bundle(tmp_path / "rotated", gauge_rotated=True))

    identity_result = _exchange_result(identity)
    rotated_result = _exchange_result(rotated)
    np.testing.assert_allclose(
        rotated_result[0], identity_result[0], rtol=2.0e-11, atol=2.0e-11
    )
    for key in ("jiso_tb2j", "dmi_tb2j", "J_gamma_r", "J_aab_full_r"):
        np.testing.assert_allclose(
            rotated_result[2][key],
            identity_result[2][key],
            rtol=2.0e-11,
            atol=2.0e-11,
        )


def test_spin_product_separability_identifies_single_and_mixed_axes() -> None:
    collinear = np.zeros((2, 4, 4), dtype=np.complex128)
    collinear[:, :2, :2] = np.diag((0.7, -0.2))
    collinear[:, 2:, 2:] = -np.diag((0.7, -0.2))
    exact = spin_product_separability(collinear)
    assert exact["residual"] < 1.0e-15
    np.testing.assert_allclose(exact["axis"], (0.0, 0.0, 1.0), atol=1.0e-15)
    assert exact["spin_dependent_norm"] > 0.0

    mixed = np.zeros_like(collinear)
    mixed[0, :2, 2:] = np.diag((1.0, 0.0))
    mixed[0, 2:, :2] = np.diag((1.0, 0.0))
    mixed[1, :2, :2] = np.diag((0.0, 1.0))
    mixed[1, 2:, 2:] = -np.diag((0.0, 1.0))
    nonseparable = spin_product_separability(mixed)
    assert nonseparable["residual"] == 0.5
    np.testing.assert_allclose(nonseparable["eigenvalues"], (0.5, 0.5, 0.0))


def test_rank_one_pauli_diagnostic_does_not_auto_certify_partner_gauge(
    tmp_path: Path,
) -> None:
    up = np.diag((-0.8, -0.3)).astype(np.complex128)
    rotation = np.asarray(((1.0, 1.0), (-1.0, 1.0))) / np.sqrt(2.0)
    down = rotation.conj().T @ np.diag((0.4, 0.9)) @ rotation
    spin_major = np.zeros((1, 4, 4), dtype=np.complex128)
    spin_major[0, :2, :2] = up
    spin_major[0, 2:, 2:] = down
    diagnostic = spin_product_separability(spin_major)
    assert diagnostic["residual"] < 1.0e-15

    hr = tmp_path / "independent_spin_gauges_hr.dat"
    write_wannier_hr(hr, 4, (1,), {(0, 0, 0): spin_major[0]})
    args = Namespace(
        axes="xyz",
        kmesh=(1, 1, 1),
        spinor_hr=str(hr),
        up_hr=None,
        dn_hr=None,
        amn=None,
        eig=None,
        spn=None,
        u_mat=None,
        u_dis_mat=None,
        spin_operator="auto",
        u_dis_layout=None,
        kernel="tb2j",
        soc_manifolds=(),
        soc="",
        lambda_te=0.0,
        intersite_soc=False,
        dynamic_soc=False,
        apply_degeneracy=True,
        hr_unit="ev",
        groupby="spin",
        win=None,
        centres=None,
    )
    with pytest.raises(ValueError, match="cannot certify.*orbital partner gauge"):
        j_wannier.run(args)


def test_public_projected_run_two_rank_matches_serial_and_loads_on_root_once(
    tmp_path: Path,
) -> None:
    bundle = _make_bundle(tmp_path / "bundle", gauge_rotated=True)
    serial_request = _projected_run_request(
        bundle,
        tmp_path / "serial",
        prefix="serial",
    )
    _module, _function, serial_args = build_namespace(serial_request)
    j_wannier.run(serial_args)
    serial = _read_exchange_h5(serial_request.output.h5_path)

    mpi_request = _projected_run_request(
        bundle,
        tmp_path / "mpi",
        prefix="mpi",
    )
    state = _CollectiveState(2)
    original_loader = j_wannier.load_projected_wannier_context
    original_compute = j_wannier.compute_projected_tb2j
    loader_calls = []
    local_energy_counts = []
    record_lock = threading.Lock()

    def counted_loader(**kwargs):
        with record_lock:
            loader_calls.append(threading.get_ident())
        return original_loader(**kwargs)

    def counted_compute(context, pair_meta, energy_mesh, axes, **kwargs):
        with record_lock:
            local_energy_counts.append(len(energy_mesh))
        return original_compute(context, pair_meta, energy_mesh, axes, **kwargs)

    with (
        mock.patch.object(
            j_wannier,
            "load_projected_wannier_context",
            side_effect=counted_loader,
        ),
        mock.patch.object(
            j_wannier,
            "compute_projected_tb2j",
            side_effect=counted_compute,
        ),
    ):
        assert _run_two_thread_ranks(mpi_request, state) == [None, None]

    assert len(loader_calls) == 1
    assert sorted(local_energy_counts) == [2, 3]
    assert len(state.gather) == 1
    distributed = _read_exchange_h5(mpi_request.output.h5_path)
    assert serial["mpi_size"] == 1
    assert distributed["mpi_size"] == 2
    assert serial["nE"] == distributed["nE"] == 5
    np.testing.assert_allclose(
        serial["spin_direction"], (0.0, 0.0, 1.0), atol=2.0e-14
    )
    np.testing.assert_allclose(
        distributed["spin_direction"], serial["spin_direction"]
    )
    assert serial["spin_direction_source"] == (
        "projected_time_reversal_odd_field_site_0"
    )
    assert distributed["spin_direction_source"] == serial["spin_direction_source"]
    assert serial["groupby_semantics"] == "amn_projection_columns"
    np.testing.assert_allclose(serial["native_kpoints"], _KPOINTS)
    np.testing.assert_allclose(
        distributed["native_kpoints"], serial["native_kpoints"]
    )
    assert serial["additional_soc_mode"] == "none"
    assert serial["base_soc_provenance"] == (
        "unknown_embedded_content_of_spinor_hr"
    )
    assert serial["collinear_override"] is True
    assert serial["selector_index_basis"] == "win_spatial_projection"
    np.testing.assert_allclose(
        distributed["tensor"], serial["tensor"], rtol=2.0e-12, atol=2.0e-12
    )
    np.testing.assert_allclose(
        distributed["trace"], serial["trace"], rtol=2.0e-12, atol=2.0e-12
    )
    assert distributed["extra"].keys() == serial["extra"].keys()
    for name in serial["extra"]:
        np.testing.assert_allclose(
            distributed["extra"][name],
            serial["extra"][name],
            rtol=2.0e-12,
            atol=2.0e-12,
        )


def test_projected_rank_local_failure_is_synchronized_before_gather(
    tmp_path: Path,
) -> None:
    bundle = _make_bundle(tmp_path / "bundle", gauge_rotated=True)
    request = _projected_run_request(
        bundle,
        tmp_path / "failure",
        prefix="failure",
    )
    state = _CollectiveState(2)
    original_compute = j_wannier.compute_projected_tb2j

    def fail_short_partition(context, pair_meta, energy_mesh, axes, **kwargs):
        if len(energy_mesh) == 2:
            raise ArithmeticError("synthetic projected integration failure")
        return original_compute(context, pair_meta, energy_mesh, axes, **kwargs)

    def run_rank(rank: int):
        _module, _function, namespace = build_namespace(request)
        return j_wannier.run(namespace, comm=_ThreadComm(state, rank))

    with (
        mock.patch.object(
            j_wannier,
            "compute_projected_tb2j",
            side_effect=fail_short_partition,
        ),
        concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor,
    ):
        futures = [executor.submit(run_rank, rank) for rank in range(2)]
        errors = []
        for future in futures:
            try:
                future.result(timeout=30.0)
            except RuntimeError as exc:
                errors.append(str(exc))

    expected_error = (
        "native exchange MPI work failed: rank 1 ArithmeticError: "
        "synthetic projected integration failure"
    )
    assert errors == [expected_error, expected_error]
    assert state.gather == {}
    assert not Path(request.output.h5_path).exists()
