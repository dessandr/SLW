"""Native exchange-striction contraction in the phonon mode basis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from slw.cli.mpi import MPIContext

from .derivative import ExchangeDerivativeModel
from .model import ExchangeModel, ExchangeRepresentation
from .parallel import distributed_array_map
from .phonon import PhononCache, ZeroPointDisplacement


def _readonly(value: object, *, dtype: np.dtype) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ModeResolvedExchangeDerivative:
    """Phonon-mode derivative ``Lambda(q,nu,bond)`` in meV."""

    q_points_frac: NDArray[np.float64]
    phonon_energy_mev: NDArray[np.float64]
    lambda_mev: NDArray[np.complex128]
    target_atom_indices: NDArray[np.int64]
    target_coverage_complete: bool
    frequency_regularized: NDArray[np.bool_]
    q_mesh_shape: tuple[int, int, int] | None = None
    phase_convention: str = "sum_Rp exp(+i2pi_q_dot_Rp) dJ(R,Rp) u(q)"
    unit: str = "meV"

    def __post_init__(self) -> None:
        q_points = _readonly(self.q_points_frac, dtype=np.dtype(np.float64))
        energies = _readonly(self.phonon_energy_mev, dtype=np.dtype(np.float64))
        coupling = _readonly(self.lambda_mev, dtype=np.dtype(np.complex128))
        targets = _readonly(self.target_atom_indices, dtype=np.dtype(np.int64))
        regularized = _readonly(self.frequency_regularized, dtype=np.dtype(bool))
        if q_points.ndim != 2 or q_points.shape[1:] != (3,):
            raise ValueError("q_points_frac must have shape (nq,3)")
        if energies.ndim != 2 or energies.shape[0] != q_points.shape[0]:
            raise ValueError("phonon_energy_mev must have shape (nq,nmode)")
        if coupling.ndim != 3 or coupling.shape[:2] != energies.shape:
            raise ValueError("lambda_mev must have shape (nq,nmode,nbond)")
        if targets.ndim != 1 or targets.size == 0 or np.any(targets < 0):
            raise ValueError("target_atom_indices must be nonempty and non-negative")
        if regularized.shape != energies.shape:
            raise ValueError("frequency_regularized must match phonon_energy_mev")
        if not np.all(np.isfinite(q_points)) or not np.all(np.isfinite(energies)):
            raise ValueError("mode-resolved derivative contains non-finite metadata")
        if not np.all(np.isfinite(coupling)):
            raise ValueError("lambda_mev contains non-finite values")
        if np.any(energies <= 0.0):
            raise ValueError("mode-resolved phonon energies must be strictly positive")
        if not isinstance(self.target_coverage_complete, (bool, np.bool_)):
            raise TypeError("target_coverage_complete must be boolean")
        q_mesh_shape: tuple[int, int, int] | None
        if self.q_mesh_shape is None:
            q_mesh_shape = None
        else:
            raw_mesh = np.asarray(self.q_mesh_shape)
            numeric_mesh = np.asarray(raw_mesh, dtype=np.float64)
            if (
                numeric_mesh.shape != (3,)
                or not np.all(np.isfinite(numeric_mesh))
                or not np.array_equal(numeric_mesh, np.rint(numeric_mesh))
                or np.any(numeric_mesh <= 0.0)
            ):
                raise ValueError("q_mesh_shape must contain three positive integers")
            q_mesh_shape = (
                int(numeric_mesh[0]),
                int(numeric_mesh[1]),
                int(numeric_mesh[2]),
            )
            if int(np.prod(q_mesh_shape)) != q_points.shape[0]:
                raise ValueError("q_mesh_shape product must equal the q-point count")
            _validate_uniform_q_grid(q_points, q_mesh_shape, 1.0e-8)
        if self.unit != "meV":
            raise ValueError("mode-resolved derivative unit must be meV")
        object.__setattr__(self, "q_points_frac", q_points)
        object.__setattr__(self, "phonon_energy_mev", energies)
        object.__setattr__(self, "lambda_mev", coupling)
        object.__setattr__(self, "target_atom_indices", targets)
        object.__setattr__(self, "frequency_regularized", regularized)
        object.__setattr__(self, "q_mesh_shape", q_mesh_shape)
        object.__setattr__(
            self, "target_coverage_complete", bool(self.target_coverage_complete)
        )

    @property
    def nq(self) -> int:
        return int(self.lambda_mev.shape[0])

    @property
    def nmode(self) -> int:
        return int(self.lambda_mev.shape[1])

    @property
    def n_bonds(self) -> int:
        return int(self.lambda_mev.shape[2])


def _validate_uniform_q_grid(
    q_points: NDArray[np.float64], mesh: tuple[int, int, int], tolerance: float
) -> None:
    scaled = q_points * np.asarray(mesh, dtype=np.float64)[None, :]
    nearest = np.rint(scaled)
    residual = np.abs(scaled - nearest)
    if float(np.max(residual, initial=0.0)) > tolerance:
        raise ValueError(
            "phonon q points are not commensurate with the derivative q mesh"
        )
    keys = np.mod(nearest.astype(np.int64), np.asarray(mesh, dtype=np.int64))
    if np.unique(keys, axis=0).shape[0] != int(np.prod(mesh)):
        raise ValueError("phonon q points do not form one complete derivative q mesh")


def _prepare_isotropic_contraction(
    exchange: ExchangeModel,
    derivative: ExchangeDerivativeModel,
    phonons: PhononCache,
    zero_point: ZeroPointDisplacement,
    *,
    require_complete_targets: bool,
    q_tolerance: float,
    q_chunk_size: int | None,
    bond_chunk_size: int | None,
) -> tuple[NDArray[np.int64], bool, int, int]:
    """Validate the common contraction contract and resolve chunk sizes."""

    if derivative.static_exchange_source != exchange.source:
        raise ValueError(
            "derivative/static source mismatch; load the derivative against the "
            "same ExchangeModel used for this calculation"
        )
    if derivative.representation is not ExchangeRepresentation.ISOTROPIC:
        raise NotImplementedError(
            "the first native self-energy route accepts isotropic dJ/du only"
        )
    if derivative.n_bonds != exchange.n_bonds:
        raise ValueError("static and derivative bond counts differ")
    source_bonds = derivative.source_static_bond_indices
    if source_bonds is None:  # pragma: no cover - canonicalized by the model
        raise AssertionError("derivative source bond indices were not canonicalized")
    source_mates = exchange.mirror_index[source_bonds]
    if not np.array_equal(np.sort(source_mates), np.sort(source_bonds)):
        raise ValueError(
            "derivative bond subset must be closed under directed-bond mates"
        )
    if phonons.q_mesh_shape != derivative.q_mesh_shape:
        raise ValueError(
            f"phonon q mesh {phonons.q_mesh_shape} != derivative q mesh "
            f"{derivative.q_mesh_shape}"
        )
    tolerance = float(q_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("q_tolerance must be finite and non-negative")
    _validate_uniform_q_grid(phonons.q_points_frac, derivative.q_mesh_shape, tolerance)
    expected_displacement_shape = (
        phonons.nq,
        phonons.nmode,
        phonons.nat,
        3,
    )
    if zero_point.values_ang.shape != expected_displacement_shape:
        raise ValueError(
            f"zero-point displacement shape {zero_point.values_ang.shape} != "
            f"{expected_displacement_shape}"
        )
    if not np.array_equal(zero_point.input_frequencies_mev, phonons.frequencies_mev):
        raise ValueError(
            "zero-point displacements were not built from this phonon cache"
        )
    targets = derivative.target_atom_indices
    if int(np.max(targets)) >= phonons.nat:
        raise ValueError("exchange-derivative target exceeds the phonon atom count")
    complete_targets = bool(
        targets.size == phonons.nat
        and np.array_equal(np.sort(targets), np.arange(phonons.nat))
    )
    if require_complete_targets and not complete_targets:
        raise ValueError(
            "native lifetime calculation requires dJ/du for every phonon atom; "
            f"targets={np.sort(targets).tolist()}, nat={phonons.nat}"
        )

    q_chunk = phonons.nq if q_chunk_size is None else int(q_chunk_size)
    bond_chunk = exchange.n_bonds if bond_chunk_size is None else int(bond_chunk_size)
    if q_chunk < 1 or bond_chunk < 1:
        raise ValueError("q_chunk_size and bond_chunk_size must be positive")
    return targets, complete_targets, q_chunk, bond_chunk


def _contract_isotropic_q_indices(
    exchange: ExchangeModel,
    derivative: ExchangeDerivativeModel,
    phonons: PhononCache,
    zero_point: ZeroPointDisplacement,
    targets: NDArray[np.int64],
    q_indices: NDArray[np.int64],
    *,
    q_chunk_size: int,
    bond_chunk_size: int,
) -> NDArray[np.complex128]:
    """Contract one rank-local list of q indices with vectorized inner axes."""

    output = np.empty(
        (q_indices.size, phonons.nmode, exchange.n_bonds), dtype=np.complex128
    )
    if q_indices.size == 0:
        return output
    derivative_values = derivative.isotropic_mev_per_ang
    rp = derivative.rp_cell_shifts.astype(np.float64, copy=False)
    for local_start in range(0, q_indices.size, q_chunk_size):
        local_stop = min(local_start + q_chunk_size, q_indices.size)
        selected = q_indices[local_start:local_stop]
        q_points = phonons.q_points_frac[selected]
        phase = np.exp(2.0j * np.pi * (q_points @ rp.T))
        displacement = zero_point.values_ang[selected][:, :, targets, :]
        for bond_start in range(0, exchange.n_bonds, bond_chunk_size):
            bond_stop = min(bond_start + bond_chunk_size, exchange.n_bonds)
            output[local_start:local_stop, :, bond_start:bond_stop] = np.einsum(
                "qmta,tbra,qr->qmb",
                displacement,
                derivative_values[:, bond_start:bond_stop],
                phase,
                optimize=True,
            )
    return output


def _mode_resolved_result(
    derivative: ExchangeDerivativeModel,
    phonons: PhononCache,
    zero_point: ZeroPointDisplacement,
    targets: NDArray[np.int64],
    complete_targets: bool,
    values: NDArray[np.complex128],
) -> ModeResolvedExchangeDerivative:
    return ModeResolvedExchangeDerivative(
        q_points_frac=phonons.q_points_frac,
        phonon_energy_mev=zero_point.effective_frequencies_mev,
        lambda_mev=values,
        target_atom_indices=targets,
        target_coverage_complete=complete_targets,
        frequency_regularized=zero_point.regularized,
        q_mesh_shape=derivative.q_mesh_shape,
    )


def build_mode_resolved_isotropic_derivative(
    exchange: ExchangeModel,
    derivative: ExchangeDerivativeModel,
    phonons: PhononCache,
    zero_point: ZeroPointDisplacement,
    *,
    require_complete_targets: bool = True,
    q_tolerance: float = 1.0e-8,
    q_chunk_size: int | None = None,
    bond_chunk_size: int | None = None,
) -> ModeResolvedExchangeDerivative:
    """Contract scalar ``dJ/du`` with physical zero-point displacements.

    q and bond chunks bound temporary phase/contraction arrays.  The target,
    displacement, and Rp axes are contracted by optimized NumPy ``einsum``.
    """

    targets, complete_targets, q_chunk, bond_chunk = _prepare_isotropic_contraction(
        exchange,
        derivative,
        phonons,
        zero_point,
        require_complete_targets=require_complete_targets,
        q_tolerance=q_tolerance,
        q_chunk_size=q_chunk_size,
        bond_chunk_size=bond_chunk_size,
    )
    q_indices = np.arange(phonons.nq, dtype=np.int64)
    output = _contract_isotropic_q_indices(
        exchange,
        derivative,
        phonons,
        zero_point,
        targets,
        q_indices,
        q_chunk_size=q_chunk,
        bond_chunk_size=bond_chunk,
    )
    return _mode_resolved_result(
        derivative,
        phonons,
        zero_point,
        targets,
        complete_targets,
        output,
    )


def build_mode_resolved_isotropic_derivative_distributed(
    exchange: ExchangeModel,
    derivative: ExchangeDerivativeModel,
    phonons: PhononCache,
    zero_point: ZeroPointDisplacement,
    *,
    require_complete_targets: bool = True,
    q_tolerance: float = 1.0e-8,
    q_chunk_size: int | None = None,
    bond_chunk_size: int | None = None,
    context: MPIContext | None = None,
    root: int = 0,
) -> ModeResolvedExchangeDerivative:
    """Build and broadcast the coupling cache with MPI-distributed q ownership."""

    targets, complete_targets, q_chunk, bond_chunk = _prepare_isotropic_contraction(
        exchange,
        derivative,
        phonons,
        zero_point,
        require_complete_targets=require_complete_targets,
        q_tolerance=q_tolerance,
        q_chunk_size=q_chunk_size,
        bond_chunk_size=bond_chunk_size,
    )

    def worker(indices: NDArray[np.int64]) -> NDArray[np.complex128]:
        return _contract_isotropic_q_indices(
            exchange,
            derivative,
            phonons,
            zero_point,
            targets,
            indices,
            q_chunk_size=q_chunk,
            bond_chunk_size=bond_chunk,
        )

    distributed = distributed_array_map(
        phonons.nq,
        worker,
        item_shape=(phonons.nmode, exchange.n_bonds),
        dtype=np.complex128,
        context=context,
        root=root,
        broadcast_result=True,
        stage="magph_mode_coupling",
    )
    if distributed.global_values is None:  # pragma: no cover - broadcast contract
        raise RuntimeError("mode-resolved coupling cache was not broadcast")
    return _mode_resolved_result(
        derivative,
        phonons,
        zero_point,
        targets,
        complete_targets,
        distributed.global_values,
    )


__all__ = [
    "ModeResolvedExchangeDerivative",
    "build_mode_resolved_isotropic_derivative",
    "build_mode_resolved_isotropic_derivative_distributed",
]
