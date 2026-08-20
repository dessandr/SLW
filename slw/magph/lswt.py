"""Native collinear linear-spin-wave theory for screened exchange models.

The implementation uses the canonical directed, mate-complete Hamiltonian

``H = -0.5 sum_(l,b) S_(i,l)^T K_b S_(j,l+R_b)``.

For unit-vector LKAG input, ``K_b = J_b/(S_i S_j)``.  Spin-operator input is
already ``K_b``.  All Fourier phases are cell-gauge phases
``exp(+i 2 pi k dot R_b)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .model import (
    ExchangeModel,
    ExchangeRepresentation,
    ExchangeSpinNormalization,
    MagneticConfiguration,
    MagneticOrder,
)


def _readonly(value: object, *, dtype: np.dtype) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class MagnonSpectrum:
    """Signed magnon channels and their band transformation on a k mesh."""

    k_points_frac: NDArray[np.float64]
    signed_energies_mev: NDArray[np.float64]
    transformation: NDArray[np.complex128]
    metric: NDArray[np.float64]
    physical_mode_count: int
    max_hermiticity_residual_mev: float
    max_eigen_residual_mev: float
    max_paraunitary_residual: float
    minimum_metric_energy_mev: float
    fourier_phase_convention: str = "exp(+i2pi_k_dot_R)"

    def __post_init__(self) -> None:
        k_points = _readonly(self.k_points_frac, dtype=np.dtype(np.float64))
        energies = _readonly(self.signed_energies_mev, dtype=np.dtype(np.float64))
        transform = _readonly(self.transformation, dtype=np.dtype(np.complex128))
        metric = _readonly(self.metric, dtype=np.dtype(np.float64))
        if k_points.ndim != 2 or k_points.shape[1:] != (3,) or k_points.shape[0] == 0:
            raise ValueError("k_points_frac must have nonempty shape (nk,3)")
        if energies.ndim != 2 or energies.shape[0] != k_points.shape[0]:
            raise ValueError("signed_energies_mev must have shape (nk,nchannel)")
        channels = energies.shape[1]
        if transform.shape != (k_points.shape[0], channels, channels):
            raise ValueError("transformation must have shape (nk,nchannel,nchannel)")
        if metric.shape != (channels,) or not np.all(np.isin(metric, (-1.0, 1.0))):
            raise ValueError("metric must contain one +/-1 entry per channel")
        physical = int(self.physical_mode_count)
        if physical < 1 or physical > channels:
            raise ValueError("physical_mode_count is outside the channel range")
        if not np.all(metric[:physical] == 1.0):
            raise ValueError("physical channels must carry positive bosonic metric")
        if not np.all(np.isfinite(k_points)) or not np.all(np.isfinite(energies)):
            raise ValueError("magnon spectrum contains non-finite real data")
        if not np.all(np.isfinite(transform)):
            raise ValueError("magnon transformation contains non-finite values")
        diagnostics = (
            self.max_hermiticity_residual_mev,
            self.max_eigen_residual_mev,
            self.max_paraunitary_residual,
        )
        if any(not np.isfinite(value) or value < 0.0 for value in diagnostics):
            raise ValueError(
                "magnon diagnostic residuals must be finite and non-negative"
            )
        minimum = float(self.minimum_metric_energy_mev)
        if not np.isfinite(minimum):
            raise ValueError("minimum_metric_energy_mev must be finite")
        object.__setattr__(self, "k_points_frac", k_points)
        object.__setattr__(self, "signed_energies_mev", energies)
        object.__setattr__(self, "transformation", transform)
        object.__setattr__(self, "metric", metric)
        object.__setattr__(self, "physical_mode_count", physical)
        object.__setattr__(self, "minimum_metric_energy_mev", minimum)

    @property
    def physical_energies_mev(self) -> NDArray[np.float64]:
        values = self.signed_energies_mev[:, : self.physical_mode_count]
        values.setflags(write=False)
        return values


def uniform_fractional_mesh(
    mesh: tuple[int, int, int] | ArrayLike,
    *,
    shift: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Build a vectorized fractional mesh with an explicit grid-unit shift."""

    raw = np.asarray(mesh)
    if raw.shape != (3,) or np.iscomplexobj(raw):
        raise ValueError("mesh must contain three positive integers")
    numeric = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(numeric)) or not np.array_equal(
        numeric, np.rint(numeric)
    ):
        raise ValueError("mesh must contain three positive integers")
    dimensions = numeric.astype(np.int64)
    if np.any(dimensions <= 0):
        raise ValueError("mesh must contain three positive integers")
    if shift is None:
        offset = np.zeros(3, dtype=np.float64)
    else:
        offset = np.asarray(shift, dtype=np.float64)
        if offset.shape != (3,) or not np.all(np.isfinite(offset)):
            raise ValueError("shift must be a finite length-3 vector in grid units")
    grid = np.indices(tuple(int(value) for value in dimensions), dtype=np.float64)
    points = np.stack(
        [
            (grid[axis].reshape(-1) + offset[axis]) / float(dimensions[axis])
            for axis in range(3)
        ],
        axis=1,
    )
    points %= 1.0
    points.setflags(write=False)
    return points


def _local_frames(configuration: MagneticConfiguration) -> NDArray[np.float64]:
    axis = configuration.quantization_axis
    cartesian = np.eye(3, dtype=np.float64)
    reference = cartesian[int(np.argmin(np.abs(cartesian @ axis)))]
    first = reference - np.dot(reference, axis) * axis
    first /= np.linalg.norm(first)
    second = np.cross(axis, first)
    frames = np.empty((configuration.n_magnetic_sites, 3, 3), dtype=np.float64)
    frames[:, :, 0] = first[None, :]
    frames[:, :, 1] = configuration.spin_pattern[:, None] * second[None, :]
    frames[:, :, 2] = configuration.spin_directions
    return frames


def _spin_operator_tensor(
    exchange: ExchangeModel, configuration: MagneticConfiguration
) -> NDArray[np.float64]:
    if (
        exchange.convention.spin_normalization
        is ExchangeSpinNormalization.SPIN_OPERATOR
    ):
        return np.asarray(exchange.tensor_mev, dtype=np.float64)
    magnitudes = configuration.spin_magnitudes
    denominator = magnitudes[exchange.bond_i] * magnitudes[exchange.bond_j]
    return exchange.tensor_mev / denominator[:, None, None]


def _quadratic_blocks(
    exchange: ExchangeModel,
    configuration: MagneticConfiguration,
    k_points: NDArray[np.float64],
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    n_site = exchange.n_magnetic_sites
    n_k = k_points.shape[0]
    frames = _local_frames(configuration)
    tensor = _spin_operator_tensor(exchange, configuration)
    local = np.einsum(
        "bai,bac,bcj->bij",
        frames[exchange.bond_i],
        tensor,
        frames[exchange.bond_j],
        optimize=True,
    )
    spin_i = configuration.spin_magnitudes[exchange.bond_i]
    spin_j = configuration.spin_magnitudes[exchange.bond_j]
    hp_i = np.sqrt(spin_i / 2.0)[:, None] * np.asarray(
        (1.0, -1.0j, 0.0), dtype=np.complex128
    )
    hp_j = np.sqrt(spin_j / 2.0)[:, None] * np.asarray(
        (1.0, -1.0j, 0.0), dtype=np.complex128
    )
    weight = exchange.convention.directed_bond_weight
    normal_ij = -weight * np.einsum(
        "bi,bij,bj->b", hp_i.conj(), local, hp_j, optimize=True
    )
    normal_ji = -weight * np.einsum(
        "bi,bij,bj->b", hp_i, local, hp_j.conj(), optimize=True
    )
    pair_create = -weight * np.einsum(
        "bi,bij,bj->b", hp_i.conj(), local, hp_j.conj(), optimize=True
    )
    onsite = np.zeros(n_site, dtype=np.complex128)
    np.add.at(onsite, exchange.bond_i, weight * local[:, 2, 2] * spin_j)
    np.add.at(onsite, exchange.bond_j, weight * local[:, 2, 2] * spin_i)

    phase = np.exp(2.0j * np.pi * (k_points @ exchange.cell_shift.astype(np.float64).T))
    normal = np.zeros((n_k, n_site, n_site), dtype=np.complex128)
    diagonal = np.arange(n_site)
    normal[:, diagonal, diagonal] = onsite[None, :]
    k_index = np.arange(n_k, dtype=np.int64)[:, None]
    np.add.at(
        normal,
        (k_index, exchange.bond_i[None, :], exchange.bond_j[None, :]),
        phase * normal_ij[None, :],
    )
    np.add.at(
        normal,
        (k_index, exchange.bond_j[None, :], exchange.bond_i[None, :]),
        phase.conj() * normal_ji[None, :],
    )
    pairing = np.zeros_like(normal)
    np.add.at(
        pairing,
        (k_index, exchange.bond_i[None, :], exchange.bond_j[None, :]),
        phase * pair_create[None, :],
    )
    # ``pair_create`` is the coefficient C_ij of a_i^dagger a_j^dagger.
    # The BdG block appears with 1/2, so B=C+C^T(-k).  Mate completeness makes
    # those two contributions identical in the canonical isotropic model.
    pairing *= 2.0
    return normal, pairing


def _metric_gram_schmidt(
    candidates: list[tuple[float, NDArray[np.complex128]]],
    metric: NDArray[np.float64],
    sign: float,
    count: int,
    tolerance: float,
) -> tuple[list[float], list[NDArray[np.complex128]]]:
    energies: list[float] = []
    modes: list[NDArray[np.complex128]] = []
    for energy, raw in candidates:
        vector = np.array(raw, dtype=np.complex128, copy=True)
        for previous in modes:
            overlap = np.vdot(previous, metric * vector)
            vector -= previous * (sign * overlap)
        norm = float(np.real(np.vdot(vector, metric * vector)))
        if sign * norm <= tolerance:
            continue
        vector /= np.sqrt(abs(norm))
        energies.append(float(energy))
        modes.append(vector)
        if len(modes) == count:
            break
    return energies, modes


def _solve_bdg_one(
    hamiltonian: NDArray[np.complex128],
    metric: NDArray[np.float64],
    *,
    metric_tolerance: float,
    imaginary_tolerance_mev: float,
) -> tuple[NDArray[np.float64], NDArray[np.complex128], float, float]:
    dynamic = metric[:, None] * hamiltonian
    eigenvalues, eigenvectors = np.linalg.eig(dynamic)
    max_imaginary = float(np.max(np.abs(eigenvalues.imag), initial=0.0))
    if max_imaginary > imaginary_tolerance_mev:
        raise ValueError(
            "bosonic BdG spectrum contains complex modes; "
            f"maximum imaginary part={max_imaginary:.6g} meV"
        )
    real_values = eigenvalues.real
    norms = np.real(
        np.einsum(
            "ci,c,ci->i", eigenvectors.conj(), metric, eigenvectors, optimize=True
        )
    )
    n_positive = int(np.count_nonzero(metric > 0.0))
    positive_candidates = sorted(
        (
            (float(real_values[index]), eigenvectors[:, index])
            for index in range(real_values.size)
            if norms[index] > metric_tolerance
        ),
        key=lambda item: item[0],
    )
    negative_candidates = sorted(
        (
            (float(real_values[index]), eigenvectors[:, index])
            for index in range(real_values.size)
            if norms[index] < -metric_tolerance
        ),
        key=lambda item: item[0],
        reverse=True,
    )
    positive_energy, positive_modes = _metric_gram_schmidt(
        positive_candidates, metric, 1.0, n_positive, metric_tolerance
    )
    negative_energy, negative_modes = _metric_gram_schmidt(
        negative_candidates,
        metric,
        -1.0,
        metric.size - n_positive,
        metric_tolerance,
    )
    if (
        len(positive_modes) != n_positive
        or len(negative_modes) != metric.size - n_positive
    ):
        raise ValueError(
            "bosonic BdG eigenvectors have zero/ill-conditioned metric norm; "
            "an exact Goldstone point may require an explicit shifted mesh"
        )
    energies = np.asarray((*positive_energy, *negative_energy), dtype=np.float64)
    transform = np.column_stack((*positive_modes, *negative_modes))
    eigen_residual = float(
        np.max(
            np.abs(dynamic @ transform - transform * energies[None, :]),
            initial=0.0,
        )
    )
    paraunitary = transform.conj().T @ (metric[:, None] * transform)
    paraunitary_residual = float(
        np.max(np.abs(paraunitary - np.diag(metric)), initial=0.0)
    )
    return energies, transform, eigen_residual, paraunitary_residual


def solve_isotropic_lswt(
    exchange: ExchangeModel,
    configuration: MagneticConfiguration,
    k_points_frac: ArrayLike,
    *,
    stability_tolerance_mev: float = 1.0e-9,
    imaginary_tolerance_mev: float = 1.0e-9,
    hermiticity_tolerance_mev: float = 1.0e-10,
    metric_tolerance: float = 1.0e-10,
    goldstone_tolerance_mev: float = 1.0e-10,
) -> MagnonSpectrum:
    """Solve FM or bipartite-AFM isotropic LSWT on explicit k points."""

    if exchange.representation is not ExchangeRepresentation.ISOTROPIC:
        raise NotImplementedError(
            "native tensor LSWT is not yet enabled; an isotropic projection must be explicit"
        )
    if configuration.n_magnetic_sites != exchange.n_magnetic_sites:
        raise ValueError("magnetic configuration and exchange site counts differ")
    if not configuration.locally_stable:
        raise ValueError("LSWT requires a locally stable magnetic configuration")
    tolerances = {
        "stability_tolerance_mev": float(stability_tolerance_mev),
        "imaginary_tolerance_mev": float(imaginary_tolerance_mev),
        "hermiticity_tolerance_mev": float(hermiticity_tolerance_mev),
        "metric_tolerance": float(metric_tolerance),
        "goldstone_tolerance_mev": float(goldstone_tolerance_mev),
    }
    if any(not np.isfinite(value) or value < 0.0 for value in tolerances.values()):
        raise ValueError("LSWT tolerances must be finite and non-negative")
    k_points = np.asarray(k_points_frac, dtype=np.float64)
    if k_points.ndim != 2 or k_points.shape[1:] != (3,) or k_points.shape[0] == 0:
        raise ValueError("k_points_frac must have nonempty shape (nk,3)")
    if not np.all(np.isfinite(k_points)):
        raise ValueError("k_points_frac contains non-finite values")

    normal, pairing = _quadratic_blocks(exchange, configuration, k_points)
    normal_minus, pairing_minus = _quadratic_blocks(exchange, configuration, -k_points)
    hermiticity = float(
        max(
            np.max(np.abs(normal - np.swapaxes(normal.conj(), 1, 2)), initial=0.0),
            np.max(np.abs(pairing - np.swapaxes(pairing_minus, 1, 2)), initial=0.0),
        )
    )
    if hermiticity > tolerances["hermiticity_tolerance_mev"]:
        raise ValueError(
            "LSWT quadratic blocks violate Hermiticity/Fourier reciprocity; "
            f"maximum residual={hermiticity:.6g} meV"
        )

    n_k = k_points.shape[0]
    n_site = exchange.n_magnetic_sites
    if configuration.order is MagneticOrder.FM:
        if (
            float(np.max(np.abs(pairing), initial=0.0))
            > tolerances["hermiticity_tolerance_mev"]
        ):
            raise ValueError("isotropic FM unexpectedly produced anomalous LSWT terms")
        hermitian = 0.5 * (normal + np.swapaxes(normal.conj(), 1, 2))
        energies, transform = np.linalg.eigh(hermitian)
        minimum = float(np.min(energies))
        if minimum < -tolerances["stability_tolerance_mev"]:
            raise ValueError(f"FM LSWT has an unstable mode at {minimum:.6g} meV")
        energies = np.maximum(energies, 0.0)
        eigen_residual = float(
            np.max(
                np.abs(hermitian @ transform - transform * energies[:, None, :]),
                initial=0.0,
            )
        )
        metric = np.ones(n_site, dtype=np.float64)
        paraunitary_residual = float(
            np.max(
                np.abs(
                    np.swapaxes(transform.conj(), 1, 2) @ transform
                    - np.eye(n_site)[None, :, :]
                ),
                initial=0.0,
            )
        )
        return MagnonSpectrum(
            k_points_frac=k_points,
            signed_energies_mev=energies,
            transformation=transform,
            metric=metric,
            physical_mode_count=n_site,
            max_hermiticity_residual_mev=hermiticity,
            max_eigen_residual_mev=eigen_residual,
            max_paraunitary_residual=paraunitary_residual,
            minimum_metric_energy_mev=float(np.min(energies)),
        )

    if configuration.order is not MagneticOrder.COLLINEAR_AFM or n_site != 2:
        raise NotImplementedError("native AFM LSWT currently supports two sublattices")
    channels = 2 * n_site
    metric = np.concatenate((np.ones(n_site), -np.ones(n_site)))
    signed_energies = np.empty((n_k, channels), dtype=np.float64)
    transformations = np.empty((n_k, channels, channels), dtype=np.complex128)
    max_eigen_residual = 0.0
    max_paraunitary_residual = 0.0
    for index in range(n_k):
        hamiltonian = np.block(
            [
                [normal[index], pairing[index]],
                [pairing_minus[index].conj(), normal_minus[index].conj()],
            ]
        )
        hamiltonian = 0.5 * (hamiltonian + hamiltonian.conj().T)
        minimum_hessian = float(np.min(np.linalg.eigvalsh(hamiltonian)))
        if minimum_hessian < -tolerances["stability_tolerance_mev"]:
            raise ValueError(
                f"AFM LSWT Hessian is unstable at {minimum_hessian:.6g} meV"
            )
        if minimum_hessian <= tolerances["goldstone_tolerance_mev"]:
            raise ValueError(
                "exact AFM Goldstone modes have zero bosonic metric norm; "
                "use an explicit shifted mesh for self-energy/lifetime calculations"
            )
        energies, transform, eigen_residual, paraunitary_residual = _solve_bdg_one(
            hamiltonian,
            metric,
            metric_tolerance=tolerances["metric_tolerance"],
            imaginary_tolerance_mev=tolerances["imaginary_tolerance_mev"],
        )
        signed_energies[index] = energies
        transformations[index] = transform
        max_eigen_residual = max(max_eigen_residual, eigen_residual)
        max_paraunitary_residual = max(max_paraunitary_residual, paraunitary_residual)
    metric_energy = signed_energies * metric[None, :]
    minimum = float(np.min(metric_energy))
    if minimum < -tolerances["stability_tolerance_mev"]:
        raise ValueError(f"AFM LSWT has an unstable metric energy {minimum:.6g} meV")
    return MagnonSpectrum(
        k_points_frac=k_points,
        signed_energies_mev=signed_energies,
        transformation=transformations,
        metric=metric,
        physical_mode_count=n_site,
        max_hermiticity_residual_mev=hermiticity,
        max_eigen_residual_mev=max_eigen_residual,
        max_paraunitary_residual=max_paraunitary_residual,
        minimum_metric_energy_mev=minimum,
    )


__all__ = ["MagnonSpectrum", "solve_isotropic_lswt", "uniform_fractional_mesh"]
