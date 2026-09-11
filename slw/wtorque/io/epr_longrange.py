"""Three-dimensional electronic polar long-range terms for EPR data.

qe2pert's electronic polar term is a finite reciprocal-space sum.  The
source form is diagonal in the Wannier basis and returns one value per
Cartesian perturbation.  The localized point-Wannier form retains exactly
that sum and multiplies each ``q + G`` term by
``exp(+2 pi i (q + G).wc_i)``.

Values in this module are in native Perturbo Ry/bohr units.  q is never
folded: the caller supplies the same reduced representative used by the
stored short-range coefficients.  The :class:`PolarLongRange3D` class also
keeps the source q-grid split consistent by subtracting the interpolated
center-minus-identity coarse values.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from slw.core.qe2pert_ws import init_rvec_images, set_wigner_seitz_cell


_GMAX = 14.0
_GAMMA_TOLERANCE = 1.0e-8
_NORM_FLOOR = 1.0e-14
_ALPHA_FLOOR = 1.0e-12
_GRID_TOLERANCE = 1.0e-10


def _finite_array(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape {shape}; got {array.shape}")
    return array


def _qpoint(value: object) -> np.ndarray:
    return _finite_array(value, (3,), "qpoint")


def _points(value: object, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or array.shape[0] < 1:
        raise ValueError(f"{name} must have shape (n, 3); got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _mesh(value: object, name: str) -> tuple[int, int, int]:
    array = np.asarray(value)
    if array.shape != (3,) or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must contain three integers; got {value!r}")
    result = tuple(int(item) for item in array)
    if any(item <= 0 for item in result):
        raise ValueError(f"{name} must contain positive integers; got {result}")
    return result


def _metadata_mapping(metadata: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        raise TypeError("polar metadata must be a mapping")
    return dict(metadata)


@dataclass(frozen=True)
class _PolarData:
    nat: int
    qc_dim: tuple[int, int, int]
    alat: float
    volume: float
    bg: np.ndarray
    epsil: np.ndarray
    zstar: np.ndarray
    tau_cart: np.ndarray
    alpha: float


def _validate_polar_metadata(
    metadata: Mapping[str, Any],
    *,
    at: object | None = None,
    tau: object | None = None,
) -> _PolarData:
    """Validate and normalize the native polar metadata contract."""

    raw = _metadata_mapping(metadata)
    try:
        nat_value = np.asarray(raw["nat"])
        if nat_value.shape != () or not np.isfinite(nat_value) or not float(nat_value).is_integer():
            raise ValueError("nat must be one finite positive integer")
        nat = int(nat_value)
        qc_dim = _mesh(raw["qc_dim"], "qc_dim")
        alat = float(raw["alat"])
        volume = float(raw["volume"])
        bg = _finite_array(raw["bg"], (3, 3), "bg")
        epsil = _finite_array(raw["epsil"], (3, 3), "epsil")
        zstar = np.asarray(raw["zstar"], dtype=np.float64)
    except KeyError as error:
        raise KeyError(f"polar metadata is missing {error.args[0]!r}") from error
    if nat <= 0:
        raise ValueError(f"nat must be positive; got {nat}")
    if not np.isfinite(alat) or alat <= 0.0:
        raise ValueError("alat must be finite and positive")
    if not np.isfinite(volume) or volume <= 0.0:
        raise ValueError("volume must be finite and positive")
    if zstar.shape != (nat, 3, 3) or not np.all(np.isfinite(zstar)):
        raise ValueError(f"zstar must be finite with shape ({nat}, 3, 3)")

    tau_cart_value = raw.get("tau_cart")
    if tau_cart_value is None:
        tau_value = tau if tau is not None else raw.get("tau")
        at_value = at if at is not None else raw.get("at")
        if tau_value is None or at_value is None:
            raise KeyError("polar metadata requires tau_cart or reduced tau together with at")
        tau_reduced = _finite_array(tau_value, (nat, 3), "tau")
        at_array = _finite_array(at_value, (3, 3), "at")
        tau_cart_value = tau_reduced @ at_array.T
    tau_cart = _finite_array(tau_cart_value, (nat, 3), "tau_cart")

    alpha = float(raw.get("polar_alpha", 1.0))
    if not np.isfinite(alpha) or alpha < _ALPHA_FLOOR:
        raise ValueError(f"polar_alpha must be finite and at least {_ALPHA_FLOOR:g}")

    # Preserve the source metric and tensor conventions.  In particular, do
    # not symmetrize Born tensors or dielectric data at this API boundary.
    metric = np.einsum("ig,ij,jg->g", bg, epsil, bg, optimize=True)
    if np.any(metric <= 0.0) or not np.all(np.isfinite(metric)):
        raise ValueError("dielectric reciprocal metric must be finite and positive")

    for name in ("system_2d", "thickness_2d"):
        if name in raw and not np.all(np.isfinite(np.asarray(raw[name], dtype=float))):
            raise ValueError(f"{name} must be finite")
    if bool(raw.get("system_2d", False)) or float(raw.get("thickness_2d", -1.0)) > 0.0:
        raise NotImplementedError("electronic long-range add-back supports 3D dipoles only")
    if bool(raw.get("lquad", False)):
        raise NotImplementedError("electronic quadrupole add-back is not implemented")

    return _PolarData(
        nat=nat,
        qc_dim=qc_dim,
        alat=alat,
        volume=volume,
        bg=bg,
        epsil=epsil,
        zstar=zstar,
        tau_cart=tau_cart,
        alpha=alpha,
    )


@dataclass(frozen=True)
class _PolarTerms:
    weights: np.ndarray
    phase: np.ndarray
    charge: np.ndarray
    qg_reduced: np.ndarray
    prefactor: complex


def _polar_terms(data: _PolarData, qpoint: object) -> _PolarTerms | None:
    """Build the exact finite q+G table used by both LR forms."""

    q = _qpoint(qpoint)
    if np.linalg.norm(q) < _GAMMA_TOLERANCE:
        return None

    maximum = _GMAX * 4.0 * data.alpha
    metric = np.einsum("ig,ij,jg->g", data.bg, data.epsil, data.bg, optimize=True)
    extents = np.ceil(np.sqrt(maximum / metric)).astype(int)
    extents[np.asarray(data.qc_dim, dtype=np.int64) < 2] = 0
    shifts = np.asarray(
        [
            (i, j, k)
            for i in range(-int(extents[0]), int(extents[0]) + 1)
            for j in range(-int(extents[1]), int(extents[1]) + 1)
            for k in range(-int(extents[2]), int(extents[2]) + 1)
        ],
        dtype=np.float64,
    )
    qg_reduced = q[None, :] + shifts
    cart = qg_reduced @ data.bg.T
    dielectric_norm = np.einsum(
        "gi,ij,gj->g", cart, data.epsil, cart, optimize=True
    )
    keep = (dielectric_norm >= _NORM_FLOOR) & (dielectric_norm <= maximum)
    qg_reduced = qg_reduced[keep]
    cart = cart[keep]
    dielectric_norm = dielectric_norm[keep]
    if qg_reduced.shape[0] == 0:
        return _PolarTerms(
            weights=np.empty(0, dtype=np.float64),
            phase=np.empty((0, data.nat), dtype=np.complex128),
            charge=np.empty((0, data.nat, 3), dtype=np.float64),
            qg_reduced=qg_reduced,
            prefactor=8j * np.pi / (data.volume * (2.0 * np.pi / data.alat)),
        )

    weights = np.exp(-dielectric_norm / (4.0 * data.alpha)) / dielectric_norm
    phase = np.exp(-2j * np.pi * (cart @ data.tau_cart.T))
    charge = np.einsum("gi,aij->gaj", cart, data.zstar, optimize=True)
    return _PolarTerms(
        weights=weights,
        phase=phase,
        charge=charge,
        qg_reduced=qg_reduced,
        prefactor=8j * np.pi / (data.volume * (2.0 * np.pi / data.alat)),
    )


def _source_and_center(
    data: _PolarData,
    qpoint: object,
    centers: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    terms = _polar_terms(data, qpoint)
    source = np.zeros((data.nat, 3), dtype=np.complex128)
    if terms is None:
        centered = None
        if centers is not None:
            centered = np.zeros((data.nat * 3, len(centers)), dtype=np.complex128)
        return source.ravel(), centered

    source = np.einsum(
        "g,ga,gaj->aj", terms.weights, terms.phase, terms.charge, optimize=True
    ) * terms.prefactor
    if centers is None:
        return source.ravel(), None
    center_phase = np.exp(2j * np.pi * (terms.qg_reduced @ centers.T))
    centered = np.einsum(
        "g,ga,gaj,gi->aji",
        terms.weights,
        terms.phase,
        terms.charge,
        center_phase,
        optimize=True,
    ) * terms.prefactor
    return source.ravel(), centered.reshape(data.nat * 3, len(centers))


def electronic_longrange_3d(
    metadata: Mapping[str, Any], qpoint: object
) -> np.ndarray:
    """Return source identity LR in native Ry/bohr, shape ``(3*nat,)``."""

    data = _validate_polar_metadata(metadata)
    source, _ = _source_and_center(data, qpoint, None)
    return source


def electronic_longrange_3d_center(
    metadata: Mapping[str, Any], qpoint: object, wannier_centers: object
) -> np.ndarray:
    """Return term-wise point-center LR in native Ry/bohr."""

    centers = _points(wannier_centers, "wannier_centers")
    data = _validate_polar_metadata(metadata)
    _, centered = _source_and_center(data, qpoint, centers)
    assert centered is not None
    return centered


def _ws_cell_parts(cell: object, key: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    if not hasattr(cell, "vectors") or not hasattr(cell, "ndeg"):
        raise TypeError(f"WS cell {key} must provide vectors and ndeg")
    vectors = np.asarray(getattr(cell, "vectors"), dtype=np.float64)
    ndeg = np.asarray(getattr(cell, "ndeg"), dtype=np.float64).reshape(-1)
    if (
        vectors.ndim != 2
        or vectors.shape[1] != 3
        or vectors.shape[0] != ndeg.size
        or vectors.shape[0] < 1
        or not np.all(np.isfinite(vectors))
        or not np.all(np.isfinite(ndeg))
        or not np.allclose(vectors, np.rint(vectors), atol=0.0, rtol=0.0)
        or not np.allclose(ndeg, np.rint(ndeg), atol=0.0, rtol=0.0)
        or np.any(ndeg <= 0.0)
    ):
        raise ValueError(f"WS cell {key} has invalid vectors/ndeg")
    return vectors, ndeg


def _normalize_ws_cells(
    cells: Mapping[tuple[int, int], object], nat: int, nwan: int
) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    if not isinstance(cells, Mapping):
        raise TypeError("ws_cells must be a mapping keyed by (atom, wannier)")
    result: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for atom in range(nat):
        for wannier in range(nwan):
            key = (atom, wannier)
            if key not in cells:
                raise ValueError(f"missing q interpolation WS cell {key}")
            result[key] = _ws_cell_parts(cells[key], key)
    return result


def _validate_coarse_grid(qpoints: np.ndarray, qmesh: tuple[int, int, int]) -> None:
    """Require one complete commensurate q mesh while retaining representatives."""

    expected_count = int(np.prod(qmesh))
    if len(qpoints) != expected_count:
        raise ValueError(
            f"coarse_qpoints must contain {expected_count} points for qmesh={qmesh}; "
            f"got {len(qpoints)}"
        )
    mesh = np.asarray(qmesh, dtype=np.int64)
    residues = np.mod(qpoints, 1.0)
    # Classify each coordinate by its nearest mesh index.  Comparing sorted
    # floating residues is unstable for centered representatives such as
    # ``-1/3 + 1e-16`` and can reject an otherwise complete source grid.
    scaled = residues * mesh[None, :]
    nearest = np.rint(scaled)
    if np.any(np.abs(scaled - nearest) > _GRID_TOLERANCE):
        raise ValueError("coarse_qpoints must be one complete qmesh modulo reciprocal integers")
    indices = np.mod(nearest.astype(np.int64), mesh[None, :])
    linear = (indices[:, 0] * mesh[1] + indices[:, 1]) * mesh[2] + indices[:, 2]
    if np.unique(linear).size != expected_count or not np.array_equal(
        np.sort(linear), np.arange(expected_count, dtype=np.int64)
    ):
        raise ValueError("coarse_qpoints must be one complete qmesh modulo reciprocal integers")


class PolarLongRange3D:
    """Source/point-center LR and source-SR re-splitting on one q mesh.

    ``coarse_qpoints`` are required for :meth:`sr_correction`; they must be a
    complete set of reduced q representatives for ``qmesh``.  Their exact
    representatives are retained in the interpolation and in every LR sum.
    When ``ws_cells`` is omitted, pair-specific q WS cells are built from
    ``qmesh``, reduced lattice ``at``, and reduced atomic positions ``tau``.
    """

    def __init__(
        self,
        polar_metadata: Mapping[str, Any],
        wannier_centers: object,
        *,
        coarse_qpoints: object | None = None,
        qmesh: object | None = None,
        at: object | None = None,
        tau: object | None = None,
        ws_cells: Mapping[tuple[int, int], object] | None = None,
    ) -> None:
        raw = _metadata_mapping(polar_metadata)
        self.wannier_centers = _points(wannier_centers, "wannier_centers")
        self._data = _validate_polar_metadata(raw, at=at, tau=tau)
        self.nat = self._data.nat
        self.nwan = len(self.wannier_centers)
        self.qmesh = _mesh(self._data.qc_dim if qmesh is None else qmesh, "qmesh")

        self.coarse_qpoints = None
        if coarse_qpoints is not None:
            selected = _points(coarse_qpoints, "coarse_qpoints")
            _validate_coarse_grid(selected, self.qmesh)
            self.coarse_qpoints = selected.copy()

        if ws_cells is None and self.coarse_qpoints is not None:
            at_value = at if at is not None else raw.get("at")
            tau_value = tau if tau is not None else raw.get("tau")
            if at_value is None or tau_value is None:
                raise KeyError("building q interpolation cells requires reduced at and tau")
            at_array = _finite_array(at_value, (3, 3), "at")
            if abs(float(np.linalg.det(at_array))) <= np.finfo(float).eps:
                raise ValueError("at must be nonsingular for q interpolation")
            tau_array = _finite_array(tau_value, (self.nat, 3), "tau")
            images = init_rvec_images(self.qmesh, at_array)
            built: dict[tuple[int, int], object] = {}
            for atom in range(self.nat):
                for wannier in range(self.nwan):
                    built[(atom, wannier)] = set_wigner_seitz_cell(
                        images,
                        at_array,
                        self.wannier_centers[wannier],
                        tau_array[atom],
                    )
            ws_cells = built
        self.ws_cells = None if ws_cells is None else _normalize_ws_cells(
            ws_cells, self.nat, self.nwan
        )

        self._coarse_difference = None
        if self.coarse_qpoints is not None:
            difference = np.empty(
                (len(self.coarse_qpoints), 3 * self.nat, self.nwan),
                dtype=np.complex128,
            )
            for index, qpoint in enumerate(self.coarse_qpoints):
                source, centered = _source_and_center(
                    self._data, qpoint, self.wannier_centers
                )
                assert centered is not None
                difference[index] = centered - source[:, None]
            self._coarse_difference = difference

    def source(self, qpoint: object) -> np.ndarray:
        """Return the source identity LR, shape ``(3*nat,)`` in Ry/bohr."""

        source, _ = _source_and_center(self._data, qpoint, None)
        return source

    def center(self, qpoint: object) -> np.ndarray:
        """Return point-center LR, shape ``(3*nat,nwan)`` in Ry/bohr."""

        _, centered = _source_and_center(self._data, qpoint, self.wannier_centers)
        assert centered is not None
        return centered

    def sr_correction(self, qpoint: object) -> np.ndarray:
        """Return ``-I_q[L_center(qc)-L_identity(qc)]`` for source SR data."""

        if self._coarse_difference is None or self.coarse_qpoints is None:
            raise ValueError("coarse_qpoints are required for sr_correction")
        if self.ws_cells is None:
            raise ValueError("q interpolation WS cells are not configured")
        q = _qpoint(qpoint)
        result = np.empty((3 * self.nat, self.nwan), dtype=np.complex128)
        for atom in range(self.nat):
            for wannier in range(self.nwan):
                vectors, ndeg = self.ws_cells[atom, wannier]
                weights = (
                    np.exp(
                        2j
                        * np.pi
                        * ((q[None, :] - self.coarse_qpoints) @ vectors.T)
                    )
                    / ndeg[None, :]
                ).sum(axis=1) / len(self.coarse_qpoints)
                result[3 * atom : 3 * atom + 3, wannier] = -(
                    weights
                    @ self._coarse_difference[:, 3 * atom : 3 * atom + 3, wannier]
                )
        return result


__all__ = [
    "PolarLongRange3D",
    "electronic_longrange_3d",
    "electronic_longrange_3d_center",
]
