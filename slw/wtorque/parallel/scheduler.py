"""Symmetry-paired q scheduling for serial, job-array, and MPI execution."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from slw.wtorque.errors import MeshCommensurabilityError


@dataclass(frozen=True)
class QPointPair:
    positive: int
    negative: int

    @property
    def self_inverse(self) -> bool:
        return self.positive == self.negative


def build_q_pair_schedule(
    qpoints: object,
    *,
    tolerance: float = 1.0e-9,
) -> tuple[QPointPair, ...]:
    mesh = np.asarray(qpoints, dtype=np.float64)
    if mesh.ndim != 2 or mesh.shape[1] != 3:
        raise ValueError("qpoints must have shape (nq,3)")
    delta = mesh[:, None, :] + mesh[None, :, :]
    residual = delta - np.rint(delta)
    distances = np.max(np.abs(residual), axis=-1)
    partners = np.argmin(distances, axis=1)
    best = distances[np.arange(mesh.shape[0]), partners]
    if np.any(best > tolerance):
        index = int(np.argmax(best))
        raise MeshCommensurabilityError(
            f"q-point {index} has no commensurate -q partner; residual={best[index]:.3e}"
        )
    visited: set[int] = set()
    pairs: list[QPointPair] = []
    for index, partner_raw in enumerate(partners):
        if index in visited:
            continue
        partner = int(partner_raw)
        if int(partners[partner]) != index:
            raise MeshCommensurabilityError("-q mapping is not involutive")
        pairs.append(QPointPair(index, partner))
        visited.add(index)
        visited.add(partner)
    return tuple(pairs)


def partition_pairs(
    pairs: tuple[QPointPair, ...] | list[QPointPair],
    rank: int,
    size: int,
) -> tuple[QPointPair, ...]:
    if size < 1 or rank < 0 or rank >= size:
        raise ValueError("rank and size are inconsistent")
    quotient, remainder = divmod(len(pairs), size)
    start = rank * quotient + min(rank, remainder)
    stop = start + quotient + (1 if rank < remainder else 0)
    return tuple(pairs[start:stop])


__all__ = ["QPointPair", "build_q_pair_schedule", "partition_pairs"]

