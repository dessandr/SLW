"""Deterministic q-pair scheduling and optional MPI ownership."""

from .scheduler import build_q_pair_schedule, partition_pairs

__all__ = ["build_q_pair_schedule", "partition_pairs"]

