"""Retarded q-pair completion for a complex physical Fourier kernel."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def finalize_q_pair(a_q: object, a_minus_q: object) -> NDArray[np.complex128]:
    """Apply WT-K03; neither raw loop is replaced by a general-q ``imag``."""

    aq = np.asarray(a_q, dtype=np.complex128)
    amq = np.asarray(a_minus_q, dtype=np.complex128)
    if aq.shape != amq.shape:
        raise ValueError("q and -q retarded loops must have matching shapes")
    return np.asarray(-(aq - amq.conj()) / (2j * np.pi), dtype=np.complex128)


def q_conjugation_residual(k_q: object, k_minus_q: object) -> tuple[float, float]:
    q = np.asarray(k_q, dtype=np.complex128)
    minus = np.asarray(k_minus_q, dtype=np.complex128)
    absolute = float(np.max(np.abs(minus - q.conj()), initial=0.0))
    scale = float(np.max(np.abs(q), initial=0.0))
    scale = max(scale, float(np.finfo(float).tiny))
    return absolute, absolute / scale


__all__ = ["finalize_q_pair", "q_conjugation_residual"]
