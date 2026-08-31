"""Central finite-difference validation helpers (WT-S02)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def central_first_derivative(
    function: Callable[[float], object],
    step: float,
) -> np.ndarray:
    if step <= 0:
        raise ValueError("finite-difference step must be positive")
    return (np.asarray(function(step)) - np.asarray(function(-step))) / (2.0 * step)


def mixed_central_derivative(
    function: Callable[[float, float], float],
    spin_step: float,
    displacement_step: float,
) -> float:
    if spin_step <= 0 or displacement_step <= 0:
        raise ValueError("mixed finite-difference steps must be positive")
    return float(
        (
            function(spin_step, displacement_step)
            - function(spin_step, -displacement_step)
            - function(-spin_step, displacement_step)
            + function(-spin_step, -displacement_step)
        )
        / (4.0 * spin_step * displacement_step)
    )


__all__ = ["central_first_derivative", "mixed_central_derivative"]

