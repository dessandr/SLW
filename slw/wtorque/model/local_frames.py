"""Validation of right-handed local transverse spin frames."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from slw.wtorque.errors import MagneticSubspaceError


def validate_local_frames(
    value: object,
    *,
    tolerance: float = 1.0e-10,
) -> NDArray[np.float64]:
    frames = np.asarray(value, dtype=np.float64)
    if frames.ndim != 3 or frames.shape[1:] != (3, 3):
        raise MagneticSubspaceError("local_frames must have shape (nmag,3,3)")
    identity = np.eye(3)
    orthogonal = float(np.max(np.abs(frames @ np.swapaxes(frames, -1, -2) - identity)))
    handed = float(np.max(np.abs(np.cross(frames[:, 0], frames[:, 1]) - frames[:, 2])))
    if max(orthogonal, handed) > tolerance:
        raise MagneticSubspaceError(
            f"local frames are not right-handed orthonormal frames: orthogonal={orthogonal:.3e}, handed={handed:.3e}"
        )
    return frames


__all__ = ["validate_local_frames"]

