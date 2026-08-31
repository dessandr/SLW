"""Feature-gated fixed-projector direct mixed vertex (WT-K04)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from slw.wtorque.model.local_projection import localize_exchange

from .vertices import rotation_vertex


def direct_rotation_vertices(
    g_xc: object | None,
    projectors: object,
    axes: object,
    *,
    include: bool = False,
    projector_policy: str = "fixed",
) -> NDArray[np.complex128] | None:
    """Build ``i[L_ell(g_XC),sigma_c]/2`` only from declared exchange DFPT."""

    if not include:
        return None
    if g_xc is None:
        raise ValueError("include_direct_vertex requires exchange-resolved g_XC")
    if projector_policy != "fixed":
        raise ValueError("moving projectors require an explicit projector derivative dataset")
    local, _ = localize_exchange(g_xc, projectors)
    axis_array = np.asarray(axes, dtype=np.float64)
    if axis_array.shape != (local.shape[0], 3):
        raise ValueError("axes must have shape (nmag,3)")
    return np.stack(
        [rotation_vertex(local[index], axis_array[index]) for index in range(local.shape[0])],
        axis=0,
    )


__all__ = ["direct_rotation_vertices"]

