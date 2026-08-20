"""Plot kind dispatch."""

from __future__ import annotations

from collections.abc import Callable

from slw.magph.legacy.plotting.config import PlotConfig
from slw.magph.legacy.plotting.magnon_h5 import plot_magnon_h5


_REGISTRY: dict[str, Callable[[PlotConfig], object]] = {
    "magnon_h5": plot_magnon_h5,
    "magnon-h5": plot_magnon_h5,
    "magnon": plot_magnon_h5,
}


def available_kinds() -> list[str]:
    return sorted(_REGISTRY)


def dispatch_plot(config: PlotConfig):
    kind = config.require("kind").strip().lower()
    try:
        func = _REGISTRY[kind]
    except KeyError as exc:
        supported = ", ".join(available_kinds())
        raise ValueError(f"Unsupported plot kind '{kind}'. Supported: {supported}") from exc
    return func(config)
