"""Modular plot.in based plotting helpers."""

from slw.magph.legacy.plotting.config import PlotConfig, load_plot_config
from slw.magph.legacy.plotting.registry import dispatch_plot

__all__ = ["PlotConfig", "load_plot_config", "dispatch_plot"]
