"""Modular plot.in based plotting helpers."""

from slw.magph.plotting.config import PlotConfig, load_plot_config
from slw.magph.plotting.registry import dispatch_plot

__all__ = ["PlotConfig", "load_plot_config", "dispatch_plot"]
