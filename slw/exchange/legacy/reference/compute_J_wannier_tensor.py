"""Archived import path for the native Wannier J kernel."""

from importlib import import_module as _import_module

_impl = _import_module("slw.exchange.kernels.j_wannier")


def __getattr__(name):
    return getattr(_impl, name)
