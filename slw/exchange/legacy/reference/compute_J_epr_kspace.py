"""Archived import path for the native scalar EPR J kernel."""

from importlib import import_module as _import_module

_impl = _import_module("slw.exchange.kernels.j_epr")


def __getattr__(name):
    return getattr(_impl, name)
