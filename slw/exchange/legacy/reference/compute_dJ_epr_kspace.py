"""Archived import path for the native scalar EPR dJ kernel."""

from importlib import import_module as _import_module

_impl = _import_module("slw.exchange.kernels.dj_epr")


def __getattr__(name):
    return getattr(_impl, name)
