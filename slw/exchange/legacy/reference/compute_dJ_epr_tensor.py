"""Archived import path for the native tensor EPR dJ kernel."""

from importlib import import_module as _import_module

_impl = _import_module("slw.exchange.kernels.dj_tensor_epr")


def __getattr__(name):
    return getattr(_impl, name)
