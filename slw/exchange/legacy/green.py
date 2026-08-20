"""Archived import path for :mod:`slw.exchange.kernels.green`."""

from importlib import import_module as _import_module

_impl = _import_module("slw.exchange.kernels.green")


def __getattr__(name):
    return getattr(_impl, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_impl)))
