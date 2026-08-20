"""Archived import path for the historical full LKAG solver."""

from importlib import import_module as _import_module

_impl = _import_module("slw.exchange.legacy.lkag_full")


def __getattr__(name):
    return getattr(_impl, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_impl)))
