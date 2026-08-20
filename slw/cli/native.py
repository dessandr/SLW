"""Protocol objects for calculation drivers migrated off the argv adapter."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any


class NativeHandlerError(ValueError):
    """Raised when a native calculation handler cannot be prepared."""


@dataclass(frozen=True)
class NativeExecutionPlan:
    """A validated native calculation ready to execute on this process.

    ``run`` is constructed independently on every MPI rank.  The common CLI
    runner decides whether it is called on rank zero only or on every rank.
    This keeps MPI policy out of the namelist/argv compatibility layer.
    """

    backend_label: str
    run: Callable[[], int | None]
    all_ranks: bool = False
    warning: str | None = None
    summary: Sequence[tuple[str, Any]] = ()


def load_handler(reference: str) -> Callable[..., NativeExecutionPlan]:
    """Load ``module:function`` without importing a scientific backend early."""

    module_name, separator, function_name = reference.partition(":")
    if not separator or not module_name or not function_name:
        raise NativeHandlerError(
            f"invalid native handler {reference!r}; expected 'module:function'"
        )
    module = importlib.import_module(module_name)
    handler = getattr(module, function_name, None)
    if not callable(handler):
        raise NativeHandlerError(f"native handler {reference!r} is not callable")
    return handler


def prepare_native(reference: str, **kwargs: Any) -> NativeExecutionPlan:
    """Load and call a native prepare function, checking its return contract."""

    handler = load_handler(reference)
    try:
        plan = handler(**kwargs)
    except NativeHandlerError:
        raise
    except (TypeError, ValueError) as exc:
        raise NativeHandlerError(str(exc)) from exc
    if not isinstance(plan, NativeExecutionPlan):
        raise NativeHandlerError(
            f"native handler {reference!r} returned {type(plan).__name__}, "
            "not NativeExecutionPlan"
        )
    return plan


def native_help(reference: str, **kwargs: Any) -> str:
    """Return handler-owned help text for a native calculation."""

    module_name, separator, _ = reference.partition(":")
    if not separator:
        raise NativeHandlerError(
            f"invalid native handler {reference!r}; expected 'module:function'"
        )
    module = importlib.import_module(module_name)
    provider = getattr(module, "format_help", None)
    if not callable(provider):
        raise NativeHandlerError(
            f"native handler module {module_name!r} does not provide format_help()"
        )
    text = provider(**kwargs)
    return str(text)
