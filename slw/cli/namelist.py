"""Fortran-namelist input helpers used by the stage executables."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class NamelistError(ValueError):
    """Raised when an SLW input file is not a valid namelist."""


def _plain_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key).lower(): _plain_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_value(item) for item in value]
    return value


def loads(text: str) -> dict[str, dict[str, Any]]:
    """Parse a QE-style Fortran namelist string into plain dictionaries."""
    try:
        import f90nml
    except ModuleNotFoundError as exc:  # pragma: no cover - packaging guard
        raise RuntimeError(
            "QE-style input support requires f90nml. Reinstall SLW with "
            "`python -m pip install -e .`."
        ) from exc

    try:
        parsed = f90nml.reads(text)
    except Exception as exc:
        raise NamelistError(f"invalid Fortran namelist: {exc}") from exc

    groups = _plain_value(parsed)
    if not groups:
        raise NamelistError("input contains no namelist groups")
    for name, values in groups.items():
        if not isinstance(values, dict):
            raise NamelistError(f"&{name} must contain key/value assignments")
    return groups


def load(path: str | Path) -> dict[str, dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return loads(handle.read())


_PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def expand_placeholders(value: Any, variables: Mapping[str, str]) -> Any:
    """Expand explicit ``${name}`` placeholders recursively.

    Environment variables are deliberately not read: every expansion comes
    from the parsed workflow configuration and is therefore reproducible.
    """
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in variables:
                known = ", ".join(sorted(variables))
                raise NamelistError(
                    f"unknown path placeholder ${{{name}}}; available: {known}"
                )
            return variables[name]

        return _PLACEHOLDER.sub(replace, value)
    if isinstance(value, list):
        return [expand_placeholders(item, variables) for item in value]
    if isinstance(value, tuple):
        return tuple(expand_placeholders(item, variables) for item in value)
    if isinstance(value, dict):
        return {
            key: expand_placeholders(item, variables)
            for key, item in value.items()
        }
    return value
