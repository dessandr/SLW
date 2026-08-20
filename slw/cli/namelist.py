"""Fortran-namelist input helpers used by the stage executables."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class NamelistError(ValueError):
    """Raised when an SLW input file is not a valid namelist."""


_SOC_HEADER = re.compile(r"^SOC\s*\(\s*([^)]+)\s*\)\s*$", re.IGNORECASE)


def _strip_card_comment(line: str) -> str:
    return line.split("!", 1)[0].split("#", 1)[0].strip()


def _extract_soc_card(text: str) -> tuple[str, dict[str, Any] | None]:
    """Remove and parse one QE-style ``SOC (atomic)`` card.

    The initial public grammar deliberately keeps the card last in the file,
    which makes malformed card contents fail closed instead of being silently
    consumed by a following namelist.
    """

    lines = text.splitlines(keepends=True)
    header_index: int | None = None
    mode = ""
    in_namelist = False
    for index, raw in enumerate(lines):
        clean = _strip_card_comment(raw)
        if clean.startswith("&"):
            in_namelist = True
        if in_namelist:
            if clean == "/" or clean.endswith("/"):
                in_namelist = False
            continue
        match = _SOC_HEADER.fullmatch(clean)
        if match is not None:
            if header_index is not None:
                raise NamelistError("input contains more than one SOC card")
            header_index = index
            mode = match.group(1).strip().lower()

    if header_index is None:
        return text, None
    if mode != "atomic":
        raise NamelistError(
            f"unsupported SOC card mode {mode!r}; only SOC (atomic) is supported"
        )

    entries: list[dict[str, Any]] = []
    for raw in lines[header_index + 1 :]:
        clean = _strip_card_comment(raw)
        if not clean:
            continue
        if clean.startswith("&") or _SOC_HEADER.fullmatch(clean):
            raise NamelistError("SOC (atomic) must be the final input card")
        fields = clean.split()
        if len(fields) != 3 or fields[0].upper() != "LAMBDA":
            raise NamelistError(
                "SOC (atomic) entries must use: LAMBDA LABEL-p|LABEL-d value_eV"
            )
        try:
            value = float(fields[2].replace("d", "e").replace("D", "E"))
        except ValueError as exc:
            raise NamelistError(
                f"SOC lambda must be a finite number in eV, got {fields[2]!r}"
            ) from exc
        if not math.isfinite(value):
            raise NamelistError(
                f"SOC lambda must be a finite number in eV, got {fields[2]!r}"
            )
        entries.append({"selector": fields[1], "lambda_ev": value})
    if not entries:
        raise NamelistError("SOC (atomic) must contain at least one LAMBDA entry")
    namelist_text = "".join(lines[:header_index])
    return namelist_text, {"mode": mode, "entries": entries}


def _plain_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key).lower(): _plain_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_value(item) for item in value]
    return value


def loads(text: str) -> dict[str, dict[str, Any]]:
    """Parse a QE-style Fortran namelist string into plain dictionaries."""
    text, soc_card = _extract_soc_card(text)
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
    if soc_card is not None:
        groups["soc_card"] = soc_card
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
