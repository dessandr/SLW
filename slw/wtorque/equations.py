"""Implemented equation registry and provenance-class lint."""

from __future__ import annotations

import re
from pathlib import Path

IMPLEMENTED_EQUATIONS: dict[str, str] = {
    "WT-E01": "STANDARD",
    "WT-E02": "SOURCE-DERIVED",
    "WT-E03": "PROJECT-EXTENSION",
    "WT-E04": "SOURCE-DERIVED",
    "WT-L01": "ORTHONORMAL-LIMIT",
    "WT-T01": "STANDARD",
    "WT-T02": "SOURCE-DERIVED",
    "WT-T03": "SOURCE-DERIVED",
    "WT-T04": "STANDARD",
    "WT-T05": "PROJECT-EXTENSION",
    "WT-K01": "STANDARD",
    "WT-K02": "PROJECT-EXTENSION",
    "WT-K03": "PROJECT-EXTENSION",
    "WT-K04": "PROJECT-EXTENSION",
    "WT-P01": "STANDARD",
    "WT-P02": "PROJECT-EXTENSION",
    "WT-M01": "STANDARD",
    "WT-M02": "PROJECT-EXTENSION",
    "WT-M03": "PROJECT-EXTENSION",
    "WT-M04": "PROJECT-EXTENSION",
    "WT-S01": "PROJECT-EXTENSION",
    "WT-S02": "STANDARD",
    "WT-S03": "STANDARD",
}

PROVENANCE_CLASSES = {
    "SOURCE-DERIVED",
    "ORTHONORMAL-LIMIT",
    "STANDARD",
    "PROJECT-EXTENSION",
}


def lint_equation_registry(equation_index: str | Path) -> tuple[str, ...]:
    text = Path(equation_index).read_text(encoding="utf-8")
    documented = set(re.findall(r"\bWT-[A-Z]\d{2}\b", text))
    errors: list[str] = []
    for equation, provenance in sorted(IMPLEMENTED_EQUATIONS.items()):
        if equation not in documented:
            errors.append(f"implemented equation {equation} is absent from the equation index")
        if provenance not in PROVENANCE_CLASSES:
            errors.append(f"implemented equation {equation} has invalid provenance {provenance}")
    undocumented = sorted(documented - IMPLEMENTED_EQUATIONS.keys())
    errors.extend(f"documented equation {equation} is not registered" for equation in undocumented)
    return tuple(errors)


__all__ = ["IMPLEMENTED_EQUATIONS", "PROVENANCE_CLASSES", "lint_equation_registry"]

