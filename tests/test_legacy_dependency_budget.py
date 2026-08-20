"""Architecture gate for the one-way legacy quarantine.

The allowlist is intentionally exact and transitional.  Removing a crossing
requires shrinking this list; adding one fails the test.  Files below either
legacy package are excluded because the archive may depend on native helpers,
but production code must move monotonically toward zero legacy references.
"""

from __future__ import annotations

import ast
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "slw"
LEGACY_PREFIXES = ("slw.exchange.legacy", "slw.magph.legacy")


TRANSITIONAL_LEGACY_REFERENCES = Counter(
    {
        ("slw/cli/registry.py", "slw.magph.legacy.reference.build_phonon_cache"): 1,
        ("slw/cli/registry.py", "slw.magph.legacy.reference.hybrid"): 1,
        ("slw/cli/registry.py", "slw.magph.legacy.reference.hybrid_berry"): 1,
        ("slw/cli/registry.py", "slw.magph.legacy.reference.solver_mpi"): 2,
        ("slw/cli/registry.py", "slw.magph.legacy.reference.plot_scattering_kbz"): 1,
        ("slw/cli/registry.py", "slw.magph.legacy.reference.plot_scattering_qbz"): 1,
        (
            "slw/cli/registry.py",
            "slw.magph.legacy.reference.analyze_rotational_coupling",
        ): 1,
        (
            "slw/cli/registry.py",
            "slw.magph.legacy.reference.run_chirality_plane_mpi",
        ): 2,
        ("slw/cli/registry.py", "slw.magph.legacy.reference.prepare_lifetime"): 1,
        ("slw/cli/registry.py", "slw.exchange.legacy.check_dJ_asr"): 1,
        ("slw/cli/registry.py", "slw.exchange.legacy.check_dJkq_symmetry"): 1,
        ("slw/cli/registry.py", "slw.exchange.legacy.dump_dJ_epr_tensor"): 1,
        ("slw/cli/registry.py", "slw.exchange.legacy.project_scalar_spin_group"): 1,
        ("slw/cli/registry.py", "slw.exchange.legacy.diagnose_J_epr_kspace"): 1,
        ("slw/cli/registry.py", "slw.exchange.legacy.audit_rpa_lkag_factor"): 1,
        ("slw/cli/registry.py", "slw.exchange.legacy.diagnose_spinflip_hr"): 1,
        (
            "slw/cli/registry.py",
            "slw.exchange.legacy.diagnose_spinflip_orbital_blocks",
        ): 1,
        ("slw/cli/registry.py", "slw.exchange.legacy.plot_epr_soc_bands"): 1,
        ("slw/cli/registry.py", "slw.magph.legacy.tensor_adapter"): 1,
        (
            "slw/cli/registry.py",
            "slw.magph.legacy.analyze_rotational_selectivity",
        ): 1,
        ("slw/cli/registry.py", "slw.magph.legacy.analyze_lifetime"): 1,
        ("slw/cli/registry.py", "slw.magph.legacy.plot_lifetime_symmetry"): 1,
        ("slw/cli/registry.py", "slw.magph.legacy.plot"): 1,
        ("slw/cli/registry.py", "slw.magph.legacy.plot_coupling_kpath"): 1,
        ("slw/cli/registry.py", "slw.magph.legacy.plot_coupling_bz"): 1,
        ("slw/cli/registry.py", "slw.magph.legacy.plot_coupling_kbz"): 1,
    }
)


def _resolved_from_module(path: Path, node: ast.ImportFrom) -> str:
    module = node.module or ""
    if node.level == 0:
        return module
    relative_path = path.relative_to(ROOT).with_suffix("")
    package_parts = list(relative_path.parts[:-1])
    parent_hops = node.level - 1
    if parent_hops:
        package_parts = package_parts[:-parent_hops]
    return (
        ".".join((*package_parts, *module.split(".")))
        if module
        else ".".join(package_parts)
    )


def _active_legacy_references() -> Counter[tuple[str, str]]:
    references: Counter[tuple[str, str]] = Counter()
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        relative = path.relative_to(ROOT)
        if "legacy" in relative.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(relative))
        for node in ast.walk(tree):
            targets: list[str] = []
            if isinstance(node, ast.Import):
                targets.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                targets.append(_resolved_from_module(path, node))
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                targets.append(node.value)
            for target in targets:
                if target.startswith(LEGACY_PREFIXES):
                    references[(relative.as_posix(), target)] += 1
    return references


class LegacyDependencyBudgetTests(unittest.TestCase):
    def test_active_legacy_references_match_the_shrinking_budget(self) -> None:
        actual = _active_legacy_references()
        self.assertEqual(
            actual,
            TRANSITIONAL_LEGACY_REFERENCES,
            "Active-to-legacy references changed. New crossings are forbidden; "
            "when a crossing is removed, shrink TRANSITIONAL_LEGACY_REFERENCES.",
        )

    def test_native_magph_has_no_legacy_reference(self) -> None:
        crossings = {
            entry: count
            for entry, count in _active_legacy_references().items()
            if entry[0].startswith("slw/magph/")
        }
        self.assertEqual(crossings, {})


if __name__ == "__main__":
    unittest.main()
