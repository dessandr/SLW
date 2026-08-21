"""Typed dispatch from the public exchange request to native kernels.

This module deliberately does not call an argparse parser, mutate ``sys.argv``,
or invoke a compatibility ``main``.  The namespace translation remains private
until each numerical kernel accepts its own typed request directly.
"""

from __future__ import annotations

import argparse
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..model import ExchangeCalculation, ExchangeRequest, ExchangeSource


@dataclass(frozen=True)
class KernelArtifacts:
    """Paths expected from one completed native-kernel run."""

    paths: tuple[str, ...]


_J_SCALAR_EPR_DEFAULTS: dict[str, Any] = {
    "atom_labels": "",
    "species_labels": "",
    "hr_unit": "ry",
    "mag_atoms_base": 0,
    "n_shells": 10,
    "d_max": 20.0,
    "emin": -25.0,
    "empoints": 500,
    "integrator": "contour",
    "cfr_beta": 400.0,
    "nproc": 1,
    "symprec": 1.0e-4,
    "angle_tolerance": -1.0,
    "orbit_grouping": "spglib",
    "debug_orbits": False,
    "debug_orbit_shell": None,
    "debug_epr_positions": False,
    "debug_shell": None,
    "debug_bond": "",
    "debug_out": "J_debug_pairs.tsv",
    "no_symmetry_orbits": False,
    "no_h5": False,
}

_J_TENSOR_EPR_DEFAULTS: dict[str, Any] = {
    "hr_unit": "ry",
    "kernel": "tb2j",
    "axes": "xyz",
    "spin_direction": (0.0, 0.0, 1.0),
    "soc": "",
    "win": None,
    "soc_element": "",
    "lambda_te": 0.0,
    "soc_p_groups": "",
    "soc_groups_base": 0,
    "p_order": "pz,px,py",
    "d_order": "dz2,dxz,dyz,dx2-y2,dxy",
    "n_shells": 10,
    "d_max": 20.0,
    "all_bonds": True,
    "nn_only": False,
    "atom_labels": "",
    "species_labels": None,
    "no_symmetry": False,
    "orbit_grouping": "spglib",
    "symprec": 1.0e-4,
    "angle_tolerance": -1.0,
    "debug_orbits": False,
    "debug_orbit_shell": None,
    "debug_epr_positions": False,
    "integrator": "contour",
    "emin": -25.0,
    "empoints": 500,
    "cfr_beta": 1000.0,
    "nproc": 1,
    "collinear_override": False,
    "spin_magnitude": 1.0,
    "d_subspace": "",
    "soc_active": "",
    "lambda_soc": None,
    "dynamic_soc": False,
}

_J_WANNIER_DEFAULTS: dict[str, Any] = {
    "up_hr": None,
    "dn_hr": None,
    "spinor_hr": None,
    "groupby": None,
    "centres": None,
    "hr_unit": "ev",
    "ref_epr_up": None,
    "ref_epr_dn": None,
    "ref_hr_unit": "ry",
    "mag_atoms_base": 0,
    "apply_degeneracy": True,
    "kernel": "tb2j",
    "axes": "xyz",
    "spin_direction": (0.0, 0.0, 1.0),
    "soc": "",
    "win": None,
    "mag_subspace": "",
    "soc_element": "",
    "lambda_te": 0.0,
    "soc_p_groups": "",
    "soc_groups_base": 0,
    "p_order": "pz,px,py",
    "d_order": "dz2,dxz,dyz,dx2-y2,dxy",
    "intersite_soc": False,
    "d_subspace": "",
    "soc_active": "",
    "lambda_soc": None,
    "e0": 0.0,
    "eta": 0.0,
    "hermitianize_soc": True,
    "n_shells": 10,
    "d_max": 20.0,
    "all_bonds": True,
    "nn_only": False,
    "orbit_grouping": "distance",
    "integrator": "contour",
    "emin": -25.0,
    "empoints": 500,
    "cfr_beta": 1000.0,
    "nproc": 1,
    "collinear_override": False,
    "spin_magnitude": 1.0,
    "dynamic_soc": False,
}

_DJ_SCALAR_DEFAULTS: dict[str, Any] = {
    "hr_unit": "ry",
    "eph_unit": "ry",
    "atom_labels": "",
    "species_labels": "",
    "targets": "all",
    "axes": "xyz",
    "mag_atoms_base": 0,
    "qmesh": None,
    "rp_idx": (0, 0, 0),
    "g_transform": "kq",
    "g_kernel": "direct",
    "n_shells": 1,
    "d_max": 20.0,
    "emin": -25.0,
    "empoints": 100,
    "integrator": "contour",
    "cfr_beta": 400.0,
    "nproc": 1,
    "omp_threads": 1,
    "precache_workers": 1,
    "rotation_mode": "none",
    "ddelta_mode": "off",
    "symprec": 1.0e-4,
    "angle_tolerance": -1.0,
    "orbit_grouping": "spglib",
    "debug_orbits": False,
    "debug_orbit_shell": None,
    "debug_epr_positions": False,
    "no_symmetry_orbits": False,
}

_DJ_TENSOR_DEFAULTS: dict[str, Any] = {
    "spinor_hr": None,
    "groupby": None,
    "spinor_hr_unit": "ev",
    "win": None,
    "centres": None,
    "mag_subspace": "",
    "apply_degeneracy": True,
    "hr_unit": "ry",
    "eph_unit": "ry",
    "qmesh": None,
    "n_shells": 10,
    "d_max": 20.0,
    "mag_atoms_base": 0,
    "targets": None,
    "disp_axes": "xyz",
    "tensor_axes": "xyz",
    "spin_direction": (0.0, 0.0, 1.0),
    "soc": "",
    "soc_element": "",
    "lambda_te": 0.0,
    "soc_p_groups": "",
    "soc_groups_base": 0,
    "p_order": "pz,px,py",
    "d_order": "dz2,dxz,dyz,dx2-y2,dxy",
    "emin": -25.0,
    "empoints": 300,
    "integrator": "contour",
    "nproc": 1,
    "numba_threads": 1,
    "blas_threads": 1,
    "progress_every": 0,
    "verbose_worker_init": 0,
    "checkpoint": 1,
    "onsite_deriv_projector": True,
}


def _slice_text(request: ExchangeRequest) -> str:
    return ",".join(
        f"{item.site}:{item.start}:{item.stop}" for item in request.slices
    )


def _normalize_aliases(options: dict[str, Any]) -> None:
    if "canonical_bonds" in options:
        canonical = bool(options.pop("canonical_bonds"))
        if "all_bonds" in options and bool(options["all_bonds"]) == canonical:
            raise ValueError("canonical_bonds conflicts with all_bonds")
        options["all_bonds"] = not canonical
    if "soc_p_groups_base" in options:
        alias = options.pop("soc_p_groups_base")
        if "soc_groups_base" in options and options["soc_groups_base"] != alias:
            raise ValueError("soc_p_groups_base conflicts with soc_groups_base")
        options["soc_groups_base"] = alias
    if "onsite_deriv_exchange_field" in options:
        alias = options.pop("onsite_deriv_exchange_field")
        if (
            "onsite_deriv_projector" in options
            and options["onsite_deriv_projector"] != alias
        ):
            raise ValueError(
                "onsite_deriv_exchange_field conflicts with onsite_deriv_projector"
            )
        options["onsite_deriv_projector"] = alias


def _core_options(request: ExchangeRequest) -> dict[str, Any]:
    files = {
        key: value
        for key, value in vars(request.files).items()
        if value is not None
    }
    options = {
        **files,
        "efermi": request.efermi,
        "kmesh": request.kmesh,
        "mag_atoms": request.mag_atoms,
        "mag_atoms_base": 0,
        "slices": _slice_text(request),
        # Absolute product paths prevent kernel writers from joining a
        # relative out_dir twice.  No directory is created during conversion.
        "out_dir": str(Path(request.output.directory).resolve()),
        "out_name": str(Path(request.output.text_path).resolve()),
        "out_h5": str(Path(request.output.h5_path).resolve()),
    }
    return options


def build_namespace(request: ExchangeRequest) -> tuple[str, str, argparse.Namespace]:
    """Return ``(module, callable, namespace)`` for one native kernel."""

    if request.calculation is ExchangeCalculation.J:
        if request.source is ExchangeSource.EPR and not request.ltensor:
            module = "slw.exchange.kernels.j_epr"
            function = "run"
            defaults = _J_SCALAR_EPR_DEFAULTS
        elif request.source is ExchangeSource.EPR:
            module = "slw.exchange.kernels.j_tensor_epr"
            function = "run"
            defaults = _J_TENSOR_EPR_DEFAULTS
        else:
            module = "slw.exchange.kernels.j_wannier"
            function = "run"
            defaults = _J_WANNIER_DEFAULTS
    elif request.ltensor:
        module = "slw.exchange.kernels.dj_tensor_epr"
        function = "run_analytic"
        defaults = _DJ_TENSOR_DEFAULTS
    else:
        module = "slw.exchange.kernels.dj_epr"
        function = "run"
        defaults = _DJ_SCALAR_DEFAULTS

    options = dict(defaults)
    advanced = request.options.as_dict()
    _normalize_aliases(advanced)
    options.update(advanced)
    options.update(_core_options(request))
    options["groupby"] = request.groupby.value if request.groupby is not None else None
    options["soc_manifolds"] = (
        tuple(
            {
                "selector": manifold.selector,
                "lambda_ev": manifold.lambda_ev,
            }
            for manifold in request.soc.manifolds
        )
        if request.soc is not None
        else ()
    )
    if request.calculation is ExchangeCalculation.J:
        if request.ltensor:
            options["kernel"] = request.tensor_kernel.value
        elif request.source is ExchangeSource.WANNIER:
            options["kernel"] = "scalar"
    return module, function, argparse.Namespace(**options)


def execute(
    request: ExchangeRequest,
    *,
    mpi: bool = False,
    comm: Any | None = None,
) -> KernelArtifacts:
    """Execute one native kernel without entering its CLI driver."""

    module_name, function_name, namespace = build_namespace(request)
    if mpi and request.calculation is ExchangeCalculation.DJ and request.ltensor:
        module_name = "slw.exchange.kernels.dj_tensor_mpi"
        function_name = "run_mpi"
    module = importlib.import_module(module_name)
    function = getattr(module, function_name)
    result = function(namespace, comm=comm) if mpi else function(namespace)
    if isinstance(result, int) and result != 0:
        raise RuntimeError(f"native kernel returned exit code {result}")

    paths = [request.output.h5_path]
    if not (request.calculation is ExchangeCalculation.DJ and request.ltensor):
        paths.append(request.output.text_path)
    has_table = (
        request.calculation is ExchangeCalculation.DJ and not request.ltensor
    ) or (
        request.calculation is ExchangeCalculation.J
        and not request.ltensor
        and request.source is ExchangeSource.EPR
    )
    if has_table:
        paths.append(request.output.table_path)
    if request.options.as_dict().get("no_h5", False):
        paths = [path for path in paths if Path(path) != Path(request.output.h5_path)]
    return KernelArtifacts(paths=tuple(paths))


__all__ = ["KernelArtifacts", "build_namespace", "execute"]
