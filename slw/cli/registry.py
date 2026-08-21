"""Calculation registry for the four public SLW stage executables."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from difflib import get_close_matches
from types import MappingProxyType
from typing import Any


class RegistryError(ValueError):
    """Raised for an unknown or ambiguous calculation request."""


@dataclass(frozen=True)
class Backend:
    module: str | None = None
    mpi_module: str | None = None
    handler: str | None = None
    injected: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (self.module is None) == (self.handler is None):
            raise ValueError("a backend must define exactly one of module or handler")

    @property
    def is_native(self) -> bool:
        return self.handler is not None


@dataclass(frozen=True)
class Action:
    name: str
    description: str
    backends: Mapping[str, Backend]
    aliases: tuple[str, ...] = ()
    alias_parameters: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    source_required: bool = False
    required_any: tuple[tuple[str, ...], ...] = ()

    @property
    def requires_source(self) -> bool:
        return self.source_required or len(self.backends) > 1


@dataclass(frozen=True)
class ResolvedAction:
    action: Action
    requested_name: str
    source: str
    backend: Backend
    parameters: dict[str, Any]


def _backend(
    module: str | None = None,
    *,
    mpi: str | None = None,
    handler: str | None = None,
    **injected: Any,
) -> Backend:
    return Backend(
        module=module,
        mpi_module=mpi,
        handler=handler,
        injected=MappingProxyType(dict(injected)),
    )


def _action(
    name: str,
    description: str,
    module: str | None = None,
    *,
    mpi: str | None = None,
    backends: Mapping[str, Backend] | None = None,
    aliases: tuple[str, ...] = (),
    alias_parameters: Mapping[str, Mapping[str, Any]] | None = None,
    source_required: bool = False,
    required_any: tuple[tuple[str, ...], ...] = (),
) -> Action:
    if backends is None:
        if module is None:
            raise ValueError(f"action {name} has no backend")
        backends = {"default": _backend(module, mpi=mpi)}
    return Action(
        name=name,
        description=description,
        backends=MappingProxyType(dict(backends)),
        aliases=aliases,
        alias_parameters=MappingProxyType(
            {
                key: MappingProxyType(dict(value))
                for key, value in (alias_parameters or {}).items()
            }
        ),
        source_required=source_required,
        required_any=required_any,
    )


_REGISTRY: dict[str, tuple[Action, ...]] = {
    "epr": (
        _action(
            "gkq",
            "Reconstruct Wannier-gauge g(k,q) from a QE qe2pert EPR file",
            "slw.epc.compute_gkq_from_epr",
        ),
        _action(
            "dispersion",
            "Validate electronic and phonon dispersions stored in EPR data",
            "slw.epc.check_qe2pert_epr_dispersion",
            required_any=(("win", "kpath", "compare_phdisp"),),
        ),
        _action(
            "phonon_cache",
            "Build the reusable phonon cache consumed by magph calculations",
            "slw.magph.legacy.reference.build_phonon_cache",
        ),
        _action(
            "kpath",
            "Generate an explicit high-symmetry path from a structure",
            "slw.core.make_kpath",
        ),
    ),
    "exchange": (
        _action(
            "j",
            "Compute LKAG exchange; ltensor selects scalar or tensor form",
            backends={
                "epr": _backend(handler="slw.exchange.engine:prepare_run"),
                "wannier": _backend(handler="slw.exchange.engine:prepare_run"),
            },
            aliases=("j_tensor",),
            alias_parameters={"j_tensor": {"ltensor": True}},
        ),
        _action(
            "dj",
            "Compute analytic dJ/du; ltensor selects scalar or tensor form",
            backends={
                "epr": _backend(handler="slw.exchange.engine:prepare_run"),
            },
            aliases=("dj_tensor",),
            alias_parameters={"dj_tensor": {"ltensor": True}},
            source_required=True,
        ),
    ),
    "magph": (
        _action(
            "dispersion",
            "Compute native MPI-distributed magnon bands and an optional plot",
            backends={
                "default": _backend(handler="slw.magph.engine:prepare_run"),
            },
        ),
        _action(
            "hybrid",
            "Build and diagonalize the hybrid magnon-phonon Hamiltonian",
            "slw.magph.legacy.reference.hybrid",
            required_any=(
                ("input_file", "s"),
                ("input_file", "spin_direction"),
                ("input_file", "kmesh"),
                ("input_file", "output"),
            ),
        ),
        _action(
            "berry",
            "Compute hybrid-band Berry curvature on a reciprocal-space plane",
            "slw.magph.legacy.reference.hybrid_berry",
            aliases=("hybrid_berry",),
        ),
        _action(
            "spectral",
            "Run the MPI-aware magnon-phonon spectral solver",
            "slw.magph.legacy.reference.solver_mpi",
            mpi="slw.magph.legacy.reference.solver_mpi",
            required_any=(("input_file",),),
        ),
        _action(
            "lifetime",
            "Compute native MPI-distributed magnon lifetimes",
            backends={
                "default": _backend(handler="slw.magph.engine:prepare_run"),
            },
        ),
        _action(
            "scattering_kbz",
            "Compute fixed-phonon-q scattering over the magnon Brillouin zone",
            "slw.magph.legacy.reference.plot_scattering_kbz",
        ),
        _action(
            "scattering_qbz",
            "Compute fixed-magnon scattering over the phonon Brillouin zone",
            "slw.magph.legacy.reference.plot_scattering_qbz",
        ),
        _action(
            "rotational_coupling",
            "Analyze rotational and chiral magnon-phonon coupling",
            "slw.magph.legacy.reference.analyze_rotational_coupling",
        ),
        _action(
            "chirality_plane",
            "Run restartable MPI chirality analysis on a reciprocal-space plane",
            "slw.magph.legacy.reference.run_chirality_plane_mpi",
            mpi="slw.magph.legacy.reference.run_chirality_plane_mpi",
        ),
        _action(
            "prepare_lifetime",
            "Create a compatibility manifest for lifetime and spectral solvers",
            "slw.magph.legacy.reference.prepare_lifetime",
        ),
    ),
    "post": (
        _action(
            "check_gkq",
            "Check g(k,q) invariants and sum rules",
            "slw.epc.check_gkq_constraints",
        ),
        _action(
            "check_spin_mz",
            "Check the spin-channel Mz relation for a compatible g(k,q) pair",
            "slw.epc.check_spin_mz_relation",
        ),
        _action(
            "compare_u_rotation",
            "Compare EPR Hamiltonians with candidate Wannier U rotations",
            "slw.epc.compare_epr_u_rotation",
        ),
        _action(
            "dj_asr",
            "Audit or project the acoustic sum rule of tensor dJ/du",
            "slw.exchange.legacy.check_dJ_asr",
        ),
        _action(
            "dj_kq_symmetry",
            "Check dJ(k,q) symmetry for an EPR/Wannier input pair",
            "slw.exchange.legacy.check_dJkq_symmetry",
        ),
        _action(
            "dump_dj",
            "Dump tensor dJ/du HDF5 data to text tables",
            "slw.exchange.legacy.dump_dJ_epr_tensor",
        ),
        _action(
            "spin_group",
            "Audit or project scalar exchange under the spin space group",
            "slw.exchange.legacy.project_scalar_spin_group",
        ),
        _action(
            "j_diagnostic",
            "Write detailed EPR k-space exchange diagnostics",
            "slw.exchange.legacy.diagnose_J_epr_kspace",
        ),
        _action(
            "rpa_lkag_audit",
            "Audit RPA and LKAG normalization conventions",
            "slw.exchange.legacy.audit_rpa_lkag_factor",
        ),
        _action(
            "spinflip_hr",
            "Inspect spin-flip blocks in spinor Wannier Hamiltonians",
            "slw.exchange.legacy.diagnose_spinflip_hr",
        ),
        _action(
            "spinflip_orbitals",
            "Inspect orbital-resolved spin-flip blocks",
            "slw.exchange.legacy.diagnose_spinflip_orbital_blocks",
        ),
        _action(
            "soc_bands",
            "Plot EPR bands with model spin-orbit coupling",
            "slw.exchange.legacy.plot_epr_soc_bands",
        ),
        _action(
            "tensor_summary",
            "Inspect or adapt exchange tensors for magph consumers",
            "slw.magph.legacy.tensor_adapter",
        ),
        _action(
            "phonon_rotation",
            "Analyze phonon rotational selectivity",
            "slw.magph.legacy.analyze_rotational_selectivity",
        ),
        _action(
            "lifetime_analysis",
            "Analyze lifetime output and symmetry channels",
            "slw.magph.legacy.analyze_lifetime",
        ),
        _action(
            "lifetime_plot",
            "Plot lifetime data with reciprocal-space symmetry",
            "slw.magph.legacy.plot_lifetime_symmetry",
        ),
        _action(
            "magnon_plot",
            "Plot a registered magnon dataset",
            "slw.magph.legacy.plot",
        ),
        _action(
            "coupling_kpath",
            "Plot coupling along a reciprocal-space path",
            "slw.magph.legacy.plot_coupling_kpath",
        ),
        _action(
            "coupling_bz",
            "Plot coupling over a Brillouin-zone plane",
            "slw.magph.legacy.plot_coupling_bz",
        ),
        _action(
            "coupling_kbz",
            "Plot fixed-q coupling over the magnon Brillouin zone",
            "slw.magph.legacy.plot_coupling_kbz",
        ),
    ),
}


def actions_for(stage: str) -> tuple[Action, ...]:
    try:
        return _REGISTRY[stage.lower()]
    except KeyError as exc:
        raise RegistryError(f"unknown SLW stage: {stage}") from exc


def find_action(stage: str, calculation: str) -> Action:
    wanted = calculation.strip().lower().replace("-", "_")
    actions = actions_for(stage)
    for action in actions:
        names = (action.name, *action.aliases)
        if wanted in names:
            return action
    choices = sorted(
        {name for action in actions for name in (action.name, *action.aliases)}
    )
    close = get_close_matches(wanted, choices, n=3)
    hint = f"; did you mean {', '.join(close)}?" if close else ""
    raise RegistryError(f"unknown calculation {calculation!r} for slw_{stage}.x{hint}")


def resolve_action(
    stage: str,
    calculation: str,
    parameters: Mapping[str, Any],
    *,
    validate_requirements: bool = True,
) -> ResolvedAction:
    action = find_action(stage, calculation)
    params = dict(parameters)
    requested_name = calculation.strip().lower().replace("-", "_")

    for key, value in action.alias_parameters.get(requested_name, {}).items():
        if key in params and params[key] != value:
            raise RegistryError(
                f"calculation={requested_name!r} fixes {key}={value!r}; "
                f"got {params[key]!r}"
            )
        params[key] = value

    source_value = params.pop("input_format", None)
    source_alias = params.pop("source", None) if action.requires_source else None
    if (
        source_value is not None
        and source_alias is not None
        and str(source_value).strip().lower() != str(source_alias).strip().lower()
    ):
        raise RegistryError("source and input_format specify different backends")
    source_value = source_value if source_value is not None else source_alias

    if action.requires_source:
        if source_value is None:
            choices = ", ".join(action.backends)
            raise RegistryError(
                f"calculation={action.name!r} requires input_format ({choices}) "
                f"in &{stage}"
            )
        source = str(source_value).strip().lower()
    else:
        source = "default"
        if source_value is not None:
            requested = str(source_value).strip().lower()
            if requested not in {"default", "epr"}:
                raise RegistryError(
                    f"calculation={action.name!r} does not support "
                    f"input_format={requested!r}"
                )

    try:
        backend = action.backends[source]
    except KeyError as exc:
        choices = ", ".join(action.backends)
        raise RegistryError(
            f"calculation={action.name!r} does not support "
            f"input_format={source!r}; choose {choices}"
        ) from exc

    for key, value in backend.injected.items():
        if key in params and params[key] != value:
            raise RegistryError(
                f"calculation={action.name!r}, input_format={source!r} fixes "
                f"{key}={value!r}; got {params[key]!r}"
            )
        params[key] = value

    if validate_requirements:
        for alternatives in action.required_any:
            if not any(
                key in params and params[key] not in (None, "", [])
                for key in alternatives
            ):
                rendered = " or ".join(alternatives)
                raise RegistryError(
                    f"calculation={action.name!r} requires an explicit {rendered} "
                    f"parameter in &{stage}"
                )

    return ResolvedAction(
        action=action,
        requested_name=requested_name,
        source=source,
        backend=backend,
        parameters=params,
    )
