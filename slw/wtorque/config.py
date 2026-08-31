"""Typed, strict YAML/JSON configuration for the wtorque workflow."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from slw.soc import (
    WANNIER90_D_ORDER,
    WANNIER90_P_ORDER,
    AtomicSOCManifold,
    AtomicSOCSpec,
)

from .basis import SpinOrder
from .errors import ConfigurationError


class BlochGauge(str, Enum):
    ATOMIC_POSITION = "atomic_position"
    CELL_PERIODIC = "cell_periodic"


class Normalization(str, Enum):
    CARTESIAN_DERIVATIVE = "cartesian_derivative"
    MASS_WEIGHTED_CARTESIAN = "mass_weighted_cartesian"
    PHONON_ZERO_POINT_MODE = "phonon_zero_point_mode"


class ExchangeExtraction(str, Enum):
    EXPLICIT = "explicit"
    TIME_REVERSAL = "time_reversal"
    COLLINEAR_SPLIT = "collinear_split"


class SiteProjection(str, Enum):
    LOCAL_PARTITION = "local_partition"
    ONSITE_ONLY = "onsite_only"
    USER_SUPPLIED = "user_supplied"
    FULL_ATOM = "full_atom"


class SpinCoordinate(str, Enum):
    TRANSVERSE_DIRECTION = "transverse_direction"
    ROTATION_ANGLE = "rotation_angle"


class SpinorLift(str, Enum):
    NATIVE_SPINOR = "native_spinor"
    SPIN_SCALAR = "spin_scalar"
    COLLINEAR_SPIN_DEPENDENT = "collinear_spin_dependent"


class ProjectorPolicy(str, Enum):
    FIXED = "fixed"
    MOVING = "moving"


def _mapping(parent: Mapping[str, Any], key: str, *, required: bool = True) -> Mapping[str, Any]:
    value = parent.get(key)
    if value is None and not required:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"configuration block {key!r} must be a mapping")
    return value


def _required(section: Mapping[str, Any], key: str, block: str) -> Any:
    if key not in section:
        raise ConfigurationError(f"{block}.{key} is required")
    return section[key]


def _enum(enum_type: type[Enum], value: Any, path: str) -> Any:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in enum_type)
        raise ConfigurationError(f"{path} must be one of: {allowed}; got {value!r}") from exc


def _path(value: Any, base_dir: Path, path: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value):
        raise ConfigurationError(f"{path} must be a non-empty path")
    candidate = Path(value).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (base_dir / candidate).resolve()


def _onsite_soc_config(
    electrons: Mapping[str, Any],
    base_dir: Path,
) -> OnsiteSOCConfig | None:
    raw = electrons.get("onsite_soc")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ConfigurationError("electrons.onsite_soc must be a mapping")
    allowed = {"win", "entries", "scale", "p_order", "d_order"}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ConfigurationError(
            f"electrons.onsite_soc has unknown keys: {', '.join(unknown)}"
        )
    entries = _required(raw, "entries", "electrons.onsite_soc")
    if not isinstance(entries, list) or not entries:
        raise ConfigurationError("electrons.onsite_soc.entries must be a non-empty list")
    manifolds: list[AtomicSOCManifold] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise ConfigurationError(
                f"electrons.onsite_soc.entries[{index}] must be a mapping"
            )
        entry_unknown = sorted(set(entry) - {"selector", "lambda_ev"})
        if entry_unknown or "selector" not in entry or "lambda_ev" not in entry:
            raise ConfigurationError(
                f"electrons.onsite_soc.entries[{index}] must contain only "
                "selector and lambda_ev"
            )
        try:
            manifolds.append(
                AtomicSOCManifold(
                    selector=str(entry["selector"]),
                    lambda_ev=entry["lambda_ev"],
                )
            )
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(str(exc)) from exc
    try:
        spec = AtomicSOCSpec(tuple(manifolds))
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from exc
    scale = float(raw.get("scale", 1.0))
    if not math.isfinite(scale):
        raise ConfigurationError("electrons.onsite_soc.scale must be finite")
    return OnsiteSOCConfig(
        win_file=_path(
            _required(raw, "win", "electrons.onsite_soc"),
            base_dir,
            "electrons.onsite_soc.win",
        ),
        spec=spec,
        scale=scale,
        p_order=str(raw.get("p_order", WANNIER90_P_ORDER)),
        d_order=str(raw.get("d_order", WANNIER90_D_ORDER)),
    )


@dataclass(frozen=True)
class OnsiteSOCConfig:
    """Input-resolved atomic SOC added to the time-reversal-even Hamiltonian."""

    win_file: Path
    spec: AtomicSOCSpec
    scale: float = 1.0
    p_order: str = WANNIER90_P_ORDER
    d_order: str = WANNIER90_D_ORDER


@dataclass(frozen=True)
class ElectronsConfig:
    file: Path
    bloch_gauge: BlochGauge
    exchange_extraction: ExchangeExtraction
    spin_order: SpinOrder
    onsite_soc: OnsiteSOCConfig | None = None


@dataclass(frozen=True)
class MagneticSiteConfig:
    atom: int
    orbitals: tuple[int, ...]


@dataclass(frozen=True)
class MagneticSubspaceConfig:
    policy: str
    sites: tuple[MagneticSiteConfig, ...]
    site_projection: SiteProjection
    spin_coordinate: SpinCoordinate


@dataclass(frozen=True)
class DFPTConfig:
    file: Path
    normalization: Normalization
    spinor_lift: SpinorLift
    include_dHsoc_du: bool = False
    final_state_representation: str = "unwrapped"
    g_xc_dataset: str | None = None


@dataclass(frozen=True)
class KernelConfig:
    include_direct_vertex: bool
    fixed_chemical_potential: bool
    q_pair_completion: bool
    projector_policy: ProjectorPolicy = ProjectorPolicy.FIXED


@dataclass(frozen=True)
class IntegrationConfig:
    backend: str
    eta_eV: float
    energy_min_eV: float | None = None
    energy_max_eV: float | None = None
    energy_points: int | None = None
    temperature_K: float = 0.0


@dataclass(frozen=True)
class OutputConfig:
    file: Path
    resume: bool


@dataclass(frozen=True)
class PerformanceConfig:
    backend: str = "numpy"
    memory_limit_mb: int = 2048
    perturbation_chunk: int | None = None


@dataclass(frozen=True)
class RunConfig:
    electrons: ElectronsConfig
    magnetic_subspace: MagneticSubspaceConfig
    dfpt: DFPTConfig
    kernel: KernelConfig
    integration: IntegrationConfig
    output: OutputConfig
    performance: PerformanceConfig = PerformanceConfig()
    phonons_file: Path | None = None
    magnons_file: Path | None = None
    raw: Mapping[str, Any] | None = None

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Any],
        *,
        base_dir: str | Path = ".",
    ) -> RunConfig:
        if not isinstance(mapping, Mapping):
            raise ConfigurationError("top-level configuration must be a mapping")
        base = Path(base_dir).resolve()
        e = _mapping(mapping, "electrons")
        electrons = ElectronsConfig(
            file=_path(_required(e, "file", "electrons"), base, "electrons.file"),
            bloch_gauge=_enum(BlochGauge, _required(e, "bloch_gauge", "electrons"), "electrons.bloch_gauge"),
            exchange_extraction=_enum(
                ExchangeExtraction,
                _required(e, "exchange_extraction", "electrons"),
                "electrons.exchange_extraction",
            ),
            spin_order=_enum(SpinOrder, _required(e, "spin_order", "electrons"), "electrons.spin_order"),
            onsite_soc=_onsite_soc_config(e, base),
        )

        m = _mapping(mapping, "magnetic_subspace")
        sites_raw = _required(m, "sites", "magnetic_subspace")
        if not isinstance(sites_raw, list) or not sites_raw:
            raise ConfigurationError("magnetic_subspace.sites must be a non-empty list")
        sites: list[MagneticSiteConfig] = []
        seen: set[int] = set()
        for index, site_raw in enumerate(sites_raw):
            if not isinstance(site_raw, Mapping):
                raise ConfigurationError(f"magnetic_subspace.sites[{index}] must be a mapping")
            atom = int(_required(site_raw, "atom", f"magnetic_subspace.sites[{index}]"))
            orbitals_raw = _required(site_raw, "orbitals", f"magnetic_subspace.sites[{index}]")
            if not isinstance(orbitals_raw, list) or not orbitals_raw:
                raise ConfigurationError(f"magnetic_subspace.sites[{index}].orbitals must be non-empty")
            orbitals = tuple(int(value) for value in orbitals_raw)
            if any(value < 0 for value in orbitals) or len(set(orbitals)) != len(orbitals):
                raise ConfigurationError("magnetic orbital indices must be unique and non-negative per site")
            overlap = seen.intersection(orbitals)
            if overlap:
                raise ConfigurationError(f"magnetic orbital masks must be disjoint; overlap={sorted(overlap)}")
            seen.update(orbitals)
            sites.append(MagneticSiteConfig(atom=atom, orbitals=orbitals))
        magnetic = MagneticSubspaceConfig(
            policy=str(_required(m, "policy", "magnetic_subspace")),
            sites=tuple(sites),
            site_projection=_enum(
                SiteProjection,
                _required(m, "site_projection", "magnetic_subspace"),
                "magnetic_subspace.site_projection",
            ),
            spin_coordinate=_enum(
                SpinCoordinate,
                _required(m, "spin_coordinate", "magnetic_subspace"),
                "magnetic_subspace.spin_coordinate",
            ),
        )

        d = _mapping(mapping, "dfpt")
        dfpt = DFPTConfig(
            file=_path(_required(d, "file", "dfpt"), base, "dfpt.file"),
            normalization=_enum(
                Normalization,
                _required(d, "normalization", "dfpt"),
                "dfpt.normalization",
            ),
            spinor_lift=_enum(
                SpinorLift,
                _required(d, "spinor_lift", "dfpt"),
                "dfpt.spinor_lift",
            ),
            include_dHsoc_du=bool(d.get("include_dHsoc_du", False)),
            final_state_representation=str(d.get("final_state_representation", "unwrapped")),
            g_xc_dataset=None if d.get("g_xc_dataset") is None else str(d["g_xc_dataset"]),
        )
        if dfpt.final_state_representation not in {"wrapped", "unwrapped"}:
            raise ConfigurationError("dfpt.final_state_representation must be wrapped or unwrapped")

        k = _mapping(mapping, "kernel")
        kernel = KernelConfig(
            include_direct_vertex=bool(_required(k, "include_direct_vertex", "kernel")),
            fixed_chemical_potential=bool(_required(k, "fixed_chemical_potential", "kernel")),
            q_pair_completion=bool(_required(k, "q_pair_completion", "kernel")),
            projector_policy=_enum(ProjectorPolicy, k.get("projector_policy", "fixed"), "kernel.projector_policy"),
        )
        if not kernel.fixed_chemical_potential:
            raise ConfigurationError("the MVP implements only fixed_chemical_potential=true")
        if not kernel.q_pair_completion:
            raise ConfigurationError("general finite-q runs require q_pair_completion=true")
        if kernel.include_direct_vertex and dfpt.g_xc_dataset is None:
            raise ConfigurationError(
                "kernel.include_direct_vertex=true requires dfpt.g_xc_dataset"
            )
        if kernel.include_direct_vertex and kernel.projector_policy is ProjectorPolicy.MOVING:
            direct = _mapping(mapping, "direct_vertex", required=False)
            if "projector_derivative_dataset" not in direct:
                raise ConfigurationError(
                    "moving-projector direct vertices require direct_vertex.projector_derivative_dataset"
                )

        i = _mapping(mapping, "integration")
        backend = str(_required(i, "backend", "integration"))
        if backend not in {"real_axis", "contour"}:
            raise ConfigurationError("integration.backend must be real_axis or contour")
        eta = float(_required(i, "eta_eV", "integration"))
        if eta <= 0:
            raise ConfigurationError("integration.eta_eV must be positive")
        integration = IntegrationConfig(
            backend=backend,
            eta_eV=eta,
            energy_min_eV=None if i.get("energy_min_eV") is None else float(i["energy_min_eV"]),
            energy_max_eV=None if i.get("energy_max_eV") is None else float(i["energy_max_eV"]),
            energy_points=None if i.get("energy_points") is None else int(i["energy_points"]),
            temperature_K=float(i.get("temperature_K", 0.0)),
        )

        o = _mapping(mapping, "output")
        output = OutputConfig(
            file=_path(_required(o, "file", "output"), base, "output.file"),
            resume=bool(_required(o, "resume", "output")),
        )
        phonons = _mapping(mapping, "phonons", required=False)
        magnons = _mapping(mapping, "magnons", required=False)
        phonons_file = None if "file" not in phonons else _path(phonons["file"], base, "phonons.file")
        magnons_file = None if "file" not in magnons else _path(magnons["file"], base, "magnons.file")
        p = _mapping(mapping, "performance", required=False)
        performance = PerformanceConfig(
            backend=str(p.get("backend", "numpy")),
            memory_limit_mb=int(p.get("memory_limit_mb", 2048)),
            perturbation_chunk=None
            if p.get("perturbation_chunk") is None
            else int(p["perturbation_chunk"]),
        )
        if performance.backend not in {"numpy", "cupy"}:
            raise ConfigurationError("performance.backend must be numpy or cupy")
        if performance.memory_limit_mb < 1:
            raise ConfigurationError("performance.memory_limit_mb must be positive")
        if performance.perturbation_chunk is not None and performance.perturbation_chunk < 1:
            raise ConfigurationError("performance.perturbation_chunk must be positive")
        return cls(
            electrons=electrons,
            magnetic_subspace=magnetic,
            dfpt=dfpt,
            kernel=kernel,
            integration=integration,
            output=output,
            performance=performance,
            phonons_file=phonons_file,
            magnons_file=magnons_file,
            raw=dict(mapping),
        )

    @classmethod
    def load(cls, path: str | Path) -> RunConfig:
        source = Path(path).resolve()
        try:
            text = source.read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigurationError(f"cannot read configuration {source}: {exc}") from exc
        if source.suffix.lower() == ".json":
            payload = json.loads(text)
        else:
            try:
                import yaml  # type: ignore[import-untyped]
            except ImportError as exc:  # pragma: no cover - declared dependency
                raise ConfigurationError("PyYAML is required to read YAML configuration") from exc
            payload = yaml.safe_load(text)
        return cls.from_mapping(payload, base_dir=source.parent)

    def resolved_dict(self) -> dict[str, Any]:
        """Return the stable, JSON-ready resolved configuration."""

        value = asdict(self)
        value.pop("raw", None)

        def normalize(item: Any) -> Any:
            if isinstance(item, Path):
                return str(item)
            if isinstance(item, Enum):
                return item.value
            if isinstance(item, dict):
                return {str(key): normalize(val) for key, val in item.items()}
            if isinstance(item, (list, tuple)):
                return [normalize(val) for val in item]
            return item

        return normalize(value)


__all__ = [
    "BlochGauge",
    "ConfigurationError",
    "ExchangeExtraction",
    "Normalization",
    "OnsiteSOCConfig",
    "ProjectorPolicy",
    "RunConfig",
    "SiteProjection",
    "SpinCoordinate",
    "SpinorLift",
]
