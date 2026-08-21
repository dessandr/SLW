"""Native public API for exchange-based magnon--phonon observables.

The array kernels and strict data contracts here are independent of the
quarantined compatibility drivers under :mod:`slw.magph.legacy`.  Native
``calculation='dispersion'`` and ``calculation='lifetime'`` have completed
their initial FM/AFM gate; unrelated registry calculations remain
compatibility-backed until replaced explicitly.
"""

from .config import (
    MagphDispersionRequest,
    MagphInputError,
    MagphLifetimeRequest,
    RestartMode,
    SingleIonAnisotropyInput,
    build_dispersion_request,
    build_lifetime_request,
)
from .coupling import (
    ModeResolvedExchangeDerivative,
    build_mode_resolved_isotropic_derivative,
    build_mode_resolved_isotropic_derivative_distributed,
)
from .derivative import (
    DerivativeASRPolicy,
    ExchangeDerivativeModel,
    ExchangeDerivativeReport,
    load_exchange_derivative_h5,
)
from .dispersion import (
    DistributedMagnonDispersion,
    MagnonDispersionResult,
    MagnonKPath,
    build_wannier90_kpath,
    compute_magnon_dispersion,
)
from .engine import MagphRunResult, run_dispersion, run_lifetime
from .lifetime import (
    LifetimeResult,
    compute_lifetime,
    lifetime_from_onshell_self_energy,
    linewidth_observables,
)
from .lswt import (
    MagnonDispersion,
    MagnonSpectrum,
    solve_isotropic_lswt,
    solve_isotropic_lswt_energies,
    uniform_fractional_mesh,
)
from .mesh import MagnonMeshCache, build_magnon_mesh_cache
from .model import (
    ExchangeCapability,
    ExchangeConvention,
    ExchangeModel,
    ExchangeRepresentation,
    ExchangeScreeningReport,
    ExchangeSpinNormalization,
    MagneticConfiguration,
    MagneticOrder,
    SingleIonAnisotropy,
)
from .output import (
    DISPERSION_OUTPUT_SCHEMA_VERSION,
    LIFETIME_OUTPUT_SCHEMA_VERSION,
    write_dispersion_npz,
    write_dispersion_plot,
    write_lifetime_npz,
)
from .parallel import (
    CollectiveExecutionError,
    DistributedArrayResult,
    IndexPartition,
    balanced_partition,
    distributed_array_map,
)
from .phonon import (
    FrequencyFloorProvenance,
    PhononCache,
    PhononInputError,
    PhononMassUnit,
    ZeroPointDisplacement,
    load_phonon_cache,
    zero_point_displacements,
)
from .pipeline import (
    DistributedLifetimeGridResult,
    LifetimeAtK,
    LifetimeGridResult,
    compute_lifetime_grid,
    evaluate_scattering_problem,
)
from .screening import load_exchange_h5, screen_magnetic_configuration
from .self_energy import (
    OnShellSelfEnergyResult,
    SelfEnergyResult,
    accumulate_onshell_self_energy_diagonal_block,
    bose_occupation,
    compute_onshell_self_energy_diagonal,
    compute_retarded_self_energy,
    normalize_q_weights,
)
from .vertex import (
    MagnonPhononScatteringProblem,
    build_bare_isotropic_vertex,
    build_bare_isotropic_vertex_block,
    build_scattering_problem,
    iter_bare_isotropic_vertex_blocks,
)

__all__ = (
    "DISPERSION_OUTPUT_SCHEMA_VERSION",
    "LIFETIME_OUTPUT_SCHEMA_VERSION",
    "CollectiveExecutionError",
    "DerivativeASRPolicy",
    "DistributedArrayResult",
    "DistributedLifetimeGridResult",
    "DistributedMagnonDispersion",
    "ExchangeCapability",
    "ExchangeConvention",
    "ExchangeDerivativeModel",
    "ExchangeDerivativeReport",
    "ExchangeModel",
    "ExchangeRepresentation",
    "ExchangeScreeningReport",
    "ExchangeSpinNormalization",
    "FrequencyFloorProvenance",
    "IndexPartition",
    "LifetimeAtK",
    "LifetimeGridResult",
    "LifetimeResult",
    "MagneticConfiguration",
    "MagneticOrder",
    "MagnonDispersion",
    "MagnonDispersionResult",
    "MagnonKPath",
    "MagnonMeshCache",
    "MagnonPhononScatteringProblem",
    "MagnonSpectrum",
    "MagphDispersionRequest",
    "MagphInputError",
    "MagphLifetimeRequest",
    "MagphRunResult",
    "ModeResolvedExchangeDerivative",
    "OnShellSelfEnergyResult",
    "PhononCache",
    "PhononInputError",
    "PhononMassUnit",
    "RestartMode",
    "SelfEnergyResult",
    "SingleIonAnisotropy",
    "SingleIonAnisotropyInput",
    "ZeroPointDisplacement",
    "accumulate_onshell_self_energy_diagonal_block",
    "balanced_partition",
    "bose_occupation",
    "build_bare_isotropic_vertex",
    "build_bare_isotropic_vertex_block",
    "build_dispersion_request",
    "build_lifetime_request",
    "build_magnon_mesh_cache",
    "build_mode_resolved_isotropic_derivative",
    "build_mode_resolved_isotropic_derivative_distributed",
    "build_scattering_problem",
    "build_wannier90_kpath",
    "compute_lifetime",
    "compute_lifetime_grid",
    "compute_magnon_dispersion",
    "compute_onshell_self_energy_diagonal",
    "compute_retarded_self_energy",
    "distributed_array_map",
    "evaluate_scattering_problem",
    "iter_bare_isotropic_vertex_blocks",
    "lifetime_from_onshell_self_energy",
    "linewidth_observables",
    "load_exchange_derivative_h5",
    "load_exchange_h5",
    "load_phonon_cache",
    "normalize_q_weights",
    "run_dispersion",
    "run_lifetime",
    "screen_magnetic_configuration",
    "solve_isotropic_lswt",
    "solve_isotropic_lswt_energies",
    "uniform_fractional_mesh",
    "write_dispersion_npz",
    "write_dispersion_plot",
    "write_lifetime_npz",
    "zero_point_displacements",
)
