"""Native public API for exchange-based magnon--phonon observables.

The array kernels and strict data contracts here are independent of the
quarantined compatibility drivers under :mod:`slw.magph.legacy`.  The current
``slw_magph.x`` registry remains compatibility-backed until native LSWT and
vertex construction complete their FM/AFM parity gates.
"""

from .config import MagphInputError, MagphLifetimeRequest, build_lifetime_request
from .coupling import (
    ModeResolvedExchangeDerivative,
    build_mode_resolved_isotropic_derivative,
)
from .derivative import (
    DerivativeASRPolicy,
    ExchangeDerivativeModel,
    ExchangeDerivativeReport,
    load_exchange_derivative_h5,
)
from .engine import MagphRunResult, run_lifetime
from .lifetime import (
    LifetimeResult,
    compute_lifetime,
    lifetime_from_onshell_self_energy,
    linewidth_observables,
)
from .lswt import MagnonSpectrum, solve_isotropic_lswt, uniform_fractional_mesh
from .model import (
    ExchangeCapability,
    ExchangeConvention,
    ExchangeModel,
    ExchangeRepresentation,
    ExchangeScreeningReport,
    ExchangeSpinNormalization,
    MagneticConfiguration,
    MagneticOrder,
)
from .output import LIFETIME_OUTPUT_SCHEMA_VERSION, write_lifetime_npz
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
    bose_occupation,
    compute_onshell_self_energy_diagonal,
    compute_retarded_self_energy,
    normalize_q_weights,
)
from .vertex import (
    MagnonPhononScatteringProblem,
    build_bare_isotropic_vertex,
    build_scattering_problem,
)

__all__ = (
    "LIFETIME_OUTPUT_SCHEMA_VERSION",
    "CollectiveExecutionError",
    "DerivativeASRPolicy",
    "DistributedArrayResult",
    "DistributedLifetimeGridResult",
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
    "MagnonPhononScatteringProblem",
    "MagnonSpectrum",
    "MagphInputError",
    "MagphLifetimeRequest",
    "MagphRunResult",
    "ModeResolvedExchangeDerivative",
    "OnShellSelfEnergyResult",
    "PhononCache",
    "PhononInputError",
    "PhononMassUnit",
    "SelfEnergyResult",
    "ZeroPointDisplacement",
    "balanced_partition",
    "bose_occupation",
    "build_bare_isotropic_vertex",
    "build_lifetime_request",
    "build_mode_resolved_isotropic_derivative",
    "build_scattering_problem",
    "compute_lifetime",
    "compute_lifetime_grid",
    "compute_onshell_self_energy_diagonal",
    "compute_retarded_self_energy",
    "distributed_array_map",
    "evaluate_scattering_problem",
    "lifetime_from_onshell_self_energy",
    "linewidth_observables",
    "load_exchange_derivative_h5",
    "load_exchange_h5",
    "load_phonon_cache",
    "normalize_q_weights",
    "run_lifetime",
    "screen_magnetic_configuration",
    "solve_isotropic_lswt",
    "uniform_fractional_mesh",
    "write_lifetime_npz",
    "zero_point_displacements",
)
