"""Typed domain errors used at wtorque input and physics boundaries."""


class WTorqueError(RuntimeError):
    """Base class for domain failures reported to users."""


class ConfigurationError(WTorqueError, ValueError):
    pass


class GaugeMismatchError(WTorqueError):
    pass


class TimeReversalMetadataError(WTorqueError):
    pass


class ExchangeDecompositionError(WTorqueError):
    pass


class MagneticSubspaceError(WTorqueError):
    pass


class LocalProjectionError(WTorqueError):
    pass


class NormalizationError(WTorqueError):
    pass


class MeshCommensurabilityError(WTorqueError, ValueError):
    pass


class ParaunitarityError(WTorqueError):
    pass


class HermiticityError(WTorqueError):
    pass


class IncompleteRunError(WTorqueError):
    pass


__all__ = [name for name in globals() if name.endswith("Error")]

