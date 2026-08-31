# Software Architecture

## Proposed tree

```text
slw/wtorque/
  cli.py
  config.py
  units.py
  errors.py
  basis.py
  io/
    hdf5.py
    wannier.py
    dfpt.py
    phonon.py
    magnon.py
  model/
    spinor_wannier.py
    time_reversal.py
    exchange_field.py
    magnetic_subspace.py
    local_projection.py
    local_frames.py
  gauge/
    conventions.py
    kq_map.py
    atomic_gauge.py
  green/
    provider.py
    real_axis.py
    contour.py
  torque/
    vertices.py
    kernel.py
    qpair.py
    direct_vertex.py
  projection/
    phonon.py
    magnon.py
    polaron.py
  validation/
    algebra.py
    time_reversal.py
    finite_difference.py
    sum_rules.py
    subspace_sensitivity.py
    symmetry.py
    rephasing.py
    reports.py
  parallel/
    scheduler.py
    mpi.py
```

## Key interfaces

```python
class ExchangeFieldExtractor(Protocol):
    def split(self, model: "SpinorWannierModel") -> tuple[ArrayC, ArrayC]: ...

class MagneticSubspace:
    projectors: ArrayC
    labels: list[list[str]]

class LocalProjector(Protocol):
    def localize(self, x: ArrayC, ell: int) -> ArrayC: ...

class TorqueVertexProvider(Protocol):
    coordinate_type: str
    def vertex_batch(self, k_red: ArrayR, q_red: ArrayR) -> ArrayC:
        """Return [nmag,2,nw,nw] for the specified k,q pair."""

class DFPTProvider(Protocol):
    def g_cart_chunk(self, iq: int, ik: int, sl: slice) -> ArrayC: ...
    def normalization(self) -> str: ...

class EnergyIntegrator(Protocol):
    def nodes_and_weights(self): ...
    def finalize_q_pair(self, aq: ArrayC, amq: ArrayC) -> ArrayC: ...
```

## Dependency direction

- I/O validates metadata and canonicalizes basis/gauge.
- Exchange extraction precedes magnetic subspace and local partition.
- The electronic kernel depends on model, vertices, gauge, Green functions, and DFPT only.
- Phonon and magnon projection operate on stored kernels, not on electronic Green functions.
- Validation may inspect all public interfaces; production modules never import test fixtures.

## Domain errors

- `GaugeMismatchError`
- `TimeReversalMetadataError`
- `ExchangeDecompositionError`
- `MagneticSubspaceError`
- `LocalProjectionError`
- `NormalizationError`
- `MeshCommensurabilityError`
- `ParaunitarityError`
- `HermiticityError`
- `IncompleteRunError`

## Logging

Every run records code version, input hashes, source-provenance IDs, resolved orbital masks, projection strategy, gauges, normalization, dimensions, memory estimate, integration settings, per-q timing, and validation residuals.
