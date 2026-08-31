# Milestones and Acceptance Criteria

## M0 - conventions and provenance

- All coordinate, gauge, source, and normalization labels frozen.
- LaTeX manuscript and source map compile without unresolved references.

## M1 - electronic model and exchange field

- Spinor Wannier I/O passes Hermiticity and wrapping tests.
- At least one exchange-field extraction route passes reconstruction and time-reversal tests.

## M2 - magnetic subspace and torque vertices

- Explicit orbital masks and local partition implemented.
- Vertex finite differences, longitudinal test, and projection sensitivity pass.

## M3 - q=0 electronic mixed response

- Bubble kernel matches analytic and numerical mixed derivatives.
- Spin-rotation-invariant SOC-off toy result vanishes.

## M4 - finite-q response

- q-pair completion, perturbation Hermiticity, and complex toy model pass.
- Rigid-translation sum rule passes.

## M5 - phonon projection

- Cartesian and mode-normalized paths agree.
- No zero-point double counting.

## M6 - external magnon projection

- FM and two-sublattice AFM fixtures pass.
- normal/anomalous blocks and rephasing invariance pass.

## M7 - restartable production workflow

- CLI, HDF5, q-level restart, merge, and validation report complete.

## M8 - MnTe reference run

- full input audit;
- q=0 plus finite-q validation;
- convergence and symmetry report;
- coupling output directly consumed by the existing magnon-polaron code.

## M9 - extensions

- direct mixed vertex;
- fixed-particle-number correction;
- spectral backend;
- tensor decomposition;
- MPI/GPU acceleration.
