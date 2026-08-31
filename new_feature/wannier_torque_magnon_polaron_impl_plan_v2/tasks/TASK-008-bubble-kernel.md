# TASK-008 - Bubble Kernel T G g G

## Goal

Compute complex retarded loops and finalize physical q-paired `K_pi_u`.

## Dependencies

TASK-004, TASK-005, TASK-006, TASK-007.

## Deliverables

Batched contraction, q-pair finalizer, chunking, restartable writer, fixed-chemical-potential metadata, and direct-term flag false by default.

## Tests

Analytic SOC toy model, mixed finite difference, spin-rotation-invariant SOC-off limit, chunk invariance, q conjugation, and a complex noncentrosymmetric finite-q fixture.

## Done when

No general-q path calls a single-loop `np.imag` finalizer.
