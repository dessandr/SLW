# TASK-004 - Magnetic Subspace, Local Projection, and Torque Vertices

## Goal

Build explicit magnetic projectors, default local partition, local frames, and both coordinate types of torque vertex.

## Dependencies

TASK-003.

## Deliverables

Magnetic-subspace audit, `local_partition`, `onsite_only`, `user_supplied`, q-dependent site selectors, a `vertex_batch(k,q)` provider, and finite-rotation helper.

## Tests

Projection closure, Hermiticity, central finite rotation, longitudinal zero, tangent-coordinate conversion, SOC isolation, finite-q phase covariance, onsite-limit reduction, and subspace sensitivity fixtures.

## Done when

All vertex and local-partition residuals pass and output lists resolved orbitals.
