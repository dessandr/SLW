# TASK-009 - Optional Direct Mixed Vertex Interface

## Goal

Define but do not silently enable the `dT/du` contribution.

## Dependencies

TASK-004, TASK-005, TASK-008.

## Deliverables

Feature gate, `g_XC` contract, fixed/moving-projector policy, separate bubble/direct/total output.

## Tests

Missing `g_XC` rejection, full-g commutator rejection, direct finite difference on a synthetic exchange field, and projector-response fixture.

## Done when

The feature remains disabled for ordinary DFPT input and is validated for a controlled synthetic dataset.
