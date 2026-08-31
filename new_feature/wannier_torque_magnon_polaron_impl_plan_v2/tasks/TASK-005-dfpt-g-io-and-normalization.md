# TASK-005 - DFPT g I/O, Spinor Lifting, and Normalization

## Goal

Load Cartesian or mode-resolved DFPT perturbations and convert to canonical spinor, units, and gauge.

## Dependencies

TASK-001, TASK-002.

## Deliverables

Native-spinor, collinear-spin-dependent, and spin-scalar adapters; normalization enum; chunk iterator; optional `g_XC` reader.

## Tests

Hand-built lifting, no double zero-point factor, undeclared normalization failure, and dimension/basis checks.

## Done when

Cartesian and precontracted mode paths reproduce the same synthetic perturbation.
