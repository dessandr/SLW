# TASK-013 - Physics Validation Suite

## Goal

Automate algebra, decomposition, finite differences, sum rules, gauge, subspace sensitivity, projection, and convergence reports.

## Dependencies

TASK-012.

## Tests

Each intentionally corrupted fixture triggers its intended failure. Reports are deterministic and preserve warn/fail distinctions.

## Done when

`templates/physics_validation_report.md` is filled automatically for a complete run.
