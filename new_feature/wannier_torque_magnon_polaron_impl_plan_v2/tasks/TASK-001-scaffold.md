# TASK-001 - Repository Scaffold and Convention Guards

## Goal

Create package skeleton, typed configuration, domain errors, logging, units, test runner, and deterministic run manifest.

## Deliverables

- tree in `docs/10_software_architecture.md`;
- `pyproject.toml`;
- configuration enums for gauge, normalization, exchange extraction, projection, and spin coordinate;
- source-provenance registry;
- CI command.

## Tests

Reject invalid enums, missing metadata, and complex64 core arrays. Verify deterministic manifests after excluding host/time fields.

## Done when

`wtorque inspect` validates a synthetic config and prints resolved conventions and provenance IDs.
