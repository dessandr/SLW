# Codex Session Prompt

Implement exactly one task from this package.

## Task

`tasks/<selected-task-file>.md`

## Required reading

1. `AGENTS.md`
2. `docs/01_notation_and_conventions.md`
3. `references/SOURCE_PROVENANCE.md`
4. all task dependencies
5. target task
6. related equation IDs in `docs/17_equation_index.md`

## Constraints

- Do not change physics conventions.
- Add tests in the same change.
- Preserve complex128.
- Reject missing gauge, time-reversal, projection, or normalization metadata.
- Do not recompute magnons.
- Do not rotate the full Hamiltonian.
- Do not add an extra bosonic metric in coefficient transforms.
- Mark project extensions explicitly.

## Response format

1. Files changed.
2. Equation IDs and source IDs implemented.
3. Design summary.
4. Tests and physical meaning.
5. Commands executed.
6. Remaining risks.
