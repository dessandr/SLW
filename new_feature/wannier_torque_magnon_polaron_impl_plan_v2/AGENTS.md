# AGENTS.md

## Mission

Build a tested Python package, provisionally named `wtorque`, that computes the spinor-Wannier mixed response, projects it onto phonons and externally supplied magnons, and writes restartable HDF5 output.

## Non-negotiable physics rules

1. **Magnons are external.** Production code must not add an exchange-extraction or magnon-diagonalization pipeline. Toy diagonalizations are allowed only in tests.
2. **Separate coordinate types.** `transverse_direction` means `pi_{ell a}`; `rotation_angle` means `theta_{ell c}`. Never store one under the name of the other.
3. **Rotate only the exchange field.** Never construct the physical torque vertex as `-i[S,H_full]`.
4. **Keep SOC in propagation.** The full spinor Hamiltonian, including SOC, defines every Green function.
5. **Use an explicit magnetic subspace.** A run must declare which Wannier orbitals represent localized magnetic degrees of freedom.
6. **Use local partition by default.** The default is `0.5*(P_ell X + X P_ell)`. `P_ell X P_ell` is a diagnostic, not the production default.
7. **MVP is bubble-only.** The direct mixed vertex is disabled unless a validated `dHxc/du` dataset is supplied.
8. **Use complex128 throughout the physics core.** No silent downcast.
9. **No implicit gauge conversion.** Bloch gauge, Fourier signs, reciprocal wrapping, and sewing matrices are metadata, not guesses.
10. **No hidden normalization.** Cartesian, mass-weighted, and zero-point mode-normalized DFPT data are distinct input types.
11. **No extra bosonic metric in coefficient transforms.** Paraunitarity uses `Sigma`; the linear operator transform uses the ordinary congruence `T_m^dagger V T_p`.
12. **No silent sign repair.** A sign mismatch is a failed convention test.
13. **No premature optimization.** The reference CPU implementation and finite-difference tests must pass first.

## Required validation gates

A production run cannot be marked valid unless all mandatory gates pass:

- Hermiticity of `H`, local exchange fields, torque vertices, and q-paired perturbations.
- Correct time-reversal reconstruction of `H_TRS` and `H_TRB` when that extraction route is used.
- Local-partition closure on the declared magnetic support.
- Central-difference check of each torque vertex.
- Longitudinal rotation vertex equal to zero within tolerance.
- Mixed electronic finite difference on a linearized Wannier model.
- `K(-q)=K(q)^*` after explicit gauge conversion.
- Rigid-translation acoustic sum rule at q=0.
- Magnetic-subspace and local-projection sensitivity report.
- Phonon and magnon rephasing invariance of physical spectra.
- Magnon paraunitarity and Nambu-order checks.

## Source discipline

Every derived equation in code documentation must point to one of:

- a primary source ID from `references/SOURCE_PROVENANCE.md`;
- a standard identity listed there;
- `PROJECT-EXTENSION`, with the assumptions stated explicitly.

Do not describe a project extension as a formula already established by either supplied paper.

## Code organization

- Put basis and metadata validation at I/O boundaries.
- Keep exchange-field extraction separate from site partition.
- Keep electronic kernels separate from phonon/magnon projection.
- Write exact array shapes and units in docstrings.
- Use typed domain exceptions rather than bare assertions for user input.
- Hash all source files and store a resolved run manifest.

## Completion report for each task

A task response must state:

1. files changed;
2. equations and source IDs implemented;
3. tests added and their physical meaning;
4. commands executed;
5. unresolved assumptions or blocked inputs.
