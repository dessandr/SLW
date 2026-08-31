# MnTe End-to-End Reference Workflow

## Purpose

Use NiAs-type A-type antiferromagnetic MnTe as the first production system after all toy tests pass. Magnons remain external.

## Input audit

Record:

- spinor Wannier Hamiltonian and interpolation error;
- exchange-field extraction route;
- Mn-centered magnetic orbital masks and sensitivity variants;
- local frames for the two antiferromagnetic sublattices;
- screened DFPT `g(k,q)` normalization and gauge;
- phonon frequencies/eigenvectors;
- external magnon energies and paraunitary matrices;
- common q path or commensurate mesh.

## Staged calculation

1. q=0 vertex and decomposition tests.
2. q=0 mixed finite difference in the linearized Wannier model.
3. one generic finite-q point and its negative.
4. one symmetry-related set to diagnose rotational residuals.
5. target q path or mesh.
6. phonon projection.
7. external magnon projection and polaron input export.

## Mandatory comparisons

- `local_partition` versus `onsite_only`;
- Mn d-only versus all Mn-centered Wannier functions;
- SOC on versus spin-rotation-invariant reference;
- q versus `-q`;
- Cartesian versus mode-normalized phonon path if both are available;
- independently rephased phonon and magnon eigenvectors.

## Convergence

Vary k mesh, energy grid/contour, broadening, Wannier window, orbital mask, and q mesh. Quote the stability of `|g_mp|` near the target crossing, not only `K_pi_u` elements.

## Completion criterion

A single resolved config reproduces the run, all mandatory tests pass or have documented warnings, and the external magnon-polaron code reads the normal/anomalous coupling datasets without conversion by hand.
