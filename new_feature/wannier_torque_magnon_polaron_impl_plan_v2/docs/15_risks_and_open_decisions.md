# Risks and Open Decisions

## Magnetic subspace

The relativistic LKAG source shows strong dependence on which orbitals are treated as localized magnetic degrees of freedom. The production run must compare multiple explicit subspaces and report the spread.

## Exchange-field extraction

A naive spin-block difference is insufficient for a general SOC spinor gauge. Prefer explicit upstream `H_XC`, otherwise require a valid time-reversal sewing matrix.

## Local projection

The site-resolved kernel depends on the local partition. Use the Hermitian local partition by default and retain onsite-only values as a diagnostic.

## Collinear DFPT lift

A lifted `g_up/g_dn` omits native spin-flip terms and often `dH_SOC/du`. Output must state that SOC acts through propagation while the lattice vertex is approximate.

## Direct mixed vertex

`dT/du` may matter. It cannot be built from the full `g` without rotating SOC. Enable only with `g_XC` and an explicit projector-derivative policy.

## Wannier projector motion

If the magnetic Wannier subspace changes with displacement, fixed projectors omit Pulay/gauge-response terms. The MVP fixes the reference projectors and labels this approximation.

## Fixed chemical potential

Metallic fixed-particle-number corrections may matter. The MVP uses fixed chemical potential consistently.

## q-space phases

Electronic, phonon, and magnon codes may use different gauges. Mandatory adapters and rephasing tests are required.

## Acoustic normalization

`1/sqrt(omega)` amplifies translation-sum-rule errors. Diagnose Cartesian kernels first.

## Double counting

The total torque kernel can contain effective intersite anisotropic contributions. Do not add a separate relativistic `dJ/du` linear coupling without a reconstruction/subtraction test.

## Open production decisions

- exact upstream source of `H_XC`;
- exact magnetic orbital masks for each material;
- whether `g_up/g_dn` share one Wannier gauge;
- magnon Fourier and local-frame convention;
- phonon mass normalization;
- real-axis versus inherited contour integration;
- treatment of projector motion in a later direct term.
