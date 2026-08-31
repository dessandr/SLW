# Validation and Test Matrix

## A. Algebra and decomposition

- `H(k)=H(k)^dagger`.
- `H_TRS` is time-reversal even.
- `H_XC` is time-reversal odd.
- decomposition reconstructs `H`.
- local frames are orthonormal and right-handed.
- local fields and torque vertices are Hermitian.
- local-partition closure holds on magnetic support.
- longitudinal rotation vertex vanishes.

## B. Vertex finite differences

For each site and local axis, compare the analytic vertex against a central finite rotation of `H_XC,ell`. Test both transverse-direction and rotation-angle coordinates and their tangent-plane conversion.

## C. Mixed electronic finite difference

Within a linearized Wannier model, evaluate

\[
K^{FD}=\frac{\mathcal F(+\delta\pi,+\delta u)
-\mathcal F(+\delta\pi,-\delta u)
-\mathcal F(-\delta\pi,+\delta u)
+\mathcal F(-\delta\pi,-\delta u)}{4\delta\pi\delta u}.
\]

Compare against the analytic bubble under the same fixed-vertex approximation.

## D. SOC isolation

- Scaling SOC must not alter a vertex built from a fixed exchange field.
- It may alter the Green function and final response.
- For a collinear stationary spin-rotation-invariant toy model, the linear transverse torque kernel must vanish when SOC and all anisotropic spin terms are removed.

## E. Gauge and q symmetry

- atomic-gauge periodicity under `k -> k+G`;
- perturbation Hermiticity between q and `-q`;
- `K(-q)=K(q)^*`;
- failure when wrapping or sublattice phases are intentionally omitted;
- noncentrosymmetric toy model demonstrating a genuinely complex `K(q)`.

## F. Translation

At q=0, a uniform displacement of every atom gives zero Cartesian kernel within tolerance. Report the residual before phonon normalization.

## G. Magnetic-subspace sensitivity

Compare at least:

- localized magnetic d-like orbitals;
- all Wannier functions assigned to each magnetic atom;
- onsite-only projection;
- local partition.

A large spread is a warning about the DFT-to-spin mapping, not a numerical error to hide.

## H. Phonon and magnon projections

- Cartesian-first and mode-first phonon paths agree.
- Phonon and magnon rephasing leave physical spectra invariant.
- Degenerate-subspace singular values are invariant.
- Magnon paraunitarity passes.
- A test deliberately inserting an extra metric fails.

## I. Convergence

Report k mesh, energy/contour grid, broadening, Wannier window, magnetic subspace, and perturbation chunk size. For MnTe, also report the residual of expected rotational relations.
