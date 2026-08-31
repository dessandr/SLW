# Scope and Source Boundary

## Primary source A: relativistic localized-orbital LKAG

Martinez-Carracedo *et al.* establish in a nonorthogonal PAO basis:

- a Hermitian local operator that assigns half of cross-boundary matrix blocks to each local sector;
- extraction of the time-reversal-broken exchange field from the spinor Kohn-Sham Hamiltonian;
- the requirement that only the exchange field, not the full SOC Hamiltonian, be rotated in the DFT-to-spin mapping;
- first- and second-order local spin-rotation vertices;
- sensitivity to the selected localized magnetic orbitals;
- sum rules and rotation-axis tests.

In the orthonormal Wannier limit, the local operator becomes

\[
\mathcal L_\ell[X]=\frac12(P_\ell X+XP_\ell).
\]

## Primary source B: spin-lattice torque response

Mankovsky *et al.* establish in relativistic KKR:

- independent spin-tilt and displacement perturbations;
- a mixed Green-function term with structure `T tau U tau`;
- a site-diagonal spin-lattice coefficient obtained from a mixed free-energy derivative;
- its interpretation as displacement-induced effective field and physical torque;
- a phonon-like finite-q displacement pattern.

## Project extension

This package proposes the following steps, which are not jointly implemented in either primary source:

1. Use an orthonormal spinor Wannier Hamiltonian and Green function.
2. Use a screened DFPT Hamiltonian derivative `g(k,q)=dH/du(q)` as the lattice vertex.
3. Evaluate a gauge-controlled finite-q momentum loop.
4. Construct the complex Fourier kernel through q-pair retarded/advanced completion.
5. Project the Cartesian kernel onto phonon eigenvectors.
6. Project the transverse-spin kernel onto externally supplied magnon paraunitary eigenvectors.
7. Export normal and anomalous magnon-phonon coupling blocks.

## Approximation hierarchy

### MVP

\[
K^{\mathrm{MVP}}=K^{\mathrm{bubble}},
\qquad
\partial \mathcal T/\partial u=0.
\]

This follows the independent-perturbation structure of the spin-lattice source.

### Total-derivative extension

\[
K=K^{\mathrm{direct}}+K^{\mathrm{bubble}},
\]

where `K_direct` requires an exchange-field-resolved displacement derivative. It is a project extension and must be reported separately.

## Claim boundary

The implementation may claim a screened-DFPT Wannier realization of the stated mixed-response approximation after validation. It must not claim exact operator equivalence to KKR, a unique decomposition into spin-model tensors, or completeness of the bubble-only response.
