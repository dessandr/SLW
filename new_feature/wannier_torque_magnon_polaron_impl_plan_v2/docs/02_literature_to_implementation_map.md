# Literature-to-Implementation Map

## Relativistic local spin vertex

The localized-orbital relativistic LKAG source gives the first-order exchange-field rotation

\[
\delta V^{(1)}_{\ell c}
=\frac{i}{2}[H_{{\rm XC},\ell},\sigma_c].
\]

The implementation names this matrix `Gamma_rotation`. It is identical to

\[
\Gamma_{\ell c}=-i[S_c,H_{{\rm XC},\ell}],
\qquad S_c=\sigma_c/2.
\]

## Mixed spin-displacement structure

The spin-lattice source obtains a site-diagonal coefficient with structure

\[
T_i\,\tau_{ik}\,U_k\,\tau_{ki}.
\]

The project uses the structural correspondence

| KKR object | Wannier object | Scope |
|---|---|---|
| spin-tilt matrix `T_i` | local `dH/dpi` or `dH/dtheta` | source-guided |
| scattering path `tau` | spinor Wannier `G` | project mapping |
| displacement matrix `U_k` | screened DFPT `g=dH/du` | project mapping |
| site path | `k -> k+q -> k` loop | project extension |
| q-dependent effective field | Fourier mixed Hessian | source-guided extension |

The table is a perturbative-structure map, not an operator identity.

## Standard supporting theories

- DFPT supplies screened first-order lattice perturbations.
- Wannier interpolation supplies a localized and gauge-trackable representation of `H` and `g`.
- Bosonic paraunitary diagonalization supplies external magnon eigenvectors.

## Project-specific completion

The following equations are explicitly marked `PROJECT-EXTENSION` in the source map:

- time-reversal sewing-matrix implementation in an arbitrary Wannier gauge;
- q-pair construction of a complex finite-q mixed kernel;
- direct mixed vertex in terms of `dHxc/du`;
- phonon zero-point projection of the torque kernel;
- external magnon Nambu projection and normal/anomalous block extraction.
