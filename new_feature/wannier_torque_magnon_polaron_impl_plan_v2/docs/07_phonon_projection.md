# Phonon Projection

## Cartesian path

For a Cartesian kernel in eV/Angstrom, define

\[
\xi_{\kappa\mu,\nu}(\mathbf q)
=\sqrt{\frac{\hbar}{2M_\kappa\omega_{\mathbf q\nu}}}
 e_{\kappa\mu,\nu}(\mathbf q).
\]

Then

\[
V_{\ell a,\nu}(\mathbf q)
=\sum_{\kappa\mu}
K_{\ell a,\kappa\mu}(\mathbf q)
\xi_{\kappa\mu,\nu}(\mathbf q).
\]

`V_pi_ph` has units of energy.

## Mode-normalized path

If upstream data already provide

\[
g_\nu=\sum_{\kappa\mu}\xi_{\kappa\mu,\nu}g_{\kappa\mu},
\]

then no additional zero-point factor is applied. The normalization enum is mandatory.

## Allowed normalization labels

- `cartesian_derivative`;
- `mass_weighted_cartesian`;
- `phonon_zero_point_mode`.

## Acoustic modes

At q=0, evaluate the rigid-translation sum rule on `K_pi_u` before dividing by `sqrt(omega)`. Exactly zero or imaginary modes are flagged; they are never hidden by a silent frequency floor.

## Rephasing

Under `e_nu(q) -> exp(i phi) e_nu(q)`, the coupling changes phase but physical spectra and subspace invariants remain unchanged.
