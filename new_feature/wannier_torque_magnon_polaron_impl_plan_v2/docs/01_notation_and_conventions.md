# Notation and Conventions

This document is normative.

## Indices

- `ell`: magnetic site or sublattice in the primitive cell.
- `a=1,2`: local transverse spin-direction coordinate.
- `c=1,2,3`: local rotation-axis coordinate.
- `kappa`: atom in the primitive cell.
- `mu=x,y,z`: Cartesian displacement direction.
- `n,m`: spinor Wannier indices.
- `nu`: phonon branch.
- `r`: positive-frequency magnon branch.
- `k,q`: reciprocal-space points, always tagged as reduced or Cartesian.

## Local spin frame

For each magnetic site,

\[
(\mathbf t_{\ell1},\mathbf t_{\ell2},\mathbf n_\ell),
\qquad
\mathbf t_{\ell1}\times\mathbf t_{\ell2}=\mathbf n_\ell.
\]

A small direction fluctuation is

\[
\mathbf e_\ell
=\mathbf n_\ell+\pi_{\ell1}\mathbf t_{\ell1}
+\pi_{\ell2}\mathbf t_{\ell2}+O(\pi^2).
\]

The stored electronic kernel uses `pi`, not a physical torque-vector component.

## Coordinate type

Every spin vertex and kernel carries one of:

- `transverse_direction`: derivative with respect to `pi_{ell a}`;
- `rotation_angle`: derivative with respect to `theta_{ell c}`.

For a right-handed local frame and rotations around `t1,t2`,

\[
\Gamma_{\ell1}=-\mathcal T_{\ell2},
\qquad
\Gamma_{\ell2}=+\mathcal T_{\ell1}.
\]

## Atomic-position Bloch gauge

\[
|w_{n\mathbf k}\rangle
=\frac1{\sqrt N}\sum_{\mathbf R}
 e^{i\mathbf k\cdot(\mathbf R+\boldsymbol\tau_n)}
|w_{n\mathbf R}\rangle.
\]

With

\[
H_{nm}(\mathbf R)=
\langle w_{n\mathbf 0}|H|w_{m\mathbf R}\rangle,
\]

the Hamiltonian matrix is

\[
H_{nm}(\mathbf k)=\sum_{\mathbf R}
 e^{i\mathbf k\cdot(\mathbf R+\boldsymbol\tau_m-\boldsymbol\tau_n)}
H_{nm}(\mathbf R).
\]

A file using the cell gauge must be converted explicitly; the package never combines the atomic-position Bloch sum with a center-free Fourier transform.

Fields use

\[
x_s(\mathbf q)=\frac1{\sqrt N}\sum_{\mathbf R}
 e^{-i\mathbf q\cdot(\mathbf R+\boldsymbol\tau_s)}x_{\mathbf Rs}.
\]

This convention applies to atomic displacements and transverse spin coordinates.

## Reciprocal wrapping

If

\[
\mathbf k+\mathbf q=\overline{\mathbf k+\mathbf q}+\mathbf G,
\]

then

\[
D_{\mathbf G}=\operatorname{diag}_n
 e^{i\mathbf G\cdot\boldsymbol\tau_n}\otimes I_{\rm spin},
\]

and

\[
H(\mathbf k+\mathbf G)=D_{\mathbf G}^\dagger H(\mathbf k)D_{\mathbf G}.
\]

The final-state index of `g(k,q)` follows the same conversion.

## Green function

\[
G^R_{\mathbf k}(\varepsilon)
=[(\varepsilon+i\eta)I-H(\mathbf k)]^{-1}.
\]

All core matrices use complex128.

## Free-energy convention

At fixed chemical potential,

\[
\frac{\partial\mathcal F}{\partial\lambda}
=-\frac1\pi\operatorname{Im}\int d\varepsilon\,f(\varepsilon)
\operatorname{Tr}[V_\lambda G^R].
\]

A fixed-particle-number correction is not part of the MVP.

## Stored kernel and physical torque

\[
K_{\ell a,\kappa\mu}(\mathbf q)
=\frac{\partial^2\mathcal F}
{\partial\pi_{\ell a}(-\mathbf q)\partial u_{\kappa\mu}(\mathbf q)}.
\]

The induced tangent effective field is

\[
H^{\rm eff}_{\ell a}=-K_{\ell a,\kappa\mu}u_{\kappa\mu},
\]

and the physical torque is

\[
\boldsymbol\tau_\ell=\mathbf n_\ell\times\mathbf H^{\rm eff}_\ell.
\]

## Units

- `H`, mode-normalized `g`, `V_pi_ph`, `g_mp`: eV.
- Cartesian `g=dH/du`, `K_pi_u`: eV/Angstrom.
- `pi`: dimensionless.
- Internal phonon energy: eV.
- Input mass: atomic mass unit with explicit conversion.

## Reality condition

After all gauge conversions,

\[
K(-\mathbf q)=K(\mathbf q)^*.
\]
