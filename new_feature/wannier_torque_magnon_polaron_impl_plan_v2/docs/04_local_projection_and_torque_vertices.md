# Local Projection, Magnetic Subspace, and Torque Vertices

## Magnetic Wannier subspace

Each production input declares disjoint projectors `P_ell` onto localized magnetic Wannier orbitals. Define

\[
P_M=\sum_\ell P_\ell.
\]

The selected subspace must be auditable by orbital label, center, spread, atom, and projection character.

## Hermitian local partition

The default local exchange field is

\[
H_{{\rm XC},\ell}=\mathcal L_\ell[H_{\rm XC}],
\qquad
\mathcal L_\ell[X]=\frac12(P_\ell X+XP_\ell).
\]

The local fields satisfy

\[
\sum_\ell H_{{\rm XC},\ell}
=\frac12(P_MH_{\rm XC}+H_{\rm XC}P_M),
\]

which is the exchange-field part touching the selected magnetic subspace. The residual outside this support is stored explicitly.

## Diagnostic projections

Implement:

1. `local_partition`: default `0.5*(P X + X P)`;
2. `onsite_only`: `P X P`, for comparison only;
3. `user_supplied`: validated local matrices from upstream;
4. `full_atom`: sensitivity test including all Wannier functions assigned to the magnetic atom.

Never choose a strategy from orbital names without writing the resolved orbital list to output.

## Transverse-direction vertex

For the rigid local exchange-field model,

\[
\mathcal T_{\ell a}
=\left.\frac{\partial H}{\partial\pi_{\ell a}}\right|_0.
\]

If

\[
H_{{\rm XC},\ell}
=\frac12\Delta_\ell\otimes
(\mathbf n_\ell\cdot\boldsymbol\sigma),
\]

then

\[
\mathcal T_{\ell a}
=\frac12\Delta_\ell\otimes
(\mathbf t_{\ell a}\cdot\boldsymbol\sigma).
\]

## Rotation-angle vertex

For rotation axis `c`,

\[
\Gamma_{\ell c}
=\frac{i}{2}
[H_{{\rm XC},\ell},\mathbf u_{\ell c}\cdot\boldsymbol\sigma]
=-i[S_{\ell c},H_{{\rm XC},\ell}].
\]

The stored coordinate convention must be explicit. For local transverse rotation axes,

\[
\Gamma_{\ell1}=-\mathcal T_{\ell2},
\qquad
\Gamma_{\ell2}=+\mathcal T_{\ell1}.
\]

## Reciprocal finite-q vertex

The local partition is a local real-space operation and is generally not equivalent to one constant onsite matrix. Define the q-dependent site selector

\[
P_\ell(\mathbf q)
=\left\langle w_{\mathbf k+\mathbf q}\middle|
\sum_{\mathbf R}e^{i\mathbf q\cdot(\mathbf R+\boldsymbol\tau_\ell)}P_{\mathbf R\ell}
\middle|w_{\mathbf k}\right\rangle.
\]

For diagonal orbital projectors in the atomic-position gauge,

\[
[P_\ell(\mathbf q)]_{nm}
=\delta_{nm}\,\delta_{s(n),\ell}
 e^{i\mathbf q\cdot(\boldsymbol\tau_\ell-\boldsymbol\tau_n)}.
\]

Let `Gamma_global,c(k)` be the uniform exchange-field rotation about the local axis chosen for site `ell`. The Fourier representation of the Hermitian local partition is

\[
\Gamma_{\ell c}(\mathbf k+\mathbf q,\mathbf k)
=\frac12\left[
P_\ell(\mathbf q)\Gamma_{c}^{\rm global}(\mathbf k)
+\Gamma_{c}^{\rm global}(\mathbf k+\mathbf q)P_\ell(\mathbf q)
\right].
\]

The transverse-direction vertex follows from the tangent-plane coordinate map. Only in the strictly onsite atomic-center limit does the vertex reduce to a k- and q-independent matrix. The production provider must therefore expose `vertex(ell,a,k,q)` or a batched equivalent.

## What remains fixed

The operation rotates `H_XC` only. `H_TRS`, including SOC and the lattice potential, remains unchanged. SOC enters the mixed response through the full Green function.

## Mandatory tests

- local frames are right-handed and orthonormal;
- each local field and vertex is Hermitian;
- a central finite rotation reproduces the analytic vertex;
- the longitudinal commutator vanishes;
- changing SOC changes `G` but not the vertex built from fixed `H_XC`;
- `local_partition` closure and residual support are reported;
- `d-only`, `full-atom`, and `onsite-only` sensitivity is quantified.
