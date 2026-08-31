# External Magnon Projection and Magnon-Polaron Assembly

## Required external data

For each q, provide:

- positive magnon energies;
- paraunitary matrix `T_m(q)`;
- spin lengths `S_ell`;
- local frames identical to the electronic calculation;
- Nambu ordering and Fourier gauge;
- paraunitarity residual from the upstream solver.

Magnon energies alone are insufficient.

## Magnon Nambu convention

\[
\Psi_m(\mathbf q)
=\begin{pmatrix}\mathbf a_{\mathbf q}\\
\mathbf a_{-\mathbf q}^\dagger\end{pmatrix}
=T_m(\mathbf q)\Gamma_m(\mathbf q),
\]

\[
T_m^\dagger\Sigma_mT_m=\Sigma_m,
\qquad
\Sigma_m=\operatorname{diag}(I,-I).
\]

## Linear spin-coordinate map

\[
\pi_{\ell1}(-\mathbf q)
=\frac{a_{\ell,-\mathbf q}+a_{\ell,\mathbf q}^\dagger}{\sqrt{2S_\ell}},
\]

\[
\pi_{\ell2}(-\mathbf q)
=\frac{-i a_{\ell,-\mathbf q}+i a_{\ell,\mathbf q}^\dagger}{\sqrt{2S_\ell}}.
\]

Define `C_m` by

\[
\boldsymbol\pi^T(-\mathbf q)=\Psi_m^\dagger(\mathbf q)C_m.
\]

## Coupling transform

For phonon Nambu spinor `Psi_p` and coordinate map `C_p`,

\[
V_{mp}(\mathbf q)=C_m V_{\pi\nu}(\mathbf q) C_p,
\]

and in the quasiparticle basis

\[
\widetilde V_{mp}(\mathbf q)
=T_m(\mathbf q)^\dagger V_{mp}(\mathbf q)T_p(\mathbf q).
\]

No bosonic metric appears in this coefficient congruence. `Sigma` is used for the eigenproblem and paraunitarity check, not inserted again here.

## Blocks

Export separately:

- `g_mp_normal`: coefficient of `alpha_q^dagger b_q`;
- `g_mp_anomalous`: coefficient of `alpha_q^dagger b_-q^dagger`;
- their Hermitian partners implied by the full Hamiltonian.

For a one-sublattice ferromagnet with no anomalous magnon mixing,

\[
g_\nu^{mp}=\frac{V_{1\nu}+iV_{2\nu}}{\sqrt{2S}}.
\]

## Polaron assembly

Near an isolated crossing, the rotating-wave block is

\[
\mathcal H_{\rm RWA}(\mathbf q)=
\begin{pmatrix}
\Omega_m & g^{mp}\\
(g^{mp})^\dagger & \Omega_{ph}
\end{pmatrix}.
\]

The existing magnon-polaron code may instead consume normal and anomalous blocks for a full bosonic BdG assembly.
