# Wannier Spin-Torque to Magnon-Polaron Implementation Plan, Version 2

This package specifies a reproducible implementation of a relativistic mixed spin-lattice response in an orthonormal spinor Wannier basis. The electronic calculation ends at the transverse-spin/displacement kernel

\[
K_{\ell a,\kappa\mu}(\mathbf q)
=\frac{\partial^2 \mathcal F}
{\partial \pi_{\ell a}(-\mathbf q)\,\partial u_{\kappa\mu}(\mathbf q)},
\]

which is projected onto phonon modes and externally supplied magnon Bogoliubov modes,

\[
K_{\pi u}(\mathbf q)
\longrightarrow V_{\pi\nu}(\mathbf q)
\longrightarrow g^{\mathrm{mp}}_{r\nu}(\mathbf q).
\]

The magnon spectrum is not recomputed by this package. Production input must provide magnon energies, local spin frames, spin lengths, Nambu ordering, and paraunitary eigenvectors.

## Version 2 changes

Version 2 replaces the original onsite-default torque construction by the relativistic local-projection framework and makes the literature boundary explicit.

- The exchange field is extracted separately from the time-reversal-symmetric Hamiltonian.
- Only the exchange field is rotated. Spin-orbit coupling remains fixed to the crystal and enters through the full spinor Green function.
- The default site partition is
  \[
  \mathcal L_\ell[X]=\tfrac12(P_\ell X+X P_\ell),
  \]
  while onsite-only projection is retained as a diagnostic.
- The magnetic Wannier subspace is selected explicitly and sensitivity to that choice is a mandatory validation.
- The default local partition produces a general `vertex(ell,a,k,q)`; a constant matrix is only an onsite-limit optimization.
- Rotation-angle vertices and transverse-direction vertices are stored as different coordinate types.
- The source-derived statements, project extensions, and standard auxiliary results are separated in `references/SOURCE_PROVENANCE.md`.
- A paper-style LaTeX derivation and compiled PDF are included under `theory/`.

See `CHANGELOG.md` for the complete migration list.

## MVP electronic expression

For a screened Cartesian DFPT perturbation

\[
g_{\kappa\mu}(\mathbf k,\mathbf q)
=\left\langle w_{\mathbf k+\mathbf q}\middle|
\frac{\partial H}{\partial u_{\kappa\mu}(\mathbf q)}
\middle|w_{\mathbf k}\right\rangle,
\]

the bubble-only retarded loop is

\[
\mathcal A^R_{\ell a,\kappa\mu}(\mathbf q)
=\frac{1}{N_k}\int d\varepsilon\,f(\varepsilon)
\sum_{\mathbf k}\operatorname{tr}
\left[
\mathcal T_{\ell a}(-\mathbf q)
G^R_{\mathbf k+\mathbf q}
 g_{\kappa\mu}(\mathbf k,\mathbf q)
G^R_{\mathbf k}
\right].
\]

A general finite-q Fourier coefficient can be complex. The physical kernel is therefore formed from the retarded q pair,

\[
K^{\mathrm{bub}}(\mathbf q)
=-\frac{\mathcal A^R(\mathbf q)-\mathcal A^R(-\mathbf q)^*}{2\pi i}.
\]

At a self-inverse q point this reduces to `-Im A^R/pi`. A single-q `np.imag` shortcut is prohibited at general q.

## Source boundary

Two supplied papers provide the primary formal foundation:

1. G. Martínez-Carracedo *et al.*, *Phys. Rev. B* **108**, 214418 (2023): relativistic LKAG rotations, exchange-field extraction, Hermitian local projection, localized magnetic subspace, and sum-rule tests.
2. S. Mankovsky *et al.*, *Phys. Rev. B* **107**, 144428 (2023): mixed spin-displacement Green-function insertion, displacement-induced effective field and torque, and phonon-like finite-q interpretation.

The finite-q screened-DFPT Wannier implementation and direct projection onto external magnon modes are project extensions. Exact bibliographic data, equation-level provenance, and SHA-256 identification of the supplied source PDFs are listed in `references/`.

## Package map

```text
README.md
AGENTS.md
CHANGELOG.md
MANIFEST.md
VERSION
CHECKSUMS.sha256
docs/                 physics and software specifications
tasks/                ordered Codex implementation tasks
templates/            implementation and validation templates
references/           exact bibliography and claim provenance
theory/               modular LaTeX manuscript and compiled PDF
```

## Recommended reading order

1. `AGENTS.md`
2. `docs/00_scope_and_source_boundary.md`
3. `docs/01_notation_and_conventions.md`
4. `docs/03_exchange_field_extraction.md`
5. `docs/04_local_projection_and_torque_vertices.md`
6. `docs/05_electronic_mixed_kernel.md`
7. `references/SOURCE_PROVENANCE.md`
8. `theory/wannier_torque_formalism.pdf`
9. Tasks in numeric order.

## Non-goals of the MVP

- Recomputing exchange tensors or magnon dispersions.
- Claiming operator-level identity between KKR displacement matrices and DFPT Hamiltonian derivatives.
- Decomposing the total linear torque kernel into DMI, symmetric anisotropic exchange, and single-ion terms.
- Native nonorthogonal-basis Green functions.
- Enabling the direct mixed vertex without an explicitly supplied exchange-field derivative.
