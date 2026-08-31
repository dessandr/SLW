# Finite-q Mapping and Gauge Control

## Exact k+q map

For each `(ik,iq)`, precompute:

- wrapped `kq_index`;
- reciprocal vector `G_wrap`;
- atomic-gauge matrix `D_G`;
- residual mesh mismatch.

The direct MVP requires commensurate meshes.

## Wrapped Green function

\[
G(\mathbf k+\mathbf q)
=D_{\mathbf G}^\dagger
G(\overline{\mathbf k+\mathbf q})D_{\mathbf G}.
\]

## DFPT vertex covariance

`g(k,q)` maps the initial k basis to the final `k+q` basis. Conversion acts on the final index according to the declared upstream gauge. No conversion is inferred from array shape.

## Perturbation Hermiticity

After conversion,

\[
g_{\kappa\mu}(\mathbf k,\mathbf q)^\dagger
=g_{\kappa\mu}(\mathbf k+\mathbf q,-\mathbf q)
\]

with wrapped indices and sublattice phases treated consistently.

## q-pair completion

The raw retarded loop is not itself the physical finite-q coefficient. The code must construct

\[
K(\mathbf q)=-[A^R(\mathbf q)-A^R(-\mathbf q)^*]/(2\pi i).
\]

Both absolute and relative q-conjugation residuals are written per q pair.

## Gauge metadata

Inputs declare:

- atomic-position or cell-periodic Bloch gauge;
- sign of each Bloch and Fourier exponential;
- orbital centers and units;
- representation of wrapped final states;
- time-reversal sewing matrix convention;
- magnon and phonon Fourier conventions.

## Degenerate mode gauge

Individual branch couplings inside a degenerate subspace are basis-dependent. Validation compares singular values, `Tr(g g^dagger)`, and polaron eigenvalues, not arbitrary branch-resolved phases.
