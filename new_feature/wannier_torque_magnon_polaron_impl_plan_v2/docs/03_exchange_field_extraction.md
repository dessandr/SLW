# Exchange-Field Extraction in a Spinor Wannier Basis

## Canonical decomposition

The runtime Hamiltonian is

\[
H(\mathbf k)=H_{\rm TRS}(\mathbf k)+H_{\rm XC}(\mathbf k),
\]

where `H_TRS` includes the scalar crystal Hamiltonian and SOC, while `H_XC` is identified with the time-reversal-broken exchange field. This identification assumes that the one-particle Hamiltonian contains no additional time-reversal-breaking term such as an external orbital or Zeeman field; any such term must be supplied and separated explicitly.

## Preferred production route: upstream decomposition

The safest input supplies `H_TRS_R` and `H_XC_R` explicitly in the same Wannier gauge. The loader must verify

\[
H_R=H_{{\rm TRS},R}+H_{{\rm XC},R}.
\]

## General spinor route: time-reversal decomposition

Let `B_Theta(k)` be the unitary sewing matrix mapping the time-reversed basis at `-k` into the basis at `k`. Define

\[
H^\Theta(\mathbf k)
=B_\Theta(\mathbf k)H(-\mathbf k)^*B_\Theta(\mathbf k)^\dagger.
\]

Then

\[
H_{\rm TRS}=\tfrac12(H+H^\Theta),
\qquad
H_{\rm XC}=\tfrac12(H-H^\Theta).
\]

In a direct real-orbital tensor spin basis, the sewing matrix reduces to `I_orb tensor i sigma_y` up to the declared Wannier gauge. The code must not assume this reduction unless metadata guarantees it.

## Collinear spin-split route

If SOC-free or scalar-relativistic spin blocks share one orbital gauge,

\[
H_{\rm avg}=\tfrac12(H_\uparrow+H_\downarrow),
\qquad
H_{\rm diff}=\tfrac12(H_\uparrow-H_\downarrow),
\]

and

\[
H_{\rm XC}=H_{\rm diff}\otimes(\mathbf n_{\rm ref}\cdot\boldsymbol\sigma).
\]

The factor convention must be recorded. A comparison against explicit spinor reconstruction is mandatory.

## Required diagnostics

- `H_TRS` is invariant under the declared time-reversal operation.
- `H_XC` changes sign under time reversal.
- `H_TRS+H_XC` reconstructs `H`.
- The extracted exchange field is Hermitian.
- Scaling or replacing `H_SOC` changes `H_TRS` but not an independently supplied `H_XC`.
- The norm of exchange-field support outside the declared magnetic subspace is reported.

## Prohibited shortcut

Do not call every spin-dependent term `H_XC`. In particular, a commutator with the full Hamiltonian rotates SOC and violates the intended DFT-to-spin mapping.
