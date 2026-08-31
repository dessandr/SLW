# Direct Mixed Vertex and Projector Motion

## Total derivative

For spin coordinate `s` and displacement `u`,

\[
\partial_u\partial_s\mathcal F
=-\frac1\pi\operatorname{Im}\int f\,\operatorname{Tr}
\left[
\frac{\partial^2H}{\partial s\partial u}G
+\frac{\partial H}{\partial s}G
 \frac{\partial H}{\partial u}G
\right].
\]

The first term is absent from the MVP.

## Fixed-projector exchange-field derivative

If an upstream calculation supplies `g_XC=dH_XC/du` in the reference Wannier gauge and projectors are fixed,

\[
\Gamma^{(1)}_{\ell c}
=\frac{i}{2}[\mathcal L_\ell(g_{\rm XC}),\sigma_c].
\]

## Projector-response terms

If `P_ell` depends on displacement,

\[
\partial_u\mathcal L_\ell[X]
=\tfrac12[(\partial_uP_\ell)X+X(\partial_uP_\ell)]
+\mathcal L_\ell[\partial_uX].
\]

The first part is a Pulay/gauge-response correction. It is not reconstructed from ordinary Cartesian `g` alone.

## Feature gate

`include_direct_vertex=true` requires:

- an exchange-field-resolved derivative;
- a declared fixed or moving projector policy;
- direct-term finite-difference validation;
- separate output for bubble, direct, and total kernels.
