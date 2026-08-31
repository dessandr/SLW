# Electronic Mixed Spin-Lattice Kernel

## Magnetic-force-theorem derivative

For a Hamiltonian parameter `lambda`,

\[
V_\lambda=\frac{\partial H}{\partial\lambda},
\qquad
\frac{\partial \mathcal F}{\partial\lambda}
=-\frac1\pi\operatorname{Im}\int d\varepsilon\,f(\varepsilon)
\operatorname{Tr}[V_\lambda G^R].
\]

For two perturbations,

\[
\frac{\partial^2\mathcal F}{\partial\lambda\partial\eta}
=-\frac1\pi\operatorname{Im}\int d\varepsilon\,f
\operatorname{Tr}
[V_{\lambda\eta}G^R+V_\lambda G^R V_\eta G^R].
\]

## DFPT displacement vertex

\[
g_{\kappa\mu}(\mathbf k,\mathbf q)
=\frac{\partial H}{\partial u_{\kappa\mu}(\mathbf q)},
\qquad
\frac{\partial G}{\partial u}=GgG.
\]

## Bubble-only finite-q loop

\[
\mathcal A^R_{\ell a,\kappa\mu}(\mathbf q)
=\frac1{N_k}\int d\varepsilon\,f(\varepsilon)
\sum_{\mathbf k}\operatorname{tr}
\left[
\mathcal T_{\ell a}(-\mathbf q)
G^R_{\mathbf k+\mathbf q}
 g_{\kappa\mu}(\mathbf k,\mathbf q)
G^R_{\mathbf k}
\right].
\]

The physical complex Fourier coefficient is

\[
K^{\rm bub}(\mathbf q)
=-\frac{\mathcal A^R(\mathbf q)-\mathcal A^R(-\mathbf q)^*}{2\pi i}.
\]

This yields `K(-q)=K(q)^*`. At a self-inverse q point it becomes `-Im A^R/pi`.

## Direct mixed vertex extension

The total derivative also permits

\[
\mathcal T^{(1)}_{\ell a,\kappa\mu}
=\frac{\partial^2 H}
{\partial\pi_{\ell a}\partial u_{\kappa\mu}},
\]

with a one-Green-function loop. It is disabled in the MVP. For rotation coordinates and fixed projectors, a possible validated construction is

\[
\Gamma^{(1)}_{\ell c,\kappa\mu}
=\frac{i}{2}
[\mathcal L_\ell(g_{{\rm XC},\kappa\mu}),
 \mathbf u_{\ell c}\cdot\boldsymbol\sigma].
\]

Never replace `g_XC` by the full `g`, because that would rotate displacement-induced SOC together with the spin.

## Physical coupling energy

\[
\mathcal H_{\rm SL}^{(1,1)}
=\sum_{\mathbf q,\ell a,\kappa\mu}
\pi_{\ell a}(-\mathbf q)
K_{\ell a,\kappa\mu}(\mathbf q)
 u_{\kappa\mu}(\mathbf q).
\]

The result is the total site-resolved electronic torque kernel within the approximation. It is projected directly onto magnons rather than decomposed into individual relativistic spin-model tensors.

## Numerical backend

The physics layer accepts either:

- real-axis quadrature with positive broadening;
- an existing complex-contour integrator.

The q-pair finalizer is backend-independent. k weights and energy weights are each applied exactly once.
