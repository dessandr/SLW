# Full spinor EPR to magnon–phonon scattering

The new numerical bridge is
`slw.exchange.kernels.spinor_epr_derivative`. It differentiates the static
spinor TB2J exchange functional using a single full spinor Hamiltonian and
Cartesian EPR derivative. It does not construct an up/down surrogate for SOC.

## Local spin model and units

Choose the local spin definition explicitly and reproduce the static exchange
before combining a new derivative with an existing magnon model.

- `periodic_spinor_transform(g, Qk, Qkq)` applies
  `Q(k+q)^dagger g(k,q) Q(k)` and changes canonical interleaved spin ordering
  to the spin-major ordering of the exchange kernel. The Fourier convention
  is cell periodic; no atomic-position displacement phase belongs here.
- Identity endpoint frames retain the Pauli operators of a declared TB2J
  interleaved Wannier-pair model. `wannier_pair_masks` checks colocated pairs
  and magnetic site membership. Geometry does not establish spin purity.
- An AMN polar frame defines a different local spin model. The static exchange
  can change substantially because its local orbital projectors change.
  Do not replace the TB2J definition by AMN projectors and rescale the result
  to conceal this difference.
- H is in eV and Cartesian g is in eV/Angstrom. `derivative_A` returns complex
  dA(q) in eV/Angstrom. Fourier transform q to Rp **before** taking the
  imaginary part with `isotropic_from_A`, which returns meV/Angstrom.
- The TB2J source convention is `H=-sum_directed J e_i.e_j`. The native
  scattering convention is `H=-1/2 sum_directed J e_i.e_j`; multiply both
  source J and source dJ by two when making that conversion.

## Response and interpolation

The kernel includes the four product-rule terms in
`d Tr[P_i G_ij P_j G_ji]`. A screened displacement changes the onsite exchange
field at every magnetic site, including when a nonmagnetic atom is displaced.
The new `onsite_projector_derivatives` argument of the tensor kernel supplies
all of these responses. Existing target-only callers retain their previous
approximation. Local exchange-field directions and orbital projectors are
frozen; projector motion and self-consistent spin relaxation are not included.

Use `DenseEPREvaluator` with the selected LR/SR models and the actual coarse
q representatives. Keep full spin-flip g and its two electronic endpoints.
Check the q-pair residual before Hermitian pair averaging. Commensurate
electronic k grids are required by this Green-function derivative kernel.
`semicircle_gauss` supplies a zero-temperature insulating contour; its lower
bound must enclose all occupied eigenvalues and the upper bound must lie in
the gap. Compare increased quadrature orders.

For dense phonon q interpolation, `slw.magph.derivative_images` places Rp
images relative to **both** magnetic bond endpoints and splits tied images
equally. It preserves all coarse samples and the actual bond reversal
`(i,j,R;Rp) <-> (j,i,-R;Rp-R)` off grid. A single periodic Rp box does not
generally preserve this relation between coarse q points.

Save raw dJ and report bond-reversal and acoustic-sum-rule residuals before
any projection. Large corrections require investigation. Do not identify a
dense integration grid with convergence of the coarser dJ sampling grid or
of its real-space bond cutoff.

## Rates

Contract dJ(q) with cell-gauge phonon zero-point displacements from
`slw.magph.phonon`. The explicit mass unit and the `e/sqrt(M)` convention are
part of the input contract. The resulting `ModeResolvedExchangeDerivative`
feeds the existing `build_scattering_problem` and retarded self-energy APIs.
This route currently consumes isotropic dJ; anisotropic exchange derivatives
are not included in its scattering vertex.

For a degenerate magnon subspace, evaluate the full self-energy at the common
energy and diagonalize `Gamma=-(Sigma-Sigma^dagger)/(2i)` within that subspace.
The diagonal of Sigma in an arbitrary degenerate basis is not an invariant
set of lifetimes. For isolated modes use the on-shell diagonal result.

The linewidth convention is HWHM Gamma, FWHM `2 Gamma`, scattering rate
`2 Gamma/hbar`, and lifetime `hbar/(2 Gamma)`. Preserve and flag negative
damping rather than taking its absolute value. Delta-function broadening is
a numerical parameter and must be compared along with the integration mesh.
The result is the Born exchange-striction lifetime of a magnon; coherent
one-magnon/one-phonon mixing alone is not an intrinsic linewidth, and a
magnon-polaron quasiparticle lifetime needs a further interaction rotation.

## Validation and execution

`tests/exchange/test_spinor_epr_derivative.py` compares the four-term response
against an independent real-space supercell finite difference at zero and
nonzero q, including complex Fourier displacements and remote-site dP. It
also checks endpoint gauge transformations with spin-flip matrices.
`tests/magph/test_derivative_images.py` checks coarse-grid preservation,
off-grid bond reversal and q conjugation.

The NiO run is material-specific input, kept outside this library:
`work/NiO/wtorque/spinor-epr-scattering-20260911-ofjb_tc3`.
It distributes derivative q points and external scattering k points with MPI,
saves atomic restartable shards, and snapshots the source. Its run README
records approximations, comparison grids and job status.
