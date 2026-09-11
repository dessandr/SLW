# Native spinor magnon–phonon mixed-response workflow

`python -m slw.wtorque run-native native.json` now connects native qe2pert
spinor data to a Cartesian spin–displacement kernel, zero-point phonon
projection, SLW magnon projection, and a rotating-wave spectrum. Independent
q tasks support MPI, for example `mpiexec -n 2 python -m slw.wtorque run-native
native.json`. Set BLAS/OpenMP thread counts explicitly for the allocated CPUs.

## Scope and conventions

This route is a **projection-anchored, fixed-projector, rigid-spin
approximation**, with isotropic TB2J magnons. A full square AMN polar frame
provides atomic spin/orbital coordinates; arbitrary MLWFs are not assumed to
have canonical Pauli matrices. H and g use the same frame at both momentum
endpoints. The model exchange is the time-reversal-odd part of H. SOC remains
in the Green functions. The local spin frames come from the actual TB2J
moments, with explicit spin lengths supplied separately.

The direct torque-displacement term is now optional and implemented. Its
native source is the derivative of the same TR-odd model exchange at fixed
AMN frame and sewing matrix. Total DFPT g is not identified with XC-only g.
Basis/projector motion remains excluded. Absent SPN data, physical-spin
closure of the polar frame is untested; rank and Gram/time-reversal covariance
are reported separately.

The stored theory defining this operator is
[`03_spinor_model.tex`](../new_feature/wannier_torque_magnon_polaron_impl_plan_v2/theory/sections/03_spinor_model.tex)
and [`C_direct.tex`](../new_feature/wannier_torque_magnon_polaron_impl_plan_v2/theory/sections/C_direct.tex),
with the operational contract in
[`19_direct_vertex_and_projector_motion.md`](../new_feature/wannier_torque_magnon_polaron_impl_plan_v2/docs/19_direct_vertex_and_projector_motion.md).
The finite-q implementation differentiates that model as

```
g_XC(k,q) = [g(k,q) - B(k+q) g(-k,-q)* B(k)†] / 2
D(k;q) = [P(-q) Gamma(g_XC(k,q)) + Gamma(g_XC(k-q,q)) P(-q)] / 2
A_direct(q) = sum_k w_k integral^mu dE Tr[D(k;q) G(k)]
K_total(q) = K_bubble(q) + K_direct(q)
```

The first equation is evaluated in the original periodic Wannier basis,
then the derivative is converted with the same atomic frame/phases as g.
The second equation closes the spin(-q)/displacement(q) momenta before
taking the diagonal-k trace. Both electronic endpoints of the k−q term
include reciprocal wrapping. Simply inserting the forward g_XC commutator
in Tr[G(k)] is incorrect at general q. The transverse-direction axes use
the same coordinate conversion as the bubble.

This is a **model exchange derivative**, not an independently separated QE
XC-potential response. Derivatives of Q, B, local projectors and frames are
held zero. The stored theory's warning against rotating total scalar/SOC g
is respected by the TR split; independent tests exclude TR-even scalar/SOC
perturbations and compare with the finite difference of the full supercell
TR-odd Hamiltonian.

The finite-q bubble uses a reverse torque insertion and a forward g:

```
A(q) = sum_k w_k integral^mu dE Tr[T(k←k+q) G(k+q) g(k+q←k) G(k)]
K(q) = -(A(q) - conj(A(-q))) / (2 i pi)
```

The zero-temperature pole integral is analytic, including nearly degenerate
energies. The lower energy endpoint, chemical potential and broadening are
explicit inputs. No finite energy-grid quadrature or finite-temperature
claim is involved. q/−q completion enforces a real-space Hermitian coefficient;
its residual is not an independent input check. The independent raw g
Hermitian-partner residual is stored before any optional pair averaging.
The direct one-G integral also uses an analytic logarithmic primitive.

Both electrons and bosonic coordinate fields use the atomic-position phase
`exp(+i 2pi q·(R+tau))`. Reciprocal wrapping, perturbation phases, phonon
conjugate partners, and magnon particle–hole partners are explicit.

The coarse `*_elph.h5` file is required: it retains the polar contribution
before EPR short-range subtraction. The reader reverses its Fortran bra/ket
storage axes and maps the PH star-major q order explicitly. g is converted
to eV/angstrom. Phonons use the native EPR IFCs with polar restoration; the
full dynamical matrices are independently checked against the PH XML files.
EPR masses are in Rydberg atomic mass units and are converted to amu before
zero-point normalization. Output mode vertices are in eV.

## Input

Paths are relative to the JSON file, or absolute. `dynamical_xml` must list
files in the exact order qe2pert read them; each file contributes its internal
Q_POINT order. qpoints are reduced reciprocal coordinates, must be on the
coarse mesh, and must include each exact signed −q partner. The current mode
projection rejects Goldstone, zero acoustic and unstable modes; it never
adds a stabilizing gap. Start with a finite q pair for an AFM.

```json
{
  "approximation": "projection_anchored_bubble",
  "epr": "seed_epr.h5",
  "elph": "out/seed_elph.h5",
  "qe_xml": "out/seed.save/data-file-schema.xml",
  "win": "seed.win",
  "amn": "seed.amn",
  "eig": "seed.eig",
  "u_mat": "seed_u.mat",
  "u_dis_mat": "seed_u_dis.mat",
  "nnkp": "seed.nnkp",
  "exchange_out": "TB2J_results/exchange.out",
  "dynamical_xml": ["ph/seed.dyn1.xml", "ph/seed.dyn2.xml"],
  "qpoints": [[0.3333333333333333, 0, 0], [-0.3333333333333333, 0, 0]],
  "u_dis_layout": "compact_outer_window",
  "atomic_spin_order": "interleaved",
  "magnetic_atom_labels": ["M1", "M2"],
  "spin_lengths": [1, 1],
  "source_directed_bond_weight": 1.0,
  "fermi_energy_eV": 15.0,
  "energy_min_eV": -30.0,
  "eta_eV": 0.01,
  "ep_energy_unit": "ry",
  "ep_displacement_unit": "bohr",
  "output": "vertex.h5"
}
```

The mesh, labels, spin lengths, Fermi level and source Hamiltonian convention
in this example must match the actual calculation. TB2J's directed
`H=-sum_ij J_ij e_i·e_j` convention has weight 1; the adapter explicitly
converts to SLW's half-weighted directed representation. Only J_iso is used;
DMI and incomplete anisotropy tensors are not silently included.

Optional inputs are `perturbation_chunk` (default 3),
`hamiltonian_tolerance_eV` (1e-5), `projector_tr_tolerance` (0.05),
`g_reciprocity_tolerance` (1e-5), and `g_pair_policy` (`raw` or
`hermitian_pair_average`). The last option averages g(k,q) with
g(k+q,−q)† in memory and preserves the original input file. It is useful for
a sensitivity comparison; it is not evidence that the underlying DFPT
error has been resolved. Overrides of the model/input tolerances are saved;
the nominal 5%/1e-5 diagnostic gate statuses are also retained.

To enable the native direct term, add the following explicit choices:

```json
{
  "approximation": "projection_anchored_frozen_frame_total",
  "include_direct_vertex": true,
  "g_xc_source": "fixed_frame_tr_odd",
  "export_g_xc": true
}
```

`export_g_xc` defaults to true when direct is enabled. It stores
`/dfpt/q_*/g_xc_cart` in eV/angstrom in the **AMN atomic basis** used in this
calculation, together with k/q points, perturbation labels, the polar frame,
orbital centers and derivative-kind metadata. It is readable as a derivative
artifact; it is not a complete strict-workflow input and must not be combined
with an untransformed original MLWF Hamiltonian. A separately supplied upstream
g_XC uses the strict HDF5 workflow described in WTORQUE.md.

## Artifacts and validation

The output HDF5 and adjacent JSON refuse to overwrite existing results.
HDF5 datasets include:

- `kernel/A_retarded`, `kernel/K_pi_u`: total loop and Cartesian kernel;
  `_bubble`, `_direct`, `_total` preserve each component separately.
- `phonons/*`, `magnons/*`: energies, vectors, masses, spin frames and lengths.
- `coupling/V_pi_ph`, `coupling/g_mp_normal`, `coupling/g_mp_anomalous`,
  `coupling/full_nambu`: zero-point and magnon mode coefficients.
  Normal/anomalous coefficients and V_pi_ph also retain `_bubble` and `_direct`.
- `bands/rwa_energies_eV`: the rotating-wave model spectrum.
- `metadata/summary_json`: resolved input, SHA256 hashes, approximations,
  independent checks and numerical diagnostics.

The RWA spectrum is an approximate downstream diagnostic; it is not a full
bosonic BdG stability test. Mode-resolved complex phases depend on the chosen
eigenvectors, especially inside degenerate spaces. Compare singular values
or summed Frobenius norms when assessing such spaces.

Tests cover direct adaptive quadrature versus analytic finite-q poles,
independent source/final gauge covariance, atomic reciprocal wrapping,
site torque finite differences, native HDF5 axes, SI zero-point factors,
PH XML reconstruction, SLW exchange normalization, bosonic partner gauges,
and an analytic kernel-to-mode output. Material calculations additionally
need mesh, broadening, integration-window and projection-model checks.
Direct-term tests additionally compare nonlocal supercell mixed finite
differences, independent one-G quadrature, the occupied-band-energy Hessian,
and finite-temperature grand-potential derivatives against bubble+direct.
Strong cancellation between the terms makes sensitivity of the **total**
coefficient the relevant numerical check.

## Dense interpolation and full magnon–polaron bands

`run-interpolated` interpolates the electronic Hamiltonian and Cartesian
electron–phonon derivative, recomputes the electronic response on an
independent uniform k mesh, and projects at every requested q. It does not
interpolate individual complex mode vertices between independently chosen
phonon or magnon eigenvectors.

```json
{
  "native_config": "native_total.json",
  "kmesh": [8, 8, 8],
  "points_per_segment": 40,
  "cache_dir": "interpolation_cache",
  "output": "dense/bands.h5"
}
```

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  mpiexec -n 2 python -m slw.wtorque run-interpolated dense.json
```

`native_config` is a valid `run-native` input. Its original commensurate
q/−q pair supplies the independent source/frame/PH validation baseline; its
`output` is not written by this command. Source files, spin lengths,
exchange convention, approximation, broadening, integration window, chemical
potential and raw/pair-average policy are inherited. All paths in the dense
JSON resolve relative to that JSON. The command works without `mpiexec` too.
MPI assigns independent q pairs to ranks; set processes and BLAS/OpenMP
threads to fit the allocated CPUs.

Choose one q sampling form:

- `points_per_segment` reads the **actual input cell's** `kpoint_path` from
  the native Wannier90 `.win`. This counts intervals, retains both endpoints
  and draws disconnected pieces separately. It defaults to 40.
- `qpoints: [[qx,qy,qz], ...]` specifies a path or a small validation/refinement
  set directly. Exact signed −q partners are evaluated automatically; they
  need not appear in this list. Distances follow the listed sequence.
- `qmesh: [nx,ny,nz]` generates a uniform three-dimensional q mesh in the
  centered representative interval `[-1/2,1/2)`. This writes the full HDF5
  dataset and JSON, without presenting the mesh ordering as a band path.

These three choices are mutually exclusive. Repeated endpoints and exact
signed q partners reuse the same response task. q+G representatives are
kept distinct. Γ and equivalent reciprocal points are explicitly omitted
before acoustic zero-point and AFM Goldstone normalization; their full-path
arrays contain NaNs with a recorded mask/status. Other unstable or singular
bare modes are located and reported. No artificial gap or stability shift
is added.

The EPR evaluator uses pair-specific Wigner–Seitz images and their stored
weights for H and the short-range g, then restores the analytic **3D dipole**
contribution in the original Wannier basis. The finite reciprocal-vector
cutoff reproduces the native Perturbo prescription and may retain small
q+G representative dependence; backend diagnostics record this limitation.
General-q matrices are not made Hermitian. The AMN polar frame is interpolated
with periodic reciprocal-space boundary conditions, and every H/g/torque
endpoint uses that same frame and its atomic-position phase. The fixed-frame
TR-odd `g_XC` derivative and direct term are recomputed at the shifted dense
momenta when enabled by the native input. The dense route retains kernels
and mode couplings rather than exporting the large intermediate g_XC arrays.

### Independent LR and SR interpolation choices

The dense workflow JSON and `DenseEPREvaluator` accept two independent options:

- `longrange_model="source"` (default) retains the source Wannier-identity
  dipole term. `"point_center"` includes `exp(+i2π(q+G)·c_i)` for each Wannier
  diagonal and every G term. This remains a point-center approximation, with
  finite orbital extent and off-diagonal electronic form factors omitted.
- `short_range_model="source"` (default) retains the original displacement
  WS cells. `"two_center"` selects displacement images by their summed distance
  to the bra center and the ket center in cell Re. The selected images are
  checked under `(i,j,Re,Rp) -> (j,i,-Re,Rp-Re)`. An insufficient candidate
  domain is rejected. No g matrix is averaged or made Hermitian by this option.

`point_center` requires a polar EPR and `longrange_coarse_qpoints`: the **complete actual reduced
q representatives used when the source EPR was produced**. The EPR mesh size
alone does not determine these representatives, and the source finite G sum
is not exactly reciprocal periodic. Supply the original q list rather than
independently folding it. The list must cover the unshifted qmesh once modulo
reciprocal integers; its original floating coordinates are retained.

Changing LR applies both `-I_selected[L_new(qc)-L_source(qc)]` to the stored SR and
`L_new(q)` on add-back. The subtraction uses the selected SR plan's actual
diagonal `Re=0` images and degeneracies. Even at `Re=0`, single-distance and
summed-distance WS criteria can select different near ties at the same finite
tolerance; substituting the original cells is then inconsistent. Changing SR
redistributes the original residue totals
among different equivalent phonon images. Both operations preserve the sampled
coarse q data; they can therefore be tested separately and together. The new
image geometry makes the interpolation compatible with the real-space
reciprocity relation, but does not remove a violation already present in the
sampled input coefficients.

Reciprocal image closure is not a magnetic-symmetry gate. In particular, it
does not ensure AFM covariance in an electronic frame that mixes Wannier
orbitals or varies with k. Check AFM and q-pair residuals independently for
each selected model and for their combination; an improvement in one residual
does not establish an improvement in the other. These options apply no AFM
projection.

For direct inspection, `evaluate_g(k,q,include_longrange=False)` returns the
re-split SR for the selected model, and `longrange_diagonal(q)` returns its
`(3*nat,nwan)` analytic diagonal in eV/Angstrom. The older `longrange(q)` method
keeps returning the **source** scalar of shape `(3*nat,)`, independently of the
selected model. To reconstruct the selected full g, embed `longrange_diagonal(q)`
on the electronic matrix diagonal and add it to the selected SR.

These choices are included in the response restart fingerprint and backend
diagnostics. They use the same rank-local evaluator and MPI q ownership as the
default workflow. A change of interpolation options requires separate output
provenance; it does not validate phonons, electronic-response convergence, or
the resulting polaron bands.

The native input's `g_pair_policy` also applies here. With
`hermitian_pair_average`, the evaluator averages g(k,q) with
g(k+q,−q)† before the TR split and response; it does not Hermitize g at one q.
Raw partner residuals are recorded before averaging. The native coarse-grid
`g_reciprocity_tolerance` is **not enforced at interpolated momenta**: this
route treats the raw residual as a diagnostic, records that the configured
coarse gate was not applied, and retains the nominal gate status. Compare
raw and averaged responses where interpolation residuals are appreciable.

For full magnon–polaron bands, both q signs supply the combined Nambu Hessian
in `(a_q,b_q,a†_-q,b†_-q)` order. With `V(q)` the complete projected
magnetic/phonon Nambu coefficient, the implementation verifies
`V(q)=X_m V(-q)* X_p`; it never assumes that finite-q pairing is symmetric at
one q. Positive Hessians use a Hermitian Cholesky reduction of the bosonic
dynamic matrix. Negative Hessian eigenvalues, complex frequencies and zero
modes remain in diagnostic datasets, while their physical-band slots are
masked. The ordinary RWA spectrum is retained for comparison.

Outputs include:

- HDF5 q points, path distances and labels, valid masks, bare energies and
  q/−q mode transformations; Cartesian `kernel/K_pi_u_bubble`, `_direct`,
  `_total` and their `_minus` partners; the corresponding complete
  `coupling/full_nambu_*` coefficients.
- `bands/energies_eV`, `bands/magnon_weights`, normalized eigenvectors, RWA
  energies, complete dynamic eigenvalues, minimum Hessian eigenvalues,
  q-pair/eigenvector/paraunitarity residuals and explicit statuses.
- Standalone PNG/PDF figures for the full spectrum and 0–80 meV view, plus
  a CSV table for path/explicit-q calculations. Color represents
  `sum_m(|u|²+|v|²)/sum_all(|u|²+|v|²)`, a bounded magnon mode character;
  it is not a neutron-scattering intensity.
- Adjacent JSON and embedded metadata containing source SHA256 hashes,
  all Python implementation hashes under `slw`, resolved inputs, source
  validation, interpolation diagnostics and per-q response diagnostics.

Progress prints the completed fraction of q-pair tasks. The shared cache
holds read-only coefficient maps and atomic per-q response shards. A restart
validates configuration, source hashes, implementation hashes and q-pair
data before reusing completed tasks. HDF5 and plots are written on rank zero.
Re-running a completed input checks its provenance and repairs missing
plot/table/JSON artifacts without recomputing the response. A changed
calculation cannot overwrite an existing result.

Dense k integration and dense q plotting **do not establish convergence of
the original coarse DFPT q mesh**. Test the k mesh, integration bounds and
broadening separately, and refine the underlying DFPT mesh when required by
the real-space tail or independent q checks. AMN spin-frame closure and the
isotropic magnon/fixed-projector assumptions remain physical limitations.
Near a crossing, compare full BdG splittings and mode-character transfer on
a refined q interval, rather than treating a coarse plotted intersection as
evidence of an avoided crossing.

Tests additionally cover finite-q nonreciprocal particle–hole pairing,
independent coupled-oscillator eigenvalues, the RWA limit, mode-gauge
covariance, degenerate metric orthogonality, explicit instability handling,
path breaks, q-task deduplication, provenance rejection, interrupted-task
restart and artifact recovery after plotting fails.
