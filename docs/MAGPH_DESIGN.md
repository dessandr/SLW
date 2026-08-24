# Native magnon--phonon redesign contract

This document defines the numerical and data-contract boundary for the new
native magnon--phonon implementation. `calculation='dispersion'` and
`calculation='lifetime'` are registered native commands; other
`slw_magph.x` calculations still use
the quarantined compatibility drivers documented in
[INPUT_REFERENCE.md](INPUT_REFERENCE.md).

The first native target is a trustworthy exchange-based pipeline for
magnon--phonon self-energy, scattering rate, and lifetime. Hybrid bands,
Berry quantities, and a later spin-torque extension are downstream work.

The first numerical foundation is now implemented as Python APIs:

- `slw.magph.screening` imports static exchange into an immutable canonical
  model and screens explicit FM or bipartite-AFM reference states;
- `slw.magph.phonon` validates schema-v3 cache units and converts
  mass-normalized polarizations to zero-point displacements;
- `slw.magph.derivative` and `slw.magph.coupling` screen scalar `dJ/du` and
  contract it with physical phonon displacements;
- `slw.magph.lswt` and `slw.magph.vertex` construct arbitrary-N FM or
  bipartite-AFM bare magnons and band-basis exchange-striction vertices;
- `slw.magph.dispersion` reads explicit Wannier90 paths, distributes path
  points with MPI, and retains exact Goldstone energies without inventing
  paraunitary eigenvectors;
- `slw.magph.self_energy` evaluates the common retarded self-energy with
  vectorized, chunked contractions;
- `slw.magph.lifetime` applies one HWHM/FWHM/rate/lifetime convention;
- `slw.magph.parallel`, `pipeline`, `engine`, and `output` distribute external
  k points, assemble results on rank zero, and write one atomic NPZ product.

## Scope and staged pipeline

The native workflow is intentionally split into independently checkable
stages:

1. Import static exchange and canonicalize bond, site, unit, and phase
   conventions.
2. Screen the requested FM or AFM reference state and its bare magnon
   spectrum.
3. Combine exchange derivatives with phonon modes to construct the
   magnon--phonon vertex.
4. Evaluate the retarded self-energy on shell or on an explicit frequency
   grid.
5. Derive linewidth, scattering rate, and lifetime from the same self-energy
   convention.

Static exchange alone is sufficient only for the first two stages. A
magnon--phonon self-energy additionally requires an exchange derivative
`dJ/du` and phonon frequencies/eigenvectors with an unambiguous displacement
normalization. Neither a static `J_iso` nor a static `J_tensor` is a coupling
vertex by itself.

The stages exchange typed arrays and explicit metadata rather than a legacy
flat input file. Material-specific meshes, magnetic sites, spin lengths, and
cutoffs remain user inputs; the package must not select them from built-in
material defaults.

## Canonical static-exchange model

The initial native model uses a directed, mate-complete bond list. Every
`(i,j,R)` entry must have `(j,i,-R)`. Scalar reciprocity requires

$$
J_{ji}(-\mathbf R)=J_{ij}(\mathbf R),
$$

and tensor reciprocity requires

$$
\mathbf J_{ji}(-\mathbf R)=\mathbf J_{ij}(\mathbf R)^T.
$$

The loader keeps global atom indices for provenance but constructs contiguous
local magnetic-site indices for numerical arrays. The number of magnetic
sites is therefore the size of that mapping, not `max(atom_index) + 1`.
Conflicting duplicates, missing mates, non-finite values, and inconsistent
bond metadata fail validation. Bond counting is part of the canonical model;
the native path does not expose a corrective `bond_factor` knob.

Static energy is stored canonically in meV. A scalar-only bond is promoted
without loss as

$$
\mathbf J_{ij}=J_{ij}^{\mathrm{iso}}\mathbf I_{3\times3}.
$$

This promotion provides one internal array shape; it does **not** create DMI,
symmetric anisotropy, torque, or any other tensor information. When scalar and
tensor representations coexist, the loader checks

$$
J_{ij}^{\mathrm{iso}}\simeq \frac13\operatorname{Tr}\mathbf J_{ij}
$$

within an explicit tolerance. It must not silently replace a tensor by its
trace or silently prefer inconsistent payloads.

The exchange Hamiltonian sign, spin normalization, directed-bond counting,
real-space translation, and Fourier phase convention must be carried in the
input provenance or selected explicitly. In particular, unit-vector LKAG
exchange and spin-operator exchange differ by the chosen spin normalization;
the importer must not guess that conversion.

The current canonical contract is
`hamiltonian_sign='minus'`,
`bond_coverage='directed_mate_complete'`,
`directed_bond_weight=0.5`, and
`realspace_gauge='i_at_0_j_at_R'`. New exchange products write these fields,
their kernel family, source spin magnitude, and either
`spin_normalization='unit_vector'` or `'spin_operator'`. Older HDF5 files that
lack them are rejected unless the Python caller supplies an explicit
`ExchangeConvention`; filename or value-sign inference is not accepted.

The scalar LKAG and direct-tensor reference routes use that one-half weight on
a mate-complete list. The retained TB2J decomposition has a different directed
pair convention and records a weight of one instead; it is therefore rejected
by the first native canonical importer until a dedicated conversion is backed
by scalar/direct/TB2J parity tests. Likewise, a canonical-half bond list and a
tensor-axis subset are valid retained exchange artifacts but are not silently
expanded by the native magnon path.

## Single-ion anisotropy

The native isotropic-exchange route may include an independent uniaxial
single-ion term

$$
H_{\mathrm{SIA}}=-\sum_i K_i(\mathbf s_i\cdot\hat{\mathbf n}_i)^2.
$$

Positive `K_i` is easy-axis and negative `K_i` is easy-plane. The spin
variable is explicit: `unit_vector` means $\mathbf s_i$ is a unit direction,
while `spin_operator` means it is the dimensionless spin operator with the
declared site spin magnitude. Consequently the two inputs have different
quadratic LSWT coefficients and are never converted by guessing. One `K` and
axis may be broadcast to every magnetic site, or site-resolved values and axes
may be supplied in magnetic-site order.

SIA participates in reference-state torque/stability screening and in every
LSWT eigensystem used by dispersion and lifetime. It changes magnon energies,
eigenvectors, and self-energy phase space. It does not create a
magnon--phonon vertex by itself: an additional derivative $dK/du$ would be
needed for an anisotropy-striction vertex, and that derivative is outside the
current native scope.

The energy-only dispersion solver admits exact AFM Goldstone points. The
strict paraunitary solver used for vertices and lifetimes continues to require
a physical SIA gap or an explicitly shifted mesh. No artificial anisotropy is
inserted as a numerical regulator. Energy-only FM dispersion also supports a
stationary transverse easy-plane axis through a Nambu energy solve. The first
FM lifetime route still uses its normal $N$-channel basis and therefore
rejects SIA that produces anomalous FM terms; extending that route requires a
tested $2N$-channel FM vertex contract.

### Scalar/tensor capability gate

The following table is an admission policy for the planned native pipeline,
not a statement that each numerical route is already registered.

| Screened inputs | Admitted native capability | Deliberately unavailable |
|---|---|---|
| `J_iso` only | Static screening; bare isotropic magnon problem after the FM/AFM stability gate | Self-energy, scattering rate, and lifetime |
| `J_iso` + `dJ_iso/du` + phonons | Isotropic exchange-striction vertex; scalar self-energy and lifetime route | Static/dynamical DMI, anisotropic exchange, tensor torque, chiral or tensor-hybrid observables |
| `J_tensor` only | Tensor equilibrium and static-spectrum screening | Any magnon--phonon observable |
| `J_tensor` + `dJ_tensor/du` + phonons | Full tensor route after tensor vertex parity is established | No automatic downgrade to the scalar route |
| Mixed scalar/tensor derivative sources | Only an explicitly selected and provenance-compatible projection | Automatic trace projection or implicit component mixing |

A calculation that requests a tensor-only observable from scalar input is
rejected with the missing capability in the error message. Users may inspect
an explicitly computed isotropic projection of tensor data, but that is a new
derived input with recorded provenance, not an implicit fallback.

## FM and AFM gates

The first end-to-end support boundary is explicit:

| Magnetic order | Initial native gate | Bosonic channels |
|---|---|---|
| FM | Any positive number of magnetic sublattices; collinear spin pattern `(+1,...,+1)` | Normal modes with metric `(+1,...,+1)` |
| Collinear bipartite AFM | Exactly two magnetic sublattices with opposite pattern `(+1,-1)` up to global reversal | Four Nambu channels with metric `(+1,+1,-1,-1)` |

The code does not infer FM versus AFM from the sign of one bond. The requested
order, spin pattern, and positive spin magnitude for every magnetic site are
part of the input. The quantization axis is also explicit for tensor exchange;
there is no material-independent default direction for an anisotropic model.
For two-site AFM, an omitted pattern may normalize to the canonical opposite
pair; for more than two AFM sublattices the initial native pipeline fails with
an unsupported-model error rather than inventing a bipartition. General
collinear and noncollinear AFM support requires a tested arbitrary-size
paraunitary solver and is a later extension.

Static screening checks the local exchange field and, for tensor input, the
residual torque of the proposed spin state. Spectral screening then samples
the requested reciprocal mesh. Complex modes, failure of bosonic metric
normalization, or a physical negative mode outside a declared numerical
tolerance blocks vertex and lifetime work. A scalar isotropic model may have
the expected Goldstone mode; it is not silently replaced by an artificial
gap.

## Planned input screening stages

The end-to-end command applies the following fail-closed stages and records
their provenance in the native lifetime product:

1. **Schema and units:** check shapes, lengths, finite values, named datasets,
   and explicit energy/length/mass units. Canonical units are meV, angstrom,
   kelvin, and picoseconds.
2. **Topology and convention:** build the local magnetic-site map, validate
   directed mates and tensor transpose reciprocity, reject conflicting
   duplicates, and fix one explicit real-space/Fourier phase convention.
3. **Magnetic state:** validate order, spin lengths/pattern, local fields, and
   tensor torque before constructing magnons.
4. **Magnon stability:** check the FM Hermitian or AFM bosonic-BdG spectrum on
   the actual calculation mesh, including metric normalization and imaginary
   parts.
5. **Dynamic coupling:** require the derivative bond map to be a
   directed-mate-complete subset of the static map, embed absent static bonds as
   exact zero derivatives, and check `dJ/du` units (normally meV/angstrom),
   rank-zero space-group covariance provenance when present, derivative
   acoustic-sum-rule status, derivative source q-grid, phonon evaluation q-grid
   and phase convention, and phonon stability. Consequently a longer-range
   static `J` model can build LSWT while
   a shorter-range `dJ/du` model supplies the vertex. Derivative bonds that are
   not present in static `J` are rejected.

Phonon mass handling is especially strict. Schema-v3 caches accepted by the
native route store the cell-gauge displacement polarization

$$
p_{\mathbf q\nu\kappa\alpha}
=\frac{e_{\mathbf q\nu\kappa\alpha}}{\sqrt{M_\kappa/m_0}}
$$

with shape `(Nq, 3*Nat, Nat, 3)` and normalization

$$
\sum_{\kappa\alpha}(M_\kappa/m_0)|p_{\mathbf q\nu\kappa\alpha}|^2=1.
$$

For qe2pert EPR data, `basic_data/mass` is $M_\kappa/m_e$; therefore the
cache must use `mass_unit='electron_mass'` and `atom_mass_electron`, rather
than reinterpret those numbers as amu. With
$E_{\mathbf q\nu}=\hbar\omega_{\mathbf q\nu}$ in meV, the physical
zero-point displacement is

$$
u^{\mathrm{zpf}}_{\mathbf q\nu\kappa\alpha}
=p_{\mathbf q\nu\kappa\alpha}
\sqrt{\frac{\hbar^2}{2m_0E_{\mathbf q\nu}}}.
$$

The per-atom mass factor is already present in `p` and is not applied a second
time. The prefactor is `61.7250525812 angstrom*sqrt(meV)` for $m_0=m_e$ and
`1.44571077426 angstrom*sqrt(meV)` for $m_0=m_u$. Missing or contradictory
mass, unit, or vector metadata is rejected. Zero-frequency modes require an
explicit recorded floor, while unstable negative modes fail; neither is
repaired by an undocumented regularizer. The effective, regularized energy is
carried with the displacement and must be the energy supplied to the
self-energy kernel.

The quarantined vertex currently multiplies electron-mass-normalized qe2pert
polarizations by the amu prefactor. It therefore underestimates the vertex
amplitude by about `42.6953` and a quadratic linewidth by about `1822.888`.
Absolute legacy linewidth/lifetime output is not a native parity oracle; the
native conversion is instead locked to the SI harmonic-oscillator expression.

## Self-energy and lifetime conventions

For external momentum `k`, the retarded self-energy uses normalized q weights
`sum_q w_q = 1` and the explicit bosonic channel metric `eta_l`:

$$
\begin{aligned}
\Sigma^R_{mn}(\mathbf k,\omega)
=\sum_{\mathbf q\nu l} w_{\mathbf q}\eta_l
g^*_{lm}(\mathbf k,\mathbf q\nu)g_{ln}(\mathbf k,\mathbf q\nu)
\bigg[&
\frac{n_B(\omega_{\mathbf q\nu})+1+n_B(E_{l,\mathbf k+\mathbf q})}
{\omega+i\delta-E_{l,\mathbf k+\mathbf q}-\omega_{\mathbf q\nu}}\\
&+\frac{n_B(\omega_{\mathbf q\nu})-n_B(E_{l,\mathbf k+\mathbf q})}
{\omega+i\delta-E_{l,\mathbf k+\mathbf q}+\omega_{\mathbf q\nu}}
\bigg].
\end{aligned}
$$

Here all energies and the positive broadening `delta` are in meV. Negative
BdG partner energies use

$$
n_B(-E)=-[1+n_B(E)],
$$

including its zero-temperature limit. The metric is supplied by the magnon
solver rather than inferred from an array length, and every internal channel
must satisfy $\eta_lE_l\geq0$. At finite temperature, an exact zero-energy
Bose mode is rejected unless the upstream magnon or phonon stage applied and
recorded an explicit signed energy floor. The kernel is dimension generic even
though the first AFM model gate is the two-sublattice case.

For a physical positive-norm mode, the on-shell convention is

$$
\gamma_m^{\mathrm{HWHM}}
=-\operatorname{Im}\Sigma^R_{mm}(\mathbf k,E_{m\mathbf k}),
\qquad
\Gamma_m^{\mathrm{FWHM}}=2\gamma_m^{\mathrm{HWHM}},
$$

$$
\mathrm{rate}_m=\frac{\Gamma_m^{\mathrm{FWHM}}}{\hbar},
\qquad
\tau_m=\frac{\hbar}{\Gamma_m^{\mathrm{FWHM}}},
\qquad
\hbar=0.6582119569\ \mathrm{meV\,ps}.
$$

Thus rate is reported in `ps^-1` and lifetime in `ps`. Zero linewidth gives
zero rate and infinite lifetime. Only negative linewidths within a declared
roundoff tolerance may be clipped to zero; a larger negative value is a
causality/stability failure. Matrix damping, when requested, follows

$$
\mathbf D=-\frac{\boldsymbol\Sigma-\boldsymbol\Sigma^\dagger}{2i}.
$$

The production lifetime path distributes the `dJ/du` to phonon-mode coupling
contraction over q points, assembles it deterministically, and broadcasts the
completed `Lambda(q,mode,bond)` cache. It then forms the exact uniform
`lcm(kmesh,phonon_qmesh)` union mesh. Its LSWT eigensystems are likewise distributed
across MPI ranks, assembled, and broadcast once, so repeated `k+q` points are
never diagonalized separately for every external k. External k points then
remain the independent MPI work axis. Within each rank, a q block is transformed
into the band basis and contracted immediately into the on-shell self-energy;
the full `(Nq,Nmode,Ninternal,Nexternal)` vertex is not materialized.
Mode/channel contractions and q/bond or q/channel chunks remain vectorized, q
weights are normalized globally before slicing, and only rank zero coordinates
final metadata/output. The two broadcast caches are currently replicated once
per MPI rank; their actual rank and maximum-node footprints are printed at run
time and recorded in output metadata.

The derivative HDF5 mesh labels the finite real-space `Rp` representation; it
does not restrict the evaluation q grid. For every phonon q point the native
coupling kernel evaluates

$$
\frac{\partial J_{ij}(\mathbf R;\mathbf q)}{\partial u}
=\sum_{\mathbf R_p}e^{+i2\pi\mathbf q\cdot\mathbf R_p}
\frac{\partial J_{ij}(\mathbf R;\mathbf R_p)}{\partial u}.
$$

Thus `phonon_qmesh` may be denser than the derivative source mesh without
recomputing electronic `dJ/du`. This is Fourier interpolation of the retained
real-space derivative, not additional information beyond its source sampling;
both meshes and the active interpolation flag are recorded in the output.

## Native dispersion output v1

`calculation='dispersion'` reads an explicit Wannier90 `kpoint_path` block;
the package does not select material-specific high-symmetry coordinates. The
exchange HDF5 must contain `lattice_ang`, which converts fractional reciprocal
coordinates to a cumulative path distance in inverse Angstrom. Path points
are distributed over MPI ranks and rank zero atomically writes an NPZ plus an
optional PNG, PDF, or SVG plot.

The NPZ contains fractional k points, reciprocal path distance, physical
magnon energies, exact-Goldstone flags, path segment boundaries, tick labels,
and JSON provenance. Exact AFM Goldstone points are valid here because no
paraunitary transform is claimed.

```fortran
&control
  calculation = 'dispersion',
  prefix = 'sample',
  outdir = './slw-tmp'
/
&parallel
  execution = 'auto',
  workers_per_rank = 1,
  threads_per_worker = 4
/
&magph
  exchange_h5 = './input/J.h5',
  magnetic_order = 'collinear_afm',
  spin_magnitudes = 2.5, 2.5,
  spin_pattern = 1, -1,
  quantization_axis = 0.0, 0.0, 1.0,
  anisotropy_model = 'uniaxial',
  anisotropy_mev = 0.05,
  anisotropy_axis = 0.0, 0.0, 1.0,
  anisotropy_normalization = 'unit_vector',
  kpath_file = './input/bands.win',
  points_per_segment = 50,
  output = '${savedir}/sample.dispersion.npz',
  plot = .true.,
  plot_output = '${savedir}/sample.dispersion.png',
  restart_mode = 'error'
/
```

Remove all four `anisotropy_*` keys for a zero-SIA model. Supplying only part
of the SIA contract is an input error.

Both native calculations use `restart_mode='error'|'restart'|'from_scratch'`.
`error` preserves no-clobber behavior. `restart` checks the output schema and
a SHA-256 signature over the scientific parameters plus source-file path,
size, and modification time before reusing a completed NPZ; dispersion also
regenerates a missing plot from that NPZ. `from_scratch` recomputes and
atomically replaces completed products. Partial lifetime checkpoints are not
yet part of the native output contract.

## Native lifetime output v1

The registered lifetime product records:

- external fractional k points and physical magnon energies;
- the complex on-shell self-energy;
- HWHM, FWHM, rate, and lifetime with the conventions and units above;
- temperature, broadening, spin state, exchange/dJ/phonon source paths,
  exchange representation/kernel, derivative ASR outcome, phonon mass/schema
  convention, derivative space-group covariance provenance,
  explicit/static/zero-filled derivative bond counts, k mesh/shift, union `k+q`
  mesh, streaming algorithm, and MPI size.

The separate typed diagnostic API can still materialize full signed
internal-channel arrays and q/mode-resolved contributions for regression work;
the production streaming path does not retain them. Serializing such optional
diagnostics requires a schema-version increment.

One canonical scientific container and a small human-readable run summary are
preferred over per-rank final files. Parallel shards are temporary products
and must be merged deterministically before the run is considered complete.
Generated HDF5/NPZ data, plots, logs, caches, and scheduler output remain
outside version control.

## Registered QE-style lifetime input

The native lifetime calculation consumes the three screened scientific
products directly; it does not use a compatibility manifest or nested flat
input file:

```fortran
&control
  calculation = 'lifetime',
  prefix = 'sample',
  outdir = './slw-tmp'
/
&parallel
  execution = 'auto',
  workers_per_rank = 1,
  threads_per_worker = 4,
  q_chunk_size = 64,
  vertex_q_chunk_size = 64,
  self_energy_q_chunk_size = 64
/
&magph
  exchange_h5 = './input/J.h5',
  derivative_h5 = './input/dJ.h5',
  phonon_cache = './input/phonons.npz',
  phonon_epr = './input/sample_epr.h5',
  phonon_qmesh = 8, 8, 8,
  phonon_loto = 'auto',
  magnetic_order = 'fm',
  spin_magnitudes = 2.5,
  spin_pattern = 1,
  quantization_axis = 0.0, 0.0, 1.0,
  ! The same complete anisotropy_* set accepted by dispersion is optional.
  kmesh = 12, 12, 8,
  kshift = 0.5, 0.5, 0.5,
  temperature_k = 300.0,
  broadening_mev = 0.2,
  output = '${savedir}/sample.lifetime.npz'
/
```

`execution='auto'` selects all discovered MPI ranks. AFM input changes
`magnetic_order` to `collinear_afm` and, in the initial gate, supplies exactly
two opposite spin-pattern entries. `kshift` is deliberately explicit because
an exact AFM Goldstone point requires a separate regularization policy. Set at
least one of `vertex_q_chunk_size` or `self_energy_q_chunk_size` to bound the
largest streamed q block; when both are present the smaller value is used.

## Migration roadmap and legacy boundary

1. Completed: typed static/dynamic exchange and phonon screening, isotropic
   LSWT/vertex construction with optional uniaxial SIA, exact-Goldstone magnon
   dispersion, self-energy/lifetime, MPI k distribution, and registered
   native dispersion/lifetime outputs for arbitrary-N FM and bipartite AFM.
2. Add generated HDF5 end-to-end fixtures and real-material opt-in parity
   artifacts without using the dimensionally inconsistent legacy absolute
   linewidth as an oracle.
3. Add restartable output and optional q/mode-resolved contribution storage.
4. Extend the same contracts to full tensor exchange, resolved scattering,
   and subsequently hybrid/spin-torque methods.

Code under `slw/magph/legacy` and `slw/magph/legacy/reference` is retained as a
parity oracle and compatibility backend. New native code must not make those
modules its public API or rely on their flat manifest as its long-term data
model. The lifetime registry no longer enters this archive. Other legacy
calculations remain available until a native replacement has formula, unit,
FM/AFM, and representative-output parity tests; quarantine is not evidence
that the replacement is already complete.
