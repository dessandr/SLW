# SLW stage input reference

This document is the exhaustive input reference for the four public stage
executables. Exchange inputs are derived from its native typed schema; tables
for stages still under migration are derived from their registered parsers.
All entries are reviewed against runtime validation and output writers.

Use it together with [CLI.md](CLI.md), which explains the workflow and MPI
model. The examples under [`examples/`](../examples/README.md) are syntax
templates only; they do not define material defaults.

- [`slw_epr.x`](#slw_eprx)
- [`slw_exchange.x`](#slw_exchangex)
- [`slw_magph.x`](#slw_magphx)
- [`slw_post.x`](#slw_postx)
- [Compatibility nested input files](#compatibility-nested-input-files)

## How to read the tables

- A namelist runs exactly one `calculation`.
- Put backend parameters in `&epr`, `&exchange`, `&magph`, or `&post`. `&input`
  is a generic alias for the stage group, but the same key must not appear in
  both groups.
- Namelist keys are case-insensitive and shown in normalized lower-case form.
  Arrays use comma-separated Fortran values; booleans use `.true.` or `.false.`.
- **Parser required** means the registered compatibility parser rejects the
  input before numerical work. In the exchange section, the equivalent column
  describes native-schema validation instead.
  **Runtime requirements** below each calculation include conditional inputs
  that the retained backend validates later.
- `none / runtime` means the parser supplies `None`; a compatibility config or
  backend-specific runtime rule may then provide a value. Read that row's help
  and the calculation's runtime-requirement paragraph.
- `cli_args` or `legacy_args` can append a quoted raw argument string as an
  emergency compatibility escape hatch. They bypass named-key discovery and
  should not be used in new production inputs.
- Some retained parsers never supplied help text for an option. Those rows are
  marked “Compatibility input retained by the backend”; their normalized key,
  type, required state, and exact parser default are still reported, while the
  calculation's runtime paragraph gives the physical constraints.

Common naming used in those retained rows:

| Pattern | Meaning |
|---|---|
| `*_h5`, `*_npz`, `*_json`, `*_dir`, `*_name` | Input/output format, directory, or basename indicated by the suffix. |
| `kmesh`, `qmesh`, `mesh`, `kpoints` | Reciprocal-space sampling; integer arrays are mesh dimensions and float arrays are fractional coordinates unless stated otherwise. |
| `*_unit`, `*_mev`, `*_ang` | Unit selection, energy in meV, or length/tolerance in angstrom. |
| `*_tol`, `*_tolerance`, `symprec`, `threshold` | Numerical or symmetry acceptance threshold. |
| `workers`, `*_chunk`, `blocks` | Retained backend spelling for local parallelism or vectorized work partitioning. Public QE-style inputs use the canonical `&parallel` keys below. |
| `s`, `spin_s`, `temperature_k`, `eta_mev` | Spin magnitude, temperature in kelvin, and broadening in meV. |
| `dj_asr`, `*_asr_*` | Exchange-derivative acoustic-sum-rule mode or tolerance. |
| `cmap`, `vmin`, `vmax`, `dpi`, `fig_*`, `panel_*` | Plot color and layout controls. |

## Common frontend input

### `&control`

| Key | Type | Required | Default | Meaning |
|---|---|---:|---|---|
| `calculation` | string | yes | — | Calculation name from the executable's registry. |
| `prefix` | string | no | `'slw'` | File-name prefix used to form `outdir/prefix.save`; path separators are rejected. |
| `outdir` | path | no | `'.'` | Workflow output directory, resolved from the launch directory. |
| `verbosity` | enum `{quiet, normal, high, debug}` | no | `'normal'` | Frontend log detail. `high` prints translated arguments; `debug` also prints backend tracebacks. |
| `dry_run` | boolean | no | `.false.` | Validate the namelist and backend arguments without calculation or directory creation. |
| `create_save` | boolean | no | `.true.` | Create `outdir/prefix.save` immediately before a real calculation. |

### `&parallel`

| Key | Type | Required | Default | Meaning |
|---|---|---:|---|---|
| `execution` | enum `{auto, serial, mpi}` | no | `'auto'` | `auto` selects a registered MPI backend only in a multi-rank world; serial backends run on rank 0 only. `mpi` rejects calculations without MPI support. |
| `workers_per_rank` | int | no | `1` | Local process workers created by each MPI rank. Native magph lifetime currently requires one; compatibility hybrid and phonon preparation translate this to their local pools. |
| `threads_per_worker` | int | no | `1` | Rank/worker compute threads. This sets the OpenMP limit and is the default for BLAS and Numba unless a safer algorithm-specific limit or an explicit override applies. |
| `precache_workers` | int | no | `workers_per_rank` | Scalar-exchange-derivative cache construction threads. Other calculations reject it. |
| `blas_threads` | int | no | algorithm-specific | Explicit BLAS threads per rank/worker. Nested Numba exchange and magph kernels default to one to prevent oversubscription. |
| `numba_threads` | int | no | `threads_per_worker` | Explicit Numba threads per rank. Only calculations with a Numba-parallel kernel consume it directly. |
| `q_chunk_size` | int | no | full local q range | q block for mode-coupling or retained rotational kernels. |
| `bond_chunk_size` | int | no | all bonds | Bond block for mode-coupling, scattering, or retained rotational kernels. |
| `vertex_q_chunk_size` | int | no | full local q range | q block for band-basis vertex assembly or retained q-BZ scattering. |
| `self_energy_q_chunk_size` | int | no | full local q range | q block accumulated into native self-energy. |
| `channel_chunk_size` | int | no | all external channels | External-channel block for native on-shell self-energy. |

All integer resource and chunk values must be positive. Historical spellings
such as `nproc`, `omp_threads`, `phonon_nproc`, `hybrid_nproc`, `num_threads`,
`q_chunk`, and `vertex_q_chunk` are rejected in public namelists with the
canonical replacement. Physical integration controls such as `integrator`,
`empoints`, and `g_kernel` remain in the stage group.

### Path placeholders

`${prefix}`, `${outdir}`, and `${savedir}` are expanded recursively in stage
parameters. They are derived only from `&control`; arbitrary environment
variables are not expanded.

The native exchange engine defaults its products to `${savedir}`. Creating the
directory does not redirect other retained backends' relative output defaults;
set their `out`, `output`, `out_dir`, or `out_prefix` explicitly when all
products must live under `${savedir}`.


## `slw_epr.x`

QE/qe2pert EPR preparation, phonon-cache construction, and path/dispersion validation.

| `calculation` | Backend source | MPI | Purpose |
|---|---|---:|---|
| `gkq` | default | no | Reconstruct Wannier-gauge g(k,q) from a QE qe2pert EPR file |
| `dispersion` | default | no | Validate electronic and phonon dispersions stored in EPR data |
| `phonon_cache` | default | no | Build the reusable phonon cache consumed by magph calculations |
| `kpath` | default | no | Generate an explicit high-symmetry path from a structure |

### `calculation='gkq'`

**Runtime requirements:** `epr` is required. If `rotate=.true.`, provide at
least one of `u_mat` and `u_dis_mat`; when both are supplied the backend
combines them. Every k+q point must be present on the EPR k mesh.

**Outputs:** HDF5 containing LZF-compressed
`g_wannier[Nq,Nk,Nw,Nw,3Nat]`, or `g_band[Nq,Nk,Nb,Nb,3Nat]` plus `kq_map`
when rotation is requested. `out` defaults to a file in the launch directory,
overwrites an existing file, and does not create its parent directory; prefer
`out='${savedir}/gkq.h5'`.

Backend: `slw.epc.compute_gkq_from_epr`.

| Namelist key | Type | Parser required | Parser default | Meaning / CLI aliases |
|---|---|---:|---|---|
| `epr` | string | yes | — | Input EPR HDF5 path<br>CLI aliases: `--epr` |
| `out` | string | no | `gkq_wannier_from_epr.h5` | Output HDF5 path<br>CLI aliases: `--out` |
| `u_mat` | string | no | none / runtime | wannier90_u.mat path (optional)<br>CLI aliases: `--u_mat` |
| `u_dis_mat` | string | no | none / runtime | wannier90_u_dis.mat path (optional)<br>CLI aliases: `--u_dis_mat` |
| `rotate` | boolean | no | .false. | Apply U_{k+q}^dagger g U_k rotation and write dataset g_band<br>CLI aliases: `--rotate` |
| `nproc` | int | no | `1` | Thread parallelism for rotation over q points<br>CLI aliases: `--nproc` |
| `mode_block` | int | no | `6` | Mode block size for batched einsum rotation<br>CLI aliases: `--mode_block` |

### `calculation='dispersion'`

**Runtime requirements:** Select one source mode: `epr`, or the complete pair
`epr_up` + `epr_dn`. Dual-spin HR comparison requires both `hr_up` and
`hr_dn`; MDRS comparison additionally requires both `wsvec_up` and
`wsvec_dn`. The frontend requires one path source: `win`, `kpath`, or
`compare_phdisp`. Prefer a `.win`, a coordinate-form `kpath`, or a reference
dispersion; label-only paths use retained built-in hexagonal coordinates.
`band_ref='vbm'` requires `nvalence`, and a physical
`band_ref='fermi'` run should set `efermi` explicitly. These conditional
pairings are checked by the backend at runtime, not by `--dry-run`.

**Outputs:** eV electronic-band tables, meV phonon tables, a dispersion NPZ,
and optional PNG plots under `out_prefix`. Dual mode writes separate up/down
band products. Without `out_prefix`, products are written relative to the
launch directory and may overwrite existing files.

Backend: `slw.epc.check_qe2pert_epr_dispersion`.

| Namelist key | Type | Parser required | Parser default | Meaning / CLI aliases |
|---|---|---:|---|---|
| `epr` | string | no | none / runtime | Input prefix_epr.h5 for single-spin mode<br>CLI aliases: `--epr` |
| `epr_up` | string | no | none / runtime | Input up-spin prefix_epr.h5<br>CLI aliases: `--epr_up` |
| `epr_dn` | string | no | none / runtime | Input down-spin prefix_epr.h5<br>CLI aliases: `--epr_dn` |
| `hr_up` | string | no | none / runtime | Optional up-spin Wannier90 *_hr.dat for band plotting<br>CLI aliases: `--hr_up` |
| `hr_dn` | string | no | none / runtime | Optional down-spin Wannier90 *_hr.dat for band plotting<br>CLI aliases: `--hr_dn` |
| `wsvec_up` | string | no | none / runtime | Optional up-spin Wannier90 *_wsvec.dat for MDRS band phases<br>CLI aliases: `--wsvec_up` |
| `wsvec_dn` | string | no | none / runtime | Optional down-spin Wannier90 *_wsvec.dat for MDRS band phases<br>CLI aliases: `--wsvec_dn` |
| `win` | string | no | none / runtime | Wannier .win file with kpoint_path<br>CLI aliases: `--win` |
| `kpath` | string | no | none / runtime | Custom path. Label form: 'G M K G' using built-in hexagonal labels. Coordinate form: 'G 0 0 0 X 0.5 0 0'. Use 'win' or omit to read --win/default.<br>CLI aliases: `--kpath` |
| `nseg` | int | no | `40` | Interpolated points per path segment<br>CLI aliases: `--nseg` |
| `asr` | optional enum {none, simple, crystal} | no | `none` | ASR mode: --asr is simple; --asr crystal applies spglib crystal IFC symmetrization plus ASR.<br>CLI aliases: `--asr`, `--sumrule` |
| `fc_symmetry` | boolean | no | .false. | Apply SLW fallback force-constant symmetrization: permutation symmetry plus iterative ASR in real space.<br>CLI aliases: `--fc_symmetry` |
| `fc_symmetry_iter` | int | no | `5` | Iterations for --fc_symmetry permutation+ASR projection.<br>CLI aliases: `--fc_symmetry_iter` |
| `no_loto` | boolean | no | .false. | Disable polar/nonanalytic LO-TO correction even when epr.h5 has lpolar=true.<br>CLI aliases: `--no_loto` |
| `loto_dim` | enum {auto, 2d, 3d, none} | no | `auto` | Override LO-TO correction dimensionality. auto follows epr.h5 system_2d; none disables LO-TO.<br>CLI aliases: `--loto_dim` |
| `no_asr_gamma_project` | boolean | no | .false. | Disable the extra Gamma acoustic-subspace projection used with --asr.<br>CLI aliases: `--no_asr_gamma_project` |
| `gamma_tol` | float | no | `1e-10` | Fractional-coordinate tolerance for treating q as Gamma in the ASR projector.<br>CLI aliases: `--gamma_tol` |
| `symprec` | float | no | `1e-05` | spglib symmetry tolerance for --asr crystal.<br>CLI aliases: `--symprec` |
| `debug_gamma` | boolean | no | .false. | Print Gamma-like q indices and acoustic frequencies before ASR projection.<br>CLI aliases: `--debug_gamma` |
| `out_prefix` | string | no | none / runtime | Compatibility input retained by the backend.<br>CLI aliases: `--out_prefix` |
| `plot` | boolean | no | .false. | Write PNG plots<br>CLI aliases: `--plot` |
| `band_ref` | enum {zero, fermi, vbm, max} | no | `zero` | Plot-only electronic energy reference. max reproduces the legacy top-of-Wannier-window shift.<br>CLI aliases: `--band_ref` |
| `efermi` | float | no | `0.0` | Fermi energy in eV used by --band_ref fermi.<br>CLI aliases: `--efermi` |
| `nvalence` | int | no | none / runtime | Occupied band count per spin used by --band_ref vbm.<br>CLI aliases: `--nvalence` |
| `compare_phdisp` | string | no | none / runtime | Optional Perturbo *.phdisp reference for phonon comparison<br>CLI aliases: `--compare_phdisp` |

### `calculation='phonon_cache'`

**Runtime requirements:** `epr`, a positive three-component `qmesh`, and a new
`output` path are required. `qshift` must be a finite three-vector and
`nproc > 0`. When `phonon_fc` is used, its force-constant mesh must agree with
the EPR `qc_dim`. The backend refuses to overwrite an existing cache.

**Outputs:** Atomic, no-clobber schema-v3 NPZ containing fractional/cartesian q
points, meV frequencies, mass-normalized complex eigenvectors, geometry,
units, phase convention, and ASR/LO-TO/source provenance.

Backend: `slw.magph.legacy.reference.build_phonon_cache`.

| Namelist key | Type | Parser required | Parser default | Meaning / CLI aliases |
|---|---|---:|---|---|
| `epr` | string | yes | — | Input qe2pert/Perturbo EPR HDF5<br>CLI aliases: `--epr`, `--epr_phonon` |
| `qmesh` | parsed string[3] | yes | — | Three positive dimensions of the uniform q mesh<br>CLI aliases: `--qmesh`, `--phonon_qmesh` |
| `qshift` | parsed string[3] | no | `[0.0, 0.0, 0.0]` | q-mesh shift in grid-index units, canonicalized modulo integers<br>CLI aliases: `--qshift`, `--phonon_qshift` |
| `nproc` | parsed string | no | `1` | Worker processes for q-point diagonalization (default: 1)<br>CLI aliases: `--nproc`, `--phonon_nproc` |
| `asr` | enum {none, simple, crystal} | no | `none` | Acoustic sum rule (default: none)<br>CLI aliases: `--asr`, `--phonon_asr` |
| `loto` | enum {auto, 2d, 3d, none} | no | `auto` | LO-TO treatment (default: auto)<br>CLI aliases: `--loto`, `--phonon_loto` |
| `phonon_fc` | string | no | none / runtime | Optional QE q2r force-constant file used instead of EPR IFCs<br>CLI aliases: `--phonon-fc` |
| `compressed` | boolean | no | .true. | Use compressed NPZ output (default: compressed)<br>CLI aliases: `--compressed`, `--no-compressed` |
| `output` | string | yes | — | New output NPZ; an existing path is never overwritten<br>CLI aliases: `--output`, `-o` |

### `calculation='kpath'`

**Runtime requirements:** `input` must name a POSCAR-like or QE structure
readable by the selected backend. `backend='seekpath'` requires the optional
Seekpath dependency. File outputs are optional; if neither `qe_out` nor
`win_out` is set, snippets are printed only to stdout.

**Outputs:** Optional QE `K_POINTS crystal_b` and Wannier90 `kpoint_path` files; otherwise stdout snippets.

Backend: `slw.core.make_kpath`.

| Namelist key | Type | Parser required | Parser default | Meaning / CLI aliases |
|---|---|---:|---|---|
| `workdir` | string | no | none / runtime | Compatibility input retained by the backend.<br>CLI aliases: `--workdir` |
| `input` | string | yes | — | POSCAR-like file or QE scf input<br>CLI aliases: `--input` |
| `backend` | enum {auto, seekpath, fallback} | no | `auto` | Compatibility input retained by the backend.<br>CLI aliases: `--backend` |
| `symprec` | float | no | `1e-05` | Compatibility input retained by the backend.<br>CLI aliases: `--symprec` |
| `points_per_segment` | int | no | `50` | Compatibility input retained by the backend.<br>CLI aliases: `--points-per-segment` |
| `qe_out` | string | no | none / runtime | Output QE K_POINTS crystal_b file<br>CLI aliases: `--qe-out` |
| `win_out` | string | no | none / runtime | Output Wannier90 kpoint_path snippet<br>CLI aliases: `--win-out` |

## `slw_exchange.x`

Static exchange and analytic exchange-derivative calculations use one native
engine. `calculation` selects `j` or `dj`; `ltensor` selects the scalar or full
tensor convention. `input_format` is always explicit. Historical
`j_tensor`/`dj_tensor` calculation names are accepted only as deprecated
aliases that inject `ltensor=.true.`.

| `calculation` | `ltensor` | Input source | MPI | Purpose |
|---|---:|---|---:|---|
| `j` | `.false.` | epr, wannier | yes | Scalar collinear LKAG exchange |
| `j` | `.true.` | epr, wannier | yes | Full exchange tensor |
| `dj` | `.false.` | epr | yes | Scalar analytic dJ/du |
| `dj` | `.true.` | epr | yes | Tensor analytic dJ/du |

All four modes require `input_format`, `efermi`, a positive three-component
`kmesh`, and explicit `mag_atoms`. Non-overlapping `slices` are required except
for spinor Wannier tensor J using either `win` + `centres` or the complete
AMN/EIG/SPN/U/U_dis projection bundle. The former assigns Wannier centres to
atoms; the latter constructs the magnetic atomic frame from `win` projections
and AMN. EPR modes require `epr_up` and `epr_dn`. Scalar Wannier J requires
`up_hr` and `dn_hr`; tensor Wannier J accepts that pair or one `spinor_hr`.
Native filename stems are
`${prefix}.j`, `${prefix}.j_tensor`, `${prefix}.dj`, and
`${prefix}.dj_tensor` under `${savedir}`. `out_dir`, `out_h5`, and `out_name`
override those paths.

With `execution='auto'`, a launch containing more than one rank uses MPI for
every mode. Scalar/tensor J partition the contour or pole energy mesh. Scalar
dJ first distributes target/displacement-axis EPC-cache ownership; when MPI
ranks outnumber those tasks, the excess ranks share that task's energy mesh.
Tensor dJ partitions target/displacement-axis tasks. Rank 0 writes the final
HDF5/text products after the collective reduction. Set
`workers_per_rank=1` under MPI for every mode except scalar dJ. Scalar dJ
supports a hybrid route where
each rank builds only its assigned EPC cache entries, publishes those arrays in
POSIX shared memory, and runs `workers_per_rank` clean local worker processes.
The maximum rank/node cache storage is printed before integration and recorded
in the HDF5 provenance. The launcher affinity
assigned to each rank must contain at least
`workers_per_rank*threads_per_worker` CPUs, and the node-local
POSIX shared-memory filesystem must be large enough for that rank's cache. In
a serial launch, `workers_per_rank>1` uses the copy-on-write local
multiprocessing path.

| Native namelist key | Type | Required | Default | Meaning |
|---|---|---:|---|---|
| `input_format` | enum {epr, wannier} | yes | — | Hamiltonian source; `dj` accepts only `epr`. |
| `ltensor` | boolean | no | `.false.` | Select the full tensor convention rather than scalar LKAG. |
| `tensor_kernel` | enum {tb2j, direct} | tensor J only | `tb2j` | Tensor-J numerical convention. Tensor dJ currently supports TB2J only. |
| `efermi` | float | yes | — | Fermi energy in eV. |
| `kmesh` | int[3] | yes | — | Positive uniform electronic mesh. |
| `mag_atoms` | list[int] | yes | — | Explicit magnetic atom indices. |
| `mag_atoms_base` | enum {0, 1} | no | `0` | Input atom-index base; normalized internally to zero-based. |
| `slices` | string | conditional | — | Non-overlapping `site:start:stop` orbital ranges. Omit for projection-anchored spinor Wannier tensor J; it is also optional for the separate `win` + `centres` matching path. |
| `out_dir` | path | no | `${savedir}` | Product directory. |
| `out_h5` | path | no | mode-derived | HDF5 product path. |
| `out_name` | path | no | mode-derived | Text product path when the mode writes one. |

### Spinor basis and atomic SOC

Without the projection bundle, raw `spinor_hr` input must declare exactly one
HR-row TB2J layout:

| Key | Allowed values | Meaning |
|---|---|---|
| `groupby` | `spin` | `[all up orbitals | all down orbitals]` |
| `groupby` | `orbital` | `[orb1 up, orb1 down, orb2 up, orb2 down, ...]` |

SLW never infers `groupby` from `wannier_centres.xyz`. On this raw-HR path both
layouts are converted internally to `groupby='spin'`. The projection-bundle
path is different: there `groupby` describes AMN trial columns and HR rows are
neither reordered nor reinterpreted. For spinor Wannier tensor J, explicit
`win` + `centres` may instead infer the magnetic orbital indices: each
spinless orbital centre is assigned to its nearest atom under periodic boundary
conditions, and `mag_atoms` selects the resulting site blocks. This matches the
TB2J centre-assignment model and does not reconstruct projection mixing through
Wannier gauge matrices. `slices` remains a manual override. By default nearest
assignment has no distance cutoff; set positive `centre_tolerance_ang` to fail
when an assigned centre is farther than that Cartesian distance from its atom.
The same spin-layout permutation is applied to HR rows, HR columns, and
Wannier-centre rows when collinear up/down inputs are exported as a spinor
model. A centre export requires both spin-channel centre files; atomic rows
must match and are written once.

#### Projection-anchored atomic-Pauli frame (SPN-validated)

For a spinor HR whose maximally-localized Wannier rows are not an independently
known orbital-spin product basis, supply this complete five-file bundle:

| Key | Wannier90 file | Role |
|---|---|---|
| `amn` | `seedname.amn` | Anchors the magnetic atomic projection columns declared by `win`. |
| `eig` | `seedname.eig` | Reconstructs the Hamiltonian before the Wannier rotations. |
| `spn` | `seedname.spn` | Supplies reference Pauli matrices in the Bloch eigenstate basis for fail-closed validation. |
| `u_mat` | `seedname_u.mat` | Final Wannier-gauge rotation. |
| `u_dis_mat` | `seedname_u_dis.mat` | Disentanglement rotation. |

The bundle is all-or-nothing and is accepted only for spinor Wannier tensor J
with an explicit `win`, `tensor_kernel='tb2j'`, and no additional `SOC
(atomic)` card. Here `groupby` declares AMN trial-column spin ordering; it does
not reinterpret HR rows. Select
`spin_operator='auto'` or `'spn'`; both use the supplied AMN-anchored,
SPN-validated atomic-Pauli path when the bundle is complete, while `'pauli'` is
rejected. `u_dis_layout` is
mandatory: use `global_bands` when U_dis rows already carry the original EIG
band indices, or `compact_outer_window` for the old packed outer-window layout.
The compact layout requires `dis_win_min` and `dis_win_max` in `win`.

Without the bundle, `spin_operator='auto'` rejects raw `spinor_hr`: a
block-diagonal or rank-one Pauli diagnostic cannot prove that independently
Wannierized spin sectors share the transverse orbital-partner gauge.
`spin_operator='pauli'` is therefore an explicit unsafe opt-in reserved for a
basis whose common orbital-spin product structure is known independently.

The magnetic subspace is derived from `mag_atoms`, the projection groups in
`win`, and AMN. Omit `slices`; manual slices are rejected. A `centres` file is
not used and is rejected. The Green function still propagates
through the full spinor Wannier Hamiltonian; only its magnetic endpoints are
expressed in the projection-anchored atomic frame.

SPN is transformed into that frame and checked against the atomic-trial Pauli
matrices, including a per-axis residual. It is a fail-closed validation, not a
k-dependent exchange vertex; the accumulator uses the atomic-trial Pauli
operator. The chosen tolerance therefore controls a documented projection
approximation and its measured residuals are stored in
`spin_operator_validation`.

The site-resolved magnetic directions are inferred from the projected
time-reversal-odd field, so `spin_direction` does not set the axis on this
path. They are stored in `magnetic_subspace/site_spin_directions`; the legacy
`basic_data/spin_direction` field records site 0's inferred direction and its
`spin_direction_source` identifies that provenance.

No k-space interpolation is performed for this path. `kmesh` dimensions must
reproduce the native U/AMN/SPN grid. The actual fractional coordinates,
ordering, and any uniform Monkhorst-Pack shift are read from U; U_dis must use
the same order and the native grid must be closed under k -> -k. The coordinates
used are stored in `basic_data/native_kpoints_crystal` in the HDF5 output. U, U_dis,
EIG, SPN, AMN, HR, and `win` must all come from the same Wannierization.
The loader fails closed when their dimensions, gauge reconstruction, projection
rank, SPN-to-atomic-Pauli validation, or magnetic-field locality checks disagree.

The validation thresholds are positive dimensionless numbers unless a unit is
shown:

| Key | Default | Validation |
|---|---:|---|
| `projection_rank_tolerance` | `1.0e-4` | Minimum allowed singular value of the selected magnetic AMN projection. |
| `spin_projection_tolerance` | `0.4` | Maximum combined or per-axis relative SPN residual in the projection-anchored atomic Pauli frame. |
| `hamiltonian_tolerance_ev` | `1.0e-4` eV | Maximum elementwise residual between HR and the U_dis/U/EIG reconstruction. |
| `noncollinear_tolerance` | `0.25` | Maximum residual from one collinear time-reversal-odd field direction per magnetic site. |
| `intersite_xc_tolerance` | `0.1` | Maximum nonlocal fraction of the time-reversal-odd field, applied to both intersite blocks and site-block k dependence. |

`collinear_override=.true.` is a diagnostic correction for a Hamiltonian known
independently to be both no-SOC and collinear. It replaces the inferred
longitudinal component by the transverse average along the detected magnetic
axis. Leave it `.false.` for every SOC calculation, because enabling it would
remove physical anisotropy.

Additional onsite SOC is a final QE-style card, not a namelist option:

```fortran
&control
  calculation = 'j'
/
&exchange
  input_format = 'wannier',
  ltensor = .true.,
  spinor_hr = 'model_hr.dat',
  groupby = 'orbital',
  win = 'model.win',
  ...
/
SOC (atomic)
  LAMBDA Te-p  0.50
  LAMBDA Mn1-d 0.05
```

`LABEL-p` and `LABEL-d` select Wannier90 projection manifolds. A species label
such as `Te-p` applies to every matching Te site; a site label such as `Mn1-d`
applies only to that site. Values are in eV. Overlapping or unmatched selectors,
missing projection metadata, and inconsistent projection counts are errors.
Wannier90 real-harmonic ordering is fixed internally; manual orbital indices
and custom p/d order controls are not accepted. No card means no *additional*
SOC. A `spinor_hr` may therefore represent either a SOC or SOC-free
noncollinear Hamiltonian, and an SOC card may add a separate term to either.

The historical keys `spinor_basis_order`, `basis_order`, `soc`, `lambda_te`,
`soc_p_groups`, `p_order`, `d_order`, `intersite_soc`, and `dynamic_soc` are not
part of the native input schema. Any such rows in archived backend documentation
refer only to quarantined code and are rejected by `slw_exchange.x`.

The option tables below retain detailed kernel tuning fields. The old parser
default shown for an output field is historical; the native output defaults
in the paragraph above take precedence. `kernel` is a compatibility alias for
native `tensor_kernel`; neither may be supplied when `ltensor=.false.`.

### `calculation='j', ltensor=.false.`

**Runtime requirements:** Set `input_format='epr'` or `'wannier'`. EPR requires
`epr_up`, `epr_dn`, `efermi`, `kmesh`, `slices`, and at least two
`mag_atoms`; spin-channel H(k) shapes and every slice bound must agree.
Wannier scalar J requires both `up_hr` and `dn_hr`, a non-empty `slices`,
`efermi`, `kmesh`, and `mag_atoms`; `spinor_hr` and all SOC additions are
rejected by the scalar kernel. Supplying `win` avoids ambiguous structure
discovery for bond construction.

**Outputs:** `${savedir}/${prefix}.j.txt` and `.h5`; EPR also writes the
derived `.all_bonds.tsv` table. By default, EPR scalar J is averaged only
within each bond orbit validated by spglib. The projected source-convention
values are stored in `J_r/value`, while the unmodified integration values
remain in `J_r/value_raw`; `J_r/projection_delta` and the `symmetry/` group
record the residual and provenance. Both datasets retain the mate-complete
source directed-bond weight recorded in `basic_data/directed_bond_weight`;
they are not silently canonicalized. Distinct symmetry orbits in the same
distance shell remain distinct. `no_h5=.true.` disables scalar EPR HDF5.
Wannier scalar J is currently stored in an isotropic `J_tensor_r` envelope
with the same explicit source weight.

#### `input_format='epr'`

Native engine: `slw.exchange.engine`; numerical kernel:
`slw.exchange.kernels.j_epr`.

| Namelist key | Type | Parser required | Parser default | Meaning / CLI aliases |
|---|---|---:|---|---|
| `epr_up` | string | yes | — | Advanced native-kernel option.<br>CLI aliases: `--epr_up` |
| `epr_dn` | string | yes | — | Advanced native-kernel option.<br>CLI aliases: `--epr_dn` |
| `atom_labels` | string | no | `''` | Comma-separated labels for EPR atoms; default Atom1,Atom2,...<br>CLI aliases: `--atom_labels` |
| `species_labels` | string | no | `''` | Comma-separated species for spglib orbit grouping, e.g. Mn,Mn,Te,Te.<br>CLI aliases: `--species_labels` |
| `hr_unit` | enum {ev, ry, ha} | no | `ry` | Advanced native-kernel option.<br>CLI aliases: `--hr_unit` |
| `mag_atoms` | list[int] | yes | — | Advanced native-kernel option.<br>CLI aliases: `--mag_atoms` |
| `mag_atoms_base` | enum {0, 1} | no | `0` | Advanced native-kernel option.<br>CLI aliases: `--mag_atoms_base` |
| `slices` | string | yes | — | Local slices as '0:0:5,1:5:10'<br>CLI aliases: `--slices` |
| `efermi` | float | yes | — | Advanced native-kernel option.<br>CLI aliases: `--efermi` |
| `kmesh` | int[3] | yes | — | Advanced native-kernel option.<br>CLI aliases: `--kmesh` |
| `n_shells` | int | no | `10` | Advanced native-kernel option.<br>CLI aliases: `--n_shells` |
| `d_max` | float | no | `20.0` | Advanced native-kernel option.<br>CLI aliases: `--d_max` |
| `emin` | float | no | `-25.0` | Advanced native-kernel option.<br>CLI aliases: `--emin` |
| `empoints` | int | no | `500` | Advanced native-kernel option.<br>CLI aliases: `--empoints` |
| `integrator` | enum {contour, cfr, cfr_ozaki} | no | `contour` | Advanced native-kernel option.<br>CLI aliases: `--integrator` |
| `cfr_beta` | float | no | `400.0` | Advanced native-kernel option.<br>CLI aliases: `--cfr_beta` |
| `symprec` | float | no | `0.0001` | spglib symmetry tolerance for orbit grouping.<br>CLI aliases: `--symprec` |
| `angle_tolerance` | float | no | `-1.0` | spglib angle tolerance in degrees; -1 uses spglib default.<br>CLI aliases: `--angle_tolerance` |
| `orbit_grouping` | enum {spglib, shell} | no | `spglib` | Orbit grouping mode. shell groups all bonds with the same shell index and distance.<br>CLI aliases: `--orbit_grouping` |
| `orbit_symmetry` | enum {report, project, fail} | no | `project` | `project` replaces every scalar J in a validated spglib orbit by that orbit's mean. `report` preserves raw values; `fail` rejects a raw within-orbit deviation above `orbit_symmetry_tolerance_mev`. Projection/failure are forbidden for shell, disabled, or fallback grouping.<br>CLI aliases: `--orbit_symmetry` |
| `orbit_symmetry_tolerance_mev` | float | no | `1.0e-8` | Non-negative tolerance used by `orbit_symmetry='fail'`; also recorded as projection provenance.<br>CLI aliases: `--orbit_symmetry_tolerance_mev` |
| `debug_orbits` | boolean | no | .false. | Print spglib operation and bond-mapping diagnostics.<br>CLI aliases: `--debug_orbits` |
| `debug_orbit_shell` | int | no | none / runtime | Restrict --debug_orbits bond diagnostics to one shell.<br>CLI aliases: `--debug_orbit_shell` |
| `debug_epr_positions` | boolean | no | .false. | Print EPR tau and Wannier-center position diagnostics.<br>CLI aliases: `--debug_epr_positions` |
| `debug_shell` | int | no | none / runtime | Write per-bond LKAG debug TSV for one shell, e.g. 2 for J2.<br>CLI aliases: `--debug_shell` |
| `debug_bond` | string | no | `''` | Write debug for one directed bond: gi,gj,R1,R2,R3.<br>CLI aliases: `--debug_bond` |
| `debug_out` | string | no | `J_debug_pairs.tsv` | Debug TSV filename inside --out_dir.<br>CLI aliases: `--debug_out` |
| `no_symmetry_orbits` | boolean | no | .false. | Fallback to distance/pair orbit grouping.<br>CLI aliases: `--no_symmetry_orbits` |
| `out_dir` | string | no | `.` | Advanced native-kernel option.<br>CLI aliases: `--out_dir` |
| `out_name` | string | no | `J_epr_kspace.txt` | Advanced native-kernel option.<br>CLI aliases: `--out_name` |
| `out_h5` | string | no | none / runtime | Optional J_r HDF5 filename/path. Default: &lt;out_name basename&gt;.Jr.h5<br>CLI aliases: `--out_h5` |
| `no_h5` | boolean | no | .false. | Disable J_r HDF5 output.<br>CLI aliases: `--no_h5` |

#### `input_format='wannier'`

Native engine: `slw.exchange.engine`; numerical kernel:
`slw.exchange.kernels.j_wannier`.

Frontend-fixed values: `kernel='scalar'`. Supplying a conflicting value is an error.

| Namelist key | Type | Parser required | Parser default | Meaning / CLI aliases |
|---|---|---:|---|---|
| `up_hr` | string | yes | — | Spin-up collinear Wannier90 hr.dat.<br>CLI aliases: `--up_hr` |
| `dn_hr` | string | yes | — | Spin-down collinear Wannier90 hr.dat.<br>CLI aliases: `--dn_hr` |
| `efermi` | float | yes | — | Fermi energy in eV<br>CLI aliases: `--efermi` |
| `hr_unit` | enum {ev, ry, ha} | no | `ev` | Unit of input hr.dat matrix elements; Wannier90 default is eV<br>CLI aliases: `--hr_unit` |
| `ref_epr_up` | string | no | none / runtime | Optional reference EPR up HDF5 for H(k) scale/gauge diagnostics<br>CLI aliases: `--ref_epr_up` |
| `ref_epr_dn` | string | no | none / runtime | Optional reference EPR down HDF5 for H(k) scale/gauge diagnostics<br>CLI aliases: `--ref_epr_dn` |
| `ref_hr_unit` | enum {ev, ry, ha} | no | `ry` | Reference EPR hopping unit<br>CLI aliases: `--ref_hr_unit` |
| `kmesh` | int[3] | yes | — | Advanced native-kernel option.<br>CLI aliases: `--kmesh` |
| `mag_atoms` | list[int] | yes | — | Magnetic atom indices<br>CLI aliases: `--mag_atoms` |
| `mag_atoms_base` | enum {0, 1} | no | `0` | Advanced native-kernel option.<br>CLI aliases: `--mag_atoms_base` |
| `slices` | string | yes | — | Manual local orbital slices, e.g. `0:0:5,1:5:10`. |
| `apply_degeneracy` | boolean | no | .true. | Divide HR blocks by Wannier90 degeneracy before H(k) construction; standard Wannier90 needs this<br>CLI aliases: `--apply_degeneracy`, `--no-apply_degeneracy` |
| `axes` | string | no | `xyz` | Isotropic tensor-envelope axes used by the current scalar writer.<br>CLI aliases: `--axes` |
| `spin_direction` | float[3] | no | `[0.0, 0.0, 1.0]` | Collinear/model input direction. The projection-anchored path instead infers every site's direction from its time-reversal-odd field.<br>CLI aliases: `--spin_direction` |
| `win` | string | no | none | Optional explicit Wannier90 structure source; scalar J does not accept an SOC card. |
| `n_shells` | int | no | `10` | Advanced native-kernel option.<br>CLI aliases: `--n_shells` |
| `d_max` | float | no | `20.0` | Advanced native-kernel option.<br>CLI aliases: `--d_max` |
| `all_bonds` | boolean | no | .true. | Keep directed bonds; default matches compute_J_epr_tensor<br>CLI aliases: `--all_bonds`, `--canonical_bonds` |
| `nn_only` | boolean | no | .false. | Advanced native-kernel option.<br>CLI aliases: `--nn_only` |
| `orbit_grouping` | enum {none, distance, shell} | no | `distance` | Advanced native-kernel option.<br>CLI aliases: `--orbit_grouping` |
| `integrator` | enum {contour, cfr_ozaki, cfr_pole} | no | `contour` | Advanced native-kernel option.<br>CLI aliases: `--integrator` |
| `emin` | float | no | `-25.0` | Advanced native-kernel option.<br>CLI aliases: `--emin` |
| `empoints` | int | no | `500` | Advanced native-kernel option.<br>CLI aliases: `--empoints` |
| `cfr_beta` | float | no | `1000.0` | Advanced native-kernel option.<br>CLI aliases: `--cfr_beta` |
| `collinear_override` | boolean | no | .false. | Override J_zz with (J_xx+J_yy)/2 in collinear calculations<br>CLI aliases: `--collinear_override` |
| `spin_magnitude` | float | no | `1.0` | Spin magnitude S to scale J by 1/S^2<br>CLI aliases: `--spin_magnitude` |
| `out_dir` | string | no | `J_wannier_tensor` | Advanced native-kernel option.<br>CLI aliases: `--out_dir` |
| `out_name` | string | no | `J_wannier_tensor.txt` | Advanced native-kernel option.<br>CLI aliases: `--out_name` |
| `out_h5` | string | no | `J_wannier_tensor.h5` | Advanced native-kernel option.<br>CLI aliases: `--out_h5` |

### `calculation='j', ltensor=.true.`

**Runtime requirements:** Set `input_format='epr'` or `'wannier'`. EPR requires
`epr_up`, `epr_dn`, `efermi`, `kmesh`, `slices`, and at least two
`mag_atoms`. Wannier requires `efermi`, `kmesh`, `mag_atoms`, and either
`spinor_hr` or the complete `up_hr` + `dn_hr` pair. Explicit `slices` are
required except when raw spinor input supplies both `win` and `centres`, which
enables automatic nearest-atom assignment, or when it supplies `win` plus all
of `amn`, `eig`, `spn`, `u_mat`, and `u_dis_mat`, which enables the
AMN-anchored atomic-Pauli, SPN-validated path. Raw spinor input always requires
`groupby='spin'|'orbital'` regardless of how the magnetic indices are built.
For the projection bundle it describes AMN trial-column spin ordering, not
MLWF/HR row ordering; outside that path it retains the HR-row meaning.
The projection bundle requires `tensor_kernel='tb2j'`, an explicit
`u_dis_layout`, and matching native Wannier k-grid dimensions; it rejects manual `slices`
and an additional `SOC (atomic)` card. Outside that path, the optional SOC card
requires `win` and may be combined with collinear or spinor input; it always
denotes an additional onsite term.

**Outputs:** `${savedir}/${prefix}.j_tensor.txt` and `.h5` with
`J_tensor_r`, `J_iso_r`, `J_gamma_r`, `J_dmi_tensor_r`, and `DMI_r`.
Wannier tensor decomposition remains under `extra/` in the current parity
schema.

Set `tensor_kernel='tb2j'` (default) or `'direct'`. This switch changes the
physical kernel convention and is dispatched before numerical integration.

#### `input_format='epr'`

Native engine: `slw.exchange.engine`; numerical kernel:
`slw.exchange.kernels.j_tensor_epr`.

| Namelist key | Type | Parser required | Parser default | Meaning / CLI aliases |
|---|---|---:|---|---|
| `epr_up` | string | yes | — | Spin-up EPR HDF5<br>CLI aliases: `--epr_up` |
| `epr_dn` | string | yes | — | Spin-down EPR HDF5<br>CLI aliases: `--epr_dn` |
| `efermi` | float | yes | — | Fermi energy in eV<br>CLI aliases: `--efermi` |
| `kmesh` | int[3] | yes | — | Advanced native-kernel option.<br>CLI aliases: `--kmesh` |
| `mag_atoms` | list[int] | yes | — | Magnetic atom indices<br>CLI aliases: `--mag_atoms` |
| `mag_atoms_base` | enum {0, 1} | no | `0` | Advanced native-kernel option.<br>CLI aliases: `--mag_atoms_base` |
| `slices` | string | yes | — | Manual local orbital slices, e.g. '0:0:5,1:5:10'<br>CLI aliases: `--slices` |
| `hr_unit` | enum {ry, ev, ha} | no | `ry` | EPR hopping unit<br>CLI aliases: `--hr_unit` |
| `kernel` | enum {tb2j, direct} | no | `tb2j` | tb2j: TB2J-like Pauli A-tensor mapping. direct: raw D_i^a G D_j^b G trace.<br>CLI aliases: `--kernel` |
| `axes` | string | no | `xyz` | Tensor axes to compute, subset of xyz<br>CLI aliases: `--axes` |
| `spin_direction` | float[3] | no | `[0.0, 0.0, 1.0]` | Advanced native-kernel option.<br>CLI aliases: `--spin_direction` |
| `win` | string | with SOC card | none | Wannier90 projections used to resolve SOC site/species manifolds. |
| `n_shells` | int | no | `10` | Advanced native-kernel option.<br>CLI aliases: `--n_shells` |
| `d_max` | float | no | `20.0` | Advanced native-kernel option.<br>CLI aliases: `--d_max` |
| `all_bonds` | boolean | no | .true. | Keep directed bonds<br>CLI aliases: `--all_bonds`, `--canonical_bonds` |
| `nn_only` | boolean | no | .false. | Advanced native-kernel option.<br>CLI aliases: `--nn_only` |
| `atom_labels` | string | no | `''` | Comma-separated atom labels<br>CLI aliases: `--atom_labels` |
| `species_labels` | list[string] | no | none / runtime | Advanced native-kernel option.<br>CLI aliases: `--species_labels` |
| `no_symmetry` | boolean | no | .false. | Advanced native-kernel option.<br>CLI aliases: `--no_symmetry` |
| `orbit_grouping` | enum {spglib, shell} | no | `spglib` | Advanced native-kernel option.<br>CLI aliases: `--orbit_grouping` |
| `symprec` | float | no | `0.0001` | Advanced native-kernel option.<br>CLI aliases: `--symprec` |
| `angle_tolerance` | float | no | `-1.0` | Advanced native-kernel option.<br>CLI aliases: `--angle_tolerance` |
| `debug_orbits` | boolean | no | .false. | Advanced native-kernel option.<br>CLI aliases: `--debug_orbits` |
| `debug_orbit_shell` | int | no | none / runtime | Advanced native-kernel option.<br>CLI aliases: `--debug_orbit_shell` |
| `debug_epr_positions` | boolean | no | .false. | Advanced native-kernel option.<br>CLI aliases: `--debug_epr_positions` |
| `integrator` | enum {contour, cfr_ozaki, cfr_pole} | no | `contour` | Advanced native-kernel option.<br>CLI aliases: `--integrator` |
| `emin` | float | no | `-25.0` | Advanced native-kernel option.<br>CLI aliases: `--emin` |
| `empoints` | int | no | `500` | Advanced native-kernel option.<br>CLI aliases: `--empoints` |
| `cfr_beta` | float | no | `1000.0` | Advanced native-kernel option.<br>CLI aliases: `--cfr_beta` |
| `collinear_override` | boolean | no | .false. | Override J_zz with (J_xx+J_yy)/2 in collinear calculations<br>CLI aliases: `--collinear_override` |
| `spin_magnitude` | float | no | `1.0` | Spin magnitude S to scale J by 1/S^2<br>CLI aliases: `--spin_magnitude` |
| `out_dir` | string | no | `J_epr_tensor` | Advanced native-kernel option.<br>CLI aliases: `--out_dir` |
| `out_name` | string | no | `J_epr_tensor.txt` | Advanced native-kernel option.<br>CLI aliases: `--out_name` |
| `out_h5` | string | no | `J_epr_tensor.h5` | Advanced native-kernel option.<br>CLI aliases: `--out_h5` |

#### `input_format='wannier'`

Native engine: `slw.exchange.engine`; numerical kernel:
`slw.exchange.kernels.j_wannier`.

| Namelist key | Type | Parser required | Parser default | Meaning / CLI aliases |
|---|---|---:|---|---|
| `up_hr` | string | conditional | — | Spin-up collinear Wannier90 hr.dat; supply together with `dn_hr`, or supply `spinor_hr` instead.<br>CLI aliases: `--up_hr` |
| `dn_hr` | string | conditional | — | Spin-down collinear Wannier90 hr.dat; supply together with `up_hr`, or supply `spinor_hr` instead.<br>CLI aliases: `--dn_hr` |
| `spinor_hr` | string | conditional | — | Full spinor Wannier90 hr.dat; mutually exclusive with the collinear pair.<br>CLI aliases: `--spinor_hr` |
| `groupby` | enum {spin, orbital} | with `spinor_hr` | — | Without the projection bundle, declares HR row ordering and is normalized to spin-major. With the bundle, declares AMN trial-column spin ordering; HR rows remain in their Wannier gauge. |
| `centres` | string | with automatic matching | none / runtime | Wannier90 centres.xyz. Together with `win`, enables TB2J-style periodic nearest-atom assignment for spinor input.<br>CLI aliases: `--centres` |
| `amn` | string | projection bundle | none / runtime | Wannier90 AMN atomic-projection matrix. Must be supplied with `eig`, `spn`, `u_mat`, and `u_dis_mat`.<br>CLI aliases: `--amn` |
| `eig` | string | projection bundle | none / runtime | Wannier90 band eigenvalues from the same run as the projection bundle.<br>CLI aliases: `--eig` |
| `spn` | string | projection bundle | none / runtime | Wannier90 reference Pauli matrices in the Bloch eigenstate basis, used to validate the AMN-anchored atomic-Pauli frame.<br>CLI aliases: `--spn` |
| `u_mat` | string | projection bundle | none / runtime | Wannier90 final U rotation.<br>CLI aliases: `--u_mat` |
| `u_dis_mat` | string | projection bundle | none / runtime | Wannier90 disentanglement rotation.<br>CLI aliases: `--u_dis_mat` |
| `centre_tolerance_ang` | float | automatic centre matching only | no cutoff | Optional positive maximum centre-to-assigned-atom distance in angstrom. It is rejected with the projection bundle, which does not use centres. |
| `efermi` | float | yes | — | Fermi energy in eV<br>CLI aliases: `--efermi` |
| `hr_unit` | enum {ev, ry, ha} | no | `ev` | Unit of input hr.dat matrix elements; Wannier90 default is eV<br>CLI aliases: `--hr_unit` |
| `ref_epr_up` | string | no | none / runtime | Optional reference EPR up HDF5 for H(k) scale/gauge diagnostics<br>CLI aliases: `--ref_epr_up` |
| `ref_epr_dn` | string | no | none / runtime | Optional reference EPR down HDF5 for H(k) scale/gauge diagnostics<br>CLI aliases: `--ref_epr_dn` |
| `ref_hr_unit` | enum {ev, ry, ha} | no | `ry` | Reference EPR hopping unit<br>CLI aliases: `--ref_hr_unit` |
| `kmesh` | int[3] | yes | — | Uniform integration-grid dimensions. With the projection bundle they must equal the native U/AMN/SPN dimensions; coordinates, ordering, and shift come from U and are not interpolated.<br>CLI aliases: `--kmesh` |
| `mag_atoms` | list[int] | yes | — | Magnetic atom indices<br>CLI aliases: `--mag_atoms` |
| `mag_atoms_base` | enum {0, 1} | no | `0` | Advanced native-kernel option.<br>CLI aliases: `--mag_atoms_base` |
| `slices` | string | conditional | `''` | Manual local orbital slices, e.g. '0:0:5,1:5:10'. Required unless spinor input uses `win` + `centres` or the complete projection bundle. A projection-anchored run rejects this key because WIN+AMN defines its magnetic frame.<br>CLI aliases: `--slices` |
| `apply_degeneracy` | boolean | no | .true. | Divide HR blocks by Wannier90 degeneracy before H(k) construction; standard Wannier90 needs this<br>CLI aliases: `--apply_degeneracy`, `--no-apply_degeneracy` |
| `tensor_kernel` | enum {direct, tb2j} | no | `tb2j` | Tensor integration/decomposition convention. The compatibility alias `kernel` accepts the same values.<br>CLI aliases: `--kernel` |
| `axes` | string | no | `xyz` | Tensor axes to compute, subset of xyz<br>CLI aliases: `--axes` |
| `spin_direction` | float[3] | outside projection bundle | `[0.0, 0.0, 1.0]` | Input/model spin axis. It is rejected with the projection bundle, where every site direction is inferred from the projected time-reversal-odd field.<br>CLI aliases: `--spin_direction` |
| `win` | string | with SOC card, automatic matching, or projection bundle | none | Wannier90 structure/projections used for centre assignment, SOC selectors, or projection-anchored AMN column selection. |
| `n_shells` | int | no | `10` | Advanced native-kernel option.<br>CLI aliases: `--n_shells` |
| `d_max` | float | no | `20.0` | Advanced native-kernel option.<br>CLI aliases: `--d_max` |
| `all_bonds` | boolean | no | .true. | Keep directed bonds; default matches compute_J_epr_tensor<br>CLI aliases: `--all_bonds`, `--canonical_bonds` |
| `nn_only` | boolean | no | .false. | Advanced native-kernel option.<br>CLI aliases: `--nn_only` |
| `orbit_grouping` | enum {none, distance, shell} | no | `distance` | Advanced native-kernel option.<br>CLI aliases: `--orbit_grouping` |
| `integrator` | enum {contour, cfr_ozaki, cfr_pole} | no | `contour` | Advanced native-kernel option.<br>CLI aliases: `--integrator` |
| `emin` | float | no | `-25.0` | Advanced native-kernel option.<br>CLI aliases: `--emin` |
| `empoints` | int | no | `500` | Advanced native-kernel option.<br>CLI aliases: `--empoints` |
| `cfr_beta` | float | no | `1000.0` | Advanced native-kernel option.<br>CLI aliases: `--cfr_beta` |
| `spin_operator` | enum {auto, pauli, spn} | no | `auto` | `auto` uses the projection-anchored path when the complete bundle is present and rejects an unverified bare `spinor_hr` otherwise. `spn` requires the bundle. `pauli` is an explicit opt-in for an independently certified orbital-spin product basis and is rejected with the bundle. |
| `u_dis_layout` | enum {global_bands, compact_outer_window} | with projection bundle | — | Explicit U_dis row convention. Compact outer-window data also requires `dis_win_min`/`dis_win_max` in `win`. |
| `projection_rank_tolerance` | float | projection bundle only | `1.0e-4` | Minimum magnetic-AMN singular value. Explicit use outside the bundle is rejected. |
| `spin_projection_tolerance` | float | projection bundle only | `0.4` | Maximum combined or per-axis SPN residual in the projection-anchored atomic-Pauli frame. Explicit use outside the bundle is rejected. |
| `hamiltonian_tolerance_ev` | float | projection bundle only | `1.0e-4` | Maximum HR versus U_dis/U/EIG reconstruction residual in eV. Explicit use outside the bundle is rejected. |
| `noncollinear_tolerance` | float | projection bundle only | `0.25` | Maximum sitewise residual from a collinear time-reversal-odd field. Explicit use outside the bundle is rejected. |
| `intersite_xc_tolerance` | float | projection bundle only | `0.1` | Maximum nonlocal fraction of the projected time-reversal-odd field, including intersite support and onsite-block k dependence. Explicit use outside the bundle is rejected. |
| `collinear_override` | boolean | no | .false. | For a known no-SOC collinear Hamiltonian only, replace the longitudinal tensor component by the transverse average along the inferred magnetic axis. Never enable for SOC input.<br>CLI aliases: `--collinear_override` |
| `spin_magnitude` | float | no | `1.0` | Spin magnitude S to scale J by 1/S^2<br>CLI aliases: `--spin_magnitude` |
| `out_dir` | string | no | `J_wannier_tensor` | Advanced native-kernel option.<br>CLI aliases: `--out_dir` |
| `out_name` | string | no | `J_wannier_tensor.txt` | Advanced native-kernel option.<br>CLI aliases: `--out_name` |
| `out_h5` | string | no | `J_wannier_tensor.h5` | Advanced native-kernel option.<br>CLI aliases: `--out_h5` |

### `calculation='dj', ltensor=.false.`

**Runtime requirements:** EPR up/down files with EPC and `qc_dim` metadata,
`efermi`, `kmesh`, `slices`, and at least two `mag_atoms` are required.
`qmesh` defaults to EPR `qc_dim`; an explicit value must divide `kmesh` on
every axis. `targets` defaults to every crystal atom. With
`g_transform='k_only_rp'`, only the selected `rp_idx` is retained instead of a
q mesh. Target atoms, displacement axes, q mesh, and Rp selection should be
explicit for production work. `g_kernel='direct'` is the full-Wannier-matrix
reference implementation. `g_kernel='spectral'` performs the exact identity
`G(k+q) g(k,q) G(k) = C(k+q) D(k+q) [C(k+q)^H g(k,q) C(k)] D(k) C(k)^H`
using every electronic eigenstate, then assembles only the requested magnetic
endpoint blocks. It does not truncate bands or change the physical expression.
Converged production comparisons against `direct` are recommended before
making `spectral` the default.

For a complete `targets='all'`, `axes='xyz'` calculation, the default
`covariant_symmetry='project'` applies the full spglib Reynolds projector on
rank zero after MPI integration. It transforms the target atom, directed bond,
periodic `Rp`, and Cartesian polar-vector component together; it is not a
distance-shell average. The target, bond, and q-mesh quotient must be closed
under every detected operation. Partial target/component diagnostics default
to `none` unless a policy is explicitly supplied.

**Outputs:** `${savedir}/${prefix}.dj.txt`, `.all_bonds.tsv`, and `.h5`. HDF5
contains displacement metadata and target/bond datasets shaped `(nRp, 3)` in
meV/A. The `symmetry/` group records policy, application status, space group,
operation count, tolerance, and raw maximum/RMS covariance residual. The
canonical `dJ_r` group contains the projected values when projection is active.

Native engine: `slw.exchange.engine`; numerical kernel:
`slw.exchange.kernels.dj_epr`.

| Namelist key | Type | Parser required | Parser default | Meaning / CLI aliases |
|---|---|---:|---|---|
| `epr_up` | string | yes | — | Advanced native-kernel option.<br>CLI aliases: `--epr_up` |
| `epr_dn` | string | yes | — | Advanced native-kernel option.<br>CLI aliases: `--epr_dn` |
| `hr_unit` | enum {ev, ry, ha} | no | `ry` | Advanced native-kernel option.<br>CLI aliases: `--hr_unit` |
| `eph_unit` | enum {ev, ry, ha} | no | `ry` | Advanced native-kernel option.<br>CLI aliases: `--eph_unit` |
| `atom_labels` | string | no | `''` | Advanced native-kernel option.<br>CLI aliases: `--atom_labels` |
| `species_labels` | string | no | `''` | Comma-separated species for spglib orbit grouping, e.g. Mn,Mn,Te,Te.<br>CLI aliases: `--species_labels` |
| `targets` | string | no | `all` | Comma labels or 0-based indices; default all<br>CLI aliases: `--targets` |
| `axes` | string | no | `xyz` | Displacement axes; compact (`xyz`) and comma-separated (`x,y,z`) forms are equivalent.<br>CLI aliases: `--axes` |
| `mag_atoms` | list[int] | yes | — | Advanced native-kernel option.<br>CLI aliases: `--mag_atoms` |
| `mag_atoms_base` | enum {0, 1} | no | `0` | Advanced native-kernel option.<br>CLI aliases: `--mag_atoms_base` |
| `slices` | string | yes | — | Local orbital slices, e.g. '0:0:5,1:5:10'<br>CLI aliases: `--slices` |
| `efermi` | float | yes | — | Advanced native-kernel option.<br>CLI aliases: `--efermi` |
| `kmesh` | int[3] | yes | — | Advanced native-kernel option.<br>CLI aliases: `--kmesh` |
| `qmesh` | int[3] | no | none / runtime | Output q mesh; default EPR basic_data/qc_dim<br>CLI aliases: `--qmesh` |
| `rp_idx` | int[3] | no | `[0, 0, 0]` | Advanced native-kernel option.<br>CLI aliases: `--rp_idx` |
| `g_transform` | enum {kq, k_only_rp} | no | `kq` | kq: FT ep_hop over Re and Rp; k_only_rp: keep selected Rp real-space and FT only Re<br>CLI aliases: `--g_transform` |
| `g_kernel` | enum {direct, spectral} | no | `direct` | Select exact GgG assembly. `direct` forms the full Wannier-space product; `spectral` rotates g once in the complete eigenbasis and forms only magnetic-endpoint blocks.<br>CLI aliases: `--g_kernel` |
| `n_shells` | int | no | `1` | Advanced native-kernel option.<br>CLI aliases: `--n_shells` |
| `d_max` | float | no | `20.0` | Advanced native-kernel option.<br>CLI aliases: `--d_max` |
| `emin` | float | no | `-25.0` | Advanced native-kernel option.<br>CLI aliases: `--emin` |
| `empoints` | int | no | `100` | Advanced native-kernel option.<br>CLI aliases: `--empoints` |
| `integrator` | enum {contour, cfr, cfr_ozaki} | no | `contour` | Advanced native-kernel option.<br>CLI aliases: `--integrator` |
| `cfr_beta` | float | no | `400.0` | Advanced native-kernel option.<br>CLI aliases: `--cfr_beta` |
| `rotation_mode` | enum {none} | no | `none` | Advanced native-kernel option.<br>CLI aliases: `--rotation_mode` |
| `ddelta_mode` | enum {off, local, onsite} | no | `off` | Include derivative of local exchange splitting Delta. off: legacy dG-only; local: use full local block of g_up-g_dn; onsite: use only electron Re=(0,0,0) onsite derivative.<br>CLI aliases: `--ddelta_mode` |
| `symprec` | float | no | `0.0001` | spglib symmetry tolerance for orbit grouping.<br>CLI aliases: `--symprec` |
| `angle_tolerance` | float | no | `-1.0` | spglib angle tolerance in degrees; -1 uses spglib default.<br>CLI aliases: `--angle_tolerance` |
| `covariant_symmetry` | enum {none, report, project, fail} | no | `project` | Space-group covariance policy for scalar `dJ/du`. `project` applies the rank-zero target/bond/Rp/Cartesian Reynolds projector; `report` preserves raw values and records the residual; `fail` rejects a residual above tolerance; `none` skips discovery. Partial targets or axes default to `none`.<br>CLI aliases: `--covariant_symmetry` |
| `covariant_symmetry_tolerance_mev_per_ang` | float | no | `1.0e-8` | Non-negative maximum-component tolerance used by `covariant_symmetry='fail'` and recorded as provenance.<br>CLI aliases: `--covariant_symmetry_tolerance_mev_per_ang` |
| `orbit_grouping` | enum {spglib, shell} | no | `spglib` | Orbit grouping mode. shell groups all bonds with the same shell index and distance.<br>CLI aliases: `--orbit_grouping` |
| `debug_orbits` | boolean | no | .false. | Print spglib operation and bond-mapping diagnostics.<br>CLI aliases: `--debug_orbits` |
| `debug_orbit_shell` | int | no | none / runtime | Restrict --debug_orbits bond diagnostics to one shell.<br>CLI aliases: `--debug_orbit_shell` |
| `debug_epr_positions` | boolean | no | .false. | Print EPR tau and Wannier-center position diagnostics.<br>CLI aliases: `--debug_epr_positions` |
| `no_symmetry_orbits` | boolean | no | .false. | Advanced native-kernel option.<br>CLI aliases: `--no_symmetry_orbits` |
| `out_dir` | string | no | `dJ_epr_kspace` | Advanced native-kernel option.<br>CLI aliases: `--out_dir` |
| `out_name` | string | no | `dJ_epr_kspace.txt` | Advanced native-kernel option.<br>CLI aliases: `--out_name` |
| `out_h5` | string | no | `dJr.h5` | Advanced native-kernel option.<br>CLI aliases: `--out_h5` |

### `calculation='dj', ltensor=.true.`

**Runtime requirements:** EPR up/down files remain required for EPC and
structure even when `spinor_hr` supplies the electronic Hamiltonian.
`efermi`, `kmesh`, `mag_atoms`, and parser-required `slices` are mandatory.
`qmesh` defaults to EPR `qc_dim` and must divide `kmesh`; `targets` currently
defaults to the first magnetic atom only. A supplied `spinor_hr` requires
explicit `groupby` and may be combined with an additional `SOC (atomic)` card.
Set `targets`, `disp_axes`, and `qmesh` explicitly for production work. Under a multi-rank launch,
`execution='auto'` selects the MPI backend; otherwise the serial backend is
used.

**Outputs:** `${savedir}/${prefix}.dj_tensor.h5` with `dA_r`, `dJ_tensor_r`,
isotropic, symmetric-anisotropic, and DMI derivatives. Serial checkpoints
update the same file and mark completion in its attributes; rank 0 writes the
same final schema under MPI.

Native engine: `slw.exchange.engine`; serial and MPI numerical kernels:
`slw.exchange.kernels.dj_tensor_epr` and
`slw.exchange.kernels.dj_tensor_mpi`. Under MPI, `workers_per_rank` must be
one; use `numba_threads` and `blas_threads` in `&parallel` for rank-local work.

| Namelist key | Type | Parser required | Parser default | Meaning / CLI aliases |
|---|---|---:|---|---|
| `epr_up` | string | yes | — | EPR up HDF5; still required for g(k+q,k) and structure metadata<br>CLI aliases: `--epr_up` |
| `epr_dn` | string | yes | — | EPR down HDF5; still required for g(k+q,k)<br>CLI aliases: `--epr_dn` |
| `spinor_hr` | string | no | none / runtime | Optional full SOC/noncollinear Wannier90 spinor hr.dat used as base Hamiltonian<br>CLI aliases: `--spinor_hr` |
| `groupby` | enum {spin, orbital} | with `spinor_hr` | — | Explicit TB2J spinor layout; normalized internally to spin-major. |
| `spinor_hr_unit` | enum {ev, ry, ha} | no | `ev` | Unit of --spinor_hr matrix elements<br>CLI aliases: `--spinor_hr_unit` |
| `win` | string | with SOC card | none | Wannier90 projections used to resolve SOC site/species manifolds. |
| `centres` | string | no | none | Optional spinor centres count validation; never used to infer `groupby`. |
| `apply_degeneracy` | boolean | no | .true. | Divide spinor_hr blocks by Wannier90 degeneracy before H(k)<br>CLI aliases: `--apply_degeneracy`, `--no-apply_degeneracy` |
| `hr_unit` | string | no | `ry` | Advanced native-kernel option.<br>CLI aliases: `--hr_unit` |
| `eph_unit` | string | no | `ry` | Advanced native-kernel option.<br>CLI aliases: `--eph_unit` |
| `efermi` | float | yes | — | Advanced native-kernel option.<br>CLI aliases: `--efermi` |
| `kmesh` | int[3] | yes | — | Advanced native-kernel option.<br>CLI aliases: `--kmesh` |
| `qmesh` | int[3] | no | none / runtime | Advanced native-kernel option.<br>CLI aliases: `--qmesh` |
| `n_shells` | int | no | `10` | Advanced native-kernel option.<br>CLI aliases: `--n_shells` |
| `d_max` | float | no | `20.0` | Advanced native-kernel option.<br>CLI aliases: `--d_max` |
| `mag_atoms` | list[int] | yes | — | Advanced native-kernel option.<br>CLI aliases: `--mag_atoms` |
| `mag_atoms_base` | enum {0, 1} | no | `0` | Advanced native-kernel option.<br>CLI aliases: `--mag_atoms_base` |
| `targets` | list[int] | no | none / runtime | Advanced native-kernel option.<br>CLI aliases: `--targets` |
| `disp_axes` | string | no | `xyz` | Advanced native-kernel option.<br>CLI aliases: `--disp_axes` |
| `tensor_axes` | string | no | `xyz` | Advanced native-kernel option.<br>CLI aliases: `--tensor_axes` |
| `spin_direction` | float[3] | no | `[0.0, 0.0, 1.0]` | Advanced native-kernel option.<br>CLI aliases: `--spin_direction` |
| `slices` | string | yes | — | Local orbital slices, e.g. '0:0:5,1:5:10'<br>CLI aliases: `--slices` |
| `emin` | float | no | `-25.0` | Advanced native-kernel option.<br>CLI aliases: `--emin` |
| `empoints` | int | no | `300` | Advanced native-kernel option.<br>CLI aliases: `--empoints` |
| `integrator` | enum {contour, cfr} | no | `contour` | Advanced native-kernel option.<br>CLI aliases: `--integrator` |
| `progress_every` | int | no | `0` | Print per-worker progress every N energy points; 0 disables<br>CLI aliases: `--progress_every` |
| `verbose_worker_init` | int | no | `0` | Print one worker-init line per local energy worker<br>CLI aliases: `--verbose_worker_init` |
| `checkpoint` | int | no | `1` | Write partial HDF5 after each completed target/axis; 0 disables<br>CLI aliases: `--checkpoint` |
| `onsite_deriv_projector` | boolean | no | .true. | Include onsite derivative of the local spinor exchange field dP_i/du. Convention: P=M.e=Delta/2 in the collinear limit, so dP=dDelta/2.<br>CLI aliases: `--onsite_deriv_exchange_field`, `--no-onsite_deriv_exchange_field`, `--onsite_deriv_projector`, `--no-onsite_deriv_projector` |
| `out_dir` | string | no | `dJ_epr_tensor_analytic` | Advanced native-kernel option.<br>CLI aliases: `--out_dir` |
| `out_h5` | string | no | `dJ_tensor_analytic.h5` | Advanced native-kernel option.<br>CLI aliases: `--out_h5` |

## `slw_magph.x`

Hybrid, Berry, lifetime, spectral, scattering, and rotational/chiral calculations.
The backend paths shown below are quarantined implementation details. Historical
`slw.magph.<module>` commands are not preserved or re-exported; use this stage
executable for all active magph drivers.

> **Current-registry boundary:** `calculation='dispersion'` and
> `calculation='lifetime'` are native magph commands. They share the strict
> FM/AFM, SIA, unit, phase, and `J_iso` screening contract in
> [MAGPH_DESIGN.md](MAGPH_DESIGN.md), with automatic MPI k distribution. The
> other calculations in this section remain quarantined compatibility
> backends.

> **Legacy absolute-unit warning:** the compatibility vertex uses an amu
> zero-point prefactor with qe2pert polarizations normalized by masses in
> electron-mass atomic units. Absolute compatibility `spectral`, scattering,
> and archived lifetime magnitudes must not be used as native parity references. See
> [MAGPH_DESIGN.md](MAGPH_DESIGN.md) for the corrected SI-validated contract.

The outer QE-style interface uses only the common `&parallel` names. At the
quarantine boundary, `workers_per_rank` maps to the hybrid/phonon local pools,
`numba_threads` maps to scattering and rotational kernels, and the canonical
q/bond chunk fields map to the corresponding retained driver arguments.
Unsupported resource requests fail instead of being silently ignored. In
particular, native lifetime distributes external k points with MPI and requires
`workers_per_rank=1`; use threads and chunking for its rank-local work.

| `calculation` | Backend source | MPI | Purpose |
|---|---|---:|---|
| `dispersion` | native | yes | Compute magnon bands, including exact Goldstone energies, and an optional plot |
| `hybrid` | default | no | Build and diagonalize the hybrid magnon-phonon Hamiltonian |
| `berry` | default | no | Compute hybrid-band Berry curvature on a reciprocal-space plane |
| `spectral` | default | yes | Run the MPI-aware magnon-phonon spectral solver |
| `lifetime` | native | yes | Compute native MPI-distributed magnon lifetimes |
| `scattering_kbz` | default | no | Compute fixed-phonon-q scattering over the magnon Brillouin zone |
| `scattering_qbz` | default | no | Compute fixed-magnon scattering over the phonon Brillouin zone |
| `rotational_coupling` | default | no | Analyze rotational and chiral magnon-phonon coupling |
| `chirality_plane` | default | yes | Run restartable MPI chirality analysis on a reciprocal-space plane |
| `prepare_lifetime` | default | no | Create a compatibility manifest for lifetime and spectral solvers |

### `calculation='dispersion'`

**Runtime requirements:** A canonical scalar-exchange HDF5 with
`basic_data/lattice_ang`, an explicit FM or bipartite-AFM state, and a file
containing a Wannier90 `begin/end kpoint_path` block. No material path is
built into the solver. Optional uniaxial single-ion anisotropy follows
`H_SIA=-K(s.n)^2`; positive K is easy-axis. Exact AFM Goldstone points are
accepted because this calculation returns energies only, not paraunitary
eigenvectors.

**MPI:** Interpolated path points are divided into balanced contiguous rank
blocks. Dense eigensolvers remain vectorized/rank-local, and only rank zero
writes the final products.

**Outputs:** One atomic NPZ containing fractional k points, inverse-Angstrom
path distance, magnon energies in meV, Goldstone flags, segment/tick data, and
JSON provenance. `plot=.true.` additionally writes an atomic PNG, PDF, or SVG.

Backend: `slw.magph.engine:prepare_run`.

| Namelist key | Type | Required | Default | Meaning |
|---|---|---:|---|---|
| `exchange_h5` | path | yes | — | Canonical static scalar-exchange HDF5 including `lattice_ang`. |
| `magnetic_order` | enum {fm, collinear_afm} | yes | — | Explicit magnetic order. |
| `spin_magnitudes` | float or float list | yes | — | Positive spin magnitude, scalar-broadcast or one per magnetic site. |
| `spin_pattern` | float list | conditional | FM all +1; AFM +1,-1 | Explicit collinear signs. |
| `quantization_axis` | float[3] | yes | — | Nonzero Cartesian spin quantization axis. |
| `anisotropy_model` | enum {uniaxial} | conditional | none | Required as a complete SIA set when anisotropy is supplied. |
| `anisotropy_mev` | float or float list | conditional | none | K in `H_SIA=-K(s.n)^2`; scalar-broadcast or one per HDF5 magnetic-site-map entry. |
| `anisotropy_axis` | float[3] or flattened site axes | conditional | none | One Cartesian uniaxial direction or one per HDF5 magnetic-site-map entry. |
| `anisotropy_normalization` | enum {unit_vector, spin_operator} | conditional | none | Defines the spin variable `s` in the SIA Hamiltonian. |
| `kpath_file` | path | yes | — | File containing a Wannier90 `kpoint_path` block. |
| `points_per_segment` | int | no | 50 | Positive interpolation count per path segment. |
| `output` | path | no | `${savedir}/${prefix}.dispersion.npz` | Native dispersion NPZ. |
| `plot` | boolean | no | .true. | Write a noninteractive band plot. |
| `plot_output` | path | conditional | `${savedir}/${prefix}.dispersion.png` | PNG, PDF, or SVG path when plotting. |
| `plot_dpi` | int | no | 180 | Positive raster resolution; ignored by vector formats. |
| `restart_mode` | enum {error, restart, from_scratch} | no | error | `error` is no-clobber; `restart` validates and reuses a completed NPZ (and recreates a missing plot); `from_scratch` atomically replaces existing products. |
| `overwrite` | boolean | deprecated | .false. | Compatibility alias: `.true.` maps to `restart_mode='from_scratch'`, `.false.` maps to `error`. It cannot be combined with `restart_mode`. |

### `calculation='hybrid'`

**Runtime requirements:** Dynamic `dj_tensor_h5` (or `tensor_h5`), static
`j_tensor_h5`, and either `phonon_cache` or `epr_phonon` are required.
`calculation_mode='from_scratch'` requires `epr_phonon`; cache mode prefers the
cache and can fall back to EPR. `structure` is required when the J HDF5 lacks
lattice/tau. Without the optional compatibility `input_file`, the frontend
also requires explicit `s`, `spin_direction`, `kmesh`, and `output`.

**Outputs:** Hybrid-band NPZ, with optional bare-band NPZ, static plots, standalone HTML/JSON, and a generated phonon cache.

Backend: `slw.magph.legacy.reference.hybrid`.

| Namelist key | Type | Parser required | Parser default | Meaning / CLI aliases |
|---|---|---:|---|---|
| `workdir` | string | no | none / runtime | Workflow root directory (default: current directory)<br>CLI aliases: `--workdir` |
| `input_file` | string | no | none / runtime | Optional input.in/magph.in with hybrid variables<br>CLI aliases: `--input_file` |
| `calculation_mode` | string | no | none / runtime | Phonon cache mode: cache/reuse or from_scratch/recompute<br>CLI aliases: `--calculation_mode` |
| `in_dir` | string | no | none / runtime | Input directory for tensor/phonon/structure files<br>CLI aliases: `--in_dir` |
| `out_dir` | string | no | none / runtime | Output directory for hybrid outputs<br>CLI aliases: `--out_dir` |
| `dj_tensor_h5` | string | no | none / runtime | dynamic dJ/du tensor HDF5 from compute_dJ_epr_tensor<br>CLI aliases: `--dJ_tensor_h5` |
| `tensor_h5` | string | no | none / runtime | Compatibility input retained by the backend.<br>CLI aliases: `--tensor_h5` |
| `epr_phonon` | string | no | none / runtime | Compatibility input retained by the backend.<br>CLI aliases: `--epr_phonon` |
| `phonon_cache` | string | no | none / runtime | Compatibility input retained by the backend.<br>CLI aliases: `--phonon_cache` |
| `phonon_cache_out` | string | no | none / runtime | Write qpath/qmesh phonon cache built from --epr_phonon<br>CLI aliases: `--phonon_cache_out` |
| `phonon_cache_compressed` | boolean | no | none / runtime | Compress --phonon_cache_out; use --no-phonon_cache_compressed for faster writes<br>CLI aliases: `--phonon_cache_compressed`, `--no-phonon_cache_compressed` |
| `phonon_qmesh` | int[3] | no | none / runtime | Compatibility input retained by the backend.<br>CLI aliases: `--phonon_qmesh` |
| `j_tensor_h5` | string | no | none / runtime | static exchange tensor HDF5 from compute_J_epr_tensor<br>CLI aliases: `--J_tensor_h5`, `--j_tensor_h5` |
| `structure` | string | no | none / runtime | POSCAR/QE input with structure; optional if J tensor HDF5 stores lattice/tau<br>CLI aliases: `--structure` |
| `component` | enum {dmi, iso, aniso, full} | no | none / runtime | Compatibility input retained by the backend.<br>CLI aliases: `--component` |
| `static_component` | string | no | none / runtime | Static exchange tensor for bare magnons: iso, aniso, dmi, full, or combinations like iso+dmi.<br>CLI aliases: `--static_component` |
| `s` | float | no | none / runtime | Compatibility input retained by the backend.<br>CLI aliases: `--S` |
| `spin_direction` | float[3] | no | none / runtime | Compatibility input retained by the backend.<br>CLI aliases: `--spin_direction` |
| `spin_pattern` | string | no | none / runtime | Compatibility input retained by the backend.<br>CLI aliases: `--spin_pattern` |
| `phase_convention` | enum {cell, basis} | no | none / runtime | Compatibility input retained by the backend.<br>CLI aliases: `--phase_convention` |
| `kmesh` | int[3] | no | none / runtime | Compatibility input retained by the backend.<br>CLI aliases: `--kmesh` |
| `qpath` | string | no | none / runtime | Band path for q, e.g. 'G M K G A'. Coordinates like '0,0,0 0.5,0,0' also work.<br>CLI aliases: `--qpath` |
| `q_points` | string | no | none / runtime | Explicit list of q-points, overriding qpath and qmesh<br>CLI aliases: `--q_points` |
| `qpath_label_style` | enum {symbol, miller, both} | no | none / runtime | X tick label style for qpath nodes.<br>CLI aliases: `--qpath_label_style` |
| `path_points` | int | no | none / runtime | Points per q-path segment<br>CLI aliases: `--path_points` |
| `plot` | string | no | none / runtime | Optional hybrid band plot colored by magnon weight<br>CLI aliases: `--plot` |
| `html_plot` | string | no | none / runtime | Optional interactive standalone HTML hybrid band plot<br>CLI aliases: `--html_plot` |
| `html_data` | string | no | none / runtime | Optional JSON sidecar path for --html_plot<br>CLI aliases: `--html_data` |
| `plot_unit` | enum {meV, THz} | no | none / runtime | Compatibility input retained by the backend.<br>CLI aliases: `--plot_unit` |
| `xtick_fontsize` | float | no | none / runtime | Font size for q-path x tick labels<br>CLI aliases: `--xtick_fontsize` |
| `color_by` | enum {magnon_weight, chirality, phonon_lz} | no | none / runtime | Compatibility input retained by the backend.<br>CLI aliases: `--color_by` |
| `overlay_bare` | boolean | no | none / runtime | Overlay bare magnon/phonon branches on the hybrid plot<br>CLI aliases: `--overlay_bare`, `--no-overlay_bare` |
| `bare_output` | string | no | none / runtime | Optional npz containing only bare magnon/phonon branches<br>CLI aliases: `--bare_output` |
| `bare_plot` | string | no | none / runtime | Optional bare magnon/phonon branch plot<br>CLI aliases: `--bare_plot` |
| `show` | boolean | no | none / runtime | Show matplotlib windows after saving plots<br>CLI aliases: `--show`, `--no-show` |
| `mpl_backend` | string | no | none / runtime | Matplotlib interactive backend for --show, e.g. TkAgg or QtAgg<br>CLI aliases: `--mpl_backend` |
| `coupling_scale` | float | no | none / runtime | Compatibility input retained by the backend.<br>CLI aliases: `--coupling_scale` |
| `bond_factor` | float | no | none / runtime | Compatibility input retained by the backend.<br>CLI aliases: `--bond_factor` |
| `anisotropy_mev` | float | no | none / runtime | Compatibility input retained by the backend.<br>CLI aliases: `--anisotropy_mev` |
| `zeeman_field_mev` | float[3] | no | none / runtime | Zeeman energy vector h=g*mu_B*B in meV, Cartesian coordinates.<br>CLI aliases: `--zeeman_field_mev` |
| `field_tesla` | float[3] | no | none / runtime | Magnetic field vector in Tesla; converted to meV using --g_factor.<br>CLI aliases: `--field_tesla` |
| `g_factor` | float | no | none / runtime | g factor for --field_tesla conversion<br>CLI aliases: `--g_factor` |
| `phonon_negative_tol_mev` | float | no | none / runtime | Clip phonon energies in [-tol, 0) to 0 before hybridization.<br>CLI aliases: `--phonon_negative_tol_mev` |
| `gamma_acoustic_zero` | int | no | none / runtime | Set the lowest N phonon modes at Gamma to 0 before hybridization, usually N=3.<br>CLI aliases: `--gamma_acoustic_zero` |
| `gamma_zero_tol` | float | no | none / runtime | Fractional-q tolerance for --gamma_acoustic_zero<br>CLI aliases: `--gamma_zero_tol` |
| `threshold` | float | no | none / runtime | Compatibility input retained by the backend.<br>CLI aliases: `--threshold` |
| `output` | string | no | none / runtime | Compatibility input retained by the backend.<br>CLI aliases: `-o`, `--output` |

### `calculation='berry'`

**Runtime requirements:** `dj_tensor_h5`, `j_tensor_h5`, either
`phonon_cache` or `epr_phonon`, and parser-required `output` are needed.
Cached q points/order must match the requested plane exactly. `structure` is
required when the J HDF5 lacks geometry. Spin state, plane/fixed coordinate,
mesh, and band/subspace selection should be explicit.

**Outputs:** Berry-curvature NPZ for a selected subspace or band-suffixed NPZ
files when several uncombined bands are requested, plus optional plots.

Calculation aliases: `hybrid_berry`.

Backend: `slw.magph.legacy.reference.hybrid_berry`.

| Namelist key | Type | Parser required | Parser default | Meaning / CLI aliases |
|---|---|---:|---|---|
| `dj_tensor_h5` | string | no | none / runtime | dynamic dJ/du tensor HDF5 from compute_dJ_epr_tensor<br>CLI aliases: `--dJ_tensor_h5` |
| `j_tensor_h5` | string | no | none / runtime | static exchange tensor HDF5 from compute_J_epr_tensor<br>CLI aliases: `--J_tensor_h5`, `--j_tensor_h5` |
| `tensor_h5` | string | no | none / runtime | Compatibility input retained by the backend.<br>CLI aliases: `--tensor_h5` |
| `epr_phonon` | string | no | none / runtime | Compatibility input retained by the backend.<br>CLI aliases: `--epr_phonon` |
| `phonon_cache` | string | no | none / runtime | Compatibility input retained by the backend.<br>CLI aliases: `--phonon_cache` |
| `component` | enum {dmi, iso, aniso, full} | no | `dmi` | Compatibility input retained by the backend.<br>CLI aliases: `--component` |
| `static_component` | string | no | `iso` | Static exchange tensor for bare magnons: iso, aniso, dmi, full, or combinations like iso+dmi<br>CLI aliases: `--static_component` |
| `s` | float | no | `2.5` | Compatibility input retained by the backend.<br>CLI aliases: `--S` |
| `spin_direction` | float[3] | no | `[0.0, 1.0, 0.0]` | Compatibility input retained by the backend.<br>CLI aliases: `--spin_direction` |
| `spin_pattern` | string | no | `auto` | Compatibility input retained by the backend.<br>CLI aliases: `--spin_pattern` |
| `phase_convention` | enum {cell, basis} | no | `basis` | Compatibility input retained by the backend.<br>CLI aliases: `--phase_convention` |
| `plane` | enum {kz, xy, ky, xz, kx, yz} | no | `kz` | Compatibility input retained by the backend.<br>CLI aliases: `--plane` |
| `fixed` | float | no | `0.0` | Compatibility input retained by the backend.<br>CLI aliases: `--fixed` |
| `mesh` | int[2] | no | `[31, 31]` | Compatibility input retained by the backend.<br>CLI aliases: `--mesh` |
| `band` | string | no | `0` | Band index, range, or list. Examples: 0, 0-5, '0 2 4', '0,2,4'<br>CLI aliases: `--band` |
| `sum_below` | boolean | no | .false. | Sum Berry curvature from band 0 through each --band value, WannierTools occupied-subspace style<br>CLI aliases: `--sum_below`, `--no-sum_below` |
| `band_sum` | string | no | none / runtime | Explicit band subspace to sum, e.g. 0-12 or '0 1 2'. Produces one summed output<br>CLI aliases: `--band_sum` |
| `threshold` | float | no | `0.0` | Compatibility input retained by the backend.<br>CLI aliases: `--threshold` |
| `structure` | string | no | none / runtime | Compatibility input retained by the backend.<br>CLI aliases: `--structure` |
| `coupling_scale` | float | no | `1.0` | Compatibility input retained by the backend.<br>CLI aliases: `--coupling_scale` |
| `berry_method` | enum {fukui, kubo, both} | no | `fukui` | Fukui is plaquette-link curvature; Kubo uses finite-difference dH/dq matrix elements<br>CLI aliases: `--berry_method` |
| `kubo_gap_floor` | float | no | `1e-08` | Energy denominator floor in meV for Kubo Berry curvature<br>CLI aliases: `--kubo_gap_floor` |
| `track_bands` | enum {energy, overlap} | no | `energy` | energy keeps per-q eigenvalue order; overlap tracks branches by nearest-neighbor eigenvector overlap<br>CLI aliases: `--track_bands` |
| `output` | string | yes | — | Compatibility input retained by the backend.<br>CLI aliases: `-o`, `--output` |
| `plot` | string | no | none / runtime | Compatibility input retained by the backend.<br>CLI aliases: `--plot` |
| `plot_quantity` | string | no | `flux` | Compatibility input retained by the backend.<br>CLI aliases: `--plot_quantity` |
| `plot_bz` | boolean | no | .false. | Draw plot as a periodic Voronoi map clipped to the first BZ<br>CLI aliases: `--plot_bz`, `--no-plot_bz` |
| `plot_tile` | int | no | `2` | Periodic image range used to fill Voronoi cells before first-BZ clipping<br>CLI aliases: `--plot_tile` |
| `plot_cmap` | string | no | none / runtime | Compatibility input retained by the backend.<br>CLI aliases: `--plot_cmap` |
| `plot_edgecolor` | string | no | `none` | Compatibility input retained by the backend.<br>CLI aliases: `--plot_edgecolor` |
| `plot_linewidth` | float | no | `0.0` | Compatibility input retained by the backend.<br>CLI aliases: `--plot_linewidth` |
| `plot_vmin` | float | no | none / runtime | Compatibility input retained by the backend.<br>CLI aliases: `--plot_vmin` |
| `plot_vmax` | float | no | none / runtime | Compatibility input retained by the backend.<br>CLI aliases: `--plot_vmax` |
| `plot_vmin_percentile` | float | no | `1.0` | Compatibility input retained by the backend.<br>CLI aliases: `--plot_vmin_percentile` |
| `plot_vmax_percentile` | float | no | `99.0` | Compatibility input retained by the backend.<br>CLI aliases: `--plot_vmax_percentile` |

### `calculation='spectral'`

**Runtime requirements:** The frontend requires compatibility `input_file`.
`manifest` must be given directly or as `manifest=` in that file and must point
to J/dJ/phonon products. `mode='execute'` additionally needs structure from
`POSCAR=`, `structure_file=`, or structure cards in the file. That file also
carries the explicit K_POINTS path and physical solver values not yet exposed
by this parser; `validate` and `plan` do not run the numerical kernel.

**Outputs:** Runtime JSON in every mode. Serial execute writes sigma and
spectral NPZ files; MPI execute writes two NPZ shards per rank and records the
unmerged chunk list in JSON.

Backend: `slw.magph.legacy.reference.solver_mpi`; MPI backend:
`slw.magph.legacy.reference.solver_mpi`.

| Namelist key | Type | Parser required | Parser default | Meaning / CLI aliases |
|---|---|---:|---|---|
| `workdir` | str | no | none / runtime | Workflow root directory (default: current directory)<br>CLI aliases: `--workdir` |
| `input_file` | str | no | none / runtime | Legacy-style input file (e.g. input.in)<br>CLI aliases: `--input_file` |
| `manifest` | str | no | none / runtime | Manifest from `slw_magph.x` with `calculation='prepare_lifetime'`<br>CLI aliases: `--manifest` |
| `mode` | enum {validate, plan, execute} | no | `plan` | validate: contract check, plan: runtime estimates, execute: run sigma + spectral<br>CLI aliases: `--mode` |
| `kernel_source` | enum {kernels, kernels_lifetime} | no | none / runtime | Kernel backend selection (default: kernels)<br>CLI aliases: `--kernel_source` |
| `mag_order` | enum {afm_bipartite, afm, fm, ferro, ferromagnetic} | no | none / runtime | Magnon order/model for solver validation (FM execute is not implemented yet)<br>CLI aliases: `--mag_order` |
| `omega_points` | int | no | `5000` | Omega axis size for solver planning<br>CLI aliases: `--omega_points` |
| `omega_max` | float | no | none / runtime | Omega axis maximum in meV (default: input file or 50)<br>CLI aliases: `--omega_max` |
| `progress_interval` | int | no | none / runtime | Print execute progress every N k-points (default: input file or 1)<br>CLI aliases: `--progress_interval` |
| `target_rank_gb` | float | no | `4.0` | Target per-rank working memory for omega chunk recommendation<br>CLI aliases: `--target_rank_gb` |
| `k_tasks` | int | no | none / runtime | Number of external k tasks for runtime planning<br>CLI aliases: `--k_tasks` |
| `out_dir` | str | no | none / runtime | Output directory for runtime metadata<br>CLI aliases: `--out_dir` |
| `out_name` | str | no | `magph_solver_runtime.json` | Output metadata filename<br>CLI aliases: `--out_name` |

### `calculation='lifetime'`

**Runtime requirements:** Static canonical exchange HDF5, scalar exchange
derivative HDF5, and either a schema-v3 phonon cache or an EPR HDF5 containing
IFCs are required. If the cache is absent, rank zero constructs it from EPR on
the explicit `phonon_qmesh`, writes it atomically, and releases the other MPI
ranks only after the completed cache is visible. Without `phonon_qmesh`, cache
construction defaults to the derivative source mesh for backward
compatibility. An existing cache is reused without rebuilding and must match
an explicit `phonon_qmesh`. The real-space derivative `dJ(R,Rp)` is Fourier
interpolated onto every phonon q point, so the phonon evaluation mesh may be
denser than the derivative source mesh. The exchange and
derivative bond maps, units, directed mates, real-space/Fourier conventions,
derivative ASR, phonon mass normalization, and q mesh are screened before any
LSWT calculation. The derivative bond map may be a strict, directed-mate-complete
subset of the static exchange map. This permits, for example, shell-10 `J` with
shell-4 `dJ/du`: static bonds absent from the derivative source are retained in
LSWT and embedded as exactly zero in the vertex. A derivative bond absent from
the static exchange map, or a subset missing its directed mate, is rejected.
`J_iso` is admitted only with compatible `dJ_iso/du` and is promoted internally
to `J_iso I` without granting tensor capabilities. FM
supports any positive number of magnetic sublattices; the initial AFM route is
restricted to exactly two collinear opposite sublattices. The initial FM
lifetime channel contract accepts collinear SIA that does not generate
anomalous FM terms; energy-only dispersion has the broader Nambu treatment.

**MPI:** `execution='auto'` uses every discovered MPI rank. The mode-resolved
`dJ/du` coupling cache is constructed with balanced phonon-q ownership and then
broadcast. The exact uniform `lcm(kmesh,phonon_qmesh)` union of all required `k+q`
points is diagonalized with the same distributed-assemble-broadcast pattern.
External k points are then distributed in balanced contiguous blocks. Within a
rank, each q block is built in the exchange-bond/site-Nambu representation,
rotated only for the required physical external channels, and accumulated
immediately into the on-shell self-energy; the full vertex is not allocated. q
weights are normalized over the complete grid before block slicing. Only rank
zero assembles and writes the scientific output, and collective Python failures
are exchanged before the next gather. The coupling and union-LSWT caches are
currently replicated per MPI rank; their actual per-rank and maximum-node sizes
are printed at run time.

**Outputs:** One atomic, no-clobber-by-default schema-v2 NPZ containing
fractional k points, physical magnon energies, complex on-shell self-energy,
HWHM, FWHM, rate in `ps^-1`, lifetime in `ps`, validity flags, and JSON
provenance. Bipartite-AFM products additionally store magnon chirality and put
all mode-resolved arrays in canonical `chi=+1, chi=-1` order. The
default path is `${savedir}/${prefix}.lifetime.npz`. Provenance records the
static, explicit derivative, and zero-filled bond counts, the derivative source
mesh, the phonon evaluation mesh, and whether Fourier interpolation was active.

Backend: `slw.magph.engine:prepare_run`.

| Namelist key | Type | Required | Default | Meaning |
|---|---|---:|---|---|
| `exchange_h5` | path | yes | — | Canonical static scalar-exchange HDF5 with explicit Hamiltonian/bond provenance. |
| `derivative_h5` | path | yes | — | Scalar `dJ/du` HDF5 with periodic `Rp`, q mesh, units, phase, and mate provenance. |
| `phonon_cache` | path | conditional | `${savedir}/${prefix}.phonon.npz` when `phonon_epr` is set | Native-compatible schema-v3 phonon cache. If present it is reused; if absent it is the output built from `phonon_epr`. |
| `phonon_epr` | path | conditional | none | qe2pert EPR HDF5 used by rank zero only when `phonon_cache` is absent. At least one of `phonon_cache` and `phonon_epr` is required. |
| `phonon_qmesh` | int[3] | no | existing cache mesh; derivative source mesh when building a missing cache | Uniform phonon/self-energy q mesh. `dJ(R,Rp)` is Fourier interpolated from its source mesh onto these q points. An existing cache must match an explicit value. |
| `phonon_loto` | enum {auto, none, 2d, 3d} | no | auto | Long-range polar correction used while building an EPR-derived cache. `auto` follows EPR metadata. |
| `phonon_imaginary_tolerance_mev` | float | no | `1.0e-6` | Non-negative roundoff window: modes with absolute signed frequency within it are stored as exact zero; more-negative modes fail cache construction. |
| `phonon_cache_compressed` | boolean | no | `.false.` | Compress a newly generated cache. Uncompressed cache writing/loading is faster and is preferred when storage is not limiting. |
| `magnetic_order` | enum {fm, collinear_afm} | yes | — | Explicit magnetic model; it is never inferred from the sign of J. |
| `spin_magnitudes` | float or float list | yes | — | Positive spin magnitude broadcast from a scalar or supplied per magnetic site. |
| `spin_pattern` | float list | conditional | FM all +1; AFM +1,-1 | Explicit collinear signs when overriding the canonical pattern. |
| `quantization_axis` | float[3] | yes | — | Nonzero Cartesian quantization axis. |
| `anisotropy_model` | enum {uniaxial} | conditional | none | Complete optional single-ion anisotropy model. |
| `anisotropy_mev` | float or float list | conditional | none | K in `H_SIA=-K(s.n)^2`; scalar-broadcast or one per HDF5 magnetic-site-map entry. |
| `anisotropy_axis` | float[3] or flattened site axes | conditional | none | One uniaxial direction or one per HDF5 magnetic-site-map entry. |
| `anisotropy_normalization` | enum {unit_vector, spin_operator} | conditional | none | Explicit spin variable used by the SIA Hamiltonian. |
| `kmesh` | int[3] | yes | — | Positive uniform external-k mesh. |
| `kshift` | float[3] | yes | — | Explicit grid-unit shift; required to avoid hidden Gamma/Goldstone policy. |
| `temperature_k` | float | yes | — | Non-negative temperature in kelvin. |
| `broadening_mev` | float | yes | — | Positive retarded broadening in meV. |
| `frequency_floor_mev` | float | no | none | Explicit positive phonon floor; required if the cache contains exact zero modes. |
| `asr_policy` | enum {fail, report, project} | no | fail | Derivative acoustic-sum-rule policy. Projection is recorded in provenance. |
| `metric_energy_tolerance_mev` | float | no | 0 | Explicit signed-BdG energy tolerance. |
| `negative_tolerance_mev` | float | no | 0 | Roundoff tolerance for slightly negative HWHM only. |
| `require_complete_targets` | boolean | no | .true. | Require `dJ/du` targets for every phonon atom. |
| `output` | path | no | `${savedir}/${prefix}.lifetime.npz` | Native lifetime NPZ. |
| `restart_mode` | enum {error, restart, from_scratch} | no | error | `error` is no-clobber; `restart` validates and reuses a completed NPZ; `from_scratch` atomically replaces it. Native lifetime has no partial checkpoint shards yet. |
| `overwrite` | boolean | deprecated | .false. | Compatibility alias for `error`/`from_scratch`; mutually exclusive with `restart_mode`. |

Native lifetime resource controls are all in `&parallel`. It requires
`workers_per_rank=1`, distributes external k points over MPI ranks, and uses
`threads_per_worker` plus the q/bond/vertex/self-energy/channel chunk fields
listed in the common table above for rank-local vectorized work. Its native
kernels are NumPy-vectorized, so an explicit `numba_threads` is rejected;
`blas_threads` controls their dense linear algebra. EPR phonon construction is
performed once on rank zero with batched, vectorized dynamical-matrix assembly
and diagonalization; `q_chunk_size` bounds that preparation stage as well.

### `calculation='scattering_kbz'`

**Runtime requirements:** Requires static `jr`/`j_cache`, dynamic
`djr`/`dj_tensor_h5`, `phonon_cache`, fixed phonon `q`, `kmesh`, and image
`output`. A separate `structure` is required if the J source lacks geometry.
AFM mode currently requires exactly two magnetic sublattices, and the selected
Voronoi slice needs at least four points.

**Outputs:** Figure plus `data_output` NPZ; when omitted, the NPZ path is
derived from the image stem.

Backend: `slw.magph.legacy.reference.plot_scattering_kbz`.

| Namelist key | Type | Parser required | Parser default | Meaning / CLI aliases |
|---|---|---:|---|---|
| `jr` | string | one of: `jr`, `jr` | none / runtime | Static Jr HDF5 (preferred) or prepared legacy J NPZ<br>CLI aliases: `--jr`, `--j-cache` |
| `djr` | string | one of: `djr`, `djr` | none / runtime | Real-space dJr HDF5<br>CLI aliases: `--djr`, `--dj-tensor-h5` |
| `phonon_cache` | string | yes | — | Prepared phonon mesh NPZ<br>CLI aliases: `--phonon-cache` |
| `structure` | string | no | none / runtime | Structure file; optional when --jr HDF5 embeds lattice and positions<br>CLI aliases: `--structure` |
| `q` | float[3] | yes | — | Requested fractional phonon momentum<br>CLI aliases: `--q` |
| `kmesh` | int[3] | yes | — | Compatibility input retained by the backend.<br>CLI aliases: `--kmesh` |
| `shift_kmesh` | boolean | no | .false. | Compatibility input retained by the backend.<br>CLI aliases: `--shift-kmesh`, `--no-shift-kmesh` |
| `mag_order` | enum {afm, fm} | no | `afm` | Compatibility input retained by the backend.<br>CLI aliases: `--mag-order` |
| `s` | float | no | `2.5` | Compatibility input retained by the backend.<br>CLI aliases: `--S` |
| `spin_pattern` | string | no | `auto` | Compatibility input retained by the backend.<br>CLI aliases: `--spin-pattern` |
| `physical_only` | boolean | no | .true. | For AFM, use only the positive-energy half of the Nambu channels<br>CLI aliases: `--physical-only`, `--no-physical-only` |
| `anisotropy_mev` | float | no | `0.0` | Compatibility input retained by the backend.<br>CLI aliases: `--anisotropy-mev` |
| `bond_factor` | float | no | `1.0` | Compatibility input retained by the backend.<br>CLI aliases: `--bond-factor` |
| `rp_idx` | int[3] | no | `[0, 0, 0]` | Compatibility input retained by the backend.<br>CLI aliases: `--rp-idx` |
| `phonon_floor_mev` | float | no | `0.001` | Compatibility input retained by the backend.<br>CLI aliases: `--phonon-floor-mev` |
| `dj_asr` | enum {none, check, project} | no | `check` | Compatibility input retained by the backend.<br>CLI aliases: `--dJ-asr` |
| `dj_asr_tolerance` | float | no | `1e-08` | Compatibility input retained by the backend.<br>CLI aliases: `--dJ-asr-tolerance` |
| `initial_mode` | list[int] | no | none / runtime | Initial magnon modes to plot; default plots all<br>CLI aliases: `--initial-mode` |
| `mode_base` | enum {0, 1} | no | `1` | Compatibility input retained by the backend.<br>CLI aliases: `--mode-base` |
| `data_output` | string | no | none / runtime | Output NPZ; default is OUTPUT with suffix .npz<br>CLI aliases: `--data-output` |
| `output` | string | yes | — | Output image<br>CLI aliases: `-o`, `--output` |
| `plane_axes` | int[2] | no | `[0, 1]` | Compatibility input retained by the backend.<br>CLI aliases: `--plane-axes` |
| `slice_value` | float | no | `0.0` | Compatibility input retained by the backend.<br>CLI aliases: `--slice` |
| `k_round` | int | no | `8` | Compatibility input retained by the backend.<br>CLI aliases: `--k-round` |
| `tile` | int | no | `2` | Compatibility input retained by the backend.<br>CLI aliases: `--tile` |
| `cmap` | string | no | `viridis` | Compatibility input retained by the backend.<br>CLI aliases: `--cmap` |
| `vmin` | float | no | none / runtime | Compatibility input retained by the backend.<br>CLI aliases: `--vmin` |
| `vmax` | float | no | none / runtime | Compatibility input retained by the backend.<br>CLI aliases: `--vmax` |
| `vmin_percentile` | float | no | `1.0` | Compatibility input retained by the backend.<br>CLI aliases: `--vmin-percentile` |
| `vmax_percentile` | float | no | `99.0` | Compatibility input retained by the backend.<br>CLI aliases: `--vmax-percentile` |
| `edgecolor` | string | no | `none` | Compatibility input retained by the backend.<br>CLI aliases: `--edgecolor` |
| `linewidth` | float | no | `0.0` | Compatibility input retained by the backend.<br>CLI aliases: `--linewidth` |
| `bz_color` | string | no | `0.35` | Compatibility input retained by the backend.<br>CLI aliases: `--bz-color` |
| `bz_lw` | float | no | `1.2` | Compatibility input retained by the backend.<br>CLI aliases: `--bz-lw` |
| `margin` | float | no | `0.04` | Compatibility input retained by the backend.<br>CLI aliases: `--margin` |
| `panel_width` | float | no | `5.0` | Compatibility input retained by the backend.<br>CLI aliases: `--panel-width` |
| `panel_height` | float | no | `4.6` | Compatibility input retained by the backend.<br>CLI aliases: `--panel-height` |
| `dpi` | int | no | `180` | Compatibility input retained by the backend.<br>CLI aliases: `--dpi` |
| `title` | string | no | none / runtime | Compatibility input retained by the backend.<br>CLI aliases: `--title` |

### `calculation='scattering_qbz'`

**Runtime requirements:** Requires static `jr`/`j_cache`, dynamic
`djr`/`dj_tensor_h5`, `phonon_cache`, initial `magnon_q`, `mag_order`, `s`, and
image `output`. A separate `structure` is required if J lacks geometry. AFM
mode currently requires exactly two magnetic sublattices, and the selected
q-slice needs at least four points.

**Outputs:** Figure plus `data_output` NPZ; when omitted, the NPZ path is
derived from the image stem.

Backend: `slw.magph.legacy.reference.plot_scattering_qbz`.

| Namelist key | Type | Parser required | Parser default | Meaning / CLI aliases |
|---|---|---:|---|---|
| `jr` | string | one of: `jr`, `jr` | none / runtime | Static Jr HDF5 (preferred) or prepared legacy J NPZ<br>CLI aliases: `--jr`, `--j-cache` |
| `djr` | string | one of: `djr`, `djr` | none / runtime | Real-space dJr HDF5<br>CLI aliases: `--djr`, `--dj-tensor-h5` |
| `phonon_cache` | string | yes | — | Prepared phonon q-mesh NPZ<br>CLI aliases: `--phonon-cache` |
| `structure` | string | no | none / runtime | Structure file; optional when --jr embeds lattice and positions<br>CLI aliases: `--structure` |
| `magnon_q` | float[3] | yes | — | Fixed fractional momentum of the initial magnon<br>CLI aliases: `--magnon-q` |
| `mag_order` | enum {afm, fm} | yes | — | Compatibility input retained by the backend.<br>CLI aliases: `--mag-order` |
| `s` | float | yes | — | Local spin magnitude<br>CLI aliases: `--S` |
| `spin_pattern` | string | no | `auto` | Compatibility input retained by the backend.<br>CLI aliases: `--spin-pattern` |
| `physical_only` | boolean | no | .true. | For AFM, retain only positive-energy Nambu channels<br>CLI aliases: `--physical-only`, `--no-physical-only` |
| `anisotropy_mev` | float | no | `0.0` | Compatibility input retained by the backend.<br>CLI aliases: `--anisotropy-mev` |
| `bond_factor` | float | no | `1.0` | Compatibility input retained by the backend.<br>CLI aliases: `--bond-factor` |
| `rp_idx` | int[3] | no | `[0, 0, 0]` | Compatibility input retained by the backend.<br>CLI aliases: `--rp-idx` |
| `phonon_floor_mev` | float | no | `0.001` | Compatibility input retained by the backend.<br>CLI aliases: `--phonon-floor-mev` |
| `dj_asr` | enum {none, check, project} | no | `check` | Compatibility input retained by the backend.<br>CLI aliases: `--dJ-asr` |
| `dj_asr_tolerance` | float | no | `1e-08` | Compatibility input retained by the backend.<br>CLI aliases: `--dJ-asr-tolerance` |
| `initial_mode` | list[int] | no | none / runtime | Initial magnon modes to plot; default plots all<br>CLI aliases: `--initial-mode` |
| `mode_base` | enum {0, 1} | no | `1` | Compatibility input retained by the backend.<br>CLI aliases: `--mode-base` |
| `data_output` | string | no | none / runtime | Output NPZ; default is OUTPUT with suffix .npz<br>CLI aliases: `--data-output` |
| `output` | string | yes | — | Output image<br>CLI aliases: `-o`, `--output` |
| `plane_axes` | int[2] | no | `[0, 1]` | Compatibility input retained by the backend.<br>CLI aliases: `--plane-axes` |
| `slice_value` | float | no | `0.0` | Compatibility input retained by the backend.<br>CLI aliases: `--slice` |
| `q_round` | int | no | `8` | Compatibility input retained by the backend.<br>CLI aliases: `--q-round` |
| `tile` | int | no | `2` | Compatibility input retained by the backend.<br>CLI aliases: `--tile` |
| `cmap` | string | no | `viridis` | Compatibility input retained by the backend.<br>CLI aliases: `--cmap` |
| `vmin` | float | no | none / runtime | Compatibility input retained by the backend.<br>CLI aliases: `--vmin` |
| `vmax` | float | no | none / runtime | Compatibility input retained by the backend.<br>CLI aliases: `--vmax` |
| `vmin_percentile` | float | no | `1.0` | Compatibility input retained by the backend.<br>CLI aliases: `--vmin-percentile` |
| `vmax_percentile` | float | no | `99.0` | Compatibility input retained by the backend.<br>CLI aliases: `--vmax-percentile` |
| `edgecolor` | string | no | `none` | Compatibility input retained by the backend.<br>CLI aliases: `--edgecolor` |
| `linewidth` | float | no | `0.0` | Compatibility input retained by the backend.<br>CLI aliases: `--linewidth` |
| `bz_color` | string | no | `0.35` | Compatibility input retained by the backend.<br>CLI aliases: `--bz-color` |
| `bz_lw` | float | no | `1.2` | Compatibility input retained by the backend.<br>CLI aliases: `--bz-lw` |
| `margin` | float | no | `0.04` | Compatibility input retained by the backend.<br>CLI aliases: `--margin` |
| `panel_width` | float | no | `5.0` | Compatibility input retained by the backend.<br>CLI aliases: `--panel-width` |
| `panel_height` | float | no | `4.6` | Compatibility input retained by the backend.<br>CLI aliases: `--panel-height` |
| `dpi` | int | no | `180` | Compatibility input retained by the backend.<br>CLI aliases: `--dpi` |
| `title` | string | no | none / runtime | Compatibility input retained by the backend.<br>CLI aliases: `--title` |

### `calculation='rotational_coupling'`

**Runtime requirements:** Requires static `jr`, dynamic `djr`, current-schema
`phonon_cache`, three-component `axis`, `spin_pattern`, at least one repeated
`k_point`, temperature, broadening, spin magnitude, bond factor, anisotropy,
and `output`. `dJ_asr='project'` requires derivative targets covering every
phonon atom; units, geometry, provenance, and stability checks are strict
unless their explicit `allow_*` overrides are selected.

**Outputs:** No-clobber analysis NPZ and `summary_json`, which defaults to the
same stem with `.json`.

Backend: `slw.magph.legacy.reference.analyze_rotational_coupling`.

| Namelist key | Type | Parser required | Parser default | Meaning / CLI aliases |
|---|---|---:|---|---|
| `jr` | string | yes | — | Static scalar J real-space file<br>CLI aliases: `--jr` |
| `djr` | string | yes | — | Scalar dJ(R,Rp) HDF5/NPZ file<br>CLI aliases: `--djr` |
| `phonon_cache` | string | yes | — | Current-schema phonon cache NPZ<br>CLI aliases: `--phonon-cache` |
| `axis` | float[3] | yes | — | Compatibility input retained by the backend.<br>CLI aliases: `--axis` |
| `spin_pattern` | string | yes | — | Two-sublattice collinear signs, e.g. 1,-1<br>CLI aliases: `--spin-pattern` |
| `k_point` | list[float] | yes | — | Fractional initial magnon k; repeat for multiple pilot points<br>CLI aliases: `--k-point` |
| `temperature_k` | float | yes | — | Compatibility input retained by the backend.<br>CLI aliases: `--temperature-k`, `--T` |
| `eta_mev` | float | yes | — | Compatibility input retained by the backend.<br>CLI aliases: `--eta-mev`, `--eta` |
| `spin_s` | float | yes | — | Compatibility input retained by the backend.<br>CLI aliases: `--spin-S`, `--S` |
| `bond_factor` | float | yes | — | Compatibility input retained by the backend.<br>CLI aliases: `--bond-factor` |
| `anisotropy_mev` | float | yes | — | Compatibility input retained by the backend.<br>CLI aliases: `--anisotropy-mev` |
| `q_stride` | int[3] | no | `[1, 1, 1]` | Uniform periodic q-grid stride; each stride must divide q_mesh_shape<br>CLI aliases: `--q-stride` |
| `exclude_shell` | list[int] | no | `[]` | Compatibility input retained by the backend.<br>CLI aliases: `--exclude-shell` |
| `exclude_shell_apply` | enum {both, J, dJ} | no | `both` | Compatibility input retained by the backend.<br>CLI aliases: `--exclude-shell-apply` |
| `shell_tol` | float | no | `0.0001` | Compatibility input retained by the backend.<br>CLI aliases: `--shell-tol` |
| `degeneracy_tol_mev` | float | no | `1e-08` | Absolute phonon-degeneracy tolerance for helicity gauge fixing<br>CLI aliases: `--degeneracy-tol-mev` |
| `chiralization_energy_offdiag_tol_mev` | float | no | `1e-10` | Maximum discarded phonon-Hamiltonian offdiagonal after finite-tolerance chiralization (default: 1e-10 meV)<br>CLI aliases: `--chiralization-energy-offdiag-tol-mev` |
| `chiralize_atom_weights` | list[float] | no | none / runtime | One signed helicity-operator weight per phonon atom<br>CLI aliases: `--chiralize-atom-weights` |
| `normalization_tol` | float | no | `1e-07` | Compatibility input retained by the backend.<br>CLI aliases: `--normalization-tol` |
| `geometry_tol` | float | no | `1e-08` | Compatibility input retained by the backend.<br>CLI aliases: `--geometry-tol` |
| `negative_phonon_tol_mev` | float | no | `1e-06` | Maximum negative phonon energy treated as acoustic numerical noise; more-negative modes fail closed<br>CLI aliases: `--negative-phonon-tol-mev` |
| `orthonormal_tol` | float | no | `1e-07` | Compatibility input retained by the backend.<br>CLI aliases: `--orthonormal-tol` |
| `angular_zero_tol` | float | no | `1e-10` | Compatibility input retained by the backend.<br>CLI aliases: `--angular-zero-tol` |
| `lte_gamma_bins` | int | no | `0` | Odd number &gt;=3 of physical LTe/hbar bins for Gamma histograms; zero disables the histogram (default: 0)<br>CLI aliases: `--lte-gamma-bins` |
| `lte_gamma_max_abs_over_hbar` | float | no | none / runtime | Optional strict symmetric physical-LTe histogram bound; the observed mode range is used when omitted<br>CLI aliases: `--lte-gamma-max-abs-over-hbar` |
| `phonon_floor_mev` | float | no | `0.001` | Compatibility input retained by the backend.<br>CLI aliases: `--phonon-floor-mev` |
| `dj_asr` | enum {none, check, project} | no | `check` | Compatibility input retained by the backend.<br>CLI aliases: `--dJ-asr`, `--dj-asr` |
| `dj_asr_tol` | float | no | `1e-08` | Compatibility input retained by the backend.<br>CLI aliases: `--dJ-asr-tol`, `--dj-asr-tol` |
| `reconstruction_tol` | float | no | `1e-09` | Compatibility input retained by the backend.<br>CLI aliases: `--reconstruction-tol` |
| `paraunitary_tol` | float | no | `1e-07` | Compatibility input retained by the backend.<br>CLI aliases: `--paraunitary-tol` |
| `metric_energy_tol_mev` | float | no | `1e-08` | Compatibility input retained by the backend.<br>CLI aliases: `--metric-energy-tol-mev` |
| `goldstone_node_policy` | enum {fail, omit} | no | `fail` | Fail on a defective/near-exact internal Goldstone quadrature node, or explicitly omit it while preserving the original q normalization<br>CLI aliases: `--goldstone-node-policy` |
| `goldstone_energy_tol_mev` | float | no | `1e-08` | Metric-positive magnon-energy threshold for Goldstone-node detection<br>CLI aliases: `--goldstone-energy-tol-mev` |
| `output` | string | yes | — | Output NPZ path<br>CLI aliases: `--output` |
| `summary_json` | string | no | none / runtime | Compatibility input retained by the backend.<br>CLI aliases: `--summary-json` |
| `save_vertices` | boolean | no | .false. | Explicitly store potentially huge component Lambda and g arrays<br>CLI aliases: `--save-vertices` |
| `allow_active_static_soc_diagnostic` | boolean | no | .false. | Allow a scalar-trace diagnostic with SOC metadata in static J; the output is marked invalid for exact SOC-off symmetry claims<br>CLI aliases: `--allow-active-static-soc-diagnostic` |
| `allow_unsafe_tensor_trace` | boolean | no | .false. | Allow diagnostic trace(dJ_tensor_r)/3 when no scalar dJ dataset exists<br>CLI aliases: `--allow-unsafe-tensor-trace` |
| `allow_unprojected_scalar_diagnostic` | boolean | no | .false. | Allow dJ without scalar_spin_group_projected provenance as diagnostic<br>CLI aliases: `--allow-unprojected-scalar-diagnostic` |
| `allow_missing_units_diagnostic` | boolean | no | .false. | Assume legacy static-J meV and/or dJ meV/angstrom only when explicit unit metadata is absent; marks output diagnostic-only<br>CLI aliases: `--allow-missing-units-diagnostic` |
| `allow_geometry_mismatch_diagnostic` | boolean | no | .false. | Allow unavailable/mismatched J/dJ/cache geometry as diagnostic-only<br>CLI aliases: `--allow-geometry-mismatch-diagnostic` |
| `geometry_epr` | string | no | none / runtime | Explicit EPR HDF5 used only to validate dynamic-dJ geometry after moving a projected file whose recorded absolute EPR path is stale<br>CLI aliases: `--geometry-epr` |
| `allow_unstable_phonon_diagnostic` | boolean | no | .false. | Clip phonons below the negative tolerance as diagnostic-only<br>CLI aliases: `--allow-unstable-phonon-diagnostic` |
| `allow_asr_none_diagnostic` | boolean | no | .false. | Allow --dJ-asr none as diagnostic-only<br>CLI aliases: `--allow-asr-none-diagnostic` |
| `allow_duplicate_bonds_diagnostic` | boolean | no | .false. | Allow duplicate J/dJ bond keys or multiplicity mismatch as diagnostic-only<br>CLI aliases: `--allow-duplicate-bonds-diagnostic` |
| `max_saved_vertices_gb` | float | no | `2.0` | Peak-memory safety limit used only with --save-vertices<br>CLI aliases: `--max-saved-vertices-gb` |
| `compressed` | boolean | no | .true. | Compatibility input retained by the backend.<br>CLI aliases: `--compressed`, `--no-compressed` |
| `overwrite` | boolean | no | .false. | Compatibility input retained by the backend.<br>CLI aliases: `--overwrite` |

### `calculation='chirality_plane'`

**Runtime requirements:** Requires the rotational inputs plus two-component
plane `kmesh`, fixed `kz`, and `output`. With more than one rank,
`numba_threads` is supplied from `&parallel` (defaulting to
`threads_per_worker`), `save_vertices` is forbidden, and only
`goldstone_node_policy='fail'` is accepted. Output, summary, and work
directories must be distinct. The calculation is restartable.

**Outputs:** Checkpoint manifest/common data and per-rank NPZ/JSON shards,
followed by merged NPZ and JSON from rank 0.

Backend: `slw.magph.legacy.reference.run_chirality_plane_mpi`; MPI backend:
`slw.magph.legacy.reference.run_chirality_plane_mpi`.

| Namelist key | Type | Parser required | Parser default | Meaning / CLI aliases |
|---|---|---:|---|---|
| `jr` | string | yes | — | Static scalar J real-space file<br>CLI aliases: `--jr` |
| `djr` | string | yes | — | Scalar dJ(R,Rp) HDF5/NPZ file<br>CLI aliases: `--djr` |
| `phonon_cache` | string | yes | — | Current-schema phonon cache NPZ<br>CLI aliases: `--phonon-cache` |
| `axis` | float[3] | yes | — | Compatibility input retained by the backend.<br>CLI aliases: `--axis` |
| `spin_pattern` | string | yes | — | Two-sublattice collinear signs, e.g. 1,-1<br>CLI aliases: `--spin-pattern` |
| `temperature_k` | float | yes | — | Compatibility input retained by the backend.<br>CLI aliases: `--temperature-k`, `--T` |
| `eta_mev` | float | yes | — | Compatibility input retained by the backend.<br>CLI aliases: `--eta-mev`, `--eta` |
| `spin_s` | float | yes | — | Compatibility input retained by the backend.<br>CLI aliases: `--spin-S`, `--S` |
| `bond_factor` | float | yes | — | Compatibility input retained by the backend.<br>CLI aliases: `--bond-factor` |
| `anisotropy_mev` | float | yes | — | Compatibility input retained by the backend.<br>CLI aliases: `--anisotropy-mev` |
| `q_stride` | int[3] | no | `[1, 1, 1]` | Uniform periodic q-grid stride; each stride must divide q_mesh_shape<br>CLI aliases: `--q-stride` |
| `exclude_shell` | list[int] | no | `[]` | Compatibility input retained by the backend.<br>CLI aliases: `--exclude-shell` |
| `exclude_shell_apply` | enum {both, J, dJ} | no | `both` | Compatibility input retained by the backend.<br>CLI aliases: `--exclude-shell-apply` |
| `shell_tol` | float | no | `0.0001` | Compatibility input retained by the backend.<br>CLI aliases: `--shell-tol` |
| `degeneracy_tol_mev` | float | no | `1e-08` | Absolute phonon-degeneracy tolerance for helicity gauge fixing<br>CLI aliases: `--degeneracy-tol-mev` |
| `chiralization_energy_offdiag_tol_mev` | float | no | `1e-10` | Maximum discarded phonon-Hamiltonian offdiagonal after finite-tolerance chiralization (default: 1e-10 meV)<br>CLI aliases: `--chiralization-energy-offdiag-tol-mev` |
| `chiralize_atom_weights` | list[float] | no | none / runtime | One signed helicity-operator weight per phonon atom<br>CLI aliases: `--chiralize-atom-weights` |
| `normalization_tol` | float | no | `1e-07` | Compatibility input retained by the backend.<br>CLI aliases: `--normalization-tol` |
| `geometry_tol` | float | no | `1e-08` | Compatibility input retained by the backend.<br>CLI aliases: `--geometry-tol` |
| `negative_phonon_tol_mev` | float | no | `1e-06` | Maximum negative phonon energy treated as acoustic numerical noise; more-negative modes fail closed<br>CLI aliases: `--negative-phonon-tol-mev` |
| `orthonormal_tol` | float | no | `1e-07` | Compatibility input retained by the backend.<br>CLI aliases: `--orthonormal-tol` |
| `angular_zero_tol` | float | no | `1e-10` | Compatibility input retained by the backend.<br>CLI aliases: `--angular-zero-tol` |
| `lte_gamma_bins` | int | no | `0` | Odd number &gt;=3 of physical LTe/hbar bins for Gamma histograms; zero disables the histogram (default: 0)<br>CLI aliases: `--lte-gamma-bins` |
| `lte_gamma_max_abs_over_hbar` | float | no | none / runtime | Optional strict symmetric physical-LTe histogram bound; the observed mode range is used when omitted<br>CLI aliases: `--lte-gamma-max-abs-over-hbar` |
| `phonon_floor_mev` | float | no | `0.001` | Compatibility input retained by the backend.<br>CLI aliases: `--phonon-floor-mev` |
| `dj_asr` | enum {none, check, project} | no | `check` | Compatibility input retained by the backend.<br>CLI aliases: `--dJ-asr`, `--dj-asr` |
| `dj_asr_tol` | float | no | `1e-08` | Compatibility input retained by the backend.<br>CLI aliases: `--dJ-asr-tol`, `--dj-asr-tol` |
| `reconstruction_tol` | float | no | `1e-09` | Compatibility input retained by the backend.<br>CLI aliases: `--reconstruction-tol` |
| `paraunitary_tol` | float | no | `1e-07` | Compatibility input retained by the backend.<br>CLI aliases: `--paraunitary-tol` |
| `metric_energy_tol_mev` | float | no | `1e-08` | Compatibility input retained by the backend.<br>CLI aliases: `--metric-energy-tol-mev` |
| `goldstone_node_policy` | enum {fail, omit} | no | `fail` | Fail on a defective/near-exact internal Goldstone quadrature node, or explicitly omit it while preserving the original q normalization<br>CLI aliases: `--goldstone-node-policy` |
| `goldstone_energy_tol_mev` | float | no | `1e-08` | Metric-positive magnon-energy threshold for Goldstone-node detection<br>CLI aliases: `--goldstone-energy-tol-mev` |
| `save_vertices` | boolean | no | .false. | Explicitly store potentially huge component Lambda and g arrays<br>CLI aliases: `--save-vertices` |
| `allow_active_static_soc_diagnostic` | boolean | no | .false. | Allow a scalar-trace diagnostic with SOC metadata in static J; the output is marked invalid for exact SOC-off symmetry claims<br>CLI aliases: `--allow-active-static-soc-diagnostic` |
| `allow_unsafe_tensor_trace` | boolean | no | .false. | Allow diagnostic trace(dJ_tensor_r)/3 when no scalar dJ dataset exists<br>CLI aliases: `--allow-unsafe-tensor-trace` |
| `allow_unprojected_scalar_diagnostic` | boolean | no | .false. | Allow dJ without scalar_spin_group_projected provenance as diagnostic<br>CLI aliases: `--allow-unprojected-scalar-diagnostic` |
| `allow_missing_units_diagnostic` | boolean | no | .false. | Assume legacy static-J meV and/or dJ meV/angstrom only when explicit unit metadata is absent; marks output diagnostic-only<br>CLI aliases: `--allow-missing-units-diagnostic` |
| `allow_geometry_mismatch_diagnostic` | boolean | no | .false. | Allow unavailable/mismatched J/dJ/cache geometry as diagnostic-only<br>CLI aliases: `--allow-geometry-mismatch-diagnostic` |
| `geometry_epr` | string | no | none / runtime | Explicit EPR HDF5 used only to validate dynamic-dJ geometry after moving a projected file whose recorded absolute EPR path is stale<br>CLI aliases: `--geometry-epr` |
| `allow_unstable_phonon_diagnostic` | boolean | no | .false. | Clip phonons below the negative tolerance as diagnostic-only<br>CLI aliases: `--allow-unstable-phonon-diagnostic` |
| `allow_asr_none_diagnostic` | boolean | no | .false. | Allow --dJ-asr none as diagnostic-only<br>CLI aliases: `--allow-asr-none-diagnostic` |
| `allow_duplicate_bonds_diagnostic` | boolean | no | .false. | Allow duplicate J/dJ bond keys or multiplicity mismatch as diagnostic-only<br>CLI aliases: `--allow-duplicate-bonds-diagnostic` |
| `max_saved_vertices_gb` | float | no | `2.0` | Peak-memory safety limit used only with --save-vertices<br>CLI aliases: `--max-saved-vertices-gb` |
| `compressed` | boolean | no | .true. | Compatibility input retained by the backend.<br>CLI aliases: `--compressed`, `--no-compressed` |
| `kmesh` | int[2] | yes | — | Periodic in-plane k mesh; ky changes fastest<br>CLI aliases: `--kmesh` |
| `kz` | float | yes | — | Fixed fractional kz<br>CLI aliases: `--kz` |
| `k_shift_grid` | float[2] | no | `[0.0, 0.0]` | Optional in-plane shift in units of one k-grid step<br>CLI aliases: `--k-shift-grid` |
| `block_size` | int | no | `8` | k points per atomic checkpoint block (default: 8)<br>CLI aliases: `--block-size` |
| `output` | string | yes | — | Merged plane NPZ<br>CLI aliases: `--output` |
| `summary_json` | string | no | none / runtime | Merged JSON report (default: output suffix changed to .json)<br>CLI aliases: `--summary-json` |
| `work_dir` | string | no | none / runtime | Checkpoint directory (default: OUTPUT stem plus .parts)<br>CLI aliases: `--work-dir` |
| `resume` | boolean | no | .true. | Reuse matching complete shards (default: true)<br>CLI aliases: `--resume`, `--no-resume` |
| `dry_run` | boolean | no | .false. | Validate inputs/configuration and print the MPI/block layout only<br>CLI aliases: `--dry-run` |
| `reference_seconds_per_k` | float | no | none / runtime | Optional measured seconds/k used only for a wall-time estimate<br>CLI aliases: `--reference-seconds-per-k` |

### `calculation='prepare_lifetime'`

**Runtime requirements:** `jr` is required and exactly one of `djr` and
`djdu_npz` must be selected. Cache/reuse mode requires `phonon_cache`;
from-scratch/recompute requires `epr_phonon` and an explicit
`phonon_qmesh`. Optional `input_file` may supply these values.

**Outputs:** `<out_name>.J_legacy.npz`, optional
`<out_name>.dJdu_legacy.npz` for non-HDF5 derivatives, optional rebuilt
phonon-cache NPZ, and always `<out_name>.manifest.json`.

Backend: `slw.magph.legacy.reference.prepare_lifetime`.

| Namelist key | Type | Parser required | Parser default | Meaning / CLI aliases |
|---|---|---:|---|---|
| `workdir` | str | no | none / runtime | Workflow root directory (default: current directory)<br>CLI aliases: `--workdir` |
| `input_file` | str | no | none / runtime | Optional input.in/magph.in with preparation variables<br>CLI aliases: `--input_file` |
| `in_dir` | str | no | none / runtime | Input directory containing dJ/du npz and phonon cache<br>CLI aliases: `--in_dir` |
| `out_dir` | str | no | none / runtime | Output directory for prepared artifacts<br>CLI aliases: `--out_dir` |
| `out_name` | str | no | `magph_lifetime_prep` | Output basename prefix<br>CLI aliases: `--out_name` |
| `djdu_npz` | str | no | none / runtime | Legacy real-space dJ NPZ<br>CLI aliases: `--djdu_npz` |
| `djr` | str | no | none / runtime | Real-space dJ input (dJr HDF5)<br>CLI aliases: `--djr`, `--djr_h5` |
| `jr` | str | no | none / runtime | Real-space J input (Jr HDF5, legacy NPZ, or text)<br>CLI aliases: `--jr`, `--j_cache` |
| `phonon_cache` | str | no | none / runtime | Phonon cache NPZ used in calculation_mode=cache<br>CLI aliases: `--phonon_cache` |
| `epr_phonon` | str | no | none / runtime | EPR HDF5 used in calculation_mode=from_scratch<br>CLI aliases: `--epr_phonon` |
| `phonon_fc` | str | no | none / runtime | Optional QE q2r force-constant file used instead of EPR IFCs<br>CLI aliases: `--phonon-fc` |
| `phonon_asr` | enum {none, simple, crystal} | no | none / runtime | Acoustic sum rule for --phonon-fc (default: none)<br>CLI aliases: `--phonon-asr` |
| `phonon_loto` | enum {auto, 2d, 3d, none} | no | none / runtime | LO-TO treatment for EPR Born charges (default: auto)<br>CLI aliases: `--phonon-loto` |
| `calculation_mode` | enum {cache, reuse, from_scratch, recompute} | no | none / runtime | Compatibility input retained by the backend.<br>CLI aliases: `--calculation_mode` |
| `phonon_qmesh` | int[3] | no | none / runtime | q mesh for --epr_phonon; default EPR basic_data/qc_dim<br>CLI aliases: `--phonon_qmesh` |
| `phonon_qshift` | parsed string[3] | no | none / runtime | Uniform q-mesh shift in grid-index units for --epr_phonon; components are canonicalized modulo integers (default: 0 0 0)<br>CLI aliases: `--phonon_qshift` |
| `phonon_cache_compressed` | boolean | no | none / runtime | Write phonon cache with np.savez_compressed; use --no-phonon_cache_compressed for faster writes<br>CLI aliases: `--phonon_cache_compressed`, `--no-phonon_cache_compressed` |
| `rp_idx` | int[3] | no | `[0, 0, 0]` | Rp index for the FM/legacy compatibility tuple; AFM atomic-gauge vertices use every Rp<br>CLI aliases: `--rp_idx` |

## `slw_post.x`

Read-only checks plus optional projections, reports, dumps, and plots.

| `calculation` | Backend source | MPI | Purpose |
|---|---|---:|---|
| `check_gkq` | default | no | Check g(k,q) invariants and sum rules |
| `check_spin_mz` | default | no | Check the spin-channel Mz relation for a compatible g(k,q) pair |
| `compare_u_rotation` | default | no | Compare EPR Hamiltonians with candidate Wannier U rotations |
| `dj_asr` | default | no | Audit or project the acoustic sum rule of tensor dJ/du |
| `dj_kq_symmetry` | default | no | Check dJ(k,q) symmetry for an EPR/Wannier input pair |
| `dump_dj` | default | no | Dump tensor dJ/du HDF5 data to text tables |
| `spin_group` | default | no | Audit or project scalar exchange under the spin space group |
| `j_diagnostic` | default | no | Write detailed EPR k-space exchange diagnostics |
| `rpa_lkag_audit` | default | no | Audit RPA and LKAG normalization conventions |
| `spinflip_hr` | default | no | Inspect spin-flip blocks in spinor Wannier Hamiltonians |
| `spinflip_orbitals` | default | no | Inspect orbital-resolved spin-flip blocks |
| `soc_bands` | default | no | Plot EPR bands with model spin-orbit coupling |
| `tensor_summary` | default | no | Inspect or adapt exchange tensors for magph consumers |
| `phonon_rotation` | default | no | Analyze phonon rotational selectivity |
| `lifetime_analysis` | default | no | Analyze lifetime output and symmetry channels |
| `lifetime_plot` | default | no | Plot lifetime data with reciprocal-space symmetry |
| `magnon_plot` | default | no | Plot a registered magnon dataset |
| `coupling_kpath` | default | no | Plot coupling along a reciprocal-space path |
| `coupling_bz` | default | no | Plot coupling over a Brillouin-zone plane |
| `coupling_kbz` | default | no | Plot fixed-q coupling over the magnon Brillouin zone |

### `calculation='check_gkq'`

**Runtime requirements:** One or more positional `inputs` are required. They
must contain the older dense `g_band`, `k_fracs`, and `q_fracs` schema and are
not directly interchangeable with current `slw_epr.x calculation='gkq'`
output. Cross-file invariant/sum-rule/shift comparisons require compatible
shapes and grids; `compare_shifts` is effective only with
`compare_invariants`.

**Outputs:** PASS/WARN/FAIL diagnostics on stdout and optional JSON report.

Backend: `slw.epc.check_gkq_constraints`.

| Namelist key | Type | Parser required | Parser default | Meaning / CLI aliases |
|---|---|---:|---|---|
| `inputs` | list[string] | yes | — | Input g(k,q) HDF5 files |
| `out_json` | str | no | none / runtime | Optional JSON output path<br>CLI aliases: `--out_json` |
| `skip_realspace` | boolean | no | .false. | Skip IFFT-based real-space relation checks<br>CLI aliases: `--skip_realspace` |
| `compare_invariants` | boolean | no | .false. | For compatible multi-file inputs, compare gauge-invariant Frobenius/SVD fingerprints<br>CLI aliases: `--compare_invariants` |
| `sum_rule` | boolean | no | .false. | For compatible multi-file inputs, check q=0 translational sum rule by summing over labels<br>CLI aliases: `--sum_rule` |
| `compare_samples` | int | no | `128` | Sample count for invariant pairwise comparisons<br>CLI aliases: `--compare_samples` |
| `compare_shifts` | boolean | no | .false. | For compatible multi-file inputs, compare real-space support maps \|\|g(Re,Rp)\|\|_F under cyclic shifts/inversions.<br>CLI aliases: `--compare_shifts` |
| `top_n` | int | no | `8` | How many dominant matrix elements to retain in reports<br>CLI aliases: `--top_n` |
| `show_top` | int | no | `5` | How many offending/top entries to print<br>CLI aliases: `--show_top` |
| `warn_rel` | float | no | `1e-06` | Relative-error threshold for PASS/WARN status<br>CLI aliases: `--warn_rel` |
| `primary_realspace_rule` | enum {minus, plus, same, none} | no | `minus` | Which real-space pairing rule participates in PASS/WARN status. Others are still reported diagnostically.<br>CLI aliases: `--primary_realspace_rule` |

### `calculation='check_spin_mz'`

**Runtime requirements:** Both `g_up` and `g_dn` are required, must contain the
older `g_band` schema, and must have identical shapes and k/q grids.

**Outputs:** Spin-Mz comparison on stdout and optional JSON report.

Backend: `slw.epc.check_spin_mz_relation`.

| Namelist key | Type | Parser required | Parser default | Meaning / CLI aliases |
|---|---|---:|---|---|
| `g_up` | string | yes | — | g(k,q) h5 for spin-up<br>CLI aliases: `--g_up` |
| `g_dn` | string | yes | — | g(k,q) h5 for spin-down<br>CLI aliases: `--g_dn` |
| `conjugate` | boolean | no | .false. | Apply complex conjugation to RHS matrix before compare<br>CLI aliases: `--conjugate` |
| `dagger` | boolean | no | .false. | Apply Hermitian conjugation to RHS matrix before compare<br>CLI aliases: `--dagger` |
| `out_json` | string | no | none / runtime | Optional report json<br>CLI aliases: `--out_json` |

### `calculation='compare_u_rotation'`

**Runtime requirements:** `win` is parser-required. Supply at least one
complete up- or down-spin `(epr, hr, u)` triplet; partial triplets are rejected
at runtime. `.win` and U-matrix k points must agree.

**Outputs:** Candidate-rotation error metrics on stdout only.

Backend: `slw.epc.compare_epr_u_rotation`.

| Namelist key | Type | Parser required | Parser default | Meaning / CLI aliases |
|---|---|---:|---|---|
| `win` | string | yes | — | Compatibility input retained by the backend.<br>CLI aliases: `--win` |
| `epr_up` | string | no | none / runtime | Compatibility input retained by the backend.<br>CLI aliases: `--epr_up` |
| `hr_up` | string | no | none / runtime | Compatibility input retained by the backend.<br>CLI aliases: `--hr_up` |
| `u_up` | string | no | none / runtime | Compatibility input retained by the backend.<br>CLI aliases: `--u_up` |
| `epr_dn` | string | no | none / runtime | Compatibility input retained by the backend.<br>CLI aliases: `--epr_dn` |
| `hr_dn` | string | no | none / runtime | Compatibility input retained by the backend.<br>CLI aliases: `--hr_dn` |
| `u_dn` | string | no | none / runtime | Compatibility input retained by the backend.<br>CLI aliases: `--u_dn` |

### `calculation='dj_asr'`

**Runtime requirements:** Positional tensor-dJ `h5` is required. Point-group
checks require `epr_h5`; setting `output` applies the ASR correction and writes
a new HDF5, while omitting it performs a read-only audit.

**Outputs:** ASR/symmetry diagnostics on stdout and, when requested, a new corrected HDF5.

Backend: `slw.exchange.legacy.check_dJ_asr`.

| Namelist key | Type | Parser required | Parser default | Meaning / CLI aliases |
|---|---|---:|---|---|
| `h5` | string | yes | — | Path to dJ_tensor HDF5 (compute_dJ_epr_tensor output) |
| `output` | string | no | none / runtime | Path to save the ASR-corrected HDF5 file. If specified, performs uniform ASR correction and writes the corrected file, then runs checks on it.<br>CLI aliases: `--output`, `-o` |
| `epr_h5` | string | no | none / runtime | Path to EPR HDF5 (epr_up.h5) to read lattice for symmetry orbit analysis. If not given, symmetry check is skipped.<br>CLI aliases: `--epr_h5` |
| `symprec` | float | no | `0.0001` | spglib symmetry tolerance for space group detection (default: 1e-4)<br>CLI aliases: `--symprec` |
| `verbose` | boolean | no | .false. | Print full per-bond 3x3 ASR residual matrices and per-orbit bond details<br>CLI aliases: `--verbose`, `-v` |
| `top_n` | int | no | `10` | Number of worst bonds to show in verbose ASR mode (default: 10)<br>CLI aliases: `--top_n` |
| `threshold` | float | no | none / runtime | Hide bonds/orbits with \|residual\| or sym_residual below this value (meV/A)<br>CLI aliases: `--threshold` |
| `cell_average` | boolean | no | .false. | Report the optional (1/n_Rp)*sum_Rp cell-average diagnostic instead of the default physical q=0 Fourier sum<br>CLI aliases: `--cell_average` |
| `no_q0_norm` | boolean | no | .false. | Deprecated compatibility alias; the physical sum_Rp is now the default (no effect)<br>CLI aliases: `--no_q0_norm` |

### `calculation='dj_kq_symmetry'`

**Runtime requirements:** `res_dir`, spin-resolved g(k,q), Fermi level,
prefixes, three-component k mesh, two-site bond pair, orbital slices, and
magnetic atoms are required. Spin files must have matching shapes, targets,
and meshes. `with_tau_phase=.true.` requires `win` or exactly one discoverable
`.win` under `res_dir`.

**Outputs:** Symmetry diagnostics on stdout and optional JSON.

Backend: `slw.exchange.legacy.check_dJkq_symmetry`.

| Namelist key | Type | Parser required | Parser default | Meaning / CLI aliases |
|---|---|---:|---|---|
| `res_dir` | string | yes | — | Directory containing Wannier90 hr.dat files<br>CLI aliases: `--res_dir` |
| `gkq_up` | string | yes | — | g(k,q) HDF5 for spin-up<br>CLI aliases: `--gkq_up` |
| `gkq_dn` | string | yes | — | g(k,q) HDF5 for spin-down<br>CLI aliases: `--gkq_dn` |
| `efermi` | float | yes | — | Fermi energy in eV<br>CLI aliases: `--efermi` |
| `prefix_up` | str | yes | — | Compatibility input retained by the backend.<br>CLI aliases: `--prefix_up` |
| `prefix_dn` | str | yes | — | Compatibility input retained by the backend.<br>CLI aliases: `--prefix_dn` |
| `kmesh` | int[3] | yes | — | Green-function k-grid<br>CLI aliases: `--kmesh` |
| `pair` | int[2] | yes | — | 0-based magnetic atom pair (i j)<br>CLI aliases: `--pair` |
| `with_tau_phase` | boolean | no | .false. | Apply exp(i 2pi q.(tau_j-tau_i)) phase in complex symmetry diagnostics<br>CLI aliases: `--with_tau_phase` |
| `win` | str | no | none / runtime | Wannier90 .win path for tau-phase correction<br>CLI aliases: `--win` |
| `slices` | string | yes | — | Local slices as '0:0:5,1:5:10'<br>CLI aliases: `--slices` |
| `mag_atoms` | list[int] | yes | — | 0-based atomic indices corresponding to magnetic-site indices<br>CLI aliases: `--mag_atoms` |
| `ddelta_mode` | enum {off, qavg} | no | `off` | Onsite dDelta in momentum space<br>CLI aliases: `--ddelta_mode` |
| `integrator` | enum {contour, cfr, cfr_ozaki} | no | `contour` | Compatibility input retained by the backend.<br>CLI aliases: `--integrator` |
| `emin` | float | no | `-25.0` | Compatibility input retained by the backend.<br>CLI aliases: `--emin` |
| `empoints` | int | no | `120` | Compatibility input retained by the backend.<br>CLI aliases: `--empoints` |
| `cfr_beta` | float | no | `400.0` | Compatibility input retained by the backend.<br>CLI aliases: `--cfr_beta` |
| `out_json` | str | no | none / runtime | Optional output JSON path<br>CLI aliases: `--out_json` |

### `calculation='dump_dj'`

**Runtime requirements:** Positional tensor-dJ `h5` is required. Unless
`all_rp=.true.`, the selected three-component `rp` must exist in the file.
Output directory/name controls select the generated text tables.

**Outputs:** Human-readable TXT and machine-readable TSV tables.

Backend: `slw.exchange.legacy.dump_dJ_epr_tensor`.

| Namelist key | Type | Parser required | Parser default | Meaning / CLI aliases |
|---|---|---:|---|---|
| `h5` | string | yes | — | dJ tensor HDF5 from compute_dJ_epr_tensor |
| `out` | string | no | none / runtime | Readable text output path<br>CLI aliases: `-o`, `--out` |
| `tsv` | string | no | none / runtime | TSV output path<br>CLI aliases: `--tsv` |
| `rp` | int[3] | no | `[0, 0, 0]` | Rp cell to dump; default 0 0 0<br>CLI aliases: `--rp` |
| `all_rp` | boolean | no | .false. | Dump every Rp instead of a single Rp<br>CLI aliases: `--all_rp` |
| `threshold` | float | no | none / runtime | Omit entries with absolute value below threshold<br>CLI aliases: `--threshold` |

### `calculation='spin_group'`

**Runtime requirements:** Positional static-J/dJ `input` and `epr_h5` are
required. Exactly one of new, no-clobber `output` and `audit_only=.true.` must
be selected.

**Outputs:** JSON-form diagnostics on stdout. Audit mode can write
`report_json`; projection writes a new HDF5 and a JSON report defaulting to
`OUTPUT.projection.json`.

Backend: `slw.exchange.legacy.project_scalar_spin_group`.

| Namelist key | Type | Parser required | Parser default | Meaning / CLI aliases |
|---|---|---:|---|---|
| `input` | string | yes | — | Input static-J and/or dJ HDF5 |
| `epr_h5` | string | yes | — | EPR HDF5 supplying lattice, positions, and species<br>CLI aliases: `--epr-h5` |
| `output` | string | one of: `output`, `audit_only` | none / runtime | New projected HDF5; existing paths are refused<br>CLI aliases: `--output` |
| `audit_only` | boolean | one of: `output`, `audit_only` | .false. | Read-only audit; no HDF5 is written<br>CLI aliases: `--audit-only` |
| `report_json` | string | no | none / runtime | JSON report path (default: OUTPUT.projection.json)<br>CLI aliases: `--report-json` |
| `symprec` | float | no | `0.0001` | Compatibility input retained by the backend.<br>CLI aliases: `--symprec` |
| `angle_tolerance` | float | no | `-1.0` | Compatibility input retained by the backend.<br>CLI aliases: `--angle-tolerance` |
| `spin_pattern` | list[float] | no | none / runtime | One collinear scalar per crystal atom; retain operations mapping it to itself or its global negative<br>CLI aliases: `--spin-pattern` |
| `enforce_asr` | boolean | no | .true. | Do not impose sum_(kappa,Rp) dJ=0 (ASR is enabled by default)<br>CLI aliases: `--no-asr` |

### `calculation='j_diagnostic'`

**Runtime requirements:** EPR up/down files, magnetic atoms, orbital slices, Fermi level, and k mesh are required.

**Outputs:** Detailed k-space diagnostic TSV files.

Backend: `slw.exchange.legacy.diagnose_J_epr_kspace`.

| Namelist key | Type | Parser required | Parser default | Meaning / CLI aliases |
|---|---|---:|---|---|
| `epr_up` | string | yes | — | Compatibility input retained by the backend.<br>CLI aliases: `--epr_up` |
| `epr_dn` | string | yes | — | Compatibility input retained by the backend.<br>CLI aliases: `--epr_dn` |
| `hr_unit` | enum {ev, ry, ha} | no | `ry` | Compatibility input retained by the backend.<br>CLI aliases: `--hr_unit` |
| `mag_atoms` | list[int] | yes | — | Compatibility input retained by the backend.<br>CLI aliases: `--mag_atoms` |
| `mag_atoms_base` | enum {0, 1} | no | `0` | Compatibility input retained by the backend.<br>CLI aliases: `--mag_atoms_base` |
| `slices` | string | yes | — | Local slices as '0:0:5,1:5:10'<br>CLI aliases: `--slices` |
| `efermi` | float | yes | — | Compatibility input retained by the backend.<br>CLI aliases: `--efermi` |
| `kmesh` | int[3] | yes | — | Compatibility input retained by the backend.<br>CLI aliases: `--kmesh` |
| `n_shells` | int | no | `1` | Compatibility input retained by the backend.<br>CLI aliases: `--n_shells` |
| `nn_only` | boolean | no | .true. | Compatibility input retained by the backend.<br>CLI aliases: `--nn_only` |
| `d_max` | float | no | `20.0` | Compatibility input retained by the backend.<br>CLI aliases: `--d_max` |
| `emin` | float | no | `-25.0` | Compatibility input retained by the backend.<br>CLI aliases: `--emin` |
| `empoints` | int | no | `500` | Compatibility input retained by the backend.<br>CLI aliases: `--empoints` |
| `integrator` | enum {contour, cfr, cfr_ozaki} | no | `contour` | Compatibility input retained by the backend.<br>CLI aliases: `--integrator` |
| `cfr_beta` | float | no | `400.0` | Compatibility input retained by the backend.<br>CLI aliases: `--cfr_beta` |
| `out_dir` | string | no | `J_epr_kdiag` | Compatibility input retained by the backend.<br>CLI aliases: `--out_dir` |
| `summary_name` | string | no | `summary.tsv` | Compatibility input retained by the backend.<br>CLI aliases: `--summary_name` |
| `energy_name` | string | no | `energy_components.tsv` | Compatibility input retained by the backend.<br>CLI aliases: `--energy_name` |

### `calculation='rpa_lkag_audit'`

**Runtime requirements:** `rpa`, `jr`, and a nonzero value per magnetic site
in `spin_pattern` are required. Site/moment counts must agree, J must use meV,
and `s` must be `'projected'` or positive.

**Outputs:** Convention-comparison TSV report.

Backend: `slw.exchange.legacy.audit_rpa_lkag_factor`.

| Namelist key | Type | Parser required | Parser default | Meaning / CLI aliases |
|---|---|---:|---|---|
| `rpa` | string | yes | — | RPA diagnostic HDF5<br>CLI aliases: `--rpa` |
| `jr` | string | yes | — | LKAG/TB2J-style J HDF5<br>CLI aliases: `--jr` |
| `j_source` | string | no | `J_iso_r` | Compatibility input retained by the backend.<br>CLI aliases: `--j_source`, `--j-source` |
| `s` | string | no | `projected` | Spin magnitude or 'projected' to use mean(abs(M_i))/2 from RPA<br>CLI aliases: `--S` |
| `spin_pattern` | list[float] | yes | — | Compatibility input retained by the backend.<br>CLI aliases: `--spin_pattern`, `--spin-pattern` |
| `bond_factors` | list[float] | no | `[1.0, 2.0]` | 1=current generic BdG; 2=TB2J double-counted i!=j Hamiltonian<br>CLI aliases: `--bond_factors`, `--bond-factors` |
| `zero_tol_mev` | float | no | `1e-05` | Compatibility input retained by the backend.<br>CLI aliases: `--zero_tol_mev`, `--zero-tol-mev` |
| `output` | string | no | `rpa_lkag_factor_audit.tsv` | Compatibility input retained by the backend.<br>CLI aliases: `--output` |

### `calculation='spinflip_hr'`

**Runtime requirements:** At least one even-dimensional, spin-major spinor HR
file is required. With `no_rspace=.true.`, provide `kmesh` or `kpoints` to
produce a k-space diagnostic.

**Outputs:** Inspection diagnostics on stdout.

Backend: `slw.exchange.legacy.diagnose_spinflip_hr`.

| Namelist key | Type | Parser required | Parser default | Meaning / CLI aliases |
|---|---|---:|---|---|
| `hr` | list[string] | yes | — | Spin-major spinor Wannier90 hr.dat file(s) |
| `labels` | list[string] | no | none / runtime | Optional labels for input files<br>CLI aliases: `--labels` |
| `top` | int | no | `20` | Number of largest R blocks to print<br>CLI aliases: `--top` |
| `tol` | float | no | `0.0` | Omit R blocks with max spin-flip norm below this value<br>CLI aliases: `--tol` |
| `kmesh` | int[3] | no | none / runtime | Also diagnose H_ud(k) on a regular fractional k mesh<br>CLI aliases: `--kmesh` |
| `kpoints` | string | no | none / runtime | Also diagnose explicit fractional k points, e.g. '0 0 0; 0.5 0 0'<br>CLI aliases: `--kpoints` |
| `no_rspace` | boolean | no | .false. | Skip R-space diagnostics<br>CLI aliases: `--no_rspace` |

### `calculation='spinflip_orbitals'`

**Runtime requirements:** Exactly one of `hr` and `eig`, plus `win` and
three-component `kmesh`, is required. The eigenvalue path additionally needs
at least one of `u_dis_mat` and `u_mat`.

**Outputs:** Orbital-block diagnostics on stdout and optional CSV.

Backend: `slw.exchange.legacy.diagnose_spinflip_orbital_blocks`.

| Namelist key | Type | Parser required | Parser default | Meaning / CLI aliases |
|---|---|---:|---|---|
| `hr` | string | one of: `hr`, `eig` | none / runtime | Spinor hr.dat source<br>CLI aliases: `--hr` |
| `eig` | string | one of: `hr`, `eig` | none / runtime | Wannier90 .eig source<br>CLI aliases: `--eig` |
| `u_dis_mat` | string | no | none / runtime | Wannier90 *_u_dis.mat for --eig reconstruction<br>CLI aliases: `--u_dis_mat` |
| `u_mat` | string | no | none / runtime | Optional Wannier90 *_u.mat; omit to stay before final U rotation<br>CLI aliases: `--u_mat` |
| `w90_rotation` | enum {udag_h_u, u_h_udag} | no | `udag_h_u` | Compatibility input retained by the backend.<br>CLI aliases: `--w90_rotation` |
| `u_match_tol` | float | no | `1e-07` | Compatibility input retained by the backend.<br>CLI aliases: `--u_match_tol` |
| `u_nearest` | boolean | no | .false. | Compatibility input retained by the backend.<br>CLI aliases: `--u_nearest` |
| `win` | string | yes | — | Wannier90 .win defining projection groups<br>CLI aliases: `--win` |
| `kmesh` | int[3] | yes | — | Compatibility input retained by the backend.<br>CLI aliases: `--kmesh` |
| `kpoints` | string | no | none / runtime | Compatibility input retained by the backend.<br>CLI aliases: `--kpoints` |
| `spin_order` | enum {spin_major, orbital_interleaved, win_interleaved} | no | `spin_major` | Compatibility input retained by the backend.<br>CLI aliases: `--spin_order` |
| `group_by` | enum {element_orbital, atom_orbital, group} | no | `element_orbital` | Compatibility input retained by the backend.<br>CLI aliases: `--group_by` |
| `blocks` | enum {ud, du, both} | no | `both` | Compatibility input retained by the backend.<br>CLI aliases: `--blocks` |
| `top` | int | no | `20` | Compatibility input retained by the backend.<br>CLI aliases: `--top` |
| `csv` | string | no | none / runtime | Optional CSV output for all block ratios<br>CLI aliases: `--csv` |

### `calculation='soc_bands'`

**Runtime requirements:** Spin-resolved EPR files and `win` are required.
`ref='vbm'` needs `nvalence`; `soc_gauge='rotated'` needs either shared
`u_mat` or the complete `u_up_mat` + `u_dn_mat` pair. SOC definitions and path
controls should be explicit.

**Outputs:** Band plot image and optional NPZ data.

Backend: `slw.exchange.legacy.plot_epr_soc_bands`.

| Namelist key | Type | Parser required | Parser default | Meaning / CLI aliases |
|---|---|---:|---|---|
| `epr_up` | string | yes | — | Compatibility input retained by the backend.<br>CLI aliases: `--epr_up` |
| `epr_dn` | string | yes | — | Compatibility input retained by the backend.<br>CLI aliases: `--epr_dn` |
| `hr_unit` | enum {ev, ry, ha} | no | `ry` | Compatibility input retained by the backend.<br>CLI aliases: `--hr_unit` |
| `efermi` | float | no | `0.0` | Fermi energy in eV subtracted from bands<br>CLI aliases: `--efermi` |
| `ref` | enum {fermi, vbm, zero} | no | `fermi` | Energy reference: fermi subtracts --efermi; vbm subtracts valence-band maximum; zero subtracts nothing<br>CLI aliases: `--ref` |
| `nvalence` | int | no | none / runtime | Number of occupied spinor bands for --ref vbm<br>CLI aliases: `--nvalence` |
| `soc` | string | no | `''` | Model SOC specs inferred from .win projections, e.g. 'I:p:0.6;Cr:d:0.05'<br>CLI aliases: `--soc` |
| `lambda_te` | float | no | `0.0` | Compatibility onsite p SOC lambda in eV; prefer --soc<br>CLI aliases: `--lambda_te` |
| `soc_p_groups` | string | no | `''` | Semicolon-separated p orbital groups, e.g. '10,11,12;25,26,27'<br>CLI aliases: `--soc_p_groups` |
| `soc_p_groups_base` | enum {0, 1} | no | `0` | Compatibility input retained by the backend.<br>CLI aliases: `--soc_p_groups_base` |
| `win` | string | yes | — | Wannier90 .win with projections and kpoint_path<br>CLI aliases: `--win` |
| `soc_gauge` | enum {wannier, rotated} | no | `wannier` | wannier adds L.S directly in Wannier index space; rotated adds U(k) L.S U(k)^dagger<br>CLI aliases: `--soc_gauge` |
| `u_mat` | string | no | none / runtime | Common Wannier90 *_u.mat, .npy, or .npz U(k) file for --soc_gauge rotated<br>CLI aliases: `--u_mat` |
| `u_up_mat` | string | no | none / runtime | Spin-up U(k) file for --soc_gauge rotated<br>CLI aliases: `--u_up_mat` |
| `u_dn_mat` | string | no | none / runtime | Spin-down U(k) file for --soc_gauge rotated<br>CLI aliases: `--u_dn_mat` |
| `soc_rotation` | enum {u_soc_udag, udag_soc_u} | no | `u_soc_udag` | Rotation convention for --soc_gauge rotated<br>CLI aliases: `--soc_rotation` |
| `u_match_tol` | float | no | `1e-07` | Fractional k tolerance for matching U(k) to the plot k-path<br>CLI aliases: `--u_match_tol` |
| `u_nearest` | boolean | no | .false. | Diagnostic only: use nearest U(k) if exact k-path matching fails<br>CLI aliases: `--u_nearest` |
| `soc_element` | string | no | `''` | Element for compatibility --lambda_te p-SOC mode<br>CLI aliases: `--soc_element` |
| `p_order` | string | no | `pz,px,py` | p orbital order inside each group<br>CLI aliases: `--p_order` |
| `d_order` | string | no | `dz2,dxz,dyz,dx2-y2,dxy` | d orbital order inside each SOC group<br>CLI aliases: `--d_order` |
| `spin_direction` | float[3] | no | `[0.0, 0.0, 1.0]` | Compatibility input retained by the backend.<br>CLI aliases: `--spin_direction` |
| `band_points` | int | no | `80` | Points per k-path segment<br>CLI aliases: `--band_points` |
| `output` | string | no | `epr_soc_bands.png` | Compatibility input retained by the backend.<br>CLI aliases: `-o`, `--output` |
| `save_npz` | string | no | none / runtime | Compatibility input retained by the backend.<br>CLI aliases: `--save_npz` |
| `ymin` | float | no | none / runtime | Compatibility input retained by the backend.<br>CLI aliases: `--ymin` |
| `ymax` | float | no | none / runtime | Compatibility input retained by the backend.<br>CLI aliases: `--ymax` |
| `fig_width` | float | no | `7.0` | Compatibility input retained by the backend.<br>CLI aliases: `--fig_width` |
| `fig_height` | float | no | `4.5` | Compatibility input retained by the backend.<br>CLI aliases: `--fig_height` |
| `dpi` | int | no | `300` | Compatibility input retained by the backend.<br>CLI aliases: `--dpi` |
| `color` | string | no | `black` | Compatibility input retained by the backend.<br>CLI aliases: `--color` |
| `linewidth` | float | no | `0.8` | Compatibility input retained by the backend.<br>CLI aliases: `--linewidth` |
| `alpha` | float | no | `0.9` | Compatibility input retained by the backend.<br>CLI aliases: `--alpha` |
| `title` | string | no | none / runtime | Compatibility input retained by the backend.<br>CLI aliases: `--title` |

### `calculation='tensor_summary'`

**Runtime requirements:** Positional tensor-dJ HDF5 `h5` is required, and each
requested component must exist. This operation prints an inspection summary
and does not write a scientific dataset.

**Outputs:** Summary and selected entries on stdout.

Backend: `slw.magph.legacy.tensor_adapter`.

| Namelist key | Type | Parser required | Parser default | Meaning / CLI aliases |
|---|---|---:|---|---|
| `h5` | string | yes | — | compute_dJ_epr_tensor HDF5 file |
| `components` | string | no | `dmi,iso,aniso` | Comma-separated components to load: dmi, iso, aniso<br>CLI aliases: `--components` |
| `show_component` | string | no | `dmi` | Component to print: dmi, iso, aniso<br>CLI aliases: `--show-component` |
| `threshold` | float | no | `0.0` | Threshold for nonzero-entry count<br>CLI aliases: `--threshold` |
| `show` | int | no | `5` | Print first N flat entries above threshold<br>CLI aliases: `--show` |

### `calculation='phonon_rotation'`

**Runtime requirements:** Current-schema `phonon_cache`, three-component
rotation `axis`, and `output` are required. Optional chiralization weights must
match the phonon atom count; an existing output requires `overwrite=.true.`.

**Outputs:** Rotational-selectivity NPZ and JSON, with JSON defaulting to the
output stem.

Backend: `slw.magph.legacy.analyze_rotational_selectivity`.

| Namelist key | Type | Parser required | Parser default | Meaning / CLI aliases |
|---|---|---:|---|---|
| `phonon_cache` | string | yes | — | Input phonon cache NPZ<br>CLI aliases: `--phonon-cache` |
| `axis` | float[3] | yes | — | Cartesian analysis axis (normalized internally)<br>CLI aliases: `--axis` |
| `output` | string | yes | — | Output analysis NPZ path<br>CLI aliases: `--output` |
| `summary_json` | string | no | none / runtime | JSON summary path (default: --output with suffix .json)<br>CLI aliases: `--summary-json` |
| `degeneracy_tol_mev` | float | no | none / runtime | Absolute energy-span tolerance for helicity diagonalization in degenerate mode blocks; omit to preserve the input mode gauge<br>CLI aliases: `--degeneracy-tol-mev` |
| `normalization_tol` | float | no | `1e-07` | Maximum allowed \|sum M\|u\|^2 - 1\| (default: 1e-7)<br>CLI aliases: `--normalization-tol` |
| `chiralization_atom_weights` | list[float] | no | none / runtime | Optional one-weight-per-atom local helicity operator used during degenerate chiralization (e.g. signed weights for a staggered mode)<br>CLI aliases: `--chiralization-atom-weights` |
| `orthonormal_tol` | float | no | `1e-07` | Maximum mass-overlap error allowed before chiralization (default: 1e-7)<br>CLI aliases: `--orthonormal-tol` |
| `chunk_size` | int | no | `256` | Number of q points processed per vectorized chunk (default: 256)<br>CLI aliases: `--chunk-size` |
| `save_chiralized_vectors` | boolean | no | .true. | Store rotated eigenvectors when chiralization is enabled (default: true)<br>CLI aliases: `--save-chiralized-vectors`, `--no-save-chiralized-vectors` |
| `compressed` | boolean | no | .true. | Compress the output NPZ (default: true)<br>CLI aliases: `--compressed`, `--no-compressed` |
| `overwrite` | boolean | no | .false. | Replace existing outputs; default behavior is no-clobber<br>CLI aliases: `--overwrite` |

### `calculation='lifetime_analysis'`

**Runtime requirements:** Positional lifetime NPZ `input` with linewidth and
k-mesh data is required. Automatic symmetry needs `structure`; explicit
symmetry needs both nine-component `rotation` and `channel_map`. The rotated
mesh must map bijectively.

**Outputs:** Analysis NPZ/JSON and optional plot outputs.

Backend: `slw.magph.legacy.analyze_lifetime`.

| Namelist key | Type | Parser required | Parser default | Meaning / CLI aliases |
|---|---|---:|---|---|
| `input` | string | yes | — | NPZ from `slw_magph.x` with `calculation='lifetime'` |
| `structure` | string | no | none / runtime | POSCAR/card structure for automatic symmetry discovery<br>CLI aliases: `--structure` |
| `linewidth_key` | string | no | `linewidth` | Compatibility input retained by the backend.<br>CLI aliases: `--linewidth-key` |
| `energy_key` | string | no | `energy` | Compatibility input retained by the backend.<br>CLI aliases: `--energy-key` |
| `kpoints_key` | string | no | `k_mesh_frac` | Compatibility input retained by the backend.<br>CLI aliases: `--kpoints-key` |
| `qpoints_key` | string | no | `q_mesh_frac` | Compatibility input retained by the backend.<br>CLI aliases: `--qpoints-key` |
| `physical_channel_count` | int | no | none / runtime | Compatibility input retained by the backend.<br>CLI aliases: `--physical-channel-count` |
| `magnetic_atom_indices` | list[int] | no | none / runtime | Compatibility input retained by the backend.<br>CLI aliases: `--magnetic-atom-indices` |
| `spin_pattern` | string | no | none / runtime | Compatibility input retained by the backend.<br>CLI aliases: `--spin-pattern` |
| `rotation` | int[9] | no | none / runtime | Explicit direct-space fractional rotation matrix<br>CLI aliases: `--rotation` |
| `channel_map` | list[int] | no | none / runtime | Compatibility input retained by the backend.<br>CLI aliases: `--channel-map` |
| `symmetry_operation_index` | int | no | none / runtime | Compatibility input retained by the backend.<br>CLI aliases: `--symmetry-operation-index` |
| `symprec` | float | no | `1e-05` | Compatibility input retained by the backend.<br>CLI aliases: `--symprec` |
| `atom_tolerance_ang` | float | no | `0.0001` | Compatibility input retained by the backend.<br>CLI aliases: `--atom-tolerance-ang` |
| `mesh_tolerance` | float | no | `1e-08` | Compatibility input retained by the backend.<br>CLI aliases: `--mesh-tolerance` |
| `workers` | int | no | `-1` | Compatibility input retained by the backend.<br>CLI aliases: `--workers` |
| `relative_floor` | float | no | none / runtime | Compatibility input retained by the backend.<br>CLI aliases: `--relative-floor` |
| `absolute_tolerance` | float | no | `1e-08` | Compatibility input retained by the backend.<br>CLI aliases: `--absolute-tolerance` |
| `energy_absolute_tolerance` | float | no | `1e-08` | Compatibility input retained by the backend.<br>CLI aliases: `--energy-absolute-tolerance` |
| `relative_tolerance` | float | no | `1e-06` | Compatibility input retained by the backend.<br>CLI aliases: `--relative-tolerance` |
| `output` | string | no | none / runtime | Compatibility input retained by the backend.<br>CLI aliases: `-o`, `--output` |
| `summary` | string | no | none / runtime | Compatibility input retained by the backend.<br>CLI aliases: `--summary` |
| `plot` | string | no | none / runtime | Compatibility input retained by the backend.<br>CLI aliases: `--plot` |
| `plane_axes` | int[2] | no | `[0, 1]` | Compatibility input retained by the backend.<br>CLI aliases: `--plane-axes` |
| `slice_value` | float | no | `0.0` | Compatibility input retained by the backend.<br>CLI aliases: `--slice-value` |
| `slice_tolerance` | float | no | `1e-08` | Compatibility input retained by the backend.<br>CLI aliases: `--slice-tolerance` |
| `dpi` | int | no | `180` | Compatibility input retained by the backend.<br>CLI aliases: `--dpi` |

### `calculation='lifetime_plot'`

**Runtime requirements:** Positional lifetime NPZ `input` and `output_dir` are
required. Native schema-v2 keys are read directly. New products embed lattice
and atom geometry; older products may supply `exchange_h5`. For schema-v1 AFM
products that file is also used to reconstruct the inexpensive LSWT chirality
labels and consistently reorder every existing observable without rerunning
self-energy. The retained
`k_mesh_frac`, `energy`, and `linewidth` array aliases remain readable.
Automatic rotation-error plots additionally need embedded magnetic/atomic
metadata, while explicit symmetry needs both `rotation` and `channel_map`.

**Outputs:** PNG/PDF/SVG figures and `plot_summary.json` in `output_dir`.

Backend: `slw.magph.lifetime_plot`.

| Namelist key | Type | Parser required | Parser default | Meaning / CLI aliases |
|---|---|---:|---|---|
| `input` | string | yes | — | NPZ from `slw_magph.x` with `calculation='lifetime'` |
| `output_dir` | string | yes | — | Figure and summary directory<br>CLI aliases: `--output-dir` |
| `exchange_h5` | string | no | metadata / runtime | Geometry and AFM-chirality reconstruction source for old native products<br>CLI aliases: `--exchange-h5` |
| `slices` | list[float] | no | `[0.0, 0.5]` | Requested periodic coordinates along the plane normal; each snaps to the closest stored plane<br>CLI aliases: `--slices` |
| `all_slices` | boolean | no | .false. | Plot every stored plane along the axis normal to `plane_axes`<br>CLI aliases: `--all-slices` |
| `plots` | list[enum {all, energy, linewidth, fwhm, scattering-rate, lifetime, splitting, rotation-error}] | no | `["all"]` | `all` generates energy, HWHM, rate, lifetime, and splitting when at least two modes exist<br>CLI aliases: `--plots` |
| `plane_axes` | int[2] | no | `[0, 1]` | Fractional reciprocal axes spanning each BZ map<br>CLI aliases: `--plane-axes` |
| `physical_mode_count` | int | no | all stored modes | Optional leading physical-mode count; old alias `physical_channel_count` is accepted<br>CLI aliases: `--physical-mode-count`, `--physical-channel-count` |
| `split_modes` | int[2] | no | `[0, 1]` | Zero-based canonical modes used for splitting. For AFM the default is signed `E_chi+ - E_chi-` and the analogous linewidth difference; old alias `split_channels` is accepted<br>CLI aliases: `--split-modes`, `--split-channels` |
| `rotation` | int[9] | no | automatic / runtime | Explicit direct-space fractional rotation for `rotation-error`<br>CLI aliases: `--rotation` |
| `channel_map` | list[int] | no | automatic / runtime | Explicit mode permutation paired with `rotation`<br>CLI aliases: `--channel-map` |
| `symmetry_operation_index` | int | no | automatic best operation | Select one spglib sublattice-transposing operation<br>CLI aliases: `--symmetry-operation-index` |
| `symprec` | float | no | `1e-05` | spglib symmetry tolerance<br>CLI aliases: `--symprec` |
| `atom_tolerance_ang` | float | no | `0.0001` | Cartesian tolerance for atom maps<br>CLI aliases: `--atom-tolerance-ang` |
| `mesh_tolerance` | float | no | `1e-08` | Maximum periodic k-mesh mapping distance for rotation comparison<br>CLI aliases: `--mesh-tolerance` |
| `workers` | int | no | `-1` | scipy k-d tree workers for rotation comparison<br>CLI aliases: `--workers` |
| `k_round` | int | no | `8` | Decimal precision used to identify stored planes<br>CLI aliases: `--k-round` |
| `formats` | list[enum {png, pdf, svg}] | no | `["png", "pdf"]` | Figure formats<br>CLI aliases: `--formats` |
| `dpi` | int | no | `300` | Raster resolution<br>CLI aliases: `--dpi` |
| `panel_width` | float | no | `4.4` | Width per magnon-mode panel<br>CLI aliases: `--panel-width` |
| `panel_height` | float | no | `4.0` | Figure panel height<br>CLI aliases: `--panel-height` |
| `tile` | int | no | `1` | Minimum periodic image range for Voronoi construction<br>CLI aliases: `--tile` |
| `bz_mode` | enum {clip, periodic} | no | `clip` | Clip to the first BZ or display periodic copies with BZ outlines<br>CLI aliases: `--bz-mode` |
| `periodic_repeats` | int | no | `1` | Neighboring BZ repeats for `periodic_view='neighbors'`<br>CLI aliases: `--periodic-repeats` |
| `periodic_view` | enum {central, neighbors} | no | `central` | Central BZ with periodic padding or multiple neighboring BZs<br>CLI aliases: `--periodic-view` |
| `periodic_padding` | float | no | `0.06` | Fractional padding outside the central first-BZ bounding box<br>CLI aliases: `--periodic-padding` |
| `margin` | float | no | `0.035` | Axis margin around clipped BZ maps<br>CLI aliases: `--margin` |
| `linewidth_scale` | float | no | `1.0` | Display multiplier applied to HWHM/FWHM and linewidth splitting<br>CLI aliases: `--linewidth-scale` |
| `linewidth_unit` | string | no | `meV` | Label paired with `linewidth_scale`<br>CLI aliases: `--linewidth-unit` |
| `scattering_rate_scale` | float | no | `1.0` | Display multiplier applied to rates<br>CLI aliases: `--scattering-rate-scale` |
| `scattering_rate_unit` | string | no | `ps$^{-1}$` | Label paired with `scattering_rate_scale`<br>CLI aliases: `--scattering-rate-unit` |
| `vmin_percentile` | float | no | `1.0` | Lower shared color percentile over selected slices/modes<br>CLI aliases: `--vmin-percentile` |
| `vmax_percentile` | float | no | `99.0` | Upper shared color percentile over selected slices/modes<br>CLI aliases: `--vmax-percentile` |
| `error_percentile` | float | no | `99.0` | Symmetric splitting/error color percentile<br>CLI aliases: `--error-percentile` |
| `energy_cmap` | string | no | `viridis` | Energy colormap<br>CLI aliases: `--energy-cmap` |
| `linewidth_cmap` | string | no | `magma` | HWHM/FWHM colormap<br>CLI aliases: `--linewidth-cmap` |
| `lifetime_cmap` | string | no | `viridis` | Log-lifetime colormap<br>CLI aliases: `--lifetime-cmap` |
| `scattering_rate_cmap` | string | no | `magma` | Scattering-rate colormap<br>CLI aliases: `--scattering-rate-cmap` |
| `error_cmap` | string | no | `cividis` | Rotation-error colormap<br>CLI aliases: `--error-cmap` |
| `diverging_cmap` | string | no | `RdBu_r` | Signed mode-splitting colormap<br>CLI aliases: `--diverging-cmap` |
| `edgecolor` | string | no | `none` | Voronoi cell edge color<br>CLI aliases: `--edgecolor` |
| `linewidth` | float | no | `0.0` | Voronoi cell edge width<br>CLI aliases: `--linewidth` |
| `bz_color` | string | no | `0.25` | BZ outline color<br>CLI aliases: `--bz-color` |
| `bz_lw` | float | no | `1.1` | Central BZ outline width<br>CLI aliases: `--bz-lw` |
| `neighbor_bz_lw` | float | no | `0.65` | Neighbor BZ outline width<br>CLI aliases: `--neighbor-bz-lw` |
| `neighbor_bz_alpha` | float | no | `0.55` | Neighbor BZ outline alpha<br>CLI aliases: `--neighbor-bz-alpha` |
| `overwrite` | boolean | no | .false. | Replace existing figures and summary<br>CLI aliases: `--overwrite` |

### `calculation='magnon_plot'`

**Runtime requirements:** `config` defaults to cwd `plot.in` and is effectively
the required second input boundary. It must define `kind` (currently only
`magnon_h5`) and an exchange HDF5 `input` containing bonds/J and lattice/tau.
Use `template=.true.` to print baseline fields or see the nested-config
subsection below.

**Outputs:** Configured image (default `magnon.png`) and optional `save_npz`
sidecar.

Backend: `slw.magph.legacy.plot`.

| Namelist key | Type | Parser required | Parser default | Meaning / CLI aliases |
|---|---|---:|---|---|
| `config` | optional string | no | `plot.in` | plot input file. Default: plot.in |
| `template` | boolean | no | .false. | print a minimal plot.in template and exit<br>CLI aliases: `--template` |
| `list` | boolean | no | .false. | list supported plot kinds and exit<br>CLI aliases: `--list` |

### `calculation='coupling_kpath'`

**Runtime requirements:** Positional hybrid NPZ `input`, image `output`, and
valid `phonon_mode` are required. The NPZ must contain `magnon_vertex` and
`kdist`, so it must come from a q-path hybrid calculation.

**Outputs:** Coupling-along-path figure.

Backend: `slw.magph.legacy.plot_coupling_kpath`.

| Namelist key | Type | Parser required | Parser default | Meaning / CLI aliases |
|---|---|---:|---|---|
| `input` | string | yes | — | hybrid.npz |
| `output` | string | yes | — | Compatibility input retained by the backend.<br>CLI aliases: `-o`, `--output` |
| `phonon_mode` | int | yes | — | Compatibility input retained by the backend.<br>CLI aliases: `--phonon_mode` |
| `magnon_mode` | int | no | none / runtime | Compatibility input retained by the backend.<br>CLI aliases: `--magnon_mode` |
| `mode_base` | enum {0, 1} | no | `1` | Compatibility input retained by the backend.<br>CLI aliases: `--mode_base` |
| `iq` | int | no | `0` | q index in magnon_vertex<br>CLI aliases: `--iq` |
| `quantity` | enum {norm, norm2, log10_abs} | no | `norm` | Compatibility input retained by the backend.<br>CLI aliases: `--quantity` |
| `color` | string | no | `black` | Compatibility input retained by the backend.<br>CLI aliases: `--color` |
| `linewidth` | float | no | `1.5` | Compatibility input retained by the backend.<br>CLI aliases: `--linewidth` |
| `fig_width` | float | no | `6.4` | Compatibility input retained by the backend.<br>CLI aliases: `--fig_width` |
| `fig_height` | float | no | `4.2` | Compatibility input retained by the backend.<br>CLI aliases: `--fig_height` |
| `dpi` | int | no | `180` | Compatibility input retained by the backend.<br>CLI aliases: `--dpi` |
| `title` | string | no | none / runtime | Compatibility input retained by the backend.<br>CLI aliases: `--title` |

### `calculation='coupling_bz'`

**Runtime requirements:** Positional hybrid NPZ `input`, image `output`, and
valid `phonon_mode` are required. The NPZ must contain `magnon_vertex` and
`qpts_frac`; the selected kz slice needs at least four points. Without
`structure`, plotting falls back to a fractional square.

**Outputs:** Brillouin-zone coupling figure.

Backend: `slw.magph.legacy.plot_coupling_bz`.

| Namelist key | Type | Parser required | Parser default | Meaning / CLI aliases |
|---|---|---:|---|---|
| `input` | string | yes | — | hybrid.npz containing magnon_vertex and qpts_frac |
| `output` | string | yes | — | Output image path<br>CLI aliases: `-o`, `--output` |
| `structure` | string | no | none / runtime | Structure file for reciprocal BZ shape; omit for fractional square coordinates<br>CLI aliases: `--structure` |
| `phonon_mode` | int | yes | — | Phonon mode number<br>CLI aliases: `--phonon_mode` |
| `magnon_mode` | int | no | none / runtime | Magnon mode number; omit to use norm over all magnon modes<br>CLI aliases: `--magnon_mode` |
| `mode_base` | enum {0, 1} | no | `1` | Whether mode numbers are 0-based or 1-based<br>CLI aliases: `--mode_base` |
| `ik` | int | no | `0` | k index in magnon_vertex<br>CLI aliases: `--ik` |
| `kz` | float | no | `0.0` | Target fractional kz slice<br>CLI aliases: `--kz` |
| `k_round` | int | no | `8` | Rounding decimals used to identify kz slices<br>CLI aliases: `--k_round` |
| `quantity` | enum {norm, norm2, log10_abs} | no | `norm` | Compatibility input retained by the backend.<br>CLI aliases: `--quantity` |
| `tile` | int | no | `2` | Periodic image range for Voronoi fill<br>CLI aliases: `--tile` |
| `cmap` | string | no | `viridis` | Compatibility input retained by the backend.<br>CLI aliases: `--cmap` |
| `vmin` | float | no | none / runtime | Compatibility input retained by the backend.<br>CLI aliases: `--vmin` |
| `vmax` | float | no | none / runtime | Compatibility input retained by the backend.<br>CLI aliases: `--vmax` |
| `vmin_percentile` | float | no | `1.0` | Compatibility input retained by the backend.<br>CLI aliases: `--vmin_percentile` |
| `vmax_percentile` | float | no | `99.0` | Compatibility input retained by the backend.<br>CLI aliases: `--vmax_percentile` |
| `edgecolor` | string | no | `none` | Compatibility input retained by the backend.<br>CLI aliases: `--edgecolor` |
| `linewidth` | float | no | `0.0` | Compatibility input retained by the backend.<br>CLI aliases: `--linewidth` |
| `bz_color` | string | no | `0.35` | Compatibility input retained by the backend.<br>CLI aliases: `--bz_color` |
| `bz_lw` | float | no | `1.2` | Compatibility input retained by the backend.<br>CLI aliases: `--bz_lw` |
| `bz_alpha` | float | no | `0.95` | Compatibility input retained by the backend.<br>CLI aliases: `--bz_alpha` |
| `margin` | float | no | `0.04` | Compatibility input retained by the backend.<br>CLI aliases: `--margin` |
| `fig_width` | float | no | `5.4` | Compatibility input retained by the backend.<br>CLI aliases: `--fig_width` |
| `fig_height` | float | no | `5.0` | Compatibility input retained by the backend.<br>CLI aliases: `--fig_height` |
| `dpi` | int | no | `180` | Compatibility input retained by the backend.<br>CLI aliases: `--dpi` |
| `title` | string | no | none / runtime | Compatibility input retained by the backend.<br>CLI aliases: `--title` |

### `calculation='coupling_kbz'`

**Runtime requirements:** Positional hybrid NPZ `input`, image `output`, and
valid `phonon_mode` are required. The NPZ needs `magnon_vertex`, `kpts_frac`,
and `qpts_frac`. `iq` and `q` are mutually exclusive (neither selects iq=0),
and the chosen k plane needs at least four unique points.

**Outputs:** Fixed-q magnon-BZ coupling figure.

Backend: `slw.magph.legacy.plot_coupling_kbz`.

| Namelist key | Type | Parser required | Parser default | Meaning / CLI aliases |
|---|---|---:|---|---|
| `input` | string | yes | — | hybrid.npz containing magnon_vertex, kpts_frac, and qpts_frac |
| `output` | string | yes | — | Output image path<br>CLI aliases: `-o`, `--output` |
| `structure` | string | no | none / runtime | Structure file used to construct the Cartesian first BZ<br>CLI aliases: `--structure` |
| `iq` | int | no | none / runtime | Zero-based q index; defaults to 0<br>CLI aliases: `--iq` |
| `q` | float[3] | no | none / runtime | Fractional q; nearest periodic mesh point is used<br>CLI aliases: `--q` |
| `phonon_mode` | int | yes | — | Compatibility input retained by the backend.<br>CLI aliases: `--phonon_mode`, `--phonon-mode` |
| `magnon_mode` | int | no | none / runtime | Omit to take the norm over all magnon modes<br>CLI aliases: `--magnon_mode`, `--magnon-mode` |
| `mode_base` | enum {0, 1} | no | `1` | Compatibility input retained by the backend.<br>CLI aliases: `--mode_base`, `--mode-base` |
| `plane_axes` | int[2] | no | `[0, 1]` | Fractional reciprocal axes spanning the plotted plane<br>CLI aliases: `--plane_axes`, `--plane-axes` |
| `slice_value` | float | no | `0.0` | Fractional coordinate along the axis normal to --plane-axes<br>CLI aliases: `--slice`, `--kz` |
| `k_round` | int | no | `8` | Compatibility input retained by the backend.<br>CLI aliases: `--k_round`, `--k-round` |
| `quantity` | enum {norm, norm2, log10_abs} | no | `norm` | Compatibility input retained by the backend.<br>CLI aliases: `--quantity` |
| `tile` | int | no | `2` | Periodic image range used to construct finite Voronoi cells<br>CLI aliases: `--tile` |
| `cmap` | string | no | `viridis` | Compatibility input retained by the backend.<br>CLI aliases: `--cmap` |
| `vmin` | float | no | none / runtime | Compatibility input retained by the backend.<br>CLI aliases: `--vmin` |
| `vmax` | float | no | none / runtime | Compatibility input retained by the backend.<br>CLI aliases: `--vmax` |
| `vmin_percentile` | float | no | `1.0` | Compatibility input retained by the backend.<br>CLI aliases: `--vmin_percentile`, `--vmin-percentile` |
| `vmax_percentile` | float | no | `99.0` | Compatibility input retained by the backend.<br>CLI aliases: `--vmax_percentile`, `--vmax-percentile` |
| `edgecolor` | string | no | `none` | Compatibility input retained by the backend.<br>CLI aliases: `--edgecolor` |
| `linewidth` | float | no | `0.0` | Compatibility input retained by the backend.<br>CLI aliases: `--linewidth` |
| `bz_color` | string | no | `0.35` | Compatibility input retained by the backend.<br>CLI aliases: `--bz_color`, `--bz-color` |
| `bz_lw` | float | no | `1.2` | Compatibility input retained by the backend.<br>CLI aliases: `--bz_lw`, `--bz-lw` |
| `bz_alpha` | float | no | `0.95` | Compatibility input retained by the backend.<br>CLI aliases: `--bz_alpha`, `--bz-alpha` |
| `margin` | float | no | `0.04` | Compatibility input retained by the backend.<br>CLI aliases: `--margin` |
| `fig_width` | float | no | `5.4` | Compatibility input retained by the backend.<br>CLI aliases: `--fig_width`, `--fig-width` |
| `fig_height` | float | no | `5.0` | Compatibility input retained by the backend.<br>CLI aliases: `--fig_height`, `--fig-height` |
| `dpi` | int | no | `180` | Compatibility input retained by the backend.<br>CLI aliases: `--dpi` |
| `title` | string | no | none / runtime | Compatibility input retained by the backend.<br>CLI aliases: `--title` |

## Compatibility nested input files

The stage namelist is the new outer interface, but three retained paths still
consume a second flat `key = value` file:

- `magph/spectral` still requires `input_file` in the compatibility path;
  native `magph/lifetime` does not read a nested input file.
- `magph/prepare_lifetime` and `magph/hybrid` may use the same file as an
  alternative to direct options.
- `post/magnon_plot` consumes its own `plot.in` file through `config`.

The magph compatibility parser accepts `#` comments and the following principal
keys. Values in this nested file are not Fortran namelist syntax.

The old resource names listed here are valid only inside that explicitly
supplied nested compatibility file. New outer stage inputs must put resources
in `&parallel`; command-line values translated from the outer group override
the corresponding retained-driver defaults.

| Area | Compatibility keys |
|---|---|
| Thermodynamic/spin | `T`, `S`, `eta`, `spin_direction`, `spin_pattern`, `mag_order`, `anisotropy_mev`, `g_factor` |
| Mesh/path | `k_mesh`, `target_k_mesh`, `phonon_qmesh`, `phonon_qshift`, `band_points`, `omega_points`, `qpath`, `q_points`, `path_points` |
| Exchange/structure | `exchange_folder`, `POSCAR`, `structure_file`, `jr`, `djr`, `djdu_npz`, `rp_idx`, `manifest` |
| Phonons | `phonon_path`, `phonon_cache`, `phonon_fc`, `phonon_asr`, `phonon_loto`, `phonon_nproc`, `phonon_cache_compressed` |
| Solver/performance | `target_rank_gb`, `kernel_source`, `progress_interval`, `vertex_q_chunk`, `vertex_bond_chunk`, `hybrid_nproc` |
| Coupling/filtering | `bond_factor`, `coupling_scale`, `dJ_asr`, `dJ_asr_tolerance`, `exclude_shells`, `shell_tol`, `exclude_shell_apply` |
| Hybrid output | `hybrid_output`, `hybrid_plot`, `hybrid_html_plot`, `hybrid_html_data`, `hybrid_bare_output`, `hybrid_bare_plot`, `hybrid_phonon_cache_out` |

For `magnon_plot`, run `slw_post.x --legacy-help magnon_plot` and set
`template=.true.` in `&post` to print the current flat-file template. The
registered `magnon_h5` kind uses these baseline keys:

| Key | Required/default | Meaning |
|---|---|---|
| `kind` | required (`magnon_h5`) | Plot backend. |
| `input` | required | Static exchange HDF5. |
| `output` | required | Image path. |
| `kpath` or `kpath_file` | one required | Explicit symbolic path or QE `K_POINTS crystal_b` file. |
| `S` | explicit | Spin magnitude. |
| `spin_pattern` | explicit | Collinear signs/magnitudes by magnetic site. |
| `band_points` | default 101 | Samples per path segment. |
| `j_source` | default `auto` | Exchange dataset selection. |
| `component` | default `iso` | Exchange component. |
| `j_prefactor` | default `auto` | Convention prefactor. |
| `solver` | default `full_bdg` | LSWT solver. |
| `bond_class` | default `sign` | Bond grouping convention. |
| `bond_factor` | default `auto` | Convert the declared source directed-bond weight to the legacy half-weight LSWT convention; an explicit finite positive override is retained for audited external files. |
| `energy_unit` | default `meV` | Plot energy unit. |
| `exclude_shells` | none | Optional 1-based shells to omit. |

## Validation and discovery commands

```bash
# Validate an EPR J template without opening scientific data
slw_exchange.x -in examples/exchange_j_epr.in --dry-run

# List calculations
slw_magph.x --list-calculations

# Show the retained backend help used to build these tables
slw_exchange.x --help-calculation j --source epr
slw_post.x --legacy-help dj_asr
```

When this reference and `--legacy-help` disagree, treat the installed backend
help as authoritative and update this document in the same change as the parser.
