# Stage-input templates

These files demonstrate supported input syntax and workflow boundaries. They
are not material defaults or converged production settings. Replace every
path, Fermi energy, mesh, magnetic atom, orbital selection, spin direction,
shell cutoff, and integration setting with values for the calculation at hand.

## Exchange templates

| Input | Calculation | Output suffix |
|---|---|---|
| `exchange_j_epr.in` | scalar J from collinear EPR up/down | `.j.h5`, `.j.txt`, `.j.all_bonds.tsv` |
| `exchange_j_tensor_epr.in` | tensor J from collinear EPR up/down | `.j_tensor.h5`, `.j_tensor.txt` |
| `exchange_dj_epr.in` | scalar dJ/du from EPR electron-phonon data | `.dj.h5`, `.dj.txt`, `.dj.all_bonds.tsv` |
| `exchange_dj_tensor_epr.in` | tensor dJ/du from EPR electron-phonon data | `.dj_tensor.h5` |
| `exchange_j_tensor_wannier_collinear.in` | tensor J from separate collinear Wannier90 HR files | `.j_tensor.h5`, `.j_tensor.txt` |
| `exchange_j_tensor_wannier_spinor.in` | tensor J from a native spinor Wannier90 HR file | `.j_tensor.h5`, `.j_tensor.txt` |
| `exchange_j_tensor_wannier_spn.in` | optional SPN validation of native spinor tensor J | `.j_tensor.h5`, `.j_tensor.txt` |

The actual output directory is `${outdir}/${prefix}.save`. The four EPR
templates intentionally share `prefix='sample'`, so their calculation-specific
filenames can coexist in the same save directory.

### Wannier90 tensor J

There are two normal input paths:

- Native spinor: provide `spinor_hr`, `win`, `centres`, and `groupby`. The HR
  already contains SOC or noncollinearity, so do not add a second SOC card.
  Magnetic orbitals are assigned automatically from the Wannier centres.
- Collinear pair: provide `up_hr`, `dn_hr`, `win`, and `slices`. The up/down
  calculations must have the same Wannier orbital order and gauge. SLW combines
  the two channels into one internal spinor Hamiltonian.

The native-spinor `groupby` value describes the order in the input HR:

- `groupby='orbital'`: `orb1-up, orb1-down, orb2-up, orb2-down, ...`
- `groupby='spin'`: all up orbitals followed by all down orbitals

The normal spinor example needs no `spin_operator` input. The longer SPN file
is an optional validation route and is the only template that sets
`spin_operator='spn'`. Its AMN/EIG/SPN/U/U_dis inputs are one complete bundle,
and its `kmesh` must equal the native Wannier90 grid represented by that bundle.
It is not the default production interface.

For EPR dJ, `qmesh` must agree with the EPR `qc_dim` and divide the electronic
`kmesh`. All meshes and the small `empoints` values in these files are syntax
examples and must be converged.

## Parallel execution

Example inputs use one local worker and one thread per MPI rank:

```ini
&parallel
  execution = 'auto',
  workers_per_rank = 1,
  threads_per_worker = 1
/
```

Choose the MPI rank count in the launcher or scheduler, not in the input file.
For example:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

mpirun -np "$SLW_MPI_RANKS" \
  slw_exchange.x -in examples/exchange_j_tensor_wannier_spinor.in > j_tensor.out
```

`SLW_MPI_RANKS` should be set from the allocated resources. Native exchange
calculations distribute their numerical work over those MPI ranks. The
`nproc=1` in `epr.in` is the portable local-worker setting for its retained
backend, not an MPI rank count.

Scalar dJ is the only exchange mode that may additionally use
`workers_per_rank>1`: each MPI rank publishes its assigned EPC cache in local
shared memory for those workers. Reserve and bind at least
`workers_per_rank*threads_per_worker` cores per rank and verify node-local
shared-memory capacity. Keep `workers_per_rank=1` for the other exchange modes.

## Validation

Exchange templates can be validated without opening scientific inputs or
creating their save directories:

```bash
for input in examples/exchange_*.in; do
  slw_exchange.x -in "$input" --dry-run
done
```

The other templates can be checked independently:

```bash
slw_epr.x -in examples/epr.in --dry-run
slw_magph.x -in examples/dispersion.in --dry-run
slw_magph.x -in examples/magph.in --dry-run
slw_magph.x -in examples/lifetime.in --dry-run
slw_post.x -in examples/post.in --dry-run
slw_post.x -in examples/lifetime_plot.in --dry-run
```

## Magnon and lifetime templates

`dispersion.in` reads the explicit Wannier90 `kpoint_path` block in
`kpath.win`. Its single-ion anisotropy is illustrative: positive `K` follows
`H_SIA=-K(s.n)^2`, and the normalization must match the supplied exchange data.

Native dispersion and lifetime use `restart_mode='error'` by default. Select
`restart` to validate and reuse a completed NPZ, or `from_scratch` to replace
an existing result atomically. Native lifetime does not resume a partial
self-energy grid.

`lifetime_plot.in` reads the native lifetime NPZ and produces mode-resolved
energy, HWHM, rate, lifetime, and mode-splitting maps. Native AFM schema-v2
files use `chi=+1, chi=-1` mode order, so the default splitting is the signed
chirality splitting. For an older result without embedded geometry, provide
its matching `exchange_h5` as indicated in the file; that also lets the plotter
reconstruct chirality without rerunning the self-energy.
