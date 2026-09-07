# SLW — Spin-Lattice Wannier

A Wannier-based toolkit for spin–lattice interactions, magnetic exchange,
spin–orbit coupling, and magnon–phonon calculations. The production code was
extracted from the original LAMP workspace.

The repository intentionally contains no legacy ABACUS/LCAO implementation,
old tests, bundled Perturbo source tree, calculation outputs, or material data.
The Perturbo **interface** remains: QE `qe2pert` EPR files can be inspected and
converted without vendoring Perturbo itself.

## Install

```bash
python -m pip install -e .
```

MPI and phonon support are optional:

```bash
python -m pip install -e '.[mpi,phonon,kpath]'
```

## Main workflows

- `slw.core`: Wannier90 `hr.dat`, `.win`, `U.mat`, structure, and k-path I/O.
- `slw.epc`: QE/Perturbo EPR inspection, `g(k,q)` reconstruction, constraints,
  real-space conversion, and interpolation.
- `slw.exchange`: EPR/Wannier LKAG scalar and tensor exchange, analytic
  `dJ/du`, symmetry/ASR checks, SOC/spin-flip construction, and diagnostics.
- `slw.soc`: typed atomic-SOC manifold input and the two TB2J spinor layouts;
  abandoned fitting and visualization experiments live under `slw.soc.legacy`.
- `slw.magph`: native exchange/phonon/SIA screening, MPI magnon dispersion,
  magnon lifetime, and one-loop phonon-renormalization APIs.
- `slw.magph.legacy`: quarantined EPR adapters, numerical kernels, MPI
  runners, analysis tools, and plotting code retained behind that stage.
- `slw.interactions`: Wannier-gauge reference density and intersite-V tools.
- `slw.phonon`: phonon parsing shared by the active workflows.
- `slw.wtorque`: gauge-explicit spinor-Wannier mixed torque response, restartable
  q-pair kernels, phonon projection, and externally supplied magnon BdG
  projection. This package is additive and does not change the existing
  `slw_*` stage interfaces.

The new torque workflow uses a strict YAML configuration and keeps magnons
external:

```bash
wtorque inspect run.yml
mpirun -np 8 wtorque compute-kernel run.yml
wtorque test-collinear-epr-q0 --help
wtorque project-phonons run.yml
wtorque project-magnons run.yml
wtorque assemble-polaron run.yml
wtorque validate run.yml --report validation.md
```

All orbital masks, local frames, meshes, gauges, normalizations, input paths,
and integration settings come from the configuration or validated HDF5 input.
See [docs/WTORQUE.md](docs/WTORQUE.md) for the strict input, MPI, restart, and
validation contracts.

## Stage executables

Install the package to expose four QE-style workflow commands:

```bash
slw_epr.x      -in epr.in      > epr.out
slw_exchange.x -in exchange.in > exchange.out
slw_magph.x    -in magph.in    > magph.out
slw_post.x     -in post.in     > post.out
```

They also accept standard input, for example
`slw_exchange.x < exchange.in > exchange.out`. MPI-aware calculations read the
input on rank 0 and broadcast it:

```bash
mpirun -np 16 slw_exchange.x -in exchange.in > exchange.out
```

See [docs/CLI.md](docs/CLI.md) for the namelist schema, calculation registry,
MPI behavior, and dry-run validation. Ready-to-edit scalar/tensor EPR J and dJ
inputs are listed in [examples/README.md](examples/README.md). The exhaustive
calculation-by-calculation option reference is in
[docs/INPUT_REFERENCE.md](docs/INPUT_REFERENCE.md).
The native magnon–phonon redesign boundary and FM/AFM admission policy are in
[docs/MAGPH_DESIGN.md](docs/MAGPH_DESIGN.md). Native
`calculation='lifetime'`, `calculation='phonon_renormalization'`, and
post-processing `calculation='lifetime_plot'` are registered; hybrid, Berry,
and spectral routes remain behind the compatibility boundary.

Native lifetime accepts an explicit dense `phonon_qmesh`. The phonons are
evaluated on that mesh while real-space `dJ(R,Rp)` is Fourier interpolated from
its source mesh, so electronic dJ does not need to be recomputed merely to make
the self-energy q integration denser.

Every material-dependent choice is explicit. In particular, exchange commands
require the magnetic atom indices, local orbital slices, and k mesh instead of
assuming a particular crystal or Wannier ordering.

The public exchange interface is the integrated stage command:

```bash
slw_exchange.x --list-calculations
slw_exchange.x --help-calculation j --source epr
```

Individual modules outside the quarantined exchange and magph packages still
provide diagnostic help where applicable:

```bash
python -m slw.epc.compute_gkq_from_epr --help
python -m slw.soc.wannier_soc --help
```

The collinear-to-spinor helper accepts only `--groupby spin|orbital` and, when
centres are written, requires both spin-channel `centres.xyz` files. Additional
onsite SOC is configured in `slw_exchange.x` with a final `SOC (atomic)` card;
it is independent of whether the input HR is already spinor/noncollinear.

Historical exchange modules are quarantined under `slw.exchange.legacy` and
are not re-exported from the public namespace. The public engine dispatches
only to `slw.exchange.kernels`; archived wrappers are not entered by
`slw_exchange.x`. New workflows should use `calculation='j'|'dj'` with
`ltensor=.true.|.false.` through the stage executable.

All four exchange modes participate in MPI when `execution='auto'` discovers
more than one rank. Scalar/tensor J split the energy integration. Scalar dJ
streams one Cartesian displacement axis at a time, distributes that axis's
target cache ownership, and uses excess ranks to split energy; tensor dJ splits
target/displacement-axis tasks. Rank 0 alone writes the final products. Use
`workers_per_rank=1` in `&parallel` per MPI rank; serial runs may raise it for
local multiprocessing. Scalar dJ additionally supports hybrid MPI with
`workers_per_rank>1`: each rank publishes only its current axis-batch EPC cache
in POSIX shared memory and clean local worker processes attach to it. Completed
axis results are reduced before that cache is released. Bind at least
`workers_per_rank*threads_per_worker` cores to each rank. Other exchange modes
still require `workers_per_rank=1` under MPI. The same canonical `&parallel`
names control native magph dispersion, lifetime, and phonon renormalization and
are translated at retained magph backend boundaries; backend-specific `nproc`
and chunk spellings are not part of the public namelist.

Historical magph modules are likewise quarantined under `slw.magph.legacy`.
The retained compatibility drivers live in `slw.magph.legacy.reference`;
helper and post-processing modules remain one level above them. Old
`slw.magph.<module>` paths have no compatibility shims and are not public
interfaces. `slw_magph.x` dispersion/lifetime/phonon-renormalization and
`slw_post.x` lifetime plots now use only native typed modules; the remaining
registered magph calculations still use the compatibility boundary.

See [docs/SCOPE.md](docs/SCOPE.md) for the extraction boundary and retained
module inventory.
