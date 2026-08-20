# Stage executables and QE-style input

SLW exposes four workflow executables. Each process reads one Fortran
namelist, performs one calculation, and writes ordinary progress messages to
standard output.

| Executable | Responsibility | MPI-aware calculations |
|---|---|---|
| `slw_epr.x` | QE/qe2pert EPR preparation and validation | none yet |
| `slw_exchange.x` | scalar/tensor `J` and analytic `dJ/du` | all calculations |
| `slw_magph.x` | hybrid bands, Berry, lifetime, spectral, and scattering calculations | `lifetime`, `spectral`, `chirality_plane` |
| `slw_post.x` | ASR/symmetry checks, analysis, dumps, and plotting | none yet |

Serial use follows the QE convention:

```bash
slw_exchange.x < exchange.in > exchange.out
slw_exchange.x -in exchange.in > exchange.out
```

For MPI jobs, `-in` is recommended because scheduler handling of standard
input varies:

```bash
mpirun -np 16 slw_exchange.x -in exchange.in > exchange.out
```

Rank 0 alone reads and validates the input and broadcasts the normalized
configuration. A calculation with an MPI implementation runs on every rank. A
serial-only calculation launched under MPI runs only on rank 0, preventing
ranks from racing to overwrite the same output. Exchange is MPI-capable in all
four modes and requires `nproc=1` per rank; serial exchange may use `nproc>1`
for local multiprocessing. Other backend-local worker settings should also be
sized to avoid oversubscription.

## Input layout

Every input has `&control`, an optional `&parallel`, and either the stage-named
group or its generic alias `&input`:

```fortran
&control
  calculation = 'j',
  prefix = 'sample',
  outdir = './slw-tmp',
  verbosity = 'normal'
/

&parallel
  execution = 'auto'
/

&exchange
  input_format = 'epr',
  ltensor = .true.,
  tensor_kernel = 'tb2j',
  epr_up = './input/up_epr.h5',
  epr_dn = './input/dn_epr.h5',
  efermi = 0.0,
  kmesh = 4, 4, 4,
  mag_atoms = 0, 1,
  slices = '0:0:5,1:5:10'
/
```

`prefix` and `outdir` define a shared directory named
`outdir/prefix.save`. The following placeholders are expanded in stage
parameters without consulting environment variables:

| Placeholder | Value |
|---|---|
| `${prefix}` | the exact input prefix |
| `${outdir}` | the absolute output directory |
| `${savedir}` | the absolute `outdir/prefix.save` directory |

The save directory is created only for a real calculation, not for a dry run.
It and common numerical products are ignored by Git.

`execution` accepts `auto`, `serial`, or `mpi`. `auto` chooses the registered
MPI backend only when more than one rank is present. `mpi` rejects calculations
that do not have an MPI implementation instead of duplicating serial work.

Scalars, arrays, quoted strings, Fortran `d` exponents, and
`.true.`/`.false.` values are parsed by `f90nml`. Exchange and magph lifetime
use native typed schemas. Other calculations currently translate their stage
parameters to retained backend option names while those handlers are migrated.

Exchange exposes only `calculation='j'` and `'dj'`. Set `ltensor=.false.` for
the scalar convention or `.true.` for the tensor convention. `input_format`
is explicit: `j` accepts `epr` or `wannier`, while `dj` currently accepts only
`epr` because the electron-phonon vertex is read from qe2pert data. Material
choices such as meshes, magnetic atoms, orbital slices, spin direction, and
reciprocal-space paths remain explicit inputs. Unless overridden, exchange
outputs are written to `${savedir}/${prefix}.<mode>.{h5,txt}`.

## Discovering calculations

List the calculations in a stage:

```bash
slw_epr.x --list-calculations
slw_exchange.x --list-calculations
slw_magph.x --list-calculations
slw_post.x --list-calculations
```

Show detailed calculation input help:

```bash
slw_exchange.x --help-calculation j --source epr
slw_magph.x --legacy-help hybrid
```

Validate an input and display the selected execution plan without reading
scientific data or creating output directories:

```bash
slw_exchange.x -in examples/exchange_j_epr.in --dry-run
```

For exchange, `verbosity='normal'` prints stable phase/progress lines and a
CPU/WALL timing summary; `high` also exposes native-kernel diagnostic lines.
For compatibility-backed stages, `high` prints the translated command.
`debug` adds tracebacks on failure.

## Current calculation registry

- EPR: `gkq`, `dispersion`, `phonon_cache`, `kpath`
- Exchange: `j`, `dj`; tensor form is selected by `ltensor=.true.`
- Magph: `hybrid`, `berry`, `spectral`, `lifetime`, `scattering_kbz`,
  `scattering_qbz`, `rotational_coupling`, `chirality_plane`,
  `prepare_lifetime`
- Post: `check_gkq`, `check_spin_mz`, `compare_u_rotation`, `dj_asr`,
  `dj_kq_symmetry`, `dump_dj`, `spin_group`, `j_diagnostic`,
  `rpa_lkag_audit`, spin-flip/SOC diagnostics, lifetime/phonon analysis,
  and coupling plots

## Migration boundary

The exchange stage now uses a native typed request and one engine dispatch.
`slw_exchange.x` reaches only `slw.exchange.kernels`; it does not import or
dispatch to `slw.exchange.legacy`. Historical modules remain archive-only and
are not re-exported at the package root. Scalar and tensor integrands
intentionally remain separate because their LKAG/TB2J conventions differ.
The current HDF5 schemas are preserved so magph screening and existing data
remain compatible while the numerical implementation is replaced internally.

Magph lifetime now runs through `slw.magph.engine` and has no dependency on
the archive. It distributes external k points over every discovered MPI rank,
keeps q/mode work vectorized within each rank, and writes one root-owned NPZ.
Other magph drivers remain under `slw.magph.legacy.reference` while numerical
contracts are replaced. Helper and post-processing modules live under
`slw.magph.legacy`; old `python -m slw.magph.<module>` paths are intentionally
not preserved.

The EPR and post stages also translate validated namelist values into retained
drivers. Their backend location is an implementation detail rather than a
public module-level command.

Two existing format boundaries still matter:

- `gkq` writes the direct qe2pert reconstruction schema. `post/check_gkq`
  currently targets the older dense band-gauge schema, so those operations must
  not be chained without an explicit conversion.
- `spectral` still consumes a compatibility manifest and legacy flat
  `input_file`. Native lifetime consumes canonical J, dJ, and phonon products
  directly from the QE-style namelist.

The complete syntax templates are in [`examples/`](../examples/README.md).
For every calculation's accepted keys, types, parser defaults, conditional
runtime requirements, and output files, see
[`INPUT_REFERENCE.md`](INPUT_REFERENCE.md).
