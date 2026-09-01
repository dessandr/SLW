# Stage-input templates

These files demonstrate syntax and workflow boundaries; they are not material
defaults or converged production settings. Replace every path, Fermi energy,
mesh, magnetic atom, orbital slice, target, shell cutoff, and numerical
integration setting with values for the calculation at hand.

## Exchange templates

| Input | Calculation | Default product under `slw-tmp/sample.save` |
|---|---|---|
| `exchange_j_epr.in` | scalar J from EPR up/down | `sample.j.h5`, `.txt`, `.all_bonds.tsv` |
| `exchange_j_tensor_epr.in` | tensor J from EPR up/down | `sample.j_tensor.h5`, `.txt` |
| `exchange_dj_epr.in` | scalar dJ/du from EPR EPC | `sample.dj.h5`, `.txt`, `.all_bonds.tsv` |
| `exchange_dj_tensor_epr.in` | tensor dJ/du from EPR EPC | `sample.dj_tensor.h5` |
| `exchange_j_tensor_wannier_soc.in` | tensor J from spinor HR plus an optional atomic-SOC card | `sample_soc.j_tensor.h5`, `.txt` |
| `exchange_j_tensor_wannier_spn.in` | projection-anchored tensor J from spinor HR and physical SPN | `sample_spn.j_tensor.h5`, `.txt` |

The four EPR templates intentionally share `prefix='sample'`, so their
distinct mode-derived filenames can coexist in one `sample.save` directory.
The listed `kmesh`, `qmesh`, `empoints`, atoms, explicit EPR slices, and SOC
strengths are only parseable examples. The SOC spinor Wannier template
demonstrates automatic `win` + `centres` assignment. The SPN template uses the
all-or-nothing AMN/EIG/SPN/U/U_dis bundle instead: `win` + AMN selects the
magnetic projection frame, so both `slices` and `centres` are omitted. Its
`kmesh` must be replaced by the native U/AMN/SPN grid dimensions; coordinates,
ordering, and any uniform shift are read from U. This path does not interpolate
a denser exchange mesh. In particular,
dJ requires EPR files containing electron-phonon data, and its explicit
`qmesh` must agree with EPR `qc_dim` and divide the electronic `kmesh`.

The standard raw-`spinor_hr` template uses the TB2J-compatible common
orbital-spin product-basis contract; `spin_operator='pauli'` is the default and
can be omitted. The SPN template is an optional validation mode and sets
`spin_operator='spn'` explicitly. The legacy `auto` value resolves to the
standard Pauli path without a bundle and to the projected path when the complete
bundle is supplied. Choose
`u_dis_layout='global_bands'` for globally indexed rows or
`'compact_outer_window'` for an old packed U_dis file. Keep
`collinear_override=.false.` for SOC data; enable it only for a Hamiltonian
known independently to be no-SOC and collinear.

All exchange templates use `execution='auto'` and `workers_per_rank=1` in
`&parallel`, which is the safe portable MPI default. Scalar dJ may instead use
multiple workers with one MPI rank per node: the rank creates one shared-memory
EPC cache and its local processes share it. Reserve and bind at least
`workers_per_rank*threads_per_worker` cores per rank, and check that node-local
POSIX shared-memory capacity is sufficient for the cache. Other MPI exchange
modes must leave `workers_per_rank=1`.

Validate every exchange template without opening a scientific input file or
creating its save directory:

```bash
for input in examples/exchange_*.in; do
  slw_exchange.x -in "$input" --dry-run
done
```

Example execution after replacing the placeholders:

```bash
mpirun -np 8 slw_exchange.x -in examples/exchange_j_epr.in > j.out
mpirun -np 8 slw_exchange.x -in examples/exchange_dj_epr.in > dj.out
```

Other stage templates can be checked independently:

```bash
slw_magph.x -in examples/dispersion.in --dry-run
slw_magph.x -in examples/lifetime.in --dry-run
slw_post.x -in examples/lifetime_plot.in --dry-run
```

`dispersion.in` uses the explicit Wannier90 `kpoint_path` block in
`kpath.win`. Its single-ion anisotropy is an illustrative input, not a
material default: positive `K` follows `H_SIA=-K(s.n)^2` and the chosen
`unit_vector` or `spin_operator` normalization is mandatory.

Native dispersion and lifetime use `restart_mode='error'` by default. Select
`restart` to validate and reuse a completed NPZ, or `from_scratch` to
atomically replace an existing result. Native lifetime does not yet resume a
partial self-energy grid.

`lifetime_plot.in` reads the native lifetime NPZ directly and produces
mode-resolved energy, HWHM, rate, lifetime, and mode-splitting BZ maps. Native
AFM schema-v2 files use `chi=+1, chi=-1` mode order, so the default splitting is
the signed chirality splitting. For old native results, set `exchange_h5` to
the static J product used by the lifetime calculation; this also lets the
plotter reconstruct chirality without rerunning the self-energy.
