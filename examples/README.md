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

The four EPR templates intentionally share `prefix='sample'`, so their
distinct mode-derived filenames can coexist in one `sample.save` directory.
The listed `kmesh`, `qmesh`, `empoints`, atoms, slices, and SOC strengths are
only parseable examples. In particular, dJ requires EPR files containing
electron-phonon data, and its explicit `qmesh` must agree with EPR `qc_dim` and
divide the electronic `kmesh`.

All exchange templates use `execution='auto'` and `nproc=1`. A multi-rank
launch therefore uses the native MPI route without nested local process pools.
A serial convergence run may raise `nproc`, while MPI runs should leave it at
one and tune only rank-local thread settings deliberately.

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
slw_magph.x -in examples/lifetime.in --dry-run
```
