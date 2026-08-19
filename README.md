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
- `slw.soc`: pure Wannier spinor/SOC construction, fitting, band/DOS and
  real-space inspection.
- `slw.magph`: EPR-to-magnon–phonon adapters, hybrid spectra, lifetimes,
  chirality/rotational analysis, MPI runners, and plotting.
- `slw.interactions`: Wannier-gauge reference density and intersite-V tools.
- `slw.phonon`: phonon parsing shared by the active workflows.

Every material-dependent choice is explicit. In particular, exchange commands
require the magnetic atom indices, local orbital slices, and k mesh instead of
assuming a particular crystal or Wannier ordering.

Use each command's `--help` as the authoritative interface, for example:

```bash
python -m slw.epc.compute_gkq_from_epr --help
python -m slw.exchange.compute_J_epr_tensor --help
python -m slw.exchange.compute_dJ_epr_tensor --help
python -m slw.exchange.compute_J_wannier_tensor --help
python -m slw.soc.wannier_soc --help
python -m slw.magph.hybrid --help
```

See [docs/SCOPE.md](docs/SCOPE.md) for the extraction boundary and retained
module inventory.
