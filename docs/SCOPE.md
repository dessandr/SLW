# SLW repository scope

SLW stands for **Spin-Lattice Wannier**. Its public Python namespace is `slw`.

## Retained production code

| Area | Retained responsibility |
|---|---|
| `slw/core` | Wannier90 HR/U-matrix/`.win` I/O, QE→Perturbo Wigner–Seitz helpers, structures, k paths, constants, and CLI path handling |
| `slw/epc` | `qe2pert` EPR validation, electronic/phonon dispersion checks, `g(k,q)` reconstruction, symmetry/constraint checks, real-space conversion, and interpolation |
| `slw/exchange` | Public native typed `j`/`dj` engine with `ltensor` dispatch |
| `slw/exchange/kernels` | Native EPR/Wannier scalar/tensor J and analytic dJ kernels with MPI work partitioning |
| `slw/exchange/legacy` | Archive-only historical LKAG implementations, diagnostics, ASR/symmetry tools, and compatibility wrappers |
| `slw/soc` | Typed atomic-SOC manifolds, Wannier90 projection resolution, and strict TB2J `groupby=spin|orbital` spinor construction |
| `slw/soc/legacy` | Archive-only SOC fitting, plotting, inspection, and superseded basis-layout experiments |
| `slw/magph` | Native exchange/dJ/phonon screening, FM/bipartite-AFM LSWT and vertex construction, retarded self-energy/lifetime kernels, MPI k distribution, and lifetime output |
| `slw/magph/legacy` | Quarantined magnon–phonon helpers, kernels, adapters, analyses, and plotting modules retained by active workflows |
| `slw/magph/legacy/reference` | Quarantined compatibility drivers retained for non-native EPR/magph calculations and scientific reference |
| `slw/interactions` | Wannier-gauge reference density and intersite-V correction |
| `slw/phonon` | Shared phonon parsing |

The EPR boundary is deliberate: file-format and workflow integration with
QE/Perturbo is retained, while the external Perturbo implementation is not.

All pre-integration exchange code is quarantined under `slw/exchange/legacy`.
Public execution goes through `exchange/engine.py` and
`exchange/kernels/`; it has no import or dispatch edge to the archive.
Historical package-root module paths are intentionally not preserved. Thin
wrappers under `legacy/reference` exist only for archived tools and parity
audits, not for public exchange calculations.

The same quarantine boundary applies to magph. Public execution goes through
`slw_magph.x` or the relevant `slw_post.x` calculation. Historical
`slw.magph.<module>` paths have no root-level compatibility shims; retained
drivers are isolated under `slw/magph/legacy/reference`, and their supporting
or post-processing modules live under `slw/magph/legacy`. Native lifetime is
registered and has no legacy import; the remaining magph actions are migrated
individually as their numerical contracts are replaced.

## Explicitly excluded

- Legacy ABACUS/LCAO Hamiltonian, overlap, CSR, finite-displacement, and LKAG
  implementations.
- Compatibility wrappers whose only backend was the removed ABACUS path.
- Historical tests and one-off test scripts from the archive repository.
- The vendored `perturbo-codes` Fortran source tree.
- Calculation inputs tied to a material, generated HDF5/NPZ/text tables,
  figures, logs, reports, caches, and build products.
- Historical planning notes and obsolete extraction manifests.

## Dependency boundary

The active package does not import the removed `core.parsing`, `core.orbital`,
`epc.bloch`, or `compute_J_abacus_*` modules. Shared Wannier functionality now
lives in `core/wannier_io.py`, material-independent structure parsing in
`core/structure.py`, and the vectorized LKAG/Green primitives in
`exchange/kernels/`.

## Input policy

No material-specific magnetic atom list, orbital slice, k mesh, SOC element, or
file prefix is selected by default. SOC site/species manifolds are explicit in
the `SOC (atomic)` card and resolved against a supplied Wannier90 `.win` file.
Spinor basis order is never inferred: raw spinor HR input declares one of the
two TB2J `groupby` layouts.
