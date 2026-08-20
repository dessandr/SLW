# SLW repository scope

SLW stands for **Spin-Lattice Wannier**. Its public Python namespace is `slw`.

## Retained production code

| Area | Retained responsibility |
|---|---|
| `slw/core` | Wannier90 HR/U-matrix/`.win` I/O, QE→Perturbo Wigner–Seitz helpers, structures, k paths, constants, and CLI path handling |
| `slw/epc` | `qe2pert` EPR validation, electronic/phonon dispersion checks, `g(k,q)` reconstruction, symmetry/constraint checks, real-space conversion, and interpolation |
| `slw/exchange` | Public native typed `j`/`dj` engine with `ltensor` dispatch |
| `slw/exchange/legacy` | Quarantined pre-native LKAG kernels, diagnostics, ASR/symmetry tools, and SOC/downfolding builders still required by retained workflows |
| `slw/soc` | Wannier spinor construction, onsite/nonlocal SOC models and fitting, band/DOS/DMI/real-space inspection, and TB2J preparation |
| `slw/magph` | Public package namespace for the integrated magnon–phonon stage; no historical driver modules are re-exported here |
| `slw/magph/legacy` | Quarantined magnon–phonon helpers, kernels, adapters, analyses, and plotting modules retained by active workflows |
| `slw/magph/legacy/reference` | Ten compatibility drivers reached by `slw_epr.x`/`slw_magph.x` while the native magph engine is developed |
| `slw/interactions` | Wannier-gauge reference density and intersite-V correction |
| `slw/phonon` | Shared phonon parsing |

The EPR boundary is deliberate: file-format and workflow integration with
QE/Perturbo is retained, while the external Perturbo implementation is not.

All pre-integration exchange code is quarantined under `slw/exchange/legacy`.
Public execution goes through `exchange/engine.py`; historical package-root
module paths are intentionally not preserved. The six parity drivers used
during kernel migration are isolated further under `legacy/reference`.

The same quarantine boundary applies to magph. Public execution goes through
`slw_magph.x` or the relevant `slw_post.x` calculation. Historical
`slw.magph.<module>` paths have no root-level compatibility shims; the ten
active stage drivers are isolated under `slw/magph/legacy/reference`, and
their supporting/post-processing modules live under `slw/magph/legacy`.

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
`core/structure.py`, and the retained vectorized Green subblock kernel in
`exchange/legacy/green.py`.

## Input policy

No material-specific magnetic atom list, orbital slice, k mesh, SOC element, or
file prefix is selected by default. Commands that need those values require
them explicitly or infer them from an unambiguous Wannier90 `.win` file.
