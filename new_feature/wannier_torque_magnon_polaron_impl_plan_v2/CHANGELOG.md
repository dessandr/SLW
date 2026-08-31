# Changelog

## 2.0.0

### Theory corrections

- Replaced `onsite_block` as the default site definition by the Hermitian local partition `0.5(PX+XP)`.
- Added time-reversal decomposition and an explicit sewing-matrix contract for extracting the exchange field from a general spinor Wannier Hamiltonian.
- Added a mandatory localized magnetic-subspace definition and a residual-support diagnostic.
- Fixed the torque construction to rotate only the exchange field while keeping SOC fixed to the lattice.
- Separated transverse-direction and rotation-angle vertices, including their local tangent-plane conversion.
- Kept the general finite-q kernel complex and retained the q-pair retarded/advanced construction.
- Classified the direct mixed vertex as a total-derivative extension rather than part of the source-paper MVP.

### Software plan changes

- Added exchange-field extraction and local-projection modules.
- Expanded HDF5 metadata for time reversal, magnetic orbital masks, projection type, coordinate type, and source provenance.
- Renumbered tasks into a 16-task dependency chain.
- Added magnetic-subspace sensitivity, local-partition closure, and quantization-axis rotation tests.
- Added a modular LaTeX theory manuscript, exact BibTeX database, and equation-level source map.

## 1.0.0

Initial Markdown implementation plan for the bubble-only `T G g G` kernel, phonon projection, and external magnon projection.
