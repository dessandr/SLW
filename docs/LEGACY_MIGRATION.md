# Legacy dependency burn-down

The directories `slw/exchange/legacy` and `slw/magph/legacy` are retained as
read-only scientific references.  Production code is moving to a one-way
boundary: an archive may call a native helper, but a native or public CLI path
must not import, dispatch to, or discover an archive module.

`tests/test_legacy_dependency_budget.py` records the exact temporary crossing
set.  The initial inventory contained 50 active references: 16 Python imports
and 34 dynamic registry references.  Moving the shared structure and EPR
metadata readers into native common layers reduced that inventory to 43.  The
native lifetime and dispersion registry entries then removed crossings. Native exchange
kernels and SOC imports removed every active exchange-runtime crossing; 28
registry-only crossings remain for post-processing and non-native magph
calculations. The allowlist may only shrink.

## Migration order

1. Completed: native magnon-phonon `dJ/du -> LSWT -> vertex -> self-energy ->
   lifetime` plus SIA-aware magnon dispersion with MPI k distribution and
   native registry entries.
2. Completed: native scalar/tensor `J` and `dJ/du` kernels with generated EPR
   and Wannier fixtures, serial/MPI numerical parity, and root-owned output.
3. Completed: active SOC imports no longer cross into exchange legacy code.
4. Replace or retire the remaining legacy post-processing and magph registry
   actions.
5. Make the dependency budget empty and stop installing legacy packages in
   production distributions.

## MPI contract

Native calculations discover their launch communicator by default. Rank zero
parses and validates input, broadcasts one canonical problem, and owns final
scientific output. Magph lifetime distributes external k points and keeps
q/mode contractions vectorized within each rank. Exchange scalar/tensor J and
scalar dJ distribute contour/pole energy points; tensor dJ distributes
target-axis tasks. Collective phases agree serializable errors before entering
the next collective. A requested multi-rank run must not silently degrade to
rank-zero serial execution. Scalar dJ may use shared-memory local workers with
`workers_per_rank>1` in `&parallel`; other exchange modes require
`workers_per_rank=1` per MPI rank. Historical backend resource names are
translated only at the quarantine boundary and are not accepted by the public
stage namelist.
