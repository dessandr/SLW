# Project rules

1. Material, lattice, orbital, mesh, and path choices must come from inputs or
   parsed files; do not hardcode a specific calculation.
2. Use vectorized, threaded, multiprocessing, MPI, or JIT paths where they are
   appropriate for numerical kernels.
3. Ask for confirmation before implementing a requested code change.
4. This workspace may reside on an SSHFS mount. Ask before exploring paths
   outside the explicitly approved repository root under `$HOME/clusters`.
