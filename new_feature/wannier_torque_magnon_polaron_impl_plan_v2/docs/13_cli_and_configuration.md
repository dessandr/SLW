# CLI and Configuration

## Commands

```text
wtorque inspect CONFIG.yml
wtorque extract-exchange CONFIG.yml
wtorque build-vertices CONFIG.yml
wtorque compute-kernel CONFIG.yml
wtorque project-phonons CONFIG.yml
wtorque project-magnons CONFIG.yml
wtorque assemble-polaron CONFIG.yml
wtorque validate CONFIG.yml
wtorque merge OUTPUT_DIR
```

## Required configuration blocks

```yaml
electrons:
  file: electrons.h5
  bloch_gauge: atomic_position
  exchange_extraction: explicit   # explicit | time_reversal | collinear_split
  spin_order: interleaved

time_reversal:
  sewing_matrix_dataset: /time_reversal/B_theta_k

magnetic_subspace:
  policy: explicit_indices
  sites:
    - atom: 0
      orbitals: [0, 1, 2, 3, 4]
    - atom: 1
      orbitals: [5, 6, 7, 8, 9]
  site_projection: local_partition
  spin_coordinate: transverse_direction

dfpt:
  file: dfpt.h5
  normalization: cartesian_derivative
  spinor_lift: native_spinor
  include_dHsoc_du: false

kernel:
  include_direct_vertex: false
  fixed_chemical_potential: true
  q_pair_completion: true

integration:
  backend: real_axis
  eta_eV: 0.01

phonons:
  file: phonons.h5

magnons:
  file: magnons.h5
  require_external_paraunitary: true

output:
  file: wtorque.h5
  resume: true
```

## Safety checks

- `time_reversal` metadata is mandatory only for that extraction route.
- `include_direct_vertex=true` requires an exchange-field-resolved DFPT dataset.
- resume is rejected if source hashes or resolved orbital masks change.
- undeclared gauge or normalization is a hard error.
