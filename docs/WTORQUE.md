# Spinor-Wannier torque workflow

`slw.wtorque` is an additive SLW subpackage.  The compatibility command remains
`wtorque`; it does not register calculations in the existing `slw_epr.x`,
`slw_exchange.x`, `slw_magph.x`, or `slw_post.x` interfaces.

The implemented MVP evaluates the complex retarded bubble

```text
T(-q) G(k+q) g(k,q) G(k)
```

and constructs the physical Fourier coefficient from both members of each
q pair. Magnons are never diagonalized in this package: positive energies,
local frames, spin lengths, Nambu order, and paraunitary eigenvectors must be
supplied by an external solver.

## Input boundary

The strict HDF5 shapes follow
`new_feature/wannier_torque_magnon_polaron_impl_plan_v2/docs/09_data_contracts.md`.
In addition:

- `/electrons` must declare `bloch_gauge`;
- `/dfpt` must declare `normalization`;
- orbital-major interleaved spin order is canonical;
- every physics-core matrix is `complex128`;
- magnetic masks in YAML must agree with `/spin/magnetic_orbital_mask` when
  that dataset is present;
- Cartesian, mass-weighted, and zero-point mode inputs are never inferred
  from shapes.

All paths, meshes, orbital masks, local frames, q points, perturbation maps,
energy grids, broadening, normalization, and memory limits come from YAML or
the parsed HDF5 files.

## Input onsite SOC

An SOC-free spinor Hamiltonian may receive input-resolved atomic onsite SOC:

```yaml
electrons:
  file: electrons.h5
  bloch_gauge: atomic_position
  exchange_extraction: explicit
  spin_order: interleaved
  onsite_soc:
    win: material.win
    scale: 1.0
    entries:
      - selector: Fe-d
        lambda_ev: 0.06
```

Selectors and orbital groups are resolved from the ordered Wannier90
`projections` block.  Site labels such as `Fe1-d` select one site; species
labels such as `Fe-d` select every matching site.  `scale`, `lambda_ev`, the
`.win` path, and optional `p_order`/`d_order` conventions are inputs and are
stored in the run manifest.

The term is added to the unique `R=(0,0,0)` block of `H_R` and `H_TRS_R`.
`H_XC_R` and its torque vertices are unchanged.  Therefore the source HDF5
must contain the SOC-free base Hamiltonian when `onsite_soc` is enabled; SOC
already present in the file must not also be specified here.

Native SLW preparation code can reuse
`slw.wtorque.io.load_collinear_wannier_hr` for common-gauge spin-resolved
Wannier90 `hr.dat` files and `slw.wtorque.io.load_collinear_slw_gkq` for the
`g_wannier` output written by `slw.epc.compute_gkq_from_epr`.  The latter
requires explicit energy and displacement units and returns canonical
eV/Angstrom, orbital-major interleaved spinor vertices.  These adapters return
validated arrays; writing the final calculation-specific HDF5 is deferred
until the actual DFPT files and their metadata are available.

## Commands

```bash
wtorque inspect CONFIG.yml
wtorque extract-exchange CONFIG.yml
wtorque build-vertices CONFIG.yml
mpirun -np 8 wtorque compute-kernel CONFIG.yml
mpirun -np 8 wtorque project-phonons CONFIG.yml
mpirun -np 8 wtorque project-magnons CONFIG.yml
wtorque assemble-polaron CONFIG.yml
wtorque validate CONFIG.yml --report validation.md
wtorque benchmark CONFIG.yml --q-index 0
wtorque merge SHARD_DIRECTORY
```

Native spinor qe2pert/Wannier90 data can be checked without first converting
it to the strict workflow HDF5:

```bash
mpirun -np 8 wtorque test-native-epr-q0 \
  --epr PREFIX_epr.h5 --win PREFIX.win --amn PREFIX.amn --eig PREFIX.eig \
  --u-mat PREFIX_u.mat --u-dis-mat PREFIX_u_dis.mat \
  --u-dis-layout compact_outer_window \
  --magnetic-projections FIRST-LAST --atomic-spin-order interleaved \
  --magnetization-direction 0 0 1 \
  --fermi-energy-ev EF --energy-min-ev EMIN --energy-max-ev EMAX \
  --energy-points NENERGY --eta-ev ETA --temperature-k TEMPERATURE \
  --ep-energy-unit ry --ep-displacement-unit bohr
```

`--magnetic-projections` selects one-based AMN trial-projection columns, not
final MLWF indices. The full spinor Hamiltonian and full EPR electron-phonon
matrix remain in every Green function. Only the time-reversal-odd exchange
field and its rotation vertex are extracted in the selected polar projection
frame and embedded back into the full Wannier space. The current command is a
labelled bubble-only, collinear magnetic-subspace q=0 test; it also reports
projection conditioning, discarded non-collinear exchange weight,
longitudinal-vertex leakage both before and after collinearization, central
finite-difference error, EPR q=0 Hermiticity, and the acoustic sum.

The legacy EPR file does not store a Fermi level or unit attributes for
`ep_hop_*`. Therefore the Fermi energy, absolute integration interval, EPR
energy unit, and displacement unit are mandatory and are never guessed.
Single-q reconstruction reads the EPR payload once on rank zero, uses an
inverse FFT along the electron real-space axis, scatters k slices, and
distributes the full-space Green-function bubble over MPI ranks.

Paired scalar collinear EPR files can be tested without Wannier90 side files:

```bash
mpirun -np 8 wtorque test-collinear-epr-q0 \
  --up-epr PREFIX_up_epr.h5 --down-epr PREFIX_dn_epr.h5 \
  --magnetic-orbitals 'FIRST1-LAST1;FIRST2-LAST2' \
  --magnetic-atoms ATOM1,ATOM2 \
  --local-direction 0 0 1 --local-direction 0 0 -1 \
  --spin-quantization-direction 0 0 1 --spin-coordinate rotation_angle \
  --fermi-energy-ev EF --energy-min-ev EMIN --energy-max-ev EMAX \
  --energy-points NENERGY --eta-ev ETA --temperature-k TEMPERATURE \
  --epr-energy-unit ry --epr-displacement-unit bohr
```

Orbital and atom indices are one based. Semicolon-separated orbital groups
are paired in order with the atoms and repeated local directions. The two EPR
files must declare `basic_data/spinor=0` and matching structures and meshes;
their final Wannier indices are paired directly as declared by the inputs.
The Hamiltonian and electron-phonon matrices are lifted into a full
orbital-major spinor block, while only the union of the selected magnetic
orbital groups is used to form and rotate `H_XC`. This makes the no-SOC
collinear selection-rule test auditable: `H`, `G`, and `g` must be
spin-conserving, the transverse torque must be spin-flipping, and the mixed
loop must vanish up to floating-point noise. Rank zero reconstructs the two
legacy EPR payloads and MPI ranks receive disjoint k slices.

Both the Hamiltonian and the spin-dependent part of `g_up/g_down` are lifted
along `--spin-quantization-direction`. For a non-z magnetic direction the
canonical z-basis matrices therefore contain spin-off-diagonal entries even
without SOC; the reported commutator with the selected quantization Pauli
matrix is the basis-independent spin-conservation diagnostic.

A controlled onsite-SOC propagation model can be added without changing
`H_XC` or the scalar DFPT vertices. Each manifold and its orbital convention
remain explicit inputs:

```bash
wtorque test-collinear-epr-q0 \
  ... \
  --onsite-soc 'p:11-13:0.5' --onsite-soc 'p:14-16:0.5' \
  --soc-p-order 'pz,px,py' \
  --integration-backend analytic_zero_temperature
```

`--onsite-soc` uses `p|d:ONE_BASED_INDICES:LAMBDA_EV` and can be repeated.
Selected manifolds must be disjoint and contain exactly three p or five d
orbitals. The constructed matrix is checked for Hermiticity and time-reversal
evenness before it is added only to the full Green-function Hamiltonian. At
q=0 and `temperature-k=0`, `analytic_zero_temperature` integrates the two
retarded Green-function poles exactly over the occupied part of the requested
energy interval; `energy-points` remains a required compatibility input but is
not used by this backend.

The same paired-scalar route can perform a Gamma-only magnon-polaron smoke
test using an explicitly projected scalar TB2J exchange file:

```bash
wtorque test-collinear-epr-q0-polaron \
  --up-epr PREFIX_up_epr.h5 --down-epr PREFIX_dn_epr.h5 \
  --magnetic-orbitals '1-5;6-10' --magnetic-atoms 1,2 \
  --local-direction 0 1 0 --local-direction 0 -1 0 \
  --spin-quantization-direction 0 1 0 --spin-coordinate rotation_angle \
  --onsite-soc 'p:11-13:0.5' --onsite-soc 'p:14-16:0.5' \
  --soc-p-order 'pz,px,py' \
  --fermi-energy-ev 11.4 --energy-min-ev 3.5 --energy-max-ev 11.4 \
  --energy-points 64 --integration-backend analytic_zero_temperature \
  --eta-ev 0.05 --temperature-k 0 \
  --epr-energy-unit ry --epr-displacement-unit bohr \
  --exchange-h5 PROJECTED_J.h5 \
  --exchange-source-directed-bond-weight 1 \
  --exchange-spin-normalization unit_vector --spin-length 2.5 \
  --anisotropy-mev 0.0005 \
  --anisotropy-spin-normalization unit_vector
```

No material value is inferred by this command. The TB2J source directed-bond
weight, spin normalization, spin length, and single-ion anisotropy are inputs.
The adapter consumes only `J_iso_r` from a file carrying
`scalar_spin_group_projected=1`; it records the conversion from the supplied
source weight to the native mate-complete weight `1/2`. The source HDF5 is not
modified.

The EPR force constants provide the Gamma phonons. Zero-frequency acoustic
modes are excluded rather than silently normalized, and the remaining modes
are contracted with physical zero-point displacements. The torque coordinates
are rotated into the exact local frame used by native LSWT before the magnon
paraunitary transform. Output includes normal and anomalous coupling, RWA
hybrid energies, degenerate-subspace coupling invariants, and a full bosonic
BdG stability diagnostic. This command does not consume `dJ/du`; that input
belongs to the exchange-striction one-phonon--two-magnon route.

MPI ownership is assigned over unique q pairs. Rank zero alone writes HDF5,
and rank failures are exchanged before the next collective so peers do not
continue into a conflicting write. A launcher-detected multi-rank run without
`mpi4py` fails immediately.

## Restart and output

Each `/q_data/q_NNNNNN` group transitions through `pending`, `running`,
`complete`, or `failed`. `complete` is written only after the payload is
flushed and checksummed. Resume checks the resolved configuration, all source
hashes, q mesh, orbital masks through the manifest, and each completed payload
checksum. Canonical stacked datasets are materialized at:

- `/kernel/K_pi_u` or `/kernel/V_pi_ph`;
- `/coupling/g_mp_normal` and `/coupling/g_mp_anomalous`;
- `/polaron/H_rwa` and `/polaron/energy`;
- `/validation/report_json`.

The ordinary production path is bubble-only. The fixed-projector direct
vertex is available as a gated library interface and requires exchange-field
resolved `g_XC`; it is not silently constructed from the full DFPT `g`.

## Validation commands

```bash
python -m pytest -q tests/wtorque
python -m ruff check slw/wtorque tests/wtorque
python -m mypy slw/wtorque --ignore-missing-imports
python -m pip wheel . --no-deps
```

The equation registry test cross-checks every implemented `WT-*` identifier
against the supplied equation index and requires an explicit provenance class.
