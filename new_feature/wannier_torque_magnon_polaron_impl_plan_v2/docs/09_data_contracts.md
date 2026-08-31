# HDF5 Data Contracts

All canonical complex arrays load as complex128 and all real arrays as float64.

## Canonical spinor basis

- `norb`: orbital Wannier functions before spin doubling.
- `nw=2*norb`.
- canonical order: `[orb0_up, orb0_dn, orb1_up, orb1_dn, ...]`.

Importers convert other orders explicitly.

## Electronic model

```text
/electrons/lattice                    [3,3] float64, Angstrom
/electrons/R_vectors                  [nR,3] int32
/electrons/H_R                        [nR,nw,nw] complex128, eV
/electrons/orbital_centers            [norb,3] float64
/electrons/orbital_site               [norb] int32
/electrons/orbital_labels             [norb] UTF-8
/electrons/kpoints                    [nk,3] float64, reduced
/electrons/weights                    [nk] float64
/electrons/fermi_energy               scalar float64, eV
```

## Exchange-field decomposition

One of the following is required.

### Explicit

```text
/electrons/H_TRS_R                    [nR,nw,nw] complex128
/electrons/H_XC_R                     [nR,nw,nw] complex128
```

### Time-reversal sewing

```text
/time_reversal/B_theta_k              [nk,nw,nw] complex128
/time_reversal/minus_k_index          [nk] int32
```

### Collinear split

```text
/electrons/H_up_R                     [nR,norb,norb] complex128
/electrons/H_dn_R                     [nR,norb,norb] complex128
```

Required metadata:

```text
exchange_extraction = explicit | time_reversal | collinear_split
exchange_factor_convention = string
```

## Magnetic subspace and local projection

```text
/spin/magnetic_atom_index             [nmag] int32
/spin/magnetic_site_position          [nmag,3] float64, fractional
/spin/magnetic_orbital_mask           [nmag,norb] bool
/spin/local_frames                    [nmag,3,3] float64, rows [t1,t2,n]
/spin/spin_length                     [nmag] float64
/spin/torque_vertex_qk                [nq,nk,nmag,2,nw,nw] complex128, optional user-supplied
```

Metadata:

```text
site_projection = local_partition | onsite_only | user_supplied
spin_coordinate = transverse_direction | rotation_angle
magnetic_subspace_policy = explicit_labels | explicit_indices
```

## DFPT Cartesian input

```text
/dfpt/qpoints                         [nq,3] float64
/dfpt/q_000000/g_cart                 [nk,npert,nw,nw] complex128
/dfpt/pert_atom                       [npert] int32
/dfpt/pert_cart                       [npert] int32
/dfpt/q_000000/kplusq_index           [nk] int32, optional cache
/dfpt/q_000000/G_wrap                 [nk,3] int32, optional cache
```

## Optional exchange-field DFPT input

Required only for the direct mixed vertex:

```text
/dfpt/q_000000/g_xc_cart              [nk,npert,nw,nw] complex128
```

## Mode-normalized DFPT input

```text
/dfpt/q_000000/g_mode                 [nk,nphonon,nw,nw] complex128, eV
/phonon/q_000000/frequency            [nphonon] float64, eV
/phonon/q_000000/eigenvector          [natom,3,nphonon] complex128
```

## Magnon input

```text
/magnon/qpoints                       [nq,3] float64
/magnon/energy                        [nq,nbranch] float64, eV
/magnon/T_para                        [nq,2*nmag,2*nmag] complex128
/magnon/metric                        [2*nmag] float64
/magnon/local_frames                  [nmag,3,3] float64
/magnon/spin_length                   [nmag] float64
```

Metadata declares Nambu order, positive/negative-norm column order, Fourier gauge, and upstream paraunitarity residual.

## Output

```text
/meta/run_manifest_json               UTF-8
/meta/source_hashes                   UTF-8 table
/meta/source_provenance_ids           UTF-8 table
/meta/spin_coordinate                 UTF-8
/meta/site_projection                 UTF-8
/meta/exchange_extraction             UTF-8
/qpoints                              [nq,3]
/kernel/A_retarded                    [nq,nmag,2,natom,3] complex128, optional
/kernel/K_pi_u                        [nq,nmag,2,natom,3] complex128, eV/Angstrom
/kernel/K_direct                      same shape, optional
/kernel/V_pi_ph                       [nq,nmag,2,nphonon] complex128, eV
/coupling/g_mp_normal                 [nq,nbranch,nphonon] complex128, eV
/coupling/g_mp_anomalous              [nq,nbranch,nphonon] complex128, eV
/validation/...                       residuals and sensitivity tables
```

## Restart contract

Each q group has `pending`, `running`, `complete`, or `failed`. `complete` is set only after flush and checksum creation.
