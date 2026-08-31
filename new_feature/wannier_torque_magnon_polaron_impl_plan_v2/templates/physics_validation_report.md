# Physics Validation Report

## Run identity

- Run name:
- Code commit:
- Input hashes:
- Source-provenance IDs:
- Backend/date:

## Conventions

- Bloch/Fourier gauge:
- Basis/spin order:
- Exchange extraction:
- Time-reversal sewing source:
- Magnetic orbital masks:
- Site projection:
- Spin coordinate:
- `g` normalization/lift:
- Direct mixed vertex:

## Algebra and decomposition

| Check | Residual | Tolerance | Status |
|---|---:|---:|---|
| H Hermiticity | | | |
| TRS time-reversal parity | | | |
| XC time-reversal parity | | | |
| H reconstruction | | | |
| Local-partition closure | | | |
| Vertex Hermiticity | | | |
| Longitudinal vertex | | | |
| Local-frame orthogonality | | | |

## Electronic response

| Test | Result | Reference | Status |
|---|---:|---:|---|
| Vertex finite difference | | | |
| Mixed finite difference | | | |
| SOC-isolation vertex | | | |
| Spin-rotation-invariant SOC-off kernel | | | |
| q conjugation | | | |
| Rigid translation | | | |

## Magnetic-subspace sensitivity

| Subspace/projection | Observable | Difference from default | Status |
|---|---:|---:|---|
| magnetic d + local partition | | 0 | |
| full magnetic atom | | | |
| onsite only | | | |

## Projection

| Test | Residual | Status |
|---|---:|---|
| Cartesian vs mode-g | | |
| phonon rephasing | | |
| magnon paraunitarity | | |
| magnon rephasing | | |
| degenerate-subspace invariant | | |
| RWA crossing | | |

## Convergence

| Parameter | Values | Observable | Conclusion |
|---|---|---|---|
| k mesh | | | |
| energy/contour mesh | | | |
| broadening | | | |
| Wannier window | | | |
| magnetic subspace | | | |
| perturbation chunk | | | |

## Final status

- [ ] Pass
- [ ] Pass with documented warnings
- [ ] Fail
