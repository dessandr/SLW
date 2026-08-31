# Implementation Checklist

## Before coding

- [ ] Read task, dependencies, conventions, and source provenance.
- [ ] Identify equation IDs.
- [ ] Confirm shapes, units, basis order, gauge, spin coordinate, exchange extraction, and site projection.
- [ ] Resolve the magnetic orbital mask.
- [ ] Add a failing test first.

## During coding

- [ ] Preserve complex128.
- [ ] Keep `H_TRS`, `H_XC`, and full `H` distinct.
- [ ] Rotate only `H_XC`.
- [ ] Avoid hidden phase or normalization conversions.
- [ ] Keep electronic and bosonic projection layers separate.
- [ ] Add explicit domain errors.
- [ ] Record assumptions and source IDs.

## Before completion

- [ ] Unit and finite-difference tests pass.
- [ ] Time-reversal and local-partition residuals pass.
- [ ] Shapes/units are documented.
- [ ] HDF5 metadata and source hashes are complete.
- [ ] Restart behavior is tested where applicable.
- [ ] No unexplained sign flip, `np.imag` shortcut, or metric insertion exists.
