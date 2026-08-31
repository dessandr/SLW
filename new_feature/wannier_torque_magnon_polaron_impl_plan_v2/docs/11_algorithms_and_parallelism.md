# Algorithms and Parallelism

## Reference q-point algorithm

For each q pair:

1. Load and validate q and `-q` metadata.
2. Build exact k+q maps and wrapping matrices.
3. For each energy node and k point, form `G_k` and `G_kq` once.
4. Build or load the local torque-vertex batch for that k,q pair.
5. Stream DFPT perturbations in chunks and contract all local vertices against `G_kq g G_k`.
6. Accumulate the complex retarded loops for q and `-q`.
7. Form the physical kernel with the q-pair finalizer.
8. Write q-group output and validation residuals atomically.

## Contraction

Avoid a tensor of shape `[ntorque,npert,nw,nw]`.

```python
A = G_kq @ g_chunk @ G_k
trace = einsum('tab,pba->tp', torque_vertices, A, optimize=True)
```

## Parallel hierarchy

1. q pairs: primary embarrassingly parallel level;
2. energy nodes or contour points;
3. k points;
4. perturbation chunks.

Start with q-level job arrays or MPI. Keep deterministic serial reduction for validation.

## Memory policy

The runtime estimates memory from `nw`, energy batching, and perturbation chunk size. It must fail before allocation when the configured limit is exceeded.

## Optimization order

- validated NumPy reference;
- batched BLAS/einsum;
- q-level multiprocessing or MPI;
- optional CuPy/spectral backends behind feature flags.

All optimized backends compare against the reference within a declared tolerance.
