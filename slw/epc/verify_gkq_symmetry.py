#!/usr/bin/env python3
import h5py
import numpy as np
import argparse

def verify_gkq_symmetry(h5_file1, h5_file2):
    print(f"Loading '{h5_file1}' and '{h5_file2}'...")

    with h5py.File(h5_file1, 'r') as f1, h5py.File(h5_file2, 'r') as f2:
        g1 = np.asarray(f1['g_band'], dtype=np.complex128)
        g2 = np.asarray(f2['g_band'], dtype=np.complex128)

        q1 = np.asarray(f1['q_fracs'])
        q2 = np.asarray(f2['q_fracs'])

    assert g1.shape == g2.shape, f"Shape mismatch: {g1.shape} vs {g2.shape}"
    Nq, Nk, dim, _ = g1.shape

    max_fro_diff = 0.0
    max_svd_diff = 0.0

    # Random sample of (q, k) points to check
    np.random.seed(42)
    sample_indices = [
        (np.random.randint(0, Nq), np.random.randint(0, Nk))
        for _ in range(min(100, Nq * Nk))
    ]
    # Always include q=0
    q0_list = np.where(np.all(np.isclose(q1, 0.0, atol=1e-12), axis=1))[0]
    if len(q0_list) > 0:
        sample_indices.append((q0_list[0], 0))

    print(f"\n--- Checking {len(sample_indices)} random (q, k) points ---")

    for iq, ik in sample_indices:
        mat1 = g1[iq, ik]
        mat2 = g2[iq, ik]

        # 1. Frobenius Norm Check
        norm1 = np.linalg.norm(mat1, 'fro')
        norm2 = np.linalg.norm(mat2, 'fro')
        fro_diff = abs(norm1 - norm2) / max(1e-15, norm1)
        max_fro_diff = max(max_fro_diff, fro_diff)

        # 2. Singular Value Spectrum Check (Gauge exactly invariant)
        sv1 = np.linalg.svd(mat1, compute_uv=False)
        sv2 = np.linalg.svd(mat2, compute_uv=False)

        sv_diff = np.linalg.norm(sv1 - sv2) / max(1e-15, np.linalg.norm(sv1))
        max_svd_diff = max(max_svd_diff, sv_diff)

    print(f"Max Frobenius Norm relative diff: {max_fro_diff:.4e}")
    print(f"Max Singular Values relative diff: {max_svd_diff:.4e}")

    if max_fro_diff > 1e-6 or max_svd_diff > 1e-6:
        print("\n❌ FAILURE: g(k,q) matrices for Mn1 and Mn2 are physically DIFFERENT.")
        print("This means the asymmetry arises before the real-space transform.")
    else:
        print("\n✅ SUCCESS: g(k,q) matrices share identical Gauge-invariant physics.")
        print("The asymmetry is just a spatial phase or index permutation that needs to be correctly mapped in the dJ/du equation.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("gkq1", help="Path to Mn1 gkq HDF5 file")
    parser.add_argument("gkq2", help="Path to Mn2 gkq HDF5 file")
    args = parser.parse_args()
    verify_gkq_symmetry(args.gkq1, args.gkq2)
