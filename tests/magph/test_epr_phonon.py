from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from slw.core.constants import RY_TO_EV
from slw.magph.epr_phonon import (
    build_epr_phonon_cache_payload,
    write_epr_phonon_cache,
)
from slw.magph.phonon import load_phonon_cache


def _write_one_atom_epr(path: Path, *, frequency_mev: float = 5.0) -> None:
    mass = 4.0
    eigenvalue_ry2 = (frequency_mev / (1000.0 * RY_TO_EV)) ** 2
    force_constant = mass * eigenvalue_ry2 * np.eye(3, dtype=np.float64)
    with h5py.File(path, "w") as handle:
        basic = handle.create_group("basic_data")
        basic.create_dataset("at", data=np.eye(3, dtype=np.float64))
        basic.create_dataset("tau", data=np.zeros((1, 3), dtype=np.float64))
        basic.create_dataset("mass", data=np.asarray((mass,), dtype=np.float64))
        basic.create_dataset("qc_dim", data=np.ones(3, dtype=np.int64))
        basic.create_dataset("nat", data=np.asarray(1, dtype=np.int64))
        basic.create_dataset("alat", data=np.asarray(1.0, dtype=np.float64))
        basic.create_dataset("lpolar", data=np.asarray(False))
        force = handle.create_group("force_constant")
        force.create_dataset("ifc1", data=force_constant[None, :, :])


class EPRPhononCacheTests(unittest.TestCase):
    def test_vectorized_builder_produces_a_loadable_mass_normalized_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            epr = root / "sample_epr.h5"
            output = root / "phonon.npz"
            _write_one_atom_epr(epr)

            report = write_epr_phonon_cache(
                epr,
                output,
                q_mesh_shape=(1, 1, 1),
                q_chunk_size=1,
                loto_mode="none",
            )
            cache = load_phonon_cache(output)

            self.assertEqual(report.q_mesh_shape, (1, 1, 1))
            self.assertEqual(report.q_point_count, 1)
            self.assertEqual(report.mode_count, 3)
            np.testing.assert_allclose(cache.frequencies_mev, 5.0, atol=1.0e-12)
            norms = np.einsum(
                "a,qmai,qmai->qm",
                cache.masses,
                cache.eigenvectors_mass_normalized.conj(),
                cache.eigenvectors_mass_normalized,
                optimize=True,
            )
            np.testing.assert_allclose(norms, 1.0, atol=1.0e-13)

    def test_imaginary_roundoff_policy_is_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            epr = Path(directory) / "sample_epr.h5"
            _write_one_atom_epr(epr, frequency_mev=0.0)
            payload, report = build_epr_phonon_cache_payload(
                epr,
                imaginary_tolerance_mev=1.0e-6,
            )

            self.assertEqual(report.rounded_frequency_count, 3)
            np.testing.assert_array_equal(payload["ph_en_flat"], 0.0)
            self.assertEqual(int(payload["rounded_frequency_count"]), 3)


if __name__ == "__main__":
    unittest.main()
