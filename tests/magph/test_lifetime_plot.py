from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from slw.magph.lifetime_plot import (
    build_parser,
    load_lifetime_plot_data,
    plot_lifetime,
)
from slw.magph.lswt import uniform_fractional_mesh


def _native_product(path: Path, *, embed_geometry: bool = True) -> Path:
    points = uniform_fractional_mesh((4, 4, 2), shift=(0.5, 0.5, 0.5))
    phase = 2.0 * np.pi * points[:, 0]
    energy = np.column_stack((2.0 + np.cos(phase), 3.0 - np.cos(phase)))
    gamma = np.column_stack(
        (0.01 + 0.002 * np.sin(phase), 0.02 - 0.002 * np.sin(phase))
    )
    metadata = {
        "calculation": "lifetime",
        "schema_version": 1,
        "kmesh": [4, 4, 2],
        "kshift": [0.5, 0.5, 0.5],
        "magnetic_atom_indices": [0, 1],
        "spin_pattern": [1.0, -1.0],
        "negative_tolerance_mev": 0.0,
    }
    if embed_geometry:
        metadata.update(
            {
                "lattice_ang": [[3.0, 0.0, 0.0], [-1.5, 2.598, 0.0], [0.0, 0.0, 5.0]],
                "tau_frac": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.5]],
                "atom_labels": ["Mn1", "Mn2"],
            }
        )
    observable = 2.0 * gamma
    np.savez_compressed(
        path,
        schema_version=np.asarray(1, dtype=np.int64),
        metadata_json=np.asarray(json.dumps(metadata)),
        k_points_frac=points,
        energy_mev=energy,
        gamma_hwhm_mev=gamma,
        fwhm_mev=observable,
        scattering_rate_ps_inv=observable / 0.6582119569,
        lifetime_ps=0.6582119569 / observable,
        valid_damping=np.ones(gamma.shape, dtype=np.bool_),
    )
    return path


class NativeLifetimePlotTests(unittest.TestCase):
    def test_native_schema_loads_directly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            product = _native_product(Path(directory) / "lifetime.npz")
            data = load_lifetime_plot_data(product)
        self.assertEqual(data.k_points_frac.shape, (32, 3))
        self.assertEqual(data.energy_mev.shape, (32, 2))
        self.assertEqual(data.mode_count, 2)
        np.testing.assert_allclose(data.fwhm_mev, 2.0 * data.gamma_hwhm_mev)
        self.assertEqual(data.atom_labels, ("Mn1", "Mn2"))

    def test_stored_chirality_reorders_every_mode_observable_together(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            product = Path(directory) / "chirality.npz"
            points = uniform_fractional_mesh((2, 1, 1), shift=(0.5, 0.0, 0.0))
            energy = np.asarray(((1.0, 2.0), (3.0, 4.0)))
            gamma = np.asarray(((0.1, 0.2), (0.3, 0.4)))
            chirality = np.asarray(((-1.0, 1.0), (1.0, -1.0)))
            metadata = {
                "schema_version": 2,
                "magnetic_order": "collinear_afm",
                "magnon_mode_order": "energy_ascending",
                "negative_tolerance_mev": 0.0,
            }
            np.savez_compressed(
                product,
                metadata_json=np.asarray(json.dumps(metadata)),
                k_points_frac=points,
                energy_mev=energy,
                gamma_hwhm_mev=gamma,
                magnon_chirality=chirality,
            )
            data = load_lifetime_plot_data(product)

        np.testing.assert_allclose(data.magnon_chirality, ((1.0, -1.0),) * 2)
        np.testing.assert_allclose(data.energy_mev, ((2.0, 1.0), (3.0, 4.0)))
        np.testing.assert_allclose(
            data.gamma_hwhm_mev, ((0.2, 0.1), (0.3, 0.4))
        )
        self.assertEqual(data.mode_labels, (r"$\chi=+1$", r"$\chi=-1$"))

    def test_old_array_aliases_remain_readable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            product = Path(directory) / "legacy.npz"
            points = uniform_fractional_mesh((2, 2, 1))
            energy = np.ones((4, 2), dtype=np.float64)
            linewidth = np.full((4, 2), 0.1, dtype=np.float64)
            np.savez(
                product,
                k_mesh_frac=points,
                energy=energy,
                linewidth=linewidth,
                physical_channel_count=np.asarray(2),
            )
            data = load_lifetime_plot_data(product)
        np.testing.assert_allclose(data.gamma_hwhm_mev, linewidth)
        np.testing.assert_allclose(data.fwhm_mev, 2.0 * linewidth)
        self.assertIsNone(data.lattice_ang)

    def test_exchange_geometry_fallback_supports_existing_native_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            product = _native_product(root / "lifetime.npz", embed_geometry=False)
            exchange = root / "J.h5"
            with h5py.File(exchange, "w") as handle:
                basic = handle.create_group("basic_data")
                basic.create_dataset("lattice_ang", data=np.eye(3))
                basic.create_dataset("tau_frac", data=np.zeros((2, 3)))
                basic.create_dataset(
                    "atom_labels",
                    data=np.asarray(("Mn1", "Mn2"), dtype=object),
                    dtype=h5py.string_dtype("utf-8"),
                )
            data = load_lifetime_plot_data(product, exchange_h5=exchange)
        np.testing.assert_allclose(data.lattice_ang, np.eye(3))

    def test_plotter_writes_native_bz_maps_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            product = _native_product(root / "lifetime.npz")
            output = root / "plots"
            args = build_parser().parse_args(
                [
                    str(product),
                    "--output-dir",
                    str(output),
                    "--slices",
                    "0.0",
                    "--formats",
                    "png",
                    "--dpi",
                    "40",
                ]
            )
            figures, summary_path, summary = plot_lifetime(args)
            self.assertEqual(len(figures), 5)
            self.assertTrue(all(path.stat().st_size > 0 for path in figures))
            self.assertTrue(summary_path.is_file())
        self.assertEqual(summary["selected_slices"], [0.25])
        self.assertEqual(
            summary["plots"],
            ["energy", "lifetime", "linewidth", "scattering-rate", "splitting"],
        )


if __name__ == "__main__":
    unittest.main()
