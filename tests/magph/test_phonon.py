import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy import constants as scipy_constants

from slw.core.constants import (
    ZPF_ANG_SQRT_MEV_AMU,
    ZPF_ANG_SQRT_MEV_ELECTRON_MASS,
)
from slw.magph.phonon import (
    CELL_GAUGE_FOURIER_PHASE_CONVENTION,
    CELL_GAUGE_VECTOR_CONVENTION,
    FrequencyFloorProvenance,
    PhononInputError,
    PhononMassUnit,
    load_phonon_cache,
    zero_point_displacements,
)


def _write_cache(
    path: Path,
    *,
    mass_unit: str = "electron_mass",
    frequency=4.0,
    masses=(4.0,),
    eigenvectors=None,
    schema_version=3,
    vector_convention: str = CELL_GAUGE_VECTOR_CONVENTION,
    fourier_phase_convention: str = CELL_GAUGE_FOURIER_PHASE_CONVENTION,
):
    mass_key = "atom_mass_electron" if mass_unit == "electron_mass" else "atom_mass_amu"
    masses_array = np.asarray(masses, dtype=np.float64)
    if eigenvectors is None:
        eigenvectors = np.zeros(
            (1, 3 * masses_array.size, masses_array.size, 3), dtype=np.complex128
        )
        for mode in range(3 * masses_array.size):
            atom, axis = divmod(mode, 3)
            eigenvectors[0, mode, atom, axis] = 1.0 / np.sqrt(masses_array[atom])
    frequencies = np.broadcast_to(
        np.asarray(frequency, dtype=np.float64),
        (1, 3 * masses_array.size),
    ).copy()
    np.savez(
        path,
        phonon_cache_schema_version=np.asarray(schema_version),
        q_mesh_flat_frac=np.zeros((1, 3)),
        q_mesh_shape=np.ones(3, dtype=np.int32),
        ph_en_flat=frequencies,
        ph_vec_flat=np.asarray(eigenvectors, dtype=np.complex128),
        mass_unit=np.asarray(mass_unit),
        energy_unit=np.asarray("meV"),
        phonon_vector_convention=np.asarray(vector_convention),
        fourier_phase_convention=np.asarray(fourier_phase_convention),
        **{mass_key: masses_array},
    )


class PhononCacheTests(unittest.TestCase):
    def test_zero_point_prefactors_match_si_harmonic_oscillator(self):
        one_mev_joule = 1.0e-3 * scipy_constants.electron_volt
        electron_mass_factor = (
            np.sqrt(
                scipy_constants.hbar**2 / (2.0 * scipy_constants.m_e * one_mev_joule)
            )
            / 1.0e-10
        )
        amu_factor = (
            np.sqrt(
                scipy_constants.hbar**2
                / (2.0 * scipy_constants.atomic_mass * one_mev_joule)
            )
            / 1.0e-10
        )

        self.assertAlmostEqual(
            ZPF_ANG_SQRT_MEV_ELECTRON_MASS,
            electron_mass_factor,
            places=9,
        )
        self.assertAlmostEqual(ZPF_ANG_SQRT_MEV_AMU, amu_factor, places=9)

    def test_electron_mass_zero_point_factor_matches_analytic_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "phonon.npz"
            _write_cache(path, mass_unit="electron_mass", frequency=4.0)
            cache = load_phonon_cache(path)
            result = zero_point_displacements(cache)

        self.assertIs(cache.mass_unit, PhononMassUnit.ELECTRON_MASS)
        expected = 0.5 * ZPF_ANG_SQRT_MEV_ELECTRON_MASS / np.sqrt(4.0)
        self.assertAlmostEqual(result.values_ang[0, 0, 0, 0].real, expected)
        self.assertFalse(result.regularized.any())
        self.assertFalse(result.values_ang.flags.writeable)
        self.assertFalse(result.input_frequencies_mev.flags.writeable)
        self.assertFalse(result.effective_frequencies_mev.flags.writeable)
        np.testing.assert_array_equal(
            result.effective_frequencies_mev,
            cache.frequencies_mev,
        )
        self.assertIs(
            result.frequency_floor_provenance,
            FrequencyFloorProvenance.NONE,
        )

    def test_amu_and_electron_mass_prefactors_are_not_interchangeable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "phonon.npz"
            _write_cache(path, mass_unit="amu", frequency=1.0)
            cache = load_phonon_cache(path)
            value = zero_point_displacements(cache).values_ang[0, 0, 0, 0].real

        self.assertAlmostEqual(value, 0.5 * ZPF_ANG_SQRT_MEV_AMU)
        ratio = ZPF_ANG_SQRT_MEV_ELECTRON_MASS / ZPF_ANG_SQRT_MEV_AMU
        self.assertGreater(ratio, 42.0)

    def test_same_physical_mass_is_unit_representation_invariant(self):
        electron_masses_per_amu = scipy_constants.atomic_mass / scipy_constants.m_e
        with tempfile.TemporaryDirectory() as tmp:
            electron_path = Path(tmp) / "electron.npz"
            amu_path = Path(tmp) / "amu.npz"
            _write_cache(
                electron_path,
                mass_unit="electron_mass",
                frequency=2.0,
                masses=(electron_masses_per_amu,),
            )
            _write_cache(
                amu_path,
                mass_unit="amu",
                frequency=2.0,
                masses=(1.0,),
            )
            electron_value = zero_point_displacements(
                load_phonon_cache(electron_path)
            ).values_ang[0, 0, 0, 0]
            amu_value = zero_point_displacements(
                load_phonon_cache(amu_path)
            ).values_ang[0, 0, 0, 0]

        self.assertAlmostEqual(electron_value.real, amu_value.real, places=9)

    def test_mass_metadata_and_array_name_must_agree(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "phonon.npz"
            _write_cache(path, mass_unit="electron_mass")
            with np.load(path, allow_pickle=False) as original:
                payload = {name: np.array(original[name]) for name in original.files}
            payload["mass_unit"] = np.asarray("amu")
            np.savez(path, **payload)
            with self.assertRaisesRegex(PhononInputError, "atom_mass_amu"):
                load_phonon_cache(path)

    def test_schema_v3_requires_three_modes_per_atom(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "phonon.npz"
            _write_cache(path)
            with np.load(path, allow_pickle=False) as original:
                payload = {name: np.array(original[name]) for name in original.files}
            payload["ph_en_flat"] = payload["ph_en_flat"][:, :2]
            payload["ph_vec_flat"] = payload["ph_vec_flat"][:, :2]
            np.savez(path, **payload)
            with self.assertRaisesRegex(PhononInputError, r"3\*nat"):
                load_phonon_cache(path)

    def test_only_integer_schema_version_three_is_supported(self):
        invalid_versions = (2, 4, 3.0, 3.5, True, "3", [3])
        with tempfile.TemporaryDirectory() as tmp:
            for index, version in enumerate(invalid_versions):
                with self.subTest(version=version):
                    path = Path(tmp) / f"phonon-{index}.npz"
                    _write_cache(path, schema_version=version)
                    with self.assertRaisesRegex(
                        PhononInputError,
                        "schema_version|schema version|schema 2",
                    ):
                        load_phonon_cache(path)

    def test_vector_and_fourier_conventions_are_exact_and_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            wrong_vector = Path(tmp) / "wrong-vector.npz"
            wrong_fourier = Path(tmp) / "wrong-fourier.npz"
            missing_fourier = Path(tmp) / "missing-fourier.npz"
            _write_cache(
                wrong_vector,
                vector_convention=CELL_GAUGE_VECTOR_CONVENTION + " ",
            )
            _write_cache(
                wrong_fourier,
                fourier_phase_convention=(
                    "u_(l,kappa) proportional exp(-i2pi q dot R_l)"
                ),
            )
            _write_cache(missing_fourier)
            with np.load(missing_fourier, allow_pickle=False) as original:
                payload = {
                    name: np.array(original[name])
                    for name in original.files
                    if name != "fourier_phase_convention"
                }
            np.savez(missing_fourier, **payload)

            with self.assertRaisesRegex(
                PhononInputError,
                "phonon_vector_convention",
            ):
                load_phonon_cache(wrong_vector)
            with self.assertRaisesRegex(
                PhononInputError,
                "fourier_phase_convention",
            ):
                load_phonon_cache(wrong_fourier)
            with self.assertRaisesRegex(
                PhononInputError,
                "missing required fields.*fourier_phase_convention",
            ):
                load_phonon_cache(missing_fourier)

    def test_direct_construction_canonicalizes_mass_unit_before_branching(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "phonon.npz"
            _write_cache(path, mass_unit="electron_mass", frequency=4.0)
            cache = load_phonon_cache(path)

        direct = replace(cache, mass_unit="electron_mass")
        self.assertIs(direct.mass_unit, PhononMassUnit.ELECTRON_MASS)
        value = zero_point_displacements(direct).values_ang[0, 0, 0, 0].real
        self.assertAlmostEqual(
            value,
            0.5 * ZPF_ANG_SQRT_MEV_ELECTRON_MASS / np.sqrt(4.0),
        )
        direct_amu = replace(cache, mass_unit="amu")
        self.assertIs(direct_amu.mass_unit, PhononMassUnit.AMU)
        amu_value = zero_point_displacements(direct_amu).values_ang[0, 0, 0, 0].real
        self.assertAlmostEqual(
            amu_value,
            0.5 * ZPF_ANG_SQRT_MEV_AMU / np.sqrt(4.0),
        )
        with self.assertRaisesRegex(PhononInputError, "schema_version"):
            replace(cache, schema_version=3.0)
        with self.assertRaisesRegex(PhononInputError, "q_mesh_flat_frac"):
            replace(cache, q_points_frac=np.zeros((1, 2)))
        with self.assertRaisesRegex(
            PhononInputError,
            "fourier_phase_convention",
        ):
            replace(cache, fourier_phase_convention="atomic gauge")

    def test_zero_mode_requires_explicit_floor(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "phonon.npz"
            _write_cache(path, frequency=0.0)
            cache = load_phonon_cache(path)

        with self.assertRaisesRegex(PhononInputError, "frequency_floor"):
            zero_point_displacements(cache)
        result = zero_point_displacements(cache, frequency_floor_mev=0.25)
        self.assertTrue(result.regularized[0, 0])
        self.assertEqual(result.frequency_floor_mev, 0.25)
        self.assertIs(
            result.frequency_floor_provenance,
            FrequencyFloorProvenance.EXPLICIT_ARGUMENT,
        )
        np.testing.assert_array_equal(
            result.effective_frequencies_mev,
            np.full((1, 3), 0.25),
        )

        with self.assertRaisesRegex(
            PhononInputError,
            "effective_frequencies_mev is inconsistent",
        ):
            replace(
                result,
                effective_frequencies_mev=np.full((1, 3), 0.5),
            )
        with self.assertRaisesRegex(
            PhononInputError,
            "provenance='explicit_argument'",
        ):
            replace(
                result,
                frequency_floor_provenance=FrequencyFloorProvenance.NONE,
            )

    def test_negative_mode_is_never_clipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "phonon.npz"
            _write_cache(path, frequency=-0.1)
            cache = load_phonon_cache(path)

        with self.assertRaisesRegex(PhononInputError, "negative phonon"):
            zero_point_displacements(cache, frequency_floor_mev=0.25)


if __name__ == "__main__":
    unittest.main()
