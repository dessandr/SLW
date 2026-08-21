from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from slw.exchange.kernels.j_epr import (
    _apply_orbit_symmetry,
    _OrbitGrouping,
    _write_h5,
)
from slw.exchange.kernels.j_tensor_epr import (
    _write_tensor_h5 as _write_epr_tensor_h5,
)
from slw.exchange.kernels.j_wannier import (
    _write_tensor_h5_simple as _write_wannier_tensor_h5,
)
from slw.magph.model import (
    ExchangeCapability,
    ExchangeConvention,
    ExchangeRepresentation,
    ExchangeSpinNormalization,
    MagneticOrder,
)
from slw.magph.screening import load_exchange_h5, screen_magnetic_configuration


def _write_bonds(
    handle: h5py.File,
    atom_i: list[int],
    atom_j: list[int],
    shifts: list[tuple[int, int, int]],
    *,
    local_i: list[int] | None = None,
    local_j: list[int] | None = None,
    mirror: list[int] | None = None,
    kernel_family: str = "direct",
) -> None:
    basic = handle.require_group("basic_data")
    basic.create_dataset("hamiltonian_sign", data=np.bytes_("minus"))
    basic.create_dataset("spin_normalization", data=np.bytes_("unit_vector"))
    basic.create_dataset("source_spin_magnitude", data=np.asarray(1.0))
    basic.create_dataset("kernel_family", data=np.bytes_(kernel_family))
    basic.create_dataset("bond_coverage", data=np.bytes_("directed_mate_complete"))
    basic.create_dataset("directed_bond_weight", data=np.asarray(0.5))
    basic.create_dataset("realspace_gauge", data=np.bytes_("i_at_0_j_at_R"))
    bonds = handle.create_group("bonds")
    bonds.create_dataset("mag_i_atom", data=np.asarray(atom_i, dtype=np.int64))
    bonds.create_dataset("mag_j_atom", data=np.asarray(atom_j, dtype=np.int64))
    bonds.create_dataset("R", data=np.asarray(shifts, dtype=np.int64))
    if local_i is not None:
        bonds.create_dataset("mag_i_local", data=np.asarray(local_i, dtype=np.int64))
    if local_j is not None:
        bonds.create_dataset("mag_j_local", data=np.asarray(local_j, dtype=np.int64))
    if mirror is not None:
        bonds.create_dataset("mirror_index", data=np.asarray(mirror, dtype=np.int64))


def _write_scalar(
    path: Path,
    values: list[float],
    *,
    atom_i: list[int],
    atom_j: list[int],
    shifts: list[tuple[int, int, int]],
    dataset: str = "J_r/value",
    unit: str = "meV",
    local_i: list[int] | None = None,
    local_j: list[int] | None = None,
) -> None:
    with h5py.File(path, "w") as handle:
        _write_bonds(
            handle,
            atom_i,
            atom_j,
            shifts,
            local_i=local_i,
            local_j=local_j,
            kernel_family="scalar_lkag",
        )
        parent, name = dataset.rsplit("/", 1) if "/" in dataset else ("", dataset)
        group = handle.require_group(parent) if parent else handle
        output = group.create_dataset(name, data=np.asarray(values, dtype=np.float64))
        output.attrs["unit"] = unit


def _two_site_scalar(path: Path, coupling: float) -> None:
    _write_scalar(
        path,
        [coupling, coupling],
        atom_i=[4, 9],
        atom_j=[9, 4],
        shifts=[(0, 0, 0), (0, 0, 0)],
    )


def test_scalar_input_is_promoted_without_inventing_tensor_capability(
    tmp_path: Path,
) -> None:
    path = tmp_path / "scalar.h5"
    _two_site_scalar(path, 2.5)

    model, report = load_exchange_h5(path)

    assert model.representation is ExchangeRepresentation.ISOTROPIC
    assert model.source_dataset == "J_r/value"
    np.testing.assert_array_equal(model.magnetic_atom_indices, [4, 9])
    np.testing.assert_array_equal(model.bond_i, [0, 1])
    np.testing.assert_array_equal(model.bond_j, [1, 0])
    np.testing.assert_allclose(
        model.tensor_mev,
        np.asarray([2.5 * np.eye(3), 2.5 * np.eye(3)]),
    )
    assert report.promoted_isotropic
    assert report.mate_complete
    assert report.supports(ExchangeCapability.ISOTROPIC_STATIC_EXCHANGE)
    assert not report.supports(ExchangeCapability.FULL_TENSOR_STATIC_EXCHANGE)
    assert "exchange_striction" not in {item.value for item in report.capabilities}
    assert model.tensor_mev.flags.writeable is False
    with pytest.raises(ValueError, match="read-only"):
        model.tensor_mev[0, 0, 0] = 0.0


def test_noncontiguous_source_local_indices_are_canonicalised(tmp_path: Path) -> None:
    path = tmp_path / "noncontiguous_local.h5"
    _write_scalar(
        path,
        [1.0, 1.0],
        atom_i=[41, 87],
        atom_j=[87, 41],
        shifts=[(1, 0, 0), (-1, 0, 0)],
        dataset="J_iso_r",
        local_i=[10, 30],
        local_j=[30, 10],
    )

    model, report = load_exchange_h5(path)

    np.testing.assert_array_equal(model.magnetic_atom_indices, [41, 87])
    np.testing.assert_array_equal(model.bond_i, [0, 1])
    np.testing.assert_array_equal(model.bond_j, [1, 0])
    assert model.source_dataset == "J_iso_r"
    assert report.supports(ExchangeCapability.COLLINEAR_AFM_BIPARTITE)


def test_full_tensor_is_retained_and_checked_with_transpose_reciprocity(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tensor.h5"
    first = np.asarray(
        [[2.0, 0.3, -0.1], [0.2, 1.0, 0.4], [0.6, -0.5, 3.0]],
        dtype=np.float64,
    )
    tensors = np.stack((first, first.T))
    with h5py.File(path, "w") as handle:
        _write_bonds(
            handle,
            [5, 12],
            [12, 5],
            [(2, -1, 0), (-2, 1, 0)],
            mirror=[1, 0],
        )
        dataset = handle.create_dataset("J_tensor_r", data=tensors)
        dataset.attrs["unit"] = "meV"
        dataset.attrs["tensor_axis_order"] = "x,y,z"

    model, report = load_exchange_h5(path)

    assert model.representation is ExchangeRepresentation.TENSOR
    assert model.source_dataset == "J_tensor_r"
    np.testing.assert_allclose(model.tensor_mev, tensors)
    np.testing.assert_allclose(model.isotropic_mev, np.trace(tensors, axis1=1, axis2=2) / 3.0)
    assert not report.promoted_isotropic
    assert report.supports(ExchangeCapability.FULL_TENSOR_STATIC_EXCHANGE)
    assert not report.supports(ExchangeCapability.ISOTROPIC_STATIC_EXCHANGE)

    with pytest.raises(ValueError, match="explicit magnetic reference"):
        screen_magnetic_configuration(
            model,
            order=MagneticOrder.FM,
            require_local_stability=False,
        )
    with pytest.raises(ValueError, match="quantization_axis"):
        screen_magnetic_configuration(
            model,
            order=MagneticOrder.FM,
            spin_magnitudes=1.0,
            require_local_stability=False,
        )
    tensor_state = screen_magnetic_configuration(
        model,
        order=MagneticOrder.FM,
        spin_magnitudes=1.0,
        quantization_axis=(0.0, 0.0, 1.0),
        require_local_stability=False,
    )
    assert tensor_state.n_magnetic_sites == 2


def test_tensor_and_scalar_decomposition_must_agree(tmp_path: Path) -> None:
    path = tmp_path / "inconsistent.h5"
    tensors = np.asarray([np.eye(3), np.eye(3)])
    with h5py.File(path, "w") as handle:
        _write_bonds(
            handle,
            [0, 1],
            [1, 0],
            [(0, 0, 0), (0, 0, 0)],
        )
        tensor = handle.create_dataset("J_tensor_r", data=tensors)
        tensor.attrs["unit"] = "meV"
        tensor.attrs["tensor_axis_order"] = "x,y,z"
        scalar = handle.create_dataset("J_iso_r", data=[1.0, 1.2])
        scalar.attrs["unit"] = "meV"

    with pytest.raises(ValueError, match=r"trace\(J_tensor_r\)/3"):
        load_exchange_h5(path)


def test_isotropic_tensor_alias_is_accepted_only_when_actually_isotropic(
    tmp_path: Path,
) -> None:
    path = tmp_path / "iso_tensor_alias.h5"
    with h5py.File(path, "w") as handle:
        _write_bonds(
            handle,
            [2, 8],
            [8, 2],
            [(0, 0, 0), (0, 0, 0)],
        )
        tensor = np.asarray([1.5 * np.eye(3), 1.5 * np.eye(3)])
        dataset = handle.create_dataset("J_iso_tensor_r", data=tensor)
        dataset.attrs["unit"] = "meV"

    model, _ = load_exchange_h5(path)
    np.testing.assert_allclose(model.isotropic_mev, [1.5, 1.5])

    with h5py.File(path, "r+") as handle:
        handle["J_iso_tensor_r"][0, 0, 1] = 0.2
    with pytest.raises(ValueError, match="not an isotropic tensor"):
        load_exchange_h5(path)


@pytest.mark.parametrize(
    ("atom_i", "atom_j", "shifts", "message"),
    [
        ([0, 1], [1, 0], [(0, 0, 0), (1, 0, 0)], "not mate-complete"),
        ([0, 0], [1, 1], [(0, 0, 0), (0, 0, 0)], "Duplicate directed bond"),
    ],
)
def test_invalid_bond_topology_is_rejected(
    tmp_path: Path,
    atom_i: list[int],
    atom_j: list[int],
    shifts: list[tuple[int, int, int]],
    message: str,
) -> None:
    path = tmp_path / "bad_bonds.h5"
    _write_scalar(
        path,
        [1.0, 1.0],
        atom_i=atom_i,
        atom_j=atom_j,
        shifts=shifts,
    )
    with pytest.raises(ValueError, match=message):
        load_exchange_h5(path)


def test_tensor_reciprocity_requires_transpose(tmp_path: Path) -> None:
    path = tmp_path / "bad_tensor_reciprocity.h5"
    first = np.asarray([[1.0, 0.2, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    with h5py.File(path, "w") as handle:
        _write_bonds(
            handle,
            [0, 1],
            [1, 0],
            [(0, 0, 0), (0, 0, 0)],
        )
        dataset = handle.create_dataset("J_tensor_r", data=np.stack((first, first)))
        dataset.attrs["unit"] = "meV"
        dataset.attrs["tensor_axis_order"] = "x,y,z"

    with pytest.raises(ValueError, match="J_mate = J.T"):
        load_exchange_h5(path)


def test_units_are_converted_and_conflicts_are_rejected(tmp_path: Path) -> None:
    converted = tmp_path / "ev.h5"
    _write_scalar(
        converted,
        [0.002, 0.002],
        atom_i=[0, 1],
        atom_j=[1, 0],
        shifts=[(0, 0, 0), (0, 0, 0)],
        unit="eV",
    )
    model, report = load_exchange_h5(converted)
    np.testing.assert_allclose(model.isotropic_mev, [2.0, 2.0])
    assert report.input_units == ("eV",)
    assert any("converted" in warning for warning in report.warnings)

    conflict = tmp_path / "unit_conflict.h5"
    with h5py.File(conflict, "w") as handle:
        _write_bonds(
            handle,
            [0, 1],
            [1, 0],
            [(0, 0, 0), (0, 0, 0)],
        )
        basic = handle.require_group("basic_data")
        basic.create_dataset("unit", data=np.bytes_("meV"))
        dataset = handle.create_dataset("J_iso_r", data=[1.0, 1.0])
        dataset.attrs["unit"] = "eV"
    with pytest.raises(ValueError, match="Conflicting exchange energy units"):
        load_exchange_h5(conflict)


def test_each_energy_dataset_requires_its_own_or_file_level_unit(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "sibling_unit_is_not_inherited.h5"
    with h5py.File(missing, "w") as handle:
        _write_bonds(handle, [0, 1], [1, 0], [(0, 0, 0), (0, 0, 0)])
        primary = handle.create_group("J_r").create_dataset("value", data=[0.001, 0.001])
        primary.attrs["unit"] = "eV"
        handle.create_dataset("J_iso_r", data=[1.0, 1.0])

    with pytest.raises(ValueError, match="missing for J_iso_r"):
        load_exchange_h5(missing)

    mixed = tmp_path / "mixed_explicit_units.h5"
    with h5py.File(mixed, "w") as handle:
        _write_bonds(handle, [0, 1], [1, 0], [(0, 0, 0), (0, 0, 0)])
        primary = handle.create_group("J_r").create_dataset("value", data=[0.001, 0.001])
        primary.attrs["unit"] = "eV"
        alias = handle.create_dataset("J_iso_r", data=[1.0, 1.0])
        alias.attrs["unit"] = "meV"

    model, report = load_exchange_h5(mixed)
    np.testing.assert_allclose(model.isotropic_mev, [1.0, 1.0])
    assert report.input_units == ("eV", "meV")


def test_fm_and_bipartite_afm_order_gates_do_not_infer_order_from_j(
    tmp_path: Path,
) -> None:
    fm_path = tmp_path / "fm_three_site.h5"
    _write_scalar(
        fm_path,
        [1.0] * 6,
        atom_i=[2, 5, 5, 11, 11, 2],
        atom_j=[5, 2, 11, 5, 2, 11],
        shifts=[(0, 0, 0)] * 6,
    )
    fm_model, fm_report = load_exchange_h5(fm_path)
    with pytest.raises(ValueError, match="spin_magnitudes"):
        screen_magnetic_configuration(fm_model, order=MagneticOrder.FM)
    fm = screen_magnetic_configuration(
        fm_model,
        order=MagneticOrder.FM,
        spin_magnitudes=1.0,
    )
    assert fm.n_magnetic_sites == 3
    assert fm.n_channels == 3
    assert fm.locally_stable
    np.testing.assert_array_equal(fm.bosonic_metric, np.ones(3))
    assert fm_report.supports(ExchangeCapability.FM_ARBITRARY_N)
    assert not fm_report.supports(ExchangeCapability.COLLINEAR_AFM_BIPARTITE)
    with pytest.raises(NotImplementedError, match="exactly two"):
        screen_magnetic_configuration(
            fm_model,
            order=MagneticOrder.COLLINEAR_AFM,
            spin_magnitudes=1.0,
        )
    with pytest.raises(ValueError, match="FM spin_pattern"):
        screen_magnetic_configuration(
            fm_model,
            order=MagneticOrder.FM,
            spin_pattern=[1.0, 1.0, -1.0],
            spin_magnitudes=1.0,
        )

    afm_path = tmp_path / "afm_two_site.h5"
    _two_site_scalar(afm_path, -2.0)
    afm_model, _ = load_exchange_h5(afm_path)
    afm = screen_magnetic_configuration(
        afm_model,
        order=MagneticOrder.COLLINEAR_AFM,
        spin_pattern=[-1.0, 1.0],
        spin_magnitudes=1.0,
    )
    assert afm.n_channels == 4
    assert afm.locally_stable
    np.testing.assert_array_equal(afm.bosonic_metric, [1.0, 1.0, -1.0, -1.0])

    with pytest.raises(ValueError, match="not locally"):
        screen_magnetic_configuration(
            afm_model,
            order=MagneticOrder.FM,
            spin_magnitudes=1.0,
        )
    forced_fm = screen_magnetic_configuration(
        afm_model,
        order=MagneticOrder.FM,
        spin_magnitudes=1.0,
        require_local_stability=False,
    )
    assert not forced_fm.locally_stable


def test_invalid_shapes_and_nonfinite_exchange_are_rejected(tmp_path: Path) -> None:
    wrong_shape = tmp_path / "wrong_shape.h5"
    with h5py.File(wrong_shape, "w") as handle:
        _write_bonds(
            handle,
            [0, 1],
            [1, 0],
            [(0, 0, 0), (0, 0, 0)],
        )
        dataset = handle.create_dataset("J_tensor_r", data=np.ones((2, 2, 2)))
        dataset.attrs["unit"] = "meV"
        dataset.attrs["tensor_axis_order"] = "x,y,z"
    with pytest.raises(ValueError, match="must have shape"):
        load_exchange_h5(wrong_shape)

    nonfinite = tmp_path / "nonfinite.h5"
    _write_scalar(
        nonfinite,
        [np.nan, np.nan],
        atom_i=[0, 1],
        atom_j=[1, 0],
        shifts=[(0, 0, 0), (0, 0, 0)],
    )
    with pytest.raises(ValueError, match="non-finite"):
        load_exchange_h5(nonfinite)


def test_integer_metadata_preserves_large_values_and_rejects_uint64_overflow(
    tmp_path: Path,
) -> None:
    large_atom = 2**53 + 1
    large = tmp_path / "large-index.h5"
    _write_scalar(
        large,
        [1.0, 1.0],
        atom_i=[large_atom, large_atom + 1],
        atom_j=[large_atom + 1, large_atom],
        shifts=[(0, 0, 0), (0, 0, 0)],
    )
    model, _ = load_exchange_h5(large)
    np.testing.assert_array_equal(
        model.magnetic_atom_indices,
        [large_atom, large_atom + 1],
    )

    overflow = tmp_path / "overflow-index.h5"
    with h5py.File(overflow, "w") as handle:
        _write_bonds(handle, [0, 1], [1, 0], [(0, 0, 0), (0, 0, 0)])
        del handle["bonds/mag_i_atom"]
        handle["bonds"].create_dataset(
            "mag_i_atom",
            data=np.asarray([np.iinfo(np.uint64).max, 1], dtype=np.uint64),
        )
        values = handle.create_group("J_r").create_dataset("value", data=[1.0, 1.0])
        values.attrs["unit"] = "meV"
    with pytest.raises(ValueError, match="outside the int64 range"):
        load_exchange_h5(overflow)


def test_tensor_axes_are_explicit_and_reordered_to_xyz(tmp_path: Path) -> None:
    missing = tmp_path / "missing_axes.h5"
    with h5py.File(missing, "w") as handle:
        _write_bonds(handle, [0, 1], [1, 0], [(0, 0, 0), (0, 0, 0)])
        dataset = handle.create_dataset("J_tensor_r", data=np.stack((np.eye(3), np.eye(3))))
        dataset.attrs["unit"] = "meV"
    with pytest.raises(ValueError, match="requires explicit tensor axes"):
        load_exchange_h5(missing)

    permuted = tmp_path / "permuted_axes.h5"
    canonical = np.asarray(
        [[2.0, 0.3, -0.4], [0.1, 3.0, 0.6], [0.7, -0.2, 5.0]],
        dtype=np.float64,
    )
    source_order = (2, 0, 1)  # z,x,y
    stored = canonical[np.ix_(source_order, source_order)]
    with h5py.File(permuted, "w") as handle:
        _write_bonds(handle, [0, 1], [1, 0], [(0, 0, 0), (0, 0, 0)])
        dataset = handle.create_dataset("J_tensor_r", data=np.stack((stored, stored.T)))
        dataset.attrs["unit"] = "meV"
        dataset.attrs["tensor_axis_order"] = "z,x,y"

    model, report = load_exchange_h5(permuted)
    np.testing.assert_allclose(model.tensor_mev, np.stack((canonical, canonical.T)))
    assert any("reordered" in warning for warning in report.warnings)


def test_scalar_lifted_tensor_does_not_gain_full_tensor_capability(
    tmp_path: Path,
) -> None:
    path = tmp_path / "scalar_lift.h5"
    with h5py.File(path, "w") as handle:
        _write_bonds(
            handle,
            [0, 1],
            [1, 0],
            [(0, 0, 0), (0, 0, 0)],
            kernel_family="scalar",
        )
        tensor = handle.create_dataset("J_tensor_r", data=np.stack((2.0 * np.eye(3), 2.0 * np.eye(3))))
        tensor.attrs["unit"] = "meV"
        tensor.attrs["tensor_axis_order"] = "x,y,z"
        handle["basic_data"].create_dataset("tensor_mode", data=np.bytes_("isotropic_from_scalar_collinear"))

    model, report = load_exchange_h5(path)
    assert model.representation is ExchangeRepresentation.ISOTROPIC
    assert report.promoted_isotropic
    assert report.supports(ExchangeCapability.ISOTROPIC_STATIC_EXCHANGE)
    assert not report.supports(ExchangeCapability.FULL_TENSOR_STATIC_EXCHANGE)


def test_missing_convention_requires_explicit_override(tmp_path: Path) -> None:
    path = tmp_path / "old_scalar.h5"
    with h5py.File(path, "w") as handle:
        bonds = handle.create_group("bonds")
        bonds.create_dataset("mag_i_atom", data=[0, 1])
        bonds.create_dataset("mag_j_atom", data=[1, 0])
        bonds.create_dataset("R", data=np.zeros((2, 3), dtype=np.int64))
        values = handle.create_group("J_r").create_dataset("value", data=[1.0, 1.0])
        values.attrs["unit"] = "meV"

    with pytest.raises(ValueError, match="convention metadata is incomplete"):
        load_exchange_h5(path)

    override = ExchangeConvention(
        spin_normalization=ExchangeSpinNormalization.UNIT_VECTOR,
        source_spin_magnitude=1.0,
        kernel_family="scalar_lkag",
    )
    model, report = load_exchange_h5(path, convention=override)
    assert model.convention == override
    assert report.convention_origin == "explicit_override"
    assert report.missing_convention_fields == (
        "basic_data/bond_coverage",
        "basic_data/directed_bond_weight",
        "basic_data/hamiltonian_sign",
        "basic_data/kernel_family",
        "basic_data/realspace_gauge",
        "basic_data/source_spin_magnitude",
        "basic_data/spin_normalization",
    )
    assert any("supplied missing metadata" in item for item in report.warnings)


def test_spin_operator_convention_requires_matching_spin_magnitude(
    tmp_path: Path,
) -> None:
    path = tmp_path / "spin_operator.h5"
    _two_site_scalar(path, 1.0)
    with h5py.File(path, "r+") as handle:
        del handle["basic_data/spin_normalization"]
        handle["basic_data"].create_dataset("spin_normalization", data=np.bytes_("spin_operator"))
        handle["basic_data/source_spin_magnitude"][...] = 2.5

    model, _ = load_exchange_h5(path)
    with pytest.raises(ValueError, match="generated for S=2.5"):
        screen_magnetic_configuration(
            model,
            order=MagneticOrder.FM,
            spin_magnitudes=2.0,
        )
    screened = screen_magnetic_configuration(
        model,
        order=MagneticOrder.FM,
        spin_magnitudes=2.5,
    )
    assert screened.locally_stable


def test_native_exchange_scalar_writer_emits_screenable_convention(
    tmp_path: Path,
) -> None:
    epr = tmp_path / "source_epr.h5"
    with h5py.File(epr, "w"):
        pass
    output = tmp_path / "j.h5"
    args = SimpleNamespace(
        kmesh=(1, 1, 1),
        efermi=0.0,
        hr_unit="ry",
        integrator="contour",
        empoints=4,
        nproc=1,
        epr_up=str(epr),
    )
    pair_meta = [
        {"gi": 2, "gj": 7, "li": 0, "lj": 1, "R": (0, 0, 0), "dist": 1.0, "shell": 1},
        {"gi": 7, "gj": 2, "li": 1, "lj": 0, "R": (0, 0, 0), "dist": 1.0, "shell": 1},
    ]
    _write_h5(
        str(output),
        args,
        ["A", "B"],
        pair_meta,
        np.asarray([3.0, 3.0]),
        [],
        0.0,
        1,
        4,
        {"n_chunks": 1},
    )

    model, report = load_exchange_h5(output)
    assert model.convention.spin_normalization is ExchangeSpinNormalization.UNIT_VECTOR
    assert model.convention.kernel_family == "scalar_lkag"
    assert model.convention.directed_bond_weight == 0.5
    assert report.supports(ExchangeCapability.ISOTROPIC_STATIC_EXCHANGE)


def test_projected_scalar_writer_preserves_raw_values_and_passes_strict_screening(
    tmp_path: Path,
) -> None:
    epr = tmp_path / "source_epr.h5"
    with h5py.File(epr, "w"):
        pass
    output = tmp_path / "projected_j.h5"
    args = SimpleNamespace(
        kmesh=(1, 1, 1),
        efermi=0.0,
        hr_unit="ry",
        integrator="contour",
        empoints=4,
        nproc=1,
        epr_up=str(epr),
    )
    pair_meta = [
        {
            "gi": 2,
            "gj": 7,
            "li": 0,
            "lj": 1,
            "R": (0, 0, 0),
            "dist": 1.0,
            "shell": 1,
        },
        {
            "gi": 7,
            "gj": 2,
            "li": 1,
            "lj": 0,
            "R": (0, 0, 0),
            "dist": 1.0,
            "shell": 1,
        },
    ]
    orbit = [
        {"i": 2, "j": 7, "R": (0, 0, 0), "distance": 1.0, "shell_idx": 1},
        {"i": 7, "j": 2, "R": (0, 0, 0), "distance": 1.0, "shell_idx": 1},
    ]
    grouping = _OrbitGrouping(
        [orbit],
        source="spglib",
        spacegroup="test #1",
        n_operations=2,
        species_source="test",
    )
    raw = np.asarray([3.0, 3.0 + 2.0e-7])
    symmetry = _apply_orbit_symmetry(
        pair_meta,
        raw,
        grouping,
        policy="project",
    )
    _write_h5(
        str(output),
        args,
        ["A", "B"],
        pair_meta,
        symmetry.values_mev,
        grouping.orbits,
        0.0,
        1,
        4,
        {"n_chunks": 1},
        raw_j_mev=symmetry.raw_values_mev,
        orbit_symmetry=symmetry,
        orbit_grouping=grouping,
    )

    with h5py.File(output) as handle:
        np.testing.assert_array_equal(handle["J_r/value_raw"], raw)
        canonical = np.asarray(handle["J_r/value"])
        assert canonical[0] == canonical[1]
        np.testing.assert_allclose(canonical, [3.0 + 1.0e-7] * 2)
        np.testing.assert_array_equal(handle["J_r/J_r_b1"][()], handle["J_r/J_r_b2"][()])
        assert handle["basic_data/orbit_symmetry_policy"].asstr()[()] == "project"
        assert bool(handle["basic_data/orbit_symmetry_applied"][()])
        assert handle["symmetry/orbit_label"].asstr()[0] == "1a"
        assert handle["symmetry/grouping_source"].asstr()[()] == "spglib"

    model, report = load_exchange_h5(output, reciprocity_atol_mev=0.0, rtol=0.0)
    assert report.max_reciprocity_error_mev == 0.0
    assert model.isotropic_mev[0] == model.isotropic_mev[1]
    np.testing.assert_allclose(model.isotropic_mev, [3.0 + 1.0e-7] * 2)


@pytest.mark.parametrize(
    ("kernel", "all_bonds", "expected_coverage", "expected_weight"),
    [
        ("direct", True, "directed_mate_complete", 0.5),
        ("direct", False, "canonical_half", 1.0),
        ("tb2j", True, "directed_mate_complete", 1.0),
        ("tb2j", False, "canonical_half", 2.0),
    ],
)
def test_epr_tensor_writer_records_audited_bond_weight(
    tmp_path: Path,
    kernel: str,
    all_bonds: bool,
    expected_coverage: str,
    expected_weight: float,
) -> None:
    epr = tmp_path / "source.h5"
    with h5py.File(epr, "w"):
        pass
    output = tmp_path / f"epr_{kernel}_{all_bonds}.h5"
    args = SimpleNamespace(
        kmesh=(1, 1, 1),
        efermi=0.0,
        spin_direction=(0.0, 0.0, 1.0),
        lambda_te=0.0,
        hr_unit="ry",
        integrator="contour",
        kernel=kernel,
        spin_magnitude=1.0,
        all_bonds=all_bonds,
        p_order="px,py,pz",
        d_order="dz2,dxz,dyz,dx2-y2,dxy",
        soc="",
        soc_element="",
        empoints=1,
        nproc=1,
        epr_up=str(epr),
    )
    pair_meta = [
        {"gi": 0, "gj": 1, "li": 0, "lj": 1, "R": (0, 0, 0), "dist": 1.0, "shell": 1},
        {"gi": 1, "gj": 0, "li": 1, "lj": 0, "R": (0, 0, 0), "dist": 1.0, "shell": 1},
    ]
    _write_epr_tensor_h5(
        str(output),
        args,
        ["A", "B"],
        pair_meta,
        np.stack((np.eye(3), np.eye(3))),
        np.zeros(2, dtype=np.complex128),
        [],
        0.0,
        1,
        1,
        {},
        ("x", "y", "z"),
    )

    with h5py.File(output, "r") as handle:
        assert handle["basic_data/bond_coverage"].asstr()[()] == expected_coverage
        assert float(handle["basic_data/directed_bond_weight"][()]) == expected_weight
    if kernel == "tb2j" and all_bonds:
        with pytest.raises(ValueError, match="directed_bond_weight=0.5"):
            load_exchange_h5(output)


@pytest.mark.parametrize(
    ("kernel", "all_bonds", "expected_family", "expected_weight"),
    [
        ("scalar", True, "scalar_lkag", 0.5),
        ("scalar", False, "scalar_lkag", 1.0),
        ("direct", True, "direct", 0.5),
        ("tb2j", True, "tb2j", 1.0),
        ("tb2j", False, "tb2j", 2.0),
    ],
)
def test_wannier_tensor_writer_records_canonical_kernel_and_weight(
    tmp_path: Path,
    kernel: str,
    all_bonds: bool,
    expected_family: str,
    expected_weight: float,
) -> None:
    output = tmp_path / f"wannier_{kernel}_{all_bonds}.h5"
    args = SimpleNamespace(
        kmesh=(1, 1, 1),
        efermi=0.0,
        apply_degeneracy=True,
        spin_direction=(0.0, 0.0, 1.0),
        kernel=kernel,
        integrator="contour",
        hr_unit="eV",
        up_hr="up_hr.dat",
        dn_hr="dn_hr.dat",
        spinor_hr=None,
        spinor_basis_order="wannier_spin",
        centres=None,
        mag_subspace="",
        win=None,
        soc="",
        ref_epr_up=None,
        ref_epr_dn=None,
        spin_magnitude=1.0,
        all_bonds=all_bonds,
    )
    pair_meta = [
        {"gi": 0, "gj": 1, "li": 0, "lj": 1, "R": (0, 0, 0), "dist": 1.0, "shell": 1},
        {"gi": 1, "gj": 0, "li": 1, "lj": 0, "R": (0, 0, 0), "dist": 1.0, "shell": 1},
    ]
    _write_wannier_tensor_h5(
        str(output),
        args,
        ["A", "B"],
        pair_meta,
        np.stack((np.eye(3), np.eye(3))),
        np.zeros(2, dtype=np.complex128),
        0.0,
        1,
        1,
        ("x", "y", "z"),
    )

    with h5py.File(output, "r") as handle:
        assert handle["basic_data/kernel_family"].asstr()[()] == expected_family
        assert float(handle["basic_data/directed_bond_weight"][()]) == expected_weight
    if kernel == "scalar" and all_bonds:
        model, report = load_exchange_h5(output)
        assert model.convention.kernel_family == "scalar_lkag"
        assert report.supports(ExchangeCapability.ISOTROPIC_STATIC_EXCHANGE)


def test_screening_tolerances_must_be_finite(tmp_path: Path) -> None:
    path = tmp_path / "scalar.h5"
    _two_site_scalar(path, 1.0)
    for tolerance in (np.nan, np.inf, -1.0):
        with pytest.raises(ValueError, match="finite and non-negative"):
            load_exchange_h5(path, reciprocity_atol_mev=tolerance)

    model, _ = load_exchange_h5(path)
    with pytest.raises(ValueError, match="finite and non-negative"):
        screen_magnetic_configuration(
            model,
            order=MagneticOrder.FM,
            spin_magnitudes=1.0,
            torque_atol_mev=np.inf,
        )
