from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest

from slw.magph.model import (
    ExchangeConvention,
    ExchangeSpinNormalization,
    MagneticOrder,
    SingleIonAnisotropy,
)
from slw.magph.screening import load_exchange_h5, screen_magnetic_configuration


def _screened_model(path: Path):
    with h5py.File(path, "w") as handle:
        basic = handle.create_group("basic_data")
        for name, value in {
            "hamiltonian_sign": "minus",
            "spin_normalization": "unit_vector",
            "kernel_family": "scalar_lkag",
            "bond_coverage": "directed_mate_complete",
            "realspace_gauge": "i_at_0_j_at_R",
        }.items():
            basic.create_dataset(name, data=np.bytes_(value))
        basic.create_dataset("source_spin_magnitude", data=1.0)
        basic.create_dataset("directed_bond_weight", data=0.5)
        bonds = handle.create_group("bonds")
        bonds.create_dataset("mag_i_atom", data=[3, 8])
        bonds.create_dataset("mag_j_atom", data=[8, 3])
        bonds.create_dataset("R", data=np.zeros((2, 3), dtype=np.int64))
        values = handle.create_group("J_r").create_dataset("value", data=[1.0, 1.0])
        values.attrs["unit"] = "meV"
    return load_exchange_h5(path)[0]


def test_exchange_convention_is_fail_closed() -> None:
    base = {
        "spin_normalization": ExchangeSpinNormalization.UNIT_VECTOR,
        "source_spin_magnitude": 1.0,
        "kernel_family": "scalar_lkag",
    }
    convention = ExchangeConvention(**base)
    assert convention.directed_bond_weight == 0.5
    with pytest.raises(ValueError, match="hamiltonian_sign"):
        ExchangeConvention(**base, hamiltonian_sign="plus")
    with pytest.raises(ValueError, match="bond_coverage"):
        ExchangeConvention(**base, bond_coverage="canonical_half")
    with pytest.raises(ValueError, match="directed_bond_weight"):
        ExchangeConvention(**base, directed_bond_weight=1.0)
    with pytest.raises(ValueError, match="source_spin_magnitude"):
        ExchangeConvention(**{**base, "source_spin_magnitude": np.nan})
    assert ExchangeConvention(**{**base, "kernel_family": "scalar"}).kernel_family == (
        "scalar_lkag"
    )
    with pytest.raises(ValueError, match="audited TB2J payloads"):
        ExchangeConvention(**{**base, "kernel_family": "tb2j"})


def test_exchange_model_rejects_lossy_integer_casts_and_bad_site_mapping(
    tmp_path: Path,
) -> None:
    model = _screened_model(tmp_path / "j.h5")
    with pytest.raises(ValueError, match="finite integers"):
        replace(model, bond_i=np.asarray([0.5, 1.0]))
    with pytest.raises(ValueError, match="inconsistent with global"):
        replace(model, bond_i_atom=np.asarray([8, 8]))
    with pytest.raises(ValueError, match="involution"):
        replace(model, mirror_index=np.asarray([1, 1]))


def test_exchange_model_direct_constructor_enforces_canonical_payload(
    tmp_path: Path,
) -> None:
    model = _screened_model(tmp_path / "canonical.h5")

    anisotropic = np.array(model.tensor_mev, copy=True)
    anisotropic[0, 0, 1] = 0.25
    with pytest.raises(ValueError, match=r"isotropic_mev\*I\(3\)"):
        replace(model, tensor_mev=anisotropic)

    unequal_scalar = np.asarray([1.0, 2.0])
    unequal_tensor = unequal_scalar[:, None, None] * np.eye(3)
    with pytest.raises(ValueError, match="mate-reciprocal"):
        replace(
            model,
            isotropic_mev=unequal_scalar,
            tensor_mev=unequal_tensor,
        )

    with pytest.raises(ValueError, match="does not cover all referenced global atoms"):
        replace(model, tau_frac=np.zeros((8, 3)))
    with pytest.raises(ValueError, match="geometry arrays disagree"):
        replace(
            model,
            tau_frac=np.zeros((9, 3)),
            tau_cart_ang=np.zeros((10, 3)),
        )


def test_magnetic_configuration_constructor_enforces_semantic_invariants(
    tmp_path: Path,
) -> None:
    model = _screened_model(tmp_path / "j.h5")
    state = screen_magnetic_configuration(
        model,
        order=MagneticOrder.FM,
        spin_magnitudes=1.0,
    )
    for array in (
        state.spin_pattern,
        state.spin_magnitudes,
        state.quantization_axis,
        state.spin_directions,
        state.local_exchange_field_mev,
        state.longitudinal_stiffness_mev,
        state.torque_mev,
        state.bosonic_metric,
    ):
        assert not array.flags.writeable

    with pytest.raises(ValueError, match=r"exactly \+/-1"):
        replace(state, spin_pattern=np.asarray([1.0, 0.0]))
    with pytest.raises(ValueError, match="strictly positive"):
        replace(state, spin_magnitudes=np.asarray([1.0, 0.0]))
    with pytest.raises(ValueError, match="normalized"):
        replace(state, quantization_axis=np.asarray([0.0, 0.0, 2.0]))
    with pytest.raises(ValueError, match=r"spin_pattern\*quantization_axis"):
        replace(state, spin_directions=np.ones((2, 3)))
    with pytest.raises(ValueError, match="bosonic_metric"):
        replace(state, bosonic_metric=np.asarray([1.0, -1.0]))
    with pytest.raises(TypeError, match="boolean"):
        replace(state, locally_stable=1)


def test_single_ion_anisotropy_is_normalized_and_immutable() -> None:
    anisotropy = SingleIonAnisotropy(
        energy_mev=(0.1, -0.2),
        axis=((0.0, 0.0, 2.0), (0.0, 3.0, 0.0)),
        spin_normalization="unit_vector",
    )
    np.testing.assert_allclose(
        anisotropy.axis,
        ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    )
    assert anisotropy.spin_normalization is ExchangeSpinNormalization.UNIT_VECTOR
    assert not anisotropy.energy_mev.flags.writeable
    assert not anisotropy.axis.flags.writeable
    with pytest.raises(ValueError, match="nonzero"):
        replace(anisotropy, axis=np.zeros((2, 3)))
    with pytest.raises(ValueError, match="uniaxial"):
        replace(anisotropy, model="cubic")
