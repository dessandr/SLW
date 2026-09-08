from __future__ import annotations

from dataclasses import replace

import h5py
import numpy as np
import pytest
from scipy.linalg import expm

from slw.wtorque.cli import _build_vertices, inspect_config
from slw.wtorque.config import BlochGauge, ProjectorPolicy
from slw.wtorque.green.real_axis import RealAxisIntegrator
from slw.wtorque.io.dfpt import HDF5DFPTProvider
from slw.wtorque.parallel.benchmark import benchmark_q
from slw.wtorque.pipeline import run_compute_kernel

from .test_io_pipeline import _config, _write_inputs


SX = np.array([[0, 1], [1, 0]], dtype=np.complex128)
SY = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
SZ = np.diag([1.0, -1.0]).astype(np.complex128)
ID = np.eye(2, dtype=np.complex128)


def _direct_config(tmp_path):
    electron_path, dfpt_path = _write_inputs(tmp_path)
    config = _config(tmp_path, electron_path, dfpt_path)
    return replace(
        config,
        dfpt=replace(config.dfpt, g_xc_dataset="exchange_response"),
        kernel=replace(config.kernel, include_direct_vertex=True),
        integration=replace(
            config.integration, energy_points=240, temperature_K=350.0,
            energy_min_eV=-1.4, energy_max_eV=0.4, eta_eV=0.08,
        ),
    )


def test_strict_direct_total_matches_finite_difference_of_free_energy(tmp_path):
    config = _direct_config(tmp_path)
    h_trs = -0.1 * ID + 0.09 * SX
    h_xc = 0.35 * SZ
    g_xc = np.stack((0.12 * SX + 0.07 * SZ, 0.08 * SY, 0.05 * SZ))
    g_fixed = np.stack((0.03 * ID, 0.04 * SX, 0.02 * ID))
    with h5py.File(config.electrons.file, "r+") as handle:
        handle["electrons/H_R"][0] = h_trs + h_xc
        handle["electrons/H_TRS_R"][0] = h_trs
    with h5py.File(config.dfpt.file, "r+") as handle:
        group = handle["dfpt/q_000000"]
        group["g_cart"][0] = g_xc + g_fixed
        group.create_dataset("exchange_response", data=g_xc[None])
        # Selecting the configured dataset must win over this conventional name.
        group.create_dataset("g_xc_cart", data=np.zeros_like(g_xc[None]))

    run_compute_kernel(config)
    integration = config.integration
    integrator = RealAxisIntegrator.gauss_legendre(
        integration.energy_min_eV, integration.energy_max_eV, integration.energy_points,
        chemical_potential_eV=0.0, temperature_K=integration.temperature_K,
    )

    def grand_potential(angle, displacement, axis, ipert):
        rotation = expm(-0.5j * angle * axis)
        h = h_trs + displacement * g_fixed[ipert]
        h = h + rotation @ (h_xc + displacement * g_xc[ipert]) @ rotation.conj().T
        energies = np.linalg.eigvalsh(h)
        log_trace = np.log(
            integrator.nodes_eV[:, None] + 1j * integration.eta_eV - energies
        ).sum(axis=1)
        return np.dot(integrator.weighted_occupations(), log_trace.imag) / np.pi

    step = 1.0e-4
    expected = np.zeros((2, 3))
    for component, axis in enumerate((SY, -SX)):
        for ipert in range(3):
            expected[component, ipert] = (
                grand_potential(step, step, axis, ipert)
                - grand_potential(step, -step, axis, ipert)
                - grand_potential(-step, step, axis, ipert)
                + grand_potential(-step, -step, axis, ipert)
            ) / (4 * step**2)
    with h5py.File(config.output.file) as handle:
        total = handle["kernel/K_pi_u_total"][0, 0, :, 0, :]
        direct = handle["kernel/K_pi_u_direct"][...]
        assert np.linalg.norm(direct) > 0.02
        np.testing.assert_allclose(total, expected, atol=4.0e-8, rtol=2.0e-6)
        for base in ("K_pi_u", "A_retarded"):
            np.testing.assert_allclose(
                handle[f"kernel/{base}_total"][...],
                handle[f"kernel/{base}_bubble"][...] + handle[f"kernel/{base}_direct"][...],
                atol=1.0e-14,
            )
            np.testing.assert_array_equal(handle[f"kernel/{base}"][...], handle[f"kernel/{base}_total"][...])
        assert bool(handle["validation/direct_term_enabled"][0])
    assert run_compute_kernel(config).computed_q_indices == ()


@pytest.mark.parametrize("problem", ["missing", "wrong_shape", "nonfinite", "total_alias"])
def test_strict_direct_rejects_invalid_exchange_response(tmp_path, problem):
    config = _direct_config(tmp_path)
    with h5py.File(config.dfpt.file, "r+") as handle:
        group = handle["dfpt/q_000000"]
        if problem == "wrong_shape":
            group.create_dataset("exchange_response", data=np.zeros((1, 2, 2, 2), dtype=np.complex128))
        elif problem == "nonfinite":
            values = np.zeros((1, 3, 2, 2), dtype=np.complex128)
            values[0, 0, 0, 0] = np.nan
            group.create_dataset("exchange_response", data=values)
        elif problem == "total_alias":
            group["exchange_response"] = group["g_cart"]
    with pytest.raises(RuntimeError, match="g_XC"):
        run_compute_kernel(config)


def test_strict_direct_keeps_moving_projector_gate_explicit(tmp_path):
    config = _direct_config(tmp_path)
    config = replace(config, kernel=replace(config.kernel, projector_policy=ProjectorPolicy.MOVING))
    with pytest.raises(NotImplementedError, match="moving-projector"):
        run_compute_kernel(config)
    assert not config.output.file.exists()


def _write_gauge_fixture(tmp_path, gauge, wrapped):
    tmp_path.mkdir()
    config = _direct_config(tmp_path)
    centers = np.array([[0.17, 0.0, 0.0], [0.42, 0.0, 0.0]])
    sites = np.array([[0.12, 0.0, 0.0]])
    kpoints = np.array([[0.0, 0.0, 0.0], [1 / 3, 0.0, 0.0], [2 / 3, 0.0, 0.0]])
    qpoints = np.array([[1 / 3, 0.0, 0.0], [-1 / 3, 0.0, 0.0]])
    h_xc = np.kron(np.array([[0.4, 0.06], [0.06, -0.23]]), SZ)
    h_trs = np.kron(np.array([[-0.1, 0.11], [0.11, 0.04]]), ID)
    h_trs = h_trs + 0.03 * np.kron(np.array([[1.0, 0.0], [0.0, -1.0]]), SX)
    with h5py.File(config.electrons.file, "r+") as handle:
        group = handle["electrons"]
        group.attrs["bloch_gauge"] = gauge.value
        arrays = {
            "H_R": (h_trs + h_xc)[None], "H_TRS_R": h_trs[None], "H_XC_R": h_xc[None],
            "orbital_centers": centers, "orbital_site": np.array([0, 1], dtype=np.int32),
            "orbital_labels": np.array(["d", "p"], dtype=h5py.string_dtype("utf-8")),
            "kpoints": kpoints, "weights": np.full(3, 1 / 3),
        }
        for name, values in arrays.items():
            del group[name]
            group.create_dataset(name, data=values)
        handle["spin/magnetic_site_position"][...] = sites
        del handle["spin/magnetic_orbital_mask"]
        handle["spin"].create_dataset("magnetic_orbital_mask", data=np.array([[True, False]]))

    rng = np.random.default_rng(7834)
    forward_xc = rng.normal(size=(3, 3, 4, 4)) + 1j * rng.normal(size=(3, 3, 4, 4))
    forward_fixed = rng.normal(size=(3, 3, 4, 4)) + 1j * rng.normal(size=(3, 3, 4, 4))

    def phases(points):
        return np.exp(2j * np.pi * points @ np.repeat(centers, 2, axis=0).T)

    with h5py.File(config.dfpt.file, "r+") as handle:
        group = handle["dfpt"]
        del group["qpoints"]
        group.create_dataset("qpoints", data=qpoints)
        del group["q_000000"]
        for iq, q in enumerate(qpoints):
            qgroup = group.create_group(f"q_{iq:06d}")
            for name, forward in (("g_cart", forward_xc + forward_fixed), ("exchange_response", forward_xc)):
                # Enforce the physical reverse link g(k+q,-q)=g(k,q)^dagger.
                values = forward if iq == 0 else np.swapaxes(forward[[2, 0, 1]].conj(), -1, -2)
                if gauge is BlochGauge.ATOMIC_POSITION:
                    final = kpoints + q
                    if wrapped:
                        final = final % 1.0
                    values = phases(final).conj()[:, None, :, None] * values * phases(kpoints)[:, None, None, :]
                qgroup.create_dataset(name, data=values.astype(np.complex128))
    return replace(
        config,
        electrons=replace(config.electrons, bloch_gauge=gauge),
        dfpt=replace(config.dfpt, final_state_representation="wrapped" if wrapped else "unwrapped"),
        integration=replace(config.integration, energy_points=80),
        performance=replace(config.performance, perturbation_chunk=1),
    )


def test_strict_finite_q_direct_is_invariant_under_bloch_gauge_and_wrap(tmp_path):
    outputs = []
    vertex_outputs = []
    for index, (gauge, wrapped) in enumerate((
        (BlochGauge.ATOMIC_POSITION, False),
        (BlochGauge.ATOMIC_POSITION, True),
        (BlochGauge.CELL_PERIODIC, False),
    )):
        config = _write_gauge_fixture(tmp_path / str(index), gauge, wrapped)
        inspection = inspect_config(config)
        assert inspection["conventions"]["direct_term_enabled"]
        assert inspection["conventions"]["g_xc_dataset"] == "exchange_response"
        with h5py.File(_build_vertices(config, None)) as handle:
            vertex_outputs.append(handle["q_000000/vertex"][...])
        run_compute_kernel(config)
        with h5py.File(config.output.file) as handle:
            outputs.append({key: handle[f"kernel/{key}"][...] for key in (
                "K_pi_u_bubble", "K_pi_u_direct", "K_pi_u_total",
                "A_retarded_bubble", "A_retarded_direct",
            )})
    for output in outputs[1:]:
        for key in outputs[0]:
            np.testing.assert_allclose(output[key], outputs[0][key], atol=3.0e-13, rtol=2.0e-12)
    for key in ("K_pi_u_bubble", "K_pi_u_direct", "K_pi_u_total"):
        np.testing.assert_allclose(outputs[0][key][1], outputs[0][key][0].conj(), atol=1.0e-14)
    kpoints = np.array([[0.0, 0.0, 0.0], [1 / 3, 0.0, 0.0], [2 / 3, 0.0, 0.0]])
    centers = np.repeat(np.array([[0.17, 0.0, 0.0], [0.42, 0.0, 0.0]]), 2, axis=0)
    source_phase = np.exp(2j * np.pi * kpoints @ centers.T)
    final_phase = np.exp(2j * np.pi * (kpoints + np.array([1 / 3, 0.0, 0.0])) @ centers.T)
    expected_atomic = (
        final_phase.conj()[:, None, None, :, None]
        * vertex_outputs[2] * source_phase[:, None, None, None, :]
    )
    np.testing.assert_allclose(vertex_outputs[0], expected_atomic, atol=2.0e-15)
    benchmark = benchmark_q(config, 0, repeats=1)
    assert benchmark.agrees
    assert benchmark.as_dict()["kernel_component"] == "bubble"


def test_provider_absolute_q_template_and_disabled_direct(tmp_path):
    config = _direct_config(tmp_path)
    with h5py.File(config.dfpt.file, "r+") as handle:
        values = np.stack((SX, SY, SZ))[None]
        handle["dfpt/q_000000"].create_dataset("exchange_response", data=values)
    with HDF5DFPTProvider(
        config.dfpt.file, normalization=config.dfpt.normalization,
        spinor_lift=config.dfpt.spinor_lift, spin_order=config.electrons.spin_order,
        norb=1, g_xc_dataset="/dfpt/q_{iq:06d}/exchange_response",
    ) as provider:
        np.testing.assert_array_equal(provider.g_xc(0), values)
    config = replace(config, kernel=replace(config.kernel, include_direct_vertex=False))
    run_compute_kernel(config)
    with h5py.File(config.output.file) as handle:
        np.testing.assert_array_equal(handle["kernel/K_pi_u_direct"][...], 0.0)
        np.testing.assert_array_equal(handle["kernel/K_pi_u"][...], handle["kernel/K_pi_u_bubble"][...])
        assert not bool(handle["validation/direct_term_enabled"][0])


def test_derivative_only_response_can_be_read_without_total_dfpt_input(tmp_path):
    config = _direct_config(tmp_path)
    values = np.stack((SX, SY, SZ))[None]
    with h5py.File(config.dfpt.file, "r+") as handle:
        group = handle["dfpt/q_000000"]
        group.create_dataset("exchange_response", data=values)
        del group["g_cart"]
        handle["dfpt"].create_dataset("kpoints", data=np.zeros((1, 3)))
    with HDF5DFPTProvider(
        config.dfpt.file, normalization=config.dfpt.normalization,
        spinor_lift=config.dfpt.spinor_lift, spin_order=config.electrons.spin_order,
        norb=1, g_xc_dataset="exchange_response",
    ) as provider:
        np.testing.assert_array_equal(provider.g_xc(0), values)
    with pytest.raises(KeyError, match="g_cart"):
        run_compute_kernel(config)
    assert not config.output.file.exists()


def test_cli_inspection_rejects_missing_configured_exchange_dataset(tmp_path):
    config = _direct_config(tmp_path)
    with pytest.raises(ValueError, match="g_XC"):
        inspect_config(config)
