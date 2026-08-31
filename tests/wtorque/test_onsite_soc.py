from __future__ import annotations

import h5py
import numpy as np

from slw.soc import atomic_d_soc_block, reorder_spinor_matrix
from slw.wtorque.config import RunConfig
from slw.wtorque.green.provider import green_matrix
from slw.wtorque.io.exchange import load_exchange_field
from slw.wtorque.io.wannier import load_spinor_wannier
from slw.wtorque.pipeline import run_compute_kernel
from slw.wtorque.provenance import build_run_manifest
from slw.wtorque.torque.kernel import contract_bubble_batch
from slw.wtorque.torque.vertices import transverse_vertices
from slw.wtorque.validation.reports import validate_run

from .test_config_and_manifest import _config


def _write_fe_win(path) -> None:
    path.write_text(
        """
begin atoms_frac
Fe1 0.0 0.0 0.0
end atoms_frac
begin projections
Fe:d
end projections
""",
        encoding="utf-8",
    )


def _write_collinear_d_model(path) -> tuple[np.ndarray, np.ndarray]:
    identity = np.eye(2, dtype=np.complex128)
    sigma_z = np.diag([1.0, -1.0]).astype(np.complex128)
    h_trs = np.kron(np.diag([-0.7, -0.3, 0.0, 0.25, 0.55]), identity)
    h_xc = np.kron(np.diag([0.35, 0.31, 0.29, 0.33, 0.27]), sigma_z)
    with h5py.File(path, "w") as handle:
        electrons = handle.create_group("electrons")
        electrons.attrs["bloch_gauge"] = "atomic_position"
        electrons.create_dataset("lattice", data=np.eye(3, dtype=np.float64))
        electrons.create_dataset(
            "R_vectors", data=np.zeros((1, 3), dtype=np.int32)
        )
        electrons.create_dataset("H_R", data=(h_trs + h_xc)[None])
        electrons.create_dataset("H_TRS_R", data=h_trs[None])
        electrons.create_dataset("H_XC_R", data=h_xc[None])
        electrons.create_dataset(
            "orbital_centers", data=np.zeros((5, 3), dtype=np.float64)
        )
        electrons.create_dataset(
            "orbital_site", data=np.zeros(5, dtype=np.int32)
        )
        electrons.create_dataset(
            "orbital_labels",
            data=np.asarray(
                ["dz2", "dxz", "dyz", "dx2-y2", "dxy"],
                dtype=h5py.string_dtype("utf-8"),
            ),
        )
        electrons.create_dataset(
            "kpoints", data=np.zeros((1, 3), dtype=np.float64)
        )
        electrons.create_dataset("weights", data=np.ones(1, dtype=np.float64))
        electrons.create_dataset("fermi_energy", data=np.float64(0.0))
        spin = handle.create_group("spin")
        spin.create_dataset("magnetic_atom_index", data=np.array([0], dtype=np.int32))
        spin.create_dataset(
            "magnetic_site_position", data=np.zeros((1, 3), dtype=np.float64)
        )
        spin.create_dataset(
            "magnetic_orbital_mask", data=np.ones((1, 5), dtype=bool)
        )
        spin.create_dataset("local_frames", data=np.eye(3, dtype=np.float64)[None])
        spin.create_dataset("spin_length", data=np.array([1.0], dtype=np.float64))
    return h_trs, h_xc


def test_input_onsite_soc_is_added_only_to_full_and_trs_hamiltonians(tmp_path):
    electrons = tmp_path / "electrons.h5"
    dfpt = tmp_path / "dfpt.h5"
    win = tmp_path / "Fe.win"
    h_trs, h_xc = _write_collinear_d_model(electrons)
    dfpt.write_bytes(b"not opened by this loader test")
    _write_fe_win(win)
    mapping = _config(str(electrons), str(dfpt), str(tmp_path / "out.h5"))
    mapping["electrons"]["onsite_soc"] = {
        "win": str(win),
        "scale": 1.0,
        "entries": [{"selector": "Fe-d", "lambda_ev": 0.06}],
    }
    config = RunConfig.from_mapping(mapping, base_dir=tmp_path)
    model = load_spinor_wannier(
        config.electrons.file,
        bloch_gauge=config.electrons.bloch_gauge,
        spin_order=config.electrons.spin_order,
        onsite_soc=config.electrons.onsite_soc,
    )
    exchange = load_exchange_field(
        config.electrons.file,
        model,
        route=config.electrons.exchange_extraction,
        spin_order=config.electrons.spin_order,
    )

    expected_soc = reorder_spinor_matrix(
        atomic_d_soc_block(0.06), source="spin", target="orbital"
    )
    np.testing.assert_allclose(model.onsite_soc_matrix, expected_soc)
    np.testing.assert_allclose(model.H_R[0], h_trs + h_xc + expected_soc)
    np.testing.assert_allclose(exchange.h_trs_r[0], h_trs + expected_soc)
    np.testing.assert_allclose(exchange.h_xc_r[0], h_xc)
    assert exchange.reconstruction_residual < 1.0e-14
    i_sigma_y = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.complex128)
    sewing = np.kron(np.eye(5, dtype=np.complex128), i_sigma_y)
    np.testing.assert_allclose(
        sewing @ expected_soc.conj() @ sewing.conj().T,
        expected_soc,
        atol=1.0e-14,
    )
    assert "onsite_soc_win" in build_run_manifest(config)["source_hashes"]


def test_soc_off_transverse_bubble_vanishes_and_soc_scaling_changes_propagation():
    identity = np.eye(2, dtype=np.complex128)
    sigma_z = np.diag([1.0, -1.0]).astype(np.complex128)
    h_trs = np.kron(np.diag([-0.7, -0.3, 0.0, 0.25, 0.55]), identity)
    h_xc = np.kron(np.diag([0.35, 0.31, 0.29, 0.33, 0.27]), sigma_z)
    onsite_soc = reorder_spinor_matrix(
        atomic_d_soc_block(0.06), source="spin", target="orbital"
    )
    vertices = transverse_vertices(h_xc, np.eye(3, dtype=np.float64))

    generator = np.random.default_rng(7)
    raw = generator.normal(size=(5, 5)) + 1j * generator.normal(size=(5, 5))
    orbital_perturbation = 0.5 * (raw + raw.conj().T)
    perturbation = np.kron(orbital_perturbation, identity).astype(np.complex128)

    responses = []
    for scale in (0.0, 0.5, 1.0):
        green = green_matrix(h_trs + h_xc + scale * onsite_soc, 0.2 + 0.04j)
        responses.append(
            contract_bubble_batch(
                vertices[None, ...],
                perturbation[None, None, ...],
                green[None, ...],
                green[None, ...],
                np.ones(1, dtype=np.float64),
            )
        )

    np.testing.assert_allclose(responses[0], 0.0, atol=1.0e-14)
    assert np.linalg.norm(responses[1]) > 1.0e-6
    assert np.linalg.norm(responses[2] - responses[1]) > 1.0e-6
    np.testing.assert_allclose(
        transverse_vertices(h_xc, np.eye(3, dtype=np.float64)), vertices
    )


def test_onsite_soc_pipeline_reports_trs_and_exchange_isolation_gates(tmp_path):
    electrons = tmp_path / "electrons.h5"
    dfpt = tmp_path / "dfpt.h5"
    win = tmp_path / "Fe.win"
    _write_collinear_d_model(electrons)
    _write_fe_win(win)
    with h5py.File(dfpt, "w") as handle:
        group = handle.create_group("dfpt")
        group.attrs["normalization"] = "cartesian_derivative"
        group.create_dataset("qpoints", data=np.zeros((1, 3), dtype=np.float64))
        group.create_dataset("pert_atom", data=np.zeros(3, dtype=np.int32))
        group.create_dataset("pert_cart", data=np.arange(3, dtype=np.int32))
        qgroup = group.create_group("q_000000")
        qgroup.create_dataset(
            "g_cart", data=np.zeros((1, 3, 5, 5), dtype=np.complex128)
        )
    mapping = _config(str(electrons), str(dfpt), str(tmp_path / "out.h5"))
    mapping["electrons"]["onsite_soc"] = {
        "win": str(win),
        "entries": [{"selector": "Fe-d", "lambda_ev": 0.06}],
    }
    mapping["magnetic_subspace"]["sites"] = [
        {"atom": 0, "orbitals": [0, 1, 2, 3, 4]}
    ]
    mapping["dfpt"]["spinor_lift"] = "spin_scalar"
    mapping["integration"].update(
        {"energy_min_eV": -2.0, "energy_max_eV": 0.0, "energy_points": 16}
    )
    config = RunConfig.from_mapping(mapping, base_dir=tmp_path)

    run_compute_kernel(config)
    report = validate_run(config)
    gates = {gate.name: gate for gate in report.gates}
    assert gates["onsite SOC Hermiticity"].status == "pass"
    assert gates["onsite SOC time reversal"].status == "pass"
    assert gates["SOC isolation from exchange"].status == "pass"
    assert report.valid
