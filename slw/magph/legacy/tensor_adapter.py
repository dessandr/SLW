"""Adapters for real-space exchange-tensor magnon-phonon prototype inputs.

The first prototype stage reads real-space dJ(R, Rp) payloads and turns them
into flat vertex lists:

    (i, j, R, kappa, mu, Rp, component)

No Fourier phases are applied in this module.
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from typing import Dict, Any, Iterable

import h5py
import numpy as np


def _decode_str_array(arr):
    out = []
    for x in np.asarray(arr).tolist():
        if isinstance(x, bytes):
            out.append(x.decode("utf-8"))
        else:
            out.append(str(x))
    return out


def _atom_label(labels, atom_index: int) -> str:
    if 0 <= atom_index < len(labels):
        label = str(labels[atom_index])
        if label:
            return label
    return f"Atom{atom_index + 1}"


def _require(h5, names, path):
    missing = [name for name in names if name not in h5]
    if missing:
        raise KeyError(f"Missing required datasets/groups in {path}: {missing}")


def _scan_target_bond_group(grp) -> Dict[tuple[int, int], str]:
    out = {}
    for key in grp.keys():
        m = re.fullmatch(r"m(\d+)_b(\d+)", str(key))
        if m is None:
            continue
        target = int(m.group(1)) - 1
        bond = int(m.group(2)) - 1
        out[(target, bond)] = str(key)
    return out


COMPONENT_SPECS = {
    "dmi": ("dDMI_r", (3,), "dDMI"),
    "iso": ("dJ_iso_r", (), "dJ_iso"),
    "aniso": ("dJ_gamma_r", (3, 3), "dJ_aniso"),
}


@dataclass
class ExchangeTensorRealSpacePayload:
    source: str
    atom_labels: list[str]
    qmesh: np.ndarray
    target_atoms: np.ndarray
    disp_axes: list[str]
    rp_grid: np.ndarray
    bond_i: np.ndarray
    bond_j: np.ndarray
    bond_R: np.ndarray
    bond_distance: np.ndarray
    dDMI: np.ndarray | None = None
    dJ_iso: np.ndarray | None = None
    dJ_aniso: np.ndarray | None = None
    units: str = "meV/A"

    @property
    def shape_summary(self) -> Dict[str, Any]:
        return {
            "n_targets": int(self.target_atoms.size),
            "n_bonds": int(self.bond_i.size),
            "n_rp": int(self.rp_grid.shape[0]),
            "n_disp_axes": int(len(self.disp_axes)),
            "dDMI_shape": None if self.dDMI is None else tuple(int(x) for x in self.dDMI.shape),
            "dJ_iso_shape": None if self.dJ_iso is None else tuple(int(x) for x in self.dJ_iso.shape),
            "dJ_aniso_shape": None if self.dJ_aniso is None else tuple(int(x) for x in self.dJ_aniso.shape),
            "units": self.units,
        }

    def component(self, name: str) -> np.ndarray:
        key = _normalize_component_name(name)
        arr = getattr(self, COMPONENT_SPECS[key][2])
        if arr is None:
            raise KeyError(f"Component {key!r} was not loaded")
        return arr

    def flat_entries(
        self,
        component: str = "dmi",
        *,
        include_zero: bool = True,
        threshold: float = 0.0,
    ) -> Dict[str, np.ndarray]:
        """Return a flat real-space vertex list for one loaded component.

        Components are stored densely as `(target, bond, Rp, disp_axis, ...)`.
        The flat output has one row per `(target, bond, Rp, disp_axis)` entry.
        """
        key = _normalize_component_name(component)
        arr = self.component(key)
        n_t, n_b, n_rp, n_disp = arr.shape[:4]
        values = arr.reshape((n_t * n_b * n_rp * n_disp,) + arr.shape[4:])

        target_idx, bond_idx, rp_idx, disp_idx = np.indices((n_t, n_b, n_rp, n_disp), sparse=False)
        target_idx = target_idx.reshape(-1)
        bond_idx = bond_idx.reshape(-1)
        rp_idx = rp_idx.reshape(-1)
        disp_idx = disp_idx.reshape(-1)

        if include_zero:
            mask = np.ones(values.shape[0], dtype=bool)
        else:
            flat_values = values.reshape(values.shape[0], -1)
            mask = np.linalg.norm(flat_values, axis=1) > float(threshold)

        disp_axis_arr = np.asarray(self.disp_axes, dtype=object)
        return {
            "component": np.full(np.count_nonzero(mask), key, dtype=object),
            "target_atom": self.target_atoms[target_idx[mask]].astype(np.int32, copy=False),
            "target_slot": target_idx[mask].astype(np.int32, copy=False),
            "disp_axis_index": disp_idx[mask].astype(np.int32, copy=False),
            "disp_axis": disp_axis_arr[disp_idx[mask]],
            "bond_index": bond_idx[mask].astype(np.int32, copy=False),
            "i_atom": self.bond_i[bond_idx[mask]].astype(np.int32, copy=False),
            "j_atom": self.bond_j[bond_idx[mask]].astype(np.int32, copy=False),
            "R": self.bond_R[bond_idx[mask]].astype(np.int32, copy=False),
            "rp_index": rp_idx[mask].astype(np.int32, copy=False),
            "Rp": self.rp_grid[rp_idx[mask]].astype(np.int32, copy=False),
            key: values[mask].astype(np.float64, copy=False),
        }


def _normalize_component_name(name: str) -> str:
    lowered = str(name).strip().lower()
    aliases = {
        "ddmi": "dmi",
        "ddmi_r": "dmi",
        "dmi_r": "dmi",
        "dj_iso": "iso",
        "dj_iso_r": "iso",
        "j_iso": "iso",
        "iso_r": "iso",
        "dj_aniso": "aniso",
        "dj_aniso_r": "aniso",
        "dj_gamma": "aniso",
        "dj_gamma_r": "aniso",
        "j_aniso": "aniso",
        "gamma": "aniso",
        "gamma_r": "aniso",
    }
    key = aliases.get(lowered, lowered)
    if key not in COMPONENT_SPECS:
        allowed = ", ".join(sorted(COMPONENT_SPECS))
        raise ValueError(f"Unknown exchange tensor component {name!r}; allowed: {allowed}")
    return key


def _normalize_components(components: Iterable[str] | str) -> list[str]:
    if isinstance(components, str):
        items = [x.strip() for x in components.split(",")]
    else:
        items = list(components)
    out = []
    for item in items:
        if not item:
            continue
        key = _normalize_component_name(item)
        if key not in out:
            out.append(key)
    if not out:
        raise ValueError("At least one exchange tensor component is required")
    return out


def _read_metadata(h5, path: str) -> Dict[str, Any]:
    required = [
        "basic_data/atom_labels",
        "basic_data/qmesh",
        "bonds/mag_i_atom",
        "bonds/mag_j_atom",
        "bonds/R",
        "displacements/target_atom",
        "displacements/axes",
        "displacements/Rp",
    ]
    _require(h5, required, path)
    bond_i = np.asarray(h5["bonds/mag_i_atom"], dtype=np.int32).reshape(-1)
    if "bonds/distance_ang" in h5:
        bond_distance = np.asarray(h5["bonds/distance_ang"], dtype=np.float64).reshape(-1)
    else:
        bond_distance = np.full(bond_i.shape, np.nan, dtype=np.float64)
    return {
        "atom_labels": _decode_str_array(h5["basic_data/atom_labels"][()]),
        "qmesh": np.asarray(h5["basic_data/qmesh"], dtype=np.int32).reshape(3),
        "target_atoms": np.asarray(h5["displacements/target_atom"], dtype=np.int32).reshape(-1),
        "disp_axes": [x.lower() for x in _decode_str_array(h5["displacements/axes"][()])],
        "rp_grid": np.asarray(h5["displacements/Rp"], dtype=np.int32).reshape(-1, 3),
        "bond_i": bond_i,
        "bond_j": np.asarray(h5["bonds/mag_j_atom"], dtype=np.int32).reshape(-1),
        "bond_R": np.asarray(h5["bonds/R"], dtype=np.int32).reshape(-1, 3),
        "bond_distance": bond_distance,
    }


def _read_component_group(h5, path: str, meta: Dict[str, Any], component_key: str) -> np.ndarray:
    group_name, tail_shape, _ = COMPONENT_SPECS[component_key]
    _require(h5, [group_name], path)
    target_atoms = meta["target_atoms"]
    n_t = int(target_atoms.size)
    n_b = int(meta["bond_i"].size)
    n_rp = int(meta["rp_grid"].shape[0])
    n_disp = int(len(meta["disp_axes"]))
    expected = (n_rp, n_disp) + tuple(tail_shape)

    grp = h5[group_name]
    key_map = _scan_target_bond_group(grp)
    arr_out = np.zeros((n_t, n_b, n_rp, n_disp) + tuple(tail_shape), dtype=np.float64)
    missing = []
    for it, target in enumerate(target_atoms):
        for ib in range(n_b):
            key = key_map.get((int(target), ib))
            if key is None:
                missing.append(f"m{int(target)+1}_b{ib+1}")
                continue
            arr = np.asarray(grp[key], dtype=np.float64)
            if arr.shape != expected:
                raise ValueError(f"/{group_name}/{key} shape={arr.shape}, expected={expected}")
            arr_out[it, ib] = arr
    if missing:
        preview = ", ".join(missing[:8])
        suffix = "" if len(missing) <= 8 else ", ..."
        raise KeyError(f"Missing /{group_name} datasets in {path}: {preview}{suffix}")
    return arr_out


def load_exchange_tensor_realspace_h5(
    h5_path: str,
    *,
    components: Iterable[str] | str = ("dmi", "iso", "aniso"),
) -> ExchangeTensorRealSpacePayload:
    """Load real-space dDMI, dJ_iso, and/or dJ_aniso from tensor HDF5."""
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"dJ tensor HDF5 not found: {h5_path}")

    component_keys = _normalize_components(components)
    data = {"dDMI": None, "dJ_iso": None, "dJ_aniso": None}
    with h5py.File(h5_path, "r") as h5:
        meta = _read_metadata(h5, h5_path)
        for key in component_keys:
            _, _, field_name = COMPONENT_SPECS[key]
            data[field_name] = _read_component_group(h5, h5_path, meta, key)

    return ExchangeTensorRealSpacePayload(
        source=os.path.abspath(h5_path),
        **meta,
        **data,
    )


def load_dmi_realspace_h5(h5_path: str) -> ExchangeTensorRealSpacePayload:
    return load_exchange_tensor_realspace_h5(h5_path, components=("dmi",))


def load_j_iso_realspace_h5(h5_path: str) -> ExchangeTensorRealSpacePayload:
    return load_exchange_tensor_realspace_h5(h5_path, components=("iso",))


def load_j_aniso_realspace_h5(h5_path: str) -> ExchangeTensorRealSpacePayload:
    return load_exchange_tensor_realspace_h5(h5_path, components=("aniso",))


def summarize_tensor_payload(
    payload: ExchangeTensorRealSpacePayload,
    *,
    component: str = "dmi",
    threshold: float = 0.0,
) -> Dict[str, Any]:
    key = _normalize_component_name(component)
    flat_all = payload.flat_entries(key, include_zero=True)
    values = flat_all[key]
    norms = np.linalg.norm(values.reshape(values.shape[0], -1), axis=1)
    nonzero = norms > float(threshold)
    out = dict(payload.shape_summary)
    out.update(
        {
            "component": key,
            "n_flat_entries": int(norms.size),
            "n_entries_above_threshold": int(np.count_nonzero(nonzero)),
            "threshold": float(threshold),
            "max_norm": float(norms.max()) if norms.size else 0.0,
            "mean_norm": float(norms.mean()) if norms.size else 0.0,
            "qmesh": tuple(int(x) for x in payload.qmesh),
            "targets_0based": [int(x) for x in payload.target_atoms],
            "disp_axes": list(payload.disp_axes),
        }
    )
    return out


def summarize_dmi_payload(payload: ExchangeTensorRealSpacePayload, *, threshold: float = 0.0) -> Dict[str, Any]:
    return summarize_tensor_payload(payload, component="dmi", threshold=threshold)


def main():
    ap = argparse.ArgumentParser(description="Inspect/load real-space exchange tensor HDF5 payload")
    ap.add_argument("h5", help="compute_dJ_epr_tensor HDF5 file")
    ap.add_argument(
        "--components",
        default="dmi,iso,aniso",
        help="Comma-separated components to load: dmi, iso, aniso",
    )
    ap.add_argument("--show-component", default="dmi", help="Component to print: dmi, iso, aniso")
    ap.add_argument("--threshold", type=float, default=0.0, help="Threshold for nonzero-entry count")
    ap.add_argument("--show", type=int, default=5, help="Print first N flat entries above threshold")
    args = ap.parse_args()

    payload = load_exchange_tensor_realspace_h5(args.h5, components=args.components)
    show_component = _normalize_component_name(args.show_component)
    summary = summarize_tensor_payload(payload, component=show_component, threshold=args.threshold)
    print("[tensor-adapter] summary")
    for key in sorted(summary):
        print(f"  {key}: {summary[key]}")

    if int(args.show) > 0:
        flat = payload.flat_entries(show_component, include_zero=False, threshold=args.threshold)
        nshow = min(int(args.show), len(flat[show_component]))
        print(f"[tensor-adapter] first {nshow} {show_component} entries above threshold")
        labels = payload.atom_labels
        for i in range(nshow):
            ia = int(flat["i_atom"][i])
            ja = int(flat["j_atom"][i])
            ka = int(flat["target_atom"][i])
            ilab = _atom_label(labels, ia)
            jlab = _atom_label(labels, ja)
            klab = _atom_label(labels, ka)
            print(
                "  "
                f"bond={int(flat['bond_index'][i])+1} {ilab}->{jlab} "
                f"R={tuple(int(x) for x in flat['R'][i])} "
                f"target={klab} axis={flat['disp_axis'][i]} "
                f"Rp={tuple(int(x) for x in flat['Rp'][i])} "
                f"{show_component}={np.asarray(flat[show_component][i]).tolist()}"
            )


if __name__ == "__main__":
    main()
