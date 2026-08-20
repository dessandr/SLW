"""Shared plotting and k-path utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np


DEFAULT_KPOINTS = {
    "g": (0.0, 0.0, 0.0),
    "gamma": (0.0, 0.0, 0.0),
    "Γ": (0.0, 0.0, 0.0),
    "m": (0.5, 0.0, 0.0),
    "k": (1.0 / 3.0, 1.0 / 3.0, 0.0),
    "a": (0.0, 0.0, 0.5),
    "l": (0.5, 0.0, 0.5),
    "h": (1.0 / 3.0, 1.0 / 3.0, 0.5),
}


def reciprocal_lattice(lattice_ang: np.ndarray) -> np.ndarray:
    lattice = np.asarray(lattice_ang, dtype=np.float64).reshape(3, 3)
    return 2.0 * np.pi * np.linalg.inv(lattice).T


def display_klabel(label: str) -> str:
    token = label.strip()
    if token.lower() in {"g", "gamma", "gam"} or token == "Γ":
        return r"$\Gamma$"
    return token


def point_table(config) -> dict[str, np.ndarray]:
    table = {k: np.asarray(v, dtype=np.float64) for k, v in DEFAULT_KPOINTS.items()}
    for key, value in config.prefixed("point").items():
        coords = [float(x) for x in value.replace(",", " ").split()]
        if len(coords) != 3:
            raise ValueError(f"point.{key} must have three fractional coordinates")
        table[key.strip().lower()] = np.asarray(coords, dtype=np.float64)
    return table


def build_kpath(config, lattice_ang: np.ndarray):
    kpath_file = config.get("kpath_file")
    if kpath_file:
        return build_kpath_from_qe_file(config.path_value("kpath_file", must_exist=True), lattice_ang, config.get_int("band_points", 101))

    labels = config.get_list("kpath", ["G", "M", "K", "G"])
    if len(labels) < 2:
        raise ValueError("kpath must contain at least two labels")

    points = point_table(config)
    n_per_segment = config.get_int("band_points", 101)
    if n_per_segment < 2:
        raise ValueError("band_points must be >= 2")

    frac_segments = []
    for start, end in zip(labels[:-1], labels[1:]):
        key0 = start.strip().lower()
        key1 = end.strip().lower()
        if key0 not in points:
            raise KeyError(f"Unknown kpath point '{start}'. Add point.{start} = kx ky kz")
        if key1 not in points:
            raise KeyError(f"Unknown kpath point '{end}'. Add point.{end} = kx ky kz")
        frac_segments.append(np.linspace(points[key0], points[key1], n_per_segment, endpoint=False))
    kfrac = np.vstack(frac_segments + [points[labels[-1].strip().lower()][None, :]])

    recip = reciprocal_lattice(lattice_ang)
    kcart = kfrac @ recip
    step = np.linalg.norm(np.diff(kcart, axis=0), axis=1)
    kdist = np.zeros(kfrac.shape[0], dtype=np.float64)
    kdist[1:] = np.cumsum(step)
    ticks = [i * n_per_segment for i in range(len(labels) - 1)] + [kfrac.shape[0] - 1]
    tick_labels = [display_klabel(label) for label in labels]
    return kfrac, kdist, ticks, tick_labels


def _strip_brackets(text: str) -> str:
    return text.replace("{", " ").replace("}", " ").replace("(", " ").replace(")", " ")


def _split_qe_kpoint_line(raw: str) -> tuple[str, str]:
    body, sep, comment = raw.partition("!")
    if not sep:
        body, sep, comment = raw.partition("#")
    return body.strip(), comment.strip()


def _is_float_token(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def _qe_label_from_comment(comment: str, fallback: str) -> str:
    if not comment:
        return fallback
    token = comment.split()[0].strip()
    return token or fallback


def read_qe_kpoints_crystal_b(path: str | Path, default_npoints: int = 101) -> list[dict]:
    """Read a QE ``K_POINTS crystal_b`` path file.

    Supported line styles after the point count are both common variants:

    ``kx ky kz n ! Label`` and ``Label kx ky kz n``.
    """
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()

    header_idx = None
    unit = ""
    for iline, raw in enumerate(lines):
        body, _comment = _split_qe_kpoint_line(raw)
        if not body:
            continue
        parts = _strip_brackets(body).split()
        if parts and parts[0].upper() == "K_POINTS":
            header_idx = iline
            unit = parts[1].lower() if len(parts) > 1 else ""
            break
    if header_idx is None:
        raise ValueError(f"{path}: missing K_POINTS header")
    if unit != "crystal_b":
        raise ValueError(f"{path}: expected K_POINTS crystal_b, got unit {unit!r}")

    cursor = header_idx + 1
    while cursor < len(lines) and not _split_qe_kpoint_line(lines[cursor])[0]:
        cursor += 1
    if cursor >= len(lines):
        raise ValueError(f"{path}: missing K_POINTS point count")
    count_body, _count_comment = _split_qe_kpoint_line(lines[cursor])
    nlabels = int(_strip_brackets(count_body).split()[0])
    cursor += 1

    records = []
    while cursor < len(lines) and len(records) < nlabels:
        body, comment = _split_qe_kpoint_line(lines[cursor])
        cursor += 1
        if not body:
            continue
        parts = _strip_brackets(body).split()
        if not parts:
            continue
        if _is_float_token(parts[0]):
            if len(parts) < 4:
                raise ValueError(f"{path}: invalid K_POINTS line: {body}")
            frac = [float(parts[0]), float(parts[1]), float(parts[2])]
            nseg = int(parts[3])
            label = parts[4] if len(parts) >= 5 else _qe_label_from_comment(comment, f"K{len(records)}")
        else:
            if len(parts) < 5:
                raise ValueError(f"{path}: invalid labelled K_POINTS line: {body}")
            label = parts[0]
            frac = [float(parts[1]), float(parts[2]), float(parts[3])]
            nseg = int(parts[4])
        records.append({"label": label, "frac": np.asarray(frac, dtype=np.float64), "npoints": max(1, int(nseg or default_npoints))})

    if len(records) != nlabels:
        raise ValueError(f"{path}: declared {nlabels} K_POINTS labels but read {len(records)}")
    if len(records) < 2:
        raise ValueError(f"{path}: K_POINTS crystal_b path needs at least two points")
    return records


def build_kpath_from_qe_file(path: str | Path, lattice_ang: np.ndarray, default_npoints: int = 101):
    records = read_qe_kpoints_crystal_b(path, default_npoints=default_npoints)
    recip = reciprocal_lattice(lattice_ang)

    k_list = []
    ticks = []
    for iseg in range(len(records) - 1):
        start = np.asarray(records[iseg]["frac"], dtype=np.float64)
        end = np.asarray(records[iseg + 1]["frac"], dtype=np.float64)
        nseg = max(1, int(records[iseg].get("npoints") or default_npoints))
        ticks.append(len(k_list))
        k_list.extend(np.linspace(start, end, nseg, endpoint=False))
    k_list.append(np.asarray(records[-1]["frac"], dtype=np.float64))
    ticks.append(len(k_list) - 1)

    kfrac = np.asarray(k_list, dtype=np.float64)
    kcart = kfrac @ recip
    step = np.linalg.norm(np.diff(kcart, axis=0), axis=1)
    kdist = np.zeros(kfrac.shape[0], dtype=np.float64)
    kdist[1:] = np.cumsum(step)
    tick_labels = [display_klabel(str(rec["label"])) for rec in records]
    return kfrac, kdist, ticks, tick_labels


def prepare_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt
