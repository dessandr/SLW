"""Plot a Wannier90 band structure along the path declared in a ``.win`` file."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from slw.core.structure import (
    read_wannier90_kpoint_path,
    read_wannier90_structure,
)
from slw.core.wannier_io import read_wannier_hr


def _sample_path(win_path: Path, points_per_segment: int):
    lattice, _, _, _ = read_wannier90_structure(win_path)
    reciprocal = np.linalg.inv(lattice).T * (2.0 * np.pi)
    segments = read_wannier90_kpoint_path(win_path)
    kpoints = []
    distances = []
    tick_positions = []
    tick_labels = []
    cumulative = 0.0
    for index, (start_label, start, end_label, end) in enumerate(segments):
        fractions = np.linspace(
            0.0,
            1.0,
            int(points_per_segment),
            endpoint=index == len(segments) - 1,
        )
        points = start[None, :] + fractions[:, None] * (end - start)[None, :]
        length = float(np.linalg.norm((end - start) @ reciprocal))
        kpoints.append(points)
        distances.append(cumulative + fractions * length)
        if not tick_positions or abs(tick_positions[-1] - cumulative) > 1.0e-12:
            tick_positions.append(cumulative)
            tick_labels.append(start_label)
        cumulative += length
        tick_positions.append(cumulative)
        tick_labels.append(end_label)
    return (
        np.concatenate(kpoints, axis=0),
        np.concatenate(distances),
        tick_positions,
        tick_labels,
    )


def _bands(hamiltonian, degeneracies, kpoints):
    r_vectors = list(hamiltonian)
    if len(r_vectors) != len(degeneracies):
        raise ValueError("Hamiltonian R points and degeneracies have different lengths")
    blocks = np.stack(
        [
            np.asarray(hamiltonian[r_vector], dtype=np.complex128)
            / float(degeneracies[index])
            for index, r_vector in enumerate(r_vectors)
        ]
    )
    phases = np.exp(
        2.0j
        * np.pi
        * (np.asarray(kpoints, dtype=np.float64) @ np.asarray(r_vectors, dtype=np.float64).T)
    )
    hk = np.einsum("kr,rij->kij", phases, blocks, optimize=True)
    hk = 0.5 * (hk + np.swapaxes(hk.conj(), 1, 2))
    return np.linalg.eigvalsh(hk)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hr", required=True, help="Wannier90 *_hr.dat Hamiltonian")
    parser.add_argument("--win", required=True, help="Wannier90 .win with structure and kpoint_path")
    parser.add_argument("--efermi", type=float, required=True, help="Fermi energy in eV")
    parser.add_argument("--reference", choices=["fermi", "vbm"], default="fermi")
    parser.add_argument("--points-per-segment", type=int, default=80)
    parser.add_argument("--ylim", type=float, nargs=2, default=None, metavar=("EMIN", "EMAX"))
    parser.add_argument("--output", default="wannier_bands.png")
    args = parser.parse_args()

    if args.points_per_segment < 2:
        raise ValueError("--points-per-segment must be at least 2")
    _, degeneracies, hamiltonian = read_wannier_hr(args.hr)
    kpoints, distances, ticks, labels = _sample_path(
        Path(args.win), args.points_per_segment
    )
    energies = _bands(hamiltonian, degeneracies, kpoints)
    occupied = energies[energies <= float(args.efermi)]
    if occupied.size == 0:
        raise ValueError("No eigenvalue lies below --efermi; cannot determine VBM")
    vbm = float(np.max(occupied))
    reference = float(args.efermi) if args.reference == "fermi" else vbm

    figure, axis = plt.subplots(figsize=(8, 6))
    axis.plot(distances, energies - reference, color="tab:red", linewidth=0.9)
    axis.axhline(0.0, color="black", linewidth=0.8)
    for position in ticks:
        axis.axvline(position, color="black", linestyle=":", alpha=0.35)
    axis.set_xticks(ticks, [r"$\Gamma$" if label.upper() in {"G", "GAMMA"} else label for label in labels])
    axis.set_xlim(float(distances[0]), float(distances[-1]))
    if args.ylim is not None:
        axis.set_ylim(*args.ylim)
    axis.set_ylabel(f"Energy relative to {args.reference} (eV)")
    axis.set_title("Wannier90 band structure")
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
