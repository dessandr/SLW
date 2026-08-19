"""Plot total or orbital-projected density of states from a Wannier90 Hamiltonian."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

from slw.core.wannier_io import read_wannier_hr


def _parse_projection(specification: str, dimension: int):
    label, start_text, stop_text = specification.split(":", 2)
    start, stop = int(start_text), int(stop_text)
    if not label or start < 0 or stop <= start or stop > dimension:
        raise ValueError(
            f"Invalid projection {specification!r}; expected LABEL:START:STOP within 0:{dimension}"
        )
    return label, slice(start, stop)


def _kmesh(shape):
    axes = [np.arange(int(size), dtype=np.float64) / float(size) for size in shape]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)


def _eigensystem(hamiltonian, degeneracies, kpoints, projections, chunk_size):
    r_vectors = list(hamiltonian)
    if len(r_vectors) != len(degeneracies):
        raise ValueError("Hamiltonian R points and degeneracies have different lengths")
    blocks = np.stack(
        [hamiltonian[r] / float(degeneracies[index]) for index, r in enumerate(r_vectors)]
    )
    r_array = np.asarray(r_vectors, dtype=np.float64)
    energies = []
    weights = {label: [] for label, _ in projections}
    for start in range(0, len(kpoints), int(chunk_size)):
        batch = kpoints[start : start + int(chunk_size)]
        phase = np.exp(2.0j * np.pi * (batch @ r_array.T))
        hk = np.einsum("kr,rij->kij", phase, blocks, optimize=True)
        hk = 0.5 * (hk + np.swapaxes(hk.conj(), 1, 2))
        if projections:
            eigenvalues, eigenvectors = np.linalg.eigh(hk)
            probability = np.abs(eigenvectors) ** 2
            for label, orbital_slice in projections:
                weights[label].append(np.sum(probability[:, orbital_slice, :], axis=1))
        else:
            eigenvalues = np.linalg.eigvalsh(hk)
        energies.append(eigenvalues)
    return np.concatenate(energies), {
        label: np.concatenate(chunks) for label, chunks in weights.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hr", required=True, help="Wannier90 *_hr.dat Hamiltonian")
    parser.add_argument("--efermi", type=float, required=True, help="Fermi energy in eV")
    parser.add_argument("--reference", choices=["fermi", "vbm"], default="fermi")
    parser.add_argument("--kmesh", type=int, nargs=3, default=[12, 12, 12])
    parser.add_argument("--project", action="append", default=[], metavar="LABEL:START:STOP")
    parser.add_argument("--sigma", type=float, default=0.08, help="Gaussian width in eV")
    parser.add_argument("--bins", type=int, default=2000)
    parser.add_argument("--energy-window", type=float, nargs=2, default=None, metavar=("EMIN", "EMAX"))
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--output", default="wannier_dos.png")
    args = parser.parse_args()

    if any(size <= 0 for size in args.kmesh):
        raise ValueError("--kmesh dimensions must be positive")
    if args.sigma <= 0.0 or args.bins < 2 or args.chunk_size < 1:
        raise ValueError("--sigma, --bins, and --chunk-size must be positive")

    dimension, degeneracies, hamiltonian = read_wannier_hr(args.hr)
    projections = [_parse_projection(item, dimension) for item in args.project]
    kpoints = _kmesh(args.kmesh)
    energies, weights = _eigensystem(
        hamiltonian, degeneracies, kpoints, projections, args.chunk_size
    )
    occupied = energies[energies <= float(args.efermi)]
    if occupied.size == 0:
        raise ValueError("No eigenvalue lies below --efermi; cannot determine VBM")
    reference = float(args.efermi) if args.reference == "fermi" else float(np.max(occupied))
    shifted = energies - reference
    window = (
        tuple(float(value) for value in args.energy_window)
        if args.energy_window is not None
        else (float(np.min(shifted)), float(np.max(shifted)))
    )
    edges = np.linspace(window[0], window[1], int(args.bins) + 1)
    energy_axis = 0.5 * (edges[:-1] + edges[1:])
    bin_width = float(edges[1] - edges[0])
    smoothing = float(args.sigma) / bin_width
    normalization = float(len(kpoints) * bin_width)

    total, _ = np.histogram(shifted.ravel(), bins=edges)
    total_dos = gaussian_filter1d(total.astype(np.float64), smoothing) / normalization
    projected_dos = {}
    for label, projection_weights in weights.items():
        histogram, _ = np.histogram(
            shifted.ravel(), bins=edges, weights=projection_weights.ravel()
        )
        projected_dos[label] = gaussian_filter1d(histogram, smoothing) / normalization

    figure, axis = plt.subplots(figsize=(7, 6))
    axis.plot(energy_axis, total_dos, color="black", label="Total")
    for label, values in projected_dos.items():
        axis.plot(energy_axis, values, label=label)
    axis.axvline(0.0, color="grey", linestyle="--", linewidth=0.8)
    axis.set_xlim(*window)
    axis.set_xlabel(f"Energy relative to {args.reference} (eV)")
    axis.set_ylabel("DOS (states/eV/cell)")
    axis.set_title("Wannier90 density of states")
    if projected_dos:
        axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
