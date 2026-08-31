"""Backend-independent Gauss-Legendre real-axis integration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import constants


@dataclass(frozen=True)
class RealAxisIntegrator:
    nodes_eV: NDArray[np.float64]
    weights_eV: NDArray[np.float64]
    chemical_potential_eV: float
    temperature_K: float = 0.0

    @classmethod
    def gauss_legendre(
        cls,
        energy_min_eV: float,
        energy_max_eV: float,
        energy_points: int,
        *,
        chemical_potential_eV: float,
        temperature_K: float = 0.0,
    ) -> RealAxisIntegrator:
        if energy_max_eV <= energy_min_eV or energy_points < 2:
            raise ValueError("energy interval must increase and contain at least two nodes")
        raw_nodes, raw_weights = np.polynomial.legendre.leggauss(int(energy_points))
        half = 0.5 * (energy_max_eV - energy_min_eV)
        center = 0.5 * (energy_max_eV + energy_min_eV)
        return cls(
            nodes_eV=np.asarray(center + half * raw_nodes, dtype=np.float64),
            weights_eV=np.asarray(half * raw_weights, dtype=np.float64),
            chemical_potential_eV=float(chemical_potential_eV),
            temperature_K=float(temperature_K),
        )

    def occupations(self) -> NDArray[np.float64]:
        if self.temperature_K < 0:
            raise ValueError("temperature_K must be non-negative")
        if self.temperature_K == 0:
            below = self.nodes_eV < self.chemical_potential_eV
            equal = np.isclose(self.nodes_eV, self.chemical_potential_eV, atol=1e-15)
            return below.astype(np.float64) + 0.5 * equal.astype(np.float64)
        kb_eV = constants.Boltzmann / constants.electron_volt
        argument = (self.nodes_eV - self.chemical_potential_eV) / (kb_eV * self.temperature_K)
        return 1.0 / (np.exp(np.clip(argument, -700.0, 700.0)) + 1.0)

    def weighted_occupations(self) -> NDArray[np.float64]:
        return self.weights_eV * self.occupations()


__all__ = ["RealAxisIntegrator"]

