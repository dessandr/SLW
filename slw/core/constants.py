"""Physical constants used throughout the SLW framework."""

# Energy conversions
RY_TO_EV = 13.605693122994
HA_TO_EV = 2 * RY_TO_EV  # 1 Hartree = 27.211 eV

# Thermodynamic and spectral conversions
HBAR_MEV_PS = 0.6582119569
KB_MEV_PER_K = 0.08617333262

# Zero-point length prefactors.  For a mode energy E in meV and a
# mass-normalized eigenvector expressed with the indicated mass unit,
# x_zpf [Angstrom] = prefactor * eigenvector / sqrt(E [meV]).
ZPF_ANG_SQRT_MEV_ELECTRON_MASS = 61.7250525812
ZPF_ANG_SQRT_MEV_AMU = 1.44571077426

# Length conversions
BOHR_TO_ANG = 0.529177249
ANG_TO_BOHR = 1.0 / BOHR_TO_ANG
