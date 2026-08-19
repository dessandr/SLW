"""
Phonon data utilities — phonopy wrappers.

Usage:
    python -m slw.phonon.parsing
"""

import numpy as np

try:
    import phonopy
    HAS_PHONOPY = True
except ImportError:
    HAS_PHONOPY = False


def load_phonon(phonon_dir, yaml_file="phonopy.yaml", force_sets="FORCE_SETS"):
    """Load phonopy object from directory.

    Parameters
    ----------
    phonon_dir : str
    yaml_file : str
    force_sets : str

    Returns
    -------
    phonon : phonopy.Phonopy
    """
    import os
    old_cwd = os.getcwd()
    os.chdir(phonon_dir)
    phonon = phonopy.load(yaml_file, force_sets_filename=force_sets)
    os.chdir(old_cwd)
    return phonon


def get_phonon_data(phonon, q_point):
    """Get mass-normalized phonon eigenvectors at a q-point.

    Returns
    -------
    frequencies : ndarray (n_modes,)
    displacements : ndarray (n_modes, n_atoms, 3) — mass-normalized
    """
    masses = phonon.primitive.masses
    sqrt_masses = np.sqrt(masses)[:, None]

    freqs, evecs = phonon.get_frequencies_with_eigenvectors(q_point)

    n_atoms = len(masses)
    n_modes = len(freqs)

    # Reshape: (n_atoms*3, n_modes) -> (n_atoms, 3, n_modes) -> (n_modes, n_atoms, 3)
    eigs = evecs.reshape(n_atoms, 3, n_modes).transpose(2, 0, 1)
    displacements = eigs / sqrt_masses

    return freqs, displacements


if __name__ == "__main__":
    if not HAS_PHONOPY:
        print("phonopy not installed.")
    else:
        import os
        phonon = load_phonon(".")
        masses = phonon.primitive.masses
        print(f"Masses: {masses}")
        freqs, disp = get_phonon_data(phonon, [0, 0, 0])
        print(f"Gamma frequencies: {freqs}")
