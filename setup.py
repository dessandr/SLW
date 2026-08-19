from setuptools import find_packages, setup


setup(
    name="slw",
    version="0.1.0",
    packages=find_packages(),
    description="Spin-Lattice Wannier toolkit for exchange and magnon-phonon interactions",
    python_requires=">=3.10",
    install_requires=[
        "h5py",
        "matplotlib",
        "numba",
        "numpy",
        "scipy",
        "spglib",
        "threadpoolctl",
    ],
    extras_require={
        "mpi": ["mpi4py"],
        "phonon": ["phonopy"],
        "kpath": ["seekpath"],
    },
)
