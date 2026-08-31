from setuptools import find_packages, setup


setup(
    name="slw",
    version="0.1.0",
    packages=find_packages(),
    description="Spin-Lattice Wannier toolkit for exchange and magnon-phonon interactions",
    python_requires=">=3.10",
    install_requires=[
        "f90nml>=1.4",
        "h5py",
        "matplotlib",
        "numba",
        "numpy",
        "PyYAML>=6",
        "scipy",
        "spglib",
        "threadpoolctl",
    ],
    extras_require={
        "mpi": ["mpi4py"],
        "phonon": ["phonopy"],
        "kpath": ["seekpath"],
    },
    entry_points={
        "console_scripts": [
            "slw_epr.x=slw.cli.epr:main",
            "slw_exchange.x=slw.cli.exchange:main",
            "slw_magph.x=slw.cli.magph:main",
            "slw_post.x=slw.cli.post:main",
            "wtorque=slw.wtorque.cli:main",
        ],
    },
)
