import os
import sys
import importlib


def _import_from_legacy(legacy_dir: str, mod_name: str):
    if not os.path.isdir(legacy_dir):
        raise FileNotFoundError(f"legacy magph directory not found: {legacy_dir}")
    if legacy_dir not in sys.path:
        sys.path.insert(0, legacy_dir)
    return importlib.import_module(mod_name)


def load_legacy_backend(legacy_dir: str, kernel_source: str = "kernels"):
    """
    Load legacy magph modules from external directory.
    Returns dict with `utils`, `kernels`.
    """
    if kernel_source not in {"kernels", "kernels_lifetime"}:
        raise ValueError(f"Unsupported kernel_source: {kernel_source}")
    if not legacy_dir or not os.path.isdir(legacy_dir):
        utils = importlib.import_module("slw.magph.legacy.utils")
        kernels = importlib.import_module("slw.magph.legacy.kernels")
        return {"utils": utils, "kernels": kernels, "kernel_source": "internal"}
    utils = _import_from_legacy(legacy_dir, "utils")
    kernels = _import_from_legacy(legacy_dir, kernel_source)
    return {"utils": utils, "kernels": kernels, "kernel_source": kernel_source}
