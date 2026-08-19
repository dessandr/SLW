import os


def _strip_comment(line: str) -> str:
    return line.split("#", 1)[0].strip()


def _to_int_list(val: str):
    clean = val.replace(",", " ").replace(";", " ")
    return [int(x) for x in clean.split()]


def _to_str_list(val: str):
    return [x for x in val.split()]


def _to_float(val: str):
    text = val.strip()
    if "/" in text:
        num, den = text.split("/", 1)
        return float(num.strip()) / float(den.strip())
    return float(text)


def _to_float_list(val: str):
    clean = val.replace(",", " ").replace(";", " ")
    return [_to_float(x) for x in clean.split()]


def _to_bool(val: str):
    text = str(val).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {val!r}")


def parse_input_file(path: str):
    """
    Parse legacy-style input.in file and return normalized dict.
    This keeps input-file workflow while allowing SLW internal standardization.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"input file not found: {path}")

    cfg = {
        "T": 300.0,
        "S": 2.5,
        "eta": 1.0,
        "k_mesh": [1, 1, 1],
        "target_k_mesh": [10, 10, 10],
        "magnetic_atoms": [],
        "exchange_folder": ".",
        "POSCAR": None,
        "structure_file": None,
        "phonon_path": None,
        "manifest": None,
        "djdu_npz": None,
        "jr": None,
        "djr": None,
        "phonon_cache": None,
        "phonon_fc": None,
        "phonon_qshift": None,
        "phonon_asr": "none",
        "phonon_loto": "auto",
        "calculation_mode": "cache",
        "band_points": 101,
        "omega_points": 5000,
        "target_rank_gb": 4.0,
        "bond_factor": 1.0,
        "anisotropy_mev": 0.0,
        "phonon_floor_mev": 1.0e-3,
        "dJ_asr": "check",
        "dJ_asr_tolerance": 1.0e-8,
        "vertex_q_chunk": 32,
        "vertex_bond_chunk": 64,
        "kernel_source": "kernels",
        "exclude_shells": [],
        "shell_tol": 1.0e-4,
        "exclude_shell_apply": "both",
        "phonon_nproc": 1,
        "hybrid_nproc": None,
        "phonon_cache_compressed": True,
        "hybrid_component": None,
        "hybrid_static_component": None,
        "hybrid_output": None,
        "hybrid_plot": None,
        "hybrid_html_plot": None,
        "hybrid_html_data": None,
        "hybrid_bare_output": None,
        "hybrid_bare_plot": None,
        "hybrid_phonon_cache_out": None,
    }

    aliases = {
        "poscar_path": "POSCAR",
        "structure": "structure_file",
        "structure_path": "structure_file",
        "phonopy_path": "phonon_path",
        "input_manifest": "manifest",
        "dJ_tensor_h5": "dJ_tensor_h5",
        "tensor_h5": "dJ_tensor_h5",
        "J_tensor_h5": "J_tensor_h5",
        "j_tensor_h5": "J_tensor_h5",
        "hybrid_dJ_tensor_h5": "dJ_tensor_h5",
        "hybrid_J_tensor_h5": "J_tensor_h5",
        "hybrid_structure": "structure_file",
        "component": "hybrid_component",
        "static_component": "hybrid_static_component",
        "output": "hybrid_output",
        "plot": "hybrid_plot",
        "html_plot": "hybrid_html_plot",
        "html_data": "hybrid_html_data",
        "bare_output": "hybrid_bare_output",
        "bare_plot": "hybrid_bare_plot",
        "phonon_cache_out": "hybrid_phonon_cache_out",
        "hybrid_phonon_cache": "phonon_cache",
        "djr_h5": "djr",
        "j_cache": "jr",
        "dj_asr": "dJ_asr",
        "dj_asr_tolerance": "dJ_asr_tolerance",
    }

    with open(path, "r") as f:
        for raw in f:
            line = _strip_comment(raw)
            if not line or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip()
            key = aliases.get(key, key)

            if key in {
                "T",
                "S",
                "eta",
                "target_rank_gb",
                "shell_tol",
                "coupling_scale",
                "bond_factor",
                "anisotropy_mev",
                "phonon_negative_tol_mev",
                "phonon_floor_mev",
                "dJ_asr_tolerance",
                "gamma_zero_tol",
                "threshold",
                "g_factor",
                "xtick_fontsize",
            }:
                cfg[key] = _to_float(val)
            elif key in {
                "band_points",
                "omega_points",
                "phonon_nproc",
                "hybrid_nproc",
                "path_points",
                "gamma_acoustic_zero",
                "vertex_q_chunk",
                "vertex_bond_chunk",
            }:
                cfg[key] = int(val)
            elif key in {"k_mesh", "target_k_mesh", "exclude_shells", "phonon_qmesh", "rp_idx", "kmesh"}:
                cfg[key] = _to_int_list(val)
            elif key in {
                "spin_direction",
                "zeeman_field_mev",
                "field_tesla",
                "phonon_qshift",
            }:
                cfg[key] = _to_float_list(val)
            elif key in {"phonon_cache_compressed", "overlay_bare", "show"}:
                cfg[key] = _to_bool(val)
            elif key == "magnetic_atoms":
                cfg[key] = _to_str_list(val)
            elif key in {
                "kernel_source",
                "exclude_shell_apply",
                "shell_filter_apply",
                "calculation_mode",
                "phonon_asr",
                "phonon_loto",
                "dJ_asr",
            }:
                cfg[key] = val.strip()
            else:
                cfg[key] = val

    return cfg
