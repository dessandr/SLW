import os


def resolve_workdir(workdir=None):
    """Return absolute workflow root (defaults to current directory)."""
    return os.path.abspath(workdir) if workdir else os.getcwd()


def resolve_path(base_dir, path):
    """Resolve path relative to base_dir unless already absolute."""
    if path is None:
        return None
    return path if os.path.isabs(path) else os.path.abspath(os.path.join(base_dir, path))


def resolve_in_dir(workdir, in_dir=None, legacy_in=None, default_subdir=None, legacy_label="--input"):
    """Resolve normalized input directory with legacy alias fallback."""
    if in_dir:
        return resolve_path(workdir, in_dir)
    if legacy_in:
        print(f"[DEPRECATED] {legacy_label} is legacy naming. Prefer --in_dir (optionally with --workdir).")
        return resolve_path(workdir, legacy_in)
    if default_subdir is None:
        return workdir
    return resolve_path(workdir, default_subdir)


def resolve_out_dir(workdir, out_dir=None, legacy_out=None, default_subdir=".", legacy_label="--output"):
    """Resolve output directory with optional legacy alias fallback."""
    if out_dir:
        out_abs = resolve_path(workdir, out_dir)
    elif legacy_out:
        print(f"[DEPRECATED] {legacy_label} is legacy naming. Prefer --out_dir (optionally with --workdir).")
        out_abs = resolve_path(workdir, legacy_out)
    else:
        out_abs = resolve_path(workdir, default_subdir)
    os.makedirs(out_abs, exist_ok=True)
    return out_abs


def resolve_out_path(
    workdir,
    out_dir=None,
    out_name=None,
    legacy_out=None,
    default_dir=".",
    default_name=None,
    legacy_label="--output",
):
    """
    Resolve output file path using --out_dir/--out_name, with legacy output path fallback.
    """
    if out_dir or out_name:
        out_abs_dir = resolve_path(workdir, out_dir) if out_dir else resolve_path(workdir, default_dir)
        os.makedirs(out_abs_dir, exist_ok=True)
        file_name = out_name if out_name else default_name
        if file_name is None:
            raise ValueError("Output file name is required: set --out_name or provide default_name.")
        return os.path.join(out_abs_dir, file_name)

    if legacy_out:
        print(f"[DEPRECATED] {legacy_label} is legacy naming. Prefer --out_dir + --out_name.")
        out_abs = resolve_path(workdir, legacy_out)
        out_parent = os.path.dirname(out_abs) or "."
        os.makedirs(out_parent, exist_ok=True)
        return out_abs

    out_abs_dir = resolve_path(workdir, default_dir)
    os.makedirs(out_abs_dir, exist_ok=True)
    if default_name is None:
        raise ValueError("default_name is required when no output argument is provided.")
    return os.path.join(out_abs_dir, default_name)
