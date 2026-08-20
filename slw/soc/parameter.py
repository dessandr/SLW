"""Fit onsite atomic SOC parameters against a spinor SOC Wannier Hamiltonian.

This module builds a spin-major model Hamiltonian

    H(k) = spinor_from_collinear(H_up(k), H_dn(k), n)
           + sum_a lambda_a LS_a

and fits lambda_a to relative band energies of a target SOC hr.dat.  Relative
energies are measured from the valence-band maximum at each k point, so the
loss is insensitive to k-dependent scalar shifts.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from slw.core.wannier_io import read_wannier_hr, write_wannier_hr
from slw.exchange.kernels.epr import _full_k_mesh
from slw.exchange.kernels.soc_downfold import _selected_groups, _soc_matrix_for_groups
from slw.exchange.kernels.spinor import (
    WANNIER90_D_ORDER,
    WANNIER90_P_ORDER,
    spinor_from_collinear,
)


@dataclass(frozen=True)
class SOCParameterSpec:
    label: str
    selector: str
    lambda0: float
    lower: float
    upper: float


def _read_wannier_hr_compat(path: str):
    parsed = read_wannier_hr(path)
    if len(parsed) == 4:
        dim, _nrpts, degens, hmap = parsed
        return int(dim), list(degens), hmap
    if len(parsed) == 3:
        dim, degens, hmap = parsed
        return int(dim), list(degens), hmap
    raise ValueError(f"Unexpected read_wannier_hr return length={len(parsed)} for {path}")


def _hmap_to_hk(hmap: dict, kpts: np.ndarray, dim: int | None = None) -> np.ndarray:
    if not hmap:
        raise ValueError("Empty Hamiltonian map")
    keys = list(hmap.keys())
    if dim is None:
        dim = int(next(iter(hmap.values())).shape[0])
    rvec = np.asarray(keys, dtype=np.float64)
    blocks = np.asarray([np.asarray(hmap[r], dtype=np.complex128) for r in keys], dtype=np.complex128)
    phase = np.exp(2.0j * np.pi * (np.asarray(kpts, dtype=np.float64) @ rvec.T))
    hk = np.einsum("kr,rij->kij", phase, blocks, optimize=True)
    return 0.5 * (hk + np.swapaxes(hk.conj(), 1, 2))


def _parse_bounds(text: str, default: tuple[float, float]) -> tuple[float, float]:
    if text is None or str(text).strip() == "":
        return default
    vals = [float(x) for x in str(text).replace(":", ",").split(",") if x.strip()]
    if len(vals) != 2:
        raise ValueError(f"Expected lower,upper bounds, got {text!r}")
    lo, hi = vals
    if hi < lo:
        raise ValueError(f"Invalid bounds {text!r}: upper < lower")
    return float(lo), float(hi)



def _strip_input_comment(line: str) -> str:
    out = str(line)
    for token in ("#", "!"):
        out = out.split(token, 1)[0]
    return out.strip()


def _read_soc_parameter_input(path: str) -> dict:
    """Read a lightweight SLW .in file for SOC parameter fitting.

    Supported syntax:

        key = value
        key value
        begin soc_params
          Cr d 0.05 0.0 0.5
          I  p 0.50 0.0 2.0
        end soc_params

    CLI arguments override values loaded from the file.
    """
    cfg: dict[str, object] = {}
    block = None
    soc_rows: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = _strip_input_comment(raw)
            if not line:
                continue
            words = line.split()
            head = words[0].lower()
            if head == "begin":
                if len(words) != 2:
                    raise ValueError(f"{path}:{lineno}: expected 'begin <block>'")
                block = words[1].lower()
                if block != "soc_params":
                    raise ValueError(f"{path}:{lineno}: unsupported block {block!r}")
                continue
            if head == "end":
                if block is None:
                    raise ValueError(f"{path}:{lineno}: 'end' without active block")
                end_name = words[1].lower() if len(words) > 1 else block
                if end_name != block:
                    raise ValueError(f"{path}:{lineno}: end {end_name!r} does not match begin {block!r}")
                block = None
                continue
            if block == "soc_params":
                row = _parse_soc_param_input_row(line, path=path, lineno=lineno)
                soc_rows.append(row)
                continue
            if block is not None:
                raise ValueError(f"{path}:{lineno}: unsupported content inside block {block!r}")
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip().lower()
                value = value.strip()
            else:
                if len(words) < 2:
                    raise ValueError(f"{path}:{lineno}: expected 'key = value' or 'key value'")
                key = words[0].strip().lower()
                value = " ".join(words[1:]).strip()
            cfg[key] = value
    if block is not None:
        raise ValueError(f"{path}: missing 'end {block}'")
    if soc_rows:
        cfg["soc_params"] = ";".join(soc_rows)
    return cfg


def _parse_soc_param_input_row(line: str, *, path: str, lineno: int) -> str:
    parts = [x.strip() for x in line.replace(",", " ").split() if x.strip()]
    if len(parts) == 3:
        elem, orb, lam0 = parts
        return f"{elem}:{orb}:{lam0}"
    if len(parts) == 5:
        elem, orb, lam0, lo, hi = parts
        return f"{elem}:{orb}:{lam0}:{lo}:{hi}"
    if len(parts) in {2, 4} and ":" in parts[0]:
        label = parts[0]
        if len(parts) == 2:
            return f"{label}:{parts[1]}"
        return f"{label}:{parts[1]}:{parts[2]}:{parts[3]}"
    raise ValueError(
        f"{path}:{lineno}: soc_params row must be 'Element orbital lambda0 [lower upper]' "
        f"or 'Element:orbital lambda0 [lower upper]', got {line!r}"
    )


def _cfg_value(cfg: dict, key: str, default=None):
    return cfg.get(str(key).lower(), default)


def _cfg_float(cfg: dict, key: str, default=None):
    value = _cfg_value(cfg, key, default)
    return None if value is None else float(value)


def _cfg_int(cfg: dict, key: str, default=None):
    value = _cfg_value(cfg, key, default)
    return None if value is None else int(value)


def _cfg_vec(cfg: dict, key: str, n: int, default=None, dtype=float):
    value = _cfg_value(cfg, key, None)
    if value is None:
        return default
    vals = [dtype(x) for x in str(value).replace(",", " ").split() if x.strip()]
    if len(vals) != int(n):
        raise ValueError(f"Expected {n} values for {key}, got {value!r}")
    return vals


def _resolve_cli_or_input(args):
    cfg = _read_soc_parameter_input(args.input) if args.input else {}

    def pick(name, default=None):
        value = getattr(args, name)
        return value if value is not None else _cfg_value(cfg, name, default)

    args.up_hr = pick("up_hr")
    args.dn_hr = pick("dn_hr")
    args.target_soc_hr = pick("target_soc_hr")
    args.win = pick("win")
    args.soc_params = pick("soc_params")
    args.efermi = float(pick("efermi")) if pick("efermi") is not None else None
    args.model_efermi = float(pick("model_efermi")) if pick("model_efermi") is not None else None
    args.target_efermi = float(pick("target_efermi")) if pick("target_efermi") is not None else None
    args.spin_direction = args.spin_direction if args.spin_direction is not None else _cfg_vec(cfg, "spin_direction", 3, [0.0, 0.0, 1.0], float)
    args.kmesh = args.kmesh if args.kmesh is not None else _cfg_vec(cfg, "kmesh", 3, [8, 8, 8], int)
    args.p_order = pick("p_order", WANNIER90_P_ORDER)
    args.d_order = pick("d_order", WANNIER90_D_ORDER)
    args.weight_alpha = float(pick("weight_alpha", 0.5))
    args.band_window = int(pick("band_window", -1))
    args.maxiter = int(pick("maxiter", 200))
    args.ftol = float(pick("ftol", 1e-12))
    args.gtol = float(pick("gtol", 1e-8))
    args.out_hr = pick("out_hr")
    args.report = pick("report")

    required = ["up_hr", "dn_hr", "target_soc_hr", "win", "soc_params"]
    missing = [name for name in required if getattr(args, name) in {None, ""}]
    if missing:
        src = f" in {args.input}" if args.input else ""
        raise ValueError(f"Missing required SOC fitting input(s){src}: {', '.join(missing)}")
    return args

def parse_soc_parameter_specs(text: str, *, default_bounds=(0.0, 2.0)) -> list[SOCParameterSpec]:
    """Parse 'Cr:d:0.05[:lo:hi];I:p:0.5[:lo:hi]' specs."""
    specs: list[SOCParameterSpec] = []
    for raw in str(text).split(";"):
        item = raw.strip()
        if not item:
            continue
        parts = [x.strip() for x in item.replace(",", ":").split(":") if x.strip()]
        if len(parts) not in {3, 5}:
            raise ValueError(
                "Each --soc_params item must be element:orbital:lambda0 or "
                f"element:orbital:lambda0:lower:upper, got {item!r}"
            )
        elem, orb = parts[0], parts[1].lower()
        if orb not in {"p", "d"}:
            raise ValueError(f"SOC parameter orbital must be p or d, got {orb!r}")
        lam0 = float(parts[2])
        lo, hi = (float(parts[3]), float(parts[4])) if len(parts) == 5 else default_bounds
        if hi < lo:
            raise ValueError(f"Invalid bounds for {item!r}: upper < lower")
        label = f"{elem}:{orb}"
        specs.append(SOCParameterSpec(label=label, selector=label, lambda0=lam0, lower=lo, upper=hi))
    if not specs:
        raise ValueError("No SOC parameters specified")
    labels = [s.label for s in specs]
    if len(set(labels)) != len(labels):
        raise ValueError(f"Duplicate SOC parameter labels: {labels}")
    return specs


def build_ls_templates(win_path: str, nwan: int, specs: Sequence[SOCParameterSpec], *, p_order=WANNIER90_P_ORDER, d_order=WANNIER90_D_ORDER):
    templates = []
    summaries = []
    for spec in specs:
        groups = _selected_groups(win_path, spec.selector, nwan)
        mat, summary = _soc_matrix_for_groups(groups, nwan, 1.0, p_order=p_order, d_order=d_order)
        templates.append(np.asarray(mat, dtype=np.complex128))
        summaries.append(summary)
    return np.asarray(templates, dtype=np.complex128), summaries


def _reference_indices(evals: np.ndarray, efermi: float) -> np.ndarray:
    below = np.sum(evals < float(efermi), axis=1) - 1
    return np.clip(below, 0, evals.shape[1] - 1).astype(np.int64)


def relative_energies(evals: np.ndarray, efermi: float) -> tuple[np.ndarray, np.ndarray]:
    refs = _reference_indices(evals, efermi)
    ref_e = evals[np.arange(evals.shape[0]), refs]
    return evals - ref_e[:, None], refs


def band_weights(nband: int, ref_idx: np.ndarray, *, alpha: float, window: int | None = None) -> np.ndarray:
    bands = np.arange(int(nband), dtype=np.int64)[None, :]
    dist = np.abs(bands - ref_idx[:, None])
    w = np.exp(-float(alpha) * dist.astype(np.float64))
    if window is not None and int(window) >= 0:
        w = np.where(dist <= int(window), w, 0.0)
    return w


class SOCParameterFitter:
    def __init__(
        self,
        hk_up,
        hk_dn,
        hk_target,
        ls_templates,
        *,
        model_efermi,
        target_efermi,
        spin_direction=(0.0, 0.0, 1.0),
        alpha=0.5,
        band_window=None,
    ):
        self.hk_up = np.asarray(hk_up, dtype=np.complex128)
        self.hk_dn = np.asarray(hk_dn, dtype=np.complex128)
        self.hk_target = np.asarray(hk_target, dtype=np.complex128)
        self.ls_templates = np.asarray(ls_templates, dtype=np.complex128)
        self.model_efermi = float(model_efermi)
        self.target_efermi = float(target_efermi)
        self.spin_direction = tuple(float(x) for x in spin_direction)
        self.alpha = float(alpha)
        self.band_window = None if band_window is None else int(band_window)

        if self.hk_up.shape != self.hk_dn.shape:
            raise ValueError(f"up/down H(k) shape mismatch: {self.hk_up.shape} vs {self.hk_dn.shape}")
        if self.hk_target.shape[-1] != 2 * self.hk_up.shape[-1]:
            raise ValueError(
                f"target spinor dimension {self.hk_target.shape[-1]} is not 2*nwan={2*self.hk_up.shape[-1]}"
            )
        if self.hk_target.shape[0] != self.hk_up.shape[0]:
            raise ValueError(f"target/model k count mismatch: {self.hk_target.shape[0]} vs {self.hk_up.shape[0]}")
        if self.ls_templates.shape[1:] != self.hk_target.shape[1:]:
            raise ValueError(f"LS template shape {self.ls_templates.shape} incompatible with target {self.hk_target.shape}")

        self.h_base = spinor_from_collinear(self.hk_up, self.hk_dn, n=self.spin_direction)
        self.e_target = np.linalg.eigvalsh(self.hk_target)
        self.rel_target, self.ref_target = relative_energies(self.e_target, self.target_efermi)
        self.weights = band_weights(
            self.e_target.shape[1],
            self.ref_target,
            alpha=self.alpha,
            window=self.band_window,
        )
        self.weight_norm = float(np.sum(self.weights))
        if self.weight_norm <= 0.0:
            raise ValueError("Band weights are all zero; increase --band_window or adjust --weight_alpha")

    def h_model(self, params: Sequence[float]) -> np.ndarray:
        p = np.asarray(params, dtype=np.float64).reshape(-1)
        if p.size != self.ls_templates.shape[0]:
            raise ValueError(f"Expected {self.ls_templates.shape[0]} parameters, got {p.size}")
        hsoc = np.einsum("a,aij->ij", p, self.ls_templates, optimize=True)
        return self.h_base + hsoc[None, :, :]

    def eigensystem(self, params: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        evals, evecs = np.linalg.eigh(self.h_model(params))
        return evals, evecs

    def residual_data(self, params: Sequence[float]):
        evals, evecs = self.eigensystem(params)
        rel_model, ref_model = relative_energies(evals, self.model_efermi)
        residual = rel_model - self.rel_target
        return evals, evecs, rel_model, ref_model, residual

    def loss(self, params: Sequence[float]) -> float:
        _evals, _evecs, _rel, _ref, residual = self.residual_data(params)
        return float(np.sum(self.weights * residual * residual) / self.weight_norm)

    def gradient(self, params: Sequence[float]) -> np.ndarray:
        _evals, evecs, _rel, ref_model, residual = self.residual_data(params)
        deps = np.einsum(
            "kib,aij,kjb->akb",
            evecs.conj(),
            self.ls_templates,
            evecs,
            optimize=True,
        ).real
        dref = deps[:, np.arange(deps.shape[1]), ref_model]
        drel = deps - dref[:, :, None]
        return 2.0 * np.einsum("kb,akb->a", self.weights * residual, drel, optimize=True) / self.weight_norm

    def rms(self, params: Sequence[float]) -> float:
        return float(np.sqrt(max(self.loss(params), 0.0)))


def fit_soc_parameters(fitter: SOCParameterFitter, x0, bounds, *, maxiter=200, ftol=1e-12, gtol=1e-8):
    from scipy.optimize import minimize

    return minimize(
        fun=fitter.loss,
        x0=np.asarray(x0, dtype=np.float64),
        jac=fitter.gradient,
        method="L-BFGS-B",
        bounds=list(bounds),
        options={"maxiter": int(maxiter), "ftol": float(ftol), "gtol": float(gtol)},
    )


def write_model_hr(path: str, h_base_r: dict, ls_templates: np.ndarray, params: Sequence[float], *, header="SLW SOC parameter model"):
    nspin = int(next(iter(h_base_r.values())).shape[0])
    hsoc = np.einsum("a,aij->ij", np.asarray(params, dtype=np.float64), np.asarray(ls_templates), optimize=True)
    out = {tuple(r): np.asarray(block, dtype=np.complex128).copy() for r, block in h_base_r.items()}
    onsite = (0, 0, 0)
    out.setdefault(onsite, np.zeros((nspin, nspin), dtype=np.complex128))
    out[onsite] = out[onsite] + hsoc
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    write_wannier_hr(path, nspin, [1] * len(out), out, header=header)


def _spinor_base_hr_from_collinear(h_up: dict, h_dn: dict, nwan: int, spin_direction) -> dict:
    keys = sorted(set(h_up) | set(h_dn))
    zero = np.zeros((int(nwan), int(nwan)), dtype=np.complex128)
    out = {}
    for r in keys:
        up = np.asarray(h_up.get(r, zero), dtype=np.complex128)
        dn = np.asarray(h_dn.get(r, zero), dtype=np.complex128)
        out[tuple(r)] = spinor_from_collinear(up, dn, n=spin_direction)
    return out


def build_arg_parser():
    ap = argparse.ArgumentParser(description="Fit onsite SOC parameters to a target SOC Wannier hr.dat")
    ap.add_argument("--input", "--in_file", default=None, help="SOC fitting .in file")
    ap.add_argument("--up_hr", default=None)
    ap.add_argument("--dn_hr", default=None)
    ap.add_argument("--target_soc_hr", default=None)
    ap.add_argument("--win", default=None)
    ap.add_argument("--soc_params", default=None, help="Short CLI form, e.g. 'Cr:d:0.05:0:0.5;I:p:0.5:0:2.0'")
    ap.add_argument("--efermi", type=float, default=None, help="Compatibility shortcut: use the same Fermi level for model and target")
    ap.add_argument("--model_efermi", type=float, default=None, help="Fermi level for the collinear model Hamiltonian")
    ap.add_argument("--target_efermi", type=float, default=None, help="Fermi level for the target SCF SOC Hamiltonian")
    ap.add_argument("--spin_direction", type=float, nargs=3, default=None)
    ap.add_argument("--kmesh", type=int, nargs=3, default=None)
    ap.add_argument("--p_order", default=None)
    ap.add_argument("--d_order", default=None)
    ap.add_argument("--weight_alpha", type=float, default=None)
    ap.add_argument("--band_window", type=int, default=None, help="Bands around target VBM index; negative uses all bands")
    ap.add_argument("--maxiter", type=int, default=None)
    ap.add_argument("--ftol", type=float, default=None)
    ap.add_argument("--gtol", type=float, default=None)
    ap.add_argument("--out_hr", default=None, help="Optional fitted model spinor hr.dat")
    ap.add_argument("--report", default=None, help="Optional text report")
    return ap


def run(args):
    args = _resolve_cli_or_input(args)
    dim_up, _deg_up, h_up = _read_wannier_hr_compat(args.up_hr)
    dim_dn, _deg_dn, h_dn = _read_wannier_hr_compat(args.dn_hr)
    dim_t, _deg_t, h_t = _read_wannier_hr_compat(args.target_soc_hr)
    if dim_up != dim_dn:
        raise ValueError(f"up/dn dimension mismatch: {dim_up} vs {dim_dn}")
    if dim_t != 2 * dim_up:
        raise ValueError(f"target_soc_hr dim={dim_t}, expected 2*nwan={2*dim_up}")

    specs = parse_soc_parameter_specs(args.soc_params)
    ls_templates, summaries = build_ls_templates(args.win, dim_up, specs, p_order=args.p_order, d_order=args.d_order)
    kpts = _full_k_mesh(tuple(int(x) for x in args.kmesh))
    hk_up = _hmap_to_hk(h_up, kpts, dim_up)
    hk_dn = _hmap_to_hk(h_dn, kpts, dim_dn)
    hk_t = _hmap_to_hk(h_t, kpts, dim_t)

    if args.model_efermi is None and args.efermi is None:
        raise ValueError("Provide --model_efermi, or --efermi as a same-Fermi shortcut")
    if args.target_efermi is None and args.efermi is None:
        raise ValueError("Provide --target_efermi, or --efermi as a same-Fermi shortcut")
    model_efermi = float(args.model_efermi if args.model_efermi is not None else args.efermi)
    target_efermi = float(args.target_efermi if args.target_efermi is not None else args.efermi)

    band_window = None if int(args.band_window) < 0 else int(args.band_window)
    fitter = SOCParameterFitter(
        hk_up,
        hk_dn,
        hk_t,
        ls_templates,
        model_efermi=model_efermi,
        target_efermi=target_efermi,
        spin_direction=args.spin_direction,
        alpha=float(args.weight_alpha),
        band_window=band_window,
    )
    x0 = np.asarray([s.lambda0 for s in specs], dtype=np.float64)
    bounds = [(s.lower, s.upper) for s in specs]
    print(f"[soc-parameter] nk={len(kpts)} nwan={dim_up} spinor_dim={dim_t}", flush=True)
    for spec, summary in zip(specs, summaries):
        print(f"[soc-parameter] param {spec.label}: x0={spec.lambda0:g} bounds=({spec.lower:g},{spec.upper:g}) groups={summary}", flush=True)
    print(f"[soc-parameter] initial loss={fitter.loss(x0):.12e} rms={fitter.rms(x0):.12e}", flush=True)

    result = fit_soc_parameters(fitter, x0, bounds, maxiter=args.maxiter, ftol=args.ftol, gtol=args.gtol)
    print(f"[soc-parameter] success={bool(result.success)} status={result.status} message={result.message}", flush=True)
    print(f"[soc-parameter] final loss={float(result.fun):.12e} rms={fitter.rms(result.x):.12e}", flush=True)
    for spec, val in zip(specs, result.x):
        print(f"[soc-parameter] lambda {spec.label} = {float(val):.12g} eV", flush=True)

    if args.out_hr:
        base_hr = _spinor_base_hr_from_collinear(h_up, h_dn, dim_up, args.spin_direction)
        write_model_hr(
            args.out_hr,
            base_hr,
            ls_templates,
            result.x,
            header="SLW fitted onsite SOC model; " + "; ".join(f"{s.label}={v:.12g}" for s, v in zip(specs, result.x)),
        )
        print(f"[soc-parameter] wrote {args.out_hr}", flush=True)

    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)) or ".", exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            f.write("# SLW SOC parameter fitting report\n")
            f.write(f"success = {bool(result.success)}\n")
            f.write(f"message = {result.message}\n")
            f.write(f"kmesh = {tuple(int(x) for x in args.kmesh)}\n")
            f.write(f"model_efermi = {model_efermi:.12g}\n")
            f.write(f"target_efermi = {target_efermi:.12g}\n")
            f.write(f"weight_alpha = {float(args.weight_alpha):.12g}\n")
            f.write(f"band_window = {band_window}\n")
            f.write(f"initial_loss = {fitter.loss(x0):.12e}\n")
            f.write(f"final_loss = {float(result.fun):.12e}\n")
            f.write(f"final_rms = {fitter.rms(result.x):.12e}\n")
            for spec, val, summary in zip(specs, result.x, summaries):
                f.write(f"lambda_{spec.label} = {float(val):.12g}\n")
                f.write(f"groups_{spec.label} = {summary}\n")
        print(f"[soc-parameter] wrote {args.report}", flush=True)
    return result


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
