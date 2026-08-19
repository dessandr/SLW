"""
Prepare Wannier files for TB2J SOC calculations.

All material-dependent paths and dimensions are explicit command-line inputs.
"""

import os
import argparse


def _read_centres_xyz(path):
    with open(path, "r") as f:
        lines = f.readlines()
    if len(lines) < 2:
        raise ValueError(f"Invalid centres file (too short): {path}")
    total = int(lines[0].strip())
    body = lines[2:]
    if len(body) < total:
        raise ValueError(f"Invalid centres file (count mismatch): {path}")
    return total, body[:total]


def _split_wann_and_atoms(lines, nwann):
    if nwann <= 0:
        raise ValueError(f"nwann must be positive, got {nwann}")
    if len(lines) < nwann:
        raise ValueError(
            f"Not enough entries to extract {nwann} Wannier centers "
            f"(got {len(lines)} total entries)."
        )
    wann = lines[:nwann]
    atoms = lines[nwann:]
    return wann, atoms


def prep_centres_duplicate(in_file, out_file, nwann_col):
    """Build spinor centres by duplicating one collinear centres file."""
    total, body = _read_centres_xyz(in_file)
    wann_lines, atom_lines = _split_wann_and_atoms(body, nwann_col)
    num_atoms = total - nwann_col

    if num_atoms < 0:
        raise ValueError(
            f"Invalid nwann={nwann_col} for total entries={total} in {in_file}"
        )

    with open(out_file, "w") as f:
        f.write(f"    {nwann_col * 2 + num_atoms}\n")
        f.write(" Wannier centres for SOC spinor\n")
        for l in wann_lines:
            f.write(l)
        for l in wann_lines:
            f.write(l)
        for l in atom_lines:
            f.write(l)

    print(f"Written {out_file} ({nwann_col * 2} Wannier + {num_atoms} atoms)")


def prep_centres_from_up_dn(up_file, dn_file, out_file, nwann_col, atom_tol=1e-6):
    """
    Build spinor centres by combining explicit up/down Wannier centres files.

    Assumes each input has:
      [nwann_col Wannier lines] + [atom lines]
    and that atom lists are identical (within tolerance).
    """
    total_up, body_up = _read_centres_xyz(up_file)
    total_dn, body_dn = _read_centres_xyz(dn_file)
    wann_up, atom_up = _split_wann_and_atoms(body_up, nwann_col)
    wann_dn, atom_dn = _split_wann_and_atoms(body_dn, nwann_col)

    if total_up - nwann_col < 0 or total_dn - nwann_col < 0:
        raise ValueError(
            f"Invalid nwann={nwann_col} for input totals up={total_up}, dn={total_dn}"
        )

    if len(atom_up) != len(atom_dn):
        raise ValueError(
            f"Atom entry count mismatch: up={len(atom_up)}, dn={len(atom_dn)}"
        )

    # Validate atom rows are effectively identical
    for i, (lu, ld) in enumerate(zip(atom_up, atom_dn), start=1):
        pu = lu.split()
        pd = ld.split()
        if len(pu) < 4 or len(pd) < 4:
            if lu.strip() != ld.strip():
                raise ValueError(f"Atom row {i} mismatch:\nup={lu}\ndn={ld}")
            continue
        if pu[0] != pd[0]:
            raise ValueError(f"Atom species mismatch at row {i}: {pu[0]} vs {pd[0]}")
        cu = [float(pu[1]), float(pu[2]), float(pu[3])]
        cd = [float(pd[1]), float(pd[2]), float(pd[3])]
        if max(abs(cu[j] - cd[j]) for j in range(3)) > atom_tol:
            raise ValueError(
                f"Atom coordinate mismatch at row {i} exceeds tol={atom_tol}: {cu} vs {cd}"
            )

    num_atoms = len(atom_up)
    with open(out_file, "w") as f:
        f.write(f"    {len(wann_up) + len(wann_dn) + num_atoms}\n")
        f.write(" Wannier centres for SOC spinor (merged from up/down)\n")
        for l in wann_up:
            f.write(l)
        for l in wann_dn:
            f.write(l)
        for l in atom_up:
            f.write(l)

    print(
        f"Written {out_file} ({len(wann_up)} up + {len(wann_dn)} dn Wannier + {num_atoms} atoms)"
    )


def prep_win(in_file, out_file, nwann_col):
    """Modify .win file for spinor num_wann."""
    with open(in_file, 'r') as f:
        lines = f.readlines()

    with open(out_file, 'w') as f:
        for line in lines:
            if line.strip().startswith("num_wann"):
                f.write(f"num_wann = {nwann_col * 2}\n")
            elif line.strip().startswith("num_bands"):
                f.write(f"num_bands = {nwann_col * 2}\n")
            elif line.strip().startswith("spin"):
                f.write("spin = spinor\n")
            else:
                f.write(line)

    print(f"Modified {out_file} (num_wann = {nwann_col * 2})")


def main():
    parser = argparse.ArgumentParser(description="Prepare TB2J files for SOC")
    parser.add_argument("--nwann", type=int, required=True, help="Collinear num_wann")
    parser.add_argument("--centres", type=str, default=None,
                        help="Single collinear centres file for duplicate mode")
    parser.add_argument("--centres_up", type=str, default=None,
                        help="Spin-up centres file for merge mode")
    parser.add_argument("--centres_dn", type=str, default=None,
                        help="Spin-down centres file for merge mode")
    parser.add_argument("--atom_tol", type=float, default=1e-6,
                        help="Tolerance for atom-coordinate consistency check in merge mode")
    parser.add_argument("--win", type=str, required=True)
    parser.add_argument("--prefix", type=str, required=True)
    args = parser.parse_args()

    out_centres = f"{args.prefix}_centres.xyz"
    if args.centres_up and args.centres_dn:
        prep_centres_from_up_dn(
            args.centres_up,
            args.centres_dn,
            out_centres,
            args.nwann,
            atom_tol=args.atom_tol,
        )
    elif args.centres_up or args.centres_dn:
        parser.error("--centres_up and --centres_dn must be supplied together")
    elif args.centres:
        prep_centres_duplicate(args.centres, out_centres, args.nwann)
    else:
        parser.error("provide --centres, or both --centres_up and --centres_dn")

    prep_win(args.win, f"{args.prefix}.win", args.nwann)


if __name__ == "__main__":
    main()
