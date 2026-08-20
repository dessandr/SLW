import numpy as np

BOHR_TO_ANG = 0.529177210903


def _strip_inline_comment(line):
    return line.split("#", 1)[0].split("!", 1)[0].strip()


def _card_name(line):
    stripped = _strip_inline_comment(line)
    if not stripped:
        return ""
    return stripped.split()[0].strip().upper()


def _card_unit(line, default):
    tokens = _strip_inline_comment(line).replace("(", " ").replace(")", " ").split()
    if len(tokens) < 2:
        return default
    return tokens[1].lower()


def _find_card(lines, names):
    names = {name.upper() for name in names}
    for idx, line in enumerate(lines):
        if _card_name(line) in names:
            return idx
    return None


def _read_three_vectors(lines, start, unit):
    rows = []
    for line in lines[start:start + 3]:
        clean = _strip_inline_comment(line)
        if not clean:
            raise ValueError("CELL_PARAMETERS card has blank vector line")
        rows.append([float(x) for x in clean.split()[:3]])
    arr = np.asarray(rows, dtype=np.float64)
    if unit in {"bohr", "a.u.", "au"}:
        arr *= BOHR_TO_ANG
    elif unit in {"angstrom", "ang", "angs", "a"}:
        pass
    else:
        raise ValueError(
            f"Unsupported CELL_PARAMETERS unit '{unit}'. Use angstrom or bohr."
        )
    return arr


def _numbered_atom_labels(species):
    counts = {}
    labels = []
    for sp in species:
        counts[sp] = counts.get(sp, 0) + 1
        labels.append(f"{sp}{counts[sp]}")
    return labels


def _parse_card_structure(path, lines):
    cell_idx = _find_card(lines, {"CELL_PARAMETERS"})
    pos_idx = _find_card(lines, {"ATOMIC_POSITIONS", "ATOMIC_POSITION"})
    if cell_idx is None or pos_idx is None:
        return None

    cell_unit = _card_unit(lines[cell_idx], "angstrom")
    r_vec = _read_three_vectors(lines, cell_idx + 1, cell_unit)

    pos_unit = _card_unit(lines[pos_idx], "crystal")
    inv_r_vec = np.linalg.inv(r_vec)
    species = []
    frac_or_cart = []
    for line in lines[pos_idx + 1:]:
        clean = _strip_inline_comment(line)
        if not clean:
            continue
        name = _card_name(clean)
        if name in {
            "CELL_PARAMETERS",
            "ATOMIC_POSITIONS",
            "ATOMIC_POSITION",
            "K_POINTS",
            "OCCUPATIONS",
            "CONSTRAINTS",
            "ATOMIC_SPECIES",
            "ATOMIC_FORCES",
        }:
            break
        parts = clean.split()
        if len(parts) < 4:
            continue
        try:
            xyz = np.asarray([float(x) for x in parts[1:4]], dtype=np.float64)
        except ValueError:
            break
        species.append(parts[0])
        frac_or_cart.append(xyz)

    if not species:
        raise ValueError(f"ATOMIC_POSITIONS card in {path} has no atoms")

    coords = np.asarray(frac_or_cart, dtype=np.float64)
    if pos_unit in {"crystal", "cryst", "fractional", "frac", "direct"}:
        atom_pos = coords @ r_vec
    elif pos_unit in {"angstrom", "ang", "angs", "cartesian", "cart", "a"}:
        atom_pos = coords
    elif pos_unit in {"bohr", "a.u.", "au"}:
        atom_pos = coords * BOHR_TO_ANG
    elif pos_unit == "alat":
        raise ValueError(
            "ATOMIC_POSITIONS alat is not supported without an explicit alat value. "
            "Use crystal or angstrom."
        )
    else:
        raise ValueError(
            f"Unsupported ATOMIC_POSITIONS unit '{pos_unit}'. "
            "Use crystal, angstrom/cartesian, or bohr."
        )

    # Wrap tiny numerical drift for crystal-derived cartesian positions only.
    if pos_unit in {"crystal", "cryst", "fractional", "frac", "direct"}:
        frac = (atom_pos @ inv_r_vec) % 1.0
        atom_pos = frac @ r_vec

    return r_vec, _numbered_atom_labels(species), list(atom_pos)


def parsing_POSCAR(poscar):
    with open(poscar) as f:
        lines = f.readlines()

    card_structure = _parse_card_structure(poscar, lines)
    if card_structure is not None:
        return card_structure

    scale = float(lines[1].split()[0])
    r_vec = scale * np.array(
        [
            list(map(float, lines[2].split()[:3])),
            list(map(float, lines[3].split()[:3])),
            list(map(float, lines[4].split()[:3])),
        ],
        dtype=np.float64,
    )
    species = lines[5].split()
    counts = list(map(int, lines[6].split()))
    atom_list = []
    for sp, n in zip(species, counts):
        for i in range(n):
            atom_list.append(f"{sp}{i + 1}")

    coord_line = 7
    if lines[coord_line].strip().lower().startswith("s"):
        coord_line += 1
    mode = lines[coord_line].strip().lower()
    start = coord_line + 1
    atom_pos = []
    for line in lines[start:start + len(atom_list)]:
        x = np.array(list(map(float, line.split()[:3])), dtype=np.float64)
        if mode.startswith("d"):
            atom_pos.append(x @ r_vec)
        else:
            atom_pos.append(x * scale)
    return r_vec, atom_list, atom_pos


def generate_k_mesh_flat(nx, ny, nz, shift=False):
    if shift:
        kx = np.linspace(0.0, 1.0, int(nx), endpoint=False) + 1.0 / (2.0 * int(nx))
        ky = np.linspace(0.0, 1.0, int(ny), endpoint=False) + 1.0 / (2.0 * int(ny))
        kz = np.linspace(0.0, 1.0, int(nz), endpoint=False) + 1.0 / (2.0 * int(nz))
    else:
        kx = np.linspace(0.0, 1.0, int(nx), endpoint=False)
        ky = np.linspace(0.0, 1.0, int(ny), endpoint=False)
        kz = np.linspace(0.0, 1.0, int(nz), endpoint=False)
    grid = np.meshgrid(kx, ky, kz, indexing="ij")
    return np.stack([g.ravel() for g in grid], axis=1) % 1.0


def generate_k_path(points_dict, path_sequence, g_vec, n_points=50):
    k_list = []
    distances = []
    current_dist = 0.0
    for iseg in range(len(path_sequence) - 1):
        start = np.asarray(points_dict[path_sequence[iseg]], dtype=np.float64)
        end = np.asarray(points_dict[path_sequence[iseg + 1]], dtype=np.float64)
        segment = np.linspace(start, end, int(n_points), endpoint=False)
        for kpt in segment:
            k_list.append(kpt)
            if len(k_list) > 1:
                current_dist += float(np.linalg.norm((k_list[-1] - k_list[-2]) @ g_vec))
            distances.append(current_dist)
    last = np.asarray(points_dict[path_sequence[-1]], dtype=np.float64)
    if k_list:
        current_dist += float(np.linalg.norm((last - k_list[-1]) @ g_vec))
    k_list.append(last)
    distances.append(current_dist)
    return np.asarray(k_list, dtype=np.float64), np.asarray(distances, dtype=np.float64)


def calculate_spectral_function(
    sigma_kw,
    magnon_k,
    omega_axis,
    eta_A=0.05,
    metric_diag=None,
):
    sigma_raw = np.asarray(sigma_kw, dtype=np.complex128)
    e_mag_k = np.asarray(magnon_k, dtype=np.float64)
    omega_axis = np.asarray(omega_axis, dtype=np.float64)
    nk, nw = sigma_raw.shape[0], sigma_raw.shape[1]
    matrix_dim = int(sigma_raw.shape[2])
    if sigma_raw.shape[3] != matrix_dim:
        raise ValueError(f"sigma_kw must be square in the last two axes, got {sigma_raw.shape}")
    if e_mag_k.shape[1] != matrix_dim:
        raise ValueError(f"magnon_k channel mismatch: e_mag={e_mag_k.shape}, sigma={sigma_raw.shape}")
    if metric_diag is None:
        metric = np.ones(matrix_dim, dtype=np.float64)
    else:
        metric = np.asarray(metric_diag, dtype=np.float64).reshape(-1)
        if metric.shape != (matrix_dim,):
            raise ValueError(
                f"metric_diag shape {metric.shape} is incompatible with matrix_dim={matrix_dim}"
            )
    eta_matrix = np.diag(metric.astype(np.complex128))
    out = np.zeros((nk, nw), dtype=np.float64)
    for ik in range(nk):
        # e_mag_k stores signed dynamical poles xi=eta*E.  Sigma is assembled
        # as a Hamiltonian-kernel self-energy, so Dyson's equation is
        # G^-1=z*eta-E-Sigma_H with E=eta*xi.
        h0 = np.diag(metric * e_mag_k[ik]).astype(np.complex128)
        for iw, w in enumerate(omega_axis):
            g_inv = (
                (float(w) + 1j * float(eta_A)) * eta_matrix
                - h0
                - sigma_raw[ik, iw]
            )
            try:
                g = np.linalg.inv(g_inv)
                out[ik, iw] = -np.imag(np.trace(eta_matrix @ g)) / np.pi
            except np.linalg.LinAlgError:
                out[ik, iw] = 0.0
    return out
