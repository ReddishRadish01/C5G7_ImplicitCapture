import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

PATH = "fluxTally_1_200cycle.txt"
K_TARGET = 10
PIN_RADIUS_CM = 0.54

def parse_core_tally(path, k_target):
    header = {}
    assemblies = {}

    current_a = None
    meta = None

    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue

            if s.startswith("CoreSize"):
                parts = s.split()
                header["core_size"] = tuple(map(float, parts[1:4]))
                continue

            if s.startswith("AssemblyNo"):
                header["assembly_no"] = int(s.split()[1])
                continue

            m = re.match(r"^Assembly\s+(\d+)$", s)
            if m:
                current_a = int(m.group(1))
                meta = {"a": current_a}
                assemblies[current_a] = {"meta": meta, "rows": []}
                continue

            if s.startswith("StartPos"):
                _, x, y, z = s.split()
                meta["start"] = (float(x), float(y), float(z))
                continue

            if s.startswith("Length"):
                parts = s.split()
                meta["length"] = (float(parts[1]), float(parts[2]), float(parts[3]))
                continue

            if s.startswith("Dims"):
                _, nx, ny, nz = s.split()
                meta["dims"] = (int(nx), int(ny), int(nz))
                continue

            if s.startswith("Format"):
                continue

            parts = s.split()
            if len(parts) == 6:
                a, k, j, i = map(int, parts[:4])
                if k == k_target:
                    pin = float(parts[4])
                    mod = float(parts[5])
                    assemblies[a]["rows"].append((j, i, pin, mod))

    return header, assemblies


def build_cell_records(header, assemblies):
    cells = []
    core_x, core_y, _ = header["core_size"]

    for a, blob in assemblies.items():
        meta = blob["meta"]
        sx, sy, _ = meta["start"]
        Lx, Ly, _ = meta["length"]
        nx, ny, _ = meta["dims"]

        dx = Lx / nx
        dy = Ly / ny

        for (j, i, pin, mod) in blob["rows"]:
            x0 = sx + i * dx
            y0 = sy + j * dy
            cells.append((x0, y0, dx, dy, pin, mod))

    return (core_x, core_y), cells


def plot_pin_mod_one_colorbar(coredims, cells, pin_radius_cm, cmap="viridis"):
    core_x, core_y = coredims

    rects = []
    rect_vals = []
    circs = []
    circ_vals = []

    for (x0, y0, dx, dy, pin, mod) in cells:
        rects.append(Rectangle((x0, y0), dx, dy))
        rect_vals.append(mod)

        if pin != 0.0:
            cx = x0 + 0.5 * dx
            cy = y0 + 0.5 * dy
            r = min(pin_radius_cm, 0.5 * min(dx, dy))
            circs.append(Circle((cx, cy), r))
            circ_vals.append(pin)

    rect_vals = np.asarray(rect_vals, dtype=float)
    circ_vals = np.asarray(circ_vals, dtype=float)

    both = rect_vals if circ_vals.size == 0 else np.concatenate([rect_vals, circ_vals])
    shared_norm = Normalize(vmin=np.nanmin(both), vmax=np.nanmax(both))

    fig, ax = plt.subplots()

    rect_coll = PatchCollection(rects, array=rect_vals, cmap=cmap, norm=shared_norm, linewidths=0.2)
    ax.add_collection(rect_coll)

    if circs:
        circ_coll = PatchCollection(circs, array=circ_vals, cmap=cmap, norm=shared_norm, linewidths=0.0)
        ax.add_collection(circ_coll)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0.0, core_x)
    ax.set_ylim(0.0, core_y)
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_title("modTally (squares) + pinTally (circles)")

    cbar = fig.colorbar(ScalarMappable(norm=shared_norm, cmap=cmap), ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("tally value (shared scale)")

    plt.tight_layout()
    plt.show()



header, assemblies = parse_core_tally(PATH, K_TARGET)
coredims, cells = build_cell_records(header, assemblies)
plot_pin_mod_one_colorbar(coredims, cells, PIN_RADIUS_CM, cmap="viridis")

