import os
import re
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.colors import Normalize, PowerNorm


@dataclass
class AssemblyBlock:
    a: int
    start: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    length: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    dims: Tuple[int, int, int] = (0, 0, 0)
    data: Dict[Tuple[int, int, int], Tuple[float, float]] = field(default_factory=dict)


@dataclass
class CycleBlock:
    cycle_no: Optional[int] = None
    keff: Optional[float] = None
    core_size: Optional[Tuple[float, float, float]] = None
    assembly_no: Optional[int] = None
    assemblies: Dict[int, AssemblyBlock] = field(default_factory=dict)


_core_re = re.compile(r"^\s*CoreSize\(cm\)\s+([0-9eE\.\+\-]+)\s+([0-9eE\.\+\-]+)\s+([0-9eE\.\+\-]+)\s*$")
_asmno_re = re.compile(r"^\s*AssemblyNo\s+(\d+)\s*$")
_cycle_re = re.compile(r"^\s*(\d+)\s+th\s+cycle:\s*$")
_keff_re = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s+multiplication\s+Factor\s*$")
_asm_re = re.compile(r"^\s*Assembly\s+(\d+)\s*$")
_start_re = re.compile(r"^\s*StartPos\s+([0-9eE\.\+\-]+)\s+([0-9eE\.\+\-]+)\s+([0-9eE\.\+\-]+)\s*$")
_len_re = re.compile(r"^\s*Length\s+([0-9eE\.\+\-]+)\s+([0-9eE\.\+\-]+)\s+([0-9eE\.\+\-]+)\s*$")
_dims_re = re.compile(r"^\s*Dims\s+(\d+)\s+(\d+)\s+(\d+)\s*$")
_format_re = re.compile(r"^\s*Format\s+a\s+k\s+j\s+i\s+pinTally\s+modTally\s*$")


def parse_single_cycle_file(path: str) -> CycleBlock:
    cyc = CycleBlock()
    cur_asm: Optional[AssemblyBlock] = None
    in_data = False

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            m = _core_re.match(line)
            if m:
                cyc.core_size = (float(m.group(1)), float(m.group(2)), float(m.group(3)))
                continue

            m = _asmno_re.match(line)
            if m:
                cyc.assembly_no = int(m.group(1))
                continue

            m = _cycle_re.match(line)
            if m:
                cyc.cycle_no = int(m.group(1))
                continue

            m = _keff_re.match(line)
            if m:
                cyc.keff = float(m.group(1))
                continue

            m = _asm_re.match(line)
            if m:
                a = int(m.group(1))
                cur_asm = cyc.assemblies.get(a)
                if cur_asm is None:
                    cur_asm = AssemblyBlock(a=a)
                    cyc.assemblies[a] = cur_asm
                in_data = False
                continue

            if cur_asm is None:
                continue

            m = _start_re.match(line)
            if m:
                cur_asm.start = (float(m.group(1)), float(m.group(2)), float(m.group(3)))
                continue

            m = _len_re.match(line)
            if m:
                cur_asm.length = (float(m.group(1)), float(m.group(2)), float(m.group(3)))
                continue

            m = _dims_re.match(line)
            if m:
                cur_asm.dims = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
                continue

            if _format_re.match(line):
                in_data = True
                continue

            if in_data:
                parts = line.split()
                if len(parts) != 6:
                    continue
                try:
                    a = int(parts[0])
                    k = int(parts[1])
                    j = int(parts[2])
                    i = int(parts[3])
                    pin = float(parts[4])
                    mod = float(parts[5])
                except ValueError:
                    continue
                asm = cyc.assemblies.get(a)
                if asm is None:
                    asm = AssemblyBlock(a=a)
                    cyc.assemblies[a] = asm
                asm.data[(k, j, i)] = (pin, mod)

    return cyc


def minmax01(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    finite = np.isfinite(v)
    out = np.zeros_like(v, dtype=float)
    if not np.any(finite):
        return out
    vmin = np.min(v[finite])
    vmax = np.max(v[finite])
    if vmax <= vmin:
        out[finite] = 1.0
        return out
    out[finite] = (v[finite] - vmin) / (vmax - vmin)
    return out


def plot_cycle_k_power01(
    cycle: CycleBlock,
    k_slice: int,
    pin_radius_cm: float = 0.54,
    outpath: Optional[str] = None,
    show: bool = False,
):
    vals = []
    for asm in cycle.assemblies.values():
        for (k, j, i), (pin, mod) in asm.data.items():
            if k != k_slice:
                continue
            vals.append(pin)
            vals.append(mod)

    if len(vals) == 0:
        raise RuntimeError(f"No data found for k={k_slice}")

    vals = np.asarray(vals, dtype=float)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        raise RuntimeError(f"All values are NaN/Inf for k={k_slice}")

    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if vmax <= vmin:
        vmax = vmin + 1.0
    #norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("turbo")
    norm = PowerNorm(gamma=1.6, vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(9, 9))

    for asm in cycle.assemblies.values():
        nx, ny, nz = asm.dims
        if nx <= 0 or ny <= 0:
            continue

        sx, sy, _ = asm.start
        lx, ly, _ = asm.length
        dx = lx / nx
        dy = ly / ny

        for (k, j, i), (pin, mod) in asm.data.items():
            if k != k_slice:
                continue

            x0 = sx + i * dx
            y0 = sy + j * dy
            cx = x0 + 0.5 * dx
            cy = y0 + 0.5 * dy

            mod01 = (mod - vmin) / (vmax - vmin) if np.isfinite(mod) else 0.0
            mod01 = float(np.clip(mod01, 0.0, 1.0))
            sq = Rectangle(
                (x0, y0),
                dx,
                dy,
                linewidth=0.02,
                edgecolor="k",
                #facecolor=plt.cm.viridis(mod01),
                facecolor=cmap(norm(mod)),
            )
            ax.add_patch(sq)

            if pin != 0.0 and np.isfinite(pin):
                pin01 = (pin - vmin) / (vmax - vmin)
                pin01 = float(np.clip(pin01, 0.0, 1.0))
                circ = Circle(
                    (cx, cy),
                    radius=pin_radius_cm,
                    linewidth=0.1,
                    alpha=0.5,
                    edgecolor="k",
                    #facecolor=plt.cm.viridis(pin01),
                    facecolor=cmap(norm(pin)),

                )
                ax.add_patch(circ)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")

    title = f"Cycle {cycle.cycle_no if cycle.cycle_no is not None else ''}, z Pos={(k_slice + 1)*1.26} (cm)"
    if cycle.keff is not None:
        title += f", keff={cycle.keff:.6f}"
    ax.set_title(title)

    #sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=Normalize(vmin=0.0, vmax=1.0))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=0.0, vmax=1.0))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Power (normalized 0–1, shared for mod+pin)")

    if cycle.core_size is not None:
        cx, cy, _ = cycle.core_size
        ax.set_xlim(0.0, cx)
        ax.set_ylim(0.0, cy)

    plt.tight_layout()

    if outpath is not None:
        os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
        plt.savefig(outpath, dpi=600)

    if show:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    infile = "flux_Tally20260703_140900.txt"
    out_dir = "plots_Total"
    k_slice = 0

    cyc = parse_single_cycle_file(infile)
    outpath = os.path.join(out_dir, f"power01_cycle{cyc.cycle_no}_k{k_slice}.png")
    plot_cycle_k_power01(cyc, k_slice=k_slice, outpath=outpath, show=True)
