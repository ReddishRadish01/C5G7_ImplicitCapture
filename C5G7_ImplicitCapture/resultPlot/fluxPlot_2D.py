import re
import math
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.colors import Normalize


@dataclass
class AssemblyBlock:
    a: int
    start: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    length: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    dims: Tuple[int, int, int] = (0, 0, 0)  # nx, ny, nz
    # map: (k, j, i) -> (pin, mod)
    data: Dict[Tuple[int, int, int], Tuple[float, float]] = field(default_factory=dict)


@dataclass
class CycleBlock:
    cycle_no: int
    keff: Optional[float] = None
    core_size: Optional[Tuple[float, float, float]] = None
    assembly_no: Optional[int] = None
    assemblies: Dict[int, AssemblyBlock] = field(default_factory=dict)


_cycle_re = re.compile(r"^\s*(\d+)\s+th\s+cycle:\s*$")
_keff_re = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s+multiplication\s+Factor\s*$")
_core_re = re.compile(r"^\s*CoreSize\(cm\)\s+([0-9eE\.\+\-]+)\s+([0-9eE\.\+\-]+)\s+([0-9eE\.\+\-]+)\s*$")
_asmno_re = re.compile(r"^\s*AssemblyNo\s+(\d+)\s*$")
_asm_re = re.compile(r"^\s*Assembly\s+(\d+)\s*$")
_start_re = re.compile(r"^\s*StartPos\s+([0-9eE\.\+\-]+)\s+([0-9eE\.\+\-]+)\s+([0-9eE\.\+\-]+)\s*$")
_len_re = re.compile(r"^\s*Length\s+([0-9eE\.\+\-]+)\s+([0-9eE\.\+\-]+)\s+([0-9eE\.\+\-]+)\s*$")
_dims_re = re.compile(r"^\s*Dims\s+(\d+)\s+(\d+)\s+(\d+)\s*$")
_format_re = re.compile(r"^\s*Format\s+a\s+k\s+j\s+i\s+pinTally\s+modTally\s*$")


def parse_flux_tally_file(path: str) -> List[CycleBlock]:
    cycles: List[CycleBlock] = []
    cur_cycle: Optional[CycleBlock] = None
    cur_asm: Optional[AssemblyBlock] = None
    in_data = False

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            m = _core_re.match(line)
            if m and cur_cycle is not None:
                cur_cycle.core_size = (float(m.group(1)), float(m.group(2)), float(m.group(3)))
                continue

            m = _asmno_re.match(line)
            if m and cur_cycle is not None:
                cur_cycle.assembly_no = int(m.group(1))
                continue

            m = _cycle_re.match(line)
            if m:
                cur_cycle = CycleBlock(cycle_no=int(m.group(1)))
                cycles.append(cur_cycle)
                cur_asm = None
                in_data = False
                continue

            if cur_cycle is None:
                continue

            m = _keff_re.match(line)
            if m:
                cur_cycle.keff = float(m.group(1))
                continue

            m = _asm_re.match(line)
            if m:
                a = int(m.group(1))
                cur_asm = cur_cycle.assemblies.get(a)
                if cur_asm is None:
                    cur_asm = AssemblyBlock(a=a)
                    cur_cycle.assemblies[a] = cur_asm
                in_data = False
                continue

            if cur_asm is not None:
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
                    if len(parts) == 6:
                        try:
                            a = int(parts[0])
                            k = int(parts[1])
                            j = int(parts[2])
                            i = int(parts[3])
                            pin = float(parts[4])
                            mod = float(parts[5])
                        except ValueError:
                            continue
                        asm = cur_cycle.assemblies.get(a)
                        if asm is None:
                            asm = AssemblyBlock(a=a)
                            cur_cycle.assemblies[a] = asm
                        asm.data[(k, j, i)] = (pin, mod)
                    continue

    return cycles


def make_shared_norm(values, mode="percentile", p_lo=1.0, p_hi=99.5, log=False):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]

    # ignore zeros when log is requested (log(0) invalid)
    if log:
        v = v[v > 0.0]

    if v.size == 0:
        return Normalize(vmin=0.0, vmax=1.0)

    if mode == "percentile":
        vmin = np.percentile(v, p_lo) if p_lo is not None else np.min(v)
        vmax = np.percentile(v, p_hi) if p_hi is not None else np.max(v)
    elif mode == "std":
        mu = np.mean(v)
        sigma = np.std(v)
        vmin = max(0.0, mu - 2.0 * sigma)
        vmax = mu + 3.0 * sigma
    else:
        vmin = np.min(v)
        vmax = np.max(v)

    if vmax <= vmin:
        vmax = vmin + 1.0

    if log:
        return LogNorm(vmin=vmin, vmax=vmax)
    return Normalize(vmin=vmin, vmax=vmax)

def plot_cycle_k(
    cycle: CycleBlock,
    k_slice: int,
    pin_radius_cm: float = 0.54,
    outpath: Optional[str] = None,
    show: bool = False,
):
    patches = []
    values = []

    # First pass: collect all pin+mod values (for shared normalization)
    for asm in cycle.assemblies.values():
        nx, ny, nz = asm.dims
        if nx <= 0 or ny <= 0:
            continue
        for (k, j, i), (pin, mod) in asm.data.items():
            if k != k_slice:
                continue
            values.append(pin)
            values.append(mod)

    vmax = max(values) if values else 1.0
    norm = make_shared_norm(values, mode="percentile", p_lo=0.0, p_hi=99.9, log=False)

    fig, ax = plt.subplots(figsize=(9, 9))

    # Second pass: draw squares (mod) and circles (pin>0)
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

            # square = moderator
            sq = Rectangle(
                (x0, y0),
                dx,
                dy,
                linewidth=0.2,
                edgecolor="k",
                facecolor=plt.cm.viridis(norm(mod)),
            )
            ax.add_patch(sq)

            # circle = pin, only if pin != 0
            if pin != 0.0:
                circ = Circle(
                    (cx, cy),
                    radius=pin_radius_cm,
                    linewidth=0.05,
                    edgecolor="k",
                    facecolor=plt.cm.viridis(norm(pin)),
                )
                ax.add_patch(circ)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")

    title = f"Cycle {cycle.cycle_no}, z Pos={1.26 * (k_slice + 1)} (cm)"
    if cycle.keff is not None:
        title += f", keff={cycle.keff:.6f}"
    ax.set_title(title)

    # one shared colorbar (same norm used for both pin & mod)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Flux tally (shared scale for mod+pin)")

    # optional: set view limits to core size if present
    if cycle.core_size is not None:
        cx, cy, _ = cycle.core_size
        ax.set_xlim(0.0, cx)
        ax.set_ylim(0.0, cy)

    plt.tight_layout()

    if outpath is not None:
        plt.savefig(outpath, dpi=600)
    if show:
        plt.show()
    plt.close(fig)


def make_all_plots(
    infile: str,
    out_dir: str = ".",
    cycles_to_plot: Optional[List[int]] = None,
):
    cycles = parse_flux_tally_file(infile)

    for cyc in cycles:
        if cycles_to_plot is not None and cyc.cycle_no not in cycles_to_plot:
            continue

        ks = set()
        for asm in cyc.assemblies.values():
            for (k, j, i) in asm.data.keys():
                ks.add(k)

        for k in sorted(ks):
            outpath = f"{out_dir}/flux_cycle{cyc.cycle_no}_k{k}.png"
            plot_cycle_k(cyc, k_slice=k, outpath=outpath, show=False)


if __name__ == "__main__":
    # Example:
    # make_all_plots("flux_Tally20260126_182315.txt", out_dir="plots")
    make_all_plots("flux_Tally20260127_030612.txt", out_dir="plots")
