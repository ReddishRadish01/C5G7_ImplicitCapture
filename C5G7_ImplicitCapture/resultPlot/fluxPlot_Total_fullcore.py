import os
import re
import argparse
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.colors import PowerNorm, Normalize


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


def available_k_slices(cycle: CycleBlock) -> List[int]:
    ks = set()
    for asm in cycle.assemblies.values():
        for (k, _j, _i) in asm.data.keys():
            ks.add(k)
    return sorted(ks)


def _collect_values_for_k(cycle: CycleBlock, k_slice: int, tally_mode: str) -> np.ndarray:
    vals = []
    for asm in cycle.assemblies.values():
        for (k, _j, _i), (pin, mod) in asm.data.items():
            if k != k_slice:
                continue
            if tally_mode in ("total", "pin"):
                vals.append(pin)
            if tally_mode in ("total", "mod"):
                vals.append(mod)
    return np.asarray(vals, dtype=float)


def _infer_quarter_size(cycle: CycleBlock) -> Tuple[float, float]:
    if cycle.core_size is not None:
        return cycle.core_size[0], cycle.core_size[1]

    xmax = 0.0
    ymax = 0.0
    for asm in cycle.assemblies.values():
        sx, sy, _ = asm.start
        lx, ly, _ = asm.length
        xmax = max(xmax, sx + lx)
        ymax = max(ymax, sy + ly)
    return xmax, ymax


def plot_cycle_k_power01(
    cycle: CycleBlock,
    k_slice: int,
    pin_radius_cm: float = 0.54,
    outpath: Optional[str] = None,
    show: bool = False,
    full_core_xy: bool = True,
    cmap_name: str = "turbo",
    gamma: float = 1.5,
    tally_mode: str = "total",
    z_pitch_cm: float = 1.26,
):
    vals = _collect_values_for_k(cycle, k_slice, tally_mode)
    if vals.size == 0:
        raise RuntimeError(
            f"No data found for k={k_slice}. Available k slices: {available_k_slices(cycle)}\n"
            "If only one k appears, your C++ DumpCoreTallyToText probably dumped only that targetZCell."
        )

    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        raise RuntimeError(f"All values are NaN/Inf for k={k_slice}")

    # Keep zero visible, but avoid one huge outlier destroying contrast.
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if vmax <= vmin:
        vmax = vmin + 1.0

    cmap = plt.get_cmap(cmap_name)
    if abs(gamma - 1.0) < 1.0e-12:
        norm = Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)

    Lx, Ly = _infer_quarter_size(cycle)
    if Lx <= 0.0 or Ly <= 0.0:
        raise RuntimeError("Could not infer core x/y size from tally file.")

    fig_size = (10.5, 9.5) if full_core_xy else (9.0, 9.0)
    fig, ax = plt.subplots(figsize=fig_size)

    def add_rect_and_pin(x0, y0, dx, dy, cx, cy, pin, mod):
        if full_core_xy:
            # Plot full core shifted to [0, 2Lx] x [0, 2Ly].
            # Original quarter-core [0,Lx]x[0,Ly] is drawn in the upper-right quadrant.
            rects = [
                (Lx + x0,      Ly + y0),       # +x, +y
                (Lx - x0 - dx, Ly + y0),       # -x, +y mirrored
                (Lx + x0,      Ly - y0 - dy),  # +x, -y mirrored
                (Lx - x0 - dx, Ly - y0 - dy),  # -x, -y mirrored
            ]
            centers = [
                (Lx + cx, Ly + cy),
                (Lx - cx, Ly + cy),
                (Lx + cx, Ly - cy),
                (Lx - cx, Ly - cy),
            ]
        else:
            rects = [(x0, y0)]
            centers = [(cx, cy)]

        if tally_mode in ("total", "mod") and np.isfinite(mod):
            for rx0, ry0 in rects:
                ax.add_patch(Rectangle(
                    (rx0, ry0),
                    dx,
                    dy,
                    linewidth=0.02,
                    edgecolor="k",
                    facecolor=cmap(norm(mod)),
                ))

        if tally_mode in ("total", "pin") and pin != 0.0 and np.isfinite(pin):
            for pcx, pcy in centers:
                ax.add_patch(Circle(
                    (pcx, pcy),
                    radius=pin_radius_cm,
                    linewidth=0.1,
                    alpha=0.75,
                    edgecolor="k",
                    facecolor=cmap(norm(pin)),
                ))

    for asm in cycle.assemblies.values():
        nx, ny, _nz = asm.dims
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
            add_rect_and_pin(x0, y0, dx, dy, cx, cy, pin, mod)

    ax.set_aspect("equal", adjustable="box")

    z_center = (k_slice + 0.5) * z_pitch_cm
    title = f"Cycle {cycle.cycle_no if cycle.cycle_no is not None else ''}, k={k_slice}, z center={z_center:.3f} cm"
    if cycle.keff is not None:
        title += f", keff={cycle.keff:.6f}"
    if full_core_xy:
        title += " | full core mirrored from x/y reflective quarter-core"
    ax.set_title(title)

    if full_core_xy:
        ax.set_xlim(0.0, 2.0 * Lx)
        ax.set_ylim(0.0, 2.0 * Ly)
        ax.axvline(Lx, color="k", linewidth=0.8)
        ax.axhline(Ly, color="k", linewidth=0.8)
        ax.set_xlabel("x (cm), mirrored full core")
        ax.set_ylabel("y (cm), mirrored full core")
    else:
        ax.set_xlim(0.0, Lx)
        ax.set_ylim(0.0, Ly)
        ax.set_xlabel("x (cm), quarter core")
        ax.set_ylabel("y (cm), quarter core")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"Fission Rate ({tally_mode}, gamma={gamma:g})")

    plt.tight_layout()

    if outpath is not None:
        outdir = os.path.dirname(outpath)
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        plt.savefig(outpath, dpi=1000)
        print(f"saved: {outpath}")

    if show:
        plt.show()

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot C5G7 fission-rate tally, optionally mirrored from quarter-core to full-core.")
    parser.add_argument("--infile", default="flux_Tally20260703_140900.txt", help="Input tally text file")
    parser.add_argument("--out-dir", default="plots_Total", help="Output directory")
    parser.add_argument("--k-slice", type=int, default=0, help="Axial cell index k to plot")
    parser.add_argument("--quarter", action="store_true", help="Plot quarter core only. Default is mirrored full core.")
    parser.add_argument("--cmap", default="turbo", help="Matplotlib colormap name, e.g. turbo, plasma, inferno")
    parser.add_argument("--gamma", type=float, default=1.5, help="PowerNorm gamma. Use 1.0 for linear.")
    parser.add_argument("--pin-radius", type=float, default=0.54, help="Pin radius in cm")
    parser.add_argument("--z-pitch", type=float, default=1.26, help="Axial cell pitch in cm")
    parser.add_argument("--mode", choices=["total", "pin", "mod"], default="total", help="Tally values used for plotting/color scaling")
    parser.add_argument("--show", action="store_true", help="Show matplotlib window")
    args = parser.parse_args()

    cycle = parse_single_cycle_file(args.infile)
    full_core = not args.quarter

    suffix = "fullcore" if full_core else "quarter"
    outname = f"fission_rate_{suffix}_cycle{cycle.cycle_no}_k{args.k_slice}_{args.cmap}_g{args.gamma:g}.png"
    outpath = os.path.join(args.out_dir, outname)

    plot_cycle_k_power01(
        cycle,
        k_slice=args.k_slice,
        pin_radius_cm=args.pin_radius,
        outpath=outpath,
        show=args.show,
        full_core_xy=full_core,
        cmap_name=args.cmap,
        gamma=args.gamma,
        tally_mode=args.mode,
        z_pitch_cm=args.z_pitch,
    )


if __name__ == "__main__":
    main()
