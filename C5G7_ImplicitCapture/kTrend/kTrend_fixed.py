#!/usr/bin/env python3
import argparse
import os
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PAIR_RE = re.compile(r"^\s*(\d+)\s+([0-9eE\.\+\-]+)\s*$")
ACTIVE_RE = re.compile(r"^\s*Active\s+cycles:\s*(\d+)\s*$")
KMEAN_RE = re.compile(r"^\s*k_mean:\s*([0-9eE\.\+\-]+)\s*$")
KSTD_RE = re.compile(r"^\s*k_stddev\(cycle\):\s*([0-9eE\.\+\-]+)\s*$")
KSTDERR_RE = re.compile(r"^\s*k_stderr\(mean\):\s*([0-9eE\.\+\-]+)\s*$")
DIFF_RE = re.compile(
    r"^\s*Difference\s+with\s+the\s+reference\s+K:\s*([0-9eE\.\+\-]+),\s*Per\s+cent\s+error:\s*([0-9eE\.\+\-]+)\s*$"
)
PCM_RE = re.compile(
    r"^\s*Reactivity\(rho\):\s*([0-9eE\.\+\-]+),\s*pcm\s+difference:\s*([0-9eE\.\+\-]+)\s*$"
)
TIME_RE = re.compile(
    r"^\s*Total\s+Time:\s*([0-9eE\.\+\-]+)seconds\.\s*Average:\s*([0-9eE\.\+\-]+)\s*seconds\s+per\s+cycle\.\s*$"
)


@dataclass
class KHistory:
    cycles: np.ndarray
    keffs: np.ndarray
    active_cycles_footer: Optional[int] = None
    k_mean_footer: Optional[float] = None
    k_stddev_footer: Optional[float] = None
    k_stderr_footer: Optional[float] = None
    diff_ref_footer: Optional[float] = None
    percent_error_footer: Optional[float] = None
    rho_footer: Optional[float] = None
    pcm_footer: Optional[float] = None
    total_time_footer: Optional[float] = None
    avg_time_footer: Optional[float] = None


def read_k_history(path: str) -> KHistory:
    cycles = []
    keffs = []

    active_cycles_footer = None
    k_mean_footer = None
    k_stddev_footer = None
    k_stderr_footer = None
    diff_ref_footer = None
    percent_error_footer = None
    rho_footer = None
    pcm_footer = None
    total_time_footer = None
    avg_time_footer = None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue

            m = PAIR_RE.match(s)
            if m:
                cycles.append(int(m.group(1)))
                keffs.append(float(m.group(2)))
                continue

            m = ACTIVE_RE.match(s)
            if m:
                active_cycles_footer = int(m.group(1))
                continue

            m = KMEAN_RE.match(s)
            if m:
                k_mean_footer = float(m.group(1))
                continue

            m = KSTD_RE.match(s)
            if m:
                k_stddev_footer = float(m.group(1))
                continue

            m = KSTDERR_RE.match(s)
            if m:
                k_stderr_footer = float(m.group(1))
                continue

            m = DIFF_RE.match(s)
            if m:
                diff_ref_footer = float(m.group(1))
                percent_error_footer = float(m.group(2))
                continue

            m = PCM_RE.match(s)
            if m:
                rho_footer = float(m.group(1))
                pcm_footer = float(m.group(2))
                continue

            m = TIME_RE.match(s)
            if m:
                total_time_footer = float(m.group(1))
                avg_time_footer = float(m.group(2))
                continue

    if not cycles:
        raise RuntimeError(f"No cycle/keff pairs parsed from {path}")

    cycles = np.asarray(cycles, dtype=int)
    keffs = np.asarray(keffs, dtype=float)

    order = np.argsort(cycles)
    cycles = cycles[order]
    keffs = keffs[order]

    return KHistory(
        cycles=cycles,
        keffs=keffs,
        active_cycles_footer=active_cycles_footer,
        k_mean_footer=k_mean_footer,
        k_stddev_footer=k_stddev_footer,
        k_stderr_footer=k_stderr_footer,
        diff_ref_footer=diff_ref_footer,
        percent_error_footer=percent_error_footer,
        rho_footer=rho_footer,
        pcm_footer=pcm_footer,
        total_time_footer=total_time_footer,
        avg_time_footer=avg_time_footer,
    )


def rolling_mean(y: np.ndarray, window: int) -> Optional[np.ndarray]:
    if window <= 1 or y.size < window:
        return None
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(y, kernel, mode="valid")


def infer_active_start_index(hist: KHistory, active_cycles_override: Optional[int]) -> int:
    n = hist.keffs.size

    active_cycles = active_cycles_override
    if active_cycles is None:
        active_cycles = hist.active_cycles_footer

    if active_cycles is None:
        # Fallback: if no footer exists, use all cycles as active.
        return 0

    if active_cycles <= 0:
        raise RuntimeError(f"active cycles must be positive, got {active_cycles}")
    if active_cycles > n:
        raise RuntimeError(f"active cycles ({active_cycles}) exceeds parsed points ({n})")

    return n - active_cycles


def make_out_png(infile: str, out_dir: str, out_png: Optional[str]) -> str:
    if out_png is not None:
        return out_png

    stem = os.path.splitext(os.path.basename(infile))[0]
    return os.path.join(out_dir, f"{stem}_k_trend.png")


def plot_k_trend(
    infile: str,
    out_png: str,
    window: int = 50,
    active_cycles_override: Optional[int] = None,
    y_zoom_active: bool = False,
    show_rolling: bool = True,
) -> str:
    hist = read_k_history(infile)

    cycles = hist.cycles
    keffs = hist.keffs

    active_start_idx = infer_active_start_index(hist, active_cycles_override)
    active_cycles = cycles[active_start_idx:]
    active_keffs = keffs[active_start_idx:]

    if active_start_idx > 0:
        active_start_cycle = int(active_cycles[0])
        cut_x = 0.5 * (cycles[active_start_idx - 1] + cycles[active_start_idx])
    else:
        active_start_cycle = int(cycles[0])
        cut_x = cycles[0]

    k_mean_computed = float(np.mean(active_keffs))
    k_stddev_computed = float(np.std(active_keffs, ddof=1)) if active_keffs.size > 1 else 0.0
    k_stderr_computed = k_stddev_computed / np.sqrt(active_keffs.size) if active_keffs.size > 0 else 0.0

    k_mean_plot = hist.k_mean_footer if hist.k_mean_footer is not None else k_mean_computed
    k_stddev_plot = hist.k_stddev_footer if hist.k_stddev_footer is not None else k_stddev_computed
    k_stderr_plot = hist.k_stderr_footer if hist.k_stderr_footer is not None else k_stderr_computed

    fig, ax = plt.subplots(figsize=(11, 5.5))

    ax.plot(cycles, keffs, linewidth=1.0, label="keff per cycle")

    if show_rolling:
        rm = rolling_mean(keffs, window)
        if rm is not None:
            ax.plot(cycles[window - 1:], rm, linewidth=2.0, label=f"rolling mean ({window})")

    if active_start_idx > 0:
        ax.axvline(cut_x, linewidth=1.8, linestyle="--", label=f"active start: cycle {active_start_cycle}")

    ax.axhline(k_mean_plot, linewidth=1.8, linestyle="-.", label=f"active mean = {k_mean_plot:.6f}")

    title_parts = ["keff history"]
    if hist.pcm_footer is not None:
        title_parts.append(f"pcm diff = {hist.pcm_footer:.3f}")
    if hist.percent_error_footer is not None:
        title_parts.append(f"percent error = {hist.percent_error_footer:.6f}%")
    ax.set_title(" | ".join(title_parts))

    ax.set_xlabel("Cycle")
    ax.set_ylabel("keff")
    ax.grid(True, alpha=0.35)

    info = [
        f"points: {len(keffs)}",
        f"active cycles: {len(active_keffs)}",
        f"k_mean: {k_mean_plot:.6f}",
        f"stddev: {k_stddev_plot:.6f}",
        f"stderr: {k_stderr_plot:.6f}",
    ]
    if hist.diff_ref_footer is not None:
        info.append(f"diff K: {hist.diff_ref_footer:.6f}")
    if hist.pcm_footer is not None:
        info.append(f"pcm: {hist.pcm_footer:.3f}")
    if hist.total_time_footer is not None and hist.avg_time_footer is not None:
        info.append(f"time: {hist.total_time_footer:.1f}s, avg {hist.avg_time_footer:.3f}s/cycle")

    ax.text(
        0.99,
        0.02,
        "\n".join(info),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", alpha=0.75),
    )

    if active_start_idx > 0:
        y_min, y_max = ax.get_ylim()
        y_text = y_max - 0.06 * (y_max - y_min)
        ax.text(cut_x, y_text, " inactive | active ", ha="center", va="top", fontsize=9)

    if y_zoom_active:
        # Optional zoom around converged/active cycles, while keeping early outliers visible only if disabled.
        margin = max(5.0 * k_stddev_computed, 5.0e-4)
        ax.set_ylim(k_mean_computed - margin, k_mean_computed + margin)

    ax.legend(loc="best")

    out_dir = os.path.dirname(out_png)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close(fig)

    print(f"Saved: {out_png}")
    print(f"Parsed points: {len(keffs)}")
    print(f"Active cycles used: {len(active_keffs)}")
    print(f"Active start cycle: {active_start_cycle}")
    print(f"k_mean used: {k_mean_plot:.8f}")
    print(f"k_mean computed from active cycles: {k_mean_computed:.8f}")
    print(f"k_stddev computed: {k_stddev_computed:.8f}")
    print(f"k_stderr computed: {k_stderr_computed:.8f}")

    return out_png


def main():
    parser = argparse.ArgumentParser(description="Plot keff trend from k_history text file.")
    parser.add_argument("--infile", default="k_history_20260703_140900.txt", help="Input k_history text file")
    parser.add_argument("--out-dir", default="plots", help="Output directory used when --out-png is omitted")
    parser.add_argument("--out-png", default=None, help="Exact output PNG path")
    parser.add_argument("--window", type=int, default=50, help="Rolling-mean window")
    parser.add_argument(
        "--active-cycles",
        type=int,
        default=None,
        help="Override active cycle count. By default, uses 'Active cycles:' footer from the file.",
    )
    parser.add_argument("--zoom-active", action="store_true", help="Zoom y-axis around the active-cycle mean")
    parser.add_argument("--no-rolling", action="store_true", help="Disable rolling mean")
    args = parser.parse_args()

    out_png = make_out_png(args.infile, args.out_dir, args.out_png)

    plot_k_trend(
        infile=args.infile,
        out_png=out_png,
        window=args.window,
        active_cycles_override=args.active_cycles,
        y_zoom_active=args.zoom_active,
        show_rolling=not args.no_rolling,
    )


if __name__ == "__main__":
    main()
