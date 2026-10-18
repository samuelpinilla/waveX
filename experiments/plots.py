"""Figures from the experiment results (runs/*/*.jsonl).

    python experiments/plots.py            # writes runs/figures/*.png

Colours are the first three categorical slots of the dataviz reference palette
(blue, orange, aqua), which pass the all-pairs colour-vision checks.
"""

from pathlib import Path

import numpy as np

from common import SUCCESS, load

WAVEX, BASELINE, CONVEX = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
OUT = Path("runs/figures")


def _style(ax, xlabel, ylabel, title):
    ax.set_title(title, color=INK, loc="left", fontsize=11)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED)


def ecdf_figure(rows, path):
    """Fraction of pulses recovered below each relative error (complete, noiseless AF)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3.8), constrained_layout=True)
    series = [("waveX", "wavex_err", WAVEX), ("waveX, convex stage only", "convex_err", CONVEX),
              ("nonconvex baseline", "baseline_err", BASELINE)]
    for label, key, color in series:
        if key not in rows[0]:
            continue
        e = np.sort([r[key] for r in rows])
        ax.step(e, np.arange(1, len(e) + 1) / len(e), where="post", color=color, linewidth=2, label=label)
    ax.axvline(SUCCESS, color=MUTED, linewidth=1, linestyle=":")
    ax.text(SUCCESS, 0.02, " success threshold", color=MUTED, fontsize=8)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    _style(ax, "relative error dist(x, q)/||x||", "fraction of pulses",
           f"Complete noiseless AF, {len(rows)} pulses")
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(path, dpi=200)


def _grouped(rows, key):
    groups = {}
    for r in rows:
        groups.setdefault(r[key], []).append(r)
    return dict(sorted(groups.items()))


def incomplete_figure(rows, path):
    """Success rate vs fraction of uniformly removed delays, split by identifiability.

    Solid lines: decimation L coprime to N. Dashed: L divides N, where each residue
    class n mod L carries a phase that only the pulse band resolves (waveX runs in band).
    """
    import math

    import matplotlib.pyplot as plt

    uniform = [r for r in rows if r.get("kind") == "uniform delays"]
    if not uniform:
        return
    n = 128
    fig, ax = plt.subplots(figsize=(6, 3.8), constrained_layout=True)
    for label, key, color, marker in [("waveX", "wavex_err", WAVEX, "o"), ("nonconvex baseline", "baseline_err", BASELINE, "s")]:
        if key not in uniform[0]:
            continue
        for ident, style, suffix in [(True, "-", "L coprime to N"), (False, "--", "L divides N")]:
            # L = 1 (complete AF) belongs to both curves as their common starting point
            sel = [r for r in uniform
                   if r["removed"] == 0.0 or (math.gcd(round(1 / (1 - r["removed"])), n) == 1) == ident]
            groups = _grouped(sel, "removed")
            if len(groups) < 2:
                continue
            xs = 100 * np.array(list(groups))
            rate = [100 * np.mean([r[key] < SUCCESS for r in g]) for g in groups.values()]
            ax.plot(xs, rate, color=color, linewidth=2, linestyle=style, marker=marker, markersize=8,
                    markeredgecolor="white", markeredgewidth=2, label=f"{label}, {suffix}")
    ax.set_ylim(-3, 103)
    _style(ax, "delays removed uniformly (%)", f"success rate (error < {SUCCESS:g}), %",
           "Incomplete AF: uniform delay decimation")
    ax.legend(frameon=False, fontsize=8, handlelength=4)
    fig.savefig(path, dpi=200)


def noisy_figure(rows, path):
    """Median relative error (interquartile band) vs SNR."""
    import matplotlib.pyplot as plt

    groups = _grouped(rows, "snr_db")
    snr = np.array(list(groups))
    fig, ax = plt.subplots(figsize=(6, 3.8), constrained_layout=True)
    for label, key, color, marker in [("waveX", "wavex_err", WAVEX, "o"), ("waveX, convex stage only", "convex_err", CONVEX, "D"),
                                      ("nonconvex baseline", "baseline_err", BASELINE, "s")]:
        if key not in rows[0]:
            continue
        q = np.array([np.percentile([r[key] for r in g], [25, 50, 75]) for g in groups.values()])
        ax.fill_between(snr, q[:, 0], q[:, 2], color=color, alpha=0.15, linewidth=0)
        ax.plot(snr, q[:, 1], color=color, linewidth=2, marker=marker, markersize=8,
                markeredgecolor="white", markeredgewidth=2, label=f"{label} (median, IQR)")
    ax.set_yscale("log")
    _style(ax, "SNR (dB)", "relative error", f"Noisy complete AF, {len(next(iter(groups.values())))} pulses per point")
    ax.legend(frameon=False)
    fig.savefig(path, dpi=200)


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in [("benchmark", ecdf_figure), ("incomplete", incomplete_figure), ("noisy", noisy_figure)]:
        folder = Path("runs") / name
        rows = [r for f in sorted(folder.glob("*.jsonl")) for r in load(f)] if folder.exists() else []
        if rows:
            fn(rows, OUT / f"{name}.png")
            print(f"wrote {OUT / name}.png")
