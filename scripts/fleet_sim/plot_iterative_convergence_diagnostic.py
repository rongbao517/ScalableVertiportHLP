#!/usr/bin/env python3
"""Plot honest fixed-point convergence diagnostics from an existing trajectory.

The upper row shows parameter levels with tight, data-driven axes.  The lower
row shows the quantities that define numerical convergence: absolute changes
between consecutive iterations.  No simulation result is modified.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def padded_limits(values: pd.Series, fraction: float = 0.18) -> tuple[float, float]:
    low, high = float(values.min()), float(values.max())
    span = high - low
    pad = max(span * fraction, max(abs(low), abs(high), 1.0) * 0.002)
    return low - pad, high + pad


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing figure: {args.output}")

    frame = pd.read_csv(args.trajectory).sort_values("t").drop_duplicates("t", keep="last")
    required = {"t", "D_t", "W_t", "R_t"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    t = frame["t"].to_numpy()
    levels = [
        (frame["D_t"] / 1e6, "Demand $D_t$ (million trips/month)", "#2166ac"),
        (frame["W_t"], "Mean waiting time $W_t$ (min)", "#d95f02"),
        (frame["R_t"] * 100, "Service rate $R_t$ (%)", "#238443"),
    ]
    changes = [
        (frame["D_t"].pct_change().abs() * 100, r"$|\Delta D_t|/D_{t-1}$ (%)", 1.0, "#2166ac"),
        (frame["W_t"].pct_change().abs() * 100, r"$|\Delta W_t|/W_{t-1}$ (%)", 5.0, "#d95f02"),
        ((frame["R_t"] - frame["R_t"].shift(1)).abs() * 100, r"$|\Delta R_t|$ (percentage points)", None, "#238443"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14.2, 6.8), layout="constrained")
    for col, (values, ylabel, color) in enumerate(levels):
        ax = axes[0, col]
        ax.plot(t, values, "o-", lw=2.1, ms=5.5, color=color)
        ax.scatter(t[0], values.iloc[0], s=80, facecolors="white", edgecolors=color,
                   linewidths=2.0, zorder=3, label="initial iteration")
        ax.scatter(t[-1], values.iloc[-1], s=62, color=color, zorder=3, label="last iteration")
        ax.set_ylim(*padded_limits(values))
        ax.set_ylabel(ylabel)
        ax.set_xticks(t)
        ax.grid(axis="y", alpha=0.28)
        ax.set_title(chr(97 + col), loc="left", fontweight="bold")
        if col == 0:
            ax.legend(frameon=False, fontsize=8, loc="best")

    for col, (values, ylabel, threshold, color) in enumerate(changes):
        ax = axes[1, col]
        valid = values.iloc[1:]
        ax.plot(t[1:], valid, "o-", lw=2.1, ms=5.5, color=color)
        if threshold is not None:
            ax.axhline(threshold, color="#b2182b", lw=1.5, ls="--",
                       label=f"stopping threshold: {threshold:g}%")
            ax.legend(frameon=False, fontsize=8, loc="upper right")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Fixed-point iteration $t$")
        ax.set_xticks(t)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.28)
        ax.set_title(chr(100 + col), loc="left", fontweight="bold")

    fig.suptitle(args.title, fontsize=14, fontweight="bold")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
