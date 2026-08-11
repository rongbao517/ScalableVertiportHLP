#!/usr/bin/env python3
"""Render a publication-style D-W-R convergence trajectory from a strict run."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def limits(values: pd.Series) -> tuple[float, float]:
    low, high = float(values.min()), float(values.max())
    pad = max((high - low) * 0.12, 0.002 * max(abs(low), abs(high), 1.0))
    return low - pad, high + pad


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing figure: {args.output}")
    data = pd.read_csv(args.trajectory).sort_values("t")
    series = [
        (data.D_t / 1e6, r"Demand $D_t$ (million trips/month)", "#2166ac"),
        (data.W_t, r"Observed wait $W_t^{out}$ (min)", "#d95f02"),
        (data.R_t * 100, r"Service rate $R_t$ (%)", "#238443"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(9.2, 9.0), sharex=True, layout="constrained")
    for index, (values, ylabel, color) in enumerate(series):
        ax = axes[index]
        ax.plot(data.t, values, "o-", color=color, lw=2.15, ms=5.5)
        ax.set_ylabel(ylabel, fontsize=16)
        ax.set_ylim(*limits(values))
        ax.grid(axis="y", alpha=0.28)
        ax.tick_params(axis="both", labelsize=16)
    axes[-1].set_xlabel("Fixed-point iteration", fontsize=16)
    axes[-1].set_xticks(data.t)
    fig.savefig(args.output, dpi=240, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
