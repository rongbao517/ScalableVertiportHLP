#!/usr/bin/env python3
"""Render a D-W-R fixed-point trajectory without changing source results."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Damped demand-operation equilibrium search")
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing figure: {args.output}")
    frame = pd.read_csv(args.trajectory)
    required = {"t", "D_t", "W_t", "R_t"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing trajectory columns: {sorted(missing)}")

    figure, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True, layout="constrained")
    series = [
        (frame["D_t"] / 1e4, r"Demand $D_t$ ($10^4$ trips/month)", "#2166dc"),
        (frame["W_t"], r"Wait time $W_t$ (min)", "#e2703a"),
        (frame["R_t"] * 100, r"Service rate $R_t$ (%)", "#1f9c4a"),
    ]
    for axis, (values, ylabel, color) in zip(axes, series):
        axis.plot(frame["t"], values, marker="o", linewidth=2, color=color)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.3)
    axes[-1].set_xlabel("Fixed-point iteration")
    axes[-1].set_xticks(frame["t"])
    figure.suptitle(args.title, fontweight="bold")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
