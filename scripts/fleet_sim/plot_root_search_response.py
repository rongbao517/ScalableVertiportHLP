#!/usr/bin/env python3
"""Create a non-destructive diagnostic figure for the bracketed root search."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trajectory", type=Path, required=True)
    p.add_argument("--validation", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    data = pd.read_csv(args.trajectory).sort_values("wait_input_min")
    validation = json.loads(args.validation.read_text())
    root_wait = validation["root_wait_min"]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    w = data.wait_input_min
    style = dict(marker="o", linewidth=1.8)
    axes[0, 0].plot(w, data.wait_observed_min, color="#e2703a", label=r"$F(W)$", **style)
    axes[0, 0].plot(w, w, color="#333333", linestyle="--", label=r"$W$")
    axes[0, 0].axvline(root_wait, color="#666666", linestyle=":")
    axes[0, 0].set_ylabel("Observed wait (min)")
    axes[0, 0].set_title("Fixed-point response")
    axes[0, 0].legend(frameon=False)
    axes[0, 1].axhline(0, color="#333333", linewidth=1)
    axes[0, 1].plot(w, data.residual_min, color="#a52a2a", **style)
    axes[0, 1].axvline(root_wait, color="#666666", linestyle=":")
    axes[0, 1].set_ylabel(r"Residual $F(W)-W$ (min)")
    axes[0, 1].set_title("Root residual")
    axes[1, 0].plot(w, data.demand_monthly / 1e4, color="#2166dc", **style)
    axes[1, 0].axvline(root_wait, color="#666666", linestyle=":")
    axes[1, 0].set_ylabel(r"Demand ($10^4$ trips/month)")
    axes[1, 0].set_xlabel("Input wait W (min)")
    axes[1, 0].set_title("Demand response")
    axes[1, 1].plot(w, data.service_rate * 100, color="#1f9c4a", **style)
    axes[1, 1].axvline(root_wait, color="#666666", linestyle=":")
    axes[1, 1].set_ylabel("Service rate R (%)")
    axes[1, 1].set_xlabel("Input wait W (min)")
    axes[1, 1].set_title("Operational response")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    figure.suptitle("LP-H8 bracketed equilibrium diagnostics", fontweight="bold")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
