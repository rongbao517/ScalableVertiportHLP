#!/usr/bin/env python3
"""Plot fixed-point residual and service feasibility without implying smooth convergence."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


BLUE = "#2166dc"
GREEN = "#1f9c4a"
RED = "#aa3333"
SHADE = "#e6d8ba"


def setup(axis):
    axis.grid(alpha=0.25, color="#777777")
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trajectory", type=Path, required=True)
    p.add_argument("--root-result", type=Path, required=True)
    p.add_argument("--out-w", type=Path, required=True)
    p.add_argument("--out-r", type=Path, required=True)
    args = p.parse_args()
    if args.out_w.exists() or args.out_r.exists():
        raise FileExistsError("Refusing to overwrite an existing figure")

    frame = pd.read_csv(args.trajectory).sort_values("wait_input_min")
    result = json.loads(args.root_result.read_text())
    bracket = result["final_bracket"]
    low, high = bracket["low_wait_min"], bracket["high_wait_min"]

    fig, ax = plt.subplots(figsize=(8.5, 5.2), layout="constrained")
    setup(ax)
    ax.axhline(0, color="#202020", linewidth=1.1, zorder=1)
    ax.axvspan(low, high, color=SHADE, alpha=0.65, label="Numerical equilibrium interval")
    ax.scatter(frame.wait_input_min, frame.residual_min, s=58, color=RED, zorder=3)
    ax.plot(frame.wait_input_min, frame.residual_min, color=RED, linewidth=1.2, alpha=0.65, zorder=2)
    ax.set_xlabel("Input waiting time, $W$ (min)")
    ax.set_ylabel(r"Fixed-point residual, $F(W)-W$ (min)")
    ax.set_title("Fixed-point residual and numerical equilibrium interval", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="upper right")
    args.out_w.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_w, dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.2), layout="constrained")
    setup(ax)
    ax.axhspan(95, 100, color="#e9f4eb", alpha=0.8, label="Feasible region ($R\geq95\%$)")
    ax.axhline(95, color="#487a4f", linestyle="--", linewidth=1.2)
    ax.axvspan(low, high, color=SHADE, alpha=0.65, label="Numerical equilibrium interval")
    ax.scatter(frame.wait_input_min, frame.service_rate * 100, s=62, color=GREEN, zorder=3)
    ax.set_xlabel("Input waiting time, $W$ (min)")
    ax.set_ylabel("Service rate, $R$ (%)")
    ax.set_title("Service-rate feasibility within the equilibrium interval", loc="left", fontweight="bold")
    ax.set_ylim(min(94.8, frame.service_rate.min() * 100 - 0.1), 100)
    ax.legend(frameon=False, loc="lower left")
    fig.savefig(args.out_r, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(args.out_w)
    print(args.out_r)


if __name__ == "__main__":
    main()
