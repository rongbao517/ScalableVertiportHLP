#!/usr/bin/env python3
"""Write separate clean D, W and R convergence line charts."""

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
    p = argparse.ArgumentParser()
    p.add_argument("--trajectory", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("--end-iteration", type=int, help="Plot through this certified terminal iteration.")
    args = p.parse_args()
    data = pd.read_csv(args.trajectory).sort_values("t")
    if args.end_iteration is not None:
        data = data.loc[data["t"] <= args.end_iteration].copy()
        if data.empty:
            raise ValueError("No observations remain at or before --end-iteration")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plots = [
        ("demand", data.D_t / 1e6, r"Demand $D_t$ (million trips/month)", "#2166ac"),
        ("wait", data.W_t, r"Observed wait $W_t^{out}$ (min)", "#d95f02"),
        ("service_rate", data.R_t * 100, r"Service rate $R_t$ (%)", "#238443"),
    ]
    for name, values, ylabel, color in plots:
        output = args.output_dir / f"{args.prefix}_{name}.png"
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite existing figure: {output}")
        fig, ax = plt.subplots(figsize=(8.0, 5.2), layout="constrained")
        ax.plot(data.t, values, color=color, lw=2.5)
        ax.set_xlabel("Fixed-point iteration", fontsize=16)
        ax.set_ylabel(ylabel, fontsize=16)
        ax.set_xticks(data.t)
        ax.set_ylim(*limits(values))
        ax.tick_params(axis="both", labelsize=16)
        fig.savefig(output, dpi=240, bbox_inches="tight")
        plt.close(fig)
        print(output)


if __name__ == "__main__":
    main()
