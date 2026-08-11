#!/usr/bin/env python3
"""Write separate plots for the three strict convergence criteria."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--trajectory", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--prefix", required=True)
    args = p.parse_args()
    data = pd.read_csv(args.trajectory).sort_values("t")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    series = [
        ("demand_change", data.delta_D_pct.abs(), r"$|\Delta D_t|/D_{t-1}$ (%)", 1.0),
        ("wait_residual", data.w_residual_min.abs(), r"$|W_t^{out}-W_t^{in}|$ (min)", 0.05),
        ("service_change", data.delta_R_pp.abs(), r"$|\Delta R_t|$ (percentage points)", 0.05),
    ]
    for name, values, ylabel, threshold in series:
        mask = values.notna()
        output = args.output_dir / f"{args.prefix}_{name}.png"
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite existing figure: {output}")
        fig, ax = plt.subplots(figsize=(8.0, 5.2), layout="constrained")
        ax.plot(data.loc[mask, "t"], values.loc[mask], color="#2166ac", lw=2.5)
        ax.set_xlabel("Fixed-point iteration", fontsize=16)
        ax.set_ylabel(ylabel, fontsize=16)
        ax.set_xticks(data.t)
        ax.tick_params(axis="both", labelsize=16)
        ax.set_ylim(bottom=0)
        fig.savefig(output, dpi=240, bbox_inches="tight")
        plt.close(fig)
        print(output)


if __name__ == "__main__":
    main()
