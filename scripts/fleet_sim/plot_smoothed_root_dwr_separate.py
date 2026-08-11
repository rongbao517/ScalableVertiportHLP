#!/usr/bin/env python3
"""Write clean separate D/W/R charts from a smoothed-root evaluation trace."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def limits(values: pd.Series) -> tuple[float, float]:
    lo, hi = float(values.min()), float(values.max())
    pad = max((hi - lo) * 0.14, 0.002 * max(abs(lo), abs(hi), 1.0))
    return lo - pad, hi + pad


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trajectory", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--prefix", required=True)
    args = p.parse_args()
    data = pd.read_csv(args.trajectory).sort_values("evaluation")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        ("demand", data.demand_monthly / 1e6, r"Demand $D$ (million trips/month)", "#2166ac"),
        ("wait", data.wait_observed_min, r"Smoothed wait $\bar W^{out}$ (min)", "#d95f02"),
        ("service_rate", data.service_rate * 100, r"Smoothed service rate $\bar R$ (%)", "#238443"),
    ]
    for name, values, ylabel, color in fields:
        output = args.output_dir / f"{args.prefix}_{name}.png"
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite existing figure: {output}")
        fig, ax = plt.subplots(figsize=(8.0, 5.2), layout="constrained")
        ax.plot(data.evaluation, values, color=color, lw=2.5, marker="o", ms=6)
        ax.set_xlabel("Smoothed root-search evaluation", fontsize=16)
        ax.set_ylabel(ylabel, fontsize=16)
        ax.set_xticks(data.evaluation)
        ax.set_ylim(*limits(values))
        ax.tick_params(axis="both", labelsize=16)
        fig.savefig(output, dpi=240, bbox_inches="tight")
        plt.close(fig)
        print(output)


if __name__ == "__main__":
    main()
