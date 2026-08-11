#!/usr/bin/env python3
"""Plot service-rate outcomes over the bracketed equilibrium-wait interval."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--trajectory", type=Path, required=True)
    p.add_argument("--result", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing figure: {args.output}")
    data = pd.read_csv(args.trajectory).sort_values("wait_input_min")
    result = json.loads(args.result.read_text())
    low = float(result["final_bracket"]["low_wait_min"])
    high = float(result["final_bracket"]["high_wait_min"])
    selected = data[data.wait_input_min.isin([low, high])].sort_values("wait_input_min")
    rates = data.service_rate * 100

    fig, ax = plt.subplots(figsize=(8.7, 5.4), layout="constrained")
    ax.plot(data.wait_input_min, rates, "o", color="#238443", ms=6,
            label="simulated service rate")
    ax.axhline(95, color="#b2182b", lw=1.7, ls="--", label="service requirement: 95%")
    ax.axvspan(low, high, color="#fdb863", alpha=0.46, label="final equilibrium bracket")
    ax.plot(selected.wait_input_min, selected.service_rate * 100, "o", color="#d95f02", ms=7,
            zorder=3, label="bracket endpoints")
    for row in selected.itertuples(index=False):
        ax.annotate(f"{row.service_rate * 100:.3f}%", (row.wait_input_min, row.service_rate * 100),
                    xytext=(7, 9), textcoords="offset points", fontsize=9, color="#7f2704")
    ax.set_xlabel(r"Candidate equilibrium waiting time $W^{in}$ (min)")
    ax.set_ylabel("Service rate $R$ (%)")
    ax.set_title("Service-rate feasibility over the equilibrium-wait bracket", fontweight="bold")
    ax.set_ylim(min(94.8, rates.min() - 0.15), rates.max() + 0.18)
    ax.grid(axis="y", alpha=0.28)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.text(0.02, 0.03,
            f"At the final bracket: R = {selected.service_rate.min() * 100:.3f}%–{selected.service_rate.max() * 100:.3f}%\n"
            "Both endpoints satisfy the 95% requirement.",
            transform=ax.transAxes, fontsize=9, va="bottom")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
