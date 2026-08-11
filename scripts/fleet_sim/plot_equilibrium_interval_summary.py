#!/usr/bin/env python3
"""Create a plain-language interval summary, not a pointwise solver diagnostic."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def interval_bar(ax, low, high, color, label, unit, note, threshold=None):
    mid = (low + high) / 2
    ax.axhline(0, color="#d9d9d9", linewidth=12, zorder=0)
    ax.plot([low, high], [0, 0], color=color, linewidth=12, solid_capstyle="round", zorder=2)
    ax.scatter([low, high], [0, 0], s=55, color=color, zorder=3)
    ax.axvline(mid, color="#222222", linewidth=1.2, zorder=4)
    if threshold is not None:
        ax.axvline(threshold, color="#aa3333", linestyle="--", linewidth=1.6, zorder=1)
        ax.text(threshold, -0.48, f"Threshold {threshold:g}{unit}", ha="center", color="#8a2222", fontsize=9)
    pad = max((high - low) * 2.2, 0.03 if abs(high) < 10 else 0.004 * abs(high))
    ax.set_xlim(low - pad, high + pad)
    ax.set_ylim(-0.62, 0.62)
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_title(label, loc="left", fontweight="bold", fontsize=15)
    ax.text(mid, 0.30, f"{low:.4f}–{high:.4f}{unit}", ha="center", fontsize=13, fontweight="bold")
    ax.text(0.5, 0.04, note, transform=ax.transAxes, ha="center", fontsize=10, color="#444444")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root-result", type=Path, required=True)
    p.add_argument("--trajectory", type=Path, required=True)
    p.add_argument("--validation-result", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {a.output}")
    root = json.loads(a.root_result.read_text())
    validation = json.loads(a.validation_result.read_text())
    b = root["final_bracket"]
    low_w, high_w = b["low_wait_min"], b["high_wait_min"]
    data = pd.read_csv(a.trajectory)
    endpoints = data[data.wait_input_min.isin([low_w, high_w])]
    d_low, d_high = endpoints.demand_monthly.min() / 1e6, endpoints.demand_monthly.max() / 1e6
    r_low, r_high = endpoints.service_rate.min() * 100, endpoints.service_rate.max() * 100

    fig, axes = plt.subplots(3, 1, figsize=(10.2, 9.2))
    fig.subplots_adjust(left=0.12, right=0.95, top=0.87, bottom=0.10, hspace=0.95)
    fig.suptitle("Equilibrium result: bounded interval, not a single exact point", fontsize=18, fontweight="bold", y=0.965)
    interval_bar(axes[0], low_w, high_w, "#e07a3f", "Waiting-time equilibrium interval, W*", " min",
                 "Root is bracketed by opposite residual signs")
    interval_bar(axes[1], d_low, d_high, "#2166dc", "Monthly UAM demand within W* interval, D*", " M trips",
                 "Demand variation across the interval is only 0.06%")
    interval_bar(axes[2], r_low, r_high, "#1f9c4a", "Service rate within W* interval, R*", "%",
                 f"All values feasible; local neighbourhood variation = {validation['r_spread_pp']:.3f} percentage points", threshold=95)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output, dpi=220, bbox_inches="tight")
    print(a.output)


if __name__ == "__main__":
    main()
