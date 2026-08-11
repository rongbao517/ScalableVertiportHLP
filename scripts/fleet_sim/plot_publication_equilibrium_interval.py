#!/usr/bin/env python3
"""Publication-style interval figure for a numerical fixed-point bracket."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def draw_interval(ax, low, high, label, color, xlabel, annotation, threshold=None):
    mid, half = (low + high) / 2, (high - low) / 2
    ax.errorbar(mid, 0, xerr=half, fmt="o", color=color, ecolor=color,
                elinewidth=3, capsize=8, markersize=9, zorder=3)
    ax.scatter([low, high], [0, 0], color=color, s=28, zorder=4)
    if threshold is not None:
        ax.axvline(threshold, color="#9b2c2c", linestyle="--", linewidth=1.5)
        ax.text(threshold, -0.28, f"feasibility threshold = {threshold:.0f}%",
                color="#9b2c2c", ha="center", fontsize=9)
    pad = max((high - low) * 3.5, 0.015 if high < 10 else 0.7)
    ax.set_xlim(low - pad, high + pad)
    ax.set_ylim(-0.55, 0.55)
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_xlabel(xlabel)
    ax.set_title(label, loc="left", fontsize=13, fontweight="bold")
    ax.annotate(annotation, (mid, 0), xytext=(0, 24), textcoords="offset points",
                ha="center", fontsize=10)
    ax.text(low, -0.13, f"{low:.4f}", ha="center", va="top", fontsize=9)
    ax.text(high, -0.13, f"{high:.4f}", ha="center", va="top", fontsize=9)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root-result", type=Path, required=True)
    p.add_argument("--trajectory", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {a.output}")
    root = json.loads(a.root_result.read_text())
    b = root["final_bracket"]
    low_w, high_w = b["low_wait_min"], b["high_wait_min"]
    frame = pd.read_csv(a.trajectory)
    endpoints = frame[frame.wait_input_min.isin([low_w, high_w])]
    low_r, high_r = endpoints.service_rate.min() * 100, endpoints.service_rate.max() * 100

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.78, bottom=0.22, wspace=0.32)
    draw_interval(
        axes[0], low_w, high_w,
        "(a) Numerical fixed-point bracket for waiting time",
        "#d66c32", "Waiting time, W (min)",
        "root bracket: g(WL)>0 and g(WU)<0",
    )
    draw_interval(
        axes[1], low_r, high_r,
        "(b) Service-rate feasibility over the W bracket",
        "#268f45", "Service rate, R (%)",
        "all bracketed outcomes are operationally feasible",
        threshold=95,
    )
    fig.suptitle("Numerical equilibrium interval and operational feasibility", fontsize=16, fontweight="bold")
    fig.text(0.5, 0.03,
             "Bars denote numerical ranges, not statistical confidence intervals. "
             "The waiting-time root is bounded but not claimed to be a unique exact point.",
             ha="center", fontsize=9, color="#444444")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output, dpi=250, bbox_inches="tight")
    print(a.output)


if __name__ == "__main__":
    main()
