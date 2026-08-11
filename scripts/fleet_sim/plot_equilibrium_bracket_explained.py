#!/usr/bin/env python3
"""Render interpretable bracket and neighbourhood figures for the reported equilibrium interval."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def style(ax):
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root-result", type=Path, required=True)
    p.add_argument("--trajectory", type=Path, required=True)
    p.add_argument("--neighbourhood", type=Path, required=True)
    p.add_argument("--out-bracket", type=Path, required=True)
    p.add_argument("--out-r", type=Path, required=True)
    a = p.parse_args()
    if a.out_bracket.exists() or a.out_r.exists():
        raise FileExistsError("Refusing to overwrite existing figure")
    root = json.loads(a.root_result.read_text())
    bracket = root["final_bracket"]
    low_w, high_w = bracket["low_wait_min"], bracket["high_wait_min"]
    trajectory = pd.read_csv(a.trajectory)
    endpoints = trajectory[trajectory.wait_input_min.isin([low_w, high_w])].sort_values("wait_input_min")
    neighbourhood = pd.read_csv(a.neighbourhood).sort_values("wait_input_min")

    fig, ax = plt.subplots(figsize=(8.3, 4.8), layout="constrained")
    style(ax)
    ax.axhline(0, color="#222222", linewidth=1.1)
    ax.axvspan(low_w, high_w, color="#e6d8ba", alpha=0.8, label="Reported equilibrium interval")
    ax.scatter(endpoints.wait_input_min, endpoints.residual_min, s=100, color=["#2166dc", "#d34a3a"], zorder=3)
    for row, label in zip(endpoints.itertuples(), ["lower bracket", "upper bracket"]):
        ax.annotate(f"{label}\nW={row.wait_input_min:.6f}\ng(W)={row.residual_min:+.4f} min",
                    (row.wait_input_min, row.residual_min), xytext=(0, 20 if row.residual_min < 0 else -54),
                    textcoords="offset points", ha="center", fontsize=10,
                    arrowprops={"arrowstyle": "-", "color": "#555555"})
    ax.set_xlim(low_w - 0.012, high_w + 0.012)
    ax.set_xlabel("Fixed input waiting time, W (min)")
    ax.set_ylabel("Residual, F(W) − W (min)")
    ax.set_title("How the reported waiting-time interval is obtained", loc="left", fontweight="bold")
    ax.text(0.01, 0.02, "A root is bracketed because the residual changes sign.\nNo single exact W is reported.",
            transform=ax.transAxes, fontsize=10, va="bottom")
    ax.legend(frameon=False, loc="upper right")
    a.out_bracket.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out_bracket, dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.3, 4.8), layout="constrained")
    style(ax)
    ax.axvspan(low_w, high_w, color="#e6d8ba", alpha=0.8, label="Reported W interval")
    ax.scatter(neighbourhood.wait_input_min, neighbourhood.service_rate * 100, s=105, color="#1f9c4a", zorder=3)
    for row in neighbourhood.itertuples():
        ax.annotate(f"W={row.wait_input_min:.4f}\nR={row.service_rate * 100:.3f}%",
                    (row.wait_input_min, row.service_rate * 100), xytext=(0, 12),
                    textcoords="offset points", ha="center", fontsize=9)
    ax.set_xlim(neighbourhood.wait_input_min.min() - 0.008, neighbourhood.wait_input_min.max() + 0.008)
    ax.set_ylim(95.75, 96.30)
    ax.set_xlabel("Input waiting time, W (min)")
    ax.set_ylabel("Service rate, R (%)")
    ax.set_title("Service-rate variation in the root neighbourhood", loc="left", fontweight="bold")
    ax.text(0.01, 0.02, "All points exceed the 95% feasibility threshold;\nlocal variation is 0.339 percentage points.",
            transform=ax.transAxes, fontsize=10, va="bottom")
    ax.legend(frameon=False, loc="upper right")
    fig.savefig(a.out_r, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(a.out_bracket)
    print(a.out_r)


if __name__ == "__main__":
    main()
