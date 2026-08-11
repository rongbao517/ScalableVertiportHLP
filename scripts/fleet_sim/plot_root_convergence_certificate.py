#!/usr/bin/env python3
"""Show convergence evidence for a bracketed numerical fixed-point solve."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def brackets(data: pd.DataFrame) -> list[tuple[int, float]]:
    ordered = data.sort_values("evaluation").reset_index(drop=True)
    pos = float(ordered.loc[ordered.residual_min > 0, "wait_input_min"].iloc[0])
    neg = float(ordered.loc[ordered.residual_min < 0, "wait_input_min"].iloc[0])
    low, high = min(pos, neg), max(pos, neg)
    result = [(int(ordered.loc[1, "evaluation"]), high - low)]
    for row in ordered.iloc[2:].itertuples(index=False):
        if row.residual_min > 0:
            low = float(row.wait_input_min)
        elif row.residual_min < 0:
            high = float(row.wait_input_min)
        low, high = min(low, high), max(low, high)
        result.append((int(row.evaluation), high - low))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing figure: {args.output}")
    data = pd.read_csv(args.trajectory).sort_values("evaluation")
    b = brackets(data)
    x_b, width = zip(*b)

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6), layout="constrained")
    ax = axes[0]
    residual = data.residual_min.abs()
    ax.semilogy(data.evaluation, residual, "o-", color="#b2182b", lw=2.1, ms=6)
    ax.scatter(data.evaluation.iloc[0], residual.iloc[0], s=74, facecolors="white",
               edgecolors="#b2182b", linewidths=2, zorder=3, label="initial evaluation")
    ax.scatter(data.evaluation.iloc[-1], residual.iloc[-1], s=58, color="#b2182b",
               zorder=3, label="final evaluation")
    ax.annotate(f"{residual.iloc[0]:.3f} min", (data.evaluation.iloc[0], residual.iloc[0]),
                xytext=(8, 9), textcoords="offset points", fontsize=9)
    ax.annotate(f"{residual.iloc[-1]:.3f} min", (data.evaluation.iloc[-1], residual.iloc[-1]),
                xytext=(-46, 9), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Model evaluation")
    ax.set_ylabel(r"Fixed-point gap $|F(W^{in})-W^{in}|$ (min, log scale)")
    ax.set_title("a  Input–output mismatch", loc="left", fontweight="bold")
    ax.grid(which="both", axis="y", alpha=0.28)
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    ax = axes[1]
    ax.semilogy(x_b, width, "o-", color="#2166ac", lw=2.2, ms=6)
    ax.scatter(x_b[0], width[0], s=74, facecolors="white", edgecolors="#2166ac",
               linewidths=2, zorder=3, label="initial bracket")
    ax.scatter(x_b[-1], width[-1], s=58, color="#2166ac", zorder=3, label="final bracket")
    ax.annotate(f"{width[0]:.3f} min", (x_b[0], width[0]), xytext=(8, 9),
                textcoords="offset points", fontsize=9)
    ax.annotate(f"{width[-1]:.4f} min", (x_b[-1], width[-1]), xytext=(-46, 9),
                textcoords="offset points", fontsize=9)
    ax.set_xlabel("Model evaluation")
    ax.set_ylabel("Sign-changing bracket width (min, log scale)")
    ax.set_title("b  Numerical uncertainty contracts", loc="left", fontweight="bold")
    ax.grid(which="both", axis="y", alpha=0.28)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("Convergence certificate for the numerical fixed-point solver", fontweight="bold", fontsize=14)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
