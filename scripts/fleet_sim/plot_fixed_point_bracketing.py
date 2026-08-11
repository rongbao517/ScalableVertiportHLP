#!/usr/bin/env python3
"""Publication-style evidence for a bracketed fixed point (without cherry-picking)."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def reconstruct_brackets(data: pd.DataFrame) -> list[tuple[int, float, float]]:
    """Reconstruct each valid sign-changing bisection bracket in evaluation order."""
    ordered = data.sort_values("evaluation")
    positive = ordered.loc[ordered.residual_min > 0].iloc[0]
    negative = ordered.loc[ordered.residual_min < 0].iloc[0]
    low, high = sorted((float(positive.wait_input_min), float(negative.wait_input_min)))
    rows = [(int(negative.evaluation), low, high)]
    for row in ordered.iloc[2:].itertuples(index=False):
        w, residual = float(row.wait_input_min), float(row.residual_min)
        if residual > 0:
            low = w
        elif residual < 0:
            high = w
        low, high = sorted((low, high))
        rows.append((int(row.evaluation), low, high))
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--trajectory", type=Path, required=True)
    p.add_argument("--result", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing figure: {args.output}")

    data = pd.read_csv(args.trajectory)
    result = json.loads(args.result.read_text())
    low = float(result["final_bracket"]["low_wait_min"])
    high = float(result["final_bracket"]["high_wait_min"])
    view = data.sort_values("wait_input_min")
    brackets = reconstruct_brackets(data)

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.3), layout="constrained")
    ax = axes[0]
    ax.axvspan(low, high, color="#fdb863", alpha=0.42, label="final sign-changing bracket")
    ax.plot(view.wait_input_min, view.wait_observed_min, "o-", color="#d95f02", lw=2.1,
            ms=5.5, label=r"simulated response $F(W^{in})$")
    bounds = np.array([min(view.wait_input_min.min(), view.wait_observed_min.min()),
                       max(view.wait_input_min.max(), view.wait_observed_min.max())])
    ax.plot(bounds, bounds, "--", color="#333333", lw=1.4, label=r"identity $W^{out}=W^{in}$")
    ax.set_xlim(bounds[0] - 0.03, bounds[1] + 0.03)
    ax.set_ylim(bounds[0] - 0.03, bounds[1] + 0.03)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"Input wait $W^{in}$ (min)")
    ax.set_ylabel(r"Simulated wait $W^{out}=F(W^{in})$ (min)")
    ax.set_title("a  Fixed-point condition", loc="left", fontweight="bold")
    ax.annotate(f"bracket: [{low:.4f}, {high:.4f}] min", xy=((low + high) / 2, (low + high) / 2),
                xytext=(0.12, 0.88), textcoords="axes fraction",
                arrowprops={"arrowstyle": "-", "color": "#666666"}, fontsize=9)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    ax = axes[1]
    y = np.arange(1, len(brackets) + 1)
    for yi, (evaluation, left, right) in zip(y, brackets):
        ax.hlines(yi, left, right, color="#2166ac", lw=3)
        ax.plot([left, right], [yi, yi], "o", color="#2166ac", ms=5)
        ax.text(right + 0.025, yi, f"eval. {evaluation}", va="center", fontsize=8)
    ax.axvspan(low, high, color="#fdb863", alpha=0.42)
    ax.set_yticks(y, [f"refinement {i}" for i in y])
    ax.invert_yaxis()
    ax.set_xlabel(r"Candidate equilibrium wait $W$ (min)")
    ax.set_title("b  Bisection bracket contracts", loc="left", fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    ax.set_xlim(-0.05, 2.28)
    ax.text(0.02, 0.03, "Each bar has opposite residual signs at its endpoints.\nFinal width = %.4f min." % (high - low),
            transform=ax.transAxes, fontsize=9, va="bottom")

    fig.suptitle("Numerical fixed-point solution for the demand–operations feedback", fontweight="bold", fontsize=14)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
