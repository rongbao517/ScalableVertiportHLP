# -*- coding: utf-8 -*-
"""Reproduce the mode-choice v3/LP fleet-operation result figures.

The default input is the converged H=8 mode-choice run used by the frozen
v3 baseline (``eqsearch_lp_mc_main_iter10``).  A different simulation run can
be plotted by passing its tag, for example::

    /usr/bin/python3.11 scripts/fleet_sim/plot_modechoice_v3lp_results.py \
        --tag eqsearch_mc_H12_fleet275_iter12 \
        --title "Mode-choice layout, LP-H12, fleet=8250" \
        --output-prefix modechoice_H12_fleet8250

For each figure both PNG and SVG are written.  This script only reads existing
simulation CSVs; it does not rerun the fleet simulation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_SIM_DIR = PROJECT_DIR / "outputs" / "fleet_sim"
DEFAULT_FIGURE_DIR = PROJECT_DIR / "outputs" / "figures"
DEFAULT_TAG = "eqsearch_lp_mc_main_iter10"
DEFAULT_PREFIX = "modechoice_v3lp"
DEFAULT_TITLE = "Mode-choice layout, LP-H8 (v3, current default)"

REAL_UNMET_COLUMNS = [
    "unmet_fleet_insufficient",
    "unmet_away_flying",
    "unmet_insufficient_battery",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=DEFAULT_TAG,
                        help="suffix shared by the fleet-simulation CSV files")
    parser.add_argument("--sim-dir", type=Path, default=DEFAULT_SIM_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--output-prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--title", default=DEFAULT_TITLE,
                        help="common title prefix used in all four figures")
    parser.add_argument("--n-bins", type=int, default=500,
                        help="number of rows/time bins to plot")
    parser.add_argument("--png-dpi", type=int, default=150)
    return parser.parse_args()


def load_csv(sim_dir: Path, stem: str, tag: str, n_bins: int) -> pd.DataFrame:
    path = sim_dir / f"{stem}_{tag}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Required simulation output does not exist: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Simulation output is empty: {path}")
    return frame.iloc[:n_bins].copy()


def require_columns(frame: pd.DataFrame, columns: list[str], source: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def save_figure(fig: plt.Figure, figure_dir: Path, basename: str,
                png_dpi: int) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    png = figure_dir / f"{basename}.png"
    svg = figure_dir / f"{basename}.svg"
    fig.savefig(png, dpi=png_dpi, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {png}")
    print(f"saved -> {svg}")


def line_figure(x: pd.Series, y: pd.Series, *, title: str, ylabel: str,
                color: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x, y, color=color, linewidth=1.8)
    ax.set_title(title)
    ax.set_xlabel("time step (30-min bin)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    ax.grid(True, color="#dddddd", linewidth=0.8)
    fig.tight_layout()
    return fig


def plot_results(args: argparse.Namespace) -> None:
    time_steps = load_csv(
        args.sim_dir, "time_step_summary", args.tag, args.n_bins
    )
    battery = load_csv(
        args.sim_dir, "battery_summary", args.tag, args.n_bins
    )
    occupancy = load_csv(
        args.sim_dir, "vertiport_occupancy", args.tag, args.n_bins
    )

    require_columns(
        time_steps,
        ["time_step", "met_demand", "unmet_demand", *REAL_UNMET_COLUMNS],
        "time_step_summary",
    )
    require_columns(
        battery, ["time_step", "low_battery_ratio"], "battery_summary"
    )
    require_columns(occupancy, ["time_step"], "vertiport_occupancy")

    met = time_steps["met_demand"]
    ordinary_denominator = met + time_steps["unmet_demand"]
    assignment_ratio = met.div(ordinary_denominator).where(
        ordinary_denominator > 0
    )

    real_unmet = time_steps[REAL_UNMET_COLUMNS].sum(axis=1)
    real_denominator = met + real_unmet
    real_assignment_ratio = met.div(real_denominator).where(real_denominator > 0)

    shown_bins = len(time_steps)
    fig = line_figure(
        time_steps["time_step"],
        assignment_ratio,
        title=f"{args.title} -- assignment ratio, first {shown_bins} bins",
        ylabel="assignment ratio",
        color="#2f78d2",
    )
    fig.axes[0].set_ylim(0, 1.05)
    save_figure(
        fig, args.figure_dir, f"{args.output_prefix}_assignratio_timeseries",
        args.png_dpi,
    )

    fig = line_figure(
        time_steps["time_step"],
        real_assignment_ratio,
        title=f"{args.title} -- REAL assignment ratio (= R*), first {shown_bins} bins",
        ylabel="assignment ratio",
        color="#2f78d2",
    )
    fig.axes[0].set_ylim(0, 1.05)
    save_figure(
        fig, args.figure_dir,
        f"{args.output_prefix}_assignratio_real_timeseries", args.png_dpi,
    )

    fig = line_figure(
        battery["time_step"],
        battery["low_battery_ratio"],
        title=f"{args.title} -- low-battery vehicle ratio, first {len(battery)} bins",
        ylabel="low-battery vehicle ratio",
        color="#ef642f",
    )
    save_figure(
        fig, args.figure_dir, f"{args.output_prefix}_lowbattery_timeseries",
        args.png_dpi,
    )

    available = occupancy.drop(columns="time_step")
    if available.shape[1] == 0:
        raise ValueError("vertiport_occupancy contains no vertiport columns")
    site_order = available.median().sort_values(ascending=False).index

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.boxplot(
        [available[site].dropna() for site in site_order],
        tick_labels=site_order,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.5},
        boxprops={"facecolor": "#8bb8e8", "edgecolor": "#6ba3df", "alpha": 0.7},
        whiskerprops={"color": "#555555"},
        capprops={"color": "#555555"},
    )
    ax.set_title(
        f"{args.title} -- per-vertiport idle-vehicle distribution, "
        f"first {len(occupancy)} bins"
    )
    ax.set_xlabel("vertiport (grid id), sorted by median availability")
    ax.set_ylabel("idle (available) vehicles")
    ax.tick_params(axis="x", rotation=90)
    ax.grid(True, axis="y", color="#dddddd", linewidth=0.8)
    fig.tight_layout()
    save_figure(
        fig, args.figure_dir,
        f"{args.output_prefix}_vertiport_available_boxplot", args.png_dpi,
    )

    print(
        "summary: "
        f"assignment_ratio_mean={assignment_ratio.mean():.6f}, "
        f"R*_bin_mean={real_assignment_ratio.mean():.6f}, "
        f"low_battery_mean={battery['low_battery_ratio'].mean():.6f}, "
        f"low_battery_max={battery['low_battery_ratio'].max():.6f}"
    )


def main() -> None:
    args = parse_args()
    if args.n_bins <= 0:
        raise ValueError("--n-bins must be positive")
    plot_results(args)


if __name__ == "__main__":
    main()
