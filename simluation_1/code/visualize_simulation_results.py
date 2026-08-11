# -*- coding: utf-8 -*-
"""Generate the frozen mode-choice LP-H8 simulation result figures.

Folder layout (parallel to ``speed_1``)::

    simluation_1/
      code/visualize_simulation_results.py
      data/time_step_summary.csv
      data/vertiport_occupancy.csv
      data/battery_summary.csv
      data/run_summary.csv
      model_run/config.json
      model_run/result_metrics.json
      figures/*.png

Run from any working directory:

    /usr/bin/python3.11 simluation_1/code/visualize_simulation_results.py

The assignment ratio plotted here is the project's operational/REAL service
rate R*, not the backlog-inflated legacy ratio:

    R*_t = met_demand_t /
           (met_demand_t
            + unmet_fleet_insufficient_t
            + unmet_away_flying_t
            + unmet_insufficient_battery_t)

``unmet_rounding_jitter`` is deliberately excluded because it is fractional
rounding noise rather than a genuine fleet-service failure.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent
DATA_DIR = ROOT / "data"
FIGURES_DIR = ROOT / "figures"
MODEL_RUN_DIR = ROOT / "model_run"

REAL_UNMET_COLUMNS = [
    "unmet_fleet_insufficient",
    "unmet_away_flying",
    "unmet_insufficient_battery",
]

BG = "#f7f6f4"
PANEL_BG = "#f2f0ed"
GRID_COLOR = "#d9d6d1"
TEXT_COLOR = "#1a1a1a"
MUTED_COLOR = "#666666"
BLUE = "#2166dc"
ORANGE = "#e2703a"
BOX_BLUE = "#8bb8e8"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    time_steps = pd.read_csv(DATA_DIR / "time_step_summary.csv")
    occupancy = pd.read_csv(DATA_DIR / "vertiport_occupancy.csv")
    battery = pd.read_csv(DATA_DIR / "battery_summary.csv")

    required = {"time_step", "met_demand", *REAL_UNMET_COLUMNS}
    missing = required.difference(time_steps.columns)
    if missing:
        raise ValueError(f"time_step_summary.csv missing columns: {sorted(missing)}")
    if "time_step" not in occupancy or len(occupancy.columns) < 2:
        raise ValueError("vertiport_occupancy.csv has no vertiport columns")
    if not {"time_step", "low_battery_ratio"}.issubset(battery.columns):
        raise ValueError("battery_summary.csv missing required columns")
    return time_steps, occupancy, battery


def add_real_assignment_ratio(time_steps: pd.DataFrame) -> pd.DataFrame:
    result = time_steps.copy()
    result["real_unmet_demand"] = result[REAL_UNMET_COLUMNS].sum(axis=1)
    denominator = result["met_demand"] + result["real_unmet_demand"]
    result["real_assignment_ratio"] = (
        result["met_demand"].div(denominator).where(denominator > 0)
    )
    return result


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(PANEL_BG)
    ax.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.tick_params(colors="black", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def save(fig: plt.Figure, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"saved -> {path}")


def plot_real_assignment_ratio(result: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 5), facecolor=BG)
    style_axis(ax)
    ax.plot(
        result["time_step"],
        result["real_assignment_ratio"],
        color=BLUE,
        linewidth=2.0,
    )
    ax.set_ylim(0.8, 1.01)
    ax.set_xlabel("Time step", fontsize=18, color="black")
    ax.set_ylabel("Real assignment ratio (R*)", fontsize=18, color="black")
    ax.tick_params(axis="both", which="major", labelsize=16, colors="black")
    fig.tight_layout()
    save(fig, "real_assignment_ratio_timeseries.png")


def plot_vehicle_distribution(occupancy: pd.DataFrame) -> None:
    available = occupancy.drop(columns="time_step")
    site_order = available.median().sort_values(ascending=False).index
    # Convert simulation grid IDs to the vertiport sequence used by the
    # selected-site layout file.  The sequence is 1-based for presentation.
    sites = pd.read_csv(DATA_DIR / "selected_sites_modechoice.csv")
    if not {"idx", "Grid ID"}.issubset(sites.columns):
        raise ValueError("selected_sites_modechoice.csv missing idx/Grid ID")
    grid_to_vertiport = {
        str(int(row["Grid ID"])): f"V{i + 1}"
        for i, (_, row) in enumerate(sites.iterrows())
    }
    tick_labels = [
        grid_to_vertiport.get(str(int(site)), str(site)) for site in site_order
    ]

    fig, ax = plt.subplots(figsize=(14, 5), facecolor=BG)
    style_axis(ax)
    ax.boxplot(
        [available[site].dropna() for site in site_order],
        tick_labels=tick_labels,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#111111", "linewidth": 1.5},
        boxprops={"facecolor": BOX_BLUE, "edgecolor": "#6ba3df", "alpha": 0.75},
        whiskerprops={"color": "#555555"},
        capprops={"color": "#555555"},
    )
    ax.set_xlabel("Vertiport", fontsize=18, color="black")
    ax.set_ylabel("Vehicles", fontsize=18, color="black")
    ax.tick_params(
        axis="both", which="major", labelsize=16, colors="black"
    )
    ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    save(fig, "vertiport_idle_vehicle_distribution.png")


def plot_low_battery_ratio(battery: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 5), facecolor=BG)
    style_axis(ax)
    ax.plot(
        battery["time_step"], battery["low_battery_ratio"],
        color=ORANGE, linewidth=2.0,
    )
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Time step", fontsize=18, color="black")
    ax.set_ylabel("Low-battery vehicle ratio", fontsize=18, color="black")
    ax.tick_params(axis="both", which="major", labelsize=16, colors="black")
    fig.tight_layout()
    save(fig, "low_battery_ratio_timeseries.png")


def plot_equilibrium_trajectory(trajectory: pd.DataFrame) -> None:
    """Plot the paper-facing D-W-R trajectory for the mode-choice layout."""
    required = {"t", "D_t", "W_t", "R_t"}
    missing = required.difference(trajectory.columns)
    if missing:
        raise ValueError(
            f"equilibrium_trajectory.csv missing columns: {sorted(missing)}"
        )

    x = trajectory["t"]
    series = [
        (trajectory["D_t"] / 10_000, "Equilibrium demand, $D_t$",
         r"$D_t$ ($10^4$ trips/month)", BLUE),
        (trajectory["W_t"], "Operational waiting time, $W_t$",
         r"$W_t$ (min)", ORANGE),
        (trajectory["R_t"] * 100, "Real assignment ratio, $R_t$",
         r"$R_t$ (%)", "#1f9c4a"),
    ]

    fig, axes = plt.subplots(
        3, 1, figsize=(10, 10), sharex=True, facecolor=BG,
        gridspec_kw={"hspace": 0.35},
    )
    for ax, (values, title, ylabel, color) in zip(axes, series):
        style_axis(ax)
        ax.axvspan(
            6.5, 10.5, color="#c8c5bf", alpha=0.22,
            label="Equilibrium window ($t=7$–$10$)",
        )
        ax.plot(
            x, values, color=color, linewidth=2.2, marker="o",
            markersize=5, zorder=3,
        )
        ax.set_title(
            title, loc="left", fontsize=14, fontweight="bold", color=TEXT_COLOR,
        )
        ax.set_ylabel(ylabel, color=TEXT_COLOR)
        ax.set_xlabel("Equilibrium iteration, $t$", color=TEXT_COLOR)
        ax.set_xticks(x)
        ax.tick_params(axis="x", labelbottom=True)
        ax.margins(x=0.03)

    axes[0].legend(frameon=False, loc="best")
    axes[2].set_ylim(min(94.5, trajectory["R_t"].min() * 100 - 0.2), 97.0)

    window = trajectory[trajectory["t"].between(7, 10)]
    d_star = window["D_t"].mean() / 10_000
    w_star = window["W_t"].mean()
    r_star = window["R_t"].mean() * 100
    fig.suptitle(
        "Mode-Choice Layout: Demand–Operation Equilibrium Convergence",
        x=0.08, y=0.985, ha="left", fontsize=18, fontweight="bold",
        color=TEXT_COLOR,
    )
    fig.text(
        0.08, 0.955,
        "LP-H8 · fleet=7,500 · equilibrium-window means: "
        f"$D^*={d_star:.2f}\\times10^4$, "
        f"$W^*={w_star:.3f}$ min, $R^*={r_star:.2f}\\%$",
        color=MUTED_COLOR, fontsize=10,
    )
    fig.subplots_adjust(top=0.91, left=0.11, right=0.97, bottom=0.07)
    save(fig, "modechoice_DWR_equilibrium_convergence.png")


def plot_equilibrium_parameters_separately(trajectory: pd.DataFrame) -> None:
    """Export D, W and R as three title-free, paper-ready standalone plots."""
    standalone_series = [
        (
            trajectory["D_t"] / 10_000,
            r"$D_t$ ($10^4$ trips/month)",
            BLUE,
            "modechoice_D_equilibrium_convergence.png",
        ),
        (
            trajectory["W_t"],
            r"$W_t$ (min)",
            ORANGE,
            "modechoice_W_equilibrium_convergence.png",
        ),
        (
            trajectory["R_t"] * 100,
            r"$R_t$ (%)",
            "#1f9c4a",
            "modechoice_R_equilibrium_convergence.png",
        ),
    ]

    for values, ylabel, color, filename in standalone_series:
        fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG)
        style_axis(ax)
        ax.plot(
            trajectory["t"], values, color=color, linewidth=2.5,
            marker="o", markersize=7, zorder=3,
        )
        ax.set_xlabel("Iteration", fontsize=18, color=TEXT_COLOR)
        ax.set_ylabel(ylabel, fontsize=18, color=TEXT_COLOR)
        ax.set_xticks(trajectory["t"])
        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.margins(x=0.04, y=0.12)
        fig.tight_layout()
        save(fig, filename)


def export_results(
    result: pd.DataFrame, occupancy: pd.DataFrame, battery: pd.DataFrame,
    trajectory: pd.DataFrame,
) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_RUN_DIR.mkdir(parents=True, exist_ok=True)

    processed_columns = [
        "time_step",
        "met_demand",
        *REAL_UNMET_COLUMNS,
        "real_unmet_demand",
        "real_assignment_ratio",
    ]
    result[processed_columns].to_csv(
        DATA_DIR / "real_assignment_ratio_by_time_step.csv", index=False
    )

    available = occupancy.drop(columns="time_step")
    equilibrium_window = trajectory[trajectory["t"].between(7, 10)]
    metrics = {
        "n_bins": int(len(result)),
        "real_assignment_ratio_formula": (
            "met_demand / (met_demand + unmet_fleet_insufficient + "
            "unmet_away_flying + unmet_insufficient_battery)"
        ),
        "real_assignment_ratio_bin_mean": float(
            result["real_assignment_ratio"].mean()
        ),
        "real_assignment_ratio_min": float(
            result["real_assignment_ratio"].min()
        ),
        "real_assignment_ratio_max": float(
            result["real_assignment_ratio"].max()
        ),
        "aggregate_real_assignment_ratio": float(
            result["met_demand"].sum()
            / (result["met_demand"].sum() + result["real_unmet_demand"].sum())
        ),
        "low_battery_ratio_mean": float(battery["low_battery_ratio"].mean()),
        "low_battery_ratio_max": float(battery["low_battery_ratio"].max()),
        "highest_median_idle_site": str(available.median().idxmax()),
        "highest_median_idle_vehicles": float(available.median().max()),
        "lowest_median_idle_site": str(available.median().idxmin()),
        "lowest_median_idle_vehicles": float(available.median().min()),
        "equilibrium_window": [7, 8, 9, 10],
        "D_star_monthly_trips": float(equilibrium_window["D_t"].mean()),
        "W_star_minutes": float(equilibrium_window["W_t"].mean()),
        "R_star": float(equilibrium_window["R_t"].mean()),
    }
    with (MODEL_RUN_DIR / "result_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"saved -> {DATA_DIR / 'real_assignment_ratio_by_time_step.csv'}")
    print(f"saved -> {MODEL_RUN_DIR / 'result_metrics.json'}")
    print(
        "aggregate real assignment ratio: "
        f"{metrics['aggregate_real_assignment_ratio']:.4%}"
    )


def main() -> None:
    time_steps, occupancy, battery = load_data()
    trajectory = pd.read_csv(DATA_DIR / "equilibrium_trajectory.csv")
    result = add_real_assignment_ratio(time_steps)
    plot_equilibrium_trajectory(trajectory)
    plot_equilibrium_parameters_separately(trajectory)
    plot_real_assignment_ratio(result)
    plot_vehicle_distribution(occupancy)
    plot_low_battery_ratio(battery)
    export_results(result, occupancy, battery, trajectory)


if __name__ == "__main__":
    main()
