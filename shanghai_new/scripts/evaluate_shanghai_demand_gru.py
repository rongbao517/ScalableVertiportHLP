# -*- coding: utf-8 -*-
"""Visualize the SiteGRU per-site demand regression results (test set)."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs"
FIGS_DIR = OUT_DIR / "figs"


def find_latest_run():
    runs = sorted((OUT_DIR / "save_shanghai_demand_gru").glob("*"), key=lambda p: p.stat().st_mtime)
    return runs[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default=None, help="run dir name under outputs/save_shanghai_demand_gru/ (default: latest by mtime)")
    ap.add_argument("--fig-prefix", type=str, default="gru_eval", help="prefix for the 4 output png filenames")
    args = ap.parse_args()

    run_dir = (OUT_DIR / "save_shanghai_demand_gru" / args.run) if args.run else find_latest_run()
    print("run:", run_dir)

    pred_log = np.load(run_dir / "test_prediction_log.npy")  # (144, 30, 2)
    true_log = np.load(run_dir / "test_groundtruth_log.npy")
    pred = np.expm1(pred_log)
    pred[pred < 0] = 0
    true = np.expm1(true_log)

    metrics = json.loads((run_dir / "test_metrics.json").read_text())
    print(json.dumps(metrics, indent=2))

    run_config = json.loads((run_dir / "config.json").read_text())
    data_name = run_config.get("data", "shanghai_demand_windows.npz")
    d = np.load(DATA_DIR / data_name)
    grid_ids = d["grid_ids"]
    ts_test = pd.to_datetime(d["ts_test"])

    # ============================================================
    # Figure 1: aggregate demand (origin), true vs pred, all sites summed
    # ============================================================
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), dpi=150, sharex=True)
    for ax, ci, label in zip(axes, [0, 1], ["origin_demand", "destination_demand"]):
        agg_true = true[:, :, ci].sum(axis=1)
        agg_pred = pred[:, :, ci].sum(axis=1)
        ax.plot(ts_test, agg_true, label="true", marker="o", ms=3, lw=1.3)
        ax.plot(ts_test, agg_pred, label="predicted", marker="x", ms=3, lw=1.3)
        ax.set_title(f"Test set (04-28..04-30): aggregate {label}, true vs predicted", fontsize=11)
        ax.set_ylabel("trips/30min\n(summed over 30 sites)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(FIGS_DIR / f"{args.fig_prefix}_1_aggregate_true_vs_pred.png")
    plt.close(fig)

    # ============================================================
    # Figure 2: scatter true vs pred, both channels pooled
    # ============================================================
    flat_true = true.flatten()
    flat_pred = pred.flatten()
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.scatter(flat_true, flat_pred, s=6, alpha=0.25)
    lim = max(flat_true.max(), flat_pred.max()) * 1.05
    ax.plot([0, lim], [0, lim], "r--", lw=1, label="y = x")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("True demand (trips/30min)")
    ax.set_ylabel("Predicted demand (trips/30min)")
    ax.set_title(f"Test set: true vs predicted (30 sites x 2 channels x 144 bins, n={flat_true.size})\n"
                 f"R2={metrics['R2']:.3f}  PCC={metrics['PCC']:.3f}")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGS_DIR / f"{args.fig_prefix}_2_scatter_true_vs_pred.png")
    plt.close(fig)

    # ============================================================
    # Figure 3: top-3 highest-demand sites, origin_demand time series
    # ============================================================
    site_totals = true[:, :, 0].sum(axis=0)
    top_idx = np.argsort(-site_totals)[:3]
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), dpi=150, sharex=True)
    for ax, i in zip(axes, top_idx):
        ax.plot(ts_test, true[:, i, 0], label="true", marker="o", ms=3, lw=1.2)
        ax.plot(ts_test, pred[:, i, 0], label="predicted", marker="x", ms=3, lw=1.2)
        ax.set_title(f"Grid {int(grid_ids[i])}: origin demand", fontsize=10)
        ax.set_ylabel("trips/30min")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("time")
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(FIGS_DIR / f"{args.fig_prefix}_3_top_sites_timeseries.png")
    plt.close(fig)

    # ============================================================
    # Figure 4: error distribution
    # ============================================================
    nz = true > 1e-5
    err = (pred - true)[nz]
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    ax.hist(err, bins=60, color="#4C72B0", alpha=0.85)
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("Prediction error (predicted - true), trips/30min")
    ax.set_ylabel("Count")
    ax.set_title(f"Test error distribution (n={err.size})  mean={err.mean():.2f}  std={err.std():.2f}")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGS_DIR / f"{args.fig_prefix}_4_error_distribution.png")
    plt.close(fig)

    # ============================================================
    # Figure 5: per-site outflow/inflow, true vs predicted (x-axis = site)
    # ============================================================
    n_sites = true.shape[1]
    site_true_out = true[:, :, 0].sum(axis=0)   # (n_sites,) total origin_demand over test set
    site_pred_out = pred[:, :, 0].sum(axis=0)
    site_true_in = true[:, :, 1].sum(axis=0)    # total destination_demand over test set
    site_pred_in = pred[:, :, 1].sum(axis=0)

    order = np.argsort(-(site_true_out + site_true_in))
    labels = [str(int(g)) for g in grid_ids[order]]
    x = np.arange(n_sites)

    def site_line(ax, true_vals, pred_vals, ylabel, title):
        ax.plot(x, true_vals, marker="o", ms=5, lw=1.5, color="#4C72B0", label="true")
        ax.plot(x, pred_vals, marker="x", ms=6, lw=1.5, color="#DD8452", label="predicted")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, axis="y")

    fig, axes = plt.subplots(2, 1, figsize=(max(11, n_sites * 0.4), 8), dpi=150)

    site_line(axes[0], site_true_out[order], site_pred_out[order],
               "total outflow\n(origin_demand, test set)",
               f"Per-site outflow (origin_demand): true vs predicted, summed over test set ({n_sites} sites)")

    site_line(axes[1], site_true_in[order], site_pred_in[order],
               "total inflow\n(destination_demand, test set)",
               f"Per-site inflow (destination_demand): true vs predicted, summed over test set ({n_sites} sites)")
    axes[1].set_xlabel("site (Grid ID), sorted by total true demand desc")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90 if n_sites > 15 else 0, fontsize=8)
        ax.set_xlim(-1, n_sites)

    plt.tight_layout()
    plt.savefig(FIGS_DIR / f"{args.fig_prefix}_5_per_site_inflow_outflow.png")
    plt.close(fig)

    print(f"saved: {args.fig_prefix}_1..5 .png")


if __name__ == "__main__":
    main()
