# -*- coding: utf-8 -*-
"""
Spatial heatmap of per-site demand at specific times of day (e.g. 08:00,
12:00, 18:00), true vs predicted side by side, on the Shanghai basemap.

Uses ONLY the test set (04-28..04-30, genuinely held-out, unlike
visualize_by_daytype.py which had to reach into in-sample train days for
weekend/holiday coverage) -- fixed hour-of-day comparison doesn't have
that problem since the 3-day test window has one real instance of every
hour. Averages the (up to) 3 test-day observations of each target hour
per site, for both true and predicted, and plots on a shared color scale
so true/predicted panels are directly, visually comparable.
"""
import argparse
import json
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import contextily as cx
from shapely.geometry import Point

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs"
FIGS_DIR = OUT_DIR / "figs"


def to_gdf(df, lat="avg_lat", lon="avg_lon"):
    g = gpd.GeoDataFrame(
        df.copy(),
        geometry=[Point(xy) for xy in zip(df[lon], df[lat])],
        crs="EPSG:4326",
    )
    return g.to_crs(epsg=3857)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, required=True, help="run dir under outputs/save_shanghai_demand_gru/")
    ap.add_argument("--sites", type=str, default="selected_sites_kmeans_K30.csv")
    ap.add_argument("--hours", type=int, nargs="+", default=[8, 12, 18])
    ap.add_argument("--fig-prefix", type=str, default="heatmap_hour")
    ap.add_argument("--calibrated", action="store_true",
                     help="use test_prediction_log_calibrated.npy (from calibrate_isotonic.py) instead of the raw model output")
    args = ap.parse_args()

    run_dir = OUT_DIR / "save_shanghai_demand_gru" / args.run
    config = json.loads((run_dir / "config.json").read_text())

    pred_name = "test_prediction_log_calibrated.npy" if args.calibrated else "test_prediction_log.npy"
    pred_log = np.load(run_dir / pred_name)   # (n_bins, n_sites, 2)
    true_log = np.load(run_dir / "test_groundtruth_log.npy")
    pred = np.expm1(pred_log); pred[pred < 0] = 0
    true = np.expm1(true_log)
    total_pred = pred.sum(axis=-1)   # (n_bins, n_sites) -- origin + destination
    total_true = true.sum(axis=-1)

    data = np.load(DATA_DIR / config["data"], allow_pickle=True)
    ts_test = pd.to_datetime(data["ts_test"])
    grid_ids = data["grid_ids"]

    sites = pd.read_csv(DATA_DIR / args.sites)
    site_meta = sites.set_index("Grid ID").loc[grid_ids][["avg_lat", "avg_lon"]].reset_index()

    vmax = max(total_true.max(), total_pred.max())

    fig, axes = plt.subplots(len(args.hours), 2, figsize=(11, 5 * len(args.hours)), dpi=150)
    if len(args.hours) == 1:
        axes = axes.reshape(1, 2)

    for row, hour in enumerate(args.hours):
        mask = (ts_test.hour == hour) & (ts_test.minute == 0)
        n_match = mask.sum()
        mean_true = total_true[mask].mean(axis=0)   # (n_sites,)
        mean_pred = total_pred[mask].mean(axis=0)

        for col, (vals, label) in enumerate([(mean_true, "true"), (mean_pred, "predicted")]):
            ax = axes[row, col]
            df = site_meta.copy()
            df["value"] = vals
            gdf = to_gdf(df)
            sizes = vals / max(vmax, 1e-6) * 500 + 40
            gdf.plot(ax=ax, column="value", cmap="plasma", vmin=0, vmax=vmax, edgecolor="white",
                      linewidth=0.6, markersize=sizes, marker="o", legend=(col == 1),
                      legend_kwds={"label": "total demand (trips/30min)", "shrink": 0.7} if col == 1 else None)
            xmin, ymin, xmax, ymax = gdf.total_bounds
            pad_x, pad_y = (xmax - xmin) * 0.1, (ymax - ymin) * 0.1
            ax.set_xlim(xmin - pad_x, xmax + pad_x)
            ax.set_ylim(ymin - pad_y, ymax + pad_y)
            cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, zoom=11)
            ax.set_axis_off()
            ax.set_title(f"{hour:02d}:00  {label}  (avg over {n_match} test days)", fontsize=10)

    plt.tight_layout()
    out_png = FIGS_DIR / f"{args.fig_prefix}_true_vs_pred.png"
    plt.savefig(out_png, bbox_inches="tight")
    print("saved:", out_png)


if __name__ == "__main__":
    main()
