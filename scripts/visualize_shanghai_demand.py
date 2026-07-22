# -*- coding: utf-8 -*-
"""
Visualize the 30-min demand series extracted for the 30 NSGA-selected
Shanghai sites (shanghai_demand_30min.csv / shanghai_demand_windows.npz).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
FIGS_DIR = BASE_DIR / "outputs" / "figs"
LONG_CSV = DATA_DIR / "shanghai_demand_30min.csv"
SITES_CSV = DATA_DIR / "selected_sites_K30.csv"
GRID_CSV = DATA_DIR / "Shanghaidata_final.csv"

TRAIN_DAYS, VAL_DAYS, TEST_DAYS = 24, 3, 3
BINS_PER_DAY = 48

df = pd.read_csv(LONG_CSV, parse_dates=["timestamp"])
sites = pd.read_csv(SITES_CSV)
grid = pd.read_csv(GRID_CSV).reset_index(drop=True)
site_meta = grid.iloc[sites["idx"].astype(int)][["Grid ID", "avg_lat", "avg_lon"]].reset_index(drop=True)

grid_ids = sorted(df["Grid ID"].unique())
n_sites = len(grid_ids)
n_bins = df["bin_idx"].nunique()

# pivot to (n_sites, n_bins) matrices
pivot_o = df.pivot(index="Grid ID", columns="bin_idx", values="origin_demand").loc[grid_ids].to_numpy()
pivot_d = df.pivot(index="Grid ID", columns="bin_idx", values="destination_demand").loc[grid_ids].to_numpy()
bin_ts = df.drop_duplicates("bin_idx").sort_values("bin_idx")["timestamp"].to_numpy()

train_end = TRAIN_DAYS * BINS_PER_DAY
val_end = (TRAIN_DAYS + VAL_DAYS) * BINS_PER_DAY

# ============================================================
# Figure 1: aggregate demand over the full month + split boundaries
# ============================================================
fig, ax = plt.subplots(figsize=(13, 4), dpi=150)
agg_o = pivot_o.sum(axis=0)
agg_d = pivot_d.sum(axis=0)
ax.plot(bin_ts, agg_o, lw=0.9, label="origin demand (30 sites total)")
ax.plot(bin_ts, agg_d, lw=0.9, alpha=0.8, label="destination demand (30 sites total)")
ax.axvline(bin_ts[train_end], color="gray", ls="--", lw=1)
ax.axvline(bin_ts[val_end], color="gray", ls="--", lw=1)
ax.text(bin_ts[train_end // 2], ax.get_ylim()[1] * 0.92, "train (24d)", ha="center", fontsize=9)
ax.text(bin_ts[(train_end + val_end) // 2], ax.get_ylim()[1] * 0.92, "val (3d)", ha="center", fontsize=9)
ax.text(bin_ts[(val_end + n_bins) // 2], ax.get_ylim()[1] * 0.92, "test (3d)", ha="center", fontsize=9)
ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
ax.set_ylabel("Trip count / 30min\n(summed over 30 sites)")
ax.set_title("Aggregate 30-min demand, 2015-04-01 to 2015-04-30 (Shanghai, 30 selected sites)")
ax.legend(loc="upper right", fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIGS_DIR / "viz_1_aggregate_timeseries.png")
plt.close(fig)

# ============================================================
# Figure 2: per-site diurnal profile heatmap (avg over 30 days), origin demand
# ============================================================
profile = np.nanmean(pivot_o.reshape(n_sites, 30, BINS_PER_DAY), axis=1)  # (n_sites, 48), skip missing 04-18
order = np.argsort(-profile.sum(axis=1))  # sort sites by total demand desc
fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
im = ax.imshow(profile[order], aspect="auto", cmap="viridis",
               extent=[0, 24, n_sites, 0])
ax.set_xlabel("Hour of day")
ax.set_ylabel("Site (sorted by total demand, desc)")
ax.set_yticks(np.arange(n_sites) + 0.5)
ax.set_yticklabels([grid_ids[i] for i in order], fontsize=7)
ax.set_title("Average diurnal origin-demand profile per site (avg over 30 days)")
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Avg trips / 30min")
plt.tight_layout()
plt.savefig(FIGS_DIR / "viz_2_diurnal_heatmap.png")
plt.close(fig)

# ============================================================
# Figure 3: spatial map, bubble = total demand per site
# ============================================================
total_demand = np.nansum(pivot_o, axis=1) + np.nansum(pivot_d, axis=1)  # skip missing 04-18
fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
sc = ax.scatter(site_meta["avg_lon"], site_meta["avg_lat"],
                 s=total_demand / total_demand.max() * 600 + 30,
                 c=total_demand, cmap="plasma", edgecolors="k", linewidths=0.5)
for _, r in site_meta.iterrows():
    ax.annotate(str(int(r["Grid ID"])), (r["avg_lon"], r["avg_lat"]),
                fontsize=6, xytext=(3, 3), textcoords="offset points")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Total demand (origin+destination, 30 days) per selected site")
cbar = fig.colorbar(sc, ax=ax)
cbar.set_label("Total trips")
plt.tight_layout()
plt.savefig(FIGS_DIR / "viz_3_spatial_demand_map.png")
plt.close(fig)

# ============================================================
# Figure 4: highest vs lowest demand site, first 7 days time series
# ============================================================
top_i, bot_i = order[0], order[-1]
week_bins = 7 * BINS_PER_DAY
fig, ax = plt.subplots(figsize=(13, 4), dpi=150)
ax.plot(bin_ts[:week_bins], pivot_o[top_i, :week_bins], label=f"Grid {grid_ids[top_i]} (highest demand)")
ax.plot(bin_ts[:week_bins], pivot_o[bot_i, :week_bins], label=f"Grid {grid_ids[bot_i]} (lowest demand)")
ax.xaxis.set_major_locator(mdates.DayLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
ax.set_ylabel("Origin trips / 30min")
ax.set_title("Origin demand, first 7 days: highest- vs lowest-demand site")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIGS_DIR / "viz_4_top_vs_bottom_site_week.png")
plt.close(fig)

print("saved: viz_1_aggregate_timeseries.png, viz_2_diurnal_heatmap.png, "
      "viz_3_spatial_demand_map.png, viz_4_top_vs_bottom_site_week.png")
print("top site:", grid_ids[top_i], "total demand:", total_demand[top_i])
print("bottom site:", grid_ids[bot_i], "total demand:", total_demand[bot_i])
