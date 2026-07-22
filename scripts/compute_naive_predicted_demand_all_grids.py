# -*- coding: utf-8 -*-
"""
Naive-forecast total trip volume (origin + destination) for EVERY grid cell,
built using ONLY the first TRAIN_DAYS days of 2015-04 -- i.e. the same data a
real siting decision would have had available, since the remaining days'
demand hadn't happened yet. This is the "predicted demand" input for site
selection, as opposed to compute_real_demand_all_grids.py's full-month
oracle total.

The "forecast" itself is intentionally trivial (just the raw historical sum
over the observed window, used as-is for K-means sample_weight -- only the
relative demand distribution across cells matters for clustering, not the
absolute scale) rather than a new long-horizon model: the existing GRU/STID/
TGCN demand models are 30-min-ahead point forecasters (LOOKBACK=48,
HORIZON=1), not built for "total demand over the rest of the month" -- a
different task. Matches TRAIN_DAYS=24 from extract_shanghai_30min_demand.py's
chronological 24/3/3 split, so this reflects "what if we'd sited using only
the train-period days."
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
GRID_CSV = DATA_DIR / "Shanghaidata_final.csv"

DUPLICATE_FILES = {"20150418.csv"}
GLOBAL_START = pd.Timestamp("2015-04-01 00:00:00")
TRAIN_DAYS = 24  # matches extract_shanghai_30min_demand.py's TRAIN_DAYS

LAT0, LON0, STEP = 31.0189, 121.2177, 0.01
N_LAT, N_LON = 39, 43


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-days", type=int, default=TRAIN_DAYS)
    ap.add_argument("--out-csv", type=str, default=str(DATA_DIR / "naive_predicted_demand_all_grids.csv"))
    args = ap.parse_args()

    train_end = GLOBAL_START + pd.Timedelta(days=args.train_days)

    grid = pd.read_csv(GRID_CSV).reset_index(drop=True)
    n_grid = len(grid)
    print(f"grid cells: {n_grid}  using first {args.train_days} days only ({GLOBAL_START} to {train_end})")

    lookup = np.full((N_LAT, N_LON), -1, dtype=np.int32)
    lat_idx = np.round((grid["avg_lat"].to_numpy() - LAT0) / STEP).astype(int)
    lon_idx = np.round((grid["avg_lon"].to_numpy() - LON0) / STEP).astype(int)
    lookup[lat_idx, lon_idx] = np.arange(n_grid)

    origin_counts = np.zeros(n_grid, dtype=np.int64)
    dest_counts = np.zeros(n_grid, dtype=np.int64)

    files = sorted(RAW_DIR.glob("201504*.csv"))
    files = [f for f in files if f.name not in DUPLICATE_FILES]
    files = [f for f in files if GLOBAL_START <= pd.to_datetime(f.stem, format="%Y%m%d") < train_end]
    print(f"using {len(files)} daily files within the training window")

    for fi, path in enumerate(files):
        df = pd.read_csv(path, usecols=["o_lng", "o_lat", "d_lng", "d_lat", "o_t"])
        o_t = pd.to_datetime(df["o_t"], format="%Y-%m-%d %H:%M:%S", errors="coerce").to_numpy()
        time_valid = (o_t >= GLOBAL_START.to_datetime64()) & (o_t < train_end.to_datetime64())

        o_lat_idx = np.round((df["o_lat"].to_numpy() - LAT0) / STEP).astype(int)
        o_lon_idx = np.round((df["o_lng"].to_numpy() - LON0) / STEP).astype(int)
        d_lat_idx = np.round((df["d_lat"].to_numpy() - LAT0) / STEP).astype(int)
        d_lon_idx = np.round((df["d_lng"].to_numpy() - LON0) / STEP).astype(int)

        o_in = (o_lat_idx >= 0) & (o_lat_idx < N_LAT) & (o_lon_idx >= 0) & (o_lon_idx < N_LON)
        d_in = (d_lat_idx >= 0) & (d_lat_idx < N_LAT) & (d_lon_idx >= 0) & (d_lon_idx < N_LON)

        o_mask = time_valid & o_in
        d_mask = time_valid & d_in

        o_site = np.full(len(df), -1, dtype=np.int64)
        o_site[o_mask] = lookup[o_lat_idx[o_mask], o_lon_idx[o_mask]]
        o_valid = o_mask & (o_site >= 0)
        origin_counts += np.bincount(o_site[o_valid], minlength=n_grid)

        d_site = np.full(len(df), -1, dtype=np.int64)
        d_site[d_mask] = lookup[d_lat_idx[d_mask], d_lon_idx[d_mask]]
        d_valid = d_mask & (d_site >= 0)
        dest_counts += np.bincount(d_site[d_valid], minlength=n_grid)

        print(f"[{fi + 1:02d}/{len(files)}] {path.name}  rows={len(df)}  "
              f"origin_matched={int(o_valid.sum())}  dest_matched={int(d_valid.sum())}")

    out = pd.DataFrame({
        "idx": np.arange(n_grid),
        "Grid ID": grid["Grid ID"],
        "avg_lat": grid["avg_lat"],
        "avg_lon": grid["avg_lon"],
        "real_origin_demand": origin_counts,
        "real_dest_demand": dest_counts,
        "real_total_demand": origin_counts + dest_counts,
    })
    out.to_csv(args.out_csv, index=False)
    print(f"saved -> {args.out_csv}")
    print(f"total trips matched (origin): {origin_counts.sum()}  (dest): {dest_counts.sum()}")
    print(f"grids with zero demand: {(out['real_total_demand'] == 0).sum()}/{n_grid}")


if __name__ == "__main__":
    main()
