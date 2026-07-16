# -*- coding: utf-8 -*-
"""
Real total trip volume (origin + destination, 29 valid days of 2015-04) for
EVERY grid cell in Shanghaidata_final.csv (1676 cells), not just the 187
NSGA candidate pool -- needed as the demand weight for whole-city
demand-weighted K-means site selection.

Same raw-data handling as compute_real_demand_candidates.py: 20150418.csv is
a byte-identical duplicate of 20150402.csv and is skipped.
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
GRID_CSV = DATA_DIR / "Shanghaidata_final.csv"
OUT_CSV = DATA_DIR / "real_demand_all_grids.csv"

DUPLICATE_FILES = {"20150418.csv"}
GLOBAL_START = pd.Timestamp("2015-04-01 00:00:00")
N_DAYS = 30

LAT0, LON0, STEP = 31.0189, 121.2177, 0.01
N_LAT, N_LON = 39, 43


def main():
    grid = pd.read_csv(GRID_CSV).reset_index(drop=True)
    n_grid = len(grid)
    print(f"grid cells: {n_grid}")

    lookup = np.full((N_LAT, N_LON), -1, dtype=np.int32)
    lat_idx = np.round((grid["avg_lat"].to_numpy() - LAT0) / STEP).astype(int)
    lon_idx = np.round((grid["avg_lon"].to_numpy() - LON0) / STEP).astype(int)
    lookup[lat_idx, lon_idx] = np.arange(n_grid)

    origin_counts = np.zeros(n_grid, dtype=np.int64)
    dest_counts = np.zeros(n_grid, dtype=np.int64)

    files = sorted(RAW_DIR.glob("201504*.csv"))
    files = [f for f in files if f.name not in DUPLICATE_FILES]
    print(f"using {len(files)} daily files (skipped {sorted(DUPLICATE_FILES)})")
    global_end = GLOBAL_START + pd.Timedelta(days=N_DAYS)

    for fi, path in enumerate(files):
        df = pd.read_csv(path, usecols=["o_lng", "o_lat", "d_lng", "d_lat", "o_t"])
        o_t = pd.to_datetime(df["o_t"], format="%Y-%m-%d %H:%M:%S", errors="coerce").to_numpy()
        time_valid = (o_t >= GLOBAL_START.to_datetime64()) & (o_t < global_end.to_datetime64())

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
    out.to_csv(OUT_CSV, index=False)
    print(f"saved -> {OUT_CSV}")
    print(f"total trips matched (origin): {origin_counts.sum()}  (dest): {dest_counts.sum()}")
    print(f"grids with zero demand: {(out['real_total_demand'] == 0).sum()}/{n_grid}")


if __name__ == "__main__":
    main()
