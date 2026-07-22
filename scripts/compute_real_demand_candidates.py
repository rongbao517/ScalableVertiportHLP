# -*- coding: utf-8 -*-
"""
Compute REAL total trip volume (origin + destination, 29 valid days of
2015-04) for each of the 187 NSGA candidate sites (predictions_shanghai_ranked.csv,
selected==1), directly from Big paper/Shanghai raw taxi GPS trips.

This replaces the old `d_mean` demand objective (from predictions_shanghai_ranked.csv,
found to have near-zero/negative correlation with real trip volume, r=-0.28)
with a demand metric that's actually grounded in the same real data the
downstream OD-demand model will be trained on.
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
RANK_CSV = DATA_DIR / "predictions_shanghai_ranked.csv"
GRID_CSV = DATA_DIR / "Shanghaidata_final.csv"
OUT_CSV = BASE_DIR / "outputs" / "candidate_real_demand.csv"

DUPLICATE_FILES = {"20150418.csv"}
GLOBAL_START = pd.Timestamp("2015-04-01 00:00:00")
N_DAYS = 30

LAT0, LON0, STEP = 31.0189, 121.2177, 0.01
N_LAT, N_LON = 39, 43


def main():
    df_rank = pd.read_csv(RANK_CSV)
    cand = df_rank[df_rank.get("selected", 0) == 1].copy().reset_index(drop=True)
    n_cand = len(cand)
    print(f"candidates: {n_cand}")

    grid = pd.read_csv(GRID_CSV).reset_index(drop=True)
    meta = grid.iloc[cand["idx"].astype(int)][["Grid ID", "avg_lat", "avg_lon"]].reset_index(drop=True)

    lookup = np.full((N_LAT, N_LON), -1, dtype=np.int32)
    lat_idx = np.round((meta["avg_lat"].to_numpy() - LAT0) / STEP).astype(int)
    lon_idx = np.round((meta["avg_lon"].to_numpy() - LON0) / STEP).astype(int)
    in_range = (lat_idx >= 0) & (lat_idx < N_LAT) & (lon_idx >= 0) & (lon_idx < N_LON)
    print(f"candidates within the 0.01-deg grid range: {in_range.sum()}/{n_cand}")
    lookup[lat_idx[in_range], lon_idx[in_range]] = np.where(in_range)[0]

    files = sorted(RAW_DIR.glob("201504*.csv"))
    files = [f for f in files if f.name not in DUPLICATE_FILES]
    print(f"using {len(files)} daily files")

    origin_counts = np.zeros(n_cand, dtype=np.int64)
    dest_counts = np.zeros(n_cand, dtype=np.int64)
    global_end = GLOBAL_START + pd.Timedelta(days=N_DAYS)

    for fi, path in enumerate(files):
        df = pd.read_csv(path, usecols=["o_lng", "o_lat", "d_lng", "d_lat", "o_t"])
        o_t = pd.to_datetime(df["o_t"], format="%Y-%m-%d %H:%M:%S", errors="coerce").to_numpy()
        time_valid = (o_t >= GLOBAL_START.to_datetime64()) & (o_t < global_end.to_datetime64())

        o_lat_idx = np.round((df["o_lat"].to_numpy() - LAT0) / STEP).astype(int)
        o_lon_idx = np.round((df["o_lng"].to_numpy() - LON0) / STEP).astype(int)
        d_lat_idx = np.round((df["d_lat"].to_numpy() - LAT0) / STEP).astype(int)
        d_lon_idx = np.round((df["d_lng"].to_numpy() - LON0) / STEP).astype(int)

        o_bounds = (o_lat_idx >= 0) & (o_lat_idx < N_LAT) & (o_lon_idx >= 0) & (o_lon_idx < N_LON)
        d_bounds = (d_lat_idx >= 0) & (d_lat_idx < N_LAT) & (d_lon_idx >= 0) & (d_lon_idx < N_LON)

        o_lat_c = np.clip(o_lat_idx, 0, N_LAT - 1); o_lon_c = np.clip(o_lon_idx, 0, N_LON - 1)
        d_lat_c = np.clip(d_lat_idx, 0, N_LAT - 1); d_lon_c = np.clip(d_lon_idx, 0, N_LON - 1)

        o_cand = lookup[o_lat_c, o_lon_c]
        d_cand = lookup[d_lat_c, d_lon_c]

        o_valid = time_valid & o_bounds & (o_cand >= 0)
        d_valid = time_valid & d_bounds & (d_cand >= 0)

        if o_valid.any():
            origin_counts += np.bincount(o_cand[o_valid], minlength=n_cand)
        if d_valid.any():
            dest_counts += np.bincount(d_cand[d_valid], minlength=n_cand)

        print(f"[{fi + 1:02d}/{len(files)}] {path.name}  origin_matched={o_valid.sum()}  dest_matched={d_valid.sum()}")

    out = cand[["idx", "d_mean", "c_mean"]].copy()
    out["Grid ID"] = meta["Grid ID"].to_numpy()
    out["real_origin_demand"] = origin_counts
    out["real_dest_demand"] = dest_counts
    out["real_total_demand"] = origin_counts + dest_counts
    out.to_csv(OUT_CSV, index=False)
    print(f"saved -> {OUT_CSV}")
    print(out[["idx", "d_mean", "real_total_demand"]].describe())
    print("\ncorrelation old d_mean vs real_total_demand:", out["d_mean"].corr(out["real_total_demand"]))


if __name__ == "__main__":
    main()
