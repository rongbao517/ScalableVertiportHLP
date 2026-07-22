# -*- coding: utf-8 -*-
"""
Real site-to-site OD demand tensor for the actual 10 final vertiports
(../selected_sites_K10_v3.csv, the NSGA-II K=10 site-selection result),
NOT the K=10 zone-centroid clustering used elsewhere in this directory
(shanghai_od_zones_k10_30min_1channel.npz / shanghai_zone_centers_k10.csv
-- those are 10 abstract KMeans zones grouping the *30* candidate sites,
built for noise-pooling reasons, and don't correspond to the 10 sites
actually chosen to be built).

This mirrors extract_shanghai_od_matrix.py (which builds the 30x30 tensor
for the ORIGINAL selected_sites_K30.csv) but restricted to the 10 sites
in selected_sites_K10_v3.csv. Only 4 of those 10 sites overlap with the
original K30 site set, so the existing 30x30 tensor cannot simply be
sliced down to these 10 -- this re-extracts real trip counts between
exactly these 10 sites directly from the raw GPS data.

OD[t, i, j] = count of trips with origin in site i's 0.01-deg grid cell
and destination in site j's grid cell, bin t (30-min, keyed by o_t).
"""
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
SITES_CSV = DATA_DIR / "selected_sites_K10_v3.csv"
GRID_CSV = DATA_DIR / "Shanghaidata_final.csv"

OUT_DIR = PROJECT_DIR / "outputs"
OUT_NPZ = OUT_DIR / "shanghai_od_final10_30min_1channel.npz"
OUT_ADJ = OUT_DIR / "shanghai_adj_matrix_final10.csv"
OUT_META = OUT_DIR / "shanghai_od_final10_meta.csv"

DUPLICATE_FILES = {"20150418.csv"}

GLOBAL_START = pd.Timestamp("2015-04-01 00:00:00")
N_DAYS = 30
BIN_MINUTES = 30
BINS_PER_DAY = 24 * 60 // BIN_MINUTES  # 48
N_BINS_FULL = N_DAYS * BINS_PER_DAY  # 1440

MISSING_DAY_IDX = 17  # 2015-04-18
MISSING_BIN_START = MISSING_DAY_IDX * BINS_PER_DAY
MISSING_BIN_END = MISSING_BIN_START + BINS_PER_DAY  # [816, 864)

LAT0, LON0, STEP = 31.0189, 121.2177, 0.01
N_LAT, N_LON = 39, 43


def load_sites():
    sites = pd.read_csv(SITES_CSV)
    grid = (
        pd.read_csv(GRID_CSV)
        .rename(columns=lambda c: c.strip())
        .drop_duplicates(subset="Grid ID")
        .set_index("Grid ID")[["avg_lat", "avg_lon"]]
    )
    meta = sites[["Grid ID"]].join(grid, on="Grid ID").reset_index(drop=True)
    assert meta["avg_lat"].notna().all(), "some Grid IDs missing from Shanghaidata_final.csv"
    return meta


def build_site_lookup(sites):
    lookup = np.full((N_LAT, N_LON), -1, dtype=np.int32)
    lat_idx = np.round((sites["avg_lat"].to_numpy() - LAT0) / STEP).astype(int)
    lon_idx = np.round((sites["avg_lon"].to_numpy() - LON0) / STEP).astype(int)
    lookup[lat_idx, lon_idx] = np.arange(len(sites))
    return lookup


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def build_adjacency(sites):
    lat = sites["avg_lat"].to_numpy()
    lon = sites["avg_lon"].to_numpy()
    n = len(sites)
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        dist[i] = haversine_km(lat[i], lon[i], lat, lon)
    sigma = dist[dist > 0].std()
    W = np.exp(-(dist ** 2) / (sigma ** 2))
    np.fill_diagonal(W, 0.0)
    return W


def main():
    sites = load_sites()
    n_sites = len(sites)
    print(f"final sites: {n_sites}")
    print(sites)

    lookup = build_site_lookup(sites)

    files = sorted(RAW_DIR.glob("201504*.csv"))
    files = [f for f in files if f.name not in DUPLICATE_FILES]
    print(f"using {len(files)} daily files (skipped {sorted(DUPLICATE_FILES)})")

    od_counts = np.zeros((N_BINS_FULL, n_sites, n_sites), dtype=np.int64)
    global_end = GLOBAL_START + pd.Timedelta(days=N_DAYS)

    for fi, path in enumerate(files):
        df = pd.read_csv(path, usecols=["o_lng", "o_lat", "d_lng", "d_lat", "o_t"])
        o_t = pd.to_datetime(df["o_t"], format="%Y-%m-%d %H:%M:%S", errors="coerce").to_numpy()
        time_valid = (o_t >= GLOBAL_START.to_datetime64()) & (o_t < global_end.to_datetime64())

        o_lat_idx = np.round((df["o_lat"].to_numpy() - LAT0) / STEP).astype(int)
        o_lon_idx = np.round((df["o_lng"].to_numpy() - LON0) / STEP).astype(int)
        d_lat_idx = np.round((df["d_lat"].to_numpy() - LAT0) / STEP).astype(int)
        d_lon_idx = np.round((df["d_lng"].to_numpy() - LON0) / STEP).astype(int)

        in_bounds = (
            (o_lat_idx >= 0) & (o_lat_idx < N_LAT) & (o_lon_idx >= 0) & (o_lon_idx < N_LON)
            & (d_lat_idx >= 0) & (d_lat_idx < N_LAT) & (d_lon_idx >= 0) & (d_lon_idx < N_LON)
        )
        base_mask = time_valid & in_bounds

        o_lat_c = np.clip(o_lat_idx, 0, N_LAT - 1)
        o_lon_c = np.clip(o_lon_idx, 0, N_LON - 1)
        d_lat_c = np.clip(d_lat_idx, 0, N_LAT - 1)
        d_lon_c = np.clip(d_lon_idx, 0, N_LON - 1)

        o_site = lookup[o_lat_c, o_lon_c]
        d_site = lookup[d_lat_c, d_lon_c]

        valid = base_mask & (o_site >= 0) & (d_site >= 0)
        n_matched = int(valid.sum())

        if n_matched:
            bin_idx = ((o_t[valid] - GLOBAL_START.to_datetime64()) // np.timedelta64(BIN_MINUTES, "m")).astype(np.int64)
            oi = o_site[valid].astype(np.int64)
            dj = d_site[valid].astype(np.int64)
            flat = bin_idx * (n_sites * n_sites) + oi * n_sites + dj
            counts = np.bincount(flat, minlength=N_BINS_FULL * n_sites * n_sites)
            od_counts += counts.reshape(N_BINS_FULL, n_sites, n_sites)

        print(f"[{fi + 1:02d}/{len(files)}] {path.name}  rows={len(df)}  od_pairs_matched={n_matched}")

    od_compressed = np.delete(od_counts, np.s_[MISSING_BIN_START:MISSING_BIN_END], axis=0)
    n_bins_compressed = od_compressed.shape[0]
    assert n_bins_compressed == N_BINS_FULL - BINS_PER_DAY  # 1392

    arr_0 = od_compressed[np.newaxis].astype(np.float32)  # (channel=1, T, 10, 10)
    np.savez_compressed(OUT_NPZ, arr_0=arr_0)
    print(f"saved OD tensor -> {OUT_NPZ}  shape={arr_0.shape}  total_trips_matched={int(od_compressed.sum())}")

    nz_frac = (od_compressed > 0).mean()
    nz_vals = od_compressed[od_compressed > 0]
    print(f"nonzero fraction: {nz_frac:.4f}  median nonzero: {np.median(nz_vals):.1f}  mean nonzero: {nz_vals.mean():.2f}")

    W = build_adjacency(sites)
    np.savetxt(OUT_ADJ, W, delimiter=",", fmt="%.6f")
    print(f"saved adjacency -> {OUT_ADJ}  shape={W.shape}")

    full_ts = [GLOBAL_START + pd.Timedelta(minutes=BIN_MINUTES * b) for b in range(N_BINS_FULL)]
    compressed_ts = [t for b, t in enumerate(full_ts) if not (MISSING_BIN_START <= b < MISSING_BIN_END)]
    meta = pd.DataFrame({"compressed_bin_idx": range(n_bins_compressed), "timestamp": compressed_ts})
    meta["grid_ids"] = ",".join(str(g) for g in sites["Grid ID"].tolist())
    meta.to_csv(OUT_META, index=False)
    print(f"saved metadata -> {OUT_META}")


if __name__ == "__main__":
    main()
