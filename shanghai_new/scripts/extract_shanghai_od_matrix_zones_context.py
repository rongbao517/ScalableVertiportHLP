# -*- coding: utf-8 -*-
"""
K-zone OD tensor with explicit time-of-day / day-of-week context channels,
in addition to the demand channel.

The ASTT architecture (uam_airspace_context_od_v2.py, untouched) has a
WeatherCalendarContextEncoder + FiLM modulation pathway that activates
whenever in_channels>1 -- built for NYC's 11 weather channels. We don't
have hourly Shanghai weather data, so earlier runs used channel=1 and left
this pathway completely dormant. But time-of-day/day-of-week are known
exactly from our own timestamps (no external data needed) and are exactly
the kind of periodic signal that context+FiLM modulation is designed to
exploit. This builds a (channel=5, T, K, K) tensor:
  channel 0: demand (as before)
  channel 1,2: sin/cos of time-of-day (period 48 bins/day)
  channel 3,4: sin/cos of day-of-week (period 7 days)
Context channels are broadcast (same value for every (i,j) cell at a given
timestep), matching how the original weather channels were structured.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
SITES_CSV = DATA_DIR / "selected_sites_K30.csv"
GRID_CSV = DATA_DIR / "Shanghaidata_final.csv"
OUT_DIR = PROJECT_DIR / "outputs"

DUPLICATE_FILES = {"20150418.csv"}
GLOBAL_START = pd.Timestamp("2015-04-01 00:00:00")
N_DAYS = 30
BIN_MINUTES = 30
BINS_PER_DAY = 24 * 60 // BIN_MINUTES
N_BINS_FULL = N_DAYS * BINS_PER_DAY
MISSING_DAY_IDX = 17
MISSING_BIN_START = MISSING_DAY_IDX * BINS_PER_DAY
MISSING_BIN_END = MISSING_BIN_START + BINS_PER_DAY

LAT0, LON0, STEP = 31.0189, 121.2177, 0.01
N_LAT, N_LON = 39, 43


def load_sites():
    sites = pd.read_csv(SITES_CSV)
    grid = pd.read_csv(GRID_CSV).reset_index(drop=True)
    meta = grid.iloc[sites["idx"].astype(int)][["Grid ID", "avg_lat", "avg_lon"]].reset_index(drop=True)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    K = args.k

    sites = load_sites()
    coords = sites[["avg_lat", "avg_lon"]].to_numpy()
    km = KMeans(n_clusters=K, random_state=args.seed, n_init=10).fit(coords)
    site_zone = km.labels_
    zone_centers = km.cluster_centers_
    print(f"K={K} zones, sites per zone: {np.bincount(site_zone)}")

    sites_out = sites.copy()
    sites_out["zone"] = site_zone
    sites_out.to_csv(OUT_DIR / f"shanghai_site_to_zone_k{K}.csv", index=False)

    lookup = build_site_lookup(sites)
    site_to_zone = np.array(site_zone)

    files = sorted(RAW_DIR.glob("201504*.csv"))
    files = [f for f in files if f.name not in DUPLICATE_FILES]

    zone_counts = np.zeros((N_BINS_FULL, K, K), dtype=np.int64)
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

        o_lat_c = np.clip(o_lat_idx, 0, N_LAT - 1); o_lon_c = np.clip(o_lon_idx, 0, N_LON - 1)
        d_lat_c = np.clip(d_lat_idx, 0, N_LAT - 1); d_lon_c = np.clip(d_lon_idx, 0, N_LON - 1)

        o_site = lookup[o_lat_c, o_lon_c]
        d_site = lookup[d_lat_c, d_lon_c]
        valid = base_mask & (o_site >= 0) & (d_site >= 0)
        n_matched = int(valid.sum())

        if n_matched:
            bin_idx = ((o_t[valid] - GLOBAL_START.to_datetime64()) // np.timedelta64(BIN_MINUTES, "m")).astype(np.int64)
            oz = site_to_zone[o_site[valid]].astype(np.int64)
            dz = site_to_zone[d_site[valid]].astype(np.int64)
            flat = bin_idx * (K * K) + oz * K + dz
            counts = np.bincount(flat, minlength=N_BINS_FULL * K * K)
            zone_counts += counts.reshape(N_BINS_FULL, K, K)
        print(f"[{fi + 1:02d}/{len(files)}] {path.name}  matched={n_matched}")

    zone_compressed = np.delete(zone_counts, np.s_[MISSING_BIN_START:MISSING_BIN_END], axis=0)
    n_bins_compressed = zone_compressed.shape[0]
    assert n_bins_compressed == N_BINS_FULL - BINS_PER_DAY

    # ---- build time-of-day / day-of-week context channels ----
    full_ts = [GLOBAL_START + pd.Timedelta(minutes=BIN_MINUTES * b) for b in range(N_BINS_FULL)]
    compressed_ts = [t for b, t in enumerate(full_ts) if not (MISSING_BIN_START <= b < MISSING_BIN_END)]
    bin_of_day = np.array([int((t - t.normalize()).total_seconds() // (BIN_MINUTES * 60)) for t in compressed_ts])
    dow = np.array([t.dayofweek for t in compressed_ts])  # 0=Mon

    tod_sin = np.sin(2 * np.pi * bin_of_day / BINS_PER_DAY)
    tod_cos = np.cos(2 * np.pi * bin_of_day / BINS_PER_DAY)
    dow_sin = np.sin(2 * np.pi * dow / 7)
    dow_cos = np.cos(2 * np.pi * dow / 7)

    def broadcast(vec):
        return np.broadcast_to(vec[:, None, None], (n_bins_compressed, K, K)).astype(np.float32)

    arr_0 = np.stack([
        zone_compressed.astype(np.float32),
        broadcast(tod_sin), broadcast(tod_cos),
        broadcast(dow_sin), broadcast(dow_cos),
    ], axis=0)  # (5, T, K, K)

    out_npz = OUT_DIR / f"shanghai_od_zones_k{K}_context_5channel.npz"
    np.savez_compressed(out_npz, arr_0=arr_0)
    print(f"saved -> {out_npz}  shape={arr_0.shape}  total_trips={int(zone_compressed.sum())}")

    dist = np.zeros((K, K))
    for i in range(K):
        dist[i] = haversine_km(zone_centers[i, 0], zone_centers[i, 1], zone_centers[:, 0], zone_centers[:, 1])
    sigma = dist[dist > 0].std()
    W = np.exp(-(dist ** 2) / (sigma ** 2))
    np.fill_diagonal(W, 0.0)
    out_adj = OUT_DIR / f"shanghai_adj_matrix_zones_k{K}.csv"
    np.savetxt(out_adj, W, delimiter=",", fmt="%.6f")
    print(f"saved adjacency -> {out_adj}")


if __name__ == "__main__":
    main()
