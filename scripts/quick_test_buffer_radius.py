# -*- coding: utf-8 -*-
"""Quick empirical test: does a 1km-radius catchment buffer (instead of the
narrow 0.01-deg grid cell) meaningfully reduce sparsity/noise in the 30-site
OD demand data? Runs on a SUBSET of days for speed (this is just to decide
whether the full rebuild is worth it)."""
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
SITES_CSV = DATA_DIR / "selected_sites_K30.csv"
GRID_CSV = DATA_DIR / "Shanghaidata_final.csv"

GLOBAL_START = pd.Timestamp("2015-04-01 00:00:00")
BIN_MINUTES = 30
RADIUS_KM = 1.0
TEST_DAYS = ["20150401.csv", "20150402.csv", "20150403.csv", "20150404.csv", "20150405.csv",
             "20150406.csv", "20150407.csv", "20150408.csv", "20150409.csv", "20150410.csv"]

sites = pd.read_csv(SITES_CSV)
grid = pd.read_csv(GRID_CSV).reset_index(drop=True)
meta = grid.iloc[sites["idx"].astype(int)][["Grid ID", "avg_lat", "avg_lon"]].reset_index(drop=True)
n_sites = len(meta)
lat0 = meta["avg_lat"].to_numpy()
lon0 = meta["avg_lon"].to_numpy()

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))

BPD = 48
n_days = len(TEST_DAYS)
counts_narrow = np.zeros((n_days * BPD, n_sites), dtype=np.int64)   # per-site origin demand
counts_buffer = np.zeros((n_days * BPD, n_sites), dtype=np.int64)

for di, fname in enumerate(TEST_DAYS):
    df = pd.read_csv(RAW_DIR / fname, usecols=["o_lng", "o_lat", "o_t"])
    o_t = pd.to_datetime(df["o_t"], format="%Y-%m-%d %H:%M:%S", errors="coerce").to_numpy()
    day_start = GLOBAL_START + pd.Timedelta(days=di)
    valid = (o_t >= day_start.to_datetime64()) & (o_t < (day_start + pd.Timedelta(days=1)).to_datetime64())
    bin_idx = np.full(len(df), -1, dtype=np.int64)
    bin_idx[valid] = ((o_t[valid] - day_start.to_datetime64()) // np.timedelta64(BIN_MINUTES, "m")).astype(np.int64)

    o_lat = df["o_lat"].to_numpy()
    o_lon = df["o_lng"].to_numpy()
    for s in range(n_sites):
        narrow_mask = valid & (np.abs(o_lat - lat0[s]) < 0.005) & (np.abs(o_lon - lon0[s]) < 0.005)
        d = haversine_km(o_lat, o_lon, lat0[s], lon0[s])
        buf_mask = valid & (d < RADIUS_KM)
        if narrow_mask.any():
            counts_narrow[di*BPD:(di+1)*BPD, s] = np.bincount(bin_idx[narrow_mask], minlength=BPD)[:BPD]
        if buf_mask.any():
            counts_buffer[di*BPD:(di+1)*BPD, s] = np.bincount(bin_idx[buf_mask], minlength=BPD)[:BPD]
    print(f"[{di+1}/{n_days}] {fname} done")

def stats(counts, name):
    nz = counts[counts > 0]
    print(f"\n=== {name} ===")
    print(f"nonzero fraction: {(counts>0).mean():.3f}")
    print(f"median nonzero: {np.median(nz):.1f}   mean nonzero: {nz.mean():.2f}")
    # variance/mean ratio across days for same (site, time-of-day)
    reshaped = counts.reshape(n_days, BPD, n_sites)
    m = reshaped.mean(axis=0)
    v = reshaped.var(axis=0)
    mask = m > 0.5
    ratio = v[mask] / m[mask]
    print(f"variance/mean ratio (Poisson noise signature, ~1.0=pure noise): median={np.median(ratio):.2f}")

stats(counts_narrow, "narrow grid cell (current method)")
stats(counts_buffer, f"{RADIUS_KM}km radius buffer")
