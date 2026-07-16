# -*- coding: utf-8 -*-
"""Same test but for OD PAIRS (30x30), which is what actually matters for
the current model's prediction task."""
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
od_narrow = np.zeros((n_days * BPD, n_sites, n_sites), dtype=np.int64)
od_buffer = np.zeros((n_days * BPD, n_sites, n_sites), dtype=np.int64)

for di, fname in enumerate(TEST_DAYS):
    df = pd.read_csv(RAW_DIR / fname, usecols=["o_lng", "o_lat", "d_lng", "d_lat", "o_t"])
    o_t = pd.to_datetime(df["o_t"], format="%Y-%m-%d %H:%M:%S", errors="coerce").to_numpy()
    day_start = GLOBAL_START + pd.Timedelta(days=di)
    valid = (o_t >= day_start.to_datetime64()) & (o_t < (day_start + pd.Timedelta(days=1)).to_datetime64())
    bin_idx = np.full(len(df), -1, dtype=np.int64)
    bin_idx[valid] = ((o_t[valid] - day_start.to_datetime64()) // np.timedelta64(BIN_MINUTES, "m")).astype(np.int64)

    o_lat = df["o_lat"].to_numpy(); o_lon = df["o_lng"].to_numpy()
    d_lat = df["d_lat"].to_numpy(); d_lon = df["d_lng"].to_numpy()

    # narrow: exact-cell membership (site index or -1), vectorized via lookup like before
    o_site_narrow = np.full(len(df), -1)
    d_site_narrow = np.full(len(df), -1)
    for s in range(n_sites):
        o_site_narrow[(np.abs(o_lat-lat0[s])<0.005)&(np.abs(o_lon-lon0[s])<0.005)] = s
        d_site_narrow[(np.abs(d_lat-lat0[s])<0.005)&(np.abs(d_lon-lon0[s])<0.005)] = s
    mask_n = valid & (o_site_narrow>=0) & (d_site_narrow>=0)
    if mask_n.any():
        flat = bin_idx[mask_n]*n_sites*n_sites + o_site_narrow[mask_n]*n_sites + d_site_narrow[mask_n]
        c = np.bincount(flat, minlength=BPD*n_sites*n_sites)
        od_narrow[di*BPD:(di+1)*BPD] += c.reshape(BPD, n_sites, n_sites)

    # buffer: nearest site within RADIUS_KM (each point assigned to its single nearest in-range site)
    o_dist = np.stack([haversine_km(o_lat,o_lon,lat0[s],lon0[s]) for s in range(n_sites)], axis=1)  # (rows, n_sites)
    d_dist = np.stack([haversine_km(d_lat,d_lon,lat0[s],lon0[s]) for s in range(n_sites)], axis=1)
    o_near = np.argmin(o_dist, axis=1); o_near_d = o_dist[np.arange(len(df)), o_near]
    d_near = np.argmin(d_dist, axis=1); d_near_d = d_dist[np.arange(len(df)), d_near]
    o_site_buf = np.where(o_near_d<RADIUS_KM, o_near, -1)
    d_site_buf = np.where(d_near_d<RADIUS_KM, d_near, -1)
    mask_b = valid & (o_site_buf>=0) & (d_site_buf>=0)
    if mask_b.any():
        flat = bin_idx[mask_b]*n_sites*n_sites + o_site_buf[mask_b]*n_sites + d_site_buf[mask_b]
        c = np.bincount(flat, minlength=BPD*n_sites*n_sites)
        od_buffer[di*BPD:(di+1)*BPD] += c.reshape(BPD, n_sites, n_sites)

    print(f"[{di+1}/{n_days}] {fname}  narrow_matched={mask_n.sum()}  buffer_matched={mask_b.sum()}")

def stats(od, name):
    nz = od[od>0]
    print(f"\n=== {name} ===")
    print(f"nonzero fraction of 30x30 cells: {(od>0).mean():.4f}")
    print(f"median nonzero: {np.median(nz):.1f}  mean nonzero: {nz.mean():.2f}")
    reshaped = od.reshape(n_days, BPD, n_sites, n_sites)
    m = reshaped.mean(axis=0); v = reshaped.var(axis=0)
    mask = m > 0.3
    ratio = v[mask]/m[mask]
    print(f"variance/mean ratio (cells with mean>0.3): median={np.median(ratio):.2f}  n={mask.sum()}")

stats(od_narrow, "OD-pair narrow grid (current method)")
stats(od_buffer, f"OD-pair {RADIUS_KM}km nearest-site-in-radius")
