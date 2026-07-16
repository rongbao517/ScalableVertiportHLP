# -*- coding: utf-8 -*-
"""
Full-resolution (1676-grid, not restricted to the 30 candidate vertiport
sites) origin-destination trip counts, aggregated over the whole month.

This is the demand table the route-assignment problem (assign every real
trip to a takeoff vertiport i and landing vertiport j) operates on: for
each trip we only need WHICH grid cell it starts/ends in and HOW MANY such
trips exist, not the 30-min time bin -- with no vertiport capacity
constraint yet, the optimal (i, j) for a given (grid_o, grid_d) pair does
not depend on time of day at all, so we aggregate across all 29 days.

A first 3-day sample already produced ~194K unique nonzero (grid_o, grid_d)
pairs, so the full month is expected to have several hundred thousand --
far too many to fit as one combined assignment LP (each pair needs its own
30x30=900 route options), even before hitting gurobipy's free-license
2000-variable cap. Stored sparse (only nonzero pairs), not as a dense
1676x1676 matrix.
"""
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
OUT_CSV = DATA_DIR / "full_grid_od_pairs_202504.csv"

DUPLICATE_FILES = {"20150418.csv"}
LAT0, LON0, STEP = 31.0189, 121.2177, 0.01
N_LAT, N_LON = 39, 43


def main():
    files = sorted(RAW_DIR.glob("201504*.csv"))
    files = [f for f in files if f.name not in DUPLICATE_FILES]
    print(f"using {len(files)} daily files (skipped {sorted(DUPLICATE_FILES)})")

    n_cells = N_LAT * N_LON
    counts = np.zeros(n_cells * n_cells, dtype=np.int64)

    total_rows = 0
    matched_rows = 0
    for fi, path in enumerate(files):
        df = pd.read_csv(path, usecols=["o_lng", "o_lat", "d_lng", "d_lat"])
        total_rows += len(df)

        o_lat_idx = np.round((df["o_lat"].to_numpy() - LAT0) / STEP).astype(int)
        o_lon_idx = np.round((df["o_lng"].to_numpy() - LON0) / STEP).astype(int)
        d_lat_idx = np.round((df["d_lat"].to_numpy() - LAT0) / STEP).astype(int)
        d_lon_idx = np.round((df["d_lng"].to_numpy() - LON0) / STEP).astype(int)

        valid = (
            (o_lat_idx >= 0) & (o_lat_idx < N_LAT) & (o_lon_idx >= 0) & (o_lon_idx < N_LON)
            & (d_lat_idx >= 0) & (d_lat_idx < N_LAT) & (d_lon_idx >= 0) & (d_lon_idx < N_LON)
        )
        matched_rows += int(valid.sum())

        o_flat = (o_lat_idx[valid] * N_LON + o_lon_idx[valid]).astype(np.int64)
        d_flat = (d_lat_idx[valid] * N_LON + d_lon_idx[valid]).astype(np.int64)
        flat = o_flat * n_cells + d_flat
        counts += np.bincount(flat, minlength=n_cells * n_cells)

        print(f"[{fi + 1:02d}/{len(files)}] {path.name}  rows={len(df)}  matched={int(valid.sum())}")

    nz = np.flatnonzero(counts)
    o_cell = nz // n_cells
    d_cell = nz % n_cells
    o_lat_idx, o_lon_idx = o_cell // N_LON, o_cell % N_LON
    d_lat_idx, d_lon_idx = d_cell // N_LON, d_cell % N_LON

    result = pd.DataFrame(
        {
            "o_lat": LAT0 + o_lat_idx * STEP,
            "o_lon": LON0 + o_lon_idx * STEP,
            "d_lat": LAT0 + d_lat_idx * STEP,
            "d_lon": LON0 + d_lon_idx * STEP,
            "trip_count": counts[nz],
        }
    )
    result.to_csv(OUT_CSV, index=False)
    print(f"total_rows={total_rows}  matched_rows={matched_rows}  unique_nonzero_pairs={len(result)}")
    print(f"saved -> {OUT_CSV}")


if __name__ == "__main__":
    main()
