# -*- coding: utf-8 -*-
"""
Time-bucketed variant of extract_full_grid_od_pairs.py: same full-resolution
(1676-grid) OD trip counts, but now split by a (hour_of_day, day_type) bucket
instead of aggregated across the whole month.

Why this is needed: the original month-aggregated table was only valid because
route_assignment_od_to_vertiports.py assumed a flat, time-invariant ground
speed -- the optimal (i, j) vertiport pair for a given (grid_o, grid_d) pair
didn't depend on time of day. Once ground speed becomes dynamic (predicted
by train_shanghai_speed_gru.py), that's no longer true: the same trip could
route differently at 8am vs 2pm. Redoing this at full 30-min-bin granularity
(1392 bins) would blow up the OD table size well past the ~487K-unique-pair
scale already noted as a near-limit in the original script. Bucketing by
(hour_of_day x day_type) = 24 x 2 = 48 buckets keeps it tractable while still
being genuinely time-varying -- day_type collapses the calendar's 3-way
workday/weekend/holiday split to a binary workday/offday (holiday days are
too few, 2 of 30, to warrant their own 24 buckets).

Stored sparse (only nonzero (o_cell, d_cell, bucket) triples), via np.unique
over a combined int64 key rather than a dense (n_cells^2 x 48) array (which
would be ~1GB+ and mostly zero) -- same sparse-storage spirit as the original.
"""
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
WEATHER_CSV = DATA_DIR / "shanghai_calendar_weather_202504.csv"
OUT_CSV = DATA_DIR / "full_grid_od_pairs_by_bucket_202504.csv"

DUPLICATE_FILES = {"20150418.csv"}
LAT0, LON0, STEP = 31.0189, 121.2177, 0.01
N_LAT, N_LON = 39, 43
N_CELLS = N_LAT * N_LON
N_BUCKETS = 48  # 24 hours x {workday, offday}


def main():
    cal = pd.read_csv(WEATHER_CSV, parse_dates=["date"])
    is_workday = dict(zip(cal["date"].dt.strftime("%Y%m%d"), cal["day_type"].eq("workday")))

    files = sorted(RAW_DIR.glob("201504*.csv"))
    files = [f for f in files if f.name not in DUPLICATE_FILES]
    print(f"using {len(files)} daily files (skipped {sorted(DUPLICATE_FILES)})")

    all_keys = []
    total_rows = 0
    matched_rows = 0

    for fi, path in enumerate(files):
        date_str = path.stem  # e.g. "20150401"
        day_is_workday = is_workday.get(date_str)
        assert day_is_workday is not None, f"no calendar entry for {date_str}"
        day_offset = 0 if day_is_workday else 1

        df = pd.read_csv(path, usecols=["o_lng", "o_lat", "d_lng", "d_lat", "o_t"])
        total_rows += len(df)

        o_t = pd.to_datetime(df["o_t"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
        hour = o_t.dt.hour.to_numpy()
        time_valid = ~o_t.isna().to_numpy()

        o_lat_idx = np.round((df["o_lat"].to_numpy() - LAT0) / STEP).astype(int)
        o_lon_idx = np.round((df["o_lng"].to_numpy() - LON0) / STEP).astype(int)
        d_lat_idx = np.round((df["d_lat"].to_numpy() - LAT0) / STEP).astype(int)
        d_lon_idx = np.round((df["d_lng"].to_numpy() - LON0) / STEP).astype(int)

        valid = (
            time_valid
            & (o_lat_idx >= 0) & (o_lat_idx < N_LAT) & (o_lon_idx >= 0) & (o_lon_idx < N_LON)
            & (d_lat_idx >= 0) & (d_lat_idx < N_LAT) & (d_lon_idx >= 0) & (d_lon_idx < N_LON)
        )
        matched_rows += int(valid.sum())

        o_flat = (o_lat_idx[valid] * N_LON + o_lon_idx[valid]).astype(np.int64)
        d_flat = (d_lat_idx[valid] * N_LON + d_lon_idx[valid]).astype(np.int64)
        bucket_id = (hour[valid].astype(np.int64) * 2 + day_offset)  # 0..47

        key = (o_flat * N_CELLS + d_flat) * N_BUCKETS + bucket_id
        all_keys.append(key)

        print(f"[{fi + 1:02d}/{len(files)}] {path.name}  rows={len(df)}  matched={int(valid.sum())}  "
              f"day_type={'workday' if day_is_workday else 'offday'}")

    all_keys = np.concatenate(all_keys)
    uniq_keys, counts = np.unique(all_keys, return_counts=True)
    print(f"unique (o_cell, d_cell, bucket) triples: {len(uniq_keys)}")

    bucket_id = uniq_keys % N_BUCKETS
    rem = uniq_keys // N_BUCKETS
    d_cell = rem % N_CELLS
    o_cell = rem // N_CELLS
    o_lat_idx, o_lon_idx = o_cell // N_LON, o_cell % N_LON
    d_lat_idx, d_lon_idx = d_cell // N_LON, d_cell % N_LON
    hour_of_day = bucket_id // 2
    day_type = np.where(bucket_id % 2 == 0, "workday", "offday")

    result = pd.DataFrame({
        "o_lat": LAT0 + o_lat_idx * STEP,
        "o_lon": LON0 + o_lon_idx * STEP,
        "d_lat": LAT0 + d_lat_idx * STEP,
        "d_lon": LON0 + d_lon_idx * STEP,
        "hour_of_day": hour_of_day,
        "day_type": day_type,
        "trip_count": counts,
    })
    result.to_csv(OUT_CSV, index=False)
    print(f"total_rows={total_rows}  matched_rows={matched_rows}  unique_nonzero_triples={len(result)}")
    print(f"saved -> {OUT_CSV}")


if __name__ == "__main__":
    main()
