# -*- coding: utf-8 -*-
"""
Empirical ground-speed ground truth, derived from the raw taxi GPS trip files
(data/raw/*.csv, columns o_lng,o_lat,d_lng,d_lat,o_t,d_t) -- the first time
this project has used d_t at all. Nothing else in the pipeline computes
duration = d_t - o_t or speed = distance/duration; GROUND_SPEED_KMH everywhere
else (route_assignment_od_to_vertiports.py, validate_route_assignment_with_gurobi.py)
is a hardcoded assumption, not derived from data.

The raw (o_lat,o_lng,o_t) -> (d_lat,d_lng,d_t) pairs are NOT clean single
passenger trips -- inspection showed many near-zero-distance/near-zero-duration
pairs (stationary GPS pings) and some long-duration/short-distance pairs
(idle time folded in), consistent with these being consecutive raw GPS
fix pairs rather than curated origin-destination trip records. We filter to
the subset that looks like a genuine single driving leg before treating the
implied distance/duration ratio as a speed observation.

Aggregated to ONE city-wide series (not per-grid-cell): per-cell would be far
too sparse once the plausibility filter below is applied, whereas a single
time-varying series still captures genuine time-of-day/rush-hour dynamics,
which is the point of making this dynamic at all.

Same 30-min-bin / missing-day convention as extract_shanghai_od_matrix_kmeans30.py
and extract_shanghai_30min_demand.py (no shared config module exists in this
project, so these constants are redefined inline here too, per that convention).
"""
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
OUT_CSV = DATA_DIR / "shanghai_ground_speed_30min.csv"

DUPLICATE_FILES = {"20150418.csv"}

GLOBAL_START = pd.Timestamp("2015-04-01 00:00:00")
N_DAYS = 30
BIN_MINUTES = 30
BINS_PER_DAY = 24 * 60 // BIN_MINUTES  # 48
N_BINS_FULL = N_DAYS * BINS_PER_DAY  # 1440

MISSING_DAY_IDX = 17  # 2015-04-18
MISSING_BIN_START = MISSING_DAY_IDX * BINS_PER_DAY
MISSING_BIN_END = MISSING_BIN_START + BINS_PER_DAY  # [816, 864)

# plausibility filter for a single driving-leg speed observation
MIN_DIST_KM = 0.3
MIN_DUR_MIN = 1.0
MAX_DUR_MIN = 90.0
MIN_SPEED_KMH = 3.0
MAX_SPEED_KMH = 60.0


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def main():
    files = sorted(RAW_DIR.glob("201504*.csv"))
    files = [f for f in files if f.name not in DUPLICATE_FILES]
    print(f"using {len(files)} daily files (skipped {sorted(DUPLICATE_FILES)})")

    global_end = GLOBAL_START + pd.Timedelta(days=N_DAYS)

    # accumulate per-bin speed samples across all files before taking the median
    bin_samples = [[] for _ in range(N_BINS_FULL)]
    total_rows = 0
    kept_rows = 0

    for fi, path in enumerate(files):
        df = pd.read_csv(path, usecols=["o_lng", "o_lat", "d_lng", "d_lat", "o_t", "d_t"])
        total_rows += len(df)

        o_t = pd.to_datetime(df["o_t"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
        d_t = pd.to_datetime(df["d_t"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
        dur_min = (d_t - o_t).dt.total_seconds().to_numpy() / 60.0
        dist_km = haversine_km(
            df["o_lat"].to_numpy(), df["o_lng"].to_numpy(),
            df["d_lat"].to_numpy(), df["d_lng"].to_numpy(),
        )

        o_t_np = o_t.to_numpy()
        time_valid = (o_t_np >= GLOBAL_START.to_datetime64()) & (o_t_np < global_end.to_datetime64())

        with np.errstate(invalid="ignore", divide="ignore"):
            speed_kmh = dist_km / (dur_min / 60.0)

        valid = (
            time_valid
            & (dist_km >= MIN_DIST_KM)
            & (dur_min >= MIN_DUR_MIN) & (dur_min <= MAX_DUR_MIN)
            & (speed_kmh >= MIN_SPEED_KMH) & (speed_kmh <= MAX_SPEED_KMH)
        )
        n_valid = int(valid.sum())
        kept_rows += n_valid

        if n_valid:
            bin_idx = ((o_t_np[valid] - GLOBAL_START.to_datetime64()) // np.timedelta64(BIN_MINUTES, "m")).astype(np.int64)
            speeds = speed_kmh[valid]
            for b, s in zip(bin_idx, speeds):
                bin_samples[b].append(s)

        print(f"[{fi + 1:02d}/{len(files)}] {path.name}  rows={len(df)}  valid_speed_obs={n_valid}")

    print(f"total rows: {total_rows}  kept as valid speed observations: {kept_rows} "
          f"({100 * kept_rows / total_rows:.2f}%)")

    median_speed = np.full(N_BINS_FULL, np.nan)
    n_samples = np.zeros(N_BINS_FULL, dtype=np.int64)
    for b in range(N_BINS_FULL):
        if bin_samples[b]:
            median_speed[b] = float(np.median(bin_samples[b]))
            n_samples[b] = len(bin_samples[b])

    # day 18 has no source file -> real gap, kept as NaN (matches
    # extract_shanghai_30min_demand.py's convention: mark missing, don't impute,
    # let downstream windowing skip any window that touches it).
    real_gap_mask = np.zeros(N_BINS_FULL, dtype=bool)
    real_gap_mask[MISSING_BIN_START:MISSING_BIN_END] = True

    empty_bins = int(np.isnan(median_speed).sum())
    sparse_but_real = empty_bins - int(real_gap_mask.sum())
    print(f"bins with zero valid samples: {empty_bins}/{N_BINS_FULL} "
          f"({sparse_but_real} sparse-but-real, {int(real_gap_mask.sum())} from the missing day)")
    if sparse_but_real > 0:
        # forward/back-fill only the sparse-but-real gaps (genuine days with too few
        # samples in that bin) -- NOT the missing-day block, which stays NaN
        s = pd.Series(median_speed)
        filled = s.ffill().bfill().to_numpy()
        median_speed = np.where(real_gap_mask, np.nan, filled)

    full_ts = [GLOBAL_START + pd.Timedelta(minutes=BIN_MINUTES * b) for b in range(N_BINS_FULL)]

    out = pd.DataFrame({
        "bin_idx": np.arange(N_BINS_FULL),
        "timestamp": full_ts,
        "median_speed_kmh": median_speed,
        "n_samples": n_samples,
    })
    out.to_csv(OUT_CSV, index=False)
    print(f"saved -> {OUT_CSV}  rows={len(out)} (uncompressed 1440-bin timeline, "
          f"missing-day block left as NaN for downstream window-skip logic)")
    print(out["median_speed_kmh"].describe())

    out["hour"] = pd.to_datetime(out["timestamp"]).dt.hour
    print("\nmedian speed by hour of day (sanity check -- expect a rush-hour dip):")
    print(out.groupby("hour")["median_speed_kmh"].mean())


if __name__ == "__main__":
    main()
