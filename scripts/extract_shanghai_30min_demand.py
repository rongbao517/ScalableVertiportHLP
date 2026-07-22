# -*- coding: utf-8 -*-
"""
Build 30-min demand time series for the 30 NSGA-selected Shanghai sites
from the raw OD trip data in `Big paper/Shanghai/*.csv` (2015-04-01..30),
then assemble sliding-window train/val/test sets for next-step forecasting.

Demand = trip count per site per 30-min bin, both directions:
  origin_demand      -> o_lat/o_lng falls inside the site's 0.01-deg grid cell
  destination_demand -> d_lat/d_lng falls inside the site's 0.01-deg grid cell

Window setup (user-confirmed): lookback=48 steps (24h) -> horizon=1 step (30min).
Split: day 1-24 train, day 25-27 val, day 28-30 test (chronological, no shuffling).

Data quality note: `20150418.csv` is a byte-for-byte duplicate of
`20150402.csv` (verified via md5, both contain only 2015-04-02 timestamps).
There is no real 2015-04-18 data in this dump. We drop the duplicate file
(which also fixes a 2x demand spike around 04-02 from double counting) and
mark every 04-18 bin as NaN (missing), excluding it from the CSV's real
counts and from any sliding window whose input/target touches that day.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--sites", type=str, default="selected_sites_K30.csv",
                 help="sites csv filename, relative to data/ (must have idx + Grid ID columns)")
ap.add_argument("--tag", type=str, default="",
                 help="suffix for output filenames, e.g. '_kmeans30' -> shanghai_demand_30min_kmeans30.csv")
ap.add_argument("--weather-csv", type=str, default=None,
                 help="optional calendar+weather csv (relative to data/, e.g. shanghai_calendar_weather_202504.csv) "
                      "-- adds day-level weather/day-type as extra context channels, broadcast across all sites")
args = ap.parse_args()

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
SITES_CSV = DATA_DIR / args.sites
GRID_CSV = DATA_DIR / "Shanghaidata_final.csv"

DUPLICATE_FILES = {"20150418.csv"}  # duplicate of 20150402.csv, no real 04-18 data

GLOBAL_START = pd.Timestamp("2015-04-01 00:00:00")
N_DAYS = 30
BIN_MINUTES = 30
N_BINS = N_DAYS * (24 * 60 // BIN_MINUTES)  # 1440
GRID_HALF = 0.005  # 0.01-deg grid cell -> +/-0.005 half-width

MISSING_DAY_IDX = 17  # 2015-04-18 (0-based day index)
MISSING_BIN_START = MISSING_DAY_IDX * (24 * 60 // BIN_MINUTES)
MISSING_BIN_END = MISSING_BIN_START + (24 * 60 // BIN_MINUTES)  # [816, 864)

LOOKBACK = 48
HORIZON = 1
TRAIN_DAYS = 24
VAL_DAYS = 3
TEST_DAYS = 3
BINS_PER_DAY = 24 * 60 // BIN_MINUTES  # 48

# NOTE: written to data/, not outputs/ -- train_shanghai_demand_gru.py reads windows
# npz from data/ (matching how shanghai_demand_windows.npz was already placed there).
OUT_LONG_CSV = DATA_DIR / f"shanghai_demand_30min{args.tag}.csv"
OUT_WINDOWS_NPZ = DATA_DIR / f"shanghai_demand_windows{args.tag}.npz"


def load_sites():
    sites = pd.read_csv(SITES_CSV)
    grid = pd.read_csv(GRID_CSV).reset_index(drop=True)
    meta = grid.iloc[sites["idx"].astype(int)][["Grid ID", "avg_lat", "avg_lon"]].reset_index(drop=True)
    return meta  # columns: Grid ID, avg_lat, avg_lon ; 30 rows


def day_files():
    files = sorted(RAW_DIR.glob("201504*.csv"))
    assert len(files) == N_DAYS, f"expected {N_DAYS} daily files, found {len(files)}"
    files = [f for f in files if f.name not in DUPLICATE_FILES]
    return files


WEATHER_NUMERIC_COLS = ["temp_max_c", "temp_min_c", "precip_mm", "windspeed_max_kmh", "humidity_pct"]


def build_context_channels(weather_csv_path, n_sites):
    """Day-level weather + day-type, tiled to 30-min bins and broadcast across all
    sites (weather/calendar is city-wide, not site-specific). Numeric weather columns
    are z-scored using TRAIN-period days only (first TRAIN_DAYS days), to avoid leaking
    val/test statistics into the normalization -- the 0/1 day-type flags are left as-is.
    Returns (N_BINS, n_sites, K) float32 array + the ordered channel-name list.
    """
    wdf = pd.read_csv(weather_csv_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    assert len(wdf) == N_DAYS, f"expected {N_DAYS} calendar rows, got {len(wdf)}"

    wdf["is_weekend_flag"] = wdf["day_type"].eq("weekend").astype(np.float32)
    wdf["is_holiday_flag"] = wdf["day_type"].eq("holiday").astype(np.float32)
    cols = WEATHER_NUMERIC_COLS + ["is_weekend_flag", "is_holiday_flag"]

    arr = wdf[cols].to_numpy().astype(np.float32)  # (N_DAYS, K)
    n_numeric = len(WEATHER_NUMERIC_COLS)
    mu = arr[:TRAIN_DAYS, :n_numeric].mean(axis=0)
    sd = arr[:TRAIN_DAYS, :n_numeric].std(axis=0)
    arr[:, :n_numeric] = (arr[:, :n_numeric] - mu) / np.where(sd > 1e-6, sd, 1.0)

    daily_tiled = np.repeat(arr, BINS_PER_DAY, axis=0)          # (N_BINS, K)
    ctx = np.tile(daily_tiled[:, None, :], (1, n_sites, 1))     # (N_BINS, n_sites, K)
    return ctx.astype(np.float32), cols


def main():
    sites = load_sites()
    n_sites = len(sites)
    print(f"sites: {n_sites}")

    origin_counts = np.zeros((n_sites, N_BINS), dtype=np.int32)
    dest_counts = np.zeros((n_sites, N_BINS), dtype=np.int32)

    site_lat = sites["avg_lat"].to_numpy()
    site_lon = sites["avg_lon"].to_numpy()

    files = day_files()
    print(f"using {len(files)} daily files (skipped {sorted(DUPLICATE_FILES)})")
    global_end = GLOBAL_START + pd.Timedelta(days=N_DAYS)

    for fi, path in enumerate(files):
        df = pd.read_csv(path, usecols=["o_lng", "o_lat", "d_lng", "d_lat", "o_t", "d_t"])
        df["o_t"] = pd.to_datetime(df["o_t"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
        df["d_t"] = pd.to_datetime(df["d_t"], format="%Y-%m-%d %H:%M:%S", errors="coerce")

        o_valid = (df["o_t"] >= GLOBAL_START) & (df["o_t"] < global_end)
        d_valid = (df["d_t"] >= GLOBAL_START) & (df["d_t"] < global_end)

        o_bin = np.full(len(df), -1, dtype=np.int64)
        o_bin[o_valid.to_numpy()] = (
            (df.loc[o_valid, "o_t"] - GLOBAL_START) // pd.Timedelta(minutes=BIN_MINUTES)
        ).to_numpy()

        d_bin = np.full(len(df), -1, dtype=np.int64)
        d_bin[d_valid.to_numpy()] = (
            (df.loc[d_valid, "d_t"] - GLOBAL_START) // pd.Timedelta(minutes=BIN_MINUTES)
        ).to_numpy()

        o_lat = df["o_lat"].to_numpy()
        o_lon = df["o_lng"].to_numpy()
        d_lat = df["d_lat"].to_numpy()
        d_lon = df["d_lng"].to_numpy()

        for i in range(n_sites):
            o_mask = (
                o_valid.to_numpy()
                & (np.abs(o_lat - site_lat[i]) < GRID_HALF)
                & (np.abs(o_lon - site_lon[i]) < GRID_HALF)
            )
            if o_mask.any():
                counts = np.bincount(o_bin[o_mask], minlength=N_BINS)[:N_BINS]
                origin_counts[i] += counts

            d_mask = (
                d_valid.to_numpy()
                & (np.abs(d_lat - site_lat[i]) < GRID_HALF)
                & (np.abs(d_lon - site_lon[i]) < GRID_HALF)
            )
            if d_mask.any():
                counts = np.bincount(d_bin[d_mask], minlength=N_BINS)[:N_BINS]
                dest_counts[i] += counts

        print(f"[{fi + 1:02d}/{len(files)}] {path.name}  rows={len(df)}")

    # day 18 has no source file -> mark as missing (NaN), not real zero demand
    origin_f = origin_counts.astype(np.float64)
    dest_f = dest_counts.astype(np.float64)
    origin_f[:, MISSING_BIN_START:MISSING_BIN_END] = np.nan
    dest_f[:, MISSING_BIN_START:MISSING_BIN_END] = np.nan

    # ---- long-format CSV ----
    bin_ts = [GLOBAL_START + pd.Timedelta(minutes=BIN_MINUTES * b) for b in range(N_BINS)]
    rows = []
    for i in range(n_sites):
        gid = sites["Grid ID"].iloc[i]
        for b in range(N_BINS):
            rows.append((gid, b, bin_ts[b], origin_f[i, b], dest_f[i, b]))
    long_df = pd.DataFrame(rows, columns=["Grid ID", "bin_idx", "timestamp", "origin_demand", "destination_demand"])
    long_df.to_csv(OUT_LONG_CSV, index=False)
    print(f"saved long-format series -> {OUT_LONG_CSV}  ({len(long_df)} rows, "
          f"{long_df['origin_demand'].isna().sum()} NaN rows from missing 04-18)")

    # ---- sliding windows: lookback=48 -> horizon=1, chronological split ----
    feat = np.stack([origin_f, dest_f], axis=-1).astype(np.float32)  # (n_sites, N_BINS, 2)
    feat = np.transpose(feat, (1, 0, 2))  # (N_BINS, n_sites, 2)
    feature_names = ["origin_demand", "destination_demand"]

    if args.weather_csv:
        ctx, ctx_cols = build_context_channels(DATA_DIR / args.weather_csv, n_sites)
        feat = np.concatenate([feat, ctx], axis=-1)  # (N_BINS, n_sites, 2+K)
        feature_names += ctx_cols
        print(f"added context channels: {ctx_cols}  feat shape: {feat.shape}")

    train_end = TRAIN_DAYS * BINS_PER_DAY          # 1152
    val_end = (TRAIN_DAYS + VAL_DAYS) * BINS_PER_DAY  # 1296
    test_end = (TRAIN_DAYS + VAL_DAYS + TEST_DAYS) * BINS_PER_DAY  # 1440
    assert test_end == N_BINS

    def make_split(t_start, t_end):
        # targets t in [t_start, t_end); each needs t-LOOKBACK >= 0, and the
        # whole [t-LOOKBACK, t+HORIZON) window must avoid the missing 04-18 bins
        Xs, ys, ts = [], [], []
        skipped = 0
        for t in range(max(t_start, LOOKBACK), t_end):
            window = feat[t - LOOKBACK:t + HORIZON]
            if np.isnan(window).any():
                skipped += 1
                continue
            Xs.append(feat[t - LOOKBACK:t])                  # (48, n_sites, 2+K)
            ys.append(feat[t:t + HORIZON][..., :2])          # (1, n_sites, 2) -- predict demand only
            ts.append(bin_ts[t])
        X = np.stack(Xs)  # (n_samples, 48, n_sites, 2)
        y = np.stack(ys)  # (n_samples, 1, n_sites, 2)
        print(f"  split [{t_start},{t_end}): {len(Xs)} samples kept, {skipped} skipped (missing-day overlap)")
        return X, y, np.array(ts, dtype="datetime64[ns]")

    X_train, y_train, ts_train = make_split(0, train_end)
    X_val, y_val, ts_val = make_split(train_end, val_end)
    X_test, y_test, ts_test = make_split(val_end, test_end)

    np.savez_compressed(
        OUT_WINDOWS_NPZ,
        X_train=X_train, y_train=y_train, ts_train=ts_train,
        X_val=X_val, y_val=y_val, ts_val=ts_val,
        X_test=X_test, y_test=y_test, ts_test=ts_test,
        grid_ids=sites["Grid ID"].to_numpy(),
        site_lat=site_lat, site_lon=site_lon,
        feature_names=np.array(feature_names),
        lookback=LOOKBACK, horizon=HORIZON, bin_minutes=BIN_MINUTES,
    )
    print(f"saved windowed dataset -> {OUT_WINDOWS_NPZ}")
    print(f"train X:{X_train.shape} y:{y_train.shape}")
    print(f"val   X:{X_val.shape} y:{y_val.shape}")
    print(f"test  X:{X_test.shape} y:{y_test.shape}")


if __name__ == "__main__":
    main()
