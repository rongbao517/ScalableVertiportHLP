# -*- coding: utf-8 -*-
"""
Sliding-window train/val/test sets for next-step ground-speed forecasting,
built from data/shanghai_ground_speed_30min.csv (extract_shanghai_ground_speed.py).

Mirrors extract_shanghai_30min_demand.py's windowing convention as closely as
possible (same LOOKBACK/HORIZON/day-split constants, same missing-day NaN-skip
logic), with n_sites=1 since ground speed here is one city-wide series, not
per-vertiport. Target is NOT log1p'd (speed isn't a count, unlike demand) --
z-scoring happens later in the training script instead, same place
train_shanghai_demand_gru.py currently does its log1p.
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
SPEED_CSV = DATA_DIR / "shanghai_ground_speed_30min.csv"
WEATHER_CSV = DATA_DIR / "shanghai_calendar_weather_202504.csv"

OUT_NPZ = DATA_DIR / "shanghai_speed_windows.npz"

GLOBAL_START = pd.Timestamp("2015-04-01 00:00:00")
N_DAYS = 30
BIN_MINUTES = 30
BINS_PER_DAY = 24 * 60 // BIN_MINUTES  # 48
N_BINS = N_DAYS * BINS_PER_DAY  # 1440

LOOKBACK = 48
HORIZON = 1
TRAIN_DAYS = 24
VAL_DAYS = 3
TEST_DAYS = 3

WEATHER_NUMERIC_COLS = ["temp_max_c", "temp_min_c", "precip_mm", "windspeed_max_kmh", "humidity_pct"]


def build_context_channels():
    """Same convention as extract_shanghai_30min_demand.py's build_context_channels:
    day-level weather/day-type tiled to 30-min bins, numeric cols z-scored using
    TRAIN-period days only. Returns (N_BINS, 1, K) float32 array + channel names."""
    wdf = pd.read_csv(WEATHER_CSV, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    assert len(wdf) == N_DAYS, f"expected {N_DAYS} calendar rows, got {len(wdf)}"

    wdf["is_weekend_flag"] = wdf["day_type"].eq("weekend").astype(np.float32)
    wdf["is_holiday_flag"] = wdf["day_type"].eq("holiday").astype(np.float32)
    cols = WEATHER_NUMERIC_COLS + ["is_weekend_flag", "is_holiday_flag"]

    arr = wdf[cols].to_numpy().astype(np.float32)  # (N_DAYS, K)
    n_numeric = len(WEATHER_NUMERIC_COLS)
    mu = arr[:TRAIN_DAYS, :n_numeric].mean(axis=0)
    sd = arr[:TRAIN_DAYS, :n_numeric].std(axis=0)
    arr[:, :n_numeric] = (arr[:, :n_numeric] - mu) / np.where(sd > 1e-6, sd, 1.0)

    daily_tiled = np.repeat(arr, BINS_PER_DAY, axis=0)  # (N_BINS, K)
    ctx = daily_tiled[:, None, :]  # (N_BINS, 1, K) -- n_sites=1
    return ctx.astype(np.float32), cols


def main():
    speed_df = pd.read_csv(SPEED_CSV, parse_dates=["timestamp"]).sort_values("bin_idx").reset_index(drop=True)
    assert len(speed_df) == N_BINS, f"expected {N_BINS} rows, got {len(speed_df)}"
    bin_ts = speed_df["timestamp"].tolist()

    speed = speed_df["median_speed_kmh"].to_numpy().astype(np.float32).reshape(N_BINS, 1, 1)  # (N_BINS, 1, 1)
    feature_names = ["median_speed_kmh"]

    ctx, ctx_cols = build_context_channels()
    feat = np.concatenate([speed, ctx], axis=-1)  # (N_BINS, 1, 1+K)
    feature_names += ctx_cols
    print(f"feat shape: {feat.shape}  channels: {feature_names}")

    train_end = TRAIN_DAYS * BINS_PER_DAY          # 1152
    val_end = (TRAIN_DAYS + VAL_DAYS) * BINS_PER_DAY  # 1296
    test_end = (TRAIN_DAYS + VAL_DAYS + TEST_DAYS) * BINS_PER_DAY  # 1440
    assert test_end == N_BINS

    def make_split(t_start, t_end):
        Xs, ys, ts = [], [], []
        skipped = 0
        for t in range(max(t_start, LOOKBACK), t_end):
            window = feat[t - LOOKBACK:t + HORIZON]
            if np.isnan(window).any():
                skipped += 1
                continue
            Xs.append(feat[t - LOOKBACK:t])                   # (48, 1, 1+K)
            ys.append(feat[t:t + HORIZON][..., :1])           # (1, 1, 1) -- predict speed only
            ts.append(bin_ts[t])
        X = np.stack(Xs)
        y = np.stack(ys)
        print(f"  split [{t_start},{t_end}): {len(Xs)} samples kept, {skipped} skipped (missing-day overlap)")
        return X, y, np.array(ts, dtype="datetime64[ns]")

    X_train, y_train, ts_train = make_split(0, train_end)
    X_val, y_val, ts_val = make_split(train_end, val_end)
    X_test, y_test, ts_test = make_split(val_end, test_end)

    np.savez_compressed(
        OUT_NPZ,
        X_train=X_train, y_train=y_train, ts_train=ts_train,
        X_val=X_val, y_val=y_val, ts_val=ts_val,
        X_test=X_test, y_test=y_test, ts_test=ts_test,
        feature_names=np.array(feature_names),
        lookback=LOOKBACK, horizon=HORIZON, bin_minutes=BIN_MINUTES,
    )
    print(f"saved windowed dataset -> {OUT_NPZ}")
    print(f"train X:{X_train.shape} y:{y_train.shape}")
    print(f"val   X:{X_val.shape} y:{y_val.shape}")
    print(f"test  X:{X_test.shape} y:{y_test.shape}")


if __name__ == "__main__":
    main()
