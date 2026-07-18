# -*- coding: utf-8 -*-
"""
Run the trained speed-forecast model (train_shanghai_speed_gru.py) across the
WHOLE month in rolling one-step-ahead mode (each bin predicted from its own
preceding 48-bin real history, not from the model's own past predictions) --
the route-assignment OD table spans all 1392 bins, not just the held-out test
slice, so we need a predicted value for every bin, not just the test period.

Aggregates the resulting predicted-speed series by (hour_of_day, day_type)
into the 48-bucket lookup table route_assignment_od_to_vertiports.py and
validate_route_assignment_with_gurobi.py read (day_type: workday vs offday,
matching extract_full_grid_od_pairs_by_bucket.py's binary collapse of the
calendar's workday/weekend/holiday split).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
SPEED_CSV = DATA_DIR / "shanghai_ground_speed_30min.csv"
WEATHER_CSV = DATA_DIR / "shanghai_calendar_weather_202504.csv"
OUT_CSV = DATA_DIR / "predicted_ground_speed_by_bucket.csv"

GLOBAL_START = pd.Timestamp("2015-04-01 00:00:00")
N_DAYS = 30
BIN_MINUTES = 30
BINS_PER_DAY = 24 * 60 // BIN_MINUTES  # 48
N_BINS = N_DAYS * BINS_PER_DAY  # 1440
LOOKBACK = 48
TRAIN_DAYS = 24

WEATHER_NUMERIC_COLS = ["temp_max_c", "temp_min_c", "precip_mm", "windspeed_max_kmh", "humidity_pct"]


class SiteGRU(nn.Module):
    """Identical to train_shanghai_speed_gru.py's SiteGRU -- must match exactly
    to load the saved state_dict."""
    def __init__(self, in_dim=2, out_dim=2, hidden=64, layers=1, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(
            in_dim, hidden, num_layers=layers, batch_first=True,
            bidirectional=True, dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        out, _ = self.gru(x)
        last = out[:, -1, :]
        return self.head(last)


def find_latest_run_dir():
    root = BASE_DIR / "outputs" / "save_shanghai_speed_gru"
    candidates = [d for d in root.iterdir() if d.is_dir() and (d / "site_gru.pt").exists()]
    if not candidates:
        raise FileNotFoundError(f"no completed speed-GRU run found under {root}")
    return max(candidates, key=lambda d: d.stat().st_mtime)


def build_feat_and_calendar():
    """Same feature construction as build_speed_windows.py -- must match exactly
    since the model was trained on this channel layout/order."""
    speed_df = pd.read_csv(SPEED_CSV, parse_dates=["timestamp"]).sort_values("bin_idx").reset_index(drop=True)
    assert len(speed_df) == N_BINS
    bin_ts = speed_df["timestamp"].tolist()
    speed = speed_df["median_speed_kmh"].to_numpy().astype(np.float32).reshape(N_BINS, 1, 1)

    wdf = pd.read_csv(WEATHER_CSV, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    assert len(wdf) == N_DAYS
    wdf["is_weekend_flag"] = wdf["day_type"].eq("weekend").astype(np.float32)
    wdf["is_holiday_flag"] = wdf["day_type"].eq("holiday").astype(np.float32)
    cols = WEATHER_NUMERIC_COLS + ["is_weekend_flag", "is_holiday_flag"]
    arr = wdf[cols].to_numpy().astype(np.float32)
    n_numeric = len(WEATHER_NUMERIC_COLS)
    mu_w = arr[:TRAIN_DAYS, :n_numeric].mean(axis=0)
    sd_w = arr[:TRAIN_DAYS, :n_numeric].std(axis=0)
    arr[:, :n_numeric] = (arr[:, :n_numeric] - mu_w) / np.where(sd_w > 1e-6, sd_w, 1.0)
    daily_tiled = np.repeat(arr, BINS_PER_DAY, axis=0)
    ctx = daily_tiled[:, None, :].astype(np.float32)  # (N_BINS, 1, K)

    feat = np.concatenate([speed, ctx], axis=-1)  # (N_BINS, 1, 1+K)

    day_is_workday = dict(zip(wdf["date"].dt.strftime("%Y%m%d"), wdf["day_type"].eq("workday")))
    return feat, bin_ts, day_is_workday


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=str, default=None,
                     help="outputs/save_shanghai_speed_gru/<run> dir; default: most recently modified completed run")
    args = ap.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else find_latest_run_dir()
    print(f"using run dir: {run_dir}")
    with open(run_dir / "config.json") as f:
        config = json.load(f)
    mu, sd = config["speed_mu_kmh"], config["speed_sd_kmh"]

    feat, bin_ts, day_is_workday = build_feat_and_calendar()
    in_dim = feat.shape[-1]

    device = torch.device("cpu")
    model = SiteGRU(in_dim=in_dim, out_dim=1, hidden=config["hidden"], layers=config["layers"],
                     dropout=config["dropout"]).to(device)
    model.load_state_dict(torch.load(run_dir / "site_gru.pt", map_location=device))
    model.eval()

    feat_z = feat.copy()
    feat_z[..., 0] = (feat_z[..., 0] - mu) / sd

    # batch every valid window into ONE forward pass instead of looping
    # model() calls one bin at a time (which was >15min of near-pure per-call
    # overhead for ~1344 tiny calls -- batching is the same fix predict()
    # already uses in train_shanghai_speed_gru.py).
    valid_ts, windows = [], []
    for t in range(LOOKBACK, N_BINS):
        window = feat_z[t - LOOKBACK:t]  # (48, 1, C)
        if np.isnan(window).any():
            continue  # touches the missing-day gap -- leave NaN, bucket avg skips it
        valid_ts.append(t)
        windows.append(window)

    predicted_kmh = np.full(N_BINS, np.nan)
    if windows:
        batch = np.stack(windows)  # (n_valid, 48, 1, C)
        x = torch.from_numpy(batch.transpose(0, 2, 1, 3).reshape(-1, LOOKBACK, batch.shape[-1]).astype(np.float32))
        with torch.no_grad():
            pred_z = model(x).squeeze(-1).numpy()  # (n_valid,)
        predicted_kmh[valid_ts] = pred_z * sd + mu

    n_predicted = int(np.isfinite(predicted_kmh).sum())
    print(f"predicted {n_predicted}/{N_BINS} bins (rest fall in the missing-day gap / lack full lookback)")

    hour_of_day = np.array([ts.hour for ts in bin_ts])
    date_str = np.array([ts.strftime("%Y%m%d") for ts in bin_ts])
    is_workday = np.array([day_is_workday.get(d, np.nan) for d in date_str])
    day_type = np.where(is_workday == 1, "workday", "offday")

    df = pd.DataFrame({
        "bin_idx": np.arange(N_BINS),
        "hour_of_day": hour_of_day,
        "day_type": day_type,
        "predicted_speed_kmh": predicted_kmh,
    })
    bucket = (
        df.dropna(subset=["predicted_speed_kmh"])
        .groupby(["hour_of_day", "day_type"])["predicted_speed_kmh"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "predicted_speed_kmh", "count": "n_bins_used"})
    )
    assert len(bucket) == 48, f"expected 48 buckets, got {len(bucket)}"
    bucket.to_csv(OUT_CSV, index=False)
    print(f"saved -> {OUT_CSV}")
    print(bucket.to_string())


if __name__ == "__main__":
    main()
