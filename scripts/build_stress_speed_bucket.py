# -*- coding: utf-8 -*-
"""
Stress-test variant of predicted_ground_speed_by_bucket.csv: the 6 workday
rush-hour buckets (hour_of_day in {6,7,8,9,17,18}) are replaced with their
empirical p10 ("worst 10% of days at that hour") ground-truth speed instead
of the mean the model-predicted table uses; every other bucket (all offday
buckets, all non-rush workday buckets) is left untouched.

Why p10 and not something more extreme: within-bucket day-to-day std is
small relative to the hour-of-day/day_type signal itself (mean within-bucket
std ~0.91 km/h vs ~3.7 km/h bucket-to-bucket -- the 48-bucket table already
captures ~86% of total variance), and with only 21 workday samples per
bucket, anything past p10 (e.g. min/p5) is a single noisy observation, not a
stable "bad day" estimate.

Purpose: quantify how much a genuinely bad rush-hour day (not just the
"typical" day the mean-bucket table represents) would degrade route-assignment
and fleet_sim results -- see build_dynspeed_full_demand.py /
run_shanghai_fleet_simulation.py's --speed-bucket-csv for where this feeds in.
"""
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"

GROUND_TRUTH_CSV = DATA_DIR / "shanghai_ground_speed_30min.csv"
WEATHER_CSV = DATA_DIR / "shanghai_calendar_weather_202504.csv"
BASE_BUCKET_CSV = DATA_DIR / "predicted_ground_speed_by_bucket.csv"
OUT_CSV = DATA_DIR / "predicted_ground_speed_by_bucket_stress_p10rush.csv"

RUSH_HOURS = [6, 7, 8, 9, 17, 18]
RUSH_DAY_TYPE = "workday"


def main():
    gt = pd.read_csv(GROUND_TRUTH_CSV, parse_dates=["timestamp"])
    cal = pd.read_csv(WEATHER_CSV, parse_dates=["date"])
    is_workday = dict(zip(cal["date"].dt.strftime("%Y%m%d"), cal["day_type"].eq("workday")))
    gt["date_str"] = gt["timestamp"].dt.strftime("%Y%m%d")
    gt["day_type"] = np.where(gt["date_str"].map(is_workday), "workday", "offday")
    gt["hour"] = gt["timestamp"].dt.hour

    p10_by_hour = (
        gt[(gt["day_type"] == RUSH_DAY_TYPE) & (gt["hour"].isin(RUSH_HOURS))]
        .groupby("hour")["median_speed_kmh"]
        .quantile(0.10)
    )

    bucket = pd.read_csv(BASE_BUCKET_CSV)
    stress = bucket.copy()
    rush_mask = (stress["day_type"] == RUSH_DAY_TYPE) & (stress["hour_of_day"].isin(RUSH_HOURS))
    for h in RUSH_HOURS:
        row_mask = rush_mask & (stress["hour_of_day"] == h)
        old_val = stress.loc[row_mask, "predicted_speed_kmh"].iloc[0]
        stress.loc[row_mask, "predicted_speed_kmh"] = p10_by_hour[h]
        print(f"hour={h} workday: {old_val:.2f} km/h (mean) -> {p10_by_hour[h]:.2f} km/h (p10 stress)")

    stress.to_csv(OUT_CSV, index=False)
    print(f"saved -> {OUT_CSV}  ({rush_mask.sum()} of 48 rows replaced with p10, rest unchanged)")


if __name__ == "__main__":
    main()
