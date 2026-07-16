# -*- coding: utf-8 -*-
"""
Day-type calendar (workday / weekend / holiday) + daily weather for the
raw data window, 2015-04-01..30 (Shanghai).

Day type: 2015 Qingming Festival holiday per the official State Council
notice (https://www.gov.cn/gongbao/content/2015/content_2799019.htm):
"清明节：4月5日放假，4月6日（星期一）补休" -- 04-05 (Sun) is the holiday,
04-06 (Mon) is a compensatory day off. No weekend workday swap was
announced for this holiday (unlike Spring Festival/National Day), so
04-04 (Sat) remains a normal weekend day. No other April 2015 date falls
on a public holiday.

Weather: Shanghai daily max/min temp, precipitation, max wind speed, mean
relative humidity, from Open-Meteo's ERA5-reanalysis historical archive
(https://open-meteo.com/en/docs/historical-weather-api), lat/lon centered
on the same grid used throughout this project (~31.23N, 121.47E). This is
reanalysis data (gridded model output blended with observations), not a
single ground station's raw readings -- adequate for a demand-model
covariate, not for exact station-observation-grade meteorology.
"""
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUT_CSV = DATA_DIR / "shanghai_calendar_weather_202504.csv"

HOLIDAYS = {"2015-04-05", "2015-04-06"}  # Qingming holiday + compensatory day off


def build_calendar():
    dates = pd.date_range("2015-04-01", "2015-04-30", freq="D")
    df = pd.DataFrame({"date": dates})
    df["day_of_week"] = df["date"].dt.day_name()
    date_str = df["date"].dt.strftime("%Y-%m-%d")
    is_holiday = date_str.isin(HOLIDAYS)
    is_weekend = df["date"].dt.weekday >= 5  # Sat=5, Sun=6
    df["is_holiday"] = is_holiday
    df["is_weekend"] = is_weekend & ~is_holiday  # holiday takes precedence in day_type
    df["day_type"] = "workday"
    df.loc[is_weekend & ~is_holiday, "day_type"] = "weekend"
    df.loc[is_holiday, "day_type"] = "holiday"
    return df


# Open-Meteo ERA5 archive API, lat/lon = project's grid center (~31.23N, 121.47E),
# fetched 2026-07-07: https://archive-api.open-meteo.com/v1/archive?latitude=31.23&
# longitude=121.47&start_date=2015-04-01&end_date=2015-04-30&daily=temperature_2m_max,
# temperature_2m_min,precipitation_sum,windspeed_10m_max,relative_humidity_2m_mean&
# timezone=Asia/Shanghai
WEATHER_RAW = {
    "temp_max_c": [21.5, 27.4, 18.9, 23.6, 15.9, 12.5, 9.2, 11.6, 13.7, 15.6, 16.4, 17.6, 14.2, 16.0, 20.6,
                   23.5, 17.7, 25.9, 25.6, 16.6, 18.0, 21.9, 22.1, 19.8, 24.3, 23.6, 22.1, 28.5, 21.6, 24.9],
    "temp_min_c": [15.2, 16.6, 11.9, 13.1, 11.6, 9.4, 5.4, 5.2, 3.6, 8.8, 5.6, 9.5, 7.5, 6.8, 9.1,
                   10.9, 10.3, 15.9, 14.2, 11.1, 8.0, 7.5, 11.5, 12.8, 10.8, 13.6, 14.7, 17.0, 15.3, 13.9],
    "precip_mm": [0.00, 5.40, 21.30, 0.00, 1.10, 12.00, 8.50, 0.00, 0.00, 0.00, 0.00, 0.10, 0.40, 1.30, 0.00,
                  0.80, 0.00, 0.00, 3.90, 2.40, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.20, 0.20, 0.00, 0.00],
    "windspeed_max_kmh": [20.0, 24.3, 16.4, 19.6, 21.1, 25.1, 24.0, 20.4, 11.1, 13.8, 13.7, 13.0, 30.9, 34.4, 17.5,
                          22.7, 26.6, 20.5, 20.2, 24.1, 11.3, 10.5, 16.2, 12.9, 10.6, 19.8, 24.9, 18.9, 21.6, 15.6],
    "humidity_pct": [92, 82, 85, 85, 87, 84, 77, 64, 75, 78, 76, 77, 70, 70, 66,
                     68, 69, 81, 85, 73, 68, 73, 58, 71, 76, 61, 78, 76, 79, 77],
}


def main():
    df = build_calendar()
    for k, v in WEATHER_RAW.items():
        assert len(v) == len(df), f"{k}: expected 30 values, got {len(v)}"
        df[k] = v

    # 04-18 has no real raw trip data (duplicate file, see extract_shanghai_30min_demand.py);
    # flag it so downstream joins can decide whether to exclude it like the demand pipeline does.
    df["has_real_trip_data"] = df["date"].dt.strftime("%Y-%m-%d") != "2015-04-18"

    df.to_csv(OUT_CSV, index=False)
    print(f"saved -> {OUT_CSV}")
    print(df.to_string(index=False))
    print(f"\nday_type counts: {df['day_type'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
