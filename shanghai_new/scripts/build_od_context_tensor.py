# -*- coding: utf-8 -*-
"""
Build the (channel, T, N, N) tensor expected by the ASTT / ContextAwareUAMNetV2
architecture (scripts/od_astt/uam_airspace_context_od_v2.py, ported from
/home/b5by/zhirong.b5by/UAM_demand/src_repo/model/ -- the "OD-preserving +
directed graph prior + FiLM weather modulation" architecture that scored best
(test log-MSE 0.2921) among everything tried on the NYC OD task).

channel 0: real site-to-site OD demand (shanghai_od_kmeans30_30min_1channel.npz),
           raw counts -- log1p'd later by the training script itself, same as
           the NYC pipeline does in load_data().
channel 1..6: calendar/weather covariates from shanghai_calendar_weather_202504.csv,
           broadcast to every (i, j) cell and every 30-min bin within a day
           (constant across the day and across all OD pairs -- same broadcasting
           convention the NYC WeatherCalendarContextEncoder expects, since it
           pools context with `.mean(dim=(1, 2))` over the O, D axes anyway):
             - is_holiday_or_weekend (0/1)
             - temp_max_c, temp_min_c, precip_mm, windspeed_max_kmh, humidity_pct
           This is DAILY-resolution weather (no real hourly Shanghai weather
           exists), unlike NYC's true hourly channels -- a real signal at
           coarser resolution, not the zero-filled dummy channels the earlier
           failed Shanghai OD attempt (R2=0.19) used.

Missing day handling matches extract_shanghai_od_matrix_kmeans30.py exactly:
day index 17 (2015-04-18, a byte-duplicate of 04-02) is dropped from the
1440-bin full timeline -> 1392 compressed bins, so this tensor's time axis
lines up 1:1 with the OD tensor's.
"""
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUT_DIR = PROJECT_DIR / "outputs"

OD_NPZ = OUT_DIR / "shanghai_od_kmeans30_30min_1channel.npz"
CALENDAR_CSV = DATA_DIR / "shanghai_calendar_weather_202504.csv"
OUT_NPZ = DATA_DIR / "shanghai_od_kmeans30_context7ch.npz"

N_DAYS = 30
BINS_PER_DAY = 48
N_BINS_FULL = N_DAYS * BINS_PER_DAY  # 1440
MISSING_DAY_IDX = 17  # 2015-04-18
MISSING_BIN_START = MISSING_DAY_IDX * BINS_PER_DAY
MISSING_BIN_END = MISSING_BIN_START + BINS_PER_DAY

CONTEXT_CHANNEL_NAMES = [
    "is_holiday_or_weekend",
    "temp_max_c",
    "temp_min_c",
    "precip_mm",
    "windspeed_max_kmh",
    "humidity_pct",
]


def main():
    od = np.load(OD_NPZ, allow_pickle=True)["arr_0"]  # (1, T_compressed, N, N)
    assert od.shape[0] == 1
    demand = od[0]  # (T_compressed, N, N)
    t_compressed, n_nodes, _ = demand.shape

    cal = pd.read_csv(CALENDAR_CSV)
    assert len(cal) == N_DAYS, f"expected {N_DAYS} calendar rows, got {len(cal)}"
    cal["is_holiday_or_weekend"] = (cal["is_holiday"] | cal["is_weekend"]).astype(float)

    # Broadcast each day's scalar covariates to BINS_PER_DAY consecutive full-timeline bins.
    full_context = np.zeros((len(CONTEXT_CHANNEL_NAMES), N_BINS_FULL), dtype=np.float32)
    for day_idx, row in cal.iterrows():
        sl = slice(day_idx * BINS_PER_DAY, (day_idx + 1) * BINS_PER_DAY)
        for c, name in enumerate(CONTEXT_CHANNEL_NAMES):
            full_context[c, sl] = float(row[name])

    context_compressed = np.delete(full_context, np.s_[MISSING_BIN_START:MISSING_BIN_END], axis=1)
    assert context_compressed.shape[1] == t_compressed, (
        f"context bins {context_compressed.shape[1]} != OD bins {t_compressed}"
    )

    # Broadcast context (channel, T) -> (channel, T, N, N), same value for every OD pair.
    context_grid = np.broadcast_to(
        context_compressed[:, :, None, None], (len(CONTEXT_CHANNEL_NAMES), t_compressed, n_nodes, n_nodes)
    )

    arr_0 = np.concatenate([demand[None], context_grid], axis=0).astype(np.float32)  # (7, T, N, N)
    np.savez_compressed(OUT_NPZ, arr_0=arr_0)
    print(f"saved -> {OUT_NPZ}  shape={arr_0.shape}")
    print("channels: [demand] +", CONTEXT_CHANNEL_NAMES)


if __name__ == "__main__":
    main()
