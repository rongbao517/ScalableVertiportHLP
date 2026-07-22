# -*- coding: utf-8 -*-
"""
Binary-logit mode-choice adjustment of route_assignment_od_to_vertiports.py's
routed demand, replacing the earlier hard "UAM faster than direct ground ->
100% adopts, else 0%" filter with a smooth adoption probability.

No revealed-preference UAM choice data exists anywhere (the mode doesn't
operate in this project's 2015 Shanghai data), so the behavioral parameters
below are NOT locally estimated -- they're a mix of literature citations and
explicitly-flagged assumptions (per user decision, "literature-parameter
logit" depth):

  VOT_UAM_CNY_PER_HOUR = 140.0
      Cited: "Integrated planning of vertiports and vertipads infrastructure
      for Urban Air Mobility", Transportation Research Part A, vol. 209
      (2026) -- a Shanghai-specific stated-preference study; UAM users'
      implied value of time from their MNL fit (vs. taxi 47.65, private car
      9.75, public transport 17.84 CNY/h in the same study).

  UAM_FARE_PER_KM = 6.0 (CNY/km, no base fee)
      Cited: AutoFlight's Shenzhen-Zhuhai eVTOL air-taxi commercial route,
      publicly announced ~6 RMB/km per passenger (2025) -- an actual
      operating fare, not an academic assumption. No literature figure was
      found for a UAM base/handling fee, so none is modeled (this likely
      understates true cost for very short flights, a documented limitation
      rather than a fabricated number).

  GROUND_BASE_FARE_CNY / GROUND_PER_KM_CNY = 10.0 / 2.40
      Shanghai taxi fare structure. The real period-correct rate for our
      raw data's date range (April 2015) was 13 CNY/3km + 2.40 CNY/km
      beyond (verified: Shanghai's flag-fall rose from 13->14 CNY only on
      2015-10-08, i.e. after this dataset); user explicitly chose to round
      the base fare to 10 CNY rather than use the period-exact 13.

  LOGIT_SCALE_CNY: no literature coefficient was recoverable (paywalled) --
      this sets how sharply P(UAM) responds to the CNY-equivalent utility
      gap and is a calibration choice, not a citation. Run at >1 value as a
      sensitivity check rather than asserting a single "true" scale.

  ADOPTION_FLOOR / ADOPTION_CEIL = 0.14 / 0.86
      Cited: a stated-preference study (8 choice scenarios) found ~14% of
      respondents always chose air taxi and ~14% never did regardless of
      time/cost -- modeled as bounds P(UAM) is clamped into, not a soft
      prior, since the source describes hard behavioral polarization.

Waiting time (per user spec): waiting is modeled as an independent generalized-
cost component, not folded into in-vehicle time at a 1:1 rate -- consistent
with UAM mode-choice studies that treat access/wait/boarding/cost as separate
attributes rather than a single "time" scalar.

    T_UAM_eff = T_UAM_base + kappa_w * W
    GC_UAM    = Fare_UAM + VOT * T_UAM_eff / 60      (VOT is CNY/hour, T in minutes)
    GC_ground = Fare_ground + VOT * T_ground / 60

  W (--wait-time-min, default 0.0): expected/realized wait time in minutes,
      fed back from fleet_sim's mean_age_at_service_bins output (bins * 30)
      -- this is what makes the demand<->operations loop an equilibrium
      rather than a one-shot pipeline. 0.0 recovers the original no-feedback
      behavior exactly.
  kappa_w (--kappa-w, default 1.5): waiting-time penalty multiplier relative
      to ordinary travel time. Not asserted as "correct" -- literature on
      wait-time aversion across transport modes commonly reports multipliers
      in the ~1.3-1.76+ range, varying by mode/context; UAM's actual waiting
      experience (reservations, real-time info, indoor lounges) plausibly
      sits closer to ride-hailing/air-travel norms than a bus-stop transit
      wait, so no single value is treated as ground truth. Main scenario
      kappa_w=1.5; sensitivity-check across {1.0, 1.5, 2.0, 2.5}.

Output columns mirror build_dynspeed_full_demand.py's routed-CSV input
contract (o_lat/o_lon/d_lat/d_lon/hour_of_day/day_type/trip_count/
takeoff_grid_id/landing_grid_id) so its --routed-csv flag can consume this
script's output directly, with trip_count replaced by expected (fractional)
UAM-adopting trip count.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUT_DIR = PROJECT_DIR / "outputs"

ROUTED_CSV = OUT_DIR / "route_assignment_kmeans30_noselfloop_dynamic_speed.csv"
SITES_CSV = DATA_DIR / "selected_sites_kmeans_K30.csv"

VOT_UAM_CNY_PER_HOUR = 140.0
UAM_FARE_PER_KM = 6.0
GROUND_BASE_FARE_CNY = 10.0
GROUND_BASE_KM = 3.0
GROUND_PER_KM_CNY = 2.40
ADOPTION_FLOOR = 0.14
ADOPTION_CEIL = 0.86


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--routed-csv", type=str, default=str(ROUTED_CSV))
    ap.add_argument("--sites-csv", type=str, default=str(SITES_CSV))
    ap.add_argument("--logit-scale", type=float, default=30.0,
                     help="CNY-equivalent spread of the logit response; not literature-derived, "
                          "a calibration choice -- rerun at another value (e.g. 60.0) to sensitivity-check.")
    ap.add_argument("--wait-time-min", type=float, default=0.0,
                     help="expected/realized UAM wait time in minutes (e.g. from fleet_sim's "
                          "mean_age_at_service_bins * 30), applied uniformly to all UAM trips this "
                          "call. 0.0 (default) recovers the original no-wait-feedback behavior.")
    ap.add_argument("--kappa-w", type=float, default=1.5,
                     help="waiting-time penalty multiplier relative to ordinary travel time; "
                          "main scenario 1.5, sensitivity-check {1.0, 1.5, 2.0, 2.5}.")
    ap.add_argument("--out-csv", type=str, required=True)
    args = ap.parse_args()

    routed = pd.read_csv(args.routed_csv)
    sites = pd.read_csv(args.sites_csv).set_index("Grid ID")[["avg_lat", "avg_lon"]]

    takeoff_lat = routed["takeoff_grid_id"].map(sites["avg_lat"]).to_numpy()
    takeoff_lon = routed["takeoff_grid_id"].map(sites["avg_lon"]).to_numpy()
    landing_lat = routed["landing_grid_id"].map(sites["avg_lat"]).to_numpy()
    landing_lon = routed["landing_grid_id"].map(sites["avg_lon"]).to_numpy()

    ground_km = haversine_km(routed["o_lat"], routed["o_lon"], routed["d_lat"], routed["d_lon"])
    uam_flight_km = haversine_km(takeoff_lat, takeoff_lon, landing_lat, landing_lon)

    fare_ground = GROUND_BASE_FARE_CNY + np.maximum(0.0, ground_km - GROUND_BASE_KM) * GROUND_PER_KM_CNY
    fare_uam = UAM_FARE_PER_KM * uam_flight_km

    t_uam_base_min = routed["uam_time_min"]
    t_uam_eff_min = t_uam_base_min + args.kappa_w * args.wait_time_min
    t_ground_min = routed["direct_ground_time_min"]

    gc_uam = fare_uam + VOT_UAM_CNY_PER_HOUR * t_uam_eff_min / 60.0
    gc_ground = fare_ground + VOT_UAM_CNY_PER_HOUR * t_ground_min / 60.0

    # utility_diff = -(GC_UAM - GC_ground): UAM adopted more readily the cheaper its
    # generalized cost is relative to ground's, matching the earlier delta_time/delta_cost
    # formulation exactly when wait_time_min=0 (GC_uam-GC_ground reduces to fare_uam-fare_ground
    # - VOT*(t_ground-t_uam)/60, i.e. delta_cost_cny - VOT*delta_time_hours, the same quantity
    # with a sign flip).
    utility_diff_cny = gc_ground - gc_uam

    p_raw = 1.0 / (1.0 + np.exp(-utility_diff_cny / args.logit_scale))
    p_uam = ADOPTION_FLOOR + (ADOPTION_CEIL - ADOPTION_FLOOR) * p_raw

    result = routed[["o_lat", "o_lon", "d_lat", "d_lon", "hour_of_day", "day_type",
                      "takeoff_grid_id", "landing_grid_id"]].copy()
    result["trip_count"] = routed["trip_count"] * p_uam
    result["p_uam"] = p_uam

    result.drop(columns=["p_uam"]).to_csv(args.out_csv, index=False)

    total_trips_ground = routed["trip_count"].sum()
    total_trips_uam = result["trip_count"].sum()
    mean_p = np.average(p_uam, weights=routed["trip_count"])
    print(f"logit_scale={args.logit_scale}  total ground trips: {total_trips_ground:.0f}  "
          f"expected UAM-adopting trips: {total_trips_uam:.0f}  "
          f"(trip-weighted mean P(UAM)={mean_p:.4f}, {total_trips_uam/total_trips_ground*100:.1f}% of ground demand)")
    print(f"saved -> {args.out_csv}")


if __name__ == "__main__":
    main()
