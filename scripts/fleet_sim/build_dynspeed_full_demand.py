# -*- coding: utf-8 -*-
"""
Converts route_assignment_od_to_vertiports.py's full-grid (1676-cell) routed
output -- real per-trip access/egress distances, real dynamic ground speed
already baked into which vertiport pair each trip uses -- into the
(takeoff_vertiport, landing_vertiport, hour_of_day, day_type) bucket table
run_shanghai_fleet_simulation.py needs to drive the fleet-capacity
simulation on this larger, more realistic demand instead of the current
~184K-trip intra-cell-only subset (shanghai_od_kmeans30_30min_1channel.npz).

route_assignment_kmeans30_noselfloop_dynamic_speed.csv's trip_count is a
MONTHLY sum per (grid_o, grid_d, hour_of_day, day_type) bucket (see
extract_full_grid_od_pairs_by_bucket.py) -- it has no per-day breakdown, so
there is no way to recover which specific day within a bucket contributed
how many trips. To turn this into a per-bin demand figure for the
simulation's 1392-bin chronological loop, this script divides each bucket's
total by how many days of that day_type actually appear in the 29-day raw
dataset (WORKDAY_DAYS=21, OFFDAY_DAYS=8 -- see day_type value_counts on
shanghai_calendar_weather_202504.csv, with 2015-04-18 excluded to match
extract_full_grid_od_pairs_by_bucket.py's DUPLICATE_FILES exclusion), giving
an average per-occurrence trip count applied identically to every bin
sharing that bucket. This necessarily loses day-to-day demand variation
within a bucket (the same simplification the speed-forecast bucket table
and the ground-speed-per-bucket lookup already make), but preserves the
aggregate monthly volume and -- crucially for this exercise -- the real,
distance-varying access/egress legs and the routing decisions that dynamic
ground speed already influenced.

Access/egress distance is aggregated as a trip-count-weighted average per
(takeoff, landing, hour_of_day, day_type) group, since many different
(grid_o, grid_d) pairs can route through the same vertiport pair in the same
bucket with different individual distances.
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
OUT_CSV = OUT_DIR / "fleet_sim_dynspeed_full_demand_bucket.csv"

# days of each type among the 29 raw daily files actually used (30 calendar
# days minus 2015-04-18, excluded everywhere upstream as a duplicate of
# 2015-04-02 -- it happens to be a "weekend" day in the calendar, so offday
# drops from 9 to 8).
DAY_TYPE_COUNTS = {"workday": 21, "offday": 8}
# each hour_of_day bucket spans TWO 30-min simulation bins per day (e.g.
# hour_of_day=7 matches both the 07:00 and 07:30 bins) -- both get the same
# per-bin average, so the divisor must be day_count * BINS_PER_HOUR, not
# just day_count, or the reconstructed per-bin total double-counts (caught
# via run_shanghai_fleet_simulation.py's own total-demand sanity print
# showing 23.15M instead of the expected 11.58M -- exactly 2x).
BINS_PER_HOUR = 2


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
    ap.add_argument("--sites-csv", type=str, default=str(SITES_CSV),
                     help="must match whichever site set --routed-csv was routed against, or "
                          "access/egress distances for any non-matching vertiport silently "
                          "zero out (NaN swallowed by groupby.sum()'s default skipna).")
    ap.add_argument("--out-csv", type=str, default=str(OUT_CSV))
    args = ap.parse_args()

    routed = pd.read_csv(args.routed_csv)
    sites = pd.read_csv(args.sites_csv).set_index("Grid ID")[["avg_lat", "avg_lon"]]

    takeoff_lat = routed["takeoff_grid_id"].map(sites["avg_lat"]).to_numpy()
    takeoff_lon = routed["takeoff_grid_id"].map(sites["avg_lon"]).to_numpy()
    landing_lat = routed["landing_grid_id"].map(sites["avg_lat"]).to_numpy()
    landing_lon = routed["landing_grid_id"].map(sites["avg_lon"]).to_numpy()

    routed["access_km"] = haversine_km(routed["o_lat"], routed["o_lon"], takeoff_lat, takeoff_lon)
    routed["egress_km"] = haversine_km(routed["d_lat"], routed["d_lon"], landing_lat, landing_lon)
    routed["w_access"] = routed["access_km"] * routed["trip_count"]
    routed["w_egress"] = routed["egress_km"] * routed["trip_count"]

    grouped = routed.groupby(["takeoff_grid_id", "landing_grid_id", "hour_of_day", "day_type"]).agg(
        total_trip_count=("trip_count", "sum"),
        w_access=("w_access", "sum"),
        w_egress=("w_egress", "sum"),
    ).reset_index()
    grouped["access_km"] = grouped["w_access"] / grouped["total_trip_count"]
    grouped["egress_km"] = grouped["w_egress"] / grouped["total_trip_count"]
    grouped["day_count"] = grouped["day_type"].map(DAY_TYPE_COUNTS)
    grouped["avg_trip_count_per_bin"] = grouped["total_trip_count"] / (grouped["day_count"] * BINS_PER_HOUR)

    result = grouped[["takeoff_grid_id", "landing_grid_id", "hour_of_day", "day_type",
                       "total_trip_count", "avg_trip_count_per_bin", "access_km", "egress_km"]]
    result.to_csv(args.out_csv, index=False)
    print(f"vertiport-pair x bucket rows: {len(result)}")
    print(f"total monthly trip_count (sanity check vs route_assignment total): {result['total_trip_count'].sum():.0f}")
    print(f"implied avg demand per bin (sum over all pairs/buckets / 48): "
          f"{result['avg_trip_count_per_bin'].sum() / 48:.1f}")
    print(f"access_km: mean={result['access_km'].mean():.2f} median={result['access_km'].median():.2f} "
          f"max={result['access_km'].max():.2f}")
    print(f"saved -> {args.out_csv}")


if __name__ == "__main__":
    main()
