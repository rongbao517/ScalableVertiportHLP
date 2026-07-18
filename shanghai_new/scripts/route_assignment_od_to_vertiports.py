# -*- coding: utf-8 -*-
"""
Route assignment: for every real trip O->D (aggregated to unique
(grid_o, grid_d, hour_of_day, day_type) buckets with a trip count, from
full_grid_od_pairs_by_bucket_202504.csv), choose which of the 30
kmeans-selected vertiport sites to take off from (i) and land at (j), so the
trip actually goes O -> vertiport_i -> vertiport_j -> D.

Objective (per user decision): minimize total door-to-door travel TIME, not
distance -- ground and air legs use different speeds, which is the entire
point of a UAM network (a longer-looking detour through vertiports can still
be faster than direct ground travel because the air leg is much quicker per km).

    cost(O, D, i, j) = access_time(O, i) + flight_time(i, j) + egress_time(j, D)
    access_time(O, i) = haversine_km(O, i) / ground_speed[bucket] * 60 + DWELL_MIN
    flight_time(i, j) = haversine_km(i, j) / AIR_SPEED_KMH * 60
    egress_time(j, D) = haversine_km(j, D) / ground_speed[bucket] * 60 + DWELL_MIN

Ground speed is now DYNAMIC, predicted per (hour_of_day, day_type) bucket by
train_shanghai_speed_gru.py (via predict_speed_by_bucket.py), instead of the
flat GROUND_SPEED_KMH=15.0 constant this script used to hardcode -- see
predict_speed_by_bucket.py / extract_shanghai_ground_speed.py for how that's
derived from real GPS trip data. This is also why the OD table itself had to
change from the old whole-month aggregate (full_grid_od_pairs_202504.csv) to
a per-bucket one (extract_full_grid_od_pairs_by_bucket.py): once ground speed
varies by time of day, the optimal (i, j) for a given (grid_o, grid_d) pair
can vary by time of day too, so trips can no longer be aggregated across the
whole month before solving.

No vertiport/corridor capacity constraint yet (per user decision) -- every
trip independently picks its own best (i, j) *within its bucket*, so this is
separable per (grid_o, grid_d, bucket) triple and solved by exact vectorized
argmin over all 30x30 combinations rather than one giant assignment LP.
See validate_route_assignment_with_gurobi.py for a Gurobi formulation of the
same per-pair problem, used to confirm this numpy solution is exactly what
Gurobi would return.

Speed/dwell assumptions (edit these for your actual operator specs):
  ground_speed  -- per-(hour_of_day, day_type) bucket, from
                    data/predicted_ground_speed_by_bucket.csv (dynamic)
  AIR_SPEED_KMH = 200 -- typical eVTOL cruise speed assumed in UAM studies (static:
                    no real eVTOL flight data exists in this project to predict it from)
  DWELL_MIN     = 5   -- fixed boarding/security/taxi overhead per vertiport leg
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUT_DIR = PROJECT_DIR / "outputs"

OD_PAIRS_CSV = DATA_DIR / "full_grid_od_pairs_by_bucket_202504.csv"
SPEED_BUCKET_CSV = DATA_DIR / "predicted_ground_speed_by_bucket.csv"
SITES_CSV = DATA_DIR / "selected_sites_kmeans_K30.csv"
# NOTE: deliberately NOT overwriting the original static-speed
# route_assignment_kmeans30_noselfloop.csv -- this dynamic-speed run is saved
# under its own filename so both results stay available for comparison.
OUT_ASSIGNMENT_CSV = OUT_DIR / "route_assignment_kmeans30_noselfloop_dynamic_speed.csv"
OUT_SUMMARY_CSV = OUT_DIR / "route_assignment_kmeans30_noselfloop_dynamic_speed_vertiport_summary.csv"

AIR_SPEED_KMH = 200.0
DWELL_MIN = 5.0

# i == j (same takeoff/landing vertiport) is excluded: a trip whose nearest
# vertiport happens to be the same one on both ends gains nothing from a
# zero-distance "flight" leg plus two dwell penalties -- it should never be
# routed through the UAM network at all. Forcing it through the nearest
# single vertiport just double-counts the dwell time for no benefit, so
# those pairs are pushed to +inf and can never be chosen as i==j; whichever
# distinct-site pair is next-best wins instead.
EXCLUDE_SELF_LOOP = True

CHUNK_SIZE = 50_000


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def haversine_matrix(lat1, lon1, lat2, lon2):
    """lat1/lon1: (n,), lat2/lon2: (m,) -> (n, m) km matrix."""
    R = 6371.0
    p1 = np.radians(lat1)[:, None]
    p2 = np.radians(lat2)[None, :]
    dphi = np.radians(lat2[None, :] - lat1[:, None])
    dlmb = np.radians(lon2[None, :] - lon1[:, None])
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed-csv", type=str, default=str(SPEED_BUCKET_CSV),
                     help="48-bucket (hour_of_day, day_type) ground-speed lookup; default is the "
                          "model-predicted mean table. Pass an alternate (e.g. a p10 rush-hour stress "
                          "variant) to test sensitivity without touching the default outputs.")
    ap.add_argument("--sites-csv", type=str, default=str(SITES_CSV),
                     help="vertiport site set; default is the oracle full-month-demand K-means "
                          "selection. Pass an alternate (e.g. the naive-forecast-based site set) to "
                          "test sensitivity without touching the default outputs.")
    ap.add_argument("--out-assignment-csv", type=str, default=str(OUT_ASSIGNMENT_CSV))
    ap.add_argument("--out-summary-csv", type=str, default=str(OUT_SUMMARY_CSV))
    args = ap.parse_args()

    pairs = pd.read_csv(OD_PAIRS_CSV)
    speed_lookup = pd.read_csv(args.speed_csv).set_index(["hour_of_day", "day_type"])["predicted_speed_kmh"]
    sites = pd.read_csv(args.sites_csv)
    n_pairs = len(pairs)
    n_sites = len(sites)
    print(f"OD (grid_o, grid_d, bucket) rows: {n_pairs}  vertiport sites: {n_sites}  speed buckets: {len(speed_lookup)}")

    site_lat = sites["avg_lat"].to_numpy()
    site_lon = sites["avg_lon"].to_numpy()
    grid_ids = sites["Grid ID"].to_numpy()

    flight_km = haversine_matrix(site_lat, site_lon, site_lat, site_lon)  # (30, 30)
    flight_time = flight_km / AIR_SPEED_KMH * 60.0  # minutes, speed-independent of ground bucket
    if EXCLUDE_SELF_LOOP:
        np.fill_diagonal(flight_time, np.inf)

    o_lat = pairs["o_lat"].to_numpy()
    o_lon = pairs["o_lon"].to_numpy()
    d_lat = pairs["d_lat"].to_numpy()
    d_lon = pairs["d_lon"].to_numpy()
    trip_count = pairs["trip_count"].to_numpy()

    best_i = np.empty(n_pairs, dtype=np.int32)
    best_j = np.empty(n_pairs, dtype=np.int32)
    best_cost = np.empty(n_pairs, dtype=np.float64)
    direct_ground_time = np.empty(n_pairs, dtype=np.float64)

    # process one (hour_of_day, day_type) bucket at a time -- each bucket has its
    # own dynamic ground speed, everything else (sites, flight_time) is shared
    bucket_keys = list(pairs.groupby(["hour_of_day", "day_type"]).groups.keys())
    print(f"processing {len(bucket_keys)} buckets")
    for bi, (hour, day_type) in enumerate(bucket_keys):
        mask = ((pairs["hour_of_day"] == hour) & (pairs["day_type"] == day_type)).to_numpy()
        idx = np.flatnonzero(mask)
        ground_speed = float(speed_lookup.loc[(hour, day_type)])

        direct_ground_time[idx] = haversine_km(o_lat[idx], o_lon[idx], d_lat[idx], d_lon[idx]) / ground_speed * 60.0

        for start in range(0, len(idx), CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, len(idx))
            chunk_idx = idx[start:end]
            access_km = haversine_matrix(o_lat[chunk_idx], o_lon[chunk_idx], site_lat, site_lon)
            egress_km = haversine_matrix(d_lat[chunk_idx], d_lon[chunk_idx], site_lat, site_lon)
            access_time = access_km / ground_speed * 60.0 + DWELL_MIN
            egress_time = egress_km / ground_speed * 60.0 + DWELL_MIN

            cost = access_time[:, :, None] + flight_time[None, :, :] + egress_time[:, None, :]
            chunk_n = len(chunk_idx)
            flat = cost.reshape(chunk_n, n_sites * n_sites)
            flat_argmin = np.argmin(flat, axis=1)
            best_i[chunk_idx] = flat_argmin // n_sites
            best_j[chunk_idx] = flat_argmin % n_sites
            best_cost[chunk_idx] = flat[np.arange(chunk_n), flat_argmin]

        print(f"bucket {bi + 1}/{len(bucket_keys)}: hour={hour} day_type={day_type} "
              f"ground_speed={ground_speed:.2f}km/h  rows={len(idx)}")

    result = pairs.copy()
    result["takeoff_grid_id"] = grid_ids[best_i]
    result["landing_grid_id"] = grid_ids[best_j]
    result["uam_time_min"] = best_cost
    result["direct_ground_time_min"] = direct_ground_time
    result["time_overhead_min"] = best_cost - direct_ground_time
    result["total_uam_time_min"] = best_cost * trip_count
    result["total_ground_time_min"] = direct_ground_time * trip_count
    result.to_csv(args.out_assignment_csv, index=False)
    print(f"saved -> {args.out_assignment_csv}")

    dep = pd.DataFrame({"grid_id": grid_ids, "site_idx": np.arange(n_sites)})
    dep_flow = result.groupby("takeoff_grid_id")["trip_count"].sum().rename("departures")
    arr_flow = result.groupby("landing_grid_id")["trip_count"].sum().rename("arrivals")
    summary = dep.set_index("grid_id").join(dep_flow).join(arr_flow).fillna(0)
    summary[["departures", "arrivals"]] = summary[["departures", "arrivals"]].astype(int)
    summary.to_csv(args.out_summary_csv)
    print(f"saved -> {args.out_summary_csv}")

    total_trips = trip_count.sum()
    total_uam_min = (best_cost * trip_count).sum()
    total_ground_min = (direct_ground_time * trip_count).sum()
    print("*" * 40)
    print(f"total trips assigned: {total_trips}")
    print(f"mean UAM door-to-door time: {total_uam_min / total_trips:.2f} min")
    print(f"mean direct ground time:    {total_ground_min / total_trips:.2f} min")
    print(f"mean overhead (UAM - ground): {(total_uam_min - total_ground_min) / total_trips:.2f} min")
    faster_frac = (result["time_overhead_min"] < 0).astype(int).mul(trip_count).sum() / total_trips
    print(f"fraction of trips where UAM is faster than direct ground: {faster_frac:.4f}")

    print("\nmean UAM time by bucket (sanity check -- expect peak-hour buckets to cost more):")
    result["total_uam_time_this_row"] = result["uam_time_min"] * result["trip_count"]
    by_bucket = result.groupby(["hour_of_day", "day_type"]).apply(
        lambda g: g["total_uam_time_this_row"].sum() / g["trip_count"].sum(), include_groups=False
    )
    print(by_bucket)


if __name__ == "__main__":
    main()
