# -*- coding: utf-8 -*-
"""
Route assignment: for every real trip O->D (aggregated to unique
(grid_o, grid_d) pairs with a trip count, from full_grid_od_pairs_202504.csv),
choose which of the 30 kmeans-selected vertiport sites to take off from (i)
and land at (j), so the trip actually goes O -> vertiport_i -> vertiport_j -> D.

Objective (per user decision): minimize total door-to-door travel TIME, not
distance -- ground and air legs use different speeds, which is the entire
point of a UAM network (a longer-looking detour through vertiports can still
be faster than direct ground travel because the air leg is much quicker per km).

    cost(O, D, i, j) = access_time(O, i) + flight_time(i, j) + egress_time(j, D)
    access_time(O, i) = haversine_km(O, i) / GROUND_SPEED_KMH * 60 + DWELL_MIN
    flight_time(i, j) = haversine_km(i, j) / AIR_SPEED_KMH * 60
    egress_time(j, D) = haversine_km(j, D) / GROUND_SPEED_KMH * 60 + DWELL_MIN

No vertiport/corridor capacity constraint yet (per user decision) -- every
trip independently picks its own best (i, j), so this is separable per
(grid_o, grid_d) pair and solved by exact vectorized argmin over all 30x30
combinations rather than one giant assignment LP (487K unique pairs x 900
route options would be ~440M decision variables in a combined formulation,
which is neither necessary here -- no shared constraints link the pairs --
nor remotely close to fitting gurobipy's free-license 2000-variable cap).
See validate_route_assignment_with_gurobi.py for a Gurobi formulation of the
same per-pair problem, used to confirm this numpy solution is exactly what
Gurobi would return, and ready to extend once capacity constraints are added
(at which point the pairs stop being independent and a real combined LP/MIP
becomes necessary).

Speed/dwell assumptions (edit these for your actual operator specs):
  GROUND_SPEED_KMH = 25   -- average congested urban driving speed
  AIR_SPEED_KMH     = 200 -- typical eVTOL cruise speed assumed in UAM studies
  DWELL_MIN         = 5   -- fixed boarding/security/taxi overhead per vertiport leg
"""
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUT_DIR = PROJECT_DIR / "outputs"

OD_PAIRS_CSV = DATA_DIR / "full_grid_od_pairs_202504.csv"
SITES_CSV = DATA_DIR / "selected_sites_kmeans_K30.csv"
OUT_ASSIGNMENT_CSV = OUT_DIR / "route_assignment_kmeans30_noselfloop.csv"
OUT_SUMMARY_CSV = OUT_DIR / "route_assignment_kmeans30_noselfloop_vertiport_summary.csv"

GROUND_SPEED_KMH = 15.0  # congested Shanghai peak-hour urban driving speed
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
    pairs = pd.read_csv(OD_PAIRS_CSV)
    sites = pd.read_csv(SITES_CSV)
    n_pairs = len(pairs)
    n_sites = len(sites)
    print(f"OD pairs: {n_pairs}  vertiport sites: {n_sites}")

    site_lat = sites["avg_lat"].to_numpy()
    site_lon = sites["avg_lon"].to_numpy()
    grid_ids = sites["Grid ID"].to_numpy()

    flight_km = haversine_matrix(site_lat, site_lon, site_lat, site_lon)  # (30, 30)
    flight_time = flight_km / AIR_SPEED_KMH * 60.0  # minutes
    if EXCLUDE_SELF_LOOP:
        np.fill_diagonal(flight_time, np.inf)

    o_lat = pairs["o_lat"].to_numpy()
    o_lon = pairs["o_lon"].to_numpy()
    d_lat = pairs["d_lat"].to_numpy()
    d_lon = pairs["d_lon"].to_numpy()
    trip_count = pairs["trip_count"].to_numpy()

    direct_ground_time = haversine_km(o_lat, o_lon, d_lat, d_lon) / GROUND_SPEED_KMH * 60.0

    best_i = np.empty(n_pairs, dtype=np.int32)
    best_j = np.empty(n_pairs, dtype=np.int32)
    best_cost = np.empty(n_pairs, dtype=np.float64)

    for start in range(0, n_pairs, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, n_pairs)
        access_km = haversine_matrix(o_lat[start:end], o_lon[start:end], site_lat, site_lon)  # (chunk, 30)
        egress_km = haversine_matrix(d_lat[start:end], d_lon[start:end], site_lat, site_lon)  # (chunk, 30)
        access_time = access_km / GROUND_SPEED_KMH * 60.0 + DWELL_MIN
        egress_time = egress_km / GROUND_SPEED_KMH * 60.0 + DWELL_MIN

        # cost[n, i, j] = access_time[n, i] + flight_time[i, j] + egress_time[n, j]
        cost = access_time[:, :, None] + flight_time[None, :, :] + egress_time[:, None, :]
        chunk_n = end - start
        flat = cost.reshape(chunk_n, n_sites * n_sites)
        flat_argmin = np.argmin(flat, axis=1)
        best_i[start:end] = flat_argmin // n_sites
        best_j[start:end] = flat_argmin % n_sites
        best_cost[start:end] = flat[np.arange(chunk_n), flat_argmin]

        print(f"assigned pairs [{start}:{end}] / {n_pairs}")

    result = pairs.copy()
    result["takeoff_grid_id"] = grid_ids[best_i]
    result["landing_grid_id"] = grid_ids[best_j]
    result["uam_time_min"] = best_cost
    result["direct_ground_time_min"] = direct_ground_time
    result["time_overhead_min"] = best_cost - direct_ground_time
    result["total_uam_time_min"] = best_cost * trip_count
    result["total_ground_time_min"] = direct_ground_time * trip_count
    result.to_csv(OUT_ASSIGNMENT_CSV, index=False)
    print(f"saved -> {OUT_ASSIGNMENT_CSV}")

    dep = pd.DataFrame({"grid_id": grid_ids, "site_idx": np.arange(n_sites)})
    dep_flow = result.groupby("takeoff_grid_id")["trip_count"].sum().rename("departures")
    arr_flow = result.groupby("landing_grid_id")["trip_count"].sum().rename("arrivals")
    summary = dep.set_index("grid_id").join(dep_flow).join(arr_flow).fillna(0)
    summary[["departures", "arrivals"]] = summary[["departures", "arrivals"]].astype(int)
    summary.to_csv(OUT_SUMMARY_CSV)
    print(f"saved -> {OUT_SUMMARY_CSV}")

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


if __name__ == "__main__":
    main()
