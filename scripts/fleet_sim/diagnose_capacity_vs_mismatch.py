# -*- coding: utf-8 -*-
"""
Diagnostic 2 (2026-07-21), job "5738592" full respec: for the fleet=12000
(v2_fleetabove) scenario that still only reaches R*=94.76% (misses the 95% gate),
determine whether the shortfall is genuine total-capacity scarcity or vehicles
sitting in the wrong place/state -- by capturing, at the moment a site rejects
demand, whether the SYSTEM (not that site) has vehicles that were actually
available.

Passive-only instrumentation (does not alter any dispatch/rebalancing decision --
every wrapper here calls straight through to the real implementation and only
copies out data that's already computed):

1. column_generation -- captures kwargs['orders'] (=total_orders, this bin's full
   demand incl. carried-forward backlog) and the returned gurobi_results, then
   replicates run_shanghai_fleet_simulation.py's OWN stage1_unmet formula
   (demand_agg - granted_agg per pair) so genuine LP-level vehicle-count
   exhaustion is captured per bin per origin site -- this piece isn't returned by
   any function and otherwise vanishes after run_iterations' loop moves on.
2. time_step_path_assignment -- passthrough capture of shortfall_by_site (already
   computed internally: away_flying/insufficient_battery/rounding_jitter/requested
   per origin site, per bin) plus assigned_routes (this bin's actually-served flow
   per pair) and launched_ids (revenue-dispatch vehicle ids, to classify later).
3. redistribute_vehicles -- passthrough capture of moves (reposition-dispatch
   vehicle ids, to classify inflight vehicles later as reposition vs revenue).
4. update_arrivals -- called once per bin with the full (vehicle_states,
   vertiport_states, vehicle_movements) triple as arguments; capturing them
   BEFORE calling through gives the fully-settled state at the end of the
   PREVIOUS bin (all of that bin's dispatch/rebalance/charging already applied),
   which is exactly "what the system looked like going into bin t" -- from this,
   per site: idle-and-fully-charged count, idle-and-charging count, inbound
   vehicles with ETA (vehicle_movements entries ending at this site), and
   outbound reposition vs revenue vehicles currently in flight (classified via
   the vid->kind map built from steps 2/3 above).

Produces one row per (bin, site) with everything needed to compute:
    P_mismatch = (# hotspot-site bins with real_unmet>0 AND other-site idle
                   vehicles available) / (# bins with any real_unmet>0)
    V_stranded = other-site idle vehicle count at the moment of rejection
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR / "scripts" / "fleet_sim"))

import run_shanghai_fleet_simulation as sim_mod  # noqa: E402
from distance_battery import set_distance_data  # noqa: E402
from initialization import initialize_states_with_time  # noqa: E402

_orig_column_generation = sim_mod.column_generation
_orig_tspa = sim_mod.time_step_path_assignment
_orig_redistribute = sim_mod.redistribute_vehicles
_orig_update_arrivals = sim_mod.update_arrivals

_CURRENT_T = -1
_LAST_STAGE1_BY_SITE = {}
_LAST_SHORTFALL_BY_SITE = {}
_LAST_ASSIGNED_BY_ORIGIN = {}
VID_KIND = {}  # vehicle id -> "revenue" | "reposition", refreshed on each (re)dispatch

BIN_SITE_LOG = []
FULLY_CHARGED_BATTERY_PCT = 99.999


def _wrapped_column_generation(*args, **kwargs):
    global _LAST_STAGE1_BY_SITE
    orders = kwargs.get("orders", args[0] if args else [])
    gurobi_results = _orig_column_generation(*args, **kwargs)

    demand_agg = {}
    for s, e, f in orders:
        demand_agg[(s, e)] = demand_agg.get((s, e), 0.0) + f
    granted_agg = {(r["takeoff"], r["landing"]): float(r["flow"]) for r in gurobi_results}

    stage1_by_site = {}
    for (s, e), d in demand_agg.items():
        shortfall = d - granted_agg.get((s, e), 0.0)
        if shortfall > 1e-9:
            stage1_by_site[s] = stage1_by_site.get(s, 0.0) + shortfall
    _LAST_STAGE1_BY_SITE = stage1_by_site
    return gurobi_results


def _wrapped_tspa(*args, **kwargs):
    global _LAST_SHORTFALL_BY_SITE, _LAST_ASSIGNED_BY_ORIGIN
    result = _orig_tspa(*args, **kwargs)
    unmet_after_assignment, assigned_routes, launched_ids, shortfall_reasons, operational_served, shortfall_by_site, unmet_for_rebalancing = result
    _LAST_SHORTFALL_BY_SITE = shortfall_by_site

    assigned_by_origin = {}
    for r in assigned_routes:
        assigned_by_origin[r["start"]] = assigned_by_origin.get(r["start"], 0.0) + r["flow"]
    _LAST_ASSIGNED_BY_ORIGIN = assigned_by_origin

    for vid in launched_ids:
        VID_KIND[vid] = "revenue"
    return result


def _wrapped_redistribute(*args, **kwargs):
    moves = _orig_redistribute(*args, **kwargs)
    for mv in moves:
        VID_KIND[mv["vehicle"]] = "reposition"
    return moves


def _wrapped_update_arrivals(*args, **kwargs):
    global _CURRENT_T
    vehicle_states = kwargs.get("vehicle_states", args[0])
    vertiport_states = kwargs.get("vertiport_states", args[1])
    vehicle_movements = kwargs.get("vehicle_movements", args[2])
    t = kwargs.get("current_step", args[3] if len(args) > 3 else _CURRENT_T)

    if _CURRENT_T >= 0:
        _snapshot_bin(_CURRENT_T, vehicle_states, vertiport_states, vehicle_movements)

    _CURRENT_T = t
    return _orig_update_arrivals(*args, **kwargs)


def _snapshot_bin(t, vehicle_states, vertiport_states, vehicle_movements):
    """Snapshot taken at the START of bin t (i.e. fully-settled END of bin t-1):
    idle/charging split, inbound-with-ETA, outbound-in-flight split by kind."""
    idle_full = {}
    idle_charging = {}
    for vid, st in vehicle_states.items():
        if st["in_service"] == 0:
            site = st["loc"]
            if st["battery"] >= FULLY_CHARGED_BATTERY_PCT:
                idle_full[site] = idle_full.get(site, 0) + 1
            else:
                idle_charging[site] = idle_charging.get(site, 0) + 1

    inbound_count = {}
    inbound_within2 = {}
    outbound_reposition = {}
    outbound_revenue = {}
    for vid, mv in vehicle_movements.items():
        end = mv["end"]
        start = mv["start"]
        inbound_count[end] = inbound_count.get(end, 0) + 1
        if mv["arrival_step"] - t <= 2:
            inbound_within2[end] = inbound_within2.get(end, 0) + 1
        kind = VID_KIND.get(vid, "revenue")
        if kind == "reposition":
            outbound_reposition[start] = outbound_reposition.get(start, 0) + 1
        else:
            outbound_revenue[start] = outbound_revenue.get(start, 0) + 1

    all_sites = list(vertiport_states.keys())
    total_idle_full_system = sum(idle_full.values())

    demand_this_bin = sim_mod.CURRENT_DEMAND_PER_BIN[t] if t < len(sim_mod.CURRENT_DEMAND_PER_BIN) else []
    new_demand_by_site = {}
    for row in demand_this_bin:
        new_demand_by_site[row["start"]] = new_demand_by_site.get(row["start"], 0.0) + float(row["flow"])

    for site in all_sites:
        away = _LAST_SHORTFALL_BY_SITE.get(site, {}).get("away_flying", 0.0)
        batt = _LAST_SHORTFALL_BY_SITE.get(site, {}).get("insufficient_battery", 0.0)
        jitter = _LAST_SHORTFALL_BY_SITE.get(site, {}).get("rounding_jitter", 0.0)
        requested = _LAST_SHORTFALL_BY_SITE.get(site, {}).get("requested", 0.0)
        stage1 = _LAST_STAGE1_BY_SITE.get(site, 0.0)
        real_unmet = stage1 + away + batt
        raw_unmet = real_unmet + jitter
        local_idle = idle_full.get(site, 0)
        BIN_SITE_LOG.append({
            "t": t, "site": site,
            "new_demand": new_demand_by_site.get(site, 0.0),
            "total_demand_incl_backlog": requested,
            "assigned_this_bin": _LAST_ASSIGNED_BY_ORIGIN.get(site, 0.0),
            "stage1_unmet": stage1,
            "away_flying": away,
            "insufficient_battery": batt,
            "rounding_jitter": jitter,
            "real_unmet": real_unmet,
            "raw_unmet": raw_unmet,
            "local_idle_full": local_idle,
            "local_idle_charging": idle_charging.get(site, 0),
            "other_idle_full": total_idle_full_system - local_idle,
            "inbound_count": inbound_count.get(site, 0),
            "inbound_within_2bins": inbound_within2.get(site, 0),
            "outbound_reposition_inflight": outbound_reposition.get(site, 0),
            "outbound_revenue_inflight": outbound_revenue.get(site, 0),
        })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites-csv", default=str(PROJECT_DIR / "data/selected_sites_kmeans_K30.csv"))
    ap.add_argument("--bucket-csv", default=str(PROJECT_DIR / "outputs/fleet_sim/fleet_sim_dynspeed_full_demand_bucket_eqsearch_v2_fleetabove_iter5.csv"))
    ap.add_argument("--vehicles-per-vertiport", type=int, default=400)
    ap.add_argument("--tag", default="diag_capmismatch_v2fleetabove_iter5")
    ap.add_argument("--n-bins", type=int, default=500)
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()

    sim_mod.column_generation = _wrapped_column_generation
    sim_mod.time_step_path_assignment = _wrapped_tspa
    sim_mod.redistribute_vehicles = _wrapped_redistribute
    sim_mod.update_arrivals = _wrapped_update_arrivals

    sites_csv = args.sites_csv
    bucket_csv = args.bucket_csv
    tag = args.tag
    n_bins = args.n_bins

    distance_csv = sim_mod.SIM_OUT_DIR / f"vertiport_distance_km.{tag}.tmp.csv"
    grid_ids = sim_mod.build_vertiport_distance_csv(sites_csv, distance_csv)
    set_distance_data(distance_csv)

    demand_per_bin, access_egress_per_bin = sim_mod.load_demand_and_access_per_bin(
        n_bins, grid_ids, bucket_csv=bucket_csv)
    sim_mod.CURRENT_DEMAND_PER_BIN = demand_per_bin  # exposed for _snapshot_bin's lookup
    ground_speed_per_bin = sim_mod.build_ground_speed_per_bin(n_bins)

    dist_df = pd.read_csv(distance_csv, index_col=0)
    dist_df.index = dist_df.index.astype(str)
    dist_df.columns = dist_df.columns.astype(str)
    distance_air = {(i, j): dist_df.loc[i, j] for i in grid_ids for j in grid_ids if i != j}

    vehicles_per_vertiport = args.vehicles_per_vertiport
    vehicles = [f"V{i}" for i in range(1, vehicles_per_vertiport * len(grid_ids) + 1)]
    vehicle_states, vertiport_states = initialize_states_with_time(vehicles, grid_ids, vehicles_per_vertiport)

    print(f"[CAP-VS-MISMATCH] vertiports={len(grid_ids)} bins={n_bins} "
          f"fleet={vehicles_per_vertiport * len(grid_ids)}", flush=True)

    sim_mod.run_iterations(
        num_bins=n_bins,
        vehicle_states=vehicle_states,
        vertiport_states=vertiport_states,
        demand_per_bin=demand_per_bin,
        charging_rate=25.0,
        discharge_rate=1.0,
        vertiports=grid_ids,
        distance_air=distance_air,
        charging_rate_per_bin=25.0,
        ground_speed_per_bin=ground_speed_per_bin,
        access_egress_per_bin=access_egress_per_bin,
        log_charging=False,
        rebalance_interval=1,
        rebalance_min_reserve=1,
        rebalance_max_idle_cap=vehicles_per_vertiport * 2.0,
        predictive_rebalancing=True,
    )
    # flush the final bin's snapshot (loop only snapshots at the START of the
    # NEXT bin's update_arrivals call, so bin n_bins-1 never gets a follow-up
    # call within this run -- harmless, we lose exactly one bin of snapshot data)

    distance_csv.unlink(missing_ok=True)

    out_csv = Path(args.out_csv) if args.out_csv else PROJECT_DIR / f"outputs/fleet_sim/_sweep_tmp/capmismatch_bin_site_log_{tag}.csv"
    pd.DataFrame(BIN_SITE_LOG).to_csv(out_csv, index=False)
    print(f"[CAP-VS-MISMATCH] saved {len(BIN_SITE_LOG)} rows -> {out_csv}", flush=True)


if __name__ == "__main__":
    main()
