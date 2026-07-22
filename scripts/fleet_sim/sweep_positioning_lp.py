# -*- coding: utf-8 -*-
"""
Round-1 test of the rolling positioning LP (2026-07-22), replacing the
net_inflow-based Tier 0. Tests H=4/8/12 on the SAME fixed-demand fleet=12000
(v2_fleetabove) scenario and 500-bin window used throughout this
investigation, so results are directly comparable to the Tier0/1/2 baseline
(R=0.8163-0.8198 across the earlier parameter sweeps) and to the offline
"clairvoyant" upper bound (R=1.0000, positioning_lp_offline_bound.py).

Captures the same per-site-per-bin real_unmet log used in every earlier
diagnostic (diagnose_capacity_vs_mismatch.py, sweep_predictive_mechanism.py)
plus positioning-LP-specific diagnostics (planned vs. executed reposition
volume per bin, solve time) so 783/928/925's shortfall trend and the
plan/execution gap can both be checked directly.
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR / "scripts" / "fleet_sim"))

import run_shanghai_fleet_simulation as sim_mod  # noqa: E402
from distance_battery import set_distance_data  # noqa: E402
from initialization import initialize_states_with_time  # noqa: E402

_orig_column_generation = sim_mod.column_generation
_orig_tspa = sim_mod.time_step_path_assignment

_LAST_STAGE1_BY_SITE = {}
_LAST_SHORTFALL_BY_SITE = {}
_LAST_ASSIGNED_BY_ORIGIN = {}
PER_SITE_BIN_LOG = []


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
    _, assigned_routes, _, _, _, shortfall_by_site, _ = result
    _LAST_SHORTFALL_BY_SITE = shortfall_by_site
    assigned_by_origin = {}
    for r in assigned_routes:
        assigned_by_origin[r["start"]] = assigned_by_origin.get(r["start"], 0.0) + r["flow"]
    _LAST_ASSIGNED_BY_ORIGIN = assigned_by_origin
    return result


_CURRENT_T = -1
_orig_update_arrivals = sim_mod.update_arrivals


def _wrapped_update_arrivals(*args, **kwargs):
    global _CURRENT_T
    t = kwargs.get("current_step", args[3] if len(args) > 3 else _CURRENT_T)
    if _CURRENT_T >= 0:
        for site in kwargs.get("vertiport_states", args[1]).keys():
            away = _LAST_SHORTFALL_BY_SITE.get(site, {}).get("away_flying", 0.0)
            batt = _LAST_SHORTFALL_BY_SITE.get(site, {}).get("insufficient_battery", 0.0)
            jitter = _LAST_SHORTFALL_BY_SITE.get(site, {}).get("rounding_jitter", 0.0)
            stage1 = _LAST_STAGE1_BY_SITE.get(site, 0.0)
            PER_SITE_BIN_LOG.append({
                "t": _CURRENT_T, "site": site,
                "assigned_this_bin": _LAST_ASSIGNED_BY_ORIGIN.get(site, 0.0),
                "real_unmet": stage1 + away + batt,
                "raw_unmet": stage1 + away + batt + jitter,
            })
    _CURRENT_T = t
    return _orig_update_arrivals(*args, **kwargs)


def run_one_config(tag, sites_csv, bucket_csv, vehicles_per_vertiport, n_bins, horizon, lam=0.001):
    global PER_SITE_BIN_LOG, _CURRENT_T
    PER_SITE_BIN_LOG = []
    _CURRENT_T = -1

    sim_mod.column_generation = _wrapped_column_generation
    sim_mod.time_step_path_assignment = _wrapped_tspa
    sim_mod.update_arrivals = _wrapped_update_arrivals

    distance_csv = sim_mod.SIM_OUT_DIR / f"vertiport_distance_km.{tag}.tmp.csv"
    grid_ids = sim_mod.build_vertiport_distance_csv(sites_csv, distance_csv)
    set_distance_data(distance_csv)

    demand_per_bin, access_egress_per_bin = sim_mod.load_demand_and_access_per_bin(
        n_bins, grid_ids, bucket_csv=bucket_csv)
    ground_speed_per_bin = sim_mod.build_ground_speed_per_bin(n_bins)

    dist_df = pd.read_csv(distance_csv, index_col=0)
    dist_df.index = dist_df.index.astype(str)
    dist_df.columns = dist_df.columns.astype(str)
    distance_air = {(i, j): dist_df.loc[i, j] for i in grid_ids for j in grid_ids if i != j}

    vehicles = [f"V{i}" for i in range(1, vehicles_per_vertiport * len(grid_ids) + 1)]
    vehicle_states, vertiport_states = initialize_states_with_time(vehicles, grid_ids, vehicles_per_vertiport)

    positioning_diag = []
    print(f"[{tag}] horizon={horizon} lambda={lam}", flush=True)
    t0 = time.time()

    (summary, cumulative_cost, assigned_routes_log, charging_log, rebalance_log,
     vertiport_occupancy_log, vertiport_total_count_log, site_assignment_stats,
     battery_summary_log, shortfall_by_site_accum) = sim_mod.run_iterations(
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
        predictive_rebalancing=False,  # old Tier 0 disabled -- LP replaces it
        positioning_lp_enabled=True,
        positioning_lp_horizon=horizon,
        positioning_lp_lambda=lam,
        positioning_lp_diag_log=positioning_diag,
    )
    elapsed = time.time() - t0

    distance_csv.unlink(missing_ok=True)

    out = sim_mod.SIM_OUT_DIR
    pd.DataFrame(summary).to_csv(out / f"time_step_summary_{tag}.csv", index=False)
    pd.DataFrame(rebalance_log).to_csv(out / f"rebalance_log_{tag}.csv", index=False)
    pd.DataFrame(PER_SITE_BIN_LOG).to_csv(out / f"_sweep_tmp/per_site_bin_{tag}.csv", index=False)
    pd.DataFrame(positioning_diag).to_csv(out / f"_sweep_tmp/positioning_diag_{tag}.csv", index=False)

    met = sum(r["met_demand"] for r in summary)
    unmet = sum(r["unmet_demand"] for r in summary)
    r_rate = met / (met + unmet) if (met + unmet) > 0 else 0.0
    n_moves = {}
    for mv in rebalance_log:
        n_moves[mv["kind"]] = n_moves.get(mv["kind"], 0) + 1
    planned_raw = sum(d["planned_total_raw"] for d in positioning_diag)
    planned = sum(d["planned_total_rounded"] for d in positioning_diag)
    executed = sum(d["executed_total"] for d in positioning_diag)
    mean_solve_s = pd.DataFrame(positioning_diag)["t"].count() and (elapsed / max(1, n_bins))
    print(f"[{tag}] DONE elapsed={elapsed:.1f}s met={met:.0f} unmet={unmet:.0f} R={r_rate:.4f} "
          f"moves={n_moves} lp_planned={planned:.0f} lp_executed={executed} "
          f"exec_rate={executed/planned if planned>0 else float('nan'):.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites-csv", default=str(PROJECT_DIR / "data/selected_sites_kmeans_K30.csv"))
    ap.add_argument("--bucket-csv", default=str(PROJECT_DIR / "outputs/fleet_sim/fleet_sim_dynspeed_full_demand_bucket_eqsearch_v2_fleetabove_iter5.csv"))
    ap.add_argument("--vehicles-per-vertiport", type=int, default=400)
    ap.add_argument("--n-bins", type=int, default=500)
    ap.add_argument("--tag-prefix", default="posLP_r1")
    ap.add_argument("--horizons", default="4,8,12")
    args = ap.parse_args()

    for h in [int(x) for x in args.horizons.split(",")]:
        tag = f"{args.tag_prefix}_H{h}"
        run_one_config(tag, args.sites_csv, args.bucket_csv, args.vehicles_per_vertiport, args.n_bins, h)


if __name__ == "__main__":
    main()
