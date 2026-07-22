# -*- coding: utf-8 -*-
"""
Robustness battery (2026-07-22): does the positioning LP's advantage over
Tier 0 generalize, or is it specific to the one fleet=12000/fleetabove
500-bin trajectory used throughout this investigation?

Note on "demand random seeds": there isn't one to vary -- this pipeline's
demand is a deterministic (hour_of_day, day_type)-bucket average
(fleet_sim_dynspeed_full_demand_bucket_*.csv), not a stochastic draw, and
every other module downstream (Gurobi/column_generation, task_assignment,
rebalancing) is itself fully deterministic given its inputs. There is no
randomness anywhere in this simulation to seed. The closest real equivalents,
already produced earlier in this project's demand-equilibrium sensitivity
work, are: different SITE LAYOUTS (different real routed-demand geometry)
and different KAPPA_W values (different mode-choice wait-time elasticity,
which converges to a genuinely different demand distribution through the
equilibrium feedback loop) -- both used below in place of a "seed".

Scenario battery:
    ref            fleet=12000, K30 oracle sites, fleetabove demand (the
                   scenario every earlier result in this thread used)
    fleet7500_mc   fleet=7500,  mode-choice sites, mc_main demand
                   (different fleet size AND layout AND demand simultaneously)
    fleet4500      fleet=4500,  K30 oracle sites, fleetbelow_damped demand
                   (own R* was already low under the old logic -- tests
                   whether LP still helps when the system may be genuinely
                   capacity-constrained, not just spatially mismatched)
    kappa20        fleet=7500,  K30 oracle sites, kappa20 demand (same
                   layout/fleet as a "kappa10-family" run, different demand
                   realization via a different wait-time elasticity)
    threshold      fleet=7500,  threshold-selected sites, threshold_main demand
                   (different site-selection method entirely)
    ref_skewed_init  same as ref, but vehicles start concentrated at LOW-
                     demand sites instead of split evenly -- tests whether
                     the LP's advantage depends on already starting from a
                     reasonable, roughly-balanced initial distribution.

For each scenario: run Tier0-baseline (threshold=1.5/gain=1.0/top_k=3/
max_drain=3, the untouched default) and the positioning LP at H=8 (the
middle of the three horizons tested on the reference scenario, a reasonable
fixed choice for a robustness check rather than re-sweeping H everywhere).
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
_orig_update_arrivals = sim_mod.update_arrivals

_LAST_STAGE1_BY_SITE = {}
_LAST_SHORTFALL_BY_SITE = {}
_LAST_ASSIGNED_BY_ORIGIN = {}
_CURRENT_T = -1
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
                "real_unmet": stage1 + away + batt,
                "raw_unmet": stage1 + away + batt + jitter,
            })
    _CURRENT_T = t
    return _orig_update_arrivals(*args, **kwargs)


def custom_initialize_skewed(grid_ids, bucket_csv, vehicles_per_vertiport):
    """Total fleet held constant at vehicles_per_vertiport*len(grid_ids), but
    concentrated at LOW-demand sites and starved at HIGH-demand sites --
    the opposite of what the system needs, to stress-test whether the LP's
    advantage depends on starting from an already-reasonable distribution."""
    df = pd.read_csv(bucket_csv)
    df["takeoff_grid_id"] = df["takeoff_grid_id"].astype(str)
    demand_by_site = df.groupby("takeoff_grid_id")["total_trip_count"].sum()
    ranked = sorted(grid_ids, key=lambda s: demand_by_site.get(s, 0.0))  # ascending: lowest demand first
    n = len(ranked)
    weights = [n - i for i in range(n)]  # highest weight to lowest-demand site
    total_fleet = vehicles_per_vertiport * n
    raw_counts = [w / sum(weights) * total_fleet for w in weights]
    counts = [int(round(c)) for c in raw_counts]
    counts[0] += total_fleet - sum(counts)  # fix rounding remainder on the single most-skewed site

    vehicles = [f"V{i}" for i in range(1, total_fleet + 1)]
    vehicle_states, vertiport_states = {}, {v: {"avail": 0, "in_service": 0} for v in grid_ids}
    idx = 0
    for site, n_here in zip(ranked, counts):
        assigned = vehicles[idx: idx + n_here]
        idx += n_here
        for vid in assigned:
            vehicle_states[vid] = {"loc": site, "battery": 100.0, "in_service": 0, "charging": 0}
        vertiport_states[site]["avail"] = n_here
    return vehicle_states, vertiport_states


def run_scenario(tag, sites_csv, bucket_csv, vehicles_per_vertiport, n_bins, mechanism, horizon=8, init_mode="uniform"):
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
    distance_csv.unlink(missing_ok=True)

    if init_mode == "skewed":
        vehicle_states, vertiport_states = custom_initialize_skewed(grid_ids, bucket_csv, vehicles_per_vertiport)
    else:
        vehicles = [f"V{i}" for i in range(1, vehicles_per_vertiport * len(grid_ids) + 1)]
        vehicle_states, vertiport_states = initialize_states_with_time(vehicles, grid_ids, vehicles_per_vertiport)

    common_kwargs = dict(
        num_bins=n_bins, vehicle_states=vehicle_states, vertiport_states=vertiport_states,
        demand_per_bin=demand_per_bin, charging_rate=25.0, discharge_rate=1.0,
        vertiports=grid_ids, distance_air=distance_air, charging_rate_per_bin=25.0,
        ground_speed_per_bin=ground_speed_per_bin, access_egress_per_bin=access_egress_per_bin,
        log_charging=False, rebalance_interval=1, rebalance_min_reserve=1,
        rebalance_max_idle_cap=vehicles_per_vertiport * 2.0,
    )

    print(f"[{tag}] mechanism={mechanism} init_mode={init_mode} fleet={vehicles_per_vertiport*len(grid_ids)}", flush=True)
    t0 = time.time()
    if mechanism == "tier0":
        result = sim_mod.run_iterations(
            **common_kwargs, predictive_rebalancing=True,
            predictive_threshold=1.5, predictive_gain=1.0, predictive_top_k=3, predictive_max_drain=3,
        )
    else:
        result = sim_mod.run_iterations(
            **common_kwargs, predictive_rebalancing=False,
            positioning_lp_enabled=True, positioning_lp_horizon=horizon, positioning_lp_lambda=0.001,
            positioning_lp_diag_log=[],
        )
    elapsed = time.time() - t0
    (summary, cumulative_cost, assigned_routes_log, charging_log, rebalance_log,
     vertiport_occupancy_log, vertiport_total_count_log, site_assignment_stats,
     battery_summary_log, shortfall_by_site_accum) = result

    out = sim_mod.SIM_OUT_DIR
    pd.DataFrame(summary).to_csv(out / f"time_step_summary_{tag}.csv", index=False)
    pd.DataFrame(rebalance_log).to_csv(out / f"rebalance_log_{tag}.csv", index=False)
    pd.DataFrame(PER_SITE_BIN_LOG).to_csv(out / f"_sweep_tmp/per_site_bin_{tag}.csv", index=False)

    met = sum(r["met_demand"] for r in summary)
    real_unmet = sum(r["unmet_fleet_insufficient"] + r["unmet_away_flying"] + r["unmet_insufficient_battery"] for r in summary)
    r_star = met / (met + real_unmet) if (met + real_unmet) > 0 else float("nan")
    print(f"[{tag}] DONE elapsed={elapsed:.1f}s R_star={r_star:.4f} met={met:.0f} real_unmet={real_unmet:.1f}", flush=True)
    return r_star


SCENARIOS = {
    "ref": dict(sites_csv=str(PROJECT_DIR / "data/selected_sites_kmeans_K30.csv"),
                bucket_csv=str(PROJECT_DIR / "outputs/fleet_sim/fleet_sim_dynspeed_full_demand_bucket_eqsearch_v2_fleetabove_iter5.csv"),
                vehicles_per_vertiport=400),
    "fleet7500_mc": dict(sites_csv=str(PROJECT_DIR / "data/selected_sites_kmeans_K30_modechoice_iter1.csv"),
                          bucket_csv=str(PROJECT_DIR / "outputs/fleet_sim/fleet_sim_dynspeed_full_demand_bucket_eqsearch_v2_mc_main_iter5.csv"),
                          vehicles_per_vertiport=250),
    "fleet4500": dict(sites_csv=str(PROJECT_DIR / "data/selected_sites_kmeans_K30.csv"),
                       bucket_csv=str(PROJECT_DIR / "outputs/fleet_sim/fleet_sim_dynspeed_full_demand_bucket_eqsearch_v2_fleetbelow_damped_iter11.csv"),
                       vehicles_per_vertiport=150),
    "kappa20": dict(sites_csv=str(PROJECT_DIR / "data/selected_sites_kmeans_K30.csv"),
                     bucket_csv=str(PROJECT_DIR / "outputs/fleet_sim/fleet_sim_dynspeed_full_demand_bucket_eqsearch_v2_kappa20_iter6.csv"),
                     vehicles_per_vertiport=250),
    "threshold": dict(sites_csv=str(PROJECT_DIR / "data/selected_sites_kmeans_K30_modechoice_threshold.csv"),
                       bucket_csv=str(PROJECT_DIR / "outputs/fleet_sim/fleet_sim_dynspeed_full_demand_bucket_eqsearch_v2_threshold_main_iter5.csv"),
                       vehicles_per_vertiport=250),
    "oracle7500": dict(sites_csv=str(PROJECT_DIR / "data/selected_sites_kmeans_K30.csv"),
                        bucket_csv=str(PROJECT_DIR / "outputs/fleet_sim/fleet_sim_dynspeed_full_demand_bucket_eqsearch_v2_oracle_main_iter5.csv"),
                        vehicles_per_vertiport=250),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True, choices=list(SCENARIOS.keys()))
    ap.add_argument("--mechanism", required=True, choices=["tier0", "lp"])
    ap.add_argument("--n-bins", type=int, default=500)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--init-mode", default="uniform", choices=["uniform", "skewed"])
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    cfg = SCENARIOS[args.scenario]
    tag = args.tag or f"robust_{args.scenario}_{args.mechanism}_{args.init_mode}"
    run_scenario(tag, cfg["sites_csv"], cfg["bucket_csv"], cfg["vehicles_per_vertiport"],
                 args.n_bins, args.mechanism, args.horizon, args.init_mode)


if __name__ == "__main__":
    main()
