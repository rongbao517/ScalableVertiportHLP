# -*- coding: utf-8 -*-
"""
First-round mechanism decomposition for Tier-0 predictive rebalancing
(2026-07-21): isolate the individual and interacting effects of
predictive_top_k / predictive_max_drain / predictive_gain / predictive_threshold
before touching Tier-1/Tier-2 or running the full demand-equilibrium loop.

Per the user's explicit design: uses a FIXED demand (the already-converged
fleet=12000/v2_fleetabove bucket_csv) and does a single fleet-sim pass per
config, NOT the full mode-choice equilibrium search -- this scenario was chosen
specifically because diagnostic 2 already showed it has abundant idle capacity
(71.8% fleet-time idle) and a real spatial/temporal mismatch, so any change here
isolates a rebalancing-mechanism effect instead of being confounded with total-
capacity effects or demand-choice feedback.

Configs (baseline + 4 single-factor perturbations; baseline is rerun WITH
instrumentation active since production's existing fleetabove run has no
predictive_diag_log capture):
    baseline: threshold=1.5, gain=1.0, top_k=3, max_drain=3  (current default)
    A:        top_k=8           (others baseline) -- widen hotspot coverage only
    B:        max_drain=8       (others baseline) -- strengthen per-hotspot supply only
    D:        gain=2.0          (others baseline) -- raise demand-responsiveness only
    F:        threshold=0.75    (others baseline) -- lower trigger sensitivity only

Captures, per config:
  - predictive_diag_log (rebalancing.py's new opt-in hook): per (bin, drain-site)
    net_inflow, raw_requested, available, n_drain_target, which of max_drain/
    available was binding -- answers "was gain actually shadowed by the cap,
    or did it change n_drain in practice" empirically instead of by assumption.
  - rebalance_log (production-standard): actual moves with distance/kind, i.e.
    real dispatch/cost, and (via groupby) donor/target contention counts within
    a bin -- no separate instrumentation needed, it's already a complete list of
    every move.
  - per-bin-per-site real_unmet (from time_step_path_assignment + stage1
    reconstruction, same technique as diagnose_capacity_vs_mismatch.py) -- lets
    the analysis correlate a drain site's OWN subsequent real shortage (did
    draining hurt it) against a target site's real_unmet reduction (did it help).
  - time_step_summary (production-standard): overall met/unmet trajectory for
    this fixed-demand pass, for a top-line before/after service comparison.
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

_LAST_STAGE1_BY_SITE = {}
_LAST_SHORTFALL_BY_SITE = {}
_LAST_ASSIGNED_BY_ORIGIN = {}
PREDICTIVE_DIAG_LOG = []
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


DONOR_FILTER_ACTIVE = True  # toggled per-run by run_one_config's donor_filter arg


def _wrapped_redistribute(*args, **kwargs):
    kwargs["predictive_diag_log"] = PREDICTIVE_DIAG_LOG
    # round-2 (C/E/G) comparison: run identical configs with the 2026-07-22
    # donor self-need filter on (its new default, 0.0) vs off (float("inf"),
    # i.e. the pre-filter behavior every earlier round-1 config actually ran
    # under) -- isolates the filter's own effect from the top_k/max_drain/gain
    # combo's effect instead of only ever seeing them together.
    kwargs["predictive_donor_max_own_need"] = 0.0 if DONOR_FILTER_ACTIVE else float("inf")
    moves = _orig_redistribute(*args, **kwargs)
    t = kwargs.get("current_step")
    all_sites = list(kwargs["vertiport_states"].keys()) if "vertiport_states" in kwargs else list(args[1].keys())
    for site in all_sites:
        away = _LAST_SHORTFALL_BY_SITE.get(site, {}).get("away_flying", 0.0)
        batt = _LAST_SHORTFALL_BY_SITE.get(site, {}).get("insufficient_battery", 0.0)
        jitter = _LAST_SHORTFALL_BY_SITE.get(site, {}).get("rounding_jitter", 0.0)
        stage1 = _LAST_STAGE1_BY_SITE.get(site, 0.0)
        PER_SITE_BIN_LOG.append({
            "t": t, "site": site,
            "assigned_this_bin": _LAST_ASSIGNED_BY_ORIGIN.get(site, 0.0),
            "real_unmet": stage1 + away + batt,
            "raw_unmet": stage1 + away + batt + jitter,
        })
    return moves


def run_one_config(tag, sites_csv, bucket_csv, vehicles_per_vertiport, n_bins,
                    predictive_threshold, predictive_gain, predictive_top_k, predictive_max_drain,
                    donor_filter=True):
    global PREDICTIVE_DIAG_LOG, PER_SITE_BIN_LOG, DONOR_FILTER_ACTIVE
    PREDICTIVE_DIAG_LOG = []
    PER_SITE_BIN_LOG = []
    DONOR_FILTER_ACTIVE = donor_filter

    sim_mod.column_generation = _wrapped_column_generation
    sim_mod.time_step_path_assignment = _wrapped_tspa
    sim_mod.redistribute_vehicles = _wrapped_redistribute

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

    print(f"[{tag}] threshold={predictive_threshold} gain={predictive_gain} "
          f"top_k={predictive_top_k} max_drain={predictive_max_drain}", flush=True)

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
        predictive_rebalancing=True,
        predictive_threshold=predictive_threshold,
        predictive_gain=predictive_gain,
        predictive_top_k=predictive_top_k,
        predictive_max_drain=predictive_max_drain,
    )

    distance_csv.unlink(missing_ok=True)

    out = sim_mod.SIM_OUT_DIR
    pd.DataFrame(summary).to_csv(out / f"time_step_summary_{tag}.csv", index=False)
    pd.DataFrame(rebalance_log).to_csv(out / f"rebalance_log_{tag}.csv", index=False)
    pd.DataFrame(PREDICTIVE_DIAG_LOG).to_csv(out / f"_sweep_tmp/predictive_diag_{tag}.csv", index=False)
    pd.DataFrame(PER_SITE_BIN_LOG).to_csv(out / f"_sweep_tmp/per_site_bin_{tag}.csv", index=False)

    met = sum(r["met_demand"] for r in summary)
    unmet = sum(r["unmet_demand"] for r in summary)
    r_rate = met / (met + unmet) if (met + unmet) > 0 else 0.0
    n_moves = {"predictive": 0, "shortage": 0, "overflow": 0}
    for mv in rebalance_log:
        n_moves[mv["kind"]] = n_moves.get(mv["kind"], 0) + 1
    print(f"[{tag}] DONE met={met:.0f} unmet={unmet:.0f} R={r_rate:.4f} "
          f"moves={n_moves} predictive_diag_rows={len(PREDICTIVE_DIAG_LOG)}", flush=True)


CONFIGS = {
    "baseline": dict(predictive_threshold=1.5, predictive_gain=1.0, predictive_top_k=3, predictive_max_drain=3),
    "A_topk8": dict(predictive_threshold=1.5, predictive_gain=1.0, predictive_top_k=8, predictive_max_drain=3),
    "B_maxdrain8": dict(predictive_threshold=1.5, predictive_gain=1.0, predictive_top_k=3, predictive_max_drain=8),
    "D_gain2": dict(predictive_threshold=1.5, predictive_gain=2.0, predictive_top_k=3, predictive_max_drain=3),
    "F_threshold075": dict(predictive_threshold=0.75, predictive_gain=1.0, predictive_top_k=3, predictive_max_drain=3),
}

# Round 2 (interaction verification, 2026-07-22): combos round 1 flagged as
# needing a joint test -- top_k+max_drain together (C), gain+max_drain together
# (E, the combo most likely to hit the donor-depletion risk round 1 first
# surfaced at site 783), and threshold+top_k+max_drain all loosened at once (G).
# Each runs twice: donor_filter=False (pre-fix behavior, matching how every
# round-1 config actually ran) and donor_filter=True (2026-07-22 self-need
# filter active) -- isolates the filter's own effect from the combo's effect.
CONFIGS_ROUND2 = {
    "C_topk8_maxdrain8": dict(predictive_threshold=1.5, predictive_gain=1.0, predictive_top_k=8, predictive_max_drain=8),
    "E_gain2_maxdrain8": dict(predictive_threshold=1.5, predictive_gain=2.0, predictive_top_k=3, predictive_max_drain=8),
    "G_thr075_topk8_maxdrain8": dict(predictive_threshold=0.75, predictive_gain=1.0, predictive_top_k=8, predictive_max_drain=8),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites-csv", default=str(PROJECT_DIR / "data/selected_sites_kmeans_K30.csv"))
    ap.add_argument("--bucket-csv", default=str(PROJECT_DIR / "outputs/fleet_sim/fleet_sim_dynspeed_full_demand_bucket_eqsearch_v2_fleetabove_iter5.csv"))
    ap.add_argument("--vehicles-per-vertiport", type=int, default=400)
    ap.add_argument("--n-bins", type=int, default=500)
    ap.add_argument("--tag-prefix", default="predsweep_r2")
    ap.add_argument("--round", choices=["1", "2"], default="2")
    args = ap.parse_args()

    if args.round == "1":
        for name, params in CONFIGS.items():
            tag = f"{args.tag_prefix}_{name}"
            run_one_config(tag, args.sites_csv, args.bucket_csv, args.vehicles_per_vertiport, args.n_bins, **params)
    else:
        for name, params in CONFIGS_ROUND2.items():
            for filt, suffix in [(False, "nofilter"), (True, "filtered")]:
                tag = f"{args.tag_prefix}_{name}_{suffix}"
                run_one_config(tag, args.sites_csv, args.bucket_csv, args.vehicles_per_vertiport, args.n_bins,
                                donor_filter=filt, **params)


if __name__ == "__main__":
    main()
