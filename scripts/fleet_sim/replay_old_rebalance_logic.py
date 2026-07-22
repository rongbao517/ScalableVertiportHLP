# -*- coding: utf-8 -*-
"""
Diagnostic (2026-07-21), job "5738591" respec: full counterfactual replay of the
PRE-FIX rebalancing signal, to answer whether the 2026-07-20 unmet_for_rebalancing
fix is over-broad -- not just "how much does the aggregated signal differ" (that
was the old, insufficient diagnose_rebalance_fix_scope.py capture) but "what does
the fleet actually do differently, bin by bin, if fed the old signal" -- including
real downstream consequences (a cancelled move today can only be judged by whether
the target site's demand a few bins later actually got worse), which requires the
old-logic dynamics to genuinely diverge and be replayed forward, not just diffed
at a single instant.

Method: monkeypatch two call sites inside run_shanghai_fleet_simulation.run_iterations,
both referenced as bare names bound at import time (looked up as globals at call
time, so patching the module attribute before run_iterations executes is enough --
same mechanism already used in diagnose_rebalance_fix_scope.py):

1. time_step_path_assignment -- passthrough capture of unmet_after_assignment
   (index 0 of the return tuple; = away_flying+insufficient_battery+rounding_jitter
   per (s,e) pair, i.e. the RAW pre-fix per-pair shortfall) and unmet_for_rebalancing
   (index 6; = away_flying+insufficient_battery only, i.e. what production's FIXED
   code actually passes to the rebalancer) each bin.

2. redistribute_vehicles -- reconstructs the pre-fix `unmet_demand` argument and
   SUBSTITUTES it in place of the fixed code's unmet_demand_for_rebalancing before
   calling through to the real (unmodified) implementation, so all downstream
   dispatch/rebalancing/vehicle-state evolution genuinely happens under the old
   signal from this bin forward.

Reconstruction: production builds its argument as
    unmet_demand_for_rebalancing = stage1_unmet + unmet_for_rebalancing   (list concat)
so slicing off the last len(unmet_for_rebalancing) elements of the argument
recovers stage1_unmet exactly (concatenation is order-preserving, no shuffling
in between) -- then old_signal = stage1_unmet + unmet_after_assignment, matching
exactly what the PRE-FIX code path would have passed.

Saves the SAME file set + naming convention run_shanghai_fleet_simulation.py's own
main() uses (just with a different --tag), so this replay's outputs are directly,
apples-to-apples diffable against the already-saved REAL production
outputs/fleet_sim/*_eqsearch_v2_mc_main_iter5.csv files (which reflect the FIXED
code, i.e. exactly what "new logic" actually did) -- no need to duplicate the new
run, it already exists on disk. Also emits a per-bin-per-site raw/real-gap log
(rebalance_scope_diag_rawreal_gap_<tag>.csv) built from the same tspa capture,
directly usable for the size-bucketed diagnostic-1 analysis (grouping cancelled
events by gap magnitude).
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

_orig_tspa = sim_mod.time_step_path_assignment
_orig_redistribute = sim_mod.redistribute_vehicles

_CURRENT_T = -1
_LAST_UNMET_AFTER_ASSIGNMENT = []   # raw, pre-fix per-pair shortfall this bin (index 0 of tspa's return)
_LAST_UNMET_FOR_REBALANCING = []    # fixed-code per-pair signal this bin (index 6 of tspa's return)

RAWREAL_GAP_LOG = []       # per-bin, per-site: raw_gap, real_gap (rebalancer-input side)
MOVE_COMPARISON_LOG = []   # per-bin, per-site (target): old-logic actual moved (this replay) vehicles


def _wrapped_tspa(*args, **kwargs):
    global _LAST_UNMET_AFTER_ASSIGNMENT, _LAST_UNMET_FOR_REBALANCING
    result = _orig_tspa(*args, **kwargs)
    _LAST_UNMET_AFTER_ASSIGNMENT = result[0]
    _LAST_UNMET_FOR_REBALANCING = result[6]

    raw_by_site = {}
    for s, e, flow in _LAST_UNMET_AFTER_ASSIGNMENT:
        raw_by_site[s] = raw_by_site.get(s, 0.0) + flow
    real_by_site = {}
    for s, e, flow in _LAST_UNMET_FOR_REBALANCING:
        real_by_site[s] = real_by_site.get(s, 0.0) + flow
    all_sites = set(raw_by_site) | set(real_by_site)
    for s in all_sites:
        RAWREAL_GAP_LOG.append({
            "t": _CURRENT_T, "site": s,
            "raw_gap": raw_by_site.get(s, 0.0),
            "real_gap": real_by_site.get(s, 0.0),
        })
    return result


def _wrapped_update_arrivals(*args, **kwargs):
    global _CURRENT_T
    _CURRENT_T = kwargs.get("current_step", args[3] if len(args) > 3 else _CURRENT_T)
    return sim_mod.update_arrivals.__wrapped_orig__(*args, **kwargs)


def _wrapped_redistribute(*args, **kwargs):
    passed_list = list(kwargs.get("unmet_demand", []))
    n_new = len(_LAST_UNMET_FOR_REBALANCING)
    stage1_list = passed_list[: len(passed_list) - n_new] if n_new > 0 else passed_list
    old_list = stage1_list + list(_LAST_UNMET_AFTER_ASSIGNMENT)
    kwargs["unmet_demand"] = old_list

    moves = _orig_redistribute(*args, **kwargs)

    shortage_moved = {}
    for mv in moves:
        if mv["kind"] == "shortage":
            shortage_moved[mv["end"]] = shortage_moved.get(mv["end"], 0) + 1
    for site, n in shortage_moved.items():
        MOVE_COMPARISON_LOG.append({"t": _CURRENT_T, "site": site, "old_logic_moved": n})

    return moves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites-csv", default=str(PROJECT_DIR / "data/selected_sites_kmeans_K30_modechoice_iter1.csv"))
    ap.add_argument("--bucket-csv", default=str(PROJECT_DIR / "outputs/fleet_sim/fleet_sim_dynspeed_full_demand_bucket_eqsearch_v2_mc_main_iter5.csv"))
    ap.add_argument("--vehicles-per-vertiport", type=int, default=250)
    ap.add_argument("--tag", default="diag_oldlogic_mc_main_iter5")
    ap.add_argument("--n-bins", type=int, default=500)
    args = ap.parse_args()

    # patch AFTER argparse so a bad invocation fails fast without the patch cost
    orig_update_arrivals = sim_mod.update_arrivals
    orig_update_arrivals_wrapper = _wrapped_update_arrivals
    orig_update_arrivals_wrapper.__wrapped_orig__ = orig_update_arrivals
    sim_mod.update_arrivals = orig_update_arrivals_wrapper
    sim_mod.time_step_path_assignment = _wrapped_tspa
    sim_mod.redistribute_vehicles = _wrapped_redistribute

    sites_csv = args.sites_csv
    bucket_csv = args.bucket_csv
    tag = args.tag
    n_bins = args.n_bins

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

    vehicles_per_vertiport = args.vehicles_per_vertiport
    vehicles = [f"V{i}" for i in range(1, vehicles_per_vertiport * len(grid_ids) + 1)]
    vehicle_states, vertiport_states = initialize_states_with_time(vehicles, grid_ids, vehicles_per_vertiport)

    print(f"[OLD-LOGIC REPLAY] vertiports={len(grid_ids)} bins={n_bins} "
          f"fleet={vehicles_per_vertiport * len(grid_ids)}", flush=True)

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
    )

    distance_csv.unlink(missing_ok=True)

    out = sim_mod.SIM_OUT_DIR
    pd.DataFrame(summary).to_csv(out / f"time_step_summary_{tag}.csv", index=False)
    pd.DataFrame(assigned_routes_log).to_csv(out / f"assigned_routes_{tag}.csv", index=False)
    pd.DataFrame(rebalance_log).to_csv(out / f"rebalance_log_{tag}.csv", index=False)
    pd.DataFrame(vertiport_occupancy_log).to_csv(out / f"vertiport_occupancy_{tag}.csv", index=False)
    pd.DataFrame(vertiport_total_count_log).to_csv(out / f"vertiport_total_count_{tag}.csv", index=False)
    pd.DataFrame(battery_summary_log).to_csv(out / f"battery_summary_{tag}.csv", index=False)

    site_stats_df = pd.DataFrame(site_assignment_stats)
    site_stats_df["total_met"] = site_stats_df["total_orders"] - site_stats_df["total_unmet"]
    site_stats_df["assignment_ratio"] = site_stats_df["total_met"] / site_stats_df["total_orders"].replace(0, pd.NA)
    site_stats_df.to_csv(out / f"vertiport_assignment_ratio_{tag}.csv", index=False)

    pd.DataFrame(RAWREAL_GAP_LOG).to_csv(out / f"_sweep_tmp/rawreal_gap_{tag}.csv", index=False)
    pd.DataFrame(MOVE_COMPARISON_LOG).to_csv(out / f"_sweep_tmp/old_logic_moves_{tag}.csv", index=False)

    print(f"[OLD-LOGIC REPLAY] done. wrote time_step_summary_{tag}.csv etc, "
          f"rawreal_gap rows={len(RAWREAL_GAP_LOG)}, old_logic_moves rows={len(MOVE_COMPARISON_LOG)}",
          flush=True)


if __name__ == "__main__":
    main()
