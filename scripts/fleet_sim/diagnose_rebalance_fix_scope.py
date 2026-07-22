# -*- coding: utf-8 -*-
"""
Diagnostic (2026-07-21): is the unmet_for_rebalancing fix (task_assignment.py,
2026-07-20) over-broad -- does it suppress real away_flying/insufficient_battery
shortage alongside rounding_jitter, or is grid 433's pattern actually system-wide?

By construction, per (origin, destination, bin) pair:
    old_signal (unmet_after_assignment) = away_flying + insufficient_battery + rounding_jitter
    new_signal (unmet_for_rebalancing)  = away_flying + insufficient_battery
so old - new == rounding_jitter exactly, always, per pair (task_assignment.py's own
invariant). What's NOT guaranteed analytically is the *aggregate* effect once you
sum across all destination pairs sharing one origin site within one bin (that's
the actual granularity fed to rebalancing.py's Tier-1 "shortage" dict) -- a site
with broad demand fan-out could accumulate many sub-1 rounding remainders into a
double-digit aggregate that behaves, in the priority-sorted donor/target logic,
just like a genuine large shortage would have. This script measures that directly
instead of assuming the per-pair bound generalizes.

Method: monkeypatch time_step_path_assignment with a passthrough wrapper that
copies its already-computed, per-bin `shortfall_by_site` return value into a log
before returning it unchanged -- zero effect on dispatch/rebalancing behavior,
pure side-channel capture. Replays the CONVERGED iteration (t=5) of the v2
(post-fix) mode-choice-candidate main scenario, i.e. the actual equilibrium point
used in the frozen comparison, using its already-computed bucket_csv (skips
re-deriving mode-choice/dynspeed demand, only reruns the fleet-sim leg).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR / "scripts" / "fleet_sim"))

import run_shanghai_fleet_simulation as sim_mod  # noqa: E402
from distance_battery import battery_consumption_required, set_distance_data  # noqa: E402
from initialization import initialize_states_with_time  # noqa: E402

BIN_LOG = []
_orig_tspa = sim_mod.time_step_path_assignment


def _wrapped_tspa(*args, **kwargs):
    global _CURRENT_T
    result = _orig_tspa(*args, **kwargs)
    shortfall_by_site = result[5]
    for s, acc in shortfall_by_site.items():
        BIN_LOG.append({
            "t": _CURRENT_T,
            "site": s,
            "away_flying": acc["away_flying"],
            "insufficient_battery": acc["insufficient_battery"],
            "rounding_jitter": acc["rounding_jitter"],
            "requested": acc["requested"],
            "idle_at_s": acc["idle_at_s"],
        })
    return result


sim_mod.time_step_path_assignment = _wrapped_tspa

# run_iterations references time_step_path_assignment as a free variable bound
# at module-import time inside run_shanghai_fleet_simulation.py's own namespace
# (`from task_assignment import ... time_step_path_assignment`), so patching
# sim_mod.time_step_path_assignment before run_iterations executes is sufficient
# -- Python looks up globals at call time, not def time, confirmed by re-reading
# run_iterations' call site (line ~262: bare name, not task_assignment.foo).

_CURRENT_T = -1
# run_iterations doesn't expose current t to callers; reconstruct it by
# wrapping update_arrivals instead, which IS called once per bin with
# current_step=t as an explicit kwarg -- cheaper than reimplementing the loop.
_orig_update_arrivals = sim_mod.update_arrivals


def _wrapped_update_arrivals(*args, **kwargs):
    global _CURRENT_T
    _CURRENT_T = kwargs.get("current_step", args[3] if len(args) > 3 else _CURRENT_T)
    return _orig_update_arrivals(*args, **kwargs)


sim_mod.update_arrivals = _wrapped_update_arrivals


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites-csv", default=str(PROJECT_DIR / "data/selected_sites_kmeans_K30_modechoice_iter1.csv"))
    ap.add_argument("--bucket-csv", default=str(PROJECT_DIR / "outputs/fleet_sim/fleet_sim_dynspeed_full_demand_bucket_eqsearch_v2_mc_main_iter5.csv"))
    ap.add_argument("--vehicles-per-vertiport", type=int, default=250)
    ap.add_argument("--tag", default="diag_rebalance_scope_v2mc_iter5")
    ap.add_argument("--out-csv", default=None)
    # run_equilibrium_search.py's own --n-bins default is 500 (not the full 1392
    # the standalone full_dynspeed CLI path defaults to) -- every production
    # eqsearch_* run this diagnostic replays used exactly 500 bins, confirmed by
    # the saved time_step_summary_*.csv / vertiport_occupancy_*.csv row counts
    # (501 = 500 + header). Must match to be a faithful replay, not a superset.
    ap.add_argument("--n-bins", type=int, default=500)
    args = ap.parse_args()

    sites_csv = args.sites_csv
    bucket_csv = args.bucket_csv
    tag = args.tag

    distance_csv = sim_mod.SIM_OUT_DIR / f"vertiport_distance_km.{tag}.tmp.csv"
    grid_ids = sim_mod.build_vertiport_distance_csv(sites_csv, distance_csv)
    set_distance_data(distance_csv)

    n_bins = args.n_bins
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

    print(f"vertiports={len(grid_ids)} bins={n_bins} fleet={vehicles_per_vertiport * len(grid_ids)}", flush=True)

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

    distance_csv.unlink(missing_ok=True)

    out_csv = Path(args.out_csv) if args.out_csv else PROJECT_DIR / "outputs/fleet_sim/_sweep_tmp/rebalance_scope_diag_bin_log.csv"
    pd.DataFrame(BIN_LOG).to_csv(out_csv, index=False)
    print(f"saved {len(BIN_LOG)} rows -> {out_csv}", flush=True)


if __name__ == "__main__":
    main()
