# -*- coding: utf-8 -*-
"""
Shanghai fleet simulation driver -- adapted from simulation_2.py's
run_iterations()/__main__ (whose imports point at modules and a dataset
that don't exist in this environment; see distance_battery.py /
initialization.py / battery_charging.py / task_assignment.py /
gurobi_optimization.py / gurobi_optimization_Colum.py / metrics.py in this
same folder for the reconstructed modules those pieces call into).

Each simulation tick = one real 30-min bin. Demand per bin comes from the
REAL site-to-site OD tensor already extracted for the 30 kmeans vertiports
(shanghai_od_kmeans30_30min_1channel.npz, 1392 bins) -- this is genuine
observed demand between vertiport pairs, not the earlier grid-to-vertiport
routing output (that answers a different question: which vertiport pair a
ground trip should use; this fleet sim asks whether a limited number of
physical vehicles can actually carry the trips that want to use each pair).

Per bin: land any vehicles due to arrive -> merge this bin's real demand
with unmet demand carried over from the previous bin -> Gurobi plans a
capacity-respecting dispatch (vehicle count per vertiport as the capacity
signal) -> physically dispatch vehicles, respecting real battery/standby
state (this is where shortfall beyond Gurobi's plan can appear) -> charge
grounded vehicles -> carry remaining unmet demand into the next bin.

assignment_ratio (分配率), matching simulation_2.py's coverage_rate exactly:
    met_demand / (met_demand + unmet_demand), summed across all bins.
"""
import argparse
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from distance_battery import haversine_km, set_distance_data, calculate_distance, battery_consumption_required
from initialization import initialize_states_with_time
from battery_charging import charging_and_battery_update, restore_vehicle_states, export_charging_log
from task_assignment import update_arrivals, time_step_path_assignment
from gurobi_optimization import run_gurobi_optimization
from gurobi_optimization_Colum import column_generation
from metrics import calculate_cost
from rebalancing import redistribute_vehicles, new_flow_history, update_flow_history
from positioning_lp import plan_and_execute_positioning

PROJECT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUT_DIR = PROJECT_DIR / "outputs"
SIM_OUT_DIR = OUT_DIR / "fleet_sim"

OD_META_CSV = OUT_DIR / "shanghai_od_kmeans30_meta.csv"
WEATHER_CSV = DATA_DIR / "shanghai_calendar_weather_202504.csv"
SPEED_BUCKET_CSV = DATA_DIR / "predicted_ground_speed_by_bucket.csv"
DYNSPEED_DEMAND_BUCKET_CSV = OUT_DIR / "fleet_sim_dynspeed_full_demand_bucket.csv"


def build_ground_speed_per_bin(n_bins, meta_csv=OD_META_CSV, weather_csv=WEATHER_CSV,
                                speed_bucket_csv=SPEED_BUCKET_CSV):
    """One ground_speed_kmh value per simulation bin t, looked up from the same
    48-row (hour_of_day, day_type) bucket table route_assignment_od_to_vertiports.py
    uses -- day_type collapses the calendar's workday/weekend/holiday split to a
    binary workday/offday, matching extract_full_grid_od_pairs_by_bucket.py's
    convention exactly, so both scripts see the same speed for the same bin."""
    meta = pd.read_csv(meta_csv, parse_dates=["timestamp"]).sort_values("compressed_bin_idx")
    assert len(meta) >= n_bins, f"meta has {len(meta)} rows, need {n_bins}"
    cal = pd.read_csv(weather_csv, parse_dates=["date"])
    is_workday = dict(zip(cal["date"].dt.strftime("%Y%m%d"), cal["day_type"].eq("workday")))
    speed_lookup = pd.read_csv(speed_bucket_csv).set_index(["hour_of_day", "day_type"])["predicted_speed_kmh"]

    speeds = np.empty(n_bins, dtype=np.float64)
    for t in range(n_bins):
        ts = meta.iloc[t]["timestamp"]
        day_type = "workday" if is_workday.get(ts.strftime("%Y%m%d")) else "offday"
        speeds[t] = float(speed_lookup.loc[(ts.hour, day_type)])
    return speeds


def load_demand_and_access_per_bin(n_bins, grid_ids, bucket_csv=DYNSPEED_DEMAND_BUCKET_CSV,
                                    meta_csv=OD_META_CSV, weather_csv=WEATHER_CSV):
    """Alternative to load_demand_per_bin(): real full-grid routed demand
    (build_dynspeed_full_demand.py, derived from route_assignment_od_to_vertiports.py's
    dynamic-speed output) instead of the ~184K-trip intra-cell-only subset --
    real access/egress ground distances (km-scale, not the ~0.4km intra-cell
    approximation task_assignment.ACCESS_EGRESS_KM uses by default) come along
    with it. bucket_csv has one row per (takeoff, landing, hour_of_day,
    day_type); every bin sharing a bucket gets that bucket's already-averaged
    avg_trip_count_per_bin (see build_dynspeed_full_demand.py's docstring for
    why day-to-day variation within a bucket can't be recovered).
    Returns (demand_per_bin, access_egress_per_bin), both length n_bins;
    access_egress_per_bin[t] is a {(start,end): (access_km, egress_km)} dict."""
    meta = pd.read_csv(meta_csv, parse_dates=["timestamp"]).sort_values("compressed_bin_idx")
    assert len(meta) >= n_bins, f"meta has {len(meta)} rows, need {n_bins}"
    cal = pd.read_csv(weather_csv, parse_dates=["date"])
    is_workday = dict(zip(cal["date"].dt.strftime("%Y%m%d"), cal["day_type"].eq("workday")))

    bucket = pd.read_csv(bucket_csv)
    bucket["takeoff_grid_id"] = bucket["takeoff_grid_id"].astype(str)
    bucket["landing_grid_id"] = bucket["landing_grid_id"].astype(str)
    valid_grid = set(grid_ids)
    assert set(bucket["takeoff_grid_id"]) <= valid_grid, "bucket_csv references a grid id not in --sites"
    assert set(bucket["landing_grid_id"]) <= valid_grid, "bucket_csv references a grid id not in --sites"

    by_bucket_demand, by_bucket_access = {}, {}
    for (hour, day_type), g in bucket.groupby(["hour_of_day", "day_type"]):
        by_bucket_demand[(hour, day_type)] = [
            {"start": s, "end": e, "flow": float(f)}
            for s, e, f in zip(g["takeoff_grid_id"], g["landing_grid_id"], g["avg_trip_count_per_bin"])
        ]
        by_bucket_access[(hour, day_type)] = {
            (s, e): (a, x)
            for s, e, a, x in zip(g["takeoff_grid_id"], g["landing_grid_id"], g["access_km"], g["egress_km"])
        }

    demand_per_bin, access_egress_per_bin = [], []
    for t in range(n_bins):
        ts = meta.iloc[t]["timestamp"]
        day_type = "workday" if is_workday.get(ts.strftime("%Y%m%d")) else "offday"
        key = (ts.hour, day_type)
        demand_per_bin.append(by_bucket_demand.get(key, []))
        access_egress_per_bin.append(by_bucket_access.get(key, {}))
    return demand_per_bin, access_egress_per_bin


def build_vertiport_distance_csv(sites_csv, out_csv):
    sites = pd.read_csv(sites_csv)
    grid_ids = [str(g) for g in sites["Grid ID"]]
    lat = sites["avg_lat"].to_numpy()
    lon = sites["avg_lon"].to_numpy()
    n = len(grid_ids)
    dist = np.zeros((n, n))
    for i in range(n):
        dist[i] = haversine_km(lat[i], lon[i], lat, lon)
    df = pd.DataFrame(dist, index=grid_ids, columns=grid_ids)
    df.to_csv(out_csv)
    return grid_ids


def load_demand_per_bin(od_npz, grid_ids):
    od = np.load(od_npz, allow_pickle=True)["arr_0"][0]  # (T, N, N)
    n_bins = od.shape[0]
    per_bin = []
    for t in range(n_bins):
        rows = []
        for i in range(len(grid_ids)):
            for j in range(len(grid_ids)):
                if i != j and od[t, i, j] > 0:
                    rows.append({"start": grid_ids[i], "end": grid_ids[j], "flow": int(od[t, i, j])})
        per_bin.append(rows)
    return per_bin


def compute_most_needed(unmet_demand_local):
    if not unmet_demand_local:
        return ("", 0)
    need = {}
    for s, e, flow in unmet_demand_local:
        need[e] = need.get(e, 0) + flow
    target = max(need, key=need.get)
    return (target, need[target])


def run_iterations(num_bins, vehicle_states, vertiport_states, demand_per_bin, charging_rate,
                    discharge_rate, vertiports, distance_air, charging_rate_per_bin,
                    ground_speed_per_bin, dwell_min=5.0, access_egress_per_bin=None, log_charging=True,
                    rebalance_interval=0, rebalance_min_reserve=1, rebalance_max_idle_cap=None,
                    predictive_rebalancing=False, predictive_threshold=1.5, predictive_gain=1.0,
                    predictive_window=30, predictive_top_k=3, predictive_max_drain=3,
                    positioning_lp_enabled=False, positioning_lp_horizon=8, positioning_lp_lambda=0.001,
                    positioning_lp_diag_log=None):
    unmet_demand = []
    time_step_summary_records = []
    cumulative_cost = 0.0
    vehicle_movements = {}
    charging_log_tracker = {}
    assigned_routes_log = []
    rebalance_log = []
    vertiport_occupancy_log = []
    vertiport_total_count_log = []
    # Diagnostic-only per-site shortfall accumulator (2026-07-20 investigation
    # of grid 433-style sites: high idle count but low own-demand assignment
    # ratio). Not used by dispatch/rebalancing, purely for post-hoc analysis.
    shortfall_by_site_accum = {}
    # one row per bin (O(n_bins), not O(fleet_size x n_bins) like the old
    # per-vehicle charging_tracker) -- safe to always collect even at large
    # fleet sizes. LOW_BATTERY_THRESHOLD_PCT=20 is a common EV-industry
    # "reserve" convention; no project-specific threshold existed already.
    battery_summary_log = []
    LOW_BATTERY_THRESHOLD_PCT = 20.0
    flow_history = new_flow_history() if predictive_rebalancing else None
    # per-origin-vertiport accounting: total demand seen (new + carried-in backlog, before
    # this bin's assignment) vs. demand still unmet after this bin's assignment. Lets us
    # compute a per-vertiport assignment_ratio, mirroring the fleet-wide one below but
    # split by site -- something the fleet-wide time_step_summary can't give us since it
    # only tracks aggregate totals.
    site_order_totals = {v: 0.0 for v in vertiports}
    site_unmet_totals = {v: 0.0 for v in vertiports}

    # Wait-time (queue-age) tracking for unmet demand, purely diagnostic --
    # doesn't feed back into dispatch (all demand for a given (s,e) pair is
    # fungible in the LP/dispatch stage; "which units get served" is an
    # arbitrary bookkeeping choice for reporting, so a FIFO convention
    # (oldest backlog served first) is used here). unmet_cohorts is keyed by
    # (s,e) -> list of [flow, age_in_bins], persisted across bins; a cohort's
    # age counts how many bins it has sat unserved so far.
    unmet_cohorts = {}
    # Parallel ledger using "continuous" (non-integer-floored) dispatch instead of the real
    # whole-vehicle dispatch -- isolates genuine capacity/battery-driven waiting from the
    # "waiting for enough fractional demand to accumulate to one whole vehicle" rounding
    # artifact that inflates unmet_cohorts' age whenever per-bin-per-pair demand is small.
    unmet_cohorts_operational = {}
    WAIT_THRESHOLD_BINS = 10  # 5 hours at 30-min bins; arbitrary "chronic backlog" cutoff for reporting

    for t in range(num_bins):
        current_unmet = unmet_demand
        unmet_demand = []

        arrivals_this_bin = update_arrivals(vehicle_states, vertiport_states, vehicle_movements,
                                             current_step=t, debug=False)
        restore_vehicle_states(vehicle_states, vehicle_movements)

        new_orders = [(row["start"], row["end"], float(row["flow"])) for row in demand_per_bin[t]]
        total_orders = current_unmet + new_orders

        from collections import Counter
        vehicle_count = Counter(vs["loc"] for vs in vehicle_states.values())
        vehicle_total = sum(vehicle_count.values())

        if not total_orders:
            gurobi_results = []
        else:
            gurobi_results = column_generation(
                orders=total_orders,
                vertiports=vertiports,
                vehicle_states=vehicle_states,
                distance_air=distance_air,
                discharge_rate=discharge_rate,
                vehicle_count=vehicle_count,
                vehicle_total=vehicle_total,
                run_gurobi_optimization=run_gurobi_optimization,
                time_step=t,
                max_iters=1,
                max_pq_per_order=3,
            )

        total_paths = [
            {"takeoff": r["takeoff"], "landing": r["landing"], "flow": float(r["flow"]), "distance": r["distance"]}
            for r in gurobi_results if float(r["flow"]) > 0
        ]

        # Stage-1 shortfall: demand the capacity-constrained LP itself couldn't
        # grant (vehicle_count budget already exhausted at that vertiport) --
        # this never reaches time_step_path_assignment at all, so it has to be
        # recovered here by diffing aggregated demand against what the LP granted,
        # or it silently vanishes instead of carrying over to the next bin.
        demand_agg = {}
        for s, e, f in total_orders:
            demand_agg[(s, e)] = demand_agg.get((s, e), 0.0) + f
        granted_agg = {(r["takeoff"], r["landing"]): float(r["flow"]) for r in gurobi_results}
        stage1_unmet = [
            (s, e, demand_agg[(s, e)] - granted_agg.get((s, e), 0.0))
            for (s, e) in demand_agg
            if demand_agg[(s, e)] - granted_agg.get((s, e), 0.0) > 1e-9
        ]

        unmet_after_assignment, assigned_routes, launched_ids, shortfall_reasons, operational_served, shortfall_by_site, unmet_for_rebalancing = time_step_path_assignment(
            gurobi_results=total_paths,
            vehicle_states=vehicle_states,
            vertiport_states=vertiport_states,
            discharge_rate=discharge_rate,
            vehicle_movements=vehicle_movements,
            current_step=t,
            ground_speed_kmh=ground_speed_per_bin[t],
            dwell_min=dwell_min,
            access_egress_lookup=access_egress_per_bin[t] if access_egress_per_bin is not None else None,
            debug=False,
        )

        unmet_demand = stage1_unmet + unmet_after_assignment
        # Fed to the reactive rebalancer instead of unmet_demand: stage1_unmet
        # is a genuine headcount shortage (keep it); unmet_for_rebalancing
        # excludes the rounding_jitter component of unmet_after_assignment
        # (see task_assignment.py's time_step_path_assignment docstring/comments).
        unmet_demand_for_rebalancing = stage1_unmet + unmet_for_rebalancing

        # accumulate diagnostic per-site shortfall (stage1 headcount-exhausted
        # + the three time_step_path_assignment reasons), across all bins
        for s, e, f in stage1_unmet:
            acc = shortfall_by_site_accum.setdefault(s, {"stage1_fleet_insufficient": 0.0, "away_flying": 0.0, "insufficient_battery": 0.0, "rounding_jitter": 0.0, "requested": 0.0})
            acc["stage1_fleet_insufficient"] += f
        for s, site_acc in shortfall_by_site.items():
            acc = shortfall_by_site_accum.setdefault(s, {"stage1_fleet_insufficient": 0.0, "away_flying": 0.0, "insufficient_battery": 0.0, "rounding_jitter": 0.0, "requested": 0.0})
            acc["away_flying"] += site_acc["away_flying"]
            acc["insufficient_battery"] += site_acc["insufficient_battery"]
            acc["rounding_jitter"] += site_acc["rounding_jitter"]
            acc["requested"] += site_acc["requested"]

        # Failure-reason breakdown for this bin's total shortfall (sums to `remaining`
        # below): stage-1 = LP's own vehicle-count-at-origin cap already exhausted
        # (genuinely too few vehicles ever positioned/assigned to that vertiport,
        # counting both idle and mid-flight); the other two decompose
        # time_step_path_assignment's dispatch-stage shortfall (see its docstring).
        unmet_fleet_insufficient = sum(f for _, _, f in stage1_unmet)
        unmet_away_flying = shortfall_reasons["away_flying"]
        unmet_insufficient_battery = shortfall_reasons["insufficient_battery"]
        unmet_rounding_jitter = shortfall_reasons["rounding_jitter"]

        # --- wait-time (queue-age) bookkeeping, diagnostic only ---
        new_orders_agg = {}
        for s, e, f in new_orders:
            new_orders_agg[(s, e)] = new_orders_agg.get((s, e), 0.0) + f
        served_agg = {}
        for r in assigned_routes:
            served_agg[(r["start"], r["end"])] = served_agg.get((r["start"], r["end"]), 0.0) + r["flow"]
        age_at_service_weighted_sum = 0.0
        age_at_service_total_qty = 0.0
        for (s, e), total_demand_pair in demand_agg.items():
            cohorts = unmet_cohorts.pop((s, e), [])
            cohorts.append([new_orders_agg.get((s, e), 0.0), 0])
            cohorts.sort(key=lambda c: -c[1])  # oldest (largest age) first
            to_serve = served_agg.get((s, e), 0.0)
            remaining_cohorts = []
            for flow, age in cohorts:
                if to_serve <= 1e-9:
                    if flow > 1e-9:
                        remaining_cohorts.append([flow, age])
                    continue
                served_here = min(flow, to_serve)
                age_at_service_weighted_sum += served_here * age
                age_at_service_total_qty += served_here
                to_serve -= served_here
                left = flow - served_here
                if left > 1e-9:
                    remaining_cohorts.append([left, age])
            if remaining_cohorts:
                unmet_cohorts[(s, e)] = remaining_cohorts
        for cohorts in unmet_cohorts.values():
            for c in cohorts:
                c[1] += 1  # carried into next bin: one bin older
        mean_age_at_service = (age_at_service_weighted_sum / age_at_service_total_qty
                                if age_at_service_total_qty > 0 else 0.0)
        max_backlog_age = max((c[1] for cohorts in unmet_cohorts.values() for c in cohorts), default=0)
        backlog_over_threshold = sum(c[0] for cohorts in unmet_cohorts.values() for c in cohorts
                                      if c[1] > WAIT_THRESHOLD_BINS)

        # --- operational-only wait ledger (continuous dispatch, no rounding artifact) ---
        operational_served_agg = {}
        for r in operational_served:
            operational_served_agg[(r["start"], r["end"])] = operational_served_agg.get((r["start"], r["end"]), 0.0) + r["flow"]
        age_op_weighted_sum = 0.0
        age_op_total_qty = 0.0
        for (s, e), total_demand_pair in demand_agg.items():
            cohorts = unmet_cohorts_operational.pop((s, e), [])
            cohorts.append([new_orders_agg.get((s, e), 0.0), 0])
            cohorts.sort(key=lambda c: -c[1])
            to_serve = operational_served_agg.get((s, e), 0.0)
            remaining_cohorts = []
            for flow, age in cohorts:
                if to_serve <= 1e-9:
                    if flow > 1e-9:
                        remaining_cohorts.append([flow, age])
                    continue
                served_here = min(flow, to_serve)
                age_op_weighted_sum += served_here * age
                age_op_total_qty += served_here
                to_serve -= served_here
                left = flow - served_here
                if left > 1e-9:
                    remaining_cohorts.append([left, age])
            if remaining_cohorts:
                unmet_cohorts_operational[(s, e)] = remaining_cohorts
        for cohorts in unmet_cohorts_operational.values():
            for c in cohorts:
                c[1] += 1
        mean_age_at_service_operational = (age_op_weighted_sum / age_op_total_qty
                                            if age_op_total_qty > 0 else 0.0)

        for s, e, f in total_orders:
            site_order_totals[s] += f
        for s, e, f in unmet_demand:
            site_unmet_totals[s] += f

        met_qty = sum(route["flow"] for route in assigned_routes)
        total_demand = sum(f for _, _, f in total_orders)
        remaining = total_demand - met_qty

        if predictive_rebalancing:
            departures_this_bin = Counter()
            for r in assigned_routes:
                departures_this_bin[r["start"]] += r["flow"]
            update_flow_history(flow_history, vertiports, arrivals_this_bin, departures_this_bin,
                                 window=predictive_window)

        moves = []
        if positioning_lp_enabled:
            # Replaces Tier 0 entirely (flow_history stays None below, since
            # predictive_rebalancing is not set when this path is used -- see
            # positioning_lp.py's docstring). Runs AFTER this bin's real
            # passenger dispatch (time_step_path_assignment, above), so
            # idle_by_vertiport already reflects this bin's revenue departures
            # -- the LP only ever decides EMPTY repositioning for bin t.
            H_eff = min(positioning_lp_horizon, num_bins - t)
            lp_moves = plan_and_execute_positioning(
                vehicle_states=vehicle_states,
                vertiport_states=vertiport_states,
                vehicle_movements=vehicle_movements,
                grid_ids=vertiports,
                distance_air=distance_air,
                current_step=t,
                horizon=H_eff,
                demand_window=demand_per_bin[t:t + H_eff],
                access_egress_window=(access_egress_per_bin[t:t + H_eff] if access_egress_per_bin is not None
                                       else [{}] * H_eff),
                ground_speed_window=ground_speed_per_bin[t:t + H_eff],
                discharge_rate=discharge_rate,
                min_reserve=rebalance_min_reserve,
                lam=positioning_lp_lambda,
                diag=positioning_lp_diag_log,
            )
            moves.extend(lp_moves)
            for mv in lp_moves:
                rebalance_log.append(mv)

        # Tier 1 (shortage) + Tier 2 (overflow cap) always run, sharing the
        # SAME vehicle_states/vertiport_states/vehicle_movements the LP just
        # updated above -- idle_by_vertiport is recomputed fresh inside
        # redistribute_vehicles from vehicle_states, so vehicles the LP just
        # dispatched are correctly excluded, no double-dispatch risk.
        if rebalance_interval > 0 and t % rebalance_interval == 0:
            tier12_moves = redistribute_vehicles(
                vehicle_states=vehicle_states,
                vertiport_states=vertiport_states,
                distance_air=distance_air,
                unmet_demand=unmet_demand_for_rebalancing,
                vehicle_movements=vehicle_movements,
                current_step=t,
                discharge_rate=discharge_rate,
                min_reserve=rebalance_min_reserve,
                max_idle_cap=rebalance_max_idle_cap,
                flow_history=flow_history,
                predictive_threshold=predictive_threshold,
                predictive_gain=predictive_gain,
                predictive_top_k=predictive_top_k,
                predictive_max_drain=predictive_max_drain,
                debug=False,
            )
            moves.extend(tier12_moves)
            for mv in tier12_moves:
                rebalance_log.append(mv)

        vertiport_occupancy_log.append({
            "time_step": t,
            **{v: vertiport_states[v]["avail"] for v in vertiports},
        })
        # Total fleet count "belonging" to each vertiport (idle + still
        # mid-flight, keyed by each vehicle's current `loc`) -- loc only
        # updates on arrival (see update_arrivals), so an in-flight vehicle
        # still counts at its departure vertiport until it lands. This is
        # the same bookkeeping convention already used for the Gurobi
        # capacity signal (vehicle_count Counter above), just re-taken here
        # after this bin's dispatch + rebalancing so it's a full distribution
        # snapshot, not just the idle subset.
        from collections import Counter as _Counter
        loc_counts = _Counter(vs["loc"] for vs in vehicle_states.values())
        vertiport_total_count_log.append({
            "time_step": t,
            **{v: loc_counts.get(v, 0) for v in vertiports},
        })

        transport_cost = calculate_cost(assigned_routes, cost_per_distance=4, distance_map=distance_air)
        reposition_cost = sum(mv["distance"] * 4 for mv in moves)
        operating_cost = len(vehicle_states) * 5
        iteration_cost = transport_cost + reposition_cost + operating_cost
        cumulative_cost += iteration_cost

        time_step_summary_records.append({
            "time_step": t,
            "met_demand": met_qty,
            "unmet_demand": remaining,
            "new_demand": sum(f for _, _, f in new_orders),
            "carried_in_unmet": sum(f for _, _, f in current_unmet),
            "carried_out_unmet": sum(f for _, _, f in unmet_demand),
            "n_rebalance_moves": len(moves),
            "iteration_cost": iteration_cost,
            "cumulative_cost": cumulative_cost,
            "n_vehicles_in_service": sum(vs["in_service"] for vs in vehicle_states.values()),
            "unmet_fleet_insufficient": unmet_fleet_insufficient,
            "unmet_away_flying": unmet_away_flying,
            "unmet_insufficient_battery": unmet_insufficient_battery,
            "unmet_rounding_jitter": unmet_rounding_jitter,
            "mean_age_at_service_bins": mean_age_at_service,
            "max_backlog_age_bins": max_backlog_age,
            "backlog_over_threshold": backlog_over_threshold,
            "mean_age_at_service_operational_bins": mean_age_at_service_operational,
        })
        for route in assigned_routes:
            assigned_routes_log.append({"time_step": t, **route})

        charging_and_battery_update(
            vehicle_states, time_interval=30, charging_rate=charging_rate_per_bin,
            current_step=t, charging_tracker=charging_log_tracker, log_enabled=log_charging,
        )

        batteries = [vs["battery"] for vs in vehicle_states.values()]
        n_low = sum(1 for b in batteries if b < LOW_BATTERY_THRESHOLD_PCT)
        # n_charging: idle (in_service==0) and not full -- matches battery_charging.py's own
        # "charging" flag condition, so this is literally how many vehicles are plugged in
        # and gaining charge this bin (charging utilization = n_charging / fleet_size).
        n_charging = sum(1 for vs in vehicle_states.values() if vs["in_service"] == 0 and vs["battery"] < 100.0)
        # energy consumed this bin, in % -battery units (no kWh capacity is defined anywhere
        # in this project -- battery is a pure 0-100% quantity, so "energy" here is only ever
        # reported in that unit, not kWh): sum of required_battery over all vehicles actually
        # dispatched this bin, recomputed from assigned_routes' own (flow, distance) since
        # task_assignment.py already discharges each dispatched vehicle by exactly this amount.
        energy_consumed_pct_this_bin = sum(
            route["flow"] * battery_consumption_required(route["distance"], discharge_rate)
            for route in assigned_routes
        )
        battery_summary_log.append({
            "time_step": t,
            "low_battery_ratio": n_low / len(batteries) if batteries else 0.0,
            "low_battery_count": n_low,
            "mean_battery_pct": sum(batteries) / len(batteries) if batteries else 0.0,
            "n_charging": n_charging,
            "charging_utilization": n_charging / len(batteries) if batteries else 0.0,
            "energy_consumed_pct_this_bin": energy_consumed_pct_this_bin,
        })

        if t % 100 == 0:
            print(f"[bin {t}/{num_bins}] met={met_qty:.0f} unmet={remaining:.0f} in_service={time_step_summary_records[-1]['n_vehicles_in_service']} rebalanced={len(moves)}")

    site_assignment_stats = [
        {"grid_id": v, "total_orders": site_order_totals[v], "total_unmet": site_unmet_totals[v]}
        for v in vertiports
    ]

    return (time_step_summary_records, cumulative_cost, assigned_routes_log, charging_log_tracker,
            rebalance_log, vertiport_occupancy_log, vertiport_total_count_log, site_assignment_stats,
            battery_summary_log, shortfall_by_site_accum)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", type=str, default=str(DATA_DIR / "selected_sites_kmeans_K30.csv"))
    ap.add_argument("--od-npz", type=str, default=str(OUT_DIR / "shanghai_od_kmeans30_30min_1channel.npz"))
    ap.add_argument("--demand-source", choices=["intracell", "full_dynspeed"], default="intracell",
                     help="intracell (default, unchanged behavior): --od-npz's ~184K-trip subset "
                          "(only trips whose origin AND destination fall inside the vertiport's own grid "
                          "cell), fixed ~0.4km access/egress approximation. full_dynspeed: "
                          "build_dynspeed_full_demand.py's full-grid routed demand (~11.6M trips/month, "
                          "real km-scale access/egress distances + the routing decisions dynamic ground "
                          "speed already influenced) -- see --demand-bucket-csv")
    ap.add_argument("--demand-bucket-csv", type=str, default=str(DYNSPEED_DEMAND_BUCKET_CSV),
                     help="only used when --demand-source=full_dynspeed")
    ap.add_argument("--speed-bucket-csv", type=str, default=str(SPEED_BUCKET_CSV),
                     help="48-bucket ground-speed lookup used for access/egress time; default is the "
                          "model-predicted mean table. Pass an alternate (e.g. a p10 rush-hour stress "
                          "variant) to test sensitivity.")
    ap.add_argument("--skip-charging-log", action="store_true",
                     help="don't accumulate the per-vehicle-per-bin charging_log_<tag>.csv detail "
                          "(battery math is unaffected, only the diagnostic log is skipped). This dict "
                          "grows O(fleet_size x n_bins) -- at large fleet sizes (10K+ vehicles) it can "
                          "reach tens of millions of entries and exceed a constrained environment's "
                          "memory limit well before the simulation does anything heavy. Not needed for "
                          "service-rate figures (those come from time_step_summary, not this log).")
    ap.add_argument("--n-bins", type=int, default=None, help="limit to first N bins (default: all 1392)")
    ap.add_argument("--vehicles-per-vertiport", type=int, default=5)
    ap.add_argument("--discharge-rate", type=float, default=1.0, help="%% battery per km flown")
    ap.add_argument("--charging-rate-per-bin", type=float, default=7.0, help="%% battery gained per 30-min bin while standby")
    ap.add_argument("--dwell-min", type=float, default=5.0,
                     help="fixed boarding/security overhead per ground leg (access + egress), minutes; "
                          "no ground-truth data exists for this, it's an assumption -- see task_assignment.py")
    ap.add_argument("--rebalance-interval", type=int, default=0,
                     help="reposition idle vehicles toward backlogged vertiports every N bins (0 = disabled)")
    ap.add_argument("--rebalance-min-reserve", type=int, default=1,
                     help="idle vehicles a vertiport keeps for itself before it can donate any to rebalancing")
    ap.add_argument("--rebalance-max-idle-multiplier", type=float, default=2.0,
                     help="hard cap on idle vehicles per vertiport, as a multiple of its initial fleet share "
                          "(vehicles-per-vertiport); excess is forced to drain toward low-idle vertiports "
                          "regardless of backlog. 0/negative disables the cap.")
    ap.add_argument("--predictive-rebalancing", action="store_true",
                     help="enable Tier 0: drain vertiports with a persistent net-inflow (real arrivals "
                          "outpacing departures over a rolling window) toward persistent net-source "
                          "vertiports, before the hard idle cap (Tier 2) ever trips")
    ap.add_argument("--predictive-threshold", type=float, default=1.5,
                     help="rolling-average net inflow (vehicles/bin) above which a vertiport is treated as a sink")
    ap.add_argument("--predictive-gain", type=float, default=1.0,
                     help="multiplier on a sink's net-inflow EMA to size how many idle vehicles to drain per bin")
    ap.add_argument("--predictive-window", type=int, default=30,
                     help="rolling window (bins) over which net inflow is averaged")
    ap.add_argument("--predictive-top-k", type=int, default=3,
                     help="only act on the k most net-inflow-heavy vertiports per bin, not every vertiport "
                          "that clears the threshold (net flow is zero-sum across the fleet, so on any given "
                          "bin roughly half of all vertiports are nominally 'positive' -- unbounded top-k "
                          "turned Tier 0 into fleet-wide churn that cost more battery than it saved)")
    ap.add_argument("--predictive-max-drain", type=int, default=3,
                     help="hard cap on idle vehicles drained per sink per bin")
    ap.add_argument("--rebalance-mechanism", choices=["tier0", "lp"], default="lp",
                     help="lp (default as of 2026-07-23): the rolling-horizon positioning LP "
                          "(positioning_lp.py, 2026-07-22) -- beat tier0 in every one of 6 fixed-demand "
                          "robustness scenarios (R* +2.3 to +4.4pp, empty mileage -2.6%% to -9.9%%, reactive "
                          "shortage dispatch -28.9%% to -84.7%%) and in full demand-equilibrium re-convergence "
                          "for all 3 site layouts (all crossed back above the 95%% feasibility gate; see "
                          "outputs/fleet_sim/LP_MIGRATION_REPORT.md). tier0: the older net_inflow-based "
                          "predictive tier, gated by --predictive-rebalancing -- kept for backward "
                          "compatibility/comparison, not recommended for new runs. --predictive-rebalancing "
                          "and the --predictive-* flags are ignored when this is 'lp' (Tier 1 shortage / "
                          "Tier 2 overflow cap still run unchanged either way).")
    ap.add_argument("--positioning-lp-horizon", type=int, default=8,
                     help="rolling lookahead window H (bins) for --rebalance-mechanism=lp; H=8 is the "
                          "2026-07-22 robustness-battery-validated default (H=12 gave only +0.26 to +0.80pp "
                          "R* over H=8 across 5 scenarios, not enough to justify its larger/more fragile LP)")
    ap.add_argument("--positioning-lp-lambda", type=float, default=0.001,
                     help="small per-km penalty on empty repositioning distance in the LP objective, purely "
                          "to break ties against pointless moves -- never large enough to compete with the "
                          "objective's dominant term (maximizing served demand)")
    ap.add_argument("--positioning-lp-diag-out", type=Path, default=None,
                    help="optional CSV path for per-bin LP rounding diagnostics; does not affect dispatch")
    ap.add_argument("--tag", type=str, default="v1")
    args = ap.parse_args()

    SIM_OUT_DIR.mkdir(parents=True, exist_ok=True)
    distance_csv = SIM_OUT_DIR / f"vertiport_distance_km.{args.tag}.tmp.csv"
    grid_ids = build_vertiport_distance_csv(args.sites, distance_csv)
    set_distance_data(distance_csv)
    distance_air = {
        (a, b): calculate_distance(a, b)
        for a in grid_ids for b in grid_ids if a != b
    }

    if args.demand_source == "full_dynspeed":
        n_bins = args.n_bins or 1392
        demand_per_bin, access_egress_per_bin = load_demand_and_access_per_bin(
            n_bins, grid_ids, bucket_csv=args.demand_bucket_csv)
        total_demand_check = sum(f for rows in demand_per_bin for _, _, f in
                                  ((r["start"], r["end"], r["flow"]) for r in rows))
        print(f"demand_source=full_dynspeed  bucket_csv={args.demand_bucket_csv}  "
              f"total real demand across {n_bins} bins: {total_demand_check:.0f}")
    else:
        demand_per_bin = load_demand_per_bin(args.od_npz, grid_ids)
        n_bins = args.n_bins or len(demand_per_bin)
        demand_per_bin = demand_per_bin[:n_bins]
        access_egress_per_bin = None
    print(f"vertiports={len(grid_ids)}  bins={n_bins}  fleet={args.vehicles_per_vertiport * len(grid_ids)} vehicles")

    ground_speed_per_bin = build_ground_speed_per_bin(n_bins, speed_bucket_csv=args.speed_bucket_csv)
    print(f"ground speed (dynamic, per-bucket): min={ground_speed_per_bin.min():.2f}  "
          f"max={ground_speed_per_bin.max():.2f}  mean={ground_speed_per_bin.mean():.2f} km/h")

    vehicles = [f"V{i}" for i in range(1, args.vehicles_per_vertiport * len(grid_ids) + 1)]
    vehicle_states, vertiport_states = initialize_states_with_time(vehicles, grid_ids, args.vehicles_per_vertiport)

    positioning_lp_diag = [] if args.positioning_lp_diag_out else None
    start = time.time()
    (summary, cumulative_cost, assigned_routes_log, charging_log, rebalance_log,
     vertiport_occupancy_log, vertiport_total_count_log, site_assignment_stats,
     battery_summary_log, shortfall_by_site_accum) = run_iterations(
        num_bins=n_bins,
        vehicle_states=vehicle_states,
        vertiport_states=vertiport_states,
        demand_per_bin=demand_per_bin,
        charging_rate=args.charging_rate_per_bin,
        discharge_rate=args.discharge_rate,
        vertiports=grid_ids,
        distance_air=distance_air,
        charging_rate_per_bin=args.charging_rate_per_bin,
        ground_speed_per_bin=ground_speed_per_bin,
        dwell_min=args.dwell_min,
        access_egress_per_bin=access_egress_per_bin,
        log_charging=not args.skip_charging_log,
        rebalance_interval=args.rebalance_interval,
        rebalance_min_reserve=args.rebalance_min_reserve,
        rebalance_max_idle_cap=(
            args.vehicles_per_vertiport * args.rebalance_max_idle_multiplier
            if args.rebalance_max_idle_multiplier > 0 else None
        ),
        predictive_rebalancing=(args.rebalance_mechanism == "tier0" and args.predictive_rebalancing),
        predictive_threshold=args.predictive_threshold,
        predictive_gain=args.predictive_gain,
        predictive_window=args.predictive_window,
        predictive_top_k=args.predictive_top_k,
        predictive_max_drain=args.predictive_max_drain,
        positioning_lp_enabled=(args.rebalance_mechanism == "lp"),
        positioning_lp_horizon=args.positioning_lp_horizon,
        positioning_lp_lambda=args.positioning_lp_lambda,
        positioning_lp_diag_log=positioning_lp_diag,
    )
    elapsed = time.time() - start
    print(f"simulation finished in {elapsed:.1f}s")

    df = pd.DataFrame(summary)
    df.to_csv(SIM_OUT_DIR / f"time_step_summary_{args.tag}.csv", index=False)
    pd.DataFrame(assigned_routes_log).to_csv(SIM_OUT_DIR / f"assigned_routes_{args.tag}.csv", index=False)
    pd.DataFrame(rebalance_log).to_csv(SIM_OUT_DIR / f"rebalance_log_{args.tag}.csv", index=False)
    pd.DataFrame(vertiport_occupancy_log).to_csv(SIM_OUT_DIR / f"vertiport_occupancy_{args.tag}.csv", index=False)
    pd.DataFrame(vertiport_total_count_log).to_csv(SIM_OUT_DIR / f"vertiport_total_count_{args.tag}.csv", index=False)
    if args.positioning_lp_diag_out:
        args.positioning_lp_diag_out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(positioning_lp_diag).to_csv(args.positioning_lp_diag_out, index=False)
    pd.DataFrame(battery_summary_log).to_csv(SIM_OUT_DIR / f"battery_summary_{args.tag}.csv", index=False)
    shortfall_by_site_df = pd.DataFrame([
        {"grid_id": s, **vals} for s, vals in shortfall_by_site_accum.items()
    ])
    shortfall_by_site_df.to_csv(SIM_OUT_DIR / f"shortfall_by_site_{args.tag}.csv", index=False)
    site_stats_df = pd.DataFrame(site_assignment_stats)
    site_stats_df["total_met"] = site_stats_df["total_orders"] - site_stats_df["total_unmet"]
    site_stats_df["assignment_ratio"] = site_stats_df["total_met"] / site_stats_df["total_orders"].replace(0, pd.NA)
    site_stats_df.to_csv(SIM_OUT_DIR / f"vertiport_assignment_ratio_{args.tag}.csv", index=False)
    if not args.skip_charging_log:
        export_charging_log(charging_log, SIM_OUT_DIR / f"charging_log_{args.tag}.csv")

    total_met = df["met_demand"].sum()
    total_unmet = df["unmet_demand"].sum()
    total_new_demand = df["new_demand"].sum()
    assignment_ratio = total_met / (total_met + total_unmet) if (total_met + total_unmet) > 0 else 0.0
    big_picture_ratio = total_met / total_new_demand if total_new_demand > 0 else 0.0

    print("*" * 40)
    print(f"vehicles_per_vertiport={args.vehicles_per_vertiport}  fleet_size={args.vehicles_per_vertiport*len(grid_ids)}")
    print(f"total real demand introduced: {total_new_demand:.0f}")
    print(f"total met: {total_met:.0f}  total unmet (per-bin, incl. carryover): {total_unmet:.0f}")
    print(f"assignment_ratio (分配率, met/(met+unmet)): {assignment_ratio:.4f}")
    print(f"big_picture_assignment_ratio (met/real_flow): {big_picture_ratio:.4f}")
    print(f"cumulative operating+transport cost: {cumulative_cost:.1f}")

    summary_row = pd.DataFrame([{
        "vehicles_per_vertiport": args.vehicles_per_vertiport,
        "fleet_size": args.vehicles_per_vertiport * len(grid_ids),
        "discharge_rate": args.discharge_rate,
        "charging_rate_per_bin": args.charging_rate_per_bin,
        "dwell_min": args.dwell_min,
        "demand_source": args.demand_source,
        "rebalance_interval": args.rebalance_interval,
        "rebalance_min_reserve": args.rebalance_min_reserve,
        "predictive_rebalancing": args.predictive_rebalancing,
        "predictive_threshold": args.predictive_threshold,
        "predictive_gain": args.predictive_gain,
        "predictive_window": args.predictive_window,
        "n_rebalance_moves_total": len(rebalance_log),
        "n_bins": n_bins,
        "total_real_demand": total_new_demand,
        "total_met": total_met,
        "total_unmet": total_unmet,
        "assignment_ratio": assignment_ratio,
        "big_picture_assignment_ratio": big_picture_ratio,
        "cumulative_cost": cumulative_cost,
    }])
    summary_row.to_csv(SIM_OUT_DIR / f"run_summary_{args.tag}.csv", index=False)
    print(f"saved -> {SIM_OUT_DIR / f'run_summary_{args.tag}.csv'}")
    Path(distance_csv).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
