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

from distance_battery import haversine_km, set_distance_data, calculate_distance
from initialization import initialize_states_with_time
from battery_charging import charging_and_battery_update, restore_vehicle_states, export_charging_log
from task_assignment import update_arrivals, time_step_path_assignment
from gurobi_optimization import run_gurobi_optimization
from gurobi_optimization_Colum import column_generation
from metrics import calculate_cost
from rebalancing import redistribute_vehicles, new_flow_history, update_flow_history

PROJECT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUT_DIR = PROJECT_DIR / "outputs"
SIM_OUT_DIR = OUT_DIR / "fleet_sim"


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
                    rebalance_interval=0, rebalance_min_reserve=1, rebalance_max_idle_cap=None,
                    predictive_rebalancing=False, predictive_threshold=1.5, predictive_gain=1.0,
                    predictive_window=30, predictive_top_k=3, predictive_max_drain=3):
    unmet_demand = []
    time_step_summary_records = []
    cumulative_cost = 0.0
    vehicle_movements = {}
    charging_log_tracker = {}
    assigned_routes_log = []
    rebalance_log = []
    vertiport_occupancy_log = []
    vertiport_total_count_log = []
    flow_history = new_flow_history() if predictive_rebalancing else None

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

        unmet_after_assignment, assigned_routes, launched_ids = time_step_path_assignment(
            gurobi_results=total_paths,
            vehicle_states=vehicle_states,
            vertiport_states=vertiport_states,
            discharge_rate=discharge_rate,
            vehicle_movements=vehicle_movements,
            current_step=t,
            debug=False,
        )

        unmet_demand = stage1_unmet + unmet_after_assignment

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
        if rebalance_interval > 0 and t % rebalance_interval == 0:
            moves = redistribute_vehicles(
                vehicle_states=vehicle_states,
                vertiport_states=vertiport_states,
                distance_air=distance_air,
                unmet_demand=unmet_demand,
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
            for mv in moves:
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
        })
        for route in assigned_routes:
            assigned_routes_log.append({"time_step": t, **route})

        charging_and_battery_update(
            vehicle_states, time_interval=30, charging_rate=charging_rate_per_bin,
            current_step=t, charging_tracker=charging_log_tracker,
        )

        if t % 100 == 0:
            print(f"[bin {t}/{num_bins}] met={met_qty:.0f} unmet={remaining:.0f} in_service={time_step_summary_records[-1]['n_vehicles_in_service']} rebalanced={len(moves)}")

    return (time_step_summary_records, cumulative_cost, assigned_routes_log, charging_log_tracker,
            rebalance_log, vertiport_occupancy_log, vertiport_total_count_log)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", type=str, default=str(DATA_DIR / "selected_sites_kmeans_K30.csv"))
    ap.add_argument("--od-npz", type=str, default=str(OUT_DIR / "shanghai_od_kmeans30_30min_1channel.npz"))
    ap.add_argument("--n-bins", type=int, default=None, help="limit to first N bins (default: all 1392)")
    ap.add_argument("--vehicles-per-vertiport", type=int, default=5)
    ap.add_argument("--discharge-rate", type=float, default=1.0, help="%% battery per km flown")
    ap.add_argument("--charging-rate-per-bin", type=float, default=7.0, help="%% battery gained per 30-min bin while standby")
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
    ap.add_argument("--tag", type=str, default="v1")
    args = ap.parse_args()

    SIM_OUT_DIR.mkdir(parents=True, exist_ok=True)
    distance_csv = SIM_OUT_DIR / "vertiport_distance_km.csv"
    grid_ids = build_vertiport_distance_csv(args.sites, distance_csv)
    set_distance_data(distance_csv)
    distance_air = {
        (a, b): calculate_distance(a, b)
        for a in grid_ids for b in grid_ids if a != b
    }

    demand_per_bin = load_demand_per_bin(args.od_npz, grid_ids)
    n_bins = args.n_bins or len(demand_per_bin)
    demand_per_bin = demand_per_bin[:n_bins]
    print(f"vertiports={len(grid_ids)}  bins={n_bins}  fleet={args.vehicles_per_vertiport * len(grid_ids)} vehicles")

    vehicles = [f"V{i}" for i in range(1, args.vehicles_per_vertiport * len(grid_ids) + 1)]
    vehicle_states, vertiport_states = initialize_states_with_time(vehicles, grid_ids, args.vehicles_per_vertiport)

    start = time.time()
    (summary, cumulative_cost, assigned_routes_log, charging_log, rebalance_log,
     vertiport_occupancy_log, vertiport_total_count_log) = run_iterations(
        num_bins=n_bins,
        vehicle_states=vehicle_states,
        vertiport_states=vertiport_states,
        demand_per_bin=demand_per_bin,
        charging_rate=args.charging_rate_per_bin,
        discharge_rate=args.discharge_rate,
        vertiports=grid_ids,
        distance_air=distance_air,
        charging_rate_per_bin=args.charging_rate_per_bin,
        rebalance_interval=args.rebalance_interval,
        rebalance_min_reserve=args.rebalance_min_reserve,
        rebalance_max_idle_cap=(
            args.vehicles_per_vertiport * args.rebalance_max_idle_multiplier
            if args.rebalance_max_idle_multiplier > 0 else None
        ),
        predictive_rebalancing=args.predictive_rebalancing,
        predictive_threshold=args.predictive_threshold,
        predictive_gain=args.predictive_gain,
        predictive_window=args.predictive_window,
        predictive_top_k=args.predictive_top_k,
        predictive_max_drain=args.predictive_max_drain,
    )
    elapsed = time.time() - start
    print(f"simulation finished in {elapsed:.1f}s")

    df = pd.DataFrame(summary)
    df.to_csv(SIM_OUT_DIR / f"time_step_summary_{args.tag}.csv", index=False)
    pd.DataFrame(assigned_routes_log).to_csv(SIM_OUT_DIR / f"assigned_routes_{args.tag}.csv", index=False)
    pd.DataFrame(rebalance_log).to_csv(SIM_OUT_DIR / f"rebalance_log_{args.tag}.csv", index=False)
    pd.DataFrame(vertiport_occupancy_log).to_csv(SIM_OUT_DIR / f"vertiport_occupancy_{args.tag}.csv", index=False)
    pd.DataFrame(vertiport_total_count_log).to_csv(SIM_OUT_DIR / f"vertiport_total_count_{args.tag}.csv", index=False)
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


if __name__ == "__main__":
    main()
