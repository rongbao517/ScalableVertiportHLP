# -*- coding: utf-8 -*-
"""
Physical dispatch and arrival handling, reconstructed to match how
simulation_2.py calls update_arrivals(...) and time_step_path_assignment(...).

Each simulation tick == one real 30-min bin (BIN_MINUTES), matching the
real demand data's granularity (shanghai_od_kmeans30_30min_1channel.npz).
A flight's duration in bins = ceil(air_time_min / BIN_MINUTES), so most
inter-vertiport hops (a few minutes at AIR_SPEED_KMH) land within the same
or next bin -- consistent with the flight-time assumptions used earlier in
route_assignment_od_to_vertiports.py (AIR_SPEED_KMH=200).

Gurobi's per-bin plan (run_gurobi_optimization) already respects a coarse
"how many vehicles are standing by at this vertiport" count, but that count
ignores individual battery levels. This module is where that gap surfaces:
a specific standby vehicle might be too low on charge for this route's
distance even though the LP assumed the vertiport had capacity, producing
additional shortfall beyond what Gurobi predicted.
"""
from collections import Counter

from distance_battery import calculate_distance, battery_consumption_required

AIR_SPEED_KMH = 200.0
BIN_MINUTES = 30.0


def flight_duration_bins(distance_km):
    air_time_min = distance_km / AIR_SPEED_KMH * 60.0
    return max(1, int(-(-air_time_min // BIN_MINUTES)))  # ceil division


def update_arrivals(vehicle_states, vertiport_states, vehicle_movements, current_step, debug=False):
    """Lands due vehicles; returns a Counter of {vertiport: n_arrivals} this bin so
    callers (e.g. rebalancing's predictive tier) can track which vertiports are
    persistently net destinations without re-deriving it from vehicle_movements."""
    landed = [vid for vid, mv in vehicle_movements.items() if mv["arrival_step"] <= current_step]
    arrivals_this_bin = Counter()
    for vid in landed:
        mv = vehicle_movements.pop(vid)
        vehicle_states[vid]["loc"] = mv["end"]
        vehicle_states[vid]["in_service"] = 0
        if vertiport_states[mv["start"]]["in_service"] > 0:
            vertiport_states[mv["start"]]["in_service"] -= 1
        vertiport_states[mv["end"]]["avail"] += 1
        arrivals_this_bin[mv["end"]] += 1
        if debug:
            print(f"[ARRIVE] {vid} landed at {mv['end']} (from {mv['start']}, step={current_step})")
    return arrivals_this_bin


def time_step_path_assignment(gurobi_results, vehicle_states, vertiport_states, discharge_rate,
                               vehicle_movements, current_step, debug=False):
    unmet_after_assignment = []
    assigned_routes = []
    launched_ids = []

    for path in gurobi_results:
        s = path["takeoff"]
        e = path["landing"]
        requested = float(path["flow"])
        distance = path.get("distance", calculate_distance(s, e))
        required_battery = battery_consumption_required(distance, discharge_rate)

        candidates = [
            vid for vid, state in vehicle_states.items()
            if state["loc"] == s and state["in_service"] == 0 and state["battery"] >= required_battery
        ]
        candidates.sort(key=lambda vid: vehicle_states[vid]["battery"], reverse=True)

        n_dispatch = min(int(requested), len(candidates))
        dispatched = candidates[:n_dispatch]

        for vid in dispatched:
            vehicle_states[vid]["in_service"] = 1
            vehicle_states[vid]["battery"] -= required_battery
            vehicle_movements[vid] = {
                "start": s, "end": e,
                "arrival_step": current_step + flight_duration_bins(distance),
                "distance": distance,
            }
            vertiport_states[s]["avail"] -= 1
            vertiport_states[s]["in_service"] += 1
            launched_ids.append(vid)

        if n_dispatch > 0:
            assigned_routes.append({"start": s, "end": e, "flow": n_dispatch, "distance": distance})

        shortfall = requested - n_dispatch
        if shortfall > 0:
            unmet_after_assignment.append((s, e, shortfall))

        if debug and shortfall > 0:
            print(f"[SHORTFALL] {s}->{e} requested={requested} dispatched={n_dispatch} shortfall={shortfall}")

    return unmet_after_assignment, assigned_routes, launched_ids
