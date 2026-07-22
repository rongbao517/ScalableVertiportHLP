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

Ground access/egress time (previously modeled nowhere in fleet_sim -- a
passenger's order was treated as appearing already AT the departure
vertiport, and the vehicle's occupation window covered only the air leg)
is now folded into trip_duration_bins() below, using the same dynamic
per-(hour_of_day, day_type)-bucket ground speed predicted by
train_shanghai_speed_gru.py / predict_speed_by_bucket.py that
route_assignment_od_to_vertiports.py already uses for the grid-to-vertiport
routing problem. flight_duration_bins() (air time only) is kept as-is and
still used for rebalancing.py's empty repositioning flights, which carry no
passenger and so have no ground leg to model.
"""
from collections import Counter

from distance_battery import calculate_distance, battery_consumption_required

AIR_SPEED_KMH = 200.0
BIN_MINUTES = 30.0

# fleet_sim's demand tensor (shanghai_od_kmeans30_30min_1channel.npz) only
# counts trips whose origin AND destination both fall inside the vertiport's
# OWN 0.01-degree grid cell (see extract_shanghai_od_matrix_kmeans30.py's
# build_site_lookup) -- so every passenger's real ground access distance is
# small and roughly fixed, not the multi-km distances
# route_assignment_od_to_vertiports.py handles for the full 1676-grid case.
# ACCESS_EGRESS_KM is the expected distance from a uniformly random point
# inside that cell (~1.11km x ~0.95km at Shanghai's latitude) to the site's
# own coordinate, estimated by Monte Carlo (2e6 samples): ~0.396km, rounded.
ACCESS_EGRESS_KM = 0.40
# same boarding/security/taxi overhead convention as
# route_assignment_od_to_vertiports.py's DWELL_MIN, applied per ground leg.
# No ground-truth boarding-time data exists in this project -- kept as a
# module-level default (matching route_assignment_od_to_vertiports.py) but
# also accepted as a trip_duration_bins() argument so it can be swept.
DWELL_MIN = 5.0


def flight_duration_bins(distance_km):
    air_time_min = distance_km / AIR_SPEED_KMH * 60.0
    return max(1, int(-(-air_time_min // BIN_MINUTES)))  # ceil division


def trip_duration_bins(distance_km, ground_speed_kmh, dwell_min=DWELL_MIN,
                        access_km=ACCESS_EGRESS_KM, egress_km=ACCESS_EGRESS_KM):
    """Passenger-carrying dispatch duration: ground access + air + ground egress,
    unlike flight_duration_bins() which is air-time-only (used for empty
    repositioning flights that have no passenger and thus no ground leg).
    access_km/egress_km default to the small intra-cell approximation but
    can be overridden with real per-(takeoff,landing,bucket) distances (see
    build_dynspeed_full_demand.py) when using the full-grid routed demand."""
    access_time_min = access_km / ground_speed_kmh * 60.0 + dwell_min
    egress_time_min = egress_km / ground_speed_kmh * 60.0 + dwell_min
    air_time_min = distance_km / AIR_SPEED_KMH * 60.0
    total_min = access_time_min + air_time_min + egress_time_min
    return max(1, int(-(-total_min // BIN_MINUTES)))  # ceil division


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
                               vehicle_movements, current_step, ground_speed_kmh, dwell_min=DWELL_MIN,
                               access_egress_lookup=None, debug=False):
    unmet_after_assignment = []
    # Rebalancing-relevant shortfall only: away_flying + insufficient_battery,
    # EXCLUDING rounding_jitter. Fix for the 2026-07-20 grid-433 investigation:
    # rounding_jitter (int(requested) flooring a small fractional per-bin,
    # per-pair demand to 0) is structurally unaddressable by adding more idle
    # vehicles -- int(0.3) is 0 no matter how many candidates exist -- so
    # feeding it into the reactive "shortage" rebalancer just chases a phantom
    # signal, permanently over-supplying low-demand sites (idle count pinned
    # at the hard cap) without ever improving their own assignment ratio.
    # unmet_after_assignment itself is untouched (still carries the full
    # shortfall, jitter included, for correct fractional-demand carry-forward
    # to the next bin and for per-site reporting) -- only the rebalancer's
    # input is filtered.
    unmet_for_rebalancing = []
    assigned_routes = []
    launched_ids = []
    operational_served = []
    # Reason-tagged shortfall totals for this bin, purely diagnostic (doesn't
    # affect dispatch): of a path's shortfall (requested - n_dispatch), how
    # much is because the vehicles Gurobi's LP counted as "at s" (loc==s,
    # regardless of in_service -- see gurobi_optimization.py) are actually
    # mid-flight elsewhere right now (away_flying) vs. physically idle at s
    # but under-charged for this route's distance (insufficient_battery).
    # idle_at_s <= vehicle_count_at_s (candidates ignore in-flight vehicles)
    # and idle_with_battery <= idle_at_s (battery filter is a strict subset),
    # so requested - min(requested, idle_with_battery) always decomposes
    # exactly into these two non-negative, non-overlapping pieces -- plus a
    # third, "rounding_jitter": n_dispatch below floors requested to a whole
    # vehicle count (int(requested)) before capping by candidates, since you
    # can't physically dispatch a fractional vehicle, but `requested` itself
    # is usually fractional (bucket-averaged demand, e.g. 344.43 "trips" this
    # bin) -- so even with an abundant candidate surplus, up to ~1 unit of
    # "shortfall" per (s,e) pair per bin is just frac(requested) carrying
    # over to next bin's total, not a real fleet/battery constraint. This is
    # bounded (never > 1 per pair; it's frac() of a running accumulator, not
    # a growing backlog) but can dominate reported unmet_demand when demand
    # per pair per bin is itself small (e.g. filtered/lower-volume scenarios).
    shortfall_reasons = {"away_flying": 0.0, "insufficient_battery": 0.0, "rounding_jitter": 0.0}
    # Diagnostic-only: same three reasons, but keyed by origin vertiport s --
    # added to trace whether a specific site's unmet demand is dominated by
    # a genuine headcount problem or by away_flying/insufficient_battery
    # (see 2026-07-20 investigation of grid 433's high-idle/low-service anomaly).
    shortfall_by_site = {}

    for path in gurobi_results:
        s = path["takeoff"]
        e = path["landing"]
        requested = float(path["flow"])
        distance = path.get("distance", calculate_distance(s, e))
        required_battery = battery_consumption_required(distance, discharge_rate)

        idle_at_s = sum(1 for state in vehicle_states.values() if state["loc"] == s and state["in_service"] == 0)
        candidates = [
            vid for vid, state in vehicle_states.items()
            if state["loc"] == s and state["in_service"] == 0 and state["battery"] >= required_battery
        ]
        candidates.sort(key=lambda vid: vehicle_states[vid]["battery"], reverse=True)

        n_dispatch = min(int(requested), len(candidates))
        dispatched = candidates[:n_dispatch]

        away_flying = max(0.0, requested - idle_at_s)
        insufficient_battery = max(0.0, min(requested, idle_at_s) - len(candidates))
        rounding_jitter = min(requested, len(candidates)) - min(int(requested), len(candidates))
        shortfall_reasons["away_flying"] += away_flying
        shortfall_reasons["insufficient_battery"] += insufficient_battery
        shortfall_reasons["rounding_jitter"] += rounding_jitter

        site_acc = shortfall_by_site.setdefault(s, {"away_flying": 0.0, "insufficient_battery": 0.0, "rounding_jitter": 0.0, "requested": 0.0, "idle_at_s": 0.0})
        site_acc["away_flying"] += away_flying
        site_acc["insufficient_battery"] += insufficient_battery
        site_acc["rounding_jitter"] += rounding_jitter
        site_acc["requested"] += requested
        site_acc["idle_at_s"] += idle_at_s

        access_km, egress_km = (
            access_egress_lookup.get((s, e), (ACCESS_EGRESS_KM, ACCESS_EGRESS_KM))
            if access_egress_lookup is not None else (ACCESS_EGRESS_KM, ACCESS_EGRESS_KM)
        )

        for vid in dispatched:
            vehicle_states[vid]["in_service"] = 1
            vehicle_states[vid]["battery"] -= required_battery
            vehicle_movements[vid] = {
                "start": s, "end": e,
                "arrival_step": current_step + trip_duration_bins(
                    distance, ground_speed_kmh, dwell_min, access_km, egress_km),
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

        rebalance_relevant_shortfall = away_flying + insufficient_battery
        if rebalance_relevant_shortfall > 0:
            unmet_for_rebalancing.append((s, e, rebalance_relevant_shortfall))

        # continuous_served: what would have been dispatched if fractional vehicles were
        # allowed (only real constraints -- idle presence + battery -- apply, no integer
        # floor). Used to build a second, parallel wait-time ledger that excludes the
        # "waiting for enough fractional demand to accumulate to one whole vehicle" effect,
        # isolating genuine capacity/battery-driven waiting from this rounding artifact.
        continuous_served = min(requested, len(candidates))
        if continuous_served > 0:
            operational_served.append({"start": s, "end": e, "flow": continuous_served})

        if debug and shortfall > 0:
            print(f"[SHORTFALL] {s}->{e} requested={requested} dispatched={n_dispatch} shortfall={shortfall}")

    return unmet_after_assignment, assigned_routes, launched_ids, shortfall_reasons, operational_served, shortfall_by_site, unmet_for_rebalancing
