# -*- coding: utf-8 -*-
"""
Idle-vehicle repositioning ("车辆再调度"), the piece simulation_2.py has
disabled/commented out (redistribute_vehicles) and this reconstruction
originally skipped.

Why this exists: demand is concentrated at a handful of vertiports (top 5
of 30 account for ~50% of all cross-vertiport trips), but the fleet starts
split evenly across all 30 and only ever moves wherever a *revenue* flight
happens to land. Idle vehicles pile up at cold vertiports while hot
vertiports run dry, and since nothing sends them back, the backlog at hot
vertiports only grows. Tier 1 below periodically sends idle vehicles from
low-backlog vertiports toward vertiports with the largest unmet backlog.

Tier 1 alone created a second failure mode: a vertiport that is a heavy net
*destination* for revenue flights (e.g. grid 783, the single busiest
landing point) almost never shows local backlog -- it keeps getting
refilled by real arrivals -- so tier 1 never flags it as needing help, and
it's rarely picked as a donor either (nearest-donor selection favors
smaller, closer surplus vertiports first). Idle vehicles pooled there
monotonically: ~500 of the 600-vehicle fleet had drifted into that one
vertiport by bin 500. Tier 2 is a hard idle-count cap, independent of
whether a vertiport currently shows backlog: any vertiport sitting above
the cap is forced to drain its excess toward whatever vertiports currently
hold the fewest idle vehicles, leveling the fleet instead of letting one
hub become a one-way sink.

Tier 2 alone turned out to be whack-a-mole, not a fix: tightening the cap
just relocates the pool to whichever vertiport is currently the *next*
biggest net destination (measured directly: at discharge_rate=5%/km,
halving the cap from 2x to 1x fleet-share cut vertiport 931's low-battery
backlog by ~98% but pushed vertiport 882's backlog up ~3x -- total
low-battery vehicle-bin instances across the fleet only fell ~8%). The cap
only reacts to *current* idle count, so it can't tell a vertiport that is
structurally net-inflow (real revenue arrivals keep landing there faster
than departures/rebalancing can drain it) from one that's just transiently
crowded. Tier 0 below fixes that by tracking each vertiport's realized
net flow (arrivals minus departures) over a rolling window and draining
structural vehicle-surplus vertiports (donors) proactively -- before the
hard cap is ever hit -- toward vertiports that are structural vehicle-deficit
sites (recipients: bleeding vehicles out faster than they're refilled),
rather than just "whichever pile is currently smallest" the way Tier 2 does.

Naming note (2026-07-22): earlier versions of this module called the donor
side "sinks" and the recipient side "sources", borrowed from the *passenger*
net-flow perspective (a vertiport is a "sink" for arriving trips, a "source"
for departing ones). That reads backwards once you're thinking about the
*vehicle* being repositioned -- the "sink" was actually giving vehicles away
and the "source" was receiving them. Renamed throughout to donor/recipient,
which matches the vehicle's own direction of travel and Tier 1's
already-correct donor/target naming below.
"""
from collections import deque

from distance_battery import battery_consumption_required
from task_assignment import flight_duration_bins


def new_flow_history():
    """Caller-owned, persisted across bins: {vertiport: deque([net_flow_per_bin, ...])}."""
    return {}


def update_flow_history(flow_history, vertiports, arrivals_this_bin, departures_this_bin, window=30):
    arrivals_this_bin = arrivals_this_bin or {}
    departures_this_bin = departures_this_bin or {}
    for v in vertiports:
        net = arrivals_this_bin.get(v, 0) - departures_this_bin.get(v, 0)
        flow_history.setdefault(v, deque(maxlen=window)).append(net)


def _net_inflow_ema(flow_history, v):
    hist = flow_history.get(v)
    if not hist:
        return 0.0
    return sum(hist) / len(hist)


def _dispatch_reposition(vehicle_states, vertiport_states, vid, start, end, distance,
                          discharge_rate, current_step, vehicle_movements):
    required_battery = battery_consumption_required(distance, discharge_rate)
    st = vehicle_states[vid]
    if st["battery"] < required_battery:
        return False
    st["in_service"] = 1
    st["battery"] -= required_battery
    vehicle_movements[vid] = {
        "start": start, "end": end,
        "arrival_step": current_step + flight_duration_bins(distance),
        "distance": distance,
    }
    vertiport_states[start]["avail"] -= 1
    vertiport_states[start]["in_service"] += 1
    return True


def redistribute_vehicles(vehicle_states, vertiport_states, distance_air, unmet_demand,
                           vehicle_movements, current_step, discharge_rate, min_reserve=1,
                           max_idle_cap=None, flow_history=None, predictive_threshold=1.5,
                           predictive_gain=1.0, predictive_top_k=3, predictive_max_drain=3,
                           predictive_min_history=None, debug=False, predictive_diag_log=None,
                           predictive_donor_max_own_need=0.0):
    idle_by_vertiport = {}
    for vid, st in vehicle_states.items():
        if st["in_service"] == 0:
            idle_by_vertiport.setdefault(st["loc"], []).append(vid)

    moves = []

    # unmet_demand here is already the rebalancing-relevant (non-rounding) real
    # shortfall signal (see run_shanghai_fleet_simulation.py's
    # unmet_demand_for_rebalancing / task_assignment.py's unmet_for_rebalancing).
    # Aggregating it by origin gives each vertiport's own current real need --
    # used below (predictive_donor_max_own_need) to stop Tier 0 from draining a
    # site that has real unmet demand of its own just because it also has a
    # persistent net inflow of *revenue* arrivals. 2026-07-21/22 investigation
    # (grid 783/928/925): these are exactly the vertiports Tier 0 targeted
    # hardest under a wider predictive_max_drain, and their OWN real_unmet got
    # WORSE, not better, as drain intensity increased -- Tier 0's net_inflow
    # signal alone can't distinguish "purely idle surplus" (safe donor, e.g.
    # grid 714/579, own real_unmet ~0 throughout) from "busy hub that also
    # can't fully serve itself" (grid 783/928/925).
    own_real_need = {}
    for s, e, flow in unmet_demand:
        own_real_need[s] = own_real_need.get(s, 0.0) + flow

    # Tier 0: predictive -- vertiports running a persistent net-inflow (real
    # arrivals outpacing departures + rebalancing over the recent rolling
    # window tracked in flow_history, see update_flow_history) get drained
    # (as donors) toward persistent net-outflow vertiports (recipients) before
    # they ever trip the hard idle cap. Opt-in: skipped entirely when
    # flow_history is None, so existing callers that don't pass it get the old
    # Tier 1/2-only behavior.
    #
    # First cut used no top-k/no per-bin cap: with threshold=0.3 nearly half
    # of all 30 vertiports cleared it on any given bin (net flow is a
    # zero-sum quantity across the fleet, so roughly half the stations are
    # "positive" at any moment even with no structural surplus at all), so
    # Tier 0 fired on ~150-200 vehicles/bin -- more of the fleet was shuttling
    # sideways for repositioning than was ever flying revenue trips, and
    # total_met collapsed by ~24% vs. doing nothing. predictive_top_k and
    # predictive_max_drain bound the intervention to the handful of vertiports
    # that are *most* persistently imbalanced, at a trickle rate, instead of
    # treating routine cross-sectional noise as a signal to act on.
    if flow_history is not None:
        all_vertiports = list(vertiport_states.keys())
        min_hist = predictive_min_history if predictive_min_history is not None else 1
        net_inflow = {
            v: (_net_inflow_ema(flow_history, v) if len(flow_history.get(v, [])) >= min_hist else 0.0)
            for v in all_vertiports
        }
        donor_candidates = sorted(
            (v for v in all_vertiports
             if net_inflow[v] > predictive_threshold
             and len(idle_by_vertiport.get(v, [])) > min_reserve
             and own_real_need.get(v, 0.0) <= predictive_donor_max_own_need),
            key=lambda v: -net_inflow[v],
        )[:predictive_top_k]
        recipient_candidates = [v for v in all_vertiports if net_inflow[v] < -predictive_threshold]

        for v in donor_candidates:
            available = len(idle_by_vertiport.get(v, [])) - min_reserve
            raw_requested = max(1, int(round(predictive_gain * net_inflow[v])))
            if available <= 0:
                # 2026-07-21 mechanism-decomposition diagnostic (opt-in, default
                # None -- zero behavior change for existing callers): records why
                # n_drain ended up what it did, so a predictive_* sensitivity sweep
                # can attribute outcomes to threshold/gain/top_k/max_drain instead
                # of just reading off the final service-rate number.
                if predictive_diag_log is not None:
                    predictive_diag_log.append({
                        "t": current_step, "site": v, "net_inflow": net_inflow[v],
                        "raw_requested": raw_requested, "available": available,
                        "n_drain_target": 0, "n_sinks_this_bin": len(donor_candidates),
                        "clipped_by_max_drain": False, "clipped_by_available": True,
                    })
                continue
            n_drain = min(available, predictive_max_drain, raw_requested)
            if predictive_diag_log is not None:
                predictive_diag_log.append({
                    "t": current_step, "site": v, "net_inflow": net_inflow[v],
                    "raw_requested": raw_requested, "available": available,
                    "n_drain_target": n_drain, "n_sinks_this_bin": len(donor_candidates),
                    "clipped_by_max_drain": n_drain == predictive_max_drain and predictive_max_drain < available and predictive_max_drain < raw_requested,
                    "clipped_by_available": n_drain == available and available < predictive_max_drain and available < raw_requested,
                })

            candidate_targets = recipient_candidates if recipient_candidates else all_vertiports
            candidate_targets = sorted((t for t in candidate_targets if t != v),
                                        key=lambda t: distance_air.get((v, t), 1e9))

            sent = 0
            for target in candidate_targets:
                if sent >= n_drain:
                    break
                distance = distance_air.get((v, target), 0.0)
                remaining = n_drain - sent
                for vid in list(idle_by_vertiport.get(v, [])):
                    if remaining <= 0:
                        break
                    if not _dispatch_reposition(vehicle_states, vertiport_states, vid, v, target,
                                                 distance, discharge_rate, current_step, vehicle_movements):
                        continue
                    idle_by_vertiport[v].remove(vid)
                    moves.append({"time_step": current_step, "start": v, "end": target,
                                  "vehicle": vid, "distance": distance, "kind": "predictive"})
                    sent += 1
                    remaining -= 1

    # Tier 1: reactive -- drain toward vertiports with unmet backlog right now.
    shortage = {}
    for s, e, flow in unmet_demand:
        shortage[s] = shortage.get(s, 0.0) + flow

    for target, need in sorted(shortage.items(), key=lambda kv: -kv[1]):
        if need <= 0:
            continue

        donors = []
        for v, vids in idle_by_vertiport.items():
            if v == target or not vids:
                continue
            spare = len(vids) - min_reserve - shortage.get(v, 0.0)
            if spare > 0:
                donors.append((v, spare))
        donors.sort(key=lambda kv: distance_air.get((kv[0], target), 1e9))

        remaining_need = need
        for v, spare in donors:
            if remaining_need <= 0:
                break
            n_send = int(min(spare, remaining_need, len(idle_by_vertiport[v])))
            if n_send <= 0:
                continue
            distance = distance_air.get((v, target), 0.0)
            sent = 0
            for vid in list(idle_by_vertiport[v]):
                if sent >= n_send:
                    break
                if not _dispatch_reposition(vehicle_states, vertiport_states, vid, v, target,
                                             distance, discharge_rate, current_step, vehicle_movements):
                    continue
                idle_by_vertiport[v].remove(vid)
                moves.append({"time_step": current_step, "start": v, "end": target,
                              "vehicle": vid, "distance": distance, "kind": "shortage"})
                sent += 1
            remaining_need -= sent

    # Tier 2: proactive overflow cap -- independent of whether a vertiport
    # currently shows backlog, nothing may hoard more than max_idle_cap idle
    # vehicles. Excess is leveled toward whichever vertiports currently hold
    # the fewest idle vehicles (updated after each single move so the
    # overflow spreads out instead of dumping onto one target).
    if max_idle_cap is not None:
        all_vertiports = list(vertiport_states.keys())
        idle_counts = {v: len(idle_by_vertiport.get(v, [])) for v in all_vertiports}
        overflow_donor_sites = sorted((v for v in all_vertiports if idle_counts[v] > max_idle_cap),
                                       key=lambda v: -idle_counts[v])

        for v in overflow_donor_sites:
            while idle_counts[v] > max_idle_cap and idle_by_vertiport.get(v):
                target = min((t for t in all_vertiports if t != v), key=lambda t: idle_counts[t])
                distance = distance_air.get((v, target), 0.0)
                moved_vid = None
                for vid in idle_by_vertiport[v]:
                    if _dispatch_reposition(vehicle_states, vertiport_states, vid, v, target,
                                             distance, discharge_rate, current_step, vehicle_movements):
                        moved_vid = vid
                        break
                if moved_vid is None:
                    break  # no idle vehicle here has enough battery left
                idle_by_vertiport[v].remove(moved_vid)
                idle_counts[v] -= 1
                idle_counts[target] += 1
                moves.append({"time_step": current_step, "start": v, "end": target,
                              "vehicle": moved_vid, "distance": distance, "kind": "overflow"})

    if debug and moves:
        n_predictive = sum(1 for m in moves if m["kind"] == "predictive")
        n_shortage = sum(1 for m in moves if m["kind"] == "shortage")
        n_overflow = sum(1 for m in moves if m["kind"] == "overflow")
        print(f"[REBALANCE] step={current_step} moved {len(moves)} "
              f"(predictive={n_predictive} shortage={n_shortage} overflow={n_overflow})")
    return moves
