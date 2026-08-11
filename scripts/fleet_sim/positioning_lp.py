# -*- coding: utf-8 -*-
"""
Rolling-horizon predictive positioning LP (2026-07-22), replacing the
net_inflow-based Tier 0 in rebalancing.py. See positioning_lp_offline_bound.py
for the "clairvoyant" full-horizon upper-bound check that motivated this
(R=1.0000 theoretical ceiling vs R=0.81-0.82 actually achieved by Tier 0/1/2 at
fleet=12000 -- confirms the shortfall is a positioning problem, not a capacity
one, so this replaces the reactive/heuristic signal with a real optimization).

Design (as agreed):
  - Solved once per bin, AFTER the current bin's real passenger dispatch has
    already happened (task_assignment.py's time_step_path_assignment) -- so
    the currently-idle count per site already reflects this bin's revenue
    departures; the LP only ever decides EMPTY (repositioning) moves.
  - Plans H bins ahead using KNOWN future demand (the same 48-bucket lookup
    table every other part of this pipeline uses -- no forecasting involved),
    but only EXECUTES the current bin's r[v,w,t] decision. Next bin, the whole
    thing is re-solved from the real (now-updated) vehicle state -- standard
    receding-horizon / MPC pattern.
  - Future bins' hypothetical passenger service (q[o,d,tau], tau>t) is a pure
    accounting device for projecting future inventory -- it is NEVER executed;
    real future dispatch is decided later, bin by bin, by the existing
    single-bin module exactly as today.
  - Arrivals already in flight (vehicle_movements, from earlier bins' revenue
    or repositioning dispatch) are exogenous, known constants for this solve,
    not decision variables -- this is what makes the rolling model different
    from (and much smaller than) the offline full-horizon model.
  - Passenger-trip arrivals lag by trip_duration_bins (ground access + air +
    ground egress + dwell); repositioning-leg arrivals lag by
    flight_duration_bins (air time only) -- matches task_assignment.py's own
    dispatch and rebalancing.py's own _dispatch_reposition exactly, so the
    LP's inventory projection is consistent with how the simulation actually
    evolves state.
  - v1: continuous LP (no per-vehicle battery bucketing, no integer vars).
    Execution maps the LP's aggregate r[v,w,t] plan onto real vehicles using
    the SAME battery-descending selection rule _dispatch_reposition/
    redistribute_vehicles already use elsewhere in this codebase (prioritize
    highest-battery vehicles for the route). Tracks planned vs. actually-
    executed count (some planned moves may fail if too few candidates have
    enough charge) so the plan/execution gap can be measured empirically
    before deciding whether battery-bucketed variables are worth adding.
  - Solved with scipy.optimize.linprog(method="highs") -- NOT gurobipy, whose
    free pip license here caps at 2000 variables/constraints (see
    positioning_lp_offline_bound.py's docstring); this rolling model is small
    (roughly 15-25k variables for H up to ~12) and HiGHS handles it in well
    under a second per solve.
"""
from collections import Counter

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from distance_battery import battery_consumption_required
from task_assignment import flight_duration_bins, trip_duration_bins, ACCESS_EGRESS_KM, DWELL_MIN
from rebalancing import _dispatch_reposition


def _exogenous_inbound(vehicle_movements, window_start, window_end):
    """Arrivals already scheduled by past dispatch (revenue or repositioning),
    landing within [window_start, window_end] -- fixed facts for this solve,
    not decisions. Keyed by (site, arrival_bin)."""
    inbound = Counter()
    for mv in vehicle_movements.values():
        a = mv["arrival_step"]
        if window_start <= a <= window_end:
            inbound[(mv["end"], a)] += 1
    return inbound


def plan_and_execute_positioning(vehicle_states, vertiport_states, vehicle_movements,
                                  grid_ids, distance_air, current_step, horizon,
                                  demand_window, access_egress_window, ground_speed_window,
                                  discharge_rate, min_reserve=1, lam=0.001, diag=None):
    """demand_window / access_egress_window / ground_speed_window: slices of
    the caller's already-computed per-bin arrays, demand_window[0] == this
    bin's demand (used only for future-bin projection, i.e. indices 1..H-1;
    index 0 is unused since this bin's real dispatch already happened),
    length == min(horizon, bins remaining in the run).
    Returns the list of executed moves (same shape as other tiers' moves)."""
    H = len(demand_window)
    if H == 0:
        return []

    idle_by_vertiport = {}
    for vid, st in vehicle_states.items():
        if st["in_service"] == 0:
            idle_by_vertiport.setdefault(st["loc"], []).append(vid)
    I_current = {s: len(idle_by_vertiport.get(s, [])) for s in grid_ids}

    exogenous_inbound = _exogenous_inbound(vehicle_movements, current_step + 1, current_step + H)

    # ---- enumerate variables ----
    # r[v,w,tau] for tau = current_step .. current_step+H-1 (tau=current_step is the ONLY executed bin)
    # q[o,d,tau] for tau = current_step+1 .. current_step+H-1 (future, projection-only)
    # I[s,tau]   for tau = current_step+1 .. current_step+H
    r_idx, q_idx, I_idx = {}, {}, {}
    D, trip_lag = {}, {}
    col = 0
    for h in range(H):
        tau = current_step + h
        for v in grid_ids:
            for w in grid_ids:
                if v == w:
                    continue
                r_idx[(v, w, tau)] = col
                col += 1
    for h in range(1, H):
        tau = current_step + h
        for row in demand_window[h]:
            o, d, flow = row["start"], row["end"], float(row["flow"])
            if flow <= 0:
                continue
            access_km, egress_km = access_egress_window[h].get((o, d), (ACCESS_EGRESS_KM, ACCESS_EGRESS_KM))
            lag = trip_duration_bins(distance_air.get((o, d), 0.0), ground_speed_window[h],
                                      DWELL_MIN, access_km, egress_km)
            trip_lag[(o, d, tau)] = lag
            D[(o, d, tau)] = flow
            q_idx[(o, d, tau)] = col
            col += 1
    for s in grid_ids:
        for h in range(1, H + 1):
            tau = current_step + h
            I_idx[(s, tau)] = col
            col += 1
    n_vars = col

    lb = np.zeros(n_vars)
    ub = np.full(n_vars, np.inf)
    for (o, d, tau), c in q_idx.items():
        ub[c] = D[(o, d, tau)]

    obj = np.zeros(n_vars)
    for (v, w, tau), c in r_idx.items():
        obj[c] = lam * distance_air.get((v, w), 0.0)
    for c in q_idx.values():
        obj[c] = -1.0

    r_out_by_site_bin, q_out_by_site_bin = {}, {}
    for (v, w, tau), c in r_idx.items():
        r_out_by_site_bin.setdefault((v, tau), []).append(c)
    for (o, d, tau), c in q_idx.items():
        q_out_by_site_bin.setdefault((o, tau), []).append(c)

    r_in_by_site_bin, q_in_by_site_bin = {}, {}
    flight_lag_cache = {}
    for (v, w, tau), c in r_idx.items():
        if (v, w) not in flight_lag_cache:
            flight_lag_cache[(v, w)] = flight_duration_bins(distance_air.get((v, w), 0.0))
        arrive = tau + flight_lag_cache[(v, w)]
        if arrive <= current_step + H:
            r_in_by_site_bin.setdefault((w, arrive), []).append(c)
    for (o, d, tau), c in q_idx.items():
        arrive = tau + trip_lag[(o, d, tau)]
        if arrive <= current_step + H:
            q_in_by_site_bin.setdefault((d, arrive), []).append(c)

    ub_rows, ub_cols, ub_data, b_ub = [], [], [], []
    eq_rows, eq_cols, eq_data, b_eq = [], [], [], []
    row = 0

    # current bin (tau=current_step): empty cap only, I is the real constant I_current
    for s in grid_ids:
        r_out = r_out_by_site_bin.get((s, current_step), [])
        for c in r_out:
            ub_rows.append(row); ub_cols.append(c); ub_data.append(1.0)
        b_ub.append(float(I_current[s] - min_reserve))
        row += 1

    # future bins (tau=current_step+1 .. current_step+H-1): revenue cap + empty cap
    for h in range(1, H):
        tau = current_step + h
        for s in grid_ids:
            q_out = q_out_by_site_bin.get((s, tau), [])
            r_out = r_out_by_site_bin.get((s, tau), [])
            Is = I_idx[(s, tau)]
            for c in q_out:
                ub_rows.append(row); ub_cols.append(c); ub_data.append(1.0)
            ub_rows.append(row); ub_cols.append(Is); ub_data.append(-1.0)
            b_ub.append(0.0)
            row += 1

            for c in r_out:
                ub_rows.append(row); ub_cols.append(c); ub_data.append(1.0)
            for c in q_out:
                ub_rows.append(row); ub_cols.append(c); ub_data.append(1.0)
            ub_rows.append(row); ub_cols.append(Is); ub_data.append(-1.0)
            b_ub.append(-float(min_reserve))
            row += 1

    # inventory recursion, for tau = current_step+1 .. current_step+H
    eqrow = 0
    for h in range(1, H + 1):
        tau = current_step + h
        prev_tau = tau - 1
        for s in grid_ids:
            q_in = q_in_by_site_bin.get((s, tau), [])
            r_in = r_in_by_site_bin.get((s, tau), [])
            exo = exogenous_inbound.get((s, tau), 0)
            eq_rows.append(eqrow); eq_cols.append(I_idx[(s, tau)]); eq_data.append(1.0)
            if prev_tau == current_step:
                prev_out_r = r_out_by_site_bin.get((s, prev_tau), [])
                for c in prev_out_r:
                    eq_rows.append(eqrow); eq_cols.append(c); eq_data.append(1.0)
                for c in q_in:
                    eq_rows.append(eqrow); eq_cols.append(c); eq_data.append(-1.0)
                for c in r_in:
                    eq_rows.append(eqrow); eq_cols.append(c); eq_data.append(-1.0)
                b_eq.append(float(I_current[s]) + float(exo))
            else:
                eq_cols_here = []
                eq_rows.append(eqrow); eq_cols.append(I_idx[(s, prev_tau)]); eq_data.append(-1.0)
                prev_out_q = q_out_by_site_bin.get((s, prev_tau), [])
                prev_out_r = r_out_by_site_bin.get((s, prev_tau), [])
                for c in prev_out_q:
                    eq_rows.append(eqrow); eq_cols.append(c); eq_data.append(1.0)
                for c in prev_out_r:
                    eq_rows.append(eqrow); eq_cols.append(c); eq_data.append(1.0)
                for c in q_in:
                    eq_rows.append(eqrow); eq_cols.append(c); eq_data.append(-1.0)
                for c in r_in:
                    eq_rows.append(eqrow); eq_cols.append(c); eq_data.append(-1.0)
                b_eq.append(float(exo))
            eqrow += 1

    A_ub = sparse.csr_matrix((ub_data, (ub_rows, ub_cols)), shape=(row, n_vars))
    A_eq = sparse.csr_matrix((eq_data, (eq_rows, eq_cols)), shape=(eqrow, n_vars))

    result = linprog(obj, A_ub=A_ub, b_ub=np.array(b_ub), A_eq=A_eq, b_eq=np.array(b_eq),
                      bounds=list(zip(lb, ub)), method="highs", options={"presolve": True})

    moves = []
    # planned_total_raw: sum of the LP's continuous solution (informational
    # only -- NOT a valid denominator for an execution rate, since summing
    # per-pair round() results can drift above OR below the continuous sum).
    # planned_total_rounded: sum of the actual per-pair integer targets
    # (n_planned_int) that execution was asked to hit -- since each pair's
    # `sent` is capped by its own n_planned_int (the `if sent >= n_planned_int:
    # break` below), executed_total can never exceed this sum. This is the
    # correct denominator: exec_rate = executed_total / planned_total_rounded
    # is mathematically guaranteed <= 1.0.
    planned_total_raw = 0.0
    planned_total_rounded = 0
    executed_total = 0
    current_plan_values = []
    if result.success:
        for (v, w, tau), c in r_idx.items():
            if tau != current_step:
                continue
            planned = result.x[c]
            current_plan_values.append(float(planned))
            if planned < 0.5:
                continue
            planned_total_raw += planned
            n_planned_int = int(round(planned))
            planned_total_rounded += n_planned_int
            distance = distance_air.get((v, w), 0.0)
            candidates = sorted(idle_by_vertiport.get(v, []),
                                 key=lambda vid: vehicle_states[vid]["battery"], reverse=True)
            sent = 0
            for vid in candidates:
                if sent >= n_planned_int:
                    break
                if not _dispatch_reposition(vehicle_states, vertiport_states, vid, v, w,
                                             distance, discharge_rate, current_step, vehicle_movements):
                    continue
                idle_by_vertiport[v].remove(vid)
                moves.append({"time_step": current_step, "start": v, "end": w,
                              "vehicle": vid, "distance": distance, "kind": "positioning"})
                sent += 1
                executed_total += 1

    if diag is not None:
        diag.append({
            "t": current_step, "H": H, "solve_status": result.status,
            "planned_total_raw": planned_total_raw,
            "planned_total_rounded": planned_total_rounded,
            "executed_total": executed_total,
            "n_current_lp_arcs": len(current_plan_values),
            "n_near_half_arcs": sum(0.45 <= value <= 0.55 for value in current_plan_values),
            "near_half_flow": sum(value for value in current_plan_values if 0.45 <= value <= 0.55),
            "n_moves_planned_pairs": sum(1 for (v, w, tau), c in r_idx.items()
                                          if tau == current_step and result.success and result.x[c] >= 0.5),
        })
    return moves
