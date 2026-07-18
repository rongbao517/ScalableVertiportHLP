# -*- coding: utf-8 -*-
"""
Per-bin capacitated dispatch-planning LP, reconstructed to match how
simulation_2.py calls run_gurobi_optimization (via column_generation).

For this bin's aggregated demand (this bin's real orders + unmet carried
over from the previous bin), decide how many trips per (takeoff, landing)
vertiport pair to plan for, subject to each vertiport's vehicle count as a
capacity ceiling on how many it can dispatch this bin:

    maximize   sum_{s,e} x[s,e]
    subject to x[s,e] <= demand[s,e]                      for every pair
               sum_e x[s,e] <= vehicle_count.get(s, 0)     for every origin s
               x[s,e] >= 0

vehicle_count is keyed by each vehicle's current `loc` regardless of
in_service status (matching the Counter(vs["loc"] ...) built in the driver)
-- this makes the LP's capacity signal optimistic (it doesn't know which of
those vehicles are mid-flight or under-charged). That's intentional: the
real constraint is enforced afterwards in task_assignment.time_step_path_assignment,
which only dispatches vehicles that are actually standby with enough
battery. This LP's job is just to decide the target flow per route/pair
within a coarse per-vertiport capacity budget; the physical dispatch step
is where any remaining shortfall (due to individual battery levels) shows up.
"""
from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB


def run_gurobi_optimization(orders, vertiports, distance_air, vehicle_count, time_step=None):
    demand = defaultdict(float)
    for s, e, f in orders:
        if s != e and f > 0:
            demand[(s, e)] += f

    if not demand:
        return []

    m = gp.Model("fleet_dispatch_plan")
    m.Params.OutputFlag = 0
    # This LP typically has multiple tied-optimal vertex solutions (many
    # (s,e) pairs competing for the same vehicle-count capacity bound with
    # equal objective coefficients), and Gurobi's default Method=-1
    # ("concurrent": several algorithms race, whichever finishes first wins)
    # picks a different tied vertex depending on thread-scheduling timing --
    # confirmed empirically: three back-to-back runs of this same simulation
    # with byte-identical inputs produced total_met of 25784/28947/31509
    # (>20% spread) purely from this. Threads=1 + a fixed deterministic
    # Method removes the race so results reproduce exactly run-to-run --
    # required for any sensitivity comparison (like a DWELL_MIN sweep) to be
    # trustworthy at all.
    m.Params.Threads = 1
    m.Params.Method = 1  # dual simplex, deterministic
    m.Params.Seed = 0

    pairs = list(demand.keys())
    x = m.addVars(pairs, lb=0.0, name="x")

    for (s, e) in pairs:
        m.addConstr(x[s, e] <= demand[(s, e)], name=f"cap_demand_{s}_{e}")

    # sorted (not a bare set comprehension): CPython set iteration order
    # depends on str hash values, which are randomized per-process
    # (PYTHONHASHSEED) unless fixed -- that reordered which capacity
    # constraint gets added to the LP first across runs, which was the
    # remaining source of tiny residual tie-break differences even after
    # Threads=1/Method=1/Seed=0 above.
    origins = sorted({s for (s, _e) in pairs})
    for s in origins:
        m.addConstr(
            gp.quicksum(x[s, e] for (s2, e) in pairs if s2 == s) <= vehicle_count.get(s, 0),
            name=f"cap_vehicles_{s}",
        )

    m.setObjective(gp.quicksum(x[p] for p in pairs), GRB.MAXIMIZE)
    m.optimize()

    if m.Status != GRB.OPTIMAL:
        return []

    results = []
    for (s, e) in pairs:
        flow = x[s, e].X
        if flow > 1e-6:
            results.append({
                "takeoff": s,
                "landing": e,
                "order_start": s,
                "order_end": e,
                "flow": flow,
                "distance": distance_air.get((s, e), 0.0),
            })
    return results
