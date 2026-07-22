# -*- coding: utf-8 -*-
"""
column_generation, reconstructed to match how simulation_2.py calls it
(passing run_gurobi_optimization in as a callback, max_iters=1,
max_pq_per_order=3).

Real column generation would iteratively price in new candidate paths
(e.g. multi-hop routes through intermediate vertiports) up to max_iters
rounds, adding at most max_pq_per_order columns per order each round. This
fleet model only ever considers direct single-leg flights between the
vertiport pair the demand is already defined on (no multi-hop routing), so
with max_iters=1 -- the value simulation_2.py actually runs with -- column
generation degenerates to exactly one call to the base LP. Kept as a
separate function (rather than calling run_gurobi_optimization directly)
so the interface matches and it's a one-line change to add real multi-hop
path generation later if needed.
"""


def column_generation(orders, vertiports, vehicle_states, distance_air, discharge_rate,
                       vehicle_count, vehicle_total, run_gurobi_optimization, time_step,
                       max_iters=1, max_pq_per_order=3):
    return run_gurobi_optimization(
        orders=orders,
        vertiports=vertiports,
        distance_air=distance_air,
        vehicle_count=vehicle_count,
        time_step=time_step,
    )
