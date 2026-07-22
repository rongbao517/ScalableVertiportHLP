# -*- coding: utf-8 -*-
"""
Cost/coverage helpers, reconstructed to match how simulation_2.py calls
calculate_coverage_rate, calculate_cost, update_demand_chart.
"""


def calculate_cost(assigned_routes, cost_per_distance, distance_map=None):
    return sum(route["flow"] * route["distance"] * cost_per_distance for route in assigned_routes)


def calculate_coverage_rate(met_demand, unmet_demand):
    total = met_demand + unmet_demand
    return met_demand / total if total > 0 else 0.0


def update_demand_chart(*args, **kwargs):
    """Not used by the Shanghai driver's active code path; kept as a no-op stub."""
    return None
