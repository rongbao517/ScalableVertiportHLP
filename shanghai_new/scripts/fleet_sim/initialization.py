# -*- coding: utf-8 -*-
"""
Fleet/vertiport state initialization, reconstructed to match how
simulation_2.py calls initialize_states_with_time(vehicles, vertiports,
vehicles_per_vertiport). Every vehicle starts fully charged, standing by
at its home vertiport (vehicles are split evenly across vertiports in
order, matching initialize_plane_status_loc's slicing in simulation_2.py).
"""


def initialize_states_with_time(vehicles, vertiports, vehicles_per_vertiport):
    vehicle_states = {}
    vertiport_states = {v: {"avail": 0, "in_service": 0} for v in vertiports}

    for i, vertiport in enumerate(vertiports):
        assigned = vehicles[i * vehicles_per_vertiport: (i + 1) * vehicles_per_vertiport]
        for vid in assigned:
            vehicle_states[vid] = {
                "loc": vertiport,
                "battery": 100.0,
                "in_service": 0,
                "charging": 0,
            }
        vertiport_states[vertiport]["avail"] = len(assigned)

    return vehicle_states, vertiport_states
