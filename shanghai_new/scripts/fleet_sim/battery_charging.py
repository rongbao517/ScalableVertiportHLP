# -*- coding: utf-8 -*-
"""
Charging and post-arrival bookkeeping, reconstructed to match how
simulation_2.py calls charging_and_battery_update / restore_vehicle_states /
export_charging_log.

Every standby vehicle (in_service == 0) not already at 100% charges by
`charging_rate` percent per `time_interval` minutes elapsed this tick --
a plug-in-while-idle assumption (no separate charging queue/slot limit).
"""
import pandas as pd


def charging_and_battery_update(vehicle_states, time_interval, charging_rate, current_step, charging_tracker):
    for vid, state in vehicle_states.items():
        if state["in_service"] == 0 and state["battery"] < 100:
            gained = charging_rate
            state["battery"] = min(100.0, state["battery"] + gained)
            state["charging"] = 1 if state["battery"] < 100 else 0
            charging_tracker.setdefault(vid, []).append(
                {"time_step": current_step, "battery_after_charge": state["battery"]}
            )
        else:
            state["charging"] = 0


def restore_vehicle_states(vehicle_states, vehicle_movements):
    """
    Consistency pass after update_arrivals: any vehicle no longer present in
    vehicle_movements (i.e. it landed) must have in_service == 0. Vehicles
    still in vehicle_movements are mid-flight and stay in_service == 1.
    """
    for vid, state in vehicle_states.items():
        if vid not in vehicle_movements:
            state["in_service"] = 0


def export_charging_log(charging_tracker, path="charging_log.csv"):
    rows = []
    for vid, entries in charging_tracker.items():
        for e in entries:
            rows.append({"vehicle_id": vid, **e})
    pd.DataFrame(rows).to_csv(path, index=False)
