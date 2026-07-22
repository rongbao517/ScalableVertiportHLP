# -*- coding: utf-8 -*-
"""
Distance and battery-consumption helpers for the Shanghai fleet simulation.

Reconstructed to match how simulation_2.py's distance_battery.py is called
(calculate_distance, battery_consumption_required, set_distance_data) -- the
original module isn't available in this environment, so this is a fresh,
documented implementation of the same contract rather than a byte-identical
port.

battery_consumption_required uses a simple linear model: battery percent
consumed = distance_km * discharge_rate. discharge_rate=1.0 (%/km) matches
the example sweep in simulation_2.py's __main__ and keeps a single flight
(sites are up to ~50km apart) from draining more than half the battery.
"""
from pathlib import Path

import numpy as np
import pandas as pd

_DISTANCE_MAP = {}


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def set_distance_data(distance_file):
    """Load a square vertiport-by-vertiport distance matrix (km) keyed by grid id string."""
    global _DISTANCE_MAP
    matrix = pd.read_csv(distance_file, index_col=0)
    matrix.index = matrix.index.astype(str)
    matrix.columns = matrix.columns.astype(str)
    _DISTANCE_MAP = {
        (a, b): float(matrix.loc[a, b])
        for a in matrix.index
        for b in matrix.columns
        if a != b
    }
    return _DISTANCE_MAP


def calculate_distance(a, b):
    if a == b:
        return 0.0
    return _DISTANCE_MAP[(str(a), str(b))]


def battery_consumption_required(distance_km, discharge_rate):
    return distance_km * discharge_rate
