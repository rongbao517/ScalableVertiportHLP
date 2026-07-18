# -*- coding: utf-8 -*-
"""
Validates route_assignment_od_to_vertiports.py's vectorized-numpy argmin
against an actual Gurobi assignment LP, on a small random sample of
(grid_o, grid_d, hour_of_day, day_type) bucket rows -- proves the numpy
shortcut (exact per-pair argmin, valid because there's no capacity constraint
coupling pairs together yet) agrees with what a real solver returns, and
gives a working Gurobi model shape to extend once vertiport/corridor capacity
constraints are added (at which point pairs are no longer independent and
this per-pair argmin stops being valid -- a combined LP/MIP will be required).

Each sampled row gets its own tiny model: 30x30=900 binary "take route (i,j)"
variables, one constraint (exactly one route chosen), objective = minimize
the same access+flight+egress time cost used in the main script, using that
row's own bucket's dynamic ground speed (previously this validator hardcoded
GROUND_SPEED_KMH=25.0, inconsistently with the main script's 15.0 -- both are
now replaced by the same per-bucket predicted-speed lookup both scripts share).
Kept small deliberately to stay well inside gurobipy's free/restricted
license limits (2000 variables/constraints).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUT_DIR = PROJECT_DIR / "outputs"

OD_PAIRS_CSV = DATA_DIR / "full_grid_od_pairs_by_bucket_202504.csv"
SPEED_BUCKET_CSV = DATA_DIR / "predicted_ground_speed_by_bucket.csv"
SITES_CSV = DATA_DIR / "selected_sites_kmeans_K30.csv"
# validates the dynamic-speed run (route_assignment_od_to_vertiports.py's own
# output file, kept separate from the original static-speed result)
ASSIGNMENT_CSV = OUT_DIR / "route_assignment_kmeans30_noselfloop_dynamic_speed.csv"

AIR_SPEED_KMH = 200.0
DWELL_MIN = 5.0
N_SAMPLE = 20
SEED = 0


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def solve_one_pair_gurobi(o_lat, o_lon, d_lat, d_lon, site_lat, site_lon, ground_speed):
    n = len(site_lat)
    access = haversine_km(o_lat, o_lon, site_lat, site_lon) / ground_speed * 60.0 + DWELL_MIN
    egress = haversine_km(d_lat, d_lon, site_lat, site_lon) / ground_speed * 60.0 + DWELL_MIN
    flight = haversine_km(site_lat[:, None], site_lon[:, None], site_lat[None, :], site_lon[None, :]) / AIR_SPEED_KMH * 60.0

    m = gp.Model("route_assignment_single_pair")
    m.Params.OutputFlag = 0
    x = m.addVars(n, n, vtype=GRB.BINARY, name="x")
    m.addConstr(x.sum() == 1, name="exactly_one_route")
    # match route_assignment_od_to_vertiports.py's EXCLUDE_SELF_LOOP=True: a
    # same-site i==j "flight" is a zero-distance leg plus two dwell penalties
    # for nothing, and the main script forbids it via np.fill_diagonal(inf).
    # Gurobi can't take an inf objective coefficient, so forbid it via a
    # constraint on the binary variable instead.
    for i in range(n):
        m.addConstr(x[i, i] == 0, name=f"no_self_loop_{i}")
    m.setObjective(
        gp.quicksum(x[i, j] * (access[i] + flight[i, j] + egress[j]) for i in range(n) for j in range(n)),
        GRB.MINIMIZE,
    )
    m.optimize()
    for i in range(n):
        for j in range(n):
            if x[i, j].X > 0.5:
                return i, j, m.ObjVal
    raise RuntimeError("no route selected")


def main():
    pairs = pd.read_csv(OD_PAIRS_CSV)
    speed_lookup = pd.read_csv(SPEED_BUCKET_CSV).set_index(["hour_of_day", "day_type"])["predicted_speed_kmh"]
    sites = pd.read_csv(SITES_CSV)
    assignment = pd.read_csv(ASSIGNMENT_CSV)
    grid_ids = sites["Grid ID"].to_numpy()
    site_lat = sites["avg_lat"].to_numpy()
    site_lon = sites["avg_lon"].to_numpy()
    id_to_idx = {gid: idx for idx, gid in enumerate(grid_ids)}

    rng = np.random.default_rng(SEED)
    sample_idx = rng.choice(len(pairs), size=N_SAMPLE, replace=False)

    n_match = 0
    for k, idx in enumerate(sample_idx):
        row = pairs.iloc[idx]
        ground_speed = float(speed_lookup.loc[(row["hour_of_day"], row["day_type"])])
        gi, gj, gurobi_cost = solve_one_pair_gurobi(
            row["o_lat"], row["o_lon"], row["d_lat"], row["d_lon"], site_lat, site_lon, ground_speed
        )
        numpy_row = assignment.iloc[idx]
        numpy_i = id_to_idx[numpy_row["takeoff_grid_id"]]
        numpy_j = id_to_idx[numpy_row["landing_grid_id"]]
        numpy_cost = numpy_row["uam_time_min"]

        match = (gi == numpy_i) and (gj == numpy_j) and abs(gurobi_cost - numpy_cost) < 1e-6
        n_match += int(match)
        status = "OK" if match else "MISMATCH"
        print(
            f"[{k+1}/{N_SAMPLE}] pair_idx={idx} hour={row['hour_of_day']} day_type={row['day_type']} "
            f"ground_speed={ground_speed:.2f}  gurobi=(i={gi},j={gj},cost={gurobi_cost:.4f})  "
            f"numpy=(i={numpy_i},j={numpy_j},cost={numpy_cost:.4f})  {status}"
        )

    print("*" * 40)
    print(f"{n_match}/{N_SAMPLE} sampled pairs match exactly between Gurobi and the numpy argmin.")


if __name__ == "__main__":
    main()
