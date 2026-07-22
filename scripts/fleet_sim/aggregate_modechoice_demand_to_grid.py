# -*- coding: utf-8 -*-
"""
Step 5 of the "predict -> site -> UAM cost -> mode choice -> update demand ->
re-site" closed loop: aggregates compute_mode_choice_demand.py's per-(grid_o,
grid_d,bucket) mode-choice-weighted trip_count back to a per-grid-cell demand
weight, in the same schema as real_demand_all_grids.csv /
naive_predicted_demand_all_grids.csv, so it can be fed straight into
run_site_selection_kmeans_K30.py --demand-csv for a second-pass site
selection based on UAM-ADOPTING demand rather than raw ground demand.

o_lat/o_lon and d_lat/d_lon in the mode-choice output are already snapped to
the 0.01-degree grid cell centroids (same LAT0/LON0/STEP convention used
throughout this project), so cells are matched by an exact centroid join
against Shanghaidata_final.csv rather than re-deriving lat/lon indices.
"""
import argparse
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_DIR / "data"
GRID_CSV = DATA_DIR / "Shanghaidata_final.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modechoice-csv", type=str, required=True,
                     help="output of compute_mode_choice_demand.py")
    ap.add_argument("--out-csv", type=str, required=True)
    args = ap.parse_args()

    mc = pd.read_csv(args.modechoice_csv, usecols=["o_lat", "o_lon", "d_lat", "d_lon", "trip_count"])
    grid = pd.read_csv(GRID_CSV).reset_index(drop=True)
    grid.columns = [c.strip().lstrip("﻿") for c in grid.columns]
    n_grid = len(grid)

    # round to the grid's own decimal precision so floating-point noise from CSV round-trips
    # doesn't break the join
    key_round = 4
    grid["_lat_r"] = grid["avg_lat"].round(key_round)
    grid["_lon_r"] = grid["avg_lon"].round(key_round)
    mc["_o_lat_r"] = mc["o_lat"].round(key_round)
    mc["_o_lon_r"] = mc["o_lon"].round(key_round)
    mc["_d_lat_r"] = mc["d_lat"].round(key_round)
    mc["_d_lon_r"] = mc["d_lon"].round(key_round)

    cell_lookup = grid.set_index(["_lat_r", "_lon_r"])["Grid ID"]

    origin_agg = mc.groupby(["_o_lat_r", "_o_lon_r"])["trip_count"].sum()
    dest_agg = mc.groupby(["_d_lat_r", "_d_lon_r"])["trip_count"].sum()

    origin_by_gridid = origin_agg.reindex(cell_lookup.index).to_numpy()
    dest_by_gridid = dest_agg.reindex(cell_lookup.index).to_numpy()
    import numpy as np
    origin_by_gridid = np.nan_to_num(origin_by_gridid)
    dest_by_gridid = np.nan_to_num(dest_by_gridid)

    out = pd.DataFrame({
        "idx": np.arange(n_grid),
        "Grid ID": grid["Grid ID"].to_numpy(),
        "avg_lat": grid["avg_lat"].to_numpy(),
        "avg_lon": grid["avg_lon"].to_numpy(),
        "real_origin_demand": origin_by_gridid,
        "real_dest_demand": dest_by_gridid,
        "real_total_demand": origin_by_gridid + dest_by_gridid,
    })
    out.to_csv(args.out_csv, index=False)

    print(f"grid cells: {n_grid}")
    print(f"total mode-choice-weighted trips (origin sum): {origin_by_gridid.sum():.0f}  "
          f"(dest sum): {dest_by_gridid.sum():.0f}")
    print(f"cells with zero UAM-adopting demand: {(out['real_total_demand'] == 0).sum()}/{n_grid}")
    print(f"saved -> {args.out_csv}")


if __name__ == "__main__":
    main()
