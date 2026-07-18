# -*- coding: utf-8 -*-
"""
Demand-weighted K-means site selection (K=30), as an alternative to the
NSGA-II approach (run_site_selection_K10_v3.py).

Method: cluster all 1676 grid cells' (avg_lat, avg_lon) into 30 clusters
with sklearn KMeans, weighting each cell by its real trip demand
(real_demand_all_grids.csv, from compute_real_demand_all_grids.py) via
sample_weight -- so cluster centers are pulled toward high-demand areas,
but with the natural K-means side effect of spreading clusters out to
cover the whole weighted mass (unlike NSGA-II's raw total-demand
objective, K-means has no way to just pile all 30 sites into one hotspot:
minimizing within-cluster variance forces spatial spread by construction).

Each cluster center is a synthetic (lat, lon), not a real buildable site --
so, within that cluster's own member cells, we pick the single highest-demand
grid cell as the buildable site (not just the nearest-to-centroid cell
regardless of its own demand: a weighted-KMeans centroid can land in a locally
quiet pocket of an otherwise busy cluster -- e.g. residential blocks between
two hot commercial strips -- so snapping to "nearest" alone was occasionally
picking a near-zero-demand cell out of a cluster with 100k+ total trips.
Picking the cluster's own top-demand cell can't collide across clusters
either, since cluster membership already partitions all cells disjointly).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs"
DEMAND_CSV = DATA_DIR / "real_demand_all_grids.csv"
OUT_CSV = DATA_DIR / "selected_sites_kmeans_K30.csv"
OUT_ASSIGNMENTS_CSV = DATA_DIR / "kmeans_K30_cluster_assignments.csv"

N_CLUSTERS = 30
SEED = 42


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demand-csv", type=str, default=str(DEMAND_CSV))
    ap.add_argument("--out-csv", type=str, default=str(OUT_CSV))
    ap.add_argument("--out-assignments-csv", type=str, default=str(OUT_ASSIGNMENTS_CSV))
    args = ap.parse_args()

    grid = pd.read_csv(args.demand_csv)
    n = len(grid)
    lat = grid["avg_lat"].to_numpy()
    lon = grid["avg_lon"].to_numpy()
    weight = grid["real_total_demand"].to_numpy().astype(float)
    print(f"grid cells: {n}  total demand: {weight.sum():.0f}  zero-demand cells: {(weight == 0).sum()}")

    coords = np.column_stack([lat, lon])
    km = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=10)
    km.fit(coords, sample_weight=weight)
    centers = km.cluster_centers_  # (30, 2) in (lat, lon)

    chosen_idx = []
    for c in range(N_CLUSTERS):
        member_idx = np.where(km.labels_ == c)[0]
        best = member_idx[np.argmax(weight[member_idx])]
        chosen_idx.append(int(best))

    sites = grid.iloc[chosen_idx].reset_index(drop=True)
    sites["cluster_label"] = range(N_CLUSTERS)
    cluster_sizes = pd.Series(km.labels_).value_counts().sort_index()
    cluster_weight = pd.Series(weight).groupby(km.labels_).sum()
    sites["cluster_n_cells"] = cluster_sizes.reindex(range(N_CLUSTERS)).to_numpy()
    sites["cluster_total_demand"] = cluster_weight.reindex(range(N_CLUSTERS)).to_numpy()

    sites.to_csv(args.out_csv, index=False)
    print(f"saved -> {args.out_csv}")

    # per-cell cluster assignment for ALL 1676 grid cells, for mapping/visualization
    assignments = grid[["idx", "Grid ID", "avg_lat", "avg_lon", "real_total_demand"]].copy()
    assignments["cluster_label"] = km.labels_
    assignments["is_selected_site"] = False
    assignments.loc[chosen_idx, "is_selected_site"] = True
    assignments.to_csv(args.out_assignments_csv, index=False)
    print(f"saved -> {args.out_assignments_csv}")
    print(f"total real demand captured by 30 chosen cells: {sites['real_total_demand'].sum():.0f}  "
          f"(vs. NSGA v3 10-site: separate comparison)")

    # spatial spread diagnostics, for comparison with the NSGA approach
    slat, slon = sites["avg_lat"].to_numpy(), sites["avg_lon"].to_numpy()
    D = np.zeros((N_CLUSTERS, N_CLUSTERS))
    for i in range(N_CLUSTERS):
        D[i] = haversine_km(slat[i], slon[i], slat, slon)
    np.fill_diagonal(D, np.inf)
    nn = D.min(axis=1)
    iu = np.triu_indices(N_CLUSTERS, 1)
    print(f"pairwise dist (km): min={D[iu][np.isfinite(D[iu])].min():.2f} "
          f"max={D[iu][np.isfinite(D[iu])].max():.2f} mean={D[iu][np.isfinite(D[iu])].mean():.2f}")
    print(f"nearest-neighbor dist per site (km): min={nn.min():.2f} median={np.median(nn):.2f}  "
          f"(sites closer than 1km to nearest neighbor: {(nn < 1).sum()})")
    print(f"lat range: {slat.min():.4f}-{slat.max():.4f}  lon range: {slon.min():.4f}-{slon.max():.4f}")


if __name__ == "__main__":
    main()
