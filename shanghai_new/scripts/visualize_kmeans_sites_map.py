# -*- coding: utf-8 -*-
"""
Visualize the demand-weighted K-means site-selection result
(run_site_selection_kmeans_K30.py) on a real OpenStreetMap basemap of
Shanghai:

  Fig 1: every one of the 1676 grid cells, colored by which of the 30
         clusters it was assigned to, with the 30 chosen sites (nearest
         real cell to each cluster's demand-weighted centroid) marked as
         large white-edged stars. Shows the clusters actually tile the
         city rather than piling into one hotspot.
  Fig 2: the 30 chosen sites only, sized/colored by real trip demand at
         that cell -- same visual convention as
         willey revise/visualize_selected_sites_map.py, for side-by-side
         comparison with the NSGA-II site maps.
"""
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import contextily as cx
from shapely.geometry import Point

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
FIGS_DIR = BASE_DIR / "outputs" / "figs"

ASSIGNMENTS_CSV = DATA_DIR / "kmeans_K30_cluster_assignments.csv"
SITES_CSV = DATA_DIR / "selected_sites_kmeans_K30.csv"


def to_gdf(df, lat="avg_lat", lon="avg_lon"):
    g = gpd.GeoDataFrame(
        df.copy(),
        geometry=[Point(xy) for xy in zip(df[lon], df[lat])],
        crs="EPSG:4326",
    )
    return g.to_crs(epsg=3857)


def cluster_palette(n_clusters):
    """tab20 only has 20 distinct colors -- with 30 clusters it wraps around and
    reuses colors for cluster i and cluster i+20, making unrelated clusters look
    identical. Concatenate tab20 + tab20b to get >=30 unique colors instead."""
    colors = list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("tab20b").colors)
    return colors[:n_clusters]


def plot_clusters(assignments, sites):
    gdf_cells = to_gdf(assignments)
    gdf_sites = to_gdf(sites)

    n_clusters = assignments["cluster_label"].nunique()
    palette = cluster_palette(n_clusters)
    cell_colors = [palette[c] for c in gdf_cells["cluster_label"]]

    fig, ax = plt.subplots(figsize=(10, 10), dpi=170)
    gdf_cells.plot(ax=ax, color=cell_colors, markersize=14, alpha=0.65)
    gdf_sites.plot(ax=ax, color="none", edgecolor="black", linewidth=1.3,
                    markersize=420, marker="*", zorder=5, label="Selected site (30)")

    xmin, ymin, xmax, ymax = gdf_cells.total_bounds
    pad_x, pad_y = (xmax - xmin) * 0.05, (ymax - ymin) * 0.05
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)

    cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, zoom=11)
    ax.set_axis_off()
    ax.legend(loc="upper right", frameon=True, fontsize=10)
    ax.set_title("Demand-weighted K-means site selection — K=30\n"
                  "(colors = cluster membership of all 1676 grid cells, stars = chosen sites)",
                  fontsize=11)
    plt.tight_layout()
    out_png = FIGS_DIR / "kmeans_K30_clusters_map.png"
    out_pdf = FIGS_DIR / "kmeans_K30_clusters_map.pdf"
    plt.savefig(out_png, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print("saved:", out_png)
    print("saved:", out_pdf)


def plot_selected_sites(sites):
    gdf_sel = to_gdf(sites)

    fig, ax = plt.subplots(figsize=(9, 9), dpi=170)
    demand = sites["real_total_demand"].to_numpy().astype(float)
    sizes = demand / demand.max() * 500 + 60
    gdf_sel.plot(ax=ax, column=demand, cmap="plasma", edgecolor="white", linewidth=0.8,
                 markersize=sizes, marker="o", legend=True,
                 legend_kwds={"label": "Real demand at site (trips, 29 days)", "shrink": 0.6})

    for _, r in sites.iterrows():
        pt = gpd.GeoSeries([Point(r["avg_lon"], r["avg_lat"])], crs="EPSG:4326").to_crs(epsg=3857).iloc[0]
        ax.annotate(str(int(r["Grid ID"])), (pt.x, pt.y), fontsize=6.5,
                    xytext=(4, 4), textcoords="offset points")

    xmin, ymin, xmax, ymax = gdf_sel.total_bounds
    pad_x, pad_y = (xmax - xmin) * 0.15, (ymax - ymin) * 0.15
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)

    cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, zoom=12)
    ax.set_axis_off()
    ax.set_title("Shanghai UAM Vertiport Site Selection — K-means (demand-weighted), K=30", fontsize=11)
    plt.tight_layout()
    out_png = FIGS_DIR / "kmeans_K30_selected_sites_map.png"
    out_pdf = FIGS_DIR / "kmeans_K30_selected_sites_map.pdf"
    plt.savefig(out_png, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print("saved:", out_png)
    print("saved:", out_pdf)


def main():
    assignments = pd.read_csv(ASSIGNMENTS_CSV)
    sites = pd.read_csv(SITES_CSV)

    slat, slon = sites["avg_lat"].to_numpy(), sites["avg_lon"].to_numpy()
    print(f"cells: {len(assignments)}  clusters: {assignments['cluster_label'].nunique()}  "
          f"selected sites: {len(sites)}")
    print(f"lat range: {slat.min():.4f}-{slat.max():.4f}  lon range: {slon.min():.4f}-{slon.max():.4f}")

    plot_clusters(assignments, sites)
    plot_selected_sites(sites)


if __name__ == "__main__":
    main()
