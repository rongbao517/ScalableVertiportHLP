"""
Visualize Grid-VNS hub selection results on a real map.
Runs Grid-VNS for given (n_b, p) pairs and saves PNG maps.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from problem import HubLocationProblem
from vns import GridVNS

try:
    import contextily as ctx
    HAS_CTX = True
except ImportError:
    HAS_CTX = False

try:
    import geopandas as gpd
    from shapely.geometry import Point
    HAS_GPD = True
except ImportError:
    HAS_GPD = False

DATA_DIR  = Path(__file__).parent.parent / "data"
OUT_DIR   = Path(__file__).parent.parent / "Results"
OUT_DIR.mkdir(exist_ok=True)


# ── bbox ────────────────────────────────────────────────────────────────────
def get_bbox():
    df = pd.read_csv(DATA_DIR / "requests.csv")
    return (
        min(df["originLat"].min(), df["destinationLat"].min()),
        max(df["originLat"].max(), df["destinationLat"].max()),
        min(df["originLon"].min(), df["destinationLon"].min()),
        max(df["originLon"].max(), df["destinationLon"].max()),
    )


def hub_to_coord(hub_id, n_b, lat_min, lat_max, lon_min, lon_max):
    col = hub_id % n_b
    row = hub_id // n_b
    lat = lat_min + (row + 0.5) * (lat_max - lat_min) / n_b
    lon = lon_min + (col + 0.5) * (lon_max - lon_min) / n_b
    return lat, lon


def node_to_coord(node_id, n_b, lat_min, lat_max, lon_min, lon_max):
    return hub_to_coord(node_id, n_b, lat_min, lat_max, lon_min, lon_max)


# ── VNS run ──────────────────────────────────────────────────────────────────
def run_vns(n_b, p, n_trials=3, seed_base=42):
    prob = HubLocationProblem(n_b=n_b)
    best_obj, best_hubs, best_assign = np.inf, None, None

    for t in range(n_trials):
        g = GridVNS(prob, time_limit=120, max_no_improve=200, seed=seed_base + t)
        hubs, obj, elapsed, _ = g.run(p)
        print(f"  [n_b={n_b}, p={p}] trial {t+1}: obj={obj:.2f}, t={elapsed:.1f}s", flush=True)
        if obj < best_obj:
            best_obj  = obj
            best_hubs = hubs
            # recompute assignment for visualization
            hub_ids = np.array(sorted(hubs))
            assign  = np.argmin(prob.C[:, hub_ids], axis=1)   # nearest-hub
            best_assign = assign

    return best_obj, sorted(best_hubs), best_assign, prob


# ── single map ───────────────────────────────────────────────────────────────
def plot_map(n_b, p, hubs, assign, prob, lat_min, lat_max, lon_min, lon_max, save_path):
    hub_set = set(hubs)
    n       = prob.n
    p_      = len(hubs)

    # colour palette per hub
    cmap   = cm.get_cmap("tab20", p_)
    colors = [cmap(i) for i in range(p_)]

    hub_idx_to_color = {h: colors[i] for i, h in enumerate(hubs)}

    # node coords & hub assignment
    node_lons, node_lats, node_colors = [], [], []
    for nd in range(n):
        lat, lon = node_to_coord(nd, n_b, lat_min, lat_max, lon_min, lon_max)
        node_lats.append(lat)
        node_lons.append(lon)
        if nd in hub_set:
            node_colors.append(hub_idx_to_color[nd])
        else:
            assigned_hub = hubs[assign[nd]]
            node_colors.append(hub_idx_to_color[assigned_hub])

    hub_lats = [node_to_coord(h, n_b, lat_min, lat_max, lon_min, lon_max)[0] for h in hubs]
    hub_lons = [node_to_coord(h, n_b, lat_min, lat_max, lon_min, lon_max)[1] for h in hubs]

    # ── figure ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 11))

    if HAS_CTX and HAS_GPD:
        # build GeoDataFrame in WGS84, reproject to Web Mercator for basemap
        # scatter nodes coloured by voronoi zone
        from shapely.geometry import box as shpbox
        cell_h = (lat_max - lat_min) / n_b
        cell_w = (lon_max - lon_min) / n_b

        records = []
        for nd in range(n):
            lat, lon = node_lats[nd], node_lons[nd]
            records.append({
                "geometry": shpbox(lon - cell_w/2, lat - cell_h/2,
                                   lon + cell_w/2, lat + cell_h/2),
                "color": node_colors[nd],
                "is_hub": nd in hub_set,
            })
        gdf = gpd.GeoDataFrame(records, crs="EPSG:4326").to_crs("EPSG:3857")

        # draw coloured grid cells (assignment zones)
        for _, row in gdf.iterrows():
            alpha = 0.55 if not row["is_hub"] else 0.85
            gpd.GeoDataFrame([row], crs="EPSG:3857").plot(
                ax=ax, color=row["color"], alpha=alpha,
                edgecolor="white", linewidth=0.3)

        # add OSM basemap beneath cells — blend with multiply
        try:
            ctx.add_basemap(ax, crs="EPSG:3857",
                            source=ctx.providers.OpenStreetMap.Mapnik,
                            alpha=0.55, zoom="auto")
        except Exception:
            try:
                ctx.add_basemap(ax, crs="EPSG:3857",
                                source=ctx.providers.CartoDB.Positron,
                                alpha=0.55, zoom="auto")
            except Exception:
                pass   # no internet — skip basemap

        # hub markers (project to 3857)
        hub_pts = gpd.GeoDataFrame(
            {"geometry": [Point(lon, lat) for lat, lon in zip(hub_lats, hub_lons)]},
            crs="EPSG:4326").to_crs("EPSG:3857")
        xs = hub_pts.geometry.x.values
        ys = hub_pts.geometry.y.values

        # outer ring
        ax.scatter(xs, ys, s=280, c="white", zorder=6, linewidths=0)
        # inner dot coloured by hub index
        ax.scatter(xs, ys, s=160,
                   c=[hub_idx_to_color[h] for h in hubs],
                   edgecolors="black", linewidths=1.5, zorder=7)

        # hub labels
        for i, (x, y, h) in enumerate(zip(xs, ys, hubs)):
            ax.annotate(f"H{i+1}", (x, y),
                        textcoords="offset points", xytext=(6, 6),
                        fontsize=8, fontweight="bold", color="black",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"))

        ax.set_axis_off()

    else:
        # fallback: plain scatter without basemap
        for nd in range(n):
            if nd not in hub_set:
                ax.scatter(node_lons[nd], node_lats[nd],
                           c=[node_colors[nd]], s=18, alpha=0.4, linewidths=0)
        ax.scatter(hub_lons, hub_lats, c=[hub_idx_to_color[h] for h in hubs],
                   s=200, edgecolors="black", linewidths=1.5, zorder=5)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    # ── legend ──────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(facecolor=colors[i], edgecolor="black",
                       label=f"H{i+1} (grid #{hubs[i]})")
        for i in range(p_)
    ]
    ax.legend(handles=legend_handles, title="Vertiport Hubs",
              loc="lower left", fontsize=7.5, title_fontsize=9,
              framealpha=0.88, ncol=2 if p_ > 8 else 1)

    ax.set_title(
        f"Vertiport Hub Location — Beijing\n"
        f"Grid: {n_b}×{n_b}  |  p = {p_} hubs  |  hubable nodes = {prob.n_h}",
        fontsize=13, pad=12)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {save_path}")


# ── main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    lat_min, lat_max, lon_min, lon_max = get_bbox()

    cases = [(15, 10), (20, 10), (25, 10)]

    for n_b, p in cases:
        print(f"\n>>> n_b={n_b}, p={p}")
        obj, hubs, assign, prob = run_vns(n_b, p, n_trials=3)
        print(f"    Best obj = {obj:.2f}")
        print(f"    Hub grid indices: {hubs}")
        coords = [hub_to_coord(h, n_b, lat_min, lat_max, lon_min, lon_max) for h in hubs]
        for i, (lat, lon) in enumerate(coords, 1):
            print(f"    H{i:>2}  ({lat:.5f}, {lon:.5f})")

        save_path = OUT_DIR / f"hub_map_nb{n_b}_p{p}.png"
        plot_map(n_b, p, hubs, assign, prob,
                 lat_min, lat_max, lon_min, lon_max, save_path)

    print("\nDone.")
