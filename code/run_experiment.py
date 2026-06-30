"""
Reproduce Table5 results: Grid-VNS vs N-VNS on small instances.
Runs 10 independent trials per instance (matching paper).
"""
import sys
import csv
import numpy as np
import pandas as pd
from pathlib import Path
from problem import HubLocationProblem
from vns import GridVNS, NVNS

RESULTS_DIR = Path(__file__).parent.parent / "Results"

DATA_DIR = Path(__file__).parent.parent / "data"

# Table5 reference values (CPLEX optimal for n_b <= 10)
TABLE5_REF = {
    (4,  3): 2761005.56,
    (6,  3): 2541079.56,
    (8,  3): 2980902.29,
    (4,  5): 2484384.31,
    (6,  5): 2186157.97,
    (8,  5): 2614842.36,
    (10, 2): 3034998.67,
    (10, 5): 2462767.00,
    (10, 10): 2211559.09,
}


def get_bbox():
    """Compute Beijing bounding box from requests.csv (same logic as data_process_cw.py)."""
    df = pd.read_csv(DATA_DIR / "requests.csv")
    city_lower = min(df["originLat"].min(), df["destinationLat"].min())
    city_upper = max(df["originLat"].max(), df["destinationLat"].max())
    city_left  = min(df["originLon"].min(), df["destinationLon"].min())
    city_right = max(df["originLon"].max(), df["destinationLon"].max())
    return city_lower, city_upper, city_left, city_right


def hub_idx_to_coord(hub_id, n_b, city_lower, city_upper, city_left, city_right):
    """Convert grid node index to (lat, lon) of grid cell center."""
    col_i = hub_id % n_b
    row_j = hub_id // n_b
    lat = city_lower + (row_j + 0.5) * (city_upper - city_lower) / n_b
    lon = city_left  + (col_i + 0.5) * (city_right - city_left)  / n_b
    return lat, lon


def main():
    cases = [
        (4,  3), (6,  3), (8,  3),
        (4,  5), (6,  5), (8,  5),
        (10, 2), (10, 5), (10, 10),
    ]
    n_trials = 10

    city_lower, city_upper, city_left, city_right = get_bbox()

    coord_rows = []

    print(f"{'n_b':>4} {'p':>3} {'Paper_best':>12}  "
          f"{'GVNS_best':>12} {'GVNS_gap%':>9} {'GVNS_#best':>10} {'GVNS_t(s)':>9}  "
          f"{'NVNS_best':>12} {'NVNS_gap%':>9} {'NVNS_#best':>10} {'NVNS_t(s)':>9}")
    print("-" * 115)

    for n_b, p in cases:
        ref  = TABLE5_REF.get((n_b, p), None)
        prob = HubLocationProblem(n_b=n_b)

        gvns_results, nvns_results = [], []

        for t in range(n_trials):
            g = GridVNS(prob, time_limit=120, max_no_improve=200, seed=t * 7)
            g_hubs, g_obj, g_time, _ = g.run(p)
            gvns_results.append((g_obj, g_time, g_hubs))

            nv = NVNS(prob, time_limit=120, max_no_improve=200, seed=t * 7)
            n_hubs, n_obj, n_time, _ = nv.run(p)
            nvns_results.append((n_obj, n_time, n_hubs))

        g_best     = min(r[0] for r in gvns_results)
        g_avg_t    = np.mean([r[1] for r in gvns_results])
        g_n_best   = sum(1 for r in gvns_results if abs(r[0] - g_best) < 1e-3)
        g_best_hubs = next(r[2] for r in gvns_results if abs(r[0] - g_best) < 1e-3)

        n_best     = min(r[0] for r in nvns_results)
        n_avg_t    = np.mean([r[1] for r in nvns_results])
        n_n_best   = sum(1 for r in nvns_results if abs(r[0] - n_best) < 1e-3)
        n_best_hubs = next(r[2] for r in nvns_results if abs(r[0] - n_best) < 1e-3)

        g_gap   = (g_best - ref) / ref * 100 if ref else float('nan')
        n_gap   = (n_best - ref) / ref * 100 if ref else float('nan')
        ref_str = f"{ref:.2f}" if ref else "N/A"

        print(f"{n_b:>4} {p:>3} {ref_str:>12}  "
              f"{g_best:>12.2f} {g_gap:>+9.3f}% {g_n_best:>10d} {g_avg_t:>9.2f}s  "
              f"{n_best:>12.2f} {n_gap:>+9.3f}% {n_n_best:>10d} {n_avg_t:>9.2f}s")

        # Hub coordinates for best GVNS solution
        g_coords = [hub_idx_to_coord(h, n_b, city_lower, city_upper, city_left, city_right)
                    for h in sorted(g_best_hubs)]
        coord_strs = ", ".join(f"({lat:.4f},{lon:.4f})" for lat, lon in g_coords)
        print(f"     GVNS hubs (idx={sorted(g_best_hubs)}): {coord_strs}")

        for rank, (hub_idx, (lat, lon)) in enumerate(zip(sorted(g_best_hubs), g_coords), start=1):
            coord_rows.append({
                "n_b": n_b,
                "p": p,
                "hub_rank": rank,
                "hub_idx": hub_idx,
                "lat": round(lat, 6),
                "lon": round(lon, 6),
                "obj": round(g_best, 4),
            })

        sys.stdout.flush()

    out_path = RESULTS_DIR / "hub_coordinates.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["n_b", "p", "hub_rank", "hub_idx", "lat", "lon", "obj"])
        writer.writeheader()
        writer.writerows(coord_rows)
    print(f"\nHub coordinates saved to {out_path}")


if __name__ == "__main__":
    main()
