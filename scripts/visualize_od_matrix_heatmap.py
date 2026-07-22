# -*- coding: utf-8 -*-
"""
Site-to-site OD flow heatmap (grid, not map): rows = origin site, columns =
destination site, cell color/annotation = real trip count between that pair
over the full observation window.

Uses the real (non-modeled) OD tensor from extract_shanghai_od_matrix_final10.py
(shanghai_od_final10_30min_1channel.npz, shape (1, n_bins, 10, 10) for the 10
NSGA-II final sites in selected_sites_K10_v3.csv) -- there is no "predicted"
counterpart because the demand models (GRU/STID/TGCN) only forecast each
site's own total origin+destination volume, not pairwise site-to-site flow.

Color scale is log1p'd since the diagonal (same-site trips) is ~2x the
combined off-diagonal mass and would otherwise wash out inter-site structure;
raw counts are annotated in each cell regardless.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs"
FIGS_DIR = OUT_DIR / "figs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--od-npz", type=str, default="shanghai_od_final10_30min_1channel.npz")
    ap.add_argument("--sites", type=str, default="selected_sites_K10_v3.csv")
    ap.add_argument("--fig-name", type=str, default="od_matrix_final10_heatmap.png")
    args = ap.parse_args()

    od = np.load(OUT_DIR / args.od_npz, allow_pickle=True)["arr_0"][0]  # (n_bins, n_sites, n_sites)
    total = od.sum(axis=0)  # (n_sites, n_sites), rows=origin, cols=dest

    sites = pd.read_csv(DATA_DIR / args.sites)
    labels = [f"grid {gid}" for gid in sites["Grid ID"]]
    n = len(labels)

    # 10 sites: big annotated cells fit fine. 30 sites: 900 numbers would be
    # unreadable clutter, so drop the per-cell text and rely on color + a
    # smaller tick font instead.
    annotate = n <= 15
    side = max(8.5, n * 0.42)
    tick_fontsize = 9 if n <= 15 else max(5, 8 - n // 10)

    fig, ax = plt.subplots(figsize=(side + 1, side), dpi=150)
    im = ax.imshow(np.log1p(total), cmap="viridis")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=tick_fontsize)
    ax.set_yticklabels(labels, fontsize=tick_fontsize)
    ax.set_xlabel("destination site")
    ax.set_ylabel("origin site")
    ax.set_title(f"site-to-site trip counts, full window ({od.shape[0]} bins x 30min, n={n} sites)")

    if annotate:
        vmax = total.max()
        for i in range(n):
            for j in range(n):
                val = total[i, j]
                color = "white" if val > vmax * 0.4 else "black"
                ax.text(j, i, f"{int(val)}", ha="center", va="center", color=color, fontsize=7)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("log1p(trip count)")

    plt.tight_layout()
    out_png = FIGS_DIR / args.fig_name
    plt.savefig(out_png, bbox_inches="tight")
    print("saved:", out_png)


if __name__ == "__main__":
    main()
