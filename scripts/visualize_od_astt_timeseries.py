# -*- coding: utf-8 -*-
"""
Per-OD-pair true vs predicted time series (line charts), for the ASTT
OD-preserving model (scripts/od_astt/train_shanghai_od_astt.py).

30 sites = 900 directed pairs -- plotting all of them is unreadable, so this
picks the top-N pairs by total true trip volume over the test window (the
same handful of hotspot cells visible on the diagonal/grid834/grid783 rows
in od_matrix_kmeans30_astt_true_vs_pred.png) and lines them up against the
real test-window timestamps from shanghai_od_kmeans30_meta.csv.
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
    ap.add_argument("--run", type=str, required=True,
                     help="run dir name under outputs/save_shanghai_od_astt/")
    ap.add_argument("--sites", type=str, default="selected_sites_kmeans_K30.csv")
    ap.add_argument("--meta", type=str, default="shanghai_od_kmeans30_meta.csv")
    ap.add_argument("--top-n", type=int, default=9)
    ap.add_argument("--fig-name", type=str, default="od_astt_top_pairs_timeseries.png")
    args = ap.parse_args()

    run_dir = OUT_DIR / "save_shanghai_od_astt" / args.run
    pred_log = np.load(run_dir / "ShanghaiASTT_1_prediction.npy")   # (n_test, N, N)
    true_log = np.load(run_dir / "ShanghaiASTT_1_groundtruth.npy")
    pred = np.expm1(pred_log); pred[pred < 0] = 0
    true = np.expm1(true_log)
    n_test = true.shape[0]

    meta = pd.read_csv(OUT_DIR / args.meta)
    ts_test = pd.to_datetime(meta["timestamp"]).to_numpy()[-n_test:]

    sites = pd.read_csv(DATA_DIR / args.sites)
    grid_ids = sites["Grid ID"].tolist()

    total = true.sum(axis=0)  # (N, N)
    flat_idx = np.argsort(total, axis=None)[::-1][: args.top_n]
    pairs = [np.unravel_index(idx, total.shape) for idx in flat_idx]

    ncols = 3
    nrows = int(np.ceil(args.top_n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows), dpi=150, sharex=True)
    axes = np.atleast_1d(axes).flatten()

    for ax, (i, j) in zip(axes, pairs):
        ax.plot(ts_test, true[:, i, j], label="true", color="tab:blue", linewidth=1.2)
        ax.plot(ts_test, pred[:, i, j], label="predicted", color="tab:orange", linewidth=1.2, linestyle="--")
        ax.set_title(f"grid {grid_ids[i]} -> grid {grid_ids[j]}  (test total={total[i, j]:.0f})", fontsize=9)
        ax.tick_params(axis="x", rotation=30, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)

    for ax in axes[len(pairs):]:
        ax.axis("off")

    axes[0].legend(fontsize=8, loc="upper right")
    fig.supylabel("trips / 30min")
    fig.suptitle(f"top-{args.top_n} OD pairs by test-window volume: true vs predicted", fontsize=12)
    plt.tight_layout()
    out_png = FIGS_DIR / args.fig_name
    plt.savefig(out_png, bbox_inches="tight")
    print("saved:", out_png)


if __name__ == "__main__":
    main()
