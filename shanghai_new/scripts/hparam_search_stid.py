# -*- coding: utf-8 -*-
"""
Random hyperparameter search for STID (train_shanghai_demand_stid.py) --
every experiment so far (SiteGRU, T-GCN, STID) used the same untuned
defaults (hidden=64, embed_dim=16, n_layers=3, dropout=0.1, lr=1e-3).
Given the dataset is tiny (1008 train samples), search is kept narrow
(model size mostly <= current default) rather than searching toward
bigger models -- every previous capacity increase in this project (ASTT,
dense-graph T-GCN) made things worse, not better.

Selection is by VALIDATION loss only (never touches test), then the
winning config's test metrics are reported once at the end -- standard
train/val/test discipline, not test-set cherry-picking.

Runs all trials in a single process/GPU allocation (one sbatch job) to
avoid per-trial submission/queue overhead; each trial gets its own run
dir under outputs/save_shanghai_demand_gru/ so the winner can be
evaluated/calibrated/visualized afterward exactly like any other run.
"""
import argparse
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import train_shanghai_demand_stid as stid_mod

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs"


SEARCH_SPACE = {
    "hidden": [32, 48, 64, 96],
    "embed_dim": [8, 16, 24, 32],
    "n_layers": [1, 2, 3, 4],
    "dropout": [0.05, 0.1, 0.15, 0.2, 0.3],
    "lr": [5e-4, 1e-3, 2e-3],
    "weight_decay": [1e-6, 1e-5, 1e-4],
    "batchsize": [32, 64, 128],
}


def sample_config(rng):
    return {k: rng.choice(v) for k, v in SEARCH_SPACE.items()}


def train_one(config, data, device, seed, epochs, patience):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader = stid_mod.make_loader(data["X_train"], data["y_train"], data["ts_train"],
                                         config["batchsize"], True, seed)
    val_loader = stid_mod.make_loader(data["X_val"], data["y_val"], data["ts_val"],
                                       config["batchsize"], False, seed)
    test_loader = stid_mod.make_loader(data["X_test"], data["y_test"], data["ts_test"],
                                        config["batchsize"], False, seed)

    in_dim = data["X_train"].shape[-1]
    out_dim = data["y_train"].shape[-1]
    lookback = data["X_train"].shape[1]
    n_sites = data["X_train"].shape[2]

    model = stid_mod.STID(in_dim=in_dim, out_dim=out_dim, lookback=lookback, n_sites=n_sites,
                           n_tod=stid_mod.BINS_PER_DAY, n_dow=7, hidden=config["hidden"],
                           embed_dim=config["embed_dim"], n_layers=config["n_layers"],
                           dropout=config["dropout"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    best_val = np.inf
    wait = 0
    best_state = None
    for epoch in range(epochs):
        stid_mod.run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss = stid_mod.run_epoch(model, val_loader, criterion, device)
        if val_loss < best_val:
            best_val = val_loss
            wait = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    pred_log, true_log = stid_mod.predict(model, test_loader, device)
    pred = np.expm1(pred_log); pred[pred < 0] = 0
    true = np.expm1(true_log)
    nz = true > 1e-5
    r2 = 1 - np.sum((true[nz] - pred[nz]) ** 2) / np.sum((true[nz] - true[nz].mean()) ** 2)
    pcc = np.corrcoef(true[nz].flatten(), pred[nz].flatten())[0, 1]
    mae = np.mean(np.abs(pred[nz] - true[nz]))
    test_metrics = dict(R2=float(r2), PCC=float(pcc), MAE=float(mae))
    return best_val, test_metrics, n_params, model, best_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="shanghai_demand_windows_kmeans30.npz")
    ap.add_argument("--n-trials", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--search-seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print("device:", device)
    data = stid_mod.load_data(DATA_DIR / args.data)
    rng = random.Random(args.search_seed)

    results = []
    best = None
    for trial in range(args.n_trials):
        config = sample_config(rng)
        t0 = time.time()
        val_loss, test_metrics, n_params, model, state = train_one(
            config, data, device, args.seed, args.epochs, args.patience)
        dt = time.time() - t0
        row = {**config, "val_loss": val_loss, "n_params": n_params, "time_s": round(dt, 1), **test_metrics}
        results.append(row)
        print(f"[trial {trial:02d}] val_loss={val_loss:.4f}  test_R2={test_metrics['R2']:.4f}  "
              f"params={n_params}  time={dt:.1f}s  cfg={config}")

        if best is None or val_loss < best["val_loss"]:
            best = row
            best_state = {k: v.clone() for k, v in state.items()}
            best_config = config

    print("\n=== SEARCH DONE ===")
    print("best by val_loss:", best)

    # save leaderboard
    import pandas as pd
    df = pd.DataFrame(results).sort_values("val_loss")
    lb_path = OUT_DIR / "stid_hparam_search_leaderboard.csv"
    df.to_csv(lb_path, index=False)
    print(f"saved leaderboard -> {lb_path}")

    # persist the winning model in the standard run-dir layout
    save_dir = OUT_DIR / "save_shanghai_demand_gru" / f"{datetime.now().strftime('%y%m%d%H%M%S')}_{os.getpid()}"
    save_dir.mkdir(parents=True, exist_ok=True)
    config = {**best_config, "seed": args.seed, "data": args.data, "model": "STID",
              "epoch": args.epochs, "patience": args.patience}
    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    torch.save(best_state, save_dir / "stid.pt")

    in_dim = data["X_train"].shape[-1]
    out_dim = data["y_train"].shape[-1]
    lookback = data["X_train"].shape[1]
    n_sites = data["X_train"].shape[2]
    model = stid_mod.STID(in_dim=in_dim, out_dim=out_dim, lookback=lookback, n_sites=n_sites,
                           n_tod=stid_mod.BINS_PER_DAY, n_dow=7, hidden=best_config["hidden"],
                           embed_dim=best_config["embed_dim"], n_layers=best_config["n_layers"],
                           dropout=best_config["dropout"]).to(device)
    model.load_state_dict(best_state)
    test_loader = stid_mod.make_loader(data["X_test"], data["y_test"], data["ts_test"],
                                        best_config["batchsize"], False, args.seed)
    pred_log, true_log = stid_mod.predict(model, test_loader, device)
    np.save(save_dir / "test_prediction_log.npy", pred_log)
    np.save(save_dir / "test_groundtruth_log.npy", true_log)
    pred = np.expm1(pred_log); pred[pred < 0] = 0
    true = np.expm1(true_log)
    nz = true > 1e-5
    mse = np.mean((pred[nz] - true[nz]) ** 2)
    r2 = 1 - np.sum((true[nz] - pred[nz]) ** 2) / np.sum((true[nz] - true[nz].mean()) ** 2)
    pcc = np.corrcoef(true[nz].flatten(), pred[nz].flatten())[0, 1]
    mae = np.mean(np.abs(pred[nz] - true[nz]))
    mape = np.mean(np.abs((pred[nz] - true[nz]) / true[nz])) * 100
    metrics = dict(MSE=float(mse), RMSE=float(np.sqrt(mse)), MAE=float(mae), MAPE=float(mape),
                    R2=float(r2), PCC=float(pcc))
    with open(save_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nsaved winning run -> {save_dir}")
    print("test metrics:", metrics)


if __name__ == "__main__":
    main()
