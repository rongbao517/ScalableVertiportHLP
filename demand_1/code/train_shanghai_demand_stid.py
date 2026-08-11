# -*- coding: utf-8 -*-
"""
STID (Spatial-Temporal Identity) style demand regressor -- Shao et al. 2022
"Spatial-Temporal Identity: A Simple yet Effective Baseline for
Multivariate Time Series Forecasting" (KDD21 workshop / later venues).
Tried as an alternative to T-GCN (train_shanghai_demand_tgcn.py) because
multiple 2023-2024 follow-up papers report that on small/overfit-prone
datasets a simple MLP with explicit learned site/time identity embeddings
matches or beats much heavier graph-convolution models -- exactly the
failure mode already hit twice in this project (uam_demand_model's ASTT,
R2=0.19; T-GCN's first, over-dense-graph attempt, R2=0.73) with only
~1000 training samples.

No graph at all here -- instead of geographic adjacency, each site gets
its own learned embedding vector (the model figures out site-to-site
similarity from the data itself, not from distance), concatenated with
learned time-of-day and day-of-week embeddings and a plain linear
encoding of the lookback window, then a small residual MLP stack.

Saves into the SAME outputs/save_shanghai_demand_gru/<ts>_<pid>/ layout as
train_shanghai_demand_gru.py / train_shanghai_demand_tgcn.py, so
evaluate_shanghai_demand_gru.py and calibrate_isotonic.py work unchanged.
"""
import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

BASE_DIR = Path(__file__).parent.parent
BINS_PER_DAY = 48  # 30-min bins


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="shanghai_demand_windows_kmeans30.npz")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--embed-dim", type=int, default=16)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batchsize", type=int, default=64)
    ap.add_argument("--epoch", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--loss", type=str, default="mse", choices=["mse", "weighted_mse", "quantile"])
    ap.add_argument("--tau", type=float, default=0.7, help="quantile for --loss quantile")
    return ap.parse_args()


def weighted_mse_loss(pred, target):
    weight = torch.expm1(target).clamp(min=0) + 1.0
    weight = weight / weight.mean().clamp(min=1e-6)
    return (weight * (pred - target) ** 2).mean()


def make_quantile_loss(tau):
    def loss_fn(pred, target):
        diff = target - pred
        return torch.maximum(tau * diff, (tau - 1) * diff).mean()
    return loss_fn


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.dropout(self.act(self.fc1(x)))
        h = self.fc2(h)
        return self.act(x + h)


class STID(nn.Module):
    def __init__(self, in_dim, out_dim, lookback, n_sites, n_tod, n_dow, hidden=64, embed_dim=16,
                 n_layers=3, dropout=0.1):
        super().__init__()
        self.input_fc = nn.Linear(lookback * in_dim, hidden)
        self.site_embed = nn.Embedding(n_sites, embed_dim)
        self.tod_embed = nn.Embedding(n_tod, embed_dim)
        self.dow_embed = nn.Embedding(n_dow, embed_dim)
        total_dim = hidden + embed_dim * 3
        self.proj = nn.Linear(total_dim, hidden)
        self.blocks = nn.ModuleList([ResidualMLPBlock(hidden, dropout) for _ in range(n_layers)])
        self.head = nn.Linear(hidden, out_dim)
        self.n_sites = n_sites

    def forward(self, x, tod_idx, dow_idx):  # x: (B, T, N, C_in); tod_idx/dow_idx: (B,)
        b, t, n, c = x.shape
        x_flat = x.permute(0, 2, 1, 3).reshape(b, n, t * c)   # (B, N, T*C)
        x_emb = self.input_fc(x_flat)                          # (B, N, hidden)

        site_e = self.site_embed(torch.arange(n, device=x.device)).unsqueeze(0).expand(b, n, -1)
        tod_e = self.tod_embed(tod_idx).unsqueeze(1).expand(b, n, -1)
        dow_e = self.dow_embed(dow_idx).unsqueeze(1).expand(b, n, -1)

        h = self.proj(torch.cat([x_emb, site_e, tod_e, dow_e], dim=-1))  # (B, N, hidden)
        for block in self.blocks:
            h = block(h)
        out = self.head(h)  # (B, N, out_dim)
        return out.reshape(b * n, -1)


def load_data(data_path):
    d = np.load(data_path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def time_indices(ts):
    ts = pd.to_datetime(ts)
    tod = (ts.hour * 2 + ts.minute // 30).to_numpy().astype(np.int64)  # 0..47
    dow = ts.dayofweek.to_numpy().astype(np.int64)                     # 0..6
    return tod, dow


def make_loader(X, y, ts, batchsize, shuffle, seed):
    X_out = X.astype(np.float32).copy()
    X_out[..., :2] = np.log1p(X_out[..., :2])
    y_log = np.log1p(y[:, 0]).astype(np.float32)
    tod, dow = time_indices(ts)
    ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X_out), torch.from_numpy(y_log),
        torch.from_numpy(tod), torch.from_numpy(dow),
    )
    gen = torch.Generator()
    gen.manual_seed(seed)
    return torch.utils.data.DataLoader(ds, batchsize, shuffle=shuffle, generator=gen)


def run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    loss_sum, n_obs = 0.0, 0
    with torch.set_grad_enabled(train):
        for X, y, tod, dow in loader:
            X, y, tod, dow = X.to(device), y.to(device), tod.to(device), dow.to(device)
            b, n, c_out = y.shape
            y_flat = y.reshape(b * n, c_out)
            pred = model(X, tod, dow)
            loss = criterion(pred, y_flat)
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            loss_sum += loss.item() * b
            n_obs += b
    return loss_sum / max(n_obs, 1)


def predict(model, loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for X, y, tod, dow in loader:
            X, tod, dow = X.to(device), tod.to(device), dow.to(device)
            b, n, c_out = y.shape
            pred = model(X, tod, dow).reshape(b, n, c_out).cpu().numpy()
            preds.append(pred)
            trues.append(y.numpy())
    return np.concatenate(preds), np.concatenate(trues)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}") if torch.cuda.is_available() else torch.device("cpu")
    print("device:", device)

    data_path = BASE_DIR / "data" / args.data
    data = load_data(data_path)
    train_loader = make_loader(data["X_train"], data["y_train"], data["ts_train"], args.batchsize, True, args.seed)
    val_loader = make_loader(data["X_val"], data["y_val"], data["ts_val"], args.batchsize, False, args.seed)
    test_loader = make_loader(data["X_test"], data["y_test"], data["ts_test"], args.batchsize, False, args.seed)
    print("train/val/test samples:", len(data["X_train"]), len(data["X_val"]), len(data["X_test"]))

    in_dim = data["X_train"].shape[-1]
    out_dim = data["y_train"].shape[-1]
    lookback = data["X_train"].shape[1]
    n_sites = data["X_train"].shape[2]
    print(f"in_dim={in_dim}  out_dim={out_dim}  lookback={lookback}  n_sites={n_sites}")

    model = STID(in_dim=in_dim, out_dim=out_dim, lookback=lookback, n_sites=n_sites,
                 n_tod=BINS_PER_DAY, n_dow=7, hidden=args.hidden, embed_dim=args.embed_dim,
                 n_layers=args.n_layers, dropout=args.dropout).to(device)
    print("params:", sum(p.numel() for p in model.parameters()))

    if args.loss == "weighted_mse":
        criterion = weighted_mse_loss
    elif args.loss == "quantile":
        criterion = make_quantile_loss(args.tau)
    else:
        criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    save_dir = BASE_DIR / "outputs" / "save_shanghai_demand_gru" / f"{datetime.now().strftime('%y%m%d%H%M%S')}_{os.getpid()}"
    save_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {"model": "STID"}
    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    best_path = save_dir / "stid.pt"

    best_val = np.inf
    wait = 0
    log_lines = []
    for epoch in range(args.epoch):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss = run_epoch(model, val_loader, criterion, device)
        if val_loss < best_val:
            best_val = val_loss
            wait = 0
            torch.save(model.state_dict(), best_path)
        else:
            wait += 1
        line = f"epoch {epoch} time {time.time()-t0:.1f}s train_loss {train_loss:.6f} val_loss {val_loss:.6f}"
        print(line)
        log_lines.append(line)
        if wait >= args.patience:
            print(f"early stopping at epoch {epoch}")
            break

    with open(save_dir / "train_log.txt", "w") as f:
        f.write("\n".join(log_lines) + "\n")

    model.load_state_dict(torch.load(best_path, map_location=device))
    pred_log, true_log = predict(model, test_loader, device)
    np.save(save_dir / "test_prediction_log.npy", pred_log)
    np.save(save_dir / "test_groundtruth_log.npy", true_log)

    pred = np.expm1(pred_log)
    pred[pred < 0] = 0
    true = np.expm1(true_log)
    nz = true > 1e-5

    mse_log = np.mean((pred_log[nz] - true_log[nz]) ** 2)
    rmse_log = np.sqrt(mse_log)
    mae_log = np.mean(np.abs(pred_log[nz] - true_log[nz]))
    mape_log = np.mean(np.abs((pred_log[nz] - true_log[nz]) / true_log[nz])) * 100
    metrics_log = dict(MSE=float(mse_log), RMSE=float(rmse_log), MAE=float(mae_log), MAPE=float(mape_log))
    print("test metrics (log1p space):", metrics_log)

    mse = np.mean((pred[nz] - true[nz]) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(pred[nz] - true[nz]))
    mape = np.mean(np.abs((pred[nz] - true[nz]) / true[nz])) * 100
    r2 = 1 - np.sum((true[nz] - pred[nz]) ** 2) / np.sum((true[nz] - true[nz].mean()) ** 2)
    pcc = np.corrcoef(true[nz].flatten(), pred[nz].flatten())[0, 1]
    metrics = dict(MSE=float(mse), RMSE=float(rmse), MAE=float(mae), MAPE=float(mape), R2=float(r2), PCC=float(pcc))
    print("test metrics (original space):", metrics)
    with open(save_dir / "test_metrics.json", "w") as f:
        json.dump({"log_space": metrics_log, "original_space": metrics}, f, indent=2)

    print("saved run ->", save_dir)


if __name__ == "__main__":
    main()
