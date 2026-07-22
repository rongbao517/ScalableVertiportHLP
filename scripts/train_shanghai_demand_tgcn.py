# -*- coding: utf-8 -*-
"""
T-GCN (Temporal Graph Convolutional Network, Zhao et al. 2019 style) demand
regressor -- a spatiotemporal upgrade of the per-site SiteGRU baseline
(train_shanghai_demand_gru.py), which treats all 30 sites as independent
(no cross-site interaction at all). This model adds a GCN layer that mixes
each site's features with its geographic neighbors (distance-weighted
adjacency) at every timestep, BEFORE the per-site GRU -- so the GRU now
sees a spatially-aware representation instead of a raw isolated series.

Not reusing willey revise's STGCN/GAT/AGG-DGRU notebooks: those are binary
*classifiers* built for the congestion-classification site-selection
pipeline (CrossEntropyLoss, num_classes=2), and their heavier attention/
multi-block architectures are exactly the kind of thing that underfit
badly (R2=0.19) on this project's small ~1000-sample regression task
(uam_demand_model/'s ASTT case). This keeps the same lightweight budget as
SiteGRU (single GCN layer + single-layer BiGRU) and just adds the one
missing ingredient -- spatial mixing -- rather than jumping to a much
bigger model.

Graph: adjacency = Gaussian kernel on haversine distance between site
centers (same convention as extract_shanghai_od_matrix_final10.py's
build_adjacency), symmetric-normalized the standard GCN way:
A_norm = D^-1/2 (A + I) D^-1/2.

Saves into the SAME outputs/save_shanghai_demand_gru/<timestamp>/ layout
(config.json, test_prediction_log.npy, test_groundtruth_log.npy,
test_metrics.json) as train_shanghai_demand_gru.py, so
evaluate_shanghai_demand_gru.py works unchanged on T-GCN runs too --
just pass --run <this-run's-timestamp>.
"""
import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

BASE_DIR = Path(__file__).parent.parent


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="shanghai_demand_windows_kmeans30.npz")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batchsize", type=int, default=64)
    ap.add_argument("--epoch", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--adj-sigma-scale", type=float, default=1.0,
                     help="Gaussian kernel bandwidth = adj_sigma_scale * std(nonzero distances)")
    ap.add_argument("--top-k", type=int, default=5,
                     help="sparsify adjacency to each site's top_k nearest neighbors (None/0 = fully dense)")
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


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def build_normalized_adjacency(site_lat, site_lon, sigma_scale=1.0, top_k=5):
    """Gaussian kernel on distance, then sparsified to each site's top_k nearest
    neighbors only. A dense kernel (every site weakly connected to every other)
    over-smooths: with sigma_scale=1.0 and no sparsification, 80% of site pairs
    get a non-negligible edge weight, which blends each site's signal with ~24
    others every timestep and crushes site-specific peaks even harder than plain
    MSE does (empirically: top-5% peak bias went from -12% to -30%). Restricting
    to nearest neighbors keeps spatial mixing local instead of a global blur."""
    n = len(site_lat)
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        dist[i] = haversine_km(site_lat[i], site_lon[i], site_lat, site_lon)
    sigma = dist[dist > 0].std() * sigma_scale
    A = np.exp(-(dist ** 2) / (sigma ** 2))
    np.fill_diagonal(A, 0.0)

    if top_k is not None and top_k < n - 1:
        mask = np.zeros_like(A, dtype=bool)
        neighbor_idx = np.argsort(-A, axis=1)[:, :top_k]
        rows = np.repeat(np.arange(n), top_k)
        mask[rows, neighbor_idx.ravel()] = True
        mask = mask | mask.T  # keep an edge if either endpoint ranks it a top-k neighbor
        A = A * mask

    A_hat = A + np.eye(n)
    deg = A_hat.sum(axis=1)  # always > 0: every node has a self-loop
    D_inv_sqrt = np.diag(np.power(deg, -0.5))
    A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt
    return A_norm.astype(np.float32)


class GCNLayer(nn.Module):
    """Self + neighbor aggregation (GraphSAGE-style), not pure Laplacian smoothing:
    a separate `skip` transform of the site's own (un-mixed) features is added to
    the neighbor-mixed branch, so the model can learn to downweight spatial mixing
    for sites where it doesn't help instead of being forced through it."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.skip = nn.Linear(in_dim, out_dim)

    def forward(self, x, a_norm):  # x: (B, N, in_dim), a_norm: (N, N)
        neighbor = torch.matmul(a_norm, x)
        return self.linear(neighbor) + self.skip(x)


class TGCN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden, dropout, a_norm):
        super().__init__()
        self.register_buffer("a_norm", a_norm)
        self.gcn = GCNLayer(in_dim, hidden)
        self.act = nn.ReLU()
        self.gru = nn.GRU(hidden, hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):  # x: (B, T, N, C_in)
        b, t, n, c = x.shape
        x = x.reshape(b * t, n, c)
        x = self.act(self.gcn(x, self.a_norm))               # (B*T, N, hidden) -- spatial mixing per timestep
        x = x.reshape(b, t, n, -1).permute(0, 2, 1, 3).reshape(b * n, t, -1)  # (B*N, T, hidden)
        out, _ = self.gru(x)
        last = out[:, -1, :]
        return self.head(last)  # (B*N, out_dim)


def load_data(data_path):
    d = np.load(data_path)
    return {k: d[k] for k in d.files}


def make_loader(X, y, batchsize, shuffle, seed):
    X_out = X.astype(np.float32).copy()
    X_out[..., :2] = np.log1p(X_out[..., :2])
    y_log = np.log1p(y[:, 0]).astype(np.float32)
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X_out), torch.from_numpy(y_log))
    gen = torch.Generator()
    gen.manual_seed(seed)
    return torch.utils.data.DataLoader(ds, batchsize, shuffle=shuffle, generator=gen)


def run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    loss_sum, n_obs = 0.0, 0
    with torch.set_grad_enabled(train):
        for X, y in loader:
            X = X.to(device)  # (B, T, N, C_in)
            y = y.to(device)  # (B, N, C_out)
            b, n, c_out = y.shape
            y_flat = y.reshape(b * n, c_out)
            pred = model(X)
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
        for X, y in loader:
            X = X.to(device)
            b, n, c_out = y.shape
            pred = model(X).reshape(b, n, c_out).cpu().numpy()
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
    train_loader = make_loader(data["X_train"], data["y_train"], args.batchsize, True, args.seed)
    val_loader = make_loader(data["X_val"], data["y_val"], args.batchsize, False, args.seed)
    test_loader = make_loader(data["X_test"], data["y_test"], args.batchsize, False, args.seed)
    print("train/val/test samples:", len(data["X_train"]), len(data["X_val"]), len(data["X_test"]))

    in_dim = data["X_train"].shape[-1]
    out_dim = data["y_train"].shape[-1]
    top_k = args.top_k if args.top_k and args.top_k > 0 else None
    a_norm_np = build_normalized_adjacency(data["site_lat"], data["site_lon"], args.adj_sigma_scale, top_k)
    a_norm = torch.from_numpy(a_norm_np).to(device)
    print(f"in_dim={in_dim}  out_dim={out_dim}  n_sites={a_norm_np.shape[0]}  "
          f"adjacency nonzero frac (off-diag)={(a_norm_np > 1e-4).mean():.3f}")

    model = TGCN(in_dim=in_dim, out_dim=out_dim, hidden=args.hidden, dropout=args.dropout, a_norm=a_norm).to(device)
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
    config = vars(args) | {"model": "TGCN"}
    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    best_path = save_dir / "tgcn.pt"

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
