# -*- coding: utf-8 -*-
"""
Purpose-built demand *regression* model for the 30 NSGA-selected Shanghai
sites, trained on shanghai_demand_windows.npz (per-site origin/destination
demand, lookback=48 steps -> horizon=1 step, 30-min bins).

This replaces the UAM_demand OD-matrix model (uam_demand_model/), which
underfit badly (test R2=0.19) because it predicts a 30x30 sparse OD matrix
from only ~1000 samples -- far too little data for that architecture
(designed for a full year of NYC data with 11 weather channels).

Also NOT reusing willey revise's DualRNN/AGG-DGRU/STGCN/GAT: those are
binary *classifiers* (congestion class 0/1, CrossEntropyLoss, num_classes=2)
built for the site-selection pipeline, not demand regressors -- their
trained weights don't transfer to this task.

Architecture: a single shared-weight (Bi)GRU applied independently to each
of the 30 sites (site dimension folded into the batch), predicting the next
30-min (origin_demand, destination_demand) in log1p space. Small and cheap
enough to fit our ~1000-sample dataset without the OD model's underfitting.
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
    ap.add_argument("--data", type=str, default="shanghai_demand_windows.npz",
                     help="windows npz filename, relative to data/ (e.g. shanghai_demand_windows_kmeans30.npz)")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batchsize", type=int, default=64)
    ap.add_argument("--epoch", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--loss", type=str, default="mse", choices=["mse", "weighted_mse", "quantile"],
                     help="weighted_mse upweights samples with larger true demand, to counteract "
                          "the systematic peak-underestimation bias of plain MSE in log1p space "
                          "(minimizing squared log-error targets the conditional geometric mean, "
                          "which is a downward-biased estimate of the arithmetic mean -- worse the "
                          "larger/more variable the true value). quantile trains the model to "
                          "predict the --tau-th conditional quantile instead of the mean (pinball "
                          "loss); tau>0.5 costs underestimation more than overestimation.")
    ap.add_argument("--tau", type=float, default=0.7, help="quantile for --loss quantile (0.5=median/symmetric)")
    return ap.parse_args()


def weighted_mse_loss(pred, target):
    """target is log1p(true_demand). Weight by the ORIGINAL-scale magnitude (recovered via
    expm1, no extra tensors needed since log1p is invertible) so peaks -- rare but large --
    pull the gradient proportionally to their size instead of being outvoted by the much more
    numerous small/zero bins. Weights are normalized to mean 1 per batch so the overall loss
    scale (and therefore effective learning rate) stays comparable to plain MSE."""
    weight = torch.expm1(target).clamp(min=0) + 1.0
    weight = weight / weight.mean().clamp(min=1e-6)
    return (weight * (pred - target) ** 2).mean()


def make_quantile_loss(tau):
    """Pinball loss in log1p space. log1p is monotonic, so the tau-th quantile of
    log1p(demand) is exactly log1p(the tau-th quantile of demand) -- training on the
    log-space target with this loss still recovers the correct original-scale quantile
    after expm1, no extra correction needed."""
    def loss_fn(pred, target):
        diff = target - pred
        return torch.maximum(tau * diff, (tau - 1) * diff).mean()
    return loss_fn


class SiteGRU(nn.Module):
    def __init__(self, in_dim=2, out_dim=2, hidden=64, layers=1, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(
            in_dim, hidden, num_layers=layers, batch_first=True,
            bidirectional=True, dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):  # x: (B*N, T, C)
        out, _ = self.gru(x)
        last = out[:, -1, :]
        return self.head(last)  # (B*N, C), can be negative -- caller clamps


def load_data(data_path):
    d = np.load(data_path)
    return {k: d[k] for k in d.files}


def to_site_batch(x):
    """(B, T, N, C) -> (B*N, T, C)"""
    b, t, n, c = x.shape
    return x.transpose(0, 2, 1, 3).reshape(b * n, t, c)


def make_loader(X, y, batchsize, shuffle, seed):
    # only the first 2 channels are demand counts (log1p-able); any extra context
    # channels (weather/day-type) are already standardized at extraction time.
    X_out = X.astype(np.float32).copy()
    X_out[..., :2] = np.log1p(X_out[..., :2])
    y_log = np.log1p(y[:, 0]).astype(np.float32)  # (B, N, 2), horizon=1 squeezed
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
            b, t, n, c_in = X.shape
            c_out = y.shape[-1]
            X_site = X.permute(0, 2, 1, 3).reshape(b * n, t, c_in)
            y_site = y.reshape(b * n, c_out)
            pred = model(X_site)
            loss = criterion(pred, y_site)
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
            b, t, n, c_in = X.shape
            c_out = y.shape[-1]
            X_site = X.permute(0, 2, 1, 3).reshape(b * n, t, c_in)
            pred = model(X_site).reshape(b, n, c_out).cpu().numpy()
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
    print(f"in_dim={in_dim}  out_dim={out_dim}  feature_names={list(data.get('feature_names', []))}")

    model = SiteGRU(in_dim=in_dim, out_dim=out_dim, hidden=args.hidden, layers=args.layers,
                     dropout=args.dropout).to(device)
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
    with open(save_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    best_path = save_dir / "site_gru.pt"

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
    mse = np.mean((pred[nz] - true[nz]) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(pred[nz] - true[nz]))
    mape = np.mean(np.abs((pred[nz] - true[nz]) / true[nz])) * 100
    r2 = 1 - np.sum((true[nz] - pred[nz]) ** 2) / np.sum((true[nz] - true[nz].mean()) ** 2)
    pcc = np.corrcoef(true[nz].flatten(), pred[nz].flatten())[0, 1]
    metrics = dict(MSE=float(mse), RMSE=float(rmse), MAE=float(mae), MAPE=float(mape), R2=float(r2), PCC=float(pcc))
    print("test metrics (original space):", metrics)
    with open(save_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("saved run ->", save_dir)


if __name__ == "__main__":
    main()
