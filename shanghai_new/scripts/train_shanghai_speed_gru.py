# -*- coding: utf-8 -*-
"""
Ground-speed forecasting model, mirroring train_shanghai_demand_gru.py's
SiteGRU architecture and train/eval conventions almost unchanged (in_dim/out_dim
read from the npz shape at runtime, so the class itself needs no modification).

Trained on data/shanghai_speed_windows.npz (build_speed_windows.py): one
city-wide 30-min ground-speed series (not per-vertiport, hence n_sites=1),
lookback=48 steps -> horizon=1 step, plus weather/day-type context channels.

Deliberate deviation from the demand pipeline: speed is NOT log1p'd (it isn't
a count) -- targets are z-scored instead, using TRAIN-period mean/std, and
predictions are un-z-scored back to km/h for evaluation/downstream use.
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
    ap.add_argument("--data", type=str, default="shanghai_speed_windows.npz")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batchsize", type=int, default=64)
    ap.add_argument("--epoch", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--gpu", type=int, default=0)
    return ap.parse_args()


class SiteGRU(nn.Module):
    """Unchanged from train_shanghai_demand_gru.py -- in_dim/out_dim are set from
    data shape at call time, so this class needs no modification for a new target."""
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
        return self.head(last)


def load_data(data_path):
    d = np.load(data_path)
    return {k: d[k] for k in d.files}


def make_loader(X, y, mu, sd, batchsize, shuffle, seed):
    # channel 0 is the speed value (z-scored); context channels (weather/day-type)
    # are already standardized at extraction time.
    X_out = X.astype(np.float32).copy()
    X_out[..., 0] = (X_out[..., 0] - mu) / sd
    y_z = ((y[:, 0, :, 0] - mu) / sd).astype(np.float32)[..., None]  # (B, N=1, 1), horizon squeezed
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X_out), torch.from_numpy(y_z))
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

    # z-score stats from TRAIN speed values only (avoid leaking val/test stats)
    mu = float(data["X_train"][..., 0].mean())
    sd = float(data["X_train"][..., 0].std())
    print(f"train speed mean={mu:.3f} std={sd:.3f} km/h")

    train_loader = make_loader(data["X_train"], data["y_train"], mu, sd, args.batchsize, True, args.seed)
    val_loader = make_loader(data["X_val"], data["y_val"], mu, sd, args.batchsize, False, args.seed)
    test_loader = make_loader(data["X_test"], data["y_test"], mu, sd, args.batchsize, False, args.seed)
    print("train/val/test samples:", len(data["X_train"]), len(data["X_val"]), len(data["X_test"]))

    in_dim = data["X_train"].shape[-1]
    out_dim = data["y_train"].shape[-1]
    print(f"in_dim={in_dim}  out_dim={out_dim}  feature_names={list(data.get('feature_names', []))}")

    model = SiteGRU(in_dim=in_dim, out_dim=out_dim, hidden=args.hidden, layers=args.layers,
                     dropout=args.dropout).to(device)
    print("params:", sum(p.numel() for p in model.parameters()))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    save_dir = BASE_DIR / "outputs" / "save_shanghai_speed_gru" / f"{datetime.now().strftime('%y%m%d%H%M%S')}_{os.getpid()}"
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "config.json", "w") as f:
        json.dump(vars(args) | {"speed_mu_kmh": mu, "speed_sd_kmh": sd}, f, indent=2)
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
    pred_z, true_z = predict(model, test_loader, device)
    np.save(save_dir / "test_prediction_z.npy", pred_z)
    np.save(save_dir / "test_groundtruth_z.npy", true_z)

    pred_kmh = pred_z * sd + mu
    true_kmh = true_z * sd + mu
    pred_kmh = np.clip(pred_kmh, 0, None)

    mse_z = np.mean((pred_z - true_z) ** 2)
    rmse_z = np.sqrt(mse_z)
    mae_z = np.mean(np.abs(pred_z - true_z))
    mape_z = np.mean(np.abs((pred_z - true_z) / true_z)) * 100
    metrics_z = dict(MSE=float(mse_z), RMSE=float(rmse_z), MAE=float(mae_z), MAPE=float(mape_z))
    print("test metrics (z-score space):", metrics_z)

    mse = np.mean((pred_kmh - true_kmh) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(pred_kmh - true_kmh))
    mape = np.mean(np.abs((pred_kmh - true_kmh) / true_kmh)) * 100
    r2 = 1 - np.sum((true_kmh - pred_kmh) ** 2) / np.sum((true_kmh - true_kmh.mean()) ** 2)
    pcc = np.corrcoef(true_kmh.flatten(), pred_kmh.flatten())[0, 1]
    metrics = dict(MSE=float(mse), RMSE=float(rmse), MAE=float(mae), MAPE=float(mape), R2=float(r2), PCC=float(pcc))
    print("test metrics (km/h space):", metrics)
    with open(save_dir / "test_metrics.json", "w") as f:
        json.dump({"z_space": metrics_z, "kmh_space": metrics, "speed_mu_kmh": mu, "speed_sd_kmh": sd}, f, indent=2)

    print("saved run ->", save_dir)


if __name__ == "__main__":
    main()
