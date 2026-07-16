# -*- coding: utf-8 -*-
"""
Post-hoc peak-bias calibration, as a surgical alternative to changing the
training loss (weighted_mse / quantile): both of those turned out to shift
predictions upward broadly (61-63% of ALL bins over-predicted, not just
peaks -- see conversation), because the correction is learned into the
SAME shared network that also produces the low/mid-range predictions.

This instead: (1) trains normally with plain MSE (best-calibrated fit
everywhere except peaks), (2) fits an isotonic regression true~predicted
on the VALIDATION set only, per channel, (3) applies that mapping to test
predictions. Isotonic regression is monotonic and non-parametric -- it
will sit near the identity line wherever validation shows pred≈true
(the low/mid range, left untouched) and only bend upward where validation
shows a systematic gap (the peak region), without needing a hand-picked
threshold or touching the model at all.

Reloads a trained run's config.json + checkpoint (works for both
train_shanghai_demand_gru.py's SiteGRU and train_shanghai_demand_tgcn.py's
TGCN, dispatched by config["model"]).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression

import train_shanghai_demand_gru as gru_mod
import train_shanghai_demand_tgcn as tgcn_mod
import train_shanghai_demand_stid as stid_mod

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs"


def rebuild_model_and_loaders(run_dir, device):
    config = json.loads((run_dir / "config.json").read_text())
    model_type = config.get("model", "SiteGRU")
    data = np.load(DATA_DIR / config["data"], allow_pickle=True)

    in_dim = data["X_train"].shape[-1]
    out_dim = data["y_train"].shape[-1]

    if model_type == "TGCN":
        mod = tgcn_mod
        val_loader = mod.make_loader(data["X_val"], data["y_val"], 64, False, config["seed"])
        test_loader = mod.make_loader(data["X_test"], data["y_test"], 64, False, config["seed"])
        top_k = config["top_k"] if config.get("top_k") else None
        a_norm_np = tgcn_mod.build_normalized_adjacency(data["site_lat"], data["site_lon"],
                                                         config["adj_sigma_scale"], top_k)
        a_norm = torch.from_numpy(a_norm_np).to(device)
        model = tgcn_mod.TGCN(in_dim=in_dim, out_dim=out_dim, hidden=config["hidden"],
                               dropout=config["dropout"], a_norm=a_norm).to(device)
        ckpt = run_dir / "tgcn.pt"
    elif model_type == "STID":
        mod = stid_mod
        val_loader = mod.make_loader(data["X_val"], data["y_val"], data["ts_val"], 64, False, config["seed"])
        test_loader = mod.make_loader(data["X_test"], data["y_test"], data["ts_test"], 64, False, config["seed"])
        lookback = data["X_train"].shape[1]
        n_sites = data["X_train"].shape[2]
        model = stid_mod.STID(in_dim=in_dim, out_dim=out_dim, lookback=lookback, n_sites=n_sites,
                               n_tod=stid_mod.BINS_PER_DAY, n_dow=7, hidden=config["hidden"],
                               embed_dim=config["embed_dim"], n_layers=config["n_layers"],
                               dropout=config["dropout"]).to(device)
        ckpt = run_dir / "stid.pt"
    else:
        mod = gru_mod
        val_loader = mod.make_loader(data["X_val"], data["y_val"], 64, False, config["seed"])
        test_loader = mod.make_loader(data["X_test"], data["y_test"], 64, False, config["seed"])
        model = gru_mod.SiteGRU(in_dim=in_dim, out_dim=out_dim, hidden=config["hidden"],
                                 layers=config["layers"], dropout=config["dropout"]).to(device)
        ckpt = run_dir / "site_gru.pt"

    model.load_state_dict(torch.load(ckpt, map_location=device))
    return model, mod, val_loader, test_loader, config


def layered_bias(true, pred, label):
    order = np.argsort(true)
    n = len(true)
    overall_bias = (pred - true).mean()
    frac_over = (pred > true).mean() * 100
    print(f"  [{label}] overall_bias={overall_bias:+.2f}  frac_overpredicted={frac_over:.1f}%")
    for lo, hi, name in [(0, 0.5, "bottom50%"), (0.5, 0.9, "mid40%"), (0.9, 1.0, "top10%")]:
        idx = order[int(n * lo):int(n * hi)]
        b = pred[idx] - true[idx]
        rel = b.mean() / max(true[idx].mean(), 1e-6) * 100
        print(f"      {name:10s} n={len(idx):5d}  mean_true={true[idx].mean():7.1f}  "
              f"bias={b.mean():+6.2f} ({rel:+.1f}%)  frac_over={100*(b>0).mean():.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, required=True, help="run dir name under outputs/save_shanghai_demand_gru/")
    args = ap.parse_args()

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    run_dir = OUT_DIR / "save_shanghai_demand_gru" / args.run

    model, mod, val_loader, test_loader, config = rebuild_model_and_loaders(run_dir, device)
    print(f"loaded run {args.run}  model={config.get('model', 'SiteGRU')}  data={config['data']}")

    pred_val_log, true_val_log = mod.predict(model, val_loader, device)
    pred_test_log, true_test_log = mod.predict(model, test_loader, device)

    pred_val = np.expm1(pred_val_log); pred_val[pred_val < 0] = 0
    true_val = np.expm1(true_val_log)
    pred_test = np.expm1(pred_test_log); pred_test[pred_test < 0] = 0
    true_test = np.expm1(true_test_log)

    channel_names = ["origin_demand", "destination_demand"]
    pred_test_calibrated = pred_test.copy()
    for c, name in enumerate(channel_names):
        iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
        iso.fit(pred_val[..., c].ravel(), true_val[..., c].ravel())
        pred_test_calibrated[..., c] = iso.predict(pred_test[..., c].ravel()).reshape(pred_test[..., c].shape)
        print(f"channel={name}  val fit range: pred[{pred_val[...,c].min():.1f},{pred_val[...,c].max():.1f}] "
              f"-> calibrated test range now [{pred_test_calibrated[...,c].min():.1f},{pred_test_calibrated[...,c].max():.1f}]")

    print("\n=== BEFORE calibration ===")
    layered_bias(true_test.flatten(), pred_test.flatten(), "before")
    print("\n=== AFTER isotonic calibration (fit on val, applied to test) ===")
    layered_bias(true_test.flatten(), pred_test_calibrated.flatten(), "after")

    def metrics(true, pred):
        nz = true > 1e-5
        mse = np.mean((pred[nz] - true[nz]) ** 2)
        r2 = 1 - np.sum((true[nz] - pred[nz]) ** 2) / np.sum((true[nz] - true[nz].mean()) ** 2)
        pcc = np.corrcoef(true[nz].flatten(), pred[nz].flatten())[0, 1]
        mae = np.mean(np.abs(pred[nz] - true[nz]))
        return dict(R2=float(r2), PCC=float(pcc), MAE=float(mae), RMSE=float(np.sqrt(mse)))

    print("\nBEFORE:", metrics(true_test, pred_test))
    print("AFTER: ", metrics(true_test, pred_test_calibrated))

    out_pred_log = np.log1p(pred_test_calibrated).astype(np.float32)
    np.save(run_dir / "test_prediction_log_calibrated.npy", out_pred_log)
    print(f"\nsaved calibrated predictions -> {run_dir / 'test_prediction_log_calibrated.npy'}")


if __name__ == "__main__":
    main()
