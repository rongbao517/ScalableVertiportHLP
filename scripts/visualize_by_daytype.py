# -*- coding: utf-8 -*-
"""
True vs predicted demand, split by day type (workday / weekend / holiday).

Data limitation: the chronological test split (04-28..04-30) is 3
consecutive workdays -- there is no weekend or holiday day in the held-out
test set, so there are no genuinely out-of-sample predictions for those
day types. Workaround: run the already-trained model's inference over
train+val+test concatenated (reconstructing predictions for ~the whole
month), then group by day_type from shanghai_calendar_weather_202504.csv.
Train-period days are IN-SAMPLE (the model was fit on them) so their
accuracy is optimistic vs true generalization -- each panel is annotated
with how many of its days are in-sample vs out-of-sample (test) so this
is not mistaken for a fair holdout comparison.

Averages the diurnal profile (mean over all days of that day_type, per
30-min bin-of-day) rather than plotting raw dates, since workday(n=21) /
weekend(n=8) / holiday(n=2) have very different day counts and individual
holiday days would otherwise be invisible next to 21 workdays overlaid.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import train_shanghai_demand_stid as stid_mod

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs"
FIGS_DIR = OUT_DIR / "figs"
CALENDAR_CSV = DATA_DIR / "shanghai_calendar_weather_202504.csv"
BINS_PER_DAY = 48


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, required=True)
    ap.add_argument("--fig-prefix", type=str, default="daytype")
    args = ap.parse_args()

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    run_dir = OUT_DIR / "save_shanghai_demand_gru" / args.run
    config = json.loads((run_dir / "config.json").read_text())
    assert config.get("model") == "STID", "this script assumes an STID run (needs tod/dow inputs)"

    data = np.load(DATA_DIR / config["data"], allow_pickle=True)
    in_dim = data["X_train"].shape[-1]
    out_dim = data["y_train"].shape[-1]
    lookback = data["X_train"].shape[1]
    n_sites = data["X_train"].shape[2]

    model = stid_mod.STID(in_dim=in_dim, out_dim=out_dim, lookback=lookback, n_sites=n_sites,
                           n_tod=BINS_PER_DAY, n_dow=7, hidden=config["hidden"], embed_dim=config["embed_dim"],
                           n_layers=config["n_layers"], dropout=config["dropout"]).to(device)
    model.load_state_dict(torch.load(run_dir / "stid.pt", map_location=device))

    preds, trues, ts_all, split_tag = [], [], [], []
    for split in ["train", "val", "test"]:
        loader = stid_mod.make_loader(data[f"X_{split}"], data[f"y_{split}"], data[f"ts_{split}"],
                                       64, False, config["seed"])
        pred_log, true_log = stid_mod.predict(model, loader, device)
        preds.append(np.expm1(pred_log))
        trues.append(np.expm1(true_log))
        ts_all.append(pd.to_datetime(data[f"ts_{split}"]))
        split_tag += [split] * len(data[f"ts_{split}"])

    pred = np.concatenate(preds)   # (n_samples, n_sites, 2)
    true = np.concatenate(trues)
    ts = pd.DatetimeIndex(np.concatenate(ts_all))
    split_tag = np.array(split_tag)
    print(f"reconstructed {len(ts)} bins spanning {ts.min()} .. {ts.max()}  "
          f"(train={sum(split_tag=='train')}, val={sum(split_tag=='val')}, test={sum(split_tag=='test')})")

    cal = pd.read_csv(CALENDAR_CSV, parse_dates=["date"])
    date_to_type = dict(zip(cal["date"].dt.date, cal["day_type"]))
    day_type = np.array([date_to_type.get(t.date(), "unknown") for t in ts])
    bin_of_day = (ts.hour * 2 + ts.minute // 30).to_numpy()

    agg_true = true.sum(axis=1)   # (n_samples, 2) -- summed over 30 sites
    agg_pred = pred.sum(axis=1)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), dpi=150, sharey="row")
    for row, (ci, label) in enumerate(zip([0, 1], ["origin_demand", "destination_demand"])):
        for col, dtype in enumerate(["workday", "weekend", "holiday"]):
            ax = axes[row, col]
            mask = day_type == dtype
            n_days = len(np.unique(ts[mask].date))
            n_train_days = len(np.unique(ts[mask & (split_tag == "train")].date))
            n_test_days = n_days - n_train_days

            profile_true = np.full(BINS_PER_DAY, np.nan)
            profile_pred = np.full(BINS_PER_DAY, np.nan)
            for b in range(BINS_PER_DAY):
                bmask = mask & (bin_of_day == b)
                if bmask.any():
                    profile_true[b] = agg_true[bmask, ci].mean()
                    profile_pred[b] = agg_pred[bmask, ci].mean()

            hours = np.arange(BINS_PER_DAY) / 2
            ax.plot(hours, profile_true, marker="o", ms=3, lw=1.3, label="true")
            ax.plot(hours, profile_pred, marker="x", ms=4, lw=1.3, label="predicted")
            tag = f"{dtype} (n={n_days}d: {n_train_days} in-sample, {n_test_days} held-out)"
            ax.set_title(tag, fontsize=9)
            ax.set_xlabel("hour of day")
            if col == 0:
                ax.set_ylabel(f"{label}\n(mean trips/30min, summed over {n_sites} sites)")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)

    plt.tight_layout()
    out_png = FIGS_DIR / f"{args.fig_prefix}_true_vs_pred.png"
    plt.savefig(out_png)
    print("saved:", out_png)


if __name__ == "__main__":
    main()
