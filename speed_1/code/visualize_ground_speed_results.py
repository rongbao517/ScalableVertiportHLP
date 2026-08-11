# -*- coding: utf-8 -*-
"""
Reconstructed visualization script for the ground-speed forecasting experiment.

The 5 figures this reproduces (ground_speed_actual_vs_predicted.png,
_timeseries_only.png, _calibration_effect.png, _actual_vs_calibrated_only.png,
and ground_speed_overview.png) were originally produced by one-off inline code
in an earlier session that was never saved to a script file. This script
regenerates them from scratch, matching their original look as closely as
possible: reruns inference from the saved SiteGRU checkpoint (site_gru.pt) to
get the real test-set predictions (no numbers are hand-copied), fits the same
validation-only isotonic calibration calibrate_isotonic.py used for the demand
models, and reproduces the "dashboard" panel style used for 4 of the 5 figures.

Expects this folder layout (matches speed_1/):
  code/train_shanghai_speed_gru.py   -- SiteGRU class + data loaders (imported)
  data/shanghai_speed_windows.npz    -- lookback=48 -> horizon=1 windows
  data/shanghai_ground_speed_30min.csv
  data/shanghai_calendar_weather_202504.csv
  data/predicted_ground_speed_by_bucket.csv
  model_run/site_gru.pt, config.json

Run from inside speed_1/code/:  python3 visualize_ground_speed_results.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.isotonic import IsotonicRegression

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import train_shanghai_speed_gru as speed_mod

CODE_DIR = Path(__file__).parent
ROOT = CODE_DIR.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model_run"
FIGS_DIR = ROOT / "figures"
FIGS_DIR.mkdir(exist_ok=True)

# ---- palette / dashboard style, matched by eye to the original figures ----
BG = "#f7f6f4"          # page background
PANEL_BG = "#f2f0ed"    # chart-panel background
GRID_COLOR = "#dddad5"
TITLE_COLOR = "#1a1a1a"
SUBTITLE_COLOR = "#666666"
C_ACTUAL = "#2166dc"      # blue, solid
C_PRED_RAW = "#e2703a"    # orange, dashed
C_PRED_CAL = "#1f9c4a"    # green, solid


def dashboard_figure(n_panels=1, width=19, height=7.5):
    """Common chrome shared by the 4 'dashboard style' figures: off-white
    canvas, bold top-left title + grey subtitle, external line-sample legend,
    and light-grey panel background(s) with only horizontal gridlines."""
    fig = plt.figure(figsize=(width, height), facecolor=BG)
    if n_panels == 1:
        axes = [fig.add_axes([0.055, 0.08, 0.915, 0.56])]
    else:
        axes = [
            fig.add_axes([0.03, 0.08, 0.62, 0.56]),
            fig.add_axes([0.685, 0.08, 0.29, 0.56]),
        ]
    for ax in axes:
        ax.set_facecolor(PANEL_BG)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(colors=SUBTITLE_COLOR, length=0)
    return fig, axes


def add_title_subtitle_legend(fig, title, subtitle_lines, legend_entries):
    fig.text(0.03, 0.94, title, fontsize=23, fontweight="bold", color=TITLE_COLOR,
              family="sans-serif", ha="left", va="top")
    y0 = 0.855
    for i, line in enumerate(subtitle_lines):
        fig.text(0.03, y0 - i * 0.035, line, fontsize=12.5, color=SUBTITLE_COLOR,
                  family="sans-serif", ha="left", va="top")
    # hand-drawn legend row (line swatch + label), matplotlib legend() boxes
    # don't match the plain "swatch + text" row used in the originals
    x = 0.03
    y = 0.755
    for label, color, ls in legend_entries:
        fig.add_artist(plt.Line2D([x, x + 0.028], [y, y], color=color, lw=3,
                                   linestyle=ls, transform=fig.transFigure,
                                   solid_capstyle="round"))
        fig.text(x + 0.034, y, label, fontsize=13.5, color="#333333",
                  family="sans-serif", ha="left", va="center")
        x += 0.034 + 0.012 * len(label) + 0.10


def day_boundary_xticks(ax, n_days, bins_per_day, day_labels):
    for d in range(1, n_days):
        ax.axvline(d * bins_per_day, color="#bbbbbb", lw=0.9, linestyle=(0, (2, 2)), zorder=1)
    centers = [d * bins_per_day + bins_per_day / 2 for d in range(n_days)]
    ax.set_xticks(centers)
    ax.set_xticklabels(day_labels, fontsize=13, color=SUBTITLE_COLOR)
    ax.set_xlim(0, n_days * bins_per_day - 1)


def style_yaxis_kmh(ax, ymin, ymax, step, suffix=" km/h"):
    ticks = np.arange(ymin, ymax + 1e-9, step)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{int(t)}{suffix}" for t in ticks], fontsize=12.5, color=SUBTITLE_COLOR)
    ax.set_ylim(ymin, ymax)
    ax.grid(axis="y", color=GRID_COLOR, lw=1.1, zorder=0)
    ax.grid(axis="x", visible=False)


def get_test_predictions():
    """Rerun inference from the saved checkpoint -- test predictions are not
    hand-copied from anywhere, they come straight out of the trained model."""
    config = json.loads((MODEL_DIR / "config.json").read_text())
    mu, sd = config["speed_mu_kmh"], config["speed_sd_kmh"]

    data = np.load(DATA_DIR / "shanghai_speed_windows.npz")
    val_loader = speed_mod.make_loader(data["X_val"], data["y_val"], mu, sd, 64, False, config["seed"])
    test_loader = speed_mod.make_loader(data["X_test"], data["y_test"], mu, sd, 64, False, config["seed"])

    in_dim = data["X_train"].shape[-1]
    device = torch.device("cpu")
    model = speed_mod.SiteGRU(in_dim=in_dim, out_dim=1, hidden=config["hidden"],
                               layers=config["layers"], dropout=config["dropout"]).to(device)
    model.load_state_dict(torch.load(MODEL_DIR / "site_gru.pt", map_location=device))

    pred_val_z, true_val_z = speed_mod.predict(model, val_loader, device)
    pred_test_z, true_test_z = speed_mod.predict(model, test_loader, device)

    pred_val = (pred_val_z * sd + mu).clip(min=0).ravel()
    true_val = (true_val_z * sd + mu).ravel()
    pred_test = (pred_test_z * sd + mu).clip(min=0).ravel()
    true_test = (true_test_z * sd + mu).ravel()

    ts_test = pd.to_datetime(data["ts_test"].ravel())
    return pred_val, true_val, pred_test, true_test, ts_test


def calibrate(pred_val, true_val, pred_test):
    """Same recipe as calibrate_isotonic.py: fit isotonic pred->true on
    validation only, apply to test -- no test-set leakage into the fit."""
    iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
    iso.fit(pred_val, true_val)
    return iso.predict(pred_test)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def metrics_block(true, pred):
    nz = true > 1e-6
    r2 = 1 - np.sum((true[nz] - pred[nz]) ** 2) / np.sum((true[nz] - true[nz].mean()) ** 2)
    mae = np.mean(np.abs(pred[nz] - true[nz]))
    mape = np.mean(np.abs((pred[nz] - true[nz]) / true[nz])) * 100
    return dict(R2=r2, RMSE=rmse(true, pred), MAE=mae, MAPE=mape)


# ---------------------------------------------------------------------------
# Figure A: ground_speed_actual_vs_predicted.png (timeseries + scatter)
# ---------------------------------------------------------------------------
def fig_actual_vs_predicted(true_test, pred_test):
    fig, (ax_ts, ax_sc) = dashboard_figure(n_panels=2)
    add_title_subtitle_legend(
        fig, "Ground-Speed Forecast — Actual vs. Predicted (held-out test set)",
        ["GRU speed model used to build the 48-bucket (hour_of_day, day_type) lookup that feeds access/egress time in the inner-loop route assignment.",
         "City-wide 30-min series, 24 train days / 3 val / 3 test (chronological split). Test-set predictions shown here, de-normalized to km/h."],
        [("Actual (ground truth)", C_ACTUAL, "-"), ("Predicted (GRU)", C_PRED_RAW, "--")],
    )

    x = np.arange(len(true_test))
    ax_ts.set_title("Ground speed over the held-out test period (3 days, 144 bins)",
                     loc="left", fontsize=14, fontweight="bold", color=TITLE_COLOR, pad=10)
    ax_ts.plot(x, true_test, color=C_ACTUAL, lw=2.2, zorder=3)
    ax_ts.plot(x, pred_test, color=C_PRED_RAW, lw=1.8, linestyle="--", zorder=3)
    ymin, ymax = 11, 32
    style_yaxis_kmh(ax_ts, ymin, ymax, 5, suffix="")
    ax_ts.set_xticks([0, 48, 95, 143])
    ax_ts.set_xticklabels(["day 1, bin 0", "day 2, bin 0", "day 2, bin 47", "day 3, bin 47"],
                           fontsize=11.5, color=SUBTITLE_COLOR)
    for d in (48, 96):
        ax_ts.axvline(d, color="#cccccc", lw=0.8, linestyle=(0, (2, 2)))

    m = metrics_block(true_test, pred_test)
    ax_sc.set_title("Predicted vs. actual (test set)", loc="left", fontsize=14,
                     fontweight="bold", color=TITLE_COLOR, pad=10)
    ax_sc.scatter(true_test, pred_test, s=22, color=C_ACTUAL, alpha=0.45, edgecolor="none", zorder=3)
    lo, hi = 12, 32
    ax_sc.plot([lo, hi], [lo, hi], color="#888888", lw=1.2, linestyle=(0, (4, 3)), zorder=2)
    ax_sc.set_xlim(lo, hi); ax_sc.set_ylim(lo, hi)
    ticks = np.arange(lo, hi + 1, 5)
    ax_sc.set_xticks(ticks); ax_sc.set_yticks(ticks)
    ax_sc.set_xticklabels([str(t) for t in ticks], fontsize=12, color=SUBTITLE_COLOR)
    ax_sc.set_yticklabels([str(t) for t in ticks], fontsize=12, color=SUBTITLE_COLOR)
    ax_sc.set_xlabel("actual speed (km/h)", fontsize=12, color=SUBTITLE_COLOR)
    ax_sc.grid(color=GRID_COLOR, lw=1.0, zorder=0)
    ax_sc.text(0.05, 0.93, f"R² = {m['R2']:.3f}", transform=ax_sc.transAxes,
               fontsize=13.5, fontweight="bold", color="#222222", va="top")
    ax_sc.text(0.05, 0.86,
               f"RMSE = {m['RMSE']:.2f} km/h\nMAE = {m['MAE']:.2f} km/h\nMAPE = {m['MAPE']:.1f}%",
               transform=ax_sc.transAxes, fontsize=12.5, color="#444444", va="top")

    fig.savefig(FIGS_DIR / "ground_speed_actual_vs_predicted.png", dpi=150, facecolor=BG)
    plt.close(fig)
    return m


# ---------------------------------------------------------------------------
# Figure B: ground_speed_actual_vs_predicted_timeseries_only.png
# ---------------------------------------------------------------------------
def fig_timeseries_only(true_test, pred_test):
    fig, (ax,) = dashboard_figure(n_panels=1, width=19, height=8)
    add_title_subtitle_legend(
        fig, "Ground-Speed Forecast — Actual vs. Predicted (held-out test set)",
        ["GRU speed model used to build the 48-bucket (hour_of_day, day_type) lookup that feeds access/egress time in the inner-loop route assignment.",
         "30-day dataset, chronological split: 24 train days / 3 validation / 3 test. Shown here: the 3 held-out test days the model never saw during training."],
        [("Actual (ground truth)", C_ACTUAL, "-"), ("Predicted (GRU)", C_PRED_RAW, "--")],
    )
    x = np.arange(len(true_test))
    ax.plot(x, true_test, color=C_ACTUAL, lw=2.6, zorder=3)
    ax.plot(x, pred_test, color=C_PRED_RAW, lw=2.0, linestyle="--", zorder=3)
    style_yaxis_kmh(ax, 11, 32, 4)
    day_boundary_xticks(ax, 3, 48, ["test day 1", "test day 2", "test day 3"])
    fig.savefig(FIGS_DIR / "ground_speed_actual_vs_predicted_timeseries_only.png", dpi=150, facecolor=BG)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure C: ground_speed_calibration_effect.png (actual + raw + calibrated)
# ---------------------------------------------------------------------------
def fig_calibration_effect(true_test, pred_test, pred_test_cal, rmse_raw, rmse_cal):
    hi_mask = true_test >= 24
    rmse_raw_hi = rmse(true_test[hi_mask], pred_test[hi_mask])
    rmse_cal_hi = rmse(true_test[hi_mask], pred_test_cal[hi_mask])

    fig, (ax,) = dashboard_figure(n_panels=1, width=19, height=8.5)
    add_title_subtitle_legend(
        fig, "Ground-Speed Forecast — Effect of Isotonic Calibration (test set)",
        ["Calibration fit on the validation set only (no test leakage), then applied to test predictions.",
         f"Test RMSE: {rmse_raw:.3f} → {rmse_cal:.3f} km/h overall; high-speed segment (≥24 km/h) RMSE: {rmse_raw_hi:.3f} → {rmse_cal_hi:.3f} km/h."],
        [("Actual (ground truth)", C_ACTUAL, "-"), ("Predicted (raw GRU)", C_PRED_RAW, "--"),
         ("Predicted (calibrated)", C_PRED_CAL, "-")],
    )
    x = np.arange(len(true_test))
    ax.plot(x, true_test, color=C_ACTUAL, lw=2.6, zorder=4)
    ax.plot(x, pred_test, color=C_PRED_RAW, lw=1.6, linestyle="--", zorder=3)
    ax.plot(x, pred_test_cal, color=C_PRED_CAL, lw=2.0, zorder=3)
    style_yaxis_kmh(ax, 11, 32, 4)
    day_boundary_xticks(ax, 3, 48, ["test day 1", "test day 2", "test day 3"])
    fig.savefig(FIGS_DIR / "ground_speed_calibration_effect.png", dpi=150, facecolor=BG)
    plt.close(fig)
    return rmse_raw_hi, rmse_cal_hi


# ---------------------------------------------------------------------------
# Figure D: ground_speed_actual_vs_calibrated_only.png
# ---------------------------------------------------------------------------
def fig_actual_vs_calibrated(true_test, pred_test_cal, rmse_cal, rmse_cal_hi):
    fig, (ax,) = dashboard_figure(n_panels=1, width=19, height=8.5)
    add_title_subtitle_legend(
        fig, "Ground-Speed Forecast — Actual vs. Calibrated Prediction (test set)",
        ["Calibration fit on the validation set only (no test leakage), then applied to test predictions.",
         f"Test RMSE (calibrated): {rmse_cal:.3f} km/h overall; high-speed segment (≥24 km/h): {rmse_cal_hi:.3f} km/h."],
        [("Actual (ground truth)", C_ACTUAL, "-"), ("Predicted (calibrated)", C_PRED_CAL, "--")],
    )
    x = np.arange(len(true_test))
    ax.plot(x, true_test, color=C_ACTUAL, lw=2.6, zorder=3)
    ax.plot(x, pred_test_cal, color=C_PRED_CAL, lw=2.0, linestyle="--", zorder=3)
    style_yaxis_kmh(ax, 11, 32, 4)
    day_boundary_xticks(ax, 3, 48, ["test day 1", "test day 2", "test day 3"])
    fig.savefig(FIGS_DIR / "ground_speed_actual_vs_calibrated_only.png", dpi=150, facecolor=BG)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure E: ground_speed_overview.png (plain matplotlib style, 2 panels)
# ---------------------------------------------------------------------------
def fig_overview():
    speed_df = pd.read_csv(DATA_DIR / "shanghai_ground_speed_30min.csv", parse_dates=["timestamp"])
    cal = pd.read_csv(DATA_DIR / "shanghai_calendar_weather_202504.csv", parse_dates=["date"])
    bucket = pd.read_csv(DATA_DIR / "predicted_ground_speed_by_bucket.csv")

    cal["is_offday"] = cal["day_type"].ne("workday")
    speed_df["date_only"] = speed_df["timestamp"].dt.normalize()
    date_to_offday = dict(zip(cal["date"], cal["is_offday"]))

    plt.style.use("default")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 11))

    ax1.plot(speed_df["timestamp"], speed_df["median_speed_kmh"], color="#1f77b4", lw=1.1)
    for d, is_off in date_to_offday.items():
        if is_off:
            ax1.axvspan(d, d + pd.Timedelta(days=1), color="#ffcc80", alpha=0.35, lw=0)
    ax1.set_title("City-wide median ground speed -- real value per 30-min bin, full month\n"
                   "(orange shading = offday; gap on 04-18 = duplicate raw file, excluded)")
    ax1.set_xlabel("date"); ax1.set_ylabel("speed (km/h)")
    ax1.grid(alpha=0.4)

    speed_df["hour_of_day"] = speed_df["timestamp"].dt.hour
    speed_df["is_offday"] = speed_df["date_only"].map(date_to_offday)
    diurnal = speed_df.groupby(["hour_of_day", "is_offday"])["median_speed_kmh"].mean().unstack()

    bucket_p = bucket.copy()
    bucket_p["is_offday"] = bucket_p["day_type"].eq("offday")
    bucket_diurnal = bucket_p.groupby(["hour_of_day", "is_offday"])["predicted_speed_kmh"].mean().unstack()

    hours = diurnal.index.to_numpy()
    ax2.plot(hours, diurnal[False], "o-", color="#d62728", label="ground truth (workday, monthly avg)")
    ax2.plot(hours, diurnal[True], "o-", color="#1f77b4", label="ground truth (offday, monthly avg)")
    ax2.plot(bucket_diurnal.index, bucket_diurnal[False], "x--", color="#f4978e",
             label="model-predicted bucket table (workday)")
    ax2.plot(bucket_diurnal.index, bucket_diurnal[True], "x--", color="#a9cce3",
             label="model-predicted bucket table (offday)")
    ax2.set_title("Hourly speed pattern: ground truth (monthly avg) vs model-predicted 48-bucket table -- close match")
    ax2.set_xlabel("hour of day"); ax2.set_ylabel("speed (km/h)")
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.4)

    fig.tight_layout()
    fig.savefig(FIGS_DIR / "ground_speed_overview.png", dpi=150)
    plt.close(fig)


def main():
    pred_val, true_val, pred_test, true_test, ts_test = get_test_predictions()
    pred_test_cal = calibrate(pred_val, true_val, pred_test)

    m_raw = fig_actual_vs_predicted(true_test, pred_test)
    print("actual_vs_predicted metrics:", m_raw)

    fig_timeseries_only(true_test, pred_test)

    rmse_raw = rmse(true_test, pred_test)
    rmse_cal = rmse(true_test, pred_test_cal)
    rmse_raw_hi, rmse_cal_hi = fig_calibration_effect(true_test, pred_test, pred_test_cal, rmse_raw, rmse_cal)
    print(f"calibration effect: overall {rmse_raw:.3f} -> {rmse_cal:.3f} km/h, "
          f"high-speed(>=24) {rmse_raw_hi:.3f} -> {rmse_cal_hi:.3f} km/h")

    fig_actual_vs_calibrated(true_test, pred_test_cal, rmse_cal, rmse_cal_hi)

    fig_overview()

    print(f"saved 5 figures -> {FIGS_DIR}")


if __name__ == "__main__":
    main()
