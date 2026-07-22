# -*- coding: utf-8 -*-
"""
Final report generator (2026-07-22): produces every headline number in the
LP-vs-Tier0 comparison directly from the underlying per-scenario CSVs --
no manually-transcribed figures. Run this whenever the underlying simulation
outputs change; every percentage/delta below is recomputed, never hardcoded.

Inputs (already-generated simulation outputs, not reproduced here):
    outputs/fleet_sim/time_step_summary_{tag}.csv   -- met/unmet decomposition
    outputs/fleet_sim/rebalance_log_{tag}.csv        -- moves, kind, distance

Core claim this script exists to support without transcription risk:
    "LP-H8 improved R*, reduced empty mileage, AND reduced reactive (shortage)
    dispatch volume, consistently, across all five tested scenarios" --
    three core operational metrics, each independently verified here.
"""
import pandas as pd
import re

SIM = "/home/b5by/zhirong.b5by/work/shanghai_new/outputs/fleet_sim"
TMP = f"{SIM}/_sweep_tmp"

# (scenario, mechanism, tag) -- tags point at the actual saved simulation output files.
RUNS = [
    ("ref", "tier0", "robust_ref_tier0_uniform"),
    ("ref", "lp_H8", "robust_ref_lp_uniform"),
    ("ref", "lp_H12", "posLP_r1_H12"),
    ("fleet7500_mc", "tier0", "robust_fleet7500_mc_tier0_uniform"),
    ("fleet7500_mc", "lp_H8", "robust_fleet7500_mc_lp_H8_solo"),
    ("fleet7500_mc", "lp_H12", "robust_fleet7500_mc_lp_H12"),
    ("fleet4500", "tier0", "robust_fleet4500_tier0_uniform"),
    ("fleet4500", "lp_H8", "robust_fleet4500_lp_H8_solo"),
    ("fleet4500", "lp_H12", "robust_fleet4500_lp_H12"),
    ("kappa20", "tier0", "robust_kappa20_tier0_uniform"),
    ("kappa20", "lp_H8", "robust_kappa20_lp_uniform"),
    ("kappa20", "lp_H12", "kappa20_H12_debug"),
    ("threshold", "tier0", "robust_threshold_tier0_uniform"),
    ("threshold", "lp_H8", "robust_threshold_lp_H8_solo"),
    ("threshold", "lp_H12", "robust_threshold_lp_H12"),
]

SCENARIO_ORDER = ["ref", "fleet7500_mc", "fleet4500", "kappa20", "threshold"]


def load_metrics():
    rows = []
    for scenario, mechanism, tag in RUNS:
        ts = pd.read_csv(f"{SIM}/time_step_summary_{tag}.csv")
        met = ts["met_demand"].sum()
        real_unmet = (ts["unmet_fleet_insufficient"] + ts["unmet_away_flying"] + ts["unmet_insufficient_battery"]).sum()
        r_star = met / (met + real_unmet) if (met + real_unmet) > 0 else float("nan")

        rebal = pd.read_csv(f"{SIM}/rebalance_log_{tag}.csv")
        mileage = rebal["distance"].sum()
        n_positioning = len(rebal[rebal["kind"].isin(["positioning", "predictive"])])
        n_shortage = len(rebal[rebal["kind"] == "shortage"])
        n_overflow = len(rebal[rebal["kind"] == "overflow"])

        rows.append({
            "scenario": scenario, "mechanism": mechanism, "R_star": r_star,
            "empty_mileage_km": mileage, "n_positioning": n_positioning,
            "n_shortage": n_shortage, "n_overflow": n_overflow,
        })
    return pd.DataFrame(rows)


def main():
    df = load_metrics()

    tier0 = df[df["mechanism"] == "tier0"].set_index("scenario")
    lp8 = df[df["mechanism"] == "lp_H8"].set_index("scenario")
    lp12 = df[df["mechanism"] == "lp_H12"].set_index("scenario")

    summary = pd.DataFrame(index=SCENARIO_ORDER)
    summary["R_star_tier0"] = tier0["R_star"]
    summary["R_star_lpH8"] = lp8["R_star"]
    summary["delta_R_pp"] = (lp8["R_star"] - tier0["R_star"]) * 100
    summary["mileage_reduction_pct"] = (tier0["empty_mileage_km"] - lp8["empty_mileage_km"]) / tier0["empty_mileage_km"] * 100
    summary["shortage_reduction_pct"] = (tier0["n_shortage"] - lp8["n_shortage"]) / tier0["n_shortage"] * 100

    pd.set_option("display.width", 200)
    print("=== Final headline table (H=8 vs Tier0, all figures code-generated) ===")
    print(summary.to_string(float_format=lambda x: f"{x:.2f}"))

    print("\n=== Ranges for the report sentence (min/mean/max, code-generated) ===")
    for col, label in [("delta_R_pp", "R* improvement (pp)"),
                        ("mileage_reduction_pct", "Empty mileage reduction (%)"),
                        ("shortage_reduction_pct", "Reactive (shortage) dispatch reduction (%)")]:
        print(f"{label}: min={summary[col].min():.1f}  mean={summary[col].mean():.1f}  max={summary[col].max():.1f}")

    print("\n=== Verification: does EVERY scenario improve on all 3 metrics (not just on average)? ===")
    print(f"R* improved in all 5:        {(summary['delta_R_pp'] > 0).all()}")
    print(f"Mileage reduced in all 5:    {(summary['mileage_reduction_pct'] > 0).all()}")
    print(f"Shortage reduced in all 5:   {(summary['shortage_reduction_pct'] > 0).all()}")

    print("\n=== H=8 vs H=12 marginal comparison (both mechanisms are LP; R* only, timing tracked separately) ===")
    h_compare = pd.DataFrame(index=SCENARIO_ORDER)
    h_compare["R_star_H8"] = lp8["R_star"]
    h_compare["R_star_H12"] = lp12["R_star"]
    h_compare["delta_H12_over_H8_pp"] = (lp12["R_star"] - lp8["R_star"]) * 100
    print(h_compare.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\nH12-over-H8 delta: min={h_compare['delta_H12_over_H8_pp'].min():.2f}pp  "
          f"mean={h_compare['delta_H12_over_H8_pp'].mean():.2f}pp  max={h_compare['delta_H12_over_H8_pp'].max():.2f}pp")

    out_csv = "/local/user/1483804531/claude-1483804531/-home-b5by-zhirong-b5by-work/1e35ed92-ff00-4c93-a5d2-47b53db06110/scratchpad/final_report_summary.csv"
    summary.to_csv(out_csv)
    print(f"\nsaved -> {out_csv}")


if __name__ == "__main__":
    main()
