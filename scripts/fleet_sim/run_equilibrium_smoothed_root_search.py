#!/usr/bin/env python3
"""Bracket a locally smoothed demand--operations fixed point.

Each response F_bar(W) is the arithmetic mean of three independently simulated
responses at W-radius, W and W+radius.  This is an explicitly separate
numerical smoothing experiment for discrete vehicle-dispatch effects.
"""

import argparse
import copy
import json
from pathlib import Path

import pandas as pd

from run_equilibrium_root_search import evaluate
from run_equilibrium_search import PROJECT_DIR, ROUTED_CSV, SITES_CSV


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True)
    p.add_argument("--sites-csv", default=str(SITES_CSV))
    p.add_argument("--routed-csv", default=str(ROUTED_CSV))
    p.add_argument("--logit-scale", type=float, default=30.0)
    p.add_argument("--kappa-w", type=float, default=1.5)
    p.add_argument("--vehicles-per-vertiport", type=int, required=True)
    p.add_argument("--discharge-rate", type=float, default=1.0)
    p.add_argument("--charging-rate-per-bin", type=float, default=25.0)
    p.add_argument("--rebalance-interval", type=int, default=1)
    p.add_argument("--n-bins", type=int, default=1000)
    p.add_argument("--rebalance-mechanism", choices=["tier0", "lp"], default="lp")
    p.add_argument("--positioning-lp-horizon", type=int, default=8)
    p.add_argument("--wait-low-min", type=float, default=0.0)
    p.add_argument("--wait-high-min", type=float, default=2.0)
    p.add_argument("--smooth-radius-min", type=float, default=0.02)
    p.add_argument("--smooth-points", type=int, default=3,
                   help="Odd number of equally spaced response samples in the local smoothing window.")
    p.add_argument("--wait-tol-min", type=float, default=0.02)
    p.add_argument("--bracket-tol-min", type=float, default=0.02)
    p.add_argument("--max-bisection-steps", type=int, default=8)
    p.add_argument("--min-service-rate", type=float, default=0.965)
    p.add_argument("--max-wait-min", type=float, default=5.0)
    return p.parse_args()


def smooth_evaluate(args, out_dir: Path, evaluation: int, center: float) -> dict:
    smooth_points = getattr(args, "smooth_points", 3)
    if smooth_points < 3 or smooth_points % 2 == 0:
        raise ValueError("smooth-points must be an odd integer of at least 3")
    if smooth_points == 3:
        samples = (("low", -args.smooth_radius_min), ("mid", 0.0), ("high", args.smooth_radius_min))
    else:
        half = smooth_points // 2
        samples = tuple(
            (f"p{index + 1:02d}", args.smooth_radius_min * index / half)
            for index in range(-half, half + 1)
        )
    components = []
    for suffix, offset in samples:
        local = copy.copy(args)
        local.label = f"smroot_{args.label}_e{evaluation:02d}_{suffix}"
        components.append(evaluate(local, out_dir, 1, max(0.0, center + offset)))
    frame = pd.DataFrame(components)
    result = {
        "evaluation": evaluation,
        "wait_input_min": center,
        "demand_monthly": frame.demand_monthly.mean(),
        "wait_observed_min": frame.wait_observed_min.mean(),
        "service_rate": frame.service_rate.mean(),
        "residual_min": frame.wait_observed_min.mean() - center,
        "demand_spread_pct": (frame.demand_monthly.max() - frame.demand_monthly.min()) / frame.demand_monthly.mean() * 100,
        "wait_spread_min": frame.wait_observed_min.max() - frame.wait_observed_min.min(),
        "service_spread_pp": (frame.service_rate.max() - frame.service_rate.min()) * 100,
    }
    print(f"[{args.label}] smooth_eval={evaluation} W={center:.5f} Fbar={result['wait_observed_min']:.5f} "
          f"gbar={result['residual_min']:.5f} D={result['demand_monthly']:.0f} "
          f"R={result['service_rate']:.5f}", flush=True)
    return result


def main():
    args = parse_args()
    if not args.wait_low_min < args.wait_high_min:
        raise ValueError("wait-low-min must be below wait-high-min")
    out_dir = PROJECT_DIR / "outputs/fleet_sim"
    records = []
    def save():
        pd.DataFrame(records).to_csv(out_dir / f"smoothed_root_trajectory_{args.label}.csv", index=False)
    low = smooth_evaluate(args, out_dir, 1, args.wait_low_min); records.append(low); save()
    high = smooth_evaluate(args, out_dir, 2, args.wait_high_min); records.append(high); save()
    status, root = "no_sign_changing_bracket", None
    if low["residual_min"] * high["residual_min"] <= 0:
        for step in range(1, args.max_bisection_steps + 1):
            mid = smooth_evaluate(args, out_dir, step + 2, (low["wait_input_min"] + high["wait_input_min"]) / 2)
            records.append(mid); save()
            if abs(mid["residual_min"]) <= args.wait_tol_min:
                root, status = mid, "residual_tolerance_met"
                break
            if low["residual_min"] * mid["residual_min"] < 0:
                high = mid
            else:
                low = mid
            if high["wait_input_min"] - low["wait_input_min"] <= args.bracket_tol_min:
                root, status = min((low, high), key=lambda row: abs(row["residual_min"])), "bracket_tolerance_met"
                break
        else:
            root, status = min((low, high), key=lambda row: abs(row["residual_min"])), "max_steps_reached"
    feasible = bool(root and root["wait_observed_min"] <= args.max_wait_min and root["service_rate"] >= args.min_service_rate)
    result = {"label": args.label, "status": status, "root_estimate": root, "feasible": feasible,
              "constraints": {"max_wait_min": args.max_wait_min, "min_service_rate": args.min_service_rate},
              "evaluations": len(records), "final_bracket": {"low": low, "high": high}}
    path = out_dir / f"smoothed_root_result_{args.label}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[{args.label}] SMOOTHED_ROOT_DONE status={status} feasible={feasible} result={path}", flush=True)


if __name__ == "__main__":
    main()
